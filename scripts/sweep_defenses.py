#!/usr/bin/env python3
"""
Defense overhead–accuracy SWEEP — turn the two live defense points (§5) into a
curve, by applying each defense at several strengths to the *undefended base
capture* and re-running the fixed attacker.

Why a simulation-on-the-base-capture (and not more live collection): the two
committed points in `defense_live.json` are REAL defended captures, but a full
strength sweep would need a fresh live capture per level.  Instead we apply the
defense transform deterministically to the undefended base pcaps (data/raw) and
re-extract features through the *exact* validated pipeline
(`features/extractor.py`).  The two live points then serve as ground-truth
ANCHORS: at the live parameters the simulation must reproduce them (see the
`validation` block) — a sanity check that the sweep did not change the method.

What is simulated, faithfully:
  * pad  (size defense)  — the deployed defense rounds every SSE *response* event
    up to the next `cell` multiple (agents/base.py `_cell_pad_len = (-len)%cell`);
    on loopback one event == one agent→client packet, so we round each response
    data packet's on-wire length (payload rounded to the cell, header kept).
    Requests and timing are untouched — matching scripts/analyze_sse_sizes.py.
    Sweeps the cell size.  Latency overhead is 0 by construction (bytes, not time).
  * rate (timing defense) — a constant-rate / minimum inter-packet-spacing model:
    impose a floor on the gap between consecutive packets within each flow.
    Sweeps the floor.  LATENCY is SCHEDULE-DERIVED (computed from the imposed
    spacing on the base capture), NOT noisy wall-clock — this fixes the
    separate-capture latency confound in defense_live.json.  Byte overhead 0.
    NOTE: this is a *timing* defense; the live "rate" point is a different
    (count-based: dummy sub-calls + reordered delegation) mechanism, so it is
    plotted as a measured anchor, not claimed to lie on this timing curve.

Attacker + metric are identical to scripts/evaluate_defense_live.py: a FIXED
attacker (RandomForest 300, balanced, seed 0) trained on UNDEFENDED traffic and
applied to the defended set, group-safe 5-fold CV on prompt_group (leak-free:
a trace's undefended version never trains the fold that tests its defended
version), macro-F1 with a percentile bootstrap 95% CI (seed 42).

Writes ${A2A_RESULTS_DIR:-data/results}/defense_curve.json.  Does NOT touch any
existing committed result.  Heavier than the rest of the suite (re-extracts the
base capture several times), so reproduce.sh runs it only under --full-suite.

Usage:
    venv/bin/python scripts/sweep_defenses.py \
        --raw data/raw --processed data/processed \
        --pad-cells 128 256 512 1024 2048 \
        --rate-gaps-ms 1 2 5 10
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.stats import bootstrap_ci  # noqa: E402
from features.extractor import FeatureExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASK = "workflow"
AGENT_PORTS = {8000, 8001, 8002, 8003}
HDR = 56           # loopback(4)+IPv4(20)+TCP+timestamp(32); see analyze_sse_sizes.py

# The deployed pad rounds the *SSE-event JSON* up to the next cell
# (agents/base.py), NOT the on-wire packet.  On the wire an event is
# JSON + a ~constant HTTP/SSE framing overhead + HDR.  We recover the JSON size
# as `wirelen - PAD_WIRE_OFFSET`, pad THAT to the cell, and add the offset back —
# so the simulation rounds in the same layer the real defense does.
#
# PAD_WIRE_OFFSET (=HDR+framing) is CALIBRATED against the committed live pad set:
# undefended response events sit at wire-payload p50 535 B and the live pad set at
# p50 773 B (= one 512-cell + 261 framing), i.e. JSON p50 ≈ 535-261 ≈ 274 → rounds
# to 512 → 773.  So PAD_WIRE_OFFSET = 56 + 261 = 317.  With this single constant the
# cell=512 sim reproduces BOTH live pad anchors (+~31% bytes, distinct 414→107,
# macro-F1 inside the live CI); the other cell sizes are then genuine predictions.
PAD_WIRE_OFFSET = 317

# A packet is (ts, size_bytes, flow_key, direction); direction +1 = agent→client
# (response / SSE), -1 = client→agent (request).  This matches FeatureExtractor.


def _rf() -> RandomForestClassifier:
    # Identical to scripts/evaluate_defense_live.py::_rf().
    return RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced"
    )


# ── Defense transforms (applied to one trace's packet list) ───────────────────

def pad_transform(pkts: list[tuple], cell: int, offset: int = PAD_WIRE_OFFSET) -> list[tuple]:
    """Round each RESPONSE event's JSON up to the next `cell` multiple.

    Mirrors the deployed pad (agents/base.py): it pads the SSE-event JSON, not the
    on-wire packet.  We recover JSON ≈ wirelen - offset, round it up to the cell,
    and add `offset` back → the on-wire size becomes ceil(JSON/cell)*cell + offset.
    Requests, ACKs (JSON<=0), and timing are untouched.  cell<=0 is a no-op.
    """
    if cell <= 0:
        return pkts
    out = []
    for ts, size, fk, d in pkts:
        json_len = size - offset
        if d == 1 and json_len > 0:          # a real agent→client SSE event
            padded = int(math.ceil(json_len / cell) * cell)
            size = padded + offset
        out.append((ts, size, fk, d))
    return out


def rate_transform(pkts: list[tuple], min_gap_s: float) -> list[tuple]:
    """Impose a minimum inter-packet gap within each flow (constant-rate model).

    Deterministic timing defense: within each flow, in chronological order, push
    each packet so it is at least `min_gap_s` after the previous one.  Sizes are
    untouched (byte overhead 0); the added duration is the schedule-derived
    latency cost.  min_gap_s<=0 is a no-op.
    """
    if min_gap_s <= 0:
        return pkts
    from collections import defaultdict
    idx_by_flow: dict[str, list[int]] = defaultdict(list)
    for i, (ts, size, fk, d) in enumerate(pkts):
        idx_by_flow[fk].append(i)
    new_ts = [ts for ts, _, _, _ in pkts]
    for fk, idxs in idx_by_flow.items():
        idxs.sort(key=lambda i: pkts[i][0])
        prev = None
        for i in idxs:
            t = pkts[i][0]
            if prev is not None and t < prev + min_gap_s:
                t = prev + min_gap_s
            new_ts[i] = t
            prev = t
    return [(new_ts[i], pkts[i][1], pkts[i][2], pkts[i][3]) for i in range(len(pkts))]


# ── Load base capture once, cache per-trace packet lists ──────────────────────

def load_base(raw_dir: Path, processed_dir: Path):
    """Read every base pcap once → (run_ids, packet_lists, y, groups).

    Labels/groups come from processed_dir/labels.json keyed by pcap stem (run_id).
    Only non-role traces of the given TASK are kept.
    """
    labels = json.loads((processed_dir / "labels.json").read_text())
    ext = FeatureExtractor(agent_ports=AGENT_PORTS, use_scapy=True)
    run_ids, pkt_lists, y, groups = [], [], [], []
    n_skip = 0
    for pcap in sorted(raw_dir.glob("*.pcap")):
        rid = pcap.stem
        info = labels.get(rid)
        if not info or "__role__" in rid or info.get(TASK) is None:
            continue
        try:
            pkts = ext._read_pcap(pcap)
        except Exception as exc:  # unreadable pcap
            logger.warning("skip %s: %s", rid, exc)
            n_skip += 1
            continue
        if not pkts:
            n_skip += 1
            continue
        run_ids.append(rid)
        pkt_lists.append(pkts)
        y.append(info[TASK])
        groups.append(info.get("prompt_group", rid))
    logger.info("loaded %d base traces (%d skipped) from %s", len(run_ids), n_skip, raw_dir)
    return run_ids, pkt_lists, np.asarray(y), np.asarray(groups), ext


def features_and_overhead(ext, pkt_lists, run_ids, transform):
    """Apply `transform` to every trace, return (X flat matrix, mean_bytes, mean_dur)."""
    X, tot_bytes, durs = [], [], []
    for rid, pkts in zip(run_ids, pkt_lists):
        tp = transform(pkts)
        tf = ext._compute_features(rid, tp)
        X.append(tf.flat_vector())
        tot_bytes.append(sum(s for _, s, _, _ in tp))
        ts = [t for t, _, _, _ in tp]
        durs.append((max(ts) - min(ts)) if len(ts) > 1 else 0.0)
    return np.asarray(X, dtype=np.float32), float(np.mean(tot_bytes)), float(np.mean(durs))


# ── Fixed-attacker evaluation (mirrors evaluate_defense_live semantics) ───────

def eval_fixed_attacker(X_undef, X_def, y, groups):
    """Group-safe CV: train on UNDEFENDED train-groups, test on DEFENDED test-groups.

    Leak-free (a trace's undefended features never train the fold that tests its
    defended features).  Pool OOF preds → macro-F1 / accuracy + bootstrap 95% CI.
    """
    n_splits = min(5, len(set(groups)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof_t, oof_p, oof_g = [], [], []
    groups_arr = np.asarray(groups)
    for tr, te in skf.split(X_undef, y, groups):
        clf = _rf().fit(X_undef[tr], y[tr])
        oof_t.extend(list(y[te]))
        oof_p.extend(list(clf.predict(X_def[te])))
        oof_g.extend(groups_arr[te].tolist())
    # CV is cluster-aware (prompt_group); the CI must be too.
    ci = bootstrap_ci(oof_t, oof_p, classes=sorted(set(y)), groups=oof_g)
    return {
        "attack_macro_f1": float(f1_score(oof_t, oof_p, average="macro")),
        "accuracy": float(accuracy_score(oof_t, oof_p)),
        "ci_low": ci["macro_f1_ci_lo"],
        "ci_high": ci["macro_f1_ci_hi"],
        "ci_method": ci["ci_method"],          # cluster (prompt_group) bootstrap — project convention
        "ci_n_clusters": ci.get("n_clusters"),
    }


def main(args: argparse.Namespace) -> None:
    raw, processed = Path(args.raw), Path(args.processed)
    if not (processed / "labels.json").exists():
        logger.error("no labels.json in %s", processed)
        return
    if not any(raw.glob("*.pcap")):
        logger.error("no pcaps in %s — the sweep needs the base capture", raw)
        return

    run_ids, pkt_lists, y, groups, ext = load_base(raw, processed)
    chance = 1.0 / len(set(y))

    # Undefended baseline (identity transform) — the fixed attacker trains on this.
    X_none, base_bytes, base_dur = features_and_overhead(ext, pkt_lists, run_ids, lambda p: p)
    base = eval_fixed_attacker(X_none, X_none, y, groups)
    logger.info("identity (none): macro_f1=%.3f (chance=%.3f, mean_bytes=%.0f)",
                base["attack_macro_f1"], chance, base_bytes)

    rows = [{
        "defense": "none_sim", "param": 0, "param_unit": "-",
        "byte_overhead": 0.0, "latency_overhead": 0.0, **base,
    }]

    # ── pad (size) sweep ──────────────────────────────────────────────────────
    for cell in args.pad_cells:
        X_def, b, _ = features_and_overhead(
            ext, pkt_lists, run_ids,
            lambda p, c=cell: pad_transform(p, c, args.pad_wire_offset))
        res = eval_fixed_attacker(X_none, X_def, y, groups)
        rows.append({
            "defense": "pad_size_sim", "param": int(cell), "param_unit": "cell_bytes",
            "byte_overhead": b / base_bytes - 1.0 if base_bytes else 0.0,
            "latency_overhead": 0.0,  # padding adds bytes, not time (schedule-derived)
            **res,
        })
        logger.info("pad cell=%-5d  byte_ohd=%+.0f%%  macro_f1=%.3f",
                    cell, 100 * rows[-1]["byte_overhead"], res["attack_macro_f1"])

    # ── rate (timing) sweep — schedule-derived latency ────────────────────────
    for gap_ms in args.rate_gaps_ms:
        gap_s = gap_ms / 1000.0
        X_def, _, d = features_and_overhead(ext, pkt_lists, run_ids, lambda p, g=gap_s: rate_transform(p, g))
        res = eval_fixed_attacker(X_none, X_def, y, groups)
        rows.append({
            "defense": "rate_timing_sim", "param": float(gap_ms), "param_unit": "min_gap_ms",
            "byte_overhead": 0.0,  # timing only
            "latency_overhead": d / base_dur - 1.0 if base_dur else 0.0,  # schedule-derived
            **res,
        })
        logger.info("rate min_gap=%-4sms  latency_ohd=%+.0f%%  macro_f1=%.3f",
                    gap_ms, 100 * rows[-1]["latency_overhead"], res["attack_macro_f1"])

    # ── Anchors + validation against the committed live measurements ──────────
    live_path = Path("data/results/defense/defense_live.json")
    measured_live, validation = {}, {}
    if live_path.exists():
        dl = json.loads(live_path.read_text())
        for k in ("none", "rate", "pad"):
            if k in dl:
                measured_live[k] = {
                    "macro_f1": dl[k]["macro_f1"],
                    "byte_overhead": dl[k].get("byte_overhead", 0.0),
                }
        pad512 = next((r for r in rows if r["defense"] == "pad_size_sim" and r["param"] == 512), None)
        validation = {
            "note": "The simulation must reproduce the live points at their parameters "
                    "(sanity check the sweep did not change the method).",
            "identity_sim_macro_f1": base["attack_macro_f1"],
            "committed_baseline_macro_f1": dl.get("none", {}).get("macro_f1"),
            "pad512_sim_macro_f1": pad512["attack_macro_f1"] if pad512 else None,
            "committed_pad_live_macro_f1": dl.get("pad", {}).get("macro_f1"),
            "pad512_sim_byte_overhead": pad512["byte_overhead"] if pad512 else None,
            "committed_pad_live_byte_overhead": dl.get("pad", {}).get("byte_overhead"),
        }

    out = {
        "task": TASK,
        "chance": chance,
        "model": "RandomForest(300, class_weight=balanced, random_state=0)",
        "method": "Fixed attacker trained on UNDEFENDED traffic, applied to the defended "
                  "set; group-safe 5-fold CV on prompt_group; macro-F1 with percentile "
                  "bootstrap 95% CI (seed 42). Identical to scripts/evaluate_defense_live.py.",
        "simulation_note": "Sweep points are the undefended base capture (data/raw) with the "
                           "defense transform applied at the packet level, then re-extracted "
                           "through features/extractor.py. Not a fresh live capture per level; "
                           "the live points in `measured_live` are the real defended captures.",
        "pad_wire_offset_bytes": args.pad_wire_offset,
        "pad_calibration_note": "pad rounds the SSE-event JSON (= wirelen - pad_wire_offset_bytes) "
                                "to the cell, matching agents/base.py. pad_wire_offset_bytes "
                                "(HDR+HTTP/SSE framing) is calibrated ONCE against the live pad set "
                                "so cell=512 reproduces it (undefended resp p50 535 B → live 773 B); "
                                "the other cell sizes are predictions. See `validation`.",
        "latency_note": "latency_overhead is SCHEDULE-DERIVED (computed from the imposed "
                        "inter-packet spacing on the base capture), NOT wall-clock — this "
                        "avoids the separate-capture latency confound in defense_live.json. "
                        "Size-padding is 0 by construction (bytes, not time).",
        "rate_caveat": "rate_timing_sim is a constant-rate TIMING model. The live 'rate' point "
                       "is a different, COUNT-based mechanism (dummy sub-calls + reordered "
                       "delegation), so it is shown as a measured anchor, not a point on this "
                       "timing curve.",
        "validation": validation,
        "measured_live": measured_live,
        "rows": rows,
    }

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "defense_curve.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  DEFENSE OVERHEAD–ACCURACY SWEEP (workflow attack, macro-F1)")
    print("=" * 74)
    print(f"  {'defense':<16}{'param':>10}{'byte ohd':>11}{'lat ohd':>10}{'macro-F1':>12}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['defense']:<16}{str(r['param'])+' '+r['param_unit']:>10}"
              f"{100*r['byte_overhead']:>10.0f}%{100*r['latency_overhead']:>9.0f}%"
              f"{r['attack_macro_f1']:>12.3f}")
    print("  " + "-" * 70)
    if validation:
        v = validation
        print(f"  VALIDATION vs committed live points:")
        print(f"    identity  sim F1={v['identity_sim_macro_f1']:.3f}  "
              f"vs live none F1={v['committed_baseline_macro_f1']:.3f}")
        if v.get("pad512_sim_macro_f1") is not None:
            print(f"    pad@512   sim F1={v['pad512_sim_macro_f1']:.3f}  "
                  f"vs live pad  F1={v['committed_pad_live_macro_f1']:.3f}  |  "
                  f"sim byte {100*v['pad512_sim_byte_overhead']:+.0f}% vs live "
                  f"{100*v['committed_pad_live_byte_overhead']:+.0f}%")
    print(f"  chance={chance:.3f}")
    print("=" * 74)
    print(f"\nWrote {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defense overhead–accuracy sweep (curve)")
    p.add_argument("--raw", default="data/raw", help="undefended base capture (pcaps)")
    p.add_argument("--processed", default="data/processed", help="labels.json source")
    p.add_argument("--pad-cells", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--rate-gaps-ms", type=float, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--pad-wire-offset", type=int, default=PAD_WIRE_OFFSET,
                   help="wirelen − offset = SSE-event JSON size that the pad rounds "
                        "(HDR+HTTP/SSE framing; calibrated to the live pad set)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
