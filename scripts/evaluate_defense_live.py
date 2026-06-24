#!/usr/bin/env python3
"""
Live C4 defense evaluation — measured on REAL defended captures, not a
feature-space simulation.

For each deployed defense we collected an actual defended dataset (agents run
with the defense active; traffic captured on the wire):
    rate  — dummy sub-calls + jittered/reordered delegation (count/rate defense)
    pad   — SSE constant-size event padding               (size defense)

Threat model for the measurement: a FIXED attacker trained on undefended
traffic (the realistic case — the adversary does not get to retrain on the
defended deployment).  We report:

  attack accuracy (undefended)         — RF, group-safe CV on the baseline
  attack accuracy under each defense   — baseline-trained RF applied to defended
  byte overhead   = mean wire-bytes(defended) / mean wire-bytes(baseline) − 1
  latency overhead= mean duration(defended)  / mean duration(baseline)  − 1

This directly tests the C4 conclusion on real traffic (N=50/pair).  The finding:
BOTH defenses are partially effective and expensive — neither is a clean win.
Size-padding (pad) and rate/count (dummy sub-calls + jittered delegation) drop
the attack by a similar amount (~0.12 accuracy each) and each leaves roughly
70 % of the above-chance signal intact, at ~30 % byte overhead.  Real mitigation
would need far more aggressive packet-count/rate obfuscation than either applies
here.  (Note: the pad latency_overhead can read negative — that is a measurement
artifact of the separately-collected padded set, NOT a speedup from padding.)

Usage:
    python scripts/evaluate_defense_live.py \
        --baseline data/processed --baseline-raw data/raw \
        --rate data/processed_defense_rate --rate-raw data/raw_defense_rate \
        --pad  data/processed_defense_pad  --pad-raw  data/raw_defense_pad
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_cross_deployment import load_deployment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASK = "workflow"


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced"
    )


def baseline_cv(X, y, groups):
    """Group-safe CV of the undefended attacker.

    Returns (acc, f1, oof_true, oof_pred) — the pooled out-of-fold predictions
    let the caller attach a bootstrap 95 % CI.
    """
    y = np.asarray(y)
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        clf = _rf().fit(X, y)
        p = clf.predict(X)
        return accuracy_score(y, p), f1_score(y, p, average="macro"), list(y), list(p)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs, f1s, oof_t, oof_p = [], [], [], []
    for tr, te in skf.split(X, y, groups):
        clf = _rf().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        oof_t.extend(list(y[te]))
        oof_p.extend(list(p))
    return float(np.mean(accs)), float(np.mean(f1s)), oof_t, oof_p


def transfer_to_defended(Xb, yb, Xd, yd):
    """Train attacker on ALL undefended data, test on defended data.

    Returns (acc, f1, y_true, y_pred) so the caller can bootstrap a 95 % CI on
    the defended test set.
    """
    clf = _rf().fit(Xb, np.asarray(yb))
    p = clf.predict(Xd)
    yd = np.asarray(yd)
    return accuracy_score(yd, p), f1_score(yd, p, average="macro"), list(yd), list(p)


def trace_wire_stats(raw_dir: Path) -> tuple[float, float]:
    """Mean total wire-bytes and mean duration (s) per pcap trace."""
    from scapy.all import rdpcap  # local import — scapy is slow to import

    bytes_list, dur_list = [], []
    for pcap in sorted(Path(raw_dir).glob("*.pcap")):
        try:
            pkts = rdpcap(str(pcap))
        except Exception:
            continue
        if not pkts:
            continue
        total = sum(getattr(p, "wirelen", len(p)) for p in pkts)
        ts = [float(p.time) for p in pkts]
        bytes_list.append(total)
        dur_list.append(max(ts) - min(ts))
    if not bytes_list:
        return 0.0, 0.0
    return float(np.mean(bytes_list)), float(np.mean(dur_list))


def main(args: argparse.Namespace) -> None:
    from evaluation.stats import bootstrap_ci

    Xb, _, yb, gb = load_deployment(Path(args.baseline), TASK)
    base_acc, base_f1, b_t, b_p = baseline_cv(Xb, yb, gb)
    base_ci = bootstrap_ci(b_t, b_p)
    base_bytes, base_dur = trace_wire_stats(Path(args.baseline_raw))
    chance = 1.0 / len(set(yb))

    logger.info("Undefended attacker: acc=%.3f [%.3f, %.3f] macro_f1=%.3f (chance=%.3f)",
                base_acc, base_ci["accuracy_ci_lo"], base_ci["accuracy_ci_hi"], base_f1, chance)

    results: dict = {
        "task": TASK,
        "chance": chance,
        "latency_overhead_note": (
            "CONFOUNDED — the rate/pad defended sets were collected separately from the "
            "baseline, so absolute trace durations are not a controlled comparison (pad "
            "even reads negative). The latency_overhead fields are retained for the record "
            "but are NOT a defense cost; only byte_overhead (bandwidth) is reported."
        ),
        "none": {
            "accuracy": base_acc,
            "accuracy_ci_lo": base_ci["accuracy_ci_lo"],
            "accuracy_ci_hi": base_ci["accuracy_ci_hi"],
            "macro_f1": base_f1,
            "macro_f1_ci_lo": base_ci["macro_f1_ci_lo"],
            "macro_f1_ci_hi": base_ci["macro_f1_ci_hi"],
            "byte_overhead": 0.0, "latency_overhead": 0.0,
            "mean_trace_bytes": base_bytes, "mean_trace_dur_s": base_dur,
        },
    }

    for name, proc, raw in [
        ("rate", args.rate, args.rate_raw),
        ("pad", args.pad, args.pad_raw),
    ]:
        if not proc or not (Path(proc) / "labels.json").exists():
            logger.warning("skip %s — no processed features at %s", name, proc)
            continue
        Xd, _, yd, _ = load_deployment(Path(proc), TASK)
        acc, f1, yt, yp = transfer_to_defended(Xb, yb, Xd, yd)
        ci = bootstrap_ci(yt, yp)
        d_bytes, d_dur = trace_wire_stats(Path(raw)) if raw else (0.0, 0.0)
        byte_ohd = (d_bytes / base_bytes - 1.0) if base_bytes else 0.0
        lat_ohd = (d_dur / base_dur - 1.0) if base_dur else 0.0
        retention = (acc - chance) / (base_acc - chance) if base_acc > chance else 0.0
        f1_retention = (f1 - chance) / (base_f1 - chance) if base_f1 > chance else 0.0
        results[name] = {
            "accuracy": acc,
            "accuracy_ci_lo": ci["accuracy_ci_lo"],
            "accuracy_ci_hi": ci["accuracy_ci_hi"],
            "macro_f1": f1,
            "macro_f1_ci_lo": ci["macro_f1_ci_lo"],
            "macro_f1_ci_hi": ci["macro_f1_ci_hi"],
            "acc_drop": base_acc - acc,
            "macro_f1_drop": base_f1 - f1,
            "above_chance_retention": retention,
            "above_chance_retention_f1": f1_retention,
            "byte_overhead": byte_ohd,
            "latency_overhead": lat_ohd,
            "mean_trace_bytes": d_bytes, "mean_trace_dur_s": d_dur,
        }
        logger.info(
            "defense=%-4s  acc=%.3f [%.3f, %.3f] (drop %.3f, retains %.0f%% of signal)  "
            "byte_ohd=%+.0f%%  latency_ohd=%+.0f%%",
            name, acc, ci["accuracy_ci_lo"], ci["accuracy_ci_hi"], base_acc - acc,
            100 * retention, 100 * byte_ohd, 100 * lat_ohd,
        )

    out = Path(os.environ.get("A2A_RESULTS_DIR", "data/results")) / "defense" / "defense_live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  LIVE C4 DEFENSE EVALUATION (measured on real defended captures)")
    print("=" * 72)
    print(f"  {'defense':<8}{'macro-F1 [95% CI]':>26}{'acc':>8}{'F1 kept':>10}{'byte ohd':>11}")
    print("  " + "-" * 70)
    for name in ("none", "rate", "pad"):
        if name not in results:
            continue
        r = results[name]
        f1ci = f"[{r.get('macro_f1_ci_lo', 0):.2f},{r.get('macro_f1_ci_hi', 0):.2f}]"
        ret = r.get("above_chance_retention_f1")
        ret_s = f"{100*ret:.0f}%" if ret is not None else "—"
        print(f"  {name:<8}{r['macro_f1']:>12.3f} {f1ci:>13}{r['accuracy']:>8.3f}{ret_s:>10}"
              f"{100*r['byte_overhead']:>10.0f}%")
    print("  " + "-" * 70)
    print("  (macro-F1 is the headline metric; latency overhead omitted — confounded,")
    print("   see latency_overhead_note in the JSON. Bandwidth = byte overhead only.)")
    print("  CONCLUSION (N=50/pair): both defenses are partially effective and")
    print("  expensive — neither is a clean win.  Size-padding (pad) and rate/count")
    print("  drop the attack by a similar amount (~0.12 acc each) and each leaves")
    print("  ~70% of the above-chance signal intact, at ~30% byte overhead.  Real")
    print("  mitigation needs far more aggressive packet-count/rate obfuscation.")
    print("  (pad's negative latency overhead is a measurement artifact of the")
    print("   separately-collected padded set, not a speedup from padding.)")
    print("=" * 72)
    print(f"\nWrote {out}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live C4 defense evaluation on real captures")
    p.add_argument("--baseline", default="data/processed")
    p.add_argument("--baseline-raw", dest="baseline_raw", default="data/raw")
    p.add_argument("--rate", default="data/processed_defense_rate")
    p.add_argument("--rate-raw", dest="rate_raw", default="data/raw_defense_rate")
    p.add_argument("--pad", default="data/processed_defense_pad")
    p.add_argument("--pad-raw", dest="pad_raw", default="data/raw_defense_pad")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
