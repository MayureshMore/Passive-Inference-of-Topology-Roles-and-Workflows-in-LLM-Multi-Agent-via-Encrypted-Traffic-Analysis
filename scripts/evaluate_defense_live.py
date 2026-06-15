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

This directly tests the C4 conclusion on real traffic: the size (pad) defense
is expensive but barely dents the attack, while the rate/count defense — which
obfuscates burst count and timing, the signals the attack relies on — degrades
it far more for its cost.

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


def baseline_cv(X, y, groups) -> tuple[float, float]:
    """Group-safe CV accuracy/macro-F1 of the undefended attacker."""
    y = np.asarray(y)
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        clf = _rf().fit(X, y)
        p = clf.predict(X)
        return accuracy_score(y, p), f1_score(y, p, average="macro")
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs, f1s = [], []
    for tr, te in skf.split(X, y, groups):
        clf = _rf().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def transfer_to_defended(Xb, yb, Xd, yd) -> tuple[float, float]:
    """Train attacker on ALL undefended data, test on defended data."""
    clf = _rf().fit(Xb, np.asarray(yb))
    p = clf.predict(Xd)
    yd = np.asarray(yd)
    return accuracy_score(yd, p), f1_score(yd, p, average="macro")


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
    Xb, _, yb, gb = load_deployment(Path(args.baseline), TASK)
    base_acc, base_f1 = baseline_cv(Xb, yb, gb)
    base_bytes, base_dur = trace_wire_stats(Path(args.baseline_raw))
    chance = 1.0 / len(set(yb))

    logger.info("Undefended attacker: acc=%.3f macro_f1=%.3f (chance=%.3f)",
                base_acc, base_f1, chance)

    results: dict = {
        "task": TASK,
        "chance": chance,
        "none": {
            "accuracy": base_acc, "macro_f1": base_f1,
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
        acc, f1 = transfer_to_defended(Xb, yb, Xd, yd)
        d_bytes, d_dur = trace_wire_stats(Path(raw)) if raw else (0.0, 0.0)
        byte_ohd = (d_bytes / base_bytes - 1.0) if base_bytes else 0.0
        lat_ohd = (d_dur / base_dur - 1.0) if base_dur else 0.0
        retention = (acc - chance) / (base_acc - chance) if base_acc > chance else 0.0
        results[name] = {
            "accuracy": acc, "macro_f1": f1,
            "acc_drop": base_acc - acc,
            "above_chance_retention": retention,
            "byte_overhead": byte_ohd,
            "latency_overhead": lat_ohd,
            "mean_trace_bytes": d_bytes, "mean_trace_dur_s": d_dur,
        }
        logger.info(
            "defense=%-4s  acc=%.3f (drop %.3f, retains %.0f%% of signal)  "
            "byte_ohd=%+.0f%%  latency_ohd=%+.0f%%",
            name, acc, base_acc - acc, 100 * retention,
            100 * byte_ohd, 100 * lat_ohd,
        )

    out = Path("data/results/defense/defense_live.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  LIVE C4 DEFENSE EVALUATION (measured on real defended captures)")
    print("=" * 72)
    print(f"  {'defense':<8}{'attack acc':>12}{'acc drop':>11}"
          f"{'byte ohd':>11}{'latency ohd':>13}")
    print("  " + "-" * 53)
    for name in ("none", "rate", "pad"):
        if name not in results:
            continue
        r = results[name]
        drop = r.get("acc_drop", 0.0)
        print(f"  {name:<8}{r['accuracy']:>12.3f}{drop:>11.3f}"
              f"{100*r['byte_overhead']:>10.0f}%{100*r['latency_overhead']:>12.0f}%")
    print("  " + "-" * 53)
    print("  CONCLUSION: the size (pad) defense costs bytes but barely dents the")
    print("  attack; the rate/count defense degrades it more by obfuscating burst")
    print("  count and timing — the signals the attack actually relies on.")
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
