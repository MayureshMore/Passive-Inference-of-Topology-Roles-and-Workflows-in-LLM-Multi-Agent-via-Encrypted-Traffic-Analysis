#!/usr/bin/env python3
"""
Feature sensitivity analysis for proposed defenses (proposal §C4 pre-study).

This script is NOT a live defense evaluation.  It synthetically modifies the
flat feature vectors to approximate what each defense would do to the features,
then measures how much RF accuracy drops.  Real defense evaluation requires
re-collecting pcap traces with the defense middleware running and re-extracting.

Defenses modelled (feature-level approximation):
  padding  — simulate constant-size padding by inflating all size features by
             a fixed fraction (padding ADDS bytes; attacker still sees totals
             but per-packet size variance collapses).
  timing   — zero out all inter-arrival and timing features (simulates
             schedule-randomisation; attacker sees no timing signal)
  dummy    — add Gaussian noise to all features (simulates dummy/cover
             traffic that inflates every statistic by a random amount)
  combined — all three together

Overhead column = fraction of feature dims modified (NOT real byte overhead).
Real per-byte / per-latency overhead requires re-collecting traces with the
live defense middleware — that is the next experiment step (C4 proper).

Usage:
    python scripts/evaluate_defense.py
    python scripts/evaluate_defense.py --task workflow
    python scripts/evaluate_defense.py --task topology --noise-std 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
RESULTS_DIR   = Path("data/results")

# ── Feature index groups (flat vector layout: 192-dim)
# pf_mean block:             indices   0–34  (35-dim)
# pf_top1 block (heaviest):  indices  35–69  (35-dim)
# pf_top2 block (2nd heavy): indices  70–104 (35-dim)
# per_system scalar block:   indices 105–121 (17 scalars)
# per_system pf_mean block:  indices 122–156 (35-dim)
# per_system pf_std block:   indices 157–191 (35-dim)
#
# pf layout offsets (same for pf_mean, pf_top1, pf_top2):
#   n_pkts_out(+0), n_pkts_in(+1), bytes_out(+2), bytes_in(+3),
#   mean_sz_out(+4), std_sz_out(+5), p25_sz_out(+6), p75_sz_out(+7),
#   mean_sz_in(+8), std_sz_in(+9), p25_sz_in(+10), p75_sz_in(+11),
#   duration_s(+12), mean_iat(+13), std_iat(+14), ...
#   bytes_out_ratio(+30), pkt_size_asymmetry(+31), n_small_inbound(+32),
#   n_response_bursts(+33), iqr_size_in(+34)
#
# per_system scalars (base 105):
#   n_flows(105), n_src(106), n_dst(107), n_pairs(108),
#   total_bytes(109), total_packets(110), total_duration(111),
#   flow_start_spread(112), flow_end_spread(113), max_concurrent(114),
#   total_bursts(115), mean_burst_rate(116), bytes_out_ratio(117),
#   heaviest_flow_bytes_frac(118), flow_bytes_cv(119),
#   n_response_heavy_flows(120), mean_flow_response_ratio(121)

# Size-related indices across all three pf blocks + per_system
SIZE_FEATURE_INDICES  = list(range(0, 12))           # pf_mean: pkt counts + size stats
SIZE_FEATURE_INDICES += [30, 31, 32, 34]             # pf_mean: asymmetry + sse + iqr
SIZE_FEATURE_INDICES += list(range(35, 47))          # pf_top1: pkt counts + size stats
SIZE_FEATURE_INDICES += [65, 66, 67, 69]             # pf_top1: asymmetry + sse + iqr
SIZE_FEATURE_INDICES += list(range(70, 82))          # pf_top2: pkt counts + size stats
SIZE_FEATURE_INDICES += [100, 101, 102, 104]         # pf_top2: asymmetry + sse + iqr
SIZE_FEATURE_INDICES += [109, 110]                   # per_system: total_bytes, total_packets

# Timing-related indices across all three pf blocks + per_system
TIMING_FEATURE_INDICES  = [12, 13, 14, 33]          # pf_mean: duration/iat/response_bursts
TIMING_FEATURE_INDICES += [47, 48, 49, 68]          # pf_top1: 35+12, 35+13, 35+14, 35+33
TIMING_FEATURE_INDICES += [82, 83, 84, 103]         # pf_top2: 70+12, 70+13, 70+14, 70+33
TIMING_FEATURE_INDICES += [111, 112, 113]           # per_system: duration, flow_start/end spread

ALL_INDICES = list(range(192))


# Per-flow (role, 35-dim) feature layout:
#   [0-11] packet counts + size stats, [12-14] timing, [15-19] burst stats,
#   [20-29] cumul_bytes, [30-34] asymmetry/sse-proxy
_SIZE_35   = list(range(0, 12)) + [30, 31, 32, 34]   # size + asymmetry + iqr
_TIMING_35 = [12, 13, 14, 18, 19, 33]                # timing + response_bursts


def _index_sets(dim: int) -> tuple[list[int], list[int]]:
    """Return (size_indices, timing_indices) appropriate for the feature vector dimension."""
    if dim == 35:
        return _SIZE_35, _TIMING_35
    return SIZE_FEATURE_INDICES, TIMING_FEATURE_INDICES


_PADDING_SCALE = 0.5  # simulate 50% extra bytes per packet added as padding


def apply_defense(X: np.ndarray, defense: str, noise_std: float = 0.3) -> np.ndarray:
    """Return a copy of X with the defense transformation applied."""
    Xd = X.copy()
    size_idx, timing_idx = _index_sets(X.shape[1])
    if defense == "padding":
        # Padding ADDS bytes — size features grow, not vanish.
        # Multiply all size features by (1 + padding_scale): attacker still sees
        # total-byte signal but per-packet variance collapses.
        Xd[:, size_idx] = Xd[:, size_idx] * (1.0 + _PADDING_SCALE)
    elif defense == "timing":
        Xd[:, timing_idx] = 0.0
    elif defense == "dummy":
        noise = np.random.default_rng(42).normal(0, noise_std, Xd.shape).astype(np.float32)
        Xd += noise
    elif defense == "combined":
        Xd[:, size_idx] = Xd[:, size_idx] * (1.0 + _PADDING_SCALE)
        Xd[:, timing_idx] = 0.0
        noise = np.random.default_rng(42).normal(0, noise_std, Xd.shape).astype(np.float32)
        Xd += noise
    return Xd


def overhead_fraction(defense: str) -> float:
    """Fraction of feature dimensions modified (proxy for protocol overhead)."""
    total = len(ALL_INDICES)
    if defense == "padding":
        return len(SIZE_FEATURE_INDICES) / total
    if defense == "timing":
        return len(TIMING_FEATURE_INDICES) / total
    if defense == "dummy":
        return 1.0
    if defense == "combined":
        modified = set(SIZE_FEATURE_INDICES) | set(TIMING_FEATURE_INDICES) | set(ALL_INDICES)
        return len(modified) / total
    return 0.0


def evaluate(task: str, noise_std: float) -> None:
    from models.random_forest import RFClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from evaluation.metrics import classification_metrics

    label_file = PROCESSED_DIR / "labels.json"
    labels_map = json.loads(label_file.read_text())

    X_list, y_list, group_list = [], [], []
    for npz_path in sorted(PROCESSED_DIR.glob("*.npz")):
        run_id = npz_path.stem
        is_role_sample = "__role__" in run_id
        if task == "role" and not is_role_sample:
            continue
        if task != "role" and is_role_sample:
            continue
        if run_id not in labels_map:
            continue
        info = labels_map[run_id]
        label = info.get(task)
        if label is None:
            continue
        d = np.load(npz_path, allow_pickle=False)
        X_list.append(d["flat"])
        y_list.append(label)
        group_list.append(info.get("prompt_group", run_id))

    if not X_list:
        raise ValueError(f"No samples for task={task}")

    X = np.stack(X_list)
    n_splits = min(5, min(y_list.count(c) for c in set(y_list)))
    if n_splits < 2:
        logger.error("Too few samples per class for task=%s", task)
        return

    defenses = ["none", "padding", "timing", "dummy", "combined"]
    results = {}

    for defense in defenses:
        Xd = apply_defense(X, defense, noise_std) if defense != "none" else X.copy()

        kf = StratifiedGroupKFold(n_splits=n_splits)
        accs, f1s = [], []
        for train_idx, val_idx in kf.split(Xd, y_list, groups=group_list):
            X_tr, X_val = Xd[train_idx], Xd[val_idx]
            y_tr = [y_list[i] for i in train_idx]
            y_val = [y_list[i] for i in val_idx]

            clf = RFClassifier(task=task)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            m = classification_metrics(y_val, preds, sorted(set(y_list)), task)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])

        mean_acc = float(np.mean(accs))
        mean_f1  = float(np.mean(f1s))
        overhead = overhead_fraction(defense)
        results[defense] = {
            "accuracy": mean_acc,
            "macro_f1": mean_f1,
            "overhead_fraction": overhead,
        }
        logger.info("defense=%-10s  acc=%.3f  f1=%.3f  overhead=%.1f%%",
                    defense, mean_acc, mean_f1, overhead * 100)

    # Print summary table
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  FEATURE SENSITIVITY ANALYSIS  task={task}")
    print(f"  (NOT live defense — feature-level approximation only)")
    print(sep)
    print(f"  {'Defense':<12} {'Accuracy':>10} {'Macro-F1':>10} {'~Overhead':>12} {'Acc drop':>10}")
    print(f"  {'-'*57}")
    baseline_acc = results["none"]["accuracy"]
    for d, r in results.items():
        drop = baseline_acc - r["accuracy"]
        print(f"  {d:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f} "
              f"{r['overhead_fraction']:>11.1%} {drop:>+10.3f}")
    print(sep)
    print("  ~Overhead = fraction of feature dims perturbed (NOT real byte/latency overhead).")
    print("  padding: size features × 1.5 (50% extra bytes; does NOT zero them).")
    print("  timing:  zeroes IAT/duration features (models schedule randomisation).")
    print("  Real overhead requires re-collecting traces with live defense middleware.")
    print(sep)

    out_dir = RESULTS_DIR / "defense"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"defense_{task}.json").write_text(json.dumps(results, indent=2))
    logger.info("Results saved → %s", out_dir / f"defense_{task}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature sensitivity analysis for proposed defenses (C4 pre-study)")
    p.add_argument("--task", choices=["workflow", "role", "parallelism", "topology"], default="workflow")
    p.add_argument("--noise-std", type=float, default=0.3,
                   help="Gaussian noise std for dummy/combined defense (default 0.3)")
    args = p.parse_args()
    evaluate(args.task, args.noise_std)
