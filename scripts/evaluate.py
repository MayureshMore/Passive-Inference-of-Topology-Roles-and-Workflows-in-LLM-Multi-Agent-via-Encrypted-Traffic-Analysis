#!/usr/bin/env python3
"""
Unified evaluation script (proposal §8.5, §10).

Loads processed features and trained models, then runs:
  - Closed-world: stratified k-fold CV for all tasks
  - Open-world: leave-one-class-out with threshold-based unknown rejection

Usage:
    python scripts/evaluate.py --mode closed_world
    python scripts/evaluate.py --mode open_world
    python scripts/evaluate.py --mode all         # default
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(
    task: str,
) -> tuple[np.ndarray, list[str], list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Returns (X, y, classes, burst_seqs, gap_seqs, groups).
    groups is a list of prompt_group hashes for GroupKFold leak-free CV.
    """
    label_file = PROCESSED_DIR / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(
            f"No labels.json in {PROCESSED_DIR} — run extract_features.py first."
        )
    labels_map: dict[str, dict] = json.loads(label_file.read_text())

    X_list, y_list, burst_list, gap_list, group_list = [], [], [], [], []
    for npz_path in sorted(PROCESSED_DIR.glob("*.npz")):
        run_id = npz_path.stem
        # Per-flow role NPZs have "__role__" in the name (30-dim flat vector).
        # Per-trace NPZs do not (103-dim flat vector).
        # Keep them strictly separate to avoid shape mismatches.
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
        burst_list.append(d["burst_sequence"])
        gap_list.append(d["gap_sequence"])
        group_list.append(info.get("prompt_group", run_id))

    if not X_list:
        raise ValueError(f"No samples for task={task} in {PROCESSED_DIR}")

    X = np.stack(X_list, axis=0)
    classes = sorted(set(y_list))
    logger.info("Loaded %d samples | task=%s | classes=%s", len(y_list), task, classes)
    return X, y_list, classes, burst_list, gap_list, group_list


# ── Closed-world evaluation ───────────────────────────────────────────────────

# All 17 per-system SCALAR indices in the 192-dim flat vector.
# Flat vector layout (192-dim):
#   [0:35]   pf_mean (35-dim)
#   [35:70]  pf_top1 — heaviest flow by total bytes (35-dim)
#   [70:105] pf_top2 — 2nd heaviest flow by total bytes (35-dim)
#   [105:192] per_system (87-dim):
#     [105:122] = 17 scalars, [122:157] = per_system pf_mean, [157:192] = pf_std
# Zeroing all 17 per-system scalars is the honest non-tautological topology test:
# n_flows, host counts, volumes, timing spreads, burst rates, AND the new
# traffic-distribution scalars (heaviest_flow_frac, flow_bytes_cv, etc.).
_ALL_SYSTEM_SCALAR_INDICES = list(range(105, 122))  # n_flows…mean_flow_response_ratio


def run_closed_world(tasks: list[str], ablate_structural: bool = False,
                     rf_only: bool = False) -> dict:
    from evaluation.closed_world import ClosedWorldEval

    all_results = {}
    out_dir = RESULTS_DIR / "closed_world"

    for task in tasks:
        try:
            X, y, classes, burst_seqs, gap_seqs, groups = load_dataset(task)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping task=%s: %s", task, exc)
            continue

        if ablate_structural and task == "topology":
            X = X.copy()
            X[:, _ALL_SYSTEM_SCALAR_INDICES] = 0.0
            logger.info("Ablation: zeroed ALL 17 per-system scalar features (indices 105–121)")

        n_splits = min(5, _min_class_count(y))
        if n_splits < 2:
            logger.warning("task=%s: fewer than 2 samples per class — skipping", task)
            continue

        evaluator = ClosedWorldEval(X, y, classes, task=task, n_splits=n_splits,
                                    groups=groups)

        logger.info("--- Closed-world RF  [%s] (group-safe CV) ---", task)
        rf_result = evaluator.run_rf(out_dir=out_dir)
        all_results[f"{task}/rf"] = rf_result

        # Transformer: uninformative at < ~1,000 traces; suppress by default.
        # Re-enable once you have 1,000–2,000+ traces per class.
        if rf_only:
            logger.info("Skipping Transformer [%s] (--rf-only; need 1k+ traces first)", task)
        elif _min_class_count(y) >= n_splits * 2:
            logger.info("--- Closed-world Transformer [%s] ---", task)
            try:
                tr_result = evaluator.run_transformer(
                    burst_sequences=burst_seqs,
                    gap_sequences=gap_seqs,
                    out_dir=out_dir,
                    n_epochs=20,
                )
                all_results[f"{task}/transformer"] = tr_result
            except Exception as exc:
                logger.warning("Transformer failed for task=%s: %s", task, exc)
        else:
            logger.info("Skipping Transformer [%s] (too few samples per fold)", task)

    return all_results


# ── Open-world evaluation ─────────────────────────────────────────────────────

def run_open_world(tasks: list[str]) -> dict:
    """
    Leave-one-class-out open-world evaluation.

    For each class C:
      - Train RF on all samples except class C
      - Evaluate: samples of known classes should be identified; samples of C
        should be rejected as "unknown" (confidence below threshold)
    """
    from models.random_forest import RFClassifier
    from evaluation.open_world import OpenWorldEval

    all_results = {}
    out_dir = RESULTS_DIR / "open_world"

    for task in tasks:
        try:
            X, y, classes, _, _, _ = load_dataset(task)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping open-world task=%s: %s", task, exc)
            continue

        if len(classes) < 3:
            logger.info("task=%s: only %d classes — skip open-world (need ≥ 3)", task, len(classes))
            continue

        task_results: list[dict] = []

        for held_out in classes:
            # Build train set: all classes except held_out
            mask_train = [label != held_out for label in y]
            X_train = X[np.array(mask_train)]
            y_train = [label for label, m in zip(y, mask_train) if m]

            # Build known test set (same classes as training)
            mask_known_test = [label != held_out for label in y]
            X_known_test = X[np.array(mask_known_test)]
            y_known_test = [label for label, m in zip(y, mask_known_test) if m]

            # Unknown test set: held-out class
            mask_unknown = [label == held_out for label in y]
            X_unknown = X[np.array(mask_unknown)]

            if len(X_train) < 4 or len(X_unknown) < 1:
                continue

            clf = RFClassifier(task=task)
            clf.fit(X_train, y_train)

            evaluator = OpenWorldEval(
                known_features=X_known_test,
                known_labels=y_known_test,
                unknown_features=X_unknown,
                task=task,
                threshold=0.6,
            )
            result = evaluator.run(model=clf, out_dir=out_dir)
            result["held_out_class"] = held_out
            task_results.append(result)

        if task_results:
            all_results[task] = task_results

    return all_results


# ── Result printer ────────────────────────────────────────────────────────────

def print_results(closed: dict, open_: dict) -> None:
    sep = "=" * 65

    if closed:
        print(f"\n{sep}")
        print("  CLOSED-WORLD RESULTS")
        print(sep)
        print(f"  {'Task / Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'vs Random':>12}")
        print(f"  {'-'*56}")

        for key, res in closed.items():
            task, model = key.rsplit("/", 1)
            if "cv" in res:
                # RF format: res["cv"]["accuracy"]["mean"]
                cv = res["cv"]
                acc     = cv.get("accuracy", {}).get("mean", 0)
                f1      = cv.get("f1_macro", {}).get("mean", 0)
                acc_std = cv.get("accuracy", {}).get("std", 0)
            else:
                # Transformer format: res["mean_accuracy"]
                acc     = res.get("mean_accuracy", 0)
                f1      = res.get("mean_macro_f1", 0)
                acc_std = res.get("std_accuracy", 0)

            n_classes = 4 if task == "workflow" else 3
            random_baseline = 1.0 / n_classes
            above = "✓" if acc > random_baseline else "✗"

            print(f"  {f'{task} / {model}':<30} {acc:.3f}±{acc_std:.3f}  {f1:.3f}       {above}")

    if open_:
        print(f"\n{sep}")
        print("  OPEN-WORLD RESULTS  (leave-one-class-out)")
        print(sep)
        print(f"  {'Task / Held-out':<35} {'Reject rate':>12} {'Avg Prec':>10}")
        print(f"  {'-'*58}")

        for task, runs in open_.items():
            for r in runs:
                held = r.get("held_out_class", "?")
                rej  = r.get("unknown_rejection_rate", 0)
                avg_prec = np.mean([
                    v["precision"] for v in r.get("per_class", {}).values()
                ]) if r.get("per_class") else 0
                print(f"  {f'{task} / -{held}':<35} {rej:.3f}        {avg_prec:.3f}")

    print(f"\n{sep}")
    print(f"  Full JSON results in data/results/")
    print(sep)


# ── Min class count helper ────────────────────────────────────────────────────

def _min_class_count(y: list[str]) -> int:
    from collections import Counter
    counts = Counter(y)
    return min(counts.values()) if counts else 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "closed_world").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "open_world").mkdir(parents=True, exist_ok=True)

    # C1 note: what's measured is topology-TYPE classification (star/chain/mesh),
    # not edge reconstruction. --ablate-structural zeroes all 17 per-system scalar
    # features (indices 105-121 in 192-dim vector) — counts, volumes, timing
    # spreads, burst rates, traffic-distribution scalars.
    # That is the honest non-tautological test.
    tasks = ["workflow", "topology", "role"]

    closed: dict = {}
    open_: dict = {}

    if args.mode in ("closed_world", "all"):
        closed = run_closed_world(tasks, ablate_structural=False, rf_only=args.rf_only)

    if args.mode in ("open_world", "all"):
        open_ = run_open_world(tasks)

    # C1 ablation: re-run topology with ALL 13 per-system scalar features zeroed.
    # This is the honest non-tautological test of whether timing/size signal survives.
    if args.ablate_structural and args.mode in ("closed_world", "all"):
        logger.info("=== C1 ABLATION: zeroing all 13 per-system scalar features (30–42) ===")
        ablation = run_closed_world(["topology"], ablate_structural=True, rf_only=args.rf_only)
        for k, v in ablation.items():
            closed[f"{k}__ablated"] = v

    print_results(closed, open_)

    # Save combined summary
    summary = {"closed_world": closed, "open_world": open_}
    summary_path = RESULTS_DIR / "summary.json"

    # JSON-serialise numpy types
    def _json_default(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    logger.info("Summary written → %s", summary_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A2A fingerprinting evaluation")
    p.add_argument("--mode", choices=["closed_world", "open_world", "all"],
                   default="all")
    p.add_argument("--ablate-structural", action="store_true",
                   help="Zero all 17 per-system scalar features (indices 105–121 in 192-dim vector) "
                        "for topology task — the honest test for non-tautological signal")
    p.add_argument("--rf-only", action="store_true",
                   help="Skip Transformer (uninformative at < ~1,000 traces per class)")
    main(p.parse_args())
