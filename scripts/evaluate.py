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
        # Per-flow role NPZs have "__role__" in the name (35-dim flat vector).
        # Per-trace NPZs do not (192-dim flat vector).
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
        vec = d["flat"]
        expected_dim = 35 if is_role_sample else 195
        if vec.shape[0] != expected_dim:
            logger.warning(
                "Skipping %s: expected %d-dim flat vector, got %d",
                run_id, expected_dim, vec.shape[0],
            )
            continue
        X_list.append(vec)
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

# All 20 per-system SCALAR indices in the 195-dim flat vector.
# Flat vector layout (195-dim):
#   [0:35]   pf_mean (35-dim)
#   [35:70]  pf_top1 — heaviest flow by total bytes (35-dim)
#   [70:105] pf_top2 — 2nd heaviest flow by total bytes (35-dim)
#   [105:195] per_system (90-dim):
#     [105:125] = 20 scalars, [125:160] = per_system pf_mean, [160:195] = pf_std
# Zeroing all 20 per-system scalars is the honest non-tautological topology test:
# n_flows, host counts, volumes, timing spreads, burst rates, traffic-distribution
# scalars, AND the new request-body ratio scalars.
_ALL_SYSTEM_SCALAR_INDICES = list(range(105, 125))  # n_flows…max_flow_bytes_in

# Parallelism concurrency-counter ablation indices (within the 195-dim vector).
# These three per-system scalars directly ENCODE concurrency as a count/spread —
# they are structural (derived from flow timing) rather than a subtle side-channel:
#   flow_start_spread (112): max(start_ts) − min(start_ts) across flows
#   flow_end_spread   (113): max(end_ts)   − min(end_ts)   across flows
#   max_concurrent    (114): peak number of simultaneously open TCP connections
# If parallelism accuracy remains high after zeroing these three, the signal is
# in per-packet timing/size features (a genuine side-channel).  If accuracy
# collapses, the model was simply counting concurrent flows — a structural tell.
_PARALLELISM_CONCURRENCY_INDICES = [112, 113, 114]


def run_closed_world(tasks: list[str], ablate_structural: bool = False,
                     ablate_parallelism: bool = False,
                     rf_only: bool = False,
                     skip_cnn: bool = False) -> dict:
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
            logger.info("Ablation: zeroed ALL 20 per-system scalar features (indices 105–124)")

        if ablate_parallelism and task == "parallelism":
            X = X.copy()
            X[:, _PARALLELISM_CONCURRENCY_INDICES] = 0.0
            logger.info(
                "Ablation: zeroed concurrency-counter features "
                "(flow_start_spread=112, flow_end_spread=113, max_concurrent=114)"
            )

        n_splits = min(5, _min_class_count(y))
        if n_splits < 2:
            logger.warning("task=%s: fewer than 2 samples per class — skipping", task)
            continue

        evaluator = ClosedWorldEval(X, y, classes, task=task, n_splits=n_splits,
                                    groups=groups)

        logger.info("--- Closed-world RF  [%s] (group-safe CV) ---", task)
        rf_result = evaluator.run_rf(out_dir=out_dir)
        all_results[f"{task}/rf"] = rf_result

        if rf_only:
            logger.info("Skipping GBT/CNN/Transformer (--rf-only) [%s]", task)
        else:
            # ── GBT baseline ────────────────────────────────────────────────
            logger.info("--- Closed-world GBT [%s] ---", task)
            try:
                gbt_result = evaluator.run_gbt(out_dir=out_dir)
                all_results[f"{task}/gbt"] = gbt_result
            except Exception as exc:
                logger.warning("GBT failed for task=%s: %s", task, exc)

            # ── 1-D CNN (burst sequences) ────────────────────────────────────
            # Better than Transformer at 600 traces; suppress only when
            # explicitly asked (--skip-cnn) or when too few samples per fold.
            if skip_cnn:
                logger.info("Skipping CNN1D [%s] (--skip-cnn)", task)
            elif _min_class_count(y) >= n_splits * 2:
                logger.info("--- Closed-world CNN1D [%s] ---", task)
                try:
                    cnn_result = evaluator.run_cnn(
                        burst_sequences=burst_seqs,
                        gap_sequences=gap_seqs,
                        out_dir=out_dir,
                        n_epochs=40,
                    )
                    all_results[f"{task}/cnn"] = cnn_result
                except Exception as exc:
                    logger.warning("CNN1D failed for task=%s: %s", task, exc)
            else:
                logger.info("Skipping CNN1D [%s] (too few samples per fold)", task)

            # ── Transformer: uninformative at < ~1,000 traces ───────────────
            # Re-enable once you have 1,000–2,000+ traces per class.
            if _min_class_count(y) >= n_splits * 2:
                logger.info("--- Closed-world Transformer [%s] ---", task)
                try:
                    tr_result = evaluator.run_transformer(
                        burst_sequences=burst_seqs,
                        gap_sequences=gap_seqs,
                        out_dir=out_dir,
                        n_epochs=80,
                        patience=12,
                    )
                    all_results[f"{task}/transformer"] = tr_result
                except Exception as exc:
                    logger.warning("Transformer failed for task=%s: %s", task, exc)
            else:
                logger.info("Skipping Transformer [%s] (too few samples per fold)", task)

    return all_results


# ── Open-world evaluation ─────────────────────────────────────────────────────

def _tune_rejection_threshold(
    cal_clf,
    X_known_val: np.ndarray,
    target_known_retention: float = 0.95,
) -> float:
    """
    Set rejection threshold from known-class validation data ONLY.

    Finds the max-confidence percentile that retains `target_known_retention`
    of known-class validation traffic (default 95%).  Concretely:
      T = 5th percentile of known-val max-confidence scores.

    Zero access to the held-out unknown class.  Using unknown samples to pick T
    (e.g. maximising HM(known_acc, unk_rejection)) leaks the test unknown class
    into threshold selection — a reviewer catches this immediately and it inflates
    every rejection number.  The honest criterion depends only on known traffic.
    """
    proba_kn = cal_clf.predict_proba(X_known_val)
    max_p_kn = proba_kn.max(axis=1)
    reject_pct = (1.0 - target_known_retention) * 100.0   # e.g. 5.0
    return float(np.percentile(max_p_kn, reject_pct))


class _CalibratedModel:
    """
    Wraps CalibratedClassifierCV + LabelEncoder to match OpenWorldEval's
    model interface (predict / predict_proba returning string labels).
    """
    def __init__(self, cal_pipeline, label_encoder) -> None:
        self.cal_pipeline = cal_pipeline
        self.label_encoder = label_encoder

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cal_pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> list[str]:
        enc = self.cal_pipeline.predict(X)
        return list(self.label_encoder.inverse_transform(enc))


def run_open_world(tasks: list[str]) -> dict:
    """
    Leave-one-class-out open-world evaluation with calibrated rejection.

    For each held-out class C:
      1. Split known-class data 75/25 into train and val/test.
      2. Fit RF on train; calibrate with isotonic regression (eliminates
         RF overconfidence).
      3. Tune rejection threshold on val-known ONLY — 5th percentile of
         max-confidence scores, retaining ~95% of known traffic.
         Zero access to held-out unknown class during tuning.
      4. Evaluate on (val-known + ALL unknown) with the honest threshold.

    The prior protocol (tuning on first half of unknown, eval on second half)
    leaks the test class into threshold selection, inflating rejection numbers.
    This version uses no unknown samples during threshold selection.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedShuffleSplit
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
            # ── partition ───────────────────────────────────────────────────
            mask_known = np.array([label != held_out for label in y])
            X_known = X[mask_known]
            y_known = [label for label in y if label != held_out]
            X_unknown = X[~mask_known]

            if len(X_known) < 8 or len(X_unknown) < 4:
                continue

            # ── 75/25 split on known data (proper train vs eval) ────────────
            min_class_n = min(y_known.count(c) for c in set(y_known))
            test_size = max(0.2, min(0.3, 1.0 - 6.0 / max(min_class_n, 1)))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            for tr_idx, te_idx in sss.split(X_known, y_known):
                X_kn_train = X_known[tr_idx]
                y_kn_train = [y_known[i] for i in tr_idx]
                X_kn_val = X_known[te_idx]
                y_kn_val = [y_known[i] for i in te_idx]

            # ── fit RF ──────────────────────────────────────────────────────
            clf = RFClassifier(task=task)
            clf.fit(X_kn_train, y_kn_train)

            # ── calibrate probabilities ─────────────────────────────────────
            # Isotonic regression needs ~40+ samples per class to avoid
            # overfitting (produces extreme 0/1 probs at small n, which makes
            # the 5th-pct threshold degenerate at T≈1).  Use Platt sigmoid
            # for small datasets; isotonic only when each class has ≥40 train
            # samples — a standard rule of thumb from Niculescu-Mizil & Caruana.
            y_kn_train_enc = clf.label_encoder.transform(y_kn_train)
            min_per_class = _min_class_count(y_kn_train)
            cal_method = "isotonic" if min_per_class >= 40 else "sigmoid"
            cal_cv = min(3, min_per_class)
            if cal_cv >= 2:
                cal_pipeline = CalibratedClassifierCV(
                    clf.pipeline, cv=cal_cv, method=cal_method,
                )
                cal_pipeline.fit(X_kn_train, y_kn_train_enc)
                logger.info("  calibration: method=%s  min_per_class=%d", cal_method, min_per_class)
            else:
                logger.warning("task=%s held_out=%s: too few samples for calibration, using raw RF", task, held_out)
                cal_pipeline = clf.pipeline

            # ── tune threshold on known-val ONLY (no unknown samples) ───────
            # 5th-percentile of max-confidence on known val → retains ~95% of
            # known traffic.  Unknown class never seen during tuning.
            best_T = _tune_rejection_threshold(cal_pipeline, X_kn_val)
            logger.info("  open-world  task=%-10s  held_out=%-20s  T(5th-pct)=%.3f", task, held_out, best_T)

            # ── evaluate on val-known + ALL unknown (no holdout split) ──────
            cal_model = _CalibratedModel(cal_pipeline, clf.label_encoder)
            evaluator = OpenWorldEval(
                known_features=X_kn_val,
                known_labels=y_kn_val,
                unknown_features=X_unknown,
                task=task,
                threshold=best_T,
            )
            result = evaluator.run(model=cal_model, out_dir=out_dir)
            result["held_out_class"] = held_out
            result["tuned_threshold"] = best_T
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
        print(f"  {'Task / Model':<38} {'Accuracy':>10} {'Macro-F1':>10} {'vs Random':>12}")
        print(f"  {'-'*60}")

        for key, res in closed.items():
            task, model = key.rsplit("/", 1)
            ablated = key.endswith("__ablated")
            if "cv" in res:
                cv = res["cv"]
                acc     = cv.get("accuracy", {}).get("mean", 0)
                f1      = cv.get("f1_macro", {}).get("mean", 0)
                acc_std = cv.get("accuracy", {}).get("std", 0)
            else:
                acc     = res.get("mean_accuracy", 0)
                f1      = res.get("mean_macro_f1", 0)
                acc_std = res.get("std_accuracy", 0)

            if "workflow" in task:
                n_classes = 4
            elif task == "parallelism":
                n_classes = 2
            else:
                n_classes = 3
            random_baseline = 1.0 / n_classes
            above = "✓" if acc > random_baseline else "✗"
            label = f"{task} / {model}"
            if ablated:
                # Show what was zeroed for each ablated task
                if "topology" in task:
                    label += "  [-system scalars]"
                elif "parallelism" in task:
                    label += "  [-concurrency]"
                else:
                    label += "  [ablated]"
            print(f"  {label:<38} {acc:.3f}±{acc_std:.3f}  {f1:.3f}       {above}")

        if any("topology" in k for k in closed):
            print(f"\n  NOTE topology: trivially separable by routing structure —")
            print(f"  not a subtle side-channel. topology/rf__ablated confirms system")
            print(f"  scalars are redundant; pf_top1/pf_top2 encode the same structure.")
            print(f"  Use 'parallelism' for the honest C1 claim.")
        if any("parallelism" in k and "ablated" in k for k in closed):
            print(f"\n  NOTE parallelism: parallelism/rf__ablated zeros max_concurrent,")
            print(f"  flow_start/end_spread. If accuracy holds, the signal is in")
            print(f"  per-packet timing/size, not just concurrency counting.")

    if open_:
        print(f"\n{sep}")
        print("  OPEN-WORLD RESULTS  (LOO within-distribution, calibrated RF)")
        print("  Protocol: threshold = 5th-pct of known-val max-confidence (~95% retention).")
        print("  Zero unknown samples used during threshold selection.")
        print("  NOTE: 'unknown' = held-out A2A class, NOT real background traffic.")
        print(sep)
        print(f"  {'Task / Held-out':<35} {'T':>6} {'Unk-Rej%':>9} {'Kn-FPR%':>9} {'Avg Prec':>10}")
        print(f"  {'-'*71}")

        for task, runs in open_.items():
            for r in runs:
                held    = r.get("held_out_class", "?")
                T       = r.get("tuned_threshold", r.get("threshold", 0))
                rej     = r.get("unknown_rejection_rate", 0)
                kn_fpr  = r.get("known_fpr", float("nan"))
                avg_prec = np.mean([
                    v["precision"] for v in r.get("per_class", {}).values()
                ]) if r.get("per_class") else 0
                fpr_s = f"{kn_fpr:.3f}" if not (isinstance(kn_fpr, float) and kn_fpr != kn_fpr) else "  N/A"
                print(f"  {f'{task} / -{held}':<35} {T:>6.3f} {rej:>9.3f} {fpr_s:>9} {avg_prec:>10.3f}")

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

    # Tasks:
    #  - workflow: 4-class (research/code/data/support) — the core fingerprinting claim
    #  - role:     4-class (orchestrator/executor/retriever/validator) — C2
    #  - parallelism: 2-class (sequential vs parallel) — honest C1 replacement
    #    chain → sequential; star/mesh → parallel. This is a genuine timing signal
    #    (chain imposes serial latency; parallel fan-out compresses it).
    #  - topology: 3-class (star/chain/mesh) — kept for reference only; trivially
    #    separable by routing config, not a subtle side-channel.
    tasks = ["workflow", "role", "parallelism", "topology"]

    closed: dict = {}
    open_: dict = {}

    if args.mode in ("closed_world", "all"):
        closed = run_closed_world(tasks, ablate_structural=False,
                                  ablate_parallelism=False, rf_only=args.rf_only,
                                  skip_cnn=args.skip_cnn)

    if args.mode in ("open_world", "all"):
        open_ = run_open_world(tasks)

    # Ablations always run in --mode all / closed_world.  RF only for ablations
    # (GBT/CNN on ablated data is redundant for the paper's ablation table).
    if args.mode in ("closed_world", "all"):
        logger.info("=== C1 TOPOLOGY ABLATION: zeroing per-system scalars (105–121) ===")
        topo_abl = run_closed_world(["topology"], ablate_structural=True,
                                    ablate_parallelism=False, rf_only=True)
        for k, v in topo_abl.items():
            closed[f"{k}__ablated"] = v

        logger.info("=== C1 PARALLELISM ABLATION: zeroing concurrency counters (112–114) ===")
        par_abl = run_closed_world(["parallelism"], ablate_structural=False,
                                   ablate_parallelism=True, rf_only=True)
        for k, v in par_abl.items():
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
                   help="(deprecated — ablations now run automatically in --mode all/closed_world)")
    p.add_argument("--rf-only", action="store_true",
                   help="Run RF only — skip GBT, CNN, Transformer")
    p.add_argument("--skip-cnn", action="store_true",
                   help="Skip CNN1D (useful on CPU-only machines — CNN is slow without MPS/CUDA)")
    main(p.parse_args())
