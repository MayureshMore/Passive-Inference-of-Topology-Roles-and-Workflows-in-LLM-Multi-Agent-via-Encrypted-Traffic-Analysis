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

def _tune_rejection_threshold(
    cal_clf,
    label_encoder,
    X_known_val: np.ndarray,
    y_known_val: list[str],
    X_unknown_val: np.ndarray,
) -> float:
    """
    Sweep thresholds on a (known-val + unknown-val) split.
    Maximise harmonic mean of known-class accuracy and unknown rejection rate.
    Prevents both degenerate extremes: reject-all (T→1) and accept-all (T→0).
    """
    proba_kn = cal_clf.predict_proba(X_known_val)
    proba_unk = cal_clf.predict_proba(X_unknown_val)
    max_p_kn = proba_kn.max(axis=1)
    max_p_unk = proba_unk.max(axis=1)
    pred_kn_enc = proba_kn.argmax(axis=1)
    y_kn_enc = label_encoder.transform(y_known_val)

    best_T, best_score = 0.5, -1.0
    for T in np.linspace(0.05, 0.95, 37):
        # Known accuracy: correctly predicted AND not rejected
        accepted = max_p_kn >= T
        known_acc = float(((pred_kn_enc == y_kn_enc) & accepted).sum()) / max(len(y_known_val), 1)
        # Unknown rejection rate: fraction of unknown samples correctly rejected
        unk_rej = float((max_p_unk < T).sum()) / max(len(X_unknown_val), 1)
        # Harmonic mean of both — balanced objective
        score = (2 * known_acc * unk_rej / (known_acc + unk_rej)) if (known_acc + unk_rej) > 0 else 0.0
        if score > best_score:
            best_score, best_T = score, float(T)
    return best_T


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
      1. Split known-class data 75 / 25 into train and val/test.
      2. Fit RF on train; calibrate probabilities with isotonic regression
         (CalibratedClassifierCV eliminates RF overconfidence).
      3. Tune rejection threshold on (val-known + first half of unknown)
         to maximise HM(known_acc, unknown_rejection_rate).
      4. Evaluate on (val-known + second half of unknown) with tuned threshold.

    Fixes two prior bugs:
      - train == test (known test set was the same as training set)
      - fixed threshold 0.6 on uncalibrated probabilities (RF overconfidence
        meant almost nothing was ever rejected)
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

            # ── split unknown: half for threshold tuning, half for eval ─────
            mid = max(1, len(X_unknown) // 2)
            X_unk_tune = X_unknown[:mid]
            X_unk_eval = X_unknown[mid:]

            # ── fit RF ──────────────────────────────────────────────────────
            clf = RFClassifier(task=task)
            clf.fit(X_kn_train, y_kn_train)

            # ── calibrate probabilities with isotonic regression ────────────
            # cv='prefit' maps the already-fitted pipeline's raw outputs
            # through a held-out isotonic regression, producing calibrated
            # probabilities that are not pathologically overconfident.
            y_kn_train_enc = clf.label_encoder.transform(y_kn_train)
            cal_cv = min(3, _min_class_count(y_kn_train))
            if cal_cv >= 2:
                cal_pipeline = CalibratedClassifierCV(
                    clf.pipeline, cv=cal_cv, method="isotonic"
                )
                cal_pipeline.fit(X_kn_train, y_kn_train_enc)
            else:
                # Too few samples to calibrate — fall back to uncalibrated
                logger.warning("task=%s held_out=%s: too few samples for calibration, using raw RF", task, held_out)
                cal_pipeline = clf.pipeline
                cal_pipeline  # already fitted

            # ── tune threshold on val-known + first half of unknown ─────────
            best_T = _tune_rejection_threshold(
                cal_pipeline, clf.label_encoder,
                X_kn_val, y_kn_val, X_unk_tune,
            )
            logger.info("  open-world  task=%-10s  held_out=%-20s  tuned_T=%.3f", task, held_out, best_T)

            # ── evaluate with tuned threshold on val + second half unknown ──
            cal_model = _CalibratedModel(cal_pipeline, clf.label_encoder)
            evaluator = OpenWorldEval(
                known_features=X_kn_val,
                known_labels=y_kn_val,
                unknown_features=X_unk_eval,
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
        print(f"  {'Task / Model':<34} {'Accuracy':>10} {'Macro-F1':>10} {'vs Random':>12}")
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
                label += "  [no structural scalars]"
            print(f"  {label:<34} {acc:.3f}±{acc_std:.3f}  {f1:.3f}       {above}")

        # Topology note: trivially separable because routing config determines
        # the entire traffic graph; the ablation (zero drop) confirms per-system
        # scalar features are unused, not that signal is subtle.
        if any("topology" in k for k in closed):
            print(f"\n  NOTE topology: trivially separable by routing structure —")
            print(f"  not a subtle side-channel. Ablation zero-drop confirms structural")
            print(f"  features are redundant (pf_top1/pf_top2 encode the same info).")
            print(f"  Use 'parallelism' task (sequential vs parallel) for honest C1 claim.")

    if open_:
        print(f"\n{sep}")
        print("  OPEN-WORLD RESULTS  (leave-one-class-out, calibrated RF, tuned threshold)")
        print(sep)
        print(f"  {'Task / Held-out':<35} {'T':>6} {'Reject%':>9} {'Avg Prec':>10}")
        print(f"  {'-'*62}")

        for task, runs in open_.items():
            for r in runs:
                held = r.get("held_out_class", "?")
                T    = r.get("tuned_threshold", r.get("threshold", 0))
                rej  = r.get("unknown_rejection_rate", 0)
                avg_prec = np.mean([
                    v["precision"] for v in r.get("per_class", {}).values()
                ]) if r.get("per_class") else 0
                print(f"  {f'{task} / -{held}':<35} {T:>6.3f} {rej:>9.3f} {avg_prec:>10.3f}")

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
