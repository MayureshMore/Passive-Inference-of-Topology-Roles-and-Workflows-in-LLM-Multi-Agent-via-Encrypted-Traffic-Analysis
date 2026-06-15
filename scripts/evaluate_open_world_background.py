#!/usr/bin/env python3
"""
Open-world evaluation using real background (non-A2A) traffic as the unknown class.

This is a strictly harder and more realistic test than the LOO open-world in
evaluate.py, which holds out one A2A workflow class as "unknown."  Here the
unknown set is genuine non-agent traffic: web browsing, file downloads, REST
polling, bare JSON-RPC calls, and direct LLM API calls.

Protocol
--------
1. Load all agentic features from data/processed/ (workflow task, 195-dim).
2. Split 75 / 25 into train and val (stratified by workflow class).
3. Fit RF on train; calibrate with isotonic regression.
4. Tune rejection threshold T on val-known ONLY:
     T = 5th percentile of max-confidence scores on known-val
     → retains ~95% of known traffic with zero access to unknowns.
5. Load background features from data/processed_background/.
6. Evaluate on (val-known + ALL background):
     - Per-category rejection rate (fraction of bg traces rejected at threshold T)
     - Known-class FPR (fraction of known val wrongly rejected)
7. Report precision at a realistic base rate (1% A2A vs 99% background).

Usage
-----
    python scripts/evaluate_open_world_background.py
    python scripts/evaluate_open_world_background.py --processed data/processed \
        --bg-processed data/processed_background
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

PROCESSED_DIR    = Path("data/processed")
BG_PROCESSED_DIR = Path("data/processed_background")
RESULTS_DIR      = Path("data/results")

SOFT_CATS = {"bg_web_browse", "bg_file_download", "bg_api_rest"}
HARD_CATS = {"bg_jsonrpc", "bg_multi_rest", "bg_llm_direct"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_agentic(processed_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load 195-dim per-trace flat vectors for workflow classification."""
    label_file = processed_dir / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"No labels.json in {processed_dir}")
    labels_map = json.loads(label_file.read_text())

    X_list, y_list = [], []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        run_id = npz_path.stem
        if "__role__" in run_id:
            continue
        if run_id not in labels_map:
            continue
        wf = labels_map[run_id].get("workflow")
        if wf is None:
            continue
        d = np.load(npz_path, allow_pickle=False)
        vec = d["flat"]
        if vec.shape[0] != 195:
            continue
        X_list.append(vec)
        y_list.append(wf)

    if not X_list:
        raise ValueError(f"No workflow samples in {processed_dir}")
    logger.info("Agentic: %d samples, classes=%s", len(y_list), sorted(set(y_list)))
    return np.stack(X_list), y_list


def load_background(bg_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load 195-dim flat vectors for background traces; return (X, categories)."""
    label_file = bg_dir / "labels_background.json"
    if not label_file.exists():
        raise FileNotFoundError(
            f"No labels_background.json in {bg_dir} — "
            "run scripts/collect_background.py first."
        )
    labels_map = json.loads(label_file.read_text())

    X_list, cat_list = [], []
    for npz_path in sorted(bg_dir.glob("*.npz")):
        run_id = npz_path.stem
        if run_id not in labels_map:
            continue
        cat = labels_map[run_id].get("category", "unknown")
        d = np.load(npz_path, allow_pickle=False)
        vec = d["flat"]
        if vec.shape[0] != 195:
            logger.warning("Skipping %s: dim=%d (expected 195)", run_id, vec.shape[0])
            continue
        X_list.append(vec)
        cat_list.append(cat)

    if not X_list:
        raise ValueError(f"No valid background samples in {bg_dir}")
    logger.info("Background: %d samples across categories=%s",
                len(cat_list), sorted(set(cat_list)))
    return np.stack(X_list), cat_list


# ── Threshold tuning (known-val only) ────────────────────────────────────────

def tune_threshold(cal_clf, X_known_val: np.ndarray, retention: float = 0.95) -> float:
    """5th-percentile of max-confidence on known-val → retains ~95% of known traffic."""
    proba = cal_clf.predict_proba(X_known_val)
    return float(np.percentile(proba.max(axis=1), (1.0 - retention) * 100.0))


# ── Main evaluation ───────────────────────────────────────────────────────────

def run(
    processed_dir: Path,
    bg_dir: Path,
    results_dir: Path,
) -> None:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedShuffleSplit
    from models.random_forest import RFClassifier

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X_ag, y_ag = load_agentic(processed_dir)
    X_bg, cats_bg = load_background(bg_dir)

    # ── 75/25 split on agentic known data ────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tr_idx, te_idx in sss.split(X_ag, y_ag):
        X_train  = X_ag[tr_idx]
        y_train  = [y_ag[i] for i in tr_idx]
        X_kn_val = X_ag[te_idx]
        y_kn_val = [y_ag[i] for i in te_idx]

    logger.info("Train: %d samples, Val (known): %d, Background: %d",
                len(y_train), len(y_kn_val), len(X_bg))

    # ── Fit RF ────────────────────────────────────────────────────────────────
    clf = RFClassifier(task="workflow")
    clf.fit(X_train, y_train)

    # ── Calibrate ────────────────────────────────────────────────────────────
    from collections import Counter
    min_per_class = min(Counter(y_train).values())
    cal_method = "isotonic" if min_per_class >= 40 else "sigmoid"
    cal_cv = min(3, min_per_class)
    y_train_enc = clf.label_encoder.transform(y_train)
    if cal_cv >= 2:
        cal_clf = CalibratedClassifierCV(clf.pipeline, cv=cal_cv, method=cal_method)
        cal_clf.fit(X_train, y_train_enc)
        logger.info("Calibration: method=%s, min_per_class=%d", cal_method, min_per_class)
    else:
        logger.warning("Too few samples for calibration — using raw RF probabilities")
        cal_clf = clf.pipeline

    # ── Tune threshold on known-val ONLY ─────────────────────────────────────
    T = tune_threshold(cal_clf, X_kn_val)
    logger.info("Rejection threshold T=%.4f (5th-pct of known-val max-confidence)", T)
    if T >= 0.999:
        logger.warning(
            "T≈1.000 — testbed too deterministic; background evaluation may still be informative "
            "since background traffic is genuinely different from A2A."
        )

    # ── Evaluate known-val ───────────────────────────────────────────────────
    proba_kn = cal_clf.predict_proba(X_kn_val)
    max_p_kn = proba_kn.max(axis=1)
    kn_rejected = (max_p_kn < T).mean()
    kn_retained = 1.0 - kn_rejected

    # ── Evaluate background by category ──────────────────────────────────────
    proba_bg = cal_clf.predict_proba(X_bg)
    max_p_bg = proba_bg.max(axis=1)
    rejected_bg = max_p_bg < T

    all_cats = sorted(set(cats_bg))
    per_cat: dict[str, dict] = {}
    for cat in all_cats:
        mask = np.array([c == cat for c in cats_bg])
        n_cat = mask.sum()
        if n_cat == 0:
            continue
        rej_rate = rejected_bg[mask].mean()
        per_cat[cat] = {
            "n": int(n_cat),
            "rejection_rate": float(rej_rate),
            "type": "soft" if cat in SOFT_CATS else "hard",
        }

    overall_rej = rejected_bg.mean()

    # ── Realistic base-rate precision ────────────────────────────────────────
    # At 1% A2A traffic (1 A2A : 99 background), the FPR on background
    # inflates false positives substantially.
    base_rate = 0.01
    # Known-val TPR at threshold T (fraction correctly identified as A2A)
    tpr = kn_retained
    # Background FPR (fraction of background that passes as A2A)
    fpr_bg = 1.0 - overall_rej
    # Bayes-corrected precision
    if tpr * base_rate + fpr_bg * (1.0 - base_rate) > 0:
        precision_realistic = (tpr * base_rate) / (
            tpr * base_rate + fpr_bg * (1.0 - base_rate)
        )
    else:
        precision_realistic = 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("  OPEN-WORLD (REAL BACKGROUND) — RF calibrated, T tuned on known-val only")
    print(f"  T = {T:.4f}   Known-val retention = {kn_retained:.1%}   FPR(known) = {kn_rejected:.1%}")
    print(sep)
    print(f"  {'Category':<25} {'Type':>6} {'N':>5} {'Rejected%':>12} {'Passed%':>10}")
    print(f"  {'-'*60}")
    for cat, info in sorted(per_cat.items(), key=lambda x: (x[1]["type"], x[0])):
        rej_pct  = info["rejection_rate"] * 100
        pass_pct = 100 - rej_pct
        print(f"  {cat:<25} {info['type']:>6} {info['n']:>5} {rej_pct:>11.1f}% {pass_pct:>9.1f}%")
    print(f"  {'-'*60}")
    n_total = len(X_bg)
    print(f"  {'ALL BACKGROUND':<25} {'mixed':>6} {n_total:>5} {overall_rej*100:>11.1f}% {(1-overall_rej)*100:>9.1f}%")
    print()
    print(f"  Realistic-base-rate precision (1% A2A vs 99% background):")
    print(f"    TPR(A2A)       = {tpr:.3f}")
    print(f"    FPR(background)= {fpr_bg:.3f}")
    print(f"    Precision      = {precision_realistic:.3f}")
    print()
    print(f"  Interpretation:")
    print(f"    Soft negatives (web/download/REST) are expected to reject at ≥80%.")
    print(f"    Hard negatives (JSON-RPC/multi-REST/LLM-direct) are the true test:")
    print(f"    rejection < 60% on hard negatives means A2A structural signal is weak;")
    print(f"    rejection > 80% means the timing + flow-count signal is robust.")
    print(sep)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "threshold": T,
        "known_val_retention": float(kn_retained),
        "known_val_fpr": float(kn_rejected),
        "overall_background_rejection": float(overall_rej),
        "precision_at_1pct_base_rate": float(precision_realistic),
        "per_category": per_cat,
    }
    out_path = results_dir / "open_world_background.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written → %s", out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    try:
        run(
            processed_dir=Path(args.processed),
            bg_dir=Path(args.bg_processed),
            results_dir=RESULTS_DIR,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Open-world evaluation with real background traffic")
    p.add_argument("--processed",    default="data/processed",
                   help="Agentic feature directory (default: data/processed)")
    p.add_argument("--bg-processed", default="data/processed_background",
                   help="Background feature directory (default: data/processed_background)")
    main(p.parse_args())
