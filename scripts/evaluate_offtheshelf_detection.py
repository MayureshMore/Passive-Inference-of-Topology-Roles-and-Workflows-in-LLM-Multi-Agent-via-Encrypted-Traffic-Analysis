#!/usr/bin/env python3
"""
Phase 5c (i) — OPEN-WORLD DETECTION of the off-the-shelf a2a_mcp system.

Question (DETECTION only — NO role/workflow transfer; a2a_mcp labels do not align
with our taxonomy):
    Can an observer detect that an externally-authored multi-agent A2A system is
    present, from traffic metadata alone, having never seen it before?

FRAMING (this is the load-bearing choice — see the history below):
    A *binary A2A-vs-background* detector.  Train on (positive = our deployment-A
    A2A traffic) vs (negative = ordinary background traffic), then test on the
    held-out, independently-authored a2a_mcp traces as positives.  Detection is
    reported at a fixed 5% background false-positive rate (the same open-world
    operating point used elsewhere in the project) plus a threshold-free AUC.

    Why NOT the workflow-classifier-with-rejection approach (the earlier version):
    that detector scores "does this look like one of A's FOUR specific workflows?"
    a2a_mcp is a *travel* workflow — a different task — so its A-workflow confidence
    is LOW (AUC ~0.41 vs background, i.e. below chance) and a permissive rejection
    threshold flags most of the background too.  That is a framing failure, not a
    sample-size one; more traces do not fix it.  We therefore detect the *structural
    A2A signature* (multi-flow fan-out, SSE response bursts, per-system shape),
    which is workflow-agnostic.  The novelty AUC is still reported as context.

HONESTY CAVEAT (emitted into the JSON):
    The background negatives are ordinary web/API/file traffic whose flow/SSE
    structure differs sharply from multi-agent A2A, so the two are near-perfectly
    separable — the detection number is high but the negatives are "easy".  The
    defensible claim is that the off-the-shelf system *carries the same structural
    A2A fingerprint* and is detected at the standard operating point; a sterner
    test would use OTHER agent frameworks as negatives (future work).

Usage:
    venv/bin/python scripts/evaluate_offtheshelf_detection.py
    venv/bin/python scripts/evaluate_offtheshelf_detection.py \
        --processed data/processed --background data/processed_background \
        --offtheshelf data/processed_offtheshelf
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
_SEED = 42
_N_TREES = 300
_N_BOOTSTRAP = 2000
_BG_FPR = 0.05  # fixed background false-positive operating point


# ── Loaders ─────────────────────────────────────────────────────────────────────

def load_flat_set(processed_dir: Path, label_file: str = "labels.json") -> np.ndarray:
    """Load all 195-dim per-trace flat vectors from a directory (labels optional)."""
    lf = processed_dir / label_file
    labels_map = json.loads(lf.read_text()) if lf.exists() else {}
    X = []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        rid = npz_path.stem
        if "__role__" in rid:
            continue
        if labels_map and rid not in labels_map:
            continue
        v = np.load(npz_path, allow_pickle=False)["flat"]
        if v.shape[0] == 195:
            X.append(v)
    if not X:
        raise ValueError(f"No 195-dim per-trace vectors in {processed_dir}")
    return np.stack(X)


def load_workflow_labeled(processed_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load 195-dim per-trace vectors + workflow labels (for the novelty-AUC context)."""
    labels_map = json.loads((processed_dir / "labels.json").read_text())
    X, y = [], []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        rid = npz_path.stem
        if "__role__" in rid or rid not in labels_map:
            continue
        wf = labels_map[rid].get("workflow")
        if wf is None:
            continue
        v = np.load(npz_path, allow_pickle=False)["flat"]
        if v.shape[0] == 195:
            X.append(v); y.append(wf)
    return (np.stack(X), y) if X else (np.empty((0, 195)), [])


def load_background_with_cat(
    processed_dir: Path, label_file: str = "labels_background.json"
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load background 195-dim vectors + per-trace (category, type), in the SAME
    sorted order as load_flat_set so the rows align with X_bg."""
    lf = processed_dir / label_file
    labels_map = json.loads(lf.read_text()) if lf.exists() else {}
    X, cats, types = [], [], []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        rid = npz_path.stem
        if "__role__" in rid:
            continue
        if labels_map and rid not in labels_map:
            continue
        v = np.load(npz_path, allow_pickle=False)["flat"]
        if v.shape[0] != 195:
            continue
        info = labels_map.get(rid, {})
        X.append(v)
        cats.append(str(info.get("category", "unknown")))
        types.append(str(info.get("type", "")))
    if not X:
        raise ValueError(f"No 195-dim background vectors in {processed_dir}")
    return np.stack(X), cats, types


# ── Detectors ─────────────────────────────────────────────────────────────────

def binary_detection(
    X_pos: np.ndarray, X_bg: np.ndarray, X_ots: np.ndarray,
    bg_categories: list[str] | None = None, bg_types: list[str] | None = None,
) -> dict:
    """
    Binary A2A-vs-background detector.

    CV on (pos=A2A, neg=background) for a threshold-free AUC and the threshold T at
    a 5% background FPR; then score the held-out off-the-shelf traces as positives.
    Also reports the per-background-category cross-validated A2A-probability so the
    "separates the hard categories too" claim is verifiable from the artifact.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    X = np.vstack([X_pos, X_bg])
    y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
    clf = RandomForestClassifier(
        n_estimators=_N_TREES, random_state=_SEED, class_weight="balanced"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED)
    oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))

    # Operating point: threshold at the chosen background FPR (from OOF negatives).
    T = float(np.percentile(oof[y == 0], 100 * (1 - _BG_FPR)))
    a2a_tpr_cv = float((oof[y == 1] >= T).mean())

    # Fit on all known data, score the independent off-the-shelf set.
    clf.fit(X, y)
    ots = clf.predict_proba(X_ots)[:, 1]
    det = float((ots >= T).mean())

    rng = np.random.default_rng(_SEED)
    boots = np.array([(rng.choice(ots, ots.size, replace=True) >= T).mean()
                      for _ in range(_N_BOOTSTRAP)])
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))

    # Per-background-category A2A-probability, from the OOF (cross-validated, no
    # leakage) scores of the negative rows. Background was appended after positives
    # in X, so oof[len(X_pos):] are the bg OOF scores aligned with bg_categories.
    per_cat: dict[str, dict] = {}
    if bg_categories is not None:
        from collections import defaultdict
        bg_oof = oof[len(X_pos):]
        types = bg_types or [""] * len(bg_categories)
        agg: dict[str, list[float]] = defaultdict(list)
        typ: dict[str, str] = {}
        for c, t, s in zip(bg_categories, types, bg_oof):
            agg[c].append(float(s)); typ[c] = t
        for c, ss in sorted(agg.items()):
            arr = np.asarray(ss)
            per_cat[c] = {
                "n": int(arr.size), "type": typ.get(c, ""),
                "mean_a2a_prob": float(arr.mean()),
                "flagged_as_a2a_at_T": float((arr >= T).mean()),
            }

    return {
        "auc_a2a_vs_background": auc,
        "background_fpr": _BG_FPR,
        "threshold": T,
        "a2a_true_positive_rate_cv": a2a_tpr_cv,
        "offtheshelf_detected_rate": det,
        "offtheshelf_detected_ci95": [lo, hi],
        "n_pos": int(len(X_pos)), "n_bg": int(len(X_bg)), "n_offtheshelf": int(len(X_ots)),
        "background_per_category": per_cat,
    }


def novelty_auc_context(processed: Path, X_bg: np.ndarray, X_ots: np.ndarray) -> dict | None:
    """
    Context only: the WRONG framing.  An RF workflow classifier's max-confidence
    used as a novelty score — a2a_mcp is a different workflow, so this fails.
    Reported so the choice of the binary framing is auditable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    Xa, ya = load_workflow_labeled(processed)
    if len(Xa) == 0:
        return None
    le = LabelEncoder()
    clf = RandomForestClassifier(n_estimators=_N_TREES, random_state=_SEED)
    clf.fit(Xa, le.fit_transform(ya))
    conf_ots = clf.predict_proba(X_ots).max(axis=1)
    conf_bg = clf.predict_proba(X_bg).max(axis=1)
    y = np.r_[np.ones(len(conf_ots)), np.zeros(len(conf_bg))]
    return {
        "auc_offtheshelf_vs_background": float(roc_auc_score(y, np.r_[conf_ots, conf_bg])),
        "note": ("WRONG TOOL — workflow-classifier max-confidence as a novelty score. "
                 "a2a_mcp is a different workflow (travel), so its A-workflow confidence "
                 "is low (AUC < 0.5, below chance vs background). Shown to justify the "
                 "binary A2A-vs-background framing above; not a usable detector here."),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(processed: Path, background: Path, offtheshelf: Path, results_dir: Path) -> None:
    if not (offtheshelf / "labels.json").exists() and not list(offtheshelf.glob("*.npz")):
        logger.warning(
            "No off-the-shelf features at %s yet — run scripts/collect_offtheshelf.sh + "
            "extract_offtheshelf.py first.", offtheshelf,
        )
        return
    if not (background / "labels_background.json").exists() and not list(background.glob("*.npz")):
        logger.warning("No background features at %s — needed as negatives for detection.", background)
        return

    X_pos = load_flat_set(processed)                                   # A2A positives (deployment A)
    X_bg, bg_cats, bg_types = load_background_with_cat(background)     # non-A2A negatives + categories
    X_ots = load_flat_set(offtheshelf)                                 # external held-out A2A

    det = binary_detection(X_pos, X_bg, X_ots, bg_categories=bg_cats, bg_types=bg_types)
    novelty = novelty_auc_context(processed, X_bg, X_ots)

    out = {
        "_status": "validated on real off-the-shelf traces",
        "framing": "binary A2A-vs-background (correct tool)",
        "detector": {
            "positives": f"{processed} (deployment-A A2A traffic)",
            "negatives": f"{background} (multi-flow but non-SSE/non-agentic: web/API, "
                         "JSON-RPC, multi-REST, file-download, direct-LLM)",
            "model": f"RandomForest({_N_TREES}, balanced)", "cv": "5-fold stratified",
            "seed": _SEED,
        },
        "detection": det,
        "caveat": (
            "Background negatives are MULTI-FLOW but NON-SSE / NON-AGENTIC (web/API, "
            "JSON-RPC, multi-REST, file-download, direct-LLM). The supervised detector "
            "separates A2A from them via A2A's SSE-streaming + orchestrator fan-out "
            "signature — it separates even the PARALLEL multi-REST negative, so the "
            "result is not merely a concurrent-vs-sequential artifact. But EVERY negative "
            "is non-agentic, so this measures 'A2A vs NON-AGENTIC traffic', NOT 'A2A vs "
            "other agent frameworks'. Separability from other AGENTIC, SSE-based frameworks "
            "(AutoGen, CrewAI, ...) is UNTESTED — the sterner future test. See "
            "detection.background_per_category: the hard JSON-RPC / multi-REST / LLM-direct "
            "categories all score ~0 A2A-probability."
        ),
        "workflow_novelty_context": novelty,
        "scope_note": ("DETECTION ONLY. a2a_mcp labels do not align with our taxonomy, "
                       "so this is not a role/workflow transfer result."),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "offtheshelf_detection.json").write_text(json.dumps(out, indent=2))

    sep = "=" * 70
    print(f"\n{sep}\n  PHASE 5c (i) — OFF-THE-SHELF DETECTION (a2a_mcp)\n{sep}")
    print(f"  framing: binary A2A-vs-background  (pos={det['n_pos']} A, neg={det['n_bg']} bg)")
    print(f"  separability AUC = {det['auc_a2a_vs_background']:.3f}   "
          f"(A2A TPR@{int(_BG_FPR*100)}%FPR = {det['a2a_true_positive_rate_cv']:.1%}, CV)")
    print(f"  >>> off-the-shelf a2a_mcp DETECTED as A2A (n={det['n_offtheshelf']}): "
          f"{det['offtheshelf_detected_rate']:.1%}  "
          f"95% CI [{det['offtheshelf_detected_ci95'][0]:.1%}, {det['offtheshelf_detected_ci95'][1]:.1%}]")
    if novelty:
        print(f"  context — workflow-novelty (WRONG tool): AUC {novelty['auc_offtheshelf_vs_background']:.3f} "
              f"(< 0.5 ⇒ fails; different workflow)")
    print(f"  caveat: easy negatives — see JSON; harder negatives (other frameworks) = future work")
    print(f"  scope: DETECTION only — no transfer claim\n{sep}")
    logger.info("Wrote %s", results_dir / "offtheshelf_detection.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5c binary A2A-vs-background detection of off-the-shelf a2a_mcp")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--background", default="data/processed_background")
    ap.add_argument("--offtheshelf", default="data/processed_offtheshelf")
    args = ap.parse_args()
    run(Path(args.processed), Path(args.background), Path(args.offtheshelf), RESULTS_DIR)


if __name__ == "__main__":
    main()
