#!/usr/bin/env python3
"""
Ablation: retriever phase-count and role classification accuracy.

Answers the question: is the retriever's role signal real, or is it an
artefact of our 3-LLM-call design?

We compare role classification accuracy when the retriever runs:
  - 3-phase (decompose → per-term retrieve → synthesise)  [default, data/processed]
  - 1-phase (single direct-QA call)                        [data/processed_1phase]

If signal is genuine, accuracy should be similar at both settings.
If signal was manufactured by our design, 1-phase accuracy drops sharply.

Usage:
    # Step 1: collect 1-phase traces
    sudo venv/bin/python scripts/run_pilot.py --retriever-phases 1 \\
        --n 5 --out data/raw_1phase

    # Step 2: extract features from 1-phase traces
    python scripts/extract_features.py --raw data/raw_1phase \\
        --out data/processed_1phase --scapy

    # Step 3: run this script
    python scripts/ablation_retriever.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE3_DIR = Path("data/processed")
PHASE1_DIR = Path("data/processed_1phase")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_role_dataset(processed_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load per-flow role samples from a processed directory."""
    label_file = processed_dir / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(
            f"No labels.json in {processed_dir} — run extract_features.py first."
        )
    labels_map: dict[str, dict] = json.loads(label_file.read_text())

    X_list, y_list, group_list = [], [], []
    for npz_path in sorted(processed_dir.glob("*__role__*.npz")):
        run_id = npz_path.stem
        if run_id not in labels_map:
            continue
        info = labels_map[run_id]
        role = info.get("role")
        if role is None:
            continue
        d = np.load(npz_path, allow_pickle=False)
        X_list.append(d["flat"])
        y_list.append(role)
        group_list.append(info.get("prompt_group", run_id))

    if not X_list:
        raise ValueError(f"No role samples in {processed_dir}")

    X = np.stack(X_list, axis=0)
    logger.info("Loaded %d role samples from %s  classes=%s",
                len(y_list), processed_dir, sorted(set(y_list)))
    return X, y_list, group_list


# ── CV evaluation ─────────────────────────────────────────────────────────────

def _eval_role_accuracy(X: np.ndarray, y: list[str], groups: list[str],
                        label: str) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold
    from models.random_forest import RFClassifier
    from sklearn.metrics import accuracy_score, f1_score

    classes = sorted(set(y))
    n_classes = len(classes)
    n_splits = min(5, min(y.count(c) for c in classes) if n_classes else 1)
    if n_splits < 2:
        logger.warning("%s: too few samples per class for CV (min=%d)", label, n_splits)
        return {"label": label, "accuracy": float("nan"), "f1_macro": float("nan"),
                "n_samples": len(y), "n_classes": n_classes}

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr = X[tr_idx]; y_tr = [y[i] for i in tr_idx]
        X_te = X[te_idx]; y_te = [y[i] for i in te_idx]
        clf = RFClassifier(task="role")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        logger.info("  %s fold %d/%d  acc=%.3f  f1=%.3f", label, fold + 1, n_splits,
                    accs[-1], f1s[-1])

    return {
        "label": label,
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro": float(np.mean(f1s)),
        "n_samples": len(y),
        "n_classes": n_classes,
        "n_splits": n_splits,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results = {}

    # ── 3-phase results (default behaviour) ──────────────────────────────────
    if not PHASE3_DIR.exists():
        logger.error("No data/processed directory — run collect_traces.py and "
                     "extract_features.py first.")
        sys.exit(1)

    try:
        X3, y3, g3 = load_role_dataset(PHASE3_DIR)
        results["3-phase"] = _eval_role_accuracy(X3, y3, g3, "3-phase (default)")
    except ValueError as exc:
        logger.error("Failed to load 3-phase role data: %s", exc)
        sys.exit(1)

    # ── 1-phase results (ablation) ────────────────────────────────────────────
    if not PHASE1_DIR.exists():
        print()
        print("=" * 65)
        print("  RETRIEVER PHASE ABLATION — INSTRUCTIONS")
        print("=" * 65)
        print()
        print("  No 1-phase data found at data/processed_1phase/")
        print()
        print("  To collect it, run:")
        print()
        print("    sudo venv/bin/python scripts/run_pilot.py \\")
        print("        --retriever-phases 1 --n 5 --out data/raw_1phase")
        print()
        print("    python scripts/extract_features.py \\")
        print("        --raw data/raw_1phase --out data/processed_1phase --scapy")
        print()
        print("  Then re-run this script.")
        print()
        print(f"  3-phase role accuracy (baseline): "
              f"{results['3-phase']['accuracy']:.3f} ± "
              f"{results['3-phase']['accuracy_std']:.3f}")
        return

    try:
        X1, y1, g1 = load_role_dataset(PHASE1_DIR)
        results["1-phase"] = _eval_role_accuracy(X1, y1, g1, "1-phase (ablation)")
    except ValueError as exc:
        logger.error("Failed to load 1-phase role data: %s", exc)
        sys.exit(1)

    # ── Print comparison ──────────────────────────────────────────────────────
    sep = "=" * 65
    print()
    print(sep)
    print("  RETRIEVER PHASE ABLATION: ROLE CLASSIFICATION")
    print(sep)
    print(f"  {'Condition':<30} {'Accuracy':>10} {'Macro-F1':>10} {'N':>6}")
    print(f"  {'-'*58}")
    for key in ("3-phase", "1-phase"):
        r = results[key]
        acc_s = f"{r['accuracy']:.3f}±{r['accuracy_std']:.3f}" if not np.isnan(r['accuracy']) else "  N/A  "
        f1_s  = f"{r['f1_macro']:.3f}" if not np.isnan(r['f1_macro']) else " N/A"
        print(f"  {r['label']:<30} {acc_s:>10} {f1_s:>10} {r['n_samples']:>6}")

    print()
    if not np.isnan(results["3-phase"]["accuracy"]) and not np.isnan(results["1-phase"]["accuracy"]):
        delta = results["3-phase"]["accuracy"] - results["1-phase"]["accuracy"]
        print(f"  Δ accuracy (3-phase − 1-phase): {delta:+.3f}")
        if abs(delta) <= 0.10:
            print("  INTERPRETATION: signal survives phase reduction (|Δ| ≤ 0.10).")
            print("  Role fingerprint is NOT solely an artefact of 3-phase design.")
        else:
            print(f"  INTERPRETATION: large drop — phase count materially affects signal.")
            print(f"  Consider noting this dependence in the paper (§ agent design).")
    print(sep)


if __name__ == "__main__":
    main()
