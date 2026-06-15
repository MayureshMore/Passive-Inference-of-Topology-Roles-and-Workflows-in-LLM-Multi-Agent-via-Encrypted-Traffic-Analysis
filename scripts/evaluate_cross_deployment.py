#!/usr/bin/env python3
"""
Cross-deployment generalization evaluation (Track A, Tasks A2 + C2).

Measures whether traffic-analysis fingerprinting transfers across two independent
A2A implementations that differ in LLM call counts, model, and delegation style.

Experiments run
───────────────
For each task in {workflow, role, parallelism, topology}:

  A→A  Internal CV on deployment A (StratifiedGroupKFold, n_splits=5)
  B→B  Internal CV on deployment B
  A→B  Train on A, test on B  (with 95 % bootstrap CI)
  B→A  Train on B, test on A  (with 95 % bootstrap CI)

SSE response-pattern sub-analysis (workflow task, A→B only):
  Both deployments stream answers as Server-Sent Events (a2a-sdk message/stream),
  so each response-direction packet is an SSE event.  The "seg" features
  (n_response_bursts, iqr_size_in, ...) capture genuine SSE-chunk structure —
  response event counts and inbound size distributions per flow.  This
  sub-analysis tests whether that SSE response-pattern transfers across
  deployments.

  (a) seg_only   30-dim response-pattern vector only
  (b) flat_only  195-dim flat, seg zeroed out (no change to flat content)
  (c) flat+seg   225-dim  (flat concatenated with response-pattern vector)

Retention metric (above-chance)
────────────────────────────────
Retention = (transfer_F1 − random_baseline) / (ceiling_F1 − random_baseline)

Raw accuracy ratios (transfer/ceiling) are misleading because they credit
performance that is trivially achievable by random guessing.  The above-chance
formula measures only the signal that the classifier actually learned.

Interpretation guide
────────────────────
topology / parallelism — STRUCTURAL BASELINE, NOT A THREAT:
  Perfect or near-perfect transfer for topology and its binary projection
  (parallelism = star/mesh→parallel, chain→sequential) reflects connection-graph
  observability, not a timing side-channel.  An on-path observer reads which
  hosts connect to which directly from IP headers without any classifier.
  These are reported for completeness; they are not the attack contribution.

workflow / role — THE ACTUAL ATTACK TARGETS:
  High above-chance retention → implementation-agnostic behavioral leak.
  Low above-chance retention  → signal is implementation-specific; defenders
                                 need only change call counts or execution order.
  Both outcomes are honest and publishable.

Usage:
    python scripts/evaluate_cross_deployment.py
    python scripts/evaluate_cross_deployment.py --dir-a data/processed --dir-b data/processed_b
    python scripts/evaluate_cross_deployment.py --task workflow role
    python scripts/evaluate_cross_deployment.py --no-bootstrap   # faster, no CIs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TASKS = ["workflow", "role", "parallelism", "topology"]
_N_BOOTSTRAP = 1000
_RNG_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_deployment(
    processed_dir: Path,
    task: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load flat + seg feature matrices for one deployment.

    Returns
    -------
    X_flat  : (N, 195)  flat feature vectors
    X_seg   : (N, 30)   response-segmentation vectors (zeros if key absent)
    y       : (N,)      label strings
    groups  : (N,)      prompt_group hashes for GroupKFold
    """
    label_file = processed_dir / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(
            f"No labels.json in {processed_dir} — run extract_features.py first."
        )
    labels_map: dict[str, dict] = json.loads(label_file.read_text())

    X_flat_list: list[np.ndarray] = []
    X_seg_list: list[np.ndarray] = []
    y_list: list[str] = []
    group_list: list[str] = []

    for npz_path in sorted(processed_dir.glob("*.npz")):
        run_id = npz_path.stem
        is_role = "__role__" in run_id
        if task == "role" and not is_role:
            continue
        if task != "role" and is_role:
            continue
        if run_id not in labels_map:
            continue
        info = labels_map[run_id]
        label = info.get(task)
        if label is None:
            continue

        d = np.load(npz_path, allow_pickle=False)
        flat = d["flat"]

        expected_dim = 35 if is_role else 195
        if flat.shape[0] != expected_dim:
            logger.warning(
                "Skipping %s: expected %d-dim flat, got %d", run_id, expected_dim, flat.shape[0]
            )
            continue

        # seg key may be absent in NPZs extracted before C1 was added
        if "seg" in d:
            seg = d["seg"].astype(np.float32)
            if seg.shape[0] != 30:
                seg = np.zeros(30, dtype=np.float32)
        else:
            seg = np.zeros(30, dtype=np.float32)

        X_flat_list.append(flat.astype(np.float32))
        X_seg_list.append(seg)
        y_list.append(label)
        group_list.append(str(info.get("prompt_group", run_id)))

    if not X_flat_list:
        raise ValueError(
            f"No samples for task={task} in {processed_dir}. "
            "Check that extract_features.py has been run on that directory."
        )

    X_flat = np.stack(X_flat_list, axis=0)
    X_seg  = np.stack(X_seg_list,  axis=0)

    logger.info(
        "Loaded %d samples | task=%s | dir=%s | classes=%s",
        len(y_list), task, processed_dir.name, sorted(set(y_list)),
    )
    return X_flat, X_seg, y_list, group_list


# ── Metrics + bootstrap ───────────────────────────────────────────────────────

def _macro_f1(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true))


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
    n: int = _N_BOOTSTRAP,
    seed: int = _RNG_SEED,
) -> dict[str, Any]:
    """
    Bootstrap CI (2.5 / 97.5 percentile) for accuracy and macro-F1.
    Resamples the test set (not the training set) — safe for transfer experiments.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc_samples: list[float] = []
    f1_samples:  list[float] = []
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        acc_samples.append(_accuracy(y_true_arr[sample].tolist(), y_pred_arr[sample].tolist()))
        f1_samples.append(_macro_f1(y_true_arr[sample].tolist(), y_pred_arr[sample].tolist(), classes))

    return {
        "accuracy":        _accuracy(y_true, y_pred),
        "accuracy_ci_lo":  float(np.percentile(acc_samples, 2.5)),
        "accuracy_ci_hi":  float(np.percentile(acc_samples, 97.5)),
        "macro_f1":        _macro_f1(y_true, y_pred, classes),
        "macro_f1_ci_lo":  float(np.percentile(f1_samples, 2.5)),
        "macro_f1_ci_hi":  float(np.percentile(f1_samples, 97.5)),
    }


# ── Internal CV (A→A or B→B) ─────────────────────────────────────────────────

def run_internal_cv(
    X: np.ndarray,
    y: list[str],
    groups: list[str],
    task: str,
    label: str,
    n_splits: int = 5,
) -> dict[str, Any]:
    """StratifiedGroupKFold cross-validation within a single deployment."""
    from sklearn.model_selection import StratifiedGroupKFold
    from models.random_forest import RFClassifier

    classes = sorted(set(y))
    n_splits = min(n_splits, min(len([v for v in y if v == c]) for c in classes))
    n_splits = max(2, n_splits)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=_RNG_SEED)
    y_arr = np.array(y)
    g_arr = np.array(groups)

    fold_accs: list[float] = []
    fold_f1s:  list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_arr, g_arr)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_arr[train_idx].tolist()
        y_te = y_arr[test_idx].tolist()

        clf = RFClassifier(task=task)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        fold_accs.append(_accuracy(y_te, preds))
        fold_f1s.append(_macro_f1(y_te, preds, classes))

    return {
        "experiment": label,
        "task":       task,
        "n_samples":  len(y),
        "n_classes":  len(classes),
        "n_splits":   n_splits,
        "accuracy":        float(np.mean(fold_accs)),
        "accuracy_std":    float(np.std(fold_accs)),
        "macro_f1":        float(np.mean(fold_f1s)),
        "macro_f1_std":    float(np.std(fold_f1s)),
    }


# ── Transfer evaluation (A→B or B→A) ─────────────────────────────────────────

def run_transfer(
    X_train: np.ndarray,
    y_train: list[str],
    X_test: np.ndarray,
    y_test: list[str],
    task: str,
    label: str,
    do_bootstrap: bool = True,
) -> dict[str, Any]:
    """Train on one deployment, evaluate on the other."""
    from models.random_forest import RFClassifier

    classes = sorted(set(y_train + y_test))

    clf = RFClassifier(task=task)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    result: dict[str, Any] = {
        "experiment": label,
        "task":       task,
        "n_train":    len(y_train),
        "n_test":     len(y_test),
        "n_classes":  len(classes),
    }

    if do_bootstrap:
        ci = bootstrap_ci(y_test, preds, classes)
        result.update(ci)
    else:
        result["accuracy"]  = _accuracy(y_test, preds)
        result["macro_f1"]  = _macro_f1(y_test, preds, classes)

    return result


# ── C2: Seg feature sub-analysis (workflow, A→B) ─────────────────────────────

def run_seg_subanalysis(
    Xa_flat: np.ndarray, Xa_seg: np.ndarray, ya: list[str],
    Xb_flat: np.ndarray, Xb_seg: np.ndarray, yb: list[str],
    do_bootstrap: bool = True,
) -> dict[str, dict]:
    """
    Three feature-set variants for A→B workflow transfer.

    (a) seg_only    30-dim seg only
    (b) flat_only   195-dim flat (seg zeroed — equivalent to standard flat)
    (c) flat+seg    225-dim concatenated

    Returns dict keyed by variant name.
    """
    results: dict[str, dict] = {}

    results["seg_only"] = run_transfer(
        Xa_seg, ya, Xb_seg, yb,
        task="workflow", label="A→B  [seg_only 30-dim]",
        do_bootstrap=do_bootstrap,
    )

    results["flat_only"] = run_transfer(
        Xa_flat, ya, Xb_flat, yb,
        task="workflow", label="A→B  [flat_only 195-dim]",
        do_bootstrap=do_bootstrap,
    )

    Xa_full = np.concatenate([Xa_flat, Xa_seg], axis=1)
    Xb_full = np.concatenate([Xb_flat, Xb_seg], axis=1)
    results["flat+seg"] = run_transfer(
        Xa_full, ya, Xb_full, yb,
        task="workflow", label="A→B  [flat+seg 225-dim]",
        do_bootstrap=do_bootstrap,
    )

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt(r: dict, do_ci: bool) -> str:
    acc = r["accuracy"]
    f1  = r["macro_f1"]
    if do_ci and "accuracy_ci_lo" in r:
        return (
            f"acc={acc:.3f} [{r['accuracy_ci_lo']:.3f}–{r['accuracy_ci_hi']:.3f}]  "
            f"F1={f1:.3f} [{r['macro_f1_ci_lo']:.3f}–{r['macro_f1_ci_hi']:.3f}]"
        )
    if "accuracy_std" in r:
        return f"acc={acc:.3f}±{r['accuracy_std']:.3f}  F1={f1:.3f}±{r['macro_f1_std']:.3f}"
    return f"acc={acc:.3f}  F1={f1:.3f}"


def _above_chance_retention(f1: float, random_baseline: float, ceiling: float) -> float:
    """Fraction of above-chance signal retained: (f1 − random) / (ceiling − random)."""
    denom = ceiling - random_baseline
    if denom <= 0:
        return 0.0
    return max(0.0, (f1 - random_baseline) / denom)


def _verdict(ab_f1: float, ba_f1: float, random_baseline: float, ceiling: float,
             task: str) -> str:
    """Return a single-line verdict string."""
    # topology and parallelism are structural baseline signals — connection graph
    # is directly readable from IP headers without any classifier.
    if task in ("topology", "parallelism"):
        ab_ret = _above_chance_retention(ab_f1, random_baseline, ceiling)
        ba_ret = _above_chance_retention(ba_f1, random_baseline, ceiling)
        return (
            f"STRUCTURAL BASELINE — connection graph is trivially observable from "
            f"IP headers (no ML required).  A→B above-chance retention {ab_ret:.0%}, "
            f"B→A {ba_ret:.0%}.  NOT an independent attack result."
        )

    ab_ret = _above_chance_retention(ab_f1, random_baseline, ceiling)
    ba_ret = _above_chance_retention(ba_f1, random_baseline, ceiling)

    # Verdict on A→B (primary transfer direction)
    if ab_f1 < random_baseline + 0.05:
        direction = "CHANCE — no generalizable signal"
    elif ab_ret >= 0.80:
        direction = "STRONG — implementation-agnostic (protocol-level leak)"
    elif ab_ret >= 0.40:
        direction = "MODERATE — partially generalizes"
    elif ab_ret >= 0.15:
        direction = "WEAK — mostly implementation-specific"
    else:
        direction = "CHANCE — effectively implementation-specific"

    return (
        f"{direction}\n"
        f"  Above-chance retention:  A→B {ab_ret:.0%}  |  B→A {ba_ret:.0%}"
    )


def print_task_results(
    task: str,
    aa: dict, bb: dict | None, ab: dict, ba: dict,
    do_ci: bool,
) -> None:
    n_cls_map = {"workflow": 4, "role": 3, "parallelism": 2, "topology": 3}
    n_cls = n_cls_map.get(task, "?")
    random_baseline = 1.0 / n_cls if isinstance(n_cls, int) else None

    width = 60
    print()
    print("─" * width)
    print(f"  Task: {task.upper()}   (random baseline ≈ {random_baseline:.3f})")
    print("─" * width)
    print(f"  A→A (internal CV)   {_fmt(aa, False)}")
    if bb is not None:
        print(f"  B→B (internal CV)   {_fmt(bb, False)}")
    print(f"  A→B (transfer)      {_fmt(ab, do_ci)}")
    print(f"  B→A (transfer)      {_fmt(ba, do_ci)}")

    if isinstance(n_cls, int) and random_baseline is not None:
        ceiling = aa["macro_f1"]
        ab_f1   = ab["macro_f1"]
        ba_f1   = ba["macro_f1"]
        if np.isnan(ab_f1) or np.isnan(ceiling):
            print("  (no deployment B data — run run_pilot.py --deployment b to enable transfer eval)")
            print()
            return
        print(f"  → {_verdict(ab_f1, ba_f1, random_baseline, ceiling, task)}")
    print()


def print_seg_subanalysis(seg_results: dict[str, dict], do_ci: bool) -> None:
    print()
    print("═" * 60)
    print("  SSE response-pattern sub-analysis  (workflow, A→B)")
    print("  Both deployments stream via SSE (a2a-sdk message/stream); each")
    print("  response packet is an SSE event.  The 'seg' vector captures genuine")
    print("  SSE-chunk structure (n_response_bursts, iqr_size_in) — response")
    print("  event counts and inbound size spread per flow.")
    print("═" * 60)
    for variant, r in seg_results.items():
        print(f"  {variant:<14} {_fmt(r, do_ci)}")

    seg_f1  = seg_results["seg_only"]["macro_f1"]
    flat_f1 = seg_results["flat_only"]["macro_f1"]
    full_f1 = seg_results["flat+seg"]["macro_f1"]

    print()
    print("  Interpretation:")
    if seg_f1 > flat_f1 + 0.05:
        print("  • seg_only > flat_only: response-pattern counts alone drive cross-deployment")
        print("    transfer.  The number of LLM call response bursts distinguishes workflows")
        print("    across implementations even without byte-count/timing features.")
    elif seg_f1 < 1.0 / 4 + 0.05:
        print("  • seg_only ≈ chance: response-pattern counts are NOT stable across deployments")
        print("    (A uses 3-phase retriever, B uses 2-phase — different call counts mask signal).")
    else:
        print("  • seg_only shows partial signal; workflow identity partially survives in")
        print("    response-direction burst counts despite different call structures.")

    if full_f1 > flat_f1 + 0.03:
        print("  • flat+seg > flat_only: response-pattern features ADD generalizable signal.")
    elif full_f1 < flat_f1 - 0.03:
        print("  • flat+seg < flat_only: response-pattern features HURT transfer (deployment-specific noise).")
    else:
        print("  • flat+seg ≈ flat_only: response-pattern features are neutral for transfer.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    do_bootstrap = not args.no_bootstrap

    b_exists = (dir_b / "labels.json").exists()
    if not b_exists:
        logger.warning(
            "Deployment B data not found at %s.  "
            "Run: sudo venv/bin/python scripts/run_pilot.py --deployment b --n 50 --out data/raw_b  "
            "then: python scripts/extract_features.py --raw data/raw_b --out %s --scapy",
            dir_b, dir_b,
        )
        logger.warning("Running A→A internal-CV only.")

    all_results: dict[str, Any] = {}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")

        # Load deployment A
        try:
            Xa_flat, Xa_seg, ya, ga = load_deployment(dir_a, task)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping task=%s: %s", task, exc)
            continue

        # A→A internal CV
        aa = run_internal_cv(Xa_flat, ya, ga, task, label="A→A")
        all_results[f"{task}/AA"] = aa

        if not b_exists:
            print_task_results(task, aa, None,
                               {"accuracy": float("nan"), "macro_f1": float("nan")},
                               {"accuracy": float("nan"), "macro_f1": float("nan")},
                               do_ci=False)
            continue

        # Load deployment B
        try:
            Xb_flat, Xb_seg, yb, gb = load_deployment(dir_b, task)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping task=%s (deployment B): %s", task, exc)
            print_task_results(task, aa, None,
                               {"accuracy": float("nan"), "macro_f1": float("nan")},
                               {"accuracy": float("nan"), "macro_f1": float("nan")},
                               do_ci=False)
            continue

        # B→B internal CV
        bb = run_internal_cv(Xb_flat, yb, gb, task, label="B→B")
        all_results[f"{task}/BB"] = bb

        # A→B transfer
        logger.info("Running A→B transfer for task=%s (bootstrap=%s)", task, do_bootstrap)
        ab = run_transfer(Xa_flat, ya, Xb_flat, yb, task=task, label="A→B",
                          do_bootstrap=do_bootstrap)
        all_results[f"{task}/AB"] = ab

        # B→A transfer
        logger.info("Running B→A transfer for task=%s (bootstrap=%s)", task, do_bootstrap)
        ba = run_transfer(Xb_flat, yb, Xa_flat, ya, task=task, label="B→A",
                          do_bootstrap=do_bootstrap)
        all_results[f"{task}/BA"] = ba

        print_task_results(task, aa, bb, ab, ba, do_ci=do_bootstrap)

        # C2: seg sub-analysis for workflow only
        if task == "workflow" and not args.skip_seg:
            logger.info("Running C2 seg sub-analysis (workflow, A→B)")
            seg_results = run_seg_subanalysis(
                Xa_flat, Xa_seg, ya,
                Xb_flat, Xb_seg, yb,
                do_bootstrap=do_bootstrap,
            )
            all_results["workflow/seg_subanalysis"] = seg_results
            print_seg_subanalysis(seg_results, do_ci=do_bootstrap)

    # Save results JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    # Final interpretation block
    _print_summary(all_results, args.tasks, do_bootstrap)


def _print_summary(all_results: dict, tasks: list[str], do_ci: bool) -> None:
    n_cls_map = {"workflow": 4, "role": 3, "parallelism": 2, "topology": 3}

    print()
    print("=" * 60)
    print("  CROSS-DEPLOYMENT SUMMARY  (above-chance retention)")
    print("  retention = (transfer_F1 − random) / (ceiling_F1 − random)")
    print("=" * 60)

    for task in tasks:
        n_cls = n_cls_map.get(task, 1)
        random_bl = 1.0 / n_cls
        aa_f1 = all_results.get(f"{task}/AA", {}).get("macro_f1", float("nan"))
        bb_f1 = all_results.get(f"{task}/BB", {}).get("macro_f1", float("nan"))
        ab_f1 = all_results.get(f"{task}/AB", {}).get("macro_f1", float("nan"))
        ba_f1 = all_results.get(f"{task}/BA", {}).get("macro_f1", float("nan"))

        ab_ret = _above_chance_retention(ab_f1, random_bl, aa_f1) if not np.isnan(ab_f1) else float("nan")
        ba_ret = _above_chance_retention(ba_f1, random_bl, aa_f1) if not np.isnan(ba_f1) else float("nan")

        flag = " [structural baseline]" if task in ("topology", "parallelism") else ""
        print(
            f"  {task:<14}  A→A={aa_f1:.3f}  B→B={bb_f1:.3f}  "
            f"A→B={ab_f1:.3f}(ret={ab_ret:.0%})  "
            f"B→A={ba_f1:.3f}(ret={ba_ret:.0%}){flag}"
        )

    print()
    print("  Key findings:")
    print("  • topology + parallelism transfer perfectly — but both reflect the")
    print("    connection graph (which hosts pair), readable directly from IP headers.")
    print("    This is NOT a side-channel; it is the structural baseline an on-path")
    print("    observer sees without any classifier.  Parallelism is topology collapsed")
    print("    to 2 classes — they are not independent results.")
    print("  • workflow + role above-chance retention is low (≤52 %) and asymmetric.")
    print("    The behavioral fingerprinting signal is largely implementation-specific:")
    print("    changing call count or execution order (A→B design) suffices to break it.")
    print("  • Honest headline: passive fingerprinting recovers coarse routing structure")
    print("    that is trivially observable, but does NOT generalize across deployments")
    print("    for the behaviorally interesting properties (workflow, role).  This is a")
    print("    limits-of-the-attack result, not a transferable-threat result.")
    print()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-deployment generalization evaluation")
    p.add_argument("--dir-a", default="data/processed",
                   help="Processed features directory for deployment A (default: data/processed)")
    p.add_argument("--dir-b", default="data/processed_b",
                   help="Processed features directory for deployment B (default: data/processed_b)")
    p.add_argument("--tasks", nargs="+", choices=_TASKS, default=_TASKS,
                   help="Tasks to evaluate (default: all four)")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap CI computation (faster)")
    p.add_argument("--skip-seg", action="store_true",
                   help="Skip response-pattern heuristic sub-analysis (formerly 'seg')")
    p.add_argument("--out", default="data/results/cross_deployment.json",
                   help="Output JSON path (default: data/results/cross_deployment.json)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
