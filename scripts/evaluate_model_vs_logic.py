#!/usr/bin/env python3
"""
Model-vs-logic disentanglement experiment.

Deployment A confounds two differences simultaneously: the LLM (llama3.2:3b vs
qwen2.5:7b) and the agent logic (parallel orchestrator + 3-phase retriever vs
sequential + 2-phase).  Cross-deployment transfer failed (workflow A→B 28% retention,
role B→A 20%).  This experiment isolates which factor is responsible.

Four conditions
───────────────
  A         llama3.2:3b  +  A-logic (parallel orch, 3-phase retriever)  [baseline]
  A_model   qwen2.5:7b   +  A-logic                                      [model-only change]
  B_logic   llama3.2:3b  +  B-logic (sequential orch, 2-phase retriever) [logic-only change]
  B         qwen2.5:7b   +  B-logic                                      [both changed, original]

Transfer experiments
────────────────────
  A → A_model   model-only effect on transfer
  A → B_logic   logic-only effect on transfer
  A → B         both changed (original cross-deployment result, for comparison)

If A→A_model ≈ A→A ceiling  → model switch alone does not break transfer; logic is the culprit.
If A→B_logic ≈ A→A ceiling  → logic switch alone does not break transfer; model is the culprit.
If both are low              → model and logic independently degrade transfer.
If A→B_logic << A→B         → logic is the dominant factor; model is secondary.

Data collection (run these first):
    # Model-only variant: A-logic + qwen2.5:7b
    sudo venv/bin/python scripts/run_pilot.py \\
        --model qwen2.5:7b --n 50 --out data/raw_amodel

    # Logic-only variant: B-logic + llama3.2:3b
    sudo venv/bin/python scripts/run_pilot.py --deployment b \\
        --model llama3.2:3b --n 50 --out data/raw_blogic

    # Extract features for both
    python scripts/extract_features.py --raw data/raw_amodel --out data/processed_amodel --scapy
    python scripts/extract_features.py --raw data/raw_blogic --out data/processed_blogic --scapy

Usage:
    python scripts/evaluate_model_vs_logic.py
    python scripts/evaluate_model_vs_logic.py --tasks workflow role
    python scripts/evaluate_model_vs_logic.py --no-bootstrap
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

_TASKS = ["workflow", "role"]  # topology/parallelism are structural; excluded here
_N_BOOTSTRAP = 1000
_RNG_SEED = 42

_N_CLS = {"workflow": 4, "role": 3}


# ── Re-use helpers from evaluate_cross_deployment ────────────────────────────

def _load(processed_dir: Path, task: str):
    """Load flat features, labels, groups from a processed directory."""
    label_file = processed_dir / "labels.json"
    if not label_file.exists():
        return None, None, None
    labels_map = json.loads(label_file.read_text())

    X_list, y_list, g_list = [], [], []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        run_id = npz_path.stem
        is_role = "__role__" in run_id
        if task == "role" and not is_role:
            continue
        if task != "role" and is_role:
            continue
        if run_id not in labels_map:
            continue
        label = labels_map[run_id].get(task)
        if label is None:
            continue
        d = np.load(npz_path, allow_pickle=False)
        flat = d["flat"].astype(np.float32)
        expected = 35 if is_role else 195
        if flat.shape[0] != expected:
            continue
        X_list.append(flat)
        y_list.append(label)
        g_list.append(str(labels_map[run_id].get("prompt_group", run_id)))

    if not X_list:
        return None, None, None
    return np.stack(X_list), y_list, g_list


def _accuracy(yt, yp):
    return sum(a == b for a, b in zip(yt, yp)) / len(yt)


def _f1(yt, yp, classes):
    from sklearn.metrics import f1_score
    return float(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))


def _bootstrap(yt, yp, classes, n=_N_BOOTSTRAP, seed=_RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yt))
    yta, ypa = np.array(yt), np.array(yp)
    accs, f1s = [], []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        accs.append(_accuracy(yta[s].tolist(), ypa[s].tolist()))
        f1s.append(_f1(yta[s].tolist(), ypa[s].tolist(), classes))
    return {
        "accuracy": _accuracy(yt, yp),
        "accuracy_ci_lo": float(np.percentile(accs, 2.5)),
        "accuracy_ci_hi": float(np.percentile(accs, 97.5)),
        "macro_f1": _f1(yt, yp, classes),
        "macro_f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "macro_f1_ci_hi": float(np.percentile(f1s, 97.5)),
    }


def _internal_cv(X, y, groups, task, n_splits=5):
    from sklearn.model_selection import StratifiedGroupKFold
    from models.random_forest import RFClassifier

    classes = sorted(set(y))
    n_splits = max(2, min(n_splits, min(sum(1 for v in y if v == c) for c in classes)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=_RNG_SEED)
    ya, ga = np.array(y), np.array(groups)
    accs, f1s = [], []
    for _, (tr, te) in enumerate(skf.split(X, ya, ga)):
        clf = RFClassifier(task=task)
        clf.fit(X[tr], ya[tr].tolist())
        preds = clf.predict(X[te])
        accs.append(_accuracy(ya[te].tolist(), preds))
        f1s.append(_f1(ya[te].tolist(), preds, classes))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))


def _transfer(X_tr, y_tr, X_te, y_te, task, do_bootstrap):
    from models.random_forest import RFClassifier
    classes = sorted(set(y_tr + y_te))
    clf = RFClassifier(task=task)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    if do_bootstrap:
        return _bootstrap(y_te, preds, classes)
    return {"accuracy": _accuracy(y_te, preds), "macro_f1": _f1(y_te, preds, classes)}


def _ret(f1, random, ceiling):
    d = ceiling - random
    return max(0.0, (f1 - random) / d) if d > 0 else 0.0


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt(r: dict, do_ci: bool) -> str:
    acc, f1 = r["accuracy"], r["macro_f1"]
    if do_ci and "accuracy_ci_lo" in r:
        return (f"acc={acc:.3f}[{r['accuracy_ci_lo']:.3f}–{r['accuracy_ci_hi']:.3f}]  "
                f"F1={f1:.3f}[{r['macro_f1_ci_lo']:.3f}–{r['macro_f1_ci_hi']:.3f}]")
    return f"acc={acc:.3f}  F1={f1:.3f}"


def _print_task(task, results, do_ci):
    random_bl = 1.0 / _N_CLS[task]
    ceiling   = results["AA"]["macro_f1"]

    print()
    print("─" * 70)
    print(f"  Task: {task.upper()}   random={random_bl:.3f}   A→A ceiling={ceiling:.3f}")
    print("─" * 70)

    rows = [
        ("A→A  (baseline CV)",           "AA"),
        ("A→A_model  (model only)",       "A_Amodel"),
        ("A→B_logic  (logic only)",       "A_Blogic"),
        ("A→B        (both changed)",     "AB"),
    ]
    for label, key in rows:
        if key not in results:
            print(f"  {label:<36}  (no data)")
            continue
        r = results[key]
        f1  = r["macro_f1"]
        ret = _ret(f1, random_bl, ceiling)
        tag = f"ret={ret:.0%}"
        print(f"  {label:<36}  {_fmt(r, do_ci and 'ci_lo' not in label)}  {tag}")

    # Interpretation
    am_f1 = results.get("A_Amodel", {}).get("macro_f1")
    bl_f1 = results.get("A_Blogic", {}).get("macro_f1")
    ab_f1 = results.get("AB",       {}).get("macro_f1")

    if am_f1 is None or bl_f1 is None:
        print()
        return

    am_ret = _ret(am_f1, random_bl, ceiling)
    bl_ret = _ret(bl_f1, random_bl, ceiling)

    print()
    if am_ret > 0.70 and bl_ret < 0.40:
        conclusion = "LOGIC is the dominant factor — model switch alone preserves transfer; " \
                     "agent logic change (call count, execution order) breaks it."
    elif bl_ret > 0.70 and am_ret < 0.40:
        conclusion = "MODEL is the dominant factor — logic change alone preserves transfer; " \
                     "LLM switch (response size, token distribution) breaks it."
    elif am_ret < 0.40 and bl_ret < 0.40:
        conclusion = "BOTH factors independently degrade transfer — model AND logic each " \
                     "contribute; the attack is fragile along both axes."
    elif am_ret > 0.70 and bl_ret > 0.70:
        conclusion = "NEITHER factor alone breaks transfer — the combined change is needed " \
                     "to degrade the attack; each individually preserves it."
    else:
        conclusion = (f"MIXED — model-only ret={am_ret:.0%}, logic-only ret={bl_ret:.0%}. "
                      f"Both factors contribute, with "
                      + ("logic" if bl_ret < am_ret else "model") + " being the larger factor.")

    print(f"  → {conclusion}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    dir_a      = Path(args.dir_a)
    dir_amodel = Path(args.dir_amodel)
    dir_blogic = Path(args.dir_blogic)
    dir_b      = Path(args.dir_b)
    do_bs      = not args.no_bootstrap

    # Check availability
    have_amodel = (dir_amodel / "labels.json").exists()
    have_blogic = (dir_blogic / "labels.json").exists()
    have_b      = (dir_b      / "labels.json").exists()

    if not have_amodel or not have_blogic:
        missing = []
        if not have_amodel:
            missing.append(f"  data/raw_amodel:  sudo venv/bin/python scripts/run_pilot.py "
                           f"--model qwen2.5:7b --n 50 --out data/raw_amodel\n"
                           f"                   python scripts/extract_features.py "
                           f"--raw data/raw_amodel --out {dir_amodel} --scapy")
        if not have_blogic:
            missing.append(f"  data/raw_blogic:  sudo venv/bin/python scripts/run_pilot.py --deployment b "
                           f"--model llama3.2:3b --n 50 --out data/raw_blogic\n"
                           f"                   python scripts/extract_features.py "
                           f"--raw data/raw_blogic --out {dir_blogic} --scapy")
        print("\n" + "=" * 70)
        print("  MODEL-VS-LOGIC EXPERIMENT — DATA MISSING")
        print("=" * 70)
        print("  Collect the missing controlled conditions first:\n")
        for m in missing:
            print(m)
        print()
        if not have_amodel and not have_blogic:
            return

    all_output = {}

    for task in args.tasks:
        print(f"\n{'='*70}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*70}")

        Xa, ya, ga = _load(dir_a, task)
        if Xa is None:
            logger.warning("No data for task=%s in %s", task, dir_a)
            continue

        results = {}

        # A→A internal CV (ceiling)
        acc_m, acc_s, f1_m, f1_s = _internal_cv(Xa, ya, ga, task)
        results["AA"] = {"accuracy": acc_m, "accuracy_std": acc_s,
                         "macro_f1": f1_m, "macro_f1_std": f1_s}
        logger.info("A→A  task=%s  F1=%.3f±%.3f", task, f1_m, f1_s)

        # A → A_model (model-only)
        if have_amodel:
            Xam, yam, _ = _load(dir_amodel, task)
            if Xam is not None:
                results["A_Amodel"] = _transfer(Xa, ya, Xam, yam, task, do_bs)
                logger.info("A→A_model  task=%s  F1=%.3f", task, results["A_Amodel"]["macro_f1"])

        # A → B_logic (logic-only)
        if have_blogic:
            Xbl, ybl, _ = _load(dir_blogic, task)
            if Xbl is not None:
                results["A_Blogic"] = _transfer(Xa, ya, Xbl, ybl, task, do_bs)
                logger.info("A→B_logic  task=%s  F1=%.3f", task, results["A_Blogic"]["macro_f1"])

        # A → B (both changed — original result for comparison)
        if have_b:
            Xb, yb, _ = _load(dir_b, task)
            if Xb is not None:
                results["AB"] = _transfer(Xa, ya, Xb, yb, task, do_bs)
                logger.info("A→B  task=%s  F1=%.3f", task, results["AB"]["macro_f1"])

        _print_task(task, results, do_ci=do_bs)
        all_output[task] = results

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialisation
    def _serial(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        raise TypeError(type(obj))
    out.write_text(json.dumps(all_output, indent=2, default=_serial))
    logger.info("Results saved → %s", out)

    # Summary
    print()
    print("=" * 70)
    print("  DISENTANGLEMENT SUMMARY")
    print("  (above-chance retention: (F1−random)/(ceiling−random))")
    print("=" * 70)
    for task in args.tasks:
        if task not in all_output:
            continue
        random_bl = 1.0 / _N_CLS[task]
        res = all_output[task]
        ceiling = res["AA"]["macro_f1"]
        for cond, key in [("A→A_model", "A_Amodel"), ("A→B_logic", "A_Blogic"), ("A→B", "AB")]:
            f1 = res.get(key, {}).get("macro_f1")
            if f1 is None:
                continue
            ret = _ret(f1, random_bl, ceiling)
            print(f"  {task:<12}  {cond:<14}  F1={f1:.3f}  ret={ret:.0%}")
    print()
    print("  Model condition:  run_pilot.py    --model qwen2.5:7b   (A-logic, qwen)")
    print("  Logic condition:  run_pilot.py --deployment b  --model llama3.2:3b  (B-logic, llama)")
    print()


def _parse():
    p = argparse.ArgumentParser(description="Model-vs-logic disentanglement experiment")
    p.add_argument("--dir-a",      default="data/processed",
                   help="Deployment A processed features (baseline)")
    p.add_argument("--dir-amodel", default="data/processed_amodel",
                   help="Model-only variant: A-logic + qwen2.5:7b")
    p.add_argument("--dir-blogic", default="data/processed_blogic",
                   help="Logic-only variant: B-logic + llama3.2:3b")
    p.add_argument("--dir-b",      default="data/processed_b",
                   help="Deployment B: B-logic + qwen2.5:7b (original cross-deployment)")
    p.add_argument("--tasks", nargs="+", choices=_TASKS, default=_TASKS)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--out", default="data/results/model_vs_logic.json")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
