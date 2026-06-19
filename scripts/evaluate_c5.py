#!/usr/bin/env python3
"""
C5 cross-network evaluation (proposal §11.3) — one command.

For each task (workflow, role, parallelism, topology) it reports three numbers
with 95% bootstrap CIs, plus a summary figure + JSON:

  (1) LAN-internal   — RF group-safe 5-fold CV on the LAN testbed   (in-deployment ceiling)
  (2) WAN-internal   — RF group-safe 5-fold CV on the WAN capture    (WAN-vantage ceiling)
  (3) LAN→WAN xfer   — RF trained on LAN, tested on WAN              (the cross-network result)

The WAN-internal baseline is the key control: it separates
  "the WAN vantage is impoverished"        → low  WAN-internal, from
  "the model doesn't transfer vantages"    → high WAN-internal but low LAN→WAN.

Everything is RF + macro-F1 so the three columns are directly comparable.

Usage:
    venv/bin/python scripts/evaluate_c5.py --local data/processed --wan data/processed_wan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.stats import bootstrap_ci
from models.random_forest import RFClassifier

CHANCE = {"workflow": 0.25, "role": 1.0 / 3, "parallelism": 0.5, "topology": 1.0 / 3}
TASKS = ["workflow", "role", "parallelism", "topology"]


def _load(proc_dir: Path, task: str):
    """Build (X, y, groups) for one task directly from labels.json + per-trace npz.

    role uses the per-agent `__role__` npz (35-dim per-flow vector); the other
    tasks use the per-trace npz (195-dim flat vector).  group = prompt hash, so
    the CV is leakage-safe across identical prompts.
    """
    labels = json.loads((proc_dir / "labels.json").read_text())
    X, y, g = [], [], []
    for rid, meta in labels.items():
        is_role = "__role__" in rid
        if (task == "role") != is_role:
            continue
        if task not in meta:
            continue
        npz = proc_dir / f"{rid}.npz"
        if not npz.exists():
            continue
        X.append(np.load(npz)["flat"])
        y.append(meta[task])
        g.append(meta.get("prompt_group", rid))
    return np.array(X), y, g


def _internal_cv(proc_dir: Path, task: str) -> dict:
    X, y, g = _load(proc_dir, task)
    res = RFClassifier(task=task).cross_validate(X, y, n_splits=5, groups=g)
    f = res["f1_macro"]
    return {
        "f1": f["mean"], "ci_lo": f["ci_lo"], "ci_hi": f["ci_hi"],
        "acc": res["accuracy"]["mean"], "n": len(y), "classes": sorted(set(y)),
    }


def _transfer(local_dir: Path, wan_dir: Path, task: str) -> dict:
    Xl, yl, _ = _load(local_dir, task)
    Xw, yw, _ = _load(wan_dir, task)
    clf = RFClassifier(task=task)
    clf.fit(Xl, yl)
    pred = clf.predict(Xw)
    classes = sorted(set(yl) | set(yw))
    ci = bootstrap_ci(yw, pred, classes=classes)
    return {
        "f1": ci["macro_f1"], "ci_lo": ci["macro_f1_ci_lo"], "ci_hi": ci["macro_f1_ci_hi"],
        "acc": ci["accuracy"], "n": len(yw),
        "train_classes": sorted(set(yl)), "test_classes": sorted(set(yw)),
    }


def _figure(out: Path, results: dict, tasks: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(tasks))
    w = 0.26
    conds = [
        ("lan_internal", "LAN-internal (ceiling)", "#4c72b0"),
        ("wan_internal", "WAN-internal (vantage ceiling)", "#55a868"),
        ("transfer", "LAN→WAN transfer", "#c44e52"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label, color) in enumerate(conds):
        vals = [results[t][key]["f1"] for t in tasks]
        lo = [max(0.0, results[t][key]["f1"] - results[t][key]["ci_lo"]) for t in tasks]
        hi = [max(0.0, results[t][key]["ci_hi"] - results[t][key]["f1"]) for t in tasks]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1})
    for j, t in enumerate(tasks):
        ax.hlines(CHANCE[t], x[j] - 1.5 * w, x[j] + 1.5 * w,
                  colors="k", linestyles="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("macro-F1 (RF, 5-fold group CV / transfer; 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title("C5: attack survives the WAN vantage (WAN-internal) but does not transfer LAN→WAN")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.005, 0.012, "dashed = chance", transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    local_dir, wan_dir = Path(args.local), Path(args.wan)
    if not (wan_dir / "labels.json").exists():
        print(f"No WAN features at {wan_dir} — run extract_features on data/raw_wan first.")
        return

    results: dict[str, dict] = {}
    for task in args.tasks:
        print(f"[{task}] LAN-internal CV ...", flush=True)
        lan = _internal_cv(local_dir, task)
        print(f"[{task}] WAN-internal CV ...", flush=True)
        wan = _internal_cv(wan_dir, task)
        print(f"[{task}] LAN→WAN transfer ...", flush=True)
        xfer = _transfer(local_dir, wan_dir, task)
        results[task] = {"chance": CHANCE[task], "lan_internal": lan,
                         "wan_internal": wan, "transfer": xfer}

    out_json = Path("data/results/c5_cross_network.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    fig_path = Path("data/results/figures/c5_cross_network.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _figure(fig_path, results, list(args.tasks))

    print("\n" + "=" * 74)
    print("  C5 CROSS-NETWORK  (macro-F1, RF)")
    print("=" * 74)
    print(f"  {'task':<13}{'chance':>8}{'LAN-internal':>15}{'WAN-internal':>15}{'LAN→WAN':>13}")
    print("  " + "-" * 66)
    for t in args.tasks:
        r = results[t]
        print(f"  {t:<13}{r['chance']:>8.2f}"
              f"{r['lan_internal']['f1']:>15.3f}"
              f"{r['wan_internal']['f1']:>15.3f}"
              f"{r['transfer']['f1']:>13.3f}")
    print("=" * 74)
    print(f"  n(WAN)={results[args.tasks[0]]['transfer']['n']}  "
          f"JSON → {out_json}   FIGURE → {fig_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C5 cross-network eval: transfer + internal baselines + figure")
    p.add_argument("--local", default="data/processed")
    p.add_argument("--wan", default="data/processed_wan")
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
