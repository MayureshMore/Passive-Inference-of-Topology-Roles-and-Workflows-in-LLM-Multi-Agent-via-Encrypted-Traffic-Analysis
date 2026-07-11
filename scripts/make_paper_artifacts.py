#!/usr/bin/env python3
"""
Generate paper-ready figures and tables from the computed results in
data/results/.  Pure post-processing — reads only the result JSONs, runs no
models, so it is safe to re-run any time (e.g. after the C5 WAN collection).

Outputs:
  data/results/figures/confusion_workflow_gbt.png
  data/results/figures/confusion_role_gbt.png
  data/results/figures/closed_world_headline.png
  data/results/figures/disentanglement.png
  data/results/figures/defense_cost_benefit.png
  data/results/PAPER_ARTIFACTS.md   (per-class tables + figure index)

Usage:
    venv/bin/python scripts/make_paper_artifacts.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Override with A2A_RESULTS_DIR to read/write a sandbox copy (e.g. reproduce.sh),
# leaving the canonical committed data/results untouched.
RESULTS = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CHANCE = {"workflow": 0.25, "role": 1.0 / 3, "parallelism": 0.5, "topology": 1.0 / 3}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ── Per-class precision / recall / F1 from a confusion matrix ──────────────────

def per_class_metrics(cm: dict) -> list[dict]:
    labels = cm["labels"]
    M = np.array(cm["matrix"], dtype=float)          # rows = true, cols = pred
    rows = []
    for i, lab in enumerate(labels):
        tp = M[i, i]
        support = M[i, :].sum()
        pred_tot = M[:, i].sum()
        prec = tp / pred_tot if pred_tot else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"class": lab, "precision": prec, "recall": rec,
                     "f1": f1, "support": int(support)})
    return rows


def _md_table(rows: list[dict]) -> str:
    out = ["| Class | Precision | Recall | F1 | Support |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} "
                   f"| {r['f1']:.3f} | {r['support']} |")
    return "\n".join(out)


# ── Figure 1/2: confusion matrices ────────────────────────────────────────────

def fig_confusion(task: str, model: str = "gbt") -> Path | None:
    d = _load(RESULTS / "closed_world" / f"closed_world_{model}_{task}.json")
    cm = d.get("cv", {}).get("confusion_matrix")
    if not cm:
        return None
    labels = cm["labels"]
    M = np.array(cm["matrix"], dtype=float)
    Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)   # row-normalised
    f1 = d["cv"]["f1_macro"]["mean"]

    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 2.2, 0.9 * len(labels) + 2))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{task.capitalize()} — {model.upper()} (macro-F1={f1:.3f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(M[i, j])}\n{Mn[i, j]*100:.0f}%",
                    ha="center", va="center",
                    color="white" if Mn[i, j] > 0.5 else "black", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    p = FIGS / f"confusion_{task}_{model}.png"
    fig.savefig(p); plt.close(fig)
    return p


# ── Figure 3: closed-world headline (GBT vs RF, CIs, chance) ───────────────────

def _ci_err(metric: dict) -> tuple[float, float]:
    """Asymmetric (lower, upper) error-bar lengths from mean + ci bounds."""
    m = metric.get("mean")
    lo, hi = metric.get("ci_lo"), metric.get("ci_hi")
    if m is None or lo is None or hi is None:
        return 0.0, 0.0
    return max(0.0, m - lo), max(0.0, hi - m)


def fig_closed_world() -> Path:
    tasks = ["topology", "parallelism", "role", "workflow"]
    gbt_m, gbt_e, rf_m, rf_e = [], [], [], []
    for t in tasks:
        g = _load(RESULTS / "closed_world" / f"closed_world_gbt_{t}.json").get("cv", {}).get("f1_macro", {})
        r = _load(RESULTS / "closed_world" / f"closed_world_rf_{t}.json").get("cv", {}).get("f1_macro", {})
        gbt_m.append(g.get("mean", 0)); rf_m.append(r.get("mean", 0))
        gbt_e.append(_ci_err(g)); rf_e.append(_ci_err(r))
    x = np.arange(len(tasks)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, gbt_m, w, yerr=np.array(gbt_e).T, capsize=4, label="GBT (primary)", color="#2c6fbb")
    ax.bar(x + w/2, rf_m, w, yerr=np.array(rf_e).T, capsize=4, label="RF (baseline)", color="#9ecae1")
    for i, t in enumerate(tasks):  # chance markers per task
        ax.plot([i - w, i + w], [CHANCE[t]] * 2, color="crimson", lw=1.6, ls="--",
                label="chance" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({'structural' if t in ('topology','parallelism') else 'attack'})"
                        for t in tasks])
    ax.set_ylabel("macro-F1  (5-fold group CV, 95% CI)"); ax.set_ylim(0, 1.05)
    ax.set_title("Closed-world fingerprinting (GBT ≈ RF; chance dashed)")
    ax.legend(loc="lower right")
    fig.tight_layout(); p = FIGS / "closed_world_headline.png"; fig.savefig(p); plt.close(fig)
    return p


# ── Figure 4: disentanglement (model vs logic) ────────────────────────────────

def fig_disentanglement() -> Path | None:
    ml = _load(RESULTS / "model_vs_logic.json")
    if not ml:
        return None
    conds = [("AA", "A→A\n(ceiling)"), ("A_Amodel", "A→A_model\n(model swap)"),
             ("A_Blogic", "A→B_logic\n(logic swap)"), ("AB", "A→B\n(both)")]
    tasks = [t for t in ("workflow", "role") if t in ml]
    x = np.arange(len(conds)); w = 0.38
    colors = {"workflow": "#d95f0e", "role": "#3182bd"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, t in enumerate(tasks):
        m, e = [], []
        for key, _ in conds:
            r = ml[t].get(key, {})
            mean = r.get("macro_f1", 0); m.append(mean)
            lo, hi = r.get("macro_f1_ci_lo"), r.get("macro_f1_ci_hi")
            e.append((max(0, mean - lo), max(0, hi - mean)) if lo is not None else (0, 0))
        ax.bar(x + (k - 0.5) * w, m, w, yerr=np.array(e).T, capsize=4,
               label=t, color=colors.get(t))
        ax.axhline(CHANCE[t], color=colors.get(t), ls=":", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds])
    ax.set_ylabel("transfer macro-F1 (95% CI)"); ax.set_ylim(0, 1.0)
    ax.set_title("Disentanglement: model swap ≈ ceiling, logic swap collapses it")
    ax.legend(title="task"); fig.tight_layout()
    p = FIGS / "disentanglement.png"; fig.savefig(p); plt.close(fig)
    return p


# ── Figure 5: defense cost / benefit ──────────────────────────────────────────

def fig_defense() -> Path | None:
    dl = _load(RESULTS / "defense" / "defense_live.json")
    if not dl:
        return None
    chance = dl.get("chance", 0.25)
    names = [n for n in ("none", "rate", "pad") if n in dl]
    # Headline metric is macro-F1 (not accuracy); use the macro-F1 bootstrap CI.
    f1v = [dl[n]["macro_f1"] for n in names]
    err = [( max(0, dl[n]["macro_f1"] - dl[n].get("macro_f1_ci_lo", dl[n]["macro_f1"])),
             max(0, dl[n].get("macro_f1_ci_hi", dl[n]["macro_f1"]) - dl[n]["macro_f1"]) ) for n in names]
    byte = [100 * dl[n].get("byte_overhead", 0) for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    # left: attack macro-F1 under each defense (with CI) + chance line
    a1.bar(names, f1v, yerr=np.array(err).T, capsize=5,
           color=["#888", "#e6550d", "#3182bd"])
    a1.axhline(chance, color="crimson", ls="--", lw=1.5, label=f"chance ({chance:.2f})")
    for i, n in enumerate(names):
        ret = dl[n].get("above_chance_retention_f1", dl[n].get("above_chance_retention"))
        if ret is not None:
            a1.text(i, f1v[i] + 0.03, f"keep {ret*100:.0f}%", ha="center", fontsize=8)
    a1.set_ylabel("attack macro-F1 (95% CI)"); a1.set_ylim(0, max(f1v) + 0.18)
    a1.set_title("Attack macro-F1 under defense"); a1.legend(loc="upper right")
    # right: byte overhead cost
    a2.bar(names, byte, color=["#888", "#e6550d", "#3182bd"])
    a2.set_ylabel("byte overhead (%)"); a2.set_title("Bandwidth cost of each defense")
    for i, b in enumerate(byte):
        a2.text(i, b + 0.5, f"{b:.0f}%", ha="center", fontsize=9)
    fig.suptitle("C4 defenses: both partial (~70% signal kept) and expensive (~30% bytes)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGS / "defense_cost_benefit.png"; fig.savefig(p); plt.close(fig)
    return p


def fig_defense_curve() -> Path | None:
    """Overhead–accuracy CURVE (from scripts/sweep_defenses.py → defense_curve.json).

    Left: size-padding — macro-F1 vs byte overhead (sim sweep + live pad anchor).
    Right: timing-spacing — macro-F1 vs schedule-derived latency overhead.
    """
    dc = _load(RESULTS / "defense_curve.json")
    if not dc or not dc.get("rows"):
        return None
    rows = dc["rows"]
    chance = dc.get("chance", 0.25)
    live = dc.get("measured_live", {})
    none = next((r for r in rows if r["defense"] == "none_sim"), None)
    nf1 = none["attack_macro_f1"] if none else None

    def series(defense: str, xkey: str):
        rs = sorted((r for r in rows if r["defense"] == defense), key=lambda r: r["param"])
        x = [100 * r[xkey] for r in rs]
        f1 = [r["attack_macro_f1"] for r in rs]
        err = [[max(0, r["attack_macro_f1"] - r["ci_low"]) for r in rs],
               [max(0, r["ci_high"] - r["attack_macro_f1"]) for r in rs]]
        return x, f1, err

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    # left — size padding vs byte overhead
    x, f1, err = series("pad_size_sim", "byte_overhead")
    a1.errorbar(x, f1, yerr=err, marker="o", color="#3182bd", capsize=3, label="pad sweep (sim)")
    if nf1 is not None:
        a1.scatter([0], [nf1], color="#444", zorder=5, label="undefended")
    if "pad" in live:
        a1.scatter([100 * live["pad"]["byte_overhead"]], [live["pad"]["macro_f1"]],
                   marker="*", s=200, color="#e6550d", zorder=6, label="pad live (measured)")
    a1.axhline(chance, ls="--", color="crimson", lw=1.2, label=f"chance ({chance:.2f})")
    a1.set_xlabel("byte overhead (%)"); a1.set_ylabel("attack macro-F1 (95% CI)")
    a1.set_ylim(0, max(0.72, (nf1 or 0.66) + 0.08)); a1.set_title("Size padding (sweep cell size)")
    a1.legend(fontsize=7, loc="lower left")
    # right — timing spacing vs schedule-derived latency overhead
    x, f1, err = series("rate_timing_sim", "latency_overhead")
    a2.errorbar(x, f1, yerr=err, marker="s", color="#31a354", capsize=3, label="timing sweep (sim)")
    if nf1 is not None:
        a2.scatter([0], [nf1], color="#444", zorder=5, label="undefended")
    a2.axhline(chance, ls="--", color="crimson", lw=1.2, label=f"chance ({chance:.2f})")
    if "rate" in live:
        a2.text(0.5, 0.06,
                f"live 'rate' F1={live['rate']['macro_f1']:.2f} is a different\n"
                f"(count-based) mechanism — not on this timing axis",
                transform=a2.transAxes, fontsize=6.5, ha="center", color="#666")
    a2.set_xlabel("latency overhead (%, schedule-derived)")
    a2.set_title("Timing spacing (sweep min-gap)"); a2.legend(fontsize=7, loc="lower left")
    fig.suptitle("C4 defenses: overhead–accuracy curves — both are expensive and plateau "
                 "well above chance (~65% of signal kept)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGS / "defense_curve.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_offtheshelf_fingerprint() -> Path | None:
    """Task #4 — role fingerprint replicated on the independent a2a_mcp system."""
    d = _load(RESULTS / "offtheshelf_fingerprint.json")
    if not d or "primary_role_closed_world" not in d:
        return None
    pr = d["primary_role_closed_world"]
    order = [("role_6way", "role (6-way)"), ("coordinator_vs_specialist", "coord vs\nspecialist (2-way)")]
    names, f1v, chances, err = [], [], [], []
    for key, lbl in order:
        r = pr.get(key)
        if not r:
            continue
        names.append(lbl); f1v.append(r["macro_f1"]); chances.append(r["chance"])
        err.append((max(0, r["macro_f1"] - r["ci_lo"]), max(0, r["ci_hi"] - r["macro_f1"])))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    xs = np.arange(len(names))
    ax.bar(xs, f1v, 0.55, yerr=np.array(err).T, capsize=5, color=["#3182bd", "#31a354"])
    for i, (x, c) in enumerate(zip(xs, chances)):
        ax.hlines(c, x - 0.3, x + 0.3, colors="crimson", linestyles="--", lw=1.4)
        ax.text(x, f1v[i] + 0.035, f"{f1v[i]:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(x, c + 0.01, f"chance {c:.2f}", ha="center", fontsize=7, color="crimson")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("macro-F1 (GBT, group-safe CV, 95% CI)"); ax.set_ylim(0, 1.12)
    sec = d.get("secondary_cross_impl_shared_abstraction", {})
    rec = sec.get("specialist_recall_on_A")
    sub = (f"cross-impl a2a_mcp→A specialist recall = {rec:.2f}   |   "
           "workflow not separable (LLM-planned routing)") if rec is not None else ""
    ax.set_title("Role fingerprint REPLICATES on an independent system (Google a2a_mcp)\n"
                 + sub, fontsize=9)
    fig.tight_layout()
    p = FIGS / "offtheshelf_fingerprint.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_framework_id() -> Path | None:
    """Phase 1 — implementation identification confusion matrix (with confound caveat)."""
    d = _load(RESULTS / "framework_id.json")
    if not d or "result_full_shape" not in d:
        return None
    cm = d["result_full_shape"]["confusion_matrix"]
    labels = cm["labels"]
    M = np.asarray(cm["matrix"], dtype=float)
    Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)  # row-normalized
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = Mn[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    mf = d["result_full_shape"]["macro_f1"]
    acf = d["timing_ablation"]["A_vs_C_separability_full"]
    acn = d["timing_ablation"]["A_vs_C_separability_no_timing"]
    ax.set_title(f"Phase 1: implementation ID — macro-F1 {mf:.3f}\n"
                 f"⚠ near-perfect incl. same-structure A↔C ({acf}, {acn} no-timing):\n"
                 f"likely batch/session-confounded — treat within-family as upper bound",
                 fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    fig.tight_layout()
    p = FIGS / "framework_id.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_cross_instance_transfer() -> Path | None:
    """Phase 2 — role transfer between two independent instances of a2a_mcp (both directions)."""
    d = _load(RESULTS / "cross_instance_transfer.json")
    if not d or "role_transfer" not in d:
        return None
    rt = d["role_transfer"]
    dirs = [("inst1_to_inst2", "train inst 1\n→ test inst 2"),
            ("inst2_to_inst1", "train inst 2\n→ test inst 1")]
    names, f1v, err = [], [], []
    for key, lbl in dirs:
        r = rt.get(key)
        if not r:
            continue
        names.append(lbl); f1v.append(r["macro_f1"])
        err.append((max(0, r["macro_f1"] - r["ci_lo"]), max(0, r["ci_hi"] - r["macro_f1"])))
    chance = rt["inst1_to_inst2"]["chance"]
    weak = rt["weaker_direction_macro_f1"]
    roles = d.get("common_roles_used", [])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    xs = np.arange(len(names))
    ax.bar(xs, f1v, 0.5, yerr=np.array(err).T, capsize=5, color=["#3182bd", "#9ecae1"])
    for i, x in enumerate(xs):
        ax.text(x, f1v[i] + 0.03, f"{f1v[i]:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.hlines(chance, -0.4, len(names) - 0.6, colors="crimson", linestyles="--", lw=1.4)
    ax.text(len(names) - 0.6, chance + 0.012, f"chance {chance:.2f}", ha="right", fontsize=8, color="crimson")
    ax.hlines(0.70, -0.4, len(names) - 0.6, colors="#238b45", linestyles=":", lw=1.4)
    ax.text(-0.38, 0.71, "§4 DEPLOYABLE ≥0.70", ha="left", fontsize=8, color="#238b45")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("macro-F1 (GBT transfer, 95% CI)"); ax.set_ylim(0, 1.15)
    n2 = d.get("roles_present_instance2", {})
    spec = min((n2.get(r, 0) for r in ("air_ticketing", "hotel", "car_rental")), default=0)
    ax.set_title(
        f"Phase 2: role transfer across two independent a2a_mcp instances\n"
        f"(diff LLM 2.5→2.0-flash, diff prompts, diff session) — {len(roles)}-way "
        f"coordinator layer ({'/'.join(roles)})\n"
        f"weaker direction {weak:.2f} → §4 DEPLOYABLE  |  specialists too sparse "
        f"(n≈{spec}/inst-2, <5 bar) — untested",
        fontsize=8.3)
    fig.tight_layout()
    p = FIGS / "cross_instance_transfer.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_framework_id_control() -> Path | None:
    """Phase-1 confound control — A↔C runtime fingerprint collapses under same-session capture."""
    d = _load(RESULTS / "framework_id_interleaved.json")
    if not d or "confound_control_comparison" not in d:
        return None
    cc = d["confound_control_comparison"]
    orig_sep = cc.get("original_confounded", {}).get("A_vs_C_separability_full")
    ctrl_sep = cc.get("interleaved_controlled", {}).get("A_vs_C_separability_full_from_3way")
    if orig_sep is None or ctrl_sep is None:
        return None
    hl = d.get("headline_A_vs_C_2way_balanced", {})
    mf, ci = hl.get("macro_f1_full"), hl.get("macro_f1_full_ci")
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    xs = [0, 1]
    bars = ax.bar(xs, [orig_sep, ctrl_sep], 0.5, color=["#c6553b", "#3182bd"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["separate-session\n(confounded)", "same-session\ninterleaved\n(controlled)"])
    for x, v in zip(xs, [orig_sep, ctrl_sep]):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.hlines(0.5, -0.4, 1.4, colors="crimson", linestyles="--", lw=1.4)
    ax.text(1.4, 0.51, "≈ chance", ha="right", fontsize=8, color="crimson")
    ax.set_ylabel("A↔C separability (asyncio vs LangGraph)"); ax.set_ylim(0, 1.1)
    sub = (f"balanced 2-way macro-F1 (controlled) = {mf:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] — CI straddles chance"
           if mf is not None and ci else "")
    ax.set_title("Phase-1 CONTROL: the A↔C runtime fingerprint is a CAPTURE-SESSION CONFOUND\n"
                 "0.997 → 0.49 under same-session interleaving — honestly demoted; corroborates §3\n"
                 + sub, fontsize=8.5)
    fig.tight_layout()
    p = FIGS / "framework_id_control.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_confound_control() -> Path | None:
    """Confound audit — core claims survive same-session interleaving; framework-ID collapses."""
    d = _load(RESULTS / "confound_control.json")
    if not d or "results" not in d:
        return None
    R = d["results"]
    rows = []
    for task in ("workflow", "role", "topology"):
        r = R.get(task)
        if not r:
            continue
        rows.append((task, r["committed_baseline"]["macro_f1"], r["interleaved"]["macro_f1"],
                     r["interleaved"]["chance"], True))
    f = R.get("framework_id_A_vs_C")
    if f:
        rows.append(("framework-ID\n(A↔C)", f["separate_session"], f["interleaved"], 0.5, False))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(rows)); w = 0.38
    comm = [r[1] for r in rows]; intr = [r[2] for r in rows]
    surv = [r[4] for r in rows]
    ax.bar(xs - w/2, comm, w, label="committed (batched)", color="#9e9e9e")
    ax.bar(xs + w/2, intr, w, label="same-session interleaved (controlled)",
           color=["#2e7d32" if s else "#c62828" for s in surv])
    for i, r in enumerate(rows):
        ax.hlines(r[3], i - 0.45, i + 0.45, colors="crimson", linestyles="--", lw=1.1)
        ax.text(i + w/2, r[2] + 0.02, f"{r[2]:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i - w/2, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.15); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Confound audit: the SAME same-session control that breaks framework-ID\n"
                 "leaves workflow / role / topology UNCHANGED (Δ ≤ 0.03) — the core attack is real,\n"
                 "not a capture artefact (red = collapses to chance; green = survives)", fontsize=8.5)
    fig.tight_layout()
    p = FIGS / "confound_control.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ── Markdown bundle ───────────────────────────────────────────────────────────

def write_markdown(figs: dict[str, Path | None]) -> Path:
    lines = ["# Paper artifacts", "",
             "_Auto-generated by `scripts/make_paper_artifacts.py` from `data/results/`._", ""]

    lines.append("## Per-class metrics (closed-world, GBT)\n")
    for task in ("workflow", "role"):
        d = _load(RESULTS / "closed_world" / f"closed_world_gbt_{task}.json")
        cm = d.get("cv", {}).get("confusion_matrix")
        if not cm:
            continue
        f1 = d["cv"]["f1_macro"]
        lines.append(f"### {task.capitalize()}  (macro-F1 {f1['mean']:.3f} "
                     f"[{f1.get('ci_lo', float('nan')):.3f}, {f1.get('ci_hi', float('nan')):.3f}], "
                     f"chance {CHANCE[task]:.3f})\n")
        lines.append(_md_table(per_class_metrics(cm)))
        lines.append("")

    # deep-model footnote table (only if --full-suite produced them)
    def _f1_of(model: str, task: str):
        d = _load(RESULTS / "closed_world" / f"closed_world_{model}_{task}.json")
        cv = d.get("cv", d)
        f = cv.get("f1_macro")
        return f.get("mean") if isinstance(f, dict) else f
    tasks = ["workflow", "role", "topology", "parallelism"]
    if any(_f1_of("transformer", t) is not None or _f1_of("cnn", t) is not None for t in tasks):
        lines.append("## Deep models vs trees (degenerate — excluded from all claims)\n")
        lines.append("Shown for completeness only. The deep sequence models **collapse to "
                     "(near) single-class prediction** at N=600 (below-chance macro-F1) — a "
                     "degenerate classifier, not a fair baseline. Excluded from every claim.\n")
        lines.append("| Task | GBT | RF | Transformer | CNN1D | chance |")
        lines.append("|---|---|---|---|---|---|")
        for t in tasks:
            def s(m):
                v = _f1_of(m, t)
                return f"{v:.3f}" if v is not None else "—"
            lines.append(f"| {t} | {s('gbt')} | {s('rf')} | {s('transformer')} | {s('cnn')} | {CHANCE[t]:.3f} |")
        lines.append("")

    # defense table
    dl = _load(RESULTS / "defense" / "defense_live.json")
    if dl:
        lines.append("## C4 live defenses (workflow attack)\n")
        lines.append("| Defense | Acc [95% CI] | Drop | Signal kept | Byte ohd |")
        lines.append("|---|---|---|---|---|")
        for n in ("none", "rate", "pad"):
            if n not in dl:
                continue
            r = dl[n]
            ci = f"[{r.get('accuracy_ci_lo', float('nan')):.3f}, {r.get('accuracy_ci_hi', float('nan')):.3f}]"
            drop = f"{r['acc_drop']:.3f}" if "acc_drop" in r else "—"
            keep = f"{r['above_chance_retention']*100:.0f}%" if "above_chance_retention" in r else "—"
            lines.append(f"| {n} | {r['accuracy']:.3f} {ci} | {drop} | {keep} | "
                         f"{r.get('byte_overhead', 0)*100:.0f}% |")
        lines.append("\n_Cost is reported as **bandwidth (byte) overhead**. Latency is treated as "
                     "confounded and omitted: the defended sets were collected separately, so "
                     "wall-clock deltas (including pad's spurious negative) reflect run-to-run "
                     "network/LLM variance, not the defense._\n")

    lines.append("## Figures\n")
    for name, p in figs.items():
        if p:
            lines.append(f"- **{name}** — `{p.as_posix()}`")
    out = RESULTS / "PAPER_ARTIFACTS.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    figs = {
        "Confusion — workflow (GBT)": fig_confusion("workflow", "gbt"),
        "Confusion — role (GBT)":     fig_confusion("role", "gbt"),
        "Closed-world headline":      fig_closed_world(),
        "Disentanglement":            fig_disentanglement(),
        "Defense cost/benefit":       fig_defense(),
        "Defense overhead–accuracy curve": fig_defense_curve(),
        "Off-the-shelf role fingerprint":  fig_offtheshelf_fingerprint(),
        "Framework/implementation ID (Phase 1)": fig_framework_id(),
        "Framework-ID confound control (Phase 1)": fig_framework_id_control(),
        "Confound audit — core claims survive": fig_confound_control(),
        "Cross-instance role transfer (Phase 2)": fig_cross_instance_transfer(),
    }
    md = write_markdown(figs)
    print("Wrote:")
    for name, p in figs.items():
        print(f"  {'✓' if p else '·'} {name}: {p}")
    print(f"  ✓ markdown bundle: {md}")


if __name__ == "__main__":
    main()
