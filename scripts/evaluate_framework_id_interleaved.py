#!/usr/bin/env python3
"""
PHASE 1 CONFOUND CONTROL — framework ID on the SAME-SESSION INTERLEAVED capture.

evaluate_framework_id.py found near-perfect implementation separability but flagged it as
possibly a capture-SESSION / batch confound (each implementation collected in a separate
run). scripts/collect_interleaved.sh re-collects A / B / C_langgraph ROUND-ROBINED in one
continuous session, so session-drift artefacts are shared across labels and de-correlated
from the label. A and C use the SAME model (llama3.2:3b) and identical workflows/topologies,
so A vs C differs ONLY in the orchestration runtime (asyncio vs LangGraph).

This script re-runs the IDENTICAL analysis (same 190-dim traffic-shape vector, same GBT,
same group-safe CV, same timing ablation) on the interleaved data and contrasts the
confound-controlled A↔C separability against the original (confounded) number.

Pre-registered reading (no massaging — either outcome is publishable):
  * A↔C separability SURVIVES interleaving (stays ~comparable, CI clear of chance)
        -> GENUINE runtime fingerprint. The confound was not the whole story; the recon
           signal is real. Phase 1 is UPGRADED from "upper bound" to a clean result.
  * A↔C separability COLLAPSES toward chance under interleaving
        -> the original 0.998 was batch-inflated. Phase 1 within-family ID is HONESTLY
           DEMOTED to "not separable once the session confound is removed" — a clean
           negative that strengthens the paper by removing an attackable claim.

Reads the interleaved processed dirs; writes ${A2A_RESULTS_DIR:-data/results}/
framework_id_interleaved.json. Additive; touches nothing committed.

Usage:
    venv/bin/python scripts/evaluate_framework_id_interleaved.py
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
from features.names import FLAT_FEATURE_NAMES  # noqa: E402
from scripts.evaluate_framework_id import (  # noqa: E402
    EXCLUDE_NAMES, is_timing, load_impl, run_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Interleaved processed dirs (produced by collect_interleaved.sh + extract_features.py).
DEPLOYMENTS = {
    "A": "data/processed_interleaved_a",
    "B": "data/processed_interleaved_b",
    "C_langgraph": "data/processed_interleaved_langgraph",
}
ORIG_RESULT = "data/results/framework_id.json"  # the confounded baseline for side-by-side


def ac(res):  # A↔C separability if both present
    return res["pairwise_separability"].get("A vs C_langgraph", {}).get("separability")


def main(args: argparse.Namespace) -> None:
    names = FLAT_FEATURE_NAMES()
    keep_idx = [i for i, n in enumerate(names) if n not in EXCLUDE_NAMES]
    kept_names = [names[i] for i in keep_idx]
    notiming_cols = [k for k, n in enumerate(kept_names) if not is_timing(n)]
    timing_dropped = [n for n in kept_names if is_timing(n)]

    X_parts, y, groups, counts = [], [], [], {}
    for label, d in DEPLOYMENTS.items():
        p = Path(d)
        if not (p / "labels.json").exists() and not any(p.glob("*.npz")):
            logger.warning("skip %s — no features at %s "
                           "(run collect_interleaved.sh + extract_features.py first)", label, d)
            continue
        Xi, yi, gi = load_impl(p, label)
        if len(Xi) == 0:
            logger.warning("skip %s — 0 usable traces at %s", label, d)
            continue
        X_parts.append(Xi[:, keep_idx]); y += yi; groups += gi
        counts[label] = len(Xi)
        logger.info("loaded %-12s n=%d", label, len(Xi))

    if len(counts) < 3:
        raise SystemExit(f"blocked: fewer than 3 interleaved implementations have features "
                         f"(found {list(counts)}). Run collect_interleaved.sh then extract "
                         f"each raw_interleaved_* dir to processed_interleaved_*.")

    X = np.vstack(X_parts); y = np.asarray(y)
    chance = 1.0 / len(counts)

    full = run_cv(X, y, groups, chance)
    notiming = run_cv(X[:, notiming_cols], y, groups, chance)
    ac_full, ac_nt = ac(full), ac(notiming)

    # ── HEADLINE: dedicated, BALANCED A↔C 2-way (asyncio vs LangGraph) ────────────
    # This is the clean control: A and C share model (llama3.2:3b) + call logic and both
    # cover all 12 (workflow×topology) conditions equally, so the ONLY systematic difference
    # is the orchestration runtime. B is excluded here because its orchestrator deterministically
    # fails chain/mesh on 3 of 4 workflows (see class_condition_coverage), giving B narrower
    # coverage that would let a 3-way separate B by topology-coverage artefact rather than shape.
    ga = np.asarray(groups, dtype=object)
    m_ac = np.isin(y, ["A", "C_langgraph"])
    two = two_nt = None
    if m_ac.sum() and len(set(y[m_ac].tolist())) == 2:
        Xac, yac, gac = X[m_ac], y[m_ac], list(ga[m_ac])
        two = run_cv(Xac, yac, gac, 0.5)
        two_nt = run_cv(Xac[:, notiming_cols], yac, gac, 0.5)
    ac2_f = two["macro_f1"] if two else None
    ac2_nt = two_nt["macro_f1"] if two_nt else None

    # Original (confounded) A↔C for the side-by-side.
    orig = {}
    op = Path(ORIG_RESULT)
    if op.exists():
        od = json.loads(op.read_text())
        orig = {
            "macro_f1_full": od.get("result_full_shape", {}).get("macro_f1"),
            "A_vs_C_separability_full": od.get("timing_ablation", {}).get("A_vs_C_separability_full"),
            "A_vs_C_separability_no_timing": od.get("timing_ablation", {}).get("A_vs_C_separability_no_timing"),
        }

    # Pre-registered verdict on the A↔C runtime question. Headline number = the BALANCED 2-way
    # A↔C macro-F1 under interleaving (ac2_f, chance 0.5); fall back to the 3-way-derived
    # separability only if the 2-way could not be built.
    COLLAPSE = 0.60   # at/below this ~= confound-driven; runtime signal did not survive the control
    SURVIVE = 0.85    # clearly retained
    ac_head = ac2_f if ac2_f is not None else ac_full
    if ac_head is None:
        ac_verdict = "A or C absent — cannot judge A↔C."
    elif ac_head >= SURVIVE:
        ac_verdict = ("SURVIVES — A↔C stays separable under same-session interleaving, so it is a "
                      "GENUINE runtime fingerprint (asyncio vs LangGraph, same model+logic), not a "
                      "session artefact. Phase-1 within-family ID is UPGRADED from 'upper bound' to a "
                      "clean recon signal.")
    elif ac_head <= COLLAPSE:
        ac_verdict = ("COLLAPSES — A↔C falls toward chance once the session confound is removed, so "
                      "the original near-perfect number was BATCH-INFLATED. Phase-1 within-family "
                      "(A/B/C) ID is HONESTLY DEMOTED to 'not separable under control'. (Distinct-"
                      "framework ID, e.g. a2a_mcp vs ours, is unaffected — genuinely different structure.)")
    else:
        ac_verdict = ("PARTIAL — A↔C drops but stays above chance; a real-but-weaker runtime signal "
                      "partly inflated by the session confound. Report the interleaved number as the "
                      "honest magnitude.")

    out = {
        "task": "Phase-1 CONFOUND CONTROL — framework ID on same-session interleaved capture",
        "why": "Original framework_id.json separability may be capture-session/batch confounded; "
               "this re-runs the identical analysis on round-robin interleaved A/B/C so session "
               "drift is shared across labels. A & C share model+logic, differ only in runtime.",
        "method": "IDENTICAL to evaluate_framework_id.py (190-dim traffic-shape vector = 195 minus 5 "
                  "structural counts; GBT; group-safe 5-fold StratifiedGroupKFold by prompt_group; "
                  "macro-F1 + pooled-OOF bootstrap 95% CI; timing ablation). Port never a feature.",
        "n_classes": len(counts), "chance": chance, "n_per_class": counts,
        "headline_A_vs_C_2way_balanced": {
            "note": "THE control result — dedicated GBT on A vs C_langgraph only (asyncio vs "
                    "LangGraph), same model+logic, both fully covering all 12 conditions; chance 0.5.",
            "macro_f1_full": ac2_f,
            "macro_f1_full_ci": [two["macro_f1_ci_lo"], two["macro_f1_ci_hi"]] if two else None,
            "macro_f1_no_timing": ac2_nt,
            "n_A": counts.get("A"), "n_C": counts.get("C_langgraph"),
        },
        "class_condition_coverage": {
            "note": "Deployment B's orchestrator (sequential pipeline) DETERMINISTICALLY fails "
                    "chain/mesh for 3 of 4 workflows, so B only yields star + support_triage traffic "
                    "(narrower coverage than A/C, which cover all 12). B's 3-way separability is "
                    "therefore partly a TOPOLOGY-COVERAGE artefact — the clean runtime test is the "
                    "balanced A↔C 2-way above, which excludes B.",
        },
        "result_full_shape": full,
        "timing_ablation": {
            "n_timing_features_dropped": len(timing_dropped),
            "result_no_timing": notiming,
            "A_vs_C_separability_full": ac_full,
            "A_vs_C_separability_no_timing": ac_nt,
        },
        "confound_control_comparison": {
            "original_confounded": orig,
            "interleaved_controlled": {
                "A_vs_C_2way_macro_f1_full": ac2_f,
                "A_vs_C_2way_macro_f1_no_timing": ac2_nt,
                "A_vs_C_separability_full_from_3way": ac_full,
                "macro_f1_full_3way": full["macro_f1"],
            },
            "thresholds": {"collapse_at_or_below": COLLAPSE, "survive_at_or_above": SURVIVE},
        },
        "verdict_A_vs_C": ac_verdict,
        "verdict_full_3way": (
            f"RECON CONFIRMED under control (macro-F1 {full['macro_f1']:.3f} ≥ 0.90, CI clear of chance)"
            if full["macro_f1"] >= 0.90 and full["macro_f1_ci_lo"] > chance else
            f"PARTIAL under control (macro-F1 {full['macro_f1']:.3f}, CI above chance)"
            if full["macro_f1_ci_lo"] > chance else
            f"NOT SEPARABLE under control (macro-F1 {full['macro_f1']:.3f}, CI touches chance)"),
    }

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "framework_id_interleaved.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 74)
    print("  PHASE 1 CONFOUND CONTROL — same-session interleaved framework ID")
    print("=" * 74)
    print(f"  classes={list(counts)}  n_per_class={counts}  chance(3way)={chance:.3f}")
    if two:
        print(f"  ★ HEADLINE  A↔C 2-way (asyncio vs LangGraph, balanced, chance 0.50):")
        print(f"      interleaved macro-F1 = {ac2_f:.3f} [{two['macro_f1_ci_lo']:.3f}, {two['macro_f1_ci_hi']:.3f}]"
              f"   (no-timing {ac2_nt:.3f})")
    print(f"  full 3-way macro-F1 = {full['macro_f1']:.3f} "
          f"[{full['macro_f1_ci_lo']:.3f}, {full['macro_f1_ci_hi']:.3f}]  acc={full['accuracy']:.3f}"
          f"  (B coverage-imbalanced — see coverage note)")
    print("  A↔C separability   original(confounded) -> interleaved(controlled):")
    print(f"      full     : {orig.get('A_vs_C_separability_full')}  ->  {ac_full}")
    print(f"      no-timing: {orig.get('A_vs_C_separability_no_timing')}  ->  {ac_nt}")
    print(f"  VERDICT A↔C : {out['verdict_A_vs_C']}")
    print(f"  VERDICT 3way: {out['verdict_full_3way']}")
    print("=" * 74)
    print(f"\nWrote {out_dir / 'framework_id_interleaved.json'}")


def _parse() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Phase-1 interleaved confound control").parse_args()


if __name__ == "__main__":
    main(_parse())
