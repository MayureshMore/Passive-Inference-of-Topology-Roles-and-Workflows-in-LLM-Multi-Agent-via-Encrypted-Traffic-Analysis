#!/usr/bin/env python3
"""
PHASE 2 — Cross-INSTANCE transfer on the public a2a_mcp framework (the deployable-attack test).

Train a role classifier on ONE instance of a2a_mcp and test it on a SECOND, independently
stood-up instance of the SAME framework:

    instance 1 = the committed 150-trip set        (data/raw_offtheshelf,       gemini-2.5-flash)
    instance 2 = a fresh, independent capture       (data/raw_offtheshelf_inst2, gemini-2.0-flash,
                 reworded prompts, separate session) — same six roles by port (10100-10105).

If a classifier trained on instance 1 recovers roles on instance 2 (and vice-versa), an attacker
can train on their OWN copy of a popular framework and attack a victim who merely runs it.

Representation + method are IDENTICAL to evaluate_offtheshelf_fingerprint.py (35-dim per-agent
traffic-shape vector; port is the LABEL only, never a feature) and to the project's transfer
pattern (_transfer: fit on train instance, predict on test instance; macro-F1 + bootstrap CI).

Roles are restricted to those present in BOTH instances with ≥ MIN_N samples (a2a_mcp's
LLM-planned routing fans out to the air/hotel/car specialists only sometimes, so specialists
are sparse; the mcp/orchestrator/planner coordinators appear every trip). Per-role n is reported.

Pre-registered verdict (brief §4, weaker direction is the headline):
    ≥ 0.70 & CI clear of chance → DEPLOYABLE ATTACK
    0.40 – 0.70                 → PARTIAL
    < 0.40 or CI touches chance → BOUNDED
coordinator-vs-specialist 2-way is reported too but flagged PARTLY STRUCTURAL (hubs carry more
traffic than leaves, so it rides on volume like topology — not the behavioural headline).

Writes ${A2A_RESULTS_DIR:-data/results}/cross_instance_transfer.json. Additive; touches nothing.
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
from evaluation.stats import bootstrap_ci  # noqa: E402
from models.gradient_boosted import GBTClassifier  # noqa: E402
from scripts.evaluate_offtheshelf_fingerprint import extract_role_samples, coarse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_N = 5  # a role must have ≥ this many samples in BOTH instances to enter the transfer


def counts(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def transfer(Xtr, ytr, Xte, yte, label):
    clf = GBTClassifier(task="role").fit(Xtr, list(ytr))
    pred = clf.predict(Xte)
    classes = sorted(set(ytr) | set(yte))
    ci = bootstrap_ci(list(yte), list(pred), classes=classes)
    chance = 1.0 / len(sorted(set(yte)))
    logger.info("[%s] macro-F1=%.3f [%.3f,%.3f] acc=%.3f (n_test=%d, chance=%.3f)",
                label, ci["macro_f1"], ci["macro_f1_ci_lo"], ci["macro_f1_ci_hi"],
                ci["accuracy"], len(yte), chance)
    return {"macro_f1": ci["macro_f1"], "ci_lo": ci["macro_f1_ci_lo"], "ci_hi": ci["macro_f1_ci_hi"],
            "accuracy": ci["accuracy"], "n_test": int(len(yte)), "chance": chance,
            "test_classes": sorted(set(yte))}


def restrict(X, y, roles):
    m = np.isin(y, list(roles))
    return X[m], y[m]


def band(mf, ci_lo, chance):
    if mf >= 0.70 and ci_lo > chance:
        return "DEPLOYABLE ATTACK (≥0.70, CI clear of chance)"
    if 0.40 <= mf < 0.70 and ci_lo > chance:
        return "PARTIAL (0.40–0.70; transfers but instance drift degrades it)"
    return "BOUNDED (<0.40 or CI touches chance)"


def main(args: argparse.Namespace) -> None:
    r1, r2 = Path(args.inst1), Path(args.inst2)
    if not any(r1.glob("*.pcap")):
        raise SystemExit(f"blocked: no instance-1 pcaps at {r1}")
    if not any(r2.glob("*.pcap")):
        raise SystemExit(f"blocked: no instance-2 pcaps at {r2} — run collect_offtheshelf_inst2.sh first")

    X1, y1, _ = extract_role_samples(r1)
    X2, y2, _ = extract_role_samples(r2)
    c1, c2 = counts(y1), counts(y2)
    logger.info("instance-1 roles: %s", c1)
    logger.info("instance-2 roles: %s", c2)

    common = sorted({r for r in set(c1) & set(c2) if c1[r] >= MIN_N and c2[r] >= MIN_N})
    if len(common) < 2:
        raise SystemExit(f"blocked: <2 roles shared with ≥{MIN_N} samples in both instances "
                         f"(inst1={c1}, inst2={c2}). Collect more instance-2 trips.")

    X1c, y1c = restrict(X1, y1, common)
    X2c, y2c = restrict(X2, y2, common)

    # 6-way (whatever roles are common) — both directions.
    f_1to2 = transfer(X1c, y1c, X2c, y2c, f"{len(common)}-way inst1→inst2")
    f_2to1 = transfer(X2c, y2c, X1c, y1c, f"{len(common)}-way inst2→inst1")
    weak = min(f_1to2["macro_f1"], f_2to1["macro_f1"])
    weak_dir = f_1to2 if f_1to2["macro_f1"] <= f_2to1["macro_f1"] else f_2to1
    verdict = band(weak, weak_dir["ci_lo"], weak_dir["chance"])

    # coordinator-vs-specialist 2-way (partly structural) — both directions, if both classes exist.
    coarse_out = None
    y1k = np.array([coarse(r) for r in y1]); y2k = np.array([coarse(r) for r in y2])
    if len({*y1k}) == 2 and len({*y2k}) == 2:
        c_1to2 = transfer(X1, y1k, X2, y2k, "coord-vs-spec inst1→inst2")
        c_2to1 = transfer(X2, y2k, X1, y1k, "coord-vs-spec inst2→inst1")
        coarse_out = {
            "caveat": "PARTLY STRUCTURAL — hubs carry more traffic than leaves, so this rides on "
                      "connection volume (like topology), not subtle per-agent behaviour. The "
                      "behavioural headline is the multi-role transfer above.",
            "inst1_to_inst2": c_1to2, "inst2_to_inst1": c_2to1,
        }

    out = {
        "task": "cross-INSTANCE role transfer on a2a_mcp (Phase 2 — deployable-attack test)",
        "hypothesis": "two independent instances of the SAME framework share call structure and "
                      "differ only in surface variables (LLM, prompts, session), so role transfer "
                      "SHOULD work — a prediction, not a guarantee.",
        "independence_of_instance2": {
            "different_llm": "gemini-2.0-flash (instance 1 = gemini-2.5-flash), via LITELLM_MODEL",
            "different_prompts": "reworded query template, different dates/party-size/class/nights",
            "separate_session": True,
            "shared": "the a2a_mcp framework's fixed six roles by port (10100-10105)",
        },
        "representation": "35-dim per-agent traffic-shape vector (features/per_flow.py); port is "
                          "the LABEL only, never a feature.",
        "method": "fit GBT on train instance, predict test instance (_transfer pattern); macro-F1 "
                  "with bootstrap 95% CI (seed 42); chance = 1/n_test_roles.",
        "roles_present_instance1": c1,
        "roles_present_instance2": c2,
        "common_roles_used": common,
        "min_samples_per_role": MIN_N,
        "n_way": len(common),
        "role_transfer": {
            "inst1_to_inst2": f_1to2,
            "inst2_to_inst1": f_2to1,
            "weaker_direction_macro_f1": weak,
        },
        "coordinator_vs_specialist": coarse_out,
        "verdict_phase2": verdict,
        "verdict_basis": "weaker of the two directions (brief §4).",
        "caveats": "Single second instance; specialist roles (air/hotel/car) are sparse because "
                   "a2a_mcp's LLM-planned routing fans out to them only sometimes, so the transfer "
                   "is dominated by the always-present coordinator roles (mcp/orchestrator/planner) "
                   "— see roles_present_*. Report is on the roles that met the ≥N-sample bar.",
    }

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_instance_transfer.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 74)
    print("  PHASE 2 — CROSS-INSTANCE ROLE TRANSFER (a2a_mcp inst1 ⇄ inst2)")
    print("=" * 74)
    print(f"  common roles (≥{MIN_N} in both): {common}  ({len(common)}-way)")
    print(f"  inst1→inst2  macro-F1 = {f_1to2['macro_f1']:.3f} [{f_1to2['ci_lo']:.3f},{f_1to2['ci_hi']:.3f}]"
          f"  (n_test={f_1to2['n_test']}, chance={f_1to2['chance']:.3f})")
    print(f"  inst2→inst1  macro-F1 = {f_2to1['macro_f1']:.3f} [{f_2to1['ci_lo']:.3f},{f_2to1['ci_hi']:.3f}]"
          f"  (n_test={f_2to1['n_test']}, chance={f_2to1['chance']:.3f})")
    print(f"  weaker direction = {weak:.3f}")
    print(f"  VERDICT (§4): {verdict}")
    print("=" * 74)
    print(f"\nWrote {out_dir / 'cross_instance_transfer.json'}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 cross-instance role transfer on a2a_mcp")
    p.add_argument("--inst1", default="data/raw_offtheshelf")
    p.add_argument("--inst2", default="data/raw_offtheshelf_inst2")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
