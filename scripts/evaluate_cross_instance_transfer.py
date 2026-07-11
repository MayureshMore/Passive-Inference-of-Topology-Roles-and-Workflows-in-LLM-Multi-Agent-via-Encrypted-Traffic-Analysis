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

MIN_N = int(os.environ.get("MIN_N", "5"))  # a role needs ≥ this many samples in BOTH instances
# (default 5 reproduces the committed coordinator result; the 6-way run sets MIN_N=10 so a noisy
#  4-5-sample specialist can never enter — per-class F1 on n=5 is noise, worse than none.)


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


SPECIALISTS = ("air_ticketing", "hotel", "car_rental")


def specialist_distribution_check(X1, y1, X2, y2):
    """Instance-2's specialist samples were collected with the fan-out-BOOSTED driver +
    fully-specified prompts (a confound axis). Compare per-agent feature distributions of each
    specialist role across instances so a driver artefact cannot masquerade as a result:
      * if the 6-way is ≥0.70, comparable distributions show the positive is robust (indeed the
        driver difference makes it a HARDER test: train on inst-1 NATURAL specialists, test on
        inst-2 FORCED ones);
      * if the 6-way is <0.70, a large distribution gap flags the driver as a possible contributor
        to the drop — it must be named, not folded silently into a 'behaviour-doesn't-transfer'
        narrative.
    Metric: per-feature standardized mean difference (|SMD| = |mean1-mean2| / pooled_std); a role
    is 'comparable' if median |SMD| < 0.5 and < 25% of features exceed |SMD| = 1."""
    y1 = np.asarray(y1); y2 = np.asarray(y2)
    out = {}
    for role in SPECIALISTS:
        A = X1[y1 == role]; B = X2[y2 == role]
        if len(A) < 3 or len(B) < 3:
            out[role] = {"n_inst1": int(len(A)), "n_inst2": int(len(B)), "note": "too few to compare"}
            continue
        mu1, mu2 = A.mean(0), B.mean(0)
        pooled = np.sqrt((A.var(0) + B.var(0)) / 2.0) + 1e-9
        smd = np.abs(mu1 - mu2) / pooled
        cos = float(np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-9))
        out[role] = {
            "n_inst1": int(len(A)), "n_inst2": int(len(B)),
            "median_abs_smd": round(float(np.median(smd)), 3),
            "max_abs_smd": round(float(np.max(smd)), 3),
            "frac_features_abs_smd_gt_1": round(float(np.mean(smd > 1.0)), 3),
            "mean_vector_cosine": round(cos, 4),
            "comparable": bool(np.median(smd) < 0.5 and np.mean(smd > 1.0) < 0.25),
        }
    n_comp = sum(1 for r in out.values() if r.get("comparable"))
    out["_summary"] = {
        "specialists_comparable": f"{n_comp}/{len(SPECIALISTS)}",
        "all_comparable": n_comp == len(SPECIALISTS),
    }
    return out


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

    # Clean coordinator-only 3-way (mcp/orchestrator/planner) — these fire on EVERY trip so their
    # samples are NATURAL in both instances (no boosted driver), giving the unconfounded deployable
    # result to contrast against the (driver-boosted) 6-way. Reported whenever ≥2 coordinators qualify.
    coord_roles = [r for r in common if r in ("mcp", "orchestrator", "planner")]
    coord_out = None
    if len(coord_roles) >= 2:
        X1k, y1k2 = restrict(X1, y1, coord_roles); X2k, y2k2 = restrict(X2, y2, coord_roles)
        cc12 = transfer(X1k, y1k2, X2k, y2k2, f"{len(coord_roles)}-way COORD inst1→inst2")
        cc21 = transfer(X2k, y2k2, X1k, y1k2, f"{len(coord_roles)}-way COORD inst2→inst1")
        cweak = min(cc12["macro_f1"], cc21["macro_f1"])
        cwd = cc12 if cc12["macro_f1"] <= cc21["macro_f1"] else cc21
        coord_out = {
            "note": "NATURAL both instances (coordinators fire on every trip; no boosted driver) — "
                    "the clean, unconfounded deployable result.",
            "roles": coord_roles, "inst1_to_inst2": cc12, "inst2_to_inst1": cc21,
            "weaker_direction_macro_f1": cweak, "verdict": band(cweak, cwd["ci_lo"], cwd["chance"]),
        }

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

    # Specialist distribution check (Correction 2): inst-2 specialists were collected with the
    # fan-out-boosted driver, so compare their per-agent feature distributions to inst-1's natural
    # specialists — a driver artefact must not masquerade as a positive OR a boundary finding.
    dist_check = specialist_distribution_check(X1, y1, X2, y2)
    specialists_in_way = [r for r in common if r in SPECIALISTS]
    all_comparable = dist_check.get("_summary", {}).get("all_comparable", False)
    if specialists_in_way and weak >= 0.70:
        driver_interpretation = (
            "6-way ≥0.70 WITH specialists in the transfer. Instance-2's specialists were collected "
            "with the boosted driver, which if anything makes this a HARDER test (train on inst-1 "
            "NATURAL specialists, test on inst-2 FORCED ones). "
            + ("Specialist feature distributions are COMPARABLE across instances (see "
               "specialist_distribution_check), so the positive is not a driver artefact — it is "
               "robust." if all_comparable else
               "BUT specialist distributions are NOT all comparable across instances — the boosted "
               "driver may be inflating similarity; treat the positive with caution and see "
               "specialist_distribution_check."))
    elif specialists_in_way and weak < 0.70:
        driver_interpretation = (
            "6-way <0.70 with specialists in the transfer. The boosted driver used for inst-2 "
            "specialists is a POSSIBLE CONTRIBUTOR to the drop (train-natural / test-forced "
            "mismatch), NOT necessarily 'behaviour doesn't transfer'. This must be named as a "
            "candidate confound. "
            + ("However specialist distributions are comparable across instances, which argues "
               "against the driver being the whole story." if all_comparable else
               "Specialist distributions differ across instances, consistent with the driver "
               "contributing — the structure-vs-behaviour reading is confounded here."))
    else:
        driver_interpretation = ("No specialists met the ≥%d bar — this is the coordinator-layer "
                                 "result; the boosted-driver axis does not apply." % MIN_N)

    out = {
        "task": "cross-INSTANCE role transfer on a2a_mcp (Phase 2 — deployable-attack test)",
        "hypothesis": "two independent instances of the SAME framework share call structure and "
                      "differ only in surface variables (LLM, prompts, session, driver), so role "
                      "transfer SHOULD work — a prediction, not a guarantee.",
        "independence_of_instance2": {
            "different_llm": "gemini-2.0-flash (instance 1 = gemini-2.5-flash), via LITELLM_MODEL",
            "different_prompts": "reworded query template, different dates/party-size/class/nights",
            "separate_session": True,
            "different_driver_for_specialists": "instance-2's specialist samples were topped up with "
                "the fan-out-BOOSTED driver (drive_orch_boost.py) + fully-specified queries, because "
                "a2a_mcp's natural fan-out to air/hotel/car is only ~6-11% of trips. This is an ADDED "
                "axis of difference (cuts both ways — see driver_confound_interpretation and "
                "specialist_distribution_check); the coordinator samples remain natural.",
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
        "coordinator_layer_3way_clean": coord_out,
        "coordinator_vs_specialist": coarse_out,
        "specialist_distribution_check": dist_check,
        "driver_confound_interpretation": driver_interpretation,
        "verdict_phase2": verdict,
        "verdict_basis": "weaker of the two directions (brief §4); verdict field matches the number.",
        "caveats": "Single second instance. Specialist samples (air/hotel/car) in instance-2 were "
                   "collected with the boosted driver (see independence_of_instance2 + "
                   "specialist_distribution_check); coordinator samples are natural.",
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
