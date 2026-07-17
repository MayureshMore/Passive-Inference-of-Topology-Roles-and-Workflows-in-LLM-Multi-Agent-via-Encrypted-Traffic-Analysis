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
from features.per_flow import PerFlowFeatures  # noqa: E402
from models.gradient_boosted import GBTClassifier  # noqa: E402
from scripts.evaluate_offtheshelf_fingerprint import extract_role_samples, coarse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_N = int(os.environ.get("MIN_N", "5"))  # a role needs ≥ this many samples in BOTH instances
# (default 5 reproduces the committed coordinator result; the 6-way run sets MIN_N=10 so a noisy
#  4-5-sample specialist can never enter — per-class F1 on n=5 is noise, worse than none.)

# ── volume-ablation mask (§9a) ────────────────────────────────────────────────
# The coordinator "deployable" result could be behavioural (per-agent traffic SHAPE)
# or could ride on the connection-VOLUME signal already demoted to a topology baseline
# (a hub carries more bytes/packets than a leaf). To separate the two — exactly parallel
# to the framework-ID timing ablation (0.46→0.38) — re-run the transfer on a SHAPE-ONLY
# feature set: drop every raw count / raw-byte-magnitude dimension (whatever scales with
# how much traffic an agent carries), keep only per-packet size shape, IAT/duration/gap
# timing, burst-duration shape, and normalized ratios.
_FEATURE_NAMES = PerFlowFeatures.FEATURE_NAMES()  # 35 names, matches to_vector() order
_VOLUME_FEATURES = (
    {"n_pkts_out", "n_pkts_in", "bytes_out", "bytes_in",      # raw packet/byte totals
     "n_bursts", "mean_burst_bytes", "std_burst_bytes",       # burst count + byte magnitudes
     "n_small_inbound", "n_response_bursts"}                  # raw event counts
    | {f"cumul_bytes_q{i}" for i in range(10)}                # cumulative byte trajectory (raw volume)
)
_SHAPE_MASK = np.array([n not in _VOLUME_FEATURES for n in _FEATURE_NAMES], dtype=bool)
_DROPPED_FEATURES = [n for n in _FEATURE_NAMES if n in _VOLUME_FEATURES]
_KEPT_FEATURES = [n for n in _FEATURE_NAMES if n not in _VOLUME_FEATURES]


def counts(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def transfer(Xtr, ytr, Xte, yte, gte, label):
    """gte = test-side cluster labels (trip). Several flows come from one trip, so the CI resamples
    whole TRIPS (project convention, evaluation/stats) — an i.i.d. interval here is over-confident."""
    clf = GBTClassifier(task="role").fit(Xtr, list(ytr))
    pred = clf.predict(Xte)
    classes = sorted(set(ytr) | set(yte))
    ci = bootstrap_ci(list(yte), list(pred), classes=classes, groups=list(gte))
    chance = 1.0 / len(sorted(set(yte)))
    logger.info("[%s] macro-F1=%.3f [%.3f,%.3f] acc=%.3f (n_test=%d, %d trips, chance=%.3f)",
                label, ci["macro_f1"], ci["macro_f1_ci_lo"], ci["macro_f1_ci_hi"],
                ci["accuracy"], len(yte), ci["n_clusters"] or 0, chance)
    return {"macro_f1": ci["macro_f1"], "ci_lo": ci["macro_f1_ci_lo"], "ci_hi": ci["macro_f1_ci_hi"],
            "accuracy": ci["accuracy"], "n_test": int(len(yte)), "chance": chance,
            "test_classes": sorted(set(yte)),
            "ci_method": ci["ci_method"], "ci_n_clusters": ci["n_clusters"]}


def restrict(X, y, g, roles):
    m = np.isin(y, list(roles))
    return X[m], y[m], np.asarray(g)[m]


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


def inst2_specialist_mode(r2, y2, g2):
    """Classify instance-2's SPECIALIST samples as NATURAL or BOOSTED by reading each sample's
    per-trip sidecar 'boosted' flag. The distribution check + 6-way band are interpreted
    CONDITIONALLY on this: boosted → the driver is a candidate confound; natural → the confound is
    removed, so any residual distribution gap is the legitimate LLM/session independence, and a
    sub-0.70 is partly LLM-attributable (must be stated, not swept)."""
    y2 = np.asarray(y2); g2 = np.asarray(g2)
    n_boost = n_nat = n_unknown = 0
    drivers = set()
    for role, trip in zip(y2, g2):
        if role not in SPECIALISTS:
            continue
        sc = Path(r2) / f"{trip}.json"
        boosted = drv = None
        if sc.exists():
            try:
                d = json.loads(sc.read_text()); boosted = d.get("boosted"); drv = d.get("driver")
            except Exception:
                pass
        if drv:
            drivers.add(drv)
        if boosted is True:
            n_boost += 1
        elif boosted is False:
            n_nat += 1
        else:
            n_unknown += 1
    total = n_boost + n_nat + n_unknown
    if total == 0:
        mode = "none"
    elif n_boost > 0:
        mode = "boosted"          # ANY boosted specialist ⇒ treat the set as driver-affected (safe)
    elif n_nat > 0:
        mode = "natural"          # explicit-natural, no boosted present
    else:
        mode = "boosted"          # all legacy/unknown ⇒ the committed boosted run (conservative)
    return {"mode": mode, "n_specialist_samples": int(total), "n_boosted": int(n_boost),
            "n_natural": int(n_nat), "n_unknown_legacy": int(n_unknown), "drivers": sorted(drivers)}


def main(args: argparse.Namespace) -> None:
    r1, r2 = Path(args.inst1), Path(args.inst2)
    if not any(r1.glob("*.pcap")):
        raise SystemExit(f"blocked: no instance-1 pcaps at {r1}")
    if not any(r2.glob("*.pcap")):
        raise SystemExit(f"blocked: no instance-2 pcaps at {r2} — run collect_offtheshelf_inst2.sh first")

    X1, y1, g1 = extract_role_samples(r1)
    X2, y2, g2 = extract_role_samples(r2)
    c1, c2 = counts(y1), counts(y2)
    logger.info("instance-1 roles: %s", c1)
    logger.info("instance-2 roles: %s", c2)

    common = sorted({r for r in set(c1) & set(c2) if c1[r] >= MIN_N and c2[r] >= MIN_N})
    if len(common) < 2:
        raise SystemExit(f"blocked: <2 roles shared with ≥{MIN_N} samples in both instances "
                         f"(inst1={c1}, inst2={c2}). Collect more instance-2 trips.")

    X1c, y1c, g1c = restrict(X1, y1, g1, common)
    X2c, y2c, g2c = restrict(X2, y2, g2, common)

    # 6-way (whatever roles are common) — both directions.
    f_1to2 = transfer(X1c, y1c, X2c, y2c, g2c, f"{len(common)}-way inst1→inst2")
    f_2to1 = transfer(X2c, y2c, X1c, y1c, g1c, f"{len(common)}-way inst2→inst1")
    weak = min(f_1to2["macro_f1"], f_2to1["macro_f1"])
    weak_dir = f_1to2 if f_1to2["macro_f1"] <= f_2to1["macro_f1"] else f_2to1
    verdict = band(weak, weak_dir["ci_lo"], weak_dir["chance"])

    # Clean coordinator-only 3-way (mcp/orchestrator/planner) — these fire on EVERY trip so their
    # samples are NATURAL in both instances (no boosted driver), giving the unconfounded deployable
    # result to contrast against the (driver-boosted) 6-way. Reported whenever ≥2 coordinators qualify.
    coord_roles = [r for r in common if r in ("mcp", "orchestrator", "planner")]
    coord_out = None
    if len(coord_roles) >= 2:
        X1k, y1k2, g1k = restrict(X1, y1, g1, coord_roles); X2k, y2k2, g2k = restrict(X2, y2, g2, coord_roles)
        cc12 = transfer(X1k, y1k2, X2k, y2k2, g2k, f"{len(coord_roles)}-way COORD inst1→inst2")
        cc21 = transfer(X2k, y2k2, X1k, y1k2, g1k, f"{len(coord_roles)}-way COORD inst2→inst1")
        cweak = min(cc12["macro_f1"], cc21["macro_f1"])
        cwd = cc12 if cc12["macro_f1"] <= cc21["macro_f1"] else cc21
        coord_out = {
            "note": "NATURAL both instances (coordinators fire on every trip; no boosted driver) — "
                    "the clean, unconfounded deployable result.",
            "roles": coord_roles, "inst1_to_inst2": cc12, "inst2_to_inst1": cc21,
            "weaker_direction_macro_f1": cweak, "verdict": band(cweak, cwd["ci_lo"], cwd["chance"]),
        }

    # ── VOLUME ABLATION on the clean coordinator transfer (§9a gate) ──────────────
    # Same data, same pipeline; only the feature columns change. Drop every raw
    # count/volume dimension (_VOLUME_FEATURES) and re-run _transfer both directions.
    # Verdict rule (no re-stamp): shape-only weaker-direction ≥0.70 → the coordinator
    # attack is BEHAVIOURAL (shape), "DEPLOYABLE" stands with evidence; <0.70 → it is
    # VOLUME-DRIVEN, reframe §9a as "coordinator structure is readable across instances"
    # and qualify "deployable". Number reported as-is either way.
    shape_out = None
    if coord_out is not None:
        X1k, y1k2, g1k = restrict(X1, y1, g1, coord_roles); X2k, y2k2, g2k = restrict(X2, y2, g2, coord_roles)
        X1s, X2s = X1k[:, _SHAPE_MASK], X2k[:, _SHAPE_MASK]
        sc12 = transfer(X1s, y1k2, X2s, y2k2, g2k, f"{len(coord_roles)}-way COORD shape-only inst1→inst2")
        sc21 = transfer(X2s, y2k2, X1s, y1k2, g1k, f"{len(coord_roles)}-way COORD shape-only inst2→inst1")
        sweak = min(sc12["macro_f1"], sc21["macro_f1"])
        swd = sc12 if sc12["macro_f1"] <= sc21["macro_f1"] else sc21
        sverdict = band(sweak, swd["ci_lo"], swd["chance"])
        behavioural = sweak >= 0.70 and swd["ci_lo"] > swd["chance"]
        shape_out = {
            "purpose": "Volume ablation on the §9a coordinator transfer — is DEPLOYABLE behavioural "
                       "(shape) or the demoted connection-volume signal? Parallel to the framework-ID "
                       "timing ablation (0.46→0.38).",
            "roles": coord_roles,
            "features_kept": _KEPT_FEATURES,
            "features_dropped": _DROPPED_FEATURES,
            "n_features_kept": int(_SHAPE_MASK.sum()),
            "n_features_dropped": int((~_SHAPE_MASK).sum()),
            "inst1_to_inst2": sc12, "inst2_to_inst1": sc21,
            "weaker_direction_macro_f1": sweak,
            "verdict": sverdict,
            "interpretation": (
                "SHAPE-ONLY weaker direction ≥0.70 with CI clear of chance: the coordinator transfer "
                "is BEHAVIOURAL (per-agent traffic shape/timing), not merely the connection-volume "
                "signal — 'DEPLOYABLE' in §9a stands with ablation evidence."
                if behavioural else
                "SHAPE-ONLY weaker direction <0.70 (or CI touches chance): the coordinator transfer "
                "rides substantially on connection VOLUME (the signal already demoted to a topology "
                "baseline). Reframe §9a as 'coordinator STRUCTURE is readable across instances' and "
                "qualify 'deployable' accordingly."),
        }
        logger.info("[COORD shape-only] weaker=%.3f -> %s", sweak, sverdict)

    # coordinator-vs-specialist 2-way (partly structural) — both directions, if both classes exist.
    coarse_out = None
    y1k = np.array([coarse(r) for r in y1]); y2k = np.array([coarse(r) for r in y2])
    if len({*y1k}) == 2 and len({*y2k}) == 2:
        c_1to2 = transfer(X1, y1k, X2, y2k, g2, "coord-vs-spec inst1→inst2")
        c_2to1 = transfer(X2, y2k, X1, y1k, g1, "coord-vs-spec inst2→inst1")
        coarse_out = {
            "caveat": "PARTLY STRUCTURAL — hubs carry more traffic than leaves, so this rides on "
                      "connection volume (like topology), not subtle per-agent behaviour. The "
                      "behavioural headline is the multi-role transfer above.",
            "inst1_to_inst2": c_1to2, "inst2_to_inst1": c_2to1,
        }

    # ── SPECIALIST DISTRIBUTION CHECK = the GATE for the 6-way band (required addition) ──
    # Compare inst-2 specialists' per-agent feature distributions to inst-1's (always natural).
    # The MEANING of the 6-way band is conditional on this, and on HOW inst-2's specialists were
    # collected (boosted vs natural):
    #   * boosted  → a driver artefact must not masquerade as a positive OR a boundary finding.
    #   * natural  → the confound is removed; any residual gap is the legitimate LLM/session
    #                independence, so a sub-0.70 is PARTLY LLM-attributable, not a clean
    #                "specialist behaviour does not transfer". Stated, not swept.
    dist_check = specialist_distribution_check(X1, y1, X2, y2)
    specialists_in_way = [r for r in common if r in SPECIALISTS]
    all_comparable = dist_check.get("_summary", {}).get("all_comparable", False)
    spec_mode = inst2_specialist_mode(r2, y2, g2)
    mode = spec_mode["mode"]

    if not specialists_in_way:
        driver_interpretation = ("No specialists met the ≥%d bar — coordinator-layer result; the "
                                 "specialist-collection axis does not apply." % MIN_N)
    elif mode == "boosted":
        driver_interpretation = (
            ("6-way ≥0.70 WITH specialists. Instance-2's specialists were collected with the BOOSTED "
             "driver, which if anything makes this a HARDER test (train inst-1 NATURAL, test inst-2 "
             "FORCED). " + ("Distributions are COMPARABLE — the positive is robust, not a driver "
             "artefact." if all_comparable else "BUT distributions are NOT all comparable — the "
             "boosted driver may inflate similarity; treat with caution."))
            if weak >= 0.70 else
            ("6-way <0.70 WITH specialists collected by the BOOSTED driver — a POSSIBLE CONTRIBUTOR "
             "to the drop (train-natural / test-forced mismatch), NOT necessarily 'behaviour doesn't "
             "transfer'; named as a candidate confound. " + ("However distributions are comparable, "
             "arguing against the driver being the whole story." if all_comparable else
             "Distributions differ, consistent with the driver contributing — confounded here. Run "
             "the NATURAL collection (scripts/collect_offtheshelf_natural.sh) to de-confound.")))
    elif mode in ("natural", "mixed"):
        prefix = ("MIXED natural+boosted specialists — read with the boosted caveat; a fully-natural "
                  "re-collection is cleaner. " if mode == "mixed"
                  else "NATURAL specialists — the boosted-driver confound is REMOVED. ")
        if all_comparable:
            driver_interpretation = prefix + (
                "Specialist distributions are COMPARABLE to instance-1 (specialist_distribution_check) "
                "⇒ the confound is gone and the pre-registered §4 band means what it says: "
                + ("≥0.70 → CLEAN full-6-way cross-instance transfer including specialists."
                   if weak >= 0.70 else
                   "<0.70 → a GENUINE partial/bounded specialist transfer (real instance drift), not "
                   "a driver artefact."))
        else:
            driver_interpretation = prefix + (
                "Specialist distributions are NOT all comparable to instance-1. With the boosted "
                "driver removed, the residual difference is the LEGITIMATE independence axes "
                "(different LLM gemini-2.0-flash, separate session), NOT a driver artefact: "
                + ("≥0.70 → transfers despite that independence gap → robust."
                   if weak >= 0.70 else
                   "<0.70 → the drop is PARTLY LLM/SESSION-ATTRIBUTABLE, not a clean 'specialist "
                   "behaviour does not transfer'. This is STATED, not swept."))
    else:
        driver_interpretation = "No specialist samples found to classify collection mode."

    out = {
        "task": "cross-INSTANCE role transfer on a2a_mcp (Phase 2 — deployable-attack test)",
        "hypothesis": "two independent instances of the SAME framework share call structure and "
                      "differ only in surface variables (LLM, prompts, session, driver), so role "
                      "transfer SHOULD work — a prediction, not a guarantee.",
        "independence_of_instance2": {
            "different_llm": "gemini-2.0-flash (instance 1 = gemini-2.5-flash), via LITELLM_MODEL",
            "different_prompts": "reworded query template, different dates/party-size/class/nights",
            "separate_session": True,
            "specialist_collection": ("BOOSTED (drive_orch_boost.py + forced full-service prompt) — a "
                "driver confound; see driver_confound_interpretation + specialist_distribution_check."
                if mode == "boosted" else
                "NATURAL (drive_orch_natural_fixed.py: bug-fixed-but-not-forced, instance-2's own "
                "reworded prompt) — the driver confound is REMOVED; residual gap = legitimate LLM/"
                "session independence. See specialist_distribution_check (the band's gate)."
                if mode == "natural" else
                "MIXED natural+boosted specialist samples — interpret with caution."),
            "shared": "the a2a_mcp framework's fixed six roles by port (10100-10105)",
        },
        "specialist_collection_mode": spec_mode,
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
        "coordinator_shape_only_ablation": shape_out,
        "coordinator_vs_specialist": coarse_out,
        "specialist_distribution_check": dist_check,
        "driver_confound_interpretation": driver_interpretation,
        "verdict_phase2": verdict,
        "verdict_basis": "weaker of the two directions (brief §4); verdict field matches the number. "
                         "For the 6-way the band's MEANING is gated on specialist_distribution_check + "
                         "specialist_collection_mode (boosted → driver confound; natural → residual gap "
                         "is legitimate LLM/session independence).",
        "caveats": ("Single second instance. Instance-2 specialists collected via " + mode +
                    " driver (see specialist_collection_mode); coordinator samples are natural in "
                    "both. The specialist_distribution_check is the gate for reading the 6-way band."),
    }

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = os.environ.get("CIT_OUT", "cross_instance_transfer.json")   # natural run overrides
    (out_dir / out_name).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 74)
    print("  PHASE 2 — CROSS-INSTANCE ROLE TRANSFER (a2a_mcp inst1 ⇄ inst2)")
    print("=" * 74)
    print(f"  common roles (≥{MIN_N} in both): {common}  ({len(common)}-way)")
    print(f"  inst1→inst2  macro-F1 = {f_1to2['macro_f1']:.3f} [{f_1to2['ci_lo']:.3f},{f_1to2['ci_hi']:.3f}]"
          f"  (n_test={f_1to2['n_test']}, chance={f_1to2['chance']:.3f})")
    print(f"  inst2→inst1  macro-F1 = {f_2to1['macro_f1']:.3f} [{f_2to1['ci_lo']:.3f},{f_2to1['ci_hi']:.3f}]"
          f"  (n_test={f_2to1['n_test']}, chance={f_2to1['chance']:.3f})")
    print(f"  weaker direction = {weak:.3f}")
    # ── distribution check reported BEFORE the verdict — it gates the band's meaning ──
    if specialists_in_way:
        print("  ── specialist distribution GATE (inst-1 vs inst-2 specialists) ──")
        print(f"     collection mode: {mode.upper()}  "
              f"(boosted={spec_mode['n_boosted']} natural={spec_mode['n_natural']} "
              f"legacy={spec_mode['n_unknown_legacy']})")
        for sp in SPECIALISTS:
            dc = dist_check.get(sp, {})
            if "median_abs_smd" in dc:
                print(f"     {sp:13s} median|SMD|={dc['median_abs_smd']:.2f} cos={dc['mean_vector_cosine']:.3f}"
                      f"  comparable={dc['comparable']}")
        print(f"     specialists comparable: {dist_check.get('_summary', {}).get('specialists_comparable')}"
              f"  -> band is {'CLEAN' if all_comparable else 'CONDITIONAL (see interpretation)'}")
    print(f"  VERDICT (§4): {verdict}")
    if coord_out:
        print(f"  coordinator 3-way (natural)      weaker = {coord_out['weaker_direction_macro_f1']:.3f}"
              f"  -> {coord_out['verdict'].split(' (')[0]}")
    if shape_out:
        print(f"  coordinator 3-way SHAPE-ONLY     weaker = {shape_out['weaker_direction_macro_f1']:.3f}"
              f"  ({shape_out['n_features_kept']}/35 feats)  -> {shape_out['verdict'].split(' (')[0]}")
    print("=" * 74)
    print(f"\nWrote {out_dir / out_name}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 cross-instance role transfer on a2a_mcp")
    p.add_argument("--inst1", default="data/raw_offtheshelf")
    p.add_argument("--inst2", default="data/raw_offtheshelf_inst2")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
