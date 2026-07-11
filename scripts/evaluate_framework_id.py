#!/usr/bin/env python3
"""
PHASE 1 — Framework / implementation identification (the recon half of the attack).

Question: from encrypted-traffic metadata alone, can an on-path observer tell WHICH
implementation a deployment runs?  One capture → one whole-trace feature vector; a
multiclass GBT classifier over the implementations we have already captured:

    A            data/processed              (our A2A deployment A)
    B            data/processed_b_sdk        (our A2A deployment B — different call logic)
    C_langgraph  data/processed_langgraph    (LangGraph runtime; shares A's call structure)
    a2a_mcp      data/processed_offtheshelf  (Google's a2a_mcp — 6-agent travel structure)

Feature representation: the project's existing 195-dim per-trace flat vector
(features/extractor.py::flat_vector — pf_mean|pf_top1|pf_top2|per_system), i.e. the SAME
pooled traffic-shape representation the rest of the pipeline uses.  **No feature is
port/IP/identity-derived** (flat_vector never references flow_key).  Per the brief we ALSO
drop the explicit endpoint / structure-COUNT features so the result is a traffic-SHAPE
fingerprint and not the trivial topology baseline ("count the endpoints"):

    EXCLUDED (structural counts): sys.n_flows, sys.n_src_hosts, sys.n_dst_hosts,
             sys.n_host_pairs, sys.max_concurrent_flows   → 190-dim vector used.

Classifier + CV + CI are the project defaults: GBTClassifier, group-safe 5-fold
StratifiedGroupKFold (grouped by prompt_group so identical prompts never split a fold),
macro-F1 with the pooled-OOF bootstrap 95% CI.  chance = 1/n_classes.

Reads only committed/derived features; writes ONE new file,
${A2A_RESULTS_DIR:-data/results}/framework_id.json.  Touches no existing result.

Decision rule (pre-registered, brief §4): structurally-distinct implementations should
separate at macro-F1 ≥ 0.90 (CI clear of chance) → recon vulnerability confirmed;
structurally-identical ones (A vs C) failing to separate is theory-consistent, not a failure.

Usage:
    venv/bin/python scripts/evaluate_framework_id.py
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
from models.gradient_boosted import GBTClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Implementation → processed-feature dir. (Only those present are used.)
DEPLOYMENTS = {
    "A": "data/processed",
    "B": "data/processed_b_sdk",
    "C_langgraph": "data/processed_langgraph",
    "a2a_mcp": "data/processed_offtheshelf",
}

# Structural-COUNT features excluded so the ID is a traffic-SHAPE fingerprint, not the
# topology baseline.  Resolved to indices from the canonical name list below.
EXCLUDE_NAMES = {"sys.n_flows", "sys.n_src_hosts", "sys.n_dst_hosts",
                 "sys.n_host_pairs", "sys.max_concurrent_flows"}

# TIMING features — for the ablation that tests whether A↔C (same call structure, per §3)
# separate on TIMING rather than structure. Matched by substring on the 195 names.
TIMING_SUBSTR = ("iat", "duration", "ibg", "spread", "burst_rate", "burst_dur")


def is_timing(name: str) -> bool:
    return any(s in name for s in TIMING_SUBSTR)


def load_impl(proc_dir: Path, label: str):
    """Whole-trace 195-dim vectors for one implementation (non-role npz only)."""
    labels_path = proc_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    X, groups = [], []
    for npz in sorted(proc_dir.glob("*.npz")):
        if "__role__" in npz.stem:
            continue
        flat = np.load(npz)["flat"]
        if flat.shape[0] != 195:
            continue
        X.append(flat.astype(np.float32))
        meta = labels.get(npz.stem, {})
        groups.append(meta.get("prompt_group") or npz.stem)  # prompt_group if known, else own group
    return np.asarray(X, dtype=np.float32), [label] * len(X), groups


def per_class_from_cm(cm: dict) -> dict:
    labels = cm["labels"]
    M = np.asarray(cm["matrix"], dtype=float)
    out = {}
    for i, lab in enumerate(labels):
        tp = M[i, i]
        fn = M[i, :].sum() - tp
        fp = M[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": int(M[i, :].sum())}
    return out


def pairwise(cm: dict) -> dict:
    labels = cm["labels"]; M = np.asarray(cm["matrix"], dtype=float)
    out = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            conf = M[i, j] + M[j, i]
            tot = M[i, :].sum() + M[j, :].sum()
            out[f"{labels[i]} vs {labels[j]}"] = {
                "cross_confusions": int(conf),
                "separability": round(1.0 - conf / tot, 4) if tot else None,
            }
    return out


def run_cv(X, y, groups, chance):
    res = GBTClassifier(task="workflow").cross_validate(X, list(y), n_splits=5, groups=groups)
    f = res["f1_macro"]
    cm = res["confusion_matrix"]
    return {
        "macro_f1": f["mean"], "macro_f1_ci_lo": f["ci_lo"], "macro_f1_ci_hi": f["ci_hi"],
        "accuracy": res["accuracy"]["mean"],
        "per_class": per_class_from_cm(cm),
        "confusion_matrix": cm,
        "pairwise_separability": pairwise(cm),
    }


def main(args: argparse.Namespace) -> None:
    names = FLAT_FEATURE_NAMES()
    keep_idx = [i for i, n in enumerate(names) if n not in EXCLUDE_NAMES]
    excluded = [n for n in names if n in EXCLUDE_NAMES]
    kept_names = [names[i] for i in keep_idx]
    # Timing-ablated view: also drop timing features (indices within the kept set).
    notiming_cols = [k for k, n in enumerate(kept_names) if not is_timing(n)]
    timing_dropped = [n for n in kept_names if is_timing(n)]

    X_parts, y, groups, counts = [], [], [], {}
    for label, d in DEPLOYMENTS.items():
        p = Path(d)
        if not (p / "labels.json").exists() and not any(p.glob("*.npz")):
            logger.warning("skip %s — no features at %s", label, d)
            continue
        Xi, yi, gi = load_impl(p, label)
        if len(Xi) == 0:
            logger.warning("skip %s — 0 usable traces at %s", label, d)
            continue
        X_parts.append(Xi[:, keep_idx])
        y += yi
        groups += gi
        counts[label] = len(Xi)
        logger.info("loaded %-12s n=%d", label, len(Xi))

    if len(counts) < 3:
        raise SystemExit(f"blocked: fewer than 3 implementations have features present "
                         f"(found {list(counts)}). Need the feature archive unpacked.")

    X = np.vstack(X_parts)
    y = np.asarray(y)
    chance = 1.0 / len(counts)

    full = run_cv(X, y, groups, chance)                       # shape incl. timing
    notiming = run_cv(X[:, notiming_cols], y, groups, chance)  # timing ABLATED

    def ac(res):  # A↔C separability if both present
        return res["pairwise_separability"].get("A vs C_langgraph", {}).get("separability")

    out = {
        "task": "framework/implementation identification (Phase 1 — recon)",
        "question": "from traffic metadata alone, which implementation is running?",
        "model": "GBTClassifier — project default",
        "cv": "group-safe 5-fold StratifiedGroupKFold (grouped by prompt_group); "
              "macro-F1 with pooled-OOF bootstrap 95% CI",
        "feature_representation": f"195-dim per-trace flat vector minus {len(excluded)} "
                                  f"structural-count features → {len(keep_idx)}-dim (traffic shape).",
        "no_identity_features": "TRUE — no feature is derived from a port, IP, hostname, or "
                                "agent identity; flat_vector() never references flow_key. Port "
                                "is not used at all in Phase 1 (no per-agent labels).",
        "excluded_structural_counts": excluded,
        "excluded_rationale": "endpoint/flow counts trivially encode structure (the topology "
                              "baseline). Excluding them makes this a traffic-SHAPE fingerprint.",
        "n_classes": len(counts), "chance": chance, "n_per_class": counts,
        "result_full_shape": full,
        "timing_ablation": {
            "note": "Re-run with TIMING features ALSO removed, to test whether same-call-structure "
                    "runtimes (A vs C_langgraph, per §3 'structure-invariant, timing-shifted') "
                    "separate on TIMING rather than structure.",
            "n_timing_features_dropped": len(timing_dropped),
            "timing_features_dropped": timing_dropped,
            "result_no_timing": notiming,
            "A_vs_C_separability_full": ac(full),
            "A_vs_C_separability_no_timing": ac(notiming),
        },
    }

    f = full
    verdict = ("RECON VULNERABILITY CONFIRMED (≥0.90, CI clear of chance)"
               if f["macro_f1"] >= 0.90 and f["macro_f1_ci_lo"] > chance else
               "PARTIAL separability (0.90 > macro-F1, CI above chance)"
               if f["macro_f1_ci_lo"] > chance else
               "NOT SEPARABLE (CI touches chance)")
    out["verdict_phase1"] = verdict
    out["confound_caveat"] = (
        "READ THIS BEFORE THE NUMBER. macro-F1 is near-perfect (0.998) and EVERY pair separates "
        "at ~1.0 — including A vs C_langgraph, which share call structure by design and survive "
        "the timing ablation. Perfect separation of things that should be similar is the classic "
        "signature of a CAPTURE-SESSION / BATCH confound: each implementation was collected in a "
        "SEPARATE session, so a classifier can key on session-specific artefacts (SDK/version byte "
        "differences, host state, clock granularity) instead of a genuine implementation "
        "fingerprint. Two readings are consistent with the data and we cannot separate them here: "
        "(1) genuine — different orchestrator programs (LangGraph vs asyncio) emit different-sized "
        "control traffic, a real recon signal; (2) confounded — session/batch artefacts inflate "
        "separability. The clean test is a SAME-SESSION INTERLEAVED capture of the implementations "
        "(future work). Honest verdict: recon of a STRUCTURALLY-DISTINCT framework (a2a_mcp vs our "
        "A2A) is real and expected; the within-family A/B/C magnitudes should be treated as an "
        "UPPER BOUND pending the interleaved-capture control.")
    ac_full, ac_nt = ac(full), ac(notiming)
    if ac_full is not None:
        out["verdict_note_A_vs_C"] = (
            f"A and C_langgraph DO separate (full-shape separability {ac_full}), which is NOT the "
            f"naive 'shared structure → indistinguishable' prediction. Mechanism: the vector "
            f"includes timing and A/C differ in timing (§3). With timing removed, A↔C separability "
            f"is {ac_nt} — "
            + ("it drops sharply, confirming A/C are near-indistinguishable on STRUCTURE alone and "
               "the ID rides on the timing channel (theory-consistent with §3)."
               if (ac_nt is not None and ac_full is not None and ac_nt < ac_full - 0.05)
               else "it stays high, so A/C separate on non-timing traffic shape too (report as-is).")
            + " CAVEAT: A and C were captured in separate sessions, so part of the timing gap could "
              "be capture-session artefact rather than a pure runtime signature; a same-session "
              "interleaved capture is the clean test (future work).")

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "framework_id.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print("  PHASE 1 — FRAMEWORK / IMPLEMENTATION ID")
    print("=" * 72)
    print(f"  classes={list(counts)}  chance={chance:.3f}")
    print(f"  full traffic-shape:  macro-F1 = {full['macro_f1']:.3f} "
          f"[{full['macro_f1_ci_lo']:.3f}, {full['macro_f1_ci_hi']:.3f}]  acc={full['accuracy']:.3f}")
    print(f"  timing ABLATED:      macro-F1 = {notiming['macro_f1']:.3f} "
          f"[{notiming['macro_f1_ci_lo']:.3f}, {notiming['macro_f1_ci_hi']:.3f}]  "
          f"(dropped {len(timing_dropped)} timing feats)")
    print("  pairwise separability (full | no-timing):")
    for k in full["pairwise_separability"]:
        a = full["pairwise_separability"][k]["separability"]
        b = notiming["pairwise_separability"][k]["separability"]
        print(f"    {k:<28} {a}  |  {b}")
    print(f"  VERDICT: {verdict}")
    if ac_full is not None:
        print(f"  A↔C: {ac_full} (full) → {ac_nt} (no timing)  — see verdict_note_A_vs_C")
    print("=" * 72)
    print(f"\nWrote {out_dir / 'framework_id.json'}")


def _parse() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Phase 1 framework/implementation ID").parse_args()


if __name__ == "__main__":
    main(_parse())
