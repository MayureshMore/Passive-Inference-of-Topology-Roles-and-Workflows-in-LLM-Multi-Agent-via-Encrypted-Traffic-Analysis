#!/usr/bin/env python3
"""
Runtime-invariance traffic diagnostic — deployment A vs deployment C (LangGraph).

WHAT THIS IS FOR
────────────────
Deployment C re-implements deployment A's orchestrator in LangGraph while reusing
A's specialists, call structure, and task prompts unchanged — so only the runtime
that *schedules* the inter-agent calls differs.  The cross-deployment classifier
transfers A→C near A's within-deployment ceiling (cross_framework.json).  Before
that result can be written up, we must rule out the trivial explanation that the
two runtimes simply emit byte-identical traffic (in which case "transfer" is
vacuous).  Conversely, if C has a *distinct* wire signature yet the fingerprint
still transfers, the runtime-invariance claim is stronger.

This script characterises, per trace, the wire-level shape that an on-path
observer sees, directly from the FROZEN 195-dim flat feature vectors (the same
artefact every model trains on), and quantifies how far A and C diverge:

  flow count            sys.n_flows
  packet count          sys.total_packets
  total bytes           sys.total_bytes
  trace duration (s)    sys.total_duration_s
  orchestrator-flow shape:
      max_concurrent_flows        fan-out degree (parallel vs sequential dispatch)
      heaviest_flow_bytes_frac    byte concentration on the dominant flow
      flow_bytes_cv               per-flow byte dispersion (star=low, chain=high)
      bytes_out_ratio             outbound/total byte ratio
      mean_flow_response_ratio    mean inbound/total per flow

Comparisons (most→least confounded):
  by_cell      matched (workflow × topology) cells — same prompts, same structure;
               the cleanest apples-to-apples and the basis for the verdict.
  by_topology  pooled within each topology (controls the dominant structural var).
  overall      all per-trace samples (label mixes are balanced & matched, so this
               is fair, but the by_cell numbers are load-bearing).

Divergence measures, per feature:
  ks_stat / ks_pvalue   two-sample Kolmogorov–Smirnov (distribution shape).
  cohens_d              standardised mean difference (effect size).  With N in the
                        hundreds, KS p-values go ~0 for any real gap, so the
                        EFFECT SIZE — not the p-value — drives interpretation.
  median_ratio_c_over_a + 95% bootstrap CI (seed=42, additive-invariant safe).

Honesty note: this is a CONTROL, not a generalization result.  C is a runtime
variant of A, not an independently-authored framework.

Usage:
    venv/bin/python scripts/diagnose_runtime_traffic.py
    venv/bin/python scripts/diagnose_runtime_traffic.py --dir-a data/processed \
        --dir-c data/processed_langgraph --out data/results/runtime_traffic_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SEED = 42
_N_BOOTSTRAP = 1000

# Headline wire features (flat-vector names → human description).  Resolved to
# column indices at runtime via features.names.FLAT_FEATURE_NAMES().
_HEADLINE: list[tuple[str, str]] = [
    ("sys.n_flows", "flow count"),
    ("sys.total_packets", "packet count"),
    ("sys.total_bytes", "total bytes"),
    ("sys.total_duration_s", "trace duration (s)"),
    ("sys.max_concurrent_flows", "max concurrent flows (fan-out degree)"),
    ("sys.heaviest_flow_bytes_frac", "heaviest-flow byte fraction (orch-flow concentration)"),
    ("sys.flow_bytes_cv", "per-flow byte CV (star low / chain high)"),
    ("sys.bytes_out_ratio", "outbound byte ratio"),
    ("sys.mean_flow_response_ratio", "mean per-flow response ratio"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_trace(processed_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the 195-dim per-trace flat matrix + (workflow, topology) labels.

    Mirrors evaluate_cross_deployment.load_deployment's filtering: per-trace NPZs
    only (skip ``__role__`` agent vectors), 195-dim flat only.
    """
    labels_map: dict[str, dict] = json.loads((processed_dir / "labels.json").read_text())
    X_list: list[np.ndarray] = []
    wf_list: list[str] = []
    tp_list: list[str] = []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        run_id = npz_path.stem
        if "__role__" in run_id:
            continue
        info = labels_map.get(run_id)
        if not info or info.get("workflow") is None:
            continue
        flat = np.load(npz_path, allow_pickle=False)["flat"]
        if flat.shape[0] != 195:
            continue
        X_list.append(flat.astype(np.float64))
        wf_list.append(str(info["workflow"]))
        tp_list.append(str(info.get("topology", "unknown")))
    if not X_list:
        raise ValueError(f"No per-trace samples in {processed_dir}")
    logger.info("Loaded %d per-trace samples from %s", len(X_list), processed_dir.name)
    return np.stack(X_list, axis=0), wf_list, tp_list


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _summary(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (a−b) with pooled SD; 0 if no variance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    sd = float(np.sqrt(pooled))
    if sd < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sd)


def _effect(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _median_ratio_ci(a_col: np.ndarray, c_col: np.ndarray) -> tuple[float, list[float]]:
    """median(C)/median(A) + 95% bootstrap CI (seed=42, deterministic)."""
    a_col = np.asarray(a_col, dtype=np.float64)
    c_col = np.asarray(c_col, dtype=np.float64)
    med_a = float(np.median(a_col))
    if abs(med_a) < 1e-12:
        return float("nan"), [float("nan"), float("nan")]
    point = float(np.median(c_col) / med_a)
    rng = np.random.default_rng(_SEED)
    ratios = np.empty(_N_BOOTSTRAP, dtype=np.float64)
    for i in range(_N_BOOTSTRAP):
        ba = rng.choice(a_col, size=a_col.size, replace=True)
        bc = rng.choice(c_col, size=c_col.size, replace=True)
        ma = np.median(ba)
        ratios[i] = (np.median(bc) / ma) if abs(ma) > 1e-12 else np.nan
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return point, [float(lo), float(hi)]


def _ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp
    r = ks_2samp(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return float(r.statistic), float(r.pvalue)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dir_a: Path, dir_c: Path, out: Path) -> None:
    from features.names import FLAT_FEATURE_NAMES

    names = FLAT_FEATURE_NAMES()
    idx = {name: names.index(name) for name, _ in _HEADLINE}

    Xa, wfa, tpa = load_per_trace(dir_a)
    Xc, wfc, tpc = load_per_trace(dir_c)
    wfa, tpa = np.array(wfa), np.array(tpa)
    wfc, tpc = np.array(wfc), np.array(tpc)

    def _counts(wf, tp) -> dict:
        from collections import Counter
        return {"n": int(len(wf)), "workflow": dict(Counter(wf.tolist())),
                "topology": dict(Counter(tp.tolist()))}

    result: dict = {
        "meta": {
            "generated_by": "scripts/diagnose_runtime_traffic.py",
            "dir_a": str(dir_a), "dir_c": str(dir_c),
            "deployment_a": "A — asyncio.gather orchestrator (agents/)",
            "deployment_c": "C — LangGraph StateGraph orchestrator (agents_langgraph/)",
            "control_note": (
                "C reuses A's specialists, call structure, and task prompts unchanged; "
                "only the orchestration runtime differs. This is a runtime-invariance "
                "CONTROL, not a generalization result."
            ),
            "seed": _SEED, "n_bootstrap": _N_BOOTSTRAP,
            "features": {name: desc for name, desc in _HEADLINE},
        },
        "label_counts": {"a": _counts(wfa, tpa), "c": _counts(wfc, tpc)},
        "overall": {},
        "by_topology": {},
        "by_cell": {},
    }

    # ── overall (matched, balanced mixes) ──────────────────────────────────────
    for name, desc in _HEADLINE:
        a = Xa[:, idx[name]]
        c = Xc[:, idx[name]]
        ks_s, ks_p = _ks(a, c)
        d = _cohens_d(c, a)  # sign: C relative to A
        ratio, ci = _median_ratio_ci(a, c)
        result["overall"][name] = {
            "desc": desc,
            "a": _summary(a), "c": _summary(c),
            "ks_stat": ks_s, "ks_pvalue": ks_p,
            "cohens_d_c_minus_a": d, "effect": _effect(d),
            "median_ratio_c_over_a": ratio, "median_ratio_ci95": ci,
        }

    # ── by topology (controls the dominant structural variable) ────────────────
    for topo in sorted(set(tpa.tolist()) & set(tpc.tolist())):
        ma = tpa == topo
        mc = tpc == topo
        block: dict = {}
        for name, _ in _HEADLINE:
            a = Xa[ma, idx[name]]
            c = Xc[mc, idx[name]]
            ks_s, ks_p = _ks(a, c)
            d = _cohens_d(c, a)
            block[name] = {
                "a_mean": float(np.mean(a)), "c_mean": float(np.mean(c)),
                "a_median": float(np.median(a)), "c_median": float(np.median(c)),
                "ks_stat": ks_s, "ks_pvalue": ks_p,
                "cohens_d_c_minus_a": d, "effect": _effect(d),
            }
        result["by_topology"][topo] = {"n_a": int(ma.sum()), "n_c": int(mc.sum()), **block}

    # ── by matched (workflow × topology) cell — load-bearing comparison ─────────
    # Decompose effect sizes into STRUCTURE (volume/shape — what a static observer
    # reads from sizes & connection graph) vs TIMING (wall-clock duration — runtime
    # scheduling/host load).  The runtime-invariance claim lives in the structural
    # channel; timing is expected to shift with the scheduler.
    timing_features = {"sys.total_duration_s"}
    struct_abs_d: list[float] = []
    timing_abs_d: list[float] = []
    cells = sorted(
        {(w, t) for w, t in zip(wfa.tolist(), tpa.tolist())}
        & {(w, t) for w, t in zip(wfc.tolist(), tpc.tolist())}
    )
    for wf, tp in cells:
        ma = (wfa == wf) & (tpa == tp)
        mc = (wfc == wf) & (tpc == tp)
        if ma.sum() < 2 or mc.sum() < 2:
            continue
        cell: dict = {}
        for name, _ in _HEADLINE:
            a = Xa[ma, idx[name]]
            c = Xc[mc, idx[name]]
            d = _cohens_d(c, a)
            (timing_abs_d if name in timing_features else struct_abs_d).append(abs(d))
            med_a = float(np.median(a))
            cell[name] = {
                "a_median": med_a, "c_median": float(np.median(c)),
                "ratio_c_over_a": (float(np.median(c) / med_a) if abs(med_a) > 1e-12 else None),
                "cohens_d_c_minus_a": d, "effect": _effect(d),
            }
        result["by_cell"][f"{wf}|{tp}"] = {"n_a": int(ma.sum()), "n_c": int(mc.sum()), **cell}

    # ── verdict (structure drives it; timing reported separately) ──────────────
    s_arr = np.array(struct_abs_d) if struct_abs_d else np.array([0.0])
    t_arr = np.array(timing_abs_d) if timing_abs_d else np.array([0.0])
    struct_max = float(s_arr.max())
    struct_mean = float(s_arr.mean())
    timing_max = float(t_arr.max())
    timing_mean = float(t_arr.mean())

    timing_shifted = timing_mean >= 0.5

    if struct_max < 0.5:
        headline = "structure-invariant" + ("-timing-shifted" if timing_shifted else "")
        recommendation = (
            "Within every matched (workflow x topology) cell, all VOLUME/STRUCTURE wire "
            "features (bytes, packets, flow count, fan-out degree, byte concentration, "
            "directionality) are near-identical between A and C (structural |d| < 0.5). "
            + (
                "Only wall-clock duration differs substantially (C is faster), as expected "
                "from a different scheduler/host load. "
                if timing_shifted else ""
            )
            + "Frame §5.2 as: the orchestration runtime changes timing but NOT the "
            "size/structure of the traffic, and the fingerprint transfers A->C near "
            "ceiling despite that timing shift -- i.e. the transferable signal rides the "
            "runtime-invariant structural channel. A control, not a generalization result."
        )
    elif struct_max < 0.8:
        headline = "structure-mostly-similar"
        recommendation = (
            "Volume/structure is mostly similar (struct |d| < 0.8) with small-to-medium "
            "differences in some cells; timing differs more. Keep §5.2 as a control: the "
            "runtime perturbs traffic modestly without breaking the fingerprint."
        )
    else:
        headline = "structure-distinct-but-transfers"
        recommendation = (
            "C carries a distinct structural wire signature (>=1 cell-feature with large "
            "effect) yet the classifier still transfers A->C near ceiling. The "
            "runtime-invariance point may be stated slightly more strongly: the fingerprint "
            "survives a measurable change in raw traffic structure."
        )

    result["verdict"] = {
        "n_cells": len(result["by_cell"]),
        "structure_within_cell_max_abs_cohens_d": struct_max,
        "structure_within_cell_mean_abs_cohens_d": struct_mean,
        "timing_within_cell_max_abs_cohens_d": timing_max,
        "timing_within_cell_mean_abs_cohens_d": timing_mean,
        "timing_shifted": bool(timing_shifted),
        "headline": headline,
        "recommendation": recommendation,
        "honesty_guardrail": (
            "C is a runtime variant of A, not an independent framework. Do NOT present this "
            "as independent validation or as closing the circularity critique; §7 must still "
            "list generalization across independently-structured frameworks as future work."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out)

    # console summary
    print("\n=== A vs C runtime-traffic diagnostic ===")
    print(f"A (asyncio): n={len(wfa)}   C (LangGraph): n={len(wfc)}")
    print("\nOverall per-feature (median A -> median C, Cohen's d, effect):")
    for name, desc in _HEADLINE:
        o = result["overall"][name]
        print(f"  {name:32s} {o['a']['median']:>12.2f} -> {o['c']['median']:>12.2f}  "
              f"d={o['cohens_d_c_minus_a']:+.2f} ({o['effect']})   {desc}")
    v = result["verdict"]
    print(f"\nWithin-cell |d|:  STRUCTURE max={v['structure_within_cell_max_abs_cohens_d']:.2f} "
          f"mean={v['structure_within_cell_mean_abs_cohens_d']:.2f}  |  "
          f"TIMING max={v['timing_within_cell_max_abs_cohens_d']:.2f} "
          f"mean={v['timing_within_cell_mean_abs_cohens_d']:.2f}")
    print(f"VERDICT: {v['headline']}")
    print(f"\n{v['recommendation']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A-vs-C runtime-invariance traffic diagnostic")
    ap.add_argument("--dir-a", default="data/processed")
    ap.add_argument("--dir-c", default="data/processed_langgraph")
    ap.add_argument("--out", default="data/results/runtime_traffic_diagnostic.json")
    args = ap.parse_args()
    main(Path(args.dir_a), Path(args.dir_c), Path(args.out))
