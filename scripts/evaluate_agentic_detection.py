#!/usr/bin/env python3
"""
AGENTIC-vs-AGENTIC DETECTION (Exp 1) — A2A flows vs AutoGen flows.

Answers the referee's sharpest deployability point: the existing detection AUC 1.0
(offtheshelf_detection.json) is only against NON-agentic negatives, so "detect an A2A system in
the wild vs another agent framework" was untested. We now have real AutoGen gRPC traffic (§10;
~/autogen-xframework/), so we run a binary detector:

    positive = A2A flows     (a2a_mcp; Starlette/JSON-RPC/SSE over HTTP; data/raw_offtheshelf)
    negative = AutoGen flows (autogen-core distributed gRPC star-through-host; local ollama)

Method = the project pipeline: 35-dim per-flow traffic-shape features (features/per_flow.py, port
NEVER a feature), GBT (HistGradientBoosting via GBTClassifier), group-safe 5-fold
StratifiedGroupKFold by TRIP (a trip's flows never split folds), leakage-free OOF predictions.
Report AUROC + macro-F1 with percentile bootstrap 95% CI, plus n per class. Also run SHAPE-ONLY
(volume-ablated) with the SAME mask as Task 1 / §10.

Pre-registered bands (headline AUROC; chance 0.50; no re-stamp):
    AUROC ≥ 0.90 & CI clear of 0.50 → STRONG (A2A separable from an independent agentic framework)
    0.70 – 0.90                     → MODERATE
    < 0.70 or CI touches 0.50       → WEAK

HONEST SCOPING (in the JSON): AutoGen uses a DIFFERENT TRANSPORT (gRPC/HTTP2) than A2A
(SSE-over-HTTP), so a positive shows A2A is separable from an INDEPENDENT agentic framework — real,
and it converts the detection section from "open problem" to a result — but the HARDER case, another
SSE-over-HTTP agentic framework sharing A2A's transport, remains untested future work. If a positive
survives the shape-only ablation it is not purely raw transport volume; the driving features are
reported so a reviewer can see whether it is transport-framing packet sizes.

Writes ${A2A_RESULTS_DIR:-data/results}/agentic_detection.json + figures/agentic_detection.png.
Additive; touches no committed result. Blocked-and-report if AutoGen pcaps are absent.

Usage: venv/bin/python scripts/evaluate_agentic_detection.py
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
from models.gradient_boosted import GBTClassifier  # noqa: E402
from scripts.evaluate_offtheshelf_fingerprint import extract_role_samples as extract_a2a  # noqa: E402
from scripts.evaluate_cross_framework_autogen import extract_autogen_role_samples  # noqa: E402
from scripts.evaluate_cross_instance_transfer import _SHAPE_MASK, _KEPT_FEATURES, _DROPPED_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SEED = 42
_N_BOOT = 2000
POS = "A2A"          # positive label
NEG = "AutoGen"      # negative label


def gbt_oof(X, y, groups, mask=None):
    """Leakage-free group-safe OOF via GBT + StratifiedGroupKFold. Returns (proba_pos, pred)."""
    from sklearn.model_selection import StratifiedGroupKFold
    Xi = X[:, mask] if mask is not None else X
    y = np.asarray(y)
    proba = np.zeros(len(y), dtype=float)
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED)
    for tr, te in cv.split(Xi, y, groups):
        clf = GBTClassifier(task="role").fit(Xi[tr], list(y[tr]))
        classes = list(clf.label_encoder.classes_)
        pcol = classes.index(POS)
        proba[te] = clf.predict_proba(Xi[te])[:, pcol]
        pred[te] = clf.predict(Xi[te])
    return proba, pred


def metrics_with_ci(y, proba, pred):
    from sklearn.metrics import roc_auc_score, f1_score
    yb = (np.asarray(y) == POS).astype(int)
    auc = float(roc_auc_score(yb, proba))
    f1 = float(f1_score(y, pred, average="macro"))
    rng = np.random.default_rng(_SEED)
    n = len(y)
    aucs, f1s = [], []
    for _ in range(_N_BOOT):
        idx = rng.integers(0, n, n)
        if len(set(yb[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yb[idx], proba[idx]))
        f1s.append(f1_score(np.asarray(y)[idx], np.asarray(pred)[idx], average="macro"))
    a_lo, a_hi = (float(v) for v in np.percentile(aucs, [2.5, 97.5]))
    f_lo, f_hi = (float(v) for v in np.percentile(f1s, [2.5, 97.5]))
    return {"auroc": auc, "auroc_ci95": [a_lo, a_hi], "macro_f1": f1, "macro_f1_ci95": [f_lo, f_hi]}


def band(auc, ci_lo):
    if auc >= 0.90 and ci_lo > 0.50:
        return "STRONG (A2A separable from an independent agentic framework; ≥0.90, CI clear of chance)"
    if 0.70 <= auc < 0.90 and ci_lo > 0.50:
        return "MODERATE (0.70–0.90)"
    return "WEAK (<0.70 or CI touches chance 0.50)"


def make_figure(full, shape, n_pos, n_neg, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    groups = ["AUROC", "macro-F1"]
    xs = np.arange(len(groups)); w = 0.36
    for j, (res, lab, col) in enumerate([(full, "full (35 feat)", "#2b6cb0"),
                                         (shape, "shape-only (16 feat)", "#dd6b20")]):
        vals = [res["auroc"], res["macro_f1"]]
        los = [res["auroc"] - res["auroc_ci95"][0], res["macro_f1"] - res["macro_f1_ci95"][0]]
        his = [res["auroc_ci95"][1] - res["auroc"], res["macro_f1_ci95"][1] - res["macro_f1"]]
        ax.bar(xs + (j - 0.5) * w, vals, w, yerr=[los, his], capsize=4, label=lab, color=col)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance (0.50)")
    ax.set_xticks(xs); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"A2A vs AutoGen agentic detection (n={n_pos} A2A / {n_neg} AutoGen flows)\n"
                 "group-safe CV, GBT — survives shape-only, but transport-driven (see §11)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def main(args: argparse.Namespace) -> None:
    a2a_raw = Path(os.path.expanduser(args.a2a_raw))
    ag_raw = Path(os.path.expanduser(args.autogen_raw))
    if not any(a2a_raw.glob("*.pcap")):
        raise SystemExit(f"BLOCKED: no A2A pcaps at {a2a_raw}")
    if not any(ag_raw.glob("trip_*.pcap")):
        raise SystemExit(f"BLOCKED: no AutoGen pcaps at {ag_raw} — run ~/autogen-xframework/collect_trips.sh")

    Xa, _, ga = extract_a2a(a2a_raw)
    Xg, _, gg = extract_autogen_role_samples(ag_raw)
    logger.info("A2A flows: %d (%d trips) | AutoGen flows: %d (%d trips)",
                len(Xa), len(set(ga)), len(Xg), len(set(gg)))

    X = np.vstack([Xa, Xg]).astype(np.float32)
    y = np.array([POS] * len(Xa) + [NEG] * len(Xg))
    groups = np.array([f"a2a:{g}" for g in ga] + [f"autogen:{g}" for g in gg])

    p_full, pred_full = gbt_oof(X, y, groups)
    p_shape, pred_shape = gbt_oof(X, y, groups, mask=_SHAPE_MASK)
    full = metrics_with_ci(y, p_full, pred_full)
    shape = metrics_with_ci(y, p_shape, pred_shape)
    logger.info("FULL  AUROC=%.3f %s  macroF1=%.3f", full["auroc"], full["auroc_ci95"], full["macro_f1"])
    logger.info("SHAPE AUROC=%.3f %s  macroF1=%.3f", shape["auroc"], shape["auroc_ci95"], shape["macro_f1"])

    # Which single features drive it? (AUROC=1.0 demands showing WHY; transport-framing packet
    # sizes near-constant per framework ⇒ a transport fingerprint.)
    from sklearn.metrics import roc_auc_score
    from features.per_flow import PerFlowFeatures
    names = PerFlowFeatures.FEATURE_NAMES(); yb = (y == POS).astype(int)
    ranked = sorted(((float(max(roc_auc_score(yb, X[:, i]), 1 - roc_auc_score(yb, X[:, i]))),
                      names[i], bool(_SHAPE_MASK[i])) for i in range(X.shape[1])), reverse=True)
    top = [{"feature": n, "single_feature_auroc": round(a, 3), "is_shape_feature": s} for a, n, s in ranked[:6]]

    survives = shape["auroc"] >= 0.90 and shape["auroc_ci95"][0] > 0.50
    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_png = out_dir / "figures" / "agentic_detection.png"
    make_figure(full, shape, len(Xa), len(Xg), fig_png)

    out = {
        "task": "agentic-vs-agentic detection — A2A vs AutoGen (Exp 1)",
        "positive_class": {"label": POS, "desc": "A2A flows (a2a_mcp; Starlette/JSON-RPC/SSE over HTTP)",
                           "n_flows": int(len(Xa)), "n_trips": int(len(set(ga)))},
        "negative_class": {"label": NEG, "desc": "AutoGen flows (autogen-core distributed gRPC; local ollama)",
                           "n_flows": int(len(Xg)), "n_trips": int(len(set(gg)))},
        "representation": "35-dim per-flow traffic-shape vector (features/per_flow.py); port NEVER a feature.",
        "method": "GBT (HistGradientBoosting); group-safe 5-fold StratifiedGroupKFold by trip; leakage-free "
                  "OOF; AUROC + macro-F1 with percentile bootstrap 95% CI (2000 resamples); seed 42.",
        "full_features": full,
        "shape_only_ablation": {**shape, "n_features": int(_SHAPE_MASK.sum()),
                                "features_kept": _KEPT_FEATURES, "features_dropped": _DROPPED_FEATURES},
        "top_discriminating_features": top,
        "verdict": band(full["auroc"], full["auroc_ci95"][0]),
        "shape_only_verdict": band(shape["auroc"], shape["auroc_ci95"][0]),
        "interpretation": (
            "A2A is separable from an independent agentic framework (AutoGen) at the flow level"
            + (" and the separation SURVIVES the shape-only ablation, so it is not purely raw connection "
               "volume. BUT the driving features are TRANSPORT-LEVEL packet-size percentiles (near-constant "
               "per framework: gRPC/HTTP2 framing vs HTTP/SSE framing — see top_discriminating_features), so "
               "'survives shape-only' means the signal lives in shape features that are themselves "
               "transport-linked." if survives else
               ", but the separation weakens under the shape-only ablation — it rides on raw transport volume.")),
        "honest_scoping": "AutoGen uses a DIFFERENT TRANSPORT (gRPC/HTTP2) than A2A (SSE-over-HTTP), and the "
                          "separation is largely a TRANSPORT fingerprint. This converts detection from "
                          "'only vs non-agentic negatives' to 'A2A distinguishable from an INDEPENDENT agentic "
                          "framework' — real — but does NOT show separability from a SAME-TRANSPORT agentic "
                          "framework (another SSE-over-HTTP system, e.g. CrewAI), the hardest untested case. "
                          "Single AutoGen topology/LLM. Reported as-is.",
        "verdict_basis": "pre-registered AUROC bands; verdict field matches the number (no re-stamp).",
        "figure": str(fig_png),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agentic_detection.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 76)
    print("  EXP 1 — AGENTIC DETECTION: A2A vs AutoGen (GBT, group-safe)")
    print("=" * 76)
    print(f"  n: A2A={len(Xa)} flows ({len(set(ga))} trips)  AutoGen={len(Xg)} flows ({len(set(gg))} trips)")
    print(f"  FULL  (35)  AUROC={full['auroc']:.3f} {full['auroc_ci95']}  macroF1={full['macro_f1']:.3f} "
          f"{full['macro_f1_ci95']}  -> {out['verdict'].split(' (')[0]}")
    print(f"  SHAPE (16)  AUROC={shape['auroc']:.3f} {shape['auroc_ci95']}  macroF1={shape['macro_f1']:.3f} "
          f"-> {out['shape_only_verdict'].split(' (')[0]}")
    print(f"  top driver: {top[0]['feature']} (single-feat AUROC {top[0]['single_feature_auroc']}, "
          f"{'shape' if top[0]['is_shape_feature'] else 'volume'})")
    print("=" * 76)
    print(f"\nWrote {out_dir / 'agentic_detection.json'}\nWrote {fig_png}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 1 — A2A vs AutoGen agentic detection (GBT)")
    p.add_argument("--a2a-raw", default="data/raw_offtheshelf")
    p.add_argument("--autogen-raw", default="~/autogen-xframework/data/raw")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
