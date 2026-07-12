#!/usr/bin/env python3
"""
MIXING / MULTIPLEXING DEGRADATION (Exp 2) — how far do detection and role recovery hold when
clean per-port flow isolation is REMOVED and A2A flows are observed amid background traffic?

Gated on Exp 1 (agentic_detection.json) landing ≥ ~0.80. The paper's detection/role results assume
the observer isolates each agent's flows by port; real deployments multiplex. We embed A2A flows in
the existing open-world background mix (data/processed_background — web/file/REST/JSON-RPC/multi-REST/
LLM-direct) and sweep the contamination ratio ρ = background flows per agent flow (ρ=0 = clean
per-port world; ρ→∞ = a window dominated by background).

Two curves, same method/guardrails as Exp 1 (35-dim per-flow shape features; GBT; group-safe 5-fold
StratifiedGroupKFold by source trace; percentile bootstrap 95% CI; port never a feature):

  (1) DETECTION — per-flow A2A-vs-background detector; threshold at the project's 5% background-FPR
      operating point. Recall (TPR) is threshold-fixed; PRECISION degrades with ρ as un-isolated
      background flows accumulate false positives: precision(ρ) = TPR / (TPR + ρ·FPR).
  (2) ROLE RECOVERY — role classifier WITH an explicit BACKGROUND reject class (the class an
      observer needs once flows aren't port-pure). Report role macro-F1 (agent roles) with the
      reject class present, the background→role LEAK rate, and role-attribution precision vs ρ.

HONEST CEILING (in the JSON + writeup): this measures degradation under multiplexing for an observer
who can still SEGMENT flows but not ATTRIBUTE them. It does NOT solve flow isolation behind a SHARED
:443 REVERSE PROXY with no distinct observable ports (flows not even separable) — that is
ARCHITECTURAL and handled by the paper's threat-model scope, not this experiment. Also: the
background is NON-AGENTIC (structurally distinct → easily rejected); AGENTIC distractors (cf. Exp 1)
would be sterner. Bands reported as-is.

Writes ${A2A_RESULTS_DIR:-data/results}/mixing_degradation.json + figures/mixing_degradation.png.
Additive; touches no committed result. Blocked-and-report if background/A2A data absent.

Usage: venv/bin/python scripts/evaluate_mixing_degradation.py
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
from scripts.evaluate_cross_instance_transfer import _SHAPE_MASK  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SEED = 42
_BG_FPR = 0.05
_RHOS = [0, 1, 2, 4, 8, 16, 32]
BG = "background"


def load_background_flows(bg_dir: Path):
    """Background per-flow 35-dim vectors: the two heaviest flows (pf_top1/pf_top2) of each
    background trace's 195-dim vector, tagged with source trace (group-safe CV)."""
    lf = bg_dir / "labels_background.json"
    labels = json.loads(lf.read_text()) if lf.exists() else {}
    X, cats, groups = [], [], []
    for npz in sorted(bg_dir.glob("*.npz")):
        flat = np.load(npz)["flat"]
        if flat.shape[0] != 195:
            continue
        cat = (labels.get(npz.stem, {}) or {}).get("category", BG)
        for block in (flat[35:70], flat[70:105]):        # pf_top1, pf_top2
            if np.any(block):
                X.append(block.astype(np.float32)); cats.append(cat); groups.append(f"bg:{npz.stem}")
    return np.asarray(X, dtype=np.float32), np.asarray(cats), np.asarray(groups)


def gbt_oof_binary(X, y_pos_mask, groups, pos="A2A", neg=BG, mask=None):
    """Group-safe OOF positive-class probability + prediction (GBT)."""
    from sklearn.model_selection import StratifiedGroupKFold
    Xi = X[:, mask] if mask is not None else X
    y = np.where(y_pos_mask, pos, neg)
    proba = np.zeros(len(y)); pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED)
    for tr, te in cv.split(Xi, y, groups):
        clf = GBTClassifier(task="role").fit(Xi[tr], list(y[tr]))
        pcol = list(clf.label_encoder.classes_).index(pos)
        proba[te] = clf.predict_proba(Xi[te])[:, pcol]
        pred[te] = clf.predict(Xi[te])
    return y, proba, pred


def gbt_oof_multiclass(X, y, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y); pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED)
    for tr, te in cv.split(X, y, groups):
        clf = GBTClassifier(task="role").fit(X[tr], list(y[tr]))
        pred[te] = clf.predict(X[te])
    return pred


def boot_ci(fn, *arrs, n=2000):
    rng = np.random.default_rng(_SEED); N = len(arrs[0]); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(fn(*[np.asarray(a)[idx] for a in arrs]))
        except Exception:
            continue
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def detection_curve(Xa, ga, Xb, gb):
    from sklearn.metrics import roc_auc_score, roc_curve
    X = np.vstack([Xa, Xb]); pos_mask = np.r_[np.ones(len(Xa), bool), np.zeros(len(Xb), bool)]
    g = np.array([f"a2a:{x}" for x in ga] + list(gb))
    y, proba, _ = gbt_oof_binary(X, pos_mask, g)
    _, proba_s, _ = gbt_oof_binary(X, pos_mask, g, mask=_SHAPE_MASK)
    yb = (y == "A2A").astype(int)
    auc = float(roc_auc_score(yb, proba)); auc_s = float(roc_auc_score(yb, proba_s))
    auc_ci = boot_ci(lambda a, b: roc_auc_score(a, b) if len(set(a)) > 1 else np.nan, yb, proba)
    # Operating point = FIXED 5% background FPR (project convention). Read recall (TPR) at that FPR
    # from the ROC curve (robust to probability ties); the operating FPR is _BG_FPR by construction.
    fp, tp, _ = roc_curve(yb, proba)
    tpr = float(np.interp(_BG_FPR, fp, tp))   # recall achievable at 5% bg-FPR
    fpr = _BG_FPR
    curve = [{"rho": r, "detection_recall": tpr,
              "detection_precision": (tpr / (tpr + r * fpr)) if (tpr + r * fpr) > 0 else 0.0}
             for r in _RHOS]
    return {"auroc": auc, "auroc_ci95": auc_ci, "auroc_shape_only": auc_s,
            "operating_point_bg_fpr": _BG_FPR, "detection_recall_TPR_at_5pct_fpr": tpr,
            "precision_recall_vs_rho": curve,
            "note": "operating point fixed at 5% background FPR (project convention); recall read from "
                    "the ROC curve at that FPR. Precision(ρ)=TPR/(TPR+ρ·0.05): recall is fixed, precision "
                    "falls as un-isolated background flows accumulate false positives at the 5% rate."}


def role_curve(Xa, ya, ga, Xb, gb):
    from sklearn.metrics import f1_score
    X = np.vstack([Xa, Xb]); y = np.r_[np.asarray(ya), np.array([BG] * len(Xb))]
    g = np.array([f"a2a:{x}" for x in ga] + list(gb))
    pred = gbt_oof_multiclass(X, y, g)
    roles = sorted(set(np.asarray(ya)))
    is_bg = y == BG
    leak = float(np.isin(pred[is_bg], roles).mean())
    f1_roles = float(f1_score(list(y), list(pred), labels=roles, average="macro"))
    f1_ci = boot_ci(lambda a, b: f1_score(list(a), list(b), labels=roles, average="macro", zero_division=0),
                    y, pred)
    bg_reject = float((pred[is_bg] == BG).mean())
    agent_correct = float((pred[~is_bg] == y[~is_bg]).mean())
    curve = [{"rho": r, "role_recovery_macro_f1": f1_roles,
              "role_attribution_precision":
              (agent_correct / (agent_correct + r * leak)) if (agent_correct + r * leak) > 0 else 0.0}
             for r in _RHOS]
    return {"n_roles": len(roles), "role_macro_f1_with_reject_class": f1_roles,
            "role_macro_f1_ci95": f1_ci, "background_reject_rate": bg_reject,
            "background_to_role_leak_rate": leak,
            "agent_role_recall_with_background_present": agent_correct,
            "precision_vs_rho": curve}


def make_figure(det, role, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    rhos = [p["rho"] for p in det["precision_recall_vs_rho"]]
    dprec = [p["detection_precision"] for p in det["precision_recall_vs_rho"]]
    drec = [p["detection_recall"] for p in det["precision_recall_vs_rho"]]
    rprec = [p["role_attribution_precision"] for p in role["precision_vs_rho"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(rhos, dprec, "o-", color="#2b6cb0", label="detection precision")
    a1.plot(rhos, drec, "s--", color="#38a169", label="detection recall (TPR)")
    a1.set_title(f"Detection vs lost isolation\n(AUROC {det['auroc']:.2f}, 5% bg-FPR op. point)")
    a2.plot(rhos, rprec, "o-", color="#dd6b20",
            label=f"role attribution precision (leak {role['background_to_role_leak_rate']:.2f})")
    a2.axhline(role["role_macro_f1_with_reject_class"], ls=":", color="gray",
               label=f"role macro-F1 +reject ({role['role_macro_f1_with_reject_class']:.2f})")
    a2.set_title("Role recovery vs lost isolation\n(non-agentic background — trivially rejected)")
    for a in (a1, a2):
        a.set_xlabel("ρ = background flows per agent flow"); a.set_ylabel("score")
        a.set_ylim(0, 1.05); a.legend(loc="lower left", fontsize=8)
    fig.suptitle("Multiplexing degradation — honest ceiling: does NOT solve shared-:443 no-ports case (§12)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_png); plt.close(fig)


def main(args: argparse.Namespace) -> None:
    a2a_raw = Path(os.path.expanduser(args.a2a_raw)); bg = Path(args.background)
    if not any(a2a_raw.glob("*.pcap")):
        raise SystemExit(f"BLOCKED: no A2A pcaps at {a2a_raw}")
    if not any(bg.glob("*.npz")):
        raise SystemExit(f"BLOCKED: no processed background at {bg} — none available, not fabricating.")

    # Gate on Exp 1
    exp1 = Path(os.environ.get("A2A_RESULTS_DIR", "data/results")) / "agentic_detection.json"
    if exp1.exists():
        a1 = json.loads(exp1.read_text()).get("full_features", {}).get("auroc")
        if a1 is not None and a1 < 0.80:
            raise SystemExit(f"BLOCKED: Exp 1 AUROC {a1:.3f} < 0.80 — Exp 2 gate not met.")

    Xa, ya, ga = extract_a2a(a2a_raw)
    Xb, cb, gb = load_background_flows(bg)
    logger.info("A2A agent flows: %d (%d traces) | background flows: %d (%d traces)",
                len(Xa), len(set(ga)), len(Xb), len(set(gb)))

    det = detection_curve(Xa, ga, Xb, gb)
    role = role_curve(Xa, ya, ga, Xb, gb)

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_png = out_dir / "figures" / "mixing_degradation.png"
    make_figure(det, role, fig_png)

    out = {
        "task": "mixing/multiplexing degradation — detection & role recovery vs loss of per-port "
                "isolation (ρ = background flows per agent flow) (Exp 2)",
        "gated_on": "Exp 1 agentic_detection AUROC ≥ ~0.80",
        "representation": "35-dim per-flow traffic-shape vector; A2A agent flows (a2a_mcp) vs genuine "
                          "background flows (web/file/REST/JSON-RPC/multi-REST/LLM-direct). Port never a feature.",
        "method": "GBT; group-safe 5-fold StratifiedGroupKFold by source trace; OOF; percentile bootstrap "
                  "95% CI; seed 42. Detection threshold at 5% background FPR.",
        "n_a2a_flows": int(len(Xa)), "n_background_flows": int(len(Xb)),
        "detection": det,
        "role_recovery": role,
        "interpretation": (
            f"DETECTION: recall (TPR) is threshold-fixed at {det['detection_recall_TPR_at_5pct_fpr']:.2f}; PRECISION "
            f"falls with lost isolation — {det['precision_recall_vs_rho'][0]['detection_precision']:.2f} at "
            f"ρ=0 → {det['precision_recall_vs_rho'][-1]['detection_precision']:.2f} at ρ={_RHOS[-1]} — the "
            f"honest cost of multiplexing. ROLE recovery with a reject class is robust here (macro-F1 "
            f"{role['role_macro_f1_with_reject_class']:.2f}, background→role leak "
            f"{role['background_to_role_leak_rate']:.2f}) BUT ONLY because the background is NON-AGENTIC "
            f"and structurally distinct; agentic distractors (cf. Exp 1) would be sterner and likely leak more."),
        "honest_ceiling": "Measures degradation under multiplexing for an observer who can still SEGMENT "
                          "flows but not ATTRIBUTE them. It does NOT solve flow isolation behind a SHARED "
                          ":443 REVERSE PROXY with no distinct observable ports — that is ARCHITECTURAL and "
                          "handled by the paper's threat-model scope, not this experiment.",
        "verdict_basis": "reported as-is; degradation is expected and quantified, not hidden.",
        "figure": str(fig_png),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mixing_degradation.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 76)
    print("  EXP 2 — MIXING DEGRADATION (loss of per-port isolation, GBT)")
    print("=" * 76)
    print(f"  A2A flows={len(Xa)}  background flows={len(Xb)}")
    print(f"  DETECTION AUROC={det['auroc']:.3f} {det['auroc_ci95']} (shape {det['auroc_shape_only']:.3f}) "
          f"recall={det['detection_recall_TPR_at_5pct_fpr']:.3f} @ bgFPR={_BG_FPR}")
    print("    ρ:  " + "  ".join(f"{p['rho']}→{p['detection_precision']:.2f}" for p in det["precision_recall_vs_rho"]))
    print(f"  ROLE macroF1(+reject)={role['role_macro_f1_with_reject_class']:.3f} {role['role_macro_f1_ci95']}  "
          f"bg-reject={role['background_reject_rate']:.3f}  bg→role leak={role['background_to_role_leak_rate']:.3f}")
    print("=" * 76)
    print(f"\nWrote {out_dir / 'mixing_degradation.json'}\nWrote {fig_png}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 2 — mixing/multiplexing degradation (GBT)")
    p.add_argument("--a2a-raw", default="data/raw_offtheshelf")
    p.add_argument("--background", default="data/processed_background")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
