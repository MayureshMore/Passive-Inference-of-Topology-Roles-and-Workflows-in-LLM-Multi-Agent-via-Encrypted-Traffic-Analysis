#!/usr/bin/env python3
"""C4 — GROUP (cluster) bootstrap check on the two headline CIs.

WHY. Every CI in the paper comes from evaluation/stats.bootstrap_ci, which resamples the evaluated
(y_true, y_pred) pairs i.i.d.:  `idx = rng.integers(0, n_obs, n_obs)`. But the data are CLUSTERED —
closed-world samples share a `prompt_group` (the same prompt family), and cross-instance role samples
share a `trip`. The cross-validation is already group-safe (StratifiedGroupKFold), so the *splitting*
respects clusters while the *interval* does not. Within-cluster correlation means the i.i.d. bootstrap
treats correlated observations as independent, which can make CIs too NARROW (over-confident).

WHAT. Recompute the two headline numbers' CIs with a CLUSTER bootstrap — resample whole groups with
replacement (drawing every observation of each drawn group) — and compare against the i.i.d. interval:

  1. §1  closed-world WORKFLOW on deployment A (GBT, StratifiedGroupKFold by prompt_group, pooled OOF)
         headline macro-F1 0.708
  2. §9a coordinator 3-way CROSS-INSTANCE transfer, weaker direction (groups = trip)
         headline macro-F1 0.866, §4 band "DEPLOYABLE" requires >=0.70 with CI clear of chance

DECISION RULE (pre-stated, no re-stamping):
  * If the group interval is materially wider (>10% wider, or it changes a §4 band verdict), the paper
    reports the GROUP bootstrap for that number.
  * If not, one line notes the choice does not matter and the i.i.d. interval stands.
Either way the POINT estimates are untouched — this only re-estimates uncertainty.

Additive: writes ${A2A_RESULTS_DIR:-data/results}/group_bootstrap_check.json. No canonical result is
modified. Seed 42, 2000 resamples, same percentile convention (2.5/97.5) as evaluation/stats.

Usage: python scripts/check_group_bootstrap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gradient_boosted import GBTClassifier  # noqa: E402
from scripts.evaluate_cross_deployment import load_deployment  # noqa: E402
from scripts.evaluate_offtheshelf_fingerprint import extract_role_samples  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SEED = 42
_N_BOOT = 2000
_COORDS = ("mcp", "orchestrator", "planner")


def _macro_f1(yt, yp, classes):
    return float(f1_score(yt, yp, labels=list(classes), average="macro", zero_division=0))


def iid_ci(y_true, y_pred, classes, n=_N_BOOT, seed=_SEED):
    """The CURRENT method: resample (y_true, y_pred) pairs i.i.d. — mirrors evaluation/stats."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        if len(set(yt[idx].tolist())) < 2:
            continue
        vals.append(_macro_f1(yt[idx], yp[idx], classes))
    lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return {"macro_f1": _macro_f1(yt, yp, classes), "ci_lo": lo, "ci_hi": hi,
            "width": round(hi - lo, 4), "method": "i.i.d. bootstrap over observations (current)"}


def group_ci(y_true, y_pred, groups, classes, n=_N_BOOT, seed=_SEED):
    """CLUSTER bootstrap: resample whole GROUPS with replacement, taking all observations of each
    drawn group. Respects within-group correlation the i.i.d. bootstrap ignores."""
    yt, yp, g = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[u] for u in drawn])
        if len(set(yt[idx].tolist())) < 2:
            continue
        vals.append(_macro_f1(yt[idx], yp[idx], classes))
    lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return {"macro_f1": _macro_f1(yt, yp, classes), "ci_lo": lo, "ci_hi": hi,
            "width": round(hi - lo, 4), "n_groups": int(len(uniq)),
            "method": "cluster/group bootstrap — resample groups with replacement"}


def compare(iid, grp, chance, band_bar=None):
    """Widening + whether any §4 band verdict would change (pre-stated rule)."""
    ratio = (grp["width"] / iid["width"]) if iid["width"] > 0 else float("inf")
    material = ratio > 1.10
    band_change = False
    if band_bar is not None:
        # DEPLOYABLE needs point >= band_bar AND ci_lo > chance; check the CI-clear-of-chance half.
        band_change = (iid["ci_lo"] > chance) != (grp["ci_lo"] > chance)
    return {
        "width_iid": iid["width"], "width_group": grp["width"],
        "width_ratio_group_over_iid": round(ratio, 3),
        "materially_wider_gt10pct": bool(material),
        "band_verdict_changes": bool(band_change),
        "decision": ("REPORT GROUP BOOTSTRAP (materially wider or band changes)"
                     if (material or band_change) else
                     "CHOICE DOES NOT MATTER — i.i.d. interval stands (group interval not materially wider, "
                     "no band change); one line in the paper suffices"),
    }


def headline_closed_world_workflow(processed: Path) -> dict:
    """§1 workflow on deployment A: group-safe CV by prompt_group, pooled OOF, both intervals."""
    X, _seg, y, groups = load_deployment(processed, "workflow")
    y = np.asarray(y); groups = np.asarray(groups)
    classes = sorted(set(y.tolist()))
    cv = StratifiedGroupKFold(n_splits=5)
    oof_t, oof_p, oof_g = [], [], []
    for tr, te in cv.split(X, y, groups):
        clf = GBTClassifier(task="workflow").fit(X[tr], list(y[tr]))
        oof_t.extend(y[te].tolist()); oof_p.extend(clf.predict(X[te])); oof_g.extend(groups[te].tolist())
    i_ = iid_ci(oof_t, oof_p, classes)
    g_ = group_ci(oof_t, oof_p, oof_g, classes)
    chance = 1.0 / len(classes)
    return {
        "what": "§1 closed-world WORKFLOW, deployment A (GBT, StratifiedGroupKFold by prompt_group, pooled OOF)",
        "published_headline_macro_f1": 0.708,
        "note_point_estimate": "The published 0.708 is the MEAN OF FOLD macro-F1s; the pooled-OOF point "
                               "below is the bootstrap's own point estimate. Only the INTERVAL is under "
                               "test here — point estimates are untouched.",
        "cluster_unit": "prompt_group", "n_obs": int(len(oof_t)), "chance": chance,
        "iid_bootstrap": i_, "group_bootstrap": g_, "comparison": compare(i_, g_, chance),
    }


def headline_coord_transfer(r1: Path, r2: Path) -> dict:
    """§9a coordinator 3-way cross-instance transfer, weaker direction; groups = trip."""
    X1, y1, g1 = extract_role_samples(r1)
    X2, y2, g2 = extract_role_samples(r2)
    m1 = np.isin(y1, list(_COORDS)); m2 = np.isin(y2, list(_COORDS))
    X1, y1, g1 = X1[m1], np.asarray(y1)[m1], np.asarray(g1)[m1]
    X2, y2, g2 = X2[m2], np.asarray(y2)[m2], np.asarray(g2)[m2]
    roles = sorted(set(y1.tolist()) & set(y2.tolist()))
    if len(roles) < 2:
        raise SystemExit(f"BLOCKED: <2 shared coordinator roles (inst1={set(y1)}, inst2={set(y2)})")

    def one_dir(Xtr, ytr, Xte, yte, gte, label):
        clf = GBTClassifier(task="role").fit(Xtr, list(ytr))
        pred = clf.predict(Xte)
        classes = sorted(set(ytr.tolist()) | set(yte.tolist()))
        i_ = iid_ci(list(yte), list(pred), classes)
        g_ = group_ci(list(yte), list(pred), list(gte), classes)
        chance = 1.0 / len(sorted(set(yte.tolist())))
        return {"direction": label, "chance": chance, "n_test": int(len(yte)),
                "iid_bootstrap": i_, "group_bootstrap": g_,
                "comparison": compare(i_, g_, chance, band_bar=0.70)}

    d12 = one_dir(X1, y1, X2, y2, g2, "inst1→inst2")
    d21 = one_dir(X2, y2, X1, y1, g1, "inst2→inst1")
    weak = d12 if d12["iid_bootstrap"]["macro_f1"] <= d21["iid_bootstrap"]["macro_f1"] else d21
    return {
        "what": "§9a coordinator 3-way CROSS-INSTANCE transfer (weaker direction is the headline)",
        "published_headline_macro_f1": 0.866,
        "cluster_unit": "trip", "roles": roles,
        "both_directions": [d12, d21],
        "weaker_direction": weak["direction"],
        "weaker_direction_result": weak,
        "band_note": "§4 DEPLOYABLE requires macro-F1 >=0.70 AND CI clear of chance; the group interval "
                     "is checked against the same bar (no re-stamping).",
    }


def main(a: argparse.Namespace) -> None:
    processed = Path(a.processed)
    r1, r2 = Path(os.path.expanduser(a.inst1)), Path(os.path.expanduser(a.inst2))
    out: dict = {
        "task": "C4 — group/cluster bootstrap check on the headline CIs",
        "rationale": "evaluation/stats.bootstrap_ci resamples observations i.i.d., but the data are "
                     "clustered (prompt_group / trip) and the CV is already group-safe. An i.i.d. "
                     "bootstrap over correlated observations can be over-confident (too narrow). This "
                     "recomputes the intervals with a cluster bootstrap that resamples whole groups.",
        "method": "percentile bootstrap, 2000 resamples, seed 42, 2.5/97.5 — identical to "
                  "evaluation/stats except the resampling unit (observation vs group).",
        "decision_rule": "group interval >10% wider, or a §4 band verdict flips → paper reports the "
                         "group bootstrap; otherwise a one-line note that the choice does not matter.",
        "point_estimates_unchanged": True,
    }
    results = []
    if (processed / "labels.json").exists():
        results.append(headline_closed_world_workflow(processed))
    else:
        results.append({"what": "§1 closed-world workflow", "blocked": f"no labels.json in {processed}"})
    if any(r1.glob("*.pcap")) and any(r2.glob("*.pcap")):
        results.append(headline_coord_transfer(r1, r2))
    else:
        results.append({"what": "§9a coordinator transfer",
                        "blocked": f"missing pcaps in {r1} or {r2}"})
    out["headlines"] = results

    changed = [r for r in results if r.get("comparison", {}).get("materially_wider_gt10pct")
               or r.get("weaker_direction_result", {}).get("comparison", {}).get("materially_wider_gt10pct")
               or r.get("weaker_direction_result", {}).get("comparison", {}).get("band_verdict_changes")]
    out["summary"] = ("At least one headline interval is materially wider under the group bootstrap — "
                      "report the group interval for it." if changed else
                      "Neither headline interval is materially wider under the group bootstrap and no §4 "
                      "band verdict changes — the resampling-unit choice does not matter; the published "
                      "i.i.d. intervals stand (one-line note in the paper).")

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "group_bootstrap_check.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 82)
    print("  C4 — GROUP/CLUSTER BOOTSTRAP CHECK (headline CIs)")
    print("=" * 82)
    for r in results:
        if r.get("blocked"):
            print(f"  BLOCKED: {r['what']} — {r['blocked']}"); continue
        if "comparison" in r:
            i_, g_, c = r["iid_bootstrap"], r["group_bootstrap"], r["comparison"]
            print(f"  {r['what'][:60]}")
            print(f"    iid  : F1={i_['macro_f1']:.3f} [{i_['ci_lo']:.3f}, {i_['ci_hi']:.3f}] width={i_['width']:.4f}")
            print(f"    group: F1={g_['macro_f1']:.3f} [{g_['ci_lo']:.3f}, {g_['ci_hi']:.3f}] width={g_['width']:.4f}"
                  f"  ({g_['n_groups']} groups)")
            print(f"    -> width ratio {c['width_ratio_group_over_iid']}x | {c['decision'][:60]}")
        else:
            w = r["weaker_direction_result"]; i_, g_, c = w["iid_bootstrap"], w["group_bootstrap"], w["comparison"]
            print(f"  {r['what'][:60]}  (weaker: {r['weaker_direction']})")
            print(f"    iid  : F1={i_['macro_f1']:.3f} [{i_['ci_lo']:.3f}, {i_['ci_hi']:.3f}] width={i_['width']:.4f}")
            print(f"    group: F1={g_['macro_f1']:.3f} [{g_['ci_lo']:.3f}, {g_['ci_hi']:.3f}] width={g_['width']:.4f}"
                  f"  ({g_['n_groups']} groups)")
            print(f"    -> width ratio {c['width_ratio_group_over_iid']}x | band changes: {c['band_verdict_changes']}")
    print("=" * 82)
    print(f"  {out['summary'][:150]}")
    print(f"\nWrote {out_dir / 'group_bootstrap_check.json'}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C4 — group/cluster bootstrap on headline CIs")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--inst1", default="data/raw_offtheshelf")
    p.add_argument("--inst2", default="data/raw_offtheshelf_inst2")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
