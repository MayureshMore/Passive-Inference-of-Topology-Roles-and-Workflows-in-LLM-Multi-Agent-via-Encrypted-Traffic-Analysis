"""
Shared statistics utilities — bootstrap confidence intervals.

A single percentile-bootstrap implementation used everywhere a headline number is reported
(closed-world CV, cross-deployment transfer, disentanglement, live defense) so every CI in the
paper is computed the same way (2.5 / 97.5 percentile = 95 % CI).

RESAMPLING UNIT — the CI convention (C4)
────────────────────────────────────────
The data are CLUSTERED: closed-world traces share a `prompt_group` (same prompt family), and
per-agent role samples share a `trip` (several flows from one run). Observations inside a cluster
are correlated, so resampling individual observations i.i.d. treats correlated points as
independent and produces intervals that are TOO NARROW (over-confident). The cross-validation is
already cluster-aware (StratifiedGroupKFold); the interval must be too.

`bootstrap_ci(..., groups=...)` therefore performs a CLUSTER (group) bootstrap: it resamples whole
groups with replacement, taking every observation of each drawn group. This is the project's
default convention — pass `groups` wherever a cluster variable exists.

`groups=None` falls back to the i.i.d. bootstrap and is correct ONLY when observations are
genuinely independent (no repeated prompt/trip). The returned dict records which was used in
`ci_method`, so every result is self-describing and the convention is machine-checkable.

Empirically (data/results/group_bootstrap_check.json) the group interval is 14 % wider on the §1
workflow headline and 24 % wider on the §9a transfer headline; no verdict changes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Defaults kept identical to scripts/evaluate_cross_deployment.py so CIs match.
_N_BOOTSTRAP = 2000
_RNG_SEED = 42


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence | None = None,
    groups: Sequence | None = None,
    n: int = _N_BOOTSTRAP,
    seed: int = _RNG_SEED,
) -> dict[str, Any]:
    """
    Percentile bootstrap 95 % CI for accuracy and macro-F1.

    Resamples the evaluated (y_true, y_pred) pairs (the test predictions / pooled out-of-fold
    predictions) — never the training set.

    Parameters
    ----------
    groups : cluster label per observation (e.g. prompt_group, trip). When given, whole clusters
        are resampled with replacement (the project convention — see module docstring). When None,
        observations are resampled i.i.d., which is only valid if they are independent.

    Returns the point estimate, 2.5/97.5 bounds for both metrics, `ci_method`
    ("group_bootstrap" | "iid_bootstrap"), and `n_clusters` when clustered.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    n_obs = len(y_true_arr)
    if classes is None:
        classes = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))

    def _acc(a, b):
        return float(accuracy_score(a, b))

    def _f1(a, b):
        return float(f1_score(a, b, labels=list(classes), average="macro", zero_division=0))

    point_acc = _acc(y_true_arr, y_pred_arr) if n_obs else 0.0
    point_f1 = _f1(y_true_arr, y_pred_arr) if n_obs else 0.0

    if n_obs == 0:
        return {
            "accuracy": 0.0, "accuracy_ci_lo": 0.0, "accuracy_ci_hi": 0.0,
            "macro_f1": 0.0, "macro_f1_ci_lo": 0.0, "macro_f1_ci_hi": 0.0,
            "ci_method": "none (empty input)", "n_clusters": 0,
        }

    rng = np.random.default_rng(seed)
    accs = np.empty(n, dtype=float)
    f1s = np.empty(n, dtype=float)

    if groups is not None:
        # CLUSTER bootstrap — resample whole groups, take every observation of each drawn group.
        g_arr = np.asarray(list(groups))
        if len(g_arr) != n_obs:
            raise ValueError(f"groups length {len(g_arr)} != n observations {n_obs}")
        uniq = np.unique(g_arr)
        idx_by_group = {u: np.where(g_arr == u)[0] for u in uniq}
        for i in range(n):
            drawn = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_group[u] for u in drawn])
            accs[i] = _acc(y_true_arr[idx], y_pred_arr[idx])
            f1s[i] = _f1(y_true_arr[idx], y_pred_arr[idx])
        method, n_clusters = "group_bootstrap", int(len(uniq))
    else:
        # i.i.d. bootstrap — only valid when observations are independent.
        for i in range(n):
            idx = rng.integers(0, n_obs, n_obs)
            accs[i] = _acc(y_true_arr[idx], y_pred_arr[idx])
            f1s[i] = _f1(y_true_arr[idx], y_pred_arr[idx])
        method, n_clusters = "iid_bootstrap", None

    return {
        "accuracy":       point_acc,
        "accuracy_ci_lo": float(np.percentile(accs, 2.5)),
        "accuracy_ci_hi": float(np.percentile(accs, 97.5)),
        "macro_f1":       point_f1,
        "macro_f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "macro_f1_ci_hi": float(np.percentile(f1s, 97.5)),
        "ci_method":      method,
        "n_clusters":     n_clusters,
    }
