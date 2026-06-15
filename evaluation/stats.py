"""
Shared statistics utilities — bootstrap confidence intervals.

A single percentile-bootstrap implementation used everywhere a headline number
is reported (closed-world CV, cross-deployment transfer, disentanglement, live
defense) so every CI is computed the same way (2.5 / 97.5 percentile = 95 % CI,
resampling the evaluated (y_true, y_pred) pairs).
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
    n: int = _N_BOOTSTRAP,
    seed: int = _RNG_SEED,
) -> dict[str, Any]:
    """
    Percentile bootstrap 95 % CI for accuracy and macro-F1.

    Resamples the evaluated (y_true, y_pred) pairs (the test predictions / pooled
    out-of-fold predictions) — never the training set. Returns the point estimate
    plus 2.5/97.5 percentile bounds for both metrics.
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
        }

    rng = np.random.default_rng(seed)
    accs = np.empty(n, dtype=float)
    f1s = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        accs[i] = _acc(y_true_arr[idx], y_pred_arr[idx])
        f1s[i] = _f1(y_true_arr[idx], y_pred_arr[idx])

    return {
        "accuracy":       point_acc,
        "accuracy_ci_lo": float(np.percentile(accs, 2.5)),
        "accuracy_ci_hi": float(np.percentile(accs, 97.5)),
        "macro_f1":       point_f1,
        "macro_f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "macro_f1_ci_hi": float(np.percentile(f1s, 97.5)),
    }
