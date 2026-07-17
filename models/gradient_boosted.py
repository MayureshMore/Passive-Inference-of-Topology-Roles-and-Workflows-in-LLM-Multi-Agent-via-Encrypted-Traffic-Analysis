"""
Gradient-Boosted Trees classifier (scikit-learn HistGradientBoosting backend).

Same interface as RFClassifier so it can be swapped in anywhere RF is used.
HistGradientBoostingClassifier uses histogram-based binning (LightGBM-style)
and outperforms RandomForest on small-n tabular features (180–600 traces)
because it corrects its own errors sequentially rather than averaging
independent trees.  It also handles class imbalance natively via class_weight.

Why HistGradientBoosting over XGBoost: XGBoost 3.x has a runtime segfault
on macOS ARM64 (M3) when called inside a sklearn Pipeline — an OpenMP
thread-pool conflict that persists even after libomp installation.
HistGradientBoosting is pure sklearn, already installed, and stable on all
platforms.  Accuracy is comparable on structured feature vectors.

Supports three classification tasks:
  - "workflow"  : predict workflow class (C3)
  - "role"      : predict agent role per flow (C2)
  - "topology"  : predict topology type (C1, stretch goal)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class GBTClassifier:
    """
    Thin wrapper around HistGradientBoostingClassifier that handles label
    encoding, cross-validation, and serialization — mirroring the
    RFClassifier API exactly.

    Note: HistGradientBoosting normalises features internally via binning,
    so StandardScaler is a no-op here.  It is kept in the Pipeline to
    make the API identical to RFClassifier (where scaling matters for RF).
    """

    def __init__(
        self,
        task: str = "workflow",
        max_iter: int = 400,
        max_depth: int | None = None,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gbt",
                    HistGradientBoostingClassifier(
                        max_iter=max_iter,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: list[str]) -> "GBTClassifier":
        y_enc = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_enc)
        self._is_fitted = True
        # Cache permutation importance so feature_importances() works without
        # storing X (HistGBT has no built-in feature_importances_ attribute).
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(
            self.pipeline, X, y_enc, n_repeats=5, random_state=42, n_jobs=-1,
        )
        self._perm_importances = pi.importances_mean
        logger.info(
            "GBT [%s] fitted on %d samples, %d classes",
            self.task, len(y), len(self.label_encoder.classes_),
        )
        return self

    def predict(self, X: np.ndarray) -> list[str]:
        y_enc = self.pipeline.predict(X)
        return list(self.label_encoder.inverse_transform(y_enc))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def cross_validate(
        self,
        X: np.ndarray,
        y: list[str],
        n_splits: int = 5,
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Stratified k-fold CV with per-fold confusion matrix accumulation.
        Identical contract to RFClassifier.cross_validate.
        """
        self.label_encoder.fit(sorted(set(y)))
        class_names = list(self.label_encoder.classes_)
        y_enc = self.label_encoder.transform(y)

        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=n_splits)
            split_args = (X, y_enc, groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_args = (X, y_enc)

        fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []
        oof_true, oof_pred, oof_groups = [], [], []  # pooled out-of-fold preds (+ clusters) for CI
        cm_sum = np.zeros((len(class_names), len(class_names)), dtype=int)
        groups_arr = np.asarray(groups) if groups is not None else None

        for train_idx, test_idx in cv.split(*split_args):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y_enc[train_idx]
            y_te = y_enc[test_idx]

            self.pipeline.fit(X_tr, y_tr)
            y_pred = self.pipeline.predict(X_te)

            fold_accs.append(accuracy_score(y_te, y_pred))
            fold_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            fold_precs.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
            fold_recs.append(recall_score(y_te, y_pred, average="macro", zero_division=0))
            oof_true.extend(y_te.tolist())
            oof_pred.extend(list(y_pred))
            if groups_arr is not None:
                oof_groups.extend(groups_arr[test_idx].tolist())
            cm_sum += confusion_matrix(y_te, y_pred, labels=list(range(len(class_names))))

        # 95 % bootstrap CI on the pooled out-of-fold predictions. When the CV is cluster-aware
        # (groups given) the CI must be too — resample whole clusters, not correlated observations
        # (project convention; see evaluation/stats).
        from evaluation.stats import bootstrap_ci
        ci = bootstrap_ci(oof_true, oof_pred, classes=list(range(len(class_names))),
                          groups=oof_groups if groups_arr is not None else None)

        summary: dict[str, Any] = {
            "accuracy":         {"mean": float(np.mean(fold_accs)),  "std": float(np.std(fold_accs)),
                                 "ci_lo": ci["accuracy_ci_lo"], "ci_hi": ci["accuracy_ci_hi"]},
            "f1_macro":         {"mean": float(np.mean(fold_f1s)),   "std": float(np.std(fold_f1s)),
                                 "ci_lo": ci["macro_f1_ci_lo"], "ci_hi": ci["macro_f1_ci_hi"]},
            "precision_macro":  {"mean": float(np.mean(fold_precs)), "std": float(np.std(fold_precs))},
            "recall_macro":     {"mean": float(np.mean(fold_recs)),  "std": float(np.std(fold_recs))},
            "confusion_matrix": {
                "labels": class_names,
                "matrix": cm_sum.tolist(),
            },
            "ci_method": ci["ci_method"],
            "ci_n_clusters": ci["n_clusters"],
        }
        logger.info("GBT CV [%s]: %s", self.task, {k: v for k, v in summary.items() if k != "confusion_matrix"})
        return summary

    def feature_importances(self, feature_names: list[str] | None = None) -> dict[str, float]:
        importances = getattr(self, "_perm_importances", None)
        if importances is None:
            return {}
        if feature_names and len(feature_names) == len(importances):
            return dict(sorted(zip(feature_names, importances.tolist()), key=lambda x: -x[1]))
        return {str(i): float(v) for i, v in enumerate(importances)}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"pipeline": self.pipeline, "label_encoder": self.label_encoder, "task": self.task},
                f,
            )
        logger.info("GBT model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "GBTClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(task=state["task"])
        obj.pipeline = state["pipeline"]
        obj.label_encoder = state["label_encoder"]
        obj._is_fitted = True
        return obj
