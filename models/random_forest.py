"""
Random Forest baseline classifier.

Operates on flat feature vectors (per-flow mean + per-system stats).
Used to establish how much signal classic engineered features capture
before applying the Transformer model.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class RFClassifier:
    """
    Thin wrapper around sklearn RandomForest that handles label encoding,
    feature scaling, cross-validation, and serialization.
    """

    def __init__(
        self,
        task: str = "workflow",
        n_estimators: int = 300,
        max_depth: int | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: list[str]) -> "RFClassifier":
        y_enc = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_enc)
        self._is_fitted = True
        logger.info(
            "RF [%s] fitted on %d samples, %d classes",
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
        Stratified k-fold CV. When groups is provided, uses StratifiedGroupKFold
        so that traces from the same prompt never span train and test folds.
        """
        y_enc = self.label_encoder.fit_transform(y)
        if groups is not None:
            # StratifiedGroupKFold does not shuffle; order is deterministic
            cv = StratifiedGroupKFold(n_splits=n_splits)
            cv_groups = groups
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_groups = None
        scores = cross_validate(
            self.pipeline,
            X,
            y_enc,
            cv=cv,
            groups=cv_groups,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            return_train_score=False,
        )
        summary: dict[str, Any] = {}
        for k, v in scores.items():
            if k.startswith("test_"):
                name = k[5:]
                summary[name] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        logger.info("CV results [%s]: %s", self.task, summary)
        return summary

    def feature_importances(self, feature_names: list[str] | None = None) -> dict[str, float]:
        rf = self.pipeline.named_steps["rf"]
        importances = rf.feature_importances_
        if feature_names and len(feature_names) == len(importances):
            return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))
        return {str(i): float(v) for i, v in enumerate(importances)}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "label_encoder": self.label_encoder, "task": self.task}, f)
        logger.info("RF model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "RFClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(task=state["task"])
        obj.pipeline = state["pipeline"]
        obj.label_encoder = state["label_encoder"]
        obj._is_fitted = True
        return obj
