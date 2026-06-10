"""
Closed-world evaluation (proposal §8.5).

Assumes the observer knows the complete set of agents and workflow classes
(closed-world assumption).  This establishes whether signal exists in the
traffic before moving to the harder open-world setting.

Runs stratified k-fold cross-validation for both RF baseline and Transformer,
then saves per-fold metrics and aggregate summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from .metrics import classification_metrics

logger = logging.getLogger(__name__)


class ClosedWorldEval:
    """
    Closed-world evaluation driver.

    Parameters
    ----------
    features : np.ndarray  (N, D) flat feature vectors
    labels   : list[str]   length N
    classes  : list[str]   complete class list
    task     : str         "workflow" | "role" | "topology"
    n_splits : int         k in k-fold CV
    groups   : list[str] | None  prompt-group keys for leak-free CV.
               When provided, uses StratifiedGroupKFold so that traces
               from the same prompt never appear in both train and test.
               Omitting this falls back to StratifiedKFold (use only for
               sanity checks — group leakage inflates accuracy).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: list[str],
        classes: list[str],
        task: str = "workflow",
        n_splits: int = 5,
        random_state: int = 42,
        groups: list[str] | None = None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.classes = classes
        self.task = task
        self.n_splits = n_splits
        self.random_state = random_state
        self.groups = groups  # prompt-group keys for leak-free CV

    def _make_splitter(self):
        """Return (splitter, use_groups) for CV fold generation."""
        if self.groups is not None:
            return StratifiedGroupKFold(n_splits=self.n_splits), True
        return StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                               random_state=self.random_state), False

    def run_rf(self, out_dir: Path | None = None) -> dict[str, Any]:
        """Run RF baseline with stratified (group-safe) CV."""
        from models.random_forest import RFClassifier

        clf = RFClassifier(task=self.task)
        cv_results = clf.cross_validate(
            self.features, self.labels,
            n_splits=self.n_splits,
            groups=self.groups,
        )

        # Also train on full data to get feature importances with human-readable names
        clf.fit(self.features, self.labels)
        try:
            from features.names import FLAT_FEATURE_NAMES, ROLE_FEATURE_NAMES
            feat_names = ROLE_FEATURE_NAMES() if self.task == "role" else FLAT_FEATURE_NAMES()
        except Exception:
            feat_names = None
        importances = clf.feature_importances(feature_names=feat_names)

        result = {
            "model": "random_forest",
            "task": self.task,
            "cv": cv_results,
            "top_features": list(importances.items())[:20],
        }

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"closed_world_rf_{self.task}.json").write_text(
                json.dumps(result, indent=2)
            )

        logger.info("Closed-world RF [%s]: %s", self.task, cv_results)
        return result

    def run_gbt(self, out_dir: Path | None = None) -> dict[str, Any]:
        """Run GBT (XGBoost) baseline with stratified (group-safe) CV."""
        from models.gradient_boosted import GBTClassifier

        clf = GBTClassifier(task=self.task)
        cv_results = clf.cross_validate(
            self.features, self.labels,
            n_splits=self.n_splits,
            groups=self.groups,
        )

        clf.fit(self.features, self.labels)
        try:
            from features.names import FLAT_FEATURE_NAMES, ROLE_FEATURE_NAMES
            feat_names = ROLE_FEATURE_NAMES() if self.task == "role" else FLAT_FEATURE_NAMES()
        except Exception:
            feat_names = None
        importances = clf.feature_importances(feature_names=feat_names)

        result = {
            "model": "gradient_boosted",
            "task": self.task,
            "cv": cv_results,
            "top_features": list(importances.items())[:20],
        }

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"closed_world_gbt_{self.task}.json").write_text(
                json.dumps(result, indent=2)
            )

        logger.info("Closed-world GBT [%s]: %s", self.task, cv_results)
        return result

    def run_cnn(
        self,
        burst_sequences: list[np.ndarray],
        gap_sequences: list[np.ndarray],
        out_dir: Path | None = None,
        n_epochs: int = 40,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> dict[str, Any]:
        """Run 1-D CNN on burst sequences with stratified CV."""
        from models.cnn1d import cnn_cross_validate

        cv_results = cnn_cross_validate(
            burst_seqs=burst_sequences,
            gap_seqs=gap_sequences,
            y=self.labels,
            task=self.task,
            n_splits=self.n_splits,
            groups=self.groups,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        result = {
            "model": "cnn1d",
            "task": self.task,
            "cv": cv_results,
        }

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"closed_world_cnn_{self.task}.json").write_text(
                json.dumps(result, indent=2)
            )

        logger.info("Closed-world CNN1D [%s]: %s", self.task, cv_results)
        return result

    def run_transformer(
        self,
        burst_sequences: list[np.ndarray],
        gap_sequences: list[np.ndarray],
        out_dir: Path | None = None,
        n_epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict[str, Any]:
        """Run Transformer with stratified CV."""
        import torch
        from models.transformer import (
            BurstDataset,
            BurstTransformer,
            collate_fn,
            train_one_epoch,
        )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info("Transformer device: %s", device)

        le = LabelEncoder()
        y_enc = le.fit_transform(self.labels)
        n_classes = len(le.classes_)

        kf, use_groups = self._make_splitter()
        fold_metrics: list[dict] = []

        split_args = (self.features, y_enc, self.groups) if use_groups else (self.features, y_enc)
        for fold, (train_idx, val_idx) in enumerate(kf.split(*split_args)):
            logger.info("Transformer fold %d/%d", fold + 1, self.n_splits)

            train_bs = [torch.tensor(burst_sequences[i], dtype=torch.float32) for i in train_idx]
            train_gs = [torch.tensor(gap_sequences[i], dtype=torch.float32) for i in train_idx]
            train_y = torch.tensor([y_enc[i] for i in train_idx], dtype=torch.long)

            val_bs = [torch.tensor(burst_sequences[i], dtype=torch.float32) for i in val_idx]
            val_gs = [torch.tensor(gap_sequences[i], dtype=torch.float32) for i in val_idx]
            val_y_enc = [y_enc[i] for i in val_idx]

            train_ds = BurstDataset(train_bs, train_gs, train_y)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )

            model = BurstTransformer(
                burst_dim=train_bs[0].size(-1) if train_bs[0].ndim > 1 else 10,
                n_classes=n_classes,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            for epoch in range(n_epochs):
                loss = train_one_epoch(model, train_loader, optimizer, device)
                scheduler.step()

            # Validation inference
            model.eval()
            val_preds: list[str] = []
            with torch.no_grad():
                for i, (bs, gs) in enumerate(zip(val_bs, val_gs)):
                    bs_t = bs.unsqueeze(0).to(device)
                    gs_t = gs.unsqueeze(0).to(device) if gs.numel() > 0 else torch.zeros(1, 0).to(device)
                    logits = model(bs_t, gs_t)
                    pred_idx = logits.argmax(dim=-1).item()
                    val_preds.append(str(le.inverse_transform([pred_idx])[0]))

            val_true = [self.labels[i] for i in val_idx]
            fold_m = classification_metrics(val_true, val_preds, list(le.classes_), self.task)
            fold_m["fold"] = fold
            fold_metrics.append(fold_m)

        # Aggregate
        agg: dict[str, Any] = {
            "model": "transformer",
            "task": self.task,
            "n_folds": self.n_splits,
            "folds": fold_metrics,
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
            "mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
            "std_macro_f1": float(np.std([m["macro_f1"] for m in fold_metrics])),
        }

        if out_dir:
            (Path(out_dir) / f"closed_world_transformer_{self.task}.json").write_text(
                json.dumps(agg, indent=2)
            )

        return agg
