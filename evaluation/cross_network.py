"""
Cross-network / robustness evaluation (C5, proposal §8.5).

Train on traffic captured on one network path (e.g., US local testbed)
and test on another (e.g., US-India WAN link), measuring how much
accuracy degrades under distribution shift.

Also runs the data-volume scaling ablation: train on fractions of the
training set and plot accuracy vs. number of traces.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import classification_metrics

logger = logging.getLogger(__name__)


class CrossNetworkEval:
    """
    Train-on-A / test-on-B evaluation.

    Parameters
    ----------
    train_features : np.ndarray  (N_train, D)
    train_labels   : list[str]
    test_features  : np.ndarray  (N_test, D)
    test_labels    : list[str]
    train_network  : str   human label for training network (e.g. "us_local")
    test_network   : str   human label for test network (e.g. "us_india_wan")
    task           : str
    """

    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: list[str],
        test_features: np.ndarray,
        test_labels: list[str],
        train_network: str = "train",
        test_network: str = "test",
        task: str = "workflow",
    ) -> None:
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.train_network = train_network
        self.test_network = test_network
        self.task = task
        self.classes = sorted(set(train_labels + test_labels))

    def run_rf(self, out_dir: Path | None = None) -> dict[str, Any]:
        from models.random_forest import RFClassifier

        clf = RFClassifier(task=self.task)
        clf.fit(self.train_features, self.train_labels)
        preds = clf.predict(self.test_features)

        result = {
            "model": "random_forest",
            "task": self.task,
            "train_network": self.train_network,
            "test_network": self.test_network,
            "n_train": len(self.train_labels),
            "n_test": len(self.test_labels),
            **classification_metrics(self.test_labels, preds, self.classes, self.task),
        }

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"cross_network_rf_{self.task}.json").write_text(
                json.dumps(result, indent=2)
            )

        logger.info(
            "Cross-network RF [%s] %s→%s: accuracy=%.3f",
            self.task, self.train_network, self.test_network, result["accuracy"],
        )
        return result

    def data_volume_ablation(
        self,
        fractions: list[float] | None = None,
        n_repeats: int = 5,
        out_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        Train on increasing fractions of the training set, test on the full
        test set.  Shows how much data is needed for stable results.
        """
        from models.random_forest import RFClassifier

        fractions = fractions or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        results: list[dict] = []

        for frac in fractions:
            fold_accs: list[float] = []
            fold_f1s: list[float] = []
            n_train = max(int(len(self.train_labels) * frac), 2)

            for _ in range(n_repeats):
                idx = np.random.choice(len(self.train_labels), size=n_train, replace=False)
                X_sub = self.train_features[idx]
                y_sub = [self.train_labels[i] for i in idx]

                # Skip if only one class represented
                if len(set(y_sub)) < 2:
                    continue

                clf = RFClassifier(task=self.task)
                clf.fit(X_sub, y_sub)
                preds = clf.predict(self.test_features)
                m = classification_metrics(self.test_labels, preds, self.classes, self.task)
                fold_accs.append(m["accuracy"])
                fold_f1s.append(m["macro_f1"])

            if fold_accs:
                results.append({
                    "fraction": frac,
                    "n_train": n_train,
                    "mean_accuracy": float(np.mean(fold_accs)),
                    "std_accuracy": float(np.std(fold_accs)),
                    "mean_macro_f1": float(np.mean(fold_f1s)),
                    "std_macro_f1": float(np.std(fold_f1s)),
                })

        agg = {
            "task": self.task,
            "ablation": "data_volume",
            "train_network": self.train_network,
            "test_network": self.test_network,
            "points": results,
        }

        if out_dir:
            (Path(out_dir) / f"ablation_volume_{self.task}.json").write_text(
                json.dumps(agg, indent=2)
            )

        return agg
