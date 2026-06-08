"""
Evaluation metrics for all three inference tasks (C1/C2/C3).

Topology metrics (C1): graph-level evaluation — edge F1, graph edit distance.
Classification metrics (C2/C3): accuracy, macro-F1, per-class P/R/F1.
All results compared against random and majority-class baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class TopologyMetrics:
    """
    Edge-level metrics for C1 topology reconstruction.

    Treating topology inference as an edge prediction task:
      - true_edges: set of (src, dst) tuples from ground truth
      - pred_edges: set of (src, dst) tuples from model prediction
    """
    true_edges: set[tuple[str, str]]
    pred_edges: set[tuple[str, str]]

    @property
    def edge_precision(self) -> float:
        if not self.pred_edges:
            return 0.0
        return len(self.true_edges & self.pred_edges) / len(self.pred_edges)

    @property
    def edge_recall(self) -> float:
        if not self.true_edges:
            return 0.0
        return len(self.true_edges & self.pred_edges) / len(self.true_edges)

    @property
    def edge_f1(self) -> float:
        p, r = self.edge_precision, self.edge_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def graph_edit_distance(self) -> int:
        """
        Simplified GED: insertions + deletions needed to transform
        pred_edges into true_edges.
        """
        insertions = len(self.true_edges - self.pred_edges)
        deletions = len(self.pred_edges - self.true_edges)
        return insertions + deletions

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "graph_edit_distance": self.graph_edit_distance(),
            "true_edges": list(self.true_edges),
            "pred_edges": list(self.pred_edges),
        }


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
    task_name: str = "",
) -> dict[str, Any]:
    """
    Full classification metrics for C2 (roles) and C3 (workflow) tasks.
    Returns a dict ready to be serialised to JSON.
    """
    from collections import Counter
    n = len(y_true)
    assert n > 0

    result: dict[str, Any] = {
        "task": task_name,
        "n_samples": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class breakdown
    result["per_class"] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result["per_class"][cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    # Baselines
    n_classes = max(len(set(y_true)), 1)
    result["random_baseline_accuracy"] = 1.0 / n_classes
    majority = Counter(y_true).most_common(1)[0][0]
    result["majority_baseline_accuracy"] = float(accuracy_score(y_true, [majority] * n))
    result["majority_baseline_f1"] = float(
        f1_score(y_true, [majority] * n, average="macro", zero_division=0)
    )
    result["above_random"] = result["accuracy"] > result["random_baseline_accuracy"]
    result["above_majority"] = result["accuracy"] > result["majority_baseline_accuracy"]

    return result


def ablation_feature_importance(importances: dict[str, float], top_k: int = 20) -> list[tuple[str, float]]:
    """Return top-k features sorted by importance."""
    return sorted(importances.items(), key=lambda x: -x[1])[:top_k]
