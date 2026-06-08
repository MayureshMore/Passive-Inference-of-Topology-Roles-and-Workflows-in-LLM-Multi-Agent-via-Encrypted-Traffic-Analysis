"""
Evaluation utilities shared by closed-world, open-world, and cross-network
evaluation scripts.  All metrics are reported against random/majority-class
baselines so performance above chance is interpretable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """
    Compute and persist evaluation metrics for a single model/task combination.

    Metrics produced:
      - accuracy, macro-F1, macro-precision, macro-recall
      - per-class precision, recall, F1
      - confusion matrix
      - comparison against random and majority-class baselines
    """

    def __init__(self, task: str, classes: list[str]) -> None:
        self.task = task
        self.classes = classes

    def evaluate(
        self,
        y_true: list[str],
        y_pred: list[str],
        y_prob: np.ndarray | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        result["accuracy"] = float(accuracy_score(y_true, y_pred))
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        result["macro_precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        result["macro_recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )

        result["per_class"] = json.loads(
            json.dumps(
                classification_report(
                    y_true, y_pred, output_dict=True, zero_division=0
                )
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        result["confusion_matrix"] = cm.tolist()
        result["class_labels"] = self.classes

        # Baselines
        n = len(y_true)
        result["random_baseline_accuracy"] = 1.0 / max(len(self.classes), 1)
        from collections import Counter
        majority_class = Counter(y_true).most_common(1)[0][0]
        majority_preds = [majority_class] * n
        result["majority_baseline_accuracy"] = float(accuracy_score(y_true, majority_preds))
        result["majority_baseline_f1"] = float(
            f1_score(y_true, majority_preds, average="macro", zero_division=0)
        )

        result["above_random"] = result["accuracy"] > result["random_baseline_accuracy"]
        result["above_majority"] = result["accuracy"] > result["majority_baseline_accuracy"]

        return result

    def save(self, result: dict[str, Any], out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))

    def print_summary(self, result: dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print(f"Task: {self.task}")
        print(f"{'='*60}")
        print(f"  Accuracy:          {result['accuracy']:.3f}")
        print(f"  Macro F1:          {result['macro_f1']:.3f}")
        print(f"  Macro Precision:   {result['macro_precision']:.3f}")
        print(f"  Macro Recall:      {result['macro_recall']:.3f}")
        print(f"  Random baseline:   {result['random_baseline_accuracy']:.3f}")
        print(f"  Majority baseline: {result['majority_baseline_accuracy']:.3f}")
        print(f"  Above random?      {result['above_random']}")
        print(f"  Above majority?    {result['above_majority']}")
        print()


def compute_open_world_metrics(
    y_true: list[str],
    y_pred: list[str],
    y_prob: np.ndarray,
    known_classes: list[str],
    unknown_label: str = "unknown",
) -> dict[str, Any]:
    """
    Open-world evaluation: precision and recall for each known class against
    a background of unknown traffic.  This is the primary metric for the
    paper (not aggregate accuracy) per proposal §10.
    """
    result: dict[str, Any] = {"open_world": True, "known_classes": known_classes}
    for cls in known_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return result
