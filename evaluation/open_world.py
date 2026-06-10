"""
Open-world evaluation (proposal §8.5, §10).

The open-world setting is MANDATORY per the proposal.  Injects unseen
agents, workflows, and background traffic.  Reports precision and recall
at realistic base rates rather than aggregate accuracy (which is misleading
under class imbalance).

Background classes:
  (a) ordinary web/API traffic — captured from non-agent HTTP traffic
  (b) non-target agentic traffic — A2A traffic from a different workflow
      class or a different agent system
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .metrics import classification_metrics, TopologyMetrics

logger = logging.getLogger(__name__)

# Label used for traffic that belongs to none of the known classes
UNKNOWN_LABEL = "unknown"


class OpenWorldEval:
    """
    Open-world evaluator.

    Parameters
    ----------
    known_features : np.ndarray  features from known workflow/role classes
    known_labels   : list[str]
    unknown_features : np.ndarray  features from background / unknown classes
    task : str
    threshold : float  probability threshold below which a sample is
                       classified as "unknown" rather than a known class
    """

    def __init__(
        self,
        known_features: np.ndarray,
        known_labels: list[str],
        unknown_features: np.ndarray,
        task: str = "workflow",
        threshold: float = 0.6,
    ) -> None:
        self.known_features = known_features
        self.known_labels = known_labels
        self.unknown_features = unknown_features
        self.task = task
        self.threshold = threshold
        self.classes = sorted(set(known_labels))

    def run(
        self,
        model,  # fitted RFClassifier or Transformer wrapper
        out_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate on (known + unknown) combined test set.
        Reports precision/recall/F1 per known class with UNKNOWN_LABEL as background.
        """
        # Build combined test set
        all_features = np.concatenate([self.known_features, self.unknown_features], axis=0)
        all_true = self.known_labels + [UNKNOWN_LABEL] * len(self.unknown_features)

        # Predict with threshold-based rejection
        proba = model.predict_proba(all_features)  # (N, n_classes)
        max_proba = proba.max(axis=1)
        raw_preds = model.predict(all_features)
        preds = [
            p if max_proba[i] >= self.threshold else UNKNOWN_LABEL
            for i, p in enumerate(raw_preds)
        ]

        # Per-class precision/recall against UNKNOWN background
        result: dict[str, Any] = {
            "task": self.task,
            "threshold": self.threshold,
            "n_known": len(self.known_labels),
            "n_unknown": len(self.unknown_features),
            "per_class": {},
        }

        all_classes = self.classes + [UNKNOWN_LABEL]
        for cls in self.classes:
            tp = sum(1 for t, p in zip(all_true, preds) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(all_true, preds) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(all_true, preds) if t == cls and p != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            result["per_class"][cls] = {
                "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Unknown rejection rate: fraction of unknown samples correctly rejected
        n_unknown_correctly_rejected = sum(
            1 for t, p in zip(all_true, preds)
            if t == UNKNOWN_LABEL and p == UNKNOWN_LABEL
        )
        result["unknown_rejection_rate"] = (
            n_unknown_correctly_rejected / max(len(self.unknown_features), 1)
        )

        # Known-class false positive rate: fraction of KNOWN samples incorrectly
        # abstained on (rejected as unknown).  By design the threshold retains
        # ~95% of known traffic, so this should be ~5%; deviations indicate the
        # calibration is not well-fit to this class distribution.
        n_known_rejected = sum(
            1 for t, p in zip(all_true, preds)
            if t != UNKNOWN_LABEL and p == UNKNOWN_LABEL
        )
        result["known_fpr"] = n_known_rejected / max(len(self.known_labels), 1)

        # Precision-recall curves at multiple thresholds
        pr_curves: dict[str, list] = {}
        for cls in self.classes:
            cls_true = [1 if t == cls else 0 for t in all_true]
            # Use class index probability as score
            le = LabelEncoder().fit(self.classes)
            if cls in le.classes_:
                cls_idx = list(le.classes_).index(cls)
                if cls_idx < proba.shape[1]:
                    scores = proba[:, cls_idx].tolist()
                else:
                    scores = max_proba.tolist()
            else:
                scores = max_proba.tolist()
            pr_curves[cls] = _precision_recall_at_thresholds(cls_true, scores)
        result["pr_curves"] = pr_curves

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"open_world_{self.task}.json").write_text(
                json.dumps(result, indent=2)
            )

        logger.info("Open-world [%s] results: %s", self.task, result["per_class"])
        return result


def _precision_recall_at_thresholds(
    y_binary: list[int],
    scores: list[float],
    n_thresholds: int = 20,
) -> list[dict]:
    """Compute precision/recall at N evenly-spaced thresholds."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    points = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for a, b in zip(y_binary, preds) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_binary, preds) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_binary, preds) if a == 1 and b == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        points.append({"threshold": float(t), "precision": prec, "recall": rec})
    return points
