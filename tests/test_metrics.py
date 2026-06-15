"""Unit tests for evaluation metrics and the above-chance retention formula."""
import math

from evaluation.metrics import classification_metrics


def test_classification_metrics_basic():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "a"]  # 3/4 correct
    m = classification_metrics(y_true, y_pred, classes=["a", "b"], task_name="t")
    assert math.isclose(m["accuracy"], 0.75)
    assert m["n_samples"] == 4
    assert math.isclose(m["random_baseline_accuracy"], 0.5)
    assert m["above_random"] is True
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert set(m["per_class"]) == {"a", "b"}


def test_classification_metrics_perfect():
    y = ["x", "y", "z"]
    m = classification_metrics(y, y, classes=["x", "y", "z"])
    assert m["accuracy"] == 1.0
    assert math.isclose(m["macro_f1"], 1.0)


def above_chance_retention(transfer_acc, ceiling_acc, chance):
    """The metric used throughout the writeup."""
    return (transfer_acc - chance) / (ceiling_acc - chance)


def test_above_chance_retention_formula():
    # workflow: transfer 0.395, ceiling 0.735, chance 0.25 -> ~0.30
    r = above_chance_retention(0.395, 0.735, 0.25)
    assert math.isclose(r, (0.395 - 0.25) / (0.735 - 0.25), rel_tol=1e-9)
    assert 0.29 < r < 0.31
    # at chance -> 0 retention; at ceiling -> 1.0
    assert math.isclose(above_chance_retention(0.25, 0.735, 0.25), 0.0)
    assert math.isclose(above_chance_retention(0.735, 0.735, 0.25), 1.0)
