#!/usr/bin/env python3
"""
Model training script.

Loads processed feature files from data/processed/, trains the selected
model (RF baseline or Transformer), and saves the checkpoint + metrics.

Usage:
    # RF baseline, workflow classification
    python scripts/train_models.py --task workflow --model rf

    # Transformer, role classification
    python scripts/train_models.py --task role --model transformer --epochs 50

    # RF baseline, all tasks
    python scripts/train_models.py --task all --model rf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
RESULTS_DIR = Path("data/results")


def load_dataset(
    task: str,
) -> tuple[np.ndarray, list[str], list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load flat features, labels, burst sequences, gap sequences, and prompt groups
    from .npz files in data/processed/.

    Returns (X, y, classes, burst_seqs, gap_seqs, groups)
    groups is a list of prompt_group hashes for GroupKFold leak-free CV.
    """
    npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files in {PROCESSED_DIR}. "
            "Run scripts/extract_features.py first."
        )

    X_list, y_list, burst_list, gap_list, group_list = [], [], [], [], []
    label_file = PROCESSED_DIR / "labels.json"

    if not label_file.exists():
        raise FileNotFoundError(
            f"No labels.json in {PROCESSED_DIR}. "
            "Run scripts/extract_features.py first."
        )

    labels_map: dict[str, dict] = json.loads(label_file.read_text())

    for npz_path in npz_files:
        run_id = npz_path.stem
        is_role_sample = "__role__" in run_id
        if task == "role" and not is_role_sample:
            continue
        if task != "role" and is_role_sample:
            continue
        if run_id not in labels_map:
            continue
        info = labels_map[run_id]

        label = info.get(task)
        if label is None:
            continue

        d = np.load(npz_path, allow_pickle=False)
        X_list.append(d["flat"])
        y_list.append(label)
        burst_list.append(d["burst_sequence"])
        gap_list.append(d["gap_sequence"])
        group_list.append(info.get("prompt_group", run_id))

    if not X_list:
        raise ValueError(f"No labeled samples found for task={task}")

    X = np.stack(X_list, axis=0)
    classes = sorted(set(y_list))
    logger.info("Loaded %d samples for task=%s, classes=%s", len(y_list), task, classes)
    return X, y_list, classes, burst_list, gap_list, group_list


def train_rf(task: str) -> None:
    from evaluation.closed_world import ClosedWorldEval

    X, y, classes, _, _, groups = load_dataset(task)
    evaluator = ClosedWorldEval(X, y, classes, task=task, groups=groups)
    result = evaluator.run_rf(out_dir=RESULTS_DIR / "closed_world")

    model_path = MODELS_DIR / f"rf_{task}.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from models.random_forest import RFClassifier
    clf = RFClassifier(task=task)
    clf.fit(X, y)
    clf.save(model_path)
    logger.info("RF model saved → %s", model_path)
    logger.info("CV results: %s", result["cv"])


def train_gbt(task: str) -> None:
    from evaluation.closed_world import ClosedWorldEval

    X, y, classes, _, _, groups = load_dataset(task)
    evaluator = ClosedWorldEval(X, y, classes, task=task, groups=groups)
    result = evaluator.run_gbt(out_dir=RESULTS_DIR / "closed_world")

    model_path = MODELS_DIR / f"gbt_{task}.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from models.gradient_boosted import GBTClassifier
    clf = GBTClassifier(task=task)
    clf.fit(X, y)
    clf.save(model_path)
    logger.info("GBT model saved → %s", model_path)
    logger.info("CV results: %s", result["cv"])


def train_cnn(task: str, epochs: int = 40) -> None:
    from evaluation.closed_world import ClosedWorldEval

    X, y, classes, burst_seqs, gap_seqs, groups = load_dataset(task)
    evaluator = ClosedWorldEval(X, y, classes, task=task, groups=groups)
    result = evaluator.run_cnn(
        burst_sequences=burst_seqs,
        gap_sequences=gap_seqs,
        out_dir=RESULTS_DIR / "closed_world",
        n_epochs=epochs,
    )
    cv = result["cv"]
    logger.info(
        "CNN1D CV [%s]: accuracy=%.3f±%.3f  macro_f1=%.3f±%.3f",
        task,
        cv["accuracy"]["mean"], cv["accuracy"]["std"],
        cv["f1_macro"]["mean"], cv["f1_macro"]["std"],
    )


def train_transformer(task: str, epochs: int = 30) -> None:
    from evaluation.closed_world import ClosedWorldEval

    X, y, classes, burst_seqs, gap_seqs, groups = load_dataset(task)
    evaluator = ClosedWorldEval(X, y, classes, task=task, groups=groups)
    result = evaluator.run_transformer(
        burst_sequences=burst_seqs,
        gap_sequences=gap_seqs,
        out_dir=RESULTS_DIR / "closed_world",
        n_epochs=epochs,
    )
    logger.info(
        "Transformer CV: accuracy=%.3f±%.3f, macro_f1=%.3f±%.3f",
        result["mean_accuracy"], result["std_accuracy"],
        result["mean_macro_f1"], result["std_macro_f1"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2A fingerprinting models")
    parser.add_argument("--task", required=True,
                        choices=["workflow", "role", "parallelism", "topology", "all"])
    parser.add_argument("--model", required=True,
                        choices=["rf", "gbt", "cnn", "transformer", "all"])
    parser.add_argument("--epochs", type=int, default=40,
                        help="Epochs for CNN / Transformer training (default 40)")
    args = parser.parse_args()

    tasks = ["workflow", "role", "parallelism", "topology"] if args.task == "all" else [args.task]

    for task in tasks:
        logger.info("=== Training task: %s ===", task)
        if args.model in ("rf", "all"):
            try:
                train_rf(task)
            except Exception as exc:
                logger.error("RF training failed for %s: %s", task, exc)

        if args.model in ("gbt", "all"):
            try:
                train_gbt(task)
            except Exception as exc:
                logger.error("GBT training failed for %s: %s", task, exc)

        if args.model in ("cnn", "all"):
            try:
                train_cnn(task, epochs=args.epochs)
            except Exception as exc:
                logger.error("CNN1D training failed for %s: %s", task, exc)

        if args.model in ("transformer", "all"):
            try:
                train_transformer(task, epochs=args.epochs)
            except Exception as exc:
                logger.error("Transformer training failed for %s: %s", task, exc)
