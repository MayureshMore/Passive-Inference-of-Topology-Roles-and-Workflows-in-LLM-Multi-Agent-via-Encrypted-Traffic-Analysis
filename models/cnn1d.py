"""
1-D CNN classifier for burst sequences.

Treats each trace as a sequence of burst feature vectors (T × burst_dim)
and applies two Conv1d blocks followed by global average pooling.

Why CNN instead of Transformer at 180–600 traces:
  - Transformer needs self-attention over all pairs of bursts — O(T²) parameters
    relative to sequence complexity, overfit-prone at small n.
  - 1-D CNN has local receptive fields: adjacent bursts correlate (one
    delegation round triggers the next), so a kernel-3 conv captures the
    right inductive bias with far fewer parameters.
  - Shared weights across time positions mean the model generalises well
    even when traces have varying numbers of bursts.

Reuses BurstDataset and collate_fn from transformer.py to minimise
duplication.  The forward signature accepts gap_seq and pad_mask for
API compatibility but ignores them (CNN uses only burst features).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Model ──────────────────────────────────────────────────────────────────────

class BurstCNN1D:
    """
    Thin wrapper that owns model, optimizer, and label encoder,
    mirroring the RFClassifier / GBTClassifier interface for use
    inside ClosedWorldEval.run_cnn().
    """

    def __init__(
        self,
        task: str = "workflow",
        n_classes: int = 4,
        burst_dim: int = 10,
        channels: tuple[int, ...] = (64, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 40,
        batch_size: int = 16,
        random_state: int = 42,
    ) -> None:
        import torch
        from sklearn.preprocessing import LabelEncoder

        self.task = task
        self.n_classes = n_classes
        self.burst_dim = burst_dim
        self.channels = channels
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._model: "_CNN1DNet | None" = None

    def _build_model(self, burst_dim: int, n_classes: int) -> "_CNN1DNet":
        return _CNN1DNet(burst_dim, n_classes, self.channels, self.dropout).to(self.device)


class _CNN1DNet:
    """Pure PyTorch nn.Module — separated so BurstCNN1D stays picklable."""
    pass


import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNN1DModule(nn.Module):
    def __init__(
        self,
        burst_dim: int,
        n_classes: int,
        channels: tuple[int, ...] = (64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = burst_dim
        for out_ch in channels[:-1]:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            ]
            in_ch = out_ch
        # Last conv + global average pool (no MaxPool)
        out_ch = channels[-1]
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        ]
        self.conv_blocks = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, out_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch // 2, n_classes),
        )

    def forward(
        self,
        burst_seq: torch.Tensor,    # (B, T, burst_dim)
        gap_seq: torch.Tensor | None = None,   # unused — API compat
        pad_mask: torch.Tensor | None = None,  # unused — API compat
    ) -> torch.Tensor:
        x = burst_seq.transpose(1, 2)   # (B, burst_dim, T)
        x = self.conv_blocks(x)         # (B, channels[-1], 1)
        return self.classifier(x)       # (B, n_classes)


# ── Training utilities ─────────────────────────────────────────────────────────

def _train_cnn(
    model: _CNN1DModule,
    burst_seqs: list[np.ndarray],
    gap_seqs: list[np.ndarray],
    y_enc: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    from models.transformer import BurstDataset, collate_fn

    y_tensor = torch.tensor(y_enc, dtype=torch.long)
    bs_tensors = [torch.tensor(b, dtype=torch.float32) for b in burst_seqs]
    gs_tensors = [torch.tensor(g, dtype=torch.float32) for g in gap_seqs]
    dataset = BurstDataset(bs_tensors, gs_tensors, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for _ in range(n_epochs):
        for burst_seq, gap_seq, pad_mask, labels in loader:
            burst_seq = burst_seq.to(device)
            labels = labels.to(device)
            logits = model(burst_seq)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()


def _eval_cnn(
    model: _CNN1DModule,
    burst_seqs: list[np.ndarray],
    gap_seqs: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Return (N, n_classes) softmax probabilities."""
    from models.transformer import BurstDataset, collate_fn

    dummy_labels = torch.zeros(len(burst_seqs), dtype=torch.long)
    bs_tensors = [torch.tensor(b, dtype=torch.float32) for b in burst_seqs]
    gs_tensors = [torch.tensor(g, dtype=torch.float32) for g in gap_seqs]
    dataset = BurstDataset(bs_tensors, gs_tensors, dummy_labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for burst_seq, _, _, _ in loader:
            burst_seq = burst_seq.to(device)
            logits = model(burst_seq)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ── Cross-validation entry point ───────────────────────────────────────────────

def cnn_cross_validate(
    burst_seqs: list[np.ndarray],
    gap_seqs: list[np.ndarray],
    y: list[str],
    task: str,
    n_splits: int = 5,
    groups: list[str] | None = None,
    n_epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Stratified k-fold CV for CNN1D on burst sequences.
    Returns the same summary dict structure as RFClassifier.cross_validate.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    )
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(sorted(set(y)))
    class_names = list(le.classes_)
    y_enc = le.transform(y)
    n_classes = len(class_names)

    # Infer burst_dim from first non-empty burst sequence
    burst_dim = next(
        (b.shape[-1] if b.ndim > 1 else 1 for b in burst_seqs if len(b) > 0), 10
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("CNN1D device: %s  burst_dim=%d  n_classes=%d", device, burst_dim, n_classes)

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        split_args_base = (np.zeros((len(y), 1)), y_enc, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_args_base = (np.zeros((len(y), 1)), y_enc)

    fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []
    cm_sum = np.zeros((n_classes, n_classes), dtype=int)

    torch.manual_seed(random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(*split_args_base)):
        logger.info("CNN1D fold %d/%d", fold_idx + 1, n_splits)

        burst_tr = [burst_seqs[i] for i in train_idx]
        gap_tr   = [gap_seqs[i]   for i in train_idx]
        y_tr     = y_enc[train_idx]

        burst_te = [burst_seqs[i] for i in test_idx]
        gap_te   = [gap_seqs[i]   for i in test_idx]
        y_te     = y_enc[test_idx]

        model = _CNN1DModule(burst_dim, n_classes, dropout=0.3).to(device)
        _train_cnn(model, burst_tr, gap_tr, y_tr, n_epochs, batch_size, lr, weight_decay, device)

        probs = _eval_cnn(model, burst_te, gap_te, device)
        y_pred = probs.argmax(axis=1)

        fold_accs.append(accuracy_score(y_te, y_pred))
        fold_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        fold_precs.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        fold_recs.append(recall_score(y_te, y_pred, average="macro", zero_division=0))
        cm_sum += confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))

    summary: dict[str, Any] = {
        "accuracy":         {"mean": float(np.mean(fold_accs)),  "std": float(np.std(fold_accs))},
        "f1_macro":         {"mean": float(np.mean(fold_f1s)),   "std": float(np.std(fold_f1s))},
        "precision_macro":  {"mean": float(np.mean(fold_precs)), "std": float(np.std(fold_precs))},
        "recall_macro":     {"mean": float(np.mean(fold_recs)),  "std": float(np.std(fold_recs))},
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm_sum.tolist(),
        },
    }
    logger.info("CNN1D CV [%s]: %s", task, {k: v for k, v in summary.items() if k != "confusion_matrix"})
    return summary
