"""
Regression tests for deep-model sequence-length handling.

The SSE migration produces burst sequences far longer than the models' max_seq_len
(observed up to ~1260 vs 512).  Two bugs followed and are guarded here:
  1. BurstDataset truncated bursts to max_len but gaps to the FULL length, so
     collate_fn crashed assigning an over-long gap into a (max_len-1) buffer.
  2. The Transformer validation path fed raw (untruncated) sequences straight to
     the model, overflowing the positional-encoding buffer.

Both must now handle over-long sequences without crashing.
"""

from __future__ import annotations

import numpy as np
import torch

from models.transformer import BurstDataset, collate_fn, BurstTransformer

BURST_DIM = 10
MAX_LEN = 512


def _seq(n):
    return torch.tensor(np.random.RandomState(0).randn(n, BURST_DIM), dtype=torch.float32)


def test_collate_handles_oversize_sequences():
    # Mixed batch incl. a sequence well past max_len; gaps len = burst len - 1.
    lengths = [30, 600, 1260]
    bursts = [_seq(n) for n in lengths]
    gaps = [torch.tensor(np.random.RandomState(1).randn(n - 1), dtype=torch.float32) for n in lengths]
    ds = BurstDataset(bursts, gaps, torch.tensor([0, 1, 2]), max_len=MAX_LEN)
    batch = [ds[i] for i in range(len(ds))]
    padded_bursts, padded_gaps, pad_mask, labels = collate_fn(batch)
    T = padded_bursts.size(1)
    assert T <= MAX_LEN                         # truncated to the cap
    assert padded_gaps.size(1) == max(1, T - 1)  # gaps align with bursts
    assert pad_mask.size(1) == T + 1             # +1 for CLS


def test_transformer_forward_survives_oversize_input():
    model = BurstTransformer(burst_dim=BURST_DIM, n_classes=4, max_seq_len=MAX_LEN).eval()
    # Raw, untruncated 1260-long sequence fed directly (the validation-path case).
    burst = _seq(1260).unsqueeze(0)             # (1, 1260, dim)
    gap = torch.zeros(1, 1259)
    with torch.no_grad():
        out = model(burst, gap)
    assert out.shape == (1, 4)                  # no crash, correct output shape
