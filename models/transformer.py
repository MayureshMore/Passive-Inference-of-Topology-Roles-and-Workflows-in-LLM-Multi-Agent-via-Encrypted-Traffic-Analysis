"""
Lightweight Transformer classifier for burst sequences.

Input: variable-length sequence of burst feature vectors (T × burst_dim)
       plus the inter-burst gap sequence (T-1,).
Output: class logits over workflow classes / topologies / roles.

Design rationale (from proposal §8.4):
  - Multi-agent workflows are long-running, asynchronous, and stateful.
  - Substantial idle gaps between delegation rounds carry timing signal.
  - A static model (RF) discards temporal ordering; a Transformer preserves it.
  - We keep the architecture minimal: one encoder with positional encoding
    derived from actual timestamps (not just sequence position), so gap
    magnitudes are encoded directly.
  - Runs on CPU/Metal; no GPU required.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GapAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that uses actual inter-burst gap durations rather
    than uniform step positions.  Each burst at absolute time t_i gets
    encoding PE(t_i), making the model aware of real-time spacing.
    """

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        self.d_model = d_model
        # Pre-compute standard sinusoidal table for quantised time bins
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, T, d_model)
        timestamps: (batch, T) absolute timestamps in seconds (optional).
                    If provided, quantise to 10ms bins for the PE lookup.
        """
        T = x.size(1)
        if timestamps is not None:
            # Normalise to bins of 10ms, clamp to max_len
            bins = (timestamps * 100).long().clamp(0, self.pe.size(0) - 1)
            # Use per-sample, per-step PE
            pe_vals = self.pe[bins]  # (batch, T, d_model)
        else:
            pe_vals = self.pe[:T].unsqueeze(0)  # (1, T, d_model)
        return x + pe_vals


class BurstTransformer(nn.Module):
    """
    Transformer-based classifier for A2A burst sequences.

    Architecture:
      - Input projection: burst_dim → d_model
      - Gap embedding: scalar gap → d_model / 4, concatenated after projection
      - Gap-aware positional encoding
      - N-layer Transformer encoder (PyTorch built-in)
      - CLS token pooling → classification head
    """

    def __init__(
        self,
        burst_dim: int = 10,
        n_classes: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(burst_dim, d_model)

        # Gap embedding (prepended between bursts)
        self.gap_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = GapAwarePositionalEncoding(d_model, max_len=max_seq_len * 2)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(
        self,
        burst_seq: torch.Tensor,       # (B, T, burst_dim)
        gap_seq: torch.Tensor,         # (B, T-1) inter-burst gaps
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T+1) True = pad
    ) -> torch.Tensor:
        # Guard: cap sequence length to max_seq_len so any caller (training via
        # BurstDataset, the validation loop that feeds raw sequences, or inference)
        # stays within the positional-encoding capacity and matches the truncated
        # length the model is trained on.  SSE burst sequences can exceed 1000.
        if burst_seq.size(1) > self.max_seq_len:
            burst_seq = burst_seq[:, : self.max_seq_len, :]
            if gap_seq is not None and gap_seq.dim() == 2 and gap_seq.size(1) > self.max_seq_len - 1:
                gap_seq = gap_seq[:, : self.max_seq_len - 1]
            if src_key_padding_mask is not None and src_key_padding_mask.size(1) > self.max_seq_len + 1:
                src_key_padding_mask = src_key_padding_mask[:, : self.max_seq_len + 1]

        B, T, _ = burst_seq.shape

        # Project burst features
        x = self.input_proj(burst_seq)  # (B, T, d_model)

        # Inject gap information as an additive signal on each burst
        if T > 1:
            gaps = gap_seq.unsqueeze(-1)  # (B, T-1, 1)
            gap_emb = self.gap_proj(gaps)   # (B, T-1, d_model)
            # Shift: gap[i] is the gap *before* burst[i+1], so add to x[:,1:,:]
            x = x.clone()
            x[:, 1:, :] = x[:, 1:, :] + gap_emb

        # Add positional encoding
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool via CLS token
        cls_out = x[:, 0, :]  # (B, d_model)
        return self.classifier(cls_out)  # (B, n_classes)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "burst_dim": self.input_proj.in_features,
                    "n_classes": self.classifier[-1].out_features,
                    "d_model": self.d_model,
                    "n_heads": self.encoder.layers[0].self_attn.num_heads,
                    "n_layers": len(self.encoder.layers),
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BurstTransformer":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


# ── Training utilities ────────────────────────────────────────────────────────

class BurstDataset(torch.utils.data.Dataset):
    """Dataset wrapping (burst_sequence, gap_sequence, label) triples."""

    def __init__(
        self,
        burst_seqs: list[torch.Tensor],
        gap_seqs: list[torch.Tensor],
        labels: torch.Tensor,
        max_len: int = 512,
    ) -> None:
        self.burst_seqs = burst_seqs
        self.gap_seqs = gap_seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        bs = self.burst_seqs[idx][: self.max_len]
        # Truncate gaps CONSISTENTLY with the (possibly truncated) burst sequence:
        # there are len(bs) - 1 inter-burst gaps among the kept bursts.  Using the
        # full (untruncated) burst length here produced gap tensors longer than the
        # burst buffer once SSE traffic pushed sequences past max_len → collate crash.
        gs = self.gap_seqs[idx][: max(len(bs) - 1, 0)]
        return bs, gs, self.labels[idx]


def collate_fn(batch):
    """Pad burst sequences to the same length within a batch."""
    burst_seqs, gap_seqs, labels = zip(*batch)
    max_t = max(b.size(0) for b in burst_seqs)
    burst_dim = burst_seqs[0].size(-1) if burst_seqs[0].ndim > 1 else 10

    padded_bursts = torch.zeros(len(batch), max_t, burst_dim)
    padded_gaps = torch.zeros(len(batch), max(1, max_t - 1))
    pad_mask = torch.ones(len(batch), max_t + 1, dtype=torch.bool)  # +1 for CLS

    for i, (bs, gs) in enumerate(zip(burst_seqs, gap_seqs)):
        t = bs.size(0)
        padded_bursts[i, :t] = bs
        pad_mask[i, 1 : t + 1] = False  # CLS always unmasked (index 0)
        # Defensive clamp: never write more gaps than the buffer (or burst) holds.
        gn = min(gs.size(0), padded_gaps.size(1), max(t - 1, 0))
        if gn > 0:
            padded_gaps[i, :gn] = gs[:gn]

    return padded_bursts, padded_gaps, pad_mask, torch.stack(labels)


def train_one_epoch(
    model: BurstTransformer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for burst_seq, gap_seq, pad_mask, labels in loader:
        burst_seq = burst_seq.to(device)
        gap_seq = gap_seq.to(device)
        pad_mask = pad_mask.to(device)
        labels = labels.to(device)

        logits = model(burst_seq, gap_seq, src_key_padding_mask=pad_mask)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
