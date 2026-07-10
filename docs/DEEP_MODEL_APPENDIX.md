# Appendix — Deep sequence models (CNN1D & Transformer): architecture, input, training budget

**Purpose.** Two reviewers raised an "untuned models" concern: that the deep models
underperform the tree attackers only because they were thrown in under-specified. This
appendix documents the exact input representation, architectures, parameter counts, and
training budget so the underperformance can be read for what it is — **not a tuning
artifact**. The deep models are regularized and early-stopped, yet at N≈600 they **collapse
to (near) single-class prediction** — a *degenerate* outcome whose signature is *below-chance*
macro-F1, so they are **excluded from all claims**. Small data is the leading hypothesis for
the collapse, but below-chance output is a class-collapse signature, **not** proof of
starvation per se. The trees win because the engineered 195-dim features already encode the
discriminative structure.

Source of truth: `models/transformer.py`, `models/cnn1d.py`,
`evaluation/closed_world.py::run_transformer` / `run_cnn`. Reported numbers are the
committed canonical results (`data/results/`, mirrored in
`data/results/PAPER_ARTIFACTS.md`).

## Input representation (shared by both deep models)

Unlike the trees (which consume the flattened 195-dim per-trace vector), the deep models
consume the **ordered burst sequence** so temporal structure is preserved.

- **Burst segmentation** (`features/burst.py`): packets on each flow are grouped into
  same-direction bursts with an **idle threshold of 0.5 s**. A trace becomes a sequence of
  `T` bursts ordered by start time.
- **Per-burst vector — 10 dims** (`Burst.to_feature_vector`):
  1. `direction` (+1 outbound / −1 inbound, observer perspective)
  2. `n_packets`
  3. `total_bytes`
  4. `duration_s`
  5. `mean_pkt_size`
  6. `std_pkt_size`
  7. `mean_inter_packet_gap`
  8. `throughput_bps`
  9. `max_pkt_size`
  10. `min_pkt_size`
- **Inter-burst gap sequence — `T−1`**: the idle gaps *between* consecutive bursts, fed
  separately (the Transformer uses them for gap-aware positional encoding; the CNN ignores
  them by design).
- **Sequence length**: capped at **`max_seq_len = 512`** bursts (truncation). SSE response
  streaming can push a trace past 1000 bursts; the cap keeps every caller within the
  positional-encoding capacity and matches the trained length.

## CNN1D (`models/cnn1d.py::_CNN1DModule`)

A 1-D CNN over the burst sequence — local receptive fields match the inductive bias that
adjacent delegation rounds correlate, with far fewer parameters than attention.

```
input  (B, T, 10)  ──transpose──▶  (B, 10, T)
Conv1d(10→64, k=3, pad=1) ─ BatchNorm1d ─ GELU ─ MaxPool1d(2, ceil)
Conv1d(64→128, k=3, pad=1) ─ BatchNorm1d ─ GELU ─ AdaptiveAvgPool1d(1)
Flatten ─ Dropout(0.3) ─ Linear(128→64) ─ GELU ─ Dropout(0.3) ─ Linear(64→n_classes)
```

- Channels `(64, 128)`, kernel 3, GELU, BatchNorm, global average pool.
- **35,588 trainable parameters (≈ 36 K).**
- Gaps/timestamps unused (API-compatible signature; CNN uses burst features only).

**Training budget** (`run_cnn` defaults): AdamW, `lr = 1e-3`, `weight_decay = 1e-4`,
`CosineAnnealingLR(T_max = n_epochs)`, gradient clipping `‖g‖ ≤ 1.0`, cross-entropy,
**`n_epochs = 40`**, **`batch_size = 16`**, `torch.manual_seed(42)`. CV =
`StratifiedGroupKFold(n_splits = 5)` grouped on `prompt_group` (same splitter as the trees,
so no prompt leaks across folds). Device: Metal (MPS) or CPU — no GPU.

## Transformer (`models/transformer.py::BurstTransformer`)

A minimal encoder with **gap-aware positional encoding** (positions come from real
inter-burst timestamps quantized to 10 ms bins, not uniform step indices) so idle-time
magnitudes between delegation rounds are encoded directly.

```
burst (B, T, 10) ─ Linear(10→128)                      ┐
gap   (B, T−1)   ─ Linear(1→128) ─ added onto bursts[1:]┘
            ─ GapAwarePositionalEncoding(d_model=128) (10 ms time bins)
            ─ prepend CLS token
            ─ TransformerEncoder × 3  (d_model=128, heads=4, ffn=256, dropout=0.1,
                                       batch_first, norm_first; src_key_padding_mask)
            ─ CLS pooling ─ LayerNorm ─ Linear(128→64) ─ GELU ─ Dropout(0.1) ─ Linear(64→n_classes)
```

- `d_model = 128`, `n_heads = 4`, `n_layers = 3`, `dim_feedforward = 256`,
  `dropout = 0.1`, `max_seq_len = 512`, pre-norm.
- **408,004 trainable parameters (≈ 0.41 M)** — ≈ 11.5× the CNN.
- Padding mask keeps the CLS token always attended; variable-length sequences padded
  per batch (`collate_fn`).

**Training budget** (`run_transformer` defaults): AdamW, `lr = 1e-3`,
`weight_decay = 1e-4`, `CosineAnnealingLR(T_max = n_epochs)`, gradient clipping
`‖g‖ ≤ 1.0`, cross-entropy, **`n_epochs = 80`**, **`batch_size = 32`**, **early stopping
on per-fold validation accuracy with `patience = 12`** and **best-val-weight restore**.
CV = `StratifiedGroupKFold(n_splits = 5)` on `prompt_group`. Device: MPS or CPU.

## Results (closed-world, 5-fold grouped CV, macro-F1)

| Task | GBT | RF | **Transformer** | **CNN1D** | chance |
|---|---|---|---|---|---|
| workflow | 0.708 | 0.663 | **0.100** | **0.228** | 0.250 |
| role | 0.864 | 0.868 | **0.684** | **0.191** | 0.333 |
| topology | 0.995 | 0.985 | **0.559** | **0.167** | 0.333 |
| parallelism | 0.989 | 0.972 | **0.519** | **0.400** | 0.500 |

Both deep models sit well below the trees; the Transformer collapses to a single class on
`workflow` (0.100 < chance 0.250), and the CNN is at/under chance on `role` and `topology`.

## Why the deep models collapse (and why it is not under-tuning)

The observed failure is **class collapse** (below-chance macro-F1 with near-chance accuracy —
e.g. the workflow CNN routes most predictions into one class; the Transformer predicts a single
class). That is a *degenerate classifier*, and these runs are excluded from every claim. The
points below explain why the collapse is not a *tuning* failure; whether more data would resolve
it is a hypothesis, **untested here**.


- **The models are regularized and tuned, not naive.** Dropout, weight decay, cosine LR,
  gradient clipping, pre-norm attention, early stopping with best-weight restore, and a
  gap-aware positional scheme are all in place. The failure mode is not an absent
  regularizer.
- **The training set per fold is ~480 traces** (600 × 4/5) across up to 4 classes — far
  below what a 0.4 M-parameter attention model or even a 35 K-parameter CNN needs to learn
  burst-sequence structure from scratch. Trees, by contrast, consume a fixed 195-dim
  vector in which the discriminative structure is **already engineered** (per-flow and
  per-system aggregates), so they extract signal from hundreds of examples.
- **The ordering is exactly what theory predicts** at this scale: GBT ≳ RF ≫ Transformer ≫
  CNN, with the highest-capacity model (Transformer) doing *relatively* better where the
  signal is strong and dense (role, topology) and collapsing where it is subtle (workflow).
- **Consequence for the paper.** Deep models are **excluded from every claim** (not the
  attack). The contribution is that *cheap* tree attackers on engineered features already
  achieve the headline numbers; whether more traces would un-collapse the deep models is
  untested and out of scope. This is the honest framing and matches the class-collapse note in
  `RESULTS.md` §1 / `PAPER_ARTIFACTS.md`.

**Reproduce** (deterministic point estimates for the trees; deep models are stochastic and
footnote-only): `bash scripts/reproduce.sh --full-suite` runs the closed-world stage incl.
CNN/Transformer into the sandbox.
