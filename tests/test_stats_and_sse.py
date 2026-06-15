"""
Tests for the shared bootstrap-CI utility and SSE-chunk feature extraction.

These back two claims that must hold for the paper:
  1. Every headline number carries a well-formed 95 % bootstrap CI.
  2. The features genuinely summarise on-wire SSE chunks (each response-direction
     data packet is one SSE event; ACK-sized packets are excluded).
"""

from __future__ import annotations

import numpy as np

from evaluation.stats import bootstrap_ci
from features.extractor import FeatureExtractor, TraceFeatures


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_well_formed_and_brackets_point():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 60)
    y_pred = y_true.copy()
    y_pred[:12] = rng.integers(0, 3, 12)  # inject ~20% errors
    r = bootstrap_ci(y_true, y_pred, n=500, seed=1)
    for m in ("accuracy", "macro_f1"):
        lo, hi = r[f"{m}_ci_lo"], r[f"{m}_ci_hi"]
        assert 0.0 <= lo <= hi <= 1.0
        assert lo - 1e-9 <= r[m] <= hi + 1e-9      # point estimate inside CI


def test_bootstrap_ci_deterministic_with_seed():
    yt, yp = [0, 1, 0, 1, 2, 2, 1, 0], [0, 1, 1, 1, 2, 0, 1, 0]
    assert bootstrap_ci(yt, yp, n=300, seed=7) == bootstrap_ci(yt, yp, n=300, seed=7)


def test_bootstrap_ci_empty_is_safe():
    r = bootstrap_ci([], [])
    assert r["accuracy"] == 0.0 and r["accuracy_ci_lo"] == 0.0


# ── SSE-chunk feature extraction ─────────────────────────────────────────────

def test_seg_features_count_sse_chunks_excluding_acks():
    # (ts, size, direction): three response data chunks (>60B) + one ACK (<=60B)
    # + one request.  n_segs must count only the three response data chunks.
    pkts = [(0.0, 150, 1), (0.1, 200, 1), (0.2, 210, 1), (0.25, 40, 1), (0.3, 150, -1)]
    seg = FeatureExtractor._compute_seg_features(pkts)
    assert seg.shape == (10,)
    assert seg[0] == 3.0                      # n_segs = SSE data chunks, ACK excluded
    assert seg[1] > 60.0                      # mean chunk size in data range


def test_feature_vector_dimensions_stable():
    tf = TraceFeatures(run_id="empty")
    assert tf.flat_vector().shape == (195,)   # attacker headline vector
    assert tf.seg_vector().shape == (30,)     # SSE-chunk stats (mean|top1|top2)
