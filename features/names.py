"""
Human-readable feature names for the 195-dim flat and 30-dim seg feature vectors.

Flat vector layout (195 dimensions):
  [0:35]    pf_mean   — per-flow stats averaged across ALL flows in the trace
  [35:70]   pf_top1   — per-flow stats for the heaviest flow (by total bytes)
  [70:105]  pf_top2   — per-flow stats for the 2nd heaviest flow
  [105:195] per_system — 90-dim system-wide aggregate:
              [105:125]  20 scalar stats (n_flows, timing spreads, request-body ratios…)
              [125:160]  per-system pf_mean (mean of each pf dim across flows)
              [160:195]  per-system pf_std  (std  of each pf dim across flows)

Seg vector layout (30 dimensions, stored as separate NPZ key "seg"):
  [0:10]   pf_seg_mean  — mean response-segmentation features across all flows
  [10:20]  pf_seg_top1  — seg features of flow with most response segments
  [20:30]  pf_seg_top2  — seg features of flow with 2nd most response segments

  Each 10-dim block:
    [0] n_segs            count of response-direction pkts with wirelen > 60B
    [1] seg_mean_sz       mean segment size (bytes)
    [2] seg_std_sz        std of segment sizes
    [3] seg_p25_sz        25th percentile size
    [4] seg_p75_sz        75th percentile size
    [5] seg_iqr_sz        p75 - p25
    [6] seg_mean_gap_ms   mean inter-segment gap (ms)
    [7] seg_std_gap_ms    std of gaps
    [8] seg_min_gap_ms    min gap
    [9] seg_max_gap_ms    max gap

The 35-dim per-flow (pf) sub-vector layout is defined in PerFlowFeatures.FEATURE_NAMES().
The 20-dim per-system scalar layout is in PerSystemFeatures.FEATURE_NAMES()[:20].

Usage:
    from features.names import FLAT_FEATURE_NAMES, ROLE_FEATURE_NAMES, SEG_FEATURE_NAMES
    names = FLAT_FEATURE_NAMES()   # 195 names for per-trace NPZs
    names = ROLE_FEATURE_NAMES()   # 35 names for per-agent role NPZs
    names = SEG_FEATURE_NAMES()    # 30 names for the "seg" NPZ key
"""

from __future__ import annotations


def _pf_names() -> list[str]:
    from .per_flow import PerFlowFeatures
    return PerFlowFeatures.FEATURE_NAMES()  # 35 names


def FLAT_FEATURE_NAMES() -> list[str]:
    """Return the 195 feature names for per-trace flat vectors."""
    pf = _pf_names()
    from .per_system import PerSystemFeatures
    sys_names = PerSystemFeatures.FEATURE_NAMES()  # 90 names

    names: list[str] = []
    # pf_mean block [0:35]
    names += [f"pf_mean.{n}" for n in pf]
    # pf_top1 block [35:70]
    names += [f"pf_top1.{n}" for n in pf]
    # pf_top2 block [70:105]
    names += [f"pf_top2.{n}" for n in pf]
    # per_system block [105:195]
    names += [f"sys.{n}" for n in sys_names]

    assert len(names) == 195, f"Expected 195 names, got {len(names)}"
    return names


def ROLE_FEATURE_NAMES() -> list[str]:
    """Return the 35 feature names for per-agent role flat vectors."""
    return _pf_names()


_SEG_DIMS = [
    "n_segs",
    "seg_mean_sz",
    "seg_std_sz",
    "seg_p25_sz",
    "seg_p75_sz",
    "seg_iqr_sz",
    "seg_mean_gap_ms",
    "seg_std_gap_ms",
    "seg_min_gap_ms",
    "seg_max_gap_ms",
]


def SEG_FEATURE_NAMES() -> list[str]:
    """Return the 30 feature names for the per-trace 'seg' NPZ key."""
    names: list[str] = []
    names += [f"seg_mean.{d}" for d in _SEG_DIMS]   # [0:10]
    names += [f"seg_top1.{d}" for d in _SEG_DIMS]   # [10:20]
    names += [f"seg_top2.{d}" for d in _SEG_DIMS]   # [20:30]
    assert len(names) == 30, f"Expected 30 seg names, got {len(names)}"
    return names
