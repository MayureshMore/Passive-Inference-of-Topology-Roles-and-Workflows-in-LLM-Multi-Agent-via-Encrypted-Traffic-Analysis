"""
Human-readable feature names for the 195-dim flat feature vector.

Flat vector layout (195 dimensions):
  [0:35]    pf_mean   — per-flow stats averaged across ALL flows in the trace
  [35:70]   pf_top1   — per-flow stats for the heaviest flow (by total bytes)
  [70:105]  pf_top2   — per-flow stats for the 2nd heaviest flow
  [105:195] per_system — 90-dim system-wide aggregate:
              [105:125]  20 scalar stats (n_flows, timing spreads, request-body ratios…)
              [125:160]  per-system pf_mean (mean of each pf dim across flows)
              [160:195]  per-system pf_std  (std  of each pf dim across flows)

The 35-dim per-flow (pf) sub-vector layout is defined in PerFlowFeatures.FEATURE_NAMES().
The 20-dim per-system scalar layout is in PerSystemFeatures.FEATURE_NAMES()[:20].

Usage:
    from features.names import FLAT_FEATURE_NAMES, ROLE_FEATURE_NAMES
    names = FLAT_FEATURE_NAMES()   # 195 names for per-trace NPZs
    names = ROLE_FEATURE_NAMES()   # 35 names for per-agent role NPZs
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
