"""
Per-system features — aggregate statistics across all flows in one trace.

These capture the multi-party collaboration structure that single-flow
features miss: the number of distinct host pairs communicating, the
coordination timing between parallel delegations, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .burst import Burst
from .per_flow import PerFlowFeatures


@dataclass
class PerSystemFeatures:
    # Flow-count statistics
    n_flows: int = 0
    n_unique_src_hosts: int = 0
    n_unique_dst_hosts: int = 0
    n_unique_host_pairs: int = 0

    # Total traffic volume
    total_bytes: int = 0
    total_packets: int = 0
    total_duration_s: float = 0.0

    # Cross-flow coordination (timing of parallel delegations)
    # Std of per-flow start times relative to trace start
    flow_start_spread_s: float = 0.0
    # Std of per-flow end times
    flow_end_spread_s: float = 0.0
    # Max simultaneous open flows (proxy for fan-out degree)
    max_concurrent_flows: int = 0

    # Burst-level cross-flow features
    total_bursts: int = 0
    mean_burst_rate_per_s: float = 0.0
    # Directionality ratio across all flows
    bytes_out_ratio: float = 0.0

    # Per-flow feature statistics (mean and std of each per-flow dimension)
    pf_mean: np.ndarray = field(default_factory=lambda: np.zeros(30, dtype=np.float32))
    pf_std: np.ndarray = field(default_factory=lambda: np.zeros(30, dtype=np.float32))

    def to_vector(self) -> np.ndarray:
        scalar = np.array(
            [
                self.n_flows,
                self.n_unique_src_hosts,
                self.n_unique_dst_hosts,
                self.n_unique_host_pairs,
                self.total_bytes,
                self.total_packets,
                self.total_duration_s,
                self.flow_start_spread_s,
                self.flow_end_spread_s,
                self.max_concurrent_flows,
                self.total_bursts,
                self.mean_burst_rate_per_s,
                self.bytes_out_ratio,
            ],
            dtype=np.float32,
        )
        return np.concatenate([scalar, self.pf_mean, self.pf_std])  # 73-dim

    @staticmethod
    def FEATURE_NAMES() -> list[str]:
        from .per_flow import PerFlowFeatures
        scalar_names = [
            "n_flows", "n_src_hosts", "n_dst_hosts", "n_host_pairs",
            "total_bytes", "total_packets", "total_duration_s",
            "flow_start_spread", "flow_end_spread", "max_concurrent_flows",
            "total_bursts", "mean_burst_rate", "bytes_out_ratio",
        ]
        pf_names = PerFlowFeatures.FEATURE_NAMES()
        pf_mean_names = [f"pf_mean_{n}" for n in pf_names]
        pf_std_names = [f"pf_std_{n}" for n in pf_names]
        return scalar_names + pf_mean_names + pf_std_names


def compute_per_system(
    per_flow_features: list[PerFlowFeatures],
    flow_bursts: dict[str, list[Burst]],
    flow_time_ranges: dict[str, tuple[float, float]],  # flow_key → (start, end)
) -> PerSystemFeatures:
    """
    Aggregate per-flow features and burst data into a single per-system
    feature vector for a complete trace.
    """
    feat = PerSystemFeatures()

    if not per_flow_features:
        return feat

    feat.n_flows = len(per_flow_features)

    # Host-pair counting from flow keys ("srcip:srcport→dstip:dstport")
    src_hosts: set[str] = set()
    dst_hosts: set[str] = set()
    host_pairs: set[str] = set()
    for fk in flow_time_ranges:
        parts = fk.split("→")
        if len(parts) == 2:
            src_host = parts[0].rsplit(":", 1)[0]
            dst_host = parts[1].rsplit(":", 1)[0]
            src_hosts.add(src_host)
            dst_hosts.add(dst_host)
            host_pairs.add(f"{src_host}→{dst_host}")  # host-level pair, not 5-tuple
    feat.n_unique_src_hosts = len(src_hosts)
    feat.n_unique_dst_hosts = len(dst_hosts)
    feat.n_unique_host_pairs = len(host_pairs)

    # Volume
    feat.total_bytes = sum(pf.total_bytes_out + pf.total_bytes_in for pf in per_flow_features)
    feat.total_packets = sum(pf.n_packets_out + pf.n_packets_in for pf in per_flow_features)

    # Duration (wall-clock span across all flows)
    if flow_time_ranges:
        all_starts = [s for s, _ in flow_time_ranges.values()]
        all_ends = [e for _, e in flow_time_ranges.values()]
        trace_start = min(all_starts)
        trace_end = max(all_ends)
        feat.total_duration_s = trace_end - trace_start
        feat.flow_start_spread_s = float(np.std([s - trace_start for s in all_starts]))
        feat.flow_end_spread_s = float(np.std([e - trace_start for e in all_ends]))

        # Max concurrent flows: sweep line over start/end events
        events = [(t, +1) for t in all_starts] + [(t, -1) for t in all_ends]
        events.sort()
        concurrent = max_concurrent = 0
        for _, delta in events:
            concurrent += delta
            max_concurrent = max(max_concurrent, concurrent)
        feat.max_concurrent_flows = max_concurrent

    # Burst stats
    all_bursts = [b for bl in flow_bursts.values() for b in bl]
    feat.total_bursts = len(all_bursts)
    if all_bursts and feat.total_duration_s > 0:
        feat.mean_burst_rate_per_s = feat.total_bursts / feat.total_duration_s

    if feat.total_bytes > 0:
        total_out = sum(pf.total_bytes_out for pf in per_flow_features)
        feat.bytes_out_ratio = total_out / feat.total_bytes

    # Per-flow feature matrix → mean and std across flows
    pf_matrix = np.stack([pf.to_vector() for pf in per_flow_features], axis=0)
    feat.pf_mean = pf_matrix.mean(axis=0)
    feat.pf_std = pf_matrix.std(axis=0) if len(per_flow_features) > 1 else np.zeros_like(feat.pf_mean)

    return feat
