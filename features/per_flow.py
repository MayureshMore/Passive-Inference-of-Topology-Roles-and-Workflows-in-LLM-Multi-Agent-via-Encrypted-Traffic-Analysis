"""
Per-flow features — aggregate statistics for a single TCP flow (5-tuple).

These are the "classic" website-fingerprinting features adapted for A2A:
packet-size distribution, directional counts, timing statistics, and
cumulative trace representations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence

from .burst import Burst


@dataclass
class PerFlowFeatures:
    flow_key: str

    # Packet-level statistics
    n_packets_out: int = 0
    n_packets_in: int = 0
    total_bytes_out: int = 0
    total_bytes_in: int = 0

    # Size distribution (outbound)
    mean_size_out: float = 0.0
    std_size_out: float = 0.0
    p25_size_out: float = 0.0
    p75_size_out: float = 0.0

    # Size distribution (inbound)
    mean_size_in: float = 0.0
    std_size_in: float = 0.0
    p25_size_in: float = 0.0
    p75_size_in: float = 0.0

    # Timing
    total_duration_s: float = 0.0
    mean_iat_s: float = 0.0
    std_iat_s: float = 0.0

    # Burst-level
    n_bursts: int = 0
    mean_burst_bytes: float = 0.0
    std_burst_bytes: float = 0.0
    mean_burst_duration: float = 0.0
    mean_inter_burst_gap: float = 0.0

    # Cumulative bytes at fixed time quantiles (10 points)
    cumulative_bytes: list[float] = None  # type: ignore[assignment]

    # Asymmetry / SSE-proxy features (indices 30–34)
    bytes_out_ratio: float = 0.5        # bytes_out / total_bytes — role discriminator
    pkt_size_asymmetry: float = 0.5     # mean_sz_out / (mean_sz_out + mean_sz_in)
    n_small_inbound: int = 0            # inbound pkts ≤200 B — SSE chunk proxy
    n_response_bursts: int = 0          # bursts in response direction — streaming depth
    iqr_size_in: float = 0.0            # p75 - p25 of inbound sizes — SSE regularity

    def __post_init__(self):
        if self.cumulative_bytes is None:
            self.cumulative_bytes = [0.0] * 10

    def to_vector(self) -> np.ndarray:
        base = np.array(
            [
                self.n_packets_out,
                self.n_packets_in,
                self.total_bytes_out,
                self.total_bytes_in,
                self.mean_size_out,
                self.std_size_out,
                self.p25_size_out,
                self.p75_size_out,
                self.mean_size_in,
                self.std_size_in,
                self.p25_size_in,
                self.p75_size_in,
                self.total_duration_s,
                self.mean_iat_s,
                self.std_iat_s,
                self.n_bursts,
                self.mean_burst_bytes,
                self.std_burst_bytes,
                self.mean_burst_duration,
                self.mean_inter_burst_gap,
            ],
            dtype=np.float32,
        )
        cumul = np.array(self.cumulative_bytes, dtype=np.float32)
        extra = np.array(
            [
                self.bytes_out_ratio,
                self.pkt_size_asymmetry,
                self.n_small_inbound,
                self.n_response_bursts,
                self.iqr_size_in,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, cumul, extra])  # 35-dim

    @staticmethod
    def FEATURE_NAMES() -> list[str]:
        names = [
            "n_pkts_out", "n_pkts_in", "bytes_out", "bytes_in",
            "mean_sz_out", "std_sz_out", "p25_sz_out", "p75_sz_out",
            "mean_sz_in", "std_sz_in", "p25_sz_in", "p75_sz_in",
            "duration_s", "mean_iat", "std_iat",
            "n_bursts", "mean_burst_bytes", "std_burst_bytes",
            "mean_burst_dur", "mean_ibg",
        ]
        names += [f"cumul_bytes_q{i}" for i in range(10)]
        names += [
            "bytes_out_ratio", "pkt_size_asymmetry",
            "n_small_inbound", "n_response_bursts", "iqr_size_in",
        ]
        return names  # 35 names


def compute_per_flow(
    flow_key: str,
    bursts: list[Burst],
    all_packets: list[tuple[float, int, int]],  # (ts, size, direction)
) -> PerFlowFeatures:
    """Compute PerFlowFeatures for one flow from its bursts and raw packets."""
    feat = PerFlowFeatures(flow_key=flow_key)

    if not all_packets:
        return feat

    # Directional packet/byte counts
    sizes_out = [s for _, s, d in all_packets if d == 1]
    sizes_in = [s for _, s, d in all_packets if d == -1]
    feat.n_packets_out = len(sizes_out)
    feat.n_packets_in = len(sizes_in)
    feat.total_bytes_out = sum(sizes_out)
    feat.total_bytes_in = sum(sizes_in)

    if sizes_out:
        a = np.array(sizes_out, dtype=np.float32)
        feat.mean_size_out = float(np.mean(a))
        feat.std_size_out = float(np.std(a))
        feat.p25_size_out = float(np.percentile(a, 25))
        feat.p75_size_out = float(np.percentile(a, 75))

    if sizes_in:
        a = np.array(sizes_in, dtype=np.float32)
        feat.mean_size_in = float(np.mean(a))
        feat.std_size_in = float(np.std(a))
        feat.p25_size_in = float(np.percentile(a, 25))
        feat.p75_size_in = float(np.percentile(a, 75))

    # Timing
    timestamps = sorted(ts for ts, _, _ in all_packets)
    if len(timestamps) > 1:
        feat.total_duration_s = timestamps[-1] - timestamps[0]
        iats = np.diff(timestamps).astype(np.float32)
        feat.mean_iat_s = float(np.mean(iats))
        feat.std_iat_s = float(np.std(iats))

    # Burst stats
    if bursts:
        feat.n_bursts = len(bursts)
        burst_bytes = np.array([b.total_bytes for b in bursts], dtype=np.float32)
        burst_durs = np.array([b.duration_s for b in bursts], dtype=np.float32)
        feat.mean_burst_bytes = float(np.mean(burst_bytes))
        feat.std_burst_bytes = float(np.std(burst_bytes))
        feat.mean_burst_duration = float(np.mean(burst_durs))
        if len(bursts) > 1:
            gaps = [
                bursts[i + 1].start_ts - bursts[i].end_ts
                for i in range(len(bursts) - 1)
            ]
            feat.mean_inter_burst_gap = float(np.mean(gaps))

    # Cumulative bytes at 10 equally-spaced time quantiles
    if feat.total_duration_s > 0:
        t0 = timestamps[0]
        all_sorted = sorted(all_packets, key=lambda x: x[0])
        quantile_times = np.linspace(t0, t0 + feat.total_duration_s, 10)
        cumul = 0.0
        pkt_idx = 0
        for q, qt in enumerate(quantile_times):
            while pkt_idx < len(all_sorted) and all_sorted[pkt_idx][0] <= qt:
                cumul += all_sorted[pkt_idx][1]
                pkt_idx += 1
            feat.cumulative_bytes[q] = cumul

    # Asymmetry / SSE-proxy features
    total_bytes = feat.total_bytes_out + feat.total_bytes_in
    feat.bytes_out_ratio = feat.total_bytes_out / (total_bytes + 1e-8)

    sz_out = feat.mean_size_out
    sz_in = feat.mean_size_in
    feat.pkt_size_asymmetry = sz_out / (sz_out + sz_in + 1e-8)

    # Small inbound packets (≤200 B) as a proxy for SSE chunks:
    # each SSE data line arrives as its own small TCP segment
    feat.n_small_inbound = sum(1 for s in sizes_in if s <= 200)

    # Response bursts: bursts where the agent is the sender (direction == 1)
    feat.n_response_bursts = sum(1 for b in bursts if b.direction == 1)

    # IQR of inbound packet sizes — low IQR = uniform SSE chunk sizes
    feat.iqr_size_in = feat.p75_size_in - feat.p25_size_in

    return feat
