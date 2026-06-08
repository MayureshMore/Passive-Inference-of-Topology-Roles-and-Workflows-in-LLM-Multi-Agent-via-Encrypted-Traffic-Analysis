"""
Session-state burst segmentation.

Multi-agent workflows are long-running and asynchronous with significant idle
gaps between delegation rounds.  Classic website-fingerprinting treats captures
as one continuous sequence; that does not work here.  Instead we segment each
flow into *bursts* separated by idle gaps, then extract features per burst and
model the gap durations between them.

A burst is a maximal run of packets in one direction with no inter-packet gap
exceeding `idle_threshold_s` seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Burst:
    """A single burst in one direction on one flow."""
    flow_key: str          # "src:port→dst:port"
    direction: int         # +1 = outbound, -1 = inbound (from observer perspective)
    start_ts: float
    end_ts: float
    packet_sizes: list[int] = field(default_factory=list)
    inter_packet_gaps: list[float] = field(default_factory=list)

    # ── Derived statistics ────────────────────────────────────────────────────

    @property
    def n_packets(self) -> int:
        return len(self.packet_sizes)

    @property
    def total_bytes(self) -> int:
        return sum(self.packet_sizes)

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def mean_pkt_size(self) -> float:
        return float(np.mean(self.packet_sizes)) if self.packet_sizes else 0.0

    @property
    def std_pkt_size(self) -> float:
        return float(np.std(self.packet_sizes)) if len(self.packet_sizes) > 1 else 0.0

    @property
    def mean_ipg(self) -> float:
        return float(np.mean(self.inter_packet_gaps)) if self.inter_packet_gaps else 0.0

    @property
    def throughput_bps(self) -> float:
        return self.total_bytes / self.duration_s if self.duration_s > 0 else 0.0

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.direction,
                self.n_packets,
                self.total_bytes,
                self.duration_s,
                self.mean_pkt_size,
                self.std_pkt_size,
                self.mean_ipg,
                self.throughput_bps,
                float(max(self.packet_sizes)) if self.packet_sizes else 0.0,
                float(min(self.packet_sizes)) if self.packet_sizes else 0.0,
            ],
            dtype=np.float32,
        )


class BurstSegmenter:
    """
    Segments a list of (timestamp, size, flow_key, direction) packet records
    into bursts using a configurable idle threshold.
    """

    def __init__(self, idle_threshold_s: float = 0.5) -> None:
        self.idle_threshold_s = idle_threshold_s

    def segment(
        self,
        packets: list[tuple[float, int, str, int]],
    ) -> list[Burst]:
        """
        Parameters
        ----------
        packets : list of (timestamp, size_bytes, flow_key, direction)
            Sorted by timestamp.

        Returns
        -------
        list[Burst] ordered by start_ts.
        """
        if not packets:
            return []

        bursts: list[Burst] = []
        # Group by flow_key first, then segment within each flow
        from collections import defaultdict
        by_flow: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for ts, size, fk, direction in packets:
            by_flow[fk].append((ts, size, direction))

        for fk, pkts in by_flow.items():
            pkts.sort(key=lambda x: x[0])
            current: Burst | None = None
            prev_ts: float | None = None

            for ts, size, direction in pkts:
                if current is None:
                    current = Burst(
                        flow_key=fk,
                        direction=direction,
                        start_ts=ts,
                        end_ts=ts,
                    )
                else:
                    gap = ts - prev_ts  # type: ignore[operator]
                    if gap > self.idle_threshold_s or direction != current.direction:
                        bursts.append(current)
                        current = Burst(
                            flow_key=fk,
                            direction=direction,
                            start_ts=ts,
                            end_ts=ts,
                        )
                    else:
                        current.inter_packet_gaps.append(gap)

                current.packet_sizes.append(size)
                current.end_ts = ts
                prev_ts = ts

            if current and current.packet_sizes:
                bursts.append(current)

        bursts.sort(key=lambda b: b.start_ts)
        return bursts

    def gap_sequence(self, bursts: list[Burst]) -> np.ndarray:
        """Inter-burst idle gap durations (seconds) between consecutive bursts."""
        if len(bursts) < 2:
            return np.array([], dtype=np.float32)
        return np.array(
            [bursts[i + 1].start_ts - bursts[i].end_ts for i in range(len(bursts) - 1)],
            dtype=np.float32,
        )
