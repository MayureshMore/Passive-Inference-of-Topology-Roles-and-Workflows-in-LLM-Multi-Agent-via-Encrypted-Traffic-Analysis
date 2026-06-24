"""
Top-level feature extractor.

Reads a .pcap file (via pyshark or scapy), runs burst segmentation, then
computes per-flow and per-system features.  Returns a TraceFeatures object
that is saved to data/processed/ as a .npz file alongside its label JSON.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .burst import BurstSegmenter
from .per_flow import PerFlowFeatures, compute_per_flow
from .per_system import PerSystemFeatures, compute_per_system

logger = logging.getLogger(__name__)

# Agent ports used to filter only inter-agent flows (avoids background traffic)
DEFAULT_AGENT_PORTS = {8000, 8001, 8002, 8003}


@dataclass
class TraceFeatures:
    run_id: str
    per_flow: list[PerFlowFeatures] = field(default_factory=list)
    per_system: PerSystemFeatures = field(default_factory=PerSystemFeatures)
    # Burst sequence as a (T, burst_feature_dim) array for the Transformer
    burst_sequence: np.ndarray = field(default_factory=lambda: np.zeros((0, 10), dtype=np.float32))
    # Gap sequence between bursts (T-1,)
    gap_sequence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Per-flow response-segmentation features: one 10-dim vector per flow.
    # Each vector counts response-direction packets > 60 B (non-ACK segments)
    # and their size/timing distributions.  See SEG_FEATURE_NAMES() for layout.
    seg_features: list[np.ndarray] = field(default_factory=list)

    def flat_vector(self) -> np.ndarray:
        """
        Flat feature vector for the Random Forest baseline.

        Layout (195-dim):
          [0:35]   pf_mean  — mean over all flows
          [35:70]  pf_top1  — heaviest flow by total bytes
          [70:105] pf_top2  — 2nd heaviest flow by total bytes
          [105:195] per_system (90-dim)
        """
        zero35 = np.zeros(35, dtype=np.float32)
        if self.per_flow:
            vecs = [pf.to_vector() for pf in self.per_flow]
            pf_mean = np.mean(vecs, axis=0).astype(np.float32)
            sorted_pf = sorted(
                self.per_flow,
                key=lambda pf: pf.total_bytes_out + pf.total_bytes_in,
                reverse=True,
            )
            pf_top1 = sorted_pf[0].to_vector() if len(sorted_pf) >= 1 else zero35
            pf_top2 = sorted_pf[1].to_vector() if len(sorted_pf) >= 2 else zero35
        else:
            pf_mean = zero35
            pf_top1 = zero35
            pf_top2 = zero35
        ps_vec = self.per_system.to_vector()  # 87-dim
        return np.concatenate([pf_mean, pf_top1, pf_top2, ps_vec]).astype(np.float32)  # 195-dim

    def seg_vector(self) -> np.ndarray:
        """
        30-dim response-segmentation vector: pf_seg_mean | pf_seg_top1 | pf_seg_top2.

        Each block is 10-dim (see SEG_FEATURE_NAMES for field layout).
        Top-1/top-2 flows are sorted by n_segs (most response segments first).

        With the official a2a-sdk, agents stream their answers as Server-Sent
        Events (JSON-RPC message/stream), so on the wire each response-direction
        packet corresponds to an SSE event — a streamed token/coalesced chunk or
        the final result artifact.  These features therefore capture genuine
        SSE-chunk structure: how many response events each flow emits and their
        size/timing spread, directly reflecting the agent's streaming behaviour
        and per-invocation LLM roundtrips.
        """
        zero10 = np.zeros(10, dtype=np.float32)
        if not self.seg_features:
            return np.concatenate([zero10, zero10, zero10])
        pf_seg_mean = np.mean(self.seg_features, axis=0).astype(np.float32)
        sorted_segs = sorted(self.seg_features, key=lambda x: x[0], reverse=True)
        pf_seg_top1 = sorted_segs[0].astype(np.float32) if len(sorted_segs) >= 1 else zero10
        pf_seg_top2 = sorted_segs[1].astype(np.float32) if len(sorted_segs) >= 2 else zero10
        return np.concatenate([pf_seg_mean, pf_seg_top1, pf_seg_top2]).astype(np.float32)

    def save(self, out_path: Path) -> None:
        np.savez_compressed(
            out_path,
            flat=self.flat_vector(),
            burst_sequence=self.burst_sequence,
            gap_sequence=self.gap_sequence,
            per_system=self.per_system.to_vector(),
            seg=self.seg_vector(),
        )

    @staticmethod
    def load(npz_path: Path, run_id: str = "") -> "TraceFeatures":
        d = np.load(npz_path, allow_pickle=False)
        tf = TraceFeatures(run_id=run_id or npz_path.stem)
        tf.burst_sequence = d["burst_sequence"]
        tf.gap_sequence = d["gap_sequence"]
        return tf


class FeatureExtractor:
    """
    Extracts TraceFeatures from a single .pcap file.

    Requires pyshark (which wraps tshark) or can fall back to scapy.
    The extractor records only packet size, timestamp, and direction —
    it never reads payload bytes.
    """

    def __init__(
        self,
        agent_ports: set[int] | None = None,
        idle_threshold_s: float = 0.5,
        use_scapy: bool = False,
    ) -> None:
        self.agent_ports = agent_ports or DEFAULT_AGENT_PORTS
        self.segmenter = BurstSegmenter(idle_threshold_s=idle_threshold_s)
        self.use_scapy = use_scapy

    def extract(self, pcap_path: Path, run_id: str = "") -> TraceFeatures | None:
        run_id = run_id or pcap_path.stem
        try:
            packets = self._read_pcap(pcap_path)
        except Exception as exc:
            logger.error("failed to read %s: %s", pcap_path, exc)
            return None

        if not packets:
            # FAIL LOUD — zero packets on the agent ports almost always means a
            # systematic capture/parse bug (wrong ports, or an IPv4-vs-IPv6
            # mismatch — services on ::1 loopback), NOT a legitimately empty trace.
            # Returning a silent None / zero-vector here is exactly how the IPv6
            # binding bug went unnoticed ("Extracted 0/N" looked like success).
            raise ValueError(
                f"no valid A2A flows in {pcap_path}: 0 packets matched agent_ports "
                f"{sorted(self.agent_ports)} — check the capture ports and the IP "
                f"version (IPv4 vs IPv6 ::1 loopback)."
            )

        return self._compute_features(run_id, packets)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_pcap(
        self, pcap_path: Path
    ) -> list[tuple[float, int, str, int]]:
        """
        Returns list of (timestamp, size_bytes, flow_key, direction).
        direction: +1 if src_port in agent_ports (outbound), -1 otherwise.
        """
        if self.use_scapy:
            return self._read_with_scapy(pcap_path)
        return self._read_with_pyshark(pcap_path)

    def _read_with_pyshark(
        self, pcap_path: Path
    ) -> list[tuple[float, int, str, int]]:
        import pyshark

        packets: list[tuple[float, int, str, int]] = []
        cap = pyshark.FileCapture(
            str(pcap_path),
            display_filter="tcp",
            only_summaries=False,
            keep_packets=False,
        )
        try:
            for pkt in cap:
                try:
                    if not hasattr(pkt, "tcp"):
                        continue
                    src_port = int(pkt.tcp.srcport)
                    dst_port = int(pkt.tcp.dstport)
                    if src_port not in self.agent_ports and dst_port not in self.agent_ports:
                        continue
                    ts = float(pkt.sniff_timestamp)
                    size = int(pkt.length)
                    # Support IPv4 and IPv6 (services binding to ::1 loopback).
                    if hasattr(pkt, "ip"):
                        src_ip = str(pkt.ip.src)
                        dst_ip = str(pkt.ip.dst)
                    elif hasattr(pkt, "ipv6"):
                        src_ip = str(pkt.ipv6.src)
                        dst_ip = str(pkt.ipv6.dst)
                    else:
                        continue
                    # Canonical key: agent port always on the right (as "dst").
                    # This merges both directions of a TCP connection into one flow.
                    # direction=1 (from agent) = response; direction=-1 (to agent) = request.
                    if dst_port in self.agent_ports:
                        flow_key = f"{src_ip}:{src_port}→{dst_ip}:{dst_port}"
                        direction = -1  # request: client → agent
                    else:
                        flow_key = f"{dst_ip}:{dst_port}→{src_ip}:{src_port}"
                        direction = 1   # response: agent → client (key flipped)
                    packets.append((ts, size, flow_key, direction))
                except AttributeError:
                    continue
        finally:
            cap.close()
        return packets

    def _read_with_scapy(
        self, pcap_path: Path
    ) -> list[tuple[float, int, str, int]]:
        from scapy.all import rdpcap, IP, IPv6, TCP

        packets: list[tuple[float, int, str, int]] = []
        scapy_pkts = rdpcap(str(pcap_path))
        for pkt in scapy_pkts:
            if TCP not in pkt:
                continue
            # Support both IPv4 and IPv6 (e.g. services that bind to ::1 loopback).
            # The IPv4 path is unchanged; IPv6 is additive.
            if IP in pkt:
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
            elif IPv6 in pkt:
                src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
            else:
                continue
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            if src_port not in self.agent_ports and dst_port not in self.agent_ports:
                continue
            ts = float(pkt.time)
            size = pkt.wirelen  # actual on-wire length; len(pkt) would give snaplen-truncated size
            # Canonical key: agent port always on the right (as "dst").
            # Merges both TCP directions into one bidirectional flow.
            if dst_port in self.agent_ports:
                flow_key = f"{src_ip}:{src_port}→{dst_ip}:{dst_port}"
                direction = -1  # request: client → agent
            else:
                flow_key = f"{dst_ip}:{dst_port}→{src_ip}:{src_port}"
                direction = 1   # response: agent → client (key flipped)
            packets.append((ts, size, flow_key, direction))
        return packets

    @staticmethod
    def _compute_seg_features(
        packets: list[tuple[float, int, int]],  # (ts, size, direction)
    ) -> np.ndarray:
        """
        10-dim response-segmentation features for one flow.

        Response segments: direction==1 (agent→client) AND wirelen > 60 B.
        The 60-byte threshold excludes pure TCP ACKs (typically 40-54 bytes on
        loopback) and retains actual HTTP response data packets.

        On loopback with MTU=65536, each LLM call typically produces one large
        response segment, so n_segs ≈ n_LLM_roundtrips — the primary signal
        distinguishing deployment A (different call counts per role) from B.

        Feature layout (10-dim):
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
        """
        segs = [(ts, sz) for ts, sz, d in packets if d == 1 and sz > 60]
        if not segs:
            return np.zeros(10, dtype=np.float32)

        sizes = np.array([sz for _, sz in segs], dtype=np.float32)
        timestamps = sorted(ts for ts, _ in segs)

        p25 = float(np.percentile(sizes, 25))
        p75 = float(np.percentile(sizes, 75))

        if len(timestamps) > 1:
            gaps_ms = np.diff(timestamps).astype(np.float64) * 1000.0
            mean_gap = float(np.mean(gaps_ms))
            std_gap  = float(np.std(gaps_ms))
            min_gap  = float(np.min(gaps_ms))
            max_gap  = float(np.max(gaps_ms))
        else:
            mean_gap = std_gap = min_gap = max_gap = 0.0

        return np.array([
            float(len(segs)),
            float(np.mean(sizes)),
            float(np.std(sizes)) if len(sizes) > 1 else 0.0,
            p25,
            p75,
            p75 - p25,
            mean_gap,
            std_gap,
            min_gap,
            max_gap,
        ], dtype=np.float32)

    def _compute_features(
        self,
        run_id: str,
        packets: list[tuple[float, int, str, int]],
    ) -> TraceFeatures:
        from collections import defaultdict

        # Group by flow key
        by_flow: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for ts, size, fk, direction in packets:
            by_flow[fk].append((ts, size, direction))

        # Burst segmentation per flow
        bursts_by_flow: dict[str, list] = {}
        for fk, pkts in by_flow.items():
            flow_packets = [(ts, size, fk, d) for ts, size, d in pkts]
            bursts_by_flow[fk] = self.segmenter.segment(flow_packets)

        # Per-flow features + response-segmentation features
        pf_features: list[PerFlowFeatures] = []
        seg_features: list[np.ndarray] = []
        flow_time_ranges: dict[str, tuple[float, float]] = {}
        for fk, pkts in by_flow.items():
            pf = compute_per_flow(fk, bursts_by_flow.get(fk, []), pkts)
            pf_features.append(pf)
            seg_features.append(self._compute_seg_features(pkts))
            if pkts:
                ts_list = [ts for ts, _, _ in pkts]
                flow_time_ranges[fk] = (min(ts_list), max(ts_list))

        # Per-system features
        ps_feat = compute_per_system(pf_features, bursts_by_flow, flow_time_ranges)

        # Burst sequence for Transformer (all bursts, chronologically)
        all_bursts = sorted(
            [b for bl in bursts_by_flow.values() for b in bl],
            key=lambda b: b.start_ts,
        )
        if all_bursts:
            burst_seq = np.stack(
                [b.to_feature_vector() for b in all_bursts], axis=0
            )
            gap_seq = self.segmenter.gap_sequence(all_bursts)
        else:
            burst_seq = np.zeros((0, 10), dtype=np.float32)
            gap_seq = np.zeros(0, dtype=np.float32)

        return TraceFeatures(
            run_id=run_id,
            per_flow=pf_features,
            per_system=ps_feat,
            burst_sequence=burst_seq,
            gap_sequence=gap_seq,
            seg_features=seg_features,
        )
