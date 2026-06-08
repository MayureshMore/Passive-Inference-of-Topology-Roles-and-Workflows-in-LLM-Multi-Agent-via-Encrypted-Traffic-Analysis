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

    def flat_vector(self) -> np.ndarray:
        """
        Flat feature vector for the Random Forest baseline.

        Layout (192-dim):
          [0:35]   pf_mean  — mean over all flows
          [35:70]  pf_top1  — heaviest flow by total bytes
          [70:105] pf_top2  — 2nd heaviest flow by total bytes
          [105:192] per_system (87-dim)
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
        return np.concatenate([pf_mean, pf_top1, pf_top2, ps_vec]).astype(np.float32)  # 192-dim

    def save(self, out_path: Path) -> None:
        np.savez_compressed(
            out_path,
            flat=self.flat_vector(),
            burst_sequence=self.burst_sequence,
            gap_sequence=self.gap_sequence,
            per_system=self.per_system.to_vector(),
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
            logger.warning("no A2A packets found in %s", pcap_path)
            return None

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
                    src_ip = str(pkt.ip.src)
                    dst_ip = str(pkt.ip.dst)
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
        from scapy.all import rdpcap, IP, TCP

        packets: list[tuple[float, int, str, int]] = []
        scapy_pkts = rdpcap(str(pcap_path))
        for pkt in scapy_pkts:
            if IP not in pkt or TCP not in pkt:
                continue
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            if src_port not in self.agent_ports and dst_port not in self.agent_ports:
                continue
            ts = float(pkt.time)
            size = len(pkt)
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
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

        # Per-flow features
        pf_features: list[PerFlowFeatures] = []
        flow_time_ranges: dict[str, tuple[float, float]] = {}
        for fk, pkts in by_flow.items():
            pf = compute_per_flow(fk, bursts_by_flow.get(fk, []), pkts)
            pf_features.append(pf)
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
        )
