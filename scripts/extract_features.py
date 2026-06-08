#!/usr/bin/env python3
"""
Feature extraction script.

Reads all .pcap files from data/raw/ (with matching .json label sidecars),
extracts TraceFeatures for each, and writes:
  - data/processed/<run_id>.npz          per-trace flat+burst features
  - data/processed/<run_id>__role__<port>.npz  per-flow features for C2
  - data/processed/labels.json           run_id → {workflow, topology, prompt_group}
                                         + <run_id>__role__<port> → {role, workflow, topology}

C2 (role classification) design:
  Each flow_key encodes "srcip:srcport→dstip:dstport".  We extract the
  destination port and look it up in the sidecar's agent_endpoints dict to
  assign a role label.  This produces ~4-5 per-flow samples per trace
  (one per agent connection), each labelled with the destination agent's role.

Usage:
    python scripts/extract_features.py --raw data/raw --out data/processed
    python scripts/extract_features.py --raw data/raw --out data/processed --scapy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _port_to_role(agent_endpoints: dict[str, str]) -> dict[str, str]:
    """Build reverse map: "8001" → "executor" from sidecar agent_endpoints."""
    mapping: dict[str, str] = {}
    for role, addr in agent_endpoints.items():
        port = addr.rsplit(":", 1)[-1]
        mapping[port] = role
    return mapping


def _prompt_group(prompt: str) -> str:
    """Stable 8-char hash of input_prompt used as GroupKFold group key."""
    return hashlib.sha1(prompt.encode()).hexdigest()[:8]


def main(raw_dir: Path, out_dir: Path, use_scapy: bool = False) -> None:
    from capture.labeler import TraceLabeler
    from features.extractor import FeatureExtractor
    from features.per_flow import compute_per_flow

    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(use_scapy=use_scapy)

    pcap_files = sorted(raw_dir.glob("*.pcap"))
    logger.info("Found %d pcap files in %s", len(pcap_files), raw_dir)

    labels_map: dict[str, dict] = {}
    n_ok = n_fail = n_role = 0

    for pcap_path in pcap_files:
        label_path = pcap_path.with_suffix(".json")
        if not label_path.exists():
            logger.warning("No label sidecar for %s — skipping", pcap_path.name)
            continue

        sidecar = json.loads(label_path.read_text())
        run = TraceLabeler.read(pcap_path)
        if not run.success:
            logger.debug("Skipping failed run %s", run.run_id)
            continue

        features = extractor.extract(pcap_path, run_id=run.run_id)
        if features is None:
            logger.warning("No features extracted from %s", pcap_path.name)
            n_fail += 1
            continue

        # ── Per-trace NPZ (workflow + topology labels) ────────────────────────
        npz_path = out_dir / f"{run.run_id}.npz"
        features.save(npz_path)
        prompt = sidecar.get("input_prompt", "")
        labels_map[run.run_id] = {
            "workflow": run.workflow_class.value,
            "topology": run.topology.value,
            "prompt_group": _prompt_group(prompt),
        }
        n_ok += 1

        # ── Per-flow NPZ (role labels — C2) ───────────────────────────────────
        agent_endpoints: dict[str, str] = sidecar.get("agent_endpoints", {})
        if not agent_endpoints:
            continue

        port_role = _port_to_role(agent_endpoints)

        # Re-read packets to group by flow and extract per-flow features
        packets = extractor._read_pcap(pcap_path)
        if not packets:
            continue

        from collections import defaultdict
        by_flow: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for ts, size, fk, direction in packets:
            by_flow[fk].append((ts, size, direction))

        for fk, pkts in by_flow.items():
            # Destination port from flow_key "srcip:srcport→dstip:dstport"
            parts = fk.split("→")
            if len(parts) != 2:
                continue
            dst_port = parts[1].rsplit(":", 1)[-1]
            role = port_role.get(dst_port)
            if role is None:
                continue

            # Burst-segment this flow
            flow_packets_full = [(ts, size, fk, d) for ts, size, d in pkts]
            flow_bursts = extractor.segmenter.segment(flow_packets_full)
            pf = compute_per_flow(fk, flow_bursts, pkts)
            pf_vec = pf.to_vector()  # 30-dim

            # Save: flat=per-flow vector, burst_sequence/gap_sequence from this flow
            if flow_bursts:
                burst_seq = np.stack([b.to_feature_vector() for b in flow_bursts], axis=0)
                gap_seq = extractor.segmenter.gap_sequence(flow_bursts)
            else:
                burst_seq = np.zeros((0, 10), dtype=np.float32)
                gap_seq = np.zeros(0, dtype=np.float32)

            role_run_id = f"{run.run_id}__role__{dst_port}"
            role_npz = out_dir / f"{role_run_id}.npz"
            np.savez_compressed(
                role_npz,
                flat=pf_vec,
                burst_sequence=burst_seq,
                gap_sequence=gap_seq,
            )
            labels_map[role_run_id] = {
                "role": role,
                "workflow": run.workflow_class.value,
                "topology": run.topology.value,
                "prompt_group": _prompt_group(prompt),
            }
            n_role += 1

    (out_dir / "labels.json").write_text(json.dumps(labels_map, indent=2))
    logger.info(
        "Extraction complete: %d traces ok, %d failed, %d role samples. Labels → %s",
        n_ok, n_fail, n_role, out_dir / "labels.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from pcap files")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--scapy", action="store_true",
                        help="Use scapy instead of pyshark for pcap parsing")
    args = parser.parse_args()
    main(Path(args.raw), Path(args.out), use_scapy=args.scapy)
