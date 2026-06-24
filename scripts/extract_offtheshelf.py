#!/usr/bin/env python3
"""
Extract 195-dim features from the OFF-THE-SHELF a2a_mcp pcaps (Phase 5).

Reuses the SAME core `features.extractor.FeatureExtractor` as the main pipeline
(identical 195-dim computation), but deliberately bypasses `TraceLabeler` /
`WorkflowRun`: the external system does not fit our WorkflowClass / TopologyType
enums, and Phase 5 only needs (a) feature vectors for open-world DETECTION and
(b) the host/port graph for TOPOLOGY OBSERVABILITY — never role/workflow labels.

Usage:
    venv/bin/python scripts/extract_offtheshelf.py --raw data/raw_offtheshelf \
        --out data/processed_offtheshelf --scapy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(raw: Path, out: Path, use_scapy: bool) -> None:
    from features.extractor import FeatureExtractor

    out.mkdir(parents=True, exist_ok=True)
    pcaps = sorted(raw.glob("*.pcap"))
    logger.info("Found %d off-the-shelf pcaps in %s", len(pcaps), raw)

    # Agent ports from the sidecars (a2a_mcp uses 10100-10105).
    ports: set[int] = set()
    for p in pcaps:
        sc = p.with_suffix(".json")
        if sc.exists():
            for addr in json.loads(sc.read_text()).get("agent_endpoints", {}).values():
                ps = addr.rsplit(":", 1)[-1]
                if ps.isdigit():
                    ports.add(int(ps))
    if not ports:
        ports = {10100, 10101, 10102, 10103, 10104, 10105}
    logger.info("Agent ports: %s", sorted(ports))

    extractor = FeatureExtractor(agent_ports=ports, use_scapy=use_scapy)
    labels: dict[str, dict] = {}
    ok = 0
    for p in pcaps:
        sc = p.with_suffix(".json")
        meta = json.loads(sc.read_text()) if sc.exists() else {}
        rid = p.stem
        try:
            feats = extractor.extract(p, run_id=rid)
        except ValueError as exc:  # zero valid A2A flows — fail loud (per-pcap)
            logger.error("ZERO-FLOW extraction failure: %s", exc)
            feats = None
        if feats is None:
            logger.warning("No features from %s", p.name)
            continue
        feats.save(out / f"{rid}.npz")
        labels[rid] = {
            "system": "a2a_mcp",
            "kind": "off_the_shelf_external",
            "input_prompt": meta.get("input_prompt", "")[:200],
        }
        ok += 1

    # FAIL HARD on a systematic zero-extraction (the IPv6-bug class).
    if pcaps and ok == 0:
        raise SystemExit(
            f"FATAL: extracted 0/{len(pcaps)} off-the-shelf traces from {raw} — "
            f"systematic zero-flow failure (check agent ports / IPv4-vs-IPv6). "
            f"Refusing to write an empty dataset."
        )

    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    logger.info("Extracted %d/%d traces -> %s", ok, len(pcaps), out / "labels.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract features from off-the-shelf a2a_mcp pcaps")
    ap.add_argument("--raw", default="data/raw_offtheshelf")
    ap.add_argument("--out", default="data/processed_offtheshelf")
    ap.add_argument("--scapy", action="store_true", help="Use scapy instead of pyshark")
    args = ap.parse_args()
    main(Path(args.raw), Path(args.out), args.scapy)
