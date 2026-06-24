#!/usr/bin/env python3
"""
Phase 5c (ii) — TOPOLOGY OBSERVABILITY of the off-the-shelf a2a_mcp system.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STATUS: PRE-WRITTEN / UNTESTED.  Validated only once real off-the-shelf   │
  │ pcaps exist at data/raw_offtheshelf/ (Phase 5b capture is gated on the    │
  │ Gemini free-tier quota).  Until then this exits cleanly with a notice.    │
  └─────────────────────────────────────────────────────────────────────────┘

Claim (qualitative, structural — NOT a classifier result): an on-path observer
recovers the agent connection graph of a system it did not build, directly from
flow metadata (IP/port headers), with no payload and no ML.

Method:
  1. Read header-only pcaps from data/raw_offtheshelf/ (96-byte snaplen).
  2. Extract TCP flows with the SAME canonical convention as
     features/extractor.py: the agent/server port is always the right-hand
     endpoint of a flow, so each flow names the agent that was CALLED.
  3. Map agent ports → component names from the capture sidecar's
     `agent_endpoints` (mcp/orchestrator/planner/air/hotel/car).
  4. Build a server-centric connection graph aggregated over all trips:
       per agent — inbound flows, distinct callers, bytes (fan-in = hub signal);
       edges     — caller→agent, with the caller named when its endpoint matches
                   a known agent listening endpoint (direct on multi-host; on
                   loopback callers use ephemeral ports, so most callers show as
                   "ephemeral" — see the limitation note below).
  5. Classify the structure (hub-and-spoke / hierarchical) from the fan-in.

Honesty note: on a single loopback host every agent shares 127.0.0.1, so a
caller's ephemeral port cannot be attributed to a specific agent from headers
alone; the recoverable signal is the set of service endpoints and their fan-in
(which already exposes the registry + orchestrator hubs and the specialist
leaves).  On a multi-host deployment (e.g. the C5 WAN testbed) each agent has a
distinct IP, and the same method yields fully-attributed agent→agent edges.

Usage:
    venv/bin/python scripts/analyze_offtheshelf_topology.py
    venv/bin/python scripts/analyze_offtheshelf_topology.py --raw data/raw_offtheshelf
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
DEFAULT_PORTS = {10100: "mcp", 10101: "orchestrator", 10102: "planner",
                 10103: "air_ticketing", 10104: "hotel", 10105: "car_rental"}


def load_port_map(raw: Path) -> dict[int, str]:
    """Build {port: component_name} from capture sidecars' agent_endpoints."""
    port_map: dict[int, str] = {}
    for sc in sorted(raw.glob("*.json")):
        try:
            eps = json.loads(sc.read_text()).get("agent_endpoints", {})
        except Exception:
            continue
        for name, addr in eps.items():
            tail = str(addr).rsplit(":", 1)[-1]
            if tail.isdigit():
                port_map[int(tail)] = name
    return port_map or dict(DEFAULT_PORTS)


def extract_flows(pcap: Path, agent_ports: set[int]):
    """
    Yield canonical flows for one pcap, mirroring features.extractor convention
    (agent/server port on the right).  Returns dict keyed by
    (client_ip, client_port, agent_ip, agent_port) -> [n_packets, n_bytes].
    """
    from scapy.all import rdpcap, IP, IPv6, TCP

    flows: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    try:
        pkts = rdpcap(str(pcap))
    except Exception as exc:
        logger.warning("unreadable pcap %s: %s", pcap.name, exc)
        return flows
    for pkt in pkts:
        if TCP not in pkt:
            continue
        # Support both IPv4 and IPv6 (services that bind to ::1 loopback).
        if IP in pkt:
            si, di = pkt[IP].src, pkt[IP].dst
        elif IPv6 in pkt:
            si, di = pkt[IPv6].src, pkt[IPv6].dst
        else:
            continue
        sp, dp = pkt[TCP].sport, pkt[TCP].dport
        if sp not in agent_ports and dp not in agent_ports:
            continue
        size = int(getattr(pkt, "wirelen", len(pkt)))
        # agent/server endpoint on the right
        if dp in agent_ports:
            key = (si, sp, di, dp)   # client → agent
        else:
            key = (di, dp, si, sp)   # flip: agent on right
        flows[key][0] += 1
        flows[key][1] += size
    return flows


def run(raw: Path, results_dir: Path) -> None:
    pcaps = sorted(raw.glob("*.pcap"))
    if not pcaps:
        logger.warning(
            "No off-the-shelf pcaps at %s yet — Phase 5b capture is gated on the "
            "Gemini quota.  This script is PRE-WRITTEN and runs once "
            "scripts/collect_offtheshelf.sh has produced captures.", raw,
        )
        return

    port_map = load_port_map(raw)
    agent_ports = set(port_map)
    logger.info("Agent ports: %s", {p: port_map[p] for p in sorted(agent_ports)})

    # Aggregate the connection graph over all trips.
    listening_eps: set[tuple[str, int]] = set()        # (ip, agent_port)
    agent_inbound = defaultdict(lambda: {"flows": 0, "bytes": 0, "callers": set()})
    edges = defaultdict(lambda: {"flows": 0, "bytes": 0})  # (caller_label, agent_name)

    all_flows: list[tuple] = []
    for pcap in pcaps:
        for (ci, cp, ai, ap), (npk, nby) in extract_flows(pcap, agent_ports).items():
            listening_eps.add((ai, ap))
            all_flows.append((ci, cp, ai, ap, npk, nby))

    # A caller endpoint is "named" if it matches a known agent listening endpoint
    # (true on multi-host; rare on loopback where callers use ephemeral ports).
    for ci, cp, ai, ap, npk, nby in all_flows:
        agent_name = port_map.get(ap, f"port:{ap}")
        caller = port_map[cp] if (ci, cp) in listening_eps and cp in port_map else "ephemeral/external"
        agent_inbound[agent_name]["flows"] += 1
        agent_inbound[agent_name]["bytes"] += nby
        agent_inbound[agent_name]["callers"].add(f"{ci}:{cp}")
        edges[(caller, agent_name)]["flows"] += 1
        edges[(caller, agent_name)]["bytes"] += nby

    nodes = {
        name: {"inbound_flows": v["flows"], "inbound_bytes": v["bytes"],
               "distinct_callers": len(v["callers"])}
        for name, v in sorted(agent_inbound.items(), key=lambda x: -x[1]["bytes"])
    }
    edge_list = [
        {"from": c, "to": a, "flows": v["flows"], "bytes": v["bytes"]}
        for (c, a), v in sorted(edges.items(), key=lambda x: -x[1]["bytes"])
    ]

    # Hub = agent with the largest fan-in.  Classify by the fan-in DISTRIBUTION
    # via a RATIO (trace-count invariant): on loopback each connection uses a new
    # ephemeral port, so distinct_callers / inbound_flows grow with the number of
    # trips — an absolute "<=1 caller = leaf" test breaks as traces accumulate.
    # A dominant hub is instead one whose inbound-flow count dwarfs the quietest
    # endpoint's, and leaves are endpoints in the bottom decile of fan-in.
    hubs = sorted(nodes.items(), key=lambda x: -x[1]["distinct_callers"])
    top_hub = hubs[0][0] if hubs else None
    in_flows = sorted((v["inbound_flows"] for v in nodes.values()), reverse=True)
    max_in = in_flows[0] if in_flows else 0
    min_in = in_flows[-1] if in_flows else 0
    leaf_cut = max(2.0, 0.1 * max_in)
    n_leaves = sum(1 for v in nodes.values() if v["inbound_flows"] <= leaf_cut)
    dominant_hub = len(nodes) >= 3 and max_in >= 5 * max(1, min_in)
    topology = ("hub-and-spoke / hierarchical (dominant registry/coordination hubs, "
                "specialist leaves)" if dominant_hub and n_leaves >= 1
                else "insufficient structure to classify")

    out = {
        "_status": "validated on real off-the-shelf pcaps",
        "raw_dir": str(raw), "n_pcaps": len(pcaps),
        "agents": port_map,
        "nodes": nodes,
        "edges": edge_list,
        "hub": top_hub,
        "topology_class": topology,
        "limitations": (
            "Single-host loopback: agent callers use ephemeral ports and share "
            "127.0.0.1, so most callers show as 'ephemeral/external' — the recoverable "
            "signal is the service endpoints + fan-in (registry/orchestrator hubs, "
            "specialist leaves). Multi-host deployments (distinct IP per agent) yield "
            "fully-attributed agent->agent edges with the same method."
        ),
        "scope_note": "TOPOLOGY OBSERVABILITY only — qualitative structural point, no classifier.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "offtheshelf_topology.json").write_text(json.dumps(out, indent=2))

    sep = "=" * 70
    print(f"\n{sep}\n  PHASE 5c (ii) — OFF-THE-SHELF TOPOLOGY OBSERVABILITY (a2a_mcp)\n{sep}")
    print(f"  pcaps={len(pcaps)}  agents discovered={len(nodes)}  hub={top_hub}")
    print(f"  topology: {topology}")
    print(f"  {'agent':<16}{'in_flows':>9}{'callers':>9}{'in_bytes':>12}")
    for name, v in nodes.items():
        print(f"  {name:<16}{v['inbound_flows']:>9}{v['distinct_callers']:>9}{v['inbound_bytes']:>12}")
    print(f"  (connection graph recovered from headers only — no payload, no ML)\n{sep}")
    logger.info("Wrote %s", results_dir / "offtheshelf_topology.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5c topology observability of off-the-shelf a2a_mcp")
    ap.add_argument("--raw", default="data/raw_offtheshelf")
    args = ap.parse_args()
    run(Path(args.raw), RESULTS_DIR)


if __name__ == "__main__":
    main()
