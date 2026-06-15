"""
Contract tests for the capture↔analysis seam.

These guard the two load-bearing invariants the graph flagged as the system's
biggest cross-community bridge (TraceLabeler / WorkflowRun + flow canonicalization):

  1. Flow canonicalization: both directions of one TCP connection — and, under
     the SDK's HTTP keep-alive, every request/response reusing one connection —
     collapse to a single canonical bidirectional flow key.
  2. WorkflowRun sidecar round-trips through TraceLabeler and carries every field
     the feature extractor / evaluation consume.
"""
from pathlib import Path

import pytest

from capture.labeler import TraceLabeler
from features.extractor import FeatureExtractor
from workflows.base import TopologyType, WorkflowClass, WorkflowRun

scapy_all = pytest.importorskip("scapy.all")
Ether, IP, TCP, wrpcap = scapy_all.Ether, scapy_all.IP, scapy_all.TCP, scapy_all.wrpcap

AGENT = 8001
CLIENT = 55000


def _pkt(sport, dport, payload=b"x" * 40):
    return Ether() / IP(src="127.0.0.1", dst="127.0.0.1") / TCP(sport=sport, dport=dport) / payload


def test_flow_canonicalization_merges_both_directions(tmp_path):
    pcap = tmp_path / "two_dir.pcap"
    wrpcap(str(pcap), [_pkt(CLIENT, AGENT), _pkt(AGENT, CLIENT)])  # request, response
    fx = FeatureExtractor(agent_ports={AGENT}, use_scapy=True)
    recs = fx._read_with_scapy(pcap)
    keys = {r[2] for r in recs}
    dirs = sorted(r[3] for r in recs)
    assert len(recs) == 2
    assert len(keys) == 1, f"both directions must share one flow key, got {keys}"
    assert dirs == [-1, 1], "request=-1 (to agent), response=+1 (from agent)"


def test_keepalive_one_connection_one_flow(tmp_path):
    # Three request/response pairs reusing the SAME 4-tuple (HTTP keep-alive) —
    # the SDK pools connections, so this must still be exactly one flow.
    pcap = tmp_path / "keepalive.pcap"
    pkts = []
    for _ in range(3):
        pkts += [_pkt(CLIENT, AGENT), _pkt(AGENT, CLIENT)]
    wrpcap(str(pcap), pkts)
    fx = FeatureExtractor(agent_ports={AGENT}, use_scapy=True)
    recs = fx._read_with_scapy(pcap)
    assert len({r[2] for r in recs}) == 1, "keep-alive reuse must map to one canonical flow"
    assert len(recs) == 6


def test_non_agent_traffic_is_dropped(tmp_path):
    pcap = tmp_path / "noise.pcap"
    wrpcap(str(pcap), [_pkt(40000, 40001)])  # neither port is an agent port
    fx = FeatureExtractor(agent_ports={AGENT}, use_scapy=True)
    assert fx._read_with_scapy(pcap) == []


def test_workflowrun_sidecar_roundtrips(tmp_path):
    pcap = tmp_path / "research_retrieval_star_abc123.pcap"
    pcap.write_bytes(b"")  # placeholder; labeler keys off the path
    run = WorkflowRun(
        workflow_class=WorkflowClass.RESEARCH_RETRIEVAL,
        topology=TopologyType.STAR,
        agent_endpoints={"orchestrator": "127.0.0.1:8000", "executor": "127.0.0.1:8001"},
        topology_edges=[["orchestrator", "executor"]],
        input_prompt="hello",
        pcap_path=str(pcap),
        success=True,
        deployment="b",
    )
    TraceLabeler.write(run)
    back = TraceLabeler.read(pcap)
    assert back.workflow_class == WorkflowClass.RESEARCH_RETRIEVAL
    assert back.topology == TopologyType.STAR
    assert back.topology_edges == [["orchestrator", "executor"]]
    assert back.deployment == "b"
    assert back.end_ts > 0  # writer stamps end_ts


def test_workflowrun_exposes_consumer_fields():
    # Fields the feature extractor + evaluation depend on must exist.
    run = WorkflowRun(workflow_class=WorkflowClass.CODE_REVIEW, topology=TopologyType.CHAIN)
    for field in ("workflow_class", "topology", "topology_edges",
                  "agent_endpoints", "input_prompt", "deployment", "success"):
        assert hasattr(run, field), f"WorkflowRun missing consumer field: {field}"
