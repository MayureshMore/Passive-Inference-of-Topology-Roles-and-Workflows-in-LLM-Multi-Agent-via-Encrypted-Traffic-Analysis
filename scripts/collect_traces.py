#!/usr/bin/env python3
"""
Automated trace collection driver (proposal §8.3).

Runs N workflow executions for a given (workflow_class, topology) pair,
capturing pcap + JSON label for each run.  Must be run with agents already
started on the configured hosts/ports.

Usage:
    python scripts/collect_traces.py \
        --workflow research_retrieval \
        --topology star \
        --n 50 \
        --out data/raw \
        --config configs/testbed_local.yaml

Requires sudo access for tcpdump (or set CAP_NET_RAW).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, AgentRole
from agents.orchestrator import OrchestratorAgent
from capture.automation import TraceCollector
from capture.recorder import PacketRecorder
from workflows import WORKFLOW_REGISTRY, WorkflowClass
from workflows.base import TopologyType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_testbed_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Testbed config not found at %s — using localhost defaults", config_path)
        return {
            "orchestrator": "127.0.0.1:8000",
            "executor": "127.0.0.1:8001",
            "retriever": "127.0.0.1:8002",
            "validator": "127.0.0.1:8003",
            "interface": "lo0",
        }
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_orchestrator(topology: str, testbed: dict) -> OrchestratorAgent:
    """
    Build an OrchestratorAgent with downstream agents configured for the
    given topology.
    """
    host_port = testbed.get("orchestrator", "127.0.0.1:8000").split(":")
    orch_host = host_port[0]
    orch_port = int(host_port[1]) if len(host_port) > 1 else 8000

    def _url(role: str) -> str:
        hp = testbed.get(role, f"127.0.0.1:{8000 + ['orchestrator','executor','retriever','validator'].index(role)}")
        return f"http://{hp}"

    if topology == "star":
        downstream = [_url("executor"), _url("retriever"), _url("validator")]
    elif topology == "chain":
        downstream = [_url("executor")]  # chain continues inside executor
    elif topology == "mesh":
        downstream = [_url("executor"), _url("retriever")]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    cfg = AgentConfig(
        role=AgentRole.ORCHESTRATOR,
        host=orch_host,
        port=orch_port,
        downstream_agents=downstream,
        ollama_model=testbed.get("ollama_model", "llama3.2:3b"),
        ollama_base_url=testbed.get("ollama_base_url", "http://localhost:11434"),
    )
    return OrchestratorAgent(cfg)


async def main(args: argparse.Namespace) -> None:
    wf_class = WorkflowClass(args.workflow)
    topo = TopologyType(args.topology)
    out_dir = Path(args.out)
    testbed = load_testbed_config(Path(args.config))

    workflow_cls = WORKFLOW_REGISTRY[wf_class]
    workflow = workflow_cls()
    prompts = workflow.sample_prompts(n=args.n)

    orchestrator = build_orchestrator(args.topology, testbed)

    agent_endpoints = {
        role: testbed.get(role, f"127.0.0.1:{8000+i}")
        for i, role in enumerate(["orchestrator", "executor", "retriever", "validator"])
    }

    # Topology edges — must match run_pilot.py TOPO_EDGES exactly so labels
    # are consistent between collect_traces.py and run_pilot.py collections.
    topo_edges = {
        "star": [["orchestrator","executor"],["orchestrator","retriever"],["orchestrator","validator"]],
        "chain": [["orchestrator","executor"],["executor","retriever"],["retriever","validator"]],
        "mesh": [["orchestrator","executor"],["orchestrator","retriever"],
                 ["executor","retriever"],["retriever","validator"]],
    }

    recorder = PacketRecorder(
        output_dir=out_dir,
        interface=testbed.get("interface", "any"),
        agent_ports=[int(v.split(":")[1]) for v in agent_endpoints.values()],
    )

    async def exec_workflow(prompt: str):
        async with orchestrator:
            await orchestrator.handle_task(task_id="collect_run", content=prompt)

    collector = TraceCollector(
        orchestrator_fn=exec_workflow,
        recorder=recorder,
        workflow_class=wf_class,
        topology=topo,
        agent_endpoints=agent_endpoints,
        topology_edges=topo_edges[args.topology],
        output_dir=out_dir,
    )

    logger.info(
        "Collecting %d traces for workflow=%s topology=%s → %s",
        args.n, args.workflow, args.topology, out_dir,
    )
    runs = await collector.collect(prompts, inter_run_delay=args.delay)

    from capture.labeler import TraceLabeler
    summary = TraceLabeler.summary(runs)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect A2A traffic traces")
    parser.add_argument("--workflow", required=True,
                        choices=[wf.value for wf in WorkflowClass])
    parser.add_argument("--topology", required=True,
                        choices=[t.value for t in TopologyType])
    parser.add_argument("--n", type=int, default=50, help="Number of traces")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--config", default="configs/testbed_local.yaml",
                        help="Testbed YAML config")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Inter-run delay in seconds")
    args = parser.parse_args()
    asyncio.run(main(args))
