#!/usr/bin/env python3
"""
Weeks 1-2 Pilot: full end-to-end data collection across all topologies
and workflow classes.

Starts executor/retriever/validator with topology-specific downstream
routing, captures N pcap traces per (workflow, topology) pair, then
tears down agents and moves to the next topology.

Usage:
    sudo venv/bin/python scripts/run_pilot.py
    sudo venv/bin/python scripts/run_pilot.py --n 5 --topology star
    sudo venv/bin/python scripts/run_pilot.py --n 3 --workflow research_retrieval
    sudo venv/bin/python scripts/run_pilot.py --n 10 --model llama3.2:3b --out data/raw
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, AgentRole
from agents.executor import ExecutorAgent
from agents.orchestrator import OrchestratorAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from capture.automation import TraceCollector
from capture.recorder import PacketRecorder
from workflows import WORKFLOW_REGISTRY, WorkflowClass
from workflows.base import TopologyType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Port assignments ──────────────────────────────────────────────────────────

PORTS: dict[str, int] = {
    "orchestrator": 8000,
    "executor": 8001,
    "retriever": 8002,
    "validator": 8003,
}

# ── Ground-truth topology edges (role → role) ─────────────────────────────────
#
# mesh: validator has no downstream to avoid the unbounded cycle
#   validator→executor→retriever→validator→...
# The mesh is still topologically distinct via the parallel orchestrator
# fan-out to executor AND retriever, plus executor→retriever cross-link.

TOPO_EDGES: dict[str, list[list[str]]] = {
    "star": [
        ["orchestrator", "executor"],
        ["orchestrator", "retriever"],
        ["orchestrator", "validator"],
    ],
    "chain": [
        ["orchestrator", "executor"],
        ["executor", "retriever"],
        ["retriever", "validator"],
    ],
    "mesh": [
        ["orchestrator", "executor"],
        ["orchestrator", "retriever"],
        ["executor", "retriever"],
        ["retriever", "validator"],
    ],
}


def _url(role: str, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{PORTS[role]}"


# ── Agent config builders ─────────────────────────────────────────────────────

def _agent_configs(topology: str, model: str) -> dict[str, AgentConfig]:
    """
    Return per-role AgentConfig for executor/retriever/validator with
    downstream_agents set to match the topology's routing.
    """
    if topology == "star":
        return {
            "executor": AgentConfig(
                role=AgentRole.EXECUTOR, port=PORTS["executor"],
                downstream_agents=[], ollama_model=model,
            ),
            "retriever": AgentConfig(
                role=AgentRole.RETRIEVER, port=PORTS["retriever"],
                downstream_agents=[], ollama_model=model,
            ),
            "validator": AgentConfig(
                role=AgentRole.VALIDATOR, port=PORTS["validator"],
                downstream_agents=[], ollama_model=model,
            ),
        }

    if topology == "chain":
        return {
            "executor": AgentConfig(
                role=AgentRole.EXECUTOR, port=PORTS["executor"],
                downstream_agents=[_url("retriever")], ollama_model=model,
            ),
            "retriever": AgentConfig(
                role=AgentRole.RETRIEVER, port=PORTS["retriever"],
                downstream_agents=[_url("validator")], ollama_model=model,
            ),
            "validator": AgentConfig(
                role=AgentRole.VALIDATOR, port=PORTS["validator"],
                downstream_agents=[], ollama_model=model,
            ),
        }

    if topology == "mesh":
        # executor forwards to retriever; retriever forwards to validator.
        # validator has no downstream (avoids the unbounded retry loop).
        return {
            "executor": AgentConfig(
                role=AgentRole.EXECUTOR, port=PORTS["executor"],
                downstream_agents=[_url("retriever")], ollama_model=model,
            ),
            "retriever": AgentConfig(
                role=AgentRole.RETRIEVER, port=PORTS["retriever"],
                downstream_agents=[_url("validator")], ollama_model=model,
            ),
            "validator": AgentConfig(
                role=AgentRole.VALIDATOR, port=PORTS["validator"],
                downstream_agents=[], ollama_model=model,
            ),
        }

    raise ValueError(f"Unknown topology: {topology}")


def _orchestrator(topology: str, model: str) -> OrchestratorAgent:
    if topology == "star":
        downstream = [_url("executor"), _url("retriever"), _url("validator")]
    elif topology == "chain":
        downstream = [_url("executor")]
    elif topology == "mesh":
        downstream = [_url("executor"), _url("retriever")]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    cfg = AgentConfig(
        role=AgentRole.ORCHESTRATOR,
        port=PORTS["orchestrator"],
        downstream_agents=downstream,
        ollama_model=model,
    )
    return OrchestratorAgent(cfg)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _kill_port(port: int) -> None:
    proc = await asyncio.create_subprocess_shell(
        f"lsof -ti tcp:{port} 2>/dev/null | xargs kill -9 2>/dev/null; true",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def _wait_ready(role: str, retries: int = 30) -> bool:
    url = _url(role)
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{url}/.well-known/agent.json")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        if attempt == 0:
            logger.info("  waiting for %s at %s ...", role, url)
        await asyncio.sleep(0.4)
    return False


# ── Per-topology run ──────────────────────────────────────────────────────────

async def run_topology(
    topology: str,
    workflows: list[str],
    n: int,
    model: str,
    out_dir: Path,
) -> dict[str, dict]:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TOPOLOGY: %s", topology.upper())
    logger.info("=" * 60)

    # Free downstream agent ports
    for role in ("executor", "retriever", "validator"):
        await _kill_port(PORTS[role])
    await asyncio.sleep(0.6)

    # Start downstream agents
    configs = _agent_configs(topology, model)
    agent_classes = {
        "executor": ExecutorAgent,
        "retriever": RetrieverAgent,
        "validator": ValidatorAgent,
    }
    tasks: list[asyncio.Task] = []
    for role, AgentClass in agent_classes.items():
        agent = AgentClass(configs[role])
        tasks.append(asyncio.create_task(agent.run(), name=f"{topology}-{role}"))

    await asyncio.sleep(1.2)  # let uvicorn bind all three ports

    # Health check
    for role in ("executor", "retriever", "validator"):
        if not await _wait_ready(role):
            logger.error("  %s agent never came up — aborting topology %s", role, topology)
            for t in tasks:
                t.cancel()
            return {}
    logger.info("  All downstream agents ready.")

    recorder = PacketRecorder(
        output_dir=out_dir,
        interface="lo0",
        agent_ports=list(PORTS.values()),
    )
    agent_endpoints = {role: f"127.0.0.1:{port}" for role, port in PORTS.items()}
    edges = TOPO_EDGES[topology]

    stats: dict[str, dict] = {}

    for wf_name in workflows:
        wf_class = WorkflowClass(wf_name)
        wf_instance = WORKFLOW_REGISTRY[wf_class]()
        prompts = wf_instance.sample_prompts(n=n)

        # Build a fresh orchestrator for each workflow run (new HTTP client)
        orch = _orchestrator(topology, model)

        async def exec_fn(prompt: str, _orch: OrchestratorAgent = orch) -> None:
            tid = f"pilot_{uuid.uuid4().hex[:8]}"
            async with _orch:
                await _orch.handle_task(task_id=tid, content=prompt)

        collector = TraceCollector(
            orchestrator_fn=exec_fn,
            recorder=recorder,
            workflow_class=wf_class,
            topology=TopologyType(topology),
            agent_endpoints=agent_endpoints,
            topology_edges=edges,
            output_dir=out_dir,
        )

        logger.info(
            "  Collecting %d traces  workflow=%-20s topology=%s",
            n, wf_name, topology,
        )
        runs = await collector.collect(prompts, inter_run_delay=1.5)
        ok = sum(1 for r in runs if r.success)
        stats[f"{topology}/{wf_name}"] = {"total": n, "success": ok, "failed": n - ok}
        logger.info("    → %d/%d successful", ok, n)

    # Tear down downstream agents
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    # Force-kill lingering port holders and wait for OS to release sockets.
    # 0.3 s is too short — uvicorn TCP sockets stay in TIME_WAIT; next topology
    # fails to bind. 3 s covers TIME_WAIT on loopback + SO_REUSEADDR delay.
    for role in ("executor", "retriever", "validator"):
        await _kill_port(PORTS[role])
    await asyncio.sleep(3.0)

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        logger.warning(
            "Not running as root — tcpdump BPF capture will be empty. "
            "Re-run with: sudo venv/bin/python scripts/run_pilot.py"
        )

    topologies = [args.topology] if args.topology else ["star", "chain", "mesh"]
    workflows  = [args.workflow]  if args.workflow  else [wf.value for wf in WorkflowClass]
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pilot config: topologies=%s  workflows=%s  n=%d  model=%s  out=%s",
                topologies, workflows, args.n, args.model, out_dir)

    all_stats: dict[str, dict] = {}
    for topology in topologies:
        stats = await run_topology(
            topology=topology,
            workflows=workflows,
            n=args.n,
            model=args.model,
            out_dir=out_dir,
        )
        all_stats.update(stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  PILOT SUMMARY")
    print("=" * 60)
    total_ok = total_all = 0
    for key, s in all_stats.items():
        tag = "✓" if s["failed"] == 0 else f"{s['success']}/{s['total']}"
        print(f"  {key:<42} {tag}")
        total_ok  += s["success"]
        total_all += s["total"]

    pcap_files = list(out_dir.glob("*.pcap"))
    print()
    print(f"  Traces successful : {total_ok}/{total_all}")
    print(f"  pcap files written: {len(pcap_files)}  →  {out_dir}/")
    print("=" * 60)
    print()

    if total_ok > 0:
        print("Next steps:")
        print("  python scripts/extract_features.py --in data/raw --out data/processed")
        print("  python scripts/train_models.py --task workflow --model rf")
        print("  python scripts/evaluate.py --mode closed_world")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A2A Pilot: collect traces for all topology × workflow pairs")
    p.add_argument("--topology", choices=["star", "chain", "mesh"],
                   help="Run one topology only (default: all three)")
    p.add_argument("--workflow",
                   choices=[wf.value for wf in WorkflowClass],
                   help="Run one workflow only (default: all four)")
    p.add_argument("--n", type=int, default=5,
                   help="Traces per (workflow, topology) pair (default: 5)")
    p.add_argument("--model", default="llama3.2:3b",
                   help="Ollama model name")
    p.add_argument("--out", default="data/raw",
                   help="Output directory for pcap + label files")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse()))
