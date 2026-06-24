#!/usr/bin/env python3
"""
C5 WAN harness — REMOTE side (run this on the India Dell host).

Serves the three specialist agents (executor, retriever, validator) bound to
0.0.0.0 so the US orchestrator can reach them across the wide-area network.
Each agent uses THIS host's local Ollama (keeping inference traffic local, so
only A2A flows cross the WAN — see the inference-latency confound note in the
proposal §8.1).

The agents keep running until interrupted (Ctrl-C).  Start them here, then run
scripts/collect_wan.py on the US host pointing --remote-host at this machine.

Usage (on the India host):
    # Deployment A specialists on the default ports (8001-8003):
    venv/bin/python scripts/serve_agents.py --topology star --deployment a
    # Deployment B specialists:
    venv/bin/python scripts/serve_agents.py --topology chain --deployment b

Topology note: chain/mesh forwarding between specialists stays LOCAL on this
host (127.0.0.1); only the orchestrator->specialist hops cross the WAN.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, AgentRole
from agents.executor import ExecutorAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from agents_b.executor_b import ExecutorB
from agents_b.retriever_b import RetrieverB
from agents_b.validator_b import ValidatorB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Deployment A uses 8001-8003; B uses 8011-8013; C (LangGraph) uses 8031-8033.
# Deployment C reuses A's specialist classes unchanged (only its orchestrator,
# driven by the collector, is LangGraph); these are the specialist servers.
PORTS = {
    "a": {"executor": 8001, "retriever": 8002, "validator": 8003},
    "b": {"executor": 8011, "retriever": 8012, "validator": 8013},
    "langgraph": {"executor": 8031, "retriever": 8032, "validator": 8033},
}
CLASSES = {
    "a": {"executor": ExecutorAgent, "retriever": RetrieverAgent, "validator": ValidatorAgent},
    "b": {"executor": ExecutorB, "retriever": RetrieverB, "validator": ValidatorB},
    "langgraph": {"executor": ExecutorAgent, "retriever": RetrieverAgent, "validator": ValidatorAgent},
}
DEFAULT_MODEL = {"a": "llama3.2:3b", "b": "qwen2.5:7b", "langgraph": "llama3.2:3b"}


def _downstream(role: str, topology: str, ports: dict[str, int]) -> list[str]:
    """India-local forwarding URLs for chain/mesh (specialists are co-located)."""
    local = lambda r: f"http://127.0.0.1:{ports[r]}"
    if topology in ("chain", "mesh"):
        if role == "executor":
            return [local("retriever")]
        if role == "retriever":
            return [local("validator")]
    return []  # star: all specialists are leaves


async def main(args: argparse.Namespace) -> None:
    dep = args.deployment
    ports = PORTS[dep]
    classes = CLASSES[dep]
    model = args.model or DEFAULT_MODEL[dep]
    num_predict = args.num_predict if args.num_predict > 0 else None

    agents = []
    tasks = []
    for role, AgentClass in classes.items():
        cfg = AgentConfig(
            role=AgentRole(role),
            host="0.0.0.0",
            port=ports[role],
            downstream_agents=_downstream(role, args.topology, ports),
            ollama_model=model,
            ollama_num_predict=num_predict,
            defense=args.defense,
        )
        agent = AgentClass(cfg)
        agents.append(agent)
        tasks.append(asyncio.create_task(agent.run(), name=f"{dep}-{role}"))

    logger.info(
        "Serving deployment-%s specialists on 0.0.0.0 ports %s "
        "(topology=%s model=%s defense=%s).  Ctrl-C to stop.",
        dep.upper(), list(ports.values()), args.topology, model, args.defense,
    )
    logger.info("From the US host run: scripts/collect_wan.py --remote-host <THIS_HOST_IP> "
                "--deployment %s --topology %s", dep, args.topology)
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for agent in agents:
            await agent.shutdown()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C5 WAN harness — serve specialist agents (remote/India host)")
    p.add_argument("--topology", required=True, choices=["star", "chain", "mesh"],
                   help="Wire specialists for this topology — MUST match collect_wan --topology on the US host")
    p.add_argument("--deployment", choices=["a", "b", "langgraph"], default="a")
    p.add_argument("--model", default=None, help="Override Ollama model (default per deployment)")
    p.add_argument("--num-predict", type=int, default=256, dest="num_predict",
                   help="Cap Ollama output tokens (match the local collection: 256; 0 = unlimited)")
    p.add_argument("--defense", default="none", choices=["none", "pad", "rate", "both"])
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(_parse()))
    except KeyboardInterrupt:
        pass
