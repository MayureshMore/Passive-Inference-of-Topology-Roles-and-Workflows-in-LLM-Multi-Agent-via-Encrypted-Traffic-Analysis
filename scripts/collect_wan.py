#!/usr/bin/env python3
"""
C5 WAN harness — LOCAL side (run this on the US MacBook host).

Drives the orchestrator locally while the specialist agents run on the remote
India host (started with scripts/serve_agents.py).  Every orchestrator->
specialist call therefore crosses the real wide-area network and is captured on
the WAN interface, giving genuine WAN traffic for the C5 robustness experiment.

Both sites run their own local Ollama, so only A2A flows cross the WAN (the
inference-latency confound is avoided — proposal §8.1).

Run ONE topology per invocation: the remote specialists are wired for a single
topology at startup, so --topology here MUST match the --topology serve_agents
was started with.  Repeat for star / chain / mesh (restart serve_agents each
time), all writing to the same --out dir.

num_predict MUST match the local collection (256) — otherwise WAN responses
differ in token count and the cross-network comparison measures token length,
not the network.  Both defaults are 256.

Prereqs:
  1. On the India host:  venv/bin/python scripts/serve_agents.py --topology star --deployment a
  2. Confirm reachability: curl http://<INDIA_IP>:8001/.well-known/agent-card.json

Usage (on the US host) — once per topology, matching the remote:
    sudo venv/bin/python scripts/collect_wan.py \
        --remote-host 203.0.113.7 --iface en0 \
        --deployment a --topology star --n 50 --num-predict 256 --out data/raw_wan

Then extract + evaluate:
    venv/bin/python scripts/extract_features.py --raw data/raw_wan --out data/processed_wan --scapy
    venv/bin/python scripts/evaluate_cross_network.py --local data/processed --wan data/processed_wan
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, AgentRole
from agents.orchestrator import OrchestratorAgent
from agents_b.orchestrator_b import OrchestratorB
from capture.automation import TraceCollector
from capture.recorder import PacketRecorder
from workflows import WORKFLOW_REGISTRY, WorkflowClass
from workflows.base import TopologyType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORTS = {"a": {"orchestrator": 8000, "executor": 8001, "retriever": 8002, "validator": 8003},
         "b": {"orchestrator": 8010, "executor": 8011, "retriever": 8012, "validator": 8013}}
ORCH = {"a": OrchestratorAgent, "b": OrchestratorB}
DEFAULT_MODEL = {"a": "llama3.2:3b", "b": "qwen2.5:7b"}


def _remote_downstream(topology: str, host: str, ports: dict[str, int]) -> list[str]:
    url = lambda r: f"http://{host}:{ports[r]}"
    if topology == "star":
        return [url("executor"), url("retriever"), url("validator")]
    if topology == "chain":
        return [url("executor")]
    if topology == "mesh":
        return [url("executor"), url("retriever")]
    raise ValueError(topology)


TOPO_EDGES = {
    "star":  [["orchestrator", "executor"], ["orchestrator", "retriever"], ["orchestrator", "validator"]],
    "chain": [["orchestrator", "executor"], ["executor", "retriever"], ["retriever", "validator"]],
    "mesh":  [["orchestrator", "executor"], ["orchestrator", "retriever"],
              ["executor", "retriever"], ["retriever", "validator"]],
}


async def _reachable(host: str, port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"http://{host}:{port}/.well-known/agent-card.json")
            return r.status_code == 200
    except Exception:
        return False


async def main(args: argparse.Namespace) -> None:
    dep = args.deployment
    ports = PORTS[dep]
    model = args.model or DEFAULT_MODEL[dep]
    num_predict = args.num_predict if args.num_predict > 0 else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One topology per run — must match how serve_agents was started remotely.
    topologies = [args.topology]
    workflows = [args.workflow] if args.workflow else [wf.value for wf in WorkflowClass]

    # Confirm the remote specialists are up before capturing.
    if not await _reachable(args.remote_host, ports["executor"]):
        logger.error("Remote executor not reachable at %s:%d — start serve_agents.py "
                     "on the India host first.", args.remote_host, ports["executor"])
        return

    recorder = PacketRecorder(
        output_dir=out_dir,
        interface=args.iface,                 # WAN interface, e.g. en0 (NOT lo0)
        agent_ports=list(ports.values()),
    )
    # Endpoints are the REMOTE host:port pairs (that is where capture sees them).
    agent_endpoints = {role: f"{args.remote_host}:{port}" for role, port in ports.items()}

    all_stats: dict[str, dict] = {}
    for topology in topologies:
        downstream = _remote_downstream(topology, args.remote_host, ports)
        for wf_name in workflows:
            wf_class = WorkflowClass(wf_name)
            wf_instance = WORKFLOW_REGISTRY[wf_class]()
            # Same seed scheme as the pilots → identical prompts to the local runs.
            _TOPOS = ["star", "chain", "mesh"]
            _WFS = [wf.value for wf in WorkflowClass]
            seed = _TOPOS.index(topology) * len(_WFS) + _WFS.index(wf_name) + 100
            prompts = wf_instance.sample_prompts(n=args.n, seed=seed)

            cfg = AgentConfig(role=AgentRole.ORCHESTRATOR, port=ports["orchestrator"],
                              downstream_agents=downstream, ollama_model=model,
                              ollama_num_predict=num_predict)
            orch = ORCH[dep](cfg)

            async def exec_fn(prompt: str, _orch=orch) -> None:
                async with _orch:
                    await _orch.handle_task(task_id=f"wan_{uuid.uuid4().hex[:8]}", content=prompt)

            collector = TraceCollector(
                orchestrator_fn=exec_fn, recorder=recorder,
                workflow_class=wf_class, topology=TopologyType(topology),
                agent_endpoints=agent_endpoints, topology_edges=TOPO_EDGES[topology],
                output_dir=out_dir, deployment=dep,
            )
            logger.info("WAN collect  workflow=%-20s topology=%s  n=%d  remote=%s",
                        wf_name, topology, args.n, args.remote_host)
            runs = await collector.collect(prompts, inter_run_delay=0.5)
            ok = sum(1 for r in runs if r.success)
            all_stats[f"{topology}/{wf_name}"] = {"total": args.n, "success": ok}

    print("\n" + "=" * 56 + "\n  WAN COLLECTION SUMMARY\n" + "=" * 56)
    for k, s in all_stats.items():
        print(f"  {k:<40} {s['success']}/{s['total']}")
    print(f"\n  pcaps → {out_dir}/  ({len(list(out_dir.glob('*.pcap')))} files)")
    print("  next: extract_features.py --raw %s --out data/processed_wan --scapy" % out_dir)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C5 WAN harness — collect cross-WAN traces (US host)")
    p.add_argument("--remote-host", required=True, help="India host IP/hostname running serve_agents.py")
    p.add_argument("--iface", default="en0", help="WAN capture interface (default en0; do NOT use lo0)")
    p.add_argument("--deployment", choices=["a", "b"], default="a")
    p.add_argument("--topology", required=True, choices=["star", "chain", "mesh"],
                   help="MUST match the --topology serve_agents was started with on the remote host")
    p.add_argument("--workflow", choices=[wf.value for wf in WorkflowClass])
    p.add_argument("--n", type=int, default=50, help="Traces per (workflow, topology) pair (match local: 50)")
    p.add_argument("--model", default=None)
    p.add_argument("--num-predict", type=int, default=256, dest="num_predict",
                   help="Ollama output-token cap — MUST match the local collection (256) for a valid comparison")
    p.add_argument("--out", default="data/raw_wan")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse()))
