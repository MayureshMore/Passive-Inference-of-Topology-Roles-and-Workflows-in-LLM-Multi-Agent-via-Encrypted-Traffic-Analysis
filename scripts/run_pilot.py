#!/usr/bin/env python3
"""
Pilot: end-to-end data collection across all topologies and workflow classes,
for EITHER agent deployment (A or B) — one driver, no duplication.

Deployment A (default):  agents/    — parallel orchestrator (asyncio.gather),
  3-phase retriever, llama3.2:3b, ports 8000-8003.
Deployment B (--deployment b):  agents_b/ — sequential pipeline orchestrator,
  2-phase retriever, single-shot validator, qwen2.5:7b, ports 8010-8013.

The per-deployment differences live ENTIRELY in the DEPLOYMENTS registry below
(agent classes, ports, default model).  Everything else — capture, labeling,
seeds, teardown, the live C4 defenses — is shared and identical, which is exactly
what the cross-deployment (A vs B) experiment requires.

Usage:
    sudo venv/bin/python scripts/run_pilot.py --n 50                          # A (llama)
    sudo venv/bin/python scripts/run_pilot.py --n 50 --deployment b           # B (qwen)
    sudo venv/bin/python scripts/run_pilot.py --n 50 --model qwen2.5:7b  --out data/raw_amodel_sdk   # amodel (A-logic + qwen)
    sudo venv/bin/python scripts/run_pilot.py --n 50 --deployment b --model llama3.2:3b --out data/raw_blogic_sdk  # blogic
    sudo venv/bin/python scripts/run_pilot.py --n 50 --defense rate --out data/raw_defense_rate
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
from agents_b.executor_b import ExecutorB
from agents_b.orchestrator_b import OrchestratorB
from agents_b.retriever_b import RetrieverB
from agents_b.validator_b import ValidatorB
from agents_langgraph.orchestrator_langgraph import OrchestratorLangGraph
from capture.automation import TraceCollector
from capture.recorder import PacketRecorder
from workflows import WORKFLOW_REGISTRY, WorkflowClass
from workflows.base import TopologyType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_NUM_PREDICT = 256  # cap Ollama tokens per call; None = unlimited

# ── Deployment registry — the ONLY per-deployment differences ─────────────────
# Ports differ so A and B can run back-to-back without colliding.  qwen2.5:7b is
# deployment B's "native" model; pass --model to swap it (the model-vs-logic
# disentanglement runs A-logic+qwen and B-logic+llama via --model overrides).

DEPLOYMENTS: dict[str, dict] = {
    "a": {
        "ports": {"orchestrator": 8000, "executor": 8001, "retriever": 8002, "validator": 8003},
        "orchestrator": OrchestratorAgent,
        "executor": ExecutorAgent,
        "retriever": RetrieverAgent,
        "validator": ValidatorAgent,
        "default_model": "llama3.2:3b",
        "default_out": "data/raw",
    },
    "b": {
        "ports": {"orchestrator": 8010, "executor": 8011, "retriever": 8012, "validator": 8013},
        "orchestrator": OrchestratorB,
        "executor": ExecutorB,
        "retriever": RetrieverB,
        "validator": ValidatorB,
        "default_model": "qwen2.5:7b",
        "default_out": "data/raw_b",
    },
    # Deployment C — runtime-invariance control (LangGraph StateGraph orchestrator).
    # Reuses A's specialists, call structure, and prompts unchanged (label alignment);
    # only the orchestration RUNTIME differs — so it is a control, NOT an independent
    # framework.  Ports 8030-8033 so C runs back-to-back with A/B (8021 is occupied
    # by a system service on the primary testbed Mac).
    "langgraph": {
        "ports": {"orchestrator": 8030, "executor": 8031, "retriever": 8032, "validator": 8033},
        "orchestrator": OrchestratorLangGraph,
        "executor": ExecutorAgent,
        "retriever": RetrieverAgent,
        "validator": ValidatorAgent,
        "default_model": "llama3.2:3b",
        "default_out": "data/raw_langgraph",
    },
}

# ── Ground-truth topology edges (role → role) ─────────────────────────────────
# mesh: validator has no downstream (avoids the unbounded validator→executor
# cycle).  The mesh is still distinct via the orchestrator fan-out to executor
# AND retriever plus the executor→retriever cross-link, so retriever sees two
# inbound flows (the captured mesh signal).

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


def _url(role: str, ports: dict[str, int], host: str = "127.0.0.1") -> str:
    return f"http://{host}:{ports[role]}"


# ── Agent config builders ─────────────────────────────────────────────────────

def _agent_configs(dep: dict, topology: str, model: str,
                   n_retrieval_phases: int = 3,
                   num_predict: int | None = None,
                   defense: str = "none",
                   ollama_url: str = "http://localhost:11434") -> dict[str, AgentConfig]:
    """
    Per-role AgentConfig for executor/retriever/validator with downstream routing
    for the topology.  n_retrieval_phases controls deployment-A retriever depth
    (deployment-B agents ignore it — RetrieverB is fixed 2-phase).  defense selects
    the live C4 network defense.  ollama_url points agents at a specific Ollama
    instance (use a second instance to run A and B in parallel without contention).
    """
    ports = dep["ports"]

    def _cfg(role, port, downstream, phases=None):
        return AgentConfig(
            role=role, port=port, downstream_agents=downstream,
            ollama_model=model, ollama_num_predict=num_predict, defense=defense,
            ollama_base_url=ollama_url,
            **({"n_retrieval_phases": phases} if phases is not None else {}),
        )

    def _retriever_cfg(downstream):
        return _cfg(AgentRole.RETRIEVER, ports["retriever"], downstream,
                    phases=n_retrieval_phases)

    if topology == "star":
        # All three are leaves; orchestrator fans out to each directly.
        return {
            "executor":  _cfg(AgentRole.EXECUTOR,  ports["executor"],  []),
            "retriever": _retriever_cfg([]),
            "validator": _cfg(AgentRole.VALIDATOR, ports["validator"], []),
        }

    if topology in ("chain", "mesh"):
        # Subagent routing is identical for chain and mesh (executor→retriever→
        # validator); the mesh distinction is purely the orchestrator fan-out
        # (see _orchestrator), which gives retriever a second inbound flow.
        return {
            "executor":  _cfg(AgentRole.EXECUTOR,  ports["executor"],  [_url("retriever", ports)]),
            "retriever": _retriever_cfg([_url("validator", ports)]),
            "validator": _cfg(AgentRole.VALIDATOR, ports["validator"], []),
        }

    raise ValueError(f"Unknown topology: {topology}")


def _orchestrator(dep: dict, topology: str, model: str,
                  num_predict: int | None = None,
                  defense: str = "none",
                  ollama_url: str = "http://localhost:11434"):
    ports = dep["ports"]
    if topology == "star":
        downstream = [_url("executor", ports), _url("retriever", ports), _url("validator", ports)]
    elif topology == "chain":
        downstream = [_url("executor", ports)]
    elif topology == "mesh":
        downstream = [_url("executor", ports), _url("retriever", ports)]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    cfg = AgentConfig(
        role=AgentRole.ORCHESTRATOR,
        port=ports["orchestrator"],
        downstream_agents=downstream,
        ollama_model=model,
        ollama_num_predict=num_predict,
        defense=defense,
        ollama_base_url=ollama_url,
    )
    return dep["orchestrator"](cfg)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _kill_port(port: int) -> None:
    # Exclude the current PID: lsof matches any socket where the port appears as
    # LOCAL **or** REMOTE, so our own CLOSE_WAIT / TIME_WAIT sockets would
    # otherwise cause us to kill ourselves with SIGKILL.
    own_pid = os.getpid()
    proc = await asyncio.create_subprocess_shell(
        f"lsof -ti tcp:{port} 2>/dev/null | grep -v '^{own_pid}$' | xargs kill -9 2>/dev/null; true",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def _wait_ready(role: str, ports: dict[str, int], retries: int = 30) -> bool:
    url = _url(role, ports)
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
    dep: dict,
    deployment: str,
    topology: str,
    workflows: list[str],
    n: int,
    model: str,
    out_dir: Path,
    n_retrieval_phases: int = 3,
    num_predict: int | None = None,
    defense: str = "none",
    ollama_url: str = "http://localhost:11434",
    seed_offset: int = 0,
) -> dict[str, dict]:
    ports = dep["ports"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TOPOLOGY: %s  (deployment %s)", topology.upper(), deployment.upper())
    logger.info("=" * 60)

    # Free downstream agent ports
    for role in ("executor", "retriever", "validator"):
        await _kill_port(ports[role])
    await asyncio.sleep(0.6)

    # Start downstream agents — keep agent refs for graceful shutdown
    configs = _agent_configs(dep, topology, model, n_retrieval_phases=n_retrieval_phases,
                             num_predict=num_predict, defense=defense, ollama_url=ollama_url)
    agent_classes = {
        "executor": dep["executor"],
        "retriever": dep["retriever"],
        "validator": dep["validator"],
    }
    agents: list = []
    tasks: list[asyncio.Task] = []
    for role, AgentClass in agent_classes.items():
        agent = AgentClass(configs[role])
        agents.append(agent)
        tasks.append(asyncio.create_task(agent.run(), name=f"{deployment}-{topology}-{role}"))

    await asyncio.sleep(1.2)  # let uvicorn bind all three ports

    # Health check
    for role in ("executor", "retriever", "validator"):
        if not await _wait_ready(role, ports):
            logger.error("  %s agent never came up — aborting topology %s", role, topology)
            for agent in agents:
                await agent.shutdown()
            for t in tasks:
                t.cancel()
            return {}
    logger.info("  All downstream agents ready.")

    recorder = PacketRecorder(
        output_dir=out_dir,
        interface="lo0",
        agent_ports=list(ports.values()),
    )
    agent_endpoints = {role: f"127.0.0.1:{port}" for role, port in ports.items()}
    edges = TOPO_EDGES[topology]

    stats: dict[str, dict] = {}

    for wf_name in workflows:
        wf_class = WorkflowClass(wf_name)
        wf_instance = WORKFLOW_REGISTRY[wf_class]()
        # Same seed formula across deployments → identical prompts, so A vs B
        # differences are due to implementation, not input.  `seed_offset` (default 0 →
        # unchanged) lets a caller draw a DISJOINT prompt sample on repeated runs, e.g. a
        # temporal-interleaving confound control that must spread distinct prompts across time.
        _TOPOS = ["star", "chain", "mesh"]
        _WFS   = [wf.value for wf in WorkflowClass]
        _seed  = _TOPOS.index(topology) * len(_WFS) + _WFS.index(wf_name) + 100 + seed_offset
        prompts = wf_instance.sample_prompts(n=n, seed=_seed)

        # Build a fresh orchestrator for each workflow run (new HTTP client)
        orch = _orchestrator(dep, topology, model, num_predict=num_predict,
                             defense=defense, ollama_url=ollama_url)

        async def exec_fn(prompt: str, _orch=orch) -> None:
            tid = f"pilot_{deployment}_{uuid.uuid4().hex[:8]}"
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
            deployment=deployment,
        )

        logger.info(
            "  Collecting %d traces  workflow=%-20s topology=%s",
            n, wf_name, topology,
        )
        runs = await collector.collect(prompts, inter_run_delay=0.5)
        ok = sum(1 for r in runs if r.success)
        stats[f"{topology}/{wf_name}"] = {"total": n, "success": ok, "failed": n - ok}
        logger.info("    → %d/%d successful", ok, n)

    # Tear down downstream agents — graceful uvicorn exit, then force-kill ports.
    logger.info("  Tearing down %s agents...", topology)
    for agent in agents:
        await agent.shutdown()
    done, pending = await asyncio.wait(tasks, timeout=4.0)
    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except BaseException:
            pass
    for role in ("executor", "retriever", "validator"):
        await _kill_port(ports[role])
    await asyncio.sleep(3.0)
    logger.info("  Teardown complete.")

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        logger.warning(
            "Not running as root — tcpdump BPF capture may be empty. "
            "Re-run with: sudo venv/bin/python scripts/run_pilot.py"
        )

    dep = DEPLOYMENTS[args.deployment]
    model = args.model or dep["default_model"]
    out_dir = Path(args.out or dep["default_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    num_predict = args.num_predict if args.num_predict > 0 else None

    topologies = [args.topology] if args.topology else ["star", "chain", "mesh"]
    workflows  = [args.workflow] if args.workflow else [wf.value for wf in WorkflowClass]

    # Confirm the model is pulled on the target Ollama before a long run
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{args.ollama_url}/api/tags")
            if r.status_code == 200:
                have = [m["name"] for m in r.json().get("models", [])]
                if not any(model.split(":")[0] in m for m in have):
                    logger.warning("Model '%s' not found on %s. Pull it: ollama pull %s",
                                   model, args.ollama_url, model)
    except Exception:
        logger.warning("Ollama not reachable at %s — agents will fail.", args.ollama_url)

    logger.info(
        "Pilot config: deployment=%s  topologies=%s  workflows=%s  n=%d  model=%s  "
        "retriever_phases=%d  num_predict=%s  defense=%s  out=%s",
        args.deployment, topologies, workflows, args.n, model, args.retriever_phases,
        num_predict if num_predict else "unlimited", args.defense, out_dir,
    )

    all_stats: dict[str, dict] = {}
    for topology in topologies:
        stats = await run_topology(
            dep=dep,
            deployment=args.deployment,
            topology=topology,
            workflows=workflows,
            n=args.n,
            model=model,
            out_dir=out_dir,
            n_retrieval_phases=args.retriever_phases,
            seed_offset=args.seed_offset,
            num_predict=num_predict,
            defense=args.defense,
            ollama_url=args.ollama_url,
        )
        all_stats.update(stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  PILOT SUMMARY  (deployment {args.deployment.upper()})")
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
        print(f"  python scripts/extract_features.py --raw {out_dir} --out data/processed --scapy")
        print("  python scripts/evaluate.py --mode all --rf-only")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A2A pilot: collect traces for all topology × workflow pairs (deployment A or B)")
    p.add_argument("--deployment", choices=["a", "b", "langgraph"], default="a",
                   help="Agent deployment: a=agents/ (parallel, llama default), "
                        "b=agents_b/ (sequential, qwen default), "
                        "langgraph=agents_langgraph/ (LangGraph StateGraph orchestrator, "
                        "A's specialists, llama). Default: a.")
    p.add_argument("--topology", choices=["star", "chain", "mesh"],
                   help="Run one topology only (default: all three)")
    p.add_argument("--workflow",
                   choices=[wf.value for wf in WorkflowClass],
                   help="Run one workflow only (default: all four)")
    p.add_argument("--n", type=int, default=5,
                   help="Traces per (workflow, topology) pair (default: 5)")
    p.add_argument("--model", default=None,
                   help="Ollama model name (default: deployment's native model)")
    p.add_argument("--out", default=None,
                   help="Output dir for pcap + label files (default: deployment's default)")
    p.add_argument("--retriever-phases", type=int, default=3, choices=[1, 2, 3],
                   dest="retriever_phases",
                   help="Deployment-A retriever LLM phases: 1=direct QA, 2=decompose+synth, "
                        "3=decompose+retrieve+synth (default). Ignored by deployment B.")
    p.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
                   dest="num_predict",
                   help=f"Cap Ollama response tokens per call (default: {DEFAULT_NUM_PREDICT}). "
                        "Set 0 for unlimited. Lower = faster collection.")
    p.add_argument("--seed-offset", type=int, default=0, dest="seed_offset",
                   help="Added to the per-workflow prompt seed (default 0 → unchanged). Vary "
                        "across repeated runs to draw DISJOINT prompt samples — e.g. a temporal-"
                        "interleaving confound control that spreads distinct prompts across time.")
    p.add_argument("--defense", default="none",
                   choices=["none", "pad", "rate", "both"],
                   help="Live C4 network defense applied during collection: none; "
                        "pad=SSE cell-size padding (size defense); rate=dummy sub-calls "
                        "+ jittered/reordered delegation (count/rate defense); both. "
                        "Use a separate --out dir per defense.")
    p.add_argument("--ollama-url", dest="ollama_url", default="http://localhost:11434",
                   help="Ollama base URL the agents call (default http://localhost:11434). "
                        "Point deployment B at a SECOND Ollama instance (e.g. "
                        "http://127.0.0.1:11435) to run A and B in parallel without "
                        "LLM contention corrupting the timing features.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse()))
