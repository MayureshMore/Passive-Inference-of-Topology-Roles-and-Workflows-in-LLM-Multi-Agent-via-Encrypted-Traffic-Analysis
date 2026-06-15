"""
End-to-end trace collection driver.

Orchestrates repeated workflow executions with randomised input variation,
automated packet capture, and ground-truth labeling.  This is the script
that must be run (with tcpdump sudo privileges) to build the dataset.

Usage:
    python -m capture.automation \
        --workflow research_retrieval \
        --topology star \
        --n 100 \
        --out data/raw
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Callable

from capture.labeler import TraceLabeler
from capture.recorder import PacketRecorder
from workflows.base import TopologyType, WorkflowClass, WorkflowRun

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Drives N workflow executions, capturing traffic for each run.

    Parameters
    ----------
    orchestrator_fn:
        Async callable that executes a workflow given a prompt string.
        Returns (success, error_str).  The caller is responsible for
        having the agents already running before calling `collect()`.
    recorder:
        PacketRecorder instance configured for this testbed.
    workflow_class:
        Label applied to all traces in this collection run.
    topology:
        Label applied to all traces.
    agent_endpoints:
        Dict mapping role → "host:port" for ground-truth metadata.
    topology_edges:
        List of [src, dst] role pairs for ground-truth graph.
    """

    def __init__(
        self,
        orchestrator_fn: Callable[[str], object],
        recorder: PacketRecorder,
        workflow_class: WorkflowClass,
        topology: TopologyType,
        agent_endpoints: dict[str, str],
        topology_edges: list[list[str]],
        output_dir: Path,
        deployment: str = "a",
    ) -> None:
        self.orchestrator_fn = orchestrator_fn
        self.recorder = recorder
        self.workflow_class = workflow_class
        self.topology = topology
        self.agent_endpoints = agent_endpoints
        self.topology_edges = topology_edges
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deployment = deployment

    async def collect(
        self,
        prompts: list[str],
        inter_run_delay: float = 2.0,
    ) -> list[WorkflowRun]:
        """
        Execute each prompt with capture.  Returns completed WorkflowRun records.
        Adds a jitter delay between runs to prevent timing correlation artifacts.
        """
        import random

        completed: list[WorkflowRun] = []
        total = len(prompts)

        for i, prompt in enumerate(prompts, 1):
            run_id = (
                f"{self.workflow_class.value}_{self.topology.value}_{uuid.uuid4().hex[:6]}"
            )
            logger.info("[%d/%d] run_id=%s", i, total, run_id)

            run = WorkflowRun(
                run_id=run_id,
                workflow_class=self.workflow_class,
                topology=self.topology,
                agent_endpoints=self.agent_endpoints,
                topology_edges=self.topology_edges,
                input_prompt=prompt,
                start_ts=time.time(),
                pcap_path=str(self.output_dir / f"{run_id}.pcap"),
                deployment=self.deployment,
            )

            try:
                async def _exec(p=prompt):
                    return await self.orchestrator_fn(p)

                await self.recorder.record_async(
                    run_id=run_id,
                    coro=_exec(),
                )
                run.success = True
            except Exception as exc:
                logger.error("run %s failed: %s", run_id, exc)
                run.error = str(exc)

            run.end_ts = time.time()
            TraceLabeler.write(run)
            completed.append(run)

            # Randomised inter-run jitter (±50 % of inter_run_delay)
            jitter = inter_run_delay * (0.5 + random.random())
            await asyncio.sleep(jitter)

        logger.info(
            "Collection complete: %d/%d successful",
            sum(1 for r in completed if r.success),
            total,
        )
        return completed
