"""
Deployment-B Orchestrator — sequential delegation, no synthesis step.

Behavioral differences from deployment A (agents/orchestrator.py):

  1. Sequential fan-out: calls each downstream agent one-at-a-time, waiting
     for each result before calling the next.  Deployment A fans out in
     parallel via asyncio.gather().
     Realistic because: LangChain SequentialChain, Haystack Pipeline, and
     many production automation frameworks delegate serially to simplify
     error propagation and avoid concurrent API billing.  Projects with tight
     token-per-minute rate limits also prefer sequential to predictable cost.

  2. Pipeline context passing: each agent receives the previous agent's output
     as its context, not the original broadcast task.  Deployment A sends the
     full task to every agent simultaneously.
     Realistic because: pipeline orchestration (each stage transforms the
     previous output) is the default pattern in data-processing workflows,
     Haystack Document Stores, and simple LangChain pipelines.

  3. No synthesis step: returns the final downstream agent's output directly.
     Deployment A calls a second LLM to synthesize N parallel results.
     Realistic because: when delegation is sequential and each agent refines
     the previous output, the last agent's response IS the final answer;
     an extra synthesis pass adds latency without new information.

On-wire consequence (important for the cross-deployment experiment):
  Sequential delegation produces one flow per agent that starts AFTER the
  previous flow ends — flow_start_spread is proportional to cumulative agent
  latency, not to fan-out degree.  In star and mesh topologies this makes
  deployment B look like deployment A's chain topology from the classifier's
  perspective.  The A→B transfer experiment will show whether the fingerprinting
  signal learned from parallel fan-out (deployment A) transfers to
  sequential execution (deployment B).
"""

from __future__ import annotations

import asyncio
import logging
import random

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class OrchestratorB(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.ORCHESTRATOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[orchestrator_b] received task %s", task_id)

        # Single LLM call: produce a brief execution plan for downstream agents
        plan = await self.llm_generate(
            f"You are a task orchestrator. Produce a brief execution plan "
            f"(2-3 numbered steps) for specialist agents to handle this task.\n\n"
            f"TASK: {content[:500]}"
        )

        if not self.config.downstream_agents:
            logger.warning("[orchestrator_b] no downstream agents configured")
            return plan

        # C4 rate/count defense: inject dummy sub-calls concurrently and add a
        # jittered delay before each real (sequential) delegation.  Pipeline
        # ordering is preserved (B is a refine-the-previous pipeline), but the
        # burst count and inter-burst timing are obfuscated.
        defended = self.config.defense in ("rate", "both")
        dummy_task = None
        if defended:
            from defense.dummy import DummyInteractionInjector

            async def _send(url: str, tid: str, c: str):
                try:
                    return await self.send_task(url, tid, c)
                except Exception as exc:  # noqa: BLE001
                    return exc

            injector = DummyInteractionInjector(
                dummy_pool=self.config.downstream_agents,
                n_per_round=2, payload_size_bytes=256, concurrent=True,
            )
            dummy_task = asyncio.create_task(injector.inject(_send))

        # Sequential pipeline: each agent receives the output of the previous
        context = f"Execution plan:\n{plan}\n\nOriginal task:\n{content}"
        for i, agent_url in enumerate(self.config.downstream_agents):
            if defended:
                await asyncio.sleep(0.05 + random.uniform(0, 0.4))
            response = await self.send_task(
                target_url=agent_url,
                task_id=f"{task_id}_s{i}",
                content=context,
            )
            context = response.output  # pipeline: each stage refines previous

        if dummy_task is not None:
            try:
                await dummy_task
            except Exception:  # noqa: BLE001
                pass

        return context  # final agent's output is the answer; no synthesis LLM call
