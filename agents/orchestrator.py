"""
Orchestrator agent — the hub in a star topology, head of a chain, or one peer
in a mesh.  Receives the top-level task, decomposes it, and fans out sub-tasks
to downstream agents via A2A (SDK streaming client → SSE on the wire).
"""

from __future__ import annotations

import asyncio
import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.ORCHESTRATOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[orchestrator] received task %s", task_id)

        # Step 1: decompose the task (internal reasoning — blocking, not streamed)
        decomposition_prompt = (
            f"You are a task orchestrator. Break the following task into 2-3 "
            f"concise sub-tasks that can be delegated to specialist agents "
            f"(executor, retriever, validator). Return a numbered list only.\n\n"
            f"TASK: {content}"
        )
        subtask_plan = await self.llm_generate(decomposition_prompt)
        logger.debug("[orchestrator] plan:\n%s", subtask_plan)

        if not self.config.downstream_agents:
            logger.warning("[orchestrator] no downstream agents configured")
            return await self.llm_stream(decomposition_prompt, emit)

        # Step 2: fan out to each downstream agent concurrently.  Each call is an
        # SDK streaming request (message/stream), so SSE chunks flow on every
        # hop.  We do NOT relay these concurrent streams into our own emit (they
        # would interleave); the orchestrator streams only its final synthesis.
        delegations = [
            (
                agent_url,
                f"{task_id}_{i}",
                f"Sub-task {i+1} of {len(self.config.downstream_agents)}:\n"
                f"{subtask_plan}\n\n"
                f"FULL TASK CONTEXT (use as needed):\n{content}",
            )
            for i, agent_url in enumerate(self.config.downstream_agents)
        ]
        if self.config.defense in ("rate", "both"):
            # C4 rate/count defense: jittered + reordered dispatch + dummy calls.
            agent_results = await self.defended_fanout(delegations)
        else:
            agent_results = await asyncio.gather(
                *(self.send_task(u, t, c) for (u, t, c) in delegations),
                return_exceptions=True,
            )

        results: list[str] = []
        for i, r in enumerate(agent_results):
            if isinstance(r, Exception):
                logger.error("[orchestrator] downstream %d failed: %s", i, r)
                results.append(f"[agent {i} error: {r}]")
            else:
                results.append(r.output)

        # Step 3: synthesise results — final answer streams as SSE
        synthesis_prompt = (
            f"You are a task orchestrator. Synthesise the following agent results "
            f"into a final answer for the original task.\n\n"
            f"ORIGINAL TASK: {content}\n\n"
            + "\n\n".join(
                f"AGENT {i} OUTPUT:\n{res}" for i, res in enumerate(results)
            )
        )
        return await self.llm_stream(synthesis_prompt, emit)
