"""
Orchestrator agent — the hub in a star topology, head of a chain, or one peer
in a mesh.  Receives the top-level task, decomposes it, and fans out sub-tasks
to downstream agents via A2A.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from .base import AgentConfig, AgentRole, BaseA2AAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.ORCHESTRATOR
        super().__init__(config)

    async def handle_task(self, task_id: str, content: str) -> str:
        logger.info("[orchestrator] received task %s", task_id)

        # Step 1: use the local LLM to decompose the task into sub-tasks
        decomposition_prompt = (
            f"You are a task orchestrator. Break the following task into 2-3 "
            f"concise sub-tasks that can be delegated to specialist agents "
            f"(executor, retriever, validator). Return a numbered list only.\n\n"
            f"TASK: {content}"
        )
        subtask_plan = await self.llm_generate(decomposition_prompt)
        logger.debug("[orchestrator] plan:\n%s", subtask_plan)

        # Step 2: fan out to each downstream agent concurrently
        if not self.config.downstream_agents:
            logger.warning("[orchestrator] no downstream agents configured")
            return subtask_plan

        results: list[str] = []
        coros = [
            self.send_task(
                target_url=agent_url,
                task_id=f"{task_id}_{i}",
                # Pass the full original task content to downstream agents.
                # In production systems (AutoGen, CrewAI), orchestrators broadcast
                # the full context so each specialist can use the parts it needs.
                # This also means data_analysis (large CSV) and code_review (large
                # code file) naturally produce larger A2A request payloads — a
                # realistic consequence of full-context routing, not a tuning choice.
                content=(
                    f"Sub-task {i+1} of {len(self.config.downstream_agents)}:\n"
                    f"{subtask_plan}\n\n"
                    f"FULL TASK CONTEXT (use as needed):\n{content}"
                ),
            )
            for i, agent_url in enumerate(self.config.downstream_agents)
        ]
        agent_results = await asyncio.gather(*coros, return_exceptions=True)

        for i, r in enumerate(agent_results):
            if isinstance(r, Exception):
                logger.error("[orchestrator] downstream %d failed: %s", i, r)
                results.append(f"[agent {i} error: {r}]")
            else:
                results.append(r.output)

        # Step 3: synthesise results with the local LLM
        synthesis_prompt = (
            f"You are a task orchestrator. Synthesise the following agent results "
            f"into a final answer for the original task.\n\n"
            f"ORIGINAL TASK: {content}\n\n"
            + "\n\n".join(
                f"AGENT {i} OUTPUT:\n{res}" for i, res in enumerate(results)
            )
        )
        final_answer = await self.llm_generate(synthesis_prompt)
        return final_answer
