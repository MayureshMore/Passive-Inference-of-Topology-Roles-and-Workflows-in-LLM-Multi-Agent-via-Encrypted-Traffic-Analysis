"""
Executor agent — performs the main computational/reasoning work of a sub-task.
In a star topology this is a leaf (streams its result back); in a chain it
forwards results to the next agent and relays that stream onward.
"""

from __future__ import annotations

import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.EXECUTOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[executor] received task %s", task_id)

        prompt = (
            f"You are a task executor. Carry out the following instruction and "
            f"produce a concrete result. Be thorough but concise.\n\n"
            f"INSTRUCTION: {content}"
        )

        # Chain: do the work internally (blocking), forward to the next agent,
        # and relay the downstream SSE stream back to our caller.
        if self.config.downstream_agents:
            result = await self.llm_generate(prompt)
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Previous executor output:\n{result}\n\nOriginal instruction: {content}",
                emit=emit,
            )
            return forwarded.output

        # Leaf: stream our own generation back as SSE.
        return await self.llm_stream(prompt, emit)
