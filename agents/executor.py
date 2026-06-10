"""
Executor agent — performs the main computational/reasoning work of a sub-task.
In a star topology this is a leaf; in a chain it passes results forward.
"""

from __future__ import annotations

import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.EXECUTOR
        super().__init__(config)

    async def handle_task(self, task_id: str, content: str) -> str:
        logger.info("[executor] received task %s", task_id)

        prompt = (
            f"You are a task executor. Carry out the following instruction and "
            f"produce a concrete result. Be thorough but concise.\n\n"
            f"INSTRUCTION: {content}"
        )
        result = await self.llm_generate(prompt)

        # In a chain topology, pass the result to the next agent if configured
        if self.config.downstream_agents:
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Previous executor output:\n{result}\n\nOriginal instruction: {content}",
            )
            return forwarded.output

        return result