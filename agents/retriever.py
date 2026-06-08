"""
Retriever agent — performs context/knowledge retrieval for a task.
In real deployments this would query a vector store or search API; in the
testbed it uses the local LLM to simulate retrieval (avoiding external API
calls that would add non-A2A traffic to the captures).
"""

from __future__ import annotations

import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.RETRIEVER
        super().__init__(config)

    async def handle_task(self, task_id: str, content: str) -> str:
        logger.info("[retriever] received task %s", task_id)

        # Simulate retrieval: generate "retrieved" context with the local LLM.
        # Deliberately uses multiple short LLM calls to mimic chunk-level retrieval
        # and produce the burst pattern characteristic of retrieval workflows.
        prompt_extract = (
            f"You are a knowledge retriever. Extract the key information need "
            f"and list 3-5 relevant facts or context items that would help answer "
            f"the following query. Format as a numbered list.\n\nQUERY: {content}"
        )
        retrieved_chunks = await self.llm_generate(prompt_extract)

        prompt_rank = (
            f"You are a knowledge retriever. Given the following retrieved chunks, "
            f"rank them by relevance and return the top 3 with a one-sentence "
            f"summary each.\n\nCHUNKS:\n{retrieved_chunks}\n\nQUERY: {content}"
        )
        ranked_context = await self.llm_generate(prompt_rank)

        # Forward to downstream if configured (e.g. in a chain)
        if self.config.downstream_agents:
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Retrieved context:\n{ranked_context}\n\nOriginal query: {content}",
            )
            return forwarded.output

        return ranked_context
