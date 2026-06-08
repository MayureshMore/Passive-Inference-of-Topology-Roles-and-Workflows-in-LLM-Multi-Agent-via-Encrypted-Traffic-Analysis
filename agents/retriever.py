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

        # Three-phase retrieval simulation — produces 3 LLM round-trips
        # (vs executor=1, validator=1), creating a distinctive latency signature.

        # Phase 1: extract key search terms
        terms = await self.llm_generate(
            f"You are a knowledge retriever. Extract 3-5 precise search terms "
            f"from this query. Return a numbered list only.\n\nQUERY: {content[:400]}"
        )

        # Phase 2: retrieve relevant facts for each term
        retrieved = await self.llm_generate(
            f"You are a knowledge retriever. For each search term below, provide "
            f"2-3 relevant facts from your knowledge. Use 'Term N:' headers.\n\n"
            f"TERMS:\n{terms}\n\nORIGINAL QUERY: {content[:400]}"
        )

        # Phase 3: synthesise into a coherent retrieval report
        report = await self.llm_generate(
            f"You are a knowledge retriever. Synthesise the following retrieved "
            f"facts into a structured retrieval report with sections: Summary, "
            f"Key Findings (numbered), Gaps.\n\nFACTS:\n{retrieved}"
        )

        if self.config.downstream_agents:
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Retrieved context:\n{report}\n\nOriginal query: {content}",
            )
            return forwarded.output

        return report
