"""
Retriever agent — performs context/knowledge retrieval for a task.

In real deployments this agent maps to production RAG architectures:
  - 1-phase (direct QA): LlamaIndex SimpleRetriever, GPT-4 with no retrieval
  - 2-phase (decompose + synthesise): standard dense/BM25 retrieval pipelines
  - 3-phase (decompose → per-term retrieval → synthesise): LangChain FLARE,
    AutoGen's RetrieveAssistantAgent, HyDE-based pipelines

The testbed uses the local Ollama LLM to simulate retrieval steps (avoiding
external API calls that would add non-A2A traffic to captures).  The call
count is realistic — production multi-step retrievers routinely make 3+ LLM
calls per query for decomposition, retrieval, and synthesis.

The n_retrieval_phases attribute (set via AgentConfig) enables the
ablation experiment in scripts/ablation_retriever.py: shows that role
signal survives when retriever phases are reduced to 1 (proving the
classifier isn't entirely reliant on our specific phase-count choice).
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
        logger.info("[retriever] received task %s (phases=%d)", task_id,
                    self.config.n_retrieval_phases)
        n = self.config.n_retrieval_phases

        # ── Phase 1: query decomposition ──────────────────────────────────────
        # Present in 2-phase and 3-phase variants.
        # Mirrors query-decomposition steps in LangChain FLARE and AutoGen RAG.
        if n >= 2:
            terms = await self.llm_generate(
                f"You are a knowledge retriever. Extract 3-5 precise search terms "
                f"from this query. Return a numbered list only.\n\nQUERY: {content[:400]}"
            )

        # ── Phase 2: per-term fact retrieval ──────────────────────────────────
        # Present in 3-phase variant only.
        # Mirrors the per-document retrieval step in HyDE and FLARE pipelines.
        if n >= 3:
            retrieved = await self.llm_generate(
                f"You are a knowledge retriever. For each search term below, provide "
                f"2-3 relevant facts from your knowledge. Use 'Term N:' headers.\n\n"
                f"TERMS:\n{terms}\n\nORIGINAL QUERY: {content[:400]}"
            )

        # ── Final synthesis phase (always present) ────────────────────────────
        if n == 1:
            # Direct QA — single call, no decomposition
            report = await self.llm_generate(
                f"Answer this query thoroughly using your knowledge. "
                f"Cite specific facts and mechanisms.\n\nQUERY: {content[:600]}"
            )
        elif n == 2:
            # Decompose + synthesise (no separate per-term retrieval)
            report = await self.llm_generate(
                f"You are a knowledge retriever. Using these search terms and your "
                f"knowledge, produce a structured retrieval report with sections: "
                f"Summary, Key Findings, Gaps.\n\n"
                f"TERMS:\n{terms}\n\nQUERY: {content[:400]}"
            )
        else:
            # Full 3-phase: synthesise from retrieved facts
            report = await self.llm_generate(
                f"You are a knowledge retriever. Synthesise the following retrieved "
                f"facts into a structured retrieval report with sections: Summary, "
                f"Key Findings, Gaps.\n\nFACTS:\n{retrieved}"
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
