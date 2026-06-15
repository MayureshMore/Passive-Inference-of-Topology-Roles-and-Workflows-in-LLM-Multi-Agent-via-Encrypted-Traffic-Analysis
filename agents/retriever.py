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
disentanglement experiment: shows that role signal is driven by agent logic
(phase/call count), not the underlying model.
"""

from __future__ import annotations

import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.RETRIEVER
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[retriever] received task %s (phases=%d)", task_id,
                    self.config.n_retrieval_phases)
        n = self.config.n_retrieval_phases
        forwarding = bool(self.config.downstream_agents)

        # ── Phase 1: query decomposition (2-/3-phase) — internal, blocking ────
        if n >= 2:
            terms = await self.llm_generate(
                f"You are a knowledge retriever. Extract 3-5 precise search terms "
                f"from this query. Return a numbered list only.\n\nQUERY: {content[:400]}"
            )

        # ── Phase 2: per-term fact retrieval (3-phase only) — internal ────────
        if n >= 3:
            retrieved = await self.llm_generate(
                
                f"You are a knowledge retriever. For each search term below, provide "
                f"2-3 relevant facts from your knowledge. Use 'Term N:' headers.\n\n"
                f"TERMS:\n{terms}\n\nORIGINAL QUERY: {content[:400]}"
            )

        # ── Final synthesis phase ─────────────────────────────────────────────
        # Stream this when we are a leaf; keep it blocking when forwarding (the
        # downstream stream is what we relay in that case).
        if n == 1:
            final_prompt = (
                f"Answer this query thoroughly using your knowledge. "
                f"Cite specific facts and mechanisms.\n\nQUERY: {content[:600]}"
            )
        elif n == 2:
            final_prompt = (
                f"You are a knowledge retriever. Using these search terms and your "
                f"knowledge, produce a structured retrieval report with sections: "
                f"Summary, Key Findings, Gaps.\n\n"
                f"TERMS:\n{terms}\n\nQUERY: {content[:400]}"
            )
        else:
            final_prompt = (
                f"You are a knowledge retriever. Synthesise the following retrieved "
                f"facts into a structured retrieval report with sections: Summary, "
                f"Key Findings, Gaps.\n\nFACTS:\n{retrieved}"
            )

        if forwarding:
            report = await self.llm_generate(final_prompt)
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Retrieved context:\n{report}\n\nOriginal query: {content}",
                emit=emit,
            )
            return forwarded.output

        return await self.llm_stream(final_prompt, emit)
