"""
Deployment-B Retriever — two-phase: decompose + synthesize.

Behavioral difference from deployment A (agents/retriever.py):
  Deployment A: 3-phase (decompose → per-term retrieval → synthesize), 3 LLM calls.
  Deployment B: 2-phase (decompose → synthesize), 2 LLM calls.  The per-term
                retrieval pass is omitted.
  Realistic because: simple dense-retrieval pipelines — LlamaIndex BasicQueryEngine,
  Haystack basic RAG, standard BM25 pipelines, and RAG-Fusion without reranking —
  use 2-phase decompose+synthesize without spawning per-term sub-queries.  This is
  the most common RAG pattern in production systems with latency budgets, and is
  explicitly contrasted with multi-step retrieval in the RAG survey
  (Gao et al. 2024, §3.2 "Iterative Retrieval").

On-wire consequence:
  One fewer Ollama roundtrip per retriever invocation reduces the number of
  response-direction bursts in this agent's flow by ~33% compared to deployment
  A's 3-phase retriever (2 bursts vs 3 bursts).  This is the primary timing
  difference used by the role classifier to distinguish retriever from executor;
  whether this difference persists across deployments is what the A→B transfer
  experiment measures.
"""

from __future__ import annotations

import logging

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class RetrieverB(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.RETRIEVER
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[retriever_b] received task %s", task_id)

        # Phase 1: query decomposition — extract search terms (internal)
        terms = await self.llm_generate(
            f"You are a knowledge retriever. Extract 3-5 precise search terms "
            f"from this query. Return a numbered list only.\n\nQUERY: {content[:400]}"
        )

        # Phase 2: direct synthesis — no per-term retrieval step
        report_prompt = (
            f"You are a knowledge retriever. Using these search terms and your "
            f"knowledge, produce a structured retrieval report with sections: "
            f"Summary, Key Findings, Gaps.\n\n"
            f"TERMS:\n{terms}\n\nORIGINAL QUERY: {content[:400]}"
        )

        if self.config.downstream_agents:
            report = await self.llm_generate(report_prompt)
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Retrieved context:\n{report}\n\nOriginal query: {content}",
                emit=emit,
            )
            return forwarded.output

        return await self.llm_stream(report_prompt, emit)
