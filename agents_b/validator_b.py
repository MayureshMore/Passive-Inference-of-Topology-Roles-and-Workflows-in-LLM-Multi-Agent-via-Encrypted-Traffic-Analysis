"""
Deployment-B Validator — single-shot review, no conditional retry.

Behavioral difference from deployment A (agents/validator.py):
  Deployment A: conditional retry — if verdict is FAIL and a downstream retry
                agent is configured, sends feedback and re-validates the revised
                output (up to _MAX_REVIEW_ROUNDS=2, so 1 or 2 LLM calls total).
  Deployment B: unconditional single-shot — always returns after exactly one
                LLM call regardless of verdict.
  Realistic because: production CI/CD quality gates (GitHub Actions checks,
  automated lint/type-check bots, Semgrep, compliance scanners) are single-pass
  by design.  Retry loops with LLM feedback are an advanced pattern used in
  agentic code-generation systems (Reflexion, self-refine) but are NOT the
  default in most deployed validators.  Single-shot validators are used in
  CrewAI's simple Quality Assurance Agent and LangChain's basic QA evaluators.

On-wire consequence:
  Deployment B's validator flow always has exactly 1 response-direction burst
  (one Ollama roundtrip).  Deployment A's validator has 1 or 2 bursts depending
  on the FAIL branch (stochastic, driven by LLM output).  This reduces variance
  in the role classifier's flow-level features for deployment B's validator, and
  removes the conditional second flow that was a distinctive deployment-A pattern.
"""

from __future__ import annotations

import logging

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class ValidatorB(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.VALIDATOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[validator_b] received task %s", task_id)

        review_prompt = (
            f"You are a strict quality validator. Reply in EXACTLY this format "
            f"(no other text):\n"
            f"VERDICT: PASS or FAIL\n"
            f"SCORE: X/10\n"
            f"REASON: one sentence max 20 words\n\n"
            f"CONTENT:\n{content[:600]}"
        )
        # Single-shot: stream the verdict back as SSE; no conditional retry.
        return await self.llm_stream(review_prompt, emit)
