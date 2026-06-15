"""
Validator agent — reviews the output of other agents for correctness,
completeness, or policy compliance and returns a pass/fail verdict with
optional feedback.  Creates the distinctive back-and-forth traffic pattern
used to fingerprint orchestrator-validator loops.
"""

from __future__ import annotations

import logging

from .base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.VALIDATOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[validator] received task %s", task_id)

        prompt_review = (
            f"You are a strict quality validator. Reply in EXACTLY this format "
            f"(no other text):\n"
            f"VERDICT: PASS or FAIL\n"
            f"SCORE: X/10\n"
            f"REASON: one sentence max 20 words\n\n"
            f"CONTENT:\n{content[:600]}"
        )

        # Validator is terminal in all configured topologies (no downstream),
        # so its verdict streams back as SSE.  The FAIL-retry branch below only
        # runs if a downstream agent is configured.
        if not self.config.downstream_agents:
            return await self.llm_stream(prompt_review, emit)

        review = await self.llm_generate(prompt_review)
        logger.debug("[validator] verdict for %s:\n%s", task_id, review)

        if "FAIL" in review.upper():
            retry_url = self.config.downstream_agents[0]
            retry_result = await self.send_task(
                target_url=retry_url,
                task_id=f"{task_id}_retry",
                content=(
                    f"Your previous output failed validation. Validator feedback:\n"
                    f"{review}\n\nPlease revise and resubmit.\n\n"
                    f"Original content:\n{content}"
                ),
            )
            prompt_recheck = (
                f"You are a strict quality validator. This is a revised submission. "
                f"Review it and provide VERDICT, SCORE, ISSUES, FEEDBACK.\n\n"
                f"REVISED CONTENT:\n{retry_result.output}"
            )
            return await self.llm_stream(prompt_recheck, emit)

        return review
