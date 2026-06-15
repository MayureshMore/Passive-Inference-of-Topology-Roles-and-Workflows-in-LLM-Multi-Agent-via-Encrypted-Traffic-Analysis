"""
Deployment-B Executor — two-step: plan then execute.

Behavioral difference from deployment A (agents/executor.py):
  Deployment A: single LLM call — "carry out this instruction."
  Deployment B: two LLM calls — first decomposes the instruction into 2-3
                concrete ordered sub-steps (planning), then executes those
                steps to produce the result (execution).
  Realistic because: ReAct (Yao et al. 2022), AutoGen TaskComposer, and
  production code-generation agents (GitHub Copilot Workspace, Devin) routinely
  emit an explicit planning step before execution.  Two calls per invocation is
  the norm in frameworks that expose intermediate reasoning, e.g. LangChain
  structured-output chains and CrewAI's plan-then-act pattern.

On-wire consequence:
  Two Ollama roundtrips per executor invocation doubles the number of
  response-direction traffic bursts in this agent's flow compared to deployment
  A.  The first (plan) LLM response is typically shorter than the second
  (execution) response, producing a short-burst → long-burst pattern within the
  executor flow.  This is the primary timing/size difference the role classifier
  must bridge for A→B transfer.
"""

from __future__ import annotations

import logging

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class ExecutorB(BaseA2AAgent):
    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.EXECUTOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[executor_b] received task %s", task_id)

        # Step 1: planning — decompose into concrete ordered sub-steps (internal)
        plan = await self.llm_generate(
            f"You are a task planner. Break this instruction into 2-3 concrete, "
            f"ordered sub-steps. Return a numbered list only.\n\n"
            f"INSTRUCTION: {content[:400]}"
        )

        # Step 2: execution — carry out the plan and produce a concrete result
        exec_prompt = (
            f"You are a task executor. Carry out the following steps and produce "
            f"a complete, concrete result.\n\n"
            f"STEPS:\n{plan}\n\n"
            f"ORIGINAL INSTRUCTION: {content[:400]}"
        )

        if self.config.downstream_agents:
            result = await self.llm_generate(exec_prompt)
            next_url = self.config.downstream_agents[0]
            forwarded = await self.send_task(
                target_url=next_url,
                task_id=f"{task_id}_fwd",
                content=f"Executor output:\n{result}\n\nOriginal instruction: {content}",
                emit=emit,
            )
            return forwarded.output

        return await self.llm_stream(exec_prompt, emit)
