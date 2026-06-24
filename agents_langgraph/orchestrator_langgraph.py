"""
Deployment-C Orchestrator — LangGraph-orchestrated implementation of the A taxonomy.

The orchestration RUNTIME differs only: call count, ordering, and parallel fan-out
are produced by a **LangGraph `StateGraph`** (Pregel supersteps + a reducer channel),
NOT by hand-rolled `asyncio.gather` (deployment A).  Everything that defines the
*labels* — and the call structure itself — is held identical to A:

  * specialists are A's exact A2A servers (agents/executor|retriever|validator),
    reused unchanged, so role/workflow behavior and the §5.1 payload-size
    discipline carry over verbatim;
  * task prompts come from the shared `WORKFLOW_REGISTRY` with the same seed
    formula run_pilot uses for every deployment → identical inputs;
  * topology is realized by the same downstream wiring as A (star = fan out to
    3 specialists; chain = call executor, specialists forward; mesh = fan out to
    2 + specialist forwarding).

Only the *engine that schedules the delegation* differs.  A→C is therefore a
runtime-invariance CONTROL — it tests whether the traffic fingerprint survives a
swap of the orchestration runtime alone.  It is NOT a cross-framework
generalization result (C is not an independently-structured framework like B).

Model llama3.2:3b, ports 8030-8033.  Logical flow mirrors agents/orchestrator
(decompose → concurrent fan-out → synthesize); the on-wire difference is whatever
LangGraph's scheduler produces.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

logger = logging.getLogger(__name__)


class _OrchState(TypedDict):
    """Shared graph state.  `results` uses an additive reducer so the parallel
    fan-out branches can each append their specialist's output without the
    concurrent-write error LangGraph raises for un-reduced shared keys."""
    task_id: str
    content: str
    plan: str
    results: Annotated[list[str], operator.add]
    synthesis: str


class OrchestratorLangGraph(BaseA2AAgent):
    """A2A orchestrator whose delegation is driven by a LangGraph StateGraph.

    Like deployments A and B, the orchestrator is driven in-process by the
    collector (its `handle_task` is called directly; it is not served as an HTTP
    agent).  The captured A2A traffic is the `send_task` fan-out to the specialist
    servers; `decompose`/`synthesize` are local Ollama calls (off the captured
    wire), exactly as in deployment A.
    """

    def __init__(self, config: AgentConfig) -> None:
        config.role = AgentRole.ORCHESTRATOR
        super().__init__(config)

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        logger.info("[orchestrator_langgraph] received task %s", task_id)

        downstream = list(self.config.downstream_agents)
        if not downstream:
            logger.warning("[orchestrator_langgraph] no downstream agents configured")
            return await self.llm_stream(
                "You are a task orchestrator. Answer the following task.\n\n"
                f"TASK: {content}",
                emit,
            )

        graph = self._build_graph(downstream)
        final = await graph.ainvoke(
            {"task_id": task_id, "content": content, "plan": "", "results": [], "synthesis": ""}
        )
        return final.get("synthesis") or "\n\n".join(final.get("results", []))

    # ── LangGraph assembly ────────────────────────────────────────────────────

    def _build_graph(self, downstream: list[str]):
        """decompose → (parallel call_0..call_n) → synthesize.

        The fan-out edges from `decompose` to each `call_i` make LangGraph run the
        specialist calls concurrently within one superstep; the `call_i →
        synthesize` edges form the fan-in barrier.  This is the LangGraph-native
        equivalent of deployment A's asyncio.gather fan-out + synthesis."""
        sg = StateGraph(_OrchState)

        async def decompose(state: _OrchState) -> dict:
            plan = await self.llm_generate(
                "You are a task orchestrator. Break the following task into 2-3 "
                "concise sub-tasks that can be delegated to specialist agents "
                "(executor, retriever, validator). Return a numbered list only.\n\n"
                f"TASK: {state['content']}"
            )
            return {"plan": plan}

        async def synthesize(state: _OrchState) -> dict:
            synthesis_prompt = (
                "You are a task orchestrator. Synthesise the following agent "
                "results into a final answer for the original task.\n\n"
                f"ORIGINAL TASK: {state['content']}\n\n"
                + "\n\n".join(
                    f"AGENT {i} OUTPUT:\n{res}" for i, res in enumerate(state["results"])
                )
            )
            final = await self.llm_stream(synthesis_prompt, None)
            return {"synthesis": final}

        sg.add_node("decompose", decompose)
        sg.add_node("synthesize", synthesize)
        sg.add_edge(START, "decompose")

        n = len(downstream)
        for i, url in enumerate(downstream):
            sg.add_node(f"call_{i}", self._make_call(i, url, n))
            sg.add_edge("decompose", f"call_{i}")   # fan-out (parallel superstep)
            sg.add_edge(f"call_{i}", "synthesize")  # fan-in barrier

        sg.add_edge("synthesize", END)
        return sg.compile()

    def _make_call(self, idx: int, url: str, total: int):
        """Build a graph node that delegates sub-task `idx` to one specialist via
        a real A2A streaming call (this is the traffic the attack observes)."""

        async def _call(state: _OrchState) -> dict:
            task_id = state["task_id"]
            message = (
                f"Sub-task {idx + 1} of {total}:\n{state['plan']}\n\n"
                f"FULL TASK CONTEXT (use as needed):\n{state['content']}"
            )
            try:
                resp = await self.send_task(url, f"{task_id}_{idx}", message)
                out = resp.output
            except Exception as exc:  # noqa: BLE001
                logger.error("[orchestrator_langgraph] downstream %d failed: %s", idx, exc)
                out = f"[agent {idx} error: {exc}]"
            return {"results": [out]}

        return _call
