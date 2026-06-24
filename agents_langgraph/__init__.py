"""
Deployment C — LangGraph-orchestrated implementation of the deployment-A taxonomy.

Runtime-invariance CONTROL for deployment A: the **orchestration runtime** differs
(a LangGraph `StateGraph` in place of A's asyncio.gather), while A's logic, prompts,
call structure, and label space are held identical.  (Distinct from deployment B /
agents_b/, a hand-built variant that deliberately changes the call logic.)  C is a
control, NOT an independent framework, and does not support a generalization claim.

Therefore C reuses deployment A's specialist A2A servers **unchanged** —
re-exported here rather than re-implemented — so that role/workflow behavior,
the §5.1 overlapping-payload discipline, and the by-port role mapping are
byte-for-byte the same as A.  Only the orchestrator is new
(`OrchestratorLangGraph`).  Recreating the specialists would introduce
behavioral drift and break the "labels identical, only the runtime different"
contract that makes A→C a clean runtime-invariance control (not generalization).

Ports: orchestrator 8030, executor 8031, retriever 8032, validator 8033
(8021 is held by a system service on the primary testbed Mac).
Model: llama3.2:3b (same as A).
"""

from agents.executor import ExecutorAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent

from .orchestrator_langgraph import OrchestratorLangGraph

# Aliases that name the reused specialists as deployment C's, for symmetry with
# agents_b's ExecutorB / RetrieverB / ValidatorB at the call sites.
ExecutorC = ExecutorAgent
RetrieverC = RetrieverAgent
ValidatorC = ValidatorAgent

__all__ = [
    "OrchestratorLangGraph",
    "ExecutorAgent",
    "RetrieverAgent",
    "ValidatorAgent",
    "ExecutorC",
    "RetrieverC",
    "ValidatorC",
]
