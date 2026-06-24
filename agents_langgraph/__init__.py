"""
Deployment C — LangGraph-orchestrated implementation of the deployment-A taxonomy.

Cross-framework counterpart to deployment B (agents_b/, a hand-built variant).
The scientific point of C is that the **orchestration engine** is independent
(a LangGraph `StateGraph`), while the *label space* is held identical to A.

Therefore C reuses deployment A's specialist A2A servers **unchanged** —
re-exported here rather than re-implemented — so that role/workflow behavior,
the §5.1 overlapping-payload discipline, and the by-port role mapping are
byte-for-byte the same as A.  Only the orchestrator is new
(`OrchestratorLangGraph`).  Recreating the specialists would introduce
behavioral drift and break the "labels identical, implementation different"
contract that makes A→C a clean transfer experiment.

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
