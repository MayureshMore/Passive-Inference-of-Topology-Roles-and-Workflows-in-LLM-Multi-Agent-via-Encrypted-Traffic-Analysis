"""
Base workflow types and the WorkflowRun record that is written alongside each
packet capture for ground-truth labeling.
"""

from __future__ import annotations

import enum
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class WorkflowClass(str, enum.Enum):
    RESEARCH_RETRIEVAL = "research_retrieval"
    CODE_REVIEW = "code_review"
    DATA_ANALYSIS = "data_analysis"
    SUPPORT_TRIAGE = "support_triage"


class TopologyType(str, enum.Enum):
    STAR = "star"
    CHAIN = "chain"
    MESH = "mesh"


class WorkflowRun(BaseModel):
    """Ground-truth record written as JSON alongside every pcap."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_class: WorkflowClass
    topology: TopologyType
    # Participating agents: role → "host:port"
    agent_endpoints: dict[str, str] = {}
    # Directed edges in the interaction graph: list of [src_role, dst_role]
    topology_edges: list[list[str]] = []
    input_prompt: str = ""
    start_ts: float = Field(default_factory=time.time)
    end_ts: float = 0.0
    pcap_path: str = ""
    success: bool = False
    error: str = ""

    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


class BaseWorkflow:
    """
    Abstract workflow.  Sub-classes set `workflow_class` and implement
    `generate_prompt()` to produce varied task inputs for data collection.
    """

    workflow_class: WorkflowClass

    def generate_prompt(self) -> str:
        raise NotImplementedError

    def sample_prompts(self, n: int = 10) -> list[str]:
        return [self.generate_prompt() for _ in range(n)]
