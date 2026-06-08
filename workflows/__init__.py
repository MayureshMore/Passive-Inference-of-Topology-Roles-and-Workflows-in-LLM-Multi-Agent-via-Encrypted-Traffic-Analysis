from .base import WorkflowClass, WorkflowRun
from .research_retrieval import ResearchRetrievalWorkflow
from .code_review import CodeReviewWorkflow
from .data_analysis import DataAnalysisWorkflow
from .support_triage import SupportTriageWorkflow

WORKFLOW_REGISTRY: dict[WorkflowClass, type] = {
    WorkflowClass.RESEARCH_RETRIEVAL: ResearchRetrievalWorkflow,
    WorkflowClass.CODE_REVIEW: CodeReviewWorkflow,
    WorkflowClass.DATA_ANALYSIS: DataAnalysisWorkflow,
    WorkflowClass.SUPPORT_TRIAGE: SupportTriageWorkflow,
}

__all__ = [
    "WorkflowClass",
    "WorkflowRun",
    "ResearchRetrievalWorkflow",
    "CodeReviewWorkflow",
    "DataAnalysisWorkflow",
    "SupportTriageWorkflow",
    "WORKFLOW_REGISTRY",
]
