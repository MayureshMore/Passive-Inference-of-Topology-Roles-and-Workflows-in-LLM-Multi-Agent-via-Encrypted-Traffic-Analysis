"""
Research/retrieval workflow — question answering with document retrieval.
Characteristic traffic: orchestrator → retriever (long SSE stream of chunks)
followed by orchestrator → executor (reasoning) followed by validator check.
"""

from __future__ import annotations

import random

from .base import BaseWorkflow, WorkflowClass

_TOPICS = [
    "quantum computing applications in cryptography",
    "impact of transformer architectures on NLP benchmarks",
    "climate change feedback loops in permafrost regions",
    "CRISPR-Cas9 off-target effects mitigation strategies",
    "Byzantine fault tolerance in distributed consensus protocols",
    "dark matter detection experiments and null results",
    "reinforcement learning from human feedback alignment techniques",
    "epidemiological modelling of zoonotic disease spillover",
    "post-quantum key exchange standards (NIST PQC)",
    "economic impacts of autonomous vehicle adoption",
    "neuroplasticity mechanisms after traumatic brain injury",
    "federated learning privacy guarantees under heterogeneous data",
    "large language model inference optimisation techniques",
    "mRNA vaccine platform applicability beyond COVID-19",
    "carbon capture storage scalability and cost projections",
]

_QUESTION_TEMPLATES = [
    "What are the current limitations and open problems in {topic}?",
    "Summarise the key findings and debates in recent literature on {topic}.",
    "Compare the leading approaches to {topic} and evaluate their trade-offs.",
    "What does the evidence say about {topic}? Cite specific mechanisms or studies.",
    "Explain {topic} to a knowledgeable non-specialist and identify gaps in understanding.",
]


class ResearchRetrievalWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.RESEARCH_RETRIEVAL

    def generate_prompt(self) -> str:
        topic = random.choice(_TOPICS)
        template = random.choice(_QUESTION_TEMPLATES)
        return template.format(topic=topic)
