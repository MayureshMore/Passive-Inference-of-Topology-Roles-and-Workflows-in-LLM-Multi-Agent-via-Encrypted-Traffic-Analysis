"""
Research/retrieval workflow — question answering with document retrieval.
Characteristic traffic: orchestrator → retriever (long SSE stream of chunks)
followed by orchestrator → executor (reasoning) followed by validator check.

Questions are intentionally mixed: short bare questions (~200-400B) and long
questions with research background paragraphs (~1200-2500B).  The longer
questions overlap with CR/ST payload sizes so the classifier must rely on
structural delegation signals rather than payload size.
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


# Long questions with research background (~1200-2500B): overlap with CR/ST payload sizes
_LONG_QUESTIONS = [
    """\
Background: Recent benchmarks (MMLU, HumanEval, MATH) have shown rapid capability \
scaling across LLM generations, but critics argue these benchmarks are increasingly \
contaminated — models may have seen test items during training. A 2024 paper introduced \
a "benchmark contamination" detection method showing that several frontier models exhibit \
statistically significant memorization of popular benchmarks. Simultaneously, the \
BIG-Bench Hard collaboration found that chain-of-thought prompting dramatically changes \
relative model rankings compared to direct prompting, raising questions about what is \
actually being measured.

Given this context: How should the NLP evaluation community redesign its benchmarking \
infrastructure to reliably track genuine reasoning capabilities rather than benchmark \
memorization? What are the fundamental tensions between easy-to-compute standardized \
benchmarks and valid measurement of generalization, and what methodological reforms \
have the strongest empirical support?""",

    """\
Context: The 2022-2024 NIST Post-Quantum Cryptography standardization process selected \
CRYSTALS-Kyber (KEM) and CRYSTALS-Dilithium, FALCON, SPHINCS+ (signatures). Several \
concerns remain: (1) SIKE was completely broken in 2022 by a classical attack running in \
hours on a laptop; (2) lattice-based schemes depend on LWE/RLWE hardness assumptions less \
studied than RSA/ECC, and recent preprints have questioned concrete security parameters; \
(3) hybrid deployment (classical + PQC) introduces protocol complexity and downgrade attack \
surfaces; (4) performance: Kyber-768 public keys are 1,184 bytes vs 32 bytes for X25519.

Against this background: What is the current threat model for "harvest now, decrypt later" \
attacks, and how should organisations prioritise PQC migration across their infrastructure? \
What are the realistic risks if the lattice assumptions underpinning the NIST selections \
prove weaker than expected, and which migration strategies are most robust to that scenario?""",

    """\
Research Brief: The RLHF paradigm has become the dominant alignment technique for \
instruction-tuned LLMs, following InstructGPT (Ouyang et al., 2022). Key subsequent \
developments: Constitutional AI (Anthropic 2022) uses written principles and AI \
self-critique to reduce human labelling burden. RLAIF (Lee et al., 2023) replaces human \
feedback with AI-generated preferences, achieving comparable performance at much lower cost. \
DPO (Rafailov et al., 2023) shows RLHF with KL-constrained reward maximisation is \
equivalent to a supervised classification on preference pairs, eliminating the need for a \
separate reward model. SPIN (Chen et al., 2024) uses game-theoretic self-play without \
explicit preference labels. Active criticisms span all methods: Goodhart's Law concerns, \
reward hacking, constitutional principles are themselves value-laden, DPO's implicit reward \
model may be poorly calibrated, and all methods assume high-quality preference data.

Question: Across these alignment paradigms, what evidence exists that any of them produce \
genuinely aligned behaviour vs. surface-level compliance that evades evaluation? How should \
practitioners choose between RLHF, DPO, Constitutional AI, and RLAIF given their different \
cost/quality trade-offs, and what does the research say about generalisation of alignment \
to novel out-of-distribution scenarios?""",

    """\
Technical Background: Federated learning (FL) distributes training across devices that \
keep data locally, transmitting only gradient updates. Several attacks undermine privacy \
in practice: Gradient Inversion (Zhu et al. 2019; Geiping et al. 2020) reconstructs \
training data from gradients with high fidelity for small batch sizes. Membership Inference \
(Shokri et al. 2017; Carlini et al. 2022) can identify whether specific records appeared \
in a training batch with up to 80% success on healthcare tabular data. Byzantine-robust \
aggregation (Krum, coordinate-wise median) is defeated by adaptive attackers with as few as \
10% malicious clients. Differential Privacy (DP-SGD) provides formal guarantees but incurs \
3-8% accuracy losses at ε=8, with difficult privacy budget calibration under heterogeneous \
non-IID participation. Secure Aggregation adds 2-5× communication overhead and is \
vulnerable to dropout attacks.

Question: What privacy guarantees does federated learning actually provide in practice \
and under what conditions? For a health system considering FL for multi-site clinical \
prediction models, what is the honest risk assessment, and how do available defences — \
differential privacy, secure aggregation, gradient clipping — stack up against realistic \
adversary models with partial system access?""",
]


class ResearchRetrievalWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.RESEARCH_RETRIEVAL

    def generate_prompt(self) -> str:
        if random.random() < 0.40:
            # Long question with research background to overlap with CR/ST payload sizes
            return random.choice(_LONG_QUESTIONS)
        topic = random.choice(_TOPICS)
        template = random.choice(_QUESTION_TEMPLATES)
        return template.format(topic=topic)
