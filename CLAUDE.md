# A2A Fingerprinting Research Project

## What this project is
Passive traffic-analysis attack on LLM multi-agent systems communicating via the A2A protocol.
Goal: reconstruct topology (C1), agent roles (C2), workflow class (C3) from encrypted metadata alone.
Then design and evaluate network-layer defenses (C4) and test robustness under WAN conditions (C5).

## Directory layout
```
agents/       A2A agent implementations (orchestrator, executor, retriever, validator)
workflows/    Workflow definitions (research, code_review, data_analysis, support_triage)
capture/      Packet capture automation (tcpdump/tshark wrappers, labeler, automation driver)
features/     Feature extraction pipeline (burst segmentation, per-flow, per-system features)
models/       ML models: Random Forest baseline + Transformer main model
defense/      Traffic-shaping defenses (padding, scheduling randomization, dummy interactions)
evaluation/   Evaluation scripts (closed-world, open-world, cross-network, metrics)
scripts/      Runnable entry points (PoC, collect traces, train, evaluate)
configs/      Global YAML configs (testbed hosts, topology definitions, workflow assignments)
data/         GITIGNORED — raw pcaps, processed traces, trained model checkpoints
```

## Key design decisions
- **A2A SDK**: use `a2a-sdk` (PyPI, Linux Foundation a2aproject). Each agent is a separate
  Starlette HTTP service speaking JSON-RPC 2.0 + SSE. Do NOT use in-process frameworks
  (LangGraph, AutoGen) without the SDK wrapper — they produce no network traffic.
- **Local LLMs**: each site runs Ollama with a quantized model. Mac uses Metal; Dell uses CPU.
  Keep inference traffic local so only A2A flows cross the WAN.
- **Capture**: `tcpdump`/`tshark` at observer host. Record metadata only (packet size,
  timestamp, direction, host pair). Never log payload content.
- **Feature extraction**: session-state segmented bursts (not a single continuous sequence).
  Features: packet-size sequences, inter-arrival timing, burst stats, SSE-chunk patterns,
  cumulative representations, per-flow and per-system aggregates.
- **Models**: RF baseline (scikit-learn) + lightweight Transformer on time-windowed bursts
  (PyTorch). Temporal GNN only if C1 topology reconstruction proves viable. No GPU needed.
- **Defense two groups**: (1) website-fingerprinting classics (padding, batching, cover traffic);
  (2) A2A-specific (randomized delegation scheduling, dummy agent interactions).

## Testbed sites
| Site       | Machine                 | Role                                    |
|------------|-------------------------|-----------------------------------------|
| US         | MacBook Pro M3 Max      | Primary agents, analysis, local LLM     |
| India      | Dell PowerEdge R730xd   | Remote agents, WAN vantage, local LLM   |
| Secondary  | ASUS Vivobook 15 i7     | Cross-network generalization (C5)        |

## Phase 0 gate (do first)
Run `scripts/run_poc.py` to verify two agents exchange JSON-RPC over HTTP/SSE across hosts
and that packets appear on the wire via tcpdump. If no traffic is captured, the project cannot
proceed — investigate before writing any other infrastructure.

## Topologies implemented
- **star**: orchestrator ← all others; orchestrator fans out tasks
- **chain**: orchestrator → executor → retriever → validator (sequential pipeline)
- **mesh**: orchestrator + peer-to-peer cross-links between specialised agents

## Workflow classes (4 core)
1. `research_retrieval` — question-answering with web/doc retrieval
2. `code_review` — code submitted, reviewed, validated
3. `data_analysis` — CSV/data analysed and summarised
4. `support_triage` — support ticket classified and escalated

## Evaluation discipline
- Always report precision AND recall (not just accuracy) under open-world conditions.
- Closed-world first to confirm signal exists; open-world is mandatory.
- Background classes: (a) ordinary web/API traffic, (b) non-target agentic traffic.
- All results against random/majority-class baselines so performance above chance is clear.
- Topology reconstruction (C1) is a stretch goal — do not let it block the schedule.

## Running things
```bash
# Install deps
pip install -r requirements.txt

# Phase 0 PoC (two agents, local loopback or two hosts)
python scripts/run_poc.py --mode local

# Collect traces — preferred: run_pilot collects all topology×workflow pairs
sudo venv/bin/python scripts/run_pilot.py --n 15 --out data/raw 2>&1 | tee logs/pilot.log

# Or per-(workflow, topology) pair via collect_traces.py (agents must be pre-started)
python scripts/collect_traces.py --workflow research_retrieval --topology star --n 50

# Extract features (use --scapy if tshark is not installed)
python scripts/extract_features.py --raw data/raw --out data/processed --scapy

# Install deps (includes xgboost for GBT)
pip install -r requirements.txt

# Train models — --model choices: rf | gbt | cnn | transformer | all
python scripts/train_models.py --task workflow --model rf
python scripts/train_models.py --task workflow --model gbt
python scripts/train_models.py --task workflow --model cnn  --epochs 40

# Evaluate (ablations run automatically in --mode all)
# --rf-only         : RF baseline only (fast)
# --skip-cnn        : RF + GBT, skip CNN (for CPU-only runs)
# (no extra flags)  : RF + GBT + CNN + Transformer
python scripts/evaluate.py --mode all --rf-only
python scripts/evaluate.py --mode all                 # full suite (RF+GBT+CNN)

# ── 600-trace collection (50 per workflow×topology pair) ─────────────────
# 50 × 4 workflows × 3 topologies = 600 traces
sudo venv/bin/python scripts/run_pilot.py --n 50 --out data/raw 2>&1 | tee logs/pilot_600.log

# ── Phase 3: real background traffic for open-world evaluation ────────────
# Collect ~150 background traces (25 per category × 6 categories)
# Requires sudo for tcpdump; servers start automatically on ports 9100–9107.
sudo venv/bin/python scripts/collect_background.py --n 25 --out data/raw_background \
  --processed data/processed_background --scapy 2>&1 | tee logs/background.log

# Open-world evaluation with real background as unknowns
python scripts/evaluate_open_world_background.py
```

## Data directory (gitignored)
```
data/raw/                   raw .pcap files, named <workflow>_<topology>_<run_id>.pcap
data/raw_background/        background traffic .pcap files for Phase 3 open-world test
data/processed/             feature matrices as .npz, labels as .json
data/processed_background/  background feature matrices + labels_background.json
data/models/                trained model checkpoints (.pkl for RF, .pt for Transformer)
data/traces/                metadata-only CSV traces (safe to share; no payload)
```

## Critical constraints
- Never capture or store packet payload — metadata only.
- Never target systems you do not own.
- Run `sudo` only for tcpdump; drop privileges immediately after.
- The India link is a real WAN — budget for latency variance in timing features.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
