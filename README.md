# Passive Inference of Topology, Roles, and Workflows in LLM Multi-Agent Systems via Encrypted Traffic Analysis

A network traffic-analysis attack on LLM multi-agent systems communicating via the [A2A protocol](https://a2aproject.org). The system reconstructs **workflow class**, **agent roles**, and **system topology** from packet metadata alone — no payload inspection required.

---

## Overview

Modern LLM applications are increasingly built as multi-agent pipelines where specialized agents (orchestrators, retrievers, executors, validators) collaborate via structured protocols. This project demonstrates that an **external passive observer** — seeing only encrypted TCP metadata (packet sizes, inter-arrival times, flow directions) — can fingerprint what a multi-agent system is doing with high accuracy.

### Research Goals

| Goal | Task | Description |
|------|------|-------------|
| C1 | Topology reconstruction | Infer star / chain / mesh from traffic graph shape |
| C2 | Role inference | Classify each agent as orchestrator / executor / retriever / validator |
| C3 | Workflow fingerprinting | Identify the active workflow class (research, code review, data analysis, support triage) |
| C4 | Defense evaluation | Measure how much padding, scheduling randomization, and dummy traffic degrade attack accuracy |
| C5 | Cross-network generalization | Test whether models trained on LAN traffic generalize to real WAN conditions (US ↔ India) |

The attack is **metadata-only**: packet payloads are never captured, stored, or examined. The only signal is packet size, timestamp, and direction — the same information visible to any network-layer observer regardless of TLS.

---

## Threat Model

```
┌─────────────────────────────────────────────────┐
│  LLM Multi-Agent System (A2A over HTTPS/TLS)    │
│                                                  │
│  Orchestrator ──→ Retriever ──→ Executor        │
│       └──────────────────────→ Validator        │
└─────────────────────────────────────────────────┘
                      │
              Encrypted traffic
                      │
              ┌───────▼────────┐
              │ Passive Observer│  (tcpdump / tshark)
              │  sees only:     │
              │  - packet sizes │
              │  - timestamps   │
              │  - flow pairs   │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │ ML Classifier  │
              │ → workflow?    │
              │ → role?        │
              │ → topology?    │
              └────────────────┘
```

**Attacker capabilities:** passive tap on any network segment the A2A traffic traverses — ISP, enterprise router, VPN endpoint, or cloud egress. No decryption. No active injection.

**Defender capabilities:** any network-layer shaping applied to the TLS stream (padding, scheduling randomization, dummy interactions).

---

## Architecture

### A2A Agent System

Each agent is a standalone Starlette HTTP service speaking [JSON-RPC 2.0](https://www.jsonrpc.org/specification) with Server-Sent Events (SSE) streaming, implemented with the official [a2a-sdk](https://pypi.org/project/a2a-sdk/) from the Linux Foundation A2A project.

```
agents/
  orchestrator.py   — fans out tasks, aggregates results, decides workflow routing
  executor.py       — performs the primary LLM reasoning step
  retriever.py      — fetches relevant context (RAG-style)
  validator.py      — checks output quality and consistency
```

Agents run on ports 8000–8003. Each agent is backed by a local [Ollama](https://ollama.com) instance — inference never leaves the local machine, ensuring only A2A control/response traffic crosses the network.

### Topologies

```
star      orchestrator ←→ all three sub-agents in parallel
chain     orchestrator → executor → retriever → validator (sequential pipeline)
mesh      orchestrator + additional peer-to-peer cross-links between agents
```

### Workflow Classes

| ID | Class | Characteristic traffic pattern |
|----|-------|--------------------------------|
| `research_retrieval` | Question answering with retrieval | Long SSE streams from retriever; multi-hop delegation |
| `code_review` | Code submitted, reviewed, validated | Structured back-and-forth; validator always active |
| `data_analysis` | CSV/tabular data analysed | Large initial payload; aggregation-heavy executor response |
| `support_triage` | Ticket classified and escalated | Short initial ticket; single escalation hop |

---

## Feature Pipeline

### Capture (metadata only)

```
tcpdump -n -s 96 -i <iface> -w trace.pcap "tcp and port 8000-8003"
```

The `-s 96` snaplen captures only the IP/TCP headers — payload bytes are never written to disk.

### Feature Extraction (195-dim flat vector)

```
features/
  burst.py      — idle-gap burst segmentation (0.5s threshold)
  per_flow.py   — 35 features per TCP flow (sizes, timing, direction asymmetry, SSE chunk stats)
  per_system.py — 90 features aggregated across all flows (hop counts, parallelism, timing spread)
  extractor.py  — combines per-flow + per-system into TraceFeatures
```

**Feature vector layout:**

```
[0  :35 ]   per-flow mean        (mean over all A2A flows)
[35 :70 ]   per-flow top-1       (heaviest flow by total bytes)
[70 :105]   per-flow top-2       (2nd heaviest flow)
[105:195]   per-system           (90-dim system-wide aggregate)
```

Key per-flow features include packet size distribution, inter-arrival timing, directional asymmetry, burst count, and SSE chunk patterns. Per-system features capture hop counts, fan-out degree, temporal overlap between flows, and cumulative byte asymmetry.

### Models

| Model | File | Notes |
|-------|------|-------|
| Random Forest | `models/random_forest.py` | 300-tree ensemble on flat 195-dim vector; baseline |
| Gradient Boosted Trees | `models/gradient_boosted.py` | HistGradientBoostingClassifier; second baseline |
| 1-D CNN | `models/cnn1d.py` | Operates on burst sequence; learns local temporal patterns |
| Transformer | `models/transformer.py` | Lightweight attention over burst sequence; main model |

---

## Defenses

Two groups of defenses are implemented and evaluated:

**Website-fingerprinting classics** (adapted for A2A):
- `defense/padding.py` — packet padding to fixed or randomized bucket sizes
- `defense/scheduling.py` — randomized delegation scheduling to blur timing patterns

**A2A-specific**:
- `defense/dummy.py` — dummy agent interactions that mimic other workflow patterns

---

## Testbed

| Site | Machine | Role |
|------|---------|------|
| US | MacBook Pro M3 Max | Primary agents, analysis, local LLM (Metal) |
| India | Dell PowerEdge R730xd | Remote agents, WAN vantage, local LLM (CPU) |
| Secondary | ASUS Vivobook 15 i7 | Cross-network generalization (C5) |

Local LLMs run via Ollama at each site. Only A2A JSON-RPC traffic crosses the WAN — inference stays local.

---

## Installation

**Requirements:** Python 3.11+, tcpdump, Ollama (for real LLM mode)

```bash
git clone <repo>
cd <repo>

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Key dependencies:**

```
a2a-sdk>=0.2.0        # A2A protocol SDK (Linux Foundation)
ollama>=0.3.3          # Local LLM client
starlette + uvicorn    # Agent HTTP servers
scapy / pyshark        # Packet capture parsing
scikit-learn>=1.4      # RF + GBT baselines
torch>=2.2             # CNN + Transformer models
```

---

## Running the Pipeline

### Phase 0 — Verify A2A traffic is on the wire

Must pass before any data collection.

```bash
# Local loopback (no Ollama needed)
python scripts/run_poc.py --mode local

# With a real local LLM
python scripts/run_poc.py --mode local --use-llm --model llama3.2:3b

# Distributed (executor on remote host)
python scripts/run_poc.py --mode distributed --executor-host 192.168.1.100
```

Expected output: `OVERALL : PASS` with HTTP JSON-RPC and wire-level traffic verified.

### Data Collection

```bash
# Collect all 12 topology×workflow pairs (50 runs each = 600 traces)
sudo venv/bin/python scripts/run_pilot.py --n 50 --out data/raw 2>&1 | tee logs/pilot.log

# Or collect a single pair (agents must be pre-started)
python scripts/collect_traces.py --workflow research_retrieval --topology star --n 50
```

`sudo` is required only for tcpdump BPF capture; privilege is dropped immediately after the capture subprocess starts.

### Feature Extraction

```bash
# Use scapy (recommended; no tshark required)
python scripts/extract_features.py --raw data/raw --out data/processed --scapy

# Use tshark/pyshark (if installed)
python scripts/extract_features.py --raw data/raw --out data/processed
```

### Training

```bash
# Individual models
python scripts/train_models.py --task workflow --model rf
python scripts/train_models.py --task workflow --model gbt
python scripts/train_models.py --task workflow --model cnn --epochs 40

# All tasks (workflow / role / topology)
python scripts/train_models.py --task all --model all
```

### Evaluation

```bash
# Fast: RF baseline only
python scripts/evaluate.py --mode all --rf-only

# Full suite: RF + GBT + CNN (skip Transformer for CPU-only runs)
python scripts/evaluate.py --mode all --skip-cnn

# Complete suite including Transformer
python scripts/evaluate.py --mode all
```

Evaluation runs **closed-world** (held-out traces from seen classes) and **open-world** (unknown-class rejection at 5% FPR) automatically. Results are written to `data/results/`.

### Defense Evaluation

Measured on **real defended captures** (agents run with the defense active, traffic
captured on the wire) — not a feature-space simulation:

```bash
python scripts/evaluate_defense_live.py \
  --baseline data/processed              --baseline-raw data/raw \
  --rate     data/processed_defense_rate --rate-raw     data/raw_defense_rate \
  --pad      data/processed_defense_pad  --pad-raw      data/raw_defense_pad
```

---

## Evaluation Discipline

Results are reported against **random and majority-class baselines** to establish that performance is above chance. All evaluations report:

- Accuracy, F1-macro, Precision, Recall (closed-world)
- Unknown-rejection rate at 5% FPR (open-world)
- Per-class confusion matrix

Open-world evaluation is mandatory, not optional. Background classes include: (a) ordinary web/API traffic, (b) non-target agentic traffic.

---

## Data Directory (gitignored)

```
data/
  raw/          .pcap files named <workflow>_<topology>_<run_id>.pcap
  processed/    feature matrices (.npz) + labels (.json)
  models/       trained checkpoints (.pkl for RF/GBT, .pt for CNN/Transformer)
  traces/       metadata-only CSV traces (safe to share; no payload bytes)
  results/      evaluation JSON output files
```

Raw pcap files and trained models are gitignored. Only metadata-only CSV traces in `data/traces/` are safe to share publicly.

---

## Ethical Constraints

- **No payload capture.** tcpdump is invoked with `-s 96` (header-only snaplen). Payload bytes are never written to disk, stored, or examined.
- **No targeting of external systems.** All captures run on infrastructure owned and operated by the researchers.
- **Privilege minimization.** `sudo` is used only to launch tcpdump. The Python process itself runs without elevated privileges.
- **Responsible disclosure.** Defense evaluation (C4) is conducted before any public release of attack-accuracy results, so the community has mitigation guidance alongside the attack findings.

---

## Project Structure

```
agents/       A2A agent implementations (orchestrator, executor, retriever, validator)
workflows/    Workflow prompt generators (research_retrieval, code_review, data_analysis, support_triage)
capture/      Packet capture automation (tcpdump/tshark wrappers, trace labeler, automation driver)
features/     Feature extraction pipeline (burst segmentation, per-flow, per-system, flat vector)
models/       ML models: Random Forest, GBT, 1D-CNN, Transformer
defense/      Traffic-shaping defenses (padding, scheduling randomization, dummy interactions)
evaluation/   Evaluation scripts (closed-world, open-world, cross-network, metrics)
scripts/      Runnable entry points (PoC, collect, train, evaluate, ablation)
configs/      YAML configs (testbed hosts, topology definitions, workflow assignments)
data/         GITIGNORED — raw pcaps, features, model checkpoints
logs/         Collection and training logs
```

---

## Citation

If you use this code or the methodology in your research, please cite:

```bibtex
@misc{more2026a2afingerprinting,
  title   = {Passive Inference of Topology, Roles, and Workflows in LLM
             Multi-Agent Systems via Encrypted Traffic Analysis},
  author  = {More, Mayuresh},
  year    = {2026},
  note    = {GitHub repository}
}
```

---

## License

This project is released for academic research purposes. See [LICENSE](LICENSE) for details.

All experiments were conducted on researcher-owned infrastructure. This code must not be used to monitor or infer the activities of multi-agent systems without explicit authorization from the system owner.
