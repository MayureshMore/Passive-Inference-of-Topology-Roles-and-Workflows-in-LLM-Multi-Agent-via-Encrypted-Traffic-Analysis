# Passive Inference of Topology, Roles, and Workflows in LLM Multi-Agent Systems via Encrypted Traffic Analysis

A network traffic-analysis attack on LLM multi-agent systems communicating via the [A2A protocol](https://a2aproject.org). The system reconstructs **workflow class**, **agent roles**, and **system topology** from packet metadata alone — no payload inspection required.

🚀 **New here? [START_HERE.md](START_HERE.md) is the one-command-per-step quickstart** (install → run → capture → extract → train → evaluate → reproduce).

📊 **Headline results, with scope and caveats, are in [RESULTS.md](RESULTS.md).**

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

### Deployment variants (disentanglement)

To separate *what leaks* from *how a particular system happens to be built*, the same task
taxonomy — identical workflows, prompts, and label space — is implemented several ways. A
classifier trained on one deployment is tested on another; whichever change breaks the
fingerprint tells us what the signal actually depends on.

| Deployment | Where | What differs from A | Isolates |
|------------|-------|---------------------|----------|
| **A** (headline) | `agents/` | — (llama3.2:3b, `asyncio.gather` orchestrator) | baseline |
| **B** | `agents_b/` | different model **and** call logic (qwen2.5:7b, sequential) | model + logic together |
| **A-model** | `data/processed_amodel_*` | A's logic with B's model | model only |
| **B-logic** | `data/processed_blogic_*` | B's logic with A's model | logic only |
| **C / LangGraph** | `agents_langgraph/` | A's specialists, prompts, and call structure unchanged — only the **orchestration runtime** swapped to a LangGraph `StateGraph` | runtime only |

This makes the central finding precise: the fingerprint is **invariant to the model and to
the orchestration runtime** (A→C transfers near A's within-deployment ceiling), but
**sensitive to the inter-agent call structure** (A→B collapses). Deployment **C is a
runtime-invariance control, not a cross-framework generalization claim** — generalization
across independently-structured frameworks (AutoGen, CrewAI, …) remains future work. A
companion diagnostic ([`scripts/diagnose_runtime_traffic.py`](scripts/diagnose_runtime_traffic.py))
confirms A and C emit near-identical traffic **structure** and differ only in **timing**.

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
| Gradient Boosted Trees | `models/gradient_boosted.py` | HistGradientBoostingClassifier on the flat 195-dim vector; **primary headline attacker** (marginal-best) |
| Random Forest | `models/random_forest.py` | 300-tree ensemble; headline tree attacker, statistically equivalent to GBT (overlapping CIs) |
| 1-D CNN | `models/cnn1d.py` | Operates on the burst sequence; data-starved at the current ~600-trace scale — reported as a scale-check, not a headline |
| Transformer | `models/transformer.py` | Gap-aware attention over the burst sequence; data-starved at current scale (learns role, where it has ~1,750 samples) — scale-check only |

The two tree models are the headline attackers; the two deep sequence models are retained as a data-scale sensitivity check and would need ~1,500–2,000 traces per class to compete. Their full architectures, burst-sequence input representation, parameter counts, and training budget — documented to pre-empt the "untuned models" critique — are in [docs/DEEP_MODEL_APPENDIX.md](docs/DEEP_MODEL_APPENDIX.md).

---

## Defenses

Two groups of defenses are implemented and evaluated **live on defended captures** (the defense runs inside the agents and the resulting traffic is captured on the wire — not a feature-space simulation):

**Group 1 — website-fingerprinting classics** (adapted to the A2A layer):
- SSE-cell size padding — each SSE event is padded up to the next fixed cell multiple at the agent (`agents/base.py`, `_cell_pad_len`)

**Group 2 — A2A-specific** (no analogue in the website-fingerprinting setting):
- `defense/scheduling.py` — randomized, jittered, reordered delegation scheduling to blur inter-agent timing and structure
- `defense/dummy.py` — injected dummy agent interactions (spurious sub-calls) that obscure genuine collaboration

---

## Off-the-shelf system (external corroboration)

Beyond the researcher-built deployments, one **externally-authored** multi-agent system —
Google's [`a2a_mcp`](https://github.com/a2aproject/a2a-samples) travel-planning sample (an
MCP agent registry + orchestrator + LangGraph planner + ADK air/hotel/car specialists, all
talking A2A) — is captured to test the attack on a system **we did not build**. Its labels
do not align with our taxonomy, so it corroborates **detection** and **topology
observability** only — **not** a role/workflow transfer number.

```
scripts/collect_offtheshelf.sh          drive the orchestrator over A2A + capture (ports 10100–10105)
scripts/extract_offtheshelf.py          features (reuses the core extractor; IPv4 AND IPv6 — the sample binds ::1)
scripts/evaluate_offtheshelf_detection.py   run the A-trained open-world detector on it
scripts/analyze_offtheshelf_topology.py     recover the agent connection graph from flow headers alone
```

The first capture's topology recovers cleanly as **hub-and-spoke** — the MCP registry is the
hub (every agent queries it) with the specialists as leaves — from headers only, no payload,
no ML. The external setup, the compatibility patches, and the multi-turn driver are recorded
under [`third_party/a2a_mcp/`](third_party/a2a_mcp/) and
[docs/PHASE5_A2A_MCP_PATCHES.md](docs/PHASE5_A2A_MCP_PATCHES.md). Capture is Gemini
free-tier quota-limited (≈2 traces/run); accumulate over daily runs.

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
a2a-sdk[http-server]==0.3.26   # A2A SDK (Linux Foundation); pinned — 1.x is gRPC-only, unsuitable
ollama>=0.3.3          # Local LLM client
starlette + uvicorn    # Agent HTTP servers
scapy / pyshark        # Packet capture parsing
scikit-learn>=1.4      # RF + GBT baselines
torch>=2.2             # CNN + Transformer models
```

---

## Reproducing the paper

All tables and figures regenerate **deterministically** from the published feature
matrices — no GPU, no network, no Ollama, no testbed; minutes, not hours. (Raw
*collection* is stochastic by nature — a fresh collection yields a different
dataset with point estimates inside the published CIs, not the identical numbers;
see [DATA.md](DATA.md) and the [C5 runbook](docs/C5_WAN_RUNBOOK.md) for that.)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download + unpack the published feature archive into data/  (see DATA.md)
#   archive: <ARCHIVE_URL — filled in after the Zenodo/figshare/OSF upload>
bash scripts/reproduce.sh
```

`scripts/reproduce.sh` is a short demonstration run: it re-analyses the frozen
features into a **sandbox** (`data/results_demo/`) and prints a side-by-side check
against the committed canonical results — it **never overwrites `data/results/`**.
RF/GBT/transfer point estimates use a fixed seed, so they reproduce the committed
numbers exactly. (Deep CNN/Transformer models are stochastic, footnote-only, and
off by default; add `--full-suite` to include them.)

To re-run the full suite into a **fresh copy** instead of the sandbox — the
canonical `data/results/` is still never touched — redirect the output dir:

```bash
A2A_RESULTS_DIR=data/results_$(date +%F) bash scripts/reproduce.sh --full-suite
```

> The committed `data/results/` is the frozen, canonical paper output. No normal
> run overwrites or deletes it; every re-run writes to a sandbox or a fresh copy.

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

`sudo` is required for tcpdump's raw-socket/BPF access. Capture is scoped to the target agent host:port flows by an explicit BPF filter and to a 96-byte snaplen, so only header metadata of the agents under test is recorded — never payload, never unrelated hosts. (Alternatively, grant BPF read access once with `chmod o+r /dev/bpf*` to run the collector without `sudo`.)

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

### Cross-deployment & cross-framework

```bash
# A vs B (different model + logic) + the model-only / logic-only disentanglement
python scripts/evaluate_cross_deployment.py --dir-a data/processed --dir-b data/processed_b_sdk
python scripts/evaluate_model_vs_logic.py

# A vs C (LangGraph runtime swap) — runtime-invariance CONTROL (relabels the B-slot as C,
# suppresses the A↔B "generalizes" verdict, which is false for a control)
python scripts/evaluate_cross_deployment.py --dir-a data/processed \
    --dir-b data/processed_langgraph --label-b C --control \
    --out data/results/cross_framework.json
python scripts/diagnose_runtime_traffic.py        # A-vs-C structure-vs-timing diagnostic
```

### Cross-Network Evaluation (C5 — US ⇄ India WAN)

The full step-by-step WAN procedure (both hosts, capture vantage, gates, collection, evaluation, troubleshooting) lives in [docs/C5_WAN_RUNBOOK.md](docs/C5_WAN_RUNBOOK.md). In short — serve the specialist agents on the remote (India) host, drive the orchestrator from the local (US) host so A2A crosses the VPN, and capture **post-decapsulation on the tunnel interface** (not the physical NIC):

```bash
# remote (India):  venv/bin/python scripts/serve_agents.py --topology star --deployment a
# local  (US):     sudo venv/bin/python scripts/collect_wan.py \
#                    --remote-host <INDIA_IP> --iface <vpn-tunnel-iface> \
#                    --deployment a --topology star --n 50 --num-predict 256 --out data/raw_wan
venv/bin/python scripts/extract_features.py --raw data/raw_wan --out data/processed_wan --scapy
venv/bin/python scripts/evaluate_c5.py --local data/processed --wan data/processed_wan
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
- **Capture minimization.** `sudo` is required for tcpdump's raw-socket/BPF access. Capture is constrained to the target agent host:port flows via an explicit BPF filter and to a 96-byte snaplen, so only header metadata of the agents under test is ever recorded — never payload, never unrelated hosts. Granting `/dev/bpf*` read access lets the collector run without `sudo` entirely.
- **Responsible disclosure.** Defense evaluation (C4) is conducted before any public release of attack-accuracy results, so the community has mitigation guidance alongside the attack findings.

---

## Project Structure

```
agents/           A2A agent implementations (orchestrator, executor, retriever, validator) — deployment A
agents_b/         Second, deliberately different deployment (B) — cross-deployment / model-vs-logic disentanglement
agents_langgraph/ Deployment C — LangGraph runtime swap (reuses A's specialists; runtime-invariance control)
workflows/        Workflow prompt generators (research_retrieval, code_review, data_analysis, support_triage)
capture/          Packet capture automation (tcpdump/tshark wrappers, trace labeler, automation driver)
features/         Feature extraction pipeline (burst segmentation, per-flow, per-system, flat vector; IPv4 + IPv6)
models/           ML models: Random Forest, GBT, 1D-CNN, Transformer
defense/          Traffic-shaping defenses (padding, scheduling randomization, dummy interactions)
evaluation/       Evaluation scripts (closed-world, open-world, cross-network, metrics)
scripts/          Runnable entry points (PoC, collect, train, evaluate, cross-deployment, off-the-shelf, reproduce)
tests/            Unit + regression tests (features, defense, metrics, stats/SSE, deep-model shapes)
configs/          YAML configs (testbed hosts, topology definitions, workflow assignments)
docs/             Runbooks & appendices (C5 WAN, deep-model appendix, cross-framework plan, a2a_mcp patches)
third_party/      External a2a_mcp compat patch + driver for the off-the-shelf capture (no vendored code)
data/             GITIGNORED (except data/results/) — raw pcaps, features, model checkpoints
logs/             Collection and training logs
```

See [DATA.md](DATA.md) for the published feature/pcap archive layout and reproduction
inputs, and the [docs/](docs/) directory for the C5 WAN runbook, the deep-model appendix,
the cross-framework plan, and the off-the-shelf (`a2a_mcp`) setup notes.

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

This project is released for **academic research purposes**. (A formal `LICENSE` file —
e.g. MIT, BSD-3, or CC-BY for the data — should be added before public release.)

All experiments were conducted on researcher-owned infrastructure. This code must not be used to monitor or infer the activities of multi-agent systems without explicit authorization from the system owner.
