# Phase 5 — Off-the-shelf system (Google `a2a_mcp`): external setup & compat patches

## Why this exists

Phase 5 captures a **third-party, independently-authored** A2A multi-agent system to
corroborate that our traffic-analysis attack works on a system **we did not build**.
The target is Google's `a2a_mcp` travel-planning sample from
[`google-a2a/a2a-samples`](https://github.com/a2aproject/a2a-samples)
(`samples/python/agents/a2a_mcp`): an MCP agent-registry server (SSE + embedding
`find_agent`), an orchestrator (`WorkflowGraph`), a LangGraph planner, and three ADK
travel specialists (air / hotel / car), all talking **A2A over the network**, backed by
Gemini.

**Scope (honesty guardrail).** This system's labels do **not** align with our
WorkflowClass / TopologyType taxonomy, so it yields **no role/workflow transfer number**.
It corroborates **open-world DETECTION** ("is an A2A multi-agent system present?") and
**TOPOLOGY OBSERVABILITY** ("read the agent connection graph from headers"), nothing more.
See `scripts/collect_offtheshelf.sh` and `scripts/extract_offtheshelf.py`.

## Reproducibility gap this closes

The sample is an **external git checkout**, not vendored into this repo, and it does not
run as-shipped against the current Gemini API / a2a-sdk. We made 8 small compatibility
fixes there. Those edits were un-versioned (a real reproducibility hole), so this folder
records them:

```
third_party/a2a_mcp/
  a2a_mcp_compat.patch   git diff of the 8 fixes (7 source files; uv.lock excluded)
  drive_orch.py          NEW: multi-turn A2A driver that auto-answers the planner's
                         clarifying questions and runs the full workflow end-to-end
docs/PHASE5_A2A_MCP_PATCHES.md   (this file)
```

The patch is against upstream commit **`22b48d5`** ("chore: refactors hello world
example (#566)"). `.env` (which holds the Gemini API key) is **never** copied or
committed — it is gitignored upstream and stays only on the capture host.

## How to reproduce the external setup

```bash
# 1. Clone the sample repo at the pinned upstream commit
git clone https://github.com/a2aproject/a2a-samples ~/a2a-samples
git -C ~/a2a-samples checkout 22b48d5

# 2. Apply our compatibility patch
git -C ~/a2a-samples apply /path/to/this-repo/third_party/a2a_mcp/a2a_mcp_compat.patch

# 3. Drop in the multi-turn driver (untracked NEW file)
cp /path/to/this-repo/third_party/a2a_mcp/drive_orch.py \
   ~/a2a-samples/samples/python/agents/a2a_mcp/drive_orch.py

# 4. Provide the Gemini key locally (NOT committed)
cd ~/a2a-samples/samples/python/agents/a2a_mcp
printf 'GOOGLE_API_KEY=YOUR_KEY\nGOOGLE_GENAI_USE_VERTEXAI=FALSE\n' > .env

# 5. Resolve deps (regenerates uv.lock -> a2a-sdk 0.3.26, replacing yanked 0.3.0)
uv sync

# 6. One-time, for non-root tcpdump on lo0:
sudo chmod o+r /dev/bpf*

# 7. Capture + extract (from THIS repo)
bash scripts/collect_offtheshelf.sh 4
venv/bin/python scripts/extract_offtheshelf.py --raw data/raw_offtheshelf \
    --out data/processed_offtheshelf --scapy

# 8. Phase 5c analysis (detection + topology observability — NO transfer claim).
#    Both scripts are pre-written; they exit cleanly with a notice until traces exist.
venv/bin/python scripts/evaluate_offtheshelf_detection.py     # open-world detection
venv/bin/python scripts/analyze_offtheshelf_topology.py       # topology from headers
```

**Phase 5c scope.** `evaluate_offtheshelf_detection.py` runs the A-trained open-world
detector on the off-the-shelf features and reports the flagged-as-A2A rate (vs background
true-negatives). `analyze_offtheshelf_topology.py` recovers the agent connection graph from
flow headers alone. Both are DETECTION / TOPOLOGY only — a2a_mcp labels do not align with
our taxonomy, so neither yields a role/workflow transfer number.

**Gotcha — IPv6 loopback.** The a2a_mcp services bind `localhost`, which on macOS resolves
to **`::1` (IPv6)**, so all captured A2A traffic is IPv6 — unlike the main testbed (IPv4).
`features/extractor.py` and `scripts/analyze_offtheshelf_topology.py` parse **both** IPv4
and IPv6 (additive; the IPv4 path is byte-for-byte unchanged, verified). The `tcpdump`
`tcp portrange 10100-10105` filter already captures both families, so no capture change is
needed. First validation run (1 trace, Tokyo trip): topology recovered cleanly as
hub-and-spoke — **MCP registry is the hub** (every agent queries `find_agent`), orchestrator
+ planner coordinate, air/hotel/car are leaves. Detection needs more than 1 trace to be
meaningful (accumulate over daily quota-limited runs).

**Gemini free-tier quota.** `gemini-2.5-flash` allows ~20 `generateContent`/day; one trip
uses ~8–12 calls, so expect ~2 traces/run. `collect_offtheshelf.sh` stops on the first
429 (`RESOURCE_EXHAUSTED`); re-run after the daily reset to accumulate, or enable billing
for a one-shot ~50–100 trace capture.

## The 8 compatibility fixes

Each was an *as-shipped vs current-API* breakage, not a behavioral change to the system we
measure — the agent logic, prompts, and A2A call structure are untouched.

| # | File | Symptom | Root cause → fix |
|---|------|---------|------------------|
| 1 | `pyproject.toml` (+ `uv.lock`) | `uv sync` fails / resolves yanked build | `a2a-sdk[sql]>=0.3.0` pulled the **yanked** 0.3.0 → pin `>=0.3.3,<0.4` (resolves to **0.3.26**). |
| 2 | `mcp/server.py` | embedding call 404s | `models/embedding-001` (and `text-embedding-004`) **retired** → `models/gemini-embedding-001` (what the key can access). |
| 3 | `common/workflow.py` | downstream call → JSON-RPC **-32001** | Message carried `'taskId': task_id` for a **not-yet-created** task → omit it; the server creates a fresh task per call. |
| 4 | `common/workflow.py` | specialist calls time out | `httpx.AsyncClient()` default ~5 s timeout vs multi-second LLM calls → `httpx.Timeout(180.0)`. |
| 5 | `agents/adk_travel_agent.py`, `agents/langgraph_planner_agent.py`, `agents/orchestrator_agent.py` (×2) | `gemini-2.0-flash` quota = 0 | Free tier gives **0** req for 2.0-flash → `gemini-2.5-flash` (4 occurrences across 3 files). |
| 6 | `common/agent_executor.py` | proxied event → **-32602** task mismatch | Proxied downstream `TaskStatus/Artifact` events kept the child's ids; server rejects the mismatch. The sample's comment intended an id-rewrite but the code was missing → `event.model_copy(update={'task_id': task.id, 'context_id': task.context_id})` before enqueue. |
| 7 | `mcp/server.py` | `find_agent` returns empty `content[0].text` | Tool is declared `-> str` and callers do `json.loads(result.content[0].text)`, but a **dict** return lands in `structuredContent` → `json.dumps(...)` the agent card. |
| 8 | `agents/adk_travel_agent.py` | `NameError: os is not defined` | `os.getenv(...)` used without `import os` (latent until fix #5's line ran) → add `import os`. |

Verified outcome: with all 8 applied, the full workflow runs end-to-end over A2A
(orchestrator → planner → `find_agent`(MCP) → air/hotel/car specialists), confirmed by
incoming POSTs in the specialist logs and the captured pcaps on ports 10100–10105.

## `drive_orch.py`

The sample ships interactive CLIs but no non-interactive end-to-end driver. `drive_orch.py`
sends the trip request to the orchestrator over A2A, and auto-answers the planner's
clarifying questions (dates, travel class, budget) from the prompt so one invocation
produces one complete multi-agent trace. It contains **no credentials**; it reads the key
from the environment (`GEMINI_API_KEY`/`GOOGLE_API_KEY`) like the rest of the sample.
