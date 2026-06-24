# Cross-framework validation — LangGraph deployment C + off-the-shelf sample

> **STATUS — SUPERSEDED FRAMING (read this first).** This is the *original plan*.
> The result came in as **A→C high**, and after the runtime-traffic diagnostic
> (`scripts/diagnose_runtime_traffic.py` → `data/results/runtime_traffic_diagnostic.json`)
> the framing was settled as a **runtime-invariance CONTROL, not a generalization
> result**. Deployment C reuses A's specialists, call structure, and task prompts
> unchanged — only the orchestration *runtime* differs — so it is **NOT
> "independently-structured"** and supports **no** "generalizes across frameworks"
> or "transfer tracks structural similarity" claim. It is one or two sentences in
> **§5.2 prose only — no new figure and no added figure bar**. §7 keeps
> generalization across independently-structured frameworks (AutoGen/CrewAI) as
> future work. Wherever the planning text below says "independent framework",
> "across frameworks", "generalizes", or "validation", treat it as **superseded**
> by this banner and the diagnostic JSON's `verdict`.

## Objective
Add a third, **independently-structured** deployment (LangGraph, "C") that runs the
*exact same task taxonomy* as deployment A, then run a train-on-X / test-on-Y
transfer experiment (A→C). This converts the paper's central claim — "the
fingerprint is implementation-specific and does not transfer" — from "two
deployments I built to differ" into "holds across an independent execution model."
Secondarily, stand up one off-the-shelf `a2a-samples` system as a real external
data point.

All collection is **LAN-local on the 3B model**. Nothing here touches the
India/WAN path. The WAN experiment (§5.6) is about *vantage* and stays frozen;
this work is about *implementation* and lives next to the existing
cross-deployment result (§5.3). Keep the two axes separate.

---

## NON-NEGOTIABLE INVARIANTS (read first)

1. **Additive only.** Do not modify, re-collect, or overwrite deployment A
   (`data/processed/`, `data/raw/`), deployment B (`data/processed_b_sdk/`), the
   WAN data (`data/processed_wan/`), or the canonical results
   (`data/results/*.json`). Every existing committed artifact must be
   **byte-identical** after this work. Verify at the end (Phase 7).
2. **Reuse the pipeline unchanged.** Capture (`scripts/run_pilot.py`, tcpdump
   96-byte snaplen, wire-length sizes), feature extraction
   (`scripts/extract_features.py`, 195-dim, canonical flows), and evaluation
   (`scripts/evaluate.py`, `scripts/evaluate_cross_deployment.py`) are
   framework-agnostic because they operate on network metadata. Do **not** fork
   or special-case them for LangGraph. New code is confined to the agent
   implementation.
3. **LAN-local, 3B model.** Back C with `llama3.2:3b` (same as A). The 2×2 already
   proved the model doesn't drive the fingerprint, so a 3B-backed C is
   scientifically valid and collects ~2–3× faster than the 7B runs. No VPN.
4. **Determinism.** `random_state=42` everywhere; respect the `A2A_RESULTS_DIR`
   sandbox convention for any re-runs.
5. **The two make-or-break conditions are label alignment (below) and the
   within-C gate (Phase 4, Step 1).** If either fails, the transfer numbers are
   meaningless. Do not proceed past the gate on faith.

---

## LABEL-ALIGNMENT CONTRACT (the experiment is void without this)

C must emit the **identical label space** as A so that "train on A, test on C" is
well-posed:

- **Workflows (4, same names):** `code_review`, `data_analysis`,
  `research_retrieval`, `support_triage`.
- **Topologies (3, same definitions):** `star`, `chain`, `mesh` — realized as the
  *same connection patterns* as A:
  - `star`: orchestrator calls executor, retriever, validator directly;
    specialists do not call each other.
  - `chain`: orchestrator → executor → retriever → validator (each hop is a real
    network call to the next).
  - `mesh`: all-to-all among the three specialists (plus the orchestrator).
- **Roles (3, same semantics, mapped by port):** `executor`, `retriever`,
  `validator`.
- **Prompts/tasks:** import and reuse **A's exact prompt pool** (the same
  per-workflow task generator A uses). `code_review` in C must draw from the same
  content distribution as `code_review` in A. Do not write new prompts.
- **Payload-size discipline:** reuse A's overlapping-payload construction so
  workflow is not readable off characteristic message sizes (preserves the §5.1
  correctness fix).
- **Grouping key:** features must carry the same `prompt_group` hash field A uses,
  so `StratifiedGroupKFold` works identically.

The point of C is that the **orchestration** (call count, ordering,
parallel-vs-sequential structure) is produced by a **LangGraph `StateGraph`**, not
by hand-rolled Python — that is the independent execution model. The *labels* are
identical; the *implementation that produces the traffic* is genuinely different.

---

## Phase 0 — Setup
1. Create a working branch (e.g. `cross-framework-c`).
2. Add LangGraph to `requirements.txt` (pin a version), install into the venv.
3. Reserve ports **8020 (orchestrator), 8021 (executor), 8022 (retriever),
   8023 (validator)** for C — distinct from A (80xx) and B (801x). Register the
   role→port map in config so `extract_features.py` labels C's roles by the same
   rule it uses for A/B.
4. Write `configs/testbed_langgraph.yaml` modeled on `configs/testbed_local.yaml`
   (same workflows/topologies, C's ports, `llama3.2:3b`).

## Phase 1 — Build LangGraph deployment C
1. Create `agents_langgraph/` (mirror the structure of `agents_b/`). Document at
   the top: model `llama3.2:3b`, ports 8020–8023, "LangGraph-orchestrated
   implementation of the A taxonomy."
2. **Specialists** (executor, retriever, validator): expose each as an A2A server
   using the **existing A2A server scaffold** (the same `A2AStarletteApplication`
   / `AgentExecutor` wrapper A and B use), backed by 3B, doing its role's task.
   They must speak the same A2A JSON-RPC + SSE streaming as A so the collector
   sees genuine A2A traffic.
3. **Orchestrator**: implement as a LangGraph `StateGraph` whose nodes invoke the
   specialist A2A servers via `A2AClient`. Build **three graph variants**
   (star/chain/mesh) matching the topology definitions above — the graph engine
   decides ordering/fan-out, which is the independent structure.
4. **Workflows**: each of the 4 workflows parameterizes the node prompts from A's
   reused prompt pool.
5. Wire C into `scripts/serve_agents.py` as a new `--deployment langgraph` option
   (analogous to how `--deployment a` / `b` work), so the existing collection
   harness drives it unchanged.

## Phase 1.5 — GATE: verify A2A traffic is genuinely on the wire
1. Launch C (`serve_agents.py --deployment langgraph --topology star`) and
   confirm, via tcpdump, that inter-agent calls are **real network flows**
   carrying A2A JSON-RPC + SSE (not in-process LangGraph calls). This is the same
   Phase-0 wire check the README mandates before any collection.
2. If the LangGraph nodes call specialists in-process instead of over the network,
   the topology won't appear in metadata — fix before collecting.

## Phase 2 — Collect C (LAN-local)
1. Collect `--n 30..40` per (workflow × topology) cell into `data/raw_langgraph/`,
   reusing `scripts/run_pilot.py` (or the existing collector) with C served —
   **same tcpdump 96-byte snaplen capture as A**. Total ≈ 360–480 traces.
2. Sanity-check: trace counts per cell, non-empty flows, all four roles observed
   in star, the expected per-topology flow structure present.

## Phase 3 — Extract features for C
1. Run `scripts/extract_features.py --raw data/raw_langgraph --out
   data/processed_langgraph` (reuse unchanged).
2. Verify: 195-dim vectors, `labels.json` present with
   workflow/role/topology/parallelism labels matching the contract, and the
   `prompt_group` field populated.

## Phase 4 — Cross-framework transfer experiment (A→C)
1. **GATE — within-C must work first.** Run closed-world on C alone
   (`scripts/evaluate.py` pointed at `data/processed_langgraph`, or read the
   within-C cells from Step 2). Confirm role and workflow macro-F1 are **clearly
   above chance** (same order as A's ~0.86 / ~0.71 — they need not be identical,
   just clearly learnable). **If within-C is near chance, STOP and debug C** — a
   transfer failure is only meaningful if the fingerprint demonstrably exists
   within C. Do not write any transfer claim until this passes.
2. Run the existing harness for A→C and write to a **new** file:
   ```
   python scripts/evaluate_cross_deployment.py \
       --dir-a data/processed \
       --dir-b data/processed_langgraph \
       --out  data/results/cross_framework.json
   ```
   This yields, for every task (workflow, role, topology, parallelism): within-A,
   within-C, A→C, C→A. (The "B" slot in the output = LangGraph/C; relabel in the
   paper. If trivial, add a `--label-b langgraph` cosmetic flag.)
3. **Interpretation guide (decide which story the data tells):**

   | Pattern | Meaning | Action |
   |---|---|---|
   | within-A high, within-C high, **A→C high** ← **ACTUAL OUTCOME** | A's fingerprint survives a runtime swap that holds logic/prompts/call-structure fixed | **Runtime-invariance CONTROL — NOT generalization.** Report as 1–2 sentences in §5.2 prose; corroborate with the structure-vs-timing diagnostic (`runtime_traffic_diagnostic.json`). Do **not** claim it generalizes across frameworks, and do **not** use the phrase "transfer tracks structural similarity". |
   | within-A high, within-C high, A→C low | (did not occur) would have meant the fingerprint is sensitive even to a runtime change | n/a |
   | within-C low | C produces no learnable fingerprint | **Gate failure** — debug C, do not interpret transfer. |

## Phase 5 — Off-the-shelf sample (secondary, supporting)
1. Pick one multi-agent example from `a2a-samples` (prefer a LangGraph- or
   ADK-based multi-agent demo so it stands up cleanly). Run it **as-is** — do not
   bend it to your taxonomy.
2. Collect ~50–100 traces with varied inputs into `data/raw_offtheshelf/` →
   `scripts/extract_features.py` → `data/processed_offtheshelf/` (same capture
   pipeline, LAN-local).
3. Because its labels do **not** align with A, run only:
   - **Open-world / detection:** feed its traces to the A-trained open-world
     detector (`scripts/evaluate_open_world_background.py` path) and report
     whether this genuinely external A2A system is flagged as agentic vs rejected.
   - **Topology observability:** show its connection graph (host/port structure)
     is readable from metadata — a qualitative structural point.
   - Do **not** compute role/workflow transfer accuracy on it (no aligned labels).
     State this limitation explicitly.
4. This is a one-paragraph external-system corroboration, not the main result.

## Phase 6 — Paper & artifact integration
1. **§5.3** (cross-implementation): add a cross-framework table (within-A,
   within-C, A→C, C→A for role/workflow; topology/parallelism as baselines) next
   to the existing A→B result. Reframe the opening claim: non-transfer now holds
   across *both* a hand-built variant (B) *and* a runtime swap (C — control).
   (Superseded: C is a runtime-invariance control, not an independent framework.)
2. **Abstract + §1 contributions**: update the non-transfer statement to "across
   implementations *and frameworks*"; the framework result is a distinct, citable
   strengthening.
3. **§5.4**: add the off-the-shelf external-system open-world observation.
4. **No new figure** (DECIDED — superseded). The runtime-swap result is not
   comparable to the model-swap / logic-swap disentanglement cells, so it is **not**
   added as a figure or a figure bar. It lives in §5.2 prose +
   `data/results/runtime_traffic_diagnostic.json`. Do not modify
   `scripts/make_paper_artifacts.py` for it.
5. **§7 limitations**: update — three implementations including one independent
   framework; note the set of frameworks tested is still bounded (AutoGen/CrewAI
   remain future work).
6. **`scripts/reproduce.sh`**: add `DIR_LANGGRAPH=data/processed_langgraph` and a
   cross-framework stage (reusing `evaluate_cross_deployment.py --out
   $SANDBOX/cross_framework.json`); add the off-the-shelf stage. **`DATA.md`**:
   add `data/processed_langgraph/`, `data/processed_offtheshelf/` (and the raw
   dirs) to the archive table.
7. Mirror the key §5.3 / abstract / limitations updates into the proposal `.docx`
   so the two documents stay in sync.

## Phase 7 — Verification (must pass before merge)
1. **Additive proof:** `git status` clean except new files; `data/processed/`,
   `data/processed_b_sdk/`, `data/processed_wan/`, and `data/results/*.json` (the
   pre-existing ones) are **byte-identical** to before — confirm by md5 /
   `git diff --stat`.
2. **Determinism:** re-running the cross-framework eval into a sandbox
   (`A2A_RESULTS_DIR=data/results_rerun`) reproduces `cross_framework.json`
   exactly.
3. **Consistency:** every cross-framework number in the paper/proposal matches
   `cross_framework.json` exactly.
4. **reproduce.sh sandbox run** lists the new stages and leaves canonical
   `data/results/` untouched.

---

## Optional (paper-only, no collection) — deep-model appendix
Independent of the above: add a short appendix giving the CNN/Transformer
**architecture, sequence length, training budget, and hyperparameters**. Two
reviewers flagged "the deep models weren't tuned"; this preempts it cheaply and
lets the §4.3 narrative stay short (the metrics move to the appendix rather than
being deleted outright).

## Do NOT
- Do not re-collect or alter A, B, or WAN data, or the committed
  `data/results/*.json`.
- Do not route C or the off-the-shelf sample through the VPN/India path.
- Do not back C with the 7B model (slower, and unnecessary per the 2×2).
- Do not write any transfer conclusion before the within-C gate (Phase 4, Step 1)
  passes.
- Do not fork the capture/feature/eval scripts for LangGraph — reuse them.
