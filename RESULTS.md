# Results

Headline results, with honest scope and caveats. All numbers are read from the
committed JSONs under [`data/results/`](data/results/) and regenerate deterministically
via `bash scripts/reproduce.sh --full-suite` (RF/GBT seed-fixed; deep models opt-in via
`--with-deep`). Metric is **macro-F1** unless stated; 95% CIs are percentile bootstrap
(2000 resamples, seed 42).

> **Reporting conventions.** (1) **macro-F1** is the headline metric everywhere (accuracy
> is reported only as context). (2) **Bandwidth only** — latency/duration overheads are
> *not* reported: the defended sets were collected separately, so absolute durations are
> confounded (the raw field is retained in the JSON, marked `latency_overhead_note`).
> (3) topology/parallelism are a **structural baseline** (the connection graph is readable
> from IP headers without ML) — reported for completeness, not as the attack contribution.

---

## 1. Closed-world attack — deployment A

Held-out traces of seen classes; StratifiedGroupKFold on prompt_group (no prompt leaks
across folds). `data/results/closed_world/`.

| Task | Chance | RF | **GBT (headline)** |
|---|---|---|---|
| **workflow** | 0.25 | 0.663 | **0.708 ±0.04** |
| **role** | 0.33 | 0.868 | **0.864 ±0.02** |
| topology | 0.33 | 0.985 | **0.995** |
| parallelism | 0.50 | 0.972 | **0.989** |

Workflow and agent role — the real attack targets — are recovered far above chance from
metadata alone.

**Deep models (footnote, data-starved at N=600):** CNN/Transformer underperform the trees
(e.g. workflow CNN 0.228 / Transformer 0.100; role Transformer 0.684). Architectures,
input representation, parameter counts, and training budget are documented in
[`docs/DEEP_MODEL_APPENDIX.md`](docs/DEEP_MODEL_APPENDIX.md) to pre-empt an "untuned models"
read — the gap is sample size, not tuning. They are **off by default** in `reproduce.sh`.

---

## 2. What the fingerprint depends on — model vs. logic (the causal result)

Same taxonomy, prompts, and label space; swap one factor at a time. `data/results/model_vs_logic.json`.

| Task | A→A (ceiling) | Model swap | **Logic swap** | Both (A→B) |
|---|---|---|---|---|
| workflow | 0.678 | 0.588 | **0.321** | 0.289 |
| role | 0.856 | 0.829 | **0.517** | 0.568 |

Swapping the **LLM model** barely moves the fingerprint; swapping the **call logic /
structure** collapses it (workflow → ≈chance). **The leak is in the inter-agent call
structure, not the model.**

---

## 3. Runtime-invariance control — A ↔ C (LangGraph)

Deployment C re-implements A's orchestrator in **LangGraph**, reusing A's specialists,
prompts, and call structure unchanged — only the orchestration runtime differs.
`data/results/cross_framework.json`.

| Task | A→A | C→C | **A→C** | C→A |
|---|---|---|---|---|
| workflow | 0.678 | 0.613 | **0.644** | 0.579 |
| role | 0.856 | 0.882 | **0.830** | 0.843 |
| topology | 0.982 | 0.985 | 0.994 | 0.983 |

Transfer stays near the within-A ceiling. The companion diagnostic
(`runtime_traffic_diagnostic.json`) finds A and C **structure-invariant, timing-shifted**
(structural |Cohen's d|max = 0.34 across matched cells; only wall-clock duration differs).

> **Honest scope:** C is a **control**, not a generalization result — it shares A's exact
> structure, so it cannot speak to *cross-framework generalization*. Read §2 and §3
> together: change the **runtime** → survives (A→C 0.64/0.83); change the **structure** →
> breaks (A→B 0.29/0.57). Generalization across independently-structured frameworks
> (AutoGen, CrewAI, …) remains future work.

---

## 4. Cross-network — US ⇄ India WAN (C5)

`data/results/c5_cross_network.json`.

| Task | Chance | LAN | WAN (in-domain) | LAN→WAN transfer | n_wan |
|---|---|---|---|---|---|
| workflow | 0.25 | 0.663 | **0.616** | 0.196 | 595 |
| role | 0.33 | 0.869 | **0.871** | 0.506 | 1195 |
| topology | 0.33 | 0.985 | **0.997** | 0.583 | 595 |

The attack **works on real WAN traffic when trained in-domain**. A LAN-trained model **does
not transfer** to WAN conditions (workflow 0.196 ≈ chance) — absolute timing shifts across
networks, so the attacker must train under the target network's conditions.

---

## 5. Live C4 defenses (workflow attack, real defended captures)

Fixed attacker trained on undefended traffic, applied to real defended captures
(N=50/pair). `data/results/defense/defense_live.json`. **Headline = macro-F1; bandwidth = byte overhead.**

| Defense | macro-F1 [95% CI] | Accuracy | Above-chance F1 retained | Byte overhead |
|---|---|---|---|---|
| none | 0.656 [0.62, 0.69] | 0.657 | — | 0% |
| rate | 0.531 [0.49, 0.57] | 0.530 | **69%** | +35% |
| pad | 0.544 [0.50, 0.58] | 0.540 | **72%** | +31% |

Defenses **degrade but don't defeat** the attack — ~70% of above-chance signal survives at
a ~30% bandwidth cost. (macro-F1 ≈ accuracy here — predictions stay balanced across the 4
classes — so the metric choice does not change the conclusion. Latency overhead is
**excluded as confounded**.)

---

## 6. Open-world — two distinct experiments (do not conflate)

### §6 Open-set class rejection — `data/results/open_world/`
Train on the *known* classes, hold out one class as *unknown*; can the classifier **reject
a novel class** at a fixed ~5% known-FPR?

| Task | Reject-unknown rate | Known-FPR |
|---|---|---|
| topology | **100%** | 5.0% |
| role | 6.7% | 5.2% |
| workflow | 3.3% | 5.2% |

Topology rejects unknown structures cleanly; **workflow/role rejection is weak** (a novel
workflow still scores high confidence on the known classes). Honest limitation of the
confidence-threshold approach.

### §5.4 Real-background detection — `data/results/open_world_background.json`
Is **ordinary internet traffic** (web/API/JSON-RPC/file/LLM) distinguishable from A2A?
Two framings give very different answers:

- **As pure novelty detection** (the deployed workflow-detector + rejection, never trained
  on background): rejects only **39.7%** of real background at 5.3% known-FPR — file-download
  (100%) and direct-LLM (90%) are caught, but REST-API, JSON-RPC, and multi-REST background
  pass as A2A (0% rejected). Precision at a 1% A2A base rate is **1.6%**. **Weak** — pure
  novelty detection is not a usable A2A detector.
- **As supervised binary A2A-vs-background** (background as negatives — see §7): **AUC
  1.000**, separating A2A from the *same* background categories, including the hard,
  multi-flow JSON-RPC / multi-REST / LLM-direct traffic that the novelty detector accepts at
  0% rejection. Per-category cross-validated A2A-probability is near-zero for those hard
  categories — committed in `detection.background_per_category`: **jsonrpc 0.016, multi-REST
  0.013, LLM-direct 0.001**. Requires background samples to train.

**Same or different negatives? (definite)** Different *directories* — novelty uses
`data/processed_background_sdk` (n=300), supervised uses `data/processed_background`
(n=150) — but the **identical 6-category composition** from the same background collector:
both contain all three `type=hard` categories (`bg_jsonrpc`, `bg_multi_rest`,
`bg_llm_direct`), 25–50 traces each. So the supervised detector's AUC 1.0 **is measured
against the same hard categories the novelty detector accepts**, not against easier traffic
— and it separates even the *parallel* multi-REST negative, so the result is **not merely a
concurrent-vs-sequential artifact**. **The §5.4-vs-§7 gap is therefore detector type (novelty
vs. supervised), not a contradiction.** Honest limit: **every negative is non-agentic
(multi-flow but non-SSE)**, and the detector keys on A2A's SSE-streaming + orchestrator
fan-out signature — so this measures **"A2A vs non-agentic traffic," not "A2A vs other agent
frameworks."** Separability from other **agentic, SSE-based** frameworks (AutoGen, CrewAI, …)
is **untested**.

---

## 7. Off-the-shelf external system — Google `a2a_mcp` (Phase 5)

An independently-authored multi-agent system (MCP registry + orchestrator + LangGraph
planner + ADK air/hotel/car specialists), captured over A2A (n=150). **Detection +
topology only — labels do not align with our taxonomy, so no role/workflow transfer
number.** `data/results/offtheshelf_{detection,topology}.json`.

- **Detection (binary A2A-vs-background, the correct framing):** **AUC 1.000**; the external
  a2a_mcp is **detected as A2A at 100%** [100%, 100%] at a 5% background-FPR operating point.
  - *Caveat (identical to the `caveat` field in the JSON):* the negatives are **multi-flow
    but non-SSE / non-agentic** (web/API, JSON-RPC, multi-REST, file-download, direct-LLM).
    The detector separates A2A via its **SSE-streaming + orchestrator fan-out signature** — it
    separates even the hard JSON-RPC / multi-REST / LLM-direct categories (cross-validated
    A2A-probability in `background_per_category`: **0.016 / 0.013 / 0.001**). But **every
    negative is non-agentic**, so this measures **"A2A vs non-agentic traffic," not "A2A vs
    other agent frameworks"**; separability from other **agentic, SSE-based** frameworks
    (AutoGen, CrewAI, …) is **untested** — the sterner future test. a2a_mcp **carries the
    same structural A2A fingerprint** and is detected at the standard operating point.
  - The naive workflow-novelty detector **fails** here (AUC 0.47 < 0.5 — a2a_mcp is a
    different workflow), which is why the binary framing is used; recorded as context.
- **Topology:** **hub-and-spoke / hierarchical**, hub = **MCP registry** (every agent
  queries it), specialists as leaves — recovered from flow headers alone, no payload, no ML.

---

## Bottom line

A passive observer, seeing only encrypted-traffic metadata, recovers **workflow** (GBT
0.71) and **agent role** (0.86) far above chance. The signal is **caused by the inter-agent
call structure** — invariant to the **LLM model** (0.68→0.59) and to the **orchestration
runtime** (A→C 0.64), destroyed by changing the **structure** (A→B 0.29). It holds on **real
WAN traffic** in-domain, **survives current defenses** (~70% F1 retained at ~30% bandwidth),
and the **structural A2A fingerprint is detectable on a system we did not build** (a2a_mcp,
100% @5% FPR — separating it even from hard JSON-RPC/multi-REST background, where a
confidence-threshold novelty detector fails; a sterner test would use other agent frameworks
as negatives).

**Not claimed:** LAN→WAN transfer, cross-*framework* generalization (C is a control),
strong open-set rejection of novel workflows/roles, or a detector robust to hard agentic
negatives — all explicitly future work.
