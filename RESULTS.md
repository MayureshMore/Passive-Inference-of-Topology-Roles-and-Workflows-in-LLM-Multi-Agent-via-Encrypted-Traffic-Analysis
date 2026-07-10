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

**Deep models — NOT a fair baseline (data-starved at N=600).** At this scale the
CNN/Transformer collapse toward single-class prediction, so their macro-F1 lands *at or
below chance* on several tasks (workflow CNN 0.228 / Transformer 0.100; role Transformer
0.684). **These sub-chance numbers are a data-scale sensitivity check, not a fair
trees-vs-deep comparison** — they show the sequence models are *starved*, not that deep
architectures are worse than trees on this problem. Architectures, input representation,
parameter counts, and training budget are in
[`docs/DEEP_MODEL_APPENDIX.md`](docs/DEEP_MODEL_APPENDIX.md) (to pre-empt an "untuned models"
read — the gap is sample size, not tuning); they would need ~1,500–2,000 traces/class to be
comparable. **Off by default** in `reproduce.sh` (opt in with `--with-deep`), and excluded
from every headline claim.

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

### 5.1 Overhead–accuracy curve (defense sweep)

The two rows above are single operating points; sweeping each defense across strengths turns
them into a curve. `data/results/defense_curve.json` +
[`figures/defense_curve.png`](data/results/figures/defense_curve.png). The method is
identical to the live eval (fixed RF attacker trained on undefended traffic, group-safe CV,
bootstrap CI); the sweep applies each defense **as a deterministic transform on the
undefended base capture** and re-extracts through the real feature pipeline, so the two live
points serve as ground-truth anchors. **Latency here is schedule-derived** (computed from the
imposed inter-packet spacing), removing the separate-capture confound that made the live
latency uninterpretable.

| Defense (swept) | Overhead range | Attack macro-F1 | Reading |
|---|---|---|---|
| **size padding** (cell 64→2048 B) | **+6% → +239% bytes** | 0.619 → **0.585** | huge bandwidth cost, attack barely dented |
| **timing spacing** (min-gap 25→200 ms) | **+6% → +166% latency** | 0.565 → **0.540** | drops ~0.12, then **plateaus** at ~0.54 |

**Validation — the sweep reproduces the live points:** identity transform → macro-F1 **0.657**
(live undefended 0.656); size-padding at the deployed 512-B cell → **+33% bytes / F1 0.595**
(live pad +31% / 0.544 — same ballpark; the simulation isolates the *pure size effect on the
same traces*, so it degrades slightly less than the separately-collected live set).

**Both curves stay far above chance (0.25) at every operating point.** Neither defense
approaches a clean defeat, and both plateau near ~0.54 (≈65% of above-chance signal retained):
the size defense is purely expensive, and the timing defense *saturates* — consistent with the
fingerprint being **structural, not timing-borne** (§2–§3). *(The live "rate" defense is a
distinct count-based mechanism — dummy sub-calls + reordered delegation — so it is a measured
anchor, not a point on the timing curve. SOTA-strength defenses remain future work.)*

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
  0.013, LLM-direct 0.001**. Requires background samples to train. **This is not a real-world
  detectability claim — all negatives are non-agentic; see §7 for why it is an open problem.**

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

- **Detection — a stated OPEN PROBLEM, not a real-world detectability claim.** In a
  *supervised binary A2A-vs-background* setup the external a2a_mcp is separated from our
  background set with **AUC 1.000** (detected as A2A 100% [100%, 100%] at a 5% background-FPR
  point). **We do not present this as evidence that A2A is detectable in the wild**, for one
  structural reason: **every negative in the background set is non-agentic** — multi-flow but
  non-SSE (web/API, JSON-RPC, multi-REST, file-download, direct-LLM). The detector separates
  A2A via its **SSE-streaming + orchestrator fan-out signature**, so the AUC 1.000 measures
  only **"A2A vs non-agentic traffic," not "A2A vs other agent frameworks."**
  - *Why even the 1.000 is fragile:* on *average* the hard categories score near-zero
    A2A-probability (`background_per_category.mean_a2a_prob`: 0.016 / 0.013 / 0.001), **but at
    the 5%-FPR threshold they sit on the A2A boundary** — `flagged_as_a2a_at_T` flags
    **multi-REST 20%** and **JSON-RPC 8%** (≈0% for the soft categories), consuming almost the
    entire false-positive budget. The 1.000 is a perfect ranking on **only 75 hard negatives
    (25/category)** and would likely not survive more hard-negative data.
  - *The test that actually matters (untested — this is the open problem):* separability from
    other **agentic, SSE-based** frameworks (AutoGen, CrewAI, …). Until A2A is distinguished
    from *those*, real-world A2A detectability is **unproven**. The naive workflow-novelty
    detector already **fails** here (AUC 0.47 < chance — a2a_mcp is a different workflow),
    which is why the binary framing was used; recorded as context.
- **Topology:** **hub-and-spoke / hierarchical**, hub = **MCP registry** (every agent
  queries it), specialists as leaves — recovered from flow headers alone, no payload, no ML.

### 7.1 Role fingerprint REPLICATES on a2a_mcp (independent-implementation replication)

Detection/topology above use a2a_mcp only as external corroboration. Going further: does the
**behavioural role fingerprint** (§1–§2) replicate on this system we did not build?
`data/results/offtheshelf_fingerprint.json` +
[`figures/offtheshelf_fingerprint.png`](data/results/figures/offtheshelf_fingerprint.png).
Each trip's flows are pooled by the agent port they target and classified into a2a_mcp's **own**
roles from the **35-dim per-agent traffic shape** — the same representation as §1's role task,
with the port used only for the *label*, never as a feature. GBT, group-safe CV by trip, bootstrap CI.

| Role task on a2a_mcp | Chance | macro-F1 [95% CI] | n |
|---|---|---|---|
| **6-way** (mcp / orchestrator / planner / air / hotel / car) | 0.167 | **0.906 [0.848, 0.954]** | 501 |
| **coordinator vs specialist** (2-way) | 0.50 | **1.000 [1.000, 1.000]** | 501 |

**Agent role is recovered far above chance on an independently-authored system** — the
fingerprint is not an artefact of our own deployments. This is the result that upgrades the
cross-implementation story from "existence proof on deployment B (which we *built* to differ)"
to "**replicated on a system we did not author**."

**Cross-implementation transfer — the honest limit.** A and a2a_mcp have **disjoint role
taxonomies** (executor/retriever/validator vs registry + coordinator + travel specialists), so a
labelled A↔a2a_mcp transfer is undefined. The one coarse abstraction both share is
coordinator-vs-specialist; A has **only** specialists, so the definable direction is
**a2a_mcp→A**: an a2a_mcp-trained coordinator-vs-specialist model classifies **67%** of A's
specialists correctly (specialist recall 0.671, n=1747) — a **partial** transfer (the
"specialist" traffic-shape partly generalises; the rest reads as coordinator). The reverse is
undefined. **True cross-*framework* label transfer needs a framework sharing A's role taxonomy
(AutoGen/CrewAI) — future work.**

**No workflow closed-world (and why).** a2a_mcp's routing is **LLM-planned**, so requests do not
map to a clean specialist fan-out: a live probe found a *flight-only* request triggered **no**
specialist fan-out while a *hotel-only* request fanned out to **all three** specialists
(`workflow_probe` in the JSON). Workflow-path classes are not cleanly separable here; a workflow
fingerprint on an external system needs deterministic routing — future work.

---

## Bottom line

A passive observer, seeing only encrypted-traffic metadata, recovers **workflow** (GBT
0.71) and **agent role** (0.86) far above chance. The signal is **caused by the inter-agent
call structure** — invariant to the **LLM model** (0.68→0.59) and to the **orchestration
runtime** (A→C 0.64), destroyed by changing the **structure** (A→B 0.29). It holds on **real
WAN traffic** in-domain, and **survives current defenses** (~70% F1 retained at ~30%
bandwidth). Crucially, the **role fingerprint REPLICATES on a system we did not build**
(a2a_mcp: 6-way role macro-F1 **0.906**, coordinator-vs-specialist **1.000**; topology also
recovers as hub-and-spoke from headers alone) — moving the cross-implementation claim beyond our
own deployments. A2A-vs-background *detection* separates a2a_mcp at AUC 1.000, but **only against
non-agentic negatives** — so real-world detectability (vs. other agentic SSE frameworks) remains
an **open problem**, not a claim.

**Not claimed:** LAN→WAN transfer, cross-*framework label* transfer (A↔a2a_mcp taxonomies are
disjoint — role *replicates* independently, but shared-label transfer is undefined; only a
partial coordinator-vs-specialist transfer at 0.67), a workflow fingerprint on an external
(LLM-planned) system, cross-*framework* generalization (C is a control), strong open-set
rejection of novel workflows/roles, or a detector robust to hard agentic negatives — all
explicitly future work.
