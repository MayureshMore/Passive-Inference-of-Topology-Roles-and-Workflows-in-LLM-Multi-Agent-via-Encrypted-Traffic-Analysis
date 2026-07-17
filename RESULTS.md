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
> (4) **Capture provenance (machine-checkable).** Every result is derived from a **loopback**
> capture (all agents on one host, tcpdump on `lo0`) **except §4/C5**, whose traces are genuinely
> **cross-host** (agents on a remote VM; endpoints on a routable address, captured post-decapsulation
> on the VPN tunnel). This is derived from the traces themselves — each capture's endpoint hosts (and,
> where absent, the pcap's own IPs) — not asserted in prose:
> `data/results/capture_interface_manifest.json`. (5) Headline CIs use a **cluster (group) bootstrap**
> — see the C4 notes in §1 and §9a.

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

> **Interval convention (C4).** Samples are clustered by **prompt_group** (the CV is already group-safe),
> so the headline interval is a **cluster (group) bootstrap** — resampling whole prompt-groups:
> workflow **[0.665, 0.745]** (289 groups). The i.i.d. interval ([0.672, 0.743]) is **14% narrower** and
> over-confident, since same-prompt-family traces are correlated. Point estimate unchanged; the
> conclusion (far above the 0.25 chance line) is unchanged. `data/results/group_bootstrap_check.json`.

**Deep models — known-DEGENERATE pipeline, excluded from all claims.** The CNN/Transformer
do not merely underperform: they **collapse to near-single-class prediction**, which is why
their macro-F1 falls *below chance* (workflow CNN 0.228 / Transformer 0.100; chance 0.25).
**This is a degenerate classifier, not a "data-starved" baseline** — a genuinely under-powered
but functional model sits *near* chance with balanced predictions, whereas here the workflow
CNN routes **326/600** predictions into one class and predicts `research_retrieval` for only 18
(recall 4.7%), and the Transformer predicts essentially a single class (accuracy = chance 0.25).
Below-chance macro-F1 alongside near-chance accuracy is the signature of **class collapse, not
sample size** — so we do not attribute it to N and we **exclude these runs from every claim**
(the headline attacker is RF/GBT). Whether the collapse is fixable with more data or a training
fix is untested and out of scope; architectures, input representation, parameter counts, and
budget are documented in [`docs/DEEP_MODEL_APPENDIX.md`](docs/DEEP_MODEL_APPENDIX.md). **Off by
default** in `reproduce.sh` (opt in with `--with-deep`).

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
frameworks."** Separability from another **independent agentic framework** is now tested in
**§11** (A2A vs AutoGen, AUC 1.0 — but transport-driven; a same-transport SSE framework like
CrewAI remains future work), and the loss-of-port-isolation concern is quantified in **§12**.

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
| **6-way** (mcp / orchestrator / planner / air / hotel / car) — *the behavioral result* | 0.167 | **0.906 [0.848, 0.954]** | 501 |
| coordinator vs specialist (2-way) — *partly structural (see note)* | 0.50 | 1.000 [1.000, 1.000] | 501 |

**The 6-way 0.906 is the headline: agent role is recovered far above chance on an
independently-authored system, from per-agent traffic shape alone** — the behavioral
fingerprint is not an artefact of our own deployments. This is the result that upgrades the
cross-implementation story from "existence proof on deployment B (which we *built* to differ)"
to "**replicated on a system we did not author**."

> **Read the coordinator-vs-specialist 1.000 with less weight — it is _partly structural_, not
> purely behavioral.** Coordinator hubs (registry / orchestrator / planner) carry far more
> connection volume and fan-in than the specialist leaves, so the 2-way split rides largely on
> the **same header-readable connection-graph signal as topology** (the intro's structural-baseline
> caveat (3)), not on subtle per-agent behavior. The genuinely behavioral claim is the **6-way**
> number.

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

## 8. Framework / implementation identification (Phase 1 — recon)

Can a passive observer tell *which implementation* a deployment runs, from traffic alone (the
reconnaissance half of an attack)? Multiclass GBT over the four captured implementations
(A, B, C/LangGraph, a2a_mcp), one whole-trace vector each, group-safe CV, bootstrap CI.
`data/results/framework_id.json` +
[`figures/framework_id.png`](data/results/figures/framework_id.png). **Feature = the 195-dim
traffic-shape vector minus 5 explicit endpoint/flow-count features** (`n_flows`, `n_src_hosts`,
`n_dst_hosts`, `n_host_pairs`, `max_concurrent_flows`) so the result is a *shape* fingerprint,
not the trivial "count the endpoints" topology baseline. No feature is port/IP/identity-derived.

| Feature set | macro-F1 [95% CI] | chance | A↔C separability |
|---|---|---|---|
| full traffic-shape (separate-session capture) | 0.998 [0.996, 1.000] | 0.25 | 0.997 |
| timing ablated (−29 timing feats) | 0.993 [0.989, 0.996] | 0.25 | 0.989 |

> **⚠ This near-perfect number is a CAPTURE-SESSION CONFOUND — we ran the control and it does not
> survive.** Every pair separated at ~1.0, *including A vs C_langgraph* which share call structure
> by design. Perfect separation of things that should be *similar* is the classic signature of a
> batch confound: each implementation was collected in a *separate* session, so the classifier can
> key on session artefacts (host state, clock granularity, ephemeral-port ranges) rather than a
> genuine fingerprint. We did not leave this as a caveat — we tested it.

### 8.1 Same-session interleaved control (the confound test — **run, not deferred**)

We re-collected A, B, and C **round-robined in one continuous session**
(`scripts/collect_interleaved.sh`; 6 cycles × 12 workflow×topology conditions), so any
session-drift artefact is *shared across labels* and can no longer predict the label. A and C
use the **same model** (`llama3.2:3b`), same call logic, and identical condition coverage, so the
**only** systematic difference is the orchestration runtime (asyncio vs LangGraph).
`data/results/framework_id_interleaved.json` +
[`figures/framework_id_control.png`](data/results/figures/framework_id_control.png).

| A↔C (asyncio vs LangGraph), balanced 2-way | macro-F1 [95% CI] | chance |
|---|---|---|
| **separate-session (confounded)** | separability **0.997** | 0.50 |
| **same-session interleaved (controlled)** | **0.460 [0.381, 0.542]** | 0.50 |
| controlled, timing also ablated | 0.383 | 0.50 |

**Verdict: the A↔C fingerprint COLLAPSES to chance under the control** (0.997 → 0.46; CI straddles
0.50). The near-perfect separate-session number was **batch-inflated**. Honest consequence:

- **Within-family implementation ID (A/B/C) is NOT a real recon signal** — it was a capture
  artefact. We *demote it* rather than defend it. (The 3-way stays at 0.61 only because deployment
  B's orchestrator deterministically fails chain/mesh, so B is separable by *topology coverage*, a
  second artefact — see `class_condition_coverage`; not a behavioural fingerprint either.)
- **Distinct-framework ID survives for a real reason.** a2a_mcp (a 6-agent travel system) vs our
  4-agent A2A separates because the call *structure* genuinely differs — that is topology, not a
  session artefact, and it is unaffected by this control.
- **This CONFIRMS the §3 runtime-invariance thesis from the dual direction.** §3 shows the attack
  *transfers* A→C because they share call structure; the control shows you *cannot even fingerprint*
  A vs C — same structure → same traffic shape. Both point to one fact: the signal is **structure**,
  and structure is invariant to the runtime. The earlier "structure-invariant, *timing-shifted*"
  gap (§3 diagnostic) was itself largely session drift — under same-session capture, timing adds
  almost nothing (0.46 → 0.38 when timing is removed).

**Net for the paper:** framework/implementation fingerprinting is honestly reported as **negative
under control** (a strengthening, not a weakness — it removes the most confound-vulnerable claim and
corroborates §3). The recon claim that remains is the narrow, defensible one: an observer can tell
apart **structurally-distinct** frameworks, because that is just topology recovery by another name.

### 8.2 Confound audit — the SAME control leaves the core attack UNCHANGED

§8.1 raises the obvious question a reviewer will ask next: *if the framework-ID number was a
capture-session artefact, why trust workflow / role / topology?* We answer it directly — by
re-running the **three core closed-world tasks on same-session interleaved captures** and comparing
to the committed (batch-collected) baselines. `data/results/confound_control.json` +
[`figures/confound_control.png`](data/results/figures/confound_control.png).

- **Role & topology** use the powered interleaved-A capture (`collect_interleaved.sh`; deployment A,
  llama3.2:3b, all four workflows round-robined across the session — 239 traces).
- **Workflow** uses a dedicated prompt-diverse interleaved capture across **all three topologies**
  (`collect_wf_interleaved.sh` with `star chain mesh`; 432 traces, 4 workflows round-robined in short
  blocks, **fresh prompts each cycle** via the new `run_pilot --seed-offset` so the group-CV — which
  holds out whole prompts — has real prompt diversity: 50–76 prompt-groups/workflow). Compared against
  the committed **all-topology** 0.708 baseline; a star-only capture corroborates (0.690).

| Core task | committed (batched) | same-session interleaved (controlled) | Δ | verdict |
|---|---|---|---|---|
| **workflow** | 0.708 [0.672, 0.743] | **0.651 [0.616, 0.697]** | −0.06 | **SURVIVES** (CI overlap; star-only 0.69 corroborates) |
| **role** | 0.864 [0.847, 0.879] | **0.886 [0.853, 0.903]** | +0.02 | **SURVIVES** |
| **topology** | 0.995 [0.988, 1.000] | **1.000 [1.000, 1.000]** | +0.01 | **SURVIVES** |
| **parallelism** | 0.989 [0.979, 0.996] | **1.000 [1.000, 1.000]** | +0.01 | **SURVIVES** |
| framework-ID (A↔C) | 0.997 | **0.46** | −0.51 | COLLAPSES → demoted (§8.1) |

**All four core recovery claims are unchanged under the same control that demolished framework-ID.**
The interleaved macro-F1s land within noise of the committed baselines (CIs overlap; |Δ| ≤ 0.06),
while framework-ID falls from 0.997 to chance. This is the crux of the paper's internal validity: a
capture-session confound inflates classification *between separately-captured classes* (framework-ID:
A, B, C each in their own session) but **cannot** create the core signal, because workflow / role /
topology labels all co-occur *within the same continuous capture* — so no session artefact tracks the
label. The control confirms exactly that, empirically. The attack recovers **genuine call-structure
traffic shape**, not the setup it was captured in.

**What about the remaining claims?** The *transfer* results — model-vs-logic (§2), runtime-invariance
(§3), and cross-instance role transfer (§9) — are **already conservative** with respect to this
confound and need no separate control: they *train on one capture and test on a different one*, so a
capture-session artefact would make train/test **mismatch** and *degrade* transfer, never inflate it.
That A→C transfers at 0.64/0.83 and the cross-instance coordinator layer at 0.87–0.91 (§9a) *across*
sessions is therefore a floor, not a confound-inflated ceiling. The confound direction matters: it inflates *separability
between separately-captured classes* (framework-ID) and *lowers* cross-capture *transfer* — so §8.2's
within-capture controls plus the cross-capture transfers together cover every classification claim.

> **A note on scientific honesty (why this section exists).** We went looking for the confound that
> would sink this paper, found one (framework-ID), and report it as a clean negative — then showed
> with the identical instrument that the core claims are unaffected. That asymmetry (one auxiliary
> claim demoted, three core claims corroborated) is *stronger* evidence than an unaudited table of
> high numbers: it demonstrates the numbers that remain are not capture artefacts.

---

## 9. Cross-instance transfer on a2a_mcp (Phase 2 — the deployable-attack test)

§7.1 showed the role fingerprint *replicates* when we re-train on a2a_mcp. The stronger,
deployability-relevant question is **transfer**: can an attacker train a role classifier on
**their own** copy of a popular framework and read roles off a **victim's independent copy**?
We stood up a **second, independent instance** of a2a_mcp and tested train-on-one → test-on-other,
both directions. `data/results/cross_instance_transfer.json` +
[`figures/cross_instance_transfer.png`](data/results/figures/cross_instance_transfer.png).

**Instance-2 independence axes:** different LLM (`gemini-2.0-flash` vs instance-1's
`gemini-2.5-flash`, via a2a_mcp's own `LITELLM_MODEL`), reworded prompts, separate session. Method
is identical to §7.1 (35-dim per-agent traffic-shape vector; **port is the label, never a feature**;
GBT `_transfer`, macro-F1 + bootstrap 95% CI, seed 42). Instance 2 = **82 trips**
(`data/raw_offtheshelf_inst2`, gemini-2.0-flash). Weaker direction is the headline; verdict fields
match the numbers.

### 9a. Coordinator layer (natural both instances) — DEPLOYABLE

The three coordinators (`mcp`/`orchestrator`/`planner`) fire on **every** trip, so their samples are
natural in both instances. Cross-instance transfer, both directions:

| Direction | 3-way coordinator macro-F1 [95% CI] | chance |
|---|---|---|
| train inst-1 → test inst-2 | **0.866 [0.821, 0.907]** | 0.333 |
| train inst-2 → test inst-1 | 0.996 [0.988, 1.000] | 0.333 |

**Verdict (§4): DEPLOYABLE ATTACK** — weaker direction **0.866**, **group-bootstrap CI [0.808, 0.916]**
far clear of chance, above the ≥0.70 bar. Train on your own copy of a2a_mcp, read the coordinator roles
off an independent victim copy despite different LLM/prompts/session. (An earlier 67-trip instance-2
gave 0.912; the number is stable as instance-2 grows.) This is the paper's clean deployable-attack result.

> **Interval convention (C4).** Role samples are clustered by **trip**, so the interval reported above is
> a **cluster (group) bootstrap** — resampling whole trips, not individual flows. The i.i.d. interval
> ([0.821, 0.907]) is **24% narrower** and therefore over-confident, because flows from one trip are
> correlated. The point estimate is identical and **the DEPLOYABLE verdict is unchanged** (floor 0.808 ≫
> chance 0.333). `data/results/group_bootstrap_check.json`.

**Volume ablation — the result is behavioural, not connection-volume.** Re-running the same transfer on a
**shape-only** feature set (16/35 dims: per-packet size shape, IAT/duration/gap timing, burst-duration
shape, and ratios — *dropping* all 19 raw count/byte-magnitude dims: packet/byte totals, cumulative-byte
trajectory, burst byte magnitudes, and event counts) barely moves it: weaker direction **0.840
[0.792, 0.884]**, still DEPLOYABLE and far above the ≥0.70 bar (`coordinator_shape_only_ablation`). Unlike
the framework-ID timing ablation, where removing the signal collapsed it (0.46→0.38), removing volume
here costs only 0.026 — the coordinator layer is recovered from per-agent traffic **shape/timing**, not
the connection-volume signal already demoted to a topology baseline.

### 9b. Full six roles (with specialists) — PARTIAL, and honestly **driver-confounded**

To test the three **specialists** (air/hotel/car — the structurally-identical leaves, i.e. the genuine
*behavioural* test), we needed ≥15 samples each in instance-2. a2a_mcp's LLM-planned routing fans out
to specialists only ~6% of natural trips, so we topped up instance-2 with a **fan-out-boosted driver**
(`drive_orch_boost.py` + fully-specified queries; the original ~6% was a Tokyo-hardcoded canned answer
sabotaging the planner on other destinations). This reached air/hotel/car = 15/15/15 for **~$0.75**.

| Direction | 6-way macro-F1 [95% CI] | chance |
|---|---|---|
| train inst-1 → test inst-2 | 0.682 [0.634, 0.723] | 0.167 |
| train inst-2 → test inst-1 | **0.605 [0.555, 0.657]** | 0.167 |

**Verdict (§4): PARTIAL** — weaker direction **0.605** (0.40–0.70), well above the 0.167 chance line
but below the 0.70 deployable bar. **We report the band the number lands in, not the one we hoped for.**

> **⚠ This 6-way is CONFOUNDED by the boosted driver — read before interpreting.** The boosted driver
> is an *added* axis of difference: instance-2's specialists were collected under forced full-service
> prompts, instance-1's under natural routing. A per-agent feature-distribution check (inst-1 natural
> vs inst-2 boosted; standardized mean difference) finds them **not comparable — 0/3 specialists**
> (median |SMD| = air 3.0, hotel 0.95, car 1.3; mean-vector cosine 0.97–0.99, i.e. same direction but
> a real magnitude/scale shift, consistent with the more verbose forced prompts). So the drop below
> 0.70 **cannot be cleanly attributed to "behaviour doesn't transfer"** — the train-natural/test-forced
> distribution shift is a live candidate contributor. We name it rather than fold it into a tidy
> "structure transfers, behaviour is structure-gated" story, and we **resolve it in §9b′** below.
> `specialist_distribution_check` + `driver_confound_interpretation` in the JSON.

### 9b′. De-confounded natural re-run — the distribution check is the gate

We removed the driver confound: a **bug-fixed-but-not-forced** natural driver (`drive_orch_natural_fixed.py`
— destination-agnostic so it doesn't hit the ~6% Tokyo bug, but *not* completion-forcing so it doesn't
shift distributions) driven by **instance-2's own reworded prompt** and LLM (gemini-2.0-flash), so
instance-2 stays internally self-consistent. Natural fan-out was **26.7%** (vs the bugged 6% and the
forced ~90%), reaching air/hotel/car = 15/15/15 for **~$1.57** (well under the $15 cap; probe-then-project
gate armed). `data/results/cross_instance_transfer_natural.json`.

**The specialist distribution check is the gate, reported before the verdict.** Natural inst-1 vs natural
inst-2 specialists: still **0/3 comparable**, but the gap **shrinks** vs the boosted run (median |SMD|:
air 3.02→**2.54**, car 1.32→**1.00**, hotel 0.95→0.93) — evidence the boosted driver *was* contributing,
now removed. A residual gap persists.

| Run | 6-way weaker-dir macro-F1 | specialists comparable |
|---|---|---|
| boosted (§9b) | 0.605 | 0/3 (median|SMD| air 3.0) |
| **natural (§9b′)** | **0.594** | 0/3 (air 2.5 — gap shrunk) |

**Verdict (§4): PARTIAL, and now de-confounded (no re-stamp).** Removing the driver barely moved the
number (0.605→**0.594**) — so the boosted driver was **not** masking a clean ≥0.70 positive. But because
the distributions are still not comparable *with the driver gone*, the residual difference is now the
**legitimate independence axes** (different LLM gemini-2.0-flash vs 2.5-flash, separate session), **not a
driver artefact**. Per the pre-registered gate, a sub-0.70 here is therefore **partly LLM/session-
attributable — not a clean "specialist behaviour does not transfer."** The honest status: the specialist
cross-instance transfer is **genuinely PARTIAL (~0.59), robust to the driver confound**, with the residual
gap owed to legitimate cross-instance independence. (The coordinator layer in the same natural run
**corroborates §9a even more strongly: weaker 0.942, shape-only 0.942 — DEPLOYABLE**.)

**Coordinator-vs-specialist (2-way):** weaker direction **0.758 [0.696, 0.813]**, stronger 0.875 —
transfers, but flagged **partly structural** (hub-vs-leaf rides on connection volume like topology).

---

## 10. Cross-framework replication + transfer on **AutoGen** (Task 3 pilot — independent, non-A2A)

The prior cross-framework question (§7 future work) was *does the attack generalize past A2A?* We
stood up an **independently-structured, networked** system: AutoGen's **distributed gRPC runtime**
(autogen-core 0.7.5) — a message-routing host with orchestrator/worker agents as gRPC *clients*
(star-through-host), a fundamentally different protocol/serialization/control-flow than a2a's
Starlette/JSON-RPC/SSE per-agent servers. An orchestrator routes sub-tasks to three specialists
(researcher/writer/reviewer), each calling a **local ollama llama3.2:3b** (no API spend). Each
agent runs in its own process ⇒ one gRPC TCP flow with a distinct ephemeral port; capture is
lo0:50051 at 96-byte snaplen, role attributed by source-port sidecar (**port = label, never a
feature**). Same 35-dim per-agent representation. n = 120 (30 trips × 4 roles), 25 topics for
group-safe CV. `data/results/cross_framework_autogen.json`.

**(a) The attack REPLICATES on AutoGen — and it is behavioural.** 4-way role recovery
(orchestrator/researcher/writer/reviewer) **macro-F1 0.966 [0.931, 0.992]** (chance 0.25). Under
the Task-1 **volume ablation** (drop all raw count/byte-magnitude dims, keep 16/35 shape+timing+
ratio features) it is **unchanged at 0.966** — the fingerprint is per-agent *behaviour* (orchestrator
dur ≈ 3.8 s / bytes-out-ratio 0.39 hub; writer longest response; reviewer 0.35 s one-liner), not
connection volume. The vulnerability class is **not A2A-specific**.

**(b) But a trained classifier does NOT portably transfer across frameworks.** On the only shared
label space (coordinator-vs-specialist), cross-framework transfer is **asymmetric**: AutoGen→a2a_mcp
**0.786 [0.712, 0.851]**, but a2a_mcp→AutoGen **0.429** (a2a's HTTP/JSON-RPC coordinators don't match
AutoGen's gRPC orchestrator signature → predicts all-specialist). **Weaker direction 0.429 → BOUNDED**
(§4, no re-stamp; shape-only identical). This **bounds portability, not the vulnerability**, and is
consistent with the paper's **implementation-specificity** thesis: every framework is fingerprintable
when you retrain on it, but the fingerprint is framework-specific — a defender cannot assume an
attacker's off-the-shelf model, yet an attacker who retrains on the target framework succeeds.

*Pilot caveats: single deployment topology, one LLM, AutoGen's own specialist roles (so a **fine**
a2a↔AutoGen 6-way label transfer is undefined — disjoint taxonomies); CrewAI, a second topology,
and a second LLM are the natural robustness extensions.*

---

## 11. Agentic-vs-agentic detection — A2A vs AutoGen (closes the §7 "open problem")

The referee's sharpest deployability point: the AUC 1.0 detector (§7) separates A2A only from
**non-agentic** negatives, so "detect an A2A system in the wild" was open. With real AutoGen traffic
we run the honest binary detector — **A2A flows (positive) vs AutoGen flows (negative)**, same 35-dim
per-flow shape vector, **GBT**, group-safe 5-fold StratifiedGroupKFold by trip, percentile bootstrap
95% CI, shape-only ablation. n = 501 A2A flows (150 trips) vs 120 AutoGen flows (25 trips).
`data/results/agentic_detection.json` (+ `figures/agentic_detection.png`).

**AUROC 1.000 [1.000, 1.000], macro-F1 1.000 — STRONG**, and it **survives the shape-only ablation
(AUROC 1.000)** — not merely raw connection volume. **Honest scoping (load-bearing):** the sanity check
shows the separation is driven by **transport-level packet-size percentiles** (`p25/p75` out-sizes,
near-constant per framework — gRPC/HTTP2 framing vs HTTP/SSE framing; `top_discriminating_features`).
So this **converts detection from "open problem / only vs non-agentic negatives" to "A2A is
distinguishable from an INDEPENDENT agentic framework"** — real and useful — but it is substantially a
**transport fingerprint**, so it does **not** by itself show A2A is separable from a **same-transport**
agentic framework (another SSE-over-HTTP system such as CrewAI). **§13 removes that transport confound**
(A2A vs CrewAI over an identical a2a-sdk JSON-RPC+SSE transport). Reported as-is.

## 12. Background-multiplexing degradation — the cost of losing port isolation

The detection/role results assume the observer isolates each agent's flows by port. Real deployments
multiplex. We embed A2A flows amid genuine background flows (web/file/REST/JSON-RPC/multi-REST/LLM-direct)
and measure degradation vs the contamination ratio **ρ = background flows per agent flow** (ρ=0 clean,
ρ→∞ background-dominated window). GBT, group-safe CV, percentile bootstrap CI.
`data/results/mixing_degradation.json` (+ `figures/mixing_degradation.png`).

- **Detection:** per-flow A2A-vs-background AUROC 1.000 (shape-only 1.000), recall 1.000 at a **fixed 5%
  background-FPR operating point**. Recall is threshold-fixed, but **precision falls predictably with ρ**
  as un-isolated background flows accumulate false positives at that 5% rate: **1.00 (ρ=0) → 0.83 (ρ=4)
  → 0.71 (ρ=8) → 0.38 (ρ=32)**. That is the honest, quantified cost of multiplexing.
- **Role recovery:** with an explicit **background reject class**, role macro-F1 **0.957 [0.909, 0.989]**
  and background→role **leak 0.00** — role attribution is **robust to this contamination**. *Honest caveat:*
  the background is **non-agentic and structurally distinct**, so it is trivially rejected; **agentic**
  distractors (cf. §11) would be a sterner test and likely leak more.

**Honest scope (threat-model, not experiment):** this is a first-order flow-level proxy — an observer
who can still **segment** flows but not **attribute** them. It does **not** address the deepest case —
agents behind a **shared reverse proxy on :443 with no distinct observable ports**, where flows aren't
even separable. That is **architectural** and is handled by the paper's threat-model framing, not by
this experiment.

## 13. Same-transport agentic detection — A2A vs **CrewAI** (removes §11's transport confound)

§11 separated A2A from AutoGen at AUROC 1.0, but the sanity scan showed the driver was a **transport**
tell (SSE-over-HTTP vs gRPC/HTTP2), so the harder question — can an observer distinguish A2A from
another agentic framework on the **same** transport — stayed open. CrewAI answers it. CrewAI's *own*
native remote-agent transport **is the A2A protocol** (`crewai.a2a`: JSON-RPC 2.0 + SSE over HTTP), so
we serve CrewAI specialists (planner + air/hotel/car, travel domain matched to the positives) on the
**a2a-sdk stack pinned to 0.3.26 — the identical server library *and version* the a2a_mcp positives
use** — each wrapping a genuine CrewAI `Agent/Task/Crew` brain (local ollama). **On-wire capture
confirms identical transport on both sides** (`POST /`, `HTTP/1.1 200`, `content-type: text/event-stream`,
`jsonrpc` / `message/stream`, `"kind":"task"`). Same 35-dim per-flow vector, **same extractor for both
sides**, GBT, group-safe 5-fold StratifiedGroupKFold by trip, percentile bootstrap 95% CI, shape-only
ablation, mandatory single-feature sanity scan. n = 501 A2A flows (150 trips) vs **120 CrewAI flows
(30 trips)**. `data/results/crewai_detection.json` (+ `figures/crewai_detection.png`).

**AUROC 1.000 [1.000, 1.000], macro-F1 1.000, survives the shape-only ablation (AUROC 1.000,
macro-F1 0.992). Verdict SCOPED** (corrected §4 band). With transport **provably held identical**,
§11's *transport*-driven escape hatch is excluded by construction — so an observer who cannot lean on
transport can still separate these agentic systems from application-layer traffic shape. **But that is
same-transport separability, not framework-code isolation**, and the number does not license the stronger
reading.

**Honest scope (load-bearing — why SCOPED, not CLOSED).** The mandatory sanity scan shows **every top
driver is an application-layer VOLUME/burst-count feature** (`n_small_inbound` 0.991, `n_response_bursts`
0.987, `n_bursts` 0.984, `n_pkts_out/in` 0.981). Those are exactly what **three uncontrolled confounds**
move: the positives use **cloud gemini-2.5-flash** while CrewAI uses **local ollama/llama3.2:3b** (different
response sizes/chunking — the *dominant* confound, not cheaply removable since the a2a_mcp positive set is
frozen canonical); a2a_mcp's orchestrator runs **multi-turn** clarifying Q&A vs CrewAI's **single-turn**
calls; and the topologies differ (**6-agent vs 4-agent**). **The AUROC 1.0 is fully explicable without any
framework-code signal at all** — the experiment traded one confound (transport) for three. Shape-only
survival does *not* rescue it: the kept features are sizes/durations/IATs — precisely what cloud-vs-local
LLM and multi- vs single-turn shift. So this is a **same-transport *detectability*** result — *agentic
systems are distinguishable on an identical transport via application-layer volume* — **not** a controlled
isolation of framework code. That stronger "framework identity with all else equal" claim is addressed by
the **matched pair in §13.1**; `confounds` and `sanity_scan_reading` in the JSON record the confounds in full.

### 13.1 Matched pair — **deployment A vs CrewAI** (LLM + transport + agents + host + domain + roles + topology all held equal)

§13 was SCOPED because the LLM backend, interaction pattern and agent count differed. This removes
all of them. **Positive = deployment A** (our own orchestrator/executor/retriever/validator, chained,
a2a-sdk 0.3.26 SSE, local ollama/llama3.2:3b). **Negative = a CrewAI deployment built to match A
exactly**: the *same 4 roles with the same per-role instructions*, chained the *same way* over the
*same a2a-sdk 0.3.26 SSE stack*, *same local ollama/llama3.2:3b*, *same host/lo0*, driven on **A's own
prompts** (10 per workflow across code_review / data_analysis / research_retrieval / support_triage,
chain topology). Controlled: transport, server library+version, LLM, agent count, host, task domain,
role semantics, call topology. What remains is **how each framework emits its calls** (plus a
chain-forwarding-format difference we wired — disclosed below); it is **not** "the framework's internal
machinery alone." Same extractor both sides, GBT, group-safe CV by trip, bootstrap CI, shape-only +
single-feature scan. n = 109 A flows (40 trips) vs 156 CrewAI flows (39 trips).
`data/results/crewai_matched_detection.json` (+ `figures/crewai_matched_detection.png`).

**AUROC 1.000 [1.000, 1.000], macro-F1 1.000, shape-only 1.000. Verdict CLOSED.** With every major
confound held equal, deployment A and CrewAI are still **perfectly separable** — so with all major
confounds controlled an observer can still tell the two implementations apart from traffic shape, and
the same-transport detection problem **closes for this pair**. But read the claim narrowly (below).

**Why (the driver, verified).** The separation is complete and size-based (zero overlap on outbound-size
features; `std_sz_out`, `pkt_size_asymmetry`, `p75_sz_out` all single-feature AUROC 1.0). The mechanism
is a **response-emission difference**: deployment A **streams token-by-token** (`llm_stream` → many
*small* outbound SSE packets, `mean_sz_out`≈90), whereas **CrewAI's `Crew.kickoff()` blocks and returns
one large final artifact** (*few large* outbound packets, `mean_sz_out`≈565, up to 1798) — flipping the
packet-size asymmetry (A large-in/small-out; CrewAI large-out/small-in). This is **not** the LLM/topology
confound that scoped §13. **Two honest caveats.** (1) *Streaming-vs-blocking is a **configuration**
property, not framework identity.* CrewAI **can** stream (`LLM(stream=True)`, step callbacks) and A could
have been written to block, so a **streaming-configured CrewAI might be indistinguishable from A**. The
defensible claim is therefore **"implementations whose response-emission behaviour differs (streaming vs
blocking) are trivially separable — here the *default idiomatic* difference between these two
frameworks,"** *not* "framework identity is detectable." (This actually fits the thesis: *how* an
implementation emits its calls is itself part of the call structure the attack reads.) (2) A *secondary*
contributor is the chain-forwarding format — A forwards "previous output + original instruction", our
CrewAI chain forwards just the upstream output; that wiring was ours (a contributor we control, disclosed),
and a fully forwarding-matched replication is future work. Net: a *systematic, size-based* separation
driven by a configuration-level emission difference — not a subtle behavioural fingerprint.
`driver_mechanism` and `scope_streaming_is_configuration` in the JSON record this in full.

---

## Bottom line

A passive observer, seeing only encrypted-traffic metadata, recovers **workflow** (GBT
0.71) and **agent role** (0.86) far above chance. The signal is **caused by the inter-agent
call structure** — invariant to the **LLM model** (0.68→0.59) and to the **orchestration
runtime** (A→C 0.64), destroyed by changing the **structure** (A→B 0.29). All four core
recovery claims are **confound-controlled**: under a same-session interleaved capture (§8.2) —
the very control that collapses the auxiliary framework-ID number from 0.997 to chance — workflow
(0.65), role (0.89), topology (1.00) and parallelism (1.00) are **unchanged** (|Δ| ≤ 0.06), proving
the signal is genuine call-structure traffic shape, not a capture artefact. It holds on **real
WAN traffic** in-domain, and **survives current defenses** (~70% F1 retained at ~30%
bandwidth). Crucially, the **role fingerprint REPLICATES on a system we did not build**
(a2a_mcp: **6-way role macro-F1 0.906** from per-agent traffic shape — the behavioral result;
the coordinator-vs-specialist 1.000 is *partly structural*, and topology recovers as hub-and-spoke
from headers alone) — and, on a **second independent instance** of that framework (different LLM,
prompts, session), a role classifier **transfers across instances at macro-F1 0.87–0.91** on the
always-present coordinator layer (§9a, weaker direction, ≥0.70 §4 bar — **DEPLOYABLE**) — the
deployable-attack result: train on your own copy, read roles off a victim's. The full six-role
transfer including the sparse specialists is **partial (~0.60)**; a **de-confounded natural re-run**
(§9b′ — bug-fixed-but-not-forced driver, instance-2's own config, 15/15/15 for ~$1.57) leaves it
essentially unchanged (0.605→**0.594**), so the driver was not masking a positive — the specialist
transfer is **genuinely PARTIAL, with the residual gap owed to the legitimate different-LLM/session
independence, not a driver artefact**. A2A-vs-background *detection* separates
a2a_mcp at AUC 1.000 against non-agentic negatives; **§11 extends this to an independent agentic
framework** (A2A vs AutoGen, AUC 1.0 — though transport-driven), and **§13 removes that transport
confound** (A2A vs CrewAI over an **identical** a2a-sdk JSON-RPC+SSE transport, AUROC 1.0 **SCOPED** —
transport is genuinely excluded, but the separation is driven by application-layer volume that the
uncontrolled LLM/interaction/topology differences move, so it shows same-transport **detectability**,
not framework-code isolation), and the **§13.1 matched pair** closes it cleanly (deployment A vs a
CrewAI built to match A on LLM, transport, agents, host, domain, roles and topology — **AUROC 1.0
CLOSED**: with all major confounds controlled the two implementations are still separable, driven by a
**streaming-vs-blocking** response-emission difference — a *configuration* property, so the claim is
"implementations differing in emission behaviour are separable," not "framework identity is detectable"),
and **§12 quantifies the loss-of-port-isolation cost** (detection precision
1.00→0.38 as background rises to 32:1 at a 5% FPR operating point; role recovery robust to non-agentic
contamination given a reject stage). The deepest no-observable-ports case is threat-model framing, not a claim. A **same-session
interleaved control** (§8.1) independently confirms the runtime-invariance: the apparent A↔C
implementation fingerprint **collapses from 0.997 to chance (0.46)** once the capture-session
confound is removed — so within-family framework/implementation ID is a batch artefact, honestly
**demoted**, and the recon claim is narrowed to telling apart *structurally-distinct* frameworks
(which is topology recovery, not a session artefact).

**Not claimed:** **within-family framework/implementation identification** (§8.1 — negative under
the same-session control; the separate-session 0.998 was batch-confounded), LAN→WAN transfer,
cross-*framework label* transfer (A↔a2a_mcp taxonomies are disjoint — role *replicates*
independently, but shared-label transfer is undefined; only a partial coordinator-vs-specialist
transfer at 0.76), **≥0.70 cross-instance transfer of the specialist (leaf) roles** (§9b/§9b′ — the
full 6-way lands at **PARTIAL ~0.60 and stays there when de-confounded** by a natural re-collection;
the specialists transfer only partially, with the residual owed to legitimate different-LLM/session
independence, not — after §9b′ — a driver artefact; the coordinator layer transfers **deployably** at
§9a/§9b′), a workflow fingerprint on an external (LLM-planned)
system, **portable** cross-*framework* classifier transfer (§10 — the attack *replicates* on
AutoGen behaviourally at 0.966, but a model trained on one framework does **not** transfer to the
other: weaker direction 0.429, BOUNDED — the fingerprint is framework-specific), strong open-set
rejection of novel workflows/roles, a detector robust to hard agentic negatives — all future work.
**Same-transport separability with all else equal is now CLAIMED, with scope:** §13 A2A-vs-CrewAI is
SCOPED (same-transport *detectability* carrying an LLM/interaction confound), but the **§13.1 matched
pair** (deployment A vs a matched CrewAI, LLM/transport/agents/host/domain/roles/topology all held equal)
**closes it — AUROC 1.0 CLOSED**. Read narrowly: the driver is a **streaming-vs-blocking** response-emission
difference, which is a **configuration** property (CrewAI can stream; A could block) — so the claim is
*"implementations whose emission behaviour differs are trivially separable, the default idiomatic gap
between these two frameworks,"* **not** "framework identity is an immutable, always-detectable signature."
A disclosed secondary chain-forwarding-wiring contributor and a fully forwarding-matched replication are
future work.
*(Deployment C remains a runtime-invariance control, not a generalization result; §10 is the actual
independent-framework data point.)*
