# Prep — Task 2 (clean natural-specialist test) & Task 3 (AutoGen/CrewAI transfer)

Task 1 (the §9a volume ablation) is **done** and folded: coordinator shape-only weaker
direction **0.840 [0.792, 0.884]** DEPLOYABLE — behavioural, not volume-driven. This doc
prepares Tasks 2 and 3 so either can launch on one word. **Nothing here has been run;
Task 2 costs money and is gated on your explicit go.**

---

## Task 2 — clean natural-specialist cross-instance test — ✅ DONE (~$1.57)

> **STATUS (done): the de-confound ran and resolved §9b.** Bug-fixed-but-not-forced natural driver
> (`drive_orch_natural_fixed.py`) + instance-2's own config → 26.7% fan-out (vs bugged 6% / forced ~90%),
> 15/15/15 specialists for ~$1.57 (probe projected $1.89, well under the $15 cap).
> **The distribution check was the gate, reported before the verdict:** still 0/3 comparable but the gap
> **shrank** (air |SMD| 3.02→2.54, car 1.32→1.00). The 6-way barely moved (0.605→**0.594**) → the driver
> was **not** masking a positive; with it removed, the residual gap is the **legitimate LLM/session
> independence**, so the sub-0.70 is **partly LLM/session-attributable, NOT a clean "behaviour doesn't
> transfer."** Coordinator natural corroborates §9a at 0.942. Results: `cross_instance_transfer_natural.json`,
> RESULTS.md §9b′. The original plan follows for reference.

## Task 2 (original plan) — clean natural-specialist cross-instance test (PAID, ~$6–12, prepared)

### What it resolves
The committed §9b 6-way (weaker 0.605 PARTIAL) is **driver-confounded**: instance-2's
specialists were topped up with the fan-out-**boosted** driver (fully-specified prompts +
completion-forcing answer), and their feature distributions came back **0/3 comparable** to
instance-1's natural specialists (`specialist_distribution_check`). So the sub-0.70 drop
cannot be cleanly read as "specialist behaviour doesn't transfer" — the driver is a candidate
cause. This task collects instance-2 specialists the **same natural way instance-1 was
collected**, removing that confound, then re-runs the 6-way.

### The de-confound (exact mirror of instance-1's method)
| axis | instance-1 (natural) | Task-2 instance-2 (natural) | boosted 6-way (confounded) |
|---|---|---|---|
| driver | `drive_orch.py` | `drive_orch.py` | `drive_orch_boost.py` |
| prompt | "Plan a trip … Book round-trip flights, a hotel …, and a rental car." | **same** | "…BOOK ALL THREE now. Do not ask clarifying questions." |
| answer to clarifying Q | natural | natural | completion-forcing |
| LLM | gemini-2.5-flash | gemini-2.0-flash | gemini-2.0-flash |
Only the intended independence axes (LLM, dates/party, session) differ — the collection
*method* matches, so specialist distributions should be comparable and the transfer clean.

### Cost reality (why the probe gate matters)
Natural fan-out to specialists is **~6–11%** of trips (instance-1: 17/150 ≈ 11%; instance-2
natural so far: 4/67 ≈ 6%). Fan-out is all-or-nothing (air/hotel/car fire together → the
17/17/17 and 4/4/4 symmetry). To reach ≥15 each **naturally**:

- **Seed already collected:** the script reuses the 67 existing natural instance-2 trips
  (4/4/4 natural specialists), so we only need **+11 each**.
- At rate *r*, top-up ≈ `11 / r` trips × ~$0.045:
  - r = 0.10 → ~110 trips → **~$5**
  - r = 0.07 → ~157 trips → **~$7**
  - r = 0.05 → ~220 trips → **~$10**
  - r = 0.04 → ~275 trips → **~$12** (near cap)
- **Probe-then-project gate (hard):** after `PROBE_N=30` new trips the script measures the
  actual rate, projects trips + cost to hit 15, and **STOPS if the projection ≥ the cap** —
  it never brute-forces. If the probe yields 0 specialists it stops and reports (no blind spend).

### Guardrails (baked into `scripts/collect_offtheshelf_natural.sh`)
- Hard cap `BUDGET_USD=15` (set lower if you want; probe gate stops early on a bad rate).
- `TARGET_SPECIALISTS=15`, `MIN_N=10` at eval (a 4–5-sample role never enters).
- Fresh dir `data/raw_offtheshelf_inst2_natural/` — **additive**, touches no committed data.
- 3-consecutive-fail stop; `RequestsPerDay` 429 stop; `PACE=12` inter-trip (avoids RPM throttle
  that degraded the last run).
- Truth = Google Cloud billing; the est is a guard, not the meter.

### Launch (only on your go)
```bash
BUDGET_USD=15 TARGET_SPECIALISTS=15 PROBE_N=30 PACE=12 \
  bash scripts/collect_offtheshelf_natural.sh
# then re-run the 6-way on the clean natural set:
MIN_N=10 venv/bin/python scripts/evaluate_cross_instance_transfer.py \
  --inst2 data/raw_offtheshelf_inst2_natural
```

### Verdict rule (no re-stamp — decided before the run)
Re-run writes a fresh `cross_instance_transfer.json`; the specialist distributions should now
be **comparable** (that is itself the headline the confound check demanded). Then the 6-way
weaker-direction number lands in a §4 band and the field matches it:
- **≥ 0.70** → specialist *behaviour* transfers across instances too → §9b upgrades to a clean
  full-6-way deployable result (the driver was the whole story).
- **0.40–0.70** → PARTIAL **and now unconfounded** → honest "coordinators transfer deployably,
  specialists partially — real instance drift, not a driver artefact."
- **< 0.40** → BOUNDED → specialists genuinely don't transfer across instances; report as-is.
Either way the coordinator §9a result (0.866 / shape-only 0.840) is untouched and stands.

### Decision for you
Worth it **only** if you want §9b's specialist question resolved cleanly. The paper already
stands: §9a is deployable (now with the volume ablation), §9b is honestly reported as
driver-confounded. This is a completeness call, ~$6–12, prepared and one command away.

---

## Task 3 — AutoGen cross-framework replication + transfer — ✅ PILOT DONE

> **STATUS UPDATE (this session): the AutoGen pilot is built and run.** Findings below;
> `data/results/cross_framework_autogen.json`, RESULTS.md §10, deployment in `~/autogen-xframework/`,
> analysis in `scripts/evaluate_cross_framework_autogen.py`. The original plan follows for context
> and for the CrewAI / robustness extensions that remain.

### Results (n=120, 30 trips × 4 roles, 25 topics, local ollama — no spend)
- **(a) The attack REPLICATES on AutoGen, behaviourally.** 4-way role recovery **0.966 [0.931, 0.992]**
  (chance 0.25); **volume-ablated (16/35 shape features) still 0.966** → per-agent behaviour, not
  connection volume. An independent gRPC framework is just as fingerprintable → the vulnerability
  class is **not A2A-specific**.
- **(b) A trained classifier does NOT portably transfer across frameworks.** Coordinator-vs-specialist
  transfer is asymmetric — AutoGen→a2a **0.786**, a2a→AutoGen **0.429**; weaker **0.429 → BOUNDED**.
  Bounds *portability*, not the vulnerability; consistent with the implementation-specificity thesis.

### What this cost / how it was de-risked
Zero spend (local ollama). The make-or-break gate — a genuinely *networked* non-A2A runtime with
per-role flow labeling — was solved: AutoGen distributed gRPC (star-through-host), each agent in its
own process ⇒ one TCP flow ⇒ `lsof` source-port→role map (port = label only). Reused the a2a feature
pipeline + Task-1 shape mask verbatim.

### Remaining (optional robustness extensions, not blocking the paper)
- **CrewAI** as a third framework (CrewAI has no built-in distributed runtime → each agent must be
  wrapped as a networked service; more custom than AutoGen's gRPC host-worker).
- A **second deployment topology** and a **second LLM** on AutoGen (robustness of the 0.966).
- A **fine** a2a↔AutoGen label transfer is **undefined** (disjoint specialist taxonomies) — the
  coordinator-vs-specialist abstraction is the only shared label space; that transfer is done (b).

---

### Original plan (for CrewAI / extensions)

The one genuine generalization gap. `docs/CROSS_FRAMEWORK_PLAN.md` already flags this as §7
future work: deployment C (LangGraph) is a **runtime-invariance control**, not a generalization
result (it reuses A's specialists/structure). A truly *independently-structured*, **networked,
non-A2A** framework with a comparable taxonomy is what converts "implementation-specific" into
"the attack class generalizes across frameworks."

### Why AutoGen/CrewAI (not another A2A sample)
- **Networked** — agents talk over sockets we can sniff at 96-byte snaplen (the whole method
  needs on-wire inter-agent traffic; single-process frameworks give nothing to capture).
- **Independently structured** — different message protocol, serialization, and control flow
  than a2a-sdk (Starlette/JSON-RPC/SSE), so a positive transfer is a real cross-framework claim,
  not a runtime re-skin.
- **Comparable taxonomy** — both express coordinator/orchestrator + worker/specialist roles, so
  the label space maps (see mapping below).

### Design (build order, all LAN-local, no spend)
1. **Networked deployment.** Stand up AutoGen (or CrewAI) in a **multi-process / socket** config
   — not the default in-process one. AutoGen: `GrpcWorkerAgentRuntime` / distributed runtime over
   gRPC. CrewAI: agents as separate networked services. Confirm inter-agent packets appear on `lo`
   before collecting anything (a single-process run is a hard blocker → report and stop).
2. **Taxonomy mapping (label bridge).**
   | our role | AutoGen | CrewAI |
   |---|---|---|
   | orchestrator/coordinator | GroupChatManager / orchestrator agent | Crew manager / hierarchical process manager |
   | planner | planning agent | planner task-agent |
   | specialist/worker (leaf) | assistant/tool worker agents | worker agents (tools) |
   Keep the mapping **coarse and defensible** (coordinator vs specialist at minimum); do not
   invent fine roles that don't exist in the other framework.
3. **Collect** with the existing pipeline unchanged (`tcpdump -s 96`, `features/per_flow.py`,
   35-dim per-agent vectors; port is a LABEL never a feature). Same prompt-group discipline for
   group-safe CV. Target ≥15 samples per mapped role.
4. **Transfer eval** — reuse the `_transfer` pattern: fit GBT on a2a_mcp roles, predict the
   AutoGen/CrewAI mapped roles and vice-versa; weaker-direction macro-F1 + bootstrap CI; §4 bands.
   New script `scripts/evaluate_cross_framework_role_transfer.py` (mirror
   `evaluate_cross_instance_transfer.py`; additive JSON `cross_framework_role_transfer.json`).
5. **Volume ablation** on any positive (reuse `_SHAPE_MASK` from Task 1) — same "behavioural vs
   volume" gate, so a cross-framework positive can't be dismissed as a hub-volume artefact.

### Verdict framing (pre-registered)
- Weaker-direction **≥ 0.70** shape-only → **cross-framework generalization** — the strongest
  possible version of the paper's claim; §7 becomes a result, venue ceiling rises.
- **0.40–0.70** → the fingerprint is *partly* framework-portable; honest partial.
- **< 0.40** → confirms **implementation-specificity across frameworks** — still a clean,
  publishable finding (it bounds the attack and matches the current cross-deployment story).
All three outcomes are paper-positive; there is no way to lose this experiment, only to spend
weeks on it.

### Effort / blockers
- **Biggest risk:** getting a genuinely *networked* multi-agent config stood up (default configs
  are in-process → nothing to sniff). Validate on-wire traffic **first**, before any collection.
- Estimate: multi-week (framework setup + collection + taxonomy validation + eval). No API spend
  if run on the local 3B/ollama models.
- **Do not start** unless aiming above mid-tier and the schedule allows. Not needed for the
  current paper — §9a (deployable, volume-ablated) + confound audit already carry it.

---

## Standing rules (applied to both)
Additive only (`git diff --stat` on committed `data/results/*.json` must be empty except this
session's own `cross_instance_transfer.json`); no re-stamping (verdict field matches the band the
number lands in); group-safe CV + bootstrap CIs; port never a feature; blocked > fabricated
(missing data/setup/budget → report and stop); 30/30 tests pass; git stays yours.
