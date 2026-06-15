#!/usr/bin/env bash
#
# Full Option-B rebuild on the official a2a-sdk + SSE stack.
#
# Re-collects every dataset with the new SDK/SSE agents, re-extracts features,
# and re-runs the entire result suite (closed-world, cross-deployment,
# model-vs-logic, open-world background, and LIVE C4 defenses).
#
# Usage:
#   bash scripts/rebuild_all.sh [N] [NUM_PREDICT]
#     N            traces per (workflow,topology) pair   (default 50 = full scale)
#     NUM_PREDICT  Ollama output-token cap per call       (default 256)
#
# DATA LAYOUT (important):
#   The ORIGINAL (pre-SDK, blocking-protocol) datasets live in root-owned dirs
#   (data/raw_b, data/raw_amodel, ...) from the earlier sudo collection and are
#   left UNTOUCHED as the archive.  New SSE data is written to fresh, user-owned
#   "*_sdk" dirs so the two are never mixed.  Deployment A is the exception:
#   evaluate.py hard-codes data/processed, so A re-collects into the (now empty,
#   user-owned) data/raw -> data/processed.
#
#   A      = A-logic (agents/)   + llama3.2:3b  -> data/raw            -> data/processed
#   amodel = A-logic (agents/)   + qwen2.5:7b   -> data/raw_amodel_sdk -> data/processed_amodel_sdk
#   B      = B-logic (agents_b/) + qwen2.5:7b   -> data/raw_b_sdk      -> data/processed_b_sdk
#   blogic = B-logic (agents_b/) + llama3.2:3b  -> data/raw_blogic_sdk -> data/processed_blogic_sdk
#
# tcpdump capture uses lo0; on this testbed BPF perms allow capture without sudo.

set -u
N="${1:-50}"
NP="${2:-256}"
PY="venv/bin/python"
LOG="logs/rebuild_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs data/results

stage() { echo ""; echo "######## $* ########" | tee -a "$LOG"; }
run()   { echo "+ $*" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 || echo "!! stage failed (continuing): $*" | tee -a "$LOG"; }

echo "Option-B rebuild  N=$N  num_predict=$NP  log=$LOG" | tee -a "$LOG"

# ── 0. Reset the user-owned dirs the new run reuses (A + all *_sdk) ───────────
stage "RESET fresh SSE output dirs"
for d in data/raw data/processed \
         data/raw_amodel_sdk data/processed_amodel_sdk \
         data/raw_b_sdk data/processed_b_sdk \
         data/raw_blogic_sdk data/processed_blogic_sdk \
         data/raw_defense_rate data/processed_defense_rate \
         data/raw_defense_pad data/processed_defense_pad \
         data/raw_background_sdk data/processed_background_sdk ; do
    rm -rf "$d" 2>/dev/null; mkdir -p "$d"
done
echo "  (original root-owned data/raw_b, data/raw_amodel, ... left intact as archive)" | tee -a "$LOG"

# ── 1. Collect all four deployments ──────────────────────────────────────────
stage "COLLECT A  (A-logic + llama3.2:3b)"
run $PY scripts/run_pilot.py   --n "$N" --model llama3.2:3b --out data/raw            --num-predict "$NP"
stage "COLLECT amodel  (A-logic + qwen2.5:7b)"
run $PY scripts/run_pilot.py   --n "$N" --model qwen2.5:7b  --out data/raw_amodel_sdk --num-predict "$NP"
stage "COLLECT B  (B-logic + qwen2.5:7b)"
run $PY scripts/run_pilot.py --deployment b --n "$N" --model qwen2.5:7b  --out data/raw_b_sdk      --num-predict "$NP"
stage "COLLECT blogic  (B-logic + llama3.2:3b)"
run $PY scripts/run_pilot.py --deployment b --n "$N" --model llama3.2:3b --out data/raw_blogic_sdk --num-predict "$NP"

# ── 2. Collect live-defended sets (A-logic) ──────────────────────────────────
stage "COLLECT defense=rate  (dummy sub-calls + jittered delegation)"
run $PY scripts/run_pilot.py --n "$N" --model llama3.2:3b --out data/raw_defense_rate --num-predict "$NP" --defense rate
stage "COLLECT defense=pad  (SSE constant-size event padding)"
run $PY scripts/run_pilot.py --n "$N" --model llama3.2:3b --out data/raw_defense_pad  --num-predict "$NP" --defense pad

# ── 3. Background traffic for open-world (real hard negatives) ────────────────
stage "COLLECT background"
run $PY scripts/collect_background.py --n "$N" --out data/raw_background_sdk --processed data/processed_background_sdk --scapy

# ── 4. Feature extraction ────────────────────────────────────────────────────
for pair in \
  "data/raw:data/processed" \
  "data/raw_amodel_sdk:data/processed_amodel_sdk" \
  "data/raw_b_sdk:data/processed_b_sdk" \
  "data/raw_blogic_sdk:data/processed_blogic_sdk" \
  "data/raw_defense_rate:data/processed_defense_rate" \
  "data/raw_defense_pad:data/processed_defense_pad" ; do
    raw="${pair%%:*}"; proc="${pair##*:}"
    stage "EXTRACT $raw -> $proc"
    run $PY scripts/extract_features.py --raw "$raw" --out "$proc" --scapy
done

# ── 5. Evaluation suite ──────────────────────────────────────────────────────
stage "EVAL closed-world (RF primary, GBT confirm)"
run $PY scripts/evaluate.py --mode all
stage "EVAL cross-deployment (A vs B)"
run $PY scripts/evaluate_cross_deployment.py --dir-a data/processed --dir-b data/processed_b_sdk
stage "EVAL model-vs-logic 2x2 disentanglement"
run $PY scripts/evaluate_model_vs_logic.py \
    --dir-a data/processed --dir-amodel data/processed_amodel_sdk \
    --dir-blogic data/processed_blogic_sdk --dir-b data/processed_b_sdk
stage "EVAL open-world vs real background"
run $PY scripts/evaluate_open_world_background.py --processed data/processed --bg-processed data/processed_background_sdk
stage "EVAL live C4 defenses (measured overhead + attack degradation)"
run $PY scripts/evaluate_defense_live.py \
    --baseline data/processed --baseline-raw data/raw \
    --rate data/processed_defense_rate --rate-raw data/raw_defense_rate \
    --pad  data/processed_defense_pad  --pad-raw  data/raw_defense_pad

stage "REBUILD COMPLETE"
echo "Results in data/results/.  Full log: $LOG"
