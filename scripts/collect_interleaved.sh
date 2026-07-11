#!/usr/bin/env bash
#
# SAME-SESSION INTERLEAVED capture — the confound control for Phase-1 framework ID.
#
# Phase 1 (evaluate_framework_id.py) found near-perfect implementation separability,
# INCLUDING same-structure A vs C_langgraph, which is the classic signature of a
# capture-SESSION / batch confound: A, B and C were each collected in a SEPARATE run,
# so a classifier could key on session artefacts (host state, clock granularity,
# ephemeral-port ranges, background load) instead of a genuine implementation fingerprint.
#
# This control removes that confound by ROUND-ROBINING short micro-batches of the three
# implementations inside ONE continuous session:
#
#   cycle 1:  [shuffled order] A-batch  B-batch  C-batch
#   cycle 2:  [shuffled order] C-batch  A-batch  B-batch
#   ...
#
# Because every label now spans many interleaved micro-sessions on the same host at the
# same time, any session-drift artefact is SHARED across labels and can no longer predict
# the label. What remains separable is genuine traffic shape. Each run_pilot invocation
# also RELAUNCHES its agents, so ephemeral-port / connection-state artefacts are randomised
# WITHIN each label too — further de-correlating session identity from the label.
#
# Critically, A and C use the SAME model (llama3.2:3b) and identical workflows/topologies,
# so A vs C differs ONLY in the orchestration runtime (asyncio vs LangGraph StateGraph).
# If A↔C separability SURVIVES this control  -> genuine runtime fingerprint (a real recon
# signal, clean of the confound). If it COLLAPSES -> the original 0.998 was batch-inflated
# and Phase 1 is honestly demoted. Either outcome makes the paper more bulletproof.
#
# ADDITIVE: writes only NEW dirs data/raw_interleaved_{a,b,langgraph}; touches nothing
# committed. Local Ollama only — $0 API cost.
#
# Usage:  bash scripts/collect_interleaved.sh [CYCLES] [NUM_PREDICT]
#   CYCLES       round-robin cycles (per-impl traces = CYCLES * 12 conditions)   default 6
#   NUM_PREDICT  Ollama output-token cap                                          default 256
set -u

CYCLES="${1:-6}"
NP="${2:-256}"
PY="venv/bin/python"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
SEED="${SEED:-42}"

OUT_A="data/raw_interleaved_a"
OUT_B="data/raw_interleaved_b"
OUT_C="data/raw_interleaved_langgraph"

LOG="logs/interleaved_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# ── Preconditions ─────────────────────────────────────────────────────────────
if ! curl -s --max-time 4 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "BLOCKED: Ollama not reachable at $OLLAMA_URL — start it with 'ollama serve'." | tee -a "$LOG"; exit 1
fi
for m in llama3.2:3b qwen2.5:7b; do
    curl -s "$OLLAMA_URL/api/tags" | grep -q "$m" || { echo "BLOCKED: model $m not pulled (ollama pull $m)" | tee -a "$LOG"; exit 1; }
done

# Reset the NEW interleaved dirs (never any committed dir).
for d in "$OUT_A" "$OUT_B" "$OUT_C"; do rm -rf "$d"; mkdir -p "$d"; done

echo "interleaved control  CYCLES=$CYCLES  NP=$NP  seed=$SEED" | tee -a "$LOG"
echo "  A=$OUT_A (a-logic/asyncio + llama3.2:3b)" | tee -a "$LOG"
echo "  B=$OUT_B (b-logic         + qwen2.5:7b)"  | tee -a "$LOG"
echo "  C=$OUT_C (a-logic/LangGraph + llama3.2:3b)  <- A & C share model+logic, differ only in runtime" | tee -a "$LOG"

# One micro-batch = run_pilot --n 1 over ALL 4 workflows x 3 topologies = 12 balanced traces.
batch() {  # $1=deployment  $2=model  $3=outdir  $4=cycle
    echo "  [cycle $4] batch $1 -> $3" | tee -a "$LOG"
    $PY scripts/run_pilot.py --deployment "$1" --model "$2" --out "$3" --n 1 \
        --num-predict "$NP" --ollama-url "$OLLAMA_URL" >>"$LOG" 2>&1 \
        || echo "  !! batch $1 (cycle $4) had a failure — continuing" | tee -a "$LOG"
}

count() { ls "$1"/*.pcap 2>/dev/null | wc -l | tr -d ' '; }

for c in $(seq 1 "$CYCLES"); do
    # Deterministic but label-decorrelated order: rotate the triple by cycle index.
    case $(( c % 3 )) in
        0) order="a b langgraph" ;;
        1) order="b langgraph a" ;;
        2) order="langgraph a b" ;;
    esac
    for dep in $order; do
        case "$dep" in
            a)         batch a         llama3.2:3b "$OUT_A" "$c" ;;
            b)         batch b         qwen2.5:7b  "$OUT_B" "$c" ;;
            langgraph) batch langgraph llama3.2:3b "$OUT_C" "$c" ;;
        esac
    done
    echo "  ── after cycle $c: A=$(count "$OUT_A")  B=$(count "$OUT_B")  C=$(count "$OUT_C") pcaps" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "── interleaved collection complete ──" | tee -a "$LOG"
echo "  A=$(count "$OUT_A")  B=$(count "$OUT_B")  C=$(count "$OUT_C") pcaps  (log $LOG)" | tee -a "$LOG"
echo "next: extract each dir then run scripts/evaluate_framework_id_interleaved.py" | tee -a "$LOG"
