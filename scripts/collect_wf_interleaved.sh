#!/usr/bin/env bash
#
# Prompt-DIVERSE, temporally-INTERLEAVED workflow capture — the confound control for the
# closed-world WORKFLOW claim (§8.2). The committed closed-world captured each workflow in one
# contiguous ~17-min block, so slow within-session drift could correlate with the workflow label.
# This control captures the 4 workflows ROUND-ROBIN (one short block each per cycle), revisiting
# every workflow across the session, and draws FRESH prompts each cycle via run_pilot --seed-offset
# (so the group-CV workflow test — which holds out whole prompts — has real prompt diversity, the
# thing a naive --n 1 interleave lacks). star topology only (workflow signal is content, not
# topology). A/llama3.2:3b, num_predict 256 — matched to the committed closed-world.
#
# ADDITIVE: writes only data/raw_wf_interleaved/. Local Ollama, $0.
# Usage: bash scripts/collect_wf_interleaved.sh [CYCLES] [N_PER_BATCH]
set -u
CYCLES="${1:-6}"; NB="${2:-6}"
PY="venv/bin/python"; OUT="data/raw_wf_interleaved"; URL="http://localhost:11434"
mkdir -p logs; rm -rf "$OUT"; mkdir -p "$OUT"
curl -s --max-time 4 "$URL/api/tags" >/dev/null 2>&1 || { echo "BLOCKED: Ollama down"; exit 1; }

WF=(research_retrieval code_review data_analysis support_triage)
count() { ls "$OUT"/*.pcap 2>/dev/null | wc -l | tr -d ' '; }

for c in $(seq 1 "$CYCLES"); do
    off=$(( c * 1000 ))
    r=$(( c % 4 ))                       # rotate starting workflow each cycle (decorrelate position)
    echo "=== cycle $c/$CYCLES  seed_offset=$off ==="
    for k in 0 1 2 3; do
        wf=${WF[$(( (k + r) % 4 ))]}
        $PY scripts/run_pilot.py --deployment a --workflow "$wf" --topology star \
            --n "$NB" --seed-offset "$off" --out "$OUT" --model llama3.2:3b \
            --num-predict 256 --ollama-url "$URL" >/dev/null 2>&1 \
            && echo "  ok $wf" || echo "  FAIL $wf"
    done
    echo "  cumulative pcaps: $(count)"
done
echo "DONE wf-interleaved: $(count) pcaps  ->  $OUT"
