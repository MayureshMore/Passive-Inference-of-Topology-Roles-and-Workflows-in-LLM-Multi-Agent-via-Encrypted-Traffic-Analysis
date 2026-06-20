#!/usr/bin/env bash
#
# reproduce.sh — regenerate every paper table and figure from FROZEN FEATURES.
#
# This is the reproducibility entry point.  Unlike scripts/rebuild_all.sh (which
# RE-COLLECTS data by running the live A2A testbed — stochastic and destructive),
# this script only RE-ANALYSES already-extracted feature matrices.  It is:
#
#   * deterministic  — RF/GBT/transfer point estimates use fixed random_state=42,
#                      so re-runs reproduce the committed numbers exactly;
#   * non-destructive — reads data/processed*/ and writes only data/results/;
#                      it never deletes raw pcaps or re-collects anything;
#   * fast / offline — minutes, no GPU, no network, no Ollama, no testbed.
#
# It expects the released feature matrices to be present under data/.  These are
# gitignored, so for a fresh checkout you must first unpack the published
# artifact (the .npz + labels.json feature archive) into data/.  Raw pcaps are
# only needed for the live-defense OVERHEAD numbers (byte/latency); every other
# stage runs from features alone.  Each stage auto-skips if its inputs are absent.
#
# Raw collection (to rebuild the testbed itself) is documented separately in
# docs/C5_WAN_RUNBOOK.md and scripts/rebuild_all.sh — it is NOT part of
# reproducing the analysis.
#
# Usage:
#   bash scripts/reproduce.sh                # RF+GBT headline (deterministic)
#   bash scripts/reproduce.sh --full-suite   # also CNN1D + Transformer (slow, stochastic, footnote-only)
#
set -u

PY="venv/bin/python"
FULL_SUITE=""
[ "${1:-}" = "--full-suite" ] && FULL_SUITE="--full-suite"

LOG="logs/reproduce_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs data/results

# ── feature directories (override via env if your layout differs) ─────────────
A="${DIR_A:-data/processed}"                         # deployment A (headline)
B="${DIR_B:-data/processed_b_sdk}"                   # deployment B (cross-deploy)
AMODEL="${DIR_AMODEL:-data/processed_amodel_sdk}"    # A-logic + qwen2.5:7b
BLOGIC="${DIR_BLOGIC:-data/processed_blogic_sdk}"    # B-logic + llama3.2:3b
BG="${DIR_BG:-data/processed_background_sdk}"        # real background
WAN="${DIR_WAN:-data/processed_wan}"                 # C5 WAN capture
RATE="${DIR_RATE:-data/processed_defense_rate}"      # live rate defense
PAD="${DIR_PAD:-data/processed_defense_pad}"         # live pad defense

n_ok=0; n_skip=0
stage() { echo ""; echo "######## $* ########" | tee -a "$LOG"; }
run()   { echo "+ $*" | tee -a "$LOG"; if "$@" >>"$LOG" 2>&1; then n_ok=$((n_ok+1));
          else echo "!! stage FAILED (see $LOG): $*" | tee -a "$LOG"; fi; }
have()  { for d in "$@"; do [ -e "$d" ] || { echo "  ↳ skip: missing $d" | tee -a "$LOG"; return 1; }; done; }
skip()  { n_skip=$((n_skip+1)); }

echo "reproduce.sh  full_suite=${FULL_SUITE:-no}  log=$LOG" | tee -a "$LOG"

# ── 1. Closed-world (C2/C3/C1 headline) — needs only deployment A ──────────────
stage "CLOSED-WORLD  (RF + GBT${FULL_SUITE:+ + CNN + Transformer})"
if have "$A/labels.json"; then run $PY scripts/evaluate.py --mode all $FULL_SUITE; else skip; fi

# ── 2. Cross-deployment generalization (A vs B) ───────────────────────────────
stage "CROSS-DEPLOYMENT  (A vs B)"
if have "$A/labels.json" "$B/labels.json"; then
    run $PY scripts/evaluate_cross_deployment.py --dir-a "$A" --dir-b "$B"; else skip; fi

# ── 3. Model-vs-logic 2x2 disentanglement ─────────────────────────────────────
stage "MODEL-vs-LOGIC  (2x2 disentanglement)"
if have "$A/labels.json" "$AMODEL/labels.json" "$BLOGIC/labels.json" "$B/labels.json"; then
    run $PY scripts/evaluate_model_vs_logic.py \
        --dir-a "$A" --dir-amodel "$AMODEL" --dir-blogic "$BLOGIC" --dir-b "$B"; else skip; fi

# ── 4. Open-world vs real background ───────────────────────────────────────────
stage "OPEN-WORLD  (vs real background)"
if have "$A/labels.json" "$BG/labels.json"; then
    run $PY scripts/evaluate_open_world_background.py --processed "$A" --bg-processed "$BG"; else skip; fi

# ── 5. Live C4 defenses — overhead numbers need the raw pcaps too ──────────────
stage "LIVE C4 DEFENSES  (attack degradation + measured overhead)"
if have "$A/labels.json" "$RATE/labels.json" "$PAD/labels.json" \
        "data/raw" "data/raw_defense_rate" "data/raw_defense_pad"; then
    run $PY scripts/evaluate_defense_live.py \
        --baseline "$A" --baseline-raw data/raw \
        --rate "$RATE" --rate-raw data/raw_defense_rate \
        --pad  "$PAD"  --pad-raw  data/raw_defense_pad; else skip; fi

# ── 6. C5 cross-network (LAN vs WAN) ──────────────────────────────────────────
stage "C5 CROSS-NETWORK  (LAN-internal / WAN-internal / LAN→WAN transfer)"
if have "$A/labels.json" "$WAN/labels.json"; then
    run $PY scripts/evaluate_c5.py --local "$A" --wan "$WAN"; else skip; fi

# ── 7. Paper artifacts (per-class tables + headline figures) ───────────────────
stage "PAPER ARTIFACTS  (tables + figures from data/results/)"
run $PY scripts/make_paper_artifacts.py

echo ""
echo "######## REPRODUCE COMPLETE ########" | tee -a "$LOG"
echo "  stages run: $n_ok   skipped (missing inputs): $n_skip" | tee -a "$LOG"
echo "  results  → data/results/        figures → data/results/figures/" | tee -a "$LOG"
echo "  full log → $LOG" | tee -a "$LOG"
