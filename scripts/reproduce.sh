#!/usr/bin/env bash
#
# reproduce.sh — demonstrate the project by re-analysing the FROZEN FEATURES.
#
# This is the reproducibility entry point.  Unlike scripts/rebuild_all.sh (which
# RE-COLLECTS data by running the live A2A testbed — stochastic and destructive),
# this only RE-ANALYSES already-extracted feature matrices, and it NEVER writes to
# the canonical committed results.
#
#   * Default (no args): a SHORT demo — closed-world (RF) + C5 — that writes to a
#     sandbox (data/results_demo/) and prints a side-by-side check against the
#     committed data/results/, to show the pipeline reproduces the headline.
#   * --full-suite: run the whole evaluation suite into the sandbox / copy
#     (closed-world RF+GBT, cross-deployment, runtime-swap control, model-vs-logic,
#     open-world, live defenses, C5, paper artifacts).
#   * --with-deep: also train the stochastic CNN/Transformer (OFF by default — the
#     deterministic RF+GBT headline reproduces without them).  Combine with --full-suite.
#
# The canonical results are protected two ways:
#   1. output dir is the sandbox (A2A_RESULTS_DIR), never data/results;
#   2. a hard guard below refuses to run if the output dir IS data/results.
# To re-run into a named fresh copy instead of the default sandbox:
#   A2A_RESULTS_DIR=data/results_$(date +%F) bash scripts/reproduce.sh --full-suite
#
# Determinism: RF/GBT/transfer point estimates use fixed random_state=42 and
# reproduce the committed numbers; CNN/Transformer are stochastic, footnote-only.
#
# Inputs: the published feature archive unpacked into data/  (see DATA.md).
#   Data archive: <ARCHIVE_URL — filled in after the Zenodo/figshare/OSF upload>
# Only the live-defense OVERHEAD stage needs raw pcaps; everything else runs from
# features alone, and each stage auto-skips if its inputs are absent.
#
# Usage:
#   bash scripts/reproduce.sh                          # short demo (sandbox)
#   bash scripts/reproduce.sh --full-suite             # full suite (RF+GBT, no deep)
#   bash scripts/reproduce.sh --full-suite --with-deep # also train CNN/Transformer
#   PYTHON=venv/bin/python bash scripts/reproduce.sh   # pick the interpreter
#
set -u

PY="${PYTHON:-python3}"          # override with PYTHON=venv/bin/python (or activate the venv)
MODE="demo"; WITH_DEEP=0
for arg in "$@"; do
    case "$arg" in
        --full-suite) MODE="full" ;;
        --with-deep)  WITH_DEEP=1 ;;   # opt-in: also train CNN/Transformer (stochastic, slow)
    esac
done

# ── Output sandbox (never the canonical data/results) ─────────────────────────
CANON="data/results"
SANDBOX="${A2A_RESULTS_DIR:-data/results_demo}"
if [ "$SANDBOX" = "$CANON" ] || [ "$SANDBOX" = "$CANON/" ]; then
    echo "REFUSING: A2A_RESULTS_DIR would overwrite the canonical $CANON." >&2
    echo "Pick a different dir, e.g. A2A_RESULTS_DIR=data/results_rerun" >&2
    exit 2
fi
export A2A_RESULTS_DIR="$SANDBOX"

LOG="logs/reproduce_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs "$SANDBOX"

A="${DIR_A:-data/processed}"
WAN="${DIR_WAN:-data/processed_wan}"

n_ok=0
stage() { echo ""; echo "######## $* ########" | tee -a "$LOG"; }
run()   { echo "+ $*" | tee -a "$LOG"; if "$@" >>"$LOG" 2>&1; then n_ok=$((n_ok+1));
          else echo "!! stage FAILED (see $LOG): $*" | tee -a "$LOG"; fi; }
have()  { for d in "$@"; do [ -e "$d" ] || { echo "  ↳ skip: missing $d" | tee -a "$LOG"; return 1; }; done; }

echo "reproduce.sh  mode=$MODE  sandbox=$SANDBOX  (canonical $CANON untouched)  log=$LOG" | tee -a "$LOG"

if [ "$MODE" = "demo" ]; then
    # ── SHORT demo: deterministic headline only ───────────────────────────────
    stage "DEMO 1/2  closed-world (RF, deterministic)"
    if have "$A/labels.json"; then run $PY scripts/evaluate.py --mode closed_world --rf-only; fi
    stage "DEMO 2/2  C5 cross-network (RF)"
    if have "$A/labels.json" "$WAN/labels.json"; then
        run $PY scripts/evaluate_c5.py --local "$A" --wan "$WAN"; fi
    stage "paper artifacts (sandbox)"
    run $PY scripts/make_paper_artifacts.py

    # ── Side-by-side check: sandbox vs committed canonical ────────────────────
    stage "CHECK  sandbox vs committed $CANON (should match on deterministic stages)"
    $PY - "$SANDBOX" "$CANON" <<'PYEOF' | tee -a "$LOG"
import json, sys
from pathlib import Path
sand, canon = Path(sys.argv[1]), Path(sys.argv[2])
def f1(root, task):
    p = root / "closed_world" / f"closed_world_rf_{task}.json"
    if not p.exists(): return None
    return json.loads(p.read_text()).get("cv", {}).get("f1_macro", {}).get("mean")
print(f"  {'task':<12}{'reproduced':>12}{'committed':>12}{'Δ':>10}")
for t in ("workflow","role","topology","parallelism"):
    a, b = f1(sand, t), f1(canon, t)
    if a is None or b is None:
        print(f"  {t:<12}{'n/a':>12}{'n/a':>12}"); continue
    print(f"  {t:<12}{a:>12.3f}{b:>12.3f}{a-b:>+10.4f}")
c = sand / "c5_cross_network.json"
if c.exists():
    d = json.loads(c.read_text())
    print(f"  C5 workflow transfer F1 = {d['workflow']['transfer']['f1']:.3f} (n={d['workflow']['transfer']['n']})")
PYEOF
else
    # ── FULL suite into the sandbox / copy ────────────────────────────────────
    B="${DIR_B:-data/processed_b_sdk}"; AMODEL="${DIR_AMODEL:-data/processed_amodel_sdk}"
    BLOGIC="${DIR_BLOGIC:-data/processed_blogic_sdk}"; BG="${DIR_BG:-data/processed_background_sdk}"
    RATE="${DIR_RATE:-data/processed_defense_rate}"; PAD="${DIR_PAD:-data/processed_defense_pad}"
    LANGGRAPH="${DIR_LANGGRAPH:-data/processed_langgraph}"

    # Deep models (CNN/Transformer) are stochastic + slow and are OFF by default;
    # pass --with-deep to include them. RF+GBT (the deterministic headline) always run.
    DEEP_FLAG=""; DEEP_LBL="(RF + GBT — deep OFF; pass --with-deep to add CNN/Transformer)"
    [ "$WITH_DEEP" = 1 ] && { DEEP_FLAG="--full-suite"; DEEP_LBL="(RF + GBT + CNN + Transformer)"; }
    stage "closed-world $DEEP_LBL"
    if have "$A/labels.json"; then run $PY scripts/evaluate.py --mode all $DEEP_FLAG; fi
    stage "cross-deployment (A vs B)"
    if have "$A/labels.json" "$B/labels.json"; then
        run $PY scripts/evaluate_cross_deployment.py --dir-a "$A" --dir-b "$B" --out "$SANDBOX/cross_deployment.json"; fi
    stage "runtime-swap control (A vs C / LangGraph — same logic, different runtime)"
    if have "$A/labels.json" "$LANGGRAPH/labels.json"; then
        run $PY scripts/evaluate_cross_deployment.py --dir-a "$A" --dir-b "$LANGGRAPH" \
            --label-b C --control --out "$SANDBOX/cross_framework.json"; fi
    stage "runtime-traffic diagnostic (A vs C — structure vs timing)"
    if have "$A/labels.json" "$LANGGRAPH/labels.json"; then
        run $PY scripts/diagnose_runtime_traffic.py --dir-a "$A" --dir-c "$LANGGRAPH" \
            --out "$SANDBOX/runtime_traffic_diagnostic.json"; fi
    stage "model-vs-logic (2x2)"
    if have "$A/labels.json" "$AMODEL/labels.json" "$BLOGIC/labels.json" "$B/labels.json"; then
        run $PY scripts/evaluate_model_vs_logic.py --dir-a "$A" --dir-amodel "$AMODEL" \
            --dir-blogic "$BLOGIC" --dir-b "$B" --out "$SANDBOX/model_vs_logic.json"; fi
    stage "open-world vs real background"
    # NB: background dirs carry labels_background.json (categories in filenames), NOT
    # labels.json — gating on labels.json here silently skipped this whole stage.
    if have "$A/labels.json" "$BG/labels_background.json"; then
        run $PY scripts/evaluate_open_world_background.py --processed "$A" --bg-processed "$BG"; fi
    stage "live C4 defenses (needs raw pcaps for overhead)"
    if have "$A/labels.json" "$RATE/labels.json" "$PAD/labels.json" \
            "data/raw" "data/raw_defense_rate" "data/raw_defense_pad"; then
        run $PY scripts/evaluate_defense_live.py \
            --baseline "$A" --baseline-raw data/raw \
            --rate "$RATE" --rate-raw data/raw_defense_rate \
            --pad  "$PAD"  --pad-raw  data/raw_defense_pad; fi
    stage "C5 cross-network"
    if have "$A/labels.json" "$WAN/labels.json"; then
        run $PY scripts/evaluate_c5.py --local "$A" --wan "$WAN"; fi
    stage "SSE size analysis vs 512 B pad cell (needs raw pcaps; logs only)"
    if have "data/raw" "data/raw_defense_pad"; then
        run $PY scripts/analyze_sse_sizes.py; fi
    stage "defense overhead–accuracy sweep (curve; heavy — re-extracts the base capture per level)"
    if have "$A/labels.json" "data/raw"; then
        run $PY scripts/sweep_defenses.py --raw data/raw --processed "$A"; fi
    stage "off-the-shelf role fingerprint (Task #4 — a2a_mcp replication; needs off-the-shelf pcaps)"
    if have "data/raw_offtheshelf" "$A/labels.json"; then
        run $PY scripts/evaluate_offtheshelf_fingerprint.py --raw data/raw_offtheshelf --a-role "$A"; fi
    stage "framework/implementation ID (Phase 1 recon; needs ≥3 impls' processed features)"
    if have "$A/labels.json" "$LANGGRAPH/labels.json"; then
        run $PY scripts/evaluate_framework_id.py; fi
    stage "framework-ID CONFOUND CONTROL (Phase 1; same-session interleaved A/B/C — new collection via collect_interleaved.sh)"
    # Reads the interleaved processed features (produced by collect_interleaved.sh + extract).
    if have "data/processed_interleaved_a/labels.json" "data/processed_interleaved_langgraph/labels.json"; then
        run $PY scripts/evaluate_framework_id_interleaved.py; fi
    stage "CONFOUND AUDIT — core claims (workflow/role/topology) under same-session interleaving"
    # Needs the powered interleaved-A capture + the prompt-diverse workflow capture (new collections
    # via collect_interleaved.sh --powered / collect_wf_interleaved.sh, then extract_features).
    if have "data/processed_interleaved_a_pwr/labels.json" "data/processed_wf_interleaved/labels.json"; then
        run $PY scripts/evaluate_confound_control.py; fi
    stage "cross-instance transfer (Phase 2; reads RAW pcaps of both a2a_mcp instances — new collection)"
    # NB: this evaluator consumes RAW pcaps directly (extract_role_samples), not processed
    # features — so gate on the raw dirs it actually reads, else the stage silently skips.
    if have "data/raw_offtheshelf" "data/raw_offtheshelf_inst2"; then
        run $PY scripts/evaluate_cross_instance_transfer.py; fi
    stage "cross-instance transfer — NATURAL de-confound (§9b′; reads the natural inst-2 set)"
    if have "data/raw_offtheshelf" "data/raw_offtheshelf_inst2_natural"; then
        run env CIT_OUT=cross_instance_transfer_natural.json \
            $PY scripts/evaluate_cross_instance_transfer.py --inst2 data/raw_offtheshelf_inst2_natural; fi
    stage "cross-FRAMEWORK replication on AutoGen (§10; reads AutoGen gRPC pcaps — external deployment ~/autogen-xframework)"
    if have "$HOME/autogen-xframework/data/raw" "data/raw_offtheshelf"; then
        run $PY scripts/evaluate_cross_framework_autogen.py; fi
    stage "agentic detection A2A-vs-AutoGen (Exp 1 / §11; reads a2a_mcp + AutoGen pcaps)"
    if have "data/raw_offtheshelf" "$HOME/autogen-xframework/data/raw"; then
        run $PY scripts/evaluate_agentic_detection.py; fi
    stage "mixing/multiplexing degradation (Exp 2 / §12; a2a_mcp flows + processed background)"
    if have "data/raw_offtheshelf" "data/processed_background/labels_background.json"; then
        run $PY scripts/evaluate_mixing_degradation.py; fi
    stage "same-transport detection A2A-vs-CrewAI (Exp 3 / §13; reads a2a_mcp + CrewAI-over-a2a-sdk pcaps — external deployment ~/crewai-xframework)"
    if have "data/raw_offtheshelf" "$HOME/crewai-xframework/data/raw"; then
        run $PY scripts/evaluate_crewai_detection.py; fi
    stage "matched-pair detection A-vs-CrewAI, ALL ELSE EQUAL (§13.1; reads deployment-A pcaps + matched CrewAI on A's roles/prompts — ~/crewai-xframework/data/raw_matched)"
    if have "data/raw" "$HOME/crewai-xframework/data/raw_matched"; then
        run $PY scripts/evaluate_crewai_matched_detection.py; fi
    stage "capture-interface provenance manifest (C1; loopback vs cross-host, derived from traces)"
    if have "data/raw"; then
        run $PY scripts/emit_capture_interface.py; fi
    stage "group/cluster bootstrap check on headline CIs (C4; prompt_group + trip clusters)"
    if have "data/processed/labels.json" "data/raw_offtheshelf" "data/raw_offtheshelf_inst2"; then
        run $PY scripts/check_group_bootstrap.py; fi
    stage "paper artifacts"
    run $PY scripts/make_paper_artifacts.py
fi

echo ""
echo "######## REPRODUCE COMPLETE ($MODE) ########" | tee -a "$LOG"
echo "  stages run: $n_ok   output → $SANDBOX/   (canonical $CANON untouched)" | tee -a "$LOG"
echo "  full log → $LOG" | tee -a "$LOG"
