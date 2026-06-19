#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# resume_wan.sh — drop-resilient C5 WAN collection for ONE topology.
#
# Survives VPN drops: before each batch it waits until Kali is reachable, then
# re-resolves the (possibly changed) tunnel interface dynamically, then collects.
# It only fills MISSING (workflow) units and re-does partials cleanly, so it is
# safe to re-run any number of times.
#
# PREREQ: serve_agents.py for THIS topology must already be running on Kali, e.g.
#   Kali:  pkill -f serve_agents.py
#          nohup venv/bin/python scripts/serve_agents.py --topology chain --deployment a >/tmp/serve.log 2>&1 &
#
# USAGE (run from the repo root, WITH sudo so tcpdump works and there are no
# mid-run password prompts):
#   sudo ./scripts/resume_wan.sh chain
#   sudo ./scripts/resume_wan.sh mesh
#
# When chain finishes, restart serve_agents on Kali for mesh, then run it again
# for mesh.  Ctrl-C any time; just re-run to continue.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

TOPO="${1:-}"
if [[ "$TOPO" != "star" && "$TOPO" != "chain" && "$TOPO" != "mesh" ]]; then
  echo "usage: sudo ./scripts/resume_wan.sh <star|chain|mesh>" >&2
  exit 2
fi

REMOTE=10.16.0.35
OUT=data/raw_wan
PY=venv/bin/python
N=50                 # target traces per (workflow, topology)
ACCEPT=45            # treat a unit with >= this many as done (don't waste near-complete work)
WORKFLOWS=(research_retrieval code_review data_analysis support_triage)

mkdir -p "$OUT"

ts()    { date '+%H:%M:%S'; }
count() { ls "$OUT"/${1}_${TOPO}_*.pcap 2>/dev/null | wc -l | tr -d ' '; }

wait_reachable() {
  local first=1
  while ! curl -s -m 8 "http://$REMOTE:8001/.well-known/agent-card.json" >/dev/null 2>&1; do
    [[ $first == 1 ]] && { echo "[$(ts)] Kali unreachable (VPN down / agents off). Polling every 30s…"; first=0; }
    sleep 30
  done
}

iface_now() { route -n get "$REMOTE" 2>/dev/null | awk '/interface:/{print $2}'; }

echo "=== resume_wan: topology=$TOPO  target=$N/unit  accept>=$ACCEPT ==="
for wf in "${WORKFLOWS[@]}"; do
  have=$(count "$wf")
  if (( have >= ACCEPT )); then
    echo "[skip] $TOPO/$wf already has $have (>= $ACCEPT)"
    continue
  fi
  while (( $(count "$wf") < N )); do
    # clear any partial so we collect a clean, prompt-aligned 50
    have=$(count "$wf")
    if (( have > 0 )); then
      echo "[clean] $TOPO/$wf had $have partial pcaps → removing for a clean run"
      rm -f "$OUT"/${wf}_${TOPO}_*.pcap
    fi
    wait_reachable
    pkill -f "tcpdump" 2>/dev/null || true   # avoid leaked BPF devices from a killed run
    IFACE=$(iface_now)
    if [[ -z "$IFACE" ]]; then echo "[$(ts)] no route to $REMOTE yet; waiting"; sleep 15; continue; fi
    echo "[$(ts)] [collect] $TOPO/$wf  iface=$IFACE  n=$N"
    "$PY" scripts/collect_wan.py --remote-host "$REMOTE" --iface "$IFACE" \
      --deployment a --topology "$TOPO" --workflow "$wf" \
      --n "$N" --num-predict 256 --out "$OUT" || true
    got=$(count "$wf")
    if (( got >= ACCEPT )); then
      echo "[$(ts)] [ok] $TOPO/$wf = $got"
      break
    fi
    echo "[$(ts)] [retry] $TOPO/$wf only $got/$N (drop mid-run?) — will clean + retry"
  done
done

echo "=== $TOPO done. data/raw_wan total: $(ls "$OUT"/*.pcap 2>/dev/null | wc -l | tr -d ' ') pcaps ==="
for wf in "${WORKFLOWS[@]}"; do printf "  %-20s %s\n" "$TOPO/$wf" "$(count "$wf")"; done
