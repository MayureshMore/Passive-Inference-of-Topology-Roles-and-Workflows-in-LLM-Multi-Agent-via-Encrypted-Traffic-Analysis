#!/usr/bin/env bash
#
# Phase 5 — capture the OFF-THE-SHELF a2a_mcp multi-agent system (Google's sample,
# externally authored). Drives the orchestrator over A2A (multi-turn: auto-answers
# the planner's clarifying questions) so it runs the full workflow
# orchestrator -> planner -> find_agent(MCP) -> air/hotel/car specialists, and
# captures each trip's A2A traffic (ports 10100-10105) on lo0 as one pcap + sidecar.
#
# Labels do NOT align with our taxonomy by design — this corroborates DETECTION +
# TOPOLOGY OBSERVABILITY of a system we did not build, NOT a transfer number.
#
# IMPORTANT — Gemini free tier = ~20 generateContent req/day (gemini-2.5-flash);
# one trip uses ~8-12 calls, so expect ~2 traces/run. The script STOPS on the
# first 429 (RESOURCE_EXHAUSTED). Re-run after the daily reset (~midnight Pacific)
# to accumulate more. (Enable billing for a full ~50-100 trace capture in one go.)
#
# Prereq (one-time, for non-root tcpdump):   sudo chmod o+r /dev/bpf*
# Run (NO sudo):                             bash scripts/collect_offtheshelf.sh [N]
#
# Requires the 8 compat fixes already applied to ~/a2a-samples (.../a2a_mcp) and
# its drive_orch.py multi-turn A2A driver.
set -u

SAMPLE_DIR="$HOME/a2a-samples/samples/python/agents/a2a_mcp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_DIR/data/raw_offtheshelf"
IFACE="lo0"; PORTS="10100-10105"
N="${1:-4}"

[ -f "$SAMPLE_DIR/drive_orch.py" ] || { echo "ERROR: $SAMPLE_DIR/drive_orch.py missing"; exit 1; }
grep -qE '^GOOGLE_API_KEY=.{10,}' "$SAMPLE_DIR/.env" 2>/dev/null || { echo "ERROR: GOOGLE_API_KEY not set in $SAMPLE_DIR/.env"; exit 1; }
mkdir -p "$OUT" "$SAMPLE_DIR/logs"
cd "$SAMPLE_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env 2>/dev/null; set +a          # load key silently
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"   # litellm reads GEMINI_API_KEY

PORTLIST="10100 10101 10102 10103 10104 10105"
clearports(){ for p in $PORTLIST; do lsof -ti tcp:$p 2>/dev/null | xargs kill -9 2>/dev/null; done; }
declare -a PIDS
start(){ eval "$1" > "logs/cap_$2.log" 2>&1 & PIDS+=($!); }
cleanup(){ clearports; kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT
clearports; sleep 1

echo "starting a2a_mcp (MCP 10100, orch 10101, planner 10102, air 10103, hotel 10104, car 10105)..."
start "uv run --env-file .env a2a-mcp --run mcp-server --transport sse --port 10100" mcp
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/orchestrator_agent.json --port 10101" orch
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/planner_agent.json --port 10102" planner
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/air_ticketing_agent.json --port 10103" air
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/hotel_booking_agent.json --port 10104" hotel
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/car_rental_agent.json --port 10105" car
echo "waiting 14s for init..."; sleep 14

DESTS=("Tokyo Japan|NRT" "Paris France|CDG" "Rome Italy|FCO" "Sydney Australia|SYD"
       "Cairo Egypt|CAI" "Reykjavik Iceland|KEF" "Bangkok Thailand|BKK" "Lisbon Portugal|LIS")
ORIG=("New York JFK" "San Francisco SFO" "Chicago ORD" "Los Angeles LAX")

ok=0
for i in $(seq 1 "$N"); do
    d=${DESTS[$(( (i-1) % ${#DESTS[@]} ))]}; dest=${d%%|*}; apt=${d##*|}
    o=${ORIG[$(( (i-1) % ${#ORIG[@]} ))]}
    q="Plan a trip from $o to $dest. Depart 2026-05-0$(( i % 9 + 1 )), return 2026-05-1$(( i % 9 )), 2 adults, economy class, budget 6000 USD, prefer airport $apt. Book round-trip flights, a hotel for 5 nights, and a rental car."
    id=$(printf "trip_%03d" "$i")

    tcpdump -i "$IFACE" -s 96 -w "$OUT/$id.pcap" "tcp portrange $PORTS" 2>/dev/null & tpid=$!
    sleep 0.8
    uv run --env-file .env python drive_orch.py "$q" > "logs/cap_drive_$id.log" 2>&1 || true
    sleep 0.8
    kill "$tpid" 2>/dev/null; wait "$tpid" 2>/dev/null

    printf '{"system":"a2a_mcp","trace_id":"%s","input_prompt":"%s","agent_endpoints":{"mcp":"127.0.0.1:10100","orchestrator":"127.0.0.1:10101","planner":"127.0.0.1:10102","air_ticketing":"127.0.0.1:10103","hotel":"127.0.0.1:10104","car_rental":"127.0.0.1:10105"}}\n' "$id" "$q" > "$OUT/$id.json"

    sz=$(stat -f%z "$OUT/$id.pcap" 2>/dev/null || echo 0)
    [ "$sz" -gt 400 ] && ok=$((ok+1))
    printf "  [%d/%d] %-22s pcap=%sB\n" "$i" "$N" "$dest" "$sz"

    if grep -qiE "RESOURCE_EXHAUSTED|RequestsPerDay|quotaValue" logs/cap_*.log 2>/dev/null; then
        echo "  !! Gemini free-tier quota hit (429) — stopping. Re-run after the daily reset."
        break
    fi
    sleep 3
done

echo ""
echo "captured $ok non-empty traces this run; total pcaps: $(ls "$OUT"/*.pcap 2>/dev/null | wc -l | tr -d ' ')  ->  $OUT"
echo "next (no quota needed):  venv/bin/python scripts/extract_offtheshelf.py --raw $OUT --out data/processed_offtheshelf --scapy"
