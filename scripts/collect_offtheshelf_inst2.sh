#!/usr/bin/env bash
#
# PHASE 2 — collect a SECOND, independent instance of the a2a_mcp framework (instance 2).
#
# Same public framework, deliberately made independent from instance 1 (the committed
# 150-trip set) along the axes the brief asks for:
#   * DIFFERENT LLM   — LITELLM_MODEL=gemini/gemini-2.0-flash (instance 1 used 2.5-flash).
#                       This is a2a_mcp's own litellm config (adk_travel_agent.py reads it).
#   * DIFFERENT PROMPTS — reworded query template, different dates/party-size/class/nights.
#   * SEPARATE SESSION — fresh run.
#   * SAME SIX ROLES by port (10100-10105) — the framework fixes them; that is the point.
#
# Writes to data/raw_offtheshelf_inst2/ (NEVER instance 1). Budget-guarded + circuit-broken,
# so it stops cleanly if the prepaid credit runs low. Additive — no committed file changes.
#
# Prereq: the external checkout at ~/a2a-samples/.../a2a_mcp with the 8 compat patches +
# drive_orch.py + a Gemini key in .env, and non-sudo BPF (chmod o+r /dev/bpf*).
#
# Usage:  BUDGET_USD=3.00 bash scripts/collect_offtheshelf_inst2.sh 120
set -u

SAMPLE_DIR="$HOME/a2a-samples/samples/python/agents/a2a_mcp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_DIR/data/raw_offtheshelf_inst2"
IFACE="lo0"; PORTS="10100-10105"
N="${1:-120}"                              # max trips to ATTEMPT (budget guard stops earlier)

INST2_MODEL="gemini/gemini-2.0-flash"      # the independence axis (different LLM)

# ── Budget / rate-limit controls ──────────────────────────────────────────────
BUDGET_USD="${BUDGET_USD:-3.00}"           # hard ceiling on ESTIMATED spend for THIS instance-2 run
COST_PER_TRACE="${COST_PER_TRACE:-0.043}"  # real per-trip rate observed on instance 1
SPENT_SO_FAR="${SPENT_SO_FAR:-0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BACKOFF_S="${BACKOFF_S:-30}"

[ -f "$SAMPLE_DIR/drive_orch.py" ] || { echo "BLOCKED: $SAMPLE_DIR/drive_orch.py missing"; exit 1; }
grep -qE '^GOOGLE_API_KEY=.{10,}' "$SAMPLE_DIR/.env" 2>/dev/null || { echo "BLOCKED: GOOGLE_API_KEY not set in $SAMPLE_DIR/.env"; exit 1; }
mkdir -p "$OUT" "$SAMPLE_DIR/logs"
cd "$SAMPLE_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env 2>/dev/null; set +a
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
export LITELLM_MODEL="$INST2_MODEL"        # <-- instance-2 independence: different specialist LLM

rm -f logs/cap_*.log

PORTLIST="10100 10101 10102 10103 10104 10105"
clearports(){ for p in $PORTLIST; do lsof -ti tcp:$p 2>/dev/null | xargs kill -9 2>/dev/null; done; }
declare -a PIDS
start(){ eval "$1" > "logs/cap_$2.log" 2>&1 & PIDS+=($!); }
cleanup(){ clearports; kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT
clearports; sleep 1

echo "instance-2 collection  LITELLM_MODEL=$LITELLM_MODEL  budget=\$$BUDGET_USD  out=$OUT"
start "uv run --env-file .env a2a-mcp --run mcp-server --transport sse --port 10100" mcp
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/orchestrator_agent.json --port 10101" orch
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/planner_agent.json --port 10102" planner
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/air_ticketing_agent.json --port 10103" air
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/hotel_booking_agent.json --port 10104" hotel
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/car_rental_agent.json --port 10105" car
echo "waiting 14s for init..."; sleep 14

# Reworded / re-parameterised queries (prompt independence from instance 1).
DESTS=("Tokyo Japan|NRT" "Paris France|CDG" "Rome Italy|FCO" "Sydney Australia|SYD"
       "Cairo Egypt|CAI" "Reykjavik Iceland|KEF" "Bangkok Thailand|BKK" "Lisbon Portugal|LIS"
       "Athens Greece|ATH" "Dubai UAE|DXB" "Singapore|SIN" "Toronto Canada|YYZ"
       "Mexico City Mexico|MEX" "Cape Town South Africa|CPT" "Oslo Norway|OSL" "Seoul Korea|ICN"
       "Buenos Aires Argentina|EZE" "Vienna Austria|VIE" "Istanbul Turkey|IST" "Nairobi Kenya|NBO")
ORIG=("New York JFK" "San Francisco SFO" "Chicago ORD" "Los Angeles LAX" "Boston BOS" "Seattle SEA")

SVC_LOGS="logs/cap_mcp.log logs/cap_orch.log logs/cap_planner.log logs/cap_air.log logs/cap_hotel.log logs/cap_car.log"
ge(){ awk "BEGIN{exit !($1 >= $2)}"; }
spend(){ awk "BEGIN{printf \"%.2f\", $SPENT_SO_FAR + $1*$COST_PER_TRACE}"; }

last=$(ls "$OUT"/trip_*.pcap 2>/dev/null | sed -E 's#.*/trip_0*([0-9]+)\.pcap#\1#' | sort -n | tail -1)
start=$(( ${last:-0} + 1 )); end=$(( start + N - 1 ))
echo "budget guard: stop at ~\$$BUDGET_USD est  (cost/trace=\$$COST_PER_TRACE)"

ok=0; skipped=0; consec_skip=0; stop_reason=""
for i in $(seq "$start" "$end"); do
    if ge "$(spend "$ok")" "$BUDGET_USD"; then stop_reason="budget ceiling (est \$$(spend "$ok") >= \$$BUDGET_USD)"; break; fi
    if [ "$consec_skip" -ge 3 ]; then stop_reason="3 consecutive failed trips (credit exhausted? services down?)"; break; fi

    d=${DESTS[$(( (i-1) % ${#DESTS[@]} ))]}; dest=${d%%|*}; apt=${d##*|}
    o=${ORIG[$(( (i-1) % ${#ORIG[@]} ))]}
    nights=$(( i % 4 + 3 )); pax=$(( i % 3 + 1 )); bud=$(( 4000 + (i%5)*1000 ))
    q="Arrange complete travel from $o to $dest for $pax traveler(s). Outbound 2026-08-1$(( i % 9 )), return 2026-08-2$(( i % 9 )), premium economy, roughly \$$bud budget, nearest airport $apt. I need round-trip air tickets, $nights nights of hotel, and a rental vehicle."
    id=$(printf "trip_%03d" "$i")

    sz=0; tries=0; skip=0
    while : ; do
        tcpdump -i "$IFACE" -s 96 -w "$OUT/$id.pcap" "tcp portrange $PORTS" 2>/dev/null & tpid=$!
        sleep 0.8
        uv run --env-file .env python drive_orch.py "$q" > "logs/cap_drive_$id.log" 2>&1 || true
        sleep 0.8
        kill "$tpid" 2>/dev/null; wait "$tpid" 2>/dev/null
        sz=$(stat -f%z "$OUT/$id.pcap" 2>/dev/null || echo 0)
        [ "${sz:-0}" -gt 400 ] && break
        if grep -qiE "PerDay|RequestsPerDay" $SVC_LOGS 2>/dev/null; then
            stop_reason="RequestsPerDay 429 — daily/credit cap hit"; sz=-1; break
        fi
        tries=$((tries+1))
        if [ "$tries" -gt "$MAX_RETRIES" ]; then echo "  ~ $id failed ${tries}x — skipping"; skip=1; break; fi
        echo "  ~ $id empty/failed — backoff ${BACKOFF_S}s then retry $tries/$MAX_RETRIES"; sleep "$BACKOFF_S"
    done
    [ "$sz" = "-1" ] && break
    if [ "$skip" = "1" ]; then rm -f "$OUT/$id.pcap"; skipped=$((skipped+1)); consec_skip=$((consec_skip+1)); sleep 2; continue; fi

    printf '{"system":"a2a_mcp","instance":2,"litellm_model":"%s","trace_id":"%s","input_prompt":"%s","agent_endpoints":{"mcp":"127.0.0.1:10100","orchestrator":"127.0.0.1:10101","planner":"127.0.0.1:10102","air_ticketing":"127.0.0.1:10103","hotel":"127.0.0.1:10104","car_rental":"127.0.0.1:10105"}}\n' "$INST2_MODEL" "$id" "$q" > "$OUT/$id.json"
    ok=$((ok+1)); consec_skip=0
    printf "  [trip %03d] %-22s pcap=%7sB | ok=%d skip=%d est≈\$%s\n" "$i" "$dest" "$sz" "$ok" "$skipped" "$(spend "$ok")"
    sleep 3
done

total=$(ls "$OUT"/*.pcap 2>/dev/null | wc -l | tr -d ' ')
echo ""; echo "── instance-2 run complete ──"
[ -n "$stop_reason" ] && echo "stopped: $stop_reason"
echo "new traces this run: $ok  (skipped: $skipped)  total inst2 pcaps: $total"
echo "estimated spend this run: ~\$$(spend "$ok")   <<< TRUTH = Google Cloud billing >>>"
echo "next: venv/bin/python scripts/extract_offtheshelf.py --raw $OUT --out data/processed_offtheshelf_inst2 --scapy"
