#!/usr/bin/env bash
#
# PHASE 2 (Task B) — accumulate a2a_mcp INSTANCE-2 SPECIALIST samples for the full 6-way
# cross-instance transfer. Appends to data/raw_offtheshelf_inst2 (same instance-2:
# gemini-2.0-flash, separate session) using FAN-OUT-BOOSTED prompts + drive_orch_boost.py so
# each trip reliably exercises the air/hotel/car specialists (the original ~6% fan-out was a
# Tokyo-hardcoded canned answer sabotaging the planner on varied destinations).
#
# STOPS as soon as all three specialists reach TARGET_SPECIALISTS samples (default 15 — the
# reviewer's bar; do NOT run the 6-way on 4-5 specialist samples), OR the hard budget ceiling,
# whichever comes first. Reports specialist counts live so the run costs only what it needs.
#
# ADDITIVE: writes only into data/raw_offtheshelf_inst2/. Real Gemini spend — TRUTH is Google
# Cloud billing; the est is a guard, not the meter.
#
# Usage:  BUDGET_USD=8 TARGET_SPECIALISTS=15 bash scripts/collect_offtheshelf_6way.sh [MAX_TRIPS]
set -u

SAMPLE_DIR="$HOME/a2a-samples/samples/python/agents/a2a_mcp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_DIR/data/raw_offtheshelf_inst2"          # APPEND to instance-2 (the binding test set)
IFACE="lo0"; PORTS="10100-10105"
N="${1:-80}"                                          # max trips to ATTEMPT (guards stop earlier)

INST2_MODEL="gemini/gemini-2.0-flash"                 # unchanged independence axis (different LLM)
DRIVER="drive_orch_boost.py"                          # destination-agnostic fan-out-forcing answer

BUDGET_USD="${BUDGET_USD:-8.00}"                      # HARD ceiling on ESTIMATED spend
COST_PER_TRACE="${COST_PER_TRACE:-0.05}"              # slightly above observed $0.04 (boosted trips do more)
TARGET_SPECIALISTS="${TARGET_SPECIALISTS:-15}"        # stop when min(air,hotel,car) reaches this
SPENT_SO_FAR="${SPENT_SO_FAR:-0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BACKOFF_S="${BACKOFF_S:-30}"

[ -f "$SAMPLE_DIR/$DRIVER" ] || { echo "BLOCKED: $SAMPLE_DIR/$DRIVER missing"; exit 1; }
grep -qE '^GOOGLE_API_KEY=.{10,}' "$SAMPLE_DIR/.env" 2>/dev/null || { echo "BLOCKED: GOOGLE_API_KEY not set in $SAMPLE_DIR/.env"; exit 1; }
mkdir -p "$OUT" "$SAMPLE_DIR/logs"
cd "$SAMPLE_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env 2>/dev/null; set +a
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
export LITELLM_MODEL="$INST2_MODEL"

rm -f logs/cap_*.log
PORTLIST="10100 10101 10102 10103 10104 10105"
clearports(){ for p in $PORTLIST; do lsof -ti tcp:$p 2>/dev/null | xargs kill -9 2>/dev/null; done; }
declare -a PIDS
start(){ eval "$1" > "logs/cap_$2.log" 2>&1 & PIDS+=($!); }
cleanup(){ clearports; kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT
clearports; sleep 1

# ── specialist counters: init from EXISTING inst2 pcaps (port 10103/4/5 traffic present) ──
spec_in_pcap(){ tcpdump -r "$1" -nn "tcp port $2" 2>/dev/null | grep -q . && echo 1 || echo 0; }
air_n=0; hot_n=0; car_n=0
for f in "$OUT"/*.pcap; do
    [ -e "$f" ] || continue
    [ "$(spec_in_pcap "$f" 10103)" = 1 ] && air_n=$((air_n+1))
    [ "$(spec_in_pcap "$f" 10104)" = 1 ] && hot_n=$((hot_n+1))
    [ "$(spec_in_pcap "$f" 10105)" = 1 ] && car_n=$((car_n+1))
done
echo "starting specialist counts (existing inst2): air=$air_n hotel=$hot_n car=$car_n  target=$TARGET_SPECIALISTS"

echo "6-way top-up  LITELLM_MODEL=$LITELLM_MODEL  budget=\$$BUDGET_USD  driver=$DRIVER  out=$OUT"
start "uv run --env-file .env a2a-mcp --run mcp-server --transport sse --port 10100" mcp
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/orchestrator_agent.json --port 10101" orch
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/planner_agent.json --port 10102" planner
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/air_ticketing_agent.json --port 10103" air
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/hotel_booking_agent.json --port 10104" hotel
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/car_rental_agent.json --port 10105" car
echo "waiting 14s for init..."; sleep 14

# Fully-specified queries (pre-empt the planner's clarifying questions → reliable fan-out).
DESTS=("Helsinki Finland|HEL" "Porto Portugal|OPO" "Marrakesh Morocco|RAK" "Zurich Switzerland|ZRH"
       "Auckland New Zealand|AKL" "Santiago Chile|SCL" "Prague Czechia|PRG" "Bali Indonesia|DPS"
       "Edinburgh Scotland|EDI" "Montreal Canada|YUL" "Lima Peru|LIM" "Krakow Poland|KRK"
       "Doha Qatar|DOH" "Bergen Norway|BGO" "Kyoto Japan|KIX" "Valencia Spain|VLC")
ORIG=("Denver DEN" "Miami MIA" "Atlanta ATL" "Dallas DFW" "Portland PDX" "Minneapolis MSP")

SVC_LOGS="logs/cap_mcp.log logs/cap_orch.log logs/cap_planner.log logs/cap_air.log logs/cap_hotel.log logs/cap_car.log"
ge(){ awk "BEGIN{exit !($1 >= $2)}"; }
spend(){ awk "BEGIN{printf \"%.2f\", $SPENT_SO_FAR + $1*$COST_PER_TRACE}"; }
minspec(){ awk "BEGIN{m=$air_n; if($hot_n<m)m=$hot_n; if($car_n<m)m=$car_n; print m}"; }

last=$(ls "$OUT"/trip_*.pcap 2>/dev/null | sed -E 's#.*/trip_0*([0-9]+)\.pcap#\1#' | sort -n | tail -1)
start_i=$(( ${last:-0} + 1 )); end_i=$(( start_i + N - 1 ))
echo "budget guard: stop at ~\$$BUDGET_USD est (cost/trace=\$$COST_PER_TRACE); or when min specialist >= $TARGET_SPECIALISTS"

ok=0; skipped=0; consec_skip=0; stop_reason=""
for i in $(seq "$start_i" "$end_i"); do
    if ge "$(spend "$ok")" "$BUDGET_USD"; then stop_reason="budget ceiling (est \$$(spend "$ok") >= \$$BUDGET_USD)"; break; fi
    if [ "$(minspec)" -ge "$TARGET_SPECIALISTS" ]; then stop_reason="target reached: all specialists >= $TARGET_SPECIALISTS"; break; fi
    if [ "$consec_skip" -ge 3 ]; then stop_reason="3 consecutive failed trips (credit exhausted? services down?)"; break; fi

    d=${DESTS[$(( (i-1) % ${#DESTS[@]} ))]}; dest=${d%%|*}; apt=${d##*|}
    o=${ORIG[$(( (i-1) % ${#ORIG[@]} ))]}
    nights=$(( i % 5 + 2 )); pax=$(( i % 3 + 1 )); bud=$(( 5000 + (i%6)*1000 ))
    q="Book a complete round trip from $o to $dest for $pax adult traveler(s). Depart 2026-09-0$(( i % 8 + 1 )), return 2026-09-1$(( i % 8 + 1 )), economy class, nearest major airport $apt, total budget about \$$bud. I want ALL of: (1) round-trip flights, (2) a 4-star hotel near the city centre for $nights nights, and (3) a mid-size rental car for the whole stay. Proceed with your best options and BOOK ALL THREE now — flights, hotel, and car. Do not ask clarifying questions; assume sensible defaults."
    id=$(printf "trip_%03d" "$i")

    sz=0; tries=0; skip=0
    while : ; do
        tcpdump -i "$IFACE" -s 96 -w "$OUT/$id.pcap" "tcp portrange $PORTS" 2>/dev/null & tpid=$!
        sleep 0.8
        uv run --env-file .env python "$DRIVER" "$q" > "logs/cap_drive_$id.log" 2>&1 || true
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

    # live specialist tally for THIS trip
    a=$(spec_in_pcap "$OUT/$id.pcap" 10103); h=$(spec_in_pcap "$OUT/$id.pcap" 10104); c=$(spec_in_pcap "$OUT/$id.pcap" 10105)
    air_n=$((air_n+a)); hot_n=$((hot_n+h)); car_n=$((car_n+c))
    printf '{"system":"a2a_mcp","instance":2,"boosted":true,"litellm_model":"%s","trace_id":"%s","input_prompt":"%s","agent_endpoints":{"mcp":"127.0.0.1:10100","orchestrator":"127.0.0.1:10101","planner":"127.0.0.1:10102","air_ticketing":"127.0.0.1:10103","hotel":"127.0.0.1:10104","car_rental":"127.0.0.1:10105"}}\n' "$INST2_MODEL" "$id" "$q" > "$OUT/$id.json"
    ok=$((ok+1)); consec_skip=0
    printf "  [trip %03d] %-24s pcap=%7sB fan[a%d h%d c%d] | tot air=%d hotel=%d car=%d | ok=%d est≈\$%s\n" \
        "$i" "$dest" "$sz" "$a" "$h" "$c" "$air_n" "$hot_n" "$car_n" "$ok" "$(spend "$ok")"
    sleep "${PACE:-3}"        # inter-trip spacing; raise (PACE=12) to avoid RPM throttle → higher fan-out yield
done

echo ""; echo "── 6-way top-up complete ──"
[ -n "$stop_reason" ] && echo "stopped: $stop_reason"
echo "new trips this run: $ok  (skipped: $skipped)"
echo "specialist totals (inst2): air=$air_n hotel=$hot_n car=$car_n  (target $TARGET_SPECIALISTS)"
echo "estimated spend this run: ~\$$(spend "$ok")   <<< TRUTH = Google Cloud billing >>>"
echo "next: extract data/raw_offtheshelf_inst2 then run scripts/evaluate_cross_instance_transfer.py"
