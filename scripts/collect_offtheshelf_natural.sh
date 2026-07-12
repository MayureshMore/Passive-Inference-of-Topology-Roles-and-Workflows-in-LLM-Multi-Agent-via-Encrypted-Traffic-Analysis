#!/usr/bin/env bash
#
# TASK 2 (paid, $15 cap) — collect a CLEAN, NATURAL instance-2 specialist set to DE-CONFOUND the
# §9b 6-way. Instance-2's committed specialists were topped up with the fan-out-BOOSTED driver
# (forced full-service prompt + completion-forcing answer), which shifted their feature
# distributions (0/3 comparable — specialist_distribution_check). This collects specialists the
# natural way, so the 6-way can be re-run without the driver confound.
#
# INSTANCE-2 SELF-CONSISTENT (the corrected profile). New specialists use instance-2's OWN natural
# config, matching the 67 reused natural trips — NOT instance-1's prompt (that would Frankenstein
# two prompt regimes inside instance-2):
#   * prompt : instance-2's reworded "Arrange complete travel …" query (identical to
#              collect_offtheshelf_inst2.sh — same DESTS/ORIG/nights/pax/budget)
#   * LLM    : gemini-2.0-flash (instance-2's independence axis)
#   * driver : drive_orch_natural_fixed.py  ← the ONE intended change: BUG-FIXED-BUT-NOT-FORCED.
#              destination-agnostic (fixes the Tokyo-hardcoded ~6% fan-out bug) yet NOT
#              completion-forcing (unlike drive_orch_boost.py, which shifts distributions).
# The de-confound is "natural driver instead of boosted", NOT "borrow the victim's prompt".
#
# SEED: reuses the 67 natural instance-2 trips already collected (4/4/4 natural specialists) by
# copying the NON-boosted ones into the fresh natural dir, so prior natural spend is not wasted;
# only +11 each are needed. Boosted trips are never copied.
#
# PROBE-THEN-PROJECT gate: after PROBE_N new trips, measure the ACTUAL natural fan-out rate,
# project trips+cost to reach ≥15 each, and STOP-AND-REPORT if the projection exceeds the $15 cap
# — it never brute-forces. A blocked outcome is acceptable: §9b then stays honestly
# driver-confounded and the future-work note stands.
#
# ADDITIVE: writes only into data/raw_offtheshelf_inst2_natural/. Real Gemini spend — TRUTH is
# Google Cloud billing; the est is a guard, not the meter.
#
# Usage (approved at $15 cap):
#   BUDGET_USD=15 TARGET_SPECIALISTS=15 PROBE_N=30 PACE=12 bash scripts/collect_offtheshelf_natural.sh
# Then (distribution check is the GATE for the 6-way verdict):
#   MIN_N=10 venv/bin/python scripts/evaluate_cross_instance_transfer.py --inst2 data/raw_offtheshelf_inst2_natural
set -u

SAMPLE_DIR="$HOME/a2a-samples/samples/python/agents/a2a_mcp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_DIR/data/raw_offtheshelf_inst2"                # seed source (natural trips reused)
OUT="$REPO_DIR/data/raw_offtheshelf_inst2_natural"        # clean natural-only instance-2 set
IFACE="lo0"; PORTS="10100-10105"
N="${1:-260}"                                             # max trips to ATTEMPT (guards stop earlier)

INST2_MODEL="gemini/gemini-2.0-flash"                     # unchanged independence axis (different LLM)
DRIVER="drive_orch_natural_fixed.py"                      # BUG-FIXED-BUT-NOT-FORCED natural driver

BUDGET_USD="${BUDGET_USD:-15.00}"                         # HARD ceiling on ESTIMATED spend
COST_PER_TRACE="${COST_PER_TRACE:-0.045}"                # ~observed natural trip cost
TARGET_SPECIALISTS="${TARGET_SPECIALISTS:-15}"           # stop when min(air,hotel,car) reaches this
PROBE_N="${PROBE_N:-30}"                                  # trips before the project-or-stop gate
SPENT_SO_FAR="${SPENT_SO_FAR:-0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BACKOFF_S="${BACKOFF_S:-30}"

[ -f "$SAMPLE_DIR/$DRIVER" ] || { echo "BLOCKED: $SAMPLE_DIR/$DRIVER missing"; exit 1; }
grep -qE '^GOOGLE_API_KEY=.{10,}' "$SAMPLE_DIR/.env" 2>/dev/null || { echo "BLOCKED: GOOGLE_API_KEY not set in $SAMPLE_DIR/.env"; exit 1; }
mkdir -p "$OUT" "$SAMPLE_DIR/logs"

# ── seed: copy NON-boosted trips from the existing inst2 dir (reuse natural spend) ──
spec_in_pcap(){ tcpdump -r "$1" -nn "tcp port $2" 2>/dev/null | grep -q . && echo 1 || echo 0; }
seeded=0
for j in "$SRC"/trip_*.json "$SRC"/trace_*.json; do
    [ -e "$j" ] || continue
    grep -q '"boosted":[[:space:]]*true' "$j" 2>/dev/null && continue   # skip boosted
    base="$(basename "${j%.json}")"; p="$SRC/$base.pcap"
    [ -e "$p" ] || continue
    if [ ! -e "$OUT/$base.pcap" ]; then cp "$p" "$OUT/$base.pcap"; cp "$j" "$OUT/$base.json"; seeded=$((seeded+1)); fi
done
echo "seeded $seeded natural trips from $SRC (boosted trips excluded)"

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

# ── specialist counters: init from the (now seeded) natural OUT dir ──
air_n=0; hot_n=0; car_n=0
for f in "$OUT"/*.pcap; do
    [ -e "$f" ] || continue
    [ "$(spec_in_pcap "$f" 10103)" = 1 ] && air_n=$((air_n+1))
    [ "$(spec_in_pcap "$f" 10104)" = 1 ] && hot_n=$((hot_n+1))
    [ "$(spec_in_pcap "$f" 10105)" = 1 ] && car_n=$((car_n+1))
done
seed_min=$air_n; [ "$hot_n" -lt "$seed_min" ] && seed_min=$hot_n; [ "$car_n" -lt "$seed_min" ] && seed_min=$car_n
echo "natural specialist counts (seed): air=$air_n hotel=$hot_n car=$car_n  target=$TARGET_SPECIALISTS"

echo "NATURAL top-up  LITELLM_MODEL=$LITELLM_MODEL  budget=\$$BUDGET_USD  driver=$DRIVER  out=$OUT"
start "uv run --env-file .env a2a-mcp --run mcp-server --transport sse --port 10100" mcp
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/orchestrator_agent.json --port 10101" orch
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/planner_agent.json --port 10102" planner
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/air_ticketing_agent.json --port 10103" air
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/hotel_booking_agent.json --port 10104" hotel
start "uv run --env-file .env src/a2a_mcp/agents/ --agent-card agent_cards/car_rental_agent.json --port 10105" car
echo "waiting 14s for init..."; sleep 14

# instance-2's OWN reworded query + destinations (identical to collect_offtheshelf_inst2.sh).
DESTS=("Tokyo Japan|NRT" "Paris France|CDG" "Rome Italy|FCO" "Sydney Australia|SYD"
       "Cairo Egypt|CAI" "Reykjavik Iceland|KEF" "Bangkok Thailand|BKK" "Lisbon Portugal|LIS"
       "Athens Greece|ATH" "Dubai UAE|DXB" "Singapore|SIN" "Toronto Canada|YYZ"
       "Mexico City Mexico|MEX" "Cape Town South Africa|CPT" "Oslo Norway|OSL" "Seoul Korea|ICN"
       "Buenos Aires Argentina|EZE" "Vienna Austria|VIE" "Istanbul Turkey|IST" "Nairobi Kenya|NBO")
ORIG=("New York JFK" "San Francisco SFO" "Chicago ORD" "Los Angeles LAX" "Boston BOS" "Seattle SEA")

SVC_LOGS="logs/cap_mcp.log logs/cap_orch.log logs/cap_planner.log logs/cap_air.log logs/cap_hotel.log logs/cap_car.log"
ge(){ awk "BEGIN{exit !($1 >= $2)}"; }
spend(){ awk "BEGIN{printf \"%.2f\", $SPENT_SO_FAR + $1*$COST_PER_TRACE}"; }
minspec(){ awk "BEGIN{m=$air_n; if($hot_n<m)m=$hot_n; if($car_n<m)m=$car_n; print m}"; }

last=$(ls "$OUT"/trip_*.pcap 2>/dev/null | sed -E 's#.*/trip_0*([0-9]+)\.pcap#\1#' | sort -n | tail -1)
start_i=$(( ${last:-0} + 1 )); end_i=$(( start_i + N - 1 ))
echo "PROBE gate: after $PROBE_N new trips, project trips+cost to reach target; STOP if projection > \$$BUDGET_USD"

ok=0; skipped=0; consec_skip=0; stop_reason=""; probed=0
for i in $(seq "$start_i" "$end_i"); do
    if ge "$(spend "$ok")" "$BUDGET_USD"; then stop_reason="budget ceiling (est \$$(spend "$ok") >= \$$BUDGET_USD)"; break; fi
    if [ "$(minspec)" -ge "$TARGET_SPECIALISTS" ]; then stop_reason="target reached: all specialists >= $TARGET_SPECIALISTS"; break; fi
    if [ "$consec_skip" -ge 3 ]; then stop_reason="3 consecutive failed trips (credit exhausted? services down?)"; break; fi

    # ── PROBE-THEN-PROJECT gate (once, after PROBE_N new trips) ──
    if [ "$probed" = 0 ] && [ "$ok" -ge "$PROBE_N" ]; then
        probed=1
        gained=$(( $(minspec) - seed_min ))                       # new min-specialists in the probe
        need=$(( TARGET_SPECIALISTS - $(minspec) ))
        rate=$(awk "BEGIN{printf \"%.4f\", ($gained>0)?$gained/$ok:0}")
        if awk "BEGIN{exit !($rate<=0)}"; then
            stop_reason="PROBE gate: 0 new specialists in $ok trips — natural fan-out ~0 this session; cannot project. STOP (raise driver quality or accept boosted 6-way)."
            break
        fi
        proj_trips=$(awk "BEGIN{printf \"%d\", ($need/$rate)+0.999}")
        proj_cost=$(awk "BEGIN{printf \"%.2f\", $SPENT_SO_FAR + ($ok+$proj_trips)*$COST_PER_TRACE}")
        echo "  [PROBE] $ok trips → +$gained min-specialists (rate=$rate/trip). need +$need more → ~$proj_trips trips, projected total est \$$proj_cost"
        if ge "$proj_cost" "$BUDGET_USD"; then
            stop_reason="PROBE gate: projected cost \$$proj_cost >= cap \$$BUDGET_USD — STOP, do not brute-force. §9b stays driver-confounded; report blocked."
            break
        fi
        echo "  [PROBE] projection \$$proj_cost within cap — continuing to target."
    fi

    d=${DESTS[$(( (i-1) % ${#DESTS[@]} ))]}; dest=${d%%|*}; apt=${d##*|}
    o=${ORIG[$(( (i-1) % ${#ORIG[@]} ))]}
    nights=$(( i % 4 + 3 )); pax=$(( i % 3 + 1 )); bud=$(( 4000 + (i%5)*1000 ))
    q="Arrange complete travel from $o to $dest for $pax traveler(s). Outbound 2026-08-1$(( i % 9 )), return 2026-08-2$(( i % 9 )), premium economy, roughly \$$bud budget, nearest airport $apt. I need round-trip air tickets, $nights nights of hotel, and a rental vehicle."
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

    a=$(spec_in_pcap "$OUT/$id.pcap" 10103); h=$(spec_in_pcap "$OUT/$id.pcap" 10104); c=$(spec_in_pcap "$OUT/$id.pcap" 10105)
    air_n=$((air_n+a)); hot_n=$((hot_n+h)); car_n=$((car_n+c))
    printf '{"system":"a2a_mcp","instance":2,"boosted":false,"driver":"drive_orch_natural_fixed.py","litellm_model":"%s","trace_id":"%s","input_prompt":"%s","agent_endpoints":{"mcp":"127.0.0.1:10100","orchestrator":"127.0.0.1:10101","planner":"127.0.0.1:10102","air_ticketing":"127.0.0.1:10103","hotel":"127.0.0.1:10104","car_rental":"127.0.0.1:10105"}}\n' "$INST2_MODEL" "$id" "$q" > "$OUT/$id.json"
    ok=$((ok+1)); consec_skip=0
    printf "  [trip %03d] %-22s pcap=%7sB fan[a%d h%d c%d] | tot air=%d hotel=%d car=%d | ok=%d est≈\$%s\n" \
        "$i" "$dest" "$sz" "$a" "$h" "$c" "$air_n" "$hot_n" "$car_n" "$ok" "$(spend "$ok")"
    sleep "${PACE:-3}"
done

echo ""; echo "── NATURAL top-up complete ──"
[ -n "$stop_reason" ] && echo "stopped: $stop_reason"
echo "new trips this run: $ok  (skipped: $skipped)"
echo "natural specialist totals: air=$air_n hotel=$hot_n car=$car_n  (target $TARGET_SPECIALISTS)"
echo "estimated spend this run: ~\$$(spend "$ok")   <<< TRUTH = Google Cloud billing >>>"
echo "next: MIN_N=10 venv/bin/python scripts/evaluate_cross_instance_transfer.py --inst2 $OUT"
