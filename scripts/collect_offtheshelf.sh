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
# QUOTA / BUDGET — one trip uses ~15-20 gemini-2.5-flash calls.
#   * Free tier (~RequestsPerDay cap): expect ~1 trace/run; the script HARD-STOPS on a
#     RequestsPerDay 429 (and warns you may still be on the free tier).
#   * Paid tier (Tier 1+): transient rate-limit 429s (RPM/TPM) are retried with backoff;
#     a per-run ESTIMATED-spend ceiling (BUDGET_USD) stops the run before it overspends.
#
# Numbering ACCUMULATES (continues from the highest existing trip), so runs are additive.
# Empty/failed captures are discarded so they never overwrite good data.
#
# Prereq (one-time, for non-root tcpdump):   sudo chmod o+r /dev/bpf*
# Run (NO sudo):
#   bash scripts/collect_offtheshelf.sh [N]                       # attempt up to N trips this run
#   BUDGET_USD=10 COST_PER_TRACE=0.11 bash scripts/collect_offtheshelf.sh 15   # calibration batch
#   SPENT_SO_FAR=0.9 COST_PER_TRACE=0.06 BUDGET_USD=10 bash scripts/collect_offtheshelf.sh 140  # bulk
#
# Requires the 8 compat fixes already applied to ~/a2a-samples (.../a2a_mcp) and
# its drive_orch.py multi-turn A2A driver.
set -u

SAMPLE_DIR="$HOME/a2a-samples/samples/python/agents/a2a_mcp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_DIR/data/raw_offtheshelf"
IFACE="lo0"; PORTS="10100-10105"
N="${1:-4}"                                # max trips to ATTEMPT this run

# ── Budget / rate-limit controls (paid tier) ──────────────────────────────────
BUDGET_USD="${BUDGET_USD:-10.00}"          # hard ceiling on ESTIMATED spend; run stops before crossing it
COST_PER_TRACE="${COST_PER_TRACE:-0.11}"   # $/trace for the guard. START conservative; LOWER it to the real
                                           #   value from the Cloud billing dashboard after the calibration batch
SPENT_SO_FAR="${SPENT_SO_FAR:-0}"          # USD already spent in earlier paid batches (carry across staged runs)
MAX_RETRIES="${MAX_RETRIES:-2}"            # transient-429 (RPM/TPM) retries per trip before giving up
BACKOFF_S="${BACKOFF_S:-30}"               # seconds to wait before retrying a transient 429

[ -f "$SAMPLE_DIR/drive_orch.py" ] || { echo "ERROR: $SAMPLE_DIR/drive_orch.py missing"; exit 1; }
grep -qE '^GOOGLE_API_KEY=.{10,}' "$SAMPLE_DIR/.env" 2>/dev/null || { echo "ERROR: GOOGLE_API_KEY not set in $SAMPLE_DIR/.env"; exit 1; }
mkdir -p "$OUT" "$SAMPLE_DIR/logs"
cd "$SAMPLE_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env 2>/dev/null; set +a          # load key silently
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"   # litellm reads GEMINI_API_KEY

rm -f logs/cap_*.log   # clear STALE logs so the 429 check sees only THIS run (else a prior run's quota hit stops us instantly)

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
       "Cairo Egypt|CAI" "Reykjavik Iceland|KEF" "Bangkok Thailand|BKK" "Lisbon Portugal|LIS"
       "Athens Greece|ATH" "Dubai UAE|DXB" "Singapore|SIN" "Toronto Canada|YYZ"
       "Mexico City Mexico|MEX" "Cape Town South Africa|CPT" "Oslo Norway|OSL" "Seoul Korea|ICN"
       "Buenos Aires Argentina|EZE" "Vienna Austria|VIE" "Istanbul Turkey|IST" "Nairobi Kenya|NBO")
ORIG=("New York JFK" "San Francisco SFO" "Chicago ORD" "Los Angeles LAX" "Boston BOS" "Seattle SEA")

SVC_LOGS="logs/cap_mcp.log logs/cap_orch.log logs/cap_planner.log logs/cap_air.log logs/cap_hotel.log logs/cap_car.log"
ge(){ awk "BEGIN{exit !($1 >= $2)}"; }                                       # exit 0 (true) iff $1 >= $2
spend(){ awk "BEGIN{printf \"%.2f\", $SPENT_SO_FAR + $1*$COST_PER_TRACE}"; }  # est USD after $1 new traces

# Continue numbering from the highest existing trip so runs ACCUMULATE (additive).
last=$(ls "$OUT"/trip_*.pcap 2>/dev/null | sed -E 's#.*/trip_0*([0-9]+)\.pcap#\1#' | sort -n | tail -1)
start=$(( ${last:-0} + 1 )); end=$(( start + N - 1 ))
[ -n "$last" ] && echo "resuming: $last trace(s) present -> attempting trip_$(printf '%03d' "$start")..$(printf '%03d' "$end")"
echo "budget guard: stop at ~\$$BUDGET_USD est  (cost/trace=\$$COST_PER_TRACE, prior spend=\$$SPENT_SO_FAR)"

ok=0; skipped=0; consec_skip=0; stop_reason=""
for i in $(seq "$start" "$end"); do
    # ── budget guard (pre-trip): never start a trip that would cross the ceiling ──
    if ge "$(spend "$ok")" "$BUDGET_USD"; then stop_reason="budget ceiling (est \$$(spend "$ok") >= \$$BUDGET_USD)"; break; fi
    # ── circuit-breaker: 3 failures in a row => credit exhausted / services down ──
    if [ "$consec_skip" -ge 3 ]; then stop_reason="3 consecutive failed trips (prepaid credit exhausted? services down?)"; break; fi

    d=${DESTS[$(( (i-1) % ${#DESTS[@]} ))]}; dest=${d%%|*}; apt=${d##*|}
    o=${ORIG[$(( (i-1) % ${#ORIG[@]} ))]}
    q="Plan a trip from $o to $dest. Depart 2026-05-0$(( i % 9 + 1 )), return 2026-05-1$(( i % 9 )), 2 adults, economy class, budget 6000 USD, prefer airport $apt. Book round-trip flights, a hotel for 5 nights, and a rental car."
    id=$(printf "trip_%03d" "$i")

    sz=0; tries=0; skip=0
    while : ; do
        tcpdump -i "$IFACE" -s 96 -w "$OUT/$id.pcap" "tcp portrange $PORTS" 2>/dev/null & tpid=$!
        sleep 0.8
        uv run --env-file .env python drive_orch.py "$q" > "logs/cap_drive_$id.log" 2>&1 || true
        sleep 0.8
        kill "$tpid" 2>/dev/null; wait "$tpid" 2>/dev/null
        sz=$(stat -f%z "$OUT/$id.pcap" 2>/dev/null || echo 0)
        [ "${sz:-0}" -gt 400 ] && break                       # usable capture -> success (pcap is the artifact)

        # trip produced no usable traffic — classify the failure
        if grep -qiE "PerDay|RequestsPerDay" $SVC_LOGS 2>/dev/null; then
            stop_reason="RequestsPerDay 429 — DAILY cap hit (still on FREE tier? confirm Tier-1 billing is active)"; sz=-1; break
        fi
        tries=$((tries+1))
        if [ "$tries" -gt "$MAX_RETRIES" ]; then echo "  ~ $id failed ${tries}x (not a daily cap) — skipping this trip"; skip=1; break; fi
        echo "  ~ $id empty/failed — backoff ${BACKOFF_S}s then retry $tries/$MAX_RETRIES"
        sleep "$BACKOFF_S"
    done
    [ "$sz" = "-1" ] && break                                  # daily cap -> hard stop the run
    if [ "$skip" = "1" ]; then rm -f "$OUT/$id.pcap"; skipped=$((skipped+1)); consec_skip=$((consec_skip+1)); sleep 2; continue; fi

    printf '{"system":"a2a_mcp","trace_id":"%s","input_prompt":"%s","agent_endpoints":{"mcp":"127.0.0.1:10100","orchestrator":"127.0.0.1:10101","planner":"127.0.0.1:10102","air_ticketing":"127.0.0.1:10103","hotel":"127.0.0.1:10104","car_rental":"127.0.0.1:10105"}}\n' "$id" "$q" > "$OUT/$id.json"
    ok=$((ok+1)); consec_skip=0
    printf "  [trip %03d] %-24s pcap=%7sB | ok=%d skip=%d est≈\$%s\n" "$i" "$dest" "$sz" "$ok" "$skipped" "$(spend "$ok")"
    sleep 3
done

total=$(ls "$OUT"/*.pcap 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "── run complete ──"
[ -n "$stop_reason" ] && echo "stopped: $stop_reason"
echo "new traces this run: $ok   (skipped: $skipped)   total pcaps now: $total"
echo "estimated spend (prior + this run): ~\$$(spend "$ok")   <<< TRUTH = Google Cloud billing dashboard >>>"
echo "extract:  venv/bin/python scripts/extract_offtheshelf.py --raw $OUT --out data/processed_offtheshelf --scapy"
