#!/bin/bash
# Behavioral test: does qwen-4B place a bed and sleep at dusk when given
# a pre-crafted bed in inventory + impending night?
#
# This is the first "isolated capability test" in the loadout pattern
# (see craft/loadouts.py — `dusk_bed`). Bed crafting + sheep finding are
# explicitly removed; we want to measure the new substrate primitive
# (sleep_in_bed) under the simplest prerequisite condition.
#
# Conditions:
#   - 5 concurrent agents
#   - --starting-loadout dusk_bed (red_bed + cooked_beef + torches)
#   - --start-phase dusk (night arrives within ~2 minutes)
#   - --random-spawn-range 20000
#   - non-peaceful (default)
#   - max_turns 30 (sleep is fast; if the agent doesn't try it in 30
#     turns, they aren't going to)
#
# Pass per agent: jsonl contains at least one tool=sleep_in_bed outcome
# starting with "slept " (vs "FAILED"). Aggregate pass = ≥3/5 agents.
#
# This test is RED against current homunculus — /bed/place and /bed/sleep
# do not exist yet. Expect "FAILED: ... transport_error ..." outcomes
# from the sleep_in_bed tool until the spec'd endpoints ship. The RED
# state is the validation that the test path is wired correctly; the
# GREEN flip happens when homunculus catches up.
#
# Usage:
#   ./scripts/sleep_loadout_test.sh

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
# Main agent model. Default qwen daily-driver; override with CRAFT_MODEL.
# Scout models stay on qwen — they're a substrate utility, not under test.
MODEL="${CRAFT_MODEL:-$QWEN}"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MINE_FORCE_XRAY=1

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/sleep-loadout-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] SLEEP LOADOUT TEST starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] loadout=dusk_bed phase=dusk max_turns=30 model=$MODEL difficulty=${SLEEP_TEST_DIFFICULTY:-peaceful}" | tee -a "$INDEX"

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — stopping" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_one() {
    local n=$1
    local port=$((25570 + n))
    local ts
    ts=$(date '+%Y%m%d-%H%M%S')
    local jsonl="$OUT/agent${n}-${ts}.jsonl"
    local log="$OUT/agent${n}-${ts}.log"
    local t0
    t0=$(date +%s)
    echo "[$(date '+%H:%M:%S')] agent${n} starting" | tee -a "$INDEX"
    HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent${n}" \
        .venv/bin/python -m craft.agent 30 bare \
            --model "$MODEL" \
            --start-phase dusk \
            --random-spawn-range 20000 \
            --starting-loadout dusk_bed \
            --difficulty "${SLEEP_TEST_DIFFICULTY:-peaceful}" \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns
    turns=$(grep -c "=== turn [0-9]*/30: planning ===" "$log" 2>/dev/null || echo "?")
    # Match the turn-record key ("tool": ...), not the LLM-transcript key
    # ("name": ...) which is duplicated across cumulative transcript writes.
    # Drop the `|| echo 0` fallback — grep -c outputs "0" on no-match (exit 1)
    # and the OR doubled the value to "0\n0". Use :- default for missing file.
    local sleep_attempts
    sleep_attempts=$(grep -c '"tool": "sleep_in_bed"' "$jsonl" 2>/dev/null)
    sleep_attempts=${sleep_attempts:-0}
    # Successful sleeps: tool outcome starts with "slept " (vs "FAILED:")
    local sleep_successes
    sleep_successes=$(grep -c '"outcome": "slept ' "$jsonl" 2>/dev/null)
    sleep_successes=${sleep_successes:-0}
    local death
    death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns} dur=${dur}s sleep_attempts=${sleep_attempts} sleep_successes=${sleep_successes}${death:+ death=$death}" | tee -a "$INDEX"
}

for n in 0 1 2 3 4; do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — test complete (out=$OUT)" | tee -a "$INDEX"
echo ""
echo "=== SUMMARY ===" | tee -a "$INDEX"
grep " ended " "$INDEX" | tee -a /dev/stderr | wc -l | xargs echo "rollouts completed:"
