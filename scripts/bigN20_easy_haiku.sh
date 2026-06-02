#!/bin/bash
# N=20 concurrent rollouts: Haiku as planner, qwen as scout (substrate
# unchanged from bigN20_easy_qwen.sh). Same difficulty, milestones, goal,
# turn budget, spawn range — the only knob flipped is the brain.
#
# Point: capability-vs-substrate measurement. Same substrate as the qwen
# daily-driver run, so any delta in tier reached / survival / failure
# distribution is attributable to the planner, not the harness. Pairs
# with results/bigN20-easy-qwen-* for A/B.
#
# Scout stays as qwen (CRAFT_SCOUT_FANOUT_MODEL/UNIFY_MODEL) deliberately:
# the prior Haiku runs (e.g. iron_to_bed_test.sh) used this split, and we
# want the *substrate* identical so the comparison isolates the planner.
#
# Prereq: agents 0..19 launched and in-world.
#
# Usage:
#   ./scripts/bigN20_easy_haiku.sh

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
HAIKU="claude-haiku-4-5"
MODEL="${CRAFT_MODEL:-$HAIKU}"
TURNS="${CRAFT_TURNS:-30}"
GOAL="${CRAFT_GOAL:-minimal}"
DIFFICULTY="${CRAFT_DIFFICULTY:-easy}"
SPAWN_RANGE="${CRAFT_SPAWN_RANGE:-20000}"
PHASE="${CRAFT_PHASE:-dawn}"
N_AGENTS="${N_AGENTS:-20}"
START_AGENT="${START_AGENT:-0}"

# Scout stays on qwen — same substrate as the daily driver.
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MINE_FORCE_XRAY=1

unset CRAFT_MILESTONES   # default chain: M1_iron_goal + M2_diamond_goal
unset CRAFT_NUDGES       # default: food_low + tiered night_bed

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set — Haiku planner needs it." >&2
    exit 2
fi

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/bigN20-easy-haiku-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
{
    echo "[$(date '+%H:%M:%S')] BIGN20 EASY HAIKU starting (out=$OUT)"
    echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "[$(date '+%H:%M:%S')] N=$N_AGENTS turns=$TURNS goal=$GOAL phase=$PHASE diff=$DIFFICULTY model=$MODEL"
    echo "[$(date '+%H:%M:%S')] milestones=default(M1+M2) nudges=default scout=qwen LOOK_AROUND_MAX_RADIUS=1 MINE_FORCE_XRAY=1"
} | tee -a "$INDEX"

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
        .venv/bin/python -m craft.agent "$TURNS" "$GOAL" \
            --model "$MODEL" \
            --start-phase "$PHASE" \
            --random-spawn-range "$SPAWN_RANGE" \
            --difficulty "$DIFFICULTY" \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns_run
    turns_run=$(grep -c "=== turn [0-9]*/${TURNS}: planning ===" "$log" 2>/dev/null || echo "?")
    local died
    died=$(grep -c "YOU DIED" "$log" 2>/dev/null)
    died=${died:-0}
    local m1
    m1=$(grep -c "M1_iron_goal" "$jsonl" 2>/dev/null)
    m1=${m1:-0}
    local m2
    m2=$(grep -c "M2_diamond_goal" "$jsonl" 2>/dev/null)
    m2=${m2:-0}
    local cause=""
    if [ "$died" -gt 0 ]; then
        cause=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    fi
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns_run} dur=${dur}s died=${died} m1_fires=${m1} m2_fires=${m2}${cause:+ cause=$cause}" | tee -a "$INDEX"
}

for n in $(seq "$START_AGENT" $((START_AGENT + N_AGENTS - 1))); do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL ROLLOUTS DONE (out=$OUT)" | tee -a "$INDEX"
