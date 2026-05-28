#!/bin/bash
# Behavioral test: does the agent complete the full wool→bed tech tree from
# 2 iron + planks + crafting_table at dusk?
#
# Chain under test:
#   shear_sheep (auto-crafts shears from 2 iron) → wool drops →
#   craft red_bed (3 wool + 3 planks) → place_bed → sleep_in_bed
#
# This stresses the new ShearReflex/AutoShears substrate AND the existing
# bed-placement/sleep code together. Compared to the pre-cooked dusk_bed
# scenario (bed already in hand, agent pre-sheltered) this is the "naked
# tech tree" version: agent starts in the open with only the raw materials.
#
# Conditions:
#   - 5 concurrent agents (agent0..agent4)
#   - --starting-loadout dusk_iron_to_bed (6 sheep pre-summoned at r=15)
#   - --start-phase dusk
#   - --random-spawn-range 20000
#   - difficulty=peaceful by default; IRON_BED_TEST_DIFFICULTY=easy to flip
#   - max_turns 40 (sleep is the win condition; if not slept by 40,
#     the agent isn't going to)
#
# Pass per agent: jsonl contains at least one tool=sleep_in_bed outcome
# starting with "slept ".
#
# Usage:
#   ./scripts/iron_to_bed_test.sh                          # peaceful
#   IRON_BED_TEST_DIFFICULTY=easy ./scripts/iron_to_bed_test.sh

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
MODEL="${CRAFT_MODEL:-$QWEN}"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MINE_FORCE_XRAY=1

DIFFICULTY="${IRON_BED_TEST_DIFFICULTY:-peaceful}"
TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/iron-to-bed-${DIFFICULTY}-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] IRON->BED TEST starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] loadout=dusk_iron_to_bed phase=dusk max_turns=40 model=$MODEL difficulty=$DIFFICULTY" | tee -a "$INDEX"

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
        .venv/bin/python -m craft.agent 40 bare \
            --model "$MODEL" \
            --start-phase dusk \
            --random-spawn-range 20000 \
            --starting-loadout dusk_iron_to_bed \
            --difficulty "$DIFFICULTY" \
            --record-video \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns
    turns=$(grep -c "=== turn [0-9]*/40: planning ===" "$log" 2>/dev/null || echo "?")
    local shear_attempts
    shear_attempts=$(grep -c '"tool": "shear_sheep"' "$jsonl" 2>/dev/null)
    shear_attempts=${shear_attempts:-0}
    local wool_outcomes
    wool_outcomes=$(grep -c '"outcome": "sheared ' "$jsonl" 2>/dev/null)
    wool_outcomes=${wool_outcomes:-0}
    local bed_crafts
    bed_crafts=$(grep -c '"outcome": "crafted .*_bed' "$jsonl" 2>/dev/null)
    bed_crafts=${bed_crafts:-0}
    local sleep_attempts
    sleep_attempts=$(grep -c '"tool": "sleep_in_bed"' "$jsonl" 2>/dev/null)
    sleep_attempts=${sleep_attempts:-0}
    local sleep_successes
    sleep_successes=$(grep -c '"outcome": "slept ' "$jsonl" 2>/dev/null)
    sleep_successes=${sleep_successes:-0}
    local death
    death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns} dur=${dur}s shear=${shear_attempts} wool=${wool_outcomes} bed_craft=${bed_crafts} sleep_try=${sleep_attempts} sleep_ok=${sleep_successes}${death:+ death=$death}" | tee -a "$INDEX"
}

for n in 0 1 2 3 4; do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — test complete (out=$OUT)" | tee -a "$INDEX"
echo ""
echo "=== SUMMARY ===" | tee -a "$INDEX"
grep " ended " "$INDEX" | tee -a /dev/stderr | wc -l | xargs echo "rollouts completed:"
