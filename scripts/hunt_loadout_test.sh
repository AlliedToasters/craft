#!/bin/bash
# Behavioral test: does qwen-4B hunt a passive mob at low hunger when
# given a sword + nearby herd (hunt_meadow) or just a sword (hunt_wild)?
#
# Second "isolated capability test" in the loadout pattern (see
# craft/loadouts.py — `hunt_meadow`, `hunt_wild`). Sword + KillAura
# is the established substrate; the new piece is hunt_passive composing
# scan_entities × baritone_goto × KillAura's passive filter.
#
# Conditions:
#   - 5 concurrent agents
#   - --starting-loadout ${CRAFT_HUNT_VARIANT:-hunt_meadow}
#       hunt_meadow: pre-summons 8 mobs around spawn (deterministic)
#       hunt_wild:   no summon, natural biome population (lottery)
#   - --start-phase noon (full day ahead, no shelter pressure)
#   - --random-spawn-range 20000
#   - difficulty easy by default — peaceful FREEZES food meter at 20
#     (hunger effect applies but does nothing), so the hunger-pressure
#     half of the test is a no-op under peaceful. easy lets hunger bite
#     while keeping mob damage manageable in daylight. flip via
#     HUNT_TEST_DIFFICULTY=peaceful to neuter food pressure on purpose.
#   - max_turns 30 (kill + cook + eat fits comfortably)
#
# Pass per agent: jsonl contains at least one tool=hunt_passive outcome
# starting with "hunted " (vs "FAILED" / "no_passives_in_range").
# Aggregate pass = ≥3/5 agents.
#
# RED canary: KillAura's passive-filter setting name is a best-guess
# ("Filter passive mobs"). If wrong, expect WARN logs in agent stdout
# but the substrate may still attack passives in some default configs.
# Smoke first with 1 agent before fan-out.
#
# Usage:
#   ./scripts/hunt_loadout_test.sh                   # hunt_meadow, peaceful
#   CRAFT_HUNT_VARIANT=hunt_wild ./scripts/hunt_loadout_test.sh
#   CRAFT_MODEL=claude-haiku-4-5 ./scripts/hunt_loadout_test.sh
#   HUNT_TEST_DIFFICULTY=easy ./scripts/hunt_loadout_test.sh

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

VARIANT="${CRAFT_HUNT_VARIANT:-hunt_meadow}"
DIFFICULTY="${HUNT_TEST_DIFFICULTY:-easy}"

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/hunt-loadout-${VARIANT}-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] HUNT LOADOUT TEST starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] loadout=$VARIANT phase=noon max_turns=30 model=$MODEL difficulty=$DIFFICULTY" | tee -a "$INDEX"

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
            --start-phase noon \
            --random-spawn-range 20000 \
            --starting-loadout "$VARIANT" \
            --difficulty "$DIFFICULTY" \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns
    turns=$(grep -c "=== turn [0-9]*/30: planning ===" "$log" 2>/dev/null || echo "?")
    # Match the turn-record key ("tool": ...), not the LLM-transcript key
    # ("name": ...) which is duplicated across cumulative transcript writes.
    local hunt_attempts
    hunt_attempts=$(grep -c '"tool": "hunt_passive"' "$jsonl" 2>/dev/null)
    hunt_attempts=${hunt_attempts:-0}
    # Successful hunts: tool outcome starts with "hunted " (vs "FAILED" / "no_passives_")
    local hunt_successes
    hunt_successes=$(grep -c '"outcome": "hunted ' "$jsonl" 2>/dev/null)
    hunt_successes=${hunt_successes:-0}
    local death
    death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns} dur=${dur}s hunt_attempts=${hunt_attempts} hunt_successes=${hunt_successes}${death:+ death=$death}" | tee -a "$INDEX"
}

for n in ${CRAFT_TEST_AGENTS:-0 1 2 3 4}; do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — test complete (out=$OUT)" | tee -a "$INDEX"
echo ""
echo "=== SUMMARY ===" | tee -a "$INDEX"
grep " ended " "$INDEX" | tee -a /dev/stderr | wc -l | xargs echo "rollouts completed:"
