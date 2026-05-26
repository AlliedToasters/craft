#!/bin/bash
# Behavioral test: does the agent cook raw meat when given the materials
# under hunger pressure?
#
# Third loadout/substrate-isolation test (after sleep + hunt). Composes
# from existing primitives only — place(furnace) → smelt(beef, coal) →
# collect_smelt → AutoEat consumes the cooked output. No new tool. The
# question is whether the model reaches for the chain.
#
# Conditions:
#   - Defaults to Haiku since qwen 4B has been observed to skip new
#     verbs (see hunt_passive fan-out 2026-05-21: 0/5 qwen calls vs
#     4 Haiku calls). qwen still selectable via CRAFT_MODEL=$QWEN.
#   - --starting-loadout cook_kitchen (raw beef + coal + furnace + sword)
#   - --start-phase noon (full day for cooking)
#   - --random-spawn-range 20000
#   - --difficulty easy by default — peaceful freezes food meter
#     (see project_hunt_loadout_findings.md).
#   - max_turns 30 (place + smelt + collect = ~5 turns; rest is buffer)
#
# Pass per agent: jsonl shows at least one of
#   (a) a smelt outcome that doesn't start with "FAILED:", OR
#   (b) inventory snapshot contains a cooked_* item at any turn, OR
#   (c) final food ≥ 18 (AutoEat consumed the cooked meat)
# Aggregate pass = ≥3/5 agents.
#
# Usage:
#   ./scripts/cook_loadout_test.sh
#   CRAFT_MODEL="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16" ./scripts/cook_loadout_test.sh
#   COOK_TEST_DIFFICULTY=peaceful ./scripts/cook_loadout_test.sh  # neutralize hunger

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
HAIKU="claude-haiku-4-5"
# Default to Haiku; the substrate question is "does any model use the
# chain?" and Haiku is the established baseline (qwen has documented
# uptake gaps on new verbs).
MODEL="${CRAFT_MODEL:-$HAIKU}"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MINE_FORCE_XRAY=1
# Cook-capability isolation: raw meat must NOT be auto-eaten before the agent
# cooks it, so the substrate stops feeding it raw beef. Requires the homunculus
# offhand-food curator + AutoEat Hands mode (preflight wires both).
export CRAFT_FOOD_POLICY="${CRAFT_FOOD_POLICY:-cooked_only}"

DIFFICULTY="${COOK_TEST_DIFFICULTY:-easy}"

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/cook-loadout-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] COOK LOADOUT TEST starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] loadout=cook_kitchen phase=noon max_turns=30 model=$MODEL difficulty=$DIFFICULTY" | tee -a "$INDEX"

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
            --starting-loadout cook_kitchen \
            --difficulty "$DIFFICULTY" \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns
    turns=$(grep -c "=== turn [0-9]*/30: planning ===" "$log" 2>/dev/null || echo "?")
    # smelt attempts (any call); successes = outcome doesn't start FAILED
    local smelt_attempts
    smelt_attempts=$(grep -c '"tool": "smelt"' "$jsonl" 2>/dev/null)
    smelt_attempts=${smelt_attempts:-0}
    # Successful smelts: outcome line doesn't start with FAILED. Match
    # smelt-tool entries that have an outcome NOT starting with FAILED.
    local smelt_successes
    smelt_successes=$(grep '"tool": "smelt"' "$jsonl" 2>/dev/null | grep -cv '"outcome": "FAILED' )
    smelt_successes=${smelt_successes:-0}
    # collect_smelt is the second half of the chain
    local collect_attempts
    collect_attempts=$(grep -c '"tool": "collect_smelt"' "$jsonl" 2>/dev/null)
    collect_attempts=${collect_attempts:-0}
    # cooked_* in any inventory snapshot = cooked something at some point
    local cooked_observed
    cooked_observed=$(grep -c '"minecraft:cooked_' "$jsonl" 2>/dev/null)
    cooked_observed=${cooked_observed:-0}
    local death
    death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns} dur=${dur}s smelt_attempts=${smelt_attempts} smelt_successes=${smelt_successes} collect_attempts=${collect_attempts} cooked_observed=${cooked_observed}${death:+ death=$death}" | tee -a "$INDEX"
}

for n in ${CRAFT_TEST_AGENTS:-0 1 2 3 4}; do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — test complete (out=$OUT)" | tee -a "$INDEX"
echo ""
echo "=== SUMMARY ===" | tee -a "$INDEX"
grep " ended " "$INDEX" | tee -a /dev/stderr | wc -l | xargs echo "rollouts completed:"
