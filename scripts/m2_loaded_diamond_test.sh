#!/bin/bash
# Behavioral test: does qwen-4B descend for diamond when started in the M2
# fire-condition (full iron armor + iron tools)?
#
# Hypothesis: the gap between "agent has iron armor + tools" and "agent
# reaches diamond" has been impossible to measure organically because
# qwen rarely assembles full iron armor on its own. With loaded rollouts
# we boot into the M2 condition directly; M2 fires turn 1; agent gets
# the y<=11 diamond-seeking nudge as part of its prompt from the start.
#
# Conditions:
#   - 5 concurrent agents (no barrier — async launch)
#   - --starting-loadout iron_armored (full iron suit + iron tools + food + torches)
#   - --start-phase dawn (start in light; will see natural dusk if survival is long)
#   - --random-spawn-range 20000 (broad biome sampling)
#   - non-peaceful (set_difficulty("easy") in _apply_setup is the default)
#   - CRAFT_MILESTONES=M1,M2 so M2 fires turn 1 with the diamond nudge
#   - max_turns 200 (enough runway for descent + tunneling + diamond mining)
#
# Pass: any rollout reaches diamond (handle_mine outcome contains
# "minecraft:diamond" or inventory shows minecraft:diamond at end).
# We're not gating on survival; even one diamond-find is signal.
#
# Usage:
#   ./scripts/m2_loaded_diamond_test.sh

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MILESTONES="M1_iron_goal,M2_diamond_goal"
# 2026-05-20: first iron-armored run had 5/5 agents pick fair=true blind
# tunneling and break their iron pickaxes long before reaching diamond.
# Force x-ray on all ore mining (exempt: mine_stone, which forces fair=true
# tool-side to avoid baritone's deep-target pathology).
export CRAFT_MINE_FORCE_XRAY=1

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/m2-loaded-diamond-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] M2 LOADED DIAMOND TEST starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] loadout=iron_armored phase=dawn max_turns=200 force_xray=${CRAFT_MINE_FORCE_XRAY}" | tee -a "$INDEX"

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
        .venv/bin/python -m craft.agent 200 bare \
            --model "$QWEN" \
            --start-phase dawn \
            --random-spawn-range 20000 \
            --starting-loadout iron_armored \
            --jsonl-out "$jsonl" \
            > "$log" 2>&1 || true
    local dur=$(($(date +%s) - t0))
    local turns
    turns=$(grep -c "=== turn [0-9]*/200: planning ===" "$log" 2>/dev/null || echo "?")
    local death
    death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
    # Diamond reach signals: handle_mine outcome containing diamond ore,
    # or final inventory record containing diamond items.
    local diamond_mined diamond_inv
    diamond_mined=$(grep -c "diamond_ore" "$log" 2>/dev/null || echo 0)
    diamond_inv=$(grep -c '"minecraft:diamond"' "$jsonl" 2>/dev/null || echo 0)
    local m2_fired
    m2_fired=$(grep -c '"milestone_fired": "M2_diamond_goal"' "$jsonl" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] agent${n} ended turns=${turns} dur=${dur}s m2=${m2_fired} diamond_mined=${diamond_mined} diamond_inv=${diamond_inv}${death:+ death=$death}" | tee -a "$INDEX"
}

for n in 0 1 2 3 4; do
    run_one "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — test complete (out=$OUT)" | tee -a "$INDEX"
echo ""
echo "=== SUMMARY ===" | tee -a "$INDEX"
grep " ended " "$INDEX" | tee -a /dev/stderr | wc -l | xargs echo "rollouts completed:"
