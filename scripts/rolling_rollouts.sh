#!/bin/bash
# Rolling rollouts: each of 5 agent slots respawns a fresh craft.agent
# whenever its current rollout exits (death, max_turns, or empty-response
# qwen quirk). The MC world clock evolves naturally — no --start-phase
# reset between rollouts, so spawn-time distribution falls out of when
# each respawn happens to fire.
#
# Stop with Ctrl-C (forwards SIGTERM to all child loops); each in-flight
# rollout finishes its current turn then exits cleanly.
#
# Each rollout's JSONL header includes a `spawn` block with day_ticks,
# biome, and xyz so the spawn-time distribution can be reconstructed
# from headers alone.

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
# 2026-05-20: force baritone x-ray on iron/coal/diamond — qwen historically
# picks fair=true (blind branch-mine) on 33% of iron and 95% of diamond
# calls. mine_stone is exempt (forces fair=true tool-side already).
export CRAFT_MINE_FORCE_XRAY=1

TODAY=$(date '+%Y%m%d')
OUT="results/rolling-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] rolling rollouts starting (out=$OUT)" | tee -a "$INDEX"

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — stopping" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_agent_forever() {
    local n=$1
    local port=$((25570 + n))
    local i=0
    while true; do
        i=$((i+1))
        local ts
        ts=$(date '+%Y%m%d-%H%M%S')
        local jsonl="$OUT/agent${n}-r${i}-${ts}.jsonl"
        local log="$OUT/agent${n}-r${i}-${ts}.log"
        local t0
        t0=$(date +%s)
        echo "[$(date '+%H:%M:%S')] agent${n} r${i} starting" | tee -a "$INDEX"
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent${n}" \
            .venv/bin/python -m craft.agent 9999 minimal \
                --model "$QWEN" \
                --random-spawn-range 20000 \
                --jsonl-out "$jsonl" \
                > "$log" 2>&1 || true
        local dur=$(($(date +%s) - t0))
        local turns
        turns=$(grep -c "=== turn [0-9]*/9999: planning ===" "$log" 2>/dev/null || echo "?")
        local death
        death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
        echo "[$(date '+%H:%M:%S')] agent${n} r${i} ended turns=${turns} dur=${dur}s${death:+ death=$death}" | tee -a "$INDEX"
        # Brief pause so Wurst's AutoRespawn finishes clicking through the
        # respawn screen before the next process tries to /position.
        sleep 4
    done
}

for n in 0 1 2 3 4; do
    run_agent_forever "$n" &
done

wait
