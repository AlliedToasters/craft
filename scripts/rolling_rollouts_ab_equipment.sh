#!/bin/bash
# A/B variant of rolling_rollouts.sh — splits each slot's infinite loop
# between CRAFT_EQUIPMENT_READOUT=1 (treatment: Equipment block in STATE)
# and =0 (control: legacy raw inventory only) via a per-rollout coin flip.
#
# Each rollout picks its arm with $((RANDOM % 2)). Over a long enough run
# both arms get ~50/50 across all 5 slots, all spawn times, and all
# biomes — removing per-slot and per-clock confounds.
#
# The JSONL header records `equipment_readout: bool`, so post-hoc analysis
# filters arms by header field. Filename and orchestrator log line also
# carry the arm tag (`-ON-` or `-OFF-`) for fast eyeballing.
#
# Stop with Ctrl-C (forwards SIGTERM); each in-flight rollout finishes
# its current turn then exits cleanly.

set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1

TODAY=$(date '+%Y%m%d')
OUT="results/rolling-ab-equipment-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] rolling A/B (equipment readout) starting (out=$OUT)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — stopping" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_agent_forever() {
    local n=$1
    local port=$((25570 + n))
    local i=0
    while true; do
        i=$((i+1))
        # Coin flip arm selection per rollout. 1 = treatment (Equipment
        # block ON), 0 = control (legacy inventory only).
        local flip=$((RANDOM % 2))
        local arm
        if [[ $flip -eq 1 ]]; then arm="ON"; else arm="OFF"; fi
        local ts
        ts=$(date '+%Y%m%d-%H%M%S')
        local jsonl="$OUT/agent${n}-r${i}-${arm}-${ts}.jsonl"
        local log="$OUT/agent${n}-r${i}-${arm}-${ts}.log"
        local t0
        t0=$(date +%s)
        echo "[$(date '+%H:%M:%S')] agent${n} r${i} arm=${arm} starting" | tee -a "$INDEX"
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent${n}" \
        CRAFT_EQUIPMENT_READOUT=$flip \
            .venv/bin/python -m craft.agent 9999 bare \
                --model "$QWEN" \
                --random-spawn-range 20000 \
                --jsonl-out "$jsonl" \
                > "$log" 2>&1 || true
        local dur=$(($(date +%s) - t0))
        local turns
        turns=$(grep -c "=== turn [0-9]*/9999: planning ===" "$log" 2>/dev/null || echo "?")
        local death
        death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
        echo "[$(date '+%H:%M:%S')] agent${n} r${i} arm=${arm} ended turns=${turns} dur=${dur}s${death:+ death=$death}" | tee -a "$INDEX"
        # Brief pause so Wurst's AutoRespawn finishes clicking through the
        # respawn screen before the next process tries to /position.
        sleep 4
    done
}

for n in 0 1 2 3 4; do
    run_agent_forever "$n" &
done

wait
