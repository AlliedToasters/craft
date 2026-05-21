#!/bin/bash
# Async A/B campaign for M2_diamond_goal with dawn-spawn (perpetual-dawn variant).
#
# DESIGN NOTE — async + --start-phase dawn:
#   Each rollout calls `/time set 0` at its own setup step. With 5 agents
#   fanning out, the world clock is continuously being reset back to 0, so
#   the world spends most of its time in early-day cycles and rarely reaches
#   genuine night. This is INTENTIONAL: agents face fewer night-mob waves,
#   higher sample counts of "did this agent reach iron/diamond?", while cave
#   mobs still spawn (light-level-driven, not time-driven) so survival is
#   still a real constraint. See [[environmental-controls]] for the "between
#   Peaceful and natural cycle" framing.
#
#   Cost: `Milestones.check` anchors `spawn_total_ticks` on turn 1; mid-
#   rollout world resets drive `ticks_alive` negative, so M1 essentially
#   never fires. M2 still fires correctly (no ticks_alive dependency). Use
#   the lockstep variant (git history of this file) when M1 fidelity matters.
#
# Arm semantics (CRAFT_MILESTONES — the generic chain selector):
#   ON  : "M1_iron_goal,M2_diamond_goal" — full chain.
#   OFF : "M1_iron_goal"                 — M1 only (M2 stalemate baseline).
#
# Each rollout coin-flips its arm; over a long run both arms get ~50/50.
# Within a long campaign each agent runs ~N/2 of each arm, removing the
# per-agent spawn-quality confound.
#
# Usage:
#   ./scripts/sync_rollouts_ab_m2.sh [N]
# where N is rollouts per agent (default 16). Total = 5*N rollouts.

set -a
. "$(dirname "$0")/../.env"
set +a

N="${1:-16}"
if ! [[ "$N" =~ ^[0-9]+$ ]] || (( N < 1 )); then
    echo "usage: $0 [N]  (positive integer; got '$N')" >&2
    exit 2
fi

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1

ARM_ON_CHAIN="M1_iron_goal,M2_diamond_goal"
ARM_OFF_CHAIN="M1_iron_goal"

TODAY=$(date '+%Y%m%d-%H%M%S')
OUT="results/sync-ab-m2-${TODAY}"
mkdir -p "$OUT"

INDEX="$OUT/_orchestrator.log"
echo "[$(date '+%H:%M:%S')] ASYNC A/B (M2_diamond_goal, --start-phase dawn) starting (out=$OUT, N=$N per agent)" | tee -a "$INDEX"
echo "[$(date '+%H:%M:%S')] commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$INDEX"

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — stopping" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_agent_async() {
    local n=$1
    local port=$((25570 + n))
    for ((i=1; i<=N; i++)); do
        # Coin-flip arm per rollout. Each agent's queue advances as fast as
        # its individual rollouts; agents do NOT wait for slow peers.
        local arm chain
        if (( RANDOM % 2 == 0 )); then
            arm="ON"; chain="$ARM_ON_CHAIN"
        else
            arm="OFF"; chain="$ARM_OFF_CHAIN"
        fi
        local ts
        ts=$(date '+%Y%m%d-%H%M%S')
        local jsonl="$OUT/agent${n}-r${i}-${arm}-${ts}.jsonl"
        local log="$OUT/agent${n}-r${i}-${arm}-${ts}.log"
        local t0
        t0=$(date +%s)
        echo "[$(date '+%H:%M:%S')] agent${n} r${i}/${N} arm=${arm} (chain=${chain}) starting" | tee -a "$INDEX"
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent${n}" \
        CRAFT_MILESTONES="$chain" \
            .venv/bin/python -m craft.agent 9999 bare \
                --model "$QWEN" \
                --start-phase dawn \
                --random-spawn-range 20000 \
                --jsonl-out "$jsonl" \
                > "$log" 2>&1 || true
        local dur=$(($(date +%s) - t0))
        local turns
        turns=$(grep -c "=== turn [0-9]*/9999: planning ===" "$log" 2>/dev/null || echo "?")
        local death
        death=$(grep "YOU DIED" "$log" 2>/dev/null | head -1 | sed 's/.*cause: //;s/).*//')
        local m1_fired m2_fired
        m1_fired=$(grep -c '"milestone_fired": "M1_iron_goal"' "$jsonl" 2>/dev/null || echo 0)
        m2_fired=$(grep -c '"milestone_fired": "M2_diamond_goal"' "$jsonl" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] agent${n} r${i}/${N} arm=${arm} ended turns=${turns} dur=${dur}s m1=${m1_fired} m2=${m2_fired}${death:+ death=$death}" | tee -a "$INDEX"
        # Wurst AutoRespawn pause before next iteration.
        sleep 4
    done
    echo "[$(date '+%H:%M:%S')] agent${n} DONE (${N} rollouts)" | tee -a "$INDEX"
}

for n in 0 1 2 3 4; do
    run_agent_async "$n" &
done

wait
echo "[$(date '+%H:%M:%S')] ALL AGENTS DONE — campaign complete (out=$OUT)" | tee -a "$INDEX"
