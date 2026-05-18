#!/bin/bash
# N=25 baseline rollouts: pure-qwen, r=1 cap, TTL cache.
# 5 waves × 5 agents. Each wave waits for all 5 to finish before the next.
# Total wall time ~25-30 minutes depending on rollout pacing.

set -e
set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
OUT=results/bigN-pure-qwen-r1-cache-20260517
mkdir -p "$OUT"

export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1

NWAVES="${NWAVES:-5}"

for wave in $(seq 1 "$NWAVES"); do
    echo "===== wave $wave/$NWAVES starting at $(date '+%H:%M:%S') ====="
    pids=()
    for n in 0 1 2 3 4; do
        port=$((25570 + n))
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME=agent$n \
            .venv/bin/python -m craft.agent 30 minimal \
                --model "$QWEN" \
                --start-phase dawn \
                --random-spawn-range 20000 \
                --jsonl-out "$OUT/wave${wave}-agent${n}.jsonl" \
                > "$OUT/wave${wave}-agent${n}.log" 2>&1 &
        pids+=("$!")
    done
    echo "wave $wave: launched 5 rollouts (pids=${pids[*]})"
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  pid $pid exited nonzero"
    done
    echo "wave $wave done at $(date '+%H:%M:%S')"
done

echo "===== all waves complete ====="
