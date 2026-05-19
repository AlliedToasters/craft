#!/bin/bash
# M1 milestone test waves. Same daily-driver config as run_bigN_pureqwen.sh,
# but rollouts are 50 turns (vs 30) to give M1 post-fire room to breathe —
# replay says median fire turn is T17, mean post-fire survival is 93 turns.
# 50 turns gives ~30 turns of post-M1 behavior to observe per rollout.
#
# NWAVES=1 by default for a quick smoke; bump for bigger N.

set -e
set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
TODAY=$(date '+%Y%m%d')
OUT="results/m1-test-${TODAY}"
mkdir -p "$OUT"

export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1

NWAVES="${NWAVES:-1}"
TURNS="${TURNS:-50}"

echo "===== M1 test: $NWAVES wave(s) × 5 agents × $TURNS turns, dawn-spawn ====="

for wave in $(seq 1 "$NWAVES"); do
    echo "===== wave $wave/$NWAVES starting at $(date '+%H:%M:%S') ====="
    pids=()
    for n in 0 1 2 3 4; do
        port=$((25570 + n))
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME=agent$n \
            .venv/bin/python -m craft.agent "$TURNS" minimal \
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

echo "===== all waves complete; results in $OUT ====="
