#!/bin/bash
# Run waves of the N=10 pure-qwen diamond driver with screen recording on,
# repeating until at least one rollout in a wave reaches a diamond. All tapes
# kept (CRAFT_RECORD_KEEP=all) so the user can review failure-mode footage.
#
# Each wave = one bigN20_easy_qwen.sh invocation with N_AGENTS=10, GOAL=diamond.
# Stops as soon as a wave's JSONLs show minecraft:diamond (raw or diamond tool).
#
# Usage: ./scripts/diamond_wave_loop.sh [max_waves]

cd "$(dirname "$0")/.."

MAX_WAVES="${1:-20}"
export CRAFT_RECORD_VIDEO=1
export CRAFT_RECORD_KEEP=all
export CRAFT_RECORD_FPS="${CRAFT_RECORD_FPS:-8}"
export N_AGENTS=10
export START_AGENT=0
export CRAFT_GOAL=diamond

LOOPLOG="results/diamond_wave_loop-$(date '+%Y%m%d-%H%M%S').log"
echo "[$(date '+%H:%M:%S')] diamond wave loop starting (max_waves=$MAX_WAVES, recording=on keep=all)" | tee -a "$LOOPLOG"

wave=1
while [ "$wave" -le "$MAX_WAVES" ]; do
    echo "[$(date '+%H:%M:%S')] === WAVE $wave/$MAX_WAVES ===" | tee -a "$LOOPLOG"

    # Find the result dir this wave creates (newest bigN20-easy-qwen-* after run).
    before=$(ls -d results/bigN20-easy-qwen-* 2>/dev/null | sort | tail -1)
    ./scripts/bigN20_easy_qwen.sh >> "$LOOPLOG" 2>&1
    OUT=$(ls -d results/bigN20-easy-qwen-* 2>/dev/null | sort | tail -1)

    # Detect any diamond across this wave's rollouts.
    dia=$(grep -lE '"minecraft:diamond"|"minecraft:diamond_(pickaxe|sword|shovel|axe)"' "$OUT"/agent*.jsonl 2>/dev/null)
    ndia=$(echo -n "$dia" | grep -c . )
    deaths=$(grep -h '"died": true\|YOU DIED' "$OUT"/agent*.jsonl "$OUT"/agent*.log 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] wave $wave done: out=$OUT diamond_rollouts=$ndia" | tee -a "$LOOPLOG"

    if [ "$ndia" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] *** DIAMOND HIT in wave $wave ***" | tee -a "$LOOPLOG"
        echo "$dia" | tee -a "$LOOPLOG"
        echo "[$(date '+%H:%M:%S')] STOPPING loop. Result dir: $OUT" | tee -a "$LOOPLOG"
        exit 0
    fi
    wave=$((wave + 1))
done

echo "[$(date '+%H:%M:%S')] max_waves reached without a diamond" | tee -a "$LOOPLOG"
exit 1
