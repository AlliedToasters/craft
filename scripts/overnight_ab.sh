#!/bin/bash
# Overnight Qwen-vs-Haiku diamond A/B accumulator.
#
# Loops wave-pairs back-to-back to build sample size while the author sleeps:
#   - Qwen planner on agents 0-9   (ports 25570-79)
#   - Haiku planner on agents 10-19 (ports 25580-89)
# concurrent, identical substrate (Fix B jar + Fix C tier-gate, scout=qwen for
# both). Only the planner brain differs. peaceful / diamond / dawn / range 20000.
#
# turns=100 by default (the author asked to bump the cap after the 45-turn pair).
#
# Autonomy guards between waves:
#   - fleet health: if status != 20/20 in-world, ./fleet.sh cycle 20.
#   - Ollama ping: warn (can't auto-restart) — qwen arm needs it.
#   - RAM guard: swap=0 is a hard cliff (FLEET.md); pause if free < 10G.
#
# Each finished wave-pair appends a diamond_tally summary to $SUMMARY so the
# morning read is one file. Result dirs are the usual per-wave timestamped
# results/bigN20-easy-{qwen,haiku}-<ts>/.
#
# Usage:
#   nohup ./scripts/overnight_ab.sh > results/overnight_ab.driver.log 2>&1 &
# Env:
#   OVERNIGHT_TURNS   (default 100)
#   OVERNIGHT_HOURS   (default 9)  — stop launching NEW waves after this many h
#   WAIT_FOR_QWEN / WAIT_FOR_HAIKU — orchestrator logs of an in-flight pair to
#                                    drain before the turns=100 loop starts.

set -a
. "$(dirname "$0")/../.env"
set +a
cd "$(dirname "$0")/.." || exit 1

TURNS="${OVERNIGHT_TURNS:-100}"
HOURS="${OVERNIGHT_HOURS:-9}"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
DIFF="${OVERNIGHT_DIFFICULTY:-peaceful}"
SUMMARY="results/overnight_ab.summary.log"
PY=.venv/bin/python

export CRAFT_TURNS=$TURNS CRAFT_GOAL=diamond CRAFT_PHASE=dawn \
       CRAFT_DIFFICULTY="$DIFF" CRAFT_SPAWN_RANGE=20000

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

log "OVERNIGHT A/B start: turns=$TURNS hours=$HOURS difficulty=$DIFF deadline=$(date -d "@$DEADLINE" '+%H:%M') commit=$(git rev-parse --short HEAD 2>/dev/null)"

# 1. Drain any in-flight pair (the turns=45 waves) before bumping to 100.
for w in "$WAIT_FOR_QWEN" "$WAIT_FOR_HAIKU"; do
    [ -n "$w" ] || continue
    log "waiting for in-flight wave to finish: $w"
    while ! grep -q "ALL ROLLOUTS DONE" "$w" 2>/dev/null; do sleep 30; done
    log "  drained: $w"
done

health_check() {
    # cycle the fleet if it isn't a clean 20/20 in-world.
    if ! ./fleet.sh status 20 >/dev/null 2>&1; then
        log "fleet degraded (status != 20/20) — cycling"
        ./fleet.sh cycle 20 >>"$SUMMARY" 2>&1 || log "  cycle returned nonzero"
    fi
    # Ollama reachability (qwen arm). Warn only — can't auto-restart it here.
    # Native /api/tags lives at the root (OLLAMA_BASE_URL carries a /v1 suffix).
    local ollama_ping="${OLLAMA_PING:-http://localhost:11434/api/tags}"
    curl -sf --max-time 5 "$ollama_ping" >/dev/null 2>&1 \
        || log "WARN: Ollama unreachable at $ollama_ping — qwen arm will error"
}

ram_guard() {
    local free
    free=$(free -g | awk '/Mem:/{print $7}')
    while [ "${free:-99}" -lt 10 ]; do
        log "LOW RAM (${free}G free) — pausing 120s (swap=0 hard cliff)"
        sleep 120
        free=$(free -g | awk '/Mem:/{print $7}')
    done
}

tally() {
    local dir=$1 brand=$2
    local line
    line=$($PY scripts/diamond_tally.py "$dir" 2>/dev/null \
        | grep -iE "rollouts=|DIAMONDS|peak pickaxe|reached iron" \
        | tr '\n' ' ')
    log "  [$brand] $line"
}

wave=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    wave=$((wave + 1))
    health_check
    ram_guard
    log "wave $wave launching (turns=$TURNS)"

    if [ "${OVERNIGHT_QWEN_ONLY:-0}" = "1" ]; then
        # Full-local: qwen only, on $OVERNIGHT_QWEN_N agents (no haiku/API cost).
        START_AGENT=0 N_AGENTS="${OVERNIGHT_QWEN_N:-10}" ./scripts/bigN20_easy_qwen.sh >/tmp/ov_qwen.out 2>&1 &
        wait "$!"
        qdir=$(ls -dt results/bigN20-easy-qwen-* 2>/dev/null | head -1)
        log "wave $wave done — qwen=$qdir (qwen-only)"
        tally "$qdir" QWEN
    else
        START_AGENT=0  N_AGENTS=10 ./scripts/bigN20_easy_qwen.sh  >/tmp/ov_qwen.out  2>&1 &
        qp=$!
        START_AGENT=10 N_AGENTS=10 ./scripts/bigN20_easy_haiku.sh >/tmp/ov_haiku.out 2>&1 &
        hp=$!
        wait "$qp" "$hp"

        qdir=$(ls -dt results/bigN20-easy-qwen-*  2>/dev/null | head -1)
        hdir=$(ls -dt results/bigN20-easy-haiku-* 2>/dev/null | head -1)
        log "wave $wave done — qwen=$qdir haiku=$hdir"
        tally "$qdir" QWEN
        tally "$hdir" HAIKU
    fi
done

log "OVERNIGHT A/B stop: deadline reached after $wave wave(s)"
