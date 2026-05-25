#!/bin/bash
# One-shot C=20 F16 daily-driver wave to validate the spectator column-spawn
# rewrite (craft/spawn.py, 2026-05-25) at fleet scale. Single wave across
# agents 0..19, then exits. Watch for: zero spawn-mechanism deaths
# (suffocation/encasement at y=100), no spawns stuck in water/bad-biome, and
# spawns landing on real surfaces incl. high-terrain biomes.
set -a
. "$(dirname "$0")/../.env"
set +a

F16="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
TURNS=30
AGENTS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
OUT="results/spawn_validate_$(date +%Y%m%d_%H%M)_C${#AGENTS[@]}_F16"
mkdir -p "$OUT"

export CRAFT_SCOUT_FANOUT_MODEL="$F16"
export CRAFT_SCOUT_UNIFY_MODEL="$F16"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1
export CRAFT_MINE_FORCE_XRAY=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "start -> $OUT"
pids=(); launched=()
for n in "${AGENTS[@]}"; do
  port=$((25570 + n))
  if ! curl -s --max-time 2 "http://127.0.0.1:$port/stats" >/dev/null 2>&1; then
    log "  agent$n: homunculus DOWN — skipping"; continue
  fi
  HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent$n" \
    .venv/bin/python -m craft.agent "$TURNS" minimal \
      --model "$F16" \
      --start-phase dawn \
      --random-spawn-range 20000 \
      --jsonl-out "$OUT/agent${n}.jsonl" \
      > "$OUT/agent${n}.log" 2>&1 &
  pids+=("$!"); launched+=("$n")
done
log "launched ${#pids[@]} agents: ${launched[*]}"
for pid in "${pids[@]}"; do wait "$pid" || true; done
log "WAVE DONE ($OUT)"
