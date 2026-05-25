#!/bin/bash
# Overnight Pareto frontier run.
# Round-robins through Q4_K_M, Q8_0, F16 at C=20 (all available agents per wave).
# Runs indefinitely — kill the screen session to stop.
#
# Usage: run inside a screen session so it survives the SSH session.
#   screen -S pareto
#   cd /home/alliedtoasters/projects/mech_interp/craft
#   bash scripts/overnight_pareto.sh
set -a; . "$(dirname "$0")/../.env"; set +a

Q4="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
Q8="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0"
F16="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"

# Round-robin order.  Equal representation across conditions.
QTAGS=(Q4_K_M Q8_0 F16)
QMODELS=("$Q4" "$Q8" "$F16")

TURNS=30
STARTPHASE=dawn
SPAWN_RANGE=20000
AGENTS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
OUT="results/pareto_overnight_$(date +%Y%m%d)_C${#AGENTS[@]}_cont"
mkdir -p "$OUT"

wave=0
qi=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait until at least one agent is reachable; give fresh clients time to boot.
log "checking agent homunculi ..."
for attempt in $(seq 1 30); do
  alive=0
  for n in "${AGENTS[@]}"; do
    curl -s --max-time 2 "http://127.0.0.1:$((25570+n))/stats" >/dev/null 2>&1 && alive=$((alive+1))
  done
  log "  $alive/${#AGENTS[@]} agents reachable (attempt $attempt)"
  [ "$alive" -ge 3 ] && break
  sleep 10
done

log "starting round-robin. output dir: $OUT"

while true; do
  wave=$((wave + 1))
  idx=$((qi % ${#QTAGS[@]}))
  qi=$((qi + 1))
  QTAG="${QTAGS[$idx]}"
  QMODEL="${QMODELS[$idx]}"

  log "===== wave $wave  quant=$QTAG ====="

  export CRAFT_SCOUT_FANOUT_MODEL="$QMODEL"
  export CRAFT_SCOUT_UNIFY_MODEL="$QMODEL"
  export CRAFT_LOOK_AROUND_MAX_RADIUS=1
  export CRAFT_MINE_FORCE_XRAY=1

  pids=()
  launched=()
  for n in "${AGENTS[@]}"; do
    port=$((25570 + n))
    if ! curl -s --max-time 2 "http://127.0.0.1:$port/stats" >/dev/null 2>&1; then
      log "  agent$n: homunculus DOWN — skipping"
      continue
    fi
    HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent$n" \
      .venv/bin/python -m craft.agent "$TURNS" minimal \
        --model "$QMODEL" \
        --start-phase "$STARTPHASE" \
        --random-spawn-range "$SPAWN_RANGE" \
        --jsonl-out "$OUT/${QTAG}_w${wave}_a${n}.jsonl" \
        > "$OUT/${QTAG}_w${wave}_a${n}.log" 2>&1 &
    pids+=("$!")
    launched+=("$n")
  done

  log "  launched ${#pids[@]} agents: ${launched[*]}"
  for pid in "${pids[@]}"; do
    wait "$pid" || true   # non-zero exit (death) is normal; don't abort the loop
  done

  # Quick count of completed rollouts per quant so far.
  for qt in "${QTAGS[@]}"; do
    n_done=$(ls "$OUT"/${qt}_w*_a*.jsonl 2>/dev/null | wc -l)
    log "  tally $qt: $n_done rollouts"
  done
  log "  wave $wave done"
done
