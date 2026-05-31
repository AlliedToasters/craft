#!/bin/bash
# §21.2 VISUAL CAPTURE — concurrent across the N-agent fleet.
#
# Fans nav_distill_capture out over agents 0..N-1 (one process per homunculus
# port), each writing its own out-root, then merges all rollout dirs into one
# capture/ for the §21.2 analysis loader. This is the concurrent form of the
# §21.0/§21.2 capture — the substrate-as-instrument thesis is that we ALWAYS run
# concurrent, so representative scheduling jitter is a feature, not noise.
#
# Clean-frame switches are applied per-agent by the driver itself (baritone/render
# off + /hud all:false + Fullbright pin); chatVisibility:2 is a per-instance
# options.txt setting that must already be set (see FLEET stand-up). Difficulty is
# set peaceful ONCE here so the 20 drivers don't serialize on the relay fighting
# over it (--no-peaceful on each).
#
# Prereq: fleet up + in-world (./fleet.sh cycle N), relay at 4747, all target
# agents' options.txt at chatVisibility:2.
#
# Usage:  ./scripts/sprint21_visual_capture.sh            # N=20, 2 rollouts each
#         N_AGENTS=10 ROLLOUTS_PER=3 ./scripts/sprint21_visual_capture.sh
set -a
. "$(dirname "$0")/../.env"
set +a
cd "$(dirname "$0")/.."

N_AGENTS="${N_AGENTS:-20}"
ROLLOUTS_PER="${ROLLOUTS_PER:-2}"
LEGS="${LEGS:-3}"
LEG_DIST="${LEG_DIST:-90}"
LEG_TIMEOUT="${LEG_TIMEOUT:-60}"
SPAWN_RANGE="${SPAWN_RANGE:-20000}"
FRAME_INTERVAL="${FRAME_INTERVAL:-0.5}"
# Per-agent seed offset: random_spawn is seeded, so every agent sharing one seed
# teleports to the SAME spot (all in one biome) — fatal for terrain variety. Give
# each agent a distinct seed so the fleet spreads across biomes.
SEED_BASE="${SEED_BASE:-21}"
RELAY="${MC_SERVER_CMD_BASE:-http://127.0.0.1:4747}"

OUT="results/sprint21_visual"
rm -rf "$OUT"; mkdir -p "$OUT"
INDEX="$OUT/_orchestrator.log"

{
  echo "[$(date '+%H:%M:%S')] §21.2 VISUAL CAPTURE  N=$N_AGENTS rollouts_per=$ROLLOUTS_PER"
  echo "[$(date '+%H:%M:%S')] commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "[$(date '+%H:%M:%S')] legs=$LEGS leg_dist=$LEG_DIST spawn_range=$SPAWN_RANGE frame_interval=$FRAME_INTERVAL"
} | tee -a "$INDEX"

# Difficulty peaceful ONCE (drivers run --no-peaceful so they don't each toggle it).
curl -s --max-time 5 -X POST "$RELAY/cmd" -d '{"cmd":"difficulty peaceful"}' >/dev/null 2>&1 \
  && echo "[$(date '+%H:%M:%S')] difficulty -> peaceful (global)" | tee -a "$INDEX"

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — stopping drivers" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_one() {
  local n=$1 port=$((25570 + n))
  local log="$OUT/agent${n}.log"
  local t0; t0=$(date +%s)
  echo "[$(date '+%H:%M:%S')] agent${n} (port $port) capture starting" | tee -a "$INDEX"
  .venv/bin/python -m experiments.codec_loop.nav_distill_capture \
      --port "$port" --player "agent${n}" \
      --rollouts "$ROLLOUTS_PER" --legs "$LEGS" --leg-dist "$LEG_DIST" \
      --leg-timeout "$LEG_TIMEOUT" --spawn-range "$SPAWN_RANGE" \
      --frames --frame-interval "$FRAME_INTERVAL" --no-peaceful \
      --seed "$((SEED_BASE + n * 1000))" \
      --out-root "$OUT/agent${n}" \
      > "$log" 2>&1 || true
  local dur=$(($(date +%s) - t0))
  local summ="$OUT/agent${n}/capture/capture_summary.json"
  local ro tk wp
  if [ -f "$summ" ]; then
    ro=$(grep -o '"rollouts":[^,]*' "$summ" | head -1)
    tk=$(grep -o '"total_ticks":[0-9]*' "$summ" | head -1)
    wp=$(grep -o '"total_ticks_with_path":[0-9]*' "$summ" | head -1)
  fi
  echo "[$(date '+%H:%M:%S')] agent${n} done dur=${dur}s ${tk:-no-summary} ${wp:-}" | tee -a "$INDEX"
}

for n in $(seq 0 $((N_AGENTS - 1))); do run_one "$n" & done
wait

# restore difficulty
curl -s --max-time 5 -X POST "$RELAY/cmd" -d '{"cmd":"difficulty easy"}' >/dev/null 2>&1 \
  && echo "[$(date '+%H:%M:%S')] difficulty -> easy (restored)" | tee -a "$INDEX"

# Merge all per-agent rollout dirs into one capture/ with globally-unique names,
# so the §21 loader (globs capture/rollout-*) reads the whole fleet's output.
MERGED="$OUT/capture"; mkdir -p "$MERGED"
nmerged=0
for n in $(seq 0 $((N_AGENTS - 1))); do
  for rd in "$OUT/agent${n}/capture"/rollout-*; do
    [ -d "$rd" ] || continue
    i=$(basename "$rd" | sed 's/rollout-//')
    mv "$rd" "$MERGED/rollout-a${n}r${i}"
    nmerged=$((nmerged + 1))
  done
done

TOTAL_ROWS=$(find "$MERGED" -name 'sidecar.jsonl.gz' | wc -l)
TOTAL_FRAMES=$(find "$MERGED" -name 'f-*.png' | wc -l)
{
  echo "[$(date '+%H:%M:%S')] MERGED $nmerged rollout dirs -> $MERGED"
  echo "[$(date '+%H:%M:%S')] sidecars=$TOTAL_ROWS frames=$TOTAL_FRAMES"
  echo "[$(date '+%H:%M:%S')] §21.2 CAPTURE DONE"
} | tee -a "$INDEX"