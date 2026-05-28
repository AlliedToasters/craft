#!/bin/bash
# A/B: does the mine_stone "descend ~5 blocks" nudge improve stone-acquisition
# efficacy for qwen-4B? Concurrent fleet fan-out — two baskets run in parallel.
#
# Scenario: --starting-loadout wood_to_stone (full wood tool set + logs +
# crafting table + sticks). The agent should infer that mining stone for a
# stone pickaxe is the next progression step (goal=diamond points the tech
# tree straight at it). On a surface spawn, fair-mode mine_stone digs locally
# at the agent's depth and finds no stone → FAILED. The nudge appends a
# "descend ~5 blocks and retry" recommendation to that failure message.
#
#   ON  basket = CRAFT_STONE_STAIRCASE=1  (descending-staircase auto-descend)
#   OFF basket = CRAFT_STONE_STAIRCASE=0  (legacy flat horizontal tunnel)
#
# (Earlier this A/B'd CRAFT_MINE_STONE_DESCEND_NUDGE, an LLM failure-message
# nudge that proved neutral; the staircase moves the recovery into the substrate.)
#
# The toggle is PYTHON-side (craft/mine.py). The homunculus entity-resolution
# refactor did NOT touch mine_stone or descend, so a mixed old/new-jar fleet
# behaves identically here — no need to redeploy/relaunch every agent first.
#
# Concurrency: each agent runs one basket. Default = 20 agents, 10 per basket
# (n=10 each), all in parallel → a signal in roughly one rollout's wall time.
# Spawns are random per agent (biome lottery — the confound; n=10/arm averages
# it out). Dead agents (no /stats) are skipped with a warning.
#
# Arms are INTERLEAVED across the agent range (ON=even, OFF=odd) on purpose: a
# block split (ON=0-9 / OFF=10-19) once lined the condition up with per-instance
# state (fresh vs jar-corrupted clients), manufacturing a false signal. Never
# let arm assignment correlate with instance identity. See
# memory: jar-deploy-over-running-corrupts.
#
# Efficacy metric (printed at the end, per basket):
#   - attempted%          : agents that called mine_stone at all
#   - stone_acquired%     : agents where mine_stone returned a positive acquire
#   - descend_after_fail% : agents where a descend followed a mine_stone FAILED
#                           (the exact behavior the nudge is trying to induce)
#
# Usage:
#   ./scripts/mine_stone_nudge_ab.sh                              # 0-9 ON, 10-19 OFF
#   CRAFT_AB_ON_AGENTS="0 1 2 3 4" CRAFT_AB_OFF_AGENTS="5 6 7 8 9" \
#       ./scripts/mine_stone_nudge_ab.sh                          # n=5/arm on 10 agents
#   CRAFT_AB_ITERS=2 ...                                          # 2 rollouts/agent (doubles n)
#   CRAFT_MODEL=claude-haiku-4-5 ...
set -a
. "$(dirname "$0")/../.env"
set +a

QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
MODEL="${CRAFT_MODEL:-$QWEN}"
# Scout models stay on qwen — substrate utility, not under test.
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN"
export CRAFT_SCOUT_UNIFY_MODEL="$QWEN"
export CRAFT_LOOK_AROUND_MAX_RADIUS=1

ON_AGENTS="${CRAFT_AB_ON_AGENTS:-0 2 4 6 8 10 12 14 16 18}"
OFF_AGENTS="${CRAFT_AB_OFF_AGENTS:-1 3 5 7 9 11 13 15 17 19}"
ITERS="${CRAFT_AB_ITERS:-1}"
TURNS="${CRAFT_AB_TURNS:-20}"
DIFFICULTY="${CRAFT_AB_DIFFICULTY:-peaceful}"

TS=$(date '+%Y%m%d-%H%M%S')
ROOT="results/mine-stone-nudge-ab-${TS}"
mkdir -p "$ROOT/on" "$ROOT/off"
INDEX="$ROOT/_orchestrator.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$INDEX"; }

log "MINE_STONE NUDGE A/B (concurrent) out=$ROOT"
log "commit=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "ON=[$ON_AGENTS] OFF=[$OFF_AGENTS] iters/agent=$ITERS turns=$TURNS difficulty=$DIFFICULTY model=$MODEL loadout=wood_to_stone goal=diamond"

# Liveness probe: only fan out to agents whose homunculus answers.
alive() { curl -s -m 2 "http://127.0.0.1:$((25570 + $1))/stats" >/dev/null 2>&1; }
filter_live() {
    local kept=""
    for n in $1; do
        if alive "$n"; then kept="$kept $n"; else log "WARN agent$n (port $((25570+n))) DOWN — skipping"; fi
    done
    echo "$kept" | xargs
}
ON_AGENTS=$(filter_live "$ON_AGENTS")
OFF_AGENTS=$(filter_live "$OFF_AGENTS")
log "live ON=[$ON_AGENTS] OFF=[$OFF_AGENTS]"
if [[ -z "$ON_AGENTS" || -z "$OFF_AGENTS" ]]; then
    log "ABORT: a basket has no live agents. Launch agents (./launch_agent.sh N) and retry."
    exit 1
fi

trap 'echo "[$(date +%H:%M:%S)] SIGTERM — killing children" | tee -a "$INDEX"; kill $(jobs -p) 2>/dev/null; wait; exit 0' INT TERM

run_agent() {
    local n=$1 arm=$2 nudge=$3
    local port=$((25570 + n))
    for i in $(seq 1 "$ITERS"); do
        local ts; ts=$(date '+%Y%m%d-%H%M%S')
        local jsonl="$ROOT/$arm/agent${n}-iter${i}-${ts}.jsonl"
        local log_f="${jsonl%.jsonl}.log"
        log "agent$n arm=$arm nudge=$nudge iter=$i starting"
        HOMUNCULUS_PORT=$port MC_PLAYER_NAME="agent${n}" \
            CRAFT_STONE_STAIRCASE="$nudge" \
            .venv/bin/python -m craft.agent "$TURNS" diamond \
                --model "$MODEL" \
                --start-phase noon \
                --random-spawn-range 20000 \
                --starting-loadout wood_to_stone \
                --difficulty "$DIFFICULTY" \
                --jsonl-out "$jsonl" \
                > "$log_f" 2>&1 || true
        log "agent$n arm=$arm iter=$i done"
    done
}

for n in $ON_AGENTS;  do run_agent "$n" on  1 & done
for n in $OFF_AGENTS; do run_agent "$n" off 0 & done
wait
log "ALL ROLLOUTS DONE — analyzing"
echo ""

ROOT="$ROOT" .venv/bin/python - <<'PY' | tee -a "$INDEX"
import glob, json, os

root = os.environ["ROOT"]

def analyze_iter(path):
    attempted = False; fails = 0; acquired = False
    descend_after_fail = False; pending_fail = False
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("_type") == "header" or "tool" not in r:
            continue
        tool = r.get("tool"); out = str(r.get("outcome") or "")
        if tool == "mine_stone":
            attempted = True
            if out.startswith("FAILED"):
                fails += 1; pending_fail = True
            elif "acquired 0" in out:
                pass
            elif "acquired" in out:
                acquired = True; pending_fail = False
        elif tool == "descend":
            if pending_fail: descend_after_fail = True
            pending_fail = False
    return attempted, fails, acquired, descend_after_fail

def arm_stats(arm):
    paths = sorted(glob.glob(os.path.join(root, arm, "*.jsonl")))
    n = len(paths); att = acq = daf = tf = 0
    for p in paths:
        a, f, ac, d = analyze_iter(p)
        att += a; acq += ac; daf += d; tf += f
    return n, att, acq, daf, tf

def pct(x, n): return f"{100*x/n:5.1f}%" if n else "  n/a"

print("=== MINE_STONE DESCEND-NUDGE A/B ===")
print(f"{'arm':<5} {'n':>3} {'attempted':>10} {'stone_acq':>10} {'descend_after_fail':>20} {'mean_fails':>11}")
for arm in ("on", "off"):
    n, att, acq, daf, tf = arm_stats(arm)
    mf = f"{tf/n:6.2f}" if n else "  n/a"
    print(f"{arm:<5} {n:>3} {pct(att,n):>10} {pct(acq,n):>10} {pct(daf,n):>20} {mf:>11}")
print()
print("If the nudge helps: ON shows higher stone_acq% and descend_after_fail%")
print("than OFF. attempted% confirms iters were informative (agent tried to mine).")
PY

log "DONE out=$ROOT"
