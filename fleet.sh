#!/usr/bin/env bash
# fleet.sh — canonical N-agent headless fleet orchestrator.
#
# The full "stand up N concurrent agents" recipe, captured as subcommands so
# we stop rediscovering it. See FLEET.md for the narrative + gotchas.
#
#   ./fleet.sh preflight        # check the externals (relay 4747 + MC server screen)
#   ./fleet.sh down             # kill all clients + reap Xvfb/watchers + clean stale sockets
#   ./fleet.sh deploy           # rebuild homunculus + distribute jar to every instance
#   ./fleet.sh up   [N]         # launch agents 0..N-1 headlessly, staggered
#   ./fleet.sh status [N]       # poll every homunculus port; report in-world count
#   ./fleet.sh fix  N           # repair one stuck agent (wipe its root + socket, relaunch)
#   ./fleet.sh cycle [N]        # down -> deploy -> up -> status  (the whole recipe)
#
# N defaults to $FLEET_N (env) or 20. Ports are 25570+n; Xvfb display :200+n.
#
# WHY a wrapper at all: launch_agent.sh launches exactly ONE agent and BLOCKS
# (it waits on the Prism PID so its EXIT trap can reap the per-agent Xvfb +
# dialog watcher). To stand up a fleet you must background one launch_agent.sh
# per agent. That loop + the teardown/socket-hygiene around it was the
# undocumented part; this is it.
set -uo pipefail
cd "$(dirname "$0")"

# Shared shell+Python config (PRISMLAUNCHER_BIN, CRAFT_MC_HOST, ...).
[ -f ./.env ] && { set -a; . ./.env; set +a; }

N_DEFAULT="${FLEET_N:-20}"
PORT_BASE=25570
DISPLAY_BASE=200
ROOTBASE="${PRISM_AGENT_ROOTS:-${XDG_DATA_HOME:-$HOME/.local/share}/pl-agents}"
RELAY="${MC_SERVER_CMD_BASE:-http://127.0.0.1:4747}"
BOOTDIR="${FLEET_BOOTDIR:-/tmp/fleet-boot}"

_port() { echo $((PORT_BASE + $1)); }

# --- in-world probe: a homunculus port answers /stats with a live HP only once
# the player has actually JOINED the world. A bound-but-not-joined port (mod
# HTTP up, client still on a menu/loading) reports no health — that gap is the
# usual "looks up but isn't" failure, so we key on HP, not on the port binding.
_hp() {
  curl -sf --max-time 2 "http://127.0.0.1:$(_port "$1")/stats" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); h=d.get('health'); print(h if h not in (None,'') else '')" 2>/dev/null
}

cmd_preflight() {
  local ok=0
  echo "== preflight =="
  # 1. MC server console relay (port 4747). world.py routes set_difficulty /
  #    set_time(--start-phase) / set_gamemode / give / clear through it. If it's
  #    down those SILENTLY no-op and rollouts run at the current world state.
  if curl -sf --max-time 3 -XPOST "$RELAY/cmd" -d '{"cmd":"time query daytime"}' >/dev/null 2>&1; then
    echo "  [ok]   relay up ($RELAY)"
  else
    echo "  [FAIL] relay DOWN ($RELAY) — start it:"
    echo "         cd ../gemmacraft/server_1.21.4 && nohup python3 mc_api.py &"
    ok=1
  fi
  # 2. The relay injects into a GNU screen session named 'server'; the MC server
  #    (Purpur/Paper) must be running inside it.
  if screen -ls 2>/dev/null | grep -q '\.server'; then
    echo "  [ok]   MC server screen session 'server' present"
  else
    echo "  [WARN] no screen session matching '.server' — relay has nothing to drive"
  fi
  # 3. PrismLauncher binary resolvable.
  if [ -x "${PRISMLAUNCHER_BIN:-}" ] || command -v "${PRISMLAUNCHER_BIN:-prismlauncher}" >/dev/null 2>&1; then
    echo "  [ok]   launcher: ${PRISMLAUNCHER_BIN:-prismlauncher}"
  else
    echo "  [FAIL] PRISMLAUNCHER_BIN not found: ${PRISMLAUNCHER_BIN:-prismlauncher}"; ok=1
  fi
  # 4. No swap = hard RAM cliff (no graceful degradation). Just a heads-up.
  local sw; sw=$(free -g | awk '/Swap:/{print $2}')
  [ "${sw:-0}" -eq 0 ] && echo "  [note] swap=0G — RAM exhaustion is a hard cliff, watch 'free -g' at high N"
  return $ok
}

cmd_down() {
  echo "== down =="
  # Kill the launch_agent.sh wrappers first so their EXIT traps reap the
  # per-agent Xvfb + dialog watcher children. Bracket-trick the pattern so
  # pkill never matches its own argv (the pkill-self-match trap).
  pkill -f '[l]aunch_agent.sh' 2>/dev/null || true
  sleep 2
  # Then the game clients: graceful, then hard.
  pkill -f '[o]rg.prismlauncher.EntryPoint' 2>/dev/null || true
  sleep 3
  pkill -9 -f '[o]rg.prismlauncher.EntryPoint' 2>/dev/null || true
  # Reap any per-agent Xvfb (:200..:231) + dialog-watcher loops that orphaned.
  pkill -9 -f '[X]vfb :2[0-3][0-9]' 2>/dev/null || true
  sleep 2
  # Stale single-instance sockets: kill -9 leaks /tmp/pl<hash> (one per app
  # root). A leftover one makes the NEXT launch print "Unable to redirect
  # command to already running instance" and exit without booting. Safe to
  # remove only once NO launcher is running — guard on that.
  if ! pgrep -f '[o]rg.prismlauncher.EntryPoint' >/dev/null 2>&1; then
    local nsock; nsock=$(ls /tmp/pl* 2>/dev/null | wc -l)
    rm -f /tmp/pl* 2>/dev/null || true
    echo "  cleared $nsock stale /tmp/pl* single-instance socket(s)"
  else
    echo "  [WARN] prism still running — skipped /tmp/pl* cleanup (would orphan a live socket)"
  fi
  # Report leftover bound homunculus ports (should be none).
  local left; left=$(ss -ltn 2>/dev/null | grep -cE ":(2557[0-9]|2558[0-9]|2559[0-9]) ")
  echo "  homunculus ports still bound: $left  (expect 0)"
}

cmd_deploy() {
  echo "== deploy (rebuild + distribute homunculus) =="
  # MUST run with no client live: cp'ing the jar over a RUNNING instance
  # corrupts its lazy class loading (silent transport_errors). 'down' first.
  if pgrep -f '[o]rg.prismlauncher.EntryPoint' >/dev/null 2>&1; then
    echo "  [FAIL] clients are running — run './fleet.sh down' before deploy"; return 1
  fi
  ( cd ../homunculus && ./move_to_instance.sh ) || { echo "  [FAIL] move_to_instance.sh"; return 1; }
}

cmd_up() {
  local n="${1:-$N_DEFAULT}"
  echo "== up (launch agents 0..$((n-1))) =="
  mkdir -p "$BOOTDIR"
  for i in $(seq 0 $((n-1))); do
    nohup ./launch_agent.sh "$i" > "$BOOTDIR/agent${i}.boot.log" 2>&1 &
    disown
    echo "  launched agent$i (display :$((DISPLAY_BASE+i)), port $(_port "$i"))"
    sleep 4   # stagger: avoid a thundering-herd of root-builds / GL inits
  done
  echo "  all $n fired — clients join ~20-40s each. Poll: ./fleet.sh status $n"
}

cmd_status() {
  local n="${1:-$N_DEFAULT}" up=0 down=""
  for i in $(seq 0 $((n-1))); do
    local hp; hp=$(_hp "$i")
    if [ -n "$hp" ]; then up=$((up+1)); else down="$down $i"; fi
  done
  echo "in-world: $up/$n${down:+   not-in-world:$down}"
  echo "prism=$(pgrep -fc '[o]rg.prismlauncher.EntryPoint')  xvfb=$(pgrep -fc '[X]vfb :2')  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"
  [ -z "$down" ] && return 0 || return 1
}

# Repair ONE stuck agent. Two known failure modes, both fixed by a clean
# rebuild: (a) a stale per-agent root whose accounts.json still has many
# accounts + activeAccount=None -> launcher loops forever in background account
# refresh and never spawns the JVM; (b) a leaked /tmp/pl socket -> "already
# running instance". Wiping the root regenerates a single-account accounts.json;
# 'down' already cleared sockets, but a single-agent relaunch may still race the
# kill's socket teardown, so we retry once.
cmd_fix() {
  local i="${1:?usage: fleet.sh fix N}"
  echo "== fix agent$i =="
  pkill -f "[l]aunch_agent.sh $i\b" 2>/dev/null || true
  pkill -9 -f "pl-agents/agent$i\b" 2>/dev/null || true
  pkill -9 -f "[X]vfb :$((DISPLAY_BASE+i))\b" 2>/dev/null || true
  sleep 2
  echo "  wiping stale root: $ROOTBASE/agent$i"
  rm -rf "${ROOTBASE:?}/agent$i"
  mkdir -p "$BOOTDIR"
  nohup ./launch_agent.sh "$i" > "$BOOTDIR/agent${i}.fix.log" 2>&1 &
  disown
  echo "  relaunched agent$i — rebuilds root then boots (~60-90s). Verify: ./fleet.sh status"
}

cmd_cycle() {
  local n="${1:-$N_DEFAULT}"
  cmd_preflight || { echo "preflight failed — fix externals first"; return 1; }
  cmd_down
  cmd_deploy || return 1
  cmd_up "$n"
  echo "Waiting for fleet to join (up to 180s)..."
  local deadline=$(( $(date +%s) + 180 ))
  while [ $(date +%s) -lt $deadline ]; do
    if cmd_status "$n" >/dev/null 2>&1; then break; fi
    sleep 6
  done
  cmd_status "$n"
}

sub="${1:-}"; shift || true
case "$sub" in
  preflight) cmd_preflight "$@";;
  down)      cmd_down "$@";;
  deploy)    cmd_deploy "$@";;
  up)        cmd_up "$@";;
  status)    cmd_status "$@";;
  fix)       cmd_fix "$@";;
  cycle)     cmd_cycle "$@";;
  *) echo "usage: $0 {preflight|down|deploy|up [N]|status [N]|fix N|cycle [N]}"; exit 2;;
esac
