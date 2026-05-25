#!/bin/bash
# Launches PrismLauncher with the 1.21.4.agent<N> instance bound to
# offline account "agent<N>". Each agent's homunculus binds 25570+N
# (configured via -Dhomunculus.port in the instance's JvmArgs).
#
# Usage: launch_agent.sh N        # N in 0..31
#
# Per-agent application root (single-instance isolation)
# ------------------------------------------------------
# PrismLauncher 10.x has single-instance IPC: the FIRST launcher becomes a
# resident "primary" and every later launch is forwarded to it, inheriting the
# primary's per-instance JVM args (so all agents would collide on one homunculus
# port). PrismLauncher scopes single-instance by its application root (-d), so we
# give each agent its OWN root under $PRISM_AGENT_ROOTS/agent<N>. The mutable
# per-agent state (the instance dir: logs/saves/options/config) is a full COPY so
# agents never share a writable file; the multi-GB read-only engine dirs
# (assets/libraries/meta/java/...) are symlinked from the canonical PrismLauncher
# dir to save disk. Roots are auto-built on first launch.
#
# Config (env vars; may live in ./.env):
#   PRISMLAUNCHER_BIN   default: prismlauncher. Point at the AppImage for that
#                       build; the script auto-adds --appimage-extract-and-run.
#   CRAFT_MC_HOST       default: 127.0.0.1. MC server to auto-join (host[:port]).
#   CRAFT_GPU_RENDER    default: unset -> force Mesa software (llvmpipe) GL so the
#                       client rasterizes on CPU, leaving the GPU for the LLM.
#                       Set =1 to render on the GPU.
#   CRAFT_XVFB_RES      default: 1280x720x24. Xvfb geometry (headless only).
#   PRISM_CANON_DIR     default: $XDG_DATA_HOME/PrismLauncher. Canonical data dir
#                       (source of accounts.json + shared engine dirs + the
#                       agent0 instance template).
#   PRISM_AGENT_ROOTS   default: $XDG_DATA_HOME/pl-agents. Where per-agent roots
#                       live. Delete a root to force a clean rebuild.
#
# Display handling:
#   - If $DISPLAY is set, it is used as-is.
#   - If headless (no $DISPLAY), a private Xvfb is started per agent via
#     `xvfb-run -a`. We deliberately do NOT route to the physical :0 GDM
#     session: that renders on the GPU, which is reserved for the model.
set -e
# Mirror craft/__init__.py: shell + Python share .env config.
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
N="${1:?usage: $0 <agent-number 0..19>}"
case "$N" in [0-9]|1[0-9]|2[0-9]|3[01]) ;; *)
  echo "agent number must be 0..31, got '$N'" >&2; exit 2;;
esac

PRISM="${PRISMLAUNCHER_BIN:-prismlauncher}"
HOST="${CRAFT_MC_HOST:-127.0.0.1}"

XDG="${XDG_DATA_HOME:-$HOME/.local/share}"
CANON="${PRISM_CANON_DIR:-$XDG/PrismLauncher}"
ROOTBASE="${PRISM_AGENT_ROOTS:-$XDG/pl-agents}"
ROOT="$ROOTBASE/agent$N"
INST="$ROOT/instances/1.21.4.agent$N"

# Offline-account ids (v3 UUID of MD5("OfflinePlayer:agent<N>"), dashless).
AGENT_UUIDS=(
  5abef2b70d3a3de98e6cf3b48558fb16  # agent0
  4312fad514e43c46a21cd9d706b4c920  # agent1
  7142801115753945bd56ab4cf504095c  # agent2
  c1f4ab4891143f5caafe206eaa641b68  # agent3
  9cda2d7deed73d999c409c48ba45bd1d  # agent4
  ef91522a1e2e3b93a08dbcd5b4960be4  # agent5
  396654a4f82e34a48b12bf828fa92a7e  # agent6
  2c3b25310e88368e94081bc651b43898  # agent7
  5de5c5e4e386367785ca6fd9ecb80a0e  # agent8
  f9e67ae5b72a3cc08984d7f9cb7ca657  # agent9
  fb8c879f29a83d54affb8eb5ef54cbab  # agent10
  21bf534730743a808215208bde341984  # agent11
  5de80f35e3fe33a1b86c3bd6516e9a66  # agent12
  a71f9e8bd70f3ab381381de48ab18ab0  # agent13
  6c6135384d31303cacd5b1f4110cbfa3  # agent14
  9ab4f304fe11381fb1e102fbf91cac7a  # agent15
  9f39b8edc2933aefbadb520913e62cee  # agent16
  50890a317ea0388c95dcaec782f987f4  # agent17
  2b98d068efb439b29f1e0b253dfd5707  # agent18
  376ea849d24f36e6becd38cdc66a9a2f  # agent19
  8ed99d1d396d37fab8015d9540014de3  # agent20
  c5c0550f802437bdb5a7c7edac098ba2  # agent21
  64f6fbf563d23678b094092b498a37bf  # agent22
  3b6fcd2b232934f1ace53657a10a5d76  # agent23
  9d68b3f939df3e6abb6cd8ba9cf2f337  # agent24
  fbb54e1ac37c31ff8911581f30f2338f  # agent25
  a3ec3c44dfc23139935c6cecbcafdfc8  # agent26
  b2753c45dc623cf1bd0dade0dbdb8619  # agent27
  6db9b458ec453dabb00b5646c430bdb9  # agent28
  3315bc2d40143042b38f6951e70503f0  # agent29
  6ea3731cc7c33aa48527857060d78133  # agent30
  07e9e71a64dd3df9bf2457de93476eef  # agent31
)

# Build the per-agent root once. Guard on a `.built` sentinel written only after
# a COMPLETE build, so an interrupted build (partial copy) self-heals on relaunch
# instead of being skipped. A stale/partial root is wiped before rebuilding.
if [ ! -f "$ROOT/.built" ]; then
  echo "[launch_agent] building per-agent root: $ROOT" >&2
  TEMPLATE="$CANON/instances/1.21.4.agent0"
  if [ ! -d "$TEMPLATE" ]; then
    echo "[launch_agent] missing instance template: $TEMPLATE" >&2; exit 3
  fi
  rm -rf "$ROOT"
  mkdir -p "$ROOT/instances"
  for d in assets libraries meta java icons iconthemes catpacks themes translations; do
    [ -e "$CANON/$d" ] && ln -sfn "$CANON/$d" "$ROOT/$d"
  done
  # Copy only the agent accounts — strip MSA/human accounts so launchers
  # don't simultaneously refresh them on startup (causes 429 rate limiting
  # and PrismLauncher hanging on a GUI auth dialog in headless mode).
  if [ -f "$CANON/accounts.json" ]; then
    python3 -c "
import json, sys
data = json.load(open('$CANON/accounts.json'))
target_uuid = sys.argv[1]
# Keep ONLY this agent's account — avoids PrismLauncher cycling through all accounts
# and ensures activeAccount points to a real entry.
data['accounts'] = [a for a in data.get('accounts', [])
                    if a.get('profile', {}).get('id', '') == target_uuid]
data['activeAccount'] = target_uuid
json.dump(data, open('$ROOT/accounts.json', 'w'))
" "${AGENT_UUIDS[$N]}" 2>/dev/null || cp -f "$CANON/accounts.json" "$ROOT/accounts.json"
  fi
  [ -f "$CANON/prismlauncher.cfg" ] && cp -f "$CANON/prismlauncher.cfg" "$ROOT/prismlauncher.cfg"
  cp -a "$TEMPLATE" "$INST"
  sed -i -E \
    -e "s|^InstanceAccountId=.*|InstanceAccountId=${AGENT_UUIDS[$N]}|" \
    -e "s|^JvmArgs=.*|JvmArgs=\"-Dhomunculus.port=$((25570+N))\"|" \
    -e "s|^name=.*|name=1.21.4.agent$N|" \
    "$INST/instance.cfg"
  rm -f "$INST/minecraft/logs/"*.log "$INST/minecraft/logs/"*.log.gz 2>/dev/null || true
  touch "$ROOT/.built"
fi

# AppImage builds launch via extract-and-run (no FUSE mount needed).
APPIMAGE_ARGS=()
case "$PRISM" in
  *.AppImage|*AppImage) APPIMAGE_ARGS=(--appimage-extract-and-run) ;;
esac

# Keep rendering off the GPU unless explicitly opted out.
if [ -z "$CRAFT_GPU_RENDER" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export GALLIUM_DRIVER=llvmpipe
  export __GLX_VENDOR_LIBRARY_NAME=mesa
fi

PRISM_ARGS=("-d" "$ROOT" "-l" "1.21.4.agent${N}" "-s" "$HOST" "-a" "agent${N}")

# Auto-dismiss PrismLauncher first-run dialogs that appear when there are no
# MSA accounts: "Quick Setup" wizard (wants MSA) and "Play demo?" (no license).
# Runs as a background loop alongside the launcher; killed on EXIT.
_prism_dialog_watcher() {
  local d="$1" x="${2:-}"
  while true; do
    sleep 3
    # Quick Setup wizard — Finish button at window-relative (573,635) in a 620×660 dialog at (320,0)
    w=$(DISPLAY=$d XAUTHORITY=$x xdotool search --name "Quick Setup" 2>/dev/null | head -1)
    [ -n "$w" ] && DISPLAY=$d XAUTHORITY=$x xdotool mousemove 893 635 click 1 2>/dev/null
    # "Play demo?" — Play Demo button at window-relative (175,100) in a 348×127 dialog at (456,257)
    w=$(DISPLAY=$d XAUTHORITY=$x xdotool search --name "Play demo" 2>/dev/null | head -1)
    [ -n "$w" ] && DISPLAY=$d XAUTHORITY=$x xdotool mousemove 631 357 click 1 2>/dev/null
  done
}

# Reap the dialog watcher (and headless Xvfb + the launcher itself) on any exit
# path. We deliberately do NOT `exec` PrismLauncher: exec replaces this shell and
# discards the EXIT trap, so the backgrounded watcher orphans and leaks on every
# launch. Running PrismLauncher as a child + `wait` keeps this shell alive to reap
# everything. The signal trap routes INT/TERM/HUP through the EXIT trap so killing
# the launcher (or the client) tears down the whole agent — no orphaned watchers.
_cleanup() {
  set +e   # a dead PID makes kill fail; under set -e that would abort cleanup
           # mid-way and leak the remaining processes (e.g. the watcher).
  [ -n "${PRISM_PID:-}" ]   && kill "$PRISM_PID"   2>/dev/null
  [ -n "${WATCHER_PID:-}" ] && kill "$WATCHER_PID" 2>/dev/null
  [ -n "${XVFB_PID:-}" ]    && kill "$XVFB_PID"    2>/dev/null
  return 0
}
trap _cleanup EXIT
trap 'exit 143' INT TERM HUP

if [ -n "$DISPLAY" ]; then
  _prism_dialog_watcher "$DISPLAY" "${XAUTHORITY:-}" &
  WATCHER_PID=$!
else
  RES="${CRAFT_XVFB_RES:-1280x720x24}"
  # Use a fixed display per agent (200+N) so we know the display before launch.
  AGENT_DISPLAY=":$((200 + N))"
  Xvfb "$AGENT_DISPLAY" -screen 0 "${RES}" -nolisten tcp &
  XVFB_PID=$!
  sleep 1
  export DISPLAY="$AGENT_DISPLAY"
  _prism_dialog_watcher "$AGENT_DISPLAY" "" &
  WATCHER_PID=$!
fi

"$PRISM" "${APPIMAGE_ARGS[@]}" "${PRISM_ARGS[@]}" &
PRISM_PID=$!
wait "$PRISM_PID"
