#!/bin/bash
# Launches PrismLauncher with the 1.21.4.agent<N> instance bound to
# offline account "agent<N>". Each agent's homunculus binds 25570+N
# (configured via -Dhomunculus.port in the instance's JvmArgs).
#
# Usage: launch_agent.sh N        # N ∈ 0..9
#
# Config (env vars):
#   PRISMLAUNCHER_BIN   default: prismlauncher (on $PATH)
#                       set to /path/to/PrismLauncher-Linux-x86_64.AppImage
#                       if you're using the AppImage build.
#   CRAFT_MC_HOST       default: 127.0.0.1
#                       hostname/IP of the Minecraft server to auto-join.
#
# Auto-detects display when invoked from SSH (no DISPLAY set):
# routes to local :0 GDM session on this box, NOT the SSH client.
set -e
# Mirror craft/__init__.py: shell + Python share .env config.
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
N="${1:?usage: $0 <agent-number 0..9>}"
case "$N" in 0|1|2|3|4|5|6|7|8|9) ;; *)
  echo "agent number must be 0..9, got '$N'" >&2; exit 2;;
esac
if [ -z "$DISPLAY" ]; then
  export DISPLAY=:0
  export XAUTHORITY=/run/user/1000/gdm/Xauthority
fi
PRISM="${PRISMLAUNCHER_BIN:-prismlauncher}"
HOST="${CRAFT_MC_HOST:-127.0.0.1}"
exec "$PRISM" \
  -l "1.21.4.agent${N}" \
  -s "$HOST" \
  -a "agent${N}"
