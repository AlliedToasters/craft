#!/bin/bash
# Launches PrismLauncher with the 1.21.4 instance and auto-connects to the
# Minecraft server.
#
# Config (env vars):
#   PRISMLAUNCHER_BIN   default: prismlauncher (on $PATH)
#                       set to /path/to/PrismLauncher-Linux-x86_64.AppImage
#                       if you're using the AppImage build.
#   CRAFT_MC_HOST       default: 127.0.0.1
#                       hostname/IP of the Minecraft server to auto-join.
#
# Auto-detects display when invoked from an SSH session (no DISPLAY set):
# routes the window to the local :0 GDM session on this box, NOT the SSH client.
if [ -z "$DISPLAY" ]; then
  export DISPLAY=:0
  export XAUTHORITY=/run/user/1000/gdm/Xauthority
fi
PRISM="${PRISMLAUNCHER_BIN:-prismlauncher}"
HOST="${CRAFT_MC_HOST:-127.0.0.1}"
exec "$PRISM" -l 1.21.4 -s "$HOST"
