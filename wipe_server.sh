#!/bin/bash
# Wipe the MC world (regenerate from the same seed). Runs a remote wipe
# script over SSH.
#
# Config (env vars):
#   CRAFT_WIPE_SSH        e.g. user@mc-host
#   CRAFT_WIPE_SCRIPT     absolute path on the remote, e.g. ~/server_scripts/mc_wipe.sh
#
# Usage: ./wipe_server.sh
set -e
: "${CRAFT_WIPE_SSH:?set CRAFT_WIPE_SSH=user@host}"
: "${CRAFT_WIPE_SCRIPT:?set CRAFT_WIPE_SCRIPT=/path/to/mc_wipe.sh on the remote}"
exec ssh "$CRAFT_WIPE_SSH" "$CRAFT_WIPE_SCRIPT"
