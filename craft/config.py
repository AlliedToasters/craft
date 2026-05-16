"""Single source for substrate connection constants.

Other modules import HOMUNCULUS_BASE / PLAYER_NAME / SERVER_CMD_BASE from
here so a subprocess can retarget the *whole* stack at a different MC
client by setting environment variables before invoking python — no
function-signature plumbing needed.

Env vars (all optional, defaults assume a single-box install):
  HOMUNCULUS_HOST       default 127.0.0.1
  HOMUNCULUS_PORT       default 25566 (main 1.21.4 instance)
  MC_PLAYER_NAME        default Player
  MC_SERVER_CMD_BASE    default http://127.0.0.1:4747

The port scheme used by launch_agent.sh:
  agent0..agent9 -> 25570..25579, with offline accounts of the same names.
A test subprocess targeting agent3 sets HOMUNCULUS_PORT=25573 and
MC_PLAYER_NAME=agent3; every HTTP call into homunculus + every console
TP/gamemode/effect command then addresses that client and player.
"""

from __future__ import annotations

import os


HOMUNCULUS_HOST: str = os.environ.get("HOMUNCULUS_HOST", "127.0.0.1")
HOMUNCULUS_PORT: int = int(os.environ.get("HOMUNCULUS_PORT", "25566"))
HOMUNCULUS_BASE: str = f"http://{HOMUNCULUS_HOST}:{HOMUNCULUS_PORT}"

PLAYER_NAME: str = os.environ.get("MC_PLAYER_NAME", "Player")

SERVER_CMD_BASE: str = os.environ.get(
    "MC_SERVER_CMD_BASE", "http://127.0.0.1:4747"
)
