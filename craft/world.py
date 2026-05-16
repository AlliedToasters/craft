"""Unified world-state mutation primitives.

Tests and rollouts share this — there should be no other place in the
codebase that POSTs raw `difficulty`, `time set`, or `gamemode` strings
to the server console. Centralization makes regressions ("difficulty
wasn't actually peaceful when X happened") easy to locate: this module
is the only place to look.

All primitives:
  - take typed/Literal values, no magic strings to typo
  - are idempotent (safe to call twice in a row)
  - return the underlying console response dict so callers can detect failure

The MC server console at SERVER_CMD_BASE accepts any vanilla command
via POST /cmd {"cmd":"..."} — these wrappers just spell the verbs
canonically.
"""

from __future__ import annotations

import random
from typing import Literal

import requests


from craft.config import SERVER_CMD_BASE, PLAYER_NAME  # noqa: F401


Difficulty = Literal["peaceful", "easy", "normal", "hard"]
Gamemode = Literal["survival", "creative", "adventure", "spectator"]
Phase = Literal["dawn", "noon", "dusk", "midnight", "random"]


# MC day cycle ticks per named phase. random → uniform tick in [0, 24000).
# Source of truth; agent.py imports PHASE_TICKS from here.
PHASE_TICKS: dict[str, int] = {
    "dawn": 0,         # 0 = sunrise (matches a fresh single-player world)
    "noon": 6000,
    "dusk": 12000,
    "midnight": 18000,
}


def _cmd(server_cmd_base: str, s: str, *, timeout: float = 5.0) -> dict:
    """POST one console command. Internal — public callers use the typed verbs."""
    try:
        r = requests.post(f"{server_cmd_base}/cmd",
                          json={"cmd": s}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": str(e)}


def set_difficulty(
    level: Difficulty,
    *,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Change global difficulty.

    'peaceful' is the only level that despawns existing hostile mobs and
    blocks new ones — tests rely on this for clean setup/teardown.
    'easy'/'normal'/'hard' all permit hostile damage.
    """
    return _cmd(server_cmd_base, f"difficulty {level}")


def set_gamemode(
    mode: Gamemode,
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Change player gamemode.

    Tests + spawn-retry use creative for safe TP (fall damage immunity,
    suffocation shield) then survival for the actual test.
    """
    return _cmd(server_cmd_base, f"gamemode {mode} {player_name}")


def resolve_phase_ticks(phase: str | int) -> int:
    """Map a phase name or explicit tick to a tick in [0, 24000)."""
    if isinstance(phase, int):
        return phase % 24000
    if phase == "random":
        return random.randint(0, 23999)
    if phase in PHASE_TICKS:
        return PHASE_TICKS[phase]
    raise ValueError(
        f"unknown phase {phase!r}; valid: "
        f"{sorted(PHASE_TICKS) + ['random']} or int tick"
    )


def set_time(
    phase: str | int,
    *,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Set MC time-of-day. Accepts a phase name or an explicit tick."""
    ticks = resolve_phase_ticks(phase)
    return _cmd(server_cmd_base, f"time set {ticks}")
