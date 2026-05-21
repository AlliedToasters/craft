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


# Inventory mutation primitives. Used by the loaded-rollout system
# (craft/loadouts.py) to materialize a deterministic starting state
# (full iron armor + tools, etc.) without grinding for it. /give is
# not idempotent — repeated calls stack counts. /clear and /item replace
# ARE idempotent.

ArmorSlot = Literal["head", "chest", "legs", "feet"]


def clear_inventory(
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Empty player inventory including armor + offhand. Idempotent."""
    return _cmd(server_cmd_base, f"clear {player_name}")


def give_item(
    item_id: str,
    count: int = 1,
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Add an item to the player's main inventory. NOT idempotent — call
    once per loadout step.
    """
    return _cmd(server_cmd_base, f"give {player_name} {item_id} {count}")


def equip_armor_slot(
    slot: ArmorSlot,
    item_id: str,
    count: int = 1,
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Place an item directly into one of the player's equipped armor
    slots, bypassing the inventory. Idempotent — replaces whatever was
    there. Use slots 'head'/'chest'/'legs'/'feet'.
    """
    return _cmd(
        server_cmd_base,
        f"item replace entity {player_name} armor.{slot} with {item_id} {count}",
    )


def give_to_main_inv_slot(
    slot: int,
    item_id: str,
    count: int = 1,
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Place an item directly into a specific main-inventory slot.

    container slot indices in MC:
      - 0-8: hotbar (visible at bottom; this is where Wurst's AutoEat
        looks for food)
      - 9-35: main inventory (3 rows × 9 cols, hidden when GUI closed)
      - 36-39: armor (head/chest/legs/feet)
      - 40: offhand
    Use 9-35 to hide an item from AutoEat / hotbar-stuck tools. Idempotent
    — replaces whatever was in the slot.

    First user is the cook_kitchen loadout: raw meat must be hidden from
    AutoEat or the agent's test materials get auto-consumed before they
    cook (validated 2026-05-21 smoke).
    """
    return _cmd(
        server_cmd_base,
        f"item replace entity {player_name} container.{slot} "
        f"with {item_id} {count}",
    )


def set_hunger(
    level: int,
    *,
    saturation: float = 0.0,  # kept for API compat; effect drains it
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Apply hunger pressure to drive foodLevel toward `level`.

    Implementation gotcha: MC blocks `data merge entity <player>`
    ("Unable to modify player data" in server log). Player NBT can't be
    written via the data command. So we apply the Hunger effect instead
    — it drains saturation, then food meter, over a few seconds.

    **Peaceful difficulty freezes the food meter at 20** regardless of
    Hunger effect strength. For this primitive to bite, the world must
    be at easy/normal/hard difficulty. Callers using set_hunger in a
    loadout should also set --difficulty easy or above.

    `level` is approximate — Hunger drains toward 0. We pick amplifier
    so the effect overcomes natural food gain (sprinting/jumping etc.):
      - level≤3 (high pressure): amp=10
      - level 4-10 (moderate):   amp=4
      - level≥11:                no effect (no pressure)

    saturation arg kept for API compat; ignored (effect handles it).
    """
    if level >= 11:
        return {"ok": True, "skipped": f"level={level} ≥ 11 → no effect needed"}
    amp = 10 if level <= 3 else 4
    return _cmd(
        server_cmd_base,
        f"effect give {player_name} minecraft:hunger 600 {amp} true",
    )


def summon_at(
    entity_id: str,
    x: float, y: float, z: float,
    *,
    nbt: str = "",
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Spawn one entity at the given world coords.

    `entity_id` is the full namespaced id (e.g. 'minecraft:cow').
    `nbt` is optional — leave empty for vanilla spawn behavior. Mirrors
    the pattern in craft/ambush.py (which uses the raw _server_cmd path
    for its 17-point ring).
    """
    suffix = f" {nbt}" if nbt else ""
    return _cmd(
        server_cmd_base,
        f"summon {entity_id} {x} {y} {z}{suffix}",
    )
