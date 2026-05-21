"""Pre-set starting inventories for "loaded rollouts".

A loaded rollout boots an agent into a deterministic high-tier state
(e.g. wearing full iron armor, holding iron tools) without grinding for
it. Two uses:

1. Mid-rollout feature testing. Substrate primitives that only matter
   once an agent reaches a high tier (M2 milestone firing, diamond
   descent, smelting interlocks) used to require an organic ~30-min run
   to even reach the test condition. With a loadout the test is a
   ~2-turn smoke. This module is the load-bearing piece behind that
   capability; see [[feedback-substrate-iteration-loop]] for the
   close-watch loop it enables.

2. A/B testing a high-tier prompt nudge without front-loading the
   "did the agent reach iron tier?" survival lottery. If the question
   is "does the M2 announcement change diamond-reach trajectory?", a
   loaded rollout tests that directly — no need to confound the answer
   with the model's iron-tier hit rate.

A loadout spec is a dict with two keys:
  - `armor`: dict mapping slot ("head"/"chest"/"legs"/"feet") to item_id
  - `main`:  list of (item_id, count) pairs to /give into the main inv

`apply_loadout(name, ...)` runs them in order via craft.world primitives.
"""

from __future__ import annotations

from craft.config import PLAYER_NAME, SERVER_CMD_BASE
from craft.world import (
    ArmorSlot,
    clear_inventory,
    equip_armor_slot,
    give_item,
)


# Registry. Add a preset here and it's selectable via --starting-loadout.
# Compose presets so each one matches a specific milestone or behavior
# we want to test. Keep them deterministic (no random selection inside a
# preset) — reproducibility is the whole point.
LOADOUTS: dict[str, dict] = {
    # The M2 fire condition: full iron armor equipped + iron tools to use
    # it + a survival buffer (food + torches). M2 predicate fires on the
    # first turn after this is applied.
    "iron_armored": {
        "armor": {
            "head":  "minecraft:iron_helmet",
            "chest": "minecraft:iron_chestplate",
            "legs":  "minecraft:iron_leggings",
            "feet":  "minecraft:iron_boots",
        },
        "main": [
            ("minecraft:iron_pickaxe",  1),
            ("minecraft:iron_sword",    1),
            ("minecraft:iron_shovel",   1),
            ("minecraft:iron_axe",      1),
            ("minecraft:cooked_beef",  16),
            ("minecraft:torch",        32),
        ],
    },
}


def apply_loadout(
    name: str,
    *,
    player_name: str = PLAYER_NAME,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> dict:
    """Apply the named loadout to `player_name`. Returns a per-step report.

    Order: clear, then armor slots, then /give for main items. Clearing
    first ensures determinism — repeated rollouts get the exact same
    inventory regardless of pre-existing junk.

    Unknown names raise ValueError so a typo in a campaign script can't
    silently no-op into a normal rollout.
    """
    spec = LOADOUTS.get(name)
    if spec is None:
        known = ", ".join(LOADOUTS) or "(none)"
        raise ValueError(
            f"Unknown loadout {name!r}. Known: {known}."
        )

    report: dict = {"name": name, "steps": []}

    clear_res = clear_inventory(
        player_name=player_name, server_cmd_base=server_cmd_base,
    )
    report["steps"].append({"step": "clear", "result": clear_res})

    for slot, item_id in (spec.get("armor") or {}).items():
        res = equip_armor_slot(
            slot, item_id,
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({
            "step": "equip", "slot": slot, "item": item_id, "result": res,
        })

    for item_id, count in (spec.get("main") or []):
        res = give_item(
            item_id, count,
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({
            "step": "give", "item": item_id, "count": count, "result": res,
        })

    report["ok"] = all(
        s.get("result", {}).get("ok") is True for s in report["steps"]
    )
    return report
