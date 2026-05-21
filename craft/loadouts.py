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
    set_gamemode,
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
    # Sleep-capability isolation: agent is pre-sheltered (cobble box via
    # handle_build_shelter) and holds a bed + stone_sword + survival kit.
    # Paired with `--start-phase dusk`, this is the cleanest prerequisite
    # condition for sleep_in_bed: safe spot already secured, bed in hand,
    # KillAura+sword handles any leakage. Removes the shelter+combat
    # confound from the sleep signal — failure modes left should be
    # timing (not_night) or model-omission (didn't call the tool).
    "dusk_bed": {
        "pre_shelter": True,
        "armor": {},
        "main": [
            ("minecraft:red_bed",      1),
            ("minecraft:stone_sword",  1),
            ("minecraft:cooked_beef",  8),
            ("minecraft:torch",        8),
        ],
    },
}


# Items given to the player during the pre_shelter step. handle_build_shelter
# pulls from inventory; cobblestone is the default tier-0 buildable, and
# diamond tools let excavate chew through any block type (mirrors the
# stress_test_shelter setup pattern — without these, excavate stalls on
# stone/leaves and the build returns "stuck"). All cleared after the build.
_PRE_SHELTER_GIVES = [
    ("minecraft:cobblestone",      128),
    ("minecraft:diamond_pickaxe",    1),
    ("minecraft:diamond_shovel",     1),
    ("minecraft:diamond_axe",        1),
]


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

    # Optional pre-shelter step: build a cobble box at the player's
    # current position before giving the rest of the inventory. Mirrors
    # the stress_test_shelter setup pattern: temp creative → give
    # build material → handle_build_shelter → survival. The shelter
    # material is cleared along with everything else in the subsequent
    # clear_inventory step, so the spec's `main` list is the *post-shelter*
    # inventory verbatim.
    if spec.get("pre_shelter"):
        # Lazy import to avoid a craft.loadouts → craft.tools cycle
        # (craft.tools imports a lot; craft.loadouts is on its hot path
        # via apply_loadout-from-agent.py).
        from craft.tools import handle_build_shelter

        gm_res = set_gamemode(
            "creative",
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({"step": "pre_shelter_gm_creative", "result": gm_res})

        for item_id, qty in _PRE_SHELTER_GIVES:
            give_res = give_item(
                item_id, qty,
                player_name=player_name, server_cmd_base=server_cmd_base,
            )
            report["steps"].append({
                "step": "pre_shelter_give",
                "item": item_id, "count": qty, "result": give_res,
            })

        # Switch to survival before the build — matches stress_test_shelter's
        # pattern and exercises the same code path the agent would experience.
        gm_res2 = set_gamemode(
            "survival",
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({"step": "pre_shelter_gm_survival", "result": gm_res2})

        # build_shelter returns a string. "shelter at (X,Y,Z); ..." on success,
        # "ABORTED: ..." or "FAILED: ..." on different failure classes.
        # Capture it but don't gate apply_loadout's overall ok on it — a
        # failed build will manifest as the agent waking up to bare ground,
        # which is the desired (visible) signal in JSONL.
        try:
            build_txt = handle_build_shelter({})
            build_err = None
        except Exception as e:
            build_txt = ""
            build_err = repr(e)
        built_ok = not (
            build_err
            or build_txt.startswith("ABORTED")
            or build_txt.startswith("FAILED")
        )
        # Surface to stdout so the agent's setup log captures pre_shelter
        # success/failure — apply_loadout's return value is discarded by
        # craft.agent, so without this print the build outcome is invisible.
        status = "ok" if built_ok else "FAILED"
        head = (build_txt or build_err or "<no output>").replace("\n", " | ")[:240]
        print(f"[loadout] pre_shelter_build {status}: {head}", flush=True)
        report["steps"].append({
            "step": "pre_shelter_build",
            "result": {"ok": built_ok, "text": build_txt[:240],
                       "exception": build_err},
        })

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
