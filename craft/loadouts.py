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

import requests

from craft.config import HOMUNCULUS_BASE, PLAYER_NAME, SERVER_CMD_BASE
from craft.world import (
    ArmorSlot,
    clear_inventory,
    equip_armor_slot,
    give_item,
    give_to_main_inv_slot,
    set_gamemode,
    set_hunger,
    summon_at,
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
    # Hunt-capability isolation, deterministic variant: a pre-summoned ring
    # of passive mobs (cow/pig/sheep/chicken) lands near the agent so the
    # hunt_passive primitive has guaranteed targets in range. Paired with
    # hunger=2 (foodLevel; saturation=0) so AutoEat triggers as soon as
    # the agent cooks any drops, exposing whether the full hunt → cook
    # → eat chain composes. No armor (capability test is for hunting,
    # not survival ceiling).
    "hunt_meadow": {
        "pre_summon_herd": True,
        "set_hunger": 2,
        "armor": {},
        "main": [
            ("minecraft:stone_sword",  1),
            ("minecraft:torch",        8),
        ],
    },
    # Hunt-capability isolation, natural variant: no pre-summon. Tests
    # combined skill (find herd in biome + hunt) — biome lottery is the
    # confound, run multiple iterations to separate signal. Same hunger
    # pressure + sword + torches as hunt_meadow so the two are directly
    # comparable on the "did the agent try to hunt?" axis.
    "hunt_wild": {
        "pre_summon_herd": False,
        "set_hunger": 2,
        "armor": {},
        "main": [
            ("minecraft:stone_sword",  1),
            ("minecraft:torch",        8),
        ],
    },
    # Cooking-capability isolation: agent starts with raw meat + fuel +
    # furnace in inventory, hunger=2 pressure, and must self-place the
    # furnace and run the smelt → collect chain. Composes from existing
    # primitives (place/smelt/collect_smelt). Tests whether the model
    # reaches for the cook chain when the materials are right there —
    # mirrors hunt_meadow's "everything ready, will the model invoke
    # the verb?" pattern.
    #
    # Critical: raw meat goes in `main_inv_only` (slots 9+) NOT `main`
    # (hotbar). Wurst's AutoEat only eats from hotbar — if raw beef
    # lands in slot 0-8 via /give it gets consumed before the agent can
    # cook (validated 2026-05-21 smoke: 8 raw beef gone by turn 13).
    # Slots 9+ are still in inventory + visible to /inventory, but
    # outside AutoEat's reach.
    "cook_kitchen": {
        "set_hunger": 2,
        "armor": {},
        "main": [
            ("minecraft:coal",         8),   # 1 coal smelts 8 items → exact fuel match
            ("minecraft:furnace",      1),
            ("minecraft:stone_sword",  1),   # survival kit
            ("minecraft:torch",        8),
        ],
        "main_inv_only": [
            ("minecraft:beef",         8),   # raw beef — hidden from AutoEat
        ],
    },
}


# Pre-summon ring for hunt_meadow. 8 mobs across 4 species in four
# clusters at ±20 blocks (N/E/S/W). Radius 20 puts the herd OUT of
# KillAura's melee range (~5 blocks); the only ways for the agent to
# engage are (a) deliberately call hunt_passive — which paths via
# /baritone/goto — or (b) walk into a cluster while doing something
# else. Radius < 32 keeps the herd inside hunt_passive's default scan
# radius so the agent's first scan will see them.
# Prior version used radius 5 (all 8 mobs adjacent) which let KillAura
# instakill the entire herd before the agent's first turn — the
# substrate fed the agent automatically and hunt_passive was never
# called. See 2026-05-21 smoke findings for details.
_HERD_RING: list[tuple[str, int, int]] = [
    # N cluster (~20 blocks)
    ("minecraft:cow",      0,  20),
    ("minecraft:pig",      2,  20),
    # E cluster
    ("minecraft:sheep",   20,   0),
    ("minecraft:chicken", 20,   2),
    # S cluster
    ("minecraft:cow",      0, -20),
    ("minecraft:pig",     -2, -20),
    # W cluster
    ("minecraft:sheep",  -20,   0),
    ("minecraft:chicken",-20,  -2),
]


def _fetch_player_pos(
    homunculus_base: str, *, timeout: float = 4.0,
) -> tuple[float, float, float] | None:
    """GET /position from homunculus. Returns (x, y, z) or None on error.

    Used by pre_summon_herd to place mobs relative to the agent's actual
    spawn location (which is set by random_spawn upstream of apply_loadout,
    so we can't precompute it).
    """
    try:
        r = requests.get(f"{homunculus_base}/position", timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except (requests.RequestException, ValueError):
        return None
    x, y, z = body.get("x"), body.get("y"), body.get("z")
    if not all(isinstance(v, (int, float)) for v in (x, y, z)):
        return None
    return (float(x), float(y), float(z))


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

    # Optional pre-summon step: spawn a passive-mob ring around the player
    # before clearing inventory + giving items. Runs first so the herd is
    # nearby when the agent boots into survival. summon_at uses /summon
    # via the server console — no homunculus changes needed.
    if spec.get("pre_summon_herd"):
        pos = _fetch_player_pos(HOMUNCULUS_BASE)
        if pos is None:
            print(
                "[loadout] pre_summon_herd FAILED: couldn't fetch player position",
                flush=True,
            )
            report["steps"].append({
                "step": "pre_summon_herd",
                "result": {"ok": False, "error": "position_fetch_failed"},
            })
        else:
            px, py, pz = pos
            ring_results: list[dict] = []
            for entity_id, dx, dz in _HERD_RING:
                res = summon_at(
                    entity_id, px + dx, py, pz + dz,
                    server_cmd_base=server_cmd_base,
                )
                ring_results.append({
                    "entity": entity_id, "offset": [dx, dz], "result": res,
                })
            spawned_n = sum(1 for r in ring_results if r["result"].get("ok"))
            print(
                f"[loadout] pre_summon_herd: spawned {spawned_n}/{len(_HERD_RING)} "
                f"mobs around ({px:.1f},{py:.1f},{pz:.1f})",
                flush=True,
            )
            report["steps"].append({
                "step": "pre_summon_herd",
                "result": {
                    "ok": spawned_n == len(_HERD_RING),
                    "spawned": spawned_n,
                    "total": len(_HERD_RING),
                    "ring": ring_results,
                },
            })

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

    # Optional `main_inv_only` items — placed in main-inventory slots
    # (9, 10, 11, ...) via /item replace entity. Used to hide items from
    # Wurst's AutoEat (which only consumes hotbar items) and from any
    # other hotbar-stuck consumer. Slot 9 is the first slot above the
    # hotbar; we increment from there.
    for i, (item_id, count) in enumerate(spec.get("main_inv_only") or []):
        slot = 9 + i
        if slot > 35:
            # Main inv is 9-35 = 27 slots; loadouts shouldn't need that many
            report["steps"].append({
                "step": "give_main_inv", "item": item_id, "count": count,
                "result": {"ok": False, "error": f"main inv slot {slot} out of range (max 35)"},
            })
            continue
        res = give_to_main_inv_slot(
            slot, item_id, count,
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({
            "step": "give_main_inv", "item": item_id, "count": count,
            "slot": slot, "result": res,
        })

    # Optional hunger-pressure step. LAST so it doesn't get reset by a
    # gamemode change (set_gamemode survival/creative refills foodLevel
    # incidentally). saturation defaults to 0 — the hidden buffer would
    # otherwise mask the foodLevel drop for several seconds.
    hunger_level = spec.get("set_hunger")
    if hunger_level is not None:
        hres = set_hunger(
            int(hunger_level),
            player_name=player_name, server_cmd_base=server_cmd_base,
        )
        report["steps"].append({
            "step": "set_hunger", "level": int(hunger_level), "result": hres,
        })

    report["ok"] = all(
        s.get("result", {}).get("ok") is True for s in report["steps"]
    )
    return report
