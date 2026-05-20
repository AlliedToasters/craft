"""Live integration test for the homunculus /equip tech-tier hierarchy.

When the agent holds multiple tiers of the same equipment class (e.g.
wooden_sword + iron_sword, leather_helmet + iron_helmet), homunculus's
/equip endpoint must place the *highest tier* into the appropriate slot.
The agent.py per-turn "Equipment:" readout depends on this contract — its
nudge ("you have no helmet!" vs "helmet: iron_helmet") is only meaningful
when the substrate consistently equips the best gear available.

Each iter exercises a battery of mixed-tier scenarios:
  - tool-only mixes (best of class wins)
  - armor-only with one tier
  - mixed leather + iron armor (iron should be equipped)
  - diamond beating iron

For every scenario we verify both:
  (1) homunculus's /inventory.armor slots contain the highest-tier item
      from what was given — proves the tech-tier hierarchy.
  (2) agent.py's `_format_inventory` reflects the same best-of-class lines
      in the Equipment block — proves the readout matches reality.

Standalone setup; setup_clean handles peaceful difficulty + clear inventory.
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from pathlib import Path

import requests

from craft.agent import _format_inventory
from craft.testkit import (
    HOMUNCULUS_BASE,
    PLAYER_NAME,
    TestLogger,
    cmd,
    inventory,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)


DEFAULT_PASS_RATE = 1.0  # all scenarios must pass


# Each scenario:
#   give              — bare item names (no "minecraft:" prefix) to /give.
#   armor_expect      — equipped armor slot expectation (head/chest/legs/feet
#                       → "minecraft:..." or None for vacant).
#   equipment_lines   — substrings that MUST appear in the rendered
#                       Equipment block.
#   equipment_forbid  — substrings that must NOT appear (used to ban
#                       lower-tier names that should have been outranked).
SCENARIOS: list[dict] = [
    {
        "name": "all_wood_tools",
        "give": ["wooden_pickaxe", "wooden_sword", "wooden_axe", "wooden_shovel"],
        "armor_expect": {"head": None, "chest": None, "legs": None, "feet": None},
        "equipment_lines": [
            "best weapon: wooden_sword",
            "best shovel: wooden_shovel",
            "best pickaxe: wooden_pickaxe",
            "best axe: wooden_axe",
            "helmet: you have no helmet!",
            "chestplate: you have no chestplate!",
            "leggings: you have no leggings!",
            "boots: you have no boots!",
        ],
        "equipment_forbid": [],
    },
    {
        "name": "mixed_tools_higher_tier_wins",
        "give": ["wooden_sword", "iron_sword", "stone_pickaxe", "wooden_pickaxe"],
        "armor_expect": {"head": None, "chest": None, "legs": None, "feet": None},
        "equipment_lines": [
            "best weapon: iron_sword",
            "best pickaxe: stone_pickaxe",
            "best shovel: you are digging barehanded!",
            "best axe: you are chopping barehanded!",
        ],
        "equipment_forbid": [
            "best weapon: wooden_sword",
            "best pickaxe: wooden_pickaxe",
        ],
    },
    {
        "name": "full_iron_armor",
        "give": ["iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots"],
        "armor_expect": {
            "head":  "minecraft:iron_helmet",
            "chest": "minecraft:iron_chestplate",
            "legs":  "minecraft:iron_leggings",
            "feet":  "minecraft:iron_boots",
        },
        "equipment_lines": [
            "helmet: iron_helmet",
            "chestplate: iron_chestplate",
            "leggings: iron_leggings",
            "boots: iron_boots",
        ],
        "equipment_forbid": [
            "you have no helmet!",
            "you have no chestplate!",
            "you have no leggings!",
            "you have no boots!",
        ],
    },
    {
        "name": "leather_plus_iron_armor_iron_equipped",
        "give": ["leather_helmet", "iron_helmet", "leather_boots", "iron_boots"],
        "armor_expect": {
            "head":  "minecraft:iron_helmet",
            "chest": None,
            "legs":  None,
            "feet":  "minecraft:iron_boots",
        },
        "equipment_lines": [
            "helmet: iron_helmet",
            "boots: iron_boots",
            "chestplate: you have no chestplate!",
            "leggings: you have no leggings!",
        ],
        "equipment_forbid": [
            "helmet: leather_helmet",
            "boots: leather_boots",
        ],
    },
    {
        "name": "diamond_beats_iron",
        "give": ["iron_sword", "diamond_sword", "iron_chestplate", "diamond_chestplate"],
        "armor_expect": {
            "head":  None,
            "chest": "minecraft:diamond_chestplate",
            "legs":  None,
            "feet":  None,
        },
        "equipment_lines": [
            "best weapon: diamond_sword",
            "chestplate: diamond_chestplate",
        ],
        "equipment_forbid": [
            "best weapon: iron_sword",
            "chestplate: iron_chestplate",
        ],
    },
]


def _equipped_armor_id(inv: dict | None, slot: str) -> str | None:
    """/inventory armor.<slot>.id, or None if vacant."""
    if not inv:
        return None
    armor = inv.get("armor") or {}
    s = armor.get(slot)
    if not s:
        return None
    return s.get("id")


def _post_equip() -> dict:
    """POST /equip — homunculus reorganizes hotbar + armor based on inventory."""
    try:
        r = requests.post(f"{HOMUNCULUS_BASE}/equip", timeout=10.0)
        r.raise_for_status()
        return r.json() or {}
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "error": repr(e)}


def _give_items(items: list[str]) -> None:
    """Give the listed bare item names (one each) via the server console."""
    for it in items:
        cmd(f"give {PLAYER_NAME} minecraft:{it} 1")
    # Server-side give settles fast; small pad so the next /inventory read sees it.
    time.sleep(0.4)


def _clear_inventory() -> None:
    cmd(f"clear {PLAYER_NAME}")
    time.sleep(0.2)


def _check_scenario(scenario: dict, verbose: bool) -> tuple[bool, str, dict]:
    """Run one scenario. Returns (passed, fail_reason, debug_info)."""
    name = scenario["name"]
    _clear_inventory()
    _give_items(scenario["give"])

    equip_resp = _post_equip()
    if not equip_resp.get("success"):
        return False, f"equip_post_failed:{equip_resp.get('error','?')}", {"equip": equip_resp}

    inv = inventory()
    if inv is None:
        return False, "inventory_fetch_failed", {}

    # Check (1): homunculus equipped the highest-tier armor.
    armor_actual = {
        slot: _equipped_armor_id(inv, slot)
        for slot in ("head", "chest", "legs", "feet")
    }
    armor_expect = scenario["armor_expect"]
    armor_mismatch = {
        slot: (armor_expect[slot], armor_actual[slot])
        for slot in armor_expect
        if armor_expect[slot] != armor_actual[slot]
    }
    if armor_mismatch:
        return False, "armor_slot_mismatch", {
            "expect": armor_expect, "actual": armor_actual, "mismatch": armor_mismatch,
        }

    # Check (2): the rendered Equipment block reflects best-of-class.
    rendered = _format_inventory(inv) or ""
    missing = [s for s in scenario["equipment_lines"] if s not in rendered]
    surprising = [s for s in scenario["equipment_forbid"] if s in rendered]
    if missing or surprising:
        return False, "equipment_readout_mismatch", {
            "missing": missing, "surprising": surprising, "rendered": rendered,
        }

    if verbose:
        print(f"  [{name}] OK", flush=True)
    return True, "", {"armor": armor_actual}


def run_iter(rec: dict, *, spawn_range: int, rng: _random.Random, verbose: bool) -> None:
    if spawn_range > 0:
        spawn_result = random_spawn(range_blocks=spawn_range, rng=rng, verbose=verbose)
        rec["spawn"] = spawn_result
        if not spawn_result.get("ok"):
            rec["passed"] = False
            rec["fail_reason"] = "spawn_retry_exhausted"
            return

    start = pos()
    if start is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_position"
        return
    anchor = (
        int(math.floor(start[0])),
        int(math.floor(start[1])),
        int(math.floor(start[2])),
    )
    rec["anchor"] = list(anchor)
    setup_clean(anchor)

    # Run every scenario; iter passes only if ALL scenarios pass. Per-scenario
    # results are recorded so failure mode is easy to read in the JSONL.
    rec["scenarios"] = []
    iter_passed = True
    first_fail: str | None = None
    for scenario in SCENARIOS:
        passed, reason, debug = _check_scenario(scenario, verbose)
        rec["scenarios"].append({
            "name": scenario["name"],
            "passed": passed,
            "fail_reason": reason if not passed else None,
            "debug": debug if not passed else None,
        })
        if not passed:
            iter_passed = False
            if first_fail is None:
                first_fail = f"{scenario['name']}:{reason}"

    _clear_inventory()

    rec["passed"] = iter_passed
    if not iter_passed:
        rec["fail_reason"] = first_fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--spawn-range", type=int, default=0)
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger("equip", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    spawn_range=args.spawn_range,
                    rng=rng,
                    verbose=not args.quiet,
                )
        except Exception as e:
            print(f"[test] iter {i} raised: {e!r}", flush=True)

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
