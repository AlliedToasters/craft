"""Live integration test for the AutoDrop whitelist-policy substrate.

The agent's craft graph depends on AutoDrop seeding a *complement* of
`craft.autodrop.ALWAYS_KEEP` into Wurst's `Items` filter every rollout
start. Two contracts must hold for the substrate to be load-bearing:

  (1) Wurst's default filter (small: flowers + rotten_flesh + wheat_seeds)
      KEEPS the useful items the agent might pick up (dirt, ingots, etc.).
      Without this, ablation runs (`CRAFT_AUTODROP_TIER=off`) lose more
      than they should.

  (2) Once seeded with bare-tier drops via /wurst/setting, Wurst DROPS the
      long-tail junk we intend to drop (sand/gravel/basalt/flowers/seeds/
      saddle/name_tag/etc.) and KEEPS the canonical keepers (ingots, gems,
      foods, tools, fuel, rare mob drops, logs+planks of every species).

Each iter exercises both contracts via /give bursts and a poll loop that
waits for AutoDrop's tick handler to settle (max ~6s). Inventory snapshot
after settle is compared against the expected keep set per scenario.

Standalone setup; setup_clean handles peaceful difficulty + clear inventory
so dying mobs don't drag items into the inventory mid-test.
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from pathlib import Path

from craft.autodrop import ALWAYS_KEEP, drop_list_for_tier
from craft.testkit import (
    PLAYER_NAME,
    TestLogger,
    cmd,
    inventory,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.wurst import seed_autodrop_from_tier, set_item_list


DEFAULT_PASS_RATE = 1.0  # all scenarios must pass


# Each scenario specifies the AutoDrop setup, the items to /give, and which
# of the given items should be kept vs. dropped once AutoDrop has ticked.
# `setup`:
#   ("reset",)         — POST /wurst/setting op=reset (back to Wurst defaults)
#   ("seed", "<tier>") — seed_autodrop_from_tier(<tier>) (our policy)
SCENARIOS: list[dict] = [
    {
        # Wurst default filter doesn't drop ingots, food, or building blocks.
        # Validates the baseline so "OFF arm" rollouts aren't broken.
        "name": "wurst_default_keeps_useful",
        "setup": ("reset",),
        "give": [
            "minecraft:dirt", "minecraft:cobblestone", "minecraft:iron_ingot",
            "minecraft:apple", "minecraft:oak_log",
        ],
        "expect_kept": {
            "minecraft:dirt", "minecraft:cobblestone", "minecraft:iron_ingot",
            "minecraft:apple", "minecraft:oak_log",
        },
        "expect_dropped": set(),
    },
    {
        # Wurst default filter does drop flowers + rotten_flesh + wheat_seeds.
        # Sanity check that AutoDrop is actually firing under "reset" state.
        "name": "wurst_default_drops_flowers",
        "setup": ("reset",),
        "give": [
            "minecraft:poppy", "minecraft:dandelion", "minecraft:sunflower",
            "minecraft:rotten_flesh", "minecraft:wheat_seeds",
        ],
        "expect_kept": set(),
        "expect_dropped": {
            "minecraft:poppy", "minecraft:dandelion", "minecraft:sunflower",
            "minecraft:rotten_flesh", "minecraft:wheat_seeds",
        },
    },
    {
        # bare-tier seed must KEEP everything in the canonical keeper set.
        # Pulls from one item per ALWAYS_KEEP semantic bucket so a category
        # regression (e.g. someone accidentally dropping `food` from
        # ALWAYS_KEEP) surfaces as a single readable scenario failure.
        "name": "bare_tier_keeps_canonical",
        "setup": ("seed", "bare"),
        "give": [
            "minecraft:dirt",         # shelter
            "minecraft:cobblestone",  # shelter + tool material
            "minecraft:iron_ingot",   # ingot
            "minecraft:raw_iron",     # raw ore
            "minecraft:diamond",      # gem
            "minecraft:oak_planks",   # planks
            "minecraft:oak_log",      # log
            "minecraft:crimson_stem", # nether stem
            "minecraft:coal",         # fuel
            "minecraft:stick",        # craft chain
            "minecraft:apple",        # food
            "minecraft:cooked_beef",  # food
            "minecraft:wheat",        # food chain
            "minecraft:string",       # mob drop (rare)
            "minecraft:leather",      # mob drop (armor)
            "minecraft:feather",      # mob drop (arrows)
            "minecraft:gunpowder",    # mob drop (rare)
            "minecraft:torch",        # lighting
            "minecraft:crafting_table",
            "minecraft:furnace",
            "minecraft:oak_door",     # shelter
            "minecraft:white_bed",    # skip-night
            "minecraft:flint_and_steel",
            "minecraft:water_bucket",
            "minecraft:wooden_pickaxe",
            "minecraft:iron_sword",
            "minecraft:iron_helmet",
        ],
        "expect_kept": "ALL_GIVEN",  # sentinel — see _resolve_expectations
        "expect_dropped": set(),
    },
    {
        # bare-tier seed must DROP the long-tail junk. Items here are
        # representative of each drop category (gravity blocks, stone variants,
        # flowers, leaves/saplings, seeds, exotic mob drops, decorative,
        # spawn eggs, bamboo). Catches over-permissive keep additions.
        "name": "bare_tier_drops_long_tail",
        "setup": ("seed", "bare"),
        "give": [
            "minecraft:sand", "minecraft:gravel",
            "minecraft:andesite", "minecraft:granite", "minecraft:diorite",
            "minecraft:tuff", "minecraft:calcite", "minecraft:dripstone_block",
            "minecraft:basalt", "minecraft:smooth_basalt", "minecraft:blackstone",
            "minecraft:end_stone", "minecraft:netherrack",
            "minecraft:mud", "minecraft:packed_mud",
            "minecraft:deepslate", "minecraft:cobbled_deepslate",
            "minecraft:poppy", "minecraft:dandelion",
            "minecraft:oak_sapling", "minecraft:oak_leaves",
            "minecraft:beetroot_seeds", "minecraft:pumpkin_seeds",
            "minecraft:rotten_flesh", "minecraft:bone", "minecraft:spider_eye",
            "minecraft:amethyst_shard", "minecraft:saddle", "minecraft:name_tag",
            "minecraft:cow_spawn_egg",
            "minecraft:bamboo",
            "minecraft:stripped_oak_log",  # stripped variants drop
        ],
        "expect_kept": set(),
        "expect_dropped": "ALL_GIVEN",  # sentinel
    },
    {
        # Stripped logs must drop even though regular logs of the same species
        # are kept. Pinpoints the keep set's tag-precision (`_log$` would
        # mistakenly keep stripped variants; explicit listing avoids this).
        "name": "bare_tier_keeps_log_drops_stripped",
        "setup": ("seed", "bare"),
        "give": [
            "minecraft:oak_log", "minecraft:birch_log",
            "minecraft:stripped_oak_log", "minecraft:stripped_birch_log",
            "minecraft:stripped_crimson_stem",
        ],
        "expect_kept": {"minecraft:oak_log", "minecraft:birch_log"},
        "expect_dropped": {
            "minecraft:stripped_oak_log", "minecraft:stripped_birch_log",
            "minecraft:stripped_crimson_stem",
        },
    },
    {
        # Bamboo progression is deferred → bamboo/bamboo_block/bamboo_planks
        # all drop even at bare tier. Pin so a future bamboo unlock has a
        # single test to flip.
        "name": "bare_tier_drops_bamboo",
        "setup": ("seed", "bare"),
        "give": [
            "minecraft:bamboo", "minecraft:bamboo_block", "minecraft:bamboo_planks",
        ],
        "expect_kept": set(),
        "expect_dropped": "ALL_GIVEN",
    },
]


def _resolve_expectations(scenario: dict) -> tuple[set[str], set[str]]:
    """Expand the ALL_GIVEN sentinel against the scenario's give list."""
    given = set(scenario["give"])
    kept = scenario["expect_kept"]
    dropped = scenario["expect_dropped"]
    if kept == "ALL_GIVEN":
        kept = given
    if dropped == "ALL_GIVEN":
        dropped = given
    return set(kept), set(dropped)


def _give_items(items: list[str]) -> None:
    """Give one of each id via the server console."""
    for it in items:
        cmd(f"give {PLAYER_NAME} {it} 1")


def _clear_inventory() -> None:
    cmd(f"clear {PLAYER_NAME}")
    time.sleep(0.2)


def _apply_setup(setup: tuple) -> tuple[bool, str]:
    """Run the per-scenario AutoDrop setup. Returns (ok, fail_reason)."""
    op = setup[0]
    if op == "reset":
        r = set_item_list("AutoDrop", "Items", [], op="reset")
        if not r.get("success"):
            return False, f"reset_failed:{r.get('reason','?')}"
        return True, ""
    if op == "seed":
        tier = setup[1]
        r = seed_autodrop_from_tier(tier, verbose=False)
        if not r.get("ok"):
            return False, f"seed_failed:{r.get('raw', {}).get('reason','?')}"
        return True, ""
    return False, f"unknown_setup_op:{op!r}"


def _inventory_ids(inv: dict | None) -> set[str]:
    """Set of item ids visible in main + offhand + armor slots."""
    if not inv:
        return set()
    ids: set[str] = set()
    for s in inv.get("main", []) or []:
        if s.get("id"):
            ids.add(s["id"])
    oh = inv.get("offhand")
    if oh and oh.get("id"):
        ids.add(oh["id"])
    for s in (inv.get("armor") or {}).values():
        if s and s.get("id"):
            ids.add(s["id"])
    return ids


def _poll_for_settle(
    expect_kept: set[str],
    expect_dropped: set[str],
    *,
    deadline_s: float = 6.0,
    poll_s: float = 0.4,
) -> tuple[set[str], set[str]]:
    """Wait until AutoDrop has settled or deadline elapses.

    Returns (kept_observed, dropped_observed). Settled when the inventory
    contains every `expect_kept` id AND none of the `expect_dropped` ids
    (or the deadline elapses, whichever first).
    """
    t0 = time.time()
    kept = set()
    while True:
        inv = inventory()
        kept = _inventory_ids(inv)
        dropped_still_present = expect_dropped & kept
        kept_all_present = expect_kept <= kept
        if kept_all_present and not dropped_still_present:
            break
        if time.time() - t0 > deadline_s:
            break
        time.sleep(poll_s)
    dropped = set(expect_dropped) - kept
    return kept, dropped


def _check_scenario(scenario: dict, verbose: bool) -> tuple[bool, str, dict]:
    """Run one scenario. Returns (passed, fail_reason, debug_info)."""
    name = scenario["name"]
    expect_kept, expect_dropped = _resolve_expectations(scenario)

    _clear_inventory()

    ok, reason = _apply_setup(scenario["setup"])
    if not ok:
        return False, reason, {}

    _give_items(scenario["give"])
    kept_observed, _ = _poll_for_settle(expect_kept, expect_dropped)

    missing_keepers = expect_kept - kept_observed
    surviving_droppers = expect_dropped & kept_observed

    if missing_keepers or surviving_droppers:
        return False, "autodrop_policy_mismatch", {
            "missing_keepers": sorted(missing_keepers),
            "surviving_droppers": sorted(surviving_droppers),
            "inventory": sorted(kept_observed),
        }

    if verbose:
        kept_n = len(expect_kept)
        drop_n = len(expect_dropped)
        print(f"  [{name}] OK ({kept_n} kept, {drop_n} dropped)", flush=True)
    return True, "", {"kept": sorted(kept_observed)}


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

    # Sanity check: the policy file's keep-set is non-empty. A regression that
    # accidentally empties ALWAYS_KEEP would silently turn this test into
    # "everything dropped"; this fails loudly instead.
    if not ALWAYS_KEEP:
        rec["passed"] = False
        rec["fail_reason"] = "ALWAYS_KEEP_is_empty"
        return
    rec["policy"] = {
        "always_keep_size": len(ALWAYS_KEEP),
        "bare_drop_count": len(drop_list_for_tier("bare")),
    }

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

    # Restore the bare-tier seed so subsequent operations on this agent
    # behave as the rollout substrate would expect (default rollout state).
    seed_autodrop_from_tier("bare", verbose=False)
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
    logger = TestLogger("autodrop", path=Path(args.out) if args.out else None)

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
