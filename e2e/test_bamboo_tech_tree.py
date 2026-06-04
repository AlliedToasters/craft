"""End-to-end validation of the bamboo wood tech tree (issue #4).

Reproduces the original bamboo_jungle gap and proves the fix. A bamboo jungle is
"dominated almost exclusively by bamboo … standard trees are heavily thinned
out" — so an agent there leans on the cane `minecraft:bamboo` as its wood. The
biome used to sit in spawn.BAD_BIOMES because `mine_wood` couldn't see bamboo
(bamboo ≠ log) and the craft chain had no bamboo path.

This test force-spawns into the (fixed-seed) bamboo jungle, then runs the whole
wood chain from bamboo:

    mine_wood (bamboo)  →  craft crafting_table  →  craft wooden_pickaxe

Spawn is forced via locate_biome (parses `/locate biome` from the relay log) +
random_spawn(require_biomes=…), an allow-list that supersedes BAD_BIOMES. The
world seed is fixed, so the located coord is deterministic; KNOWN_BAMBOO_JUNGLE
is the value observed on 2026-06-04 and used as a fallback if locate fails.

Determinism note: bamboo jungles keep SPARSE trees, so a raw spawn often has an
oak/jungle log nearer than the bamboo, and `mine_wood` would grab the log
instead of exercising the bamboo path. To pin the wood source to bamboo we plant
a short ring of bamboo columns ADJACENT to the player (nearer than any natural
tree) on the jungle's own grass — the inverse of e2e/test_mine_wood.py's planted
oak logs. The craft chain then runs on a bamboo-only inventory, which is what
actually exercises the new code (mine.LOG_TYPES bamboo candidate, the
bamboo→block→planks substitution, and the homunculus tag-aware canonicalItem
that lets bamboo_planks satisfy the #minecraft:planks recipe slots).

Per-iter pass criteria:
  - spawned into bamboo_jungle
  - mine_wood acquired bamboo (the cane is seen as wood)
  - reached --bamboo-target bamboo
  - craft(crafting_table) succeeded
  - craft(wooden_pickaxe) succeeded AND the pickaxe is in inventory

NOT in the default suite (mining + the craft chain is minutes, not seconds). Run
standalone, e.g.:
    HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 \
        uv run python -m e2e.test_bamboo_tech_tree --iters 1
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from pathlib import Path

from craft.testkit import (
    PLAYER_NAME,
    TestLogger,
    cmd,
    inventory,
    locate_biome,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.config import SERVER_CMD_BASE
from craft.tools import dispatch


BAMBOO_JUNGLE_BIOMES = frozenset({"bamboo_jungle"})
# Deterministic fallback (fixed seed): nearest bamboo_jungle from world origin,
# observed via `/locate biome` on 2026-06-04. locate_biome is tried first.
KNOWN_BAMBOO_JUNGLE = (-736, -1408)

# Enough bamboo for the full chain: crafting_table (4 planks) + wooden_pickaxe
# (3 planks + 2 sticks = 5 planks) = 9 planks. At 2 planks/block, 9 bamboo/block,
# that's 5 blocks = 45 bamboo; 54 gives a one-block margin against round-up.
DEFAULT_BAMBOO_TARGET = 54
DEFAULT_SPAWN_RANGE = 40
# mine_wood is capped at 10/call; allow a few extra rounds for partial hauls.
MAX_MINE_ROUNDS = 9
# Planted bamboo ring: 8 columns adjacent to the player, each this tall. 8×12=96
# bamboo, all at distance ~2 — nearer than the jungle's sparse trees, so the
# mine_wood probe ranks bamboo first and Baritone mines bamboo, not a log.
RING_COLUMNS = [(2, 0), (-2, 0), (0, 2), (0, -2),
                (2, 2), (2, -2), (-2, 2), (-2, -2)]
RING_HEIGHT = 12
# Generous Baritone budget: bamboo gathering + the recursive craft chain.
DEFAULT_TIMEOUT_S = 360.0
DEFAULT_PASS_RATE = 0.9

WOODEN_PICKAXE = "minecraft:wooden_pickaxe"
CRAFTING_TABLE = "minecraft:crafting_table"
BAMBOO = "minecraft:bamboo"


def _count_items(item_ids: set[str]) -> int:
    """Sum counts of `item_ids` across main + offhand. -1 on read error."""
    inv = inventory()
    if inv is None:
        return -1
    total = 0
    for slot in inv.get("main", []) or []:
        if slot.get("id") in item_ids:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") in item_ids:
        total += int(off.get("count", 0))
    return total


def _plant_bamboo_ring(anchor: tuple[int, int, int]) -> None:
    """Plant 8 bamboo columns around the anchor on the jungle's own ground.

    Each column gets a dirt base (guaranteed plantable support, so setblock'd
    bamboo doesn't pop off on the next block update) then RING_HEIGHT bamboo.
    All columns sit at distance ~2 — inside any natural tree — so mine_wood's
    distance-ranked probe selects bamboo as the wood candidate.
    """
    ax, ay, az = anchor
    for dx, dz in RING_COLUMNS:
        x, z = ax + dx, az + dz
        cmd(f"setblock {x} {ay - 1} {z} minecraft:dirt")
        for h in range(RING_HEIGHT):
            cmd(f"setblock {x} {ay + h} {z} minecraft:bamboo")
    time.sleep(0.4)


def _clear_bamboo_ring(anchor: tuple[int, int, int]) -> None:
    ax, ay, az = anchor
    for dx, dz in RING_COLUMNS:
        x, z = ax + dx, az + dz
        for h in range(-1, RING_HEIGHT):
            cmd(f"setblock {x} {ay + h} {z} minecraft:air")
    cmd("kill @e[type=item,distance=..32]")


def _spawn_into_bamboo(
    rec: dict, *, spawn_range: int, rng: _random.Random, verbose: bool
) -> bool:
    """Force-spawn into the bamboo jungle. Returns True on success."""
    xz = locate_biome("bamboo_jungle", server_cmd_base=SERVER_CMD_BASE)
    if xz is None:
        xz = KNOWN_BAMBOO_JUNGLE
        if verbose:
            print(f"[test] locate_biome failed; using fallback {xz}", flush=True)
    rec["bamboo_jungle_xz"] = list(xz)

    spawn_result = random_spawn(
        range_blocks=spawn_range,
        anchor_xz=xz,
        require_biomes=BAMBOO_JUNGLE_BIOMES,
        rng=rng,
        verbose=verbose,
    )
    rec["spawn"] = spawn_result
    if not spawn_result.get("ok"):
        rec["fail_reason"] = "spawn_not_bamboo_jungle"
        return False
    rec["spawn_biome"] = spawn_result.get("biome")
    return True


def run_iter(
    rec: dict,
    *,
    bamboo_target: int,
    timeout_s: float,
    spawn_range: int,
    rng: _random.Random,
    verbose: bool,
) -> None:
    """Run one bamboo-tech-tree iteration, populating `rec` in place."""
    rec["passed"] = False

    if not _spawn_into_bamboo(rec, spawn_range=spawn_range, rng=rng, verbose=verbose):
        return

    start = pos()
    if start is None:
        rec["fail_reason"] = "could_not_read_position"
        return
    anchor = (int(math.floor(start[0])),
              int(math.floor(start[1])),
              int(math.floor(start[2])))
    rec["anchor"] = list(anchor)

    # Clean, empty-handed, peaceful start in the natural bamboo, then plant the
    # adjacent bamboo ring so the wood source is deterministically bamboo.
    setup_clean(anchor)
    _plant_bamboo_ring(anchor)

    t0 = time.monotonic()

    # ---- Step 1: mine bamboo through mine_wood -----------------------------
    rounds: list[dict] = []
    bamboo = _count_items({BAMBOO})
    first_round_acquired = 0
    for r in range(MAX_MINE_ROUNDS):
        if bamboo >= bamboo_target:
            break
        try:
            out = dispatch("mine_wood", json.dumps({"quantity": 10}))
        except Exception as e:
            out = f"FAILED: dispatch threw {e!r}"
        after = _count_items({BAMBOO})
        gained = max(0, after - bamboo)
        if r == 0:
            first_round_acquired = gained
        rounds.append({"round": r, "outcome": out, "bamboo_after": after, "gained": gained})
        if verbose:
            print(f"[test] mine_wood round {r}: {out} (bamboo={after})", flush=True)
        if gained == 0 and out.startswith("FAILED"):
            break  # nothing reachable — original-bug signature; fail on target
        bamboo = after
    rec["mine_rounds"] = rounds
    rec["bamboo_mined"] = bamboo
    rec["first_round_acquired"] = first_round_acquired

    mine_saw_bamboo = first_round_acquired > 0

    if bamboo < bamboo_target:
        rec["mine_wall_s"] = round(time.monotonic() - t0, 2)
        rec["checks"] = {
            "spawned_bamboo_jungle": True,
            "mine_wood_saw_bamboo": mine_saw_bamboo,
            "bamboo_target_met": False,
        }
        rec["fail_reason"] = "insufficient_bamboo"
        _clear_bamboo_ring(anchor)
        cmd(f"clear {PLAYER_NAME}")
        return

    # ---- Step 2: craft a crafting_table from bamboo ------------------------
    try:
        table_out = dispatch("craft", json.dumps({"item": "crafting_table", "quantity": 1}))
    except Exception as e:
        table_out = f"FAILED: dispatch threw {e!r}"
    rec["craft_table_outcome"] = table_out
    if verbose:
        print(f"[test] craft crafting_table: {table_out}", flush=True)
    table_ok = not table_out.startswith("FAILED") and _count_items({CRAFTING_TABLE}) >= 1

    # ---- Step 3: craft a wooden_pickaxe from bamboo ------------------------
    try:
        pick_out = dispatch("craft", json.dumps({"item": "wooden_pickaxe", "quantity": 1}))
    except Exception as e:
        pick_out = f"FAILED: dispatch threw {e!r}"
    rec["craft_pickaxe_outcome"] = pick_out
    if verbose:
        print(f"[test] craft wooden_pickaxe: {pick_out}", flush=True)
    have_pickaxe = _count_items({WOODEN_PICKAXE}) >= 1

    elapsed = round(time.monotonic() - t0, 2)
    rec["mine_wall_s"] = elapsed
    rec["have_pickaxe"] = have_pickaxe

    _clear_bamboo_ring(anchor)
    cmd(f"clear {PLAYER_NAME}")

    within_budget = elapsed <= timeout_s
    pickaxe_ok = not pick_out.startswith("FAILED") and have_pickaxe
    rec["target"] = bamboo_target
    rec["checks"] = {
        "spawned_bamboo_jungle": True,
        "mine_wood_saw_bamboo": mine_saw_bamboo,
        "bamboo_target_met": bamboo >= bamboo_target,
        "crafting_table_crafted": table_ok,
        "wooden_pickaxe_crafted": pickaxe_ok,
        "within_timeout": within_budget,
    }
    rec["passed"] = (mine_saw_bamboo and table_ok and pickaxe_ok and within_budget)
    if not rec["passed"]:
        if not mine_saw_bamboo:
            rec["fail_reason"] = "mine_wood_missed_bamboo"
        elif not table_ok:
            rec["fail_reason"] = "crafting_table_failed"
        elif not pickaxe_ok:
            rec["fail_reason"] = "wooden_pickaxe_failed"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bamboo-target", type=int, default=DEFAULT_BAMBOO_TARGET,
                    help=f"bamboo to gather before crafting (default {DEFAULT_BAMBOO_TARGET})")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"per-iter wall-time budget in seconds (default {DEFAULT_TIMEOUT_S})")
    ap.add_argument("--iters", type=int, default=1,
                    help="iterations to run (default 1; >1 enables pass-rate estimation)")
    ap.add_argument("--spawn-range", type=int, default=DEFAULT_SPAWN_RANGE,
                    help=f"±range around the located bamboo jungle (default {DEFAULT_SPAWN_RANGE})")
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE,
                    help=f"exit 0 if iters_passed/iters >= this rate (default {DEFAULT_PASS_RATE})")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="JSONL output path (default: results/test-bamboo_tech_tree-<ts>.jsonl)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger("bamboo_tech_tree",
                        path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(rec, bamboo_target=args.bamboo_target,
                         timeout_s=args.timeout, spawn_range=args.spawn_range,
                         rng=rng, verbose=not args.quiet)
        except Exception as e:
            print(f"[test] iter {i} raised: {e!r}", flush=True)

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
