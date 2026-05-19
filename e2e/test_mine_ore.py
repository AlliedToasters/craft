"""Multi-iter test of mine_iron / mine_diamond / mine_coal.

Parameterized by --species. All three share the cardinal-plant pattern with
mine_wood: build an arena, drop 4 ore blocks at the cardinals, dispatch the
tool, verify inventory delta. The only per-species variables are the source
block, drop set, and required pickaxe tier.

Pass criteria per iter (mirrors test_mine_wood):
  - outcome doesn't start with FAILED
  - acquired count >= target
  - wall_s <= timeout

Overall verdict: pass-rate across iters >= --pass-rate.
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
    build_arena,
    cmd,
    inventory,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


# (source_block, drop_ids, required_pickaxe, tool_name)
SPECIES: dict[str, dict] = {
    "iron": {
        "source": "minecraft:iron_ore",
        "drops": {"minecraft:raw_iron"},
        "pickaxe": "minecraft:stone_pickaxe",
        "tool": "mine_iron",
    },
    "diamond": {
        "source": "minecraft:diamond_ore",
        "drops": {"minecraft:diamond"},
        "pickaxe": "minecraft:iron_pickaxe",
        "tool": "mine_diamond",
    },
    "coal": {
        "source": "minecraft:coal_ore",
        "drops": {"minecraft:coal"},
        "pickaxe": "minecraft:wooden_pickaxe",
        "tool": "mine_coal",
    },
}

DEFAULT_TARGET = 2
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_PASS_RATE = 0.9


def _count_drops(drop_ids: set[str]) -> int:
    inv = inventory()
    if inv is None:
        return -1
    total = 0
    for slot in inv.get("main", []) or []:
        if slot.get("id") in drop_ids:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") in drop_ids:
        total += int(off.get("count", 0))
    return total


def _plant_ores(
    anchor: tuple[int, int, int], source_block: str
) -> list[tuple[int, int, int]]:
    """Drop 4 source blocks at the cardinals, 5 blocks out."""
    ax, ay, az = anchor
    planted = [
        (ax + 5, ay, az),
        (ax - 5, ay, az),
        (ax, ay, az + 5),
        (ax, ay, az - 5),
    ]
    for x, y, z in planted:
        cmd(f"setblock {x} {y} {z} {source_block}")
    time.sleep(0.3)
    return planted


def _cleanup_blocks(planted: list[tuple[int, int, int]]) -> None:
    for x, y, z in planted:
        cmd(f"setblock {x} {y} {z} minecraft:air")
    cmd("kill @e[type=item,distance=..32]")


def run_iter(
    rec: dict,
    *,
    species: dict,
    target: int,
    timeout_s: float,
    spawn_range: int,
    rng: _random.Random,
    verbose: bool,
) -> None:
    if spawn_range > 0:
        spawn_result = random_spawn(
            range_blocks=spawn_range, rng=rng, verbose=verbose
        )
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
    rec["species"] = species["tool"]
    if verbose:
        print(f"[test] anchor = {anchor}, tool = {species['tool']}", flush=True)

    setup_clean(anchor)
    build_arena(anchor, x_radius=8)

    # Give the required pickaxe after setup_clean's clear+tp.
    cmd(f"give {PLAYER_NAME} {species['pickaxe']} 1")
    time.sleep(0.3)

    planted = _plant_ores(anchor, species["source"])
    rec["planted"] = [list(p) for p in planted]

    before = _count_drops(species["drops"])
    if before < 0:
        _cleanup_blocks(planted)
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_pre_inventory"
        return
    rec["drops_before"] = before

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch(species["tool"], json.dumps({"quantity": target}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["mine_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    after = _count_drops(species["drops"])
    acquired = after - before if (after >= 0 and before >= 0) else None
    rec["drops_after"] = after
    rec["acquired"] = acquired
    if verbose:
        print(f"[test] acquired={acquired} target={target}", flush=True)

    _cleanup_blocks(planted)
    cmd(f"clear {PLAYER_NAME}")

    is_failed = outcome is None or outcome.startswith("FAILED")
    meets_target = acquired is not None and acquired >= target
    within_budget = elapsed <= timeout_s

    rec["target"] = target
    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "acquired_meets_target": meets_target,
        "within_timeout": within_budget,
    }
    rec["passed"] = (not is_failed) and meets_target and within_budget
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not meets_target:
            rec["fail_reason"] = "target_not_met"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--species", choices=sorted(SPECIES.keys()), required=True,
        help="which ore tool to test",
    )
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--spawn-range", type=int, default=0)
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    species = SPECIES[args.species]

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger(
        species["tool"],
        path=Path(args.out) if args.out else None,
    )

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    species=species,
                    target=args.target,
                    timeout_s=args.timeout,
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
