"""Multi-iter test of the mine_stone tool.

mine_stone is FORCED blind-tunnel (no Baritone chunk-scan, no `fair` toggle).
The handler digs a 1×2 corridor forward in the player's facing direction at
the current y — intended for use underground after descend(). The test
mirrors the intended deployment: teleport the player to spawn_y - 10 (deep
enough for natural stone, shallow enough to keep iter wall time short),
force a stone-encased zone (covers natural caves/water/dirt), and dispatch.

Setup per iter:
  1. Compute (px, py, pz) from current pos; target_y = py - 10.
  2. /fill a stone-encased shell around (px, target_y, pz) so the tunneling
     zone is guaranteed stone regardless of natural terrain at this xz.
  3. Carve a 1×3 air pocket inside the shell so the player can stand.
  4. /tp player there with yaw=0 pitch=0 (faces +z deterministically).
  5. /give wooden_pickaxe.
  6. Dispatch mine_stone, verify cobblestone delta.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - acquired count >= target
  - wall_s <= timeout
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
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


STONE_DROP_IDS = {"minecraft:cobblestone", "minecraft:cobbled_deepslate"}

DEFAULT_TARGET = 2
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_PASS_RATE = 0.9


def _count_stone() -> int:
    inv = inventory()
    if inv is None:
        return -1
    total = 0
    for slot in inv.get("main", []) or []:
        if slot.get("id") in STONE_DROP_IDS:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") in STONE_DROP_IDS:
        total += int(off.get("count", 0))
    return total


UNDERGROUND_DEPTH = 10  # blocks below spawn y — plenty for natural stone


def _build_underground_chamber(
    px: int, pz: int, y: int
) -> tuple[tuple, tuple]:
    """Force a stone-encased chamber with a 1×3 air pocket for the player.

    The fill replaces ANY existing block (caves, water, dirt) so we don't
    depend on the natural terrain at this xz. The pocket is a single 1×3
    column at (px, pz); the surrounding 5×4×15 box ahead in +z gives the
    tunnel solid stone to dig through.

    Returns (lo, hi) of the stone region for cleanup.
    """
    # Stone shell — generous box that fully contains the tunnel zone (+z 1..14)
    # plus the player column and a buffer ring.
    lo = (px - 3, y - 1, pz - 2)
    hi = (px + 3, y + 3, pz + 15)
    cmd(f"fill {lo[0]} {lo[1]} {lo[2]} {hi[0]} {hi[1]} {hi[2]} minecraft:stone")
    # 1×3 air column for the player to stand in.
    cmd(f"fill {px} {y} {pz} {px} {y + 2} {pz} minecraft:air")
    time.sleep(0.3)
    return lo, hi


def _cleanup_chamber(lo: tuple, hi: tuple) -> None:
    cmd(f"fill {lo[0]} {lo[1]} {lo[2]} {hi[0]} {hi[1]} {hi[2]} minecraft:air")
    cmd("kill @e[type=item,distance=..32]")


def run_iter(
    rec: dict,
    *,
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
    px = int(math.floor(start[0]))
    py = int(math.floor(start[1]))
    pz = int(math.floor(start[2]))
    target_y = py - UNDERGROUND_DEPTH
    anchor = (px, target_y, pz)
    rec["anchor"] = list(anchor)
    rec["spawn_y"] = py
    if verbose:
        print(f"[test] anchor = {anchor} (spawn_y={py}, depth={UNDERGROUND_DEPTH})",
              flush=True)

    # Build the encased chamber FIRST (carves an air pocket and surrounds it
    # with stone), then setup_clean teleports the player into it. setup_clean
    # does creative→tp→survival, so suffocation is avoided during the trip.
    lo, hi = _build_underground_chamber(px, pz, target_y)
    rec["chamber_lo"] = list(lo)
    rec["chamber_hi"] = list(hi)

    setup_clean(anchor)

    # Pin facing direction so the tunnel runs deterministically in +z, into
    # the stone shell ahead of the player.
    cmd(f"tp {PLAYER_NAME} {px} {target_y} {pz} 0 0")
    time.sleep(0.5)

    cmd(f"give {PLAYER_NAME} minecraft:wooden_pickaxe 1")
    time.sleep(0.3)

    before = _count_stone()
    if before < 0:
        _cleanup_chamber(lo, hi)
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_pre_inventory"
        return
    rec["drops_before"] = before

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch("mine_stone", json.dumps({"quantity": target}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["mine_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    after = _count_stone()
    acquired = after - before if (after >= 0 and before >= 0) else None
    rec["drops_after"] = after
    rec["acquired"] = acquired
    if verbose:
        print(f"[test] acquired={acquired} target={target}", flush=True)

    _cleanup_chamber(lo, hi)
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
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
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
    logger = TestLogger(
        "mine_stone", path=Path(args.out) if args.out else None
    )

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
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
