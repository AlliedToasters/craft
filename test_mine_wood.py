"""Multi-iter test of the mine_wood tool.

The template for single-tool e2e tests. Per iteration:
  1. (Optional) biome-aware random TP — spawn diversity for failure-rate estimation.
  2. setup_clean — peaceful, heal, clear inventory, TP to anchor.
  3. build_arena — stone floor + air column (terrain-independent).
  4. Plant 4 oak_logs at the cardinals via /setblock.
  5. Dispatch mine_wood through craft.tools.dispatch (real handler path).
  6. Verify outcome string, inventory delta, wall-time budget.
  7. Cleanup + write one JSONL record.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - acquired count ≥ target
  - mine wall_s ≤ timeout

Overall suite verdict: pass-rate across iters ≥ --pass-rate.
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


# Default target. 2 keeps the test cheap; bumping it would mostly stretch
# Baritone wall-time without exercising additional code paths.
DEFAULT_TARGET = 2
# Generous Baritone budget: ~30s per log + slack.
DEFAULT_TIMEOUT_S = 90.0
# Multi-iter pass-rate threshold. mine_wood is one of the more deterministic
# tools (no mobs, no time pressure); we hold it to a high bar.
DEFAULT_PASS_RATE = 0.9


LOG_DROP_IDS = {
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
    "minecraft:mangrove_log", "minecraft:cherry_log", "minecraft:pale_oak_log",
    "minecraft:crimson_stem", "minecraft:warped_stem",
}


def _count_logs() -> int:
    """Sum of LOG_DROP_IDS counts across main inventory + offhand. -1 on error."""
    inv = inventory()
    if inv is None:
        return -1
    total = 0
    for slot in inv.get("main", []) or []:
        if slot.get("id") in LOG_DROP_IDS:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") in LOG_DROP_IDS:
        total += int(off.get("count", 0))
    return total


def _plant_logs(anchor: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Drop 4 oak_log blocks at the cardinals, 5 blocks out. Returns positions.

    Four targets (not 2) gives Baritone redundancy: even if one is unreachable
    the cumulative-target semantics still hits the small target count.
    """
    ax, ay, az = anchor
    planted = [
        (ax + 5, ay, az),
        (ax - 5, ay, az),
        (ax, ay, az + 5),
        (ax, ay, az - 5),
    ]
    for x, y, z in planted:
        cmd(f"setblock {x} {y} {z} minecraft:oak_log")
    time.sleep(0.3)
    return planted


def _cleanup_logs(planted: list[tuple[int, int, int]]) -> None:
    for x, y, z in planted:
        cmd(f"setblock {x} {y} {z} minecraft:air")
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
    """Run a single iteration and populate `rec` in place."""
    if spawn_range > 0:
        spawn_result = random_spawn(
            range_blocks=spawn_range, rng=rng, verbose=verbose)
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
    anchor = (int(math.floor(start[0])),
              int(math.floor(start[1])),
              int(math.floor(start[2])))
    rec["anchor"] = list(anchor)
    if verbose:
        print(f"[test] anchor = {anchor}", flush=True)

    setup_clean(anchor)
    build_arena(anchor, x_radius=8)
    planted = _plant_logs(anchor)
    rec["planted"] = [list(p) for p in planted]

    before = _count_logs()
    if before < 0:
        _cleanup_logs(planted)
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_pre_inventory"
        return
    rec["logs_before"] = before

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch("mine_wood", json.dumps({"quantity": target}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["mine_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    after = _count_logs()
    acquired = after - before if (after >= 0 and before >= 0) else None
    rec["logs_after"] = after
    rec["acquired"] = acquired
    if verbose:
        print(f"[test] acquired={acquired} target={target}", flush=True)

    _cleanup_logs(planted)
    cmd(f"clear {PLAYER_NAME}")

    is_failed = outcome is None or outcome.startswith("FAILED")
    meets_target = (acquired is not None and acquired >= target)
    within_budget = (elapsed <= timeout_s)

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
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET,
                    help=f"log count target per iter (default {DEFAULT_TARGET})")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"per-iter wall-time budget in seconds (default {DEFAULT_TIMEOUT_S})")
    ap.add_argument("--iters", type=int, default=1,
                    help="iterations to run (default 1; >1 enables pass-rate estimation)")
    ap.add_argument("--spawn-range", type=int, default=0,
                    help="if >0, biome-aware random TP within ±range each iter")
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE,
                    help=f"exit 0 if iters_passed/iters >= this rate (default {DEFAULT_PASS_RATE})")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="JSONL output path (default: results/test-mine_wood-<ts>.jsonl)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger("mine_wood",
                        path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(rec, target=args.target, timeout_s=args.timeout,
                         spawn_range=args.spawn_range, rng=rng,
                         verbose=not args.quiet)
        except Exception as e:
            # TestLogger.__exit__ already recorded fatal_error; just keep going.
            print(f"[test] iter {i} raised: {e!r}", flush=True)

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
