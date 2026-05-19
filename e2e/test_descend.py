"""Multi-iter test of the descend tool.

descend(target_y) digs straight down through stone to a target Y level.
Setup: stand the player on natural surface terrain, request descend to
spawn_y - 15, verify the post-descend Y is within 2 blocks of the target.

The descent goes through whatever natural blocks are below (dirt, stone,
gravel) — no artificial terrain prep. A wooden pickaxe in inventory is
sufficient for the dirt/stone band at spawn_y..spawn_y-15.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - |new_y - target_y| <= 2
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
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_DEPTH = 15  # blocks below current y — well under DESCEND_MAX_PER_CALL=40
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_PASS_RATE = 0.9


def run_iter(
    rec: dict,
    *,
    depth: int,
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

    setup_clean(anchor)

    # Re-read pos after setup_clean — that's our reference y for the descent.
    after_setup = pos()
    if after_setup is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_setup_position"
        return
    sx, sy, sz = after_setup
    target_y = int(math.floor(sy)) - depth
    rec["start_y"] = sy
    rec["target_y"] = target_y
    if verbose:
        print(f"[test] start_y={sy}, target_y={target_y}", flush=True)

    cmd(f"give {PLAYER_NAME} minecraft:stone_pickaxe 1")
    time.sleep(0.3)

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch("descend", json.dumps({"target_y": target_y}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["descend_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    end = pos()
    if end is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_position"
        return
    ex, ey, ez = end
    rec["end_pos"] = [ex, ey, ez]
    delta_y = ey - target_y
    rec["delta_y"] = round(delta_y, 2)

    is_failed = outcome is None or outcome.startswith("FAILED")
    reached_target = abs(delta_y) <= 2
    within_budget = elapsed <= timeout_s

    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "reached_target_y": reached_target,
        "within_timeout": within_budget,
    }
    rec["passed"] = (not is_failed) and reached_target and within_budget
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not reached_target:
            rec["fail_reason"] = "did_not_reach_target_y"
        else:
            rec["fail_reason"] = "timeout"

    cmd(f"clear {PLAYER_NAME}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
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
    logger = TestLogger("descend", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    depth=args.depth,
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
