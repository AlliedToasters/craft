"""Multi-iter test of the smelt tool (fire-and-forget start).

smelt(input, count) loads + ignites a furnace asynchronously and returns
immediately with 'smelt started: Nx <ingot> in furnace at (x,y,z); ...'.
Test: /give raw_iron + coal + furnace, dispatch smelt, verify the outcome
parses as a started smelt at a furnace near the player.

This test deliberately does NOT wait for the cook — that's
test_collect_smelt's job. We just verify the synchronous prologue.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - outcome contains 'smelt started'
  - furnace coord is parseable and within 8 blocks of the player
  - wall_s <= timeout
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import re
import sys
import time
from pathlib import Path

from craft.testkit import (
    PLAYER_NAME,
    TestLogger,
    build_arena,
    cmd,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_INPUT = "raw_iron"
DEFAULT_COUNT = 2
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_PASS_RATE = 0.9


FURNACE_RE = re.compile(r"furnace at \((-?\d+),(-?\d+),(-?\d+)\)")


def _extract_furnace_pos(outcome: str) -> tuple[int, int, int] | None:
    m = FURNACE_RE.search(outcome)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def run_iter(
    rec: dict,
    *,
    input_item: str,
    count: int,
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
    rec["input_item"] = input_item
    rec["count"] = count

    setup_clean(anchor)
    build_arena(anchor, x_radius=6)
    # /give all the inputs so smelt has no recursive crafting work to do.
    cmd(f"give {PLAYER_NAME} minecraft:furnace 1")
    cmd(f"give {PLAYER_NAME} minecraft:{input_item} {count}")
    cmd(f"give {PLAYER_NAME} minecraft:coal {count}")
    time.sleep(0.4)

    after_setup = pos()
    if after_setup is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_setup_position"
        return
    sx, sy, sz = after_setup
    rec["start_pos"] = [sx, sy, sz]

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch(
            "smelt",
            json.dumps({"input": input_item, "count": count}),
        )
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["smelt_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    started = bool(outcome) and outcome.startswith("smelt started")
    furnace_pos = _extract_furnace_pos(outcome) if outcome else None
    rec["furnace_pos"] = list(furnace_pos) if furnace_pos else None

    near_player = False
    if furnace_pos is not None:
        fx, fy, fz = furnace_pos
        dist = max(abs(fx - sx), abs(fy - sy), abs(fz - sz))
        rec["chebyshev_to_player"] = round(dist, 2)
        near_player = dist <= 8.0

    # Cleanup: clear furnace + inventory. The smelt may still be cooking
    # internally — that's OK for the next iter since we /give fresh items.
    if furnace_pos is not None:
        fx, fy, fz = furnace_pos
        cmd(f"setblock {fx} {fy} {fz} minecraft:air")
    cmd(f"clear {PLAYER_NAME}")
    cmd("kill @e[type=item,distance=..32]")

    is_failed = outcome is None or outcome.startswith("FAILED")
    within_budget = elapsed <= timeout_s

    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "smelt_started": started,
        "furnace_pos_parsed": furnace_pos is not None,
        "furnace_near_player": near_player,
        "within_timeout": within_budget,
    }
    rec["passed"] = (
        (not is_failed)
        and started
        and furnace_pos is not None
        and near_player
        and within_budget
    )
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not started:
            rec["fail_reason"] = "smelt_not_started"
        elif furnace_pos is None:
            rec["fail_reason"] = "furnace_pos_not_parsed"
        elif not near_player:
            rec["fail_reason"] = "furnace_too_far_from_player"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
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
    logger = TestLogger("smelt", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    input_item=args.input,
                    count=args.count,
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
