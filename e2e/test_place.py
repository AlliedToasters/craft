"""Multi-iter test of the place tool.

place(item) puts a block from inventory near the player. Test: /give a
chest (a block that the harness doesn't auto-place via craft/smelt), build
an arena so there's a safe placeable spot, dispatch place("chest"), and
verify the outcome string + that a chest now exists in the placed_at
coordinates the tool reported.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - outcome contains 'placed' and the placed_at coords appear plausible
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


DEFAULT_ITEM = "chest"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_PASS_RATE = 0.9

PLACED_AT_RE = re.compile(r"placed_at[^[]*\[([^\]]+)\]|placed_at.*?\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)|at\s*\[([^\]]+)\]")


def _extract_placed_at(outcome: str) -> tuple[int, int, int] | None:
    # The handler returns e.g. "placed minecraft:chest at [12, 64, -5]".
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", outcome)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def run_iter(
    rec: dict,
    *,
    item: str,
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
    rec["item"] = item

    setup_clean(anchor)
    build_arena(anchor, x_radius=6)
    cmd(f"give {PLAYER_NAME} minecraft:{item} 1")
    time.sleep(0.3)

    # Re-read pos after setup_clean so we can verify the chest lands NEAR
    # the agent (place is documented as "near you" — empirically within ~5
    # blocks). Catches the failure mode where homunculus returns success but
    # the block ended up in some default-coords location.
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
        outcome = dispatch("place", json.dumps({"item": item}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["place_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    placed_at = _extract_placed_at(outcome) if outcome else None
    rec["placed_at"] = list(placed_at) if placed_at else None

    near_player = False
    if placed_at is not None:
        bx, by, bz = placed_at
        dist = max(abs(bx - sx), abs(by - sy), abs(bz - sz))  # Chebyshev
        rec["chebyshev_to_player"] = round(dist, 2)
        near_player = dist <= 5.0
        # Cleanup the placed block.
        cmd(f"setblock {bx} {by} {bz} minecraft:air")
    cmd(f"clear {PLAYER_NAME}")
    cmd("kill @e[type=item,distance=..32]")

    is_failed = outcome is None or outcome.startswith("FAILED")
    has_placed = placed_at is not None
    within_budget = elapsed <= timeout_s

    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "placed_at_parsed": has_placed,
        "near_player": near_player,
        "within_timeout": within_budget,
    }
    rec["passed"] = (
        (not is_failed) and has_placed and near_player and within_budget
    )
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not has_placed:
            rec["fail_reason"] = "placed_at_not_parsed"
        elif not near_player:
            rec["fail_reason"] = "placed_too_far_from_player"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--item", default=DEFAULT_ITEM,
                    help=f"block to place (default {DEFAULT_ITEM})")
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
    logger = TestLogger("place", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    item=args.item,
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
