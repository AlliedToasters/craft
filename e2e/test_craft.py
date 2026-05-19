"""Multi-iter test of the craft tool.

craft(item, quantity) recursively crafts an item, auto-placing a
crafting_table when needed. Test path: /give 4 oak_log, dispatch
craft("oak_planks", 16), verify the planks count delta + outcome string.
oak_planks is a 1×1 inventory recipe so no table placement is required —
keeps the test deterministic.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - planks delta >= 16
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
    build_arena,
    cmd,
    inventory,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_TARGET = 16
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_PASS_RATE = 0.9


def _count_item(item_id: str) -> int:
    inv = inventory()
    if inv is None:
        return -1
    total = 0
    for slot in inv.get("main", []) or []:
        if slot.get("id") == item_id:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") == item_id:
        total += int(off.get("count", 0))
    return total


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
    anchor = (
        int(math.floor(start[0])),
        int(math.floor(start[1])),
        int(math.floor(start[2])),
    )
    rec["anchor"] = list(anchor)

    setup_clean(anchor)
    build_arena(anchor, x_radius=6)
    # 4 oak_log → 16 oak_planks (1:4 species recipe, no table needed).
    logs_needed = (target + 3) // 4  # ceil(target / 4)
    cmd(f"give {PLAYER_NAME} minecraft:oak_log {logs_needed}")
    time.sleep(0.3)

    before = _count_item("minecraft:oak_planks")
    if before < 0:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_pre_inventory"
        return
    rec["planks_before"] = before

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch(
            "craft",
            json.dumps({"item": "oak_planks", "quantity": target}),
        )
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["craft_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    after = _count_item("minecraft:oak_planks")
    acquired = after - before if (after >= 0 and before >= 0) else None
    rec["planks_after"] = after
    rec["acquired"] = acquired

    cmd(f"clear {PLAYER_NAME}")
    cmd("kill @e[type=item,distance=..32]")

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
    logger = TestLogger("craft", path=Path(args.out) if args.out else None)

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
