"""Multi-iter test of the collect_smelt tool.

collect_smelt() walks the player to a ready furnace and pulls ingots into
inventory. Test plan:
  1. /give furnace + raw_iron + coal.
  2. Dispatch smelt(raw_iron, 1) — count=1 keeps the cook quick (~10s).
  3. Wait ~13s for the cook to finish (10s/item + slack).
  4. Dispatch collect_smelt(), parse the outcome, verify iron_ingot delta.

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - iron_ingot count increased by >= count
  - wall_s <= timeout (covers smelt + sleep + collect)
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
    inventory,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_COUNT = 1
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_PASS_RATE = 0.9

# Per-item smelt cook time in seconds (vanilla furnace = 10s/item).
COOK_SECONDS_PER_ITEM = 10.0
SMELT_SLACK_S = 5.0


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


FURNACE_RE = re.compile(r"furnace at \((-?\d+),(-?\d+),(-?\d+)\)")


def run_iter(
    rec: dict,
    *,
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
    rec["count"] = count

    setup_clean(anchor)
    build_arena(anchor, x_radius=6)
    cmd(f"give {PLAYER_NAME} minecraft:furnace 1")
    cmd(f"give {PLAYER_NAME} minecraft:raw_iron {count}")
    cmd(f"give {PLAYER_NAME} minecraft:coal {count}")
    time.sleep(0.4)

    # Phase 1 — start smelt.
    smelt_outcome: str | None = None
    try:
        smelt_outcome = dispatch(
            "smelt",
            json.dumps({"input": "raw_iron", "count": count}),
        )
    except Exception as e:
        smelt_outcome = f"FAILED: dispatch threw {e!r}"
    rec["smelt_outcome"] = smelt_outcome

    if not smelt_outcome or not smelt_outcome.startswith("smelt started"):
        rec["passed"] = False
        rec["fail_reason"] = "smelt_did_not_start"
        cmd(f"clear {PLAYER_NAME}")
        return

    m = FURNACE_RE.search(smelt_outcome)
    furnace_pos = (
        (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    )
    rec["furnace_pos"] = list(furnace_pos) if furnace_pos else None

    # Phase 2 — wait for the cook.
    wait_s = COOK_SECONDS_PER_ITEM * count + SMELT_SLACK_S
    if verbose:
        print(f"[test] waiting {wait_s:.0f}s for smelt to finish...", flush=True)
    time.sleep(wait_s)

    # Phase 3 — collect.
    before = _count_item("minecraft:iron_ingot")
    if before < 0:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_pre_inventory"
        cmd(f"clear {PLAYER_NAME}")
        return
    rec["ingots_before"] = before

    t0 = time.monotonic()
    collect_outcome: str | None = None
    try:
        collect_outcome = dispatch("collect_smelt", json.dumps({}))
    except Exception as e:
        collect_outcome = f"FAILED: dispatch threw {e!r}"
    collect_elapsed = round(time.monotonic() - t0, 2)
    rec["collect_outcome"] = collect_outcome
    rec["collect_wall_s"] = collect_elapsed
    total_elapsed = round(wait_s + collect_elapsed, 2)
    rec["total_wall_s"] = total_elapsed
    if verbose:
        print(f"[test] collect outcome ({collect_elapsed}s): {collect_outcome}",
              flush=True)

    after = _count_item("minecraft:iron_ingot")
    acquired = after - before if (after >= 0 and before >= 0) else None
    rec["ingots_after"] = after
    rec["acquired"] = acquired

    # Cleanup: clear furnace + inventory.
    if furnace_pos is not None:
        fx, fy, fz = furnace_pos
        cmd(f"setblock {fx} {fy} {fz} minecraft:air")
    cmd(f"clear {PLAYER_NAME}")
    cmd("kill @e[type=item,distance=..32]")

    is_failed = (
        collect_outcome is None or collect_outcome.startswith("FAILED")
    )
    meets_target = acquired is not None and acquired >= count
    within_budget = total_elapsed <= timeout_s

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
    logger = TestLogger(
        "collect_smelt", path=Path(args.out) if args.out else None
    )

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
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
