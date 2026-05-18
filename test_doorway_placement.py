"""Repro: /place lands a block in the shelter doorway path.

Sets up a synthetic minimal scenario — a 7x7 stone platform with a
single oak_door three cells north of center — then teleports the
player to the center and calls POST /place. The shelter bug:
homunculus's auto-placement picks the cavity cell directly inside the
doorway (player_z - 2) because the existing doorway guard's scan box
(`Placer.java:272-286`, dx/dz ∈ [-2,+2]) is too narrow to catch a door
at distance 3.

PASS if `placed_at` is NOT in the door's 5-cell egress footprint
(door cell + 4 cardinal neighbors at door y), or if /place returns
`reason=blocks_doorway` (the guard correctly refused).
FAIL if the block lands in the doorway path.

Run (against agent0):
  HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 \\
  MC_SERVER_CMD_BASE=http://10.0.0.222:4747 \\
  .venv/bin/python test_doorway_placement.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

from craft.testkit import (
    HOMUNCULUS_BASE,
    PLAYER_NAME,
    cmd,
    pos,
    set_gamemode,
)


ARENA_X = 5000
ARENA_Y = 100
ARENA_Z = 5000
DOOR_DZ = -3  # door 3 cells north of center


def setup_arena() -> None:
    cmd(f"fill {ARENA_X - 3} {ARENA_Y} {ARENA_Z - 3} "
        f"{ARENA_X + 3} {ARENA_Y + 2} {ARENA_Z + 3} air")
    cmd(f"fill {ARENA_X - 3} {ARENA_Y - 1} {ARENA_Z - 3} "
        f"{ARENA_X + 3} {ARENA_Y - 1} {ARENA_Z + 3} stone")
    cmd(f"setblock {ARENA_X} {ARENA_Y} {ARENA_Z + DOOR_DZ} "
        f"oak_door[half=lower,facing=south,hinge=left]")
    cmd(f"setblock {ARENA_X} {ARENA_Y + 1} {ARENA_Z + DOOR_DZ} "
        f"oak_door[half=upper,facing=south,hinge=left]")


def teardown_arena() -> None:
    cmd(f"fill {ARENA_X - 3} {ARENA_Y - 1} {ARENA_Z - 3} "
        f"{ARENA_X + 3} {ARENA_Y + 2} {ARENA_Z + 3} air")


def place_block(item: str) -> dict:
    r = requests.post(f"{HOMUNCULUS_BASE}/place",
                      json={"item": item}, timeout=15.0)
    return r.json()


def _doorway_cells() -> set[tuple[int, int, int]]:
    dx = ARENA_X
    dy = ARENA_Y
    dz = ARENA_Z + DOOR_DZ
    return {
        (dx, dy, dz),       # door bottom
        (dx, dy + 1, dz),   # door upper half
        (dx, dy, dz - 1),   # outside, in line
        (dx, dy, dz + 1),   # INSIDE walkway — the one the bug hits
        (dx - 1, dy, dz),   # side wall west
        (dx + 1, dy, dz),   # side wall east
    }


def run_iter(verbose: bool = True) -> dict:
    # TP the player first so the destination chunks load; only then do
    # /fill operations actually touch blocks. /fill on an unloaded chunk
    # silently no-ops.
    set_gamemode("creative")
    cmd(f"tp {PLAYER_NAME} {ARENA_X}.5 {ARENA_Y + 5} {ARENA_Z}.5")
    time.sleep(1.5)
    setup_arena()
    time.sleep(0.8)
    # TP again now that the platform exists, then wait until the player
    # is on_ground at the floor (any small drift in y is fine).
    cmd(f"tp {PLAYER_NAME} {ARENA_X}.5 {ARENA_Y} {ARENA_Z}.5")
    on_ground = False
    for _ in range(20):
        time.sleep(0.3)
        p = pos()
        if p is None:
            continue
        if (abs(p[0] - (ARENA_X + 0.5)) < 1.5
                and abs(p[2] - (ARENA_Z + 0.5)) < 1.5
                and abs(p[1] - ARENA_Y) < 0.5):
            on_ground = True
            break
    if not on_ground:
        return {"passed": False,
                "fail_reason": f"player did not land on platform pos={pos()}"}
    set_gamemode("survival")
    cmd(f"clear {PLAYER_NAME}")
    cmd(f"give {PLAYER_NAME} crafting_table 1")
    time.sleep(0.5)

    p = pos()
    if p is None:
        return {"passed": False, "fail_reason": "could_not_read_position"}
    if abs(p[0] - ARENA_X) > 2 or abs(p[2] - ARENA_Z) > 2:
        return {"passed": False,
                "fail_reason": f"player_not_at_center pos={p}"}

    resp = place_block("minecraft:crafting_table")
    if verbose:
        print(f"[place] response: {json.dumps(resp)}")

    rec: dict = {
        "player_pos": list(p),
        "door_pos": [ARENA_X, ARENA_Y, ARENA_Z + DOOR_DZ],
        "place_response": resp,
    }

    if resp.get("success"):
        placed_at = tuple(resp["placed_at"])
        in_doorway = placed_at in _doorway_cells()
        rec["passed"] = not in_doorway
        rec["fail_reason"] = (
            f"placed at {list(placed_at)} blocks doorway "
            f"footprint of door at ({ARENA_X},{ARENA_Y},{ARENA_Z + DOOR_DZ})"
            if in_doorway else None
        )
        rec["in_doorway"] = in_doorway
        return rec

    # Place failed. If the guard refused with blocks_doorway, that's the
    # post-fix expected behavior — PASS. Any other reason is unclear
    # (no_space, no_placeable_spot, internal_error) and FAILs the test
    # so the operator notices.
    reason = resp.get("reason", "?")
    if reason == "blocks_doorway":
        rec["passed"] = True
        rec["fail_reason"] = None
        rec["guard_fired"] = True
    else:
        rec["passed"] = False
        rec["fail_reason"] = f"place_failed reason={reason}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep-arena", action="store_true",
                    help="skip teardown (useful for inspecting in-game)")
    args = ap.parse_args()

    verbose = not args.quiet
    results: list[dict] = []
    try:
        for i in range(args.iters):
            print(f"\n=== iter {i + 1}/{args.iters} (player={PLAYER_NAME}, "
                  f"homunculus={HOMUNCULUS_BASE}) ===")
            rec = run_iter(verbose=verbose)
            results.append(rec)
            tag = "PASS" if rec.get("passed") else "FAIL"
            reason = rec.get("fail_reason") or "ok"
            print(f"[{tag}] iter {i + 1}: {reason}")
    finally:
        if not args.keep_arena:
            teardown_arena()

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
