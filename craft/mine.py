"""Candidate-cycling mining via homunculus /baritone/mine.

`#mine` takes a single block id; this layer iterates candidate variants
(oak_log → birch_log, iron_ore → deepslate_iron_ore, …) until one reports
`have_target` or `already_satisfied`. Quantity is a cumulative inventory
target (Baritone's own semantics); the delta wrapper lives in tools.py.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

import requests


from craft.config import HOMUNCULUS_BASE  # noqa: F401

LOG_TYPES = [
    "oak_log",
    "birch_log",
    "spruce_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
    "mangrove_log",
    "cherry_log",
    "pale_oak_log",
    "crimson_stem",
    "warped_stem",
]

# Candidates whose drops feed stone-tier crafting. stone→cobblestone and
# deepslate→cobbled_deepslate (with wood+ pickaxe); cobblestone and
# cobbled_deepslate drop themselves. Granite/diorite/andesite drop themselves
# but don't satisfy stone_pickaxe ingredients, so they're excluded.
STONE_TYPES = [
    "stone",
    "deepslate",
    "cobblestone",
    "cobbled_deepslate",
]

# Iron ore variants. Both ores drop raw_iron with a stone+ pickaxe.
# raw_iron_block is a (rare) shortcut for compactly-stored iron.
IRON_TYPES = [
    "iron_ore",
    "deepslate_iron_ore",
    "raw_iron_block",
]

# Diamond ore variants. Both drop 1x diamond with an iron-tier+ pickaxe.
# deepslate_diamond_ore dominates (densest around Y=-58 to -64).
DIAMOND_TYPES = [
    "deepslate_diamond_ore",
    "diamond_ore",
]

# Coal ore variants. Both drop 1x coal with a wooden+ pickaxe. Coal is the
# tier-appropriate smelt fuel (1 coal = 8 smelts) — far better than burning
# planks/logs (1.5 smelts each).
COAL_TYPES = [
    "coal_ore",
    "deepslate_coal_ore",
]

# Reasons that mean "this candidate isn't here; try the next one".
# Everything else (interrupted, timeout, busy, transport, …) stops the cycle.
# "no_progress" = homunculus's mine watchdog tripped: the target is present in
# loaded chunks (so scan_nearest ranked it) but Baritone can't path to it and
# was oscillating — bail fast and try the next species instead of burning the
# full per-candidate deadline.
SKIP_REASONS = {"unreachable", "never_started", "no_progress"}
SUCCESS_REASONS = {"have_target", "already_satisfied"}

# Detail of the most recent _mine_first_reachable cycle-stop (a non-skip,
# non-success reason that halted the candidate cycle). Lets the tool-layer
# handler surface specific guidance — e.g. homunculus's `no_effective_tool`
# (issue #11, pickaxe missing/broke mid-mine) → "craft a pickaxe" — instead of
# a generic "no candidate reachable". Process-local: fleet agents run as
# separate processes, so there's no cross-agent race on this module global.
last_stop: dict | None = None

# Total wall-clock budget per candidate. Homunculus splits this into a short
# start window (~15s) and the remainder for the actual mine; default 45s
# matches the old chat-scrape budget of 15s start + 30s mining extension.
PER_CANDIDATE_TIMEOUT = 45


def _mine_one(block: str, count: int, *, timeout_seconds: int = PER_CANDIDATE_TIMEOUT) -> dict:
    resp = requests.post(
        f"{HOMUNCULUS_BASE}/baritone/mine",
        json={"block": block, "count": count, "timeout_seconds": timeout_seconds},
        timeout=timeout_seconds + 10,
    )
    resp.raise_for_status()
    return resp.json()


def _scan_nearest(candidates: list[str], radius: int, y_radius: int) -> dict[str, dict | None] | None:
    """Probe homunculus for the nearest instance of each candidate block.

    Returns a dict mapping `minecraft:<id>` → {"x","y","z","distance"} or None
    for absent ids. Returns None on transport/server failure so callers can
    fall back to the un-probed cycle.
    """
    ids = ",".join(f"minecraft:{c}" for c in candidates)
    try:
        resp = requests.get(
            f"{HOMUNCULUS_BASE}/scan_nearest",
            params={"ids": ids, "radius": radius, "y_radius": y_radius},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"  → probe transport_error: {e}, skipping probe")
        return None
    if resp.status_code == 404:
        # Older homunculus build without /scan_nearest. Caller falls back.
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if data.get("success") is False:
        print(f"  → probe failed: {data.get('reason')} ({data.get('message')}), skipping probe")
        return None
    return data.get("matches") or {}


def _mine_first_reachable(
    quantity: int,
    candidates: list[str],
    *,
    probe_radius: int = 32,
    probe_y_radius: int = 16,
) -> str | None:
    """Try /baritone/mine for each candidate until one succeeds.

    Pre-flights with /scan_nearest to drop candidate species absent from the
    surrounding box and to reorder by ascending distance. The reorder saves
    ~PER_CANDIDATE_TIMEOUT per absent species — the dominant cost in the old
    mine_wood wild-goose chase. On any probe failure we fall back to the
    untouched list.

    `quantity` is the cumulative inventory target (matches Baritone's
    `mine(int, ...)` semantics). Returns the block id that produced a
    success reason, or None if every candidate was unreachable / refused.
    On a None return caused by a hard stop (not just "nothing here"), the
    stop reason+message is recorded in the module-level `last_stop`.
    """
    global last_stop
    last_stop = None
    matches = _scan_nearest(candidates, probe_radius, probe_y_radius)
    if matches is not None:
        ranked: list[tuple[str, float]] = []
        for c in candidates:
            m = matches.get(f"minecraft:{c}")
            if m is None:
                continue
            ranked.append((c, float(m.get("distance", 0.0))))
        if not ranked:
            print(
                f"  → probe: no {candidates} within radius={probe_radius}, "
                f"y_radius={probe_y_radius}, returning None",
                flush=True,
            )
            return None
        ranked.sort(key=lambda x: x[1])
        candidates = [c for c, _ in ranked]
        print(
            f"  → probe: {len(ranked)}/{len(matches)} present, ordered "
            f"{[(c, round(d, 1)) for c, d in ranked]}",
            flush=True,
        )

    for block in candidates:
        print(f"trying {block} (target={quantity})...", flush=True)
        try:
            data = _mine_one(block, quantity)
        except requests.RequestException as e:
            last_stop = {"reason": "transport_error", "message": str(e)}
            print(f"  → transport_error: {e}, stopping cycle")
            return None
        reason = data.get("reason", "unknown")
        if data.get("success") and reason in SUCCESS_REASONS:
            print(f"  → success ({block}, {reason})")
            return block
        if reason in SKIP_REASONS:
            print(f"  → {reason}, trying next")
            continue
        # interrupted / timeout / busy / baritone_not_loaded / unknown_block /
        # no_effective_tool / internal_error — a hard stop. Record it so the
        # tool-layer can surface specific guidance (#11 no_effective_tool).
        msg = data.get("message", "")
        last_stop = {"reason": reason, "message": msg}
        print(f"  → {reason} ({msg}), stopping cycle")
        return None
    return None


# Salvage wood (issue #7). On wood-deficient spawns (deep caves, mineshafts,
# post-flood, snowy peaks) there are no surface logs, but worked wood exists in
# structures — village walls/roofs, mineshaft platforms. Only *_planks salvage
# CLEANLY: planks ARE a usable crafting input (they drop themselves and carry
# the #minecraft:planks tag, so crafting_table/sticks/pickaxe work directly,
# skipping the log→plank step). Doors / fences / slabs / stairs are deliberately
# EXCLUDED — they drop themselves with no recipe back to planks, so mining one
# yields a dead item, not a wood input. Logs (the primary set) already catch
# village log-pillars; salvage is the plank-only fallback for plank structures.
SALVAGE_WOOD_TYPES = [
    "oak_planks",
    "birch_planks",
    "spruce_planks",
    "jungle_planks",
    "acacia_planks",
    "dark_oak_planks",
    "mangrove_planks",
    "cherry_planks",
    "pale_oak_planks",
    "bamboo_planks",
    "crimson_planks",
    "warped_planks",
]


def mine_any_log(quantity: int = 1) -> str | None:
    # Trees are surface canopy — wide horizontal, narrow vertical band.
    return _mine_first_reachable(quantity, LOG_TYPES, probe_radius=64, probe_y_radius=8)


def mine_any_salvage_wood(quantity: int = 1) -> str | None:
    # Structure planks: village walls/roofs, mineshaft platforms. Wide
    # horizontal AND deep vertical band — a mineshaft sits y20-50 below a cave
    # spawn, so y_radius is generous unlike the surface-canopy log probe.
    return _mine_first_reachable(
        quantity, SALVAGE_WOOD_TYPES, probe_radius=64, probe_y_radius=32
    )


def mine_any_stone(quantity: int = 1) -> str | None:
    return _mine_first_reachable(quantity, STONE_TYPES, probe_radius=32, probe_y_radius=16)


def mine_any_iron(quantity: int = 1) -> str | None:
    return _mine_first_reachable(quantity, IRON_TYPES, probe_radius=32, probe_y_radius=32)


def mine_any_diamond(quantity: int = 1) -> str | None:
    return _mine_first_reachable(quantity, DIAMOND_TYPES, probe_radius=32, probe_y_radius=32)


def mine_any_coal(quantity: int = 1) -> str | None:
    return _mine_first_reachable(quantity, COAL_TYPES, probe_radius=32, probe_y_radius=32)


# ──────────────────────── "Fair" / blind tunnel mining ───────────────────────
# Baritone's /mine scans loaded chunks for the target block and routes there.
# For abundant materials (stone), this means the agent picks an arbitrary
# clump deep underground and tunnels straight down, gathering desired stone
# incidentally on the way. Costly on vertical travel and leaves the agent
# pillared in a pit. The "fair" mode below replaces that with blind horizontal
# tunneling at the agent's current y level: a single recyclable primitive
# (tunnel_for) usable for any drop set, gated on inventory delta + timeout.

_DIR_VEC: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


def _yaw_to_direction(yaw: float) -> str:
    """Snap MC yaw to nearest cardinal. MC: 0=S, 90=W, 180=N, 270/-90=E."""
    y = yaw % 360.0
    if y < 45.0 or y >= 315.0:
        return "south"
    if y < 135.0:
        return "west"
    if y < 225.0:
        return "north"
    return "east"


def _count_drops(drops: Iterable[str]) -> int | None:
    """Sum inventory counts of any items whose id is in `drops`."""
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/inventory", timeout=5.0)
        resp.raise_for_status()
        inv = resp.json()
    except (requests.RequestException, ValueError):
        return None
    drops_set = set(drops)
    total = 0
    for slot in inv.get("main", []):
        if slot.get("id") in drops_set:
            total += int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id") in drops_set:
        total += int(off.get("count", 0))
    return total


def _excavate_box(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
    *, timeout_seconds: int,
) -> dict:
    try:
        r = requests.post(
            f"{HOMUNCULUS_BASE}/baritone/excavate",
            json={"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2,
                  "timeoutSeconds": int(timeout_seconds)},
            timeout=timeout_seconds + 10,
        )
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


def tunnel_for(
    target_drops: Iterable[str],
    quantity: int,
    *,
    timeout: float = 120.0,
    direction: str | None = None,
    y_offset: int = -2,
    step_blocks: int = 4,
) -> str | None:
    """Blind horizontal tunnel until `quantity` more of any `target_drops`.

    Reusable across mine_* primitives — the only resource-specific bit is
    the `target_drops` set the caller supplies (LOG_DROPS, STONE_DROPS, …).
    Picks a tunnel direction from player yaw (snapped cardinal) unless
    `direction` is given. Digs a 1×2 corridor (foot=y_offset, head=y_offset+1
    relative to player feet) in `step_blocks` segments via /baritone/excavate.
    After each segment, polls inventory; returns "tunnel" if delta ≥ quantity
    or None on timeout/failure.

    Note: at y_offset=-2 the tunnel sits 2 below the agent's feet, so on a
    flat surface biome the corridor cuts through dirt/stone with no entry
    shaft needed (Baritone digs in from the side). Agent's player position
    ends inside the tunnel — agent should call surface() afterwards if a
    surface return is needed.
    """
    drops_set = set(target_drops)
    before = _count_drops(drops_set)
    if before is None:
        print("  [tunnel] couldn't read inventory; aborting", flush=True)
        return None
    target = before + quantity

    try:
        pos = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=5.0).json()
        px = int(pos["x"])
        py = int(pos["y"])
        pz = int(pos["z"])
        yaw = float(pos.get("yaw", 0.0))
    except (requests.RequestException, ValueError, KeyError):
        print("  [tunnel] couldn't read position; aborting", flush=True)
        return None

    if direction is None:
        direction = _yaw_to_direction(yaw)
    if direction not in _DIR_VEC:
        direction = "east"
    dx, dz = _DIR_VEC[direction]

    # Tunnel at player's body level: feet=py, head=py+1. As baritone
    # /excavate clears each cell, the player walks horizontally through the
    # corridor — drops spawn at feet level, within pickup radius. This is
    # the user's "blind mining at agent-specified y" framing: the agent
    # MUST descend() to the desired y FIRST. Calling fair mine_stone at
    # the surface gets dirt/grass, not cobble — agent learns to descend.
    tunnel_y_bottom = py
    tunnel_y_top = py + 1
    print(
        f"  [tunnel] from ({px},{py},{pz}) {direction} y={tunnel_y_bottom}-{tunnel_y_top} "
        f"target={target} (before={before}, want={quantity})",
        flush=True,
    )

    # Early-exit: if K consecutive segments produce zero new drops, the player
    # isn't following the tunnel (likely because baritone can't reach the box —
    # cells in unloaded chunks return success+cleared=0 instantly). haiku-fair-r2
    # ran 1281 phantom segments at 100ms each across 5km of unloaded chunks while
    # the player sat still and got zombie-killed.
    DRY_SEGMENT_LIMIT = 3
    deadline = time.monotonic() + timeout
    step = 0
    last_count = before
    dry_streak = 0
    while time.monotonic() < deadline:
        s_lo = step * step_blocks + 1
        s_hi = (step + 1) * step_blocks
        x_a, x_b = px + dx * s_lo, px + dx * s_hi
        z_a, z_b = pz + dz * s_lo, pz + dz * s_hi
        bx1, bx2 = min(x_a, x_b), max(x_a, x_b)
        bz1, bz2 = min(z_a, z_b), max(z_a, z_b)
        remaining = deadline - time.monotonic()
        if remaining < 4.0:
            break
        seg_timeout = int(min(remaining, 30.0))
        print(
            f"  [tunnel] seg {step}: ({bx1},{tunnel_y_bottom},{bz1})..({bx2},{tunnel_y_top},{bz2}) "
            f"timeout={seg_timeout}s",
            flush=True,
        )
        data = _excavate_box(bx1, tunnel_y_bottom, bz1, bx2, tunnel_y_top, bz2,
                             timeout_seconds=seg_timeout)
        if not data.get("success"):
            print(
                f"  [tunnel] seg {step} failed: {data.get('reason')} "
                f"{data.get('message', '')}",
                flush=True,
            )
            # Soft-fail: advance step and try the next segment. Two failures
            # in a row will burn through to the timeout anyway. Hostile-hit
            # evasion is handled outside the handler — see Evasion.java +
            # agent.py per-turn arm/disarm; we don't special-case it here.
        after = _count_drops(drops_set)
        if after is None:
            after = last_count
        print(f"  [tunnel] seg {step} done; count={after}/{target}", flush=True)
        if after > last_count:
            dry_streak = 0
        else:
            dry_streak += 1
            if dry_streak >= DRY_SEGMENT_LIMIT:
                print(
                    f"  [tunnel] {dry_streak} dry segments in a row; baritone "
                    f"isn't progressing (likely unreachable or unloaded chunks). "
                    f"Stopping.",
                    flush=True,
                )
                break
        last_count = after
        if after >= target:
            return "tunnel"
        step += 1

    after = _count_drops(drops_set) or last_count
    if after > before:
        print(f"  [tunnel] timeout; partial acquired={after - before}", flush=True)
    else:
        print("  [tunnel] timeout; zero acquired", flush=True)
    return None


# Drop sets — duplicated from tools.py so mine.py stays import-clean.
# Kept in sync manually; if these diverge, tunnel_for would count wrong
# items vs the handler's delta math.
LOG_DROPS = {
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
    "minecraft:mangrove_log", "minecraft:cherry_log", "minecraft:pale_oak_log",
    "minecraft:crimson_stem", "minecraft:warped_stem",
}
STONE_DROPS = {"minecraft:cobblestone", "minecraft:cobbled_deepslate"}
IRON_DROPS = {"minecraft:raw_iron"}
DIAMOND_DROPS = {"minecraft:diamond"}
COAL_DROPS = {"minecraft:coal"}


def tunnel_for_logs(quantity: int) -> "str | None":
    return tunnel_for(LOG_DROPS, quantity)


# Fair-mode stone mining digs a DESCENDING STAIRCASE (down-and-forward, never
# straight down) to reach stone. Baritone's x-ray path mis-targets dense stone
# (picks faraway clumps), so stone is forced to blind fair mining; a flat
# surface tunnel only finds dirt. So instead of a horizontal corridor we cut a
# 1×2 staircase (one block down per one forward) that passes through the dirt
# cap into stone, collecting cobble as it goes. This absorbs the recovery into
# the substrate — an A/B'd failure-message nudge telling the LLM to descend had
# no measurable effect (2026-05-27). NEVER digs straight down (lava/fall trap):
# every step is forward+down, a walkable staircase. Bounded by step cap, a
# min-Y floor, wall-clock, and a stuck-streak (nothing cleared).
_STONE_STAIR_MAX_STEPS = 24      # 45° staircase: y~70 surface → ~y46, into stone
_STONE_STAIR_MIN_Y = -40         # don't staircase toward deep lava layers
_STONE_OVERALL_TIMEOUT = 150.0   # total wall-clock budget for one mine_stone turn
_STONE_STUCK_LIMIT = 3           # consecutive steps that clear nothing → bail


def _step_is_safe(cx: int, cy: int, cz: int) -> "tuple[bool, str]":
    """Pre-dig probe for one staircase step. Scans the 3×3×3 around the foot
    cell (cx,cy,cz) and refuses the step if:
      - lava is in or adjacent to it (digging in would flood / stepping kills), or
      - the floor below (cx,cy-1) is not solid — i.e. air/void → a fall.
    Conservative: any scan failure also returns unsafe (don't dig blind). This is
    what makes the staircase safe where a blind dig-down would drop into lava."""
    try:
        data = requests.get(
            f"{HOMUNCULUS_BASE}/scan_blocks",
            params={"x1": cx - 1, "y1": cy - 1, "z1": cz - 1,
                    "x2": cx + 1, "y2": cy + 1, "z2": cz + 1},
            timeout=5.0,
        ).json()
    except (requests.RequestException, ValueError):
        return False, "scan_failed"
    blocks = data.get("blocks")
    if blocks is None:
        return False, f"scan_{data.get('reason', 'error')}"
    solid_floor = False
    for b in blocks:
        bid = b.get("id", "")
        if "lava" in bid:
            return False, f"lava@({b.get('x')},{b.get('y')},{b.get('z')})"
        # air is omitted by scan_blocks, so a present, non-passable floor cell
        # is genuine solid ground.
        if (b.get("x") == cx and b.get("y") == cy - 1 and b.get("z") == cz
                and not b.get("passable", False)):
            solid_floor = True
    if not solid_floor:
        return False, "no_solid_floor (air/void below)"
    return True, "ok"


def tunnel_for_stone(quantity: int) -> "str | None":
    """Fair-mode stone via a descending staircase (see note above).

    Gated by CRAFT_STONE_STAIRCASE (default on) so the staircase can be A/B'd.
    When off, falls back to the legacy flat horizontal tunnel at the agent's
    current y (no auto-descend) — the control arm.
    """
    if os.environ.get("CRAFT_STONE_STAIRCASE", "1").strip().lower() in ("0", "false", "no"):
        return tunnel_for(STONE_DROPS, quantity)   # legacy flat tunnel (control)
    drops = STONE_DROPS
    before = _count_drops(drops)
    if before is None:
        print("  [stone] couldn't read inventory; aborting", flush=True)
        return None
    target = before + quantity
    try:
        pos = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=5.0).json()
        px, py, pz = int(pos["x"]), int(pos["y"]), int(pos["z"])
        yaw = float(pos.get("yaw", 0.0))
    except (requests.RequestException, ValueError, KeyError):
        print("  [stone] couldn't read position; aborting", flush=True)
        return None

    direction = _yaw_to_direction(yaw)
    dx, dz = _DIR_VEC.get(direction, (1, 0))
    print(
        f"  [stone] staircase {direction} from ({px},{py},{pz}) "
        f"target={target} (before={before}, want={quantity})",
        flush=True,
    )

    deadline = time.monotonic() + _STONE_OVERALL_TIMEOUT
    last = before
    last_y = py          # lowest Y reached so far — real descent down the staircase
    stuck = 0
    for i in range(1, _STONE_STAIR_MAX_STEPS + 1):
        remaining_time = deadline - time.monotonic()
        if remaining_time < 4.0:
            break
        # Step i: one block forward + one block down (a walkable stair). Clear the
        # foot (cy) and head (cy+1) cells of the column; the block below stays as
        # the next floor. The player's own column is never dug out beneath it.
        cx, cz = px + dx * i, pz + dz * i
        cy = py - i
        if cy <= _STONE_STAIR_MIN_Y:
            break
        # Look before you dig: refuse the step if lava is in/around it or the
        # floor below is air. This is the explicit hazard check that makes the
        # staircase safe (a bare dig-down has no such guard).
        safe, why = _step_is_safe(cx, cy, cz)
        if not safe:
            print(f"  [stone] step {i} blocked at y={cy}: {why} — stopping staircase",
                  flush=True)
            break
        _excavate_box(cx, cy, cz, cx, cy + 1, cz,
                      timeout_seconds=int(min(remaining_time, 20.0)))
        # Real-progress guard (issue #10). Baritone's own box accounting
        # (volume/remaining → "cleared") is unreliable: on a partial-break→reset
        # loop it reports the box cleared even though no block actually broke, so
        # a "cleared"-keyed guard never fires and the call burns the full 150s
        # while the pickaxe wears out with zero descent (the agent9 stalemate).
        # Key the stuck-streak on REAL progress instead — inventory gain OR the
        # player genuinely descending the staircase — mirroring tunnel_for's
        # inventory-delta bail. Descent is what still lets the staircase punch
        # through the dirt cap (no cobble drop there, but the player does move
        # down) without a premature stop.
        after = _count_drops(drops)
        if after is None:
            after = last
        try:
            cur = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=5.0).json()
            cur_y = int(cur["y"])
        except (requests.RequestException, ValueError, KeyError):
            cur_y = last_y
        progressed = (after > last) or (cur_y < last_y)
        print(f"  [stone] step {i}: y={cy} player_y={cur_y} progressed={progressed} "
              f"count={after}/{target}", flush=True)
        if after >= target:
            return "tunnel"
        if progressed:
            stuck = 0
        else:
            stuck += 1
            if stuck >= _STONE_STUCK_LIMIT:
                print(f"  [stone] {stuck} steps with no real progress "
                      f"(no drops, no descent) — stopping", flush=True)
                break
        last = after
        last_y = min(last_y, cur_y)

    final = _count_drops(drops)
    if final is None:
        final = before
    if final > before:
        print(f"  [stone] done; acquired {final - before}", flush=True)
    return "tunnel" if final > before else None


def tunnel_for_iron(quantity: int) -> "str | None":
    return tunnel_for(IRON_DROPS, quantity)


def tunnel_for_diamond(quantity: int) -> "str | None":
    return tunnel_for(DIAMOND_DROPS, quantity)


def tunnel_for_coal(quantity: int) -> "str | None":
    return tunnel_for(COAL_DROPS, quantity)


if __name__ == "__main__":
    result = mine_any_log(1)
    if result:
        print(f"acquired: {result}")
    else:
        print("no log type reachable")
