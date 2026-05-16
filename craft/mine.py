"""Candidate-cycling mining via homunculus /baritone/mine.

`#mine` takes a single block id; this layer iterates candidate variants
(oak_log → birch_log, iron_ore → deepslate_iron_ore, …) until one reports
`have_target` or `already_satisfied`. Quantity is a cumulative inventory
target (Baritone's own semantics); the delta wrapper lives in tools.py.
"""

from __future__ import annotations

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
SKIP_REASONS = {"unreachable", "never_started"}
SUCCESS_REASONS = {"have_target", "already_satisfied"}

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
    """
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
            print(f"  → transport_error: {e}, stopping cycle")
            return None
        reason = data.get("reason", "unknown")
        if data.get("success") and reason in SUCCESS_REASONS:
            print(f"  → success ({block}, {reason})")
            return block
        if reason in SKIP_REASONS:
            print(f"  → {reason}, trying next")
            continue
        # interrupted / timeout / busy / baritone_not_loaded / unknown_block / internal_error
        msg = data.get("message", "")
        print(f"  → {reason} ({msg}), stopping cycle")
        return None
    return None


def mine_any_log(quantity: int = 1) -> str | None:
    # Trees are surface canopy — wide horizontal, narrow vertical band.
    return _mine_first_reachable(quantity, LOG_TYPES, probe_radius=64, probe_y_radius=8)


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


def tunnel_for_stone(quantity: int) -> "str | None":
    return tunnel_for(STONE_DROPS, quantity)


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
