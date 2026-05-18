"""Scout subagent — situational synthesis from block scans.

Companion to craft/subagent.py. Pilot for the subagent pattern: given a
16×16 chunk of block data around the player, ask the subagent to describe
the surroundings in 2-3 sentences so the planning agent can make better
direction decisions.

Layers:
- ``scan_chunk(dx, dz)`` — raw payload helper. ``compact=True`` (default)
  L3-compacts the block list to a heightmap + interesting-blocks side
  list, shrinking a typical surface chunk from ~90KB to ~5KB of JSON. The
  compaction was selected after a 2026-05-17 ablation showed Qwen3-4B
  "going meta" (describing the JSON format instead of the world) at ~90KB
  and recovering cleanly at ~5KB.
- ``describe_chunk(dx, dz)`` — scan + synthesize.
- ``describe_neighborhood(radius)`` — fans out describe_chunk across
  (2r+1)² chunks in parallel and runs a synth-of-synths over the
  per-chunk descriptions to produce a unified scout report.

Known gap: entities are not yet scanned. /scan_entities is per-type so
including hostile + passive coverage means ~13 round-trips per chunk;
defer until single-chunk synthesis is validated end-to-end with the
agent.
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import Iterable

import requests

from craft.config import HOMUNCULUS_BASE
from craft.subagent import DEFAULT_SUBAGENT_MODEL, synthesize


# Env-var defaults. Set CRAFT_SCOUT_FANOUT_MODEL / CRAFT_SCOUT_UNIFY_MODEL
# to override without touching code or argparse — same pattern as the
# HOMUNCULUS_PORT / MC_PLAYER_NAME retargeting. The intended use case is
# pure-qwen rollouts (all three roles on the same Ollama GPU) for
# stress-testing GPU saturation against the mixed-model default.
_FANOUT_MODEL_DEFAULT = os.environ.get("CRAFT_SCOUT_FANOUT_MODEL") or "claude-haiku-4-5"
_UNIFY_MODEL_DEFAULT = os.environ.get("CRAFT_SCOUT_UNIFY_MODEL") or DEFAULT_SUBAGENT_MODEL


# Chunk description cache. Keyed on (chunk_x, chunk_z, py_band,
# vertical_radius, model). py_band buckets player Y into 4-block bands so
# a small vertical wiggle reuses cached descriptions, but a descend(32)
# evicts (different slice of the chunk volume).
#
# TTL is intentionally short. Stale descriptions are misleading after
# mine_* / place / build_shelter, and the substrate doesn't (yet)
# invalidate cache entries on block mutation — this is the "TTL + log
# hit-rate first" pass. If hit rate is meaningful, the next iteration
# threads scout.invalidate_chunk() through the mutating handlers.
_CHUNK_CACHE_TTL_S = float(os.environ.get("CRAFT_SCOUT_CACHE_TTL_S", "30"))
_PY_BAND_SIZE = 4

_chunk_cache: dict[tuple, tuple[float, str]] = {}
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0, "expired": 0}


def scout_cache_stats() -> dict[str, int]:
    """Snapshot of cache hit / miss / expired counts since process start."""
    return dict(_cache_stats)


def scout_cache_reset() -> None:
    """Clear cache + stats. Mostly for tests / repeated benchmarks."""
    _chunk_cache.clear()
    _cache_stats.update({"hits": 0, "misses": 0, "expired": 0})


def _cache_get(key: tuple) -> str | None:
    import time
    entry = _chunk_cache.get(key)
    if entry is None:
        _cache_stats["misses"] += 1
        return None
    ts, desc = entry
    age = time.time() - ts
    if age > _CHUNK_CACHE_TTL_S:
        _cache_stats["expired"] += 1
        del _chunk_cache[key]
        return None
    _cache_stats["hits"] += 1
    return desc


def _cache_put(key: tuple, description: str) -> None:
    import time
    _chunk_cache[key] = (time.time(), description)


SCOUT_PROMPT = (
    "You are a scout for a Minecraft 1.21.4 agent. Given block + position "
    "data for a 16×16 chunk around the player, describe the surroundings "
    "in 2-3 sentences for the planner. Use cardinal directions (+x=east, "
    "-x=west, +z=south, -z=north). Mention terrain, hazards (water, lava, "
    "drops, exposed caves), and resources (trees, ore outcrops, "
    "structures). Be concise; the planner reads this in one breath."
)


NEIGHBORHOOD_PROMPT = (
    "You are a scout for a Minecraft 1.21.4 agent. You receive per-chunk "
    "scout reports keyed by their position relative to the player (dx, dz) "
    "where +x=east, -x=west, +z=south, -z=north; (0,0) is the player's "
    "chunk. Produce ONE unified spatial summary in 3-5 sentences: where "
    "the player is, what's in each cardinal direction (combine adjacent "
    "chunks), and which directions are safest / most resource-rich. Lead "
    "with a one-line recommendation: 'best direction to travel: <cardinal>'."
)


# Block ids worth surfacing separately in the L3-compact payload. Anything
# in this set is preserved at its true (x, y, z) instead of just contributing
# to the topmost-per-column heightmap.
INTERESTING_BLOCKS: frozenset[str] = frozenset({
    # Liquids — hazards.
    "water", "lava",
    # Ores — resources.
    "coal_ore", "deepslate_coal_ore",
    "iron_ore", "deepslate_iron_ore",
    "copper_ore", "deepslate_copper_ore",
    "gold_ore", "deepslate_gold_ore",
    "diamond_ore", "deepslate_diamond_ore",
    "redstone_ore", "deepslate_redstone_ore",
    "lapis_ore", "deepslate_lapis_ore",
    "emerald_ore", "deepslate_emerald_ore",
    # Wood — resources.
    "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "cherry_log",
    # Cave/structure markers — hazard or signal.
    "cave_air", "obsidian", "mob_spawner",
    # Village markers.
    "crafting_table", "furnace", "chest", "bed",
})


def scan_chunk(
    dx: int = 0,
    dz: int = 0,
    *,
    vertical_radius: int = 3,
    compact: bool = True,
) -> dict:
    """Snapshot one 16×16 chunk at (player_chunk + dx, dz), Y ± vertical_radius.

    Returns ``{success, chunk, box, player_pos, biome, ...}``. When
    ``compact`` (default), the block list collapses into ``surface_topmost``
    (one record per (x, z) column, the highest non-air block) plus
    ``interesting_blocks`` (full position list for ores / wood / liquids /
    structure markers). When ``compact=False``, ``blocks`` is the raw
    list-of-dicts straight from ``/scan_blocks``.

    Vertical slice defaults to ±3 (7 tall, 1792 cells) — under homunculus's
    MAX_VOLUME=2000 cap.
    """
    pos = _get("/position")
    if pos.get("success") is False:
        return {"success": False, "reason": "position_unavailable", "detail": pos}
    px, py, pz = pos["x"], pos["y"], pos["z"]

    stats = _get("/stats")
    biome = (
        stats.get("biome", "unknown")
        if stats.get("success") is not False
        else "unknown"
    )

    cx = (int(px) >> 4) + dx
    cz = (int(pz) >> 4) + dz
    x1, z1 = cx * 16, cz * 16
    x2, z2 = x1 + 15, z1 + 15
    y1 = int(py) - vertical_radius
    y2 = int(py) + vertical_radius

    scan = _get(
        "/scan_blocks",
        params={"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2},
        timeout=10.0,
    )
    if scan.get("success") is False:
        return {
            "success": False,
            "reason": "scan_failed",
            "chunk": [cx, cz],
            "box": [x1, y1, z1, x2, y2, z2],
            "detail": scan,
        }

    blocks = scan.get("blocks", [])
    base = {
        "success": True,
        "chunk": [cx, cz],
        "box": [x1, y1, z1, x2, y2, z2],
        "player_pos": [px, py, pz],
        "biome": biome,
    }
    if compact:
        base.update(_l3_compact(blocks))
    else:
        base["blocks"] = blocks
    return base


def _l3_compact(blocks: Iterable[dict]) -> dict:
    """Collapse a raw block list into surface heightmap + interesting list.

    ``surface_topmost``: one entry per (x, z) column at the highest y where
    the block is non-air, as ``[x, y, z, id_short]``. This conveys terrain
    shape without flooding the model with every solid block.

    ``interesting_blocks``: full ``[x, y, z, id_short]`` list for every
    block whose short id is in ``INTERESTING_BLOCKS``. These are the
    features that drive planner decisions (ore, wood, liquids, structures)
    and would otherwise be lost in the heightmap.
    """
    cols: dict[tuple[int, int], tuple[int, int, int, str]] = {}
    interesting: list[list] = []
    for b in blocks:
        x, y, z = b["x"], b["y"], b["z"]
        bid = b["id"].removeprefix("minecraft:")
        key = (x, z)
        cur = cols.get(key)
        if cur is None or y > cur[1]:
            cols[key] = (x, y, z, bid)
        if bid in INTERESTING_BLOCKS:
            interesting.append([x, y, z, bid])
    surface = sorted(
        ([x, y, z, bid] for (x, y, z, bid) in cols.values()),
        key=lambda r: (r[2], r[0]),
    )
    return {"surface_topmost": surface, "interesting_blocks": interesting}


def describe_chunk(
    dx: int = 0,
    dz: int = 0,
    *,
    model: str = DEFAULT_SUBAGENT_MODEL,
    vertical_radius: int = 3,
    compact: bool = True,
    prompt: str = SCOUT_PROMPT,
    use_cache: bool = True,
) -> str:
    """Scan + synthesize one chunk. Returns the description text.

    When ``use_cache`` (default), looks up an existing description keyed on
    ``(cx, cz, py_band, vertical_radius, model)`` before scanning. Cache
    misses fall through to scan + synthesize and store the result.
    """
    cache_key = None
    if use_cache:
        pos = _get("/position")
        if pos.get("success") is not False:
            px = int(pos.get("x", 0))
            py = int(pos.get("y", 0))
            pz = int(pos.get("z", 0))
            cx = (px >> 4) + dx
            cz = (pz >> 4) + dz
            py_band = py // _PY_BAND_SIZE
            cache_key = (cx, cz, py_band, vertical_radius, model)
            cached = _cache_get(cache_key)
            if cached is not None:
                return cached

    payload = scan_chunk(dx, dz, vertical_radius=vertical_radius, compact=compact)
    if not payload.get("success"):
        return f"scan_failed: {payload.get('reason')}"
    description = synthesize(prompt, payload, model=model)
    if cache_key is not None:
        _cache_put(cache_key, description)
    return description


def describe_neighborhood(
    radius: int = 2,
    *,
    model: str | None = None,
    fanout_model: str | None = None,
    unify_model: str | None = None,
    vertical_radius: int = 3,
    max_workers: int = 16,
) -> dict:
    """Fan out describe_chunk over a (2r-1)² chunk grid around the player,
    then synth-of-synths into a unified scout report.

    Radius is in chunks (1-indexed, side length = 2r-1):

    - ``radius=1`` →  1 chunk  (just the player's chunk; unify is a no-op
      and returns the single description unchanged)
    - ``radius=2`` →  9 chunks (3×3)
    - ``radius=3`` → 25 chunks (5×5)
    - ``radius=4`` → 49 chunks (7×7)

    Returns ``{per_chunk: {"<dx>,<dz>": description}, unified: str,
    timings: {fanout_s, unify_s}, models: {fanout, unify}}``.

    Mixed-model composition: ``fanout_model`` and ``unify_model`` are
    independently selectable. Smoked 2026-05-17: claude-haiku-4-5 for the
    fan-out (~9× parallel via Anthropic infra) + Qwen3-4B for the unify
    (small input, fast local) sits between pure-modes on latency; quality
    win was visible in one trial. If ``model`` is set and the role-specific
    knobs are not, both legs use ``model``. Default fan-out is Haiku;
    default unify is Qwen.
    """
    import time

    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    if fanout_model is None:
        fanout_model = model or _FANOUT_MODEL_DEFAULT
    if unify_model is None:
        unify_model = model or _UNIFY_MODEL_DEFAULT

    half = radius - 1
    offsets = [
        (dx, dz)
        for dx in range(-half, half + 1)
        for dz in range(-half, half + 1)
    ]

    def _one(off):
        dx, dz = off
        return off, describe_chunk(
            dx, dz,
            model=fanout_model, vertical_radius=vertical_radius,
        )

    hits_before = _cache_stats["hits"]
    t0 = time.time()
    per_chunk: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for off, desc in ex.map(_one, offsets):
            per_chunk[f"{off[0]},{off[1]}"] = desc
    fanout_s = time.time() - t0
    fanout_hits = _cache_stats["hits"] - hits_before

    cache_block = {"hits": fanout_hits, "total": len(per_chunk)}

    # No-unify shortcut when there's nothing to compose. The unified field
    # mirrors the single chunk description so callers get a stable contract.
    if len(per_chunk) == 1:
        only = next(iter(per_chunk.values()))
        return {
            "per_chunk": per_chunk,
            "unified": only,
            "timings": {"fanout_s": fanout_s, "unify_s": 0.0},
            "models": {"fanout": fanout_model, "unify": None},
            "cache": cache_block,
        }

    t1 = time.time()
    unified = synthesize(
        NEIGHBORHOOD_PROMPT, {"chunks": per_chunk}, model=unify_model,
    )
    unify_s = time.time() - t1

    return {
        "per_chunk": per_chunk,
        "unified": unified,
        "timings": {"fanout_s": fanout_s, "unify_s": unify_s},
        "models": {"fanout": fanout_model, "unify": unify_model},
        "cache": cache_block,
    }


def _get(path: str, *, params: dict | None = None, timeout: float = 5.0) -> dict:
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


if __name__ == "__main__":
    import json
    import sys
    import time

    args = sys.argv[1:]
    if args and args[0] == "hood":
        radius = int(args[1]) if len(args) >= 2 else 2
        print(f"--- describe_neighborhood(radius={radius}) ---")
        t0 = time.time()
        result = describe_neighborhood(radius)
        print(f"\n[total: {time.time() - t0:.2f}s; "
              f"fan-out({result['models']['fanout']}): {result['timings']['fanout_s']:.2f}s; "
              f"unify({result['models']['unify']}): {result['timings']['unify_s']:.2f}s]\n")
        print("--- per-chunk ---")
        for k, v in result["per_chunk"].items():
            print(f"\n({k}):\n  {v}")
        print("\n--- unified ---")
        print(result["unified"])
        sys.exit(0)

    dx = int(args[0]) if len(args) >= 1 else 0
    dz = int(args[1]) if len(args) >= 2 else 0
    print(f"--- scan_chunk(dx={dx}, dz={dz}) ---")
    t0 = time.time()
    payload = scan_chunk(dx, dz)
    print(f"scan: {time.time() - t0:.2f}s")
    if not payload.get("success"):
        print(json.dumps(payload, indent=2))
        sys.exit(1)
    print(
        f"biome={payload['biome']} chunk={payload['chunk']} "
        f"surface={len(payload.get('surface_topmost', []))} "
        f"interesting={len(payload.get('interesting_blocks', []))}"
    )

    print(f"\n--- describe_chunk(dx={dx}, dz={dz}) ---")
    t0 = time.time()
    print(synthesize(SCOUT_PROMPT, payload))
    print(f"synthesize: {time.time() - t0:.2f}s")
