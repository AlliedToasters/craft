"""Tool surface exposed to the LLM planner.

Each tool has an OpenAI-style schema (in TOOLS) and a handler (in HANDLERS).
Handlers return a one-line outcome string that gets routed back to the planner
as the `tool` role message for the next turn.
"""

from __future__ import annotations

import json
import math
import os

import requests

from craft.mine import (
    LOG_TYPES,
    _DIR_VEC,
    _yaw_to_direction,
    mine_any_coal,
    mine_any_diamond,
    mine_any_iron,
    mine_any_log,
    mine_any_stone,
    tunnel_for_coal,
    tunnel_for_diamond,
    tunnel_for_iron,
    tunnel_for_logs,
    tunnel_for_stone,
)

from craft.config import HOMUNCULUS_HOST, HOMUNCULUS_PORT, HOMUNCULUS_BASE  # noqa: F401

MAX_QUANTITY = 10
TRAVEL_MAX_DISTANCE = 64
DESCEND_MAX_PER_CALL = 40
SURFACE_MAX_PER_CALL = 40

# Items whose inventory counts are summed to track mining progress.
# mine_wood/mine_stone use delta semantics: target = current_count + requested.
LOG_DROPS = {f"minecraft:{lt}" for lt in LOG_TYPES}
STONE_DROPS = {"minecraft:cobblestone", "minecraft:cobbled_deepslate"}
IRON_DROPS = {"minecraft:raw_iron"}
DIAMOND_DROPS = {"minecraft:diamond"}
COAL_DROPS = {"minecraft:coal"}

# Tier 0-2 buildable blocks (mirrors homunculus Equipper.BUILDING_BLOCK_TIER).
# Used by build_shelter's pre-flight to sum inventory buildables and refuse
# a build that would partially seal — observed rollouts 6 + 9 died to mobs
# entering through ceiling holes caused by mid-build inventory exhaustion.
_SHELTER_BUILDABLE_BARE = {
    # Tier 0 — trash / pathing-friendly
    "dirt", "coarse_dirt", "rooted_dirt", "grass_block", "podzol", "mycelium",
    "mud", "packed_mud", "cobblestone", "cobbled_deepslate", "netherrack", "blackstone",
    # Tier 1 — cheap stone
    "stone", "deepslate", "granite", "diorite", "andesite", "tuff",
    "basalt", "smooth_basalt", "end_stone", "sandstone", "red_sandstone",
    "calcite", "dripstone_block",
    # Tier 2 — wood
    "oak_planks", "spruce_planks", "birch_planks", "jungle_planks", "acacia_planks",
    "dark_oak_planks", "mangrove_planks", "cherry_planks", "bamboo_planks",
    "crimson_planks", "warped_planks",
    "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "cherry_log",
    "stripped_oak_log", "stripped_spruce_log", "stripped_birch_log",
    "stripped_jungle_log", "stripped_acacia_log", "stripped_dark_oak_log",
    "stripped_mangrove_log", "stripped_cherry_log",
}
# Conservative minimum: shell geometry has ~98 cells but pre-existing terrain
# usually covers some. Observed need ~50-90 across recent rollouts; 70 leaves
# margin for low-coverage spawns (hilltops, sky islands) without false-aborting
# the typical flat-terrain build.
SHELTER_BUDGET_MIN = 70

# Items Baritone treats as freely placeable for pillar-up / bridging during
# pathing (Baritone's default `acceptableThrowawayItems`). When a substrate-
# initiated goto is about to lead into a craft that needs one of these, we
# restrict Baritone's placement set via /baritone/goto so the craft materials
# survive the trip. See memory:
# project_baritone_inventory_consumption.md for the failure case.
THROWAWAY_ITEMS = {
    "minecraft:cobblestone",
    "minecraft:dirt",
    "minecraft:netherrack",
}

# Recipes for items that require sequential substeps to craft.
# Key: output item. Value: list of (ingredient, count) tuples needed.
# When a recipe's ingredient has its own recipe, we recurse.
CRAFTING_RECIPES = {
    "minecraft:wooden_pickaxe": [
        ("minecraft:oak_planks", 3),
        ("minecraft:stick", 2),
    ],
    "minecraft:stone_pickaxe": [
        ("minecraft:cobblestone", 3),
        ("minecraft:stick", 2),
    ],
    "minecraft:iron_pickaxe": [
        ("minecraft:iron_ingot", 3),
        ("minecraft:stick", 2),
    ],
    "minecraft:diamond_pickaxe": [
        ("minecraft:diamond", 3),
        ("minecraft:stick", 2),
    ],
    "minecraft:furnace": [("minecraft:cobblestone", 8)],
    "minecraft:oak_planks": [("minecraft:oak_log", 1)],
    "minecraft:birch_planks": [("minecraft:birch_log", 1)],
    "minecraft:spruce_planks": [("minecraft:spruce_log", 1)],
    "minecraft:jungle_planks": [("minecraft:jungle_log", 1)],
    "minecraft:acacia_planks": [("minecraft:acacia_log", 1)],
    "minecraft:dark_oak_planks": [("minecraft:dark_oak_log", 1)],
    "minecraft:mangrove_planks": [("minecraft:mangrove_log", 1)],
    "minecraft:cherry_planks": [("minecraft:cherry_log", 1)],
    "minecraft:pale_oak_planks": [("minecraft:pale_oak_log", 1)],
    "minecraft:stick": [("minecraft:oak_planks", 2)],
    "minecraft:crafting_table": [("minecraft:oak_planks", 4)],
}

# Wood-species substitution table for _craft_recursive. When homunculus
# reports missing oak_planks (its default species when none are available),
# the agent may be holding spruce/birch/etc. — substitute to a species we
# actually have. Vanilla recipes accept any *_planks via the #planks tag,
# so the parent recipe still resolves. Observed in probe-validate-r2 T2:
# snowy_taiga spawn, 3x spruce_log in inventory, craft(wooden_pickaxe)
# failed because the oak_planks substep had no oak_log.
_PLANKS_LOG_BY_SPECIES = {
    "minecraft:oak_planks": "minecraft:oak_log",
    "minecraft:spruce_planks": "minecraft:spruce_log",
    "minecraft:birch_planks": "minecraft:birch_log",
    "minecraft:jungle_planks": "minecraft:jungle_log",
    "minecraft:acacia_planks": "minecraft:acacia_log",
    "minecraft:dark_oak_planks": "minecraft:dark_oak_log",
    "minecraft:mangrove_planks": "minecraft:mangrove_log",
    "minecraft:cherry_planks": "minecraft:cherry_log",
    "minecraft:pale_oak_planks": "minecraft:pale_oak_log",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mine_wood",
            "description": (
                "Mine wood logs from any nearby tree. Cycles through tree types. "
                "DELTA semantics: quantity is how many MORE logs to acquire on top "
                "of what you already have. Capped at 10 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of additional logs to mine (capped at 10).",
                    },
                    "fair": {
                        "type": "boolean",
                        "description": (
                            "If true, use BLIND TUNNELING (no chunk-wide target "
                            "search): dig a 1×2 corridor forward (player yaw) and "
                            "stop when delta acquired. For wood, this only works "
                            "if you're already adjacent to a tree — usually leave "
                            "false. Default: false."
                        ),
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_stone",
            "description": (
                "Mine stone-type blocks (stone, deepslate, cobblestone, "
                "cobbled_deepslate) to acquire cobblestone in inventory. "
                "BLIND TUNNELING at your current y: digs a 1×2 corridor forward "
                "in your facing direction and walks you through it, exiting on "
                "delta-hit. For best yield, descend() to y≤32 FIRST so the "
                "tunnel is in stone — calling at surface tunnels through dirt "
                "and produces no cobble. Requires a pickaxe in inventory — "
                "the harness equips it. DELTA semantics: quantity is how many "
                "MORE cobble drops to acquire. Capped at 10 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of additional stone-drops to mine (capped at 10).",
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_iron",
            "description": (
                "Mine iron ore (iron_ore, deepslate_iron_ore, raw_iron_block). "
                "Drops raw_iron, which must be smelted in a furnace to become "
                "iron_ingot. Requires a stone-tier or better pickaxe in inventory "
                "(wooden won't work — drops nothing). DELTA semantics: quantity "
                "is how many MORE raw_iron to acquire on top of what you have. "
                "Capped at 10 per call. May fail with 'far' if no exposed ore is "
                "in nearby loaded chunks — try mining stone first to expose more "
                "of the underground."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of additional raw_iron drops to mine (capped at 10).",
                    },
                    "fair": {
                        "type": "boolean",
                        "description": (
                            "If true, BLIND TUNNEL at player's current y. Descend "
                            "to y=16-32 first; the tunnel digs forward at body level. "
                            "Default: false (baritone chunk-scan also works for iron)."
                        ),
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_diamond",
            "description": (
                "Mine diamond ore (diamond_ore, deepslate_diamond_ore). "
                "Drops 1x diamond per ore — no smelting needed. Requires an "
                "IRON-tier or better pickaxe (stone/wood will break the block "
                "but drop nothing — wasted swing). DELTA semantics: quantity is "
                "how many MORE diamonds to acquire on top of what you have. "
                "Capped at 10 per call. Densest spawn is Y=-58 to -64 (deepslate "
                "layer); a sparser layer exists below Y=16. Use descend() to "
                "reach the target Y first. May fail with 'far' if no exposed "
                "diamond ore is in nearby loaded chunks — mine_stone first to "
                "expose more of the underground, or travel + descend to a "
                "different column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of additional diamonds to mine (capped at 10).",
                    },
                    "fair": {
                        "type": "boolean",
                        "description": (
                            "If true, BLIND TUNNEL at player's current y. Descend "
                            "to y=-58 to -64 first. Default: false (chunk-scan is "
                            "still genuinely useful for rare ores like diamond)."
                        ),
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_coal",
            "description": (
                "Mine coal ore (coal_ore, deepslate_coal_ore). Drops 1x coal per "
                "ore — the tier-appropriate smelt fuel (1 coal smelts 8 items vs "
                "1.5 for a plank). Works with a wooden+ pickaxe. DELTA semantics: "
                "quantity is how many MORE coal to acquire on top of what you "
                "have. Capped at 10 per call. Common at Y=16 to Y=64; abundant "
                "in mountains and exposed cliffs. Mine this before smelting to "
                "save your wood for crafting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of additional coal to mine (capped at 10).",
                    },
                    "fair": {
                        "type": "boolean",
                        "description": (
                            "If true, BLIND TUNNEL at player's current y. Descend "
                            "to y=40-60 first. Default: false."
                        ),
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "craft",
            "description": (
                "Craft an item. ALL substeps are handled internally: sub-recipes "
                "(planks, sticks) AND placing a crafting_table when needed. "
                "Just request the final item — do not pre-craft prerequisites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Item name without namespace, e.g. 'wooden_pickaxe', 'crafting_table', 'stick'. We add the 'minecraft:' prefix automatically.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to craft.",
                    },
                    "location": {
                        "type": "string",
                        "enum": ["auto", "home", "here"],
                        "description": (
                            "Where to perform the craft. 'home' returns to the "
                            "home base (saved when the first crafting_table is "
                            "placed) before crafting — use this when you've left "
                            "your base, e.g. went underground to mine. 'here' "
                            "crafts in-place, auto-placing a new table if needed "
                            "— use on long expeditions when home is too far. "
                            "'auto' (default) currently behaves like 'here'; the "
                            "harness will fall back to home automatically if no "
                            "table is reachable. Default: 'auto'."
                        ),
                    },
                },
                "required": ["item", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smelt",
            "description": (
                "Load a furnace and ignite it; returns IMMEDIATELY. The cook runs "
                "asynchronously (~10s per item) while you do other things. Call "
                "collect_smelt() in a later turn to retrieve the outputs. Common "
                "uses: raw_iron→iron_ingot, raw_gold→gold_ingot, "
                "sand→glass, cobblestone→stone, "
                "oak_log→charcoal. Fuel is auto-picked from inventory "
                "(sticks/planks/logs preferred over coal). If no furnace is "
                "placed, the harness will craft+place one for you (uses 8 "
                "cobblestone). The 'Active smelts' block in each turn's context "
                "will show progress; collect when status is READY."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The item to smelt (the consumable, not the result). E.g. 'raw_iron', 'sand'. We add minecraft: prefix automatically.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of items to smelt (capped at 10).",
                    },
                    "location": {
                        "type": "string",
                        "enum": ["auto", "home", "here"],
                        "description": (
                            "Where to smelt. 'home' returns to base before "
                            "smelting (uses any furnace already there). 'here' "
                            "smelts in-place, auto-placing a furnace if needed. "
                            "'auto' (default) currently behaves like 'here' with "
                            "a silent fall-back to home if no furnace is reachable."
                        ),
                    },
                },
                "required": ["input", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collect_smelt",
            "description": (
                "Retrieve outputs from one of your active smelting furnaces. Walks "
                "the player to the furnace and pulls ready ingots/items into "
                "inventory. Call this AFTER smelt() returns 'started', once the "
                "'Active smelts' block shows status READY (or PARTIAL if fuel ran "
                "out mid-batch). With no argument, collects from the nearest "
                "ready furnace. Pass furnace_pos to target a specific one when "
                "you have multiple in flight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "furnace_pos": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [x, y, z] of a specific furnace to collect from. Omit to collect from the nearest ready furnace.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": (
                "Place a block on the ground near you (the harness picks a safe "
                "spot — you don't need to aim). Rarely needed in practice: craft() "
                "auto-places crafting_tables and smelt() auto-places furnaces. "
                "Use for chests or other blocks the harness won't place for you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Block item name, e.g. 'chest', 'furnace', 'crafting_table'. We add the 'minecraft:' prefix automatically.",
                    }
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "surface",
            "description": (
                "Travel up to open sky in your current column. Use to escape a "
                "self-dug hole, a cave, or any enclosed space — Baritone will dig "
                "upward through stone if needed. No arguments. Returns FAILED if "
                "the column has no sky above (rare); in that case, mine_stone "
                f"upward or travel() to a different column. CHUNKED: each call "
                f"advances at most {SURFACE_MAX_PER_CALL} blocks; for deep "
                "ascents just call surface() again until the result no longer "
                "says 'more — call surface() again'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "descend",
            "description": (
                "Dig DOWN to a target Y level. Use this — not mine_stone — to "
                "move deeper underground. Baritone mines straight down through "
                "stone to reach the target. Reference Y levels: diamonds Y=8 to "
                "Y=16, iron Y=8 to Y=32, deepslate boundary Y=0. Requires a "
                f"pickaxe in inventory (stone-tier for ores). CHUNKED: each "
                f"call advances at most {DESCEND_MAX_PER_CALL} blocks; for deep "
                "targets just call descend(target_y) again until the result no "
                "longer says 'more — call descend... again'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_y": {
                        "type": "integer",
                        "description": "The Y level to descend to. Must be below your current Y. For diamonds use 8.",
                    },
                },
                "required": ["target_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_shelter",
            "description": (
                "Build a stone shelter around your current position. Carves a "
                "5×2×5 air cavity (room to move around) and seals it with "
                "solid walls, floor, and ceiling using whatever cheap building "
                "blocks you have. Patchwork-safe: starts with whatever's most "
                "plentiful (cobblestone, dirt, cobbled_deepslate, netherrack…) "
                "and auto-switches mid-build if one material runs out. Use when "
                "HP is low, mobs are pressuring, or to wait out the night. "
                "Places up to ~90 blocks (any mix); fewer when existing terrain "
                "already covers part of the hull. To break out, mine_stone(1) "
                "from inside. "
                "DESIGNED FOR OPEN SURFACE TERRAIN — underground / cave / "
                "encased starts often fail; surface yourself first if you're "
                "in a tunnel or pocket."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wall_in",
            "description": (
                "Tactical retreat: tunnel 3 cells into the nearest cardinal "
                "wall and seal yourself in. Picks a direction with a 3-deep "
                "solid wall at your head+feet, digs a 1×2 corridor, then "
                "places blocks 1 cell in from the entrance — leaving a 1-cell "
                "foyer + a 2-cell back cavity for you. Use when caught "
                "underground after evasion fires, or when night catches you "
                "in a cave with no shelter materials. Needs to be flush "
                "against a cave wall or hillside; on open ground use "
                "build_shelter instead. The tunnel itself produces enough "
                "stone for the seal. To break out, mine_stone(1) from inside."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "carve_alcove",
            "description": (
                "Convert a tactical wall_in pocket into a working shelter. "
                "Carves a 2×3 alcove off your back cavity (2 cells forward, "
                "3 cells wide) — enough room for a crafting_table, furnace, "
                "and bed. Requires you've already called wall_in and are "
                "still inside it. Refuses if water/lava is in the wall "
                "(cave-adjacent). The seal stays intact — you're still safe "
                "from mobs while carving. After: place crafting_table / "
                "furnace / bed with the normal place() tool inside the new "
                "chamber."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "travel",
            "description": (
                "Walk a number of blocks in a cardinal direction. Use to find "
                "flatter ground after a place-failure (no_space / no_placeable_spot), "
                "explore for resources, or move away from a cluttered work area. "
                "Baritone handles pathing and will dig/build through obstacles by "
                "default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["north", "south", "east", "west"],
                        "description": "Cardinal direction. Mojang convention: north=-z, south=+z, east=+x, west=-x.",
                    },
                    "distance": {
                        "type": "integer",
                        "description": "Number of blocks to travel (capped at 64).",
                    },
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_around",
            "description": (
                "Get a natural-language description of the surrounding terrain, "
                "hazards (water, lava, drops, exposed caves), and resources "
                "(trees, ores, structures) in the chunks around the player. "
                "A scout subagent reads the block scan and condenses it into "
                "a short paragraph with cardinal direction hints. Useful for "
                "deciding which way to travel, or confirming surroundings "
                "before committing to a plan. Latency: radius=1 ~3s (1 chunk, "
                "just current), radius=2 ~5s (3×3), radius=3 ~8s (5×5). "
                "Does NOT mine, place, or move the player — read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "integer",
                        "description": "Chunk radius. 1 = current chunk only, 2 = 3×3 (default), 3 = 5×5.",
                    },
                },
                "required": [],
            },
        },
    },
]


def _post_homunculus(path: str, payload: dict, *, timeout: float = 10.0) -> dict:
    """Single POST to homunculus. Returns parsed JSON, or a synthetic transport_error dict."""
    try:
        resp = requests.post(f"{HOMUNCULUS_BASE}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}
    except ValueError as e:
        return {"success": False, "reason": "transport_error", "message": f"bad JSON: {e}"}


def _get_homunculus(path: str, *, params: dict | None = None, timeout: float = 5.0) -> dict:
    """Single GET to homunculus. Returns parsed JSON, or a synthetic transport_error dict."""
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}
    except ValueError as e:
        return {"success": False, "reason": "transport_error", "message": f"bad JSON: {e}"}


def _position() -> dict:
    return _get_homunculus("/position")


def _scan_column(x: int | None = None, z: int | None = None) -> dict:
    params: dict = {}
    if x is not None:
        params["x"] = x
    if z is not None:
        params["z"] = z
    return _get_homunculus("/scan_column", params=params or None)


def _craft_raw(item: str, count: int) -> dict:
    print(f"  [craft] requesting {count}x {item}...", flush=True)
    return _post_homunculus("/craft", {"item": item, "count": count})


def _place_raw(item: str) -> dict:
    print(f"  [place] requesting {item}...", flush=True)
    return _post_homunculus("/place", {"item": item})


def _place_at_raw(item: str, x: int, y: int, z: int) -> dict:
    """Place a block at an explicit coordinate (no candidate search).

    Caller is responsible for positioning the bot within reach (~5.5 from
    support top). Used for the shelter doorway where /place's ring-2-first
    search would put the block on open ground outside the wall, not in
    the wall slot itself.
    """
    print(f"  [place_at] requesting {item} at ({x},{y},{z})...", flush=True)
    return _post_homunculus(
        "/place_at", {"item": item, "x": x, "y": y, "z": z}, timeout=15.0,
    )


def _smelt_raw(input_item: str, count: int, fuel: str | None = None) -> dict:
    """Fire-and-forget smelt start. Returns immediately after place+load+ignite.

    Homunculus v1.2+: /smelt does validate, auto-place (if needed), load input
    + fuel, ignite, register the smelt — then returns without waiting for the
    cook. Cook ticks asynchronously; poll /smelt_status or call /collect_smelt
    later. Timeout covers the synchronous prologue only.
    """
    body: dict = {"input": input_item, "count": count}
    if fuel:
        body["fuel"] = fuel
    timeout = 30.0
    print(f"  [smelt] starting {count}x {input_item} (fuel={fuel or 'auto'})...", flush=True)
    return _post_homunculus("/smelt", body, timeout=timeout)


def _collect_smelt_raw(furnace_pos: list[int] | None = None) -> dict:
    """Walk to an active furnace and pull ready outputs.

    Optional furnace_pos targets a specific furnace; otherwise homunculus
    picks the closest ready/partial one. Timeout is generous because the
    Baritone goto leg dominates.
    """
    body: dict = {}
    if furnace_pos is not None:
        body["furnace_pos"] = furnace_pos
    print(f"  [collect_smelt] requesting (furnace_pos={furnace_pos})...", flush=True)
    return _post_homunculus("/collect_smelt", body, timeout=90.0)


# Home is stored in-memory as an (x, y, z) tuple captured at home-set time.
# We deliberately avoid Baritone's #wp save/goto waypoints: #wp save creates
# a new waypoint per call (not an overwrite), so repeated table placements
# accumulate duplicate "home" waypoints and #wp goto then errors with
# "multiple waypoints were found". Coord-based goto avoids that entirely.
_HOME_POS: tuple[int, int, int] | None = None


def _baritone_goto(
    x: int,
    y: int,
    z: int,
    *,
    timeout_seconds: int,
    arrival_tolerance: int = 2,
    allow_place: bool = True,
    throwaway_items: list[str] | None = None,
    ensure_throwaway_in_hotbar: bool = False,
    goal_type: str | None = None,
) -> dict:
    """Drive Baritone to (x,y,z). Blocks until arrival / stuck / timeout.

    Returns the raw homunculus response. Always populates `final_position`
    when the player exists. Homunculus calls `cancelEverything()` on every
    non-success exit so no separate stop is needed.

    Protection parameters (homunculus v1.3+, ignored by v1.2):
    - `allow_place=False` → Baritone's `allowPlace` disabled for this call.
    - `throwaway_items=[...]` → restricts Baritone's `acceptableThrowawayItems`
      to the given list for this call (else Baritone defaults apply).
    - `ensure_throwaway_in_hotbar=True` → mod stages a matching throwaway
      block in hotbar slot 6 before pathing (most-plentiful wins), restores
      slot 6 on goto exit.

    Goal type (homunculus v1.4+, defaults to "block" when omitted):
    - `goal_type="y_level"` → GoalYLevel(y); x/z accepted but ignored by mod.
      Caller may still pass current x/z (schema uniformity); only y matters.

    Each parameter is sent only when non-default, keeping v1.2 a no-op.
    """
    body = {
        "x": x,
        "y": y,
        "z": z,
        "timeout_seconds": timeout_seconds,
        "arrival_tolerance": arrival_tolerance,
    }
    extras = []
    if goal_type is not None:
        body["goal_type"] = goal_type
        extras.append(f"goal_type={goal_type}")
    if not allow_place:
        body["allow_place"] = False
        extras.append("allow_place=False")
    if throwaway_items is not None:
        body["throwaway_items"] = throwaway_items
        extras.append(f"throwaway_items={throwaway_items}")
    if ensure_throwaway_in_hotbar:
        body["ensure_throwaway_in_hotbar"] = True
        extras.append("ensure_in_hotbar=True")
    tail = f" ({', '.join(extras)})" if extras else ""
    print(f"  [goto] ({x},{y},{z}) timeout={timeout_seconds}s tol={arrival_tolerance}{tail}", flush=True)
    return _post_homunculus("/baritone/goto", body, timeout=timeout_seconds + 10)


def _scan_blocks(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
) -> dict:
    params = {"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2}
    return _get_homunculus("/scan_blocks", params=params, timeout=10.0)


def _baritone_excavate(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
    *,
    timeout_seconds: int = 120,
) -> dict:
    """Clear an AABB to air via Baritone's clearArea. Synchronous.

    Volume cap 500 per call (mod-enforced). Preserves torches in the box.
    Returns the raw homunculus response; `remaining=0` is the success signal.
    """
    body = {
        "x1": x1, "y1": y1, "z1": z1,
        "x2": x2, "y2": y2, "z2": z2,
        "timeout_seconds": timeout_seconds,
    }
    print(f"  [excavate] ({x1},{y1},{z1})→({x2},{y2},{z2}) timeout={timeout_seconds}s", flush=True)
    return _post_homunculus("/baritone/excavate", body, timeout=timeout_seconds + 10)


def _trim_passable_in_aabb(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
    *,
    label: str,
) -> int:
    """Per-cell excavate any passable non-air blocks (tall grass, flowers,
    short grass, etc.) in an AABB. Returns the count of cells trimmed.

    Why this exists, in two parts:

    1. Baritone's bulk excavate (`clearArea`) builds a FillSchematic with
       target=AIR. For passable plants, the schematic comparison treats them
       as effectively satisfying the air target and skips them, so an AABB
       excavate leaves plants in place. A 1×1×1 excavate per passable cell
       sidesteps that — small enough that Baritone breaks them rather than
       treating them as schema-satisfied.
    2. Passable plants left inside the shelter occlude Baritone's
       placement raytrace (block-outline mode treats tall_grass as opaque
       for line-of-sight). A wall cell behind a tall_grass column won't get
       placed and the BuilderProcess silently no-ops each tick — the build
       stalls until Fill.java's deadline forces a retry.

    Called both before each plate fill (covers wall-position plants) and
    after the cavity excavate (covers plants inside the interior).
    """
    scan = _scan_blocks(x1, y1, z1, x2, y2, z2)
    passable = [b for b in scan.get("blocks", []) if b.get("passable")]
    if passable:
        print(f"  [trim] {label}: {len(passable)} passable cell(s)", flush=True)
    for b in passable:
        _baritone_excavate(b["x"], b["y"], b["z"], b["x"], b["y"], b["z"], timeout_seconds=15)
    return len(passable)


def _baritone_fill(
    block: str,
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
    *,
    timeout_seconds: int = 120,
) -> dict:
    """Fill all air cells in an AABB with `block` via Baritone's FillSchematic.

    Volume cap 500. Leaves existing solids alone (buildIgnoreExisting=true).
    REQUIRES the fill block in the hotbar — Baritone can't reach main inventory.
    Caller must `/equip` (or otherwise stage slot 6) before this.
    """
    body = {
        "block": block,
        "x1": x1, "y1": y1, "z1": z1,
        "x2": x2, "y2": y2, "z2": z2,
        "timeout_seconds": timeout_seconds,
    }
    print(f"  [fill] {block} ({x1},{y1},{z1})→({x2},{y2},{z2}) timeout={timeout_seconds}s", flush=True)
    return _post_homunculus("/baritone/fill", body, timeout=timeout_seconds + 10)


def _equip() -> dict:
    """Trigger homunculus auto-equip. Fixed slot scheme: 0 sword, 2 pickaxe,
    5 food, 6 building (best tier-0 cheap block — cobblestone preferred).

    The per-turn agent loop already calls this before planning, but
    intra-turn primitives that mine new blocks (e.g. build_shelter's
    excavate phase) re-invoke after to migrate freshly-mined cobble into
    slot 6 before the fill phase.
    """
    return _post_homunculus("/equip", {}, timeout=10.0)


# Tier-0 + tier-1 building blocks from Equipper's tier table, plus the Baritone
# stock defaults. Used as the throwaway-items allowlist during shelter builds:
# anything Equipper might stage into slot 6 should also be something the
# pathfinder can pillar with, else mid-build material switches trap the agent.
#
# Excluded as known-bad: mud (slow-walk physics → Baritone can't reach wall
# faces in placement budget), basalt + deepslate (pillar/axis-state blocks →
# Equipper places with wrong orientation). All three produce
# `failed plates: north=stuck, south=stuck` PARTIAL builds. Characterized at
# N=250 in 2026-05-19 iters=50 run: rates 9-15% vs ~85% for their non-axis /
# non-slow-walk siblings (packed_mud, smooth_basalt, cobbled_deepslate). Real
# rollouts rarely stage these as primary material; treating as known
# limitation rather than root-fixing Equipper/Baritone.
_SHELTER_THROWAWAY_ITEMS: tuple[str, ...] = (
    # Baritone stock
    "minecraft:dirt", "minecraft:cobblestone", "minecraft:netherrack", "minecraft:stone",
    # Equipper tier-0 (cheap throwaways)
    "minecraft:coarse_dirt", "minecraft:rooted_dirt", "minecraft:grass_block",
    "minecraft:podzol", "minecraft:mycelium", "minecraft:packed_mud",
    "minecraft:cobbled_deepslate", "minecraft:blackstone",
    # Equipper tier-1 (cheap stones)
    "minecraft:granite", "minecraft:diorite", "minecraft:andesite",
    "minecraft:tuff", "minecraft:smooth_basalt", "minecraft:end_stone",
    "minecraft:sandstone", "minecraft:red_sandstone", "minecraft:calcite",
    "minecraft:dripstone_block",
)


def _set_shelter_throwaway_items() -> bool:
    """Tell Baritone every cheap building block we might stage is a valid
    pillaring material. Without this, a fill that switches from dirt to (say)
    diorite mid-build leaves the pathfinder unable to ascend — it stalls in
    enormous A* searches looking for ground-level detours.
    """
    resp = _post_homunculus(
        "/baritone/throwaway_items",
        {"items": list(_SHELTER_THROWAWAY_ITEMS)},
        timeout=10.0,
    )
    if not resp.get("success"):
        print(
            f"  [throwaway] WARN: could not extend list "
            f"({resp.get('reason')}: {resp.get('message','')})",
            flush=True,
        )
        return False
    return True


# Accepted door variants for shelter doorway placement. All 12 vanilla wooden
# doors in 1.21.4 — picked over iron because they open on right-click (no
# redstone needed) and Baritone can interact with them during pathing.
_DOOR_ITEMS: tuple[str, ...] = (
    "minecraft:oak_door",
    "minecraft:spruce_door",
    "minecraft:birch_door",
    "minecraft:jungle_door",
    "minecraft:acacia_door",
    "minecraft:dark_oak_door",
    "minecraft:mangrove_door",
    "minecraft:cherry_door",
    "minecraft:pale_oak_door",
    "minecraft:bamboo_door",
    "minecraft:crimson_door",
    "minecraft:warped_door",
)


# Suffixes of block ids that occupy a cell but DON'T support a door above
# (no full top face). Door /place_at silently no-ops over these — observed
# 2026-05-13 on `oak_leaves` next to a shelter built into a treeline.
_NON_DOOR_SUPPORT_SUFFIXES: tuple[str, ...] = (
    "_leaves", "_fence", "_glass", "_pane", "_stairs", "_slab", "_carpet",
)
# Exact-id ids where suffix matching would over- or under-match. e.g.
# `minecraft:snow` is the thin layer (no support) but `snow_block` is full
# (supports a door).
_NON_DOOR_SUPPORT_EXACT: frozenset[str] = frozenset({
    "minecraft:cobweb", "minecraft:vine", "minecraft:snow",
    "minecraft:torch", "minecraft:wall_torch", "minecraft:soul_torch",
    "minecraft:lantern", "minecraft:soul_lantern",
    "minecraft:lily_pad", "minecraft:scaffolding", "minecraft:ladder",
    "minecraft:moss_carpet", "minecraft:hanging_roots",
    # Transparent full cubes: collision is full but isFaceSturdy(UP) = false,
    # so DoorBlock's mayPlace check rejects them.
    "minecraft:glass", "minecraft:ice", "minecraft:frosted_ice",
    "minecraft:tinted_glass",
})


def _supports_door(block_id: str | None) -> bool:
    """True iff a door placed at +Y above this block will stick.

    Heuristic — covers the common bad cases (leaves, fences, glass, stairs,
    slabs, carpet, single-layer snow, vines, etc.). Full solid blocks
    (stone, dirt, sandstone, planks, the throwaway materials) return True.
    """
    if not block_id or block_id == "minecraft:air":
        return False
    bid = block_id.lower()
    if bid in _NON_DOOR_SUPPORT_EXACT:
        return False
    return not any(bid.endswith(s) for s in _NON_DOOR_SUPPORT_SUFFIXES)


def _find_door_in_inventory() -> str | None:
    """Scan inventory and return the first door variant present, or None.

    Order follows `_DOOR_ITEMS` (oak first, nether woods last). Caller is
    responsible for hotbar staging before placement — this just answers
    "is there a door we could use?"
    """
    inv = _get_homunculus("/inventory")
    if inv.get("success") is False:
        return None
    have: set[str] = set()
    for slot in inv.get("main", []) or []:
        item = slot.get("id")
        if item:
            have.add(item)
    offhand = inv.get("offhand")
    if offhand and offhand.get("id"):
        have.add(offhand["id"])
    for door in _DOOR_ITEMS:
        if door in have:
            return door
    return None


def _set_allow_break(value: bool) -> bool:
    """Toggle Baritone's allowBreak pathfinder setting. Used to forbid
    breaking during shelter construction — without this, Baritone routinely
    breaks partially-built walls to take a shortcut up to the roof, leaving
    holes that patch passes can't fix without re-creating the same loop.

    The cost of forbid-break is that Baritone is forced to pillar inside the
    cavity instead of pathing outside; we clean up those pillars with a
    cavity re-excavate after construction.
    """
    resp = _post_homunculus(
        "/baritone/allow_break", {"value": value}, timeout=10.0,
    )
    if not resp.get("success"):
        print(
            f"  [allow_break] WARN: could not set value={value} "
            f"({resp.get('reason')}: {resp.get('message','')})",
            flush=True,
        )
        return False
    return True


def _count_shelter_buildables() -> dict[str, int]:
    """Sum inventory counts of any tier 0-2 buildable block.

    Used by build_shelter's pre-flight. Returns dict of bare item id
    (without minecraft: prefix) → total count across all stacks. Returns
    {} on inventory-read failure; caller treats empty as zero buildables.
    """
    inv_resp = _get_homunculus("/inventory")
    out: dict[str, int] = {}
    if not inv_resp or inv_resp.get("success") is False:
        return out
    for slot in (inv_resp.get("main") or []):
        item_id = (slot.get("id") or "").lower()
        bare = item_id.split(":")[-1]
        if bare in _SHELTER_BUILDABLE_BARE:
            out[bare] = out.get(bare, 0) + int(slot.get("count", 0))
    return out


def _stage_building() -> str | None:
    """Re-equip and return whatever tier-0 building block is now in hotbar
    slot 6, or None if no buildable exists in inventory.

    Equipper picks the most-plentiful tier-0 stack (cobble/dirt/deepslate/...)
    so when one material drops below another by count, the next call returns
    a different block id. That's the patchwork-shelter affordance: callers
    don't pick the material — they ask "what's available now?" between plates.
    """
    eq = _equip()
    if not eq.get("success"):
        return None
    return (eq.get("equipped") or {}).get("building")


def _set_home_waypoint() -> None:
    """Capture the player's current position as 'home' (in-memory only).

    Idempotent — overwrites the in-process tuple. No Baritone waypoint
    state is touched, so we don't pollute its persistent store.
    """
    global _HOME_POS
    pos = _position()
    if pos.get("success") is False:
        print(f"  [home] save skipped — position read failed: {pos.get('reason')}", flush=True)
        return
    xyz = _read_xyz(pos)
    if xyz is None:
        print("  [home] save skipped — malformed /position response", flush=True)
        return
    _HOME_POS = xyz
    print(f"  [home] saved at {xyz}", flush=True)


def _goto_home(
    timeout: int = 30,
    *,
    allow_place: bool = True,
    throwaway_items: list[str] | None = None,
    ensure_throwaway_in_hotbar: bool = False,
) -> bool:
    """Send the player back to home via /baritone/goto.

    Returns True if a goto was attempted (home is set), False if home has
    never been saved this session. Homunculus blocks until arrival / stuck
    / timeout — `timeout` is the worst-case cap.

    Callers from craft/smelt should derive the protection-policy triple
    from `_throwaway_policy(item, count)` and splat it in. No-op on
    homunculus v1.2; effective from v1.3.
    """
    if _HOME_POS is None:
        return False
    hx, hy, hz = _HOME_POS
    resp = _baritone_goto(
        hx, hy, hz,
        timeout_seconds=timeout,
        allow_place=allow_place,
        throwaway_items=throwaway_items,
        ensure_throwaway_in_hotbar=ensure_throwaway_in_hotbar,
    )
    reason = resp.get("reason", "unknown")
    if not resp.get("success"):
        print(f"  [home] goto {reason} (non-fatal): {resp.get('message', '')}", flush=True)
    return True


def handle_place(args: dict) -> str:
    item = args.get("item", "<missing>")
    if not item.startswith("minecraft:"):
        item = f"minecraft:{item}"
    data = _place_raw(item)
    if data.get("success"):
        if item == "minecraft:crafting_table":
            _set_home_waypoint()
        return f"placed {item} at {data.get('placed_at', [0, 0, 0])}"
    return f"FAILED: {data.get('reason', 'unknown')} ({data.get('message', '')})"


def _recipe_needs(item: str, count: int, _depth: int = 0) -> dict[str, int]:
    """Expand a recipe to its leaf-level ingredient needs.

    Walks `CRAFTING_RECIPES` recursively. Items not in the dict are treated
    as leaves (must be acquired via mining/smelting/etc). Multiplicative
    semantics: requesting `count` of `item` multiplies sub-recipe counts.
    Depth-capped at 8 to avoid pathological recursion on malformed recipes.
    """
    if _depth > 8 or item not in CRAFTING_RECIPES:
        return {item: count}
    needs: dict[str, int] = {}
    for ing, ing_count in CRAFTING_RECIPES[item]:
        sub = _recipe_needs(ing, ing_count * count, _depth=_depth + 1)
        for k, v in sub.items():
            needs[k] = needs.get(k, 0) + v
    return needs


def _throwaway_policy(item: str, count: int) -> tuple[bool, list[str] | None, bool]:
    """Decide goto protection level for an upcoming craft.

    Returns `(allow_place, throwaway_items, ensure_throwaway_in_hotbar)` —
    the three knobs on homunculus v1.3's /baritone/goto.

    Rule: any recipe ingredient that's in Baritone's throwaway set is
    *reserved* for the craft. Baritone is restricted to the remaining
    throwaway items for pillar-up / bridging on this trip. If everything's
    reserved, the goto runs with placement disabled entirely and may
    fail-unreachable — strictly better than silent inventory consumption.

    No buffer or inventory check: substrate-initiated gotos are short
    recipe-anchored trips, and Baritone's per-trip consumption is unbounded
    (49-block ascent ate 25 throwaway blocks in r4 T8). Trying to predict
    consumption with a buffer was the wrong abstraction. Long-horizon
    travel uses the `travel()` tool which keeps default behavior and
    consumes the throwaway surplus organically.
    """
    needs = _recipe_needs(item, count)
    protected = needs.keys() & THROWAWAY_ITEMS

    if not protected:
        return (True, None, False)

    permitted = sorted(THROWAWAY_ITEMS - protected)
    if not permitted:
        return (False, None, False)
    return (True, permitted, True)


def _count_inventory_items(item_ids: set[str]) -> int | None:
    """Sum counts of inventory items matching any of `item_ids`.

    Returns None on inventory-read failure so callers can distinguish
    a genuine zero count from a transport problem.
    """
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/inventory", timeout=5.0)
        resp.raise_for_status()
        inv = resp.json()
    except (requests.RequestException, ValueError):
        return None
    total = 0
    for slot in inv.get("main", []):
        if slot.get("id") in item_ids:
            total += slot.get("count", 0)
    offhand = inv.get("offhand")
    if offhand and offhand.get("id") in item_ids:
        total += offhand.get("count", 0)
    return total


def _resolve_wood_substitute(ing_id: str, count: int) -> str:
    """If `ing_id` is a *_planks species we lack, substitute one we have.

    Returns either `ing_id` unchanged (we already have enough material to
    produce that species) or an alternative planks id from a species the
    agent is actually holding. Vanilla recipes use the #planks tag so the
    parent craft accepts any species' planks.
    """
    if ing_id not in _PLANKS_LOG_BY_SPECIES:
        return ing_id
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/inventory", timeout=5.0)
        resp.raise_for_status()
        inv = resp.json()
    except (requests.RequestException, ValueError):
        return ing_id
    items_of_interest = (
        set(_PLANKS_LOG_BY_SPECIES.keys()) | set(_PLANKS_LOG_BY_SPECIES.values())
    )
    counts: dict[str, int] = {}
    for slot in inv.get("main", []):
        sid = slot.get("id")
        if sid in items_of_interest:
            counts[sid] = counts.get(sid, 0) + slot.get("count", 0)
    offhand = inv.get("offhand")
    if offhand and offhand.get("id") in items_of_interest:
        sid = offhand["id"]
        counts[sid] = counts.get(sid, 0) + offhand.get("count", 0)

    def available(planks_id: str, log_id: str) -> int:
        return counts.get(planks_id, 0) + 4 * counts.get(log_id, 0)

    log_id = _PLANKS_LOG_BY_SPECIES[ing_id]
    if available(ing_id, log_id) >= count:
        return ing_id
    for alt_planks, alt_log in _PLANKS_LOG_BY_SPECIES.items():
        if alt_planks == ing_id:
            continue
        if available(alt_planks, alt_log) >= count:
            print(
                f"  [craft] substituting {alt_planks} for {ing_id} "
                f"(need {count}, have {counts.get(alt_log, 0)} {alt_log.split(':')[-1]})",
                flush=True,
            )
            return alt_planks
    return ing_id


def _handle_mine_delta(
    label: str,
    args: dict,
    drops: set[str],
    miner,
    *,
    fair_miner=None,
) -> str:
    """Shared delta-semantics handler for mine_wood / mine_stone.

    Reads pre-mining inventory, sets Baritone's cumulative target to
    `before + delta`, then reports actual acquired count after mining.
    Avoids the cumulative-target trap where repeated calls instant-succeed.

    Partial-acquire is reported as progress, not FAILED. Baritone often
    times out after mining some but not all of the target (e.g. ran out
    of nearby trees) — the items are in inventory regardless, so calling
    that FAILED misleads the agent into switching strategies prematurely.

    `fair` (bool, default False) routes to `fair_miner` instead of the
    baritone-targeted `miner`. Fair-mode digs a blind 1×2 tunnel from the
    agent's current xz/y; trade-off is local-only mining, no cross-chunk
    target-seeking. Useful for abundant materials (stone) where baritone's
    chunk-wide search picks pathological targets deep underground.
    """
    delta = min(int(args.get("quantity", 1)), MAX_QUANTITY)
    fair = bool(args.get("fair", False))
    if fair and fair_miner is None:
        fair = False  # no fair impl registered; silent fall-through
    before = _count_inventory_items(drops)
    if before is None:
        return f"FAILED: couldn't read inventory before {label}"
    mode = "fair" if fair else "baritone"
    print(f"  [{label}] mode={mode} before={before}, delta={delta}", flush=True)
    if fair:
        # fair_miner takes the delta directly (turn ends on inventory-delta hit).
        result = fair_miner(delta)
    else:
        target = before + delta
        result = miner(target)
    after = _count_inventory_items(drops)
    if after is None:
        after = before  # transport blip — best-effort report
    acquired = max(0, after - before)
    if acquired > 0:
        # Got something. Distinguish full success (Baritone reached target)
        # from partial (cycle ended without target) so the agent knows whether
        # to call again for more or accept the partial.
        if result is None:
            return (
                f"PARTIAL: acquired {acquired} of {delta} {label}-drops "
                f"(now have {after}). Cycle ended before target — call "
                f"{label} again to keep gathering, or proceed if {after} is "
                f"enough."
            )
        return f"acquired {acquired} more (now have {after} {label}-drops; last type mined: {result})"
    if result is None:
        return f"FAILED: no candidate reachable for {label} (acquired 0)"
    return f"acquired 0 more (already had {after} {label}-drops — target was already met)"


def handle_mine_wood(args: dict) -> str:
    return _handle_mine_delta("mine_wood", args, LOG_DROPS, mine_any_log,
                              fair_miner=tunnel_for_logs)


def handle_mine_stone(args: dict) -> str:
    # mine_stone FORCES fair-mode (blind tunnel at player's current y).
    # User decision 2026-05-15: baritone's chunk-wide /mine for stone picks
    # pathological deep targets — agents shouldn't even have the option to
    # use it. Other ores keep the toggle since their rare/sparse drops
    # genuinely benefit from baritone's chunk-scan.
    args = {**args, "fair": True}
    return _handle_mine_delta("mine_stone", args, STONE_DROPS, mine_any_stone,
                              fair_miner=tunnel_for_stone)


def handle_mine_iron(args: dict) -> str:
    return _handle_mine_delta("mine_iron", args, IRON_DROPS, mine_any_iron,
                              fair_miner=tunnel_for_iron)


def handle_mine_diamond(args: dict) -> str:
    return _handle_mine_delta("mine_diamond", args, DIAMOND_DROPS, mine_any_diamond,
                              fair_miner=tunnel_for_diamond)


def handle_mine_coal(args: dict) -> str:
    return _handle_mine_delta("mine_coal", args, COAL_DROPS, mine_any_coal,
                              fair_miner=tunnel_for_coal)


def _ensure_furnace_placed() -> str:
    """Place a furnace within reach. Craft one first if not in inventory.

    Furnace is a 3×3 recipe (8 cobblestone), so its own craft requires a
    crafting_table — `_craft_recursive` handles that subgoal recursively.
    """
    place_resp = _place_raw("minecraft:furnace")
    if place_resp.get("success"):
        return f"placed furnace at {place_resp.get('placed_at')}"

    if place_resp.get("reason") == "not_in_inventory":
        sub = _craft_recursive("minecraft:furnace", 1)
        if sub.startswith("FAILED"):
            return f"FAILED: couldn't craft furnace: {sub}"
        place_resp = _place_raw("minecraft:furnace")
        if place_resp.get("success"):
            return f"placed furnace at {place_resp.get('placed_at')}"

    return f"FAILED: place furnace: {place_resp.get('reason')} ({place_resp.get('message', '')})"


def handle_smelt(args: dict) -> str:
    input_item = args.get("input", "<missing>")
    count = min(int(args.get("count", 1)), MAX_QUANTITY)
    fuel = args.get("fuel")
    location = args.get("location", "auto")
    if not input_item.startswith("minecraft:"):
        input_item = f"minecraft:{input_item}"
    if fuel and not fuel.startswith("minecraft:"):
        fuel = f"minecraft:{fuel}"
    print(f"  [handle_smelt] {count}x {input_item} (location={location})", flush=True)

    # We don't know whether home has a usable furnace; if it doesn't, the
    # downstream fallback will craft+place one (8 cobblestone). Use furnace
    # as the protection target so the cobble survives the home trip.
    allow_place, throwaway, ensure_hotbar = _throwaway_policy("minecraft:furnace", 1)

    if location == "home":
        if not _goto_home(
            allow_place=allow_place,
            throwaway_items=throwaway,
            ensure_throwaway_in_hotbar=ensure_hotbar,
        ):
            print("  [home] no home set yet — proceeding in-situ", flush=True)

    home_attempted = False
    MAX_ATTEMPTS = 3
    for _ in range(MAX_ATTEMPTS):
        resp = _smelt_raw(input_item, count, fuel)
        if resp.get("success"):
            fpos = resp.get("furnace_pos") or [0, 0, 0]
            expected = resp.get("expected_output", {})
            out_count = expected.get("count", count)
            out_id = expected.get("id", input_item)
            eta = resp.get("eta_seconds", count * 10)
            fl = resp.get("fuel_loaded", [])
            if isinstance(fl, dict):
                fl = [fl]
            fuel_str = ", ".join(f"{f.get('count', 0)}x {f.get('id', '?')}" for f in fl) or "auto"
            # Fuel-cap surfacing (homunculus v1.2 limitation #1): if only a
            # subset of the requested batch was started, tell the agent so it
            # mines more fuel and re-smelts the remainder rather than walking
            # off thinking the full batch is in flight.
            shortfall_note = ""
            if isinstance(out_count, int) and out_count < count:
                missing = count - out_count
                shortfall_note = (
                    f" PARTIAL: requested {count}, only {out_count} fueled "
                    f"({missing} short — mine more fuel and call smelt() again "
                    f"to cook the rest)."
                )
            hom_msg = resp.get("message")
            msg_tail = f" [{hom_msg}]" if hom_msg else ""
            return (
                f"smelt started: {out_count}x {out_id} in furnace at "
                f"({fpos[0]},{fpos[1]},{fpos[2]}); ETA ~{eta}s (fuel: {fuel_str}). "
                f"Continue with other actions; call collect_smelt() when ready."
                f"{shortfall_note}{msg_tail}"
            )

        reason = resp.get("reason", "unknown")
        message = resp.get("message", "")

        if reason == "requires_furnace" and not resp.get("furnace_nearby"):
            # Silent fall-back: if home is set and we haven't tried it this
            # call, return there before resorting to placing a fresh furnace.
            # No-op if home was never saved.
            if _HOME_POS is not None and not home_attempted:
                home_attempted = True
                _goto_home(
                    allow_place=allow_place,
                    throwaway_items=throwaway,
                    ensure_throwaway_in_hotbar=ensure_hotbar,
                )
                continue
            ensure = _ensure_furnace_placed()
            if ensure.startswith("FAILED"):
                return f"FAILED: couldn't ensure furnace for smelt: {ensure}"
            continue

        if reason == "not_in_inventory":
            # v1.2 /smelt auto-places from inventory but won't craft. Mirrors
            # the craft() abstraction's handling of crafting_table: caller
            # should never need to pre-acquire the substrate-required block.
            # No _place_raw — /smelt does the placement itself on retry.
            sub = _craft_recursive("minecraft:furnace", 1)
            if sub.startswith("FAILED"):
                return f"FAILED: couldn't craft furnace for smelt: {sub}"
            continue

        if reason == "missing_input":
            missing = resp.get("missing", [])
            mstr = ", ".join(f"{m['count']}x {m['id']}" for m in missing)
            return f"FAILED: missing input ({mstr}) — acquire it first (e.g. mine_iron)"

        if reason == "missing_fuel":
            missing = resp.get("missing", [])
            mstr = ", ".join(f"{m['count']}x {m['id']}" for m in missing)
            return f"FAILED: missing fuel ({mstr}) — get logs/planks/coal first"

        if reason == "no_recipe":
            return f"FAILED: no smelting recipe for {input_item}"
        if reason == "unknown_item":
            return f"FAILED: unknown item {input_item}"

        return f"FAILED: {reason} ({message})"

    return f"FAILED: exhausted retries for smelt"


def handle_collect_smelt(args: dict) -> str:
    raw_pos = args.get("furnace_pos")
    furnace_pos: list[int] | None = None
    if raw_pos is not None:
        try:
            furnace_pos = [int(raw_pos[0]), int(raw_pos[1]), int(raw_pos[2])]
        except (TypeError, ValueError, IndexError):
            return "FAILED: furnace_pos must be [x, y, z] integers"

    resp = _collect_smelt_raw(furnace_pos)

    # Fallback: if no furnace_pos was given AND homunculus replied "no active
    # smelts", consult /smelt_status directly and retry with the first listed
    # furnace position. Observed 2026-05-14 (Haiku R5): the no-arg path returned
    # no_active_smelts repeatedly while /smelt_status was reporting an active
    # furnace 4 blocks away — likely a sync gap between /collect_smelt's
    # registry view and /smelt_status's. Retrying with the explicit position
    # resolves the same call without bothering the LLM.
    if (
        furnace_pos is None
        and not resp.get("success")
        and resp.get("reason") == "no_active_smelts"
    ):
        status_resp = _get_homunculus("/smelt_status")
        smelts = (status_resp or {}).get("smelts") or []
        if smelts:
            fp = smelts[0].get("furnace_pos")
            if isinstance(fp, list) and len(fp) == 3:
                fallback = [int(fp[0]), int(fp[1]), int(fp[2])]
                print(f"  [collect_smelt] retry with /smelt_status furnace {fallback}", flush=True)
                resp = _collect_smelt_raw(fallback)
                furnace_pos = fallback
    if resp.get("success"):
        fpos = resp.get("furnace_pos") or [0, 0, 0]
        collected = resp.get("collected", [])
        if isinstance(collected, dict):
            collected = [collected]
        coll_str = ", ".join(f"{c.get('count', 0)}x {c.get('id', '?')}" for c in collected) or "nothing"
        still = resp.get("still_cooking", 0)
        status = resp.get("status", "unknown")
        if still > 0:
            eta = resp.get("eta_seconds", still * 10)
            return (
                f"collected {coll_str} from furnace ({fpos[0]},{fpos[1]},{fpos[2]}); "
                f"{still} still cooking (~{eta}s, status={status}). Call collect_smelt() "
                f"again later to pull the rest."
            )
        return (
            f"collected {coll_str} from furnace ({fpos[0]},{fpos[1]},{fpos[2]}); "
            f"smelt complete (status={status})."
        )

    reason = resp.get("reason", "unknown")
    message = resp.get("message", "")

    if reason == "no_active_smelts":
        return "FAILED: no active smelts to collect from — call smelt() first"
    if reason == "not_in_registry":
        return (
            f"FAILED: furnace_pos {furnace_pos} isn't a registered smelt — "
            f"check 'Active smelts' for valid positions, or omit furnace_pos "
            f"to collect from the nearest ready furnace"
        )
    if reason == "furnace_unreachable":
        return (
            f"FAILED: couldn't path to the furnace ({message}) — "
            f"travel closer in the right direction then call collect_smelt() "
            f"again. If the furnace is in another dimension, it isn't recoverable in v1."
        )
    if reason == "furnace_destroyed":
        return (
            f"FAILED: the furnace was destroyed; smelted items are lost. "
            f"The registry entry has been dropped."
        )
    if reason == "nothing_ready":
        return (
            f"reconciled but nothing collectable: the furnace is still cooking "
            f"with 0 ready. Do other work and call collect_smelt() again once "
            f"'Active smelts' shows READY or PARTIAL."
        )
    return f"FAILED: collect_smelt: {reason} ({message})"


def _ensure_crafting_table_placed(_depth: int) -> str:
    """Place a crafting_table within reach. Craft one first if not in inventory.

    crafting_table is a 2×2 recipe so its own craft never needs a table —
    no reentrancy hazard. Each successful place updates the 'home' waypoint
    so home tracks the most recently established craft station.
    """
    place_resp = _place_raw("minecraft:crafting_table")
    if place_resp.get("success"):
        _set_home_waypoint()
        return f"placed crafting_table at {place_resp.get('placed_at')}"

    if place_resp.get("reason") == "not_in_inventory":
        sub = _craft_recursive("minecraft:crafting_table", 1, _depth=_depth + 1)
        if sub.startswith("FAILED"):
            return f"FAILED: couldn't craft crafting_table: {sub}"
        place_resp = _place_raw("minecraft:crafting_table")
        if place_resp.get("success"):
            _set_home_waypoint()
            return f"placed crafting_table at {place_resp.get('placed_at')}"

    return f"FAILED: place crafting_table: {place_resp.get('reason')} ({place_resp.get('message', '')})"


def _craft_recursive(item: str, count: int, _depth: int = 0) -> str:
    """Reactive recursive crafter.

    Strategy: try the craft. If homunculus reports missing_ingredients, recurse
    on each missing item that we have a recipe for. If it reports
    requires_crafting_table with no nearby table, craft+place one. Then retry.

    No proactive ingredient multiplication — homunculus knows what's actually
    missing given current inventory, so we don't over-craft.
    """
    if _depth > 6:
        return f"FAILED: recursion depth exceeded for {item}"

    home_attempted = False
    MAX_ATTEMPTS = 6
    last_missing_sig: tuple | None = None
    sig_repeat_streak = 0
    for _ in range(MAX_ATTEMPTS):
        resp = _craft_raw(item, count)
        if resp.get("success"):
            crafted = resp.get("crafted", {})
            return f"crafted {crafted.get('count', count)}x {crafted.get('id', item)}"

        reason = resp.get("reason", "unknown")
        message = resp.get("message", "")

        if reason == "missing_ingredients":
            missing = resp.get("missing", [])
            if not missing:
                return f"FAILED: missing_ingredients with empty list for {item}"
            # "Same missing sig as last attempt" can be a false-positive when
            # a parallel sub-place (e.g. crafting_table consuming 4 planks)
            # silently drained the inventory our sub-craft just topped up.
            # Require 2 consecutive repeats (3 same sigs total) before
            # bailing as a real tag-canonical loop.
            current_sig = tuple(sorted((m["id"], m["count"]) for m in missing))
            if current_sig == last_missing_sig:
                sig_repeat_streak += 1
            else:
                sig_repeat_streak = 0
            if sig_repeat_streak >= 2:
                ids = ", ".join(f"{m['count']}x {m['id']}" for m in missing)
                return (
                    f"FAILED: substitution loop for {item} — homunculus repeatedly "
                    f"reports missing {ids} even after sub-crafts. Likely an old "
                    f"homunculus jar without tag-aware ingredient matching; restart "
                    f"MC after pulling the latest jar"
                )
            last_missing_sig = current_sig
            for m in missing:
                ing_id = m["id"]
                ing_count = m["count"]
                ing_id = _resolve_wood_substitute(ing_id, ing_count)
                if ing_id not in CRAFTING_RECIPES:
                    return (
                        f"FAILED: {item} needs {ing_count}x {ing_id}; "
                        f"no recipe — must be acquired (e.g. mining)"
                    )
                sub = _craft_recursive(ing_id, ing_count, _depth=_depth + 1)
                if sub.startswith("FAILED"):
                    return f"FAILED: while crafting {ing_id} for {item}: {sub}"
            # ingredients secured — loop and retry the main craft
            continue

        if reason == "requires_crafting_table" and not resp.get("crafting_table_nearby"):
            # Silent fall-back at the outermost call only: if home is set and
            # we haven't tried it this call, return there before placing a new
            # table. No-op when home was never saved.
            if _depth == 0 and _HOME_POS is not None and not home_attempted:
                home_attempted = True
                # Recipe is in scope as `item`; protect throwaway-set ingredients.
                ap, tw, ehb = _throwaway_policy(item, count)
                _goto_home(
                    allow_place=ap,
                    throwaway_items=tw,
                    ensure_throwaway_in_hotbar=ehb,
                )
                continue
            table_result = _ensure_crafting_table_placed(_depth)
            if table_result.startswith("FAILED"):
                return f"FAILED: couldn't ensure crafting_table for {item}: {table_result}"
            continue

        if reason == "no_recipe":
            return f"FAILED: homunculus has no recipe for {item} (recipe may not be unlocked)"
        if reason == "unknown_item":
            return f"FAILED: unknown item {item}"

        return f"FAILED: {reason} ({message})"

    return f"FAILED: exhausted retries for {item}"


def handle_craft(args: dict) -> str:
    item = args.get("item", "<missing>")
    quantity = int(args.get("quantity", 1))
    location = args.get("location", "auto")
    if not item.startswith("minecraft:"):
        item = f"minecraft:{item}"
    print(f"  [handle_craft] {quantity}x {item} (location={location})", flush=True)
    if location == "home":
        ap, tw, ehb = _throwaway_policy(item, quantity)
        if not _goto_home(
            allow_place=ap,
            throwaway_items=tw,
            ensure_throwaway_in_hotbar=ehb,
        ):
            print("  [home] no home set yet — proceeding in-situ", flush=True)
    return _craft_recursive(item, quantity)


def _read_xyz(pos: dict) -> tuple[int, int, int] | None:
    """Coerce a /position response's x,y,z to ints. None on malformed."""
    try:
        return int(pos["x"]), int(pos["y"]), int(pos["z"])
    except (KeyError, ValueError, TypeError):
        return None


def _final_xyz(resp: dict) -> tuple[int, int, int] | None:
    """Pull integer xyz out of /baritone/goto's `final_position` list."""
    fp = resp.get("final_position")
    if not isinstance(fp, (list, tuple)) or len(fp) < 3:
        return None
    try:
        return int(fp[0]), int(fp[1]), int(fp[2])
    except (ValueError, TypeError):
        return None


def handle_surface(args: dict) -> str:
    print(f"  [handle_surface]", flush=True)
    pos = _position()
    if pos.get("success") is False:
        return f"FAILED: read position: {pos.get('reason')} ({pos.get('message','')})"
    xyz = _read_xyz(pos)
    if xyz is None:
        return "FAILED: malformed /position response"
    px, py, pz = xyz

    scan = _scan_column()
    if scan.get("success") is False:
        return f"FAILED: scan column: {scan.get('reason')} ({scan.get('message','')})"
    surface_y = scan.get("surface_y")
    if surface_y is None:
        return "FAILED: no open sky in this column (fully enclosed) — try mining up or travel() to another column"

    dy = surface_y - py
    if abs(dy) <= 1:
        return f"already at surface (y={py}, surface_y={surface_y})"

    # Chunked ascent: clamp this call to SURFACE_MAX_PER_CALL blocks. Same
    # rationale as descend chunking — long Δy goto's PARTIAL out on Baritone.
    chunk_target_y = min(surface_y, py + SURFACE_MAX_PER_CALL)
    chunk_dy = chunk_target_y - py
    print(f"  [surface] at ({px},{py},{pz}) → chunk_y={chunk_target_y} (Δy=+{chunk_dy}, surface={surface_y})", flush=True)
    timeout = int(min(120, max(20, chunk_dy * 2)))
    resp = _baritone_goto(px, chunk_target_y, pz, timeout_seconds=timeout, goal_type="y_level")
    final = _final_xyz(resp) or xyz
    _, new_y, _ = final

    remaining = max(0, surface_y - new_y)
    if abs(new_y - chunk_target_y) <= 2:
        if remaining <= 2:
            return f"surfaced to y={new_y} (target was {surface_y})"
        return (
            f"ascended to y={new_y} (surface at y={surface_y}, {remaining} more — "
            f"call surface() again to continue)"
        )
    reason = resp.get("reason", "unknown")
    return (
        f"PARTIAL: at y={new_y} (started y={py}, chunk target y={chunk_target_y}, "
        f"surface y={surface_y}) — Baritone {reason}; switch strategy "
        f"(mine_stone(8) by hand, or travel to a different column)"
    )


def handle_descend(args: dict) -> str:
    try:
        target_y = int(args.get("target_y"))
    except (TypeError, ValueError):
        return "FAILED: target_y must be an integer"

    print(f"  [handle_descend] target_y={target_y}", flush=True)
    pos = _position()
    if pos.get("success") is False:
        return f"FAILED: read position: {pos.get('reason')} ({pos.get('message','')})"
    xyz = _read_xyz(pos)
    if xyz is None:
        return "FAILED: malformed /position response"
    px, py, pz = xyz

    if target_y >= py:
        return f"FAILED: target_y={target_y} is not below current y={py} — use surface() to go up"

    dy = py - target_y
    if dy <= 1:
        return f"already near target (y={py}, target_y={target_y})"

    # Chunked descent: clamp this call's goto to DESCEND_MAX_PER_CALL blocks.
    # Long Δy descends consistently PARTIAL out on Baritone; chunking gives the
    # planner control back every N blocks while letting it just call descend
    # again toward the same target_y to keep going.
    chunk_target_y = max(target_y, py - DESCEND_MAX_PER_CALL)
    chunk_dy = py - chunk_target_y
    print(f"  [descend] at ({px},{py},{pz}) → chunk_y={chunk_target_y} (Δy=-{chunk_dy}, final target={target_y})", flush=True)
    timeout = int(min(120, max(20, chunk_dy * 2)))
    resp = _baritone_goto(px, chunk_target_y, pz, timeout_seconds=timeout, goal_type="y_level")
    final = _final_xyz(resp) or xyz
    _, new_y, _ = final

    remaining = max(0, new_y - target_y)
    if abs(new_y - chunk_target_y) <= 2:
        if remaining <= 2:
            return f"descended to y={new_y} (target was {target_y})"
        return (
            f"descended to y={new_y} (target was {target_y}, {remaining} more — "
            f"call descend({target_y}) again to continue)"
        )
    reason = resp.get("reason", "unknown")
    return (
        f"PARTIAL: at y={new_y} (started y={py}, chunk target y={chunk_target_y}, "
        f"final target y={target_y}) — Baritone {reason}; switch strategy "
        f"(mine_stone(8) by hand, or travel to a different column)"
    )


_TRAVEL_HAZARD_IDS: frozenset[str] = frozenset({
    "minecraft:lava",
})

# Clamp the agent to N blocks before the nearest hazard. Pulls them back
# far enough that one more autonomous tick (e.g., Baritone overshoot) is
# still safe. 2 blocks chosen because Baritone's `arrival_tolerance` for
# /baritone/goto is 2 — anything tighter lands inside the tolerance band.
_TRAVEL_HAZARD_SAFETY_MARGIN: int = 2


def _travel_hazard_scan(
    px: int, py: int, pz: int,
    dx_unit: int, dz_unit: int,
    distance: int,
) -> list[tuple[int, int, int, str, int]]:
    """Pre-flight scan along a cardinal travel corridor for surface-level
    hazards (currently just lava).

    Scans the chunks the path crosses, vertical slice player Y ±1 (the
    cells the player would actually walk through). Per-chunk volume:
    16×16×3 = 768, well under the 2000 cap. Hazards that are within ±3
    blocks perpendicular of the path line and inside [0, distance] along
    it are returned, sorted by along-axis distance so callers get the
    nearest one first.

    Returns a list of ``(x, y, z, id, along_blocks)``. Empty list means
    the corridor is clear (or every touched chunk failed to scan; we treat
    "no signal" as "no hazard" rather than blocking on a transport hiccup).
    """
    path_chunks: set[tuple[int, int]] = set()
    for step in range(0, distance + 1, 4):
        x = px + dx_unit * step
        z = pz + dz_unit * step
        path_chunks.add((x >> 4, z >> 4))
    end_x = px + dx_unit * distance
    end_z = pz + dz_unit * distance
    path_chunks.add((end_x >> 4, end_z >> 4))

    hazards: list[tuple[int, int, int, str, int]] = []
    for cx, cz in path_chunks:
        x1, z1 = cx * 16, cz * 16
        x2, z2 = x1 + 15, z1 + 15
        y1, y2 = py - 1, py + 1
        scan = _scan_blocks(x1, y1, z1, x2, y2, z2)
        if scan.get("success") is False:
            continue
        for b in scan.get("blocks", []):
            bid = b.get("id")
            if bid not in _TRAVEL_HAZARD_IDS:
                continue
            bx, by, bz = b["x"], b["y"], b["z"]
            along = (bx - px) * dx_unit + (bz - pz) * dz_unit
            perp = abs((bx - px) * dz_unit - (bz - pz) * dx_unit)
            if 0 <= along <= distance and perp <= 3:
                hazards.append((bx, by, bz, bid, along))

    hazards.sort(key=lambda h: h[4])
    return hazards


def handle_travel(args: dict) -> str:
    direction = str(args.get("direction", "")).lower()
    try:
        distance = int(args.get("distance", 0))
    except (ValueError, TypeError):
        return "FAILED: distance must be an integer"
    distance = min(distance, TRAVEL_MAX_DISTANCE)
    if distance <= 0:
        return "FAILED: distance must be a positive integer"

    if direction == "north":
        dx_unit, dz_unit = 0, -1
    elif direction == "south":
        dx_unit, dz_unit = 0, 1
    elif direction == "east":
        dx_unit, dz_unit = 1, 0
    elif direction == "west":
        dx_unit, dz_unit = -1, 0
    else:
        return f"FAILED: unknown direction '{direction}' (must be north/south/east/west)"

    pos = _position()
    if pos.get("success") is False:
        return f"FAILED: read position: {pos.get('reason')} ({pos.get('message','')})"
    xyz = _read_xyz(pos)
    if xyz is None:
        return "FAILED: malformed /position response"
    px, py, pz = xyz

    # Travel-scout interlock: pre-flight scan for lava (and future hazard
    # block types) along the corridor. Clamp the distance to stop short of
    # the nearest hazard rather than refusing the call — honors the
    # no-dispatch-guards principle: the action still happens, but safely.
    hazards = _travel_hazard_scan(px, py, pz, dx_unit, dz_unit, distance)
    clamp_note = ""
    if hazards:
        hx, hy, hz, hid, along = hazards[0]
        safe_distance = max(0, along - _TRAVEL_HAZARD_SAFETY_MARGIN)
        short_id = hid.removeprefix("minecraft:")
        if safe_distance <= 0:
            return (
                f"FAILED: {short_id} at ({hx},{hy},{hz}) is right next to you "
                f"({along} blocks {direction}); refusing zero-distance travel. "
                f"Try a different direction."
            )
        if safe_distance < distance:
            print(
                f"  [travel] hazard clamp: {short_id} at ({hx},{hy},{hz}) "
                f"{along} blocks {direction}; clamping distance {distance}→{safe_distance}",
                flush=True,
            )
            clamp_note = (
                f" (clamped from {distance} — {short_id} at ({hx},{hy},{hz}) "
                f"is {along} blocks {direction}; stopped {_TRAVEL_HAZARD_SAFETY_MARGIN} short)"
            )
            distance = safe_distance

    dx = dx_unit * distance
    dz = dz_unit * distance
    tx = px + dx
    tz = pz + dz
    print(f"  [travel] {direction} {distance}: ({px},{py},{pz}) → ({tx},{py},{tz})", flush=True)
    timeout = int(min(60, max(15, distance)))
    resp = _baritone_goto(tx, py, tz, timeout_seconds=timeout)
    final = _final_xyz(resp) or xyz
    fx, fy, fz = final

    moved = max(abs(fx - px), abs(fz - pz))
    if moved < 2:
        reason = resp.get("reason", "unknown")
        return f"FAILED: barely moved ({moved} blocks {direction}) from ({px},{py},{pz}) — Baritone {reason}; try different direction{clamp_note}"
    return f"traveled {direction}: moved {moved} blocks (target {distance}); now at ({fx},{fy},{fz}){clamp_note}"


def _fill_plate(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int,
    *,
    label: str,
    timeout_seconds: int = 30,
) -> tuple[str | None, int, str | None]:
    """Pre-clear passable obstructions then fill one shelter plate.

    Returns (used_block, cells_placed, failure_reason_or_None).

    Pre-clear is surgical: we scan the plate AABB and excavate only the
    *passable* non-air cells (tall grass, flowers) one at a time. Baritone's
    fill skips any non-air cell — so passables would leave wall holes — but
    we deliberately leave SOLID non-air cells alone. That's load-bearing for
    the underground case: a shelter dug into a hillside or down in stone is
    mostly already-solid, and bulk-clearing the wall AABB would just create
    work for the fill to undo. Per-cell Baritone trips are unfortunate but
    typically 0-few cells per plate (only walls that intersect the surface
    layer have plants).

    Material retry: Baritone's fill is pinned to a single block id; if the
    staged stack runs dry, Baritone pauses with "Missing materials". The
    homunculus side surfaces this as `missing_block` only on the pre-flight
    hotbar check — mid-fill exhaustion comes back as `partial` (builder
    went inactive with cells still air). In either case the recovery is
    the same: re-stage via `_equip` and retry. Equipper may hand us back
    the SAME block id (slot 6 was empty, main inv still had more of what
    we just used → SWAP it back in) or a different one (we drained that
    material's supply everywhere). Both are valid forward progress, so we
    don't gate the retry on material identity — we gate on cell progress.
    The loop terminates when `remaining` stops shrinking across attempts
    (genuinely stuck) or Equipper can't stage anything (inventory dry).
    """
    _trim_passable_in_aabb(x1, y1, z1, x2, y2, z2, label=label)

    block = _stage_building()
    if block is None:
        return None, 0, "no_buildable_in_inventory"

    last_resp: dict = {}
    prev_remaining: int | None = None
    for _ in range(5):
        last_resp = _baritone_fill(block, x1, y1, z1, x2, y2, z2, timeout_seconds=timeout_seconds)
        if last_resp.get("success"):
            placed = last_resp.get("volume", 0) - last_resp.get("remaining", 0)
            return block, placed, None
        cur_remaining = last_resp.get("remaining", -1)
        # No cells placed since the prior attempt → not a material problem.
        # Re-staging won't help; bail with the current reason.
        if prev_remaining is not None and cur_remaining >= prev_remaining:
            break
        prev_remaining = cur_remaining
        new_block = _stage_building()
        if new_block is None:
            break
        if new_block != block:
            print(
                f"  [fill_plate] {label}: switching {block} → {new_block} "
                f"(remaining={cur_remaining}, reason={last_resp.get('reason')})",
                flush=True,
            )
        else:
            print(
                f"  [fill_plate] {label}: re-stacked {block} from main inv "
                f"(remaining={cur_remaining}, reason={last_resp.get('reason')})",
                flush=True,
            )
        block = new_block

    placed = last_resp.get("volume", 0) - last_resp.get("remaining", 0)
    return block, placed, last_resp.get("reason", "unknown")


def _inspect_shelter_shell(
    aabbs: list[tuple[int, int, int, int, int, int]],
    exclude: set[tuple[int, int, int]] | None = None,
) -> tuple[int, int, list[tuple[int, int, int]], str | None]:
    """Walk every cell in the shelter shell and verify it is solid.

    Returns (solid_count, expected_count, holes, scan_error). A "hole" is
    any expected cell that is currently air OR passable (tall_grass etc.
    that survived the trim and blocks the placement raytrace) — both fail
    the "structure intact" question we're trying to answer.

    `exclude` removes cells from the expected set — used for the door
    slot (intentionally not solid) so it doesn't show up as a hole.

    On /scan_blocks failure (chunks not loaded, client error) scan_error
    is set and the caller surfaces it instead of acting on the zeros.

    Why not trust Baritone's per-plate counts: the count is the builder's
    own view of remaining cells in its schematic; if a cell is broken
    after the plate succeeds (creeper, enderman, even Baritone digging
    during a later phase's pathfinding) the count doesn't update. Walking
    the real block state is the only way to surface degradation.
    """
    expected: set[tuple[int, int, int]] = set()
    for a, b, c, d, e, f in aabbs:
        for x in range(a, d + 1):
            for y in range(b, e + 1):
                for z in range(c, f + 1):
                    expected.add((x, y, z))
    if exclude:
        expected -= exclude
    if not expected:
        return 0, 0, [], None

    xs = [p[0] for p in expected]
    ys = [p[1] for p in expected]
    zs = [p[2] for p in expected]
    scan = _scan_blocks(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    if scan.get("success") is False:
        return 0, len(expected), [], f"{scan.get('reason')}: {scan.get('message', '')}"

    solid: set[tuple[int, int, int]] = set()
    for blk in scan.get("blocks", []):
        if not blk.get("passable"):
            solid.add((blk["x"], blk["y"], blk["z"]))

    holes = sorted(expected - solid)
    return len(expected) - len(holes), len(expected), holes, None


def handle_build_shelter(args: dict) -> str:
    """Compose excavate + fill into a patchwork-safe shelter primitive.

    Carves a 5×2×5 cavity around the player, then places six fill plates
    forming solid floor, ceiling, and four full-width perimeter walls.

    Each plate runs through `_fill_plate`, which surgically clears only
    *passable* obstructions (grass/flowers) before filling — solid terrain
    inside a plate's AABB is left alone and counts as free wall coverage.
    That makes the same primitive work for surface boxes (mostly air,
    some plants) AND underground hollows (mostly stone, a few air gaps).
    Above-ground plate cell counts (25 floor / 25 ceiling / 10 each wall)
    are upper bounds; underground we typically place far fewer.

    Build order is floor + 4 walls → reentry goto → ceiling. While the
    ceiling is still open, Baritone can pillar-and-drop the player back
    into the box if a prior fill carried them outside. If reentry fails,
    we leave the ceiling open and return PARTIAL — better to surface the
    issue than seal the agent out.

    Patchwork material switching: each plate calls `_stage_building()` via
    `_fill_plate` to get whatever tier-0 buildable is most plentiful right
    now (cobble/dirt/...), so as one stack drains across plates the slot-6
    occupant flips naturally.
    """
    pos = _position()
    if pos.get("success") is False:
        return f"FAILED: read position: {pos.get('reason')} ({pos.get('message','')})"
    xyz = _read_xyz(pos)
    if xyz is None:
        return "FAILED: malformed /position response"
    px, py, pz = xyz
    print(f"  [build_shelter] at ({px},{py},{pz})", flush=True)

    # Pre-flight 1: /stats — the fastest "agent is in water/lava right now"
    # signal. Catches sea-surface spawns where the cavity (py..py+1) sits
    # above the waterline but the player's feet are at a water block at py.
    stats_resp = _get_homunculus("/stats")
    if stats_resp.get("in_lava"):
        return (
            f"ABORTED: standing in lava at ({px},{py},{pz}) "
            f"— relocate to safe ground before building"
        )
    if stats_resp.get("in_water"):
        return (
            f"ABORTED: standing in water at ({px},{py},{pz}) "
            f"— relocate to dry land before building"
        )

    # Pre-flight 2-4: scan the cavity + floor row (py-1..py+1). One scan
    # feeds three checks: water/lava saturation, agent encasement, and
    # floor-footprint sufficiency.
    preflight = _scan_blocks(px - 2, py - 1, pz - 2, px + 2, py + 1, pz + 2)
    cells_by_xyz: dict[tuple[int, int, int], dict] = {
        (b["x"], b["y"], b["z"]): b
        for b in (preflight.get("blocks") or [])
    }

    # Pre-flight 2: water/lava saturation. Water can't be cleared cleanly
    # by /baritone/excavate (reflow); lava damages the agent. 75-cell box;
    # >25 (33%) catches ocean and lava lakes while leaving puddles ok.
    bad_cells: dict[str, int] = {}
    for b in cells_by_xyz.values():
        bid = (b.get("id") or "").lower()
        if "water" in bid or "lava" in bid:
            bad_cells[bid] = bad_cells.get(bid, 0) + 1
    bad_total = sum(bad_cells.values())
    if bad_total > 25:
        detail = ", ".join(
            f"{c} {b.split(':')[-1]}"
            for b, c in sorted(bad_cells.items(), key=lambda kv: -kv[1])
        )
        return (
            f"ABORTED: build region at ({px},{py},{pz}) is "
            f"{bad_total}/75 water/lava ({detail}) "
            f"— relocate to solid ground before building"
        )

    # Pre-flight 3: agent encasement. If BOTH the foot (py) and head (py+1)
    # cells at the agent's position are non-passable, the agent was TP'd
    # into solid rock — the build could still excavate around them but
    # they'd take suffocation damage during the carve, and the scenario
    # isn't a meaningful shelter test. Fires often for underground spawns.
    foot = cells_by_xyz.get((px, py, pz))
    head = cells_by_xyz.get((px, py + 1, pz))
    foot_blocked = foot is not None and not foot.get("passable")
    head_blocked = head is not None and not head.get("passable")
    if foot_blocked and head_blocked:
        foot_id = (foot.get("id") or "?").split(":")[-1]
        head_id = (head.get("id") or "?").split(":")[-1]
        return (
            f"ABORTED: agent encased in solid blocks at ({px},{py},{pz}): "
            f"foot={foot_id}, head={head_id} — relocate to open space"
        )

    # Pre-flight 4: need a real floor footprint to anchor the build on.
    # Skinny pinnacles / sky islands leave most of the 5x5 floor row as
    # air, and Baritone can't pillar into thin air to place wall blocks
    # — fills retry indefinitely (~20 min waste before we kill it).
    floor_solid = 0
    for fx in range(px - 2, px + 3):
        for fz in range(pz - 2, pz + 3):
            b = cells_by_xyz.get((fx, py - 1, fz))
            if b is None:
                continue
            bid = (b.get("id") or "").lower()
            if "water" not in bid and "lava" not in bid:
                floor_solid += 1
    if floor_solid < 10:
        burrow_dir = _viable_burrow_direction(px, py, pz)
        burrow_hint = (
            f" — OR call wall_in() (solid wall to your {burrow_dir}, "
            f"tunneling produces enough cobble to seal it)"
            if burrow_dir else ""
        )
        return (
            f"ABORTED: floor footprint at y={py - 1} has only "
            f"{floor_solid}/25 solid cells — relocate to flatter ground"
            f"{burrow_hint}"
        )

    # Pre-flight 5: enough buildables to seal the shell. Underestimating burns
    # 2-5 minutes of Baritone churn then leaves a mob-permeable structure
    # (rollouts 6 + 9 both died to mobs through ceiling holes here). Shell
    # geometry is ~98 cells; conservative threshold of SHELTER_BUDGET_MIN
    # leaves margin for terrain-coverage variance. Refuse with a structured
    # shortfall the agent can act on.
    buildables = _count_shelter_buildables()
    total_buildable = sum(buildables.values())
    # Day vs night changes the right advice on a budget shortfall. At day,
    # mining more is correct (buy materials, build a solid shelter). At
    # night, mining sends the agent away from cover to die — observed in
    # probe-validate-r1/r6 where shelter aborted at night, agent mined,
    # got swarmed. At night accept a leaky shelter (or a tighter geometry)
    # rather than refuse the build entirely. Threshold dropped to 35
    # (enough for a partial 5×2×5 with ~half-coverage, or a tight cluster
    # of cells around the player).
    day_ticks = stats_resp.get("day_ticks")
    is_night = isinstance(day_ticks, (int, float)) and day_ticks >= 12000
    NIGHT_BUDGET_MIN = 35
    effective_min = NIGHT_BUDGET_MIN if is_night else SHELTER_BUDGET_MIN
    if total_buildable < effective_min:
        have_str = ", ".join(
            f"{c} {b}"
            for b, c in sorted(buildables.items(), key=lambda kv: -kv[1])
        ) or "none"
        shortfall = effective_min - total_buildable
        # Clamp the suggested next-call quantity to MAX_QUANTITY so the agent
        # doesn't get told to ask for more than mine_stone can deliver in one
        # shot. Observed agent behavior (R-haiku 2026-05-14): without an
        # explicit suggested-quantity, the agent defaulted to mine_stone(1)
        # and burned ~5 turns ticking the shortfall down by 1 per call.
        suggested = min(shortfall, MAX_QUANTITY)
        burrow_dir = _viable_burrow_direction(px, py, pz)
        if is_night:
            # Burrow is the canonical fix here: agent4 prm0 T8 hit this exact
            # path (28 buildables underground), then died T11 because the
            # static prompt hint never got re-read mid-rollout.
            if burrow_dir:
                return (
                    f"ABORTED at NIGHT: only {total_buildable} buildables at "
                    f"({px},{py},{pz}) ({have_str}); need ≥{NIGHT_BUDGET_MIN} for a "
                    f"partial seal (short by {shortfall}). Call wall_in() — "
                    f"there's a solid wall to your {burrow_dir} and tunneling "
                    f"produces 6 cobble (enough to seal). If that fails, "
                    f"wait for dawn: call craft('stick', 1) until time=DAY."
                )
            return (
                f"ABORTED at NIGHT: only {total_buildable} buildables at "
                f"({px},{py},{pz}) ({have_str}); need ≥{NIGHT_BUDGET_MIN} for a "
                f"partial seal (short by {shortfall}). Do NOT call mine_* at "
                f"night — agents die mining in the open. Wait for dawn: "
                f"call craft('stick', 1) or any other idle in-place tool until "
                f"time=DAY, then mine + build."
            )
        burrow_hint = (
            f" Or call wall_in() — solid wall to your {burrow_dir}, "
            f"tunneling produces enough cobble to seal."
            if burrow_dir else ""
        )
        return (
            f"ABORTED: not enough buildable blocks at ({px},{py},{pz}) — "
            f"have {total_buildable} ({have_str}), need ~{SHELTER_BUDGET_MIN} "
            f"to seal a 5×2×5 shelter (short by {shortfall}). "
            f"Call mine_stone(quantity={suggested}) — or mine_wood(quantity={suggested}) "
            f"if trees are closer — to close the gap in one step.{burrow_hint}"
        )

    # Extend Baritone's pathfinder-throwaway allowlist BEFORE any fill — without
    # this, a mid-build material switch (e.g. dirt depletes, Equipper stages
    # diorite) leaves the pathfinder unable to pillar and stalls in long A*
    # searches looking for ground-level paths to y=65/y=66 cells.
    _set_shelter_throwaway_items()

    # Step 1: excavate 5×2×5 cavity. Player occupies (px, py..py+1, pz).
    # partial/already_clear are acceptable — we still proceed to wall in
    # whatever air now exists.
    ex = _baritone_excavate(px - 2, py, pz - 2, px + 2, py + 1, pz + 2, timeout_seconds=120)
    ex_ok = {"cleared", "already_clear", "partial"}
    if ex.get("reason") not in ex_ok:
        return f"FAILED: excavate interior: {ex.get('reason')} ({ex.get('message','')})"
    cleared = ex.get("volume", 0) - ex.get("remaining", 0)

    # Cavity excavate skips passable plants (Baritone's clearArea schematic
    # treats tall_grass/short_grass as already satisfying the AIR target).
    # If left in place, plants inside the cavity occlude the placement
    # raytrace and stall plate fills silently. Per-cell trim handles them.
    _trim_passable_in_aabb(px - 2, py, pz - 2, px + 2, py + 1, pz + 2, label="cavity")

    # Trim the shelter footprint + a 1-block exterior margin. Plants one
    # block *outside* the wall (e.g. tall_grass at (px, py, pz-4)) sit
    # outside every plate's AABB, so the per-plate trim misses them — but
    # the bot needs that cell free to stand/path there while placing wall
    # blocks from outside (and to keep raytraces unobstructed when looking
    # at outward-facing wall faces). One bulk scan covers all four sides
    # and the floor/ceiling extras' surroundings in a single pass.
    _trim_passable_in_aabb(
        px - 4, py - 1, pz - 4, px + 4, py + 2, pz + 4, label="exterior_margin",
    )

    # Walls sit one block outside the 5×5 interior boundary (at ±3).
    # Corner columns are covered by both adjacent walls — fill is idempotent.
    #
    # Anchor seeds (4 floor + 4 ceiling, 1-cell each at wall-column middles):
    # plain 5×5 floor/ceiling slabs have NO face-adjacent solid where the
    # wall base/top meets them — the wall is one z/x step outside the slab,
    # so they only meet diagonally. On flat terrain the natural surface
    # supplies enough anchors that walls find SOMETHING to place against;
    # on hanging or sloped builds it doesn't, and Baritone stalls trying to
    # place against air. Adding one cell at each cardinal wall-column middle
    # (same y as the slab, one z/x outside) yields 8 face-adjacent anchors
    # that don't depend on terrain — wall base middles, wall top middles,
    # and ceiling edge middles all chain from them.
    walls = [
        ("floor",        (px - 2, py - 1, pz - 2, px + 2, py - 1, pz + 2)),  # 25
        ("floor_ext_n",  (px,     py - 1, pz - 3, px,     py - 1, pz - 3)),  # 1 — anchors wall-N base mid
        ("floor_ext_s",  (px,     py - 1, pz + 3, px,     py - 1, pz + 3)),  # 1 — anchors wall-S base mid
        ("floor_ext_w",  (px - 3, py - 1, pz,     px - 3, py - 1, pz    )),  # 1 — anchors wall-W base mid
        ("floor_ext_e",  (px + 3, py - 1, pz,     px + 3, py - 1, pz    )),  # 1 — anchors wall-E base mid
        ("north", (px - 2, py,     pz - 3, px + 2, py + 1, pz - 3)),  # 10
        ("south", (px - 2, py,     pz + 3, px + 2, py + 1, pz + 3)),  # 10
        ("west",  (px - 3, py,     pz - 2, px - 3, py + 1, pz + 2)),  # 10
        ("east",  (px + 3, py,     pz - 2, px + 3, py + 1, pz + 2)),  # 10
    ]
    # Ceiling extras sit at y=py+2 (same as ceiling slab) at the wall-column
    # middles. After walls are up, each is face-adjacent to that side's wall
    # top middle (anchor) AND to the ceiling edge middle on that side
    # (anchors the ceiling slab during its fill). Placed BEFORE the main
    # ceiling so the slab's edge middles already have face-adjacent solids.
    eb = 0 # extra buffer
    ceiling_extras = [
        ("ceiling_ext_n", (px - eb, py + 2, pz - 3 , px + eb,  py + 2, pz - 3 )),
        ("ceiling_ext_s", (px - eb, py + 2, pz + 3 , px + eb,  py + 2, pz + 3 )),
        ("ceiling_ext_w", (px - 3 , py + 2, pz - eb, px - 3 ,  py + 2, pz + eb)),
        ("ceiling_ext_e", (px + 3 , py + 2, pz - eb, px + 3 ,  py + 2, pz + eb)),
    ]
    ceiling = ("ceiling", (px - 2, py + 2, pz - 2, px + 2, py + 2, pz + 2))  # 25

    placed = 0
    used: dict[str, int] = {}
    failures: list[str] = []

    def run_plate(label_aabb):
        nonlocal placed
        lbl, (a, b, c, d, e, f) = label_aabb
        block, delta, fail = _fill_plate(a, b, c, d, e, f, label=lbl)
        if fail is not None:
            failures.append(f"{lbl}={fail}")
            return
        placed += delta
        if block is not None:
            used[block] = used.get(block, 0) + delta

    # Forbid breaking during construction — but ONLY on surface. Two reasons
    # to keep it disabled on surface builds:
    # 1. Construction: prevents Baritone from breaking partial walls to
    #    take a shortcut onto the roof, which was creating unfixable
    #    wall holes.
    # 2. Roof-descent: at the end of ceiling phase the bot is on top of
    #    the sealed shelter. With allow_break=false the only way down
    #    is to walk to the ceiling's edge and fall — *not* break a hole
    #    straight through the roof to enter the cavity.
    #
    # Underground/cave builds are the inverse: every path to a wall cell
    # threads through stone, so allow_break=false makes the pathfinder
    # take absurd cave-corridor detours (observed live 2026-05-14 on
    # iter 2 of the cave trial). Distinguish surface vs cave by what's
    # OVERHEAD — not by the cavity itself (cave pockets are mostly air
    # like surface). Scan a 5x4x5 box just above the cavity (py+2..py+5);
    # if mostly solid → there's a cave ceiling → underground.
    overhead_scan = _scan_blocks(
        px - 2, py + 2, pz - 2, px + 2, py + 5, pz + 2,
    )
    overhead_solid = 0
    for b in overhead_scan.get("blocks", []) or []:
        if not b.get("passable"):
            overhead_solid += 1
    underground = overhead_solid > 50  # >50% of 100 cells solid
    if underground:
        print(
            f"  [build_shelter] WARNING overhead {overhead_solid}/100 solid → "
            f"underground (allow_break=True); this primitive is designed for "
            f"surface and often fails in caves — carve-mode is best-effort",
            flush=True,
        )
    else:
        print(
            f"  [build_shelter] overhead {overhead_solid}/100 solid → "
            f"surface (allow_break=False)",
            flush=True,
        )
    _set_allow_break(underground)
    try:
        if underground:
            # Carve-mode: the cavity excavate above already cleared the
            # interior, and the natural stone surrounding it IS the wall/
            # floor/ceiling. Skip the surface-style plate fills entirely
            # — they'd just thrash the pathfinder trying to reach 100
            # cells that are already solid. The inspect+patch pass below
            # handles the 1-5 cells that ARE genuine holes (cave openings
            # into the shell). Bot is already inside the cavity, so no
            # reentry goto needed either.
            print("  [build_shelter] carve-mode: skipping plate fills, "
                  "leaving inspect+patch to seal cave openings", flush=True)
            # Door-first + entry tunnel: cut the slot (z=pz-3) AND three
            # cells of tunnel beyond it (z=pz-4..pz-6). Reasons:
            # 1. The slot exists during the patch pass — some holes may
            #    be reachable only via the doorway (a wall cell that
            #    opens into a narrow side passage), and having an
            #    already-cut path lets Baritone reach without breaking
            #    the shell.
            # 2. The 3-deep tunnel makes the doorway "obviously" the
            #    cheapest path in/out, enticing the pathfinder away
            #    from breaking arbitrary shell cells for shortcuts.
            # Excavate no-ops on already-air cells, so this is cheap
            # if the cave already opens northward.
            early_door = _baritone_excavate(
                px, py, pz - 6, px, py + 1, pz - 3, timeout_seconds=30,
            )
            if early_door.get("reason") not in {"cleared", "already_clear", "partial"}:
                failures.append(
                    f"early_door_excavate={early_door.get('reason')}"
                )
            ceiling_open = False
        else:
            for plate in walls:
                run_plate(plate)

            reenter = _baritone_goto(
                px, py, pz,
                timeout_seconds=20,
                arrival_tolerance=1,
                allow_place=True,
            )
            inside = False
            post = _position()
            post_xyz = _read_xyz(post) if post.get("success") is not False else None
            if post_xyz is not None:
                ax, ay, az = post_xyz
                inside = abs(ax - px) <= 2 and abs(az - pz) <= 2 and ay >= py and ay <= py + 1
            ceiling_open = not inside
            if ceiling_open:
                failures.append(
                    f"reenter={reenter.get('reason', 'unknown')}"
                    + (f" (post_xyz={post_xyz})" if post_xyz else " (post_xyz=unknown)")
                )
            else:
                for plate in ceiling_extras:
                    run_plate(plate)
                run_plate(ceiling)

                # Walk off the roof to a point just outside the door wall
                # (default = north). With allow_break=false the only path is
                # along the ceiling to the edge, off the side. Falls are
                # cheap; chewing through the roof is forbidden, which is
                # exactly the constraint we want here.
                _baritone_goto(
                    px, py, pz - 4,
                    timeout_seconds=30,
                    arrival_tolerance=1,
                    allow_place=True,
                )
    finally:
        _set_allow_break(True)

    # Door slot: 1 cell wide, 2 cells tall, in the middle of the north wall.
    # Cells stay air permanently — excluded from the inspect's expected set.
    # Hypothesis (per-user, to test on this run): once a doorway exists,
    # Baritone prefers walking through it over breaking other shell cells
    # when it needs to enter/exit. The cavity re-excavate below is the
    # test — bot is currently outside; it needs to reach the cavity floor
    # to dig out the pillars; if it uses the doorway we're done, if it
    # breaks a different wall the final inspect surfaces that.
    #
    # Real door-block placement is a follow-up — homunculus's /place is
    # "place wherever the player is looking," which needs more orchestration
    # for doors than a single call. For now the doorway is just an open
    # 1x2x1 hole (testable, then we polish).
    door_aabb = (px, py, pz - 3, px, py + 1, pz - 3)
    door_cells: set[tuple[int, int, int]] = set()
    for x in range(door_aabb[0], door_aabb[3] + 1):
        for y in range(door_aabb[1], door_aabb[4] + 1):
            for z in range(door_aabb[2], door_aabb[5] + 1):
                door_cells.add((x, y, z))

    shell_aabbs = [aabb for _, aabb in walls + ceiling_extras + [ceiling]]
    door_status = "skipped"
    if not ceiling_open:
        # Cut the door slot. Bot is at (px, py, pz-4), face-adjacent to
        # the slot — no pathing required, just mine the two cells.
        door_ex = _baritone_excavate(
            door_aabb[0], door_aabb[1], door_aabb[2],
            door_aabb[3], door_aabb[4], door_aabb[5],
            timeout_seconds=30,
        )
        if door_ex.get("reason") not in {"cleared", "already_clear", "partial"}:
            failures.append(f"door_excavate={door_ex.get('reason')}")

        # Door install is deferred until AFTER the final_goto below. Placing
        # the door here (closed) would block Baritone's final entry — it can't
        # open closed doors mid-path. Instead we leave the slot open, let
        # Baritone route through it on the final goto, then place the door
        # from inside (reach to the slot is well under 5 blocks from center).
        door_item = _find_door_in_inventory()
        door_status = "open (no door in inventory)" if door_item is None else None

        # Cavity re-excavate. Any pillars Baritone placed during the
        # ceiling phase live at y=py..py+1 inside the footprint; this
        # cleans them. Hypothesis: bot enters via the doorway (cheap path)
        # rather than breaking through the roof.
        ex2 = _baritone_excavate(
            px - 2, py, pz - 2, px + 2, py + 1, pz + 2, timeout_seconds=60,
        )
        if ex2.get("reason") not in {"cleared", "already_clear", "partial"}:
            failures.append(f"post_excavate={ex2.get('reason')}")

        # Defensive patch pass. Excludes the door slot — those cells are
        # supposed to be air.
        _, _, pre_holes, pre_err = _inspect_shelter_shell(shell_aabbs, exclude=door_cells)
        if pre_err is None and pre_holes:
            all_plates = walls + ceiling_extras + [ceiling]
            affected: dict[str, tuple[int, int, int, int, int, int]] = {}
            for hx, hy, hz in pre_holes:
                for label, (a, b, c, d, e, f) in all_plates:
                    if a <= hx <= d and b <= hy <= e and c <= hz <= f:
                        affected[label] = (a, b, c, d, e, f)
                        break
            for label, (a, b, c, d, e, f) in affected.items():
                print(f"  [patch] re-running {label} for hole(s)", flush=True)
                blk, delta, fail = _fill_plate(a, b, c, d, e, f, label=f"patch_{label}")
                placed += delta
                if blk is not None and delta > 0:
                    used[blk] = used.get(blk, 0) + delta
                if fail is None:
                    failures = [f for f in failures if not f.startswith(f"{label}=")]
                else:
                    failures.append(f"patch_{label}={fail}")

            # The patch fills entire plate AABBs (one call per affected
            # plate) — _fill_plate has no per-cell exclusion, so it fills
            # every air cell in the AABB INCLUDING the door slot. If we
            # patched the north wall, the door cells are now solid. Re-
            # excavate door+tunnel here to undo that and restore passage.
            print("  [patch] re-cutting door+tunnel after patch", flush=True)
            redoor = _baritone_excavate(
                px, py, pz - 6, px, py + 1, pz - 3, timeout_seconds=20,
            )
            if redoor.get("reason") not in {"cleared", "already_clear", "partial"}:
                failures.append(f"post_patch_door_excavate={redoor.get('reason')}")

        # Final navigate-inside. After the patch pass the bot is typically
        # standing outside (it stepped out via the doorway to patch the
        # ceiling from above — Baritone's natural preference). Ask it to
        # come back inside. With the doorway present and allow_break=true,
        # the hypothesis is that walking through the doorway is cheaper
        # than punching a new hole in the shell. If that fails, the final
        # inspect surfaces the resulting holes — we don't loop on them.
        final_goto = _baritone_goto(
            px, py, pz,
            timeout_seconds=20,
            arrival_tolerance=1,
            allow_place=True,
        )
        if not final_goto.get("success"):
            failures.append(f"final_goto={final_goto.get('reason', 'unknown')}")

        # Now install the door from inside. Bot is at (px, py, pz); door
        # support is at (px, py-1, pz-3) — distance ~3.16, well under reach.
        # The placer's Look.faceBlockTop handles orientation. MC auto-fills
        # the upper half from the same place packet.
        if door_item is not None:
            # Verify the support block. Leaves, fences, glass, slabs, stairs,
            # carpets, thin snow etc. are non-air but don't support doors —
            # /place_at silently no-ops. Excavate and re-fill from inventory
            # if so; safe no-op if support is already doorworthy.
            sx, sy, sz = door_aabb[0], door_aabb[1] - 1, door_aabb[2]
            sscan = _scan_blocks(sx, sy, sz, sx, sy, sz)
            support_id: str | None = None
            for b in sscan.get("blocks", []) or []:
                if b.get("x") == sx and b.get("y") == sy and b.get("z") == sz:
                    support_id = b.get("id")
                    break
            if not _supports_door(support_id):
                print(
                    f"  [door_support] {support_id or 'air'} at ({sx},{sy},{sz})"
                    f" can't hold a door; excavate+refill",
                    flush=True,
                )
                _baritone_excavate(sx, sy, sz, sx, sy, sz, timeout_seconds=5)
                blk, delta, sf = _fill_plate(sx, sy, sz, sx, sy, sz,
                                              label="door_support_patch")
                if blk is not None and delta > 0:
                    used[blk] = used.get(blk, 0) + delta
                    placed += delta
                if sf is not None:
                    failures.append(f"door_support_patch={sf}")

            place_resp = _place_at_raw(
                door_item, door_aabb[0], door_aabb[1], door_aabb[2],
            )
            if place_resp.get("success"):
                door_status = door_item.split(":")[-1]
            else:
                door_status = f"open ({place_resp.get('reason', 'unknown')})"
                failures.append(
                    f"door_place={place_resp.get('reason')}: "
                    f"{place_resp.get('message', '')}"
                )

    # Final inspection (post-patch). This is the ground truth for the report.
    # Excludes the door slot when the door was actually cut (i.e. ceiling
    # built successfully) — when ceiling_open we never tried to cut the
    # door, so those cells revert to being expected-solid wall.
    final_exclude = door_cells if not ceiling_open else None
    solid_n, expected_n, holes, scan_err = _inspect_shelter_shell(
        shell_aabbs, exclude=final_exclude,
    )

    # Player position relative to cavity (after patch + any wall-break-in).
    inside_post = False
    pos_post = _position()
    xyz_post = _read_xyz(pos_post) if pos_post.get("success") is not False else None
    if xyz_post is not None:
        ax, ay, az = xyz_post
        inside_post = (
            abs(ax - px) <= 2 and abs(az - pz) <= 2
            and ay >= py and ay <= py + 1
        )
    player_str = "inside" if inside_post else "outside"

    used_str = ", ".join(
        f"{c} {b.split(':')[-1]}" for b, c in sorted(used.items(), key=lambda kv: -kv[1])
    ) or "nothing"
    base = (
        f"shelter at ({px},{py},{pz}); excavated {cleared} cells, "
        f"placed {placed} ({used_str})"
    )

    if scan_err is not None:
        inspect_str = f" inspect: scan_failed ({scan_err}); player: {player_str}"
    elif holes:
        head = ", ".join(f"({x},{y},{z})" for x, y, z in holes[:5])
        more = f" +{len(holes) - 5} more" if len(holes) > 5 else ""
        inspect_str = (
            f" inspect: {solid_n}/{expected_n} solid, "
            f"{len(holes)} hole(s) at {head}{more}; player: {player_str}"
        )
    else:
        inspect_str = f" inspect: {solid_n}/{expected_n} solid; player: {player_str}"

    door_str = (
        ""
        if ceiling_open
        else f"; door: {door_status} on north at ({px},{py},{pz - 3})"
    )
    if ceiling_open:
        return (
            f"PARTIAL: {base}; ceiling OPEN — reentry failed. "
            f"Issues: {', '.join(failures)}.{inspect_str}"
        )
    if failures:
        return (
            f"PARTIAL: {base}; failed plates: {', '.join(failures)}"
            f"{door_str}.{inspect_str}"
        )
    return (
        f"built {base}{door_str}. Walk through doorway to enter.{inspect_str}"
    )


# --- burrow ----------------------------------------------------------------
# Tactical 1×2 sideways tunnel into a wall with a 1-cell foyer + seal at
# cell-index-2 + 1-cell back cavity. Pairs with reflexive /evasion: evasion
# buys ~3s to flee, burrow converts that to a sealed pocket. Distinct from
# build_shelter (which expects open sky); burrow expects an adjacent wall.

# Seal-block preference. Tunneled cobble lands here first 99% of the time;
# the fallbacks cover dirt/stone-variant walls and the rare "I had nothing
# but I tunneled into deepslate" case.
_BURROW_SEAL_PRIORITY: tuple[str, ...] = (
    "minecraft:cobblestone",
    "minecraft:cobbled_deepslate",
    "minecraft:stone",
    "minecraft:deepslate",
    "minecraft:granite", "minecraft:diorite", "minecraft:andesite", "minecraft:tuff",
    "minecraft:dirt", "minecraft:coarse_dirt",
    "minecraft:netherrack", "minecraft:blackstone", "minecraft:basalt",
)

# Cells that disqualify a direction even if the rest of the wall is solid —
# we don't want to tunnel into water (drowning) or lava (instant death).
_BURROW_HAZARD_IDS: frozenset[str] = frozenset({
    "minecraft:water", "minecraft:lava",
    "minecraft:flowing_water", "minecraft:flowing_lava",
})

# Active burrow state, consumed by agent._fetch_stats for ambient hinting.
# Shape: {"anchor": (x,y,z), "seal": (x,y,z), "direction": str} or None.
_burrow_state: dict | None = None


def get_burrow_state() -> dict | None:
    """Return current burrow state (anchor, seal, direction) or None."""
    return _burrow_state


def clear_burrow_state() -> None:
    """Reset burrow state. Called when agent dies / between rollouts."""
    global _burrow_state
    _burrow_state = None


def _inventory_counts() -> dict[str, int]:
    """Aggregate inventory ids → counts. Returns {} on transport failure."""
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/inventory", timeout=5.0)
        r.raise_for_status()
        inv = r.json()
    except (requests.RequestException, ValueError):
        return {}
    counts: dict[str, int] = {}
    for slot in inv.get("main", []) or []:
        sid = slot.get("id")
        if sid:
            counts[sid] = counts.get(sid, 0) + int(slot.get("count", 0))
    off = inv.get("offhand")
    if off and off.get("id"):
        counts[off["id"]] = counts.get(off["id"], 0) + int(off.get("count", 0))
    return counts


def _pick_seal_item(needed: int = 2) -> str | None:
    counts = _inventory_counts()
    for candidate in _BURROW_SEAL_PRIORITY:
        if counts.get(candidate, 0) >= needed:
            return candidate
    return None


def _wall_viable(px: int, py: int, pz: int, dx: int, dz: int) -> tuple[bool, str]:
    """Check if 3-deep × 2-high wall is fully solid (no air/lava/water).

    Returns (viable, reason). reason is empty on success, else describes
    the first failure cell for diagnostic.
    """
    x_a, x_b = px + dx * 1, px + dx * 3
    z_a, z_b = pz + dz * 1, pz + dz * 3
    x1, x2 = min(x_a, x_b), max(x_a, x_b)
    z1, z2 = min(z_a, z_b), max(z_a, z_b)
    scan = _scan_blocks(x1, py, z1, x2, py + 1, z2)
    if scan.get("success") is False:
        return False, f"scan_blocks failed: {scan.get('reason')}"
    blocks = {(b["x"], b["y"], b["z"]): b for b in scan.get("blocks", [])}
    for step in (1, 2, 3):
        for dy in (0, 1):
            cell = (px + dx * step, py + dy, pz + dz * step)
            b = blocks.get(cell)
            if b is None:
                return False, f"air at cell {cell}"
            if b.get("id") in _BURROW_HAZARD_IDS:
                return False, f"hazard {b['id']} at cell {cell}"
            if b.get("passable"):
                return False, f"passable {b.get('id')} at cell {cell}"
    return True, ""


def _viable_burrow_direction(px: int, py: int, pz: int) -> str | None:
    """Return the first cardinal direction with a viable burrow wall, or None.

    Used by build_shelter's ABORT path to surface burrow as a fallback at
    the moment of decision — the strategy hint in the system prompt is
    invisible by the time these aborts fire (buried under tool outcomes).
    """
    for d in ("east", "west", "north", "south"):
        dx, dz = _DIR_VEC[d]
        ok, _ = _wall_viable(px, py, pz, dx, dz)
        if ok:
            return d
    return None


def handle_burrow(args: dict) -> str:
    global _burrow_state
    pos_resp = _position()
    if pos_resp.get("success") is False:
        return f"FAILED: position read failed ({pos_resp.get('reason')})"
    try:
        px = int(pos_resp["x"])
        py = int(pos_resp["y"])
        pz = int(pos_resp["z"])
        yaw = float(pos_resp.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError):
        return "FAILED: malformed /position response"

    # Try directions in priority order: current facing first (the most likely
    # wall the agent is already pressed against), then the others.
    facing = _yaw_to_direction(yaw)
    cardinals = [facing] + [d for d in ("east", "west", "north", "south") if d != facing]

    chosen: str | None = None
    last_reason = ""
    for d in cardinals:
        dx, dz = _DIR_VEC[d]
        ok, reason = _wall_viable(px, py, pz, dx, dz)
        if ok:
            chosen = d
            break
        last_reason = f"{d}: {reason}"
        print(f"  [burrow] {d} not viable — {reason}", flush=True)

    if chosen is None:
        return (
            f"FAILED: no_wall_found — no cardinal has a 3-deep solid wall at "
            f"y={py}-{py+1}. Last check: {last_reason}. You're not flush "
            f"against a wall — travel into a cave wall / hillside first, "
            f"or use build_shelter() if you're on open ground."
        )

    dx, dz = _DIR_VEC[chosen]
    # Excavate the 3-cell × 2-high corridor (foot + head). Cell coords:
    #   cell 1 (foyer)        = (px+dx,   py..py+1, pz+dz)
    #   cell 2 (seal target)  = (px+2dx,  py..py+1, pz+2dz)
    #   cell 3 (back cavity)  = (px+3dx,  py..py+1, pz+3dz)
    x_a, x_b = px + dx * 1, px + dx * 3
    z_a, z_b = pz + dz * 1, pz + dz * 3
    x1, x2 = min(x_a, x_b), max(x_a, x_b)
    z1, z2 = min(z_a, z_b), max(z_a, z_b)
    print(
        f"  [burrow] {chosen}: excavating ({x1},{py},{z1})→({x2},{py+1},{z2})",
        flush=True,
    )
    ex = _baritone_excavate(x1, py, z1, x2, py + 1, z2, timeout_seconds=60)
    if ex.get("success") is False:
        return (
            f"FAILED: excavate refused ({ex.get('reason')}: {ex.get('message')}). "
            f"Try again or surface() to find a different wall."
        )
    # remaining=0 is the success signal; >0 means baritone bailed early.
    remaining = ex.get("remaining")
    if isinstance(remaining, (int, float)) and remaining > 0:
        return (
            f"FAILED: tunnel only partially dug ({remaining} cells remain). "
            f"You're probably exposed at the entrance — try wall_in() again, "
            f"or build_shelter() if you have buildable blocks."
        )

    # Walk into the back cavity. Critical: player MUST NOT be standing in the
    # seal cell or /place_at returns target_blocked. Default Baritone tolerance
    # (2 blocks) is too loose — agent often stops at the seal cell mid-walk.
    # arrival_tolerance=0 forces Baritone to put the player exactly at the
    # back-cavity block.
    back_x = px + dx * 3
    back_z = pz + dz * 3
    seal_x = px + dx * 2
    seal_z = pz + dz * 2
    bgoto = _baritone_goto(back_x, py, back_z, timeout_seconds=15, arrival_tolerance=0)
    final = _final_xyz(bgoto) or (px, py, pz)
    fx, fy, fz = final
    # Verify player is NOT in the seal cell (would block placement). Float
    # position needs floor-rounding to compare to int block coords.
    fbx = math.floor(fx)
    fbz = math.floor(fz)
    if fbx == seal_x and fbz == seal_z:
        # Retry with a small forward nudge — Baritone sometimes parks on the
        # block before its goal when arrival_tolerance has odd float rounding.
        bgoto2 = _baritone_goto(back_x, py, back_z, timeout_seconds=8, arrival_tolerance=0)
        final = _final_xyz(bgoto2) or final
        fx, fy, fz = final
        fbx = math.floor(fx)
        fbz = math.floor(fz)
        if fbx == seal_x and fbz == seal_z:
            return (
                f"PARTIAL: tunnel dug but player stuck at seal cell "
                f"({fbx},{int(fy)},{fbz}) — Baritone won't advance to back "
                f"cavity ({back_x},{py},{back_z}). Manually walk forward 1 "
                f"block, then place 2 cobblestone at ({seal_x},{py},{seal_z}) "
                f"and ({seal_x},{py+1},{seal_z})."
            )

    seal_item = _pick_seal_item(needed=2)
    if seal_item is None:
        return (
            f"PARTIAL: tunnel dug at ({back_x},{py},{back_z}) facing {chosen} "
            f"but no placeable blocks for seal (need ≥2 of "
            f"cobble/stone/dirt). Mine cobble or break the wall to refill, "
            f"then call wall_in() again — or place blocks manually."
        )

    sx = px + dx * 2
    sz = pz + dz * 2
    # Foot first (cell 2 at y=py), then head (y=py+1). Head needs the foot as
    # support attachment; reverse order can fail with no_adjacent_face.
    foot = _place_at_raw(seal_item, sx, py, sz)
    if foot.get("success") is False:
        return (
            f"PARTIAL: tunnel dug but seal foot placement failed at "
            f"({sx},{py},{sz}): {foot.get('reason')}. You're exposed — "
            f"manually place 2 {seal_item.split(':')[-1]} at "
            f"({sx},{py},{sz}) and ({sx},{py+1},{sz})."
        )
    head = _place_at_raw(seal_item, sx, py + 1, sz)
    if head.get("success") is False:
        return (
            f"PARTIAL: foot sealed but head failed at ({sx},{py+1},{sz}): "
            f"{head.get('reason')}. Mob-sized gap remains — manually place "
            f"1 {seal_item.split(':')[-1]} at ({sx},{py+1},{sz})."
        )

    _burrow_state = {
        "anchor": (back_x, py, back_z),
        "seal": (sx, py, sz),
        "direction": chosen,
    }
    return (
        f"walled in at ({back_x},{py},{back_z}) facing {chosen}; "
        f"sealed with 2× {seal_item.split(':')[-1]} at "
        f"({sx},{py},{sz})+({sx},{py+1},{sz}). Foyer cell at "
        f"({px+dx},{py},{pz+dz}) is open but mob-occupiable. To exit: "
        f"mine_stone(1) to break through the seal."
    )


def handle_expand_burrow(args: dict) -> str:
    """Productive conversion: carve a 2×3 alcove off the back cavity.

    Pre-req: an active burrow (_burrow_state set) and the agent still
    inside it. Excavates 2 deeper cells + 1 lateral cell on each side at
    the deep end — fits crafting_table, furnace, and bed. Seal stays
    untouched. Refuses when water/lava is detected in the target volume.
    """
    global _burrow_state
    bstate = _burrow_state
    if bstate is None:
        return (
            "FAILED: no_active_wall_in — call wall_in() first. carve_alcove "
            "carves a 2×3 chamber off your back cavity for crafting/furnace "
            "placement, but you need a sealed pocket to extend."
        )

    direction = bstate.get("direction")
    if direction not in _DIR_VEC:
        return f"FAILED: corrupted wall_in state (direction='{direction}')"
    dx, dz = _DIR_VEC[direction]
    # Perpendicular axis (90° CCW from forward). For east (dx=1,dz=0) →
    # (0, 1) = south; for north (0,-1) → (1, 0) = east; etc.
    perp_dx, perp_dz = -dz, dx

    bx, by, bz = bstate["anchor"]

    pos_resp = _position()
    if pos_resp.get("success") is False:
        return f"FAILED: position read failed ({pos_resp.get('reason')})"
    try:
        px = int(pos_resp["x"])
        py = int(pos_resp["y"])
        pz = int(pos_resp["z"])
    except (KeyError, TypeError, ValueError):
        return "FAILED: malformed /position response"

    # Stale-burrow guard. Anchor is from when burrow() returned; if the
    # agent has wandered (e.g. broke the seal and walked out) the alcove
    # geometry is no longer relative to where they stand.
    drift = abs(px - bx) + abs(pz - bz)
    if drift > 4:
        return (
            f"FAILED: drifted_from_wall_in — anchor at ({bx},{by},{bz}) "
            f"but you're at ({px},{py},{pz}) — {drift}b away. Walk back "
            f"inside your sealed pocket before calling carve_alcove."
        )

    # Alcove footprint. Cells are at d_step ∈ {1,2} forward and
    # p_step ∈ {-1,0,1} lateral, relative to the back cavity (bx,by,bz):
    #   forward 1 + lateral {-1,0,1}  → 3 cells (one row 1 deep into wall)
    #   forward 2 + lateral {-1,0,1}  → 3 cells (one row 2 deep into wall)
    # Total 6 cells × 2 layers = 12 block breaks. AABB is the bounding box
    # of those 6 cells, which for any cardinal direction is exactly 2×3.
    corners = []
    for d_step in (1, 2):
        for p_step in (-1, 1):
            cx = bx + dx * d_step + perp_dx * p_step
            cz = bz + dz * d_step + perp_dz * p_step
            corners.append((cx, cz))
    xs = [c[0] for c in corners]
    zs = [c[1] for c in corners]
    ax1, ax2 = min(xs), max(xs)
    az1, az2 = min(zs), max(zs)

    # Hazard pre-scan. Refuse if any cell in the alcove volume is water/
    # lava — carving into them would flood the burrow.
    scan = _scan_blocks(ax1, by, az1, ax2, by + 1, az2)
    if scan.get("success") is False:
        return f"FAILED: scan refused: {scan.get('reason')}: {scan.get('message')}"
    for b in scan.get("blocks", []):
        if b.get("id") in _BURROW_HAZARD_IDS:
            return (
                f"FAILED: hazard {b['id']} at ({b['x']},{b['y']},{b['z']}) "
                f"in the wall — would flood your pocket. Stay where you are; "
                f"the seal is still intact."
            )

    print(
        f"  [expand_burrow] excavating alcove "
        f"({ax1},{by},{az1})→({ax2},{by+1},{az2}) (12 cells)",
        flush=True,
    )
    ex = _baritone_excavate(ax1, by, az1, ax2, by + 1, az2, timeout_seconds=90)
    if ex.get("success") is False:
        return (
            f"FAILED: excavate refused ({ex.get('reason')}: {ex.get('message')}). "
            f"Seal is unchanged — you're still safe inside."
        )
    remaining = ex.get("remaining")
    if isinstance(remaining, (int, float)) and remaining > 0:
        return (
            f"PARTIAL: alcove only partially carved ({remaining} cells "
            f"remain). Seal still intact. Call carve_alcove() again "
            f"to finish, or use the partial chamber as-is."
        )

    _burrow_state = {
        **bstate,
        "alcove_aabb": (ax1, by, az1, ax2, by + 1, az2),
    }
    return (
        f"alcove carved: 2×3 chamber at ({ax1},{by},{az1})→({ax2},{by+1},{az2}) "
        f"off your back cavity. Place crafting_table, furnace, and bed inside "
        f"with place(). Seal at {bstate['seal']} is unchanged — "
        f"mine_stone(1) when you want to leave."
    )


def handle_look_around(args: dict) -> str:
    from craft.scout import describe_neighborhood

    radius = int(args.get("radius", 2))
    if radius < 1 or radius > 4:
        return f"FAILED: radius must be 1-4, got {radius}"
    # Env-var cap for ablations (e.g., force r=1 to test "constant
    # scanning with smaller area"). Caps SILENTLY rather than failing —
    # we want the agent to see the result, not a parameter-rejected error.
    max_radius_env = os.environ.get("CRAFT_LOOK_AROUND_MAX_RADIUS")
    if max_radius_env:
        try:
            max_radius = int(max_radius_env)
            if radius > max_radius:
                print(f"  [look_around] capping radius {radius} → {max_radius} (CRAFT_LOOK_AROUND_MAX_RADIUS)", flush=True)
                radius = max_radius
        except ValueError:
            pass
    print(f"  [look_around] radius={radius} ({(2*radius-1)**2} chunks)...", flush=True)
    try:
        result = describe_neighborhood(radius)
    except Exception as e:
        return f"FAILED: look_around error: {e}"
    n = len(result["per_chunk"])
    t = result["timings"]
    cache = result.get("cache") or {"hits": 0, "total": n}
    return (
        f"[look_around radius={radius} chunks={n} "
        f"fanout={t['fanout_s']:.1f}s unify={t['unify_s']:.1f}s "
        f"cache_hits={cache['hits']}/{cache['total']}]\n"
        f"{result['unified']}"
    )


HANDLERS = {
    "mine_wood": handle_mine_wood,
    "mine_stone": handle_mine_stone,
    "mine_iron": handle_mine_iron,
    "mine_diamond": handle_mine_diamond,
    "mine_coal": handle_mine_coal,
    "craft": handle_craft,
    "place": handle_place,
    "smelt": handle_smelt,
    "collect_smelt": handle_collect_smelt,
    "surface": handle_surface,
    "descend": handle_descend,
    "travel": handle_travel,
    "build_shelter": handle_build_shelter,
    "wall_in": handle_burrow,
    "carve_alcove": handle_expand_burrow,
    "look_around": handle_look_around,
}


def dispatch(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        return f"FAILED: invalid args JSON ({e})"
    handler = HANDLERS.get(name)
    if handler is None:
        return f"FAILED: unknown tool '{name}'"
    return handler(args)
