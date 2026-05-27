"""Closed-loop agent driven by structured tool calls.

Each turn, the LLM proposes one tool call; the handler executes; the outcome
is routed back as a `tool` role message; the LLM proposes the next call.
Capped at `max_turns`. If the LLM emits multiple tool calls in one turn we
execute only the first and warn — the design principle is *tight return loop,
high turn count* (see feedback_tight_return_loop).
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import requests

from craft.config import PLAYER_NAME as _PLAYER_NAME, SERVER_CMD_BASE as _SERVER_CMD_BASE
from craft.llm import chat_with_tools, DEFAULT_MODEL
from craft.milestones import Milestones, resolve_milestones
from craft.nudges import resolve_nudges, render_nudges
from craft.recorder import start_rollout_recording
from craft.mine import _yaw_to_direction
from craft.spawn import random_spawn
from craft.tools import (
    HOMUNCULUS_BASE,
    TOOLS,
    dispatch,
    get_burrow_state,
)
from craft.world import (
    PHASE_TICKS as _START_PHASE_TICKS,  # back-compat alias for any external readers
    resolve_phase_ticks,
    set_difficulty,
    set_gamemode,
    set_time,
)
from craft.wurst import ensure_hacks_on as ensure_wurst_hacks_on
from craft.wurst import ensure_hacks_off as ensure_wurst_hacks_off
from craft.wurst import seed_autodrop_from_tier as _seed_autodrop_from_tier


# Hostile types the shelter watcher polls for. Tight list — covers the
# common night-pressure mobs without inflating per-turn scan cost (one
# /scan_entities call per type, ~50ms each).
_SHELTER_HOSTILE_TYPES: tuple[str, ...] = (
    "minecraft:zombie",
    "minecraft:skeleton",
    "minecraft:spider",
    "minecraft:creeper",
)

# Shelter watcher state. None when no shelter is currently armed.
# Shape: {anchor: (x,y,z), per_uuid: dict[str, dict], breach: bool,
#         breach_first_t: float | None, started_at: float}
_shelter_watch: dict | None = None


# build_shelter always emits "shelter at (X,Y,Z); ..." in its outcome string
# (see handle_build_shelter base = f"shelter at ({px},{py},{pz}); ...").
_SHELTER_ANCHOR_RE = re.compile(r"shelter at \((-?\d+),(-?\d+),(-?\d+)\)")


def _arm_shelter_watch(outcome: str) -> None:
    """Capture the cavity anchor by parsing it out of build_shelter's outcome.

    The shelter primitive logs its anchor as "shelter at (X,Y,Z)" — that IS
    the watcher's anchor. Originally we re-read /position here, but that
    raced with mid-build deaths + AutoRespawn: 2026-05-14 produced a watcher
    armed 14000 blocks from the actual shelter because the player died during
    a 162s build, respawned at world spawn, and /position returned the
    post-respawn coords. Parsing the outcome sidesteps the race entirely.
    """
    global _shelter_watch
    m = _SHELTER_ANCHOR_RE.search(outcome)
    if not m:
        print("[shelter_watch] couldn't find 'shelter at (X,Y,Z)' in outcome — not arming",
              flush=True)
        return
    anchor = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    _shelter_watch = {
        "anchor": anchor,
        "per_uuid": {},
        "breach": False,
        "breach_first_t": None,
        "started_at": time.time(),
    }
    print(f"[shelter_watch] armed at anchor={anchor}", flush=True)


def _scan_hostile(mob: str) -> list[dict]:
    """One /scan_entities call. Returns [] on transport error or empty result."""
    try:
        resp = requests.get(
            f"{HOMUNCULUS_BASE}/scan_entities",
            params={"type": mob, "radius": 16, "limit": 16},
            timeout=4.0,
        )
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError):
        return []
    if body.get("success") is False:
        return []
    return body.get("entities", []) or []


def _poll_shelter_watch() -> str | None:
    """Poll hostiles, update watcher state, return a status string or None.

    Returns a string only when there's something worth surfacing:
    - First confirmation of a breach (mob inside cavity ≥2 consecutive polls)
    - Continued breach with live mob still inside on this poll
    Otherwise None — we don't spam the LLM with "shelter ok" lines.

    Also auto-disarms when the player drifts >12 blocks from anchor — the
    anchor is stale once the agent has wandered off, e.g. after evacuating.
    """
    global _shelter_watch
    s = _shelter_watch
    if s is None:
        return None

    # Drift check. Read /position; if far from anchor, drop the watcher.
    try:
        pos_resp = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=5.0)
        pos = pos_resp.json() if pos_resp.ok else {}
    except (requests.RequestException, ValueError):
        pos = {}
    ax, ay, az = s["anchor"]
    px_p = pos.get("x")
    pz_p = pos.get("z")
    if isinstance(px_p, (int, float)) and isinstance(pz_p, (int, float)):
        if abs(px_p - ax) > 12 or abs(pz_p - az) > 12:
            print("[shelter_watch] disarmed: player drifted >12 from anchor", flush=True)
            _shelter_watch = None
            return None

    x_lo, x_hi = ax - 2, ax + 2
    z_lo, z_hi = az - 2, az + 2
    y_lo, y_hi = ay, ay + 1

    def inside_block(ex: float, ey: float, ez: float) -> bool:
        bx, by, bz = math.floor(ex), math.floor(ey), math.floor(ez)
        return x_lo <= bx <= x_hi and z_lo <= bz <= z_hi and y_lo <= by <= y_hi

    t = round(time.time() - s["started_at"], 1)
    per_uuid = s["per_uuid"]
    seen: set[str] = set()
    new_confirmed: list[tuple[str, dict]] = []

    for mob in _SHELTER_HOSTILE_TYPES:
        for ent in _scan_hostile(mob):
            uuid = ent.get("uuid")
            if not uuid:
                continue
            seen.add(uuid)
            epos = ent.get("position") or [None, None, None]
            here = (
                epos[0] is not None
                and isinstance(epos[0], (int, float))
                and inside_block(epos[0], epos[1], epos[2])
            )
            rec = per_uuid.get(uuid)
            if rec is None:
                rec = {
                    "type": mob, "first_t": t, "consec_in": 0,
                    "confirmed": False, "confirmed_t": None, "confirmed_pos": None,
                }
                per_uuid[uuid] = rec
            if here:
                rec["consec_in"] += 1
                if not rec["confirmed"] and rec["consec_in"] >= 2:
                    rec["confirmed"] = True
                    rec["confirmed_t"] = t
                    rec["confirmed_pos"] = list(epos)
                    new_confirmed.append((uuid, rec))
                    if s["breach_first_t"] is None:
                        s["breach_first_t"] = t
                    s["breach"] = True
            else:
                rec["consec_in"] = 0

    # Unseen uuids (out of scan range, chunk unloaded) reset consec.
    for uuid, rec in per_uuid.items():
        if uuid not in seen:
            rec["consec_in"] = 0

    if new_confirmed:
        uuid, rec = new_confirmed[0]
        cp = rec.get("confirmed_pos") or [None, None, None]
        mtype = rec["type"].split(":")[-1]
        return (
            f"SHELTER BREACHED at t={rec['confirmed_t']}s: "
            f"{mtype} at ({cp[0]},{cp[1]},{cp[2]}) inside cavity "
            f"(anchor=({ax},{ay},{az})). Re-call build_shelter() to repair "
            f"(re-patches the hole + re-installs the door). No other repair primitive exists."
        )

    live_inside = [
        rec for uuid, rec in per_uuid.items()
        if rec["confirmed"] and uuid in seen and rec["consec_in"] >= 1
    ]
    if live_inside and s["breach"]:
        types = ",".join(sorted({r["type"].split(":")[-1] for r in live_inside}))
        return (
            f"SHELTER STILL BREACHED: {len(live_inside)}x [{types}] in cavity. "
            f"build_shelter() to repair or evacuate."
        )

    return None


SYSTEM_PROMPT = (
    "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
    "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens — "
    "they are not part of your output schema.\n\n"
    "You are a Minecraft agent. Goal: acquire a diamond. "
    "Emit ONE tool call per response. Do not emit text content. "
    "Inventory will be shown each turn — read it, then call the next tool.\n\n"
    "Tools (use exactly one per turn):\n"
    "- mine_wood(quantity), mine_stone(quantity) [forced blind-tunnel — descend(30) first for stone layers], mine_coal(quantity, fair?), mine_iron(quantity, fair?), mine_diamond(quantity, fair?)\n"
    "  fair=true on ore mines → BLIND TUNNEL at your CURRENT y (1×2 corridor forward, player walks through). Pattern: descend() to right y, then mine_X(fair=true).\n"
    "- craft(item, quantity, location?) — recursive: handles sub-recipes and table placement\n"
    "- smelt(input, count, location?) — returns IMMEDIATELY. Cook is async (~10s/item). Auto-places furnace, auto-picks fuel.\n"
    "- collect_smelt(furnace_pos?) — pull outputs from your furnace. Call after smelt() once 'Active smelts' shows READY.\n"
    "- place(item) — rarely needed; craft/smelt place tables/furnaces\n"
    "- surface() — go up to sky\n"
    "- descend(target_y) — dig down to Y\n"
    "- travel(direction, distance) — walk N blocks N/S/E/W (cap 64)\n"
    "\nGoal completion: if your inventory already shows the goal item (a diamond), "
    "the run is OVER. Emit NO tool call. The harness ends the run when you stop "
    "emitting tool calls. Don't loop on surface() or other busywork.\n"
    "\nMovement notes:\n"
    "- descend() and surface() are CHUNKED: each call moves you part of the way and returns. For a deep target, just call the same tool again whenever the result says 'more — call ... again'. This is normal progress, not a failure.\n"
    "- A 'PARTIAL' outcome means Baritone failed mid-chunk. Don't retry the same thing — switch strategy (mine_stone(8) to dig by hand, or travel to a different column).\n"
    "- Never call surface() repeatedly when already at surface (Δy < 2). Pick a new action.\n"
)


SURVIVE_PROMPT = (
    "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
    "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens — "
    "they are not part of your output schema.\n\n"
    "You are a Minecraft agent. Goal: SURVIVE — stay alive and build useful inventory. "
    "There is no win condition; the run ends only when you hit the turn cap. "
    "Each turn you'll see your stats (HP, food, saturation, air, status flags) and inventory.\n"
    "Emit ONE tool call per response. Do not emit text content.\n\n"
    "Tools (use exactly one per turn):\n"
    "- mine_wood(quantity), mine_stone(quantity) [forced blind-tunnel — descend(30) first for stone layers], mine_coal(quantity, fair?), mine_iron(quantity, fair?), mine_diamond(quantity, fair?)\n"
    "  fair=true on ore mines → BLIND TUNNEL at your CURRENT y (1×2 corridor forward, player walks through). Pattern: descend() to right y, then mine_X(fair=true).\n"
    "- craft(item, quantity, location?) — recursive: handles sub-recipes and table placement\n"
    "- smelt(input, count, location?) — returns IMMEDIATELY. Cook is async (~10s/item). Auto-places furnace, auto-picks fuel.\n"
    "- collect_smelt(furnace_pos?) — pull outputs from your furnace. Call after smelt() once 'Active smelts' shows READY.\n"
    "- place(item) — rarely needed; craft/smelt place tables/furnaces\n"
    "- surface() — go up to sky\n"
    "- descend(target_y) — dig down to Y\n"
    "- travel(direction, distance) — walk N blocks N/S/E/W (cap 64)\n"
    "- build_shelter() — seal a 5×2×5 stone cavity around you with a north-facing door. "
    "Needs a wooden door in inventory (craft('oak_door', 1) — 6 planks → 3 doors) "
    "AND enough cobblestone/dirt/etc. to wall in (~90 blocks total). Surface-only — "
    "skip when underground. Re-calling it from inside the same shelter REPAIRS holes "
    "and re-installs the door; that's the only repair path.\n"
    "\nAuto-systems (substrate, not strategy — do NOT fight these):\n"
    "- Any mob in melee range is auto-killed by KillAura. You do NOT need to flee or avoid mobs.\n"
    "  EXCEPTION: creepers detonate before they die. Keep distance from creepers specifically.\n"
    "- Any edible item in inventory is auto-eaten when hungry (AutoEat). To recover food, ACQUIRE edibles — don't try to consume them.\n"
    "- Mobs drop food (cows/pigs/sheep → meat; zombies → rotten flesh, still edible). Travel through varied terrain when food is low; mob contact = food acquisition.\n"
    "\nPRIORITY HIERARCHY (when rules conflict, higher rank wins):\n"
    "  1. EMERGENCY — in_lava, drowning, HP<10. Drop everything. See threat response.\n"
    "  2. SHELTER — time=DAY <2min until dusk OR time=NIGHT. Stop gathering/exploring/crafting. "
    "build_shelter() if not already in one. Damage in MC arrives in BURSTS (one zombie hit = 3HP, "
    "one skeleton volley = 8-12HP, one creeper = 49HP) — reactive HP-threshold rules cannot save you. "
    "Proactive shelter before night is the ONLY safe strategy.\n"
    "  3. TECH-TREE / GATHERING / EXPLORATION — only while DAY and >2min until dusk.\n"
    "\nDay/night cycle (shown in stats as time=DAY/NIGHT Xmin until dusk/dawn):\n"
    "- DAY (>2min until dusk): surface is safe — gather, mine_wood/stone, smelt, travel freely.\n"
    "- DUSK approaching (<2min until dusk): STOP NEW tech-tree work. If not in a shelter, "
    "build_shelter() NOW (craft a door if needed: craft('oak_door', 1)). Finish current craft "
    "INSIDE the shelter. Do not start new chains on the surface.\n"
    "- NIGHT: stay in shelter. Use the time for indoor crafts/smelts. AVOID open-surface travel "
    "and AVOID descend()/mining at depth — caves spawn hostiles regardless of light, and the "
    "tech-tree can resume safely at dawn (≤8 minutes away).\n"
    "- DAWN approaching (<2min until dawn): safe to plan post-night moves; mobs burn at sunrise.\n"
    "\nThreat response (priority order, override any other plan):\n"
    "- in_lava=true → surface() immediately. Lava ticks HP fast.\n"
    "- air ≤ 0 AND in_water=true → surface() immediately, you're drowning.\n"
    "- HP < 10 → travel to open flat terrain (not caves, not edges). HP regenerates when food ≥ 18.\n"
    "\nMovement notes:\n"
    "- descend() and surface() are CHUNKED: each call moves you part of the way and returns. Call again until the result no longer says 'more — call ... again'.\n"
    "- A 'PARTIAL' outcome means Baritone failed mid-chunk. Don't retry the same thing — switch strategy.\n"
    "- Never call surface() repeatedly when already at surface (Δy < 2). Pick a new action.\n"
)


SURVIVE_FIRST_PROMPT = SURVIVE_PROMPT.replace(
    "You are a Minecraft agent. Goal: SURVIVE — stay alive and build useful inventory. ",
    (
        "You are a Minecraft agent. Goal: SURVIVE — stay alive and climb the tech tree IN ORDER. "
        "TECH-TREE ORDER (no skipping; safety-pause permitted — see PRIORITY HIERARCHY below): "
        "wooden_pickaxe → wooden_axe + wooden_shovel + wooden_sword → "
        "stone_pickaxe → stone_axe + stone_sword → "
        "coal (smelt fuel — 1 coal = 8 smelts; required BEFORE any smelt or it burns "
        "planks/logs at 1.5 items each, which is wasteful and slow) → "
        "iron ore (needs stone_pickaxe to mine; smelt raw_iron → iron_ingot with coal) → "
        "iron_pickaxe AND iron armor (iron_chestplate 8 ingots + iron_helmet 5 ingots) → "
        "diamond ore (needs iron_pickaxe; deepest at Y≈-58 to -64). "
        "Why this order matters: axe makes mine_wood ~3x faster; shovel speeds dirt/sand mining "
        "and build_shelter excavation; sword enables KillAura mob auto-kill (it needs a weapon "
        "to work); coal unlocks efficient smelting; iron armor before diamond is non-negotiable — "
        "rushing to diamond without armor is the leading cause of death. "
        "Tech-tree is RANK 3 — it PAUSES at dusk and during night (RANK 2 takes over). "
        "Resume the current step at dawn; do NOT start a new step on the surface after dusk. "
    ),
)


SURVIVE_SHELTER_PROMPT = SURVIVE_FIRST_PROMPT.replace(
    "Threat response (priority order, override any other plan):\n",
    (
        "Threat response (priority order, override any other plan):\n"
        "- HP dropping with no obvious threat (you're being attacked from out of sight) OR HP < 12 with mobs nearby → "
        "build_shelter(). PRECONDITION: a wooden door in inventory AND ≥70 cobblestone/dirt/planks/logs (mixed buildables OK). "
        "If you don't have a door, craft('oak_door', 1) first (needs 6 planks, yields 3). "
        "If you don't have wall stock, mine_stone(10) first. Build_shelter handles the rest: "
        "carves a 5×2×5 room around you, walls it in, cuts the door, installs it. "
        "On 'SHELTER BREACHED' feedback from the watcher, call build_shelter() again — "
        "it re-patches the hole and re-installs the door. That's the only repair primitive.\n"
        "- ONCE A SHELTER IS BUILT AT DUSK OR NIGHT: STAY IN IT until dawn. Do NOT travel >10 blocks "
        "from the shelter anchor (the (X,Y,Z) reported by build_shelter). After the watcher disarms "
        "you have no protection. Pre-dawn rebuilds frequently fail with 'excavate: stuck' because "
        "mobs interfere with Baritone — the right move is to USE indoor turns for crafting / smelting / "
        "collect_smelt, not to abandon the shelter. Travel + new build_shelter is a DAY-time activity.\n"
    ),
)


MINIMAL_PROMPT = (
    "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
    "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens.\n\n"
    "You are a Minecraft survival agent. Stay alive as long as possible. "
    "Each turn you will see your current stats and inventory. Call one tool.\n\n"
    "Tools:\n"
    "- mine_wood(quantity)\n"
    "- mine_stone(quantity)\n"
    "- mine_coal(quantity, fair?)\n"
    "- mine_iron(quantity, fair?)\n"
    "- mine_diamond(quantity, fair?)\n"
    "  fair=true → blind 1×2 tunnel at current Y (no candidate targeting).\n"
    "- craft(item, quantity, location?) — handles sub-recipes; places crafting table / furnace as needed\n"
    "- smelt(input, count, location?) — async; returns immediately; auto-places furnace, auto-picks fuel\n"
    "- collect_smelt(furnace_pos?) — pull finished outputs from furnace\n"
    "- place(item)\n"
    "- surface() — navigate up to sky\n"
    "- descend(target_y) — dig down to Y\n"
    "- travel(direction, distance) — walk up to 64 blocks N/S/E/W\n"
    "- look_around(radius?) — read-only scout call; describes nearby terrain, "
    "hazards, and resources with cardinal hints. radius=1 ~3s (1 chunk), "
    "radius=2 ~5s (3×3, default), radius=3 ~8s (5×5).\n"
    "- build_shelter() — carve a 5×2×5 room, wall it in, install a door. "
    "Requires: wooden door in inventory + ~90 solid blocks (cobblestone/dirt/etc). Surface only.\n"
    "- wall_in() — tunnel 3 cells into the nearest cardinal wall and seal yourself in "
    "(1-cell foyer + 2 blocks placed + 2-cell back cavity). Requires: flush against a "
    "solid wall (cave wall, hillside). Tunnel produces enough stone for the seal.\n"
    "- carve_alcove() — carve a 2×3 alcove off your back cavity for crafting_table / "
    "furnace / bed. Requires: an active wall_in pocket.\n\n"
    "Substrate facts (automatic — do not try to override):\n"
    "- Nearby mobs are auto-attacked (KillAura). Creepers detonate before dying — keep your distance.\n"
    "- Food in inventory is auto-eaten when hungry (AutoEat).\n\n"
    "Movement: descend() and surface() move in chunks. "
    "If the result says 'call again', call it again. "
    "PARTIAL = Baritone failed mid-path — switch strategy, do not retry the same call.\n"
)


BARE_PROMPT = (
    "You are playing Minecraft. Call one tool.\n\n"
    "Tools:\n"
    "- mine_wood(quantity)\n"
    "- mine_stone(quantity)\n"
    "- mine_coal(quantity, fair?)\n"
    "- mine_iron(quantity, fair?)\n"
    "- mine_diamond(quantity, fair?)\n"
    "  fair=true → blind 1×2 tunnel at current Y (no candidate targeting).\n"
    "- craft(item, quantity, location?) — handles sub-recipes; places crafting table / furnace as needed\n"
    "- smelt(input, count, location?) — async; returns immediately; auto-places furnace, auto-picks fuel\n"
    "- collect_smelt(furnace_pos?) — pull finished outputs from furnace\n"
    "- place(item)\n"
    "- surface() — navigate up to sky\n"
    "- descend(target_y) — dig down to Y\n"
    "- travel(direction, distance) — walk up to 64 blocks N/S/E/W\n"
    "- look_around(radius?) — read-only scout call; describes nearby terrain, "
    "hazards, and resources with cardinal hints. radius=1 ~3s (1 chunk), "
    "radius=2 ~5s (3×3, default), radius=3 ~8s (5×5).\n"
    "- build_shelter() — carve a 5×2×5 room, wall it in, install a door. "
    "Requires: wooden door in inventory + ~90 solid blocks (cobblestone/dirt/etc). Surface only.\n"
    "- wall_in() — tunnel 3 cells into the nearest cardinal wall and seal yourself in "
    "(1-cell foyer + 2 blocks placed + 2-cell back cavity). Requires: flush against a "
    "solid wall (cave wall, hillside). Tunnel produces enough stone for the seal.\n"
    "- carve_alcove() — carve a 2×3 alcove off your back cavity for crafting_table / "
    "furnace / bed. Requires: an active wall_in pocket.\n"
)


GOAL_PROMPTS = {
    "diamond": SYSTEM_PROMPT,
    "survive": SURVIVE_PROMPT,
    "survive_first": SURVIVE_FIRST_PROMPT,
    "survive_shelter": SURVIVE_SHELTER_PROMPT,
    "minimal": MINIMAL_PROMPT,
    "bare": BARE_PROMPT,
}


def _fetch_new_deaths(since_ms: int) -> list[dict]:
    """Poll /deaths?since=N for death records the harness hasn't yet surfaced.

    Returns a list (possibly empty) of death records. Transport failures
    silently return []; deaths are nice-to-have context, not load-bearing.
    """
    try:
        resp = requests.get(
            f"{HOMUNCULUS_BASE}/deaths",
            params={"since": since_ms},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []
    return data.get("deaths", [])


def _format_death(d: dict) -> str:
    """Render a death record as a one-line log entry for the trajectory."""
    dp = d.get("death_pos") or [None, None, None]
    msg = d.get("message", "you died")
    cause = d.get("cause", "unknown")
    return f"YOU DIED: {msg} (cause: {cause}). Died at ({dp[0]},{dp[1]},{dp[2]})."


def _fetch_stats() -> str | None:
    """Fetch /stats and render a compact one-liner for the LLM context.

    Always shows HP/food/saturation/air (they change every turn and inform
    survive-mode decisions). Status flags (in_water, in_lava, on_fire, dim,
    effects) are only rendered when non-default — keeps the line short
    when nothing's wrong.
    """
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/stats", timeout=5.0)
        resp.raise_for_status()
        s = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if s.get("success") is False:
        return None

    parts = [
        f"HP={s.get('health', '?')}/{s.get('max_health', '?')}",
        f"food={s.get('food', '?')}",
        f"sat={s.get('saturation', '?')}",
        f"air={s.get('air', '?')}",
        # Player armor rating (0–20). Full iron set ≈ 15; full diamond = 20.
        # Distinct from the Equipment block (which lists *which* armor is
        # worn) — this is the aggregate damage-reduction stat. Belongs in
        # the always-on stats line so it shows under both toggle arms.
        f"armor={s.get('armor', '?')}",
    ]
    # Position + facing. A human player gets F3 on demand; the agent used to
    # only learn coordinates from tool result strings (travel/build_shelter/
    # place) and could lose orientation between calls. Always-on pos solves
    # that — and facing helps because fair-mode mine_* tunnels in the
    # cardinal direction the player yaw is closest to.
    try:
        pr = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=5.0)
        pr.raise_for_status()
        p = pr.json()
        if isinstance(p, dict) and "x" in p and "y" in p and "z" in p:
            px = int(p["x"]) if isinstance(p["x"], (int, float)) else p["x"]
            py = int(p["y"]) if isinstance(p["y"], (int, float)) else p["y"]
            pz = int(p["z"]) if isinstance(p["z"], (int, float)) else p["z"]
            parts.append(f"pos=({px},{py},{pz})")
            yaw = p.get("yaw")
            if isinstance(yaw, (int, float)):
                parts.append(f"facing={_yaw_to_direction(float(yaw))}")
    except (requests.RequestException, ValueError):
        pass  # transport blip — keep the rest of the stats line
    # wall_in anchor hint. After a successful wall_in() the agent is in a
    # 1×3 tube with a seal at cell 2 — easy to forget without F3-equivalent.
    # Same shape as the shelter anchor hint logged by build_shelter.
    bstate = get_burrow_state()
    if bstate is not None:
        ax, ay, az = bstate["anchor"]
        sx, sy, sz = bstate["seal"]
        parts.append(
            f"wall_in=({ax},{ay},{az}) seal=({sx},{sy},{sz}) "
            f"dir={bstate.get('direction', '?')}"
        )
    # Day-time hint. MC day cycle: 0=sunrise, 12000=dusk, 24000=next dawn.
    # 24000 ticks = 20 real-time minutes ⇒ 1200 ticks/min. The agent uses
    # this to prioritize (shelter before nightfall; safe to explore at dawn).
    day_ticks = s.get("day_ticks")
    day_count = s.get("day_count")
    if isinstance(day_ticks, (int, float)):
        if day_ticks < 12000:
            mins = (12000 - day_ticks) / 1200
            tstr = f"time=DAY {mins:.1f}min until dusk"
        else:
            mins = (24000 - day_ticks) / 1200
            tstr = f"time=NIGHT {mins:.1f}min until dawn"
        if isinstance(day_count, (int, float)):
            tstr += f" (day {int(day_count)})"
        parts.append(tstr)
    # Biome hint. Strong context for survival decisions: forest→easy wood,
    # desert→no wood, ocean→stranded, mushroom→no hostiles. Stripped to
    # the short name (e.g. minecraft:plains → plains).
    biome = s.get("biome")
    if isinstance(biome, str) and biome:
        parts.append(f"biome={biome.split(':')[-1]}")
    flags = []
    if s.get("in_water"):
        flags.append("in_water")
    if s.get("in_lava"):
        flags.append("in_lava")
    if s.get("on_fire"):
        flags.append("on_fire")
    effects = s.get("effects") or []
    if effects:
        names = ",".join(e.get("id", "?").split(":")[-1] for e in effects)
        flags.append(f"effects=[{names}]")
    dim = s.get("dimension")
    if dim and dim != "minecraft:overworld":
        flags.append(f"dim={dim}")
    if flags:
        parts.append(" ".join(flags))
    return "Stats: " + " ".join(parts)


def _fetch_smelts() -> str | None:
    """Fetch /smelt_status and render active smelts as a context block.

    Returns None when the registry is empty so the prompt stays clean.
    Transport failures also return None — smelts are nice-to-have context,
    not load-bearing.
    """
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/smelt_status", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    smelts = data.get("smelts") or []
    if not smelts:
        return None

    now_ms = int(time.time() * 1000)
    lines = ["Active smelts:"]
    for s in smelts:
        fp = s.get("furnace_pos") or [0, 0, 0]
        out = s.get("output", {})
        out_id = out.get("id", "?")
        ready = out.get("count_ready", 0)
        inp = s.get("input", {})
        remaining = inp.get("count_remaining", 0)
        status = (s.get("status") or "?").upper()
        eta = s.get("eta_seconds")
        eta_str = f"~{eta}s" if isinstance(eta, (int, float)) else "unknown ETA"

        if status == "READY":
            tail = f"{ready}x {out_id} READY — call collect_smelt()"
        elif status == "PARTIAL":
            tail = (
                f"{ready}x {out_id} ready, {remaining} still cooking ({eta_str}) — "
                f"call collect_smelt() to pull what's done"
            )
        elif status == "STALE":
            last_ms = s.get("last_observed_ms")
            if isinstance(last_ms, (int, float)):
                ago_s = max(0, (now_ms - int(last_ms)) // 1000)
                ago_str = f"last seen {ago_s}s ago"
            else:
                ago_str = "chunk unloaded"
            tail = (
                f"STALE ({ago_str}): last known {ready}x {out_id} ready, "
                f"{remaining}x {inp.get('id', '?')} remaining. Call collect_smelt() "
                f"to walk back and reconcile."
            )
        elif status == "DESTROYED":
            tail = "DESTROYED — items lost"
        elif status == "EMPTY":
            tail = "empty (cleanup pending)"
        else:  # cooking or unknown
            tail = f"{remaining}x {inp.get('id', '?')} → {out_id}, {eta_str} remaining"
        lines.append(f"  furnace ({fp[0]},{fp[1]},{fp[2]}): {tail}")
    return "\n".join(lines)


# Tier ladder + slot tables for the per-turn Equipment readout. Homunculus
# auto-equips the best tier on /equip, so "best-of-class anywhere in
# inventory" matches what the agent actually wields. The readout doubles as
# an implicit nudge: a vacant slot ("you have no helmet!") signals "craft
# this" without prose. Adding pickaxe → diamond_pickaxe is just inventory
# changing; the same template flips from vacant to "diamond_pickaxe".
_TOOL_TIERS: tuple[str, ...] = ("netherite", "diamond", "iron", "stone", "golden", "wooden")
_TOOL_SLOTS: tuple[tuple[str, str, str], ...] = (
    # (label,         item suffix, vacant phrase)
    ("best weapon",  "sword",   "you are unarmed!"),
    ("best shovel",  "shovel",  "you are digging barehanded!"),
    ("best pickaxe", "pickaxe", "you cannot mine stone yet!"),
    ("best axe",     "axe",     "you are chopping barehanded!"),
)
_ARMOR_TIERS: tuple[str, ...] = ("netherite", "diamond", "iron", "chainmail", "golden", "leather")
_ARMOR_SLOTS: tuple[tuple[str, str], ...] = (
    # (label,        item suffix)
    ("helmet",     "helmet"),
    ("chestplate", "chestplate"),
    ("leggings",   "leggings"),
    ("boots",      "boots"),
)
# Materials that prove an armor tier is craftable, best-tier-first. A vacant
# armor slot only gets a line when at least one of these is in inventory —
# otherwise the readout suggests an action the agent can't take. Without this
# gating, qwen confabulates `wooden_helmet` (no such recipe) in a tight loop
# (validated 2026-05-20 replay: 5/5 → wooden_helmet without gating, 0/5 with).
_ARMOR_MATERIAL_PROBES: tuple[tuple[str, str], ...] = (
    # (material name, registry id whose presence proves the tier is craftable)
    ("netherite", "minecraft:netherite_ingot"),
    ("diamond",   "minecraft:diamond"),
    ("iron",      "minecraft:iron_ingot"),
    ("golden",    "minecraft:gold_ingot"),
    ("leather",   "minecraft:leather"),
)


def _all_item_ids(inv: dict | None) -> set[str]:
    """All item ids present anywhere: main + offhand + armor slots."""
    if not inv:
        return set()
    ids: set[str] = set()
    for slot in (inv.get("main") or []):
        if slot.get("id"):
            ids.add(slot["id"])
    oh = inv.get("offhand")
    if oh and oh.get("id"):
        ids.add(oh["id"])
    for armor_slot in (inv.get("armor") or {}).values():
        if armor_slot and armor_slot.get("id"):
            ids.add(armor_slot["id"])
    return ids


def _best_tier_id(ids: set[str], suffix: str, tier_order: tuple[str, ...]) -> str | None:
    """Highest-tier item id matching f'minecraft:{tier}_{suffix}', or None."""
    for tier in tier_order:
        candidate = f"minecraft:{tier}_{suffix}"
        if candidate in ids:
            return candidate
    return None


def _best_craftable_armor_material(ids: set[str]) -> str | None:
    """First (best-tier) material from _ARMOR_MATERIAL_PROBES present in `ids`."""
    for material, probe_id in _ARMOR_MATERIAL_PROBES:
        if probe_id in ids:
            return material
    return None


def _armor_nudge_gating_enabled() -> bool:
    """Whether armor lines are craftability-gated (default ON post-2026-05-20).

    Toggle via CRAFT_ARMOR_NUDGE_GATING ("0"/"false"/"off" reverts to the
    pre-fix legacy "you have no helmet!" nudge for every armor slot, used
    for A/B regression studies). Resolved per-call so a long-running
    orchestrator can flip per rollout.
    """
    raw = os.environ.get("CRAFT_ARMOR_NUDGE_GATING", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _render_equipment(inv: dict | None) -> list[str]:
    """Per-slot best-of-class equipment block for the STATE readout.

    Tool slots always render (vacant phrase + nudge), since wood-tier exists
    for every tool and `craft(wooden_<tool>)` is a valid recipe from logs.
    Armor slots are craftability-gated by default: render when (a) equipped
    or (b) materials are present to craft a real tier; otherwise the line
    is omitted to avoid nagging the agent into hallucinating `wooden_helmet`
    (no such recipe). Setting CRAFT_ARMOR_NUDGE_GATING=0 reverts to legacy
    "you have no helmet!" for every slot — used for A/B studies.
    """
    lines = ["Equipment:"]
    ids = _all_item_ids(inv)
    for label, suffix, vacant in _TOOL_SLOTS:
        best = _best_tier_id(ids, suffix, _TOOL_TIERS)
        if best:
            lines.append(f"  {label}: {best.split(':', 1)[-1]}")
        else:
            lines.append(f"  {label}: {vacant} (no {suffix} crafted yet)")
    gating = _armor_nudge_gating_enabled()
    material = _best_craftable_armor_material(ids) if gating else None
    for label, suffix in _ARMOR_SLOTS:
        best = _best_tier_id(ids, suffix, _ARMOR_TIERS)
        if best:
            lines.append(f"  {label}: {best.split(':', 1)[-1]}")
        elif not gating:
            lines.append(f"  {label}: you have no {label}! (no {label} crafted yet)")
        elif material is not None:
            lines.append(f"  {label}: none — {material}_{suffix} craftable")
        # else: gating on, no materials → omit the line entirely
    return lines


def _equipment_readout_enabled() -> bool:
    """Whether the Equipment block is prepended to the inventory readout.

    Toggle via CRAFT_EQUIPMENT_READOUT env var ("0"/"false"/"off" disables;
    anything else, including unset, leaves it on). Resolved per-call so a
    single in-process suite can A/B by flipping the var between rollouts.
    """
    raw = os.environ.get("CRAFT_EQUIPMENT_READOUT", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _autodrop_tier_setting() -> str:
    """Tier the AutoDrop whitelist is seeded to at startup.

    Env: CRAFT_AUTODROP_TIER (default "bare"). Set to "off" to skip seeding —
    AutoDrop reverts to Wurst's tiny default filter (flowers + rotten_flesh).
    "stone"/"iron"/"diamond" force higher tiers for testing.
    """
    raw = os.environ.get("CRAFT_AUTODROP_TIER", "bare").strip().lower()
    if raw in ("0", "false", "off", "no", ""):
        return "off"
    if raw not in ("bare", "stone", "iron", "diamond"):
        print(f"[autodrop] WARN unknown CRAFT_AUTODROP_TIER={raw!r}; defaulting to 'bare'", flush=True)
        return "bare"
    return raw


def _format_inventory(inv: dict | None) -> str | None:
    """Render inventory as (optional) Equipment block + raw slot listing.

    Returns None when `inv` is None (transport error upstream); callers
    translate that into the STATE-level "(unavailable …)" literal.

    The Equipment block is toggleable for A/B comparison — see
    `_equipment_readout_enabled`. When off, the output matches the
    pre-Equipment-block format (raw slot listing only).
    """
    if inv is None:
        return None
    lines: list[str] = []
    if _equipment_readout_enabled():
        lines.extend(_render_equipment(inv))
        lines.append("")
    lines.append("Current inventory:")
    main = inv.get("main") or []
    if main:
        for slot in main:
            slot_num = slot.get("slot", "?")
            item_id = slot.get("id", "<?unknown>")
            count = slot.get("count", 1)
            lines.append(f"  slot {slot_num}: {count}x {item_id}")
    else:
        lines.append("  (empty)")
    offhand = inv.get("offhand")
    if offhand:
        item_id = offhand.get("id", "<?unknown>")
        count = offhand.get("count", 1)
        lines.append(f"  offhand: {count}x {item_id}")
    return "\n".join(lines)


def _fetch_inventory() -> str | None:
    """Fetch current inventory and render for the prompt. None on transport error."""
    return _format_inventory(_inventory_raw())


def _build_state_chunk(
    stats_str: str | None,
    inv_str: str | None,
    smelts_str: str | None,
    nudges_str: str | None = None,
) -> str:
    """Render the per-turn STATE body delivered as its own user message.

    World state (stats / inventory / active smelts) used to be concatenated
    onto each tool result, which left the *opening* user message pinned to
    spawn-time stats forever and made SFT extraction ambiguous (which
    `Stats:` line is the live one?). Splitting state into its own message
    gives every turn a single canonical state slot the model attends to.

    Always returns a non-empty string. Transport failures surface as
    explicit "(unavailable …)" literals rather than silent omission, so a
    homunculus blip doesn't silently strip state from the prompt.

    `nudges_str`, when present, is appended last so the reactive hint is the
    final thing the model reads before choosing its next call (recency).
    """
    parts: list[str] = []
    if stats_str:
        parts.append(stats_str)
    else:
        parts.append("Stats: (unavailable — homunculus transport error)")
    if inv_str:
        parts.append(inv_str)
    else:
        parts.append("Current inventory: (unavailable — homunculus transport error)")
    if smelts_str:
        parts.append(smelts_str)
    if nudges_str:
        parts.append(nudges_str)
    return "STATE:\n" + "\n\n".join(parts)


def _render_nudges(nudge_chain: list, stats_raw: dict | None, inv_raw: dict | None) -> str | None:
    """Render the active reactive nudges from current stats + inventory.

    Sister to _build_state_chunk's `nudges_str`. None when the chain is empty,
    stats are unavailable, or no nudge's condition holds this turn.
    """
    if not nudge_chain or not stats_raw:
        return None
    state = {
        "food": stats_raw.get("food"),
        "day_ticks": stats_raw.get("day_ticks"),
        "day_count": stats_raw.get("day_count"),
        "inv": _inventory_compact(inv_raw),
    }
    return render_nudges(nudge_chain, state)


def _stats_raw() -> dict | None:
    """Raw /stats response, or None on transport error. Sister to _fetch_stats."""
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _inventory_raw() -> dict | None:
    """Raw /inventory response. Sister to _fetch_inventory."""
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/inventory", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _inventory_compact(inv: dict | None) -> dict[str, int]:
    """Flatten /inventory main+offhand+armor into {item_id: total_count}.

    The `armor` slot dict ({head,chest,legs,feet} -> {id,count} | None)
    must be included — equipped armor is the natural state we want
    predicates (e.g. M2_diamond_goal) to recognize, and worn pieces are
    NOT mirrored in `main`. Original implementation dropped `armor` on
    the floor; reproduced via `--starting-loadout iron_armored` smoke.
    """
    if not inv:
        return {}
    out: dict[str, int] = {}
    for slot in (inv.get("main") or []):
        item_id = slot.get("id")
        if item_id:
            out[item_id] = out.get(item_id, 0) + int(slot.get("count", 1))
    oh = inv.get("offhand")
    if oh and oh.get("id"):
        out[oh["id"]] = out.get(oh["id"], 0) + int(oh.get("count", 1))
    armor = inv.get("armor") or {}
    if isinstance(armor, dict):
        for piece in armor.values():
            if isinstance(piece, dict) and piece.get("id"):
                out[piece["id"]] = out.get(piece["id"], 0) + int(piece.get("count", 1))
    return out


def _evasion_arm(x: float, y: float, z: float) -> bool:
    """POST /evasion/arm. Returns True on 200/success, False otherwise (no exception).

    Per the outside-the-handler design: every turn arms with the current
    player position. The homunculus-side watcher autonomously cancels
    Baritone and routes the player back here on the first hostile-mob hit.
    Handlers don't participate.
    """
    try:
        r = requests.post(
            f"{HOMUNCULUS_BASE}/evasion/arm",
            json={"x": x, "y": y, "z": z},
            timeout=3.0,
        )
        return r.ok and r.json().get("success") is True
    except (requests.RequestException, ValueError):
        return False


def _evasion_disarm() -> None:
    """POST /evasion/disarm. Best-effort — failures aren't surfaced.

    Disarm does NOT cancel a flee in progress (Java side: just clears state).
    The player keeps walking until either arrival or the next Baritone task
    overrides. Called at the top of each turn (re-arm semantics) and at
    rollout end.
    """
    try:
        requests.post(f"{HOMUNCULUS_BASE}/evasion/disarm", timeout=3.0)
    except requests.RequestException:
        pass


def _evasion_status() -> dict | None:
    """GET /evasion/status. Returns the parsed body or None on transport error.

    Shape: {success, armed, fired, anchor, attackers, flee_state,
    flee_failure_reason?}. flee_state ∈ idle/in_progress/arrived/timeout/failed.
    """
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/evasion/status", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _format_evaded_preamble(status: dict, tool_name: str) -> str:
    """Render an EVADED preamble for the tool result.

    Names the attackers and the turn-start anchor, makes clear the flee is
    still running, and primes the LLM with the three fight-or-flight pivots
    (return to safety / run further / fight back). Existing tools cover all
    three — no new tool surface, just a contextual hint.
    """
    attackers = status.get("attackers") or []
    atk = ", ".join(a.split(":")[-1] for a in attackers) if attackers else "unknown hostile mob"
    anchor = status.get("anchor")
    if isinstance(anchor, list) and len(anchor) == 3:
        ax, ay, az = (int(v) if isinstance(v, (int, float)) else "?" for v in anchor)
        loc = f"({ax},{ay},{az})"
    else:
        loc = "turn-start anchor"
    flee = status.get("flee_state", "in_progress")
    # When the flee already finished (rare — the LLM call ran longer than
    # the walk back), the framing changes: agent IS at safety now.
    if flee == "arrived":
        progress = f"reflexive flee complete — player is back at {loc}"
    elif flee in ("timeout", "failed"):
        progress = (
            f"reflexive flee {flee} (reason: {status.get('flee_failure_reason', 'unknown')}); "
            f"may NOT be at {loc} — verify pos before assuming safety"
        )
    else:
        progress = f"reflexive flee IN PROGRESS toward {loc}; player still walking back"
    return (
        f"EVADED: attacked by {atk} during {tool_name}; {progress}. "
        f"Your next tool call will override the flee. Options: "
        f"build_shelter (if dusk/night and not already in one), "
        f"travel(<dir>, <dist>) to run further, "
        f"or continue the tech tree if you have a sword + armor and want to fight."
    )


def _water_aversion_arm() -> bool:
    """POST /water_aversion/arm. Returns True on 200/success.

    Same per-turn-rearm pattern as evasion. No anchor — the dry-land target
    is computed at fire-time by the Java-side BFS, not supplied here.
    """
    try:
        r = requests.post(f"{HOMUNCULUS_BASE}/water_aversion/arm", timeout=3.0)
        return r.ok and r.json().get("success") is True
    except (requests.RequestException, ValueError):
        return False


def _water_aversion_disarm() -> None:
    """POST /water_aversion/disarm. Best-effort.

    Like evasion: clears state but does NOT cancel an in-progress flee. The
    player keeps walking to dry land until the next Baritone task overrides.
    """
    try:
        requests.post(f"{HOMUNCULUS_BASE}/water_aversion/disarm", timeout=3.0)
    except requests.RequestException:
        pass


def _water_aversion_status() -> dict | None:
    """GET /water_aversion/status. Shape: {success, armed, fired, submerged_pos,
    dry_land_pos, flee_state, flee_failure_reason?}.
    """
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/water_aversion/status", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _format_water_aversion_preamble(status: dict, tool_name: str) -> str:
    """Render a WATER AVERSION FIRED preamble for the tool result.

    Names the submerged location and the picked dry-land target, makes clear
    the reflex flee is still running, and frames the next tool call as the
    override point. Mirrors _format_evaded_preamble.
    """
    sp = status.get("submerged_pos")
    if isinstance(sp, list) and len(sp) == 3:
        sx, sy, sz = (int(v) if isinstance(v, (int, float)) else "?" for v in sp)
        sub = f"({sx},{sy},{sz})"
    else:
        sub = "underwater"
    dp = status.get("dry_land_pos")
    if isinstance(dp, list) and len(dp) == 3:
        dx, dy, dz = (int(v) if isinstance(v, (int, float)) else "?" for v in dp)
        dst = f"({dx},{dy},{dz})"
    else:
        dst = "nearest dry land"
    flee = status.get("flee_state", "in_progress")
    if flee == "arrived":
        progress = f"reflexive flee complete — player is back on dry land at {dst}"
    elif flee in ("timeout", "failed"):
        reason = status.get("flee_failure_reason", "unknown")
        progress = (
            f"reflexive flee {flee} (reason: {reason}); player may still be in water — "
            f"verify pos and consider travel(<dir>, <short_dist>) toward visible shore"
        )
    else:
        progress = f"reflexive flee IN PROGRESS toward {dst}; player still walking out"
    return (
        f"WATER AVERSION: was submerged at {sub} during {tool_name}; {progress}. "
        f"Baritone breaks down in water; your next tool call will override the flee."
    )


def _server_cmd(cmd: str, *, timeout: float = 5.0) -> dict:
    """POST a single command to the MC server console. Returns raw JSON."""
    try:
        resp = requests.post(
            f"{_SERVER_CMD_BASE}/cmd",
            json={"cmd": cmd},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[setup] cmd {cmd!r} failed: {e}", flush=True)
        return {"ok": False, "error": str(e)}


def _apply_setup(
    *,
    start_phase: str,
    random_spawn_range: int,
    starting_loadout: str = "none",
    difficulty: str = "easy",
) -> dict | None:
    """Pre-rollout setup: random TP, time reset, heal, clean inventory,
    optional starting loadout.

    Mirrors e2e/stress_test_shelter.py's per-iter setup so agent rollouts have
    the same baseline as substrate stress tests. Skipped entirely when
    start_phase='none' AND random_spawn_range=0 AND starting_loadout='none'.

    Returns the Wurst pre-flight report (or None if skipped) so the caller
    can log it into the JSONL header — the rollout's behavior is meaningfully
    different depending on which Wurst hacks were actually on.
    """
    if (
        start_phase == "none"
        and random_spawn_range == 0
        and starting_loadout == "none"
    ):
        return None

    print(
        f"[setup] start_phase={start_phase} random_spawn_range={random_spawn_range}"
        f" starting_loadout={starting_loadout}",
        flush=True,
    )

    # Peaceful wipes any lingering hostiles so the agent starts clean.
    set_difficulty("peaceful", server_cmd_base=_SERVER_CMD_BASE)
    _server_cmd(f"effect clear {_PLAYER_NAME}")

    if random_spawn_range > 0:
        spawn_result = random_spawn(
            range_blocks=random_spawn_range,
            homunculus_base=HOMUNCULUS_BASE,
            server_cmd_base=_SERVER_CMD_BASE,
            player_name=_PLAYER_NAME,
            verbose=True,
            log=lambda msg: print(msg, flush=True),
        )
        if not spawn_result.get("ok"):
            print("[setup] spawn-retry exhausted; rollout will proceed at last position",
                  flush=True)

    if start_phase != "none":
        ticks = resolve_phase_ticks(start_phase)
        print(f"[setup] time set {ticks} (phase={start_phase})", flush=True)
        set_time(ticks, server_cmd_base=_SERVER_CMD_BASE)

    # Restore the requested rollout difficulty. Default `easy` keeps mob
    # spawning + damage live for survival rollouts; `peaceful` is the
    # capability-isolation override for sanity tests (sleep_in_bed,
    # craft chains, etc. — substrate signal without the mob confound).
    set_difficulty(difficulty, server_cmd_base=_SERVER_CMD_BASE)

    # Wurst pre-flight: KillAura/AutoEat/AutoTool/AntiKnockback/AntiSpam must
    # be ON for survival rollouts to behave as designed. Before this bridge
    # landed, these depended on the player having toggled them in the UI; a
    # missed click silently invalidated rollout outcomes.
    wurst_report = ensure_wurst_hacks_on()
    # Force unwanted persisted hacks OFF (Wurst saves hack state per profile, so a
    # stale toggle survives relaunches). Sneak left on cripples movement/pathing.
    wurst_report["forbidden_off"] = ensure_wurst_hacks_off()

    # AutoEat offhand-only eating + food policy. AutoEat is pinned to Hands mode
    # (eats only from the offhand/held slot), and homunculus's offhand-food curator
    # stages policy-approved food there. CRAFT_FOOD_POLICY=cooked_only keeps raw meat
    # out of the offhand so it can't be auto-eaten before the agent cooks it; default
    # `any` preserves daily-driver behavior (raw meat still feeds the agent).
    from craft.wurst import (
        set_autoeat_offhand_mode, set_food_policy, set_killaura_no_pvp, set_wurst_hud,
    )
    wurst_report["autoeat_offhand"] = set_autoeat_offhand_mode()
    food_policy = os.environ.get("CRAFT_FOOD_POLICY", "any")
    wurst_report["food_policy"] = set_food_policy(food_policy)
    # Fleet hygiene: stop KillAura from PvP-ing other agents (this build defaults
    # the player filter OFF). Client-side, no server.properties change.
    wurst_report["killaura_no_pvp"] = set_killaura_no_pvp()
    # Wurst HUD (logo/hacklist/TabGui) is debug-only clutter on recorded
    # rollouts — hide it. CRAFT_WURST_HUD=1 re-enables it for a debug session.
    wurst_hud_on = os.environ.get("CRAFT_WURST_HUD", "0").strip().lower() in ("1", "true", "on", "yes")
    wurst_report["wurst_hud"] = set_wurst_hud(wurst_hud_on)

    # AutoDrop policy seeding. With AutoDrop now in REQUIRED_HACKS the module
    # itself is on; this step writes the whitelist-complement drop list into
    # its `Items` setting so the policy is an *inclusion* list of "what to
    # keep" rather than Wurst's tiny exclusion default.
    tier = _autodrop_tier_setting()
    if tier != "off":
        autodrop_report = _seed_autodrop_from_tier(tier)
        wurst_report["autodrop"] = autodrop_report
    else:
        wurst_report["autodrop"] = {"ok": None, "tier": "off", "drop_count": 0}

    # Starting loadout (loaded rollouts): materialize a deterministic
    # high-tier inventory state via MC commands. Applied LAST so the
    # AutoDrop tier-keep policy is already in place — iron+gold+diamond
    # items are in ALWAYS_KEEP and won't be auto-dropped.
    if starting_loadout != "none":
        from craft.loadouts import apply_loadout
        print(f"[setup] applying loadout {starting_loadout!r}", flush=True)
        loadout_report = apply_loadout(
            starting_loadout,
            player_name=_PLAYER_NAME,
            server_cmd_base=_SERVER_CMD_BASE,
        )
        wurst_report["loadout"] = {
            "name": loadout_report["name"],
            "ok": loadout_report["ok"],
            "steps": len(loadout_report["steps"]),
        }
    return wurst_report


def run(
    max_turns: int = 8,
    goal: str = "diamond",
    *,
    start_phase: str = "none",
    random_spawn_range: int = 0,
    starting_loadout: str = "none",
    difficulty: str = "easy",
    jsonl_path: str | None = None,
    model: str = DEFAULT_MODEL,
) -> None:
    # Resolve the JSONL artifact path up front so the recorder — started before
    # spawn to capture the full rollout (spectator-drop → terminal) — shares the
    # transcript's stem (agentN-….jsonl ↔ agentN-….mp4). jsonl_path='' disables
    # the transcript; the recorder then falls back to a default video path.
    if jsonl_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        jsonl_path = f"results/rollout-{goal}-{ts}-{_PLAYER_NAME}.jsonl"
    # Best-effort screen recording (CRAFT_RECORD_VIDEO / --record-video). No-op
    # when disabled or ffmpeg/display unavailable; never raises into the rollout.
    recorder = start_rollout_recording(jsonl_path)
    video_path = recorder.path if recorder is not None else None

    wurst_report = _apply_setup(
        start_phase=start_phase,
        random_spawn_range=random_spawn_range,
        starting_loadout=starting_loadout,
        difficulty=difficulty,
    )
    prompt = GOAL_PROMPTS.get(goal)
    if prompt is None:
        raise ValueError(f"unknown goal {goal!r}; valid: {sorted(GOAL_PROMPTS)}")

    # Open JSONL sink for post-hoc summarizer. The path was resolved at the top
    # of run() (shared with the recorder); pass jsonl_path='' to disable.
    jsonl_fh = None
    if jsonl_path:
        from pathlib import Path
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = open(jsonl_path, "w", buffering=1)  # line-buffered
        print(f"[jsonl] writing per-turn records to {jsonl_path}", flush=True)
        # Header record: invocation context for post-hoc analysis. The
        # wurst_preflight summary lets the summarizer flag rollouts that ran
        # without KillAura/AutoEat/AutoTool — those outcomes shouldn't be
        # compared against rollouts where the substrate was healthy.
        header: dict = {
            "_type": "header",
            "goal": goal, "max_turns": max_turns,
            "start_phase": start_phase, "random_spawn_range": random_spawn_range,
            "starting_loadout": starting_loadout,
            "model": model,
            "player": _PLAYER_NAME,
            "video": video_path,
            "video_started_at": recorder.started_at if recorder is not None else None,
            "started_at": time.time(),
            "equipment_readout": _equipment_readout_enabled(),
            "armor_nudge_gating": _armor_nudge_gating_enabled(),
            "milestones": [
                m.name for m in resolve_milestones(
                    os.environ.get("CRAFT_MILESTONES")
                )
            ],
            "nudges": [
                n.name for n in resolve_nudges(os.environ.get("CRAFT_NUDGES"))
            ],
        }
        if wurst_report is not None:
            header["wurst_preflight"] = {
                "ok": wurst_report.get("ok"),
                "wurst_loaded": wurst_report.get("wurst_loaded"),
                "enabled": [r["name"] for r in wurst_report.get("results", []) if r.get("ok")],
                "failed": [r["name"] for r in wurst_report.get("results", []) if not r.get("ok")],
            }
            ad = wurst_report.get("autodrop") or {}
            header["autodrop"] = {
                "tier": ad.get("tier"),
                "ok": ad.get("ok"),
                "drop_count": ad.get("drop_count"),
            }
        # Spawn-time snapshot for rolling-rollout analysis. Captures the
        # precise incidental MC time-of-day each rollout lands on (not just
        # dawn/noon/dusk) so spawn-time distribution can be reconstructed
        # from headers alone, without parsing per-turn stats.
        spawn_stats = _stats_raw()
        if spawn_stats:
            header["spawn"] = {
                k: spawn_stats.get(k) for k in (
                    "day_ticks", "day_count", "biome",
                    "x", "y", "z", "dimension",
                ) if k in spawn_stats
            }
        jsonl_fh.write(json.dumps(header) + "\n")

    # Opening is pure instruction text — no spawn-time stats/inventory baked in.
    # The opening used to embed initial Stats/Inventory inline, which then went
    # stale forever (the user message at index 1 still claimed "Current
    # inventory: (empty)" 50 turns into a diamond rollout). State is now an
    # *ephemeral* per-turn injection at prompt-construction time — never
    # persisted to history — so the model sees exactly one STATE block per
    # prompt, always at the tail, always fresh.
    if goal == "bare":
        # Bare path: defer all judgement to the model's MC pretraining +
        # constraint assumptions inferable from the tool list. No STATE
        # explainer, no substrate facts, no movement notes. Validated
        # (qwen3-4B): identical decisions vs. minimal with less noise.
        opening = "Begin."
    else:
        opening = (
            "Begin. The STATE: block at the end of each prompt holds the current "
            "stats, inventory, and any active smelts — read it before deciding "
            "your next tool call. Earlier turns in the transcript show only what "
            "you did and what each tool returned; the STATE there is intentionally "
            "absent (it would be stale)."
        )

    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": opening},
    ]
    # Reactive nudge set — STATE-block hints toward under-used verbs, gated on
    # current state and recomputed every turn (ephemeral, like the equipment
    # nudge). Chain is selected from CRAFT_NUDGES (comma-separated names; unset
    # → default set; "" → control arm, no nudges). Resolved here so the initial
    # pre-loop STATE can carry a nudge too (cook_kitchen/hunt_meadow spawn at
    # hunger=2, so the food nudge is relevant from turn 1).
    nudge_chain = resolve_nudges(os.environ.get("CRAFT_NUDGES"))

    # Carried state — refreshed at the end of each turn from the post-dispatch
    # fetches, injected at the head of the next turn's prompt. Computed once
    # before the loop so turn 1's prompt isn't blind.
    pending_state = _build_state_chunk(
        _fetch_stats(), _fetch_inventory(), _fetch_smelts(),
        _render_nudges(nudge_chain, _stats_raw(), _inventory_raw()),
    )
    # Cap conversation length to the last N turns so prefill stays bounded.
    # Each turn appends exactly 2 messages (assistant tool_call + tool result)
    # to the persistent history — the STATE injection happens at prompt-build
    # time and is NOT stored. We keep messages[:2] (system + opening) and the
    # last 2*WINDOW_TURNS pairs. Set generously enough that within-rollout
    # dependencies (e.g., what the agent crafted 5 turns ago) survive, but
    # aggressively enough that 50-turn rollouts don't blow up prefill cost:
    # gemma median plan_s climbed from ~10s on early turns to 60-80s on late
    # turns in 2026-05-14 rollouts.
    WINDOW_TURNS = 8
    print(f"=== goal={goal}, max_turns={max_turns} permadeath (window={WINDOW_TURNS}) ===")

    print("starting in 3s...")
    time.sleep(3)

    # Track which deaths we've already surfaced. Initial value = now: any
    # pre-run deaths in homunculus's ring buffer are ignored.
    last_death_ts = int(time.time() * 1000)

    # Rollout-outcome trackers for the video keep-on-failure policy. `turn` is
    # also seeded so a max_turns=0 (empty loop) doesn't NameError below.
    rollout_had_death = False
    turn = 0

    # Milestone framework — staged goal progression. Predicates evaluated per
    # turn against stats + inventory. When one fires, its announcement is
    # appended to the opening (messages[1]) so it persists past the WINDOW
    # trim and lands in every subsequent prefill. Chain is selected from
    # CRAFT_MILESTONES (comma-separated names; unset → default chain).
    milestone_chain = resolve_milestones(os.environ.get("CRAFT_MILESTONES"))
    milestones = Milestones(milestones=milestone_chain)

    # Accumulators for the LLM-idle-time post-mortem. Each rollout's plan_s
    # total is the answer to "how long was the harness standing around
    # waiting for the model?" — useful for A/B comparing models or context
    # strategies (gemma vs Haiku, trim window sizes, …).
    plan_s_total = 0.0
    plan_s_count = 0
    rollout_start_t = time.time()

    # gemma occasionally returns empty (no content, no tool_call) — typically
    # a reasoning-runaway truncated by the stop-token list (observed
    # probe-validate-r4-dusk T1: 85.7s plan, both content and tool_calls empty).
    # Retry once before bailing the rollout; if it's a transient sampling
    # artifact we keep going, if it's persistent we stop after 2 attempts.
    EMPTY_RETRIES = 1

    for turn in range(1, max_turns + 1):
        turn_start = time.perf_counter()
        turn_wall_start = time.time()  # epoch anchor for post-hoc video overlay
        milestone_event = None
        print(f"\n=== turn {turn}/{max_turns}: planning ===")
        plan_start = time.perf_counter()
        # Build the prompt by appending the *current* STATE to history. This
        # is the only place STATE enters the conversation — it never lands
        # in `messages` itself, so the next turn won't see this turn's stale
        # STATE alongside its own fresh one. prompt_messages is a fresh list,
        # safe to use as both the LLM input and the JSONL snapshot.
        prompt_messages = messages + [
            {"role": "user", "content": pending_state},
        ]
        prompt_snapshot = prompt_messages
        tool_calls, content, reasoning, raw_message = chat_with_tools(prompt_messages, TOOLS, model=model)
        plan_dt = time.perf_counter() - plan_start
        plan_s_total += plan_dt
        plan_s_count += 1
        if content:
            print(f"[content] {content!r}")
        # Human-driver "quit" is an explicit choice to end the rollout; don't
        # retry-prompt them. For LLMs an empty response is usually a sampling
        # glitch and warrants one retry.
        if model != "human":
            for retry_i in range(EMPTY_RETRIES):
                if tool_calls:
                    break
                retry_start = time.perf_counter()
                print(f"[retry] empty response, retry {retry_i + 1}/{EMPTY_RETRIES}", flush=True)
                tool_calls, content, reasoning, raw_message = chat_with_tools(prompt_messages, TOOLS, model=model)
                retry_dt = time.perf_counter() - retry_start
                plan_dt += retry_dt
                plan_s_total += retry_dt
                if content:
                    print(f"[content] {content!r}")

        # Faithful per-turn LLM record (prompt + raw response) for replay /
        # SFT data extraction. Written before the no-tool-call break so even
        # failed plans are captured. `raw_message` is the full provider message
        # dump (Ollama via OpenAI SDK) so anything beyond the three parsed
        # fields — `refusal`, future schema extensions — is preserved verbatim.
        if jsonl_fh is not None:
            llm_rec = {
                "_type": "llm",
                "turn": turn,
                "model": model,
                "prompt_messages": prompt_snapshot,
                "response": {
                    "content": content or "",
                    "reasoning": reasoning or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in tool_calls
                    ],
                },
            }
            if raw_message is not None:
                llm_rec["response"]["raw_message"] = raw_message
            jsonl_fh.write(json.dumps(llm_rec) + "\n")

        if not tool_calls:
            print(f"=== no tool call returned; stopping (plan={plan_dt:.1f}s) ===")
            break

        if len(tool_calls) > 1:
            print(f"!! WARNING: planner emitted {len(tool_calls)} tool calls; executing only the first")
            for extra in tool_calls[1:]:
                print(f"   discarded: {extra.function.name}({extra.function.arguments})")

        tc = tool_calls[0]
        name = tc.function.name
        args = tc.function.arguments
        print(f"=== turn {turn}: executing {name}({args}) ===")

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            }
        )

        # Pre-dispatch death poll. Long plan times (gemma can be >100s) leave
        # the agent vulnerable mid-plan — observed r9 T17: 133.8s plan during
        # which a zombie killed the swamp agent; build_shelter then executed
        # at the respawn coords with an empty inventory. Stop the loop now
        # instead of running a wasted dispatch.
        pre_deaths = _fetch_new_deaths(last_death_ts)
        if pre_deaths:
            d = pre_deaths[-1]
            last_death_ts = int(d.get("timestamp", last_death_ts))
            preamble = _format_death(d)
            print(f"[death] {preamble} (during planning — skipping dispatch)")
            turn_dt = time.perf_counter() - turn_start
            print(f"[timing] turn {turn} aborted post-plan: total={turn_dt:.1f}s")
            # Synthetic JSONL record so post-hoc analyzers see the terminating
            # death. Skipping it under-counts permadeaths (issue #1). Fields
            # mirror the normal turn record but with no dispatch outcome and
            # exec/ctx zeroed; stats/inventory omitted (player has already
            # respawned by now, so live reads would be misleading).
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps({
                    "_type": "turn",
                    "turn": turn,
                    "t": turn_wall_start,
                    "tool": name,
                    "args": args,
                    "outcome": "aborted_pre_dispatch_due_to_death",
                    "plan_s": round(plan_dt, 3),
                    "exec_s": 0.0,
                    "ctx_s": 0.0,
                    "total_s": round(turn_dt, 3),
                    "health": 0,
                    "died": True,
                    "death": d,
                }) + "\n")
                jsonl_fh.flush()
            rollout_had_death = True
            print(f"\n=== PERMADEATH: trajectory terminated at turn {turn} ===")
            break

        exec_start = time.perf_counter()
        # Reflexive evasion: re-arm with the current player position. The
        # homunculus-side watcher autonomously cancels Baritone + flees back
        # here on any hostile-mob hit — handlers don't participate. We
        # disarm first so any in-progress flee from the prior turn's tail
        # stops being tracked (the flee itself can keep running in MC; it'll
        # be cancelled implicitly by whatever this turn's tool dispatches).
        _evasion_disarm()
        _water_aversion_disarm()
        evasion_armed = False
        try:
            pos_resp = requests.get(f"{HOMUNCULUS_BASE}/position", timeout=3.0)
            pos_for_arm = pos_resp.json() if pos_resp.ok else {}
        except (requests.RequestException, ValueError):
            pos_for_arm = {}
        ax = pos_for_arm.get("x")
        ay = pos_for_arm.get("y")
        az = pos_for_arm.get("z")
        if isinstance(ax, (int, float)) and isinstance(ay, (int, float)) and isinstance(az, (int, float)):
            evasion_armed = _evasion_arm(float(ax), float(ay), float(az))
            if evasion_armed:
                print(f"[evasion] armed at ({int(ax)},{int(ay)},{int(az)})", flush=True)
        water_aversion_armed = _water_aversion_arm()
        if water_aversion_armed:
            print("[water_aversion] armed", flush=True)
        outcome = dispatch(name, args)
        exec_dt = time.perf_counter() - exec_start

        # Post-dispatch evasion check. Don't block on flee completion — the
        # next chat_with_tools call runs concurrently with the still-walking
        # player; by the time the LLM responds and dispatches, the flee is
        # either done or gets overridden by the new Baritone task.
        evasion_status: dict | None = None
        evaded_preamble: str | None = None
        if evasion_armed:
            evasion_status = _evasion_status()
            if evasion_status and evasion_status.get("fired"):
                evaded_preamble = _format_evaded_preamble(evasion_status, name)
                print(f"[evasion] FIRED: {evaded_preamble}", flush=True)
        water_aversion_status: dict | None = None
        water_aversion_preamble: str | None = None
        if water_aversion_armed:
            water_aversion_status = _water_aversion_status()
            if water_aversion_status and water_aversion_status.get("fired"):
                water_aversion_preamble = _format_water_aversion_preamble(water_aversion_status, name)
                print(f"[water_aversion] FIRED: {water_aversion_preamble}", flush=True)
        print(f"=== turn {turn} outcome: {outcome} ===")
        print(f"[timing] turn {turn}: plan={plan_dt:.1f}s exec={exec_dt:.1f}s ({name})")

        # Arm the shelter watcher on a non-failing build_shelter. PARTIAL
        # counts — most partials still hold; let the watcher decide via
        # actual breach detection rather than the build-result heuristic.
        if name == "build_shelter" and not (
            outcome.startswith("FAILED") or outcome.startswith("ABORTED")
        ):
            _arm_shelter_watch(outcome)

        # Auto-organize hotbar + armor so the *next* turn sees a tidy
        # inventory and Wurst's autoTool has the right tools in hotbar.
        # See homunculus /equip spec for the layout. Failure is non-fatal.
        try:
            equip_resp = requests.post(f"{HOMUNCULUS_BASE}/equip", timeout=5.0)
            equip_data = equip_resp.json() if equip_resp.ok else {}
            if equip_data.get("success"):
                changes = equip_data.get("changes", [])
                if changes:
                    print(f"[equip] {len(changes)} change(s): {equip_data.get('message', '')}")
        except requests.RequestException as e:
            print(f"[equip] failed (non-fatal): {e}")

        stats_raw = _stats_raw()
        inv_raw = _inventory_raw()
        stats_str = _fetch_stats()
        inv_str = _format_inventory(inv_raw)
        smelts_str = _fetch_smelts()
        shelter_str = _poll_shelter_watch()
        if stats_str:
            print(f"[stats] {stats_str}")
        if inv_str:
            print(f"[inventory]\n{inv_str}")
        else:
            print("[inventory] failed to fetch (homunculus may be offline)")
        if smelts_str:
            print(f"[smelts]\n{smelts_str}")
        # Tool message carries only what's *tied to this turn's event*: the
        # dispatch outcome plus reflex/death preambles plus shelter alerts.
        # World snapshot (stats/inv/smelts) lives in a separate role:user
        # message appended below so SFT has a clean structural split between
        # "what the action returned" and "what the world looks like now".
        chunks = [outcome]
        # Shelter alerts are urgent — prepend so the LLM reads them first.
        if shelter_str:
            print(f"[shelter_watch] {shelter_str}")
            chunks.insert(0, shelter_str)
        # Evasion preamble outranks shelter: a hostile-mob hit just landed,
        # the reflex flee is in flight, and the LLM's next call needs that
        # context to pick its pivot (build_shelter / travel / fight).
        # Water aversion preamble sits between: less urgent than a hostile-mob
        # hit, more load-bearing than a shelter watch hint. Insert in reverse
        # priority so evasion ends up at chunks[0].
        if water_aversion_preamble:
            chunks.insert(0, water_aversion_preamble)
        if evaded_preamble:
            chunks.insert(0, evaded_preamble)
        full_outcome = "\n\n".join(chunks)
        nudges_str = _render_nudges(nudge_chain, stats_raw, inv_raw)
        if nudges_str:
            print(f"[nudge]\n{nudges_str}")
        state_chunk = _build_state_chunk(stats_str, inv_str, smelts_str, nudges_str)

        # Surface any death that landed during this turn. We use the most
        # recent record (deaths are rare; multi-death within one turn is a
        # cascade we still summarize as "the last one"). The preamble goes
        # before the outcome so the LLM sees it first.
        new_deaths = _fetch_new_deaths(last_death_ts)
        died_this_turn = False
        if new_deaths:
            d = new_deaths[-1]
            last_death_ts = int(d.get("timestamp", last_death_ts))
            preamble = _format_death(d)
            print(f"[death] {preamble}")
            full_outcome = f"{preamble}\n\n{full_outcome}"
            died_this_turn = True

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": full_outcome,
            }
        )
        # Refresh the carried STATE for the next turn's prompt. Not appended
        # to messages — STATE is ephemeral by design (one per prompt, always
        # at the tail, always fresh) so the model never sees a stale snapshot
        # mixed in with the action history.
        pending_state = state_chunk

        # Milestone check — predicates over stats + inventory. Fires at most
        # once per milestone per rollout. The announcement is appended to the
        # opening so it survives the window trim and becomes part of every
        # subsequent prefill (no system-prompt mutation → kv-cache friendly).
        # Pass the compact {item_id: count} shape that the backtest validated
        # against, not the raw homunculus {main: [...], offhand: {...}} shape.
        milestone_event = milestones.check(
            stats_raw, _inventory_compact(inv_raw), turn
        )
        if milestone_event:
            print(f"[milestone] {milestone_event.name} fired at turn {turn}")
            messages[1] = {
                **messages[1],
                "content": messages[1]["content"] + "\n\n" + milestone_event.message,
            }

        # Trim older turns to keep prefill cost bounded. Each turn appends
        # exactly 2 messages to the persistent history (assistant tool_call +
        # tool result); STATE injection happens at prompt-build time and isn't
        # stored. We keep messages[:2] (system + opening) and the most-recent
        # 2*WINDOW_TURNS entries. Pair-aligned slicing is critical —
        # orphaning an assistant tool_call without its matching tool_result
        # breaks the API contract.
        max_msgs = 2 + 2 * WINDOW_TURNS
        if len(messages) > max_msgs:
            messages = messages[:2] + messages[-2 * WINDOW_TURNS:]

        turn_total_dt = time.perf_counter() - turn_start
        ctx_dt = turn_total_dt - plan_dt - exec_dt
        print(f"[timing] turn {turn} total={turn_total_dt:.1f}s (plan={plan_dt:.1f}s + exec={exec_dt:.1f}s + ctx={ctx_dt:.1f}s)")

        # Structured per-turn record for post-hoc analysis.
        if jsonl_fh is not None:
            rec = {
                "_type": "turn",
                "turn": turn,
                "t": turn_wall_start,
                "tool": name,
                "args": args,
                "outcome": outcome,
                "plan_s": round(plan_dt, 3),
                "exec_s": round(exec_dt, 3),
                "ctx_s": round(ctx_dt, 3),
                "total_s": round(turn_total_dt, 3),
                "nudge": nudges_str,
                "shelter_armed": _shelter_watch is not None,
                "shelter_breach": (_shelter_watch or {}).get("breach", False),
                "shelter_str": shelter_str,
                "died": died_this_turn,
            }
            if stats_raw:
                for k in ("health", "food", "saturation", "air", "biome",
                         "day_ticks", "day_count", "in_water", "in_lava",
                         "on_fire", "dimension"):
                    if k in stats_raw:
                        rec[k] = stats_raw[k]
            rec["inventory"] = _inventory_compact(inv_raw)
            if evasion_status is not None:
                rec["evasion"] = {
                    "fired": bool(evasion_status.get("fired")),
                    "attackers": evasion_status.get("attackers") or [],
                    "anchor": evasion_status.get("anchor"),
                    "flee_state": evasion_status.get("flee_state"),
                }
            if water_aversion_status is not None:
                rec["water_aversion"] = {
                    "fired": bool(water_aversion_status.get("fired")),
                    "submerged_pos": water_aversion_status.get("submerged_pos"),
                    "dry_land_pos": water_aversion_status.get("dry_land_pos"),
                    "flee_state": water_aversion_status.get("flee_state"),
                }
            if milestone_event:
                rec["milestone_fired"] = milestone_event.name
            if died_this_turn and new_deaths:
                rec["death"] = new_deaths[-1]
            jsonl_fh.write(json.dumps(rec) + "\n")

        if died_this_turn:
            rollout_had_death = True
            print(f"\n=== PERMADEATH: trajectory terminated at turn {turn} ===")
            break

    # Disarm evasion + water_aversion at rollout end so a stale armed state
    # doesn't trigger a cancel-and-flee on the next /baritone/* call by
    # some other client.
    _evasion_disarm()
    _water_aversion_disarm()

    rollout_wall_s = time.time() - rollout_start_t
    mean_plan = (plan_s_total / plan_s_count) if plan_s_count else 0.0
    idle_pct = (plan_s_total / rollout_wall_s * 100) if rollout_wall_s > 0 else 0.0
    print("\n=== rollout complete ===")
    print(
        f"[llm-idle] model={model} turns={plan_s_count} "
        f"plan_total={plan_s_total:.1f}s ({plan_s_total/60:.1f}min) "
        f"mean={mean_plan:.1f}s/turn "
        f"wall={rollout_wall_s:.0f}s — agent was idle on LLM for {idle_pct:.0f}% of the rollout"
    )
    # Finalize the screen recording (idempotent; atexit also covers the
    # exception/early-return paths, and the fragmented mp4 survives a hard kill).
    # Then apply the keep-on-failure retention policy: a rollout is a "failure"
    # (worth a tape) if the agent died or the loop ended before max_turns
    # (death / no-tool-call / empty-plan bail). Clean full-length survivals are
    # discarded under CRAFT_RECORD_KEEP=failures.
    video_kept = None
    if recorder is not None:
        recorder.stop()
        rollout_failed = rollout_had_death or (turn < max_turns)
        video_kept = recorder.should_keep(failed=rollout_failed)
        if not video_kept:
            recorder.discard()
    if jsonl_fh is not None:
        jsonl_fh.write(json.dumps({
            "_type": "end",
            "ended_at": time.time(),
            "video_kept": video_kept,
            "rollout_had_death": rollout_had_death,
            "plan_s_total": round(plan_s_total, 3),
            "plan_s_mean": round(mean_plan, 3),
            "wall_s": round(rollout_wall_s, 3),
            "llm_idle_pct": round(idle_pct, 1),
        }) + "\n")
        jsonl_fh.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run a craft.agent rollout (permadeath; first death ends the trajectory).")
    ap.add_argument("turns", nargs="?", type=int, default=8,
                    help="max turns (default 8)")
    ap.add_argument("goal", nargs="?", default="diamond",
                    help=f"goal prompt: {sorted(GOAL_PROMPTS)} (default diamond)")
    ap.add_argument("--start-phase",
                    choices=["dawn", "noon", "dusk", "midnight", "random", "none"],
                    default="none",
                    help="reset MC time to this phase before launching the rollout")
    ap.add_argument("--random-spawn-range", type=int, default=20000,
                    help="TP player to a random xz offset (±N from current pos, drop y=100); "
                         "default 20000 for broad biome sampling, 0 to disable")
    # Loaded-rollouts capability: boot the agent into a deterministic
    # high-tier state (e.g. full iron armor equipped) via MC commands so
    # downstream features (M2 firing, diamond descent) are testable in a
    # 2-turn smoke rather than a 30-min organic survival run.
    from craft.loadouts import LOADOUTS as _LOADOUTS
    ap.add_argument("--starting-loadout",
                    choices=["none"] + sorted(_LOADOUTS),
                    default="none",
                    help="apply a named pre-set inventory loadout after spawn "
                         f"(presets: {sorted(_LOADOUTS)})")
    ap.add_argument("--difficulty",
                    choices=["peaceful", "easy", "normal", "hard"],
                    default="easy",
                    help="rollout difficulty (default easy). `peaceful` "
                         "isolates capability tests from mob pressure.")
    ap.add_argument("--jsonl-out", default=None,
                    help="write per-turn JSONL to PATH (default: results/rollout-<goal>-<ts>.jsonl; '' to disable)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LLM backend; gemma-* → Ollama, claude-* → Anthropic. Default: {DEFAULT_MODEL}")
    ap.add_argument("--record-video", action="store_true",
                    help="record the agent's screen (its Xvfb) to <jsonl-stem>.mp4 for the "
                         "full rollout, spawn→terminal (sets CRAFT_RECORD_VIDEO; needs ffmpeg)")
    args = ap.parse_args()

    if args.record_video:
        os.environ["CRAFT_RECORD_VIDEO"] = "1"

    run(
        max_turns=args.turns,
        goal=args.goal,
        start_phase=args.start_phase,
        random_spawn_range=args.random_spawn_range,
        starting_loadout=args.starting_loadout,
        difficulty=args.difficulty,
        jsonl_path=args.jsonl_out,
        model=args.model,
    )
