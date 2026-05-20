"""Milestone framework — staged goal progression.

A milestone is a `(predicate, message)` pair. The predicate is evaluated each
turn against the agent's current stats + inventory. When it fires, the
announcement message is appended to the conversation opening so it persists
across the WINDOW_TURNS trim and becomes part of every subsequent prefill.

Empirical M1 ("survived first night + has wooden tools") was chosen from
N=257 rolling-rollout data, 2026-05-18: ~85% of survivors (T>=20) trigger
it, ~5-10% of early-deaths. See [[project-rolling-rollouts-20260518]].

The framework is intentionally manual at v0 — predicates are author-defined.
A future "superagent" variant could have an LLM generate milestones from
recent rollout history, at which point this module becomes its execution
backend.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


def _has(inv: dict | None, item_suffix: str) -> bool:
    """True if any inventory key ends with the given item id."""
    if not inv:
        return False
    for k, v in inv.items():
        if isinstance(v, int) and v >= 1 and k.endswith(item_suffix):
            return True
    return False


@dataclass
class MilestoneEvent:
    """Returned by Milestones.check() when a new milestone fires."""
    name: str
    message: str
    turn: int


@dataclass
class Milestone:
    name: str
    # predicate(state: dict) -> bool. state has keys:
    #   day_ticks, day_count, inv (dict), ticks_alive (int)
    predicate: Callable[[dict], bool]
    message: str


# Half an MC day = 12000 ticks (~10 real-time minutes). A dawn-spawned agent
# hits this at first dusk; a noon-spawned agent at midnight; a night-spawned
# agent at next dawn. Threshold matches the empirical "survived first dusk"
# signal from the N=257 backtest (45% A_dead_early vs 98%+ B/C/D/E).
_M1_TICKS_ALIVE = 12000


def _m1_predicate(state: dict) -> bool:
    """Has wooden_pickaxe AND has been alive for >=12000 MC ticks (~10 min).

    `ticks_alive` is `(day_count*24000 + day_ticks) - spawn_total_ticks` —
    cumulative game time since rollout start. Spawn-phase independent: works
    the same whether the agent spawned at dawn, noon, dusk, or night.
    """
    inv = state.get("inv") or {}
    if not _has(inv, ":wooden_pickaxe"):
        return False
    return state.get("ticks_alive", 0) >= _M1_TICKS_ALIVE


M1 = Milestone(
    name="M1_iron_goal",
    predicate=_m1_predicate,
    message=(
        "MILESTONE REACHED. You survived your first night and have basic tools. "
        "New goal: craft a full iron tool + armor set (iron_pickaxe, iron_sword, "
        "iron_helmet, iron_chestplate, iron_leggings, iron_boots). "
        "Death is failure — but so is staying at wooden tier. Take calculated risks "
        "to mine stone, coal, then iron. You may lose this run; that's acceptable."
    ),
)


def _m2_predicate(state: dict) -> bool:
    """Has the full iron armor set: helmet + chestplate + leggings + boots.

    Empirically (sprint 2026-05-20 armor-nudge-gating campaign), surviving
    qwen agents reliably reach iron tier but stall there, hoarding redundant
    iron tools/armor instead of risking the descent. Gating M2 on the
    *complete* armor set (rather than iron tools as in an earlier draft)
    serves the trajectory we want: a deep-delve to diamond is dangerous,
    full iron armor is the protective investment that makes it survivable,
    so the milestone naturally fires *exactly when* the agent has paid the
    cost that justifies the risk. Distinct in shape from M1 (which uses
    tool + stability) — M2 uses armor completeness, no ticks_alive floor.
    """
    inv = state.get("inv") or {}
    return (
        _has(inv, ":iron_helmet")
        and _has(inv, ":iron_chestplate")
        and _has(inv, ":iron_leggings")
        and _has(inv, ":iron_boots")
    )


M2 = Milestone(
    name="M2_diamond_goal",
    predicate=_m2_predicate,
    message=(
        "MILESTONE REACHED. You have a full iron armor set. New goal: descend "
        "to y<=11 and mine diamond_ore with an iron_pickaxe. Craft "
        "diamond_pickaxe, diamond_sword, and a full diamond armor set "
        "(diamond_helmet, diamond_chestplate, diamond_leggings, diamond_boots). "
        "Bring torches, food, and watch for lava lakes. You may lose this run; "
        "that's acceptable."
    ),
)


# Ordered milestone chain. Each milestone fires at most once per rollout.
# Future milestones (e.g. M3_netherite_goal) append here.
MILESTONES: list[Milestone] = [M1, M2]

# Name -> Milestone registry, used by `resolve_milestones` to interpret
# the CRAFT_MILESTONES env var (and any future declarative chain spec).
# Add new milestones above and they become A/B-selectable automatically.
MILESTONES_BY_NAME: dict[str, Milestone] = {m.name: m for m in MILESTONES}


def resolve_milestones(spec: str | None) -> list[Milestone]:
    """Resolve a milestone chain from an env-spec string.

    spec is the raw value of `CRAFT_MILESTONES`:
    - None (env unset)        -> default chain (all of MILESTONES, in order).
    - "" (set but empty)      -> empty chain (no milestones fire).
    - "M1_iron_goal,M2_..."   -> exactly those milestones, in the order given.
    - Unknown name            -> ValueError. Silent drops would let typos turn
                                a campaign arm into a no-op without warning.

    The user-supplied order is honored — earlier entries get evaluation
    priority when multiple predicates fire on the same turn (mirrors the
    list semantics in `Milestones.check`).
    """
    if spec is None:
        return list(MILESTONES)
    tokens = [tok.strip() for tok in spec.split(",")]
    tokens = [tok for tok in tokens if tok]  # drop empties from "M1,,M2"
    chain: list[Milestone] = []
    for name in tokens:
        m = MILESTONES_BY_NAME.get(name)
        if m is None:
            known = ", ".join(MILESTONES_BY_NAME) or "(none)"
            raise ValueError(
                f"Unknown milestone '{name}' in CRAFT_MILESTONES. "
                f"Known milestones: {known}."
            )
        chain.append(m)
    return chain


class Milestones:
    """Per-rollout milestone state-tracker.

    Usage in agent loop:
        ms = Milestones()
        # ... each turn, after fetching stats + inventory:
        event = ms.check(stats_raw, inv_raw, turn)
        if event:
            # apply: append event.message to messages[1]["content"]
            # log: write event.name + event.turn to JSONL
    """

    def __init__(self, milestones: list[Milestone] = None):
        self._milestones = list(milestones) if milestones is not None else list(MILESTONES)
        self._fired: dict[str, int] = {}  # name -> turn fired
        self._spawn_total_ticks: int | None = None  # day_count*24000 + day_ticks at spawn

    def check(self, stats: dict | None, inv: dict | None, turn: int) -> MilestoneEvent | None:
        """Evaluate the next un-fired milestone. Returns event on fire, else None."""
        if not stats:
            return None
        # Contract: inv must be the flat {item_id: count} shape, NOT the raw
        # homunculus {"main": [...], "offhand": {...}} shape. Pass it through
        # craft.agent._inventory_compact() before calling this. Detect the
        # raw shape and fail loudly to prevent silent regression of the
        # original bug (caller passed raw → predicates never matched).
        if isinstance(inv, dict) and isinstance(inv.get("main"), list):
            raise ValueError(
                "Milestones.check received raw inventory shape "
                "({'main': [...], 'offhand': ...}). Apply _inventory_compact() "
                "first to get the flat {item_id: count} shape predicates expect."
            )
        dc = stats.get("day_count")
        dt = stats.get("day_ticks")
        total = None
        if isinstance(dc, (int, float)) and isinstance(dt, (int, float)):
            total = int(dc) * 24000 + int(dt)
            if self._spawn_total_ticks is None:
                self._spawn_total_ticks = total
        ticks_alive = (
            total - self._spawn_total_ticks
            if total is not None and self._spawn_total_ticks is not None
            else 0
        )

        state = {
            "day_ticks": dt,
            "day_count": dc,
            "ticks_alive": ticks_alive,
            "inv": inv or {},
        }
        # Fire milestones in order; each can fire at most once.
        for m in self._milestones:
            if m.name in self._fired:
                continue
            if m.predicate(state):
                self._fired[m.name] = turn
                return MilestoneEvent(name=m.name, message=m.message, turn=turn)
        return None

    @property
    def fired(self) -> dict[str, int]:
        """Read-only view of fired milestones: {name: turn_fired}."""
        return dict(self._fired)
