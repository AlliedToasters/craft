"""Reactive, state-gated nudges — STATE-block hints toward under-used verbs.

Unlike milestones (one-shot goal changes appended to the opening message that
persist for the rest of the rollout), nudges are reactive and *ephemeral*:
recomputed every turn and injected into the per-turn STATE block, so a nudge
appears exactly while its condition holds and vanishes once it clears. Same
vehicle as the equipment-slot nudge already living in agent.py.

Motivation (2026-05-25): qwen3-4B reaches for new named verbs (hunt_passive,
cook_meat, sleep_in_bed) far less than Haiku — a "named-verb planning ceiling".
Those verbs exist only as tool-schema JSON; the prose goal prompts (bare/
minimal/survive*) never name them, so the model gets no *when-to-use* trigger.
A nudge supplies that trigger at the precise moment state calls for it.

Whether the nudge actually moves uptake is the experiment, so each nudge is
A/B-selectable via CRAFT_NUDGES (mirror of CRAFT_MILESTONES) and the whole
block is an ablatable substrate feature: treatment runs with nudges, control
sets CRAFT_NUDGES="" to suppress them. See [[project-qwen-capability-uptake]].
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


def _has(inv: dict | None, item_suffix: str) -> bool:
    """True if any inventory key ends with the given item id suffix."""
    if not inv:
        return False
    for k, v in inv.items():
        if isinstance(v, int) and v >= 1 and k.endswith(item_suffix):
            return True
    return False


def _count(inv: dict | None, item_suffix: str) -> int:
    """Total count of items whose id ends with the given suffix.

    Used when the nudge needs to gate on "at least N" not "any". `_has`
    answers existence; `_count` answers quantity (e.g. ≥3 planks for a
    bed recipe).
    """
    if not inv:
        return 0
    return sum(
        int(v) for k, v in inv.items()
        if isinstance(v, int) and v >= 1 and k.endswith(item_suffix)
    )


# Food at or below this triggers the food nudge. AutoEat targets ~18-20 and HP
# regen needs food >= 18, so 8 is "acting late but not yet starving" — low
# enough not to nag during normal play, high enough to act before damage.
_FOOD_LOW_THRESHOLD = 8

# Raw-meat item suffixes the cook chain understands (matches cook_meat).
_RAW_MEATS = ("beef", "porkchop", "mutton", "chicken", "rabbit")

# A player can enter a bed from ~tick 12542 (night) until ~23460 (pre-dawn).
# We gate the sleep nudge on this sleepable window rather than the DAY/NIGHT
# label (which flips at dusk=12000) — firing earlier would point the agent at
# sleep_in_bed while it still fail-fasts with 'not_night'.
_BED_SLEEPABLE_TICK = 12542


@dataclass
class Nudge:
    """A reactive hint. render(state) -> hint line, or None when inert.

    state keys: food (int|None), day_ticks (int|None), day_count (int|None),
    inv (flat {item_id: count}).
    """
    name: str
    render: Callable[[dict], "str | None"]


def _food_low(state: dict) -> str | None:
    food = state.get("food")
    if not isinstance(food, (int, float)) or food > _FOOD_LOW_THRESHOLD:
        return None
    inv = state.get("inv") or {}
    if any(_has(inv, ":" + m) for m in _RAW_MEATS):
        return (
            f"Food is low (food={int(food)}) and you are carrying raw meat. "
            "Call cook_meat to cook it — the cooked output is eaten automatically."
        )
    if _has(inv, "_sword"):
        return (
            f"Food is low (food={int(food)}). Call hunt_passive to kill a nearby "
            "animal for meat, then cook_meat to cook the drops into food."
        )
    return (
        f"Food is low (food={int(food)}). Acquire food: hunt_passive (needs a "
        "sword in inventory) hunts animals, or travel to reach a herd."
    )


FOOD_LOW = Nudge(name="food_low", render=_food_low)


def _night_bed(state: dict) -> str | None:
    """Tiered night-bed nudge: surface the closest sleep-skip path the
    agent's current inventory can reach.

    Three precursor tiers, in order of "how few tool calls to sleep":

      T1 (bed in hand)             → sleep_in_bed
      T2 (wool + ≥3 planks)        → craft a colored bed + sleep
      T3 (iron + ≥3 planks)        → shear_sheep + craft bed + sleep

    Why tiered: the prior single-tier version gated on `_has(inv, "_bed")`
    which never fired in the iron→bed scenario (agents start with iron+
    planks, never get a bed). 25 rollouts × 2 models confirmed zero
    bed-tool calls when only T1 fires. Tiering moves the substrate signal
    upstream so qwen + Haiku see the bedding chain at *any* point along
    its precursor path. See [[project-qwen-capability-uptake]].
    """
    dt = state.get("day_ticks")
    if not isinstance(dt, (int, float)):
        return None
    if not (_BED_SLEEPABLE_TICK <= dt < 24000):
        return None
    inv = state.get("inv") or {}
    mins = (24000 - dt) / 1200
    prefix = f"It is night ({mins:.0f}min until dawn)"

    # T1: bed already crafted → just sleep.
    if _has(inv, "_bed"):
        return (
            f"{prefix} and you have a bed. Call sleep_in_bed to skip safely "
            "to dawn — you are invincible while asleep and the night's mob "
            "danger is skipped entirely."
        )

    planks_n = _count(inv, "_planks")

    # T2: wool + planks → one craft away from a bed.
    if _has(inv, "_wool") and planks_n >= 3:
        return (
            f"{prefix} and you carry wool + planks. Craft a colored bed "
            "(e.g. white_bed if your wool is white — recipes are color-strict, "
            "3 wool + 3 planks), then call sleep_in_bed to skip the night."
        )

    # T3: iron + planks → shear_sheep handles the shears craft + wool harvest.
    if _has(inv, ":iron_ingot") and planks_n >= 3:
        return (
            f"{prefix} and you have iron + planks. Call shear_sheep — it "
            "auto-crafts shears from 2 iron and harvests wool from nearby "
            "sheep. Then craft a colored bed (3 wool + 3 planks) and call "
            "sleep_in_bed to skip the night."
        )

    return None


NIGHT_BED = Nudge(name="night_bed", render=_night_bed)


# Registry. Add a nudge here and it becomes A/B-selectable automatically.
NUDGES: list[Nudge] = [FOOD_LOW, NIGHT_BED]
NUDGES_BY_NAME: dict[str, Nudge] = {n.name: n for n in NUDGES}


def resolve_nudges(spec: str | None) -> list[Nudge]:
    """Resolve an active nudge set from the CRAFT_NUDGES env-spec string.

    - None (env unset)      -> default set (all of NUDGES).
    - "" (set but empty)    -> empty set (control arm: no nudges fire).
    - "food_low,night_bed"  -> exactly those, in the order given.
    - Unknown name          -> ValueError (a typo silently no-op'ing a
                               campaign arm is the failure mode to avoid).
    """
    if spec is None:
        return list(NUDGES)
    tokens = [tok.strip() for tok in spec.split(",")]
    tokens = [tok for tok in tokens if tok]
    chain: list[Nudge] = []
    for name in tokens:
        n = NUDGES_BY_NAME.get(name)
        if n is None:
            known = ", ".join(NUDGES_BY_NAME) or "(none)"
            raise ValueError(
                f"Unknown nudge '{name}' in CRAFT_NUDGES. Known nudges: {known}."
            )
        chain.append(n)
    return chain


def render_nudges(nudges: list[Nudge], state: dict) -> str | None:
    """Render the active nudges' hint lines into a STATE sub-block, or None.

    A nudge whose render() raises is skipped (a nudge bug must never crash the
    rollout loop) — reactive hints are best-effort, not load-bearing.
    """
    lines: list[str] = []
    for n in nudges:
        try:
            hint = n.render(state)
        except Exception:
            hint = None
        if hint:
            lines.append("- " + hint)
    if not lines:
        return None
    return "SUGGESTED ACTIONS (based on current state):\n" + "\n".join(lines)
