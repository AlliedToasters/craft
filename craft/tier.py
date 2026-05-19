"""Tech-tier classification from inventory snapshots.

Single source of truth for "what tier did this agent reach?" Used by
analysis scripts, the milestone backtester, and any future tier-progression
nudges.

**Why this module exists**: the obvious shortcut — grep the rollout log
for "minecraft:iron_pickaxe" — produces false positives because log lines
capture *attempted* tool calls regardless of success. A failed
`craft({"item":"iron_pickaxe"})` ends up in the log even though the agent
never actually held the tool.

The correct signal is *inventory snapshots*: did the agent's inventory
ever contain a tier marker? That's what these helpers check.
"""

from __future__ import annotations
from typing import Iterable


# Tier ordering. Higher rank wins when classifying "best tier reached".
TIER_RANK: dict[str, int] = {
    "NONE": 0,
    "WOOD": 1,
    "STONE": 2,
    "IRON": 3,
    "DIAMOND": 4,
}


# Tier markers: presence of any item containing these substrings in an
# inventory implies the agent reached that tier. Pickaxes + swords only —
# axes/shovels/hoes can be wood-crafted without progression intent, so
# they're weak signals. Pickaxe-or-sword presence is the canonical marker.
_TIER_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("DIAMOND", ("diamond_pickaxe", "diamond_sword")),
    ("IRON", ("iron_pickaxe", "iron_sword")),
    ("STONE", ("stone_pickaxe", "stone_sword")),
    ("WOOD", ("wooden_pickaxe", "wooden_sword")),
]


def tier_from_inventory(inv: dict | None) -> str:
    """Return the tier label implied by items currently in `inv`.

    `inv` must be the flat {item_id: count} shape (same as JSONL turn
    records). Returns "NONE" if no tier marker is present.

    Note: this looks at one inventory snapshot. For "highest tier ever
    reached across a rollout" use `best_tier_across_turns`.
    """
    if not inv:
        return "NONE"
    keys = list(inv.keys())
    for tier, markers in _TIER_MARKERS:
        if any(any(marker in key for marker in markers) for key in keys):
            return tier
    return "NONE"


def best_tier_across_turns(turns: Iterable[dict]) -> str:
    """Highest tier observed across a sequence of turn records.

    Each turn dict is expected to have an "inventory" key with the flat
    {item_id: count} shape (matches the JSONL records written by
    craft.agent). Turns without inventory are skipped.
    """
    best = 0
    for t in turns:
        inv = t.get("inventory") if isinstance(t, dict) else None
        rank = TIER_RANK[tier_from_inventory(inv)]
        if rank > best:
            best = rank
    return next(k for k, v in TIER_RANK.items() if v == best)
