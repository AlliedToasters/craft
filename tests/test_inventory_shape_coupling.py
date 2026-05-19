"""Regression test for the inventory-shape bug between agent.py and milestones.

The bug: `_inventory_raw()` returns the homunculus shape:
    {"main": [{"id": str, "count": int}, ...], "offhand": {...} | None}

but `Milestones.check()` (and the predicates' `_has()` helper) expect the
flat shape:
    {item_id: count}

Pre-fix, `agent.py` passed `_inventory_raw()` directly to milestones, so
`_has(inv, ":wooden_pickaxe")` never matched (it iterated dict keys
"main" / "offhand" instead of item ids). M1 silently never fired in
N=10 rollouts before the bug was caught by inspecting day_ticks vs
fire counts.

The fix: `agent.py` now passes `_inventory_compact(inv_raw)` to
milestones.check. These tests pin the contract.
"""

import pytest

from craft.agent import _inventory_compact
from craft.milestones import Milestones, _has


# Sample raw shape exactly as homunculus's /inventory returns it — captured
# during a real rollout. The `main` list always has 36 slots (some with
# id=None for empty); offhand may be None.
RAW_INVENTORY_SAMPLE = {
    "main": [
        {"id": "minecraft:wooden_pickaxe", "count": 1, "slot": 0},
        {"id": "minecraft:oak_log", "count": 8, "slot": 1},
        {"id": "minecraft:dirt", "count": 23, "slot": 2},
        # ... empty slots typically appear as {"id": None, "count": 0}
    ],
    "offhand": None,
}


class TestInventoryCompact:
    def test_flattens_main(self):
        compact = _inventory_compact(RAW_INVENTORY_SAMPLE)
        assert compact["minecraft:wooden_pickaxe"] == 1
        assert compact["minecraft:oak_log"] == 8
        assert compact["minecraft:dirt"] == 23

    def test_sums_duplicate_slots(self):
        """If the same item appears in multiple slots, counts should sum."""
        raw = {
            "main": [
                {"id": "minecraft:cobblestone", "count": 64, "slot": 0},
                {"id": "minecraft:cobblestone", "count": 32, "slot": 5},
            ],
            "offhand": None,
        }
        compact = _inventory_compact(raw)
        assert compact["minecraft:cobblestone"] == 96

    def test_includes_offhand(self):
        raw = {
            "main": [{"id": "minecraft:dirt", "count": 5, "slot": 0}],
            "offhand": {"id": "minecraft:torch", "count": 4},
        }
        compact = _inventory_compact(raw)
        assert compact["minecraft:torch"] == 4
        assert compact["minecraft:dirt"] == 5

    def test_offhand_none_safe(self):
        raw = {"main": [{"id": "minecraft:dirt", "count": 5}], "offhand": None}
        compact = _inventory_compact(raw)
        assert "offhand" not in compact

    def test_empty_inventory(self):
        assert _inventory_compact({"main": [], "offhand": None}) == {}

    def test_none_inventory(self):
        assert _inventory_compact(None) == {}


class TestMilestonesAgainstCompactedInventory:
    """The regression test: M1 predicate must work with the inventory shape
    that agent.py actually passes it (compact, not raw)."""

    def test_compact_then_has_finds_pickaxe(self):
        compact = _inventory_compact(RAW_INVENTORY_SAMPLE)
        assert _has(compact, ":wooden_pickaxe")

    def test_raw_then_has_does_not_find(self):
        """Confirms the bug shape — _has against the RAW format will silently
        fail. This test is here to document the contract: callers MUST pass
        compact, not raw."""
        # _has iterates dict keys looking for endswith(":wooden_pickaxe").
        # In the raw shape, keys are "main" / "offhand" → no match.
        assert not _has(RAW_INVENTORY_SAMPLE, ":wooden_pickaxe")

    def test_milestones_check_rejects_raw_shape(self):
        """Hard contract: Milestones.check raises on the raw shape so the
        original bug fails loudly instead of silently. If this test fails,
        the defensive guard in milestones.py has been removed."""
        ms = Milestones()
        with pytest.raises(ValueError, match="raw inventory shape"):
            ms.check(
                stats={"day_count": 0, "day_ticks": 12100},
                inv=RAW_INVENTORY_SAMPLE,
                turn=20,
            )

    def test_full_data_path_agent_to_milestones(self):
        """End-to-end: simulate one turn of the agent's data path —
        _inventory_raw() shape → _inventory_compact() → Milestones.check().

        If this fails, M1 won't fire at runtime regardless of what the
        predicate says — same shape this bug originally caused.
        """
        ms = Milestones()
        # Spawn turn — anchor
        ms.check(
            stats={"day_count": 0, "day_ticks": 0},
            inv=_inventory_compact(RAW_INVENTORY_SAMPLE),
            turn=1,
        )
        # Past dusk threshold — should fire
        event = ms.check(
            stats={"day_count": 0, "day_ticks": 12100},
            inv=_inventory_compact(RAW_INVENTORY_SAMPLE),
            turn=20,
        )
        assert event is not None, (
            "M1 must fire when compact inventory has wooden_pickaxe and "
            "ticks_alive >= 12000. If this fails, check the shape passed "
            "from agent.py: it should be _inventory_compact(inv_raw), not "
            "inv_raw directly."
        )
        assert event.name == "M1_iron_goal"
