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
from craft.milestones import M1, M2, Milestone, Milestones, _has


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

    # ---- Regression 2026-05-20: armor slot was being dropped ----
    # `_inventory_compact` originally flattened only main + offhand, so
    # equipped armor (head/chest/legs/feet — the natural state we care
    # about) was invisible to predicates like M2_diamond_goal. Found via
    # `--starting-loadout iron_armored` smoke (M2 didn't fire despite the
    # agent literally wearing iron armor). These tests pin the fix.

    def test_includes_armor_slot_single(self):
        raw = {
            "main": [],
            "offhand": None,
            "armor": {
                "head": {"id": "minecraft:iron_helmet", "count": 1},
                "chest": None, "legs": None, "feet": None,
            },
        }
        compact = _inventory_compact(raw)
        assert compact["minecraft:iron_helmet"] == 1

    def test_includes_full_armor_set(self):
        raw = {
            "main": [],
            "offhand": None,
            "armor": {
                "head":  {"id": "minecraft:iron_helmet", "count": 1},
                "chest": {"id": "minecraft:iron_chestplate", "count": 1},
                "legs":  {"id": "minecraft:iron_leggings", "count": 1},
                "feet":  {"id": "minecraft:iron_boots", "count": 1},
            },
        }
        compact = _inventory_compact(raw)
        for piece in (
            "minecraft:iron_helmet", "minecraft:iron_chestplate",
            "minecraft:iron_leggings", "minecraft:iron_boots",
        ):
            assert compact[piece] == 1, f"{piece} missing from compact view"

    def test_armor_none_entries_safe(self):
        """Empty armor slots come back as None; must not crash _inventory_compact."""
        raw = {
            "main": [{"id": "minecraft:dirt", "count": 5, "slot": 0}],
            "offhand": None,
            "armor": {"head": None, "chest": None, "legs": None, "feet": None},
        }
        compact = _inventory_compact(raw)
        assert compact == {"minecraft:dirt": 5}

    def test_missing_armor_key_safe(self):
        """Older homunculus responses or tests might omit the `armor` key
        entirely. Don't crash, just skip it."""
        raw = {
            "main": [{"id": "minecraft:dirt", "count": 5, "slot": 0}],
            "offhand": None,
        }
        compact = _inventory_compact(raw)
        assert compact == {"minecraft:dirt": 5}

    def test_armor_count_sums_with_main(self):
        """Edge case: same item id worn AND in main — counts should sum.
        (You can hold a helmet in main while wearing one.)"""
        raw = {
            "main": [{"id": "minecraft:iron_helmet", "count": 2, "slot": 0}],
            "offhand": None,
            "armor": {
                "head": {"id": "minecraft:iron_helmet", "count": 1},
                "chest": None, "legs": None, "feet": None,
            },
        }
        compact = _inventory_compact(raw)
        assert compact["minecraft:iron_helmet"] == 3


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

    def test_m2_fires_with_full_armor_via_raw_shape(self):
        """End-to-end regression for the armor-slot bug.

        Boot the homunculus-shaped inventory with full iron armor in the
        `armor` slot (the natural state — equipped, NOT in main). Run it
        through _inventory_compact then through Milestones.check with M2
        in the chain. M2 must fire on the first eligible turn.

        Pre-fix: this would silently NOT fire — _inventory_compact dropped
        the armor key so the predicate saw an empty inv. Caught in 33s by
        the `--starting-loadout iron_armored` smoke (2026-05-20).
        """
        # Spawn turn: empty armor (the natural pre-iron state). M2 should
        # NOT fire here.
        raw_empty = {
            "main": [],
            "offhand": None,
            "armor": {"head": None, "chest": None, "legs": None, "feet": None},
        }
        # Post-craft turn: full iron armor equipped (the test condition).
        raw_full = {
            "main": [
                {"id": "minecraft:iron_pickaxe", "count": 1, "slot": 0},
                {"id": "minecraft:iron_sword", "count": 1, "slot": 1},
            ],
            "offhand": None,
            "armor": {
                "head":  {"id": "minecraft:iron_helmet", "count": 1},
                "chest": {"id": "minecraft:iron_chestplate", "count": 1},
                "legs":  {"id": "minecraft:iron_leggings", "count": 1},
                "feet":  {"id": "minecraft:iron_boots", "count": 1},
            },
        }
        ms = Milestones(milestones=[M2])
        # Spawn anchor — no fire (empty armor).
        e0 = ms.check(
            {"day_count": 0, "day_ticks": 0}, _inventory_compact(raw_empty), turn=1,
        )
        assert e0 is None, "M2 should not fire on empty-armor spawn turn"
        # Armor equipped — M2 fires.
        event = ms.check(
            {"day_count": 0, "day_ticks": 100}, _inventory_compact(raw_full), turn=2,
        )
        assert event is not None and event.name == "M2_diamond_goal", (
            "M2 must fire when the homunculus inventory has all four iron "
            "armor pieces in `armor` slots (the equipped state). If this "
            "fails, _inventory_compact has regressed to dropping the "
            "`armor` key — see commit history for context."
        )
