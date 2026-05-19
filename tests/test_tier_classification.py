"""Unit tests for craft.tier.

Includes a regression test for the **log-grep false-positive bug**: prior
ad-hoc tier classification grepped rollout logs for `minecraft:iron_pickaxe`,
which matched FAILED `craft({"item":"iron_pickaxe"})` attempts. Wave 1 of
the M1 milestone test (2026-05-18) initially classified agent0 as IRON
tier when it had only *attempted* to craft iron_pickaxe (the craft failed
because the agent had no iron_ingot). True tier = WOOD.

The correct signal is whether iron_pickaxe ever appeared in an **inventory
snapshot**, not in the log. These tests pin that contract.
"""

from craft.tier import (
    TIER_RANK,
    best_tier_across_turns,
    tier_from_inventory,
)


# ---------------------------------------------- tier_from_inventory


class TestTierFromInventory:
    def test_empty(self):
        assert tier_from_inventory({}) == "NONE"

    def test_none(self):
        assert tier_from_inventory(None) == "NONE"

    def test_wood(self):
        inv = {"minecraft:wooden_pickaxe": 1, "minecraft:dirt": 5}
        assert tier_from_inventory(inv) == "WOOD"

    def test_stone(self):
        inv = {"minecraft:stone_pickaxe": 1, "minecraft:wooden_pickaxe": 1}
        assert tier_from_inventory(inv) == "STONE"

    def test_iron(self):
        inv = {"minecraft:iron_sword": 1, "minecraft:cobblestone": 64}
        assert tier_from_inventory(inv) == "IRON"

    def test_diamond_outranks_iron(self):
        inv = {"minecraft:diamond_pickaxe": 1, "minecraft:iron_pickaxe": 1}
        assert tier_from_inventory(inv) == "DIAMOND"

    def test_non_progression_items_dont_count(self):
        """Wooden_axe / wooden_shovel alone shouldn't imply WOOD tier —
        only pickaxe and sword are the canonical markers."""
        inv = {"minecraft:wooden_axe": 1, "minecraft:wooden_shovel": 1}
        assert tier_from_inventory(inv) == "NONE"

    def test_sword_alone_is_enough(self):
        """A wooden_sword without pickaxe still counts as WOOD tier."""
        assert tier_from_inventory({"minecraft:wooden_sword": 1}) == "WOOD"

    def test_iron_ingot_alone_is_not_iron_tier(self):
        """Ingredient ≠ tier. iron_ingot in inventory means the agent
        mined+smelted iron but didn't necessarily craft an iron tool."""
        assert tier_from_inventory({"minecraft:iron_ingot": 5}) == "NONE"


# ---------------------------------------------- best_tier_across_turns


class TestBestTierAcrossTurns:
    def test_progresses_with_turns(self):
        turns = [
            {"turn": 1, "inventory": {"minecraft:dirt": 5}},
            {"turn": 5, "inventory": {"minecraft:wooden_pickaxe": 1}},
            {"turn": 12, "inventory": {"minecraft:stone_pickaxe": 1}},
        ]
        assert best_tier_across_turns(turns) == "STONE"

    def test_remembers_higher_tier_even_if_lost(self):
        """If the agent drops/loses iron_pickaxe later, best-tier still IRON."""
        turns = [
            {"turn": 1, "inventory": {"minecraft:wooden_pickaxe": 1}},
            {"turn": 10, "inventory": {"minecraft:iron_pickaxe": 1}},
            {"turn": 20, "inventory": {"minecraft:dirt": 5}},  # lost the iron
        ]
        assert best_tier_across_turns(turns) == "IRON"

    def test_empty_sequence(self):
        assert best_tier_across_turns([]) == "NONE"

    def test_skips_turns_without_inventory(self):
        turns = [
            {"turn": 1},  # no inventory key
            {"turn": 2, "inventory": {"minecraft:wooden_pickaxe": 1}},
        ]
        assert best_tier_across_turns(turns) == "WOOD"


# --------------------------------------- regression: log-grep false positive


class TestLogGrepFalsePositiveRegression:
    """Documents the bug that motivated this module.

    Wave 1 / M1 test, 2026-05-18: agent0 attempted `craft(iron_pickaxe)`
    at T81 but the craft FAILED (no iron_ingot). The string
    "minecraft:iron_pickaxe" appeared in the log via the executing-tool
    line. A naive grep classified the rollout as IRON tier. True tier =
    WOOD (agent never held iron_pickaxe in inventory).

    These tests pin that the inventory-based classifier returns the
    correct tier even when a tool-call ATTEMPTED higher tier crafting.
    """

    def test_attempted_iron_craft_without_inventory_iron_is_not_iron(self):
        """The exact wave 1 agent0 scenario: agent attempted craft(iron_pickaxe)
        at T81 but the craft FAILED. Snapshot has wooden_pickaxe + raw
        materials but no iron tools. Naive log-grep would have called this
        IRON (because the failed craft call wrote "iron_pickaxe" into the
        log). Inventory-based classifier correctly returns WOOD."""
        inv_at_attempt = {
            "minecraft:wooden_pickaxe": 1,
            "minecraft:oak_log": 17,
            "minecraft:cobblestone": 44,
            "minecraft:charcoal": 23,
            "minecraft:coal": 8,
            "minecraft:furnace": 1,
            "minecraft:crafting_table": 2,
            # NOTE: no iron_pickaxe, no iron_sword, no iron_ingot
        }
        result = tier_from_inventory(inv_at_attempt)
        assert result == "WOOD", (
            f"Expected WOOD (agent has wooden tools, no iron); got {result}. "
            "If this returns IRON, the classifier is matching on something "
            "other than inventory content — the original log-grep bug."
        )

    def test_failed_iron_craft_doesnt_taint_rollout_tier(self):
        """Across a rollout where the agent only attempted iron crafting,
        best_tier_across_turns should still return WOOD."""
        # Simulating agent0's actual JSONL turn trajectory: wood at T5,
        # multiple later turns with the same wood inventory (no iron ever)
        turns = [
            {"turn": 5, "inventory": {"minecraft:wooden_pickaxe": 1}},
            {"turn": 81, "inventory": {  # attempted craft iron_pickaxe HERE
                "minecraft:wooden_axe": 1,
                "minecraft:cobblestone": 44,
            }},
            {"turn": 95, "inventory": {  # rollout end
                "minecraft:wooden_axe": 1,
                "minecraft:cobblestone": 44,
            }},
        ]
        assert best_tier_across_turns(turns) == "WOOD"

    def test_inventory_iron_pickaxe_correctly_classified_iron(self):
        """Positive control: when the agent ACTUALLY has iron_pickaxe in
        inventory (i.e. successful craft), classifier should return IRON."""
        turns = [
            {"turn": 5, "inventory": {"minecraft:wooden_pickaxe": 1}},
            {"turn": 90, "inventory": {
                "minecraft:iron_pickaxe": 1,
                "minecraft:iron_ingot": 2,
            }},
        ]
        assert best_tier_across_turns(turns) == "IRON"


# --------------------------------------- tier_rank monotonicity


class TestTierRank:
    def test_strict_ordering(self):
        assert TIER_RANK["NONE"] < TIER_RANK["WOOD"]
        assert TIER_RANK["WOOD"] < TIER_RANK["STONE"]
        assert TIER_RANK["STONE"] < TIER_RANK["IRON"]
        assert TIER_RANK["IRON"] < TIER_RANK["DIAMOND"]
