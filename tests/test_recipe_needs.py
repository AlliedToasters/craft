"""Unit tests for _recipe_needs() recursive recipe expansion.

`_recipe_needs(item, count)` walks `CRAFTING_RECIPES` recursively to compute
the leaf-level ingredient bill for an end item. It's the substrate input to
`_throwaway_policy`, which decides what Baritone can place during a
craft-triggered goto — protecting recipe ingredients from being burnt as
pillar-up blocks (the 2026-05-11 r3 doom loop, per
[[project-baritone-inventory-consumption]]).

Risks covered here:
- **Depth termination**: `_depth > 8` is the only loop guard. A circular
  recipe edit would otherwise blow the stack.
- **Multiplicative semantics**: count=2 must produce exactly double the
  leaf needs of count=1. A regression to additive math would silently
  under-protect Baritone goto.
- **Leaf aggregation**: multiple paths to the same leaf (e.g.
  wooden_pickaxe needs oak_log via *both* planks-branch and stick-branch)
  must sum, not overwrite.
"""

from __future__ import annotations

import pytest

from craft import tools
from craft.tools import CRAFTING_RECIPES, _recipe_needs


# ---------------------------------------------- base cases (leaves)


class TestLeafItems:
    """Items not in CRAFTING_RECIPES are leaves — return {item: count}."""

    def test_unknown_item_returns_self(self):
        assert _recipe_needs("minecraft:dirt", 5) == {"minecraft:dirt": 5}

    def test_iron_ingot_is_leaf(self):
        """iron_ingot is a leaf — must be smelted, not crafted."""
        assert _recipe_needs("minecraft:iron_ingot", 3) == {"minecraft:iron_ingot": 3}

    def test_cobblestone_is_leaf(self):
        assert _recipe_needs("minecraft:cobblestone", 8) == {"minecraft:cobblestone": 8}

    def test_oak_log_is_leaf(self):
        assert _recipe_needs("minecraft:oak_log", 1) == {"minecraft:oak_log": 1}

    def test_diamond_is_leaf(self):
        assert _recipe_needs("minecraft:diamond", 3) == {"minecraft:diamond": 3}


# ---------------------------------------------- single-level recipes


class TestSingleLevel:
    """Items whose recipe ingredients are all leaves."""

    def test_oak_planks_expands_to_oak_log(self):
        """Recipe: 1 oak_planks <- 1 oak_log (table convention; MC's actual
        1log→4planks is handled at homunculus craft time)."""
        assert _recipe_needs("minecraft:oak_planks", 1) == {"minecraft:oak_log": 1}

    def test_oak_planks_scales_multiplicatively(self):
        assert _recipe_needs("minecraft:oak_planks", 4) == {"minecraft:oak_log": 4}

    def test_furnace_expands_to_cobblestone(self):
        """8 cobblestone per furnace, no sub-recipe."""
        assert _recipe_needs("minecraft:furnace", 1) == {"minecraft:cobblestone": 8}

    def test_furnace_scales(self):
        assert _recipe_needs("minecraft:furnace", 2) == {"minecraft:cobblestone": 16}

    def test_crafting_table_expands_to_logs(self):
        """crafting_table: 4 oak_planks <- 4 oak_log."""
        assert _recipe_needs("minecraft:crafting_table", 1) == {"minecraft:oak_log": 4}

    def test_pale_oak_planks(self):
        """Cover the newest species entry."""
        assert _recipe_needs("minecraft:pale_oak_planks", 2) == {"minecraft:pale_oak_log": 2}


# ---------------------------------------------- multi-level recipes


class TestMultiLevelRecursion:
    """Items whose recipes recurse through intermediate craftables."""

    def test_stick_expands_through_planks(self):
        """stick(1) -> 2 oak_planks -> 2 oak_log."""
        assert _recipe_needs("minecraft:stick", 1) == {"minecraft:oak_log": 2}

    def test_stick_scales(self):
        """stick(2) -> 4 oak_planks -> 4 oak_log."""
        assert _recipe_needs("minecraft:stick", 2) == {"minecraft:oak_log": 4}

    def test_wooden_pickaxe_aggregates_logs_across_branches(self):
        """wooden_pickaxe(1):
          - 3 oak_planks  -> 3 oak_log
          - 2 stick       -> 2*2=4 oak_planks -> 4 oak_log
          = 7 oak_log total.

        If this returns 3 or 4, the leaf-aggregation logic regressed.
        """
        assert _recipe_needs("minecraft:wooden_pickaxe", 1) == {"minecraft:oak_log": 7}

    def test_wooden_pickaxe_scales(self):
        """count=2 → 14 oak_log (strictly multiplicative)."""
        assert _recipe_needs("minecraft:wooden_pickaxe", 2) == {"minecraft:oak_log": 14}


# ---------------------------------------------- mixed leaves


class TestMixedLeafTypes:
    """Recipes producing multiple distinct leaves."""

    def test_stone_pickaxe(self):
        """stone_pickaxe(1):
          - 3 cobblestone (leaf)
          - 2 stick -> 4 oak_log
        """
        assert _recipe_needs("minecraft:stone_pickaxe", 1) == {
            "minecraft:cobblestone": 3,
            "minecraft:oak_log": 4,
        }

    def test_iron_pickaxe(self):
        """iron_pickaxe(1) needs 3 iron_ingot (leaf) + 4 oak_log via sticks."""
        assert _recipe_needs("minecraft:iron_pickaxe", 1) == {
            "minecraft:iron_ingot": 3,
            "minecraft:oak_log": 4,
        }

    def test_diamond_pickaxe(self):
        """diamond_pickaxe(1) needs 3 diamond + 4 oak_log via sticks."""
        assert _recipe_needs("minecraft:diamond_pickaxe", 1) == {
            "minecraft:diamond": 3,
            "minecraft:oak_log": 4,
        }

    def test_iron_pickaxe_scales(self):
        """Pin scaling on multi-leaf recipes — both leaves must double."""
        assert _recipe_needs("minecraft:iron_pickaxe", 3) == {
            "minecraft:iron_ingot": 9,
            "minecraft:oak_log": 12,
        }


# ---------------------------------------------- count=0 edge


class TestZeroCount:
    """count=0 still traverses (multiplicative), produces zero-valued leaves."""

    def test_zero_count_leaf(self):
        assert _recipe_needs("minecraft:cobblestone", 0) == {"minecraft:cobblestone": 0}

    def test_zero_count_recursive(self):
        """count=0 propagates through recursion as 0 * anything = 0."""
        assert _recipe_needs("minecraft:wooden_pickaxe", 0) == {"minecraft:oak_log": 0}


# ---------------------------------------------- depth cap


class TestDepthCap:
    """`_depth > 8` is the only protection against recipe cycles. If
    recipes are ever edited to form a cycle (e.g. someone swaps oak_planks
    to require sticks), the function must terminate — not blow the stack.

    These tests install a faux CRAFTING_RECIPES via monkeypatch and verify
    termination behavior.
    """

    def test_circular_recipe_terminates(self, monkeypatch):
        """Direct cycle: a -> b -> a. Without the depth cap this recurses
        forever. With the cap, the 9th call returns {leaf: count} as if
        the item were a leaf — limiting damage."""
        circular = {
            "x:a": [("x:b", 1)],
            "x:b": [("x:a", 1)],
        }
        monkeypatch.setattr(tools, "CRAFTING_RECIPES", circular)
        # Should not raise RecursionError
        result = _recipe_needs("x:a", 1)
        # At depth 9 the recursion stops and emits the item-at-depth-9 as a
        # leaf with count=1 (multiplicative 1*1*1...). The exact terminal
        # key alternates a/b based on depth parity but either is acceptable;
        # the critical property is termination.
        assert sum(result.values()) == 1
        assert set(result.keys()) <= {"x:a", "x:b"}

    def test_long_linear_chain_terminates(self, monkeypatch):
        """Chain longer than 8 levels: each item recipes-to next, ending in
        a leaf. Items at depth >8 are treated as leaves (premature
        termination — but at least no infinite recursion)."""
        # i1 -> i2 -> ... -> i10 -> leaf
        chain = {f"x:i{i}": [(f"x:i{i+1}", 1)] for i in range(1, 10)}
        # i10 has no recipe → leaf naturally
        monkeypatch.setattr(tools, "CRAFTING_RECIPES", chain)
        result = _recipe_needs("x:i1", 1)
        # Depth cap fires before reaching the true leaf — emits the
        # item-at-depth-8 as a quasi-leaf. Either way, terminates.
        assert sum(result.values()) == 1


# ---------------------------------------------- recipe table sanity


class TestRecipeTableSanity:
    """Catch malformed CRAFTING_RECIPES entries that would silently break
    _recipe_needs."""

    def test_all_recipes_have_at_least_one_ingredient(self):
        for item, recipe in CRAFTING_RECIPES.items():
            assert len(recipe) > 0, f"{item} has empty recipe"

    def test_all_recipe_counts_positive(self):
        for item, recipe in CRAFTING_RECIPES.items():
            for ing, count in recipe:
                assert count > 0, f"{item} has zero/negative count for {ing}"

    def test_no_recipe_references_itself_directly(self):
        """A direct self-reference would loop until depth-cap fires. Not
        catastrophic but a code smell — pin against accidental introduction."""
        for item, recipe in CRAFTING_RECIPES.items():
            for ing, _count in recipe:
                assert ing != item, f"{item} recipe references itself"

    def test_iron_pickaxe_iron_ingot_is_leaf(self):
        """Regression guard: if someone adds an iron_ingot recipe to the
        table (it should stay leaf — smelting, not crafting), this fails.
        iron_pickaxe's bill should remain {iron_ingot: 3, oak_log: 4}."""
        assert "minecraft:iron_ingot" not in CRAFTING_RECIPES, (
            "iron_ingot must remain a leaf (acquired via smelt, not craft). "
            "Adding it to CRAFTING_RECIPES will silently change every iron-tier "
            "recipe's leaf bill."
        )
