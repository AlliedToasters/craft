"""Unit tests for `_throwaway_policy()` — recipe-aware Baritone protection.

`_throwaway_policy(item, count)` decides what Baritone may pillar-up / bridge
with during a craft-triggered goto. It composes:
    - `_recipe_needs(item, count)` → leaf bill (already pinned in
      [[test_recipe_needs.py]])
    - `THROWAWAY_ITEMS` → blocks Baritone considers "free to burn"
into one of three policy outcomes:

    Branch A — no overlap between needs and throwaways:
        (True, None, False)          # default Baritone behavior
    Branch B — every throwaway is a recipe ingredient:
        (False, None, False)         # disable placement entirely
    Branch C — partial overlap:
        (True, sorted_permitted, True)  # restrict placement set

**Why this exists**: 2026-05-11 r3 doom loop — agent mined stone, called
craft(stone_pickaxe), Baritone immediately used the freshly-mined cobble
as pillar-up bridging blocks during the goto to the crafting table. Agent
arrived broke. Recipe-aware allow_place protects against this exact
pattern, see [[project-baritone-inventory-consumption]].
"""

from __future__ import annotations

import pytest

from craft import tools
from craft.tools import THROWAWAY_ITEMS, _throwaway_policy


# ---------------------------------------------- branch A: no overlap


class TestNoOverlap:
    """Recipes whose leaf bill doesn't touch any throwaway item.
    Baritone keeps default permissive behavior."""

    def test_wooden_pickaxe(self):
        """wooden_pickaxe needs {oak_log} — no overlap with throwaways."""
        result = _throwaway_policy("minecraft:wooden_pickaxe", 1)
        assert result == (True, None, False)

    def test_wooden_pickaxe_high_count(self):
        """Quantity doesn't change set-algebra outcome — pin against accidental
        count-dependent logic creeping in."""
        result = _throwaway_policy("minecraft:wooden_pickaxe", 64)
        assert result == (True, None, False)

    def test_iron_pickaxe(self):
        """iron_pickaxe leaves: {iron_ingot, oak_log} — no overlap."""
        result = _throwaway_policy("minecraft:iron_pickaxe", 1)
        assert result == (True, None, False)

    def test_diamond_pickaxe(self):
        """diamond_pickaxe leaves: {diamond, oak_log} — no overlap."""
        result = _throwaway_policy("minecraft:diamond_pickaxe", 1)
        assert result == (True, None, False)

    def test_leaf_with_no_overlap(self):
        """Bare leaf item with no recipe + no throwaway membership →
        passthrough returns the policy for the leaf itself."""
        result = _throwaway_policy("minecraft:oak_log", 5)
        assert result == (True, None, False)

    def test_iron_ingot_leaf(self):
        """iron_ingot is a leaf (smelting target); needs={iron_ingot} — no
        throwaway overlap."""
        result = _throwaway_policy("minecraft:iron_ingot", 3)
        assert result == (True, None, False)


# ---------------------------------------------- branch C: partial overlap


class TestPartialOverlap:
    """Recipes whose leaf bill includes SOME throwaway items but not all.
    Baritone permitted to place the non-reserved throwaway subset."""

    def test_stone_pickaxe_reserves_cobblestone(self):
        """The 2026-05-11 r3 fix scenario.

        stone_pickaxe leaves: {cobblestone, oak_log}.
        cobblestone is in THROWAWAY_ITEMS → reserved.
        dirt + netherrack remain permitted.
        """
        allow_place, permitted, ensure = _throwaway_policy(
            "minecraft:stone_pickaxe", 1
        )
        assert allow_place is True
        assert permitted == ["minecraft:dirt", "minecraft:netherrack"]
        assert ensure is True
        # Critical: cobble must NOT be in permitted (it's the protected ingredient)
        assert "minecraft:cobblestone" not in permitted

    def test_stone_pickaxe_high_count_same_decision(self):
        """count=64 doesn't add new ingredients; permitted set identical."""
        result = _throwaway_policy("minecraft:stone_pickaxe", 64)
        assert result == (True, ["minecraft:dirt", "minecraft:netherrack"], True)

    def test_furnace_reserves_cobblestone(self):
        """furnace recipe is purely cobblestone → still permits dirt+netherrack."""
        allow_place, permitted, ensure = _throwaway_policy("minecraft:furnace", 1)
        assert allow_place is True
        assert permitted == ["minecraft:dirt", "minecraft:netherrack"]
        assert ensure is True

    def test_cobblestone_leaf_reserves_itself(self):
        """Asking the policy for cobblestone itself (e.g. as a craft target):
        needs={cobblestone} which is a throwaway → reserved.
        Other two throwaways still permitted."""
        result = _throwaway_policy("minecraft:cobblestone", 1)
        assert result == (True, ["minecraft:dirt", "minecraft:netherrack"], True)

    def test_dirt_leaf_reserves_itself(self):
        """Symmetry pin: dirt as target reserves dirt, permits {cobble, netherrack}."""
        result = _throwaway_policy("minecraft:dirt", 1)
        assert result == (
            True,
            ["minecraft:cobblestone", "minecraft:netherrack"],
            True,
        )

    def test_netherrack_leaf_reserves_itself(self):
        result = _throwaway_policy("minecraft:netherrack", 1)
        assert result == (
            True,
            ["minecraft:cobblestone", "minecraft:dirt"],
            True,
        )


# ---------------------------------------------- branch B: all reserved


class TestAllReserved:
    """Hypothetical: every throwaway is an ingredient. No real recipe in the
    table triggers this, so we monkeypatch one in. Result: (False, None, False)
    — placement disabled entirely, goto may fail-unreachable. Strictly better
    than silent inventory consumption."""

    def test_synthetic_all_three_throwaways(self, monkeypatch):
        synthetic = {
            "test:everything_burner": [
                ("minecraft:cobblestone", 1),
                ("minecraft:dirt", 1),
                ("minecraft:netherrack", 1),
            ],
        }
        monkeypatch.setattr(tools, "CRAFTING_RECIPES", synthetic)
        result = _throwaway_policy("test:everything_burner", 1)
        assert result == (False, None, False)

    def test_synthetic_subset_via_recursion(self, monkeypatch):
        """All three throwaways reached through recursion, not direct."""
        synthetic = {
            "test:dirt_block": [("minecraft:dirt", 1)],
            "test:all_three": [
                ("minecraft:cobblestone", 1),
                ("test:dirt_block", 1),
                ("minecraft:netherrack", 1),
            ],
        }
        monkeypatch.setattr(tools, "CRAFTING_RECIPES", synthetic)
        assert _throwaway_policy("test:all_three", 1) == (False, None, False)


# ---------------------------------------------- structural / shape pins


class TestReturnShape:
    """Pin the return tuple shape since callers (handle_craft) unpack
    positionally into homunculus /baritone/goto params."""

    def test_returns_three_tuple(self):
        result = _throwaway_policy("minecraft:wooden_pickaxe", 1)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_permitted_is_list_not_set(self):
        """homunculus expects a JSON array — sets would not serialize stably."""
        _allow, permitted, _ensure = _throwaway_policy("minecraft:stone_pickaxe", 1)
        assert isinstance(permitted, list)

    def test_permitted_items_are_fully_qualified_ids(self):
        """All items must keep the `minecraft:` prefix (homunculus requires it)."""
        _allow, permitted, _ensure = _throwaway_policy("minecraft:stone_pickaxe", 1)
        for ident in permitted:
            assert ident.startswith("minecraft:"), (
                f"unprefixed id {ident!r} in permitted list"
            )


class TestPermittedListOrdering:
    """The function uses `sorted(THROWAWAY_ITEMS - protected)` so output is
    deterministic. If someone removes the sort, Baritone may see a different
    list across calls and behave non-reproducibly — pin against that drift."""

    def test_ordering_is_stable_across_calls(self):
        results = [_throwaway_policy("minecraft:stone_pickaxe", 1) for _ in range(5)]
        permitted_lists = [r[1] for r in results]
        assert all(p == permitted_lists[0] for p in permitted_lists)

    def test_ordering_is_lexicographic(self):
        """`sorted()` is lex on minecraft: ids — pin that ordering specifically."""
        _allow, permitted, _ensure = _throwaway_policy("minecraft:stone_pickaxe", 1)
        assert permitted == sorted(permitted)


# ---------------------------------------------- count=0 edge


class TestZeroCount:
    """count=0 propagates through _recipe_needs as {leaf: 0}; keys are still
    present so set-algebra still fires. This is the "Baritone should still
    protect even if quantity asked for is zero" property — defensive pin
    against accidentally short-circuiting on count=0."""

    def test_zero_count_no_overlap(self):
        """wooden_pickaxe count=0 → needs={oak_log: 0}; still no overlap."""
        result = _throwaway_policy("minecraft:wooden_pickaxe", 0)
        assert result == (True, None, False)

    def test_zero_count_partial_overlap(self):
        """stone_pickaxe count=0 → needs={cobble: 0, oak_log: 0}; cobble key
        still triggers reservation."""
        result = _throwaway_policy("minecraft:stone_pickaxe", 0)
        assert result == (True, ["minecraft:dirt", "minecraft:netherrack"], True)


# ---------------------------------------------- substrate-coupling sanity


class TestThrowawaySetSanity:
    """Pin the contents of THROWAWAY_ITEMS — these three blocks are
    load-bearing for the policy contract and changing them would silently
    alter every craft's protection profile.

    If Baritone's actual throwaway set changes upstream and we want to mirror
    it, this test fails loudly so we update both sides in lock-step instead
    of drifting."""

    def test_throwaway_set_contents(self):
        assert THROWAWAY_ITEMS == {
            "minecraft:cobblestone",
            "minecraft:dirt",
            "minecraft:netherrack",
        }

    def test_throwaway_items_are_fully_qualified(self):
        for item in THROWAWAY_ITEMS:
            assert item.startswith("minecraft:")

    def test_no_progression_tools_in_throwaway_set(self):
        """Sanity guard: no pickaxe/sword/ingot/log/ore should ever be a
        throwaway candidate (would defeat the whole protection purpose)."""
        forbidden_substrings = ("pickaxe", "sword", "ingot", "_log", "_ore", "planks")
        for item in THROWAWAY_ITEMS:
            for sub in forbidden_substrings:
                assert sub not in item, (
                    f"{item} is in THROWAWAY_ITEMS but contains {sub!r} — "
                    "progression items must never be marked throwaway"
                )
