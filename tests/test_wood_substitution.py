"""Unit tests for wood-species recipe substitution.

`_resolve_wood_substitute()` in craft.tools is the substrate-side fix for
the recurring bug "agent has spruce_log, craft(wooden_pickaxe) fails because
the oak_planks substep can't resolve." Vanilla MC recipes accept any
*_planks via the #planks tag, so the substitute returns a species the agent
actually holds.

Recurring-bug history (per CLAUDE.md "Survival rules"):
- 2026-05-14 probe-validate-r2 T2: snowy_taiga spawn, 3× spruce_log, craft
  failed → first fix landed (this function).
- 2026-05-15 r5: bug recurred on the Java side; `Recipes.canonicalItem`
  returned `ing.items().findFirst()` (oak_planks for any #planks). Java
  fix prefers in-inventory species + python loop-detect guard.

These tests pin the Python-side contract so a third regression fails loud
at unit-test time instead of mid-rollout.
"""

from __future__ import annotations

import pytest
import requests

from craft import tools
from craft.tools import (
    CRAFTING_RECIPES,
    _PLANKS_LOG_BY_SPECIES,
    _resolve_wood_substitute,
)


# ---------------------------------------------- test helpers


class _FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, data, status: int = 200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _make_inv(*items, offhand=None) -> dict:
    """Build a homunculus-shaped inventory.

    `items` is a flat sequence of (id, count) tuples; each becomes one slot.
    """
    main = [{"id": item_id, "count": count, "slot": i}
            for i, (item_id, count) in enumerate(items)]
    return {"main": main, "offhand": offhand}


@pytest.fixture
def fake_inventory(monkeypatch):
    """Install a fake requests.get that returns a pre-built inventory.

    Usage:
        fake_inventory(_make_inv(("minecraft:spruce_log", 3)))
        result = _resolve_wood_substitute("minecraft:oak_planks", 2)
    """
    def _install(inv_data, *, raise_exc: Exception | None = None):
        def fake_get(url, timeout=None):
            if raise_exc is not None:
                raise raise_exc
            return _FakeResp(inv_data)
        monkeypatch.setattr(tools.requests, "get", fake_get)
    return _install


# ---------------------------------------------- _PLANKS_LOG_BY_SPECIES table


class TestPlanksLogTable:
    """The substitution table itself is load-bearing. Pin its shape so a
    typo in the dict (wrong species pairing, missing entry) trips loud."""

    EXPECTED_SPECIES = {
        "oak", "spruce", "birch", "jungle", "acacia",
        "dark_oak", "mangrove", "cherry", "pale_oak",
    }

    def test_all_nine_species_present(self):
        """All 9 overworld wood species must be in the table."""
        prefixes = {
            k.replace("minecraft:", "").replace("_planks", "")
            for k in _PLANKS_LOG_BY_SPECIES.keys()
        }
        assert prefixes == self.EXPECTED_SPECIES, (
            f"Missing species: {self.EXPECTED_SPECIES - prefixes}, "
            f"unexpected species: {prefixes - self.EXPECTED_SPECIES}"
        )

    def test_keys_are_planks_values_are_logs(self):
        for planks, log in _PLANKS_LOG_BY_SPECIES.items():
            assert planks.endswith("_planks"), f"{planks} is not a planks id"
            assert log.endswith("_log"), f"{log} is not a log id"

    def test_key_and_value_share_species_prefix(self):
        """oak_planks must map to oak_log, not birch_log."""
        for planks, log in _PLANKS_LOG_BY_SPECIES.items():
            planks_species = planks.removeprefix("minecraft:").removesuffix("_planks")
            log_species = log.removeprefix("minecraft:").removesuffix("_log")
            assert planks_species == log_species, (
                f"Mismatched pair: {planks} → {log}"
            )

    def test_every_species_has_a_crafting_recipe(self):
        """Each *_planks in the substitution table must have a recipe.

        If we add a species to the substitution table but forget the recipe,
        substitution returns the species but craft() can't produce it.
        """
        for planks_id in _PLANKS_LOG_BY_SPECIES:
            assert planks_id in CRAFTING_RECIPES, (
                f"{planks_id} in substitution table but has no recipe"
            )

    def test_each_planks_recipe_uses_matching_log(self):
        """Recipe for spruce_planks must consume spruce_log, not oak_log."""
        for planks_id, expected_log in _PLANKS_LOG_BY_SPECIES.items():
            recipe = CRAFTING_RECIPES[planks_id]
            assert len(recipe) == 1, f"{planks_id} recipe is multi-step"
            ing_id, _count = recipe[0]
            assert ing_id == expected_log, (
                f"{planks_id} recipe uses {ing_id}, expected {expected_log}"
            )


# ---------------------------------------------- pass-through behavior


class TestPassThrough:
    """Non-planks ingredients and degenerate counts skip substitution."""

    def test_non_planks_ingredient_returns_unchanged(self, monkeypatch):
        """Stick / cobblestone are not in the substitution table → no HTTP,
        return as-is. A regression here would slam homunculus with a fetch
        on every recipe step."""
        called = []

        def fake_get(*a, **kw):
            called.append(1)
            raise AssertionError("HTTP should not be called for non-planks")

        monkeypatch.setattr(tools.requests, "get", fake_get)
        assert _resolve_wood_substitute("minecraft:stick", 2) == "minecraft:stick"
        assert _resolve_wood_substitute("minecraft:cobblestone", 8) == "minecraft:cobblestone"
        assert _resolve_wood_substitute("minecraft:iron_ingot", 3) == "minecraft:iron_ingot"
        assert called == []


# ---------------------------------------------- self-sufficient species


class TestSelfSufficient:
    """When the requested species has enough material, return it unchanged."""

    def test_exact_planks_count_sufficient(self, fake_inventory):
        fake_inventory(_make_inv(("minecraft:oak_planks", 3)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:oak_planks"

    def test_one_log_produces_four_planks(self, fake_inventory):
        """1 log = 4 planks in MC. count<=4 is satisfied by 1 log."""
        fake_inventory(_make_inv(("minecraft:oak_log", 1)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 4) == "minecraft:oak_planks"

    def test_plank_log_combo_sums(self, fake_inventory):
        """2 planks + 1 log = 6 available; sufficient for count=6."""
        fake_inventory(_make_inv(
            ("minecraft:spruce_planks", 2),
            ("minecraft:spruce_log", 1),
        ))
        assert _resolve_wood_substitute("minecraft:spruce_planks", 6) == "minecraft:spruce_planks"

    def test_boundary_exact_threshold(self, fake_inventory):
        """count == available is sufficient (>=, not >)."""
        fake_inventory(_make_inv(("minecraft:birch_log", 1)))
        # 1 log → 4 planks; count=4 boundary
        assert _resolve_wood_substitute("minecraft:birch_planks", 4) == "minecraft:birch_planks"

    def test_count_zero_always_sufficient(self, fake_inventory):
        """count=0 → trivially sufficient, return requested species."""
        fake_inventory(_make_inv(("minecraft:dirt", 64)))  # nothing wood
        assert _resolve_wood_substitute("minecraft:oak_planks", 0) == "minecraft:oak_planks"


# ---------------------------------------------- substitution path


class TestSubstitution:
    """When requested species lacks material, return an alternate species
    that has enough. The original probe-validate-r2 T2 scenario lives here."""

    def test_probe_validate_r2_scenario(self, fake_inventory):
        """The original bug: 3× spruce_log + 0 oak, craft asks for oak_planks
        count=2. Must substitute to spruce_planks (the agent's only wood)."""
        fake_inventory(_make_inv(("minecraft:spruce_log", 3)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 2) == "minecraft:spruce_planks"

    def test_substitutes_when_only_logs_in_other_species(self, fake_inventory):
        """Requesting jungle_planks with only birch_log in inv → birch_planks."""
        fake_inventory(_make_inv(("minecraft:birch_log", 2)))
        assert _resolve_wood_substitute("minecraft:jungle_planks", 4) == "minecraft:birch_planks"

    def test_substitutes_when_only_planks_in_other_species(self, fake_inventory):
        """Same but the alternate is held as planks, not logs."""
        fake_inventory(_make_inv(("minecraft:cherry_planks", 6)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 5) == "minecraft:cherry_planks"

    def test_prefers_insertion_order_for_tiebreak(self, fake_inventory):
        """When multiple alternates have enough, the first one in dict-iteration
        order wins. Dict order is oak, spruce, birch, jungle, acacia, ...
        Request oak with both spruce and birch available → spruce (earlier).

        Pinning this means future changes to dict order are visible in tests."""
        fake_inventory(_make_inv(
            ("minecraft:spruce_log", 1),
            ("minecraft:birch_log", 1),
        ))
        # Both have 4 planks-worth; spruce comes first in _PLANKS_LOG_BY_SPECIES
        result = _resolve_wood_substitute("minecraft:oak_planks", 3)
        assert result == "minecraft:spruce_planks"

    def test_no_alternate_with_enough_returns_original(self, fake_inventory):
        """If no species has enough, fall back to original (caller will fail
        the craft loudly rather than silently swap to an insufficient species)."""
        fake_inventory(_make_inv(
            ("minecraft:spruce_planks", 1),  # need 5, has 1 — insufficient
            ("minecraft:birch_planks", 2),   # also insufficient
        ))
        assert _resolve_wood_substitute("minecraft:oak_planks", 10) == "minecraft:oak_planks"

    def test_mixed_species_dont_sum_across(self, fake_inventory):
        """1 spruce_planks + 1 birch_planks ≠ 2 of either. Substitution must
        NOT pool across species — each species is independent."""
        fake_inventory(_make_inv(
            ("minecraft:spruce_planks", 1),
            ("minecraft:birch_planks", 1),
            ("minecraft:jungle_planks", 1),
        ))
        # Need 3 planks. Cross-species sum = 3, but per-species each is 1.
        # No species individually satisfies → return original (caller fails).
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:oak_planks"


# ---------------------------------------------- HTTP failure handling


class TestHttpFailureFallthrough:
    """When homunculus is unreachable, _resolve_wood_substitute must fail
    soft (return ing_id unchanged) so the parent craft call gets a real
    error from homunculus rather than a confusing substitution failure."""

    def test_connection_error_returns_unchanged(self, fake_inventory):
        fake_inventory(None, raise_exc=requests.ConnectionError("homunculus down"))
        assert _resolve_wood_substitute("minecraft:oak_planks", 2) == "minecraft:oak_planks"

    def test_timeout_returns_unchanged(self, fake_inventory):
        fake_inventory(None, raise_exc=requests.Timeout("read timed out"))
        assert _resolve_wood_substitute("minecraft:spruce_planks", 4) == "minecraft:spruce_planks"

    def test_http_500_returns_unchanged(self, fake_inventory):
        """raise_for_status() throws HTTPError, caught by requests.RequestException."""
        fake_inventory({}, raise_exc=None)
        # Override to return a 500 status (not raised at get() time, raised at raise_for_status())
        def fake_get_500(url, timeout=None):
            return _FakeResp({}, status=500)
        import unittest.mock
        with unittest.mock.patch.object(tools.requests, "get", fake_get_500):
            assert _resolve_wood_substitute("minecraft:birch_planks", 4) == "minecraft:birch_planks"

    def test_invalid_json_returns_unchanged(self, fake_inventory):
        """JSON decode error → ValueError → caught → return original."""
        fake_inventory(ValueError("invalid json"))
        assert _resolve_wood_substitute("minecraft:cherry_planks", 2) == "minecraft:cherry_planks"


# ---------------------------------------------- offhand inclusion


class TestOffhandCounted:
    """Offhand slot counts toward available material. If the agent is
    holding a stack of planks in offhand and we ignore it, we'd substitute
    away from a species the agent already has."""

    def test_offhand_only_counts(self, fake_inventory):
        """Material held only in offhand still satisfies."""
        fake_inventory({
            "main": [],
            "offhand": {"id": "minecraft:spruce_log", "count": 2},
        })
        # 2 logs = 8 planks-worth, sufficient for count=4
        assert _resolve_wood_substitute("minecraft:spruce_planks", 4) == "minecraft:spruce_planks"

    def test_offhand_and_main_sum(self, fake_inventory):
        """main + offhand of same species are pooled."""
        fake_inventory({
            "main": [{"id": "minecraft:oak_planks", "count": 2, "slot": 0}],
            "offhand": {"id": "minecraft:oak_log", "count": 1},
        })
        # 2 planks + 4 (from 1 log) = 6 available
        assert _resolve_wood_substitute("minecraft:oak_planks", 6) == "minecraft:oak_planks"

    def test_offhand_can_drive_substitution(self, fake_inventory):
        """Offhand-held alternate species triggers substitution."""
        fake_inventory({
            "main": [],
            "offhand": {"id": "minecraft:jungle_log", "count": 1},  # 4 planks-worth
        })
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:jungle_planks"

    def test_offhand_none_safe(self, fake_inventory):
        """offhand: None must not crash (common case for fresh spawn)."""
        fake_inventory({
            "main": [{"id": "minecraft:oak_log", "count": 1, "slot": 0}],
            "offhand": None,
        })
        assert _resolve_wood_substitute("minecraft:oak_planks", 4) == "minecraft:oak_planks"


# ---------------------------------------------- duplicate-slot summing


class TestDuplicateSlotsSummed:
    """Multiple slots holding the same item should sum, not overwrite."""

    def test_planks_in_two_slots_sum(self, fake_inventory):
        fake_inventory(_make_inv(
            ("minecraft:oak_planks", 32),
            ("minecraft:oak_planks", 16),
        ))
        # 48 planks; sufficient for count=40
        assert _resolve_wood_substitute("minecraft:oak_planks", 40) == "minecraft:oak_planks"

    def test_logs_in_two_slots_sum(self, fake_inventory):
        """Common case: agent stacks logs across slots after partial mines."""
        fake_inventory(_make_inv(
            ("minecraft:spruce_log", 1),
            ("minecraft:spruce_log", 1),
        ))
        # 2 logs = 8 planks-worth
        assert _resolve_wood_substitute("minecraft:spruce_planks", 8) == "minecraft:spruce_planks"
