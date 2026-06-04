"""Unit tests for the bamboo wood tech tree groundwork (issue #4).

Bamboo is the one wood species with no `*_log`. Issue #4 imagined three Python
pieces, but live investigation (2026-06-04) found the biome can't be unbanned
yet — two blockers sit OUTSIDE the craft layer:

  - SUBSTRATE: Baritone can't harvest bamboo cane (/baritone/mine deactivates
    with 0 collected; /excavate breaks it without picking up drops). So bamboo
    is NOT a mine_wood candidate (see section 1) — pending a harvest primitive.
  - VANILLA: bamboo_block is a 3×3 recipe → needs a crafting table, but the
    first table needs planks ← bamboo_block. A pure-bamboo spawn can't bootstrap
    its first table; real play uses the jungle's sparse trees for it.

What DOES work and is pinned here is the CRAFT-layer groundwork — correct once a
table exists, so bamboo is a usable SUPPLEMENTARY wood source:

  1. bamboo is intentionally not yet a mine candidate (Baritone gap).
  2. Recipe recursion knows the two-step path — bamboo_planks → bamboo_block →
     bamboo (a mined leaf) live in CRAFTING_RECIPES.
  3. Substitution points the parent craft at bamboo_planks — `_planks_available`
     models 9 bamboo → 1 block → 2 planks, and `_resolve_wood_substitute`
     chooses bamboo when it's the only wood held, without disturbing the
     standard-9 insertion-order tiebreak.
  4. AutoDrop keeps bamboo / bamboo_block / bamboo_planks (autodrop.py), so a
     bamboo haul isn't thrown away mid-tech-tree.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from craft import mine, tools
from craft.tools import (
    CRAFTING_RECIPES,
    _SUBSTITUTE_PLANKS,
    _WOOD_SUBSTITUTE_ITEMS,
    _planks_available,
    _resolve_wood_substitute,
)


# ---------------------------------------------- 1. mine_wood does NOT mine bamboo (yet)
# Baritone cannot harvest bamboo cane (/baritone/mine deactivates with 0
# collected; /excavate breaks it without collecting drops — confirmed live
# 2026-06-04). So bamboo is intentionally NOT a mine_wood candidate: if it were
# the nearest candidate, the cycle would hard-stop on "interrupted" and never
# fall through to the bamboo_jungle's sparse oak/jungle logs (which Baritone
# DOES mine). Re-add only once a working bamboo-harvest substrate exists. These
# tests pin the decision so a premature re-add trips loud.


def test_bamboo_not_a_mine_candidate():
    assert "bamboo" not in mine.LOG_TYPES


def test_bamboo_not_counted_by_mine_wood():
    assert "minecraft:bamboo" not in tools.LOG_DROPS


# ---------------------------------------------- 2. recipe recursion knows bamboo


def test_bamboo_planks_recipe():
    assert CRAFTING_RECIPES["minecraft:bamboo_planks"] == [("minecraft:bamboo_block", 1)]


def test_bamboo_block_recipe():
    assert CRAFTING_RECIPES["minecraft:bamboo_block"] == [("minecraft:bamboo", 9)]


def test_bamboo_cane_is_a_leaf():
    # The cane is mined, never crafted — _craft_recursive must bottom out on it
    # with a "must be acquired (mining)" message rather than looping.
    assert "minecraft:bamboo" not in CRAFTING_RECIPES


# ---------------------------------------------- 4. AutoDrop keeps bamboo


def test_autodrop_keeps_bamboo_forms():
    # AutoDrop's whitelist policy must keep all three bamboo forms, else a
    # bamboo haul is tick-dropped before it can be crafted (observed live).
    from craft.autodrop import ALWAYS_KEEP
    for item in ("minecraft:bamboo", "minecraft:bamboo_block", "minecraft:bamboo_planks"):
        assert item in ALWAYS_KEEP, f"{item} not kept by AutoDrop"


# ---------------------------------------------- 5. harvest_bamboo + mine_wood routing
# Baritone can't mine bamboo, so the cane is harvested through the homunculus
# /harvest_bamboo base-break primitive, and mine_wood routes there ONLY as a
# fallback when no logs are reachable (logs stay primary + bootstrap the table).


class _PostResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_bamboo_drops_set():
    assert mine.BAMBOO_DROPS == {"minecraft:bamboo"}


def test_harvest_bamboo_returns_bamboo_on_break(monkeypatch):
    monkeypatch.setattr(mine, "_reposition_to_bamboo", lambda: False)  # one round
    monkeypatch.setattr(mine.requests, "post",
                        lambda *a, **k: _PostResp({"success": True, "columns_broken": 3,
                                                   "bases_in_reach": 5}))
    assert mine.harvest_bamboo(10) == "bamboo"


def test_harvest_bamboo_none_when_no_columns(monkeypatch):
    monkeypatch.setattr(mine, "_reposition_to_bamboo", lambda: False)
    monkeypatch.setattr(mine.requests, "post",
                        lambda *a, **k: _PostResp({"success": False, "reason": "no_bamboo"}))
    assert mine.harvest_bamboo(10) is None


def test_mine_wood_routes_to_bamboo_when_no_logs():
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        if drops == tools.LOG_DROPS:
            return "FAILED: no candidate reachable for mine_wood (acquired 0)"
        if drops == tools.BAMBOO_DROPS:
            return "acquired 7 more (now have 7 mine_wood-drops; last type mined: bamboo)"
        raise AssertionError(f"unexpected drops {drops}")

    with patch.object(tools, "_handle_mine_delta", _stub):
        out = tools.handle_mine_wood({"quantity": 5})
    assert out.startswith("[wood_source=bamboo]")
    assert tools.BAMBOO_DROPS in seen


def test_mine_wood_logs_success_skips_bamboo():
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        return "acquired 5 more (now have 5 mine_wood-drops; last type mined: oak_log)"

    with patch.object(tools, "_handle_mine_delta", _stub):
        out = tools.handle_mine_wood({"quantity": 5})
    assert out.startswith("acquired 5 more")
    assert tools.BAMBOO_DROPS not in seen


# ---------------------------------------------- 3a. _planks_available math


class TestPlanksAvailableBamboo:
    P = "minecraft:bamboo_planks"

    def test_nine_bamboo_makes_two_planks(self):
        assert _planks_available(self.P, {"minecraft:bamboo": 9}) == 2

    def test_floor_division_below_a_block(self):
        # 8 < 9 bamboo can't make a block, so zero planks-worth.
        assert _planks_available(self.P, {"minecraft:bamboo": 8}) == 0

    def test_eighteen_bamboo_makes_four_planks(self):
        assert _planks_available(self.P, {"minecraft:bamboo": 18}) == 4

    def test_blocks_count_two_each(self):
        assert _planks_available(self.P, {"minecraft:bamboo_block": 3}) == 6

    def test_planks_count_themselves(self):
        assert _planks_available(self.P, {"minecraft:bamboo_planks": 5}) == 5

    def test_all_three_forms_sum(self):
        # 9 bamboo (2) + 1 block (2) + 1 plank (1) = 5
        counts = {
            "minecraft:bamboo": 9,
            "minecraft:bamboo_block": 1,
            "minecraft:bamboo_planks": 1,
        }
        assert _planks_available(self.P, counts) == 5

    def test_empty_is_zero(self):
        assert _planks_available(self.P, {}) == 0

    def test_standard_species_unaffected(self):
        # 1 log → 4 planks; the linear branch still holds for non-bamboo.
        assert _planks_available("minecraft:oak_planks", {"minecraft:oak_log": 2}) == 8


# ---------------------------------------------- 3b. substitution tables


def test_bamboo_in_substitute_set_and_last():
    assert "minecraft:bamboo_planks" in _SUBSTITUTE_PLANKS
    assert _SUBSTITUTE_PLANKS[-1] == "minecraft:bamboo_planks"


def test_bamboo_items_of_interest():
    for item in ("minecraft:bamboo", "minecraft:bamboo_block", "minecraft:bamboo_planks"):
        assert item in _WOOD_SUBSTITUTE_ITEMS


# ---------------------------------------------- 3c. _resolve_wood_substitute live path


class _FakeResp:
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
    main = [{"id": i, "count": c, "slot": n} for n, (i, c) in enumerate(items)]
    return {"main": main, "offhand": offhand}


@pytest.fixture
def fake_inventory(monkeypatch):
    def _install(inv_data):
        def fake_get(url, timeout=None):
            return _FakeResp(inv_data)
        monkeypatch.setattr(tools.requests, "get", fake_get)
    return _install


class TestBambooSubstitution:
    def test_only_bamboo_substitutes_bamboo_planks(self, fake_inventory):
        # bamboo_jungle spawn: 27 bamboo, 0 logs, craft asks for oak_planks(3).
        # 27 // 9 = 3 blocks → 6 planks ≥ 3 → substitute bamboo_planks.
        fake_inventory(_make_inv(("minecraft:bamboo", 27)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:bamboo_planks"

    def test_bamboo_block_drives_substitution(self, fake_inventory):
        fake_inventory(_make_inv(("minecraft:bamboo_block", 2)))  # 4 planks-worth
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:bamboo_planks"

    def test_held_bamboo_planks_drive_substitution(self, fake_inventory):
        fake_inventory(_make_inv(("minecraft:bamboo_planks", 6)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 5) == "minecraft:bamboo_planks"

    def test_insufficient_bamboo_no_substitute(self, fake_inventory):
        # 8 bamboo < one block → 0 planks-worth → fall back to the original
        # (caller fails the craft loudly rather than swap to nothing).
        fake_inventory(_make_inv(("minecraft:bamboo", 8)))
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:oak_planks"

    def test_standard_species_wins_tiebreak_over_bamboo(self, fake_inventory):
        # spruce_log (4 planks) AND 27 bamboo both suffice; bamboo is appended
        # last, so the standard species keeps the insertion-order tiebreak.
        fake_inventory(_make_inv(
            ("minecraft:spruce_log", 1),
            ("minecraft:bamboo", 27),
        ))
        assert _resolve_wood_substitute("minecraft:oak_planks", 3) == "minecraft:spruce_planks"

    def test_bamboo_planks_request_self_sufficient(self, fake_inventory):
        # A recipe asking directly for bamboo_planks with enough bamboo in hand
        # returns bamboo_planks unchanged (no needless swap).
        fake_inventory(_make_inv(("minecraft:bamboo", 9)))
        assert _resolve_wood_substitute("minecraft:bamboo_planks", 2) == "minecraft:bamboo_planks"

    def test_bamboo_offhand_counted(self, fake_inventory):
        fake_inventory({"main": [], "offhand": {"id": "minecraft:bamboo", "count": 18}})
        assert _resolve_wood_substitute("minecraft:oak_planks", 4) == "minecraft:bamboo_planks"


# ---------------------------------------------- locate_biome parsing


from craft import spawn  # noqa: E402


class _LogResp:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def json(self):
        return {"lines": self._lines}


@pytest.fixture
def fake_relay(monkeypatch):
    """Stub the relay: POST /cmd → {"ok": True}, GET /log → given lines."""
    def _install(log_lines, *, post_ok=True, get_exc=None):
        def fake_post(url, json=None, timeout=None):
            return _FakeResp({"ok": post_ok})

        def fake_get(url, params=None, timeout=None):
            if get_exc is not None:
                raise get_exc
            return _LogResp(log_lines)

        monkeypatch.setattr(spawn.requests, "post", fake_post)
        monkeypatch.setattr(spawn.requests, "get", fake_get)
        monkeypatch.setattr(spawn.time, "sleep", lambda *_a, **_k: None)
    return _install


_NEAREST = ("[01:08:23] [Server thread/INFO]: The nearest minecraft:bamboo_jungle "
            "is at [-736, 64, -1408] (1588 blocks away)")


class TestLocateBiome:
    def test_parses_coords(self, fake_relay):
        fake_relay([_NEAREST])
        assert spawn.locate_biome(
            "bamboo_jungle", server_cmd_base="http://x") == (-736, -1408)

    def test_namespaced_input_ok(self, fake_relay):
        fake_relay([_NEAREST])
        assert spawn.locate_biome(
            "minecraft:bamboo_jungle", server_cmd_base="http://x") == (-736, -1408)

    def test_tilde_y_is_tolerated(self, fake_relay):
        fake_relay(["The nearest minecraft:bamboo_jungle is at [10, ~, -20] (5 blocks away)"])
        assert spawn.locate_biome(
            "bamboo_jungle", server_cmd_base="http://x") == (10, -20)

    def test_takes_last_match(self, fake_relay):
        fake_relay([
            "The nearest minecraft:bamboo_jungle is at [1, 64, 2] (9 blocks away)",
            "The nearest minecraft:bamboo_jungle is at [-736, 64, -1408] (1588 blocks away)",
        ])
        assert spawn.locate_biome(
            "bamboo_jungle", server_cmd_base="http://x") == (-736, -1408)

    def test_wrong_biome_line_ignored(self, fake_relay):
        fake_relay(["The nearest minecraft:jungle is at [5, 64, 5] (1 blocks away)"])
        assert spawn.locate_biome("bamboo_jungle", server_cmd_base="http://x") is None

    def test_not_found_returns_none(self, fake_relay):
        fake_relay(["Can't find element 'minecraft:bamboo_jungle' of type ..."])
        assert spawn.locate_biome("bamboo_jungle", server_cmd_base="http://x") is None

    def test_log_transport_error_returns_none(self, fake_relay):
        fake_relay([], get_exc=requests.ConnectionError("relay down"))
        assert spawn.locate_biome("bamboo_jungle", server_cmd_base="http://x") is None
