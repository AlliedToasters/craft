"""Unit tests for the mine tier-gate (Fix C, 2026-06-01).

Context: the goal=diamond wave exposed qwen calling mine_diamond 26× across 8
rollouts that never reached iron tier — every call doomed (diamond ore needs an
iron+ pickaxe to drop). The tier-gate refuses + redirects at the tool boundary
when the required pickaxe is absent, instead of dispatching a doomed mine.

These patch the inventory read + the underlying delta handler to assert the
gate decision (block vs proceed) by pickaxe tier, env kill-switch, and the
fail-open-on-read-blip contract.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from craft import tools


def _inv_counter(fake_inventory: dict[str, int]):
    """Return a stand-in for _count_inventory_items that sums the requested ids
    against `fake_inventory` (matches the real summing semantics)."""
    def _count(item_ids):
        return sum(c for i, c in fake_inventory.items() if i in item_ids)
    return _count


def _run(handler, fake_inventory, env=None):
    """Run a mine handler with a faked inventory; return (result, dispatched)."""
    dispatched = {"called": False}

    def _stub_delta(*a, **k):
        dispatched["called"] = True
        return "acquired 1 more (now have 1 drops; last type mined: test)"

    envd = {"CRAFT_MINE_TIER_GATE": "1"}
    if env:
        envd.update(env)
    with patch.dict(os.environ, envd), \
            patch.object(tools, "_count_inventory_items", _inv_counter(fake_inventory)), \
            patch.object(tools, "_handle_mine_delta", _stub_delta):
        result = handler({"quantity": 1})
    return result, dispatched["called"]


def test_iron_blocked_without_stone_pickaxe():
    result, dispatched = _run(tools.handle_mine_iron, {"minecraft:wooden_pickaxe": 1})
    assert result.startswith("SKIPPED mine_iron")
    assert "STONE-tier" in result
    assert not dispatched


def test_iron_proceeds_with_stone_pickaxe():
    result, dispatched = _run(tools.handle_mine_iron, {"minecraft:stone_pickaxe": 1})
    assert dispatched
    assert not result.startswith("SKIPPED")


def test_iron_proceeds_with_iron_pickaxe():
    result, dispatched = _run(tools.handle_mine_iron, {"minecraft:iron_pickaxe": 1})
    assert dispatched


def test_diamond_blocked_without_iron_pickaxe():
    # A stone pickaxe satisfies iron but NOT diamond.
    result, dispatched = _run(tools.handle_mine_diamond, {"minecraft:stone_pickaxe": 1})
    assert result.startswith("SKIPPED mine_diamond")
    assert "IRON-tier" in result
    assert not dispatched


def test_diamond_proceeds_with_iron_pickaxe():
    result, dispatched = _run(tools.handle_mine_diamond, {"minecraft:iron_pickaxe": 1})
    assert dispatched


def test_diamond_proceeds_with_netherite_pickaxe():
    result, dispatched = _run(tools.handle_mine_diamond, {"minecraft:netherite_pickaxe": 1})
    assert dispatched


def test_stone_blocked_without_any_pickaxe():
    # Issue #11: barehanded stone breaks for nothing — gate it before dispatch.
    result, dispatched = _run(tools.handle_mine_stone, {})
    assert result.startswith("SKIPPED mine_stone")
    assert "WOODEN" in result
    assert not dispatched


def test_stone_proceeds_with_wooden_pickaxe():
    result, dispatched = _run(tools.handle_mine_stone, {"minecraft:wooden_pickaxe": 1})
    assert dispatched
    assert not result.startswith("SKIPPED")


def test_stone_proceeds_with_iron_pickaxe():
    # Any pickaxe >= wood satisfies stone.
    result, dispatched = _run(tools.handle_mine_stone, {"minecraft:iron_pickaxe": 1})
    assert dispatched


def test_stone_blocked_with_only_a_shovel():
    # A non-pickaxe tool is no better than barehanded for stone drops.
    result, dispatched = _run(tools.handle_mine_stone, {"minecraft:iron_shovel": 1})
    assert result.startswith("SKIPPED mine_stone")
    assert not dispatched


def test_env_kill_switch_disables_gate():
    # No pickaxe at all, but the gate is off → proceed (legacy behavior).
    result, dispatched = _run(tools.handle_mine_diamond, {}, env={"CRAFT_MINE_TIER_GATE": "0"})
    assert dispatched
    assert not result.startswith("SKIPPED")


def test_fail_open_on_inventory_read_blip():
    # _count_inventory_items returns None (transport blip) → must NOT block.
    def _stub_delta(*a, **k):
        return "acquired 1 more"
    with patch.dict(os.environ, {"CRAFT_MINE_TIER_GATE": "1"}), \
            patch.object(tools, "_count_inventory_items", lambda ids: None), \
            patch.object(tools, "_handle_mine_delta", _stub_delta) as m:
        result = tools.handle_mine_diamond({"quantity": 1})
    assert not result.startswith("SKIPPED")
