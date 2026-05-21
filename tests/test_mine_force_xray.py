"""Regression test for the CRAFT_MINE_FORCE_XRAY substrate override.

Context: 2026-05-20 iron-loadout campaign — 5/5 qwen agents chose
`fair=true` (blind 1×2 tunneling) for mine_diamond, broke their iron
pickaxes well before reaching diamond, and died trying to recover.
The substrate fix is `CRAFT_MINE_FORCE_XRAY=1`: forces fair=False at the
dispatch layer for all ore mining handlers except mine_stone (which
forces fair=True tool-side for its own reasons).

These tests pin the contract by patching the miner closures and asserting
which branch (baritone vs fair_miner) runs given env var + agent arg.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from craft import tools


def _stub_inventory(mock_count: int = 0):
    """Patch _count_inventory_items to a fixed return so the handler runs
    without touching homunculus."""
    return patch.object(tools, "_count_inventory_items", return_value=mock_count)


class TestForceXrayOverride:
    def _run_with(self, *, agent_fair: bool, env_value: str | None, label: str):
        """Run _handle_mine_delta with stubs and report which branch fired.

        Returns ("baritone", target) or ("fair", delta).
        """
        calls: list = []

        def baritone_miner(target):
            calls.append(("baritone", target))
            return "ok"

        def fair_miner(delta):
            calls.append(("fair", delta))
            return "ok"

        args = {"quantity": 4, "fair": agent_fair}
        env = dict(os.environ)
        if env_value is None:
            env.pop("CRAFT_MINE_FORCE_XRAY", None)
        else:
            env["CRAFT_MINE_FORCE_XRAY"] = env_value
        with patch.dict(os.environ, env, clear=True), _stub_inventory():
            tools._handle_mine_delta(
                label, args, drops={"minecraft:diamond"},
                miner=baritone_miner, fair_miner=fair_miner,
            )
        assert len(calls) == 1, f"expected exactly one miner invocation, got {calls}"
        return calls[0]

    def test_unset_honors_agent_fair_false(self):
        kind, _ = self._run_with(agent_fair=False, env_value=None, label="mine_diamond")
        assert kind == "baritone"

    def test_unset_honors_agent_fair_true(self):
        kind, _ = self._run_with(agent_fair=True, env_value=None, label="mine_diamond")
        assert kind == "fair"

    def test_override_clobbers_fair_true(self):
        """The substrate bug we're fixing: agent picks fair=True, env says no."""
        kind, _ = self._run_with(agent_fair=True, env_value="1", label="mine_diamond")
        assert kind == "baritone", "CRAFT_MINE_FORCE_XRAY=1 must override fair=True"

    def test_override_with_fair_false_is_noop(self):
        kind, _ = self._run_with(agent_fair=False, env_value="1", label="mine_diamond")
        assert kind == "baritone"

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_values_trigger(self, truthy):
        kind, _ = self._run_with(agent_fair=True, env_value=truthy, label="mine_iron")
        assert kind == "baritone"

    @pytest.mark.parametrize("falsy", ["0", "false", "no", ""])
    def test_falsy_values_do_not_trigger(self, falsy):
        kind, _ = self._run_with(agent_fair=True, env_value=falsy, label="mine_iron")
        assert kind == "fair", f"value {falsy!r} should not enable override"

    def test_mine_stone_exempt(self):
        """mine_stone forces fair=True tool-side to avoid baritone's
        deep-target pathology. The env override MUST NOT undo that — if it
        does, mine_stone will start picking pathological deep stone targets
        again (the bug fixed 2026-05-15)."""
        kind, _ = self._run_with(agent_fair=True, env_value="1", label="mine_stone")
        assert kind == "fair", "mine_stone must remain fair even with override on"


class TestMineWoodAlwaysXray:
    """mine_wood unconditionally forces fair=False (baritone x-ray).
    Reasoning: trees are trivially visible to a human player; modeling
    wood-getting as blind tunneling is a worse substrate than vision.
    Decision 2026-05-20 after observing 13% historical fair=true rate.
    """

    def test_schema_has_no_fair_field(self):
        """mine_wood's tool schema must not expose `fair` to the agent."""
        from craft.tools import TOOLS

        wood = next(
            t for t in TOOLS
            if t.get("function", {}).get("name") == "mine_wood"
        )
        props = wood["function"]["parameters"]["properties"]
        assert "fair" not in props, (
            "mine_wood should not expose a `fair` knob — wood is x-ray-only "
            "by substrate decision (see handle_mine_wood comment)."
        )

    def test_handle_mine_wood_clobbers_fair_true(self):
        """If something injects fair=True into mine_wood args (legacy
        replay, manual call), the handler MUST override it back to False.
        """
        from craft import tools as _tools

        calls: list = []

        def baritone_miner(target):
            calls.append(("baritone", target))
            return "ok"

        def fair_miner(delta):
            calls.append(("fair", delta))
            return "ok"

        # We patch the module's miner funcs the handler dispatches to.
        with patch.object(_tools, "mine_any_log", baritone_miner), \
             patch.object(_tools, "tunnel_for_logs", fair_miner), \
             _stub_inventory():
            _tools.handle_mine_wood({"quantity": 4, "fair": True})

        assert calls == [("baritone", 4)], (
            f"mine_wood must always route to baritone (x-ray) regardless of "
            f"args; got {calls}"
        )
