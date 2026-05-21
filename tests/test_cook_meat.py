"""Tests for the cook_meat tool — composite smelt + collect chain.

Substrate finding behind this primitive (2026-05-21 fan-out): Haiku
recognized hunt_passive when food was at 0 and called it 4×, but never
reached for the smelt+collect chain when given raw_beef + furnace + coal
in cook_kitchen loadout. 0/5 smelt attempts across the fan-out. The
chain isn't visible as a verb. cook_meat wraps it as one.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from craft import tools


class TestSchemaShape:
    def test_cook_meat_in_tools_list(self):
        names = [t["function"]["name"] for t in tools.TOOLS]
        assert "cook_meat" in names

    def test_cook_meat_optional_args(self):
        spec = next(
            t for t in tools.TOOLS if t["function"]["name"] == "cook_meat"
        )
        params = spec["function"]["parameters"]
        assert params["required"] == [], (
            "cook_meat should accept zero args — default behavior is auto-pick"
        )
        assert "meat" in params["properties"]
        assert "count" in params["properties"]

    def test_dispatch_wired(self):
        assert tools.HANDLERS.get("cook_meat") is tools.handle_cook_meat


class TestPickRawMeat:
    def test_picks_first_priority_in_inventory(self):
        # Has beef AND porkchop — beef wins (first in _COOKABLE_MEATS order)
        with patch.object(
            tools, "_inventory_counts",
            return_value={"minecraft:porkchop": 4, "minecraft:beef": 8},
        ):
            result = tools._pick_raw_meat()
        assert result == ("minecraft:beef", 8)

    def test_picks_mutton_when_no_beef_porkchop(self):
        with patch.object(
            tools, "_inventory_counts",
            return_value={"minecraft:mutton": 3},
        ):
            result = tools._pick_raw_meat()
        assert result == ("minecraft:mutton", 3)

    def test_none_when_no_raw_meat(self):
        # cooked_beef in inv must NOT count as cookable
        with patch.object(
            tools, "_inventory_counts",
            return_value={"minecraft:cobblestone": 32, "minecraft:cooked_beef": 4},
        ):
            assert tools._pick_raw_meat() is None


class TestCompositeFlow:
    def _run(
        self,
        *,
        args=None,
        inv_meat_id="minecraft:beef",
        inv_meat_count=8,
        inv_fuel={"minecraft:coal": 4},
        smelt_result=None,
        collect_result=None,
        cooked_count=8,
        smelt_status_smelts=None,
    ):
        # Build a single inventory snapshot consumed by every
        # _inventory_counts() call in the handler. cooked_count is what
        # the post-collect call sees in `cooked_<meat>`.
        snapshot: dict[str, int] = {}
        if inv_meat_count:
            snapshot[inv_meat_id] = inv_meat_count
        snapshot.update(inv_fuel)
        cooked_id = inv_meat_id.replace(
            "minecraft:", "minecraft:cooked_", 1,
        )
        snapshot[cooked_id] = cooked_count

        # Default eta=0 so the polling loop doesn't run any real seconds
        # (we mock time.sleep too, but the deadline math still uses
        # time.time which actually advances).
        smelt_result = smelt_result or {"success": True, "eta_seconds": 0}
        collect_result = collect_result or {"success": True}
        # By default the first /smelt_status poll returns a ready smelt
        # so the loop exits on first iteration.
        if smelt_status_smelts is None:
            smelt_status_smelts = [{"status": "ready"}]
        status_response = {"smelts": smelt_status_smelts}

        def fake_get(path, **_kw):
            if path == "/smelt_status":
                return status_response
            return {}

        with patch.object(tools, "_inventory_counts", return_value=snapshot), \
             patch.object(tools, "_smelt_raw", return_value=smelt_result), \
             patch.object(tools, "_collect_smelt_raw", return_value=collect_result), \
             patch.object(tools, "_get_homunculus", side_effect=fake_get), \
             patch("time.sleep", lambda *_a, **_k: None):
            return tools.handle_cook_meat(args or {})

    def test_happy_path_no_args(self):
        out = self._run()
        assert "cooked" in out
        assert "cooked_beef" in out
        assert "FAILED" not in out

    def test_happy_path_with_meat_arg(self):
        out = self._run(
            args={"meat": "porkchop"},
            inv_meat_id="minecraft:porkchop",
            inv_meat_count=4,
        )
        assert "cooked_porkchop" in out
        assert "FAILED" not in out

    def test_meat_arg_with_minecraft_prefix(self):
        out = self._run(
            args={"meat": "minecraft:mutton"},
            inv_meat_id="minecraft:mutton",
        )
        assert "cooked_mutton" in out

    def test_unknown_meat_rejected(self):
        out = self._run(args={"meat": "diamond"})
        assert out.startswith("FAILED")
        assert "not a cookable" in out

    def test_no_raw_meat_in_inventory(self):
        out = self._run(inv_meat_count=0)
        assert out.startswith("FAILED")
        assert "no raw meat" in out

    def test_no_fuel(self):
        out = self._run(inv_fuel={})
        assert out.startswith("FAILED")
        assert "no_fuel" in out

    def test_smelt_failure_bubbles_up(self):
        out = self._run(
            smelt_result={
                "success": False,
                "reason": "no_cobblestone",
                "message": "need 8 cobble to craft furnace",
            },
        )
        assert out.startswith("FAILED")
        assert "smelt didn't start" in out
        assert "no_cobblestone" in out

    def test_collect_failure_bubbles_up(self):
        out = self._run(
            collect_result={
                "success": False,
                "reason": "furnace_not_found",
                "message": "no active smelt to collect",
            },
        )
        assert out.startswith("FAILED")
        assert "collect failed" in out
        assert "furnace_not_found" in out

    def test_count_arg_caps_smelt(self):
        captured: list[tuple[str, int]] = []

        def fake_smelt(meat, count, fuel=None):
            captured.append((meat, count))
            return {"success": True, "eta_seconds": 0}

        snapshot = {"minecraft:beef": 20, "minecraft:coal": 4}

        def fake_get(path, **_kw):
            return {"smelts": [{"status": "ready"}]} if path == "/smelt_status" else {}

        with patch.object(tools, "_inventory_counts", return_value=snapshot), \
             patch.object(tools, "_smelt_raw", side_effect=fake_smelt), \
             patch.object(tools, "_collect_smelt_raw", return_value={"success": True}), \
             patch.object(tools, "_get_homunculus", side_effect=fake_get), \
             patch("time.sleep", lambda *_a, **_k: None):
            tools.handle_cook_meat({"count": 4})
        assert captured == [("minecraft:beef", 4)]

    def test_count_capped_at_per_call_max(self):
        """When inv has 20x beef and no count arg, cap at _COOK_PER_CALL_CAP."""
        captured: list[tuple[str, int]] = []

        def fake_smelt(meat, count, fuel=None):
            captured.append((meat, count))
            return {"success": True, "eta_seconds": 0}

        snapshot = {"minecraft:beef": 20, "minecraft:coal": 4}

        def fake_get(path, **_kw):
            return {"smelts": [{"status": "ready"}]} if path == "/smelt_status" else {}

        with patch.object(tools, "_inventory_counts", return_value=snapshot), \
             patch.object(tools, "_smelt_raw", side_effect=fake_smelt), \
             patch.object(tools, "_collect_smelt_raw", return_value={"success": True}), \
             patch.object(tools, "_get_homunculus", side_effect=fake_get), \
             patch("time.sleep", lambda *_a, **_k: None):
            tools.handle_cook_meat({})
        assert captured == [("minecraft:beef", tools._COOK_PER_CALL_CAP)]
