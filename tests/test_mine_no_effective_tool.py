"""Test the `no_effective_tool` surfacing path (issue #11).

The homunculus MineHandler fails a mine fast with reason `no_effective_tool`
when the target block requires a correct tool for drops but the inventory has
none — either it started toolless or the pickaxe snapped mid-mine. The python
side must:

  1. `_mine_first_reachable` records the hard-stop detail in `mine.last_stop`
     (it is NOT a skip reason — every same-resource candidate fails identically,
     so the whole cycle stops).
  2. `_handle_mine_delta` (baritone path only) surfaces that message to the agent
     as a clear "craft/upgrade a pickaxe" FAILED, not a misleading "no candidate
     reachable". The fair tunnel path must NOT consult `last_stop` (it never sets
     it, so a stale value must not leak through).
"""

from __future__ import annotations

from unittest.mock import patch

from craft import mine, tools


# ---------------------------------------------- _mine_first_reachable records the stop


class TestLastStopRecording:
    def test_no_effective_tool_recorded_and_stops_cycle(self):
        resp = {
            "success": False,
            "reason": "no_effective_tool",
            "block": "minecraft:iron_ore",
            "message": "no correct tool in inventory for minecraft:iron_ore",
        }
        with patch.object(mine, "_scan_nearest", lambda *a, **k: None), \
                patch.object(mine, "_mine_one", lambda *a, **k: resp):
            out = mine._mine_first_reachable(1, ["iron_ore", "deepslate_iron_ore"])
        assert out is None
        assert mine.last_stop is not None
        assert mine.last_stop["reason"] == "no_effective_tool"
        assert "no correct tool" in mine.last_stop["message"]

    def test_first_candidate_stops_cycle_no_second_attempt(self):
        """A tool failure is identical for every same-resource candidate, so the
        cycle must stop on the first — not waste a call on the next variant."""
        calls = {"n": 0}

        def _one(*a, **k):
            calls["n"] += 1
            return {"success": False, "reason": "no_effective_tool", "message": "m"}

        with patch.object(mine, "_scan_nearest", lambda *a, **k: None), \
                patch.object(mine, "_mine_one", _one):
            mine._mine_first_reachable(1, ["iron_ore", "deepslate_iron_ore"])
        assert calls["n"] == 1

    def test_last_stop_reset_at_cycle_start(self):
        """A fresh cycle that succeeds must clear a stale last_stop from before."""
        mine.last_stop = {"reason": "no_effective_tool", "message": "stale"}
        ok = {"success": True, "reason": "have_target", "message": "done"}
        with patch.object(mine, "_scan_nearest", lambda *a, **k: None), \
                patch.object(mine, "_mine_one", lambda *a, **k: ok):
            out = mine._mine_first_reachable(1, ["iron_ore"])
        assert out == "iron_ore"
        assert mine.last_stop is None

    def test_skip_reason_does_not_record(self):
        """`unreachable` is a skip reason (try next), not a hard stop — it must
        not populate last_stop."""
        mine.last_stop = None
        resp = {"success": False, "reason": "unreachable", "message": "x"}
        with patch.object(mine, "_scan_nearest", lambda *a, **k: None), \
                patch.object(mine, "_mine_one", lambda *a, **k: resp):
            out = mine._mine_first_reachable(1, ["iron_ore"])
        assert out is None
        assert mine.last_stop is None


# ---------------------------------------------- _handle_mine_delta surfacing


def _delta(label, *, fair, before, after, miner_result, last_stop):
    """Drive _handle_mine_delta with stubbed inventory counts + a stub miner."""
    mine.last_stop = last_stop
    counts = iter([before, after])

    def _count(_ids):
        return next(counts)

    def _miner(_target):
        return miner_result

    def _fair_miner(_delta):
        return miner_result

    args = {"quantity": 4}
    # mine_stone forces fair=True via its handler; here we exercise _handle_mine_delta
    # directly with an explicit fair flag in args.
    if fair:
        args["fair"] = True
    with patch.object(tools, "_count_inventory_items", _count):
        return tools._handle_mine_delta(
            label, args, {"minecraft:raw_iron"}, _miner, fair_miner=_fair_miner,
        )


class TestHandleMineDeltaSurfacing:
    STOP = {"reason": "no_effective_tool",
            "message": "effective tool lost mid-mine for minecraft:iron_ore (pickaxe broke?)"}

    def test_zero_acquired_surfaces_tool_message(self):
        out = _delta("mine_iron", fair=False, before=0, after=0,
                     miner_result=None, last_stop=self.STOP)
        assert out.startswith("FAILED mine_iron:")
        assert "pickaxe broke" in out

    def test_partial_then_tool_failure(self):
        # Pickaxe broke after a couple of ores: acquired>0 but stop is the tool.
        out = _delta("mine_iron", fair=False, before=0, after=2,
                     miner_result=None, last_stop=self.STOP)
        assert out.startswith("PARTIAL then FAILED")
        assert "acquired 2" in out
        assert "pickaxe broke" in out

    def test_fair_path_ignores_stale_last_stop(self):
        # The fair tunnel never sets last_stop; a stale tool stop from a prior
        # baritone call must not leak into a fair mine's message.
        out = _delta("mine_stone", fair=True, before=0, after=0,
                     miner_result=None, last_stop=self.STOP)
        assert "FAILED mine_stone:" not in out
        assert "no candidate reachable" in out

    def test_non_tool_stop_uses_generic_message(self):
        # A different hard stop (e.g. timeout) must not masquerade as a tool fail.
        out = _delta("mine_iron", fair=False, before=0, after=0, miner_result=None,
                     last_stop={"reason": "timeout", "message": "deadline"})
        assert out == "FAILED: no candidate reachable for mine_iron (acquired 0)"

    def test_success_unaffected(self):
        out = _delta("mine_iron", fair=False, before=0, after=4,
                     miner_result="iron_ore", last_stop=self.STOP)
        assert out.startswith("acquired 4 more")
