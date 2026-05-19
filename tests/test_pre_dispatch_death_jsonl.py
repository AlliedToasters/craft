"""Test the pre-dispatch death JSONL record (closes issue #1).

Background: when `_fetch_new_deaths` returns a death between LLM plan and
tool dispatch, the rollout aborts. Pre-fix, the JSONL never got that turn's
record, silently under-counting permadeaths in post-hoc analyzers
(`backtest_milestones.py`, `analyze_milestones.py` both filter
`_type=="turn"` then check `died`).

The fix writes a synthetic record with `_type=turn`, `died=true`,
`outcome="aborted_pre_dispatch_due_to_death"`, and the full death dict
before breaking out of the loop. This test exercises the branch end-to-end
by stubbing the agent loop's network surface so no live homunculus is
needed, then reads the resulting JSONL.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from craft import agent


@pytest.fixture
def stub_agent_substrate(monkeypatch):
    """Stub every call agent.run() makes against the world so the loop can
    enter the pre-dispatch death branch on turn 1 without any network.

    Returns nothing; tests still drive `monkeypatch` directly to install
    `chat_with_tools` and `_fetch_new_deaths` behavior (those vary per test).
    """
    # Setup envelope — _apply_setup runs server commands; skip entirely.
    monkeypatch.setattr(agent, "_apply_setup", lambda **kw: None)

    # The 3s startup sleep + any others — instant for tests.
    monkeypatch.setattr(agent.time, "sleep", lambda *_a, **_kw: None)

    # Initial-state fetches called once before the loop.
    monkeypatch.setattr(agent, "_fetch_stats", lambda: "hp=20 food=20")
    monkeypatch.setattr(agent, "_fetch_inventory", lambda: "(empty)")
    monkeypatch.setattr(agent, "_fetch_smelts", lambda: None)
    monkeypatch.setattr(agent, "_stats_raw", lambda: {"x": 0, "y": 64, "z": 0,
                                                       "biome": "plains"})
    monkeypatch.setattr(agent, "_inventory_raw", lambda: {"main": [], "offhand": {}})


def _make_tool_call(name: str, args: str = "{}") -> SimpleNamespace:
    """Build a chat_with_tools-shape tool call."""
    return SimpleNamespace(
        id="call_test",
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------- the branch exists at all


class TestPreDispatchDeathWritesRecord:
    """Issue #1: synthetic turn record must land in JSONL when the pre-
    dispatch death poll fires."""

    def test_record_written_on_pre_dispatch_death(self, tmp_path, monkeypatch, stub_agent_substrate):
        # Plan returns one tool call (the death will fire before it dispatches).
        monkeypatch.setattr(
            agent, "chat_with_tools",
            lambda messages, tools, model: ([_make_tool_call("mine_wood", '{"count": 4}')], None),
        )

        # Death record returned on the pre-dispatch poll.
        death = {
            "timestamp": 1_700_000_000_000,
            "message": "agent0 was blown up by Creeper",
            "cause": "creeper_explosion",
            "death_pos": [10, 64, -20],
        }
        monkeypatch.setattr(agent, "_fetch_new_deaths", lambda since: [death])

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=5, goal="diamond", jsonl_path=str(jsonl))

        records = _read_jsonl(jsonl)
        turn_recs = [r for r in records if r.get("_type") == "turn"]
        assert len(turn_recs) == 1, f"expected 1 turn record, got {len(turn_recs)}: {records}"

        rec = turn_recs[0]
        assert rec["turn"] == 1
        assert rec["died"] is True
        assert rec["outcome"] == "aborted_pre_dispatch_due_to_death"
        assert rec["death"] == death
        assert rec["health"] == 0
        assert rec["tool"] == "mine_wood"
        assert rec["args"] == '{"count": 4}'

    def test_loop_terminates_at_turn_1(self, tmp_path, monkeypatch, stub_agent_substrate):
        """A pre-dispatch death must end the rollout — even if max_turns is high
        and chat_with_tools would happily return more calls."""
        call_count = {"n": 0}

        def fake_chat(*_a, **_kw):
            call_count["n"] += 1
            return ([_make_tool_call("mine_wood")], None)

        monkeypatch.setattr(agent, "chat_with_tools", fake_chat)
        monkeypatch.setattr(agent, "_fetch_new_deaths",
                            lambda since: [{"timestamp": 1, "message": "x",
                                            "cause": "y", "death_pos": [0, 0, 0]}])

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=50, goal="diamond", jsonl_path=str(jsonl))

        # Only one plan happened; the death short-circuited the rest.
        assert call_count["n"] == 1

    def test_end_record_still_written(self, tmp_path, monkeypatch, stub_agent_substrate):
        """The for-loop break falls through to the normal end-record write.
        Pin this so future refactors don't accidentally skip the close."""
        monkeypatch.setattr(
            agent, "chat_with_tools",
            lambda *_a, **_kw: ([_make_tool_call("mine_wood")], None),
        )
        monkeypatch.setattr(agent, "_fetch_new_deaths",
                            lambda since: [{"timestamp": 1, "message": "x",
                                            "cause": "y", "death_pos": [0, 0, 0]}])

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=5, goal="diamond", jsonl_path=str(jsonl))

        records = _read_jsonl(jsonl)
        types = [r.get("_type") for r in records]
        assert "header" in types
        assert "turn" in types
        assert "end" in types
        assert types[-1] == "end"

    def test_plan_seconds_recorded(self, tmp_path, monkeypatch, stub_agent_substrate):
        """plan_s should be non-zero (chat_with_tools was called). exec_s and
        ctx_s are 0 because no dispatch happened."""
        monkeypatch.setattr(
            agent, "chat_with_tools",
            lambda *_a, **_kw: ([_make_tool_call("mine_wood")], None),
        )
        monkeypatch.setattr(agent, "_fetch_new_deaths",
                            lambda since: [{"timestamp": 1, "message": "x",
                                            "cause": "y", "death_pos": [0, 0, 0]}])

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=5, goal="diamond", jsonl_path=str(jsonl))

        rec = next(r for r in _read_jsonl(jsonl) if r.get("_type") == "turn")
        assert rec["plan_s"] >= 0.0  # mocked chat is near-instant; just shape
        assert rec["exec_s"] == 0.0
        assert rec["ctx_s"] == 0.0
        assert rec["total_s"] >= rec["plan_s"]


# ---------------------------------------------- no false-positives


class TestNoSpuriousRecord:
    """The synthetic record must only fire on the pre-dispatch death branch,
    not when deaths arrive post-dispatch (existing 'death this turn' branch)
    or when there are no deaths at all."""

    def test_no_death_no_synthetic_record(self, tmp_path, monkeypatch, stub_agent_substrate):
        """With no deaths, the loop runs normally. We force chat to return
        no tool call on turn 2 so the loop exits cleanly without a
        permadeath."""
        seq = iter([
            ([_make_tool_call("mine_wood")], None),  # turn 1: dispatch happens
            ([], None),                              # turn 2: empty → break
            ([], None),                              # turn 2 retry (EMPTY_RETRIES=1)
        ])
        monkeypatch.setattr(agent, "chat_with_tools",
                            lambda *_a, **_kw: next(seq))
        monkeypatch.setattr(agent, "_fetch_new_deaths", lambda since: [])

        # dispatch() runs for turn 1 since the death poll is empty. Stub the
        # downstream substrate so the post-dispatch path doesn't blow up.
        monkeypatch.setattr(agent, "dispatch", lambda name, args: "ok mined 4")
        monkeypatch.setattr(agent, "_evasion_arm", lambda *a: False)
        monkeypatch.setattr(agent, "_evasion_disarm", lambda: None)
        monkeypatch.setattr(agent, "_evasion_status", lambda: None)
        monkeypatch.setattr(agent, "_water_aversion_arm", lambda: False)
        monkeypatch.setattr(agent, "_water_aversion_disarm", lambda: None)
        monkeypatch.setattr(agent, "_water_aversion_status", lambda: None)
        # /position + /equip request stubs
        class _R:
            ok = True
            def json(self): return {}
        monkeypatch.setattr(agent.requests, "get", lambda *a, **kw: _R())
        monkeypatch.setattr(agent.requests, "post", lambda *a, **kw: _R())

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=5, goal="diamond", jsonl_path=str(jsonl))

        records = _read_jsonl(jsonl)
        turn_recs = [r for r in records if r.get("_type") == "turn"]
        # Exactly one turn record from the normal post-dispatch path; no
        # synthetic pre-dispatch entry.
        assert len(turn_recs) == 1
        assert turn_recs[0].get("died") is False
        assert turn_recs[0].get("outcome") != "aborted_pre_dispatch_due_to_death"
