"""Test the homunculus-unreachable early-abort guard (closes issue #9).

Background: when the homunculus HTTP bridge dies mid-rollout (client crash, MC
disconnect, prismlauncher zombie), every tool call returns transport_error and
the agent loop otherwise runs all max_turns turns producing a *null* trajectory.
That record flags as "T50, didn't die" and silently inflates the population
survival rate in post-hoc analyzers (sister to issue #1).

The fix probes liveness at the top of each turn via `_homunculus_reachable()`.
After K consecutive unreachable probes it writes a synthetic turn record with
`outcome="aborted_homunculus_unreachable"` and breaks; the `end` record carries
`rollout_aborted="homunculus_unreachable"` so analyzers can exclude null runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from craft import agent


# ---------------------------------------------- the probe itself


class TestHomunculusReachable:
    """Only a transport-level failure counts as unreachable; any completed HTTP
    exchange (even a 4xx/5xx) proves the process is up."""

    def test_completed_request_is_reachable(self, monkeypatch):
        monkeypatch.setattr(agent.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200))
        assert agent._homunculus_reachable() is True

    def test_error_status_still_reachable(self, monkeypatch):
        # A 500 still means the bridge answered — don't abort the rollout for it.
        monkeypatch.setattr(agent.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=500))
        assert agent._homunculus_reachable() is True

    def test_connection_refused_is_unreachable(self, monkeypatch):
        def _boom(*a, **kw):
            raise requests.ConnectionError("Connection refused")
        monkeypatch.setattr(agent.requests, "get", _boom)
        assert agent._homunculus_reachable() is False

    def test_timeout_is_unreachable(self, monkeypatch):
        def _boom(*a, **kw):
            raise requests.Timeout("timed out")
        monkeypatch.setattr(agent.requests, "get", _boom)
        assert agent._homunculus_reachable() is False


# ---------------------------------------------- the abort branch in run()


@pytest.fixture
def stub_agent_substrate(monkeypatch):
    """Stub every call agent.run() makes before/around the loop so it can be
    driven without a live homunculus. Tests install `_homunculus_reachable`
    and `chat_with_tools` behavior themselves."""
    monkeypatch.setattr(agent, "_apply_setup", lambda **kw: None)
    monkeypatch.setattr(agent.time, "sleep", lambda *_a, **_kw: None)
    monkeypatch.setattr(agent, "_fetch_stats", lambda: "hp=20 food=20")
    monkeypatch.setattr(agent, "_fetch_inventory", lambda: "(empty)")
    monkeypatch.setattr(agent, "_fetch_smelts", lambda: None)
    monkeypatch.setattr(agent, "_stats_raw", lambda: {"x": 0, "y": 64, "z": 0, "biome": "plains"})
    monkeypatch.setattr(agent, "_inventory_raw", lambda: {"main": [], "offhand": {}})


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestUnreachableAbort:
    def test_aborts_after_k_turns_without_planning(self, tmp_path, monkeypatch, stub_agent_substrate):
        """A dead bridge must abort within K turns and never call the LLM —
        the whole point is to stop wasting plan/dispatch on a null run."""
        monkeypatch.setattr(agent, "_homunculus_reachable", lambda *a, **kw: False)

        chat_calls = {"n": 0}

        def fake_chat(*_a, **_kw):
            chat_calls["n"] += 1
            return ([], None, "", None)
        monkeypatch.setattr(agent, "chat_with_tools", fake_chat)

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=50, goal="diamond", jsonl_path=str(jsonl))

        # No plan was ever attempted — the probe short-circuited every turn.
        assert chat_calls["n"] == 0

        records = _read_jsonl(jsonl)
        turn_recs = [r for r in records if r.get("_type") == "turn"]
        assert len(turn_recs) == 1, f"expected exactly one synthetic abort record, got {turn_recs}"
        rec = turn_recs[0]
        assert rec["outcome"] == "aborted_homunculus_unreachable"
        assert rec["died"] is False
        assert rec["tool"] is None
        # Abort trips on the K-th consecutive unreachable probe.
        assert rec["turn"] == 3

    def test_end_record_flags_abort(self, tmp_path, monkeypatch, stub_agent_substrate):
        """The `end` record must carry the abort reason so post-hoc analyzers
        can exclude the null run from population stats."""
        monkeypatch.setattr(agent, "_homunculus_reachable", lambda *a, **kw: False)
        monkeypatch.setattr(agent, "chat_with_tools",
                            lambda *_a, **_kw: ([], None, "", None))

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=50, goal="diamond", jsonl_path=str(jsonl))

        records = _read_jsonl(jsonl)
        end = next(r for r in records if r.get("_type") == "end")
        assert end["rollout_aborted"] == "homunculus_unreachable"
        # End record is still the last line — the break falls through to close.
        assert records[-1].get("_type") == "end"

    def test_transient_blip_resets_counter(self, tmp_path, monkeypatch, stub_agent_substrate):
        """A single unreachable probe followed by a good one must NOT abort —
        only a *sustained* outage trips the guard (K consecutive)."""
        # Reachability pattern: down, up, down, up, ... never K-in-a-row.
        seq = iter([False, True, False, True, False, True, False, True])

        def flaky(*a, **kw):
            try:
                return next(seq)
            except StopIteration:
                return True
        monkeypatch.setattr(agent, "_homunculus_reachable", flaky)

        # Once a probe passes, the turn proceeds to chat; return no tool call so
        # the loop exits cleanly via the empty-plan break instead of an abort.
        monkeypatch.setattr(agent, "chat_with_tools",
                            lambda *_a, **_kw: ([], None, "", None))

        jsonl = tmp_path / "rollout.jsonl"
        agent.run(max_turns=10, goal="diamond", jsonl_path=str(jsonl))

        records = _read_jsonl(jsonl)
        aborts = [r for r in records if r.get("outcome") == "aborted_homunculus_unreachable"]
        assert aborts == [], "transient blips must not trip the abort guard"
        end = next(r for r in records if r.get("_type") == "end")
        assert end["rollout_aborted"] is None
