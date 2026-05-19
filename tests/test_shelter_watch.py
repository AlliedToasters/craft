"""Unit tests for the shelter watcher in craft.agent.

Three components, in dependency order:
    1. `_SHELTER_ANCHOR_RE` — parses anchor from build_shelter outcome string.
    2. `_arm_shelter_watch(outcome)` — module-level state init.
    3. `_poll_shelter_watch()` — per-turn state machine; HTTP-bound (polls
       /position for drift + /scan_entities per hostile type).

**Why this needs unit coverage**: this code carried two real bugs:
    - 2026-05-14: watcher armed 14000 blocks from shelter because the
      original implementation re-read /position during arm, which raced
      against AutoRespawn after a mid-build death. Fix = parse the anchor
      from the outcome string, sidestepping /position. The regex IS the
      contract — if anyone refactors it, this suite fails.
    - 2026-05-14 N=18 study: the ≥2 consecutive polls debounce rule is
      load-bearing for breach confidence (single-poll mob-at-edge was
      causing false-positive re-shelter spam). Pinned here.

**Mocking strategy**: state machine code only touches the network via
`requests.get(f"{HOMUNCULUS_BASE}/position"...)` and
`requests.get(f"{HOMUNCULUS_BASE}/scan_entities"...)`. Monkeypatch
`craft.agent.requests` with a router that dispatches by URL substring.
No live homunculus needed — agent0 isn't required for these tests.
"""

from __future__ import annotations

import pytest

from craft import agent
from craft.agent import (
    _SHELTER_ANCHOR_RE,
    _arm_shelter_watch,
    _poll_shelter_watch,
)


# ---------------------------------------------- mocking infrastructure


class _FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, data, status: int = 200):
        self._data = data
        self.status_code = status
        self.ok = status < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _make_router(
    *,
    position: dict | None = None,
    scan_by_type: dict[str, list[dict]] | None = None,
    position_exc: Exception | None = None,
    scan_exc: Exception | None = None,
):
    """Build a fake `requests.get` that dispatches by URL.

    - position={"x":..,"y":..,"z":..} → /position responds with that body.
    - scan_by_type={"minecraft:zombie": [ent, ...], ...} → /scan_entities
      returns entities for the type in the query string. Missing types = [].
    - position_exc / scan_exc → simulate transport failures.
    """
    position = position or {}
    scan_by_type = scan_by_type or {}

    def fake_get(url, params=None, timeout=None):
        if "/position" in url:
            if position_exc is not None:
                raise position_exc
            return _FakeResp(position)
        if "/scan_entities" in url:
            if scan_exc is not None:
                raise scan_exc
            mob_type = (params or {}).get("type")
            entities = scan_by_type.get(mob_type, [])
            return _FakeResp({"success": True, "entities": entities})
        return _FakeResp({}, status=404)

    return fake_get


def _mob(uuid: str, mob_type: str, x: float, y: float, z: float) -> dict:
    """Build a /scan_entities entity dict matching homunculus shape."""
    return {"uuid": uuid, "type": mob_type, "position": [x, y, z]}


@pytest.fixture(autouse=True)
def _clear_shelter_state():
    """Tests must start with no shelter armed. The module-level
    `_shelter_watch` persists across tests otherwise."""
    agent._shelter_watch = None
    yield
    agent._shelter_watch = None


# ---------------------------------------------- regex


class TestAnchorRegex:
    """`_SHELTER_ANCHOR_RE` is parsed against build_shelter's outcome string.
    The canonical form (per craft.tools handle_build_shelter) is:
        f"shelter at ({px},{py},{pz}); excavated N cells, placed M (...)"
    """

    def test_positive_coords(self):
        m = _SHELTER_ANCHOR_RE.search("shelter at (12,64,300); excavated 5 cells")
        assert m is not None
        assert m.groups() == ("12", "64", "300")

    def test_negative_coords(self):
        m = _SHELTER_ANCHOR_RE.search("shelter at (-12,64,-300); placed 53")
        assert m is not None
        assert m.groups() == ("-12", "64", "-300")

    def test_mixed_signs(self):
        m = _SHELTER_ANCHOR_RE.search("shelter at (-1,-64,1)")
        assert m is not None
        assert m.groups() == ("-1", "-64", "1")

    def test_zero_coords(self):
        m = _SHELTER_ANCHOR_RE.search("shelter at (0,0,0); placed 0 (nothing)")
        assert m is not None
        assert m.groups() == ("0", "0", "0")

    def test_no_match_on_decimal_coords(self):
        """Regex requires integer coords — fractional Y from /position would
        not match. (Outcome from handle_build_shelter always emits ints.)"""
        m = _SHELTER_ANCHOR_RE.search("shelter at (12.5,64,300)")
        assert m is None

    def test_no_match_on_missing_parens(self):
        m = _SHELTER_ANCHOR_RE.search("shelter at 12,64,300")
        assert m is None

    def test_no_match_on_malformed(self):
        assert _SHELTER_ANCHOR_RE.search("") is None
        assert _SHELTER_ANCHOR_RE.search("no anchor here") is None
        assert _SHELTER_ANCHOR_RE.search("shelter at (12,)") is None
        assert _SHELTER_ANCHOR_RE.search("shelter at (a,b,c)") is None

    def test_canonical_outcome_string_matches(self):
        """The exact f-string format from craft.tools:2587 — pin against
        format drift on the handle_build_shelter side."""
        outcome = (
            "shelter at (123,64,-456); excavated 10 cells, "
            "placed 53 (40 cobblestone, 13 dirt)"
        )
        m = _SHELTER_ANCHOR_RE.search(outcome)
        assert m is not None
        assert m.groups() == ("123", "64", "-456")


# ---------------------------------------------- arm


class TestArmShelterWatch:
    """`_arm_shelter_watch` parses the outcome + initializes the module-level
    `_shelter_watch` dict. Must be parser-only (no /position read) — that's
    the load-bearing fix from the 14000-block-drift bug."""

    def test_arm_sets_anchor_as_int_tuple(self):
        _arm_shelter_watch("shelter at (12,64,-300); placed 53")
        assert agent._shelter_watch is not None
        anchor = agent._shelter_watch["anchor"]
        assert anchor == (12, 64, -300)
        # Must be ints — float coords would break math.floor checks downstream
        assert all(isinstance(c, int) for c in anchor)

    def test_arm_initializes_empty_per_uuid(self):
        _arm_shelter_watch("shelter at (0,64,0)")
        assert agent._shelter_watch["per_uuid"] == {}

    def test_arm_initializes_breach_false(self):
        _arm_shelter_watch("shelter at (0,64,0)")
        assert agent._shelter_watch["breach"] is False
        assert agent._shelter_watch["breach_first_t"] is None

    def test_arm_sets_started_at(self):
        _arm_shelter_watch("shelter at (0,64,0)")
        # Just verify the key exists and is float-ish; exact timing isn't pinned
        assert isinstance(agent._shelter_watch["started_at"], float)

    def test_arm_no_match_is_noop(self, capsys):
        """Outcome without anchor regex → state remains None + a diagnostic
        line. This is the only protection if handle_build_shelter's outcome
        format ever drifts."""
        agent._shelter_watch = None
        _arm_shelter_watch("FAILED: shelter build aborted, only 12 buildables")
        assert agent._shelter_watch is None
        captured = capsys.readouterr().out
        assert "couldn't find" in captured

    def test_arm_replaces_prior_state(self):
        """Calling arm twice should reset to the new anchor — agents do
        re-build mid-rollout and we shouldn't be tracking stale per_uuid."""
        _arm_shelter_watch("shelter at (10,64,10)")
        agent._shelter_watch["per_uuid"]["fake-uuid"] = {"stale": True}
        _arm_shelter_watch("shelter at (20,64,20)")
        assert agent._shelter_watch["anchor"] == (20, 64, 20)
        assert agent._shelter_watch["per_uuid"] == {}


# ---------------------------------------------- poll: disarmed


class TestPollDisarmed:
    def test_poll_returns_none_when_unarmed(self, monkeypatch):
        """No state → poll returns None and makes no HTTP calls."""
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _FakeResp({})

        monkeypatch.setattr(agent.requests, "get", fake_get)
        assert agent._shelter_watch is None
        result = _poll_shelter_watch()
        assert result is None
        assert call_count == 0, "should short-circuit before any network call"


# ---------------------------------------------- poll: drift auto-disarm


class TestDriftAutoDisarm:
    def test_drift_beyond_12_disarms(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        # Player at (50, 64, 0) — 50 > 12 → disarm
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position={"x": 50, "y": 64, "z": 0}),
        )
        result = _poll_shelter_watch()
        assert result is None
        assert agent._shelter_watch is None

    def test_drift_in_z_disarms(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position={"x": 0, "y": 64, "z": 13}),
        )
        result = _poll_shelter_watch()
        assert result is None
        assert agent._shelter_watch is None

    def test_within_12_does_not_disarm(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position={"x": 10, "y": 64, "z": 10}),
        )
        result = _poll_shelter_watch()
        assert result is None  # no mobs scanned, so no message
        assert agent._shelter_watch is not None

    def test_boundary_exactly_12_does_not_disarm(self, monkeypatch):
        """Strict greater-than: exactly 12 stays armed (`abs(...) > 12`)."""
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position={"x": 12, "y": 64, "z": -12}),
        )
        _poll_shelter_watch()
        assert agent._shelter_watch is not None

    def test_position_http_error_does_not_disarm(self, monkeypatch):
        """Network failure on /position must NOT silently disarm — that
        would let a transport blip wipe the only nighttime safety state."""
        import requests as real_requests
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position_exc=real_requests.ConnectionError("boom")),
        )
        _poll_shelter_watch()
        assert agent._shelter_watch is not None

    def test_position_non_numeric_does_not_disarm(self, monkeypatch):
        """isinstance check requires (int, float) — None/missing should
        skip drift check without disarming."""
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(position={"x": None, "y": None, "z": None}),
        )
        _poll_shelter_watch()
        assert agent._shelter_watch is not None


# ---------------------------------------------- poll: cavity geometry


class TestCavityGeometry:
    """Cavity occupies x∈[ax-2, ax+2], z∈[az-2, az+2], y∈[ay, ay+1].
    These tests verify the floor-based inside_block check against the
    canonical bounds."""

    def _arm_and_check(self, monkeypatch, mob_x, mob_y, mob_z, expected_breach):
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = _mob("u1", "minecraft:zombie", mob_x, mob_y, mob_z)
        # Two consecutive polls with mob in same place — debounce → confirm.
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [ent]},
            ),
        )
        _poll_shelter_watch()  # poll 1 (consec_in -> 1)
        result = _poll_shelter_watch()  # poll 2 (would confirm if inside)
        if expected_breach:
            assert result is not None and "BREACHED" in result
        else:
            assert result is None or "BREACHED" not in result

    def test_mob_at_anchor_inside(self, monkeypatch):
        self._arm_and_check(monkeypatch, 0, 64, 0, expected_breach=True)

    def test_mob_at_x_plus_2_inside(self, monkeypatch):
        self._arm_and_check(monkeypatch, 2.5, 64, 0, expected_breach=True)

    def test_mob_at_x_plus_3_outside(self, monkeypatch):
        """floor(3.0)=3 > ax+2=2 → outside."""
        self._arm_and_check(monkeypatch, 3.0, 64, 0, expected_breach=False)

    def test_mob_at_z_minus_2_inside(self, monkeypatch):
        self._arm_and_check(monkeypatch, 0, 64, -2, expected_breach=True)

    def test_mob_below_floor_outside(self, monkeypatch):
        """y=63.5 floors to 63 < ay=64 → outside."""
        self._arm_and_check(monkeypatch, 0, 63.5, 0, expected_breach=False)

    def test_mob_at_y_plus_1_inside(self, monkeypatch):
        """Cavity is 2 tall: y∈[64,65]. y=65.5 floors to 65 → inside."""
        self._arm_and_check(monkeypatch, 0, 65.5, 0, expected_breach=True)

    def test_mob_at_y_plus_2_outside(self, monkeypatch):
        """y=66.0 floors to 66 > ay+1=65 → outside."""
        self._arm_and_check(monkeypatch, 0, 66.0, 0, expected_breach=False)


# ---------------------------------------------- poll: debounce


class TestBreachDebounce:
    """≥2 consecutive polls with mob inside is the confirmation rule.
    Single-poll false positives (mob hugging the cavity wall transiently)
    must NOT fire."""

    def test_one_poll_inside_no_breach(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = _mob("u1", "minecraft:zombie", 0, 64, 0)
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [ent]},
            ),
        )
        result = _poll_shelter_watch()
        assert result is None
        # Recorded consec=1 but not confirmed
        rec = agent._shelter_watch["per_uuid"]["u1"]
        assert rec["consec_in"] == 1
        assert rec["confirmed"] is False

    def test_two_polls_confirms_breach(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = _mob("u1", "minecraft:zombie", 0, 64, 0)
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [ent]},
            ),
        )
        assert _poll_shelter_watch() is None
        result = _poll_shelter_watch()
        assert result is not None
        assert "SHELTER BREACHED" in result
        assert "zombie" in result
        assert "build_shelter" in result  # the remediation hint
        assert agent._shelter_watch["per_uuid"]["u1"]["confirmed"] is True
        assert agent._shelter_watch["breach"] is True

    def test_in_then_out_resets_consec(self, monkeypatch):
        """Mob inside on poll1, outside on poll2 → consec resets to 0.
        Mob back inside on poll3 → only consec=1 again (not confirmed)."""
        _arm_shelter_watch("shelter at (0,64,0)")
        inside = _mob("u1", "minecraft:zombie", 0, 64, 0)
        outside = _mob("u1", "minecraft:zombie", 10, 64, 10)

        # Poll 1: inside
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [inside]},
            ),
        )
        _poll_shelter_watch()
        assert agent._shelter_watch["per_uuid"]["u1"]["consec_in"] == 1

        # Poll 2: outside cavity (but still in scan range)
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [outside]},
            ),
        )
        _poll_shelter_watch()
        assert agent._shelter_watch["per_uuid"]["u1"]["consec_in"] == 0

        # Poll 3: inside again → consec=1, still not confirmed
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [inside]},
            ),
        )
        result = _poll_shelter_watch()
        assert result is None
        assert agent._shelter_watch["per_uuid"]["u1"]["consec_in"] == 1
        assert agent._shelter_watch["per_uuid"]["u1"]["confirmed"] is False


# ---------------------------------------------- poll: continued breach


class TestContinuedBreach:
    """After a confirmed breach: if the mob is still inside on subsequent
    polls, surface SHELTER STILL BREACHED so the agent can re-react."""

    def _drive_to_confirmed(self, monkeypatch, mob_ent):
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [mob_ent]},
            ),
        )
        _poll_shelter_watch()  # consec=1
        _poll_shelter_watch()  # consec=2 → confirmed, fires SHELTER BREACHED

    def test_continued_breach_after_confirm(self, monkeypatch):
        ent = _mob("u1", "minecraft:zombie", 0, 64, 0)
        self._drive_to_confirmed(monkeypatch, ent)
        # 3rd poll, mob still inside → STILL BREACHED line
        result = _poll_shelter_watch()
        assert result is not None
        assert "STILL BREACHED" in result
        assert "1x" in result
        assert "zombie" in result

    def test_no_message_when_mob_leaves_cavity(self, monkeypatch):
        """Confirmed mob → leaves cavity (still visible) → quiet."""
        inside = _mob("u1", "minecraft:zombie", 0, 64, 0)
        outside = _mob("u1", "minecraft:zombie", 10, 64, 10)
        self._drive_to_confirmed(monkeypatch, inside)
        # Now mob walks out of cavity (still seen in scan)
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [outside]},
            ),
        )
        result = _poll_shelter_watch()
        assert result is None

    def test_no_message_when_mob_out_of_scan(self, monkeypatch):
        """Confirmed mob → chunk unloads / mob despawns → unseen.
        live_inside check requires `uuid in seen` → no still-breached message."""
        ent = _mob("u1", "minecraft:zombie", 0, 64, 0)
        self._drive_to_confirmed(monkeypatch, ent)
        # Now scan returns no entities — mob fell off the world
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={},
            ),
        )
        result = _poll_shelter_watch()
        assert result is None
        # consec_in must be reset because mob was unseen
        assert agent._shelter_watch["per_uuid"]["u1"]["consec_in"] == 0


# ---------------------------------------------- poll: multi-type


class TestMultipleHostileTypes:
    def test_breach_message_has_specific_mob_type(self, monkeypatch):
        """Confirm message names the exact hostile species (not generic)."""
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = _mob("u1", "minecraft:creeper", 0, 64, 0)
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:creeper": [ent]},
            ),
        )
        _poll_shelter_watch()
        result = _poll_shelter_watch()
        assert "creeper" in result

    def test_still_breached_aggregates_types(self, monkeypatch):
        """Two confirmed mobs of different species → STILL BREACHED joins
        types with comma + sorted."""
        _arm_shelter_watch("shelter at (0,64,0)")
        zomb = _mob("u1", "minecraft:zombie", 0, 64, 0)
        spider = _mob("u2", "minecraft:spider", 0, 64, 0)
        # Confirm both via 2 polls
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={
                    "minecraft:zombie": [zomb],
                    "minecraft:spider": [spider],
                },
            ),
        )
        _poll_shelter_watch()  # poll 1
        _poll_shelter_watch()  # poll 2 → both confirmed (one fires SHELTER BREACHED)
        # poll 3 → still-breached line aggregates
        result = _poll_shelter_watch()
        assert result is not None
        assert "STILL BREACHED" in result
        assert "2x" in result
        # sorted: spider before zombie
        assert "spider,zombie" in result


# ---------------------------------------------- poll: input sanitization


class TestEntitySanitization:
    """The scan response may contain incomplete/malformed entities. Each
    should be skipped silently — no crashes, no false positives."""

    def test_entity_without_uuid_skipped(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = {"type": "minecraft:zombie", "position": [0, 64, 0]}  # no uuid
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [ent]},
            ),
        )
        _poll_shelter_watch()
        _poll_shelter_watch()
        assert agent._shelter_watch["per_uuid"] == {}
        assert agent._shelter_watch["breach"] is False

    def test_entity_without_position_skipped(self, monkeypatch):
        _arm_shelter_watch("shelter at (0,64,0)")
        ent = {"uuid": "u1", "type": "minecraft:zombie"}  # no position
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_by_type={"minecraft:zombie": [ent]},
            ),
        )
        _poll_shelter_watch()
        _poll_shelter_watch()
        # uuid IS recorded (seen.add(uuid) fires before the position check)
        # but consec_in stays 0 because `here` is False
        rec = agent._shelter_watch["per_uuid"].get("u1")
        if rec is not None:
            assert rec["confirmed"] is False
            assert rec["consec_in"] == 0
        assert agent._shelter_watch["breach"] is False

    def test_scan_transport_error_no_crash(self, monkeypatch):
        """RequestException in /scan_entities → returns [] (per _scan_hostile)."""
        import requests as real_requests
        _arm_shelter_watch("shelter at (0,64,0)")
        monkeypatch.setattr(
            agent.requests, "get",
            _make_router(
                position={"x": 0, "y": 64, "z": 0},
                scan_exc=real_requests.ConnectionError("boom"),
            ),
        )
        # Should not raise; should return None
        assert _poll_shelter_watch() is None
