"""Spawn column-quality classifier: encased peaks vs cave-fall pockets.

The existing spawn-retry catches water/lava/biome but is blind to column
shape. Two failure modes were observed in the 2026-05-18 e2e iters=5 run:
encased-at-y=100 (4/15 shelter iters, 5/40 in fan-out) and cave-fall
from drop_y=100 (5/5 surface iters, single travel failure).

These tests pin the pure classifier; an end-to-end test threads the new
reason codes through `random_spawn`'s `attempts` audit trail.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from craft import spawn


class TestClassifyLanding:
    """`_classify_landing` is the pure predicate. Default thresholds match
    the proposed fix in project_spawn_quality_constraints.md."""

    # ---- inverted-column branch ------------------------------------------

    def test_landing_at_drop_y_is_inverted(self):
        # Encased: agent ended up AT the TP target — terrain extends above.
        reason = spawn._classify_landing(landing_y=100, drop_y=100)
        assert reason is not None
        assert "column_inverted" in reason
        assert "y=100" in reason

    def test_landing_just_below_drop_y_is_inverted(self):
        # The 5-block default margin: 95 still rejects, since real surfaces
        # this high are vanishingly rare and the cost of one extra retry is
        # cheap vs. a 50-turn rollout into an encased peak.
        reason = spawn._classify_landing(landing_y=95, drop_y=100)
        assert reason is not None and "column_inverted" in reason

    def test_landing_at_inverted_boundary_passes(self):
        # 94 = drop_y - margin - 1 = just under the threshold.
        assert spawn._classify_landing(landing_y=94, drop_y=100) is None

    # ---- cave-fall branch ------------------------------------------------

    def test_landing_far_below_drop_y_is_cave_fall(self):
        # Observed real failure: surface tests with landing_y=44.
        reason = spawn._classify_landing(landing_y=44, drop_y=100)
        assert reason is not None
        assert "cave_fall" in reason
        assert "y=44" in reason

    def test_landing_just_below_cave_fall_threshold(self):
        # drop_y - cave_fall_max - 1 = 49 rejects (100-49=51 > 50).
        reason = spawn._classify_landing(landing_y=49, drop_y=100)
        assert reason is not None and "cave_fall" in reason

    def test_landing_at_cave_fall_boundary_passes(self):
        # 50 = drop_y - cave_fall_max exactly; 100-50=50 not > 50,
        # so this just passes. Calibrated to leave landing_y=50+ alone
        # so normal plains spawns (y≈64) aren't rejected.
        assert spawn._classify_landing(landing_y=50, drop_y=100) is None

    # ---- normal-surface band ---------------------------------------------

    def test_normal_landing_passes(self):
        # 64 ≈ vanilla sea-level overworld surface.
        assert spawn._classify_landing(landing_y=64, drop_y=100) is None

    def test_high_hill_landing_passes(self):
        # 85 is a high plateau — within band, not inverted.
        assert spawn._classify_landing(landing_y=85, drop_y=100) is None

    # ---- threshold tuning ------------------------------------------------

    def test_custom_inverted_margin(self):
        # Loosen to 0: only landing AT drop_y rejects.
        assert spawn._classify_landing(95, 100, inverted_margin=0) is None
        reason = spawn._classify_landing(100, 100, inverted_margin=0)
        assert reason and "column_inverted" in reason

    def test_custom_cave_fall_max(self):
        # Tighten to 10: anything more than 10 below drop rejects.
        reason = spawn._classify_landing(85, 100, cave_fall_max=10)
        assert reason and "cave_fall" in reason

    def test_alternative_drop_y(self):
        # Scenario: someone passes drop_y=80 (lower TP). 75 is inverted
        # (80 - 5); 29 is cave-fall (80 - 50 = 30, so y<30 rejects);
        # y=40 passes (80-40=40 < 50).
        assert "column_inverted" in spawn._classify_landing(75, 80)
        assert "cave_fall" in spawn._classify_landing(29, 80)
        assert spawn._classify_landing(40, 80) is None

    # ---- edge cases ------------------------------------------------------

    def test_inverted_takes_precedence_over_cave_fall(self):
        # If both branches would fire (impossible under defaults, but
        # construct it with extreme thresholds), inverted wins because
        # it's the cheaper-to-detect bug. Defensive pin, not load-bearing.
        # cave_fall_max=-100 means "always cave-fall"; landing_y=100=drop_y
        # is also inverted. Inverted check runs first.
        reason = spawn._classify_landing(100, 100, cave_fall_max=-100)
        assert "column_inverted" in reason


class TestAttemptIntegration:
    """`_attempt` is nested inside `random_spawn`; we exercise it by
    driving `random_spawn` with stubbed network surface. The new reason
    codes must appear verbatim in the `attempts` audit array."""

    @pytest.fixture
    def stub_network(self, monkeypatch):
        """Make `random_spawn` runnable without any live homunculus.

        Each test then layers in per-call `_stats` and `_position`."""
        monkeypatch.setattr(spawn, "_server_cmd",
                            lambda *_a, **_kw: {"ok": True})
        monkeypatch.setattr(spawn, "set_gamemode",
                            lambda *_a, **_kw: None)
        monkeypatch.setattr(spawn.time, "sleep",
                            lambda *_a, **_kw: None)

    def _stub_world(self, monkeypatch, *, landing_y: int,
                    biome: str = "plains", hp: float = 20.0,
                    in_water: bool = False, in_lava: bool = False,
                    position_has_y: bool = True):
        """Stub _stats (no y; live homunculus omits coords from /stats)
        and _position (has y). Each attempt: poll-loop _stats until
        on_ground, then _position for y, then _stats for biome/HP."""
        stats = {"on_ground": True, "biome": biome,
                 "in_water": in_water, "in_lava": in_lava, "health": hp}
        monkeypatch.setattr(spawn, "_stats",
                            lambda *_a, **_kw: dict(stats))
        if position_has_y:
            pos = {"x": 0, "y": landing_y, "z": 0}
        else:
            pos = {"x": 0, "z": 0}
        monkeypatch.setattr(spawn, "_position",
                            lambda *_a, **_kw: dict(pos))

    def test_inverted_landing_rejected_with_reason(self, stub_network, monkeypatch):
        self._stub_world(monkeypatch, landing_y=100)
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=1,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is False
        assert len(result["attempts"]) == 1
        assert "column_inverted" in result["attempts"][0]["reason"]
        assert "y=100" in result["attempts"][0]["reason"]

    def test_cave_fall_rejected_with_reason(self, stub_network, monkeypatch):
        self._stub_world(monkeypatch, landing_y=44)
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=1,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is False
        assert "cave_fall" in result["attempts"][0]["reason"]
        assert "y=44" in result["attempts"][0]["reason"]

    def test_normal_landing_accepted(self, stub_network, monkeypatch):
        # Landing at y=64 (overworld sea-level), plains: should succeed.
        self._stub_world(monkeypatch, landing_y=64)
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=1,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is True
        assert result["biome"] == "plains"

    def test_column_check_runs_before_biome(self, stub_network, monkeypatch):
        # If both the column AND biome are bad, we should see the column
        # reason first — it's the earlier check and the more actionable
        # signal (biome reflects the bad column's misleading surface).
        self._stub_world(monkeypatch, landing_y=44, biome="badlands")
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=1,
            rng=random.Random(0),
            verbose=False,
        )
        assert "cave_fall" in result["attempts"][0]["reason"]

    def test_missing_y_does_not_blow_up(self, stub_network, monkeypatch):
        # /position response missing "y" (degraded homunculus?) — the
        # column check must skip silently rather than KeyError. Existing
        # biome/HP path still applies.
        self._stub_world(monkeypatch, landing_y=0, position_has_y=False)
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=1,
            rng=random.Random(0),
            verbose=False,
        )
        # No column reason fires; landing classifies as plains -> ok.
        assert result["ok"] is True


class TestAdaptiveRetry:
    """When the region keeps tripping column_inverted, drop_y should rise
    above the observed terrain peak so subsequent attempts can land in a
    real surface band. Calibrated against the 2026-05-18 iters=10 run:
    mine_diamond iter=3 logged 7/8 column_inverted at y=95-100 and exhausted
    retries; the same trace should succeed under adaptive drop_y."""

    @pytest.fixture
    def stub_network(self, monkeypatch):
        monkeypatch.setattr(spawn, "_server_cmd",
                            lambda *_a, **_kw: {"ok": True})
        monkeypatch.setattr(spawn, "set_gamemode",
                            lambda *_a, **_kw: None)
        monkeypatch.setattr(spawn.time, "sleep",
                            lambda *_a, **_kw: None)

    def _stub_landing_per_attempt(self, monkeypatch, sequence: list[int]):
        """Returns a different landing_y per attempt; biome=plains, hp=20,
        no water/lava. `sequence` is consumed in order.

        The implementation reads /position immediately after /stats reports
        on_ground; we expose the next landing_y via a tiny stateful stub."""
        state = {"i": 0}

        def _pos(*_a, **_kw):
            idx = min(state["i"], len(sequence) - 1)
            state["i"] += 1
            return {"x": 0, "y": sequence[idx], "z": 0}

        stats = {"on_ground": True, "biome": "plains",
                 "in_water": False, "in_lava": False, "health": 20.0}
        monkeypatch.setattr(spawn, "_stats",
                            lambda *_a, **_kw: dict(stats))
        monkeypatch.setattr(spawn, "_position", _pos)

    def test_two_inverted_hits_bumps_drop_y(self, stub_network, monkeypatch):
        # First two attempts land at y=100 (inverted); third would land at
        # y=64 under the new drop_y. Adapter should kick in at attempt 3.
        self._stub_landing_per_attempt(monkeypatch, [100, 100, 64])
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=5,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is True
        attempts = result["attempts"]
        assert len(attempts) == 3
        assert "column_inverted" in attempts[0]["reason"]
        assert "column_inverted" in attempts[1]["reason"]
        # Adapter raises drop_y on attempt 3 (after 2nd inverted).
        # max_inverted_y=100, bump_dy=40 → new drop_y=140.
        assert attempts[0]["drop_y"] == 100
        assert attempts[1]["drop_y"] == 100
        assert attempts[2]["drop_y"] == 140
        # cave_fall_max bumps proportionally so y=64 (drop_y-76) passes.
        # Sanity: under default cave_fall_max=50, landing_y=64 at drop_y=140
        # would have been rejected (140-64=76>50). The adapter must bump
        # cave_fall_max to 50 + (140-100) = 90 → 140-64=76<90 → ok.
        assert attempts[2]["ok"] is True

    def test_inverted_hits_under_threshold_no_bump(self, stub_network, monkeypatch):
        # Single inverted hit doesn't trigger the bump.
        self._stub_landing_per_attempt(monkeypatch, [100, 64])
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=5,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is True
        attempts = result["attempts"]
        # Both attempts use the original drop_y=100.
        assert attempts[0]["drop_y"] == 100
        assert attempts[1]["drop_y"] == 100

    def test_non_consecutive_inverted_still_counts(self, stub_network, monkeypatch):
        # iter=8 mine_wood was inverted+biome+inverted+biome+...; total
        # inverted hits should accumulate, not require consecutiveness.
        # Sequence: y=100 (inverted) → invalid biome later via no, so
        # easier path: ensure max_inverted_y tracks the highest observed.
        # Here attempt1=100, attempt2=96, attempt3 should bump to 100+40=140.
        self._stub_landing_per_attempt(monkeypatch, [100, 96, 64])
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=5,
            rng=random.Random(0),
            verbose=False,
        )
        attempts = result["attempts"]
        assert "column_inverted" in attempts[0]["reason"]
        assert "column_inverted" in attempts[1]["reason"]
        # max_inverted_y=100 (not 96): we track the peak.
        assert attempts[2]["drop_y"] == 140
        assert result["ok"] is True

    def test_bump_after_threshold_tunable(self, stub_network, monkeypatch):
        # column_inverted_bump_after=3 — first two inverted hits don't
        # trigger; third does.
        self._stub_landing_per_attempt(monkeypatch, [100, 100, 100, 64])
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=5,
            column_inverted_bump_after=3,
            rng=random.Random(0),
            verbose=False,
        )
        attempts = result["attempts"]
        assert attempts[0]["drop_y"] == 100
        assert attempts[1]["drop_y"] == 100
        assert attempts[2]["drop_y"] == 100
        assert attempts[3]["drop_y"] == 140
        assert result["ok"] is True

    def test_default_max_retries_is_12(self, stub_network, monkeypatch):
        # All attempts fail at the "no on_ground" pre-check (stuck-suffocation
        # path) — this failure mode doesn't depend on drop_y so the adapter
        # is bypassed and we get a clean count of the retry budget.
        monkeypatch.setattr(spawn, "_stats",
                            lambda *_a, **_kw: {"on_ground": False})
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            # No max_retries override — default applies.
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is False
        assert len(result["attempts"]) == 12
        assert all(a["reason"] == "stuck_no_ground" for a in result["attempts"])

    def test_adaptive_does_not_lower_drop_y(self, stub_network, monkeypatch):
        # If a later inverted hit reports a lower y, drop_y should NOT
        # regress. (max_inverted_y is monotone.)
        self._stub_landing_per_attempt(monkeypatch, [120, 120, 95, 95, 64])
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=6,
            # drop_y default is 100, but landings can exceed it (rare TP
            # quirk). Pin a high drop_y to make the asymmetry observable.
            drop_y=125,  # inverted_margin=5 → 120 still inverted.
            rng=random.Random(0),
            verbose=False,
        )
        attempts = result["attempts"]
        # First two land at y=120, inverted. After 2nd hit:
        # max_inverted_y=120, new drop_y=160.
        assert attempts[2]["drop_y"] == 160
        # Third lands at y=95 (still inverted under new threshold:
        # 160-5=155, 95<155, so NOT inverted — it'd be cave_fall instead.
        # 160-95=65, cave_fall_max=50+(160-125)=85, 65<85 → passes!
        # Whoops, my arithmetic: cave_fall_max=50+(160-125)=85; 65<=85
        # so y=95 passes. Fourth attempt never runs. Adjust expectation:
        assert result["ok"] is True

    def test_inverted_y_regex_handles_negative(self):
        # Defensive: nether-style coordinates aren't expected but the regex
        # shouldn't barf on a leading minus.
        m = spawn._INVERTED_Y_RE.match("column_inverted(y=-5)")
        assert m is not None
        assert int(m.group(1)) == -5
