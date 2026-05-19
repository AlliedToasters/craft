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
