"""Spectator column-spawn: surface detection + reject-reason audit trail.

The spawn loop puts the player in spectator at a high Y to force chunk
generation, scans the 1×1 column to locate the true surface, then TPs
there and switches to survival. These tests pin:

  - `_surface_from_column` (pure): air/canopy filtering + topmost pick.
  - `random_spawn` reject codes (gen_timeout / in_water / in_lava /
    biome_* / damage_in_survival) flowing through the `attempts` audit
    array, plus the success path's `tp_to` = (x, surface+1, z).
"""

from __future__ import annotations

import random

import pytest

from craft import spawn


def _blk(bid: str, y: int, x: int = 0, z: int = 0) -> dict:
    """One /scan_blocks record. id is namespaced like the live server."""
    return {"id": f"minecraft:{bid}", "x": x, "y": y, "z": z}


class TestSurfaceFromColumn:
    """`_surface_from_column` is the pure surface predicate."""

    def test_picks_topmost_solid(self):
        col = [_blk("stone", 60), _blk("dirt", 69), _blk("grass_block", 70)]
        y, bid = spawn._surface_from_column(col)
        assert y == 70 and bid == "grass_block"

    def test_skips_air_variants(self):
        # cave_air above the grass must not count as the surface.
        col = [_blk("grass_block", 70), _blk("cave_air", 75), _blk("air", 80)]
        y, bid = spawn._surface_from_column(col)
        assert y == 70 and bid == "grass_block"

    def test_skips_tree_canopy(self):
        # A column that lands on a tree resolves to the ground UNDER the
        # trunk (so surface+1 lands inside the log → survival probe rejects).
        col = [
            _blk("dirt", 69), _blk("grass_block", 70),
            _blk("oak_log", 71), _blk("oak_log", 72),
            _blk("oak_leaves", 73), _blk("birch_wood", 74),
        ]
        y, bid = spawn._surface_from_column(col)
        assert y == 70 and bid == "grass_block"

    def test_water_surface_reported(self):
        # Ocean/lake column: topmost non-canopy block is water.
        col = [_blk("sand", 50), _blk("water", 62), _blk("water", 63)]
        y, bid = spawn._surface_from_column(col)
        assert bid == "water" and y == 63

    def test_lava_surface_reported(self):
        col = [_blk("stone", 30), _blk("lava", 31)]
        y, bid = spawn._surface_from_column(col)
        assert bid == "lava" and y == 31

    def test_empty_column(self):
        assert spawn._surface_from_column([]) == (None, None)

    def test_all_air_column(self):
        col = [_blk("air", 100), _blk("cave_air", 50)]
        assert spawn._surface_from_column(col) == (None, None)


class TestRandomSpawnFlow:
    """Drive `random_spawn` with a fully stubbed network surface and assert
    the reject reasons + success contract."""

    @pytest.fixture
    def stub_network(self, monkeypatch):
        monkeypatch.setattr(spawn, "_server_cmd", lambda *_a, **_kw: {"ok": True})
        monkeypatch.setattr(spawn, "set_gamemode", lambda *_a, **_kw: None)
        monkeypatch.setattr(spawn.time, "sleep", lambda *_a, **_kw: None)
        # Settle loop reads player Y to confirm grounding; None short-circuits
        # the position check so the stubbed on_ground flag decides settling.
        monkeypatch.setattr(spawn, "_scan_player_y", lambda *_a, **_kw: None)

    def _stub(self, monkeypatch, *, column, biome="plains", hp=20.0):
        """Stub the column scan + /stats. `column` may be a block list,
        None (gen never completes), or a callable(returning either)."""
        if callable(column):
            monkeypatch.setattr(spawn, "_scan_column", column)
        else:
            monkeypatch.setattr(spawn, "_scan_column",
                                lambda *_a, **_kw: column)
        monkeypatch.setattr(
            spawn, "_stats",
            lambda *_a, **_kw: {"biome": biome, "health": hp,
                                "on_ground": True},
        )

    def _run(self, *, max_retries: int = 1, gen_timeout_s: float = 20.0):
        return spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            max_retries=max_retries,
            gen_timeout_s=gen_timeout_s,
            rng=random.Random(0),
            verbose=False,
        )

    def test_success_places_at_surface_plus_one(self, stub_network, monkeypatch):
        self._stub(monkeypatch, column=[_blk("grass_block", 70)])
        result = self._run()
        assert result["ok"] is True
        assert result["biome"] == "plains"
        tx, ty, tz = result["tp_to"]
        assert ty == 71  # surface_y + 1
        # anchor is (0,0) so tp x/z equal the chosen offset.
        assert (tx, tz) == result["offset"]
        assert result["attempts"][-1]["surface_y"] == 70

    def test_gen_timeout(self, stub_network, monkeypatch):
        # Column never generates → poll budget exhausts.
        self._stub(monkeypatch, column=None)
        result = self._run(gen_timeout_s=0.0)
        assert result["ok"] is False
        assert result["attempts"][0]["reason"] == "gen_timeout"
        assert result["attempts"][0]["surface_y"] is None

    def test_in_water_rejected(self, stub_network, monkeypatch):
        self._stub(monkeypatch, column=[_blk("sand", 50), _blk("water", 63)])
        result = self._run()
        assert result["ok"] is False
        assert result["attempts"][0]["reason"] == "in_water"

    def test_in_lava_rejected(self, stub_network, monkeypatch):
        self._stub(monkeypatch, column=[_blk("lava", 31)])
        result = self._run()
        assert result["ok"] is False
        assert result["attempts"][0]["reason"] == "in_lava"

    def test_bad_biome_rejected(self, stub_network, monkeypatch):
        self._stub(monkeypatch, column=[_blk("sand", 70)], biome="desert")
        result = self._run()
        assert result["ok"] is False
        assert result["attempts"][0]["reason"] == "biome_desert"

    def test_damage_in_survival_rejected(self, stub_network, monkeypatch):
        # Surface looks fine + biome ok, but HP drops after survival switch
        # (encased — e.g. surface+1 sat inside a trunk).
        self._stub(monkeypatch, column=[_blk("grass_block", 70)], hp=8.0)
        result = self._run()
        assert result["ok"] is False
        assert "damage_in_survival" in result["attempts"][0]["reason"]
        assert "hp=8.0" in result["attempts"][0]["reason"]

    def test_default_max_retries_is_12(self, stub_network, monkeypatch):
        self._stub(monkeypatch, column=None)
        # Omit the max_retries override so the function default (12) applies.
        result = spawn.random_spawn(
            range_blocks=100,
            homunculus_base="http://stub",
            server_cmd_base="http://stub",
            player_name="agent0",
            anchor_xz=(0, 0),
            gen_timeout_s=0.0,
            rng=random.Random(0),
            verbose=False,
        )
        assert result["ok"] is False
        assert len(result["attempts"]) == 12
        assert all(a["reason"] == "gen_timeout" for a in result["attempts"])

    def test_exhaustion_falls_back_to_land_surface(self, stub_network, monkeypatch):
        # All attempts fail, but one found dry land (bad biome). On exhaustion
        # the player should be placed on that land surface, not left in water.
        seq = [[_blk("grass_block", 80)], [_blk("water", 63)]]
        state = {"i": 0}

        def _col(*_a, **_kw):
            c = seq[min(state["i"], len(seq) - 1)]
            state["i"] += 1
            return c

        # biome=desert → attempt 1 (grass land) rejects on biome, recording a
        # priority-2 land fallback; attempt 2 is water (rejected pre-biome).
        self._stub(monkeypatch, column=_col, biome="desert")
        result = self._run(max_retries=2)
        assert result["ok"] is False
        # tp_to is the dry land surface (y=81), not the water column.
        assert result["tp_to"] is not None and result["tp_to"][1] == 81
        assert result["attempts"][0]["reason"] == "biome_desert"
        assert result["attempts"][1]["reason"] == "in_water"

    def test_retries_until_good_column(self, stub_network, monkeypatch):
        # First two columns are water, third is solid ground → spawn on 3rd.
        seq = [
            [_blk("water", 63)],
            [_blk("water", 63)],
            [_blk("grass_block", 72)],
        ]
        state = {"i": 0}

        def _col(*_a, **_kw):
            c = seq[min(state["i"], len(seq) - 1)]
            state["i"] += 1
            return c

        self._stub(monkeypatch, column=_col)
        result = self._run(max_retries=5)
        assert result["ok"] is True
        assert len(result["attempts"]) == 3
        assert result["attempts"][0]["reason"] == "in_water"
        assert result["attempts"][1]["reason"] == "in_water"
        assert result["tp_to"][1] == 73  # surface 72 + 1
