"""Unit tests for the fair-stone staircase real-progress guard (issue #10).

Context: `mine_stone` forces fair=True → `mine.tunnel_for_stone` staircase. The
old stuck-guard keyed on Baritone Excavate's own box accounting ("cleared"),
which stays true even on a partial-break→reset loop where no block actually
breaks — so it never bailed and burned the full 150s timeout, wearing out the
pickaxe with zero descent (the agent9 stalemate tape, 2026-06-03).

The fix keys the stuck-streak on REAL progress: inventory gain OR the player
genuinely descending the staircase. These tests pin both directions:
  - stuck (no drops, no descent) → bail at _STONE_STUCK_LIMIT steps, not 150s;
  - legitimate descent (player Y falls each step) → never bails even with no
    cobble drop (the dirt-cap punch-through), runs until target/step cap.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from craft import mine


class _FakePos:
    """Stand-in for a requests Response whose .json() yields a position dict."""

    def __init__(self, y: int):
        self._y = y

    def json(self):
        return {"x": 0, "y": self._y, "z": 0, "yaw": 0.0}


def _run_staircase(*, y_sequence, count_sequence, quantity=8):
    """Drive tunnel_for_stone with scripted player-Y and inventory readings.

    `y_sequence` / `count_sequence` are popped once per /position-or-count read;
    the last value repeats once exhausted. Returns (result, excavate_calls).
    """
    ys = list(y_sequence)
    counts = list(count_sequence)
    excavate_calls = {"n": 0}

    def _fake_get(url, *a, **k):
        # tunnel_for_stone only GETs /position; pop the next scripted Y.
        y = ys.pop(0) if len(ys) > 1 else ys[0]
        return _FakePos(y)

    def _fake_count(_drops):
        return counts.pop(0) if len(counts) > 1 else counts[0]

    def _fake_excavate(*a, **k):
        excavate_calls["n"] += 1
        return {"success": True, "volume": 2, "remaining": 0}

    with patch.dict(os.environ, {"CRAFT_STONE_STAIRCASE": "1"}), \
            patch.object(mine.requests, "get", _fake_get), \
            patch.object(mine, "_count_drops", _fake_count), \
            patch.object(mine, "_step_is_safe", lambda *a, **k: (True, "ok")), \
            patch.object(mine, "_excavate_box", _fake_excavate):
        result = mine.tunnel_for_stone(quantity)
    return result, excavate_calls["n"]


def test_bails_when_no_real_progress():
    """Player never descends and never gains cobble → bail at the stuck limit,
    NOT after burning the full 150s wall-clock budget (the issue #10 hang)."""
    # Initial /position read returns y=70; every subsequent step also y=70
    # (player wedged). Inventory flat at 24. Baritone reports volume cleared.
    result, calls = _run_staircase(y_sequence=[70], count_sequence=[24])
    assert result is None  # nothing acquired
    # One excavate per step; bail at _STONE_STUCK_LIMIT consecutive stuck steps.
    assert calls == mine._STONE_STUCK_LIMIT, (
        f"expected bail after {mine._STONE_STUCK_LIMIT} stuck steps, "
        f"got {calls} excavate calls"
    )


def test_no_bail_while_descending_without_drops():
    """Player descends 1 Y/step (real progress) but gathers no cobble — the
    dirt-cap punch-through. Must NOT bail on the flat inventory; runs to the
    step cap (or min-Y), proving descent alone keeps the guard satisfied."""
    # initial pos y=70, then a long monotonic descent. Inventory stays flat.
    descent = [70] + list(range(69, 69 - 40, -1))
    result, calls = _run_staircase(y_sequence=descent, count_sequence=[24])
    assert result is None  # never reached target, but...
    # ...it kept going: more steps than the stuck limit (no premature bail).
    assert calls > mine._STONE_STUCK_LIMIT
    assert calls <= mine._STONE_STAIR_MAX_STEPS


def test_returns_tunnel_when_target_met():
    """Acquiring the requested quantity returns 'tunnel' and stops early."""
    # Descend normally; inventory jumps to target on the 2nd count read.
    result, calls = _run_staircase(
        y_sequence=[70, 69, 68, 67],
        count_sequence=[24, 32],  # before=24, target=32 reached immediately
        quantity=8,
    )
    assert result == "tunnel"
    assert calls == 1  # stopped as soon as target met
