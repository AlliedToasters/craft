"""Unit tests for the salvage-wood fallback (issue #7).

On wood-deficient spawns (deep caves, mineshafts, snowy peaks) there are no
logs in scan range, but structures hold worked wood. `handle_mine_wood` first
tries the normal log candidate cycle; only if that finds NOTHING reachable
does it fall back to mining structure *planks*.

Design decisions under test:
  - Salvage is *planks only*. Doors / fences / slabs / stairs drop themselves
    with no recipe back to planks → a dead item, so they're excluded.
  - The fallback fires only on a clean "no candidate reachable" log failure —
    NOT on a partial log haul (returned as progress) and NOT on a tool failure.
  - The salvage delta is self-contained (its own before/after on the plank drop
    set), so a no-op "already had N" salvage result must not masquerade as a win.
"""

from __future__ import annotations

from unittest.mock import patch

from craft import mine, tools


# ------------------------------------------------- the salvage set is planks-only


def test_salvage_set_is_planks_only():
    # Every salvage candidate must be a *_planks block — doors/fences/slabs/
    # stairs are deliberately excluded (no recipe back to planks).
    assert mine.SALVAGE_WOOD_TYPES, "salvage set must be non-empty"
    for t in mine.SALVAGE_WOOD_TYPES:
        assert t.endswith("_planks"), f"{t} is not a planks block"
    # Drop set mirrors the type list, namespaced.
    assert tools.SALVAGE_WOOD_DROPS == {
        f"minecraft:{t}" for t in mine.SALVAGE_WOOD_TYPES
    }
    # Salvage and logs are disjoint (salvage is a genuine fallback, not overlap).
    assert tools.SALVAGE_WOOD_DROPS.isdisjoint(tools.LOG_DROPS)


# ------------------------------------------------- mine_any_salvage_wood plumbing


def test_mine_any_salvage_wood_probes_plank_candidates():
    captured = {}

    def _stub(quantity, candidates, *, probe_radius, probe_y_radius):
        captured["quantity"] = quantity
        captured["candidates"] = candidates
        captured["probe_y_radius"] = probe_y_radius
        return "oak_planks"

    with patch.object(mine, "_mine_first_reachable", _stub):
        out = mine.mine_any_salvage_wood(5)
    assert out == "oak_planks"
    assert captured["quantity"] == 5
    assert captured["candidates"] == mine.SALVAGE_WOOD_TYPES
    # Deep vertical band (mineshafts below a cave spawn), unlike the log probe.
    assert captured["probe_y_radius"] >= 32


# ------------------------------------------------- handle_mine_wood fallback logic


def _wood_with_delta_stub(stub):
    """Run handle_mine_wood with _handle_mine_delta replaced by `stub`."""
    with patch.object(tools, "_handle_mine_delta", stub):
        return tools.handle_mine_wood({"quantity": 3})


def test_logs_found_no_salvage_attempt():
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        return "acquired 3 more (now have 3 mine_wood-drops; last type mined: oak_log)"

    out = _wood_with_delta_stub(_stub)
    assert out.startswith("acquired 3 more")
    assert tools.SALVAGE_WOOD_DROPS not in seen  # salvage never tried


def test_no_logs_then_salvage_succeeds():
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        if drops == tools.LOG_DROPS:
            return "FAILED: no candidate reachable for mine_wood (acquired 0)"
        if drops == tools.SALVAGE_WOOD_DROPS:
            return ("acquired 3 more (now have 3 mine_wood-drops; "
                    "last type mined: oak_planks)")
        raise AssertionError(f"unexpected drops {drops}")

    out = _wood_with_delta_stub(_stub)
    assert out.startswith("[wood_source=salvage]")
    assert "acquired 3 more" in out
    assert "Planks ARE planks" in out
    assert tools.LOG_DROPS in seen and tools.SALVAGE_WOOD_DROPS in seen


def test_no_logs_and_no_salvage_returns_log_failure():
    def _stub(label, args, drops, miner, *, fair_miner=None):
        return "FAILED: no candidate reachable for mine_wood (acquired 0)"

    out = _wood_with_delta_stub(_stub)
    assert out == "FAILED: no candidate reachable for mine_wood (acquired 0)"
    assert "salvage" not in out


def test_partial_log_haul_not_overridden_by_salvage():
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        return ("PARTIAL: acquired 1 of 3 mine_wood-drops (now have 1). Cycle "
                "ended before target — call mine_wood again to keep gathering.")

    out = _wood_with_delta_stub(_stub)
    assert out.startswith("PARTIAL")
    assert tools.SALVAGE_WOOD_DROPS not in seen  # partial logs ≠ no logs


def test_salvage_noop_already_had_does_not_override():
    # Log path fails; salvage finds the plank target already satisfied (no new
    # blocks). That's not progress — keep the honest log failure.
    def _stub(label, args, drops, miner, *, fair_miner=None):
        if drops == tools.LOG_DROPS:
            return "FAILED: no candidate reachable for mine_wood (acquired 0)"
        return "acquired 0 more (already had 5 mine_wood-drops — target was already met)"

    out = _wood_with_delta_stub(_stub)
    assert out == "FAILED: no candidate reachable for mine_wood (acquired 0)"


def test_tool_failure_not_overridden_by_salvage():
    # A real tool failure (not a "no candidate" miss) must not trigger salvage —
    # the message is specific guidance the agent should see.
    seen = []

    def _stub(label, args, drops, miner, *, fair_miner=None):
        seen.append(drops)
        return "FAILED mine_wood: effective tool lost mid-mine (pickaxe broke?)"

    out = _wood_with_delta_stub(_stub)
    assert out.startswith("FAILED mine_wood:")
    assert tools.SALVAGE_WOOD_DROPS not in seen
