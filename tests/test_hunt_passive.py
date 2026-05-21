"""Tests for the hunt_passive tool + hunt_meadow/hunt_wild loadouts.

These exercise the craft-side composite logic with the network helpers
mocked. The live test is scripts/hunt_loadout_test.sh which validates
end-to-end against a real homunculus + Wurst.

Test ordering (per RED-GREEN discipline):
- Schema/dispatch/loadout tests should pass right now (pure shape).
- Composite-flow tests pass with mocked helpers.
- The live loadout-test script is the integration validator: it
  uncovers the actual KillAura passive-filter setting name + whether
  the Baritone goto-then-KillAura kill loop composes as designed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from craft import tools
from craft.loadouts import LOADOUTS


class TestLoadoutEntries:
    def test_hunt_meadow_exists(self):
        assert "hunt_meadow" in LOADOUTS

    def test_hunt_wild_exists(self):
        assert "hunt_wild" in LOADOUTS

    def test_hunt_meadow_pre_summons_herd(self):
        assert LOADOUTS["hunt_meadow"]["pre_summon_herd"] is True

    def test_hunt_wild_does_not_pre_summon(self):
        # Either absent or explicitly False — both mean "no summon".
        assert not LOADOUTS["hunt_wild"].get("pre_summon_herd")

    def test_both_loadouts_have_a_sword(self):
        for name in ("hunt_meadow", "hunt_wild"):
            main = dict((it, c) for it, c in LOADOUTS[name]["main"])
            swords = [k for k in main if k.endswith("_sword")]
            assert swords, f"{name} must include some sword; got {list(main)}"

    def test_both_loadouts_have_hunger_pressure(self):
        for name in ("hunt_meadow", "hunt_wild"):
            assert LOADOUTS[name].get("set_hunger") == 2, (
                f"{name} should start at hunger=2 to exercise the eat-cooked-"
                f"meat path"
            )

    def test_both_loadouts_armor_is_empty(self):
        for name in ("hunt_meadow", "hunt_wild"):
            assert LOADOUTS[name]["armor"] == {}

    def test_hunt_meadow_no_food_buffer(self):
        """Hunt-isolation loadout: agent should rely on hunting for food,
        not on a pre-stocked buffer. If beef/porkchop is in `main`, the
        hunger pressure is short-circuited by AutoEat."""
        main = dict((it, c) for it, c in LOADOUTS["hunt_meadow"]["main"])
        edibles = [
            k for k in main
            if any(p in k for p in ("beef", "porkchop", "mutton",
                                    "chicken", "bread", "apple"))
        ]
        assert not edibles, (
            f"hunt_meadow should NOT include cooked food (defeats the "
            f"hunger-pressure test); found {edibles}"
        )


class TestSchemaShape:
    def test_hunt_passive_in_tools_list(self):
        names = [t["function"]["name"] for t in tools.TOOLS]
        assert "hunt_passive" in names

    def test_hunt_passive_optional_radius(self):
        spec = next(
            t for t in tools.TOOLS
            if t["function"]["name"] == "hunt_passive"
        )
        params = spec["function"]["parameters"]
        assert "radius" in params["properties"]
        assert params["required"] == [], (
            "hunt_passive radius should be optional — default works fine"
        )

    def test_dispatch_wired(self):
        assert tools.HANDLERS.get("hunt_passive") is tools.handle_hunt_passive


_UNSET = object()


class TestCompositeFlow:
    """Mock the network helpers and verify the scan → goto → poll →
    restore-filter state machine."""

    def _run(
        self,
        *,
        nearest=_UNSET,
        goto_result=None,
        inv_seq=None,
        toggle_on=None,
        toggle_off=None,
        args=None,
    ):
        # `nearest` semantic: _UNSET → use a default cow target; None →
        # simulate "no passives found".
        # /scan_entities returns entities with `position: [x, y, z]` —
        # mock that exactly, not bare x/y/z fields.
        if nearest is _UNSET:
            nearest_val = (
                {"position": [10, 64, 20], "distance": 8.0}, "minecraft:cow",
            )
        else:
            nearest_val = nearest
        goto_result = goto_result if goto_result is not None else {"success": True}
        # inv_seq: each call to _inventory_drop_counts returns the next
        # snapshot. Final snapshot is the "after" used for the report.
        inv_seq = inv_seq if inv_seq is not None else [
            {},                                # before
            {"minecraft:beef": 2},             # mid-poll
            {"minecraft:beef": 2, "minecraft:leather": 1},  # final
        ]
        inv_iter = iter(inv_seq)

        def fake_inv():
            try:
                return next(inv_iter)
            except StopIteration:
                return inv_seq[-1]

        toggle_on = toggle_on if toggle_on is not None else {"success": True}
        toggle_off = toggle_off if toggle_off is not None else {"success": True}
        toggle_calls = []

        def fake_toggle(*, on):
            toggle_calls.append(on)
            return toggle_on if on else toggle_off

        # Shrink the kill-wait window so the real-time loop terminates
        # in milliseconds (time.sleep is mocked but time.time() still ticks).
        with patch.object(tools, "_nearest_passive", return_value=nearest_val), \
             patch.object(tools, "_baritone_goto", return_value=goto_result), \
             patch.object(tools, "_inventory_drop_counts", side_effect=fake_inv), \
             patch.object(tools, "_killaura_attack_passives", side_effect=fake_toggle), \
             patch.object(tools, "_HUNT_KILL_WAIT_S", 0.05), \
             patch.object(tools, "_HUNT_KILL_POLL_S", 0.01), \
             patch("time.sleep", lambda *_a, **_k: None):
            out = tools.handle_hunt_passive(args or {})
        return out, toggle_calls

    def test_happy_path_returns_drop_tally(self):
        out, _ = self._run()
        assert "hunted minecraft:cow" in out
        assert "gained" in out
        assert "beef" in out
        assert "leather" in out
        assert "FAILED" not in out

    def test_no_passives_in_range_reported(self):
        out, toggles = self._run(nearest=None)
        assert "no_passives_in_range" in out
        # Toggle should NOT have run if there was nothing to hunt — no
        # point flipping KillAura's filter for a no-op.
        assert toggles == []

    def test_no_drops_after_engage_returns_failed(self):
        out, _ = self._run(
            inv_seq=[{}, {}, {}],  # nothing changed
        )
        assert out.startswith("FAILED")
        assert "no_drops_observed" in out

    def test_toggle_off_runs_even_on_failed_kill(self):
        """KillAura's passive filter must be restored regardless of
        whether the hunt succeeded — leaving it off would silently change
        downstream tool behavior."""
        _, toggles = self._run(
            inv_seq=[{}, {}, {}],  # no drops → FAILED
        )
        assert toggles == [True, False], (
            f"expected [True (on), False (restore)], got {toggles}"
        )

    def test_toggle_off_runs_on_happy_path(self):
        _, toggles = self._run()
        assert toggles == [True, False]

    def test_toggle_failure_warns_but_does_not_abort(self):
        """If the KillAura filter toggle fails (wrong setting name etc.),
        the handler proceeds — KillAura may still attack passives in
        some configs, and the post-flight drop check is the actual
        signal."""
        out, toggles = self._run(
            toggle_on={"success": False, "reason": "setting_not_found",
                       "message": "no setting named 'Filter passive mobs'"},
        )
        # Despite the toggle failing, the mocked inventory shows drops,
        # so the handler reports success.
        assert "hunted minecraft:cow" in out
        # Both toggle attempts ran (restoration is best-effort).
        assert toggles == [True, False]

    def test_radius_clamped_to_max(self):
        """Args.radius > _HUNT_MAX_RADIUS should silently clamp, not
        reject — agent UX preference."""
        captured: list[int] = []

        def fake_nearest(r):
            captured.append(r)
            return None  # short-circuit to no_passives_in_range

        with patch.object(tools, "_nearest_passive", side_effect=fake_nearest), \
             patch("time.sleep", lambda *_a, **_k: None):
            tools.handle_hunt_passive({"radius": 999})
        assert captured == [tools._HUNT_MAX_RADIUS]

    def test_radius_clamped_to_min(self):
        captured: list[int] = []

        def fake_nearest(r):
            captured.append(r)
            return None

        with patch.object(tools, "_nearest_passive", side_effect=fake_nearest), \
             patch("time.sleep", lambda *_a, **_k: None):
            tools.handle_hunt_passive({"radius": 1})
        assert captured == [4]

    def test_reads_position_list_not_xyz_fields(self):
        """Regression for 2026-05-21: /scan_entities returns `position:
        [x, y, z]` lists, not bare x/y/z fields. Earlier handler read
        `target.get("x")` and short-circuited to FAILED on every kill.
        Haiku called hunt_passive 4× during the first fan-out and every
        one bounced off this bug."""
        captured: list[tuple[int, int, int]] = []

        def fake_goto(x, y, z, *, timeout_seconds=30, arrival_tolerance=2):
            captured.append((x, y, z))
            return {"success": True}

        # Sticky inventory mock — short window mocked, the polling loop
        # may iterate a few times before the loop's real-time deadline.
        inv_calls = [0]

        def fake_inv():
            inv_calls[0] += 1
            return {} if inv_calls[0] == 1 else {"minecraft:beef": 2}

        with patch.object(
            tools, "_nearest_passive",
            return_value=(
                {"position": [123, 64, 456], "distance": 7.0},
                "minecraft:cow",
            ),
        ), \
             patch.object(tools, "_baritone_goto", side_effect=fake_goto), \
             patch.object(tools, "_inventory_drop_counts", side_effect=fake_inv), \
             patch.object(
                tools, "_killaura_attack_passives",
                return_value={"success": True},
            ), \
             patch.object(tools, "_HUNT_KILL_WAIT_S", 0.05), \
             patch.object(tools, "_HUNT_KILL_POLL_S", 0.01), \
             patch("time.sleep", lambda *_a, **_k: None):
            out = tools.handle_hunt_passive({})

        assert "hunted minecraft:cow" in out
        assert "FAILED" not in out
        assert captured == [(123, 64, 456)], (
            f"goto should have been called with extracted position; got {captured}"
        )

    def test_goto_failure_still_polls_for_drops(self):
        """KillAura may have killed the mob at range during travel even
        if Baritone reports timeout — the drops are still the signal."""
        out, toggles = self._run(
            goto_result={"success": False, "reason": "timeout"},
        )
        # Drops appeared (mocked inventory delta) → success despite goto.
        assert "hunted minecraft:cow" in out
        assert toggles == [True, False]


class TestNearestPassive:
    """Verify the multi-species scan picks the geometrically closest
    entity (not just the first species with a hit)."""

    def test_picks_closest_across_species(self):
        scans = {
            "minecraft:cow":     [{"position": [0, 0, 0],  "distance": 12.0}],
            "minecraft:pig":     [{"position": [5, 0, 0],  "distance": 3.0}],
            "minecraft:sheep":   [],
            "minecraft:rabbit":  [],
            "minecraft:chicken": [{"position": [1, 0, 0],  "distance": 5.0}],
        }

        def fake_scan(t, *, radius, limit):
            return scans.get(t, [])

        with patch.object(tools, "_scan_entities_raw", side_effect=fake_scan):
            result = tools._nearest_passive(radius=32)
        assert result is not None
        ent, species = result
        assert species == "minecraft:pig"
        assert ent["distance"] == 3.0

    def test_returns_none_when_no_species_in_range(self):
        with patch.object(tools, "_scan_entities_raw", return_value=[]):
            assert tools._nearest_passive(radius=32) is None

    def test_skips_entities_with_no_distance_field(self):
        """Defensive — homunculus may omit distance on some entity
        snapshots (e.g., un-loaded chunk edge)."""
        scans = {
            "minecraft:cow":     [{"position": [0, 0, 0]}],                     # no distance
            "minecraft:pig":     [{"position": [5, 0, 0],  "distance": 7.0}],
            "minecraft:sheep":   [],
            "minecraft:rabbit":  [],
            "minecraft:chicken": [],
        }

        def fake_scan(t, *, radius, limit):
            return scans.get(t, [])

        with patch.object(tools, "_scan_entities_raw", side_effect=fake_scan):
            result = tools._nearest_passive(radius=32)
        assert result is not None
        _, species = result
        assert species == "minecraft:pig"


class TestInventoryDropCounts:
    def test_sums_main_and_offhand(self):
        inv = {
            "main": [
                {"id": "minecraft:beef", "count": 3},
                {"id": "minecraft:cobblestone", "count": 32},  # not a drop
                {"id": "minecraft:beef", "count": 2},          # second stack
            ],
            "offhand": {"id": "minecraft:leather", "count": 1},
        }
        with patch.object(tools, "_get_homunculus", return_value=inv):
            counts = tools._inventory_drop_counts()
        assert counts == {"minecraft:beef": 5, "minecraft:leather": 1}

    def test_returns_empty_on_transport_error(self):
        with patch.object(
            tools, "_get_homunculus",
            return_value={"success": False, "reason": "transport_error"},
        ):
            assert tools._inventory_drop_counts() == {}

    def test_ignores_non_drop_items(self):
        inv = {
            "main": [
                {"id": "minecraft:stone_sword", "count": 1},
                {"id": "minecraft:torch", "count": 8},
            ],
        }
        with patch.object(tools, "_get_homunculus", return_value=inv):
            assert tools._inventory_drop_counts() == {}
