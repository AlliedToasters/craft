"""Tests for the sleep_in_bed tool + dusk_bed loadout.

These are scaffolding tests written before the homunculus /bed/* endpoints
land. They exercise the craft-side composite logic by mocking the
_post_homunculus / _get_homunculus calls. When the Java side ships,
the live test in scripts/sleep_loadout_test.sh is what validates the
end-to-end wiring.

Test ordering (per RED-GREEN discipline):
- Schema/dispatch/loadout tests should pass right now.
- Composite-flow tests should pass right now (mocked endpoints).
- The live loadout-test script is the RED canary: it'll fail with
  transport_error against current homunculus, then pass once the
  spec'd endpoints exist.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from craft import tools
from craft.loadouts import LOADOUTS


class TestLoadoutEntry:
    def test_dusk_bed_exists(self):
        assert "dusk_bed" in LOADOUTS

    def test_dusk_bed_has_a_bed(self):
        main = dict((item, count) for item, count in LOADOUTS["dusk_bed"]["main"])
        bed_items = [k for k in main if k.endswith("_bed")]
        assert bed_items, f"dusk_bed must include a bed item; got {list(main)}"
        for k in bed_items:
            assert main[k] >= 1

    def test_dusk_bed_has_food(self):
        main = dict((item, count) for item, count in LOADOUTS["dusk_bed"]["main"])
        food_keys = [k for k in main if "beef" in k or "porkchop" in k or "bread" in k]
        assert food_keys, "dusk_bed should include some food for survival buffer"

    def test_dusk_bed_armor_is_empty(self):
        """Intentionally no armor — capability test is for sleep, not
        survival ceiling. Mirrors iron_armored's pattern of giving only
        what the tested capability needs."""
        assert LOADOUTS["dusk_bed"]["armor"] == {}


class TestSchemaShape:
    def test_sleep_in_bed_in_tools_list(self):
        names = [t["function"]["name"] for t in tools.TOOLS]
        assert "sleep_in_bed" in names

    def test_sleep_in_bed_has_no_required_args(self):
        spec = next(
            t for t in tools.TOOLS
            if t["function"]["name"] == "sleep_in_bed"
        )
        params = spec["function"]["parameters"]
        assert params["properties"] == {}, (
            "sleep_in_bed should expose no args to the agent — "
            "got " + repr(params["properties"])
        )
        assert params["required"] == []

    def test_dispatch_wired(self):
        assert tools.HANDLERS.get("sleep_in_bed") is tools.handle_sleep_in_bed


class TestCompositeFlow:
    """Mock the two homunculus paths and verify the place-if-needed +
    poll-until-wake state machine."""

    def _patch_endpoints(self, post_responses: list, get_responses: list):
        """Patch _post_homunculus to return successive entries from
        post_responses, and _get_homunculus likewise. _time.sleep is
        no-op'd so the polling loop runs at full speed.
        """
        post_iter = iter(post_responses)
        get_iter = iter(get_responses)

        def fake_post(path, payload, *, timeout=10.0):
            return next(post_iter)

        def fake_get(path, *, params=None, timeout=5.0):
            return next(get_iter)

        return (
            patch.object(tools, "_post_homunculus", side_effect=fake_post),
            patch.object(tools, "_get_homunculus", side_effect=fake_get),
            patch("time.sleep", lambda *_a, **_k: None),
        )

    def _run(self, post_responses, get_responses):
        post_patch, get_patch, sleep_patch = self._patch_endpoints(
            post_responses, get_responses,
        )
        with post_patch, get_patch, sleep_patch:
            return tools.handle_sleep_in_bed({})

    def test_sleep_succeeds_on_first_call_no_place(self):
        """Happy path: bed already nearby, sleep enters immediately, wakes
        on second stats poll."""
        out = self._run(
            post_responses=[
                {
                    "success": True,
                    "bed": [10, 64, 20],
                    "day_ticks_at_sleep": 13000,
                },
            ],
            get_responses=[
                {"is_sleeping": True, "day_ticks": 13050, "is_night": True},
                {"is_sleeping": False, "day_ticks": 23500, "is_night": False},
            ],
        )
        assert "slept" in out
        assert "natural_dawn" in out
        assert "FAILED" not in out

    def test_no_bed_nearby_triggers_place_then_sleep(self):
        """RED: first /bed/sleep returns no_bed_nearby. Composite calls
        /bed/place then retries /bed/sleep, then waits for wake."""
        out = self._run(
            post_responses=[
                {"success": False, "reason": "no_bed_nearby", "message": "no bed within 6"},
                {"success": True, "foot": [10, 64, 20], "head": [10, 64, 21], "facing": "north"},
                {"success": True, "bed": [10, 64, 20], "day_ticks_at_sleep": 13000},
            ],
            get_responses=[
                {"is_sleeping": False, "day_ticks": 23500, "is_night": False},
            ],
        )
        assert "slept" in out
        assert "FAILED" not in out

    def test_place_failure_bubbles_up(self):
        out = self._run(
            post_responses=[
                {"success": False, "reason": "no_bed_nearby", "message": ""},
                {"success": False, "reason": "no_bed_in_inventory", "message": "no *_bed"},
            ],
            get_responses=[],
        )
        assert out.startswith("FAILED")
        assert "no_bed_in_inventory" in out

    def test_second_sleep_failure_bubbles_up(self):
        out = self._run(
            post_responses=[
                {"success": False, "reason": "no_bed_nearby", "message": ""},
                {"success": True, "foot": [10, 64, 20], "head": [10, 64, 21], "facing": "north"},
                {"success": False, "reason": "monsters_nearby", "message": "zombie 3 blocks away"},
            ],
            get_responses=[],
        )
        assert out.startswith("FAILED")
        assert "monsters_nearby" in out

    def test_not_night_no_place_attempted(self):
        """When sleep fails with not_night, the composite must NOT try
        to place a bed — the issue is timing, not bed availability."""
        post_calls = []

        def fake_post(path, payload, *, timeout=10.0):
            post_calls.append(path)
            return {"success": False, "reason": "not_night", "message": "it is day"}

        with patch.object(tools, "_post_homunculus", side_effect=fake_post), \
             patch.object(tools, "_get_homunculus", return_value={}), \
             patch("time.sleep", lambda *_a, **_k: None):
            out = tools.handle_sleep_in_bed({})

        assert out.startswith("FAILED")
        assert "not_night" in out
        assert post_calls == ["/bed/sleep"], (
            f"expected exactly one /bed/sleep call, got {post_calls}"
        )

    def test_stats_missing_is_sleeping_surfaces_substrate_gap(self):
        """RED canary: if /stats doesn't expose is_sleeping (homunculus
        not updated yet), the handler must surface that, not loop forever.
        """
        out = self._run(
            post_responses=[
                {"success": True, "bed": [10, 64, 20], "day_ticks_at_sleep": 13000},
            ],
            get_responses=[
                {"day_ticks": 13050},  # is_sleeping absent
            ],
        )
        assert out.startswith("FAILED")
        assert "is_sleeping" in out

    def test_interrupted_wake_labeled(self):
        """If is_night is still true at wake, it was an interrupt, not
        natural dawn."""
        out = self._run(
            post_responses=[
                {"success": True, "bed": [10, 64, 20], "day_ticks_at_sleep": 13000},
            ],
            get_responses=[
                {"is_sleeping": False, "day_ticks": 14000, "is_night": True},
            ],
        )
        assert "interrupted" in out

    def test_transport_error_on_first_sleep_does_not_loop(self):
        """If /bed/sleep returns transport_error (homunculus down/missing
        endpoint), bubble up — don't try to place a bed in response."""
        out = self._run(
            post_responses=[
                {"success": False, "reason": "transport_error",
                 "message": "Connection refused"},
            ],
            get_responses=[],
        )
        assert out.startswith("FAILED")
        assert "transport_error" in out
