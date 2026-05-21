"""Tests for the cook_kitchen loadout.

cook_kitchen is the cooking-capability isolation loadout: raw meat +
fuel + furnace + hunger pressure, no new tool. The agent must self-place
the furnace, smelt, collect, and let AutoEat fire. Composes from
existing place/smelt/collect_smelt primitives.

The live test is scripts/cook_loadout_test.sh against a real homunculus.
These unit tests cover the loadout shape only.
"""

from __future__ import annotations

import pytest

from craft.loadouts import LOADOUTS


class TestCookKitchen:
    def test_exists(self):
        assert "cook_kitchen" in LOADOUTS

    def test_has_raw_meat_hidden_from_hotbar(self):
        """Raw meat must be in `main_inv_only` (slots 9+), NOT `main`
        (hotbar). Wurst AutoEat only eats from hotbar — if raw meat
        lands there via /give, it's auto-consumed before the agent can
        cook (validated 2026-05-21 smoke)."""
        spec = LOADOUTS["cook_kitchen"]
        main = dict((item, count) for item, count in spec.get("main", []))
        main_inv = dict((item, count) for item, count in spec.get("main_inv_only", []))

        raw_meats_in_main = [
            k for k in main if k in {
                "minecraft:beef", "minecraft:porkchop", "minecraft:mutton",
                "minecraft:chicken", "minecraft:rabbit",
            }
        ]
        assert not raw_meats_in_main, (
            f"raw meat found in hotbar (main): {raw_meats_in_main}. "
            "AutoEat will consume it. Move to main_inv_only."
        )

        raw_meats_hidden = [
            k for k in main_inv if k in {
                "minecraft:beef", "minecraft:porkchop", "minecraft:mutton",
                "minecraft:chicken", "minecraft:rabbit",
            }
        ]
        assert raw_meats_hidden, (
            f"cook_kitchen must include raw meat in main_inv_only; "
            f"got main={list(main)}, main_inv_only={list(main_inv)}"
        )
        for k in raw_meats_hidden:
            assert main_inv[k] >= 4, (
                f"give enough raw meat to test multi-cycle ({k}={main_inv[k]})"
            )

    def test_has_fuel(self):
        main = dict((item, count) for item, count in LOADOUTS["cook_kitchen"]["main"])
        fuels = [
            k for k in main
            if k in {"minecraft:coal", "minecraft:charcoal"}
        ]
        assert fuels, f"cook_kitchen needs fuel for the furnace; got {list(main)}"

    def test_has_furnace(self):
        main = dict((item, count) for item, count in LOADOUTS["cook_kitchen"]["main"])
        assert main.get("minecraft:furnace", 0) >= 1, (
            f"cook_kitchen must include a furnace block; got {list(main)}"
        )

    def test_has_hunger_pressure(self):
        assert LOADOUTS["cook_kitchen"].get("set_hunger") == 2

    def test_armor_empty(self):
        assert LOADOUTS["cook_kitchen"]["armor"] == {}

    def test_no_cooked_food(self):
        """If we pre-give cooked food, AutoEat fires immediately and
        cooking pressure evaporates — defeats the test."""
        spec = LOADOUTS["cook_kitchen"]
        all_items = (
            list(spec.get("main", [])) + list(spec.get("main_inv_only", []))
        )
        cooked = [
            k for k, _ in all_items
            if k.startswith("minecraft:cooked_") or k in {
                "minecraft:bread", "minecraft:apple", "minecraft:cookie",
                "minecraft:melon_slice", "minecraft:carrot",
            }
        ]
        assert not cooked, (
            f"cook_kitchen should NOT include cooked/ready food; found {cooked}"
        )

    def test_fuel_matches_meat_ratio(self):
        """Each coal smelts 8 items. A loadout with 8x raw meat and 1x
        coal is exact-match; more coal is fine (wasted), less is starvation
        risk. Verify the user-visible ratio is sane (≥1 coal per 8 raw).
        Raw meat lives in main_inv_only (hidden from AutoEat)."""
        spec = LOADOUTS["cook_kitchen"]
        all_items = dict(
            (k, v) for k, v in (
                list(spec.get("main", [])) + list(spec.get("main_inv_only", []))
            )
        )
        raw_meat_total = sum(
            v for k, v in all_items.items()
            if k in {
                "minecraft:beef", "minecraft:porkchop", "minecraft:mutton",
                "minecraft:chicken", "minecraft:rabbit",
            }
        )
        coal_count = (
            all_items.get("minecraft:coal", 0)
            + all_items.get("minecraft:charcoal", 0)
        )
        required = (raw_meat_total + 7) // 8
        assert coal_count >= required, (
            f"need at least {required} coal for {raw_meat_total} raw meat; "
            f"got {coal_count}"
        )
