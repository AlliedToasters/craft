"""Equipment readout: per-slot best-of-class block in the STATE block.

The model's strategic nudge is structural — a vacant slot ("you have no
helmet!") signals "craft this" without prose. These tests pin:
  - tier resolution (iron beats stone beats wooden beats nothing)
  - armor located in equipped /inventory armor slots is detected
  - armor in main inventory (not yet equipped) is also detected — homunculus
    will auto-equip on next /equip cycle, so reporting it as "best" matches
    what the agent is about to wield
  - vacant phrasing per slot, exactly as the model sees it
"""

import os

import pytest

from craft.agent import (
    _all_item_ids,
    _armor_nudge_gating_enabled,
    _best_craftable_armor_material,
    _best_tier_id,
    _equipment_readout_enabled,
    _format_inventory,
    _render_equipment,
    _ARMOR_TIERS,
    _TOOL_TIERS,
)


@pytest.fixture(autouse=True)
def _equipment_readout_on(monkeypatch):
    """Default the toggle ON for every test; specific tests override."""
    monkeypatch.setenv("CRAFT_EQUIPMENT_READOUT", "1")


def _mk(main=None, offhand=None, head=None, chest=None, legs=None, feet=None):
    """Construct an /inventory-shaped dict for tests.

    main: list of {"id": str, "count": int, "slot": int} or list of ids
    offhand: id (str) or None
    armor: id (str) or None per slot
    """
    main_list = []
    if main:
        for i, item in enumerate(main):
            if isinstance(item, str):
                main_list.append({"slot": i, "id": item, "count": 1})
            else:
                main_list.append(item)
    return {
        "main": main_list,
        "offhand": {"id": offhand, "count": 1} if offhand else None,
        "armor": {
            "head":  {"id": head,  "count": 1} if head  else None,
            "chest": {"id": chest, "count": 1} if chest else None,
            "legs":  {"id": legs,  "count": 1} if legs  else None,
            "feet":  {"id": feet,  "count": 1} if feet  else None,
        },
        "selected_slot": 0,
    }


# ---------------------------------------------- _all_item_ids


class TestAllItemIds:
    def test_empty(self):
        assert _all_item_ids({}) == set()

    def test_none(self):
        assert _all_item_ids(None) == set()

    def test_main_only(self):
        inv = _mk(main=["minecraft:wooden_pickaxe", "minecraft:dirt"])
        assert _all_item_ids(inv) == {"minecraft:wooden_pickaxe", "minecraft:dirt"}

    def test_includes_offhand(self):
        inv = _mk(main=["minecraft:dirt"], offhand="minecraft:torch")
        assert _all_item_ids(inv) == {"minecraft:dirt", "minecraft:torch"}

    def test_includes_armor_slots(self):
        inv = _mk(head="minecraft:iron_helmet", feet="minecraft:leather_boots")
        assert _all_item_ids(inv) == {"minecraft:iron_helmet", "minecraft:leather_boots"}

    def test_armor_slot_with_none_does_not_leak(self):
        # Empty armor slots store None — not a {"id": None} dict.
        inv = _mk()
        assert _all_item_ids(inv) == set()


# ---------------------------------------------- _best_tier_id


class TestBestTierId:
    def test_no_match(self):
        assert _best_tier_id(set(), "sword", _TOOL_TIERS) is None

    def test_wooden(self):
        ids = {"minecraft:wooden_pickaxe", "minecraft:dirt"}
        assert _best_tier_id(ids, "pickaxe", _TOOL_TIERS) == "minecraft:wooden_pickaxe"

    def test_iron_beats_wooden(self):
        ids = {"minecraft:wooden_sword", "minecraft:iron_sword"}
        assert _best_tier_id(ids, "sword", _TOOL_TIERS) == "minecraft:iron_sword"

    def test_diamond_beats_iron(self):
        ids = {"minecraft:iron_pickaxe", "minecraft:diamond_pickaxe"}
        assert _best_tier_id(ids, "pickaxe", _TOOL_TIERS) == "minecraft:diamond_pickaxe"

    def test_netherite_is_top(self):
        ids = {"minecraft:diamond_axe", "minecraft:netherite_axe"}
        assert _best_tier_id(ids, "axe", _TOOL_TIERS) == "minecraft:netherite_axe"

    def test_armor_chainmail_beats_leather(self):
        ids = {"minecraft:leather_helmet", "minecraft:chainmail_helmet"}
        assert _best_tier_id(ids, "helmet", _ARMOR_TIERS) == "minecraft:chainmail_helmet"

    def test_armor_iron_beats_chainmail(self):
        ids = {"minecraft:chainmail_chestplate", "minecraft:iron_chestplate"}
        assert _best_tier_id(ids, "chestplate", _ARMOR_TIERS) == "minecraft:iron_chestplate"

    def test_suffix_isolation(self):
        # iron_ingot must NOT match a pickaxe/sword lookup.
        ids = {"minecraft:iron_ingot"}
        assert _best_tier_id(ids, "pickaxe", _TOOL_TIERS) is None
        assert _best_tier_id(ids, "sword", _TOOL_TIERS) is None


# ---------------------------------------------- _render_equipment


class TestRenderEquipment:
    def test_empty_inventory_tools_vacant_armor_omitted(self):
        # Tools always emit a vacant line (wood-tier is craftable from logs
        # alone; the nudge is actionable). Armor is gated on craftability —
        # without leather/iron/etc. in inventory, armor lines are omitted to
        # avoid steering the model into hallucinating wooden_helmet.
        lines = _render_equipment({})
        assert lines[0] == "Equipment:"
        body = "\n".join(lines[1:])
        assert "best weapon: you are unarmed! (no sword crafted yet)" in body
        assert "best shovel: you are digging barehanded! (no shovel crafted yet)" in body
        assert "best pickaxe: you cannot mine stone yet! (no pickaxe crafted yet)" in body
        assert "best axe: you are chopping barehanded! (no axe crafted yet)" in body
        for slot in ("helmet", "chestplate", "leggings", "boots"):
            assert slot not in body

    def test_wood_pickaxe_only(self):
        inv = _mk(main=["minecraft:wooden_pickaxe"])
        body = "\n".join(_render_equipment(inv))
        assert "best pickaxe: wooden_pickaxe" in body
        # Other tools still vacant.
        assert "best weapon: you are unarmed!" in body
        assert "best axe: you are chopping barehanded!" in body

    def test_mixed_tier_picks_best(self):
        inv = _mk(main=[
            "minecraft:wooden_sword",
            "minecraft:iron_sword",
            "minecraft:wooden_pickaxe",
        ])
        body = "\n".join(_render_equipment(inv))
        assert "best weapon: iron_sword" in body
        assert "best pickaxe: wooden_pickaxe" in body

    def test_armor_in_equipped_slots(self):
        inv = _mk(
            head="minecraft:iron_helmet",
            chest="minecraft:iron_chestplate",
            legs="minecraft:iron_leggings",
            feet="minecraft:iron_boots",
        )
        body = "\n".join(_render_equipment(inv))
        assert "helmet: iron_helmet" in body
        assert "chestplate: iron_chestplate" in body
        assert "leggings: iron_leggings" in body
        assert "boots: iron_boots" in body

    def test_armor_in_main_only_still_detected(self):
        # Agent just crafted iron_helmet but homunculus hasn't /equipped it yet.
        # Equipment readout should still surface it as the best helmet.
        inv = _mk(main=["minecraft:iron_helmet"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: iron_helmet" in body

    def test_armor_main_beats_equipped_if_higher_tier(self):
        # Wearing leather_helmet, just crafted iron_helmet in main.
        inv = _mk(main=["minecraft:iron_helmet"], head="minecraft:leather_helmet")
        body = "\n".join(_render_equipment(inv))
        assert "helmet: iron_helmet" in body
        assert "leather_helmet" not in body


# ---------------------------------------------- armor craftability gating


class TestBestCraftableArmorMaterial:
    def test_empty(self):
        assert _best_craftable_armor_material(set()) is None

    def test_logs_alone_dont_qualify(self):
        # The bug: model confabulates wooden_helmet (no such recipe). Logs
        # in inventory must NOT imply armor is craftable.
        assert _best_craftable_armor_material({"minecraft:oak_log"}) is None
        assert _best_craftable_armor_material({"minecraft:birch_planks"}) is None

    def test_leather_qualifies(self):
        assert _best_craftable_armor_material({"minecraft:leather"}) == "leather"

    def test_iron_qualifies(self):
        assert _best_craftable_armor_material({"minecraft:iron_ingot"}) == "iron"

    def test_iron_beats_leather(self):
        ids = {"minecraft:leather", "minecraft:iron_ingot"}
        assert _best_craftable_armor_material(ids) == "iron"

    def test_diamond_beats_iron(self):
        ids = {"minecraft:iron_ingot", "minecraft:diamond"}
        assert _best_craftable_armor_material(ids) == "diamond"

    def test_netherite_is_top(self):
        ids = {"minecraft:diamond", "minecraft:netherite_ingot"}
        assert _best_craftable_armor_material(ids) == "netherite"

    def test_raw_iron_does_not_qualify(self):
        # Raw iron needs smelting; armor recipe needs ingots.
        assert _best_craftable_armor_material({"minecraft:raw_iron"}) is None


class TestArmorGating:
    """Armor lines only render when equipped OR materials are present.

    Pins the fix for the 2026-05-20 wooden_helmet loop: with no armor and no
    materials, agent saw 5/5 → wooden_helmet (non-existent recipe). The
    gating removes the prompt pressure when no recipe could succeed.
    """

    def test_no_armor_no_materials_omits_all_armor(self):
        inv = _mk(main=["minecraft:wooden_pickaxe", "minecraft:birch_log"])
        body = "\n".join(_render_equipment(inv))
        for slot in ("helmet", "chestplate", "leggings", "boots"):
            assert slot not in body

    def test_leather_in_inventory_surfaces_leather_armor_hints(self):
        inv = _mk(main=["minecraft:leather"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: none — leather_helmet craftable" in body
        assert "chestplate: none — leather_chestplate craftable" in body
        assert "leggings: none — leather_leggings craftable" in body
        assert "boots: none — leather_boots craftable" in body

    def test_iron_ingot_surfaces_iron_armor_hints(self):
        inv = _mk(main=["minecraft:iron_ingot"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: none — iron_helmet craftable" in body
        assert "boots: none — iron_boots craftable" in body

    def test_iron_supersedes_leather_in_hint_text(self):
        # Best-tier material wins; agent crafts iron not leather.
        inv = _mk(main=["minecraft:leather", "minecraft:iron_ingot"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: none — iron_helmet craftable" in body
        assert "leather_helmet" not in body

    def test_equipped_armor_overrides_hint(self):
        # Wearing leather_helmet AND has iron — readout shows what's worn,
        # not a hint to upgrade. (Upgrade is the model's job to spot.)
        inv = _mk(head="minecraft:leather_helmet", main=["minecraft:iron_ingot"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: leather_helmet" in body
        # Other slots: no equipped, iron available → hint.
        assert "chestplate: none — iron_chestplate craftable" in body

    def test_partial_armor_some_equipped_others_hinted(self):
        inv = _mk(
            head="minecraft:iron_helmet",
            chest="minecraft:iron_chestplate",
            main=["minecraft:iron_ingot"],
        )
        body = "\n".join(_render_equipment(inv))
        assert "helmet: iron_helmet" in body
        assert "chestplate: iron_chestplate" in body
        assert "leggings: none — iron_leggings craftable" in body
        assert "boots: none — iron_boots craftable" in body

    def test_tools_still_emit_vacant_phrases(self):
        # Gating is armor-only — tool nudges always fire (they're actionable).
        inv = _mk(main=["minecraft:dirt"])
        body = "\n".join(_render_equipment(inv))
        assert "best weapon: you are unarmed!" in body
        assert "best pickaxe: you cannot mine stone yet!" in body


# ---------------------------------------------- _format_inventory


class TestFormatInventory:
    def test_none_returns_none(self):
        # Transport-error sentinel propagates so the STATE chunk shows
        # "(unavailable …)" instead of silently dropping the inventory.
        assert _format_inventory(None) is None

    def test_empty_renders_block_and_empty_marker(self):
        out = _format_inventory(_mk())
        assert out is not None
        assert "Equipment:" in out
        assert "Current inventory:" in out
        assert "(empty)" in out

    def test_populated_inventory_lists_slots(self):
        inv = _mk(main=[
            {"slot": 1, "id": "minecraft:spruce_log", "count": 9},
            {"slot": 6, "id": "minecraft:wooden_pickaxe", "count": 1},
        ])
        out = _format_inventory(inv)
        assert "slot 1: 9x minecraft:spruce_log" in out
        assert "slot 6: 1x minecraft:wooden_pickaxe" in out
        # Equipment line for pickaxe should also reflect it.
        assert "best pickaxe: wooden_pickaxe" in out

    def test_offhand_rendered(self):
        inv = _mk(main=["minecraft:dirt"], offhand="minecraft:torch")
        out = _format_inventory(inv)
        assert "offhand: 1x minecraft:torch" in out

    def test_block_order_equipment_then_inventory(self):
        # Equipment must come first — the model reads top-down and the
        # structured nudge is the headline; raw slots are accounting detail.
        inv = _mk(main=["minecraft:wooden_pickaxe"])
        out = _format_inventory(inv)
        eq_idx = out.find("Equipment:")
        inv_idx = out.find("Current inventory:")
        assert 0 <= eq_idx < inv_idx


# ---------------------------------------------- A/B toggle


class TestEquipmentReadoutToggle:
    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("CRAFT_EQUIPMENT_READOUT", raising=False)
        assert _equipment_readout_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "no", "FALSE", "Off"])
    def test_disabled_values(self, monkeypatch, val):
        monkeypatch.setenv("CRAFT_EQUIPMENT_READOUT", val)
        assert _equipment_readout_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", "anything"])
    def test_enabled_values(self, monkeypatch, val):
        monkeypatch.setenv("CRAFT_EQUIPMENT_READOUT", val)
        assert _equipment_readout_enabled() is True

    def test_format_inventory_off_omits_equipment_block(self, monkeypatch):
        monkeypatch.setenv("CRAFT_EQUIPMENT_READOUT", "0")
        inv = _mk(main=["minecraft:wooden_pickaxe", "minecraft:dirt"])
        out = _format_inventory(inv)
        assert "Equipment:" not in out
        assert "best pickaxe" not in out
        # Raw slot listing remains intact.
        assert "Current inventory:" in out
        assert "slot 0: 1x minecraft:wooden_pickaxe" in out

    def test_format_inventory_on_includes_equipment_block(self, monkeypatch):
        monkeypatch.setenv("CRAFT_EQUIPMENT_READOUT", "1")
        inv = _mk(main=["minecraft:wooden_pickaxe"])
        out = _format_inventory(inv)
        assert "Equipment:" in out
        assert "best pickaxe: wooden_pickaxe" in out


# ---------------------------------------------- armor-nudge-gating toggle


class TestArmorNudgeGatingToggle:
    """The gating fix can be reverted for A/B regression studies.

    `CRAFT_ARMOR_NUDGE_GATING=0` restores the pre-2026-05-20 behavior:
    every vacant armor slot emits "you have no <slot>!". Used to measure
    the impact of the fix against the same code path that produced the
    1956 wooden_helmet attempts in the previous campaign.
    """

    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("CRAFT_ARMOR_NUDGE_GATING", raising=False)
        assert _armor_nudge_gating_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "no", "FALSE", "Off"])
    def test_disabled_values(self, monkeypatch, val):
        monkeypatch.setenv("CRAFT_ARMOR_NUDGE_GATING", val)
        assert _armor_nudge_gating_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", "anything"])
    def test_enabled_values(self, monkeypatch, val):
        monkeypatch.setenv("CRAFT_ARMOR_NUDGE_GATING", val)
        assert _armor_nudge_gating_enabled() is True

    def test_gating_off_restores_legacy_nudge_for_all_slots(self, monkeypatch):
        monkeypatch.setenv("CRAFT_ARMOR_NUDGE_GATING", "0")
        body = "\n".join(_render_equipment({}))
        assert "helmet: you have no helmet! (no helmet crafted yet)" in body
        assert "chestplate: you have no chestplate! (no chestplate crafted yet)" in body
        assert "leggings: you have no leggings! (no leggings crafted yet)" in body
        assert "boots: you have no boots! (no boots crafted yet)" in body

    def test_gating_off_still_shows_equipped_armor(self, monkeypatch):
        monkeypatch.setenv("CRAFT_ARMOR_NUDGE_GATING", "0")
        inv = _mk(head="minecraft:iron_helmet")
        body = "\n".join(_render_equipment(inv))
        assert "helmet: iron_helmet" in body
        # Other slots fall back to legacy nudge.
        assert "chestplate: you have no chestplate!" in body

    def test_gating_off_ignores_materials_in_inventory(self, monkeypatch):
        # Even with leather, legacy mode shows the nag (no recipe hint).
        monkeypatch.setenv("CRAFT_ARMOR_NUDGE_GATING", "0")
        inv = _mk(main=["minecraft:leather"])
        body = "\n".join(_render_equipment(inv))
        assert "helmet: you have no helmet!" in body
        assert "leather_helmet craftable" not in body
