"""Tiered AutoDrop policy: whitelist (keep) defined; drop list is the complement.

These tests pin the high-leverage invariants:
  - every keep-set id exists in the canonical registry (the self_check in
    autodrop.py runs this at import; this test catches reintroduction)
  - drop_list_for_tier returns a sorted complement that excludes everything
    in the layered keep set
  - tier ordering is monotone (later tiers keep a superset of earlier ones)
  - known-load-bearing items are kept across all tiers (food, ingots, gems,
    tools, doors, beds, fuel) and known-junk items are dropped (saplings,
    sand, gravel, andesite, basalt, all flowers, mossy variants, etc.)
"""

from __future__ import annotations

import pytest

from craft._item_registry import ALL_ITEMS
from craft.autodrop import (
    ALWAYS_KEEP,
    TIER_KEEP,
    drop_list_for_tier,
    keep_set_for_tier,
)


_TIERS = ("bare", "stone", "iron", "diamond")


class TestRegistryConsistency:
    def test_always_keep_subset_of_registry(self):
        # Self-check in autodrop.py runs at import time; this test pins it.
        bogus = ALWAYS_KEEP - ALL_ITEMS
        assert not bogus, f"ALWAYS_KEEP has non-registry ids: {sorted(bogus)}"

    @pytest.mark.parametrize("tier", _TIERS)
    def test_tier_keep_subset_of_registry(self, tier):
        bogus = TIER_KEEP[tier] - ALL_ITEMS
        assert not bogus, f"TIER_KEEP[{tier}] has non-registry ids: {sorted(bogus)}"

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            drop_list_for_tier("nether")


class TestKeepSetMonotone:
    """A later tier must keep everything an earlier tier kept."""

    def test_stone_keeps_everything_bare_keeps(self):
        assert keep_set_for_tier("bare") <= keep_set_for_tier("stone")

    def test_iron_keeps_everything_stone_keeps(self):
        assert keep_set_for_tier("stone") <= keep_set_for_tier("iron")

    def test_diamond_keeps_everything_iron_keeps(self):
        assert keep_set_for_tier("iron") <= keep_set_for_tier("diamond")


class TestDropList:
    @pytest.mark.parametrize("tier", _TIERS)
    def test_drop_list_is_sorted(self, tier):
        drops = drop_list_for_tier(tier)
        assert drops == sorted(drops)

    @pytest.mark.parametrize("tier", _TIERS)
    def test_drop_and_keep_partition_registry(self, tier):
        drops = set(drop_list_for_tier(tier))
        keeps = keep_set_for_tier(tier)
        assert drops.isdisjoint(keeps), "drop list overlaps keep set"
        assert drops | keeps == ALL_ITEMS, "drop ∪ keep != registry"

    def test_drop_count_decreases_with_tier(self):
        # Cumulative keep => non-increasing drop count.
        sizes = [len(drop_list_for_tier(t)) for t in _TIERS]
        assert sizes == sorted(sizes, reverse=True), f"non-monotone drop sizes: {sizes}"


class TestLoadBearingItemsKept:
    """Known-good items must be in the keep set across all tiers.

    These come from prior rollout observations + the substrate-thesis principle
    that anything a current-tier agent could plausibly *use* should never be
    dropped from inventory.
    """

    @pytest.mark.parametrize("item", [
        # Fuel + recipes
        "minecraft:coal", "minecraft:charcoal", "minecraft:stick",
        # Food
        "minecraft:bread", "minecraft:apple", "minecraft:cooked_beef",
        "minecraft:beef", "minecraft:chicken", "minecraft:cooked_chicken",
        # Ingots + raw ores
        "minecraft:iron_ingot", "minecraft:gold_ingot", "minecraft:raw_iron",
        "minecraft:netherite_ingot",
        # Gems
        "minecraft:diamond", "minecraft:emerald", "minecraft:lapis_lazuli",
        # Tools (any tier)
        "minecraft:wooden_pickaxe", "minecraft:iron_pickaxe",
        "minecraft:diamond_sword", "minecraft:netherite_axe",
        # Armor
        "minecraft:iron_helmet", "minecraft:diamond_chestplate",
        "minecraft:turtle_helmet",
        # Furniture critical to craft graph
        "minecraft:crafting_table", "minecraft:furnace", "minecraft:smithing_table",
        # Doors + beds + wool for shelter / skip-night
        "minecraft:oak_door", "minecraft:iron_door",
        "minecraft:white_bed", "minecraft:red_wool",
        # Rare mob drops
        "minecraft:leather", "minecraft:string", "minecraft:feather",
        "minecraft:gunpowder", "minecraft:ender_pearl", "minecraft:blaze_rod",
        "minecraft:nether_star",
        # Utility
        "minecraft:flint", "minecraft:flint_and_steel",
        "minecraft:bucket", "minecraft:water_bucket", "minecraft:lava_bucket",
        # Lighting
        "minecraft:torch",
    ])
    @pytest.mark.parametrize("tier", _TIERS)
    def test_load_bearing_item_kept(self, item, tier):
        assert item in keep_set_for_tier(tier), (
            f"load-bearing item {item} would be dropped at tier={tier}"
        )


class TestWoodTierIncludesAllSpecies:
    @pytest.mark.parametrize("species", [
        "oak", "birch", "spruce", "dark_oak", "jungle", "acacia",
        "cherry", "mangrove", "pale_oak",
    ])
    def test_overworld_log_kept_at_bare(self, species):
        assert f"minecraft:{species}_log" in keep_set_for_tier("bare")

    @pytest.mark.parametrize("species", [
        "oak", "birch", "spruce", "dark_oak", "jungle", "acacia",
        "cherry", "mangrove", "pale_oak", "crimson", "warped",
    ])
    def test_planks_kept_at_bare(self, species):
        assert f"minecraft:{species}_planks" in keep_set_for_tier("bare")

    def test_nether_stems_kept_at_bare(self):
        assert "minecraft:crimson_stem" in keep_set_for_tier("bare")
        assert "minecraft:warped_stem" in keep_set_for_tier("bare")


class TestJunkDropped:
    """Items the agent never gainfully uses must be on the drop list."""

    @pytest.mark.parametrize("item", [
        # Saplings (no farming surface)
        "minecraft:oak_sapling", "minecraft:birch_sapling",
        # Flowers (Wurst defaults already drop these; we widen)
        "minecraft:poppy", "minecraft:dandelion", "minecraft:sunflower",
        # Sand / gravel / gravity blocks (bad shelter material)
        "minecraft:sand", "minecraft:red_sand", "minecraft:gravel",
        # Common cave/cliff stone junk
        "minecraft:andesite", "minecraft:granite", "minecraft:diorite",
        "minecraft:tuff", "minecraft:calcite", "minecraft:dripstone_block",
        # Nether/end junk
        "minecraft:basalt", "minecraft:smooth_basalt", "minecraft:blackstone",
        "minecraft:end_stone", "minecraft:netherrack",
        # Last-sprint shelter-material cluster
        "minecraft:mud", "minecraft:packed_mud", "minecraft:deepslate",
        "minecraft:cobbled_deepslate",
        # Stripped variants (agent never strips)
        "minecraft:stripped_oak_log", "minecraft:stripped_birch_log",
        # Seeds (no farming)
        "minecraft:wheat_seeds", "minecraft:beetroot_seeds",
        "minecraft:pumpkin_seeds", "minecraft:melon_seeds",
        # Mob loot the agent can't use yet
        "minecraft:rotten_flesh", "minecraft:bone",
        "minecraft:spider_eye", "minecraft:slime_ball",
        # Decorative / exotic items
        "minecraft:amethyst_shard", "minecraft:saddle", "minecraft:name_tag",
        # Spawn eggs
        "minecraft:cow_spawn_egg", "minecraft:zombie_spawn_egg",
        # Bamboo (deferred — biome filter handles spawn)
        "minecraft:bamboo", "minecraft:bamboo_block", "minecraft:bamboo_planks",
    ])
    @pytest.mark.parametrize("tier", _TIERS)
    def test_junk_dropped(self, item, tier):
        assert item in drop_list_for_tier(tier), (
            f"junk item {item} not dropped at tier={tier}"
        )


class TestTierGatingDormant:
    """Tier-gating is dormant — all current keep-items live in ALWAYS_KEEP.

    Per the policy file's directive: tier-gating gets added back only when
    actual rollout observations motivate it. Until then, TIER_KEEP buckets
    are placeholders (subsets of ALWAYS_KEEP), so drop_count is the same
    across all tiers. These tests pin that shape so accidental tier-only
    additions (which would change behavior at runtime) get flagged.
    """

    @pytest.mark.parametrize("tier", _TIERS)
    def test_tier_keep_subset_of_always_keep(self, tier):
        extra = TIER_KEEP[tier] - ALWAYS_KEEP
        assert not extra, (
            f"TIER_KEEP[{tier}] has additions outside ALWAYS_KEEP: {sorted(extra)}; "
            f"if intentional, update this test"
        )

    def test_all_tiers_have_same_keep_set(self):
        sets = [keep_set_for_tier(t) for t in _TIERS]
        assert all(s == sets[0] for s in sets[1:])
