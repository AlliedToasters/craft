"""Tiered AutoDrop whitelist policy.

Wurst's AutoDrop hack is binary per item: an id is on the drop list or it isn't.
We invert the usual mental model — the brain decides what's worth *keeping*, and
the substrate drops everything else. Keep-sets are layered: ALWAYS_KEEP for items
that are always useful regardless of progression, plus per-tier additions that
unlock as the agent progresses (wood → stone → iron → diamond).

`drop_list_for_tier(tier)` produces the complement (sorted list of ids) suitable
for `POST /wurst/setting {hack: AutoDrop, setting: Items, op: replace, value: ...}`.

Discipline: this file errs aggressive (drop more than keep). Adding an id to a
keep set should follow a substrate-thread justification — e.g., "agent now uses
X recipe so X's inputs need to stay." Removing one is cheap. The "I might want
this later" trap (slimeballs, prismarine, glow_ink_sac, ...) goes in the drop
pile until a concrete use case lands.
"""

from __future__ import annotations

from craft._item_registry import ALL_ITEMS


# -------------------- ALWAYS_KEEP categories --------------------
# Each frozenset is one semantic bucket. Composed below into ALWAYS_KEEP.
# Naming kept verbose so future-me can scan and tune one bucket at a time.

_TOOLS: frozenset[str] = frozenset({
    # 6 tiers × 5 types = 30. Includes wooden through netherite, all 5 tool kinds.
    "minecraft:wooden_sword",   "minecraft:wooden_pickaxe",   "minecraft:wooden_axe",
    "minecraft:wooden_shovel",  "minecraft:wooden_hoe",
    "minecraft:stone_sword",    "minecraft:stone_pickaxe",    "minecraft:stone_axe",
    "minecraft:stone_shovel",   "minecraft:stone_hoe",
    "minecraft:iron_sword",     "minecraft:iron_pickaxe",     "minecraft:iron_axe",
    "minecraft:iron_shovel",    "minecraft:iron_hoe",
    "minecraft:golden_sword",   "minecraft:golden_pickaxe",   "minecraft:golden_axe",
    "minecraft:golden_shovel",  "minecraft:golden_hoe",
    "minecraft:diamond_sword",  "minecraft:diamond_pickaxe",  "minecraft:diamond_axe",
    "minecraft:diamond_shovel", "minecraft:diamond_hoe",
    "minecraft:netherite_sword","minecraft:netherite_pickaxe","minecraft:netherite_axe",
    "minecraft:netherite_shovel","minecraft:netherite_hoe",
})

_ARMOR: frozenset[str] = frozenset({
    # 6 tiers × 4 pieces = 24, plus turtle_helmet (sole turtle-tier piece).
    "minecraft:leather_helmet",  "minecraft:leather_chestplate",
    "minecraft:leather_leggings","minecraft:leather_boots",
    "minecraft:chainmail_helmet","minecraft:chainmail_chestplate",
    "minecraft:chainmail_leggings","minecraft:chainmail_boots",
    "minecraft:iron_helmet",     "minecraft:iron_chestplate",
    "minecraft:iron_leggings",   "minecraft:iron_boots",
    "minecraft:golden_helmet",   "minecraft:golden_chestplate",
    "minecraft:golden_leggings", "minecraft:golden_boots",
    "minecraft:diamond_helmet",  "minecraft:diamond_chestplate",
    "minecraft:diamond_leggings","minecraft:diamond_boots",
    "minecraft:netherite_helmet","minecraft:netherite_chestplate",
    "minecraft:netherite_leggings","minecraft:netherite_boots",
    "minecraft:turtle_helmet",
})

_INGOTS_RAW: frozenset[str] = frozenset({
    "minecraft:iron_ingot", "minecraft:gold_ingot", "minecraft:copper_ingot",
    "minecraft:netherite_ingot", "minecraft:netherite_scrap",
    "minecraft:raw_iron", "minecraft:raw_gold", "minecraft:raw_copper",
})

_GEMS: frozenset[str] = frozenset({
    "minecraft:diamond", "minecraft:emerald", "minecraft:lapis_lazuli",
    "minecraft:redstone", "minecraft:quartz",
    # amethyst_shard intentionally OMITTED — purely decorative at present.
})

_FOOD: frozenset[str] = frozenset({
    # Curated to anything AutoEat would touch. Saturating foods first; emergency
    # foods (sweet_berries, dried_kelp) included so a low-hunger agent has fallbacks.
    "minecraft:bread", "minecraft:apple", "minecraft:golden_apple",
    "minecraft:enchanted_golden_apple", "minecraft:carrot", "minecraft:golden_carrot",
    "minecraft:potato", "minecraft:baked_potato", "minecraft:beetroot",
    "minecraft:melon_slice", "minecraft:sweet_berries", "minecraft:glow_berries",
    "minecraft:chorus_fruit", "minecraft:honey_bottle", "minecraft:pumpkin_pie",
    "minecraft:cake", "minecraft:cookie", "minecraft:dried_kelp",
    "minecraft:beef", "minecraft:cooked_beef",
    "minecraft:chicken", "minecraft:cooked_chicken",
    "minecraft:porkchop", "minecraft:cooked_porkchop",
    "minecraft:mutton", "minecraft:cooked_mutton",
    "minecraft:rabbit", "minecraft:cooked_rabbit",
    "minecraft:cod", "minecraft:cooked_cod",
    "minecraft:salmon", "minecraft:cooked_salmon",
    "minecraft:mushroom_stew", "minecraft:rabbit_stew", "minecraft:beetroot_soup",
    "minecraft:suspicious_stew",
    "minecraft:tropical_fish", "minecraft:pufferfish",  # not great but technically food
    "minecraft:wheat",  # not eaten but recipe → bread; kept for the chain
})

_FUEL: frozenset[str] = frozenset({
    "minecraft:coal", "minecraft:charcoal", "minecraft:coal_block",
    "minecraft:stick",  # tool recipes and torch crafting
    "minecraft:blaze_powder",  # potion fuel; rare drop
})

_FURNITURE: frozenset[str] = frozenset({
    # Minimal set load-bearing for the survival craft graph. Exotic stations
    # (loom, fletching_table, enchanting_table, …) are deliberately dropped —
    # add when an agent's craft path actually needs them.
    "minecraft:crafting_table",
    "minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker",
    "minecraft:smithing_table",  # netherite progression
    "minecraft:anvil",           # tool repair
    "minecraft:chest",           # base storage; agents don't use yet but harmless to keep
})

_LIGHTING: frozenset[str] = frozenset({
    "minecraft:torch",
    # jack_o_lantern / soul_torch / lanterns / candles OMITTED — decorative,
    # craftable variants of the same light source.
})

_DOORS: frozenset[str] = frozenset({
    "minecraft:oak_door", "minecraft:birch_door", "minecraft:spruce_door",
    "minecraft:dark_oak_door", "minecraft:jungle_door", "minecraft:acacia_door",
    "minecraft:cherry_door", "minecraft:mangrove_door", "minecraft:pale_oak_door",
    "minecraft:bamboo_door",  # craft-only; bamboo biomes are spawn-filtered
    "minecraft:crimson_door", "minecraft:warped_door",
    "minecraft:iron_door",
    # Copper door variants OMITTED — exotic, redstone-gated.
})

_BEDS: frozenset[str] = frozenset({
    "minecraft:black_bed", "minecraft:blue_bed", "minecraft:brown_bed",
    "minecraft:cyan_bed", "minecraft:gray_bed", "minecraft:green_bed",
    "minecraft:light_blue_bed", "minecraft:light_gray_bed", "minecraft:lime_bed",
    "minecraft:magenta_bed", "minecraft:orange_bed", "minecraft:pink_bed",
    "minecraft:purple_bed", "minecraft:red_bed", "minecraft:white_bed",
    "minecraft:yellow_bed",
})

_WOOL: frozenset[str] = frozenset({
    "minecraft:black_wool", "minecraft:blue_wool", "minecraft:brown_wool",
    "minecraft:cyan_wool", "minecraft:gray_wool", "minecraft:green_wool",
    "minecraft:light_blue_wool", "minecraft:light_gray_wool", "minecraft:lime_wool",
    "minecraft:magenta_wool", "minecraft:orange_wool", "minecraft:pink_wool",
    "minecraft:purple_wool", "minecraft:red_wool", "minecraft:white_wool",
    "minecraft:yellow_wool",
})

_MOB_DROPS: frozenset[str] = frozenset({
    # Conservative — only mob drops with a clear immediate or near-future craft use.
    "minecraft:leather",      # leather armor
    "minecraft:string",       # bow, wool (4→1)
    "minecraft:feather",      # arrows
    "minecraft:gunpowder",    # TNT — not crafted yet but high-cost rare drop
    "minecraft:ender_pearl",  # eye_of_ender → end portal; rare
    "minecraft:blaze_rod",    # blaze_powder fuel; rare
    "minecraft:nether_star",  # end-game; impossible to recover if dropped
    # OMITTED (long tail): bone, bone_meal, spider_eye, fermented_spider_eye,
    # magma_cream, ghast_tear, ink_sac, glow_ink_sac, prismarine_*, rabbit_foot,
    # rabbit_hide, slime_ball. Add piecemeal if/when a substrate use lands.
})

_BUCKETS: frozenset[str] = frozenset({
    "minecraft:bucket",
    "minecraft:water_bucket", "minecraft:lava_bucket", "minecraft:milk_bucket",
    # Fish/mob buckets OMITTED — exotic, accidental fills.
})

_UTILITY: frozenset[str] = frozenset({
    "minecraft:flint",
    "minecraft:flint_and_steel",
    "minecraft:shears",  # wool/leaves harvesting — 2 iron_ingot to recraft, so a
                         # /give that hit AutoDrop wasted a non-trivial cost. Load-bearing
                         # for the sheep → wool → bed chain (see ShearHandler).
    # compass, map, clock, name_tag, saddle, lead, spyglass OMITTED.
})


# -------------------- Tier-gated keep sets --------------------
# Cumulative: drop_list_for_tier("iron") unions bare + stone + iron.

# Logs (excluding stripped variants — agents don't strip; village-picked stripped
# logs are recipe-equivalent via tag-aware canonicalItem so dropping them isn't
# load-bearing, just frees inventory).
# NOTE: this has been manually disabled by the user, too complicated for first pass (sorry, claude)
_LOGS_AND_STEMS: frozenset[str] = frozenset({
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:dark_oak_log", "minecraft:jungle_log", "minecraft:acacia_log",
    "minecraft:cherry_log", "minecraft:mangrove_log", "minecraft:pale_oak_log",
    "minecraft:crimson_stem", "minecraft:warped_stem",
    # bamboo_block OMITTED — bamboo progression deferred (biome spawn-filtered).
})

_PLANKS: frozenset[str] = frozenset({
    "minecraft:oak_planks", "minecraft:birch_planks", "minecraft:spruce_planks",
    "minecraft:dark_oak_planks", "minecraft:jungle_planks", "minecraft:acacia_planks",
    "minecraft:cherry_planks", "minecraft:mangrove_planks", "minecraft:pale_oak_planks",
    "minecraft:crimson_planks", "minecraft:warped_planks",
    # bamboo_planks OMITTED — see above.
})

ALWAYS_KEEP: frozenset[str] = (
    _TOOLS | _ARMOR | _INGOTS_RAW | _GEMS | _FOOD | _FUEL
    | _FURNITURE | _LIGHTING | _DOORS | _BEDS | _WOOL
    | _MOB_DROPS | _BUCKETS | _UTILITY | _LOGS_AND_STEMS
    | _PLANKS | frozenset({
        "minecraft:cobblestone",  # universal shelter block + stone-tool material
        "minecraft:dirt",         # shelter filler
    }) | frozenset({
        "minecraft:stone",         # smelted cobble; rare in normal flow
        "minecraft:iron_nugget",   # iron_ingot/9 — kept once stone tools mine iron
    })
)

# commented out, please revisit if/when ACTUAL OBSERVATIONS MOTIVATE AN ACTUAL NEED FOR THIS
TIER_KEEP: dict[str, frozenset[str]] = {
    "bare":    _LOGS_AND_STEMS | _PLANKS | frozenset({
        # "minecraft:cobblestone",  # universal shelter block + stone-tool material
        # "minecraft:dirt",         # shelter filler
    }),
    "stone":   frozenset({
        # "minecraft:stone",         # smelted cobble; rare in normal flow
        # "minecraft:iron_nugget",   # iron_ingot/9 — kept once stone tools mine iron
    }),
    "iron":    frozenset({
        # No additions beyond ALWAYS_KEEP yet — iron tier is mostly enabled by
        # the existing ingot/gem/tool keep sets. Reserved slot for future
        # iron-gated materials (e.g., compass parts, anvil components).
    }),
    "diamond": frozenset({
        # Same — diamond progression keeps everything via ALWAYS_KEEP gems.
        # Reserved for future diamond-gated materials.
    }),
}

_TIER_ORDER: tuple[str, ...] = ("bare", "stone", "iron", "diamond")


def keep_set_for_tier(tier: str) -> frozenset[str]:
    """Union of ALWAYS_KEEP plus every TIER_KEEP up to and including `tier`."""
    if tier not in TIER_KEEP:
        raise ValueError(
            f"unknown tier {tier!r}, expected one of {_TIER_ORDER}"
        )
    keep = ALWAYS_KEEP
    for t in _TIER_ORDER:
        keep |= TIER_KEEP[t]
        if t == tier:
            break
    return keep


def drop_list_for_tier(tier: str) -> list[str]:
    """Sorted list of item ids to drop at the given tier.

    Suitable for POST /wurst/setting {hack: AutoDrop, setting: Items,
    op: replace, value: <returned list>}.
    """
    return sorted(ALL_ITEMS - keep_set_for_tier(tier))


def _self_check() -> None:
    """Catch typos: every entry in any keep set must be a real MC item id."""
    all_keeps = ALWAYS_KEEP | frozenset().union(*TIER_KEEP.values())
    bogus = all_keeps - ALL_ITEMS
    if bogus:
        raise AssertionError(
            f"autodrop.py: {len(bogus)} keep-set entries not in registry: "
            f"{sorted(bogus)}"
        )


_self_check()


if __name__ == "__main__":
    import sys
    tier = sys.argv[1] if len(sys.argv) > 1 else "bare"
    drops = drop_list_for_tier(tier)
    keep = sorted(keep_set_for_tier(tier))
    print(f"tier={tier}: keep={len(keep)} drop={len(drops)} of {len(ALL_ITEMS)} total")
    print(f"  keep sample: {keep[:10]} ...")
    print(f"  drop sample: {drops[:10]} ...")
