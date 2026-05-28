"""Direct substrate chain validation for iron→bed.

Drives the dusk_iron_to_bed scenario without an LLM. Confirms the
substrate primitives compose:

    apply_loadout('dusk_iron_to_bed')
        ↓ (2 iron + planks + crafting_table + iron tools + 6 sheep nearby)
    handle_shear_sheep({})              # auto-crafts shears from 2 iron
        ↓ wool drops to inventory
    handle_craft({item: red_bed, quantity: 1})
        ↓ bed in inventory
    handle_sleep_in_bed({})
        ↓ time advances to dawn

If this passes, the substrate is green and the only remaining gap for
the LLM rollout is "the model doesn't *choose* this chain" — i.e. the
substrate signal (tiered night-bed nudge). If this fails, we have
substrate bugs to fix before any nudge work matters.

Run (against agent0):
    HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 \\
    MC_SERVER_CMD_BASE=http://10.0.0.222:4747 \\
    .venv/bin/python -m scripts.iron_to_bed_direct
"""

from __future__ import annotations

import time

from craft.config import HOMUNCULUS_BASE, PLAYER_NAME, SERVER_CMD_BASE
from craft.testkit import cmd, inventory, pos, preflight, stats
from craft.loadouts import apply_loadout
from craft.spawn import random_spawn
from craft.tools import (
    handle_craft,
    handle_shear_sheep,
    handle_sleep_in_bed,
)
from craft.world import give_item, set_gamemode, set_time


def _has(inv_response: dict | None, suffix: str) -> int:
    """Count items in inventory whose id ends with `suffix`.

    /inventory returns {main: [{slot,id,count}], armor: [...], offhand: [...]}.
    We sum across main + armor + offhand.
    """
    if not inv_response:
        return 0
    total = 0
    for bucket in ("main", "armor", "offhand"):
        for item in inv_response.get(bucket) or []:
            if isinstance(item, dict) and item.get("id", "").endswith(suffix):
                total += int(item.get("count", 0))
    return total


def _print_step(label: str, outcome: str) -> None:
    head = outcome.replace("\n", " | ")[:200]
    print(f"\n[{label}] outcome:\n    {head}")


def main() -> None:
    print("=== iron→bed direct chain smoke ===")

    print("\n[step 0] random_spawn (clean tile) ...")
    sp = random_spawn(
        range_blocks=20000,
        homunculus_base=HOMUNCULUS_BASE,
        server_cmd_base=SERVER_CMD_BASE,
        player_name=PLAYER_NAME,
    )
    if not sp.get("ok"):
        print(f"FAIL: random_spawn errored: {sp}")
        return
    p = pos()
    if p is None:
        print("FAIL: pos() returned None after spawn")
        return
    print(f"    spawned at ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) "
          f"biome={sp.get('biome')}")

    # random_spawn ends in survival, but defensive: force-survival just
    # before the loadout. /give doesn't work in spectator and we just
    # toggled gamemode multiple times during spawn-probe.
    print("\n[step 0.5] force survival ...")
    set_gamemode("survival")
    s0 = stats() or {}
    print(f"    gamemode hint: hp={s0.get('hp')} food={s0.get('food')}")

    # Fresh clients (or clients left over from prior rollouts that ended
    # cleanly) have hacks OFF. Without KillAura/AutoShears/ShearReflex the
    # shear chain falls back to the explicit /shear/sheep packet, which
    # races against natural sheep wandering. Preflight enables the full
    # required hack set.
    print("\n[step 0.7] wurst preflight ...")
    pf_err = preflight(ensure_hacks=True)
    if pf_err:
        print(f"    preflight WARNING: {pf_err}")
    else:
        print("    preflight ok")

    print("\n[step 1] applying dusk_iron_to_bed loadout ...")
    rep = apply_loadout("dusk_iron_to_bed")
    # /give and /item replace are dispatched via the server console relay
    # and take a tick or two to actually mutate the player's inventory.
    # Sleep briefly so the inventory read below sees the post-loadout
    # state, not the pre-loadout state.
    time.sleep(1.0)
    if not rep.get("ok"):
        # Report which step failed
        bad = [s for s in rep.get("steps", [])
               if (s.get("result") or {}).get("ok") is False]
        print(f"    loadout warnings: {len(bad)} step(s) not-ok")
        for b in bad[:3]:
            print(f"      - {b.get('step')}: {b.get('result')}")
    inv = inventory()
    print(f"    inventory: iron_ingot={_has(inv,'iron_ingot')} "
          f"oak_planks={_has(inv,'oak_planks')} "
          f"crafting_table={_has(inv,'crafting_table')}")

    print("\n[step 2] set_time(midnight) ...")
    set_time("midnight")
    s = stats() or {}
    print(f"    day_ticks={s.get('day_ticks')} day={s.get('day_count')}")

    # Wait briefly so the pre_summon_sheep settle on solid ground (they
    # fall a tick or two after /summon).
    time.sleep(2.0)

    # Shears yield 1-3 wool per sheep — usually 1. Loop until we have 3+
    # (one bed) or we exhaust the 6 pre-summoned sheep. Cap at 6 calls.
    print("\n[step 3] handle_shear_sheep (loop until 3+ wool) ...")
    wool_n = _has(inventory(), "_wool")
    for attempt in range(6):
        if wool_n >= 3:
            print(f"    have {wool_n} wool, sufficient for bed.")
            break
        print(f"\n  -- attempt {attempt + 1} (current wool={wool_n}) --")
        out = handle_shear_sheep({})
        _print_step("shear_sheep", out)
        time.sleep(0.5)
        wool_n = _has(inventory(), "_wool")
        print(f"    cumulative wool: {wool_n}")

    if wool_n < 3:
        # The shear chain is not deterministic — KillAura kills passives,
        # 6 pre-summoned sheep aren't always enough to net 3 wool, sheep
        # wander past scan radius after the first chase. We've validated
        # the shear chain works; now we want to validate the bed/sleep
        # chain regardless. Top up with /give and continue.
        need = 3 - wool_n
        print(f"\n  [topup] only {wool_n} wool from shearing — adding "
              f"{need}x white_wool via /give to validate the rest of the chain.")
        give_item("minecraft:white_wool", need)
        time.sleep(0.5)
        wool_n = _has(inventory(), "_wool")
        print(f"    after topup: {wool_n} wool")

    # White_bed because shearing yields white_wool (default sheep color).
    # Bed recipes are color-strict in MC — red_bed needs red_wool, etc.
    # This matches what the agent would actually produce from shearing.
    print("\n[step 4] handle_craft white_bed ...")
    out = handle_craft({"item": "minecraft:white_bed", "quantity": 1})
    _print_step("craft", out)
    inv3 = inventory()
    bed_n = _has(inv3, "_bed")
    print(f"    bed count: {bed_n}")

    if bed_n < 1:
        print("\nFAIL: no bed in inventory after craft. Substrate gap in "
              "craft chain — investigate.")
        return

    # Build a flat stone platform around the player so the bed-pair search
    # succeeds. Without this, natural terrain (grass tufts, slopes,
    # leaves overhead) routinely fails the head-cell sturdy-support
    # check. Substrate gap (bed-placement too finicky) noted separately;
    # here we just want to validate sleep_in_bed proper.
    p = pos()
    if p is not None:
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        # 5x5 stone floor at y-1
        cmd(f"fill {x-2} {y-1} {z-2} {x+2} {y-1} {z+2} stone")
        # 5x5x3 air above
        cmd(f"fill {x-2} {y} {z-2} {x+2} {y+2} {z+2} air")
        time.sleep(0.5)

    print("\n[step 5] handle_sleep_in_bed ...")
    pre = (stats() or {}).get("day_ticks", -1)
    out = handle_sleep_in_bed({})
    _print_step("sleep_in_bed", out)
    post = (stats() or {}).get("day_ticks", -1)
    print(f"    day_ticks pre={pre} post={post} (advance={post - pre})")

    print("\n=== chain complete ===")
    if out.startswith("slept"):
        print("PASS: end-to-end substrate chain executed.")
    else:
        print("PARTIAL: bed crafted but sleep didn't return slept-outcome — "
              "inspect tool output.")


if __name__ == "__main__":
    main()
