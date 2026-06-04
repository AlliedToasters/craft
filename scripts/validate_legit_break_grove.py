"""End-to-end BAREHANDED bamboo: a small grove + the real mine.harvest_bamboo()
loop (break base -> reposition onto next, vacuuming the prior drops). Validates
break AND collection with no sword in inventory.

Run: HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 python -m scripts.validate_legit_break_grove
"""

from __future__ import annotations

import math
import re
import time

import requests

from craft import mine
from craft.testkit import cmd

H = "http://127.0.0.1:25570"
RELAY_LOG = "http://127.0.0.1:4747/log"


def server_bamboo_count() -> int:
    cmd("clear agent0 minecraft:bamboo 0")
    time.sleep(0.35)
    try:
        tail = requests.get(RELAY_LOG, params={"n": 30}, timeout=4).text
    except requests.RequestException:
        tail = ""
    m = re.findall(r"Found (\d+) matching item", tail)
    return int(m[-1]) if m else 0


def main() -> None:
    p = requests.get(f"{H}/position", timeout=4).json()
    px, py, pz = math.floor(p["x"]), math.floor(p["y"]), math.floor(p["z"])
    print(f"[pos] player at ({px},{py},{pz})")

    cmd("difficulty peaceful")
    cmd("gamemode survival agent0")
    cmd("clear agent0")  # no sword -> barehanded
    time.sleep(0.6)

    # A short row of 4 columns (3 tall) stepping +x from the player.
    cols = [(px + dx, pz) for dx in (1, 2, 3, 4)]
    for (cx, cz) in cols:
        cmd(f"setblock {cx} {py - 1} {cz} minecraft:dirt")
        for dy in (0, 1, 2):
            cmd(f"setblock {cx} {py + dy} {cz} minecraft:bamboo")
    time.sleep(0.8)
    print(f"[place] {len(cols)} bamboo columns (12 stalks) along x={cols[0][0]}..{cols[-1][0]} z={pz}")

    before = server_bamboo_count()
    print(f"[pre]   server bamboo on player = {before}")

    t0 = time.time()
    out = mine.harvest_bamboo(quantity=12, max_rounds=8)
    print(f"[harvest_bamboo()] ({time.time() - t0:.1f}s) -> {out!r}")

    time.sleep(1.5)
    after = server_bamboo_count()
    print(f"[post]  server bamboo on player = {after} (collected {after - before})")

    print("\n==== VERDICT (barehanded grove, real harvest loop) ====")
    print(f"  bamboo collected server-side : {after - before}")
    ok = (after - before) > 0
    print(f"  >>> END-TO-END BAREHANDED {'PASS' if ok else 'FAIL'} <<<")


if __name__ == "__main__":
    main()
