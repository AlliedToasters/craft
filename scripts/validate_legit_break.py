"""Live validation of the TickBreaker 'legit break' fix — BAREHANDED bamboo.

Deterministic mechanism test: clear the hotbar (no sword => forces the gradual,
tick-paced break), /setblock a bamboo column one block from the player, POST
/harvest_bamboo, then check SERVER-SIDE that the base is gone and bamboo was
collected. Probes use a per-run NONCE so stale chat-log lines can't match.

Run: HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 python -m scripts.validate_legit_break
"""

from __future__ import annotations

import math
import re
import time

import requests

from craft.testkit import cmd

H = "http://127.0.0.1:25570"
RELAY_LOG = "http://127.0.0.1:4747/log"


def _log_tail(n: int = 60) -> str:
    try:
        return requests.get(RELAY_LOG, params={"n": n}, timeout=4).text
    except requests.RequestException:
        return ""


def probe(condition: str, nonce: str) -> bool:
    """`execute <condition> run say <nonce>` server-side; True iff the marker fired."""
    cmd(f"execute {condition} run say {nonce}")
    time.sleep(0.4)
    return nonce in _log_tail()


def server_bamboo_count() -> int:
    """`clear <player> bamboo 0` counts matching items WITHOUT removing them."""
    nonce_cmd = "clear agent0 minecraft:bamboo 0"
    cmd(nonce_cmd)
    time.sleep(0.35)
    tail = _log_tail()
    # "Found N matching item(s) on player agent0"  /  "No items were found ..."
    m = re.findall(r"Found (\d+) matching item", tail)
    if m:
        return int(m[-1])
    return 0


def ground_items_near() -> bool:
    n = f"GROUNDITEM_{int(time.time() * 1000) % 100000}"
    return probe("if entity @e[type=item,distance=..10]", n)


def main() -> None:
    run = int(time.time() * 1000) % 1000000
    p = requests.get(f"{H}/position", timeout=4).json()
    px, py, pz = math.floor(p["x"]), math.floor(p["y"]), math.floor(p["z"])
    print(f"[pos] player at ({px},{py},{pz})  run={run}")

    cmd("difficulty peaceful")
    cmd("gamemode survival agent0")
    cmd("clear agent0")
    time.sleep(0.6)

    bx, by, bz = px + 1, py, pz
    cmd(f"setblock {bx} {by - 1} {bz} minecraft:dirt")
    for dy in (0, 1, 2):
        cmd(f"setblock {bx} {by + dy} {bz} minecraft:bamboo")
    time.sleep(0.6)
    placed = probe(f"if block {bx} {by} {bz} minecraft:bamboo", f"PLACED_{run}")
    print(f"[place] bamboo base ({bx},{by},{bz}) placed_server_side={placed}")
    if not placed:
        print("[ABORT] base did not place")
        return

    before = server_bamboo_count()
    print(f"[pre]   server bamboo on player = {before}")

    t0 = time.time()
    resp = requests.post(f"{H}/harvest_bamboo", json={"radius": 5}, timeout=40).json()
    print(f"[harvest] ({time.time() - t0:.1f}s) -> {resp}")

    time.sleep(2.0)  # generous drain for slow-client item pickup

    still = probe(f"if block {bx} {by} {bz} minecraft:bamboo", f"STILL_{run}")
    cleared = probe(f"unless block {bx} {by} {bz} minecraft:bamboo", f"CLEARED_{run}")
    after = server_bamboo_count()
    ground = ground_items_near()
    print(f"[post]  base still bamboo (server) = {still}")
    print(f"[post]  base cleared (server)      = {cleared}")
    print(f"[post]  server bamboo on player    = {after} (gained {after - before})")
    print(f"[post]  loose bamboo items nearby  = {ground}")

    broke = int(resp.get("columns_broken", 0) or 0)
    print("\n==== VERDICT (barehanded, no sword) ====")
    print(f"  columns_broken             : {broke}")
    print(f"  base cleared SERVER-side    : {cleared and not still}")
    print(f"  bamboo collected (server)   : {after - before}")
    print(f"  (loose items on ground      : {ground})")
    break_ok = cleared and not still
    collect_ok = (after - before) > 0
    print(f"  >>> BREAK {'PASS' if break_ok else 'FAIL'} | "
          f"COLLECT {'PASS' if collect_ok else 'FAIL'} <<<")


if __name__ == "__main__":
    main()
