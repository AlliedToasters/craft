"""Probe: are Wurst KillAura's attacks visible in the outbound packet stream?

Determines whether §13.1.4 can use rigorous capture-replay (KillAura's per-frame
target recovered from interact/ATTACK packets, directly comparable to the 0.985
offline number) or must fall back to a functional A/B.

Arms the packet recorder, summons an AI zombie wave with KillAura ON, lets it fight,
disarms, and counts interact/ATTACK packets in the capture.
"""
from __future__ import annotations

import collections
import json
import os
import time

import requests

from craft.config import HOMUNCULUS_BASE as H, SERVER_CMD_BASE as CMD

OUT = os.path.abspath("results/online_arena/ka_probe/packets.jsonl")


def cmd(s):
    return requests.post(f"{CMD}/cmd", json={"cmd": s}, timeout=5).json()


def killaura(on):
    return requests.post(f"{H}/wurst/hack",
                         json={"name": "KillAura", "enabled": on}, timeout=5).json()


def arm():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    return requests.post(f"{H}/packets/recording/arm",
                         json={"path": OUT, "append": False}, timeout=5).json()


def disarm():
    return requests.post(f"{H}/packets/recording/disarm", json={}, timeout=5).json()


def main():
    p = requests.get(f"{H}/position", timeout=5).json()
    px, py, pz = round(p["x"]), round(p["y"]), round(p["z"])
    print("player", px, py, pz)
    cmd("difficulty easy")
    cmd(f"effect give agent0 minecraft:resistance 120 4 true")
    cmd(f"effect give agent0 minecraft:regeneration 120 3 true")
    cmd("kill @e[type=minecraft:zombie,distance=..40]")
    time.sleep(1.0)
    print("killaura on:", killaura(True))
    print("arm:", arm())
    # AI-enabled zombies so they aggro and KillAura engages.
    for ox, oz in ((2, 0), (3, 1), (1, 2), (-2, 1), (0, 3)):
        sx, sy, sz = px + ox, py, pz + oz
        cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy+2} {sz+1} minecraft:air")
        cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy-1} {sz+1} minecraft:stone")
        cmd(f"summon minecraft:zombie {sx} {sy} {sz} {{PersistenceRequired:1b,Silent:1b}}")
    # let KillAura fight
    for i in range(12):
        time.sleep(1.0)
        cmd(f"effect give agent0 minecraft:regeneration 30 3 true")
    print("disarm:", disarm())
    time.sleep(0.5)

    ids = collections.Counter()
    actions = collections.Counter()
    n = 0
    with open(OUT) as f:
        for line in f:
            n += 1
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ids[r.get("id")] += 1
            if r.get("id") == "minecraft:interact":
                actions[(r.get("fields") or {}).get("action")] += 1
    print(f"\ntotal packets captured: {n}")
    print("top ids:", dict(ids.most_common(10)))
    print("interact actions:", dict(actions))
    print(f"\nVERDICT: KillAura ATTACK packets visible = "
          f"{actions.get('ATTACK', 0) > 0} (n_ATTACK={actions.get('ATTACK', 0)})")
    cmd("kill @e[type=minecraft:zombie,distance=..40]")


if __name__ == "__main__":
    main()
