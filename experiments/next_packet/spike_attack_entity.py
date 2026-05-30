"""§13.1.2 spike: prove the /attack_entity injection path lands a hit on a chosen
entity. Summons a stationary NoAI zombie next to the player, scans for it, picks
it by UUID, and attacks until dead — logging health_before/after each hit.

Done-criterion: a chosen runtime entity's health drops / it dies via /attack_entity.

Env: HOMUNCULUS_PORT=25570 (agent0). Uses the MC server cmd relay (:4747) to summon.
KillAura is turned OFF for the duration so the hits are unambiguously ours.

    HOMUNCULUS_PORT=25570 .venv/bin/python -m experiments.next_packet.spike_attack_entity
"""
from __future__ import annotations

import time

import requests

from craft.config import HOMUNCULUS_BASE, SERVER_CMD_BASE

H = HOMUNCULUS_BASE
CMD = SERVER_CMD_BASE


def cmd(s):
    r = requests.post(f"{CMD}/cmd", json={"cmd": s}, timeout=5)
    return r.json() if r.ok else {"ok": False, "status": r.status_code}


def position():
    return requests.get(f"{H}/position", timeout=5).json()


def scan(etype, radius=16, limit=16):
    r = requests.get(f"{H}/scan_entities",
                     params={"type": etype, "radius": radius, "limit": limit}, timeout=5)
    return r.json()


def attack(uuid):
    r = requests.post(f"{H}/attack_entity", json={"uuid": uuid}, timeout=8)
    return r.json()


def set_killaura(enabled):
    return requests.post(f"{H}/wurst/hack",
                         json={"name": "KillAura", "enabled": enabled}, timeout=5).json()


def main():
    print(f"homunculus={H} relay={CMD}")
    # 0. KillAura OFF so only our attacks land.
    print("killaura off:", set_killaura(False))
    cmd("difficulty easy")  # hostiles persist (peaceful would despawn them)

    pos = position()
    px, py, pz = pos["x"], pos["y"], pos["z"]
    print(f"player at ({px:.1f},{py:.1f},{pz:.1f})")

    # 1. Carve a clear pocket 3 blocks away so the zombie can't suffocate in terrain
    #    (the prior run summoned into a wall → "suffocated in a wall"), then summon a
    #    stationary, non-despawning zombie on its floor.
    sx, sy, sz = round(px) + 3, round(py), round(pz)
    cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy+2} {sz+1} minecraft:air")
    cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy-1} {sz+1} minecraft:stone")
    nbt = "{NoAI:1b,PersistenceRequired:1b,Silent:1b}"
    print("summon:", cmd(f"summon minecraft:zombie {sx} {sy} {sz} {nbt}"))
    time.sleep(1.0)

    # 2. Scan for it.
    sc = scan("minecraft:zombie")
    ents = sc.get("entities", [])
    print(f"scan found {len(ents)} zombie(s)")
    if not ents:
        print("FAIL: no zombie found after summon")
        return
    target = ents[0]
    uuid = target["uuid"]
    print(f"target uuid={uuid} type={target['type']} dist={target['distance']:.2f} "
          f"hp={target['health']}")

    # 3. Attack until dead (or 30 tries). Zombie has 20 HP. A full-strength melee
    #    needs the attack-cooldown charged (~0.6s for fist), so pace > cooldown; a
    #    no_effect just means the swing wasn't charged or hit i-frames — retry.
    landed = 0
    for i in range(30):
        res = attack(uuid)
        ok = res.get("success")
        print(f"  hit {i}: success={ok} reason={res.get('reason')} "
              f"hp {res.get('health_before')}->{res.get('health_after')} "
              f"killed={res.get('killed')} msg={res.get('message','')[:50]}")
        if ok:
            landed += 1
            if res.get("killed"):
                print(f"TARGET KILLED after {landed} landed hits")
                break
        elif res.get("reason") in ("entity_not_found", "target_gone"):
            print(f"target gone (reason={res.get('reason')})")
            break
        time.sleep(0.75)  # > attack-cooldown so each swing is charged

    print(f"\nRESULT: landed {landed} hit(s) via /attack_entity")
    # cleanup any leftover spike targets
    cmd("kill @e[type=minecraft:zombie,name=spike_target]")
    print("killaura back on:", set_killaura(True))


if __name__ == "__main__":
    main()
