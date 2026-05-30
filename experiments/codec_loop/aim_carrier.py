#!/usr/bin/env python3
"""§17.0 — Diagnose the AIM carrier: which wire field carries the attack target?

§16 proved the move-packet's rotation is near-cosmetic for NAVIGATION — the
obs-relative codec deadbands turns to 180° steps and the controller still reaches
every goto, because POSITION carries navigation and Baritone re-aims client-side
each tick. But goto-reach is rotation-insensitive by construction. §11a found the
opposite for AIM: the block/entity target collapses to GAZE. So: for an aim-
dependent action (attack), WHICH wire field actually carries the target?

This is the live probe (no new lossy aim codec — §17.0 scope). Fixed aim task =
melee-attack one stationary NoAI cow at point-blank reach, K times, and judge an
AIM-SENSITIVE outcome: did the cow's HEALTH drop (hit landed)? Cells:

  control      lossless identity codec (§14 codec-off-equivalent).
  obsrel@bN    §16 obs-relative rotation at N bits — move-packet yaw/pitch
               DEADBANDED on the wire (b2 = 180° steps, near-total rotation loss),
               position held near-lossless (pos_bits high). Tests whether the
               server gates the attack on the wire ROTATION.

VERDICT logic:
  (a) obsrel@b2 BREAKS the hit (landed-rate drops vs control) → move-rotation is
      load-bearing for aim → 17.1 finds the move-rotation aim knee.
  (b) obsrel@b2 is BENIGN (landed-rate == control) → the target does NOT ride the
      move-packet → it rides the action packet (interact.entity_id) → 17.2 is the
      real test (lossy discrete-target codec). PREDICTED case (the action packet
      names entity_id; move-yaw is render/anti-cheat).

The cow is summoned NoAI/Silent/Persistent so it never flees, takes no knockback-
walk, and stays at constant point-blank reach — every attack is a clean aim test
with position held fixed (isolates rotation). Killaura MUST be OFF (else it lands
its own hits and confounds the A/B); the harness asserts it.

Run under PEACEFUL. One agent, sequential cells.

Usage:
    .venv/bin/python -m experiments.codec_loop.aim_carrier --port 25570 \
        --bits 2,3,4 --attacks 6 --out results/sprint17/aim_carrier.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from experiments.codec_loop.run_rungs import _http, _pos, _resolve_base, _wait_in_world


def _sidecar_base(codec_url: str) -> str:
    i = codec_url.find("/codec/roundtrip")
    return codec_url[:i] if i > 0 else codec_url


def _config(sidecar: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{sidecar}/config", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _relay(relay: str, cmd: str) -> dict:
    return _http("POST", f"{relay}/cmd", {"cmd": cmd}, timeout=8)


def _rot_step_deg(bits: int) -> float:
    k = (1 << (bits - 1)) - 1
    return 180.0 / k if k > 0 else 360.0


def _killaura_off(base: str) -> bool:
    """Assert Killaura is disabled — it would land its own hits and confound."""
    st = _http("GET", f"{base}/wurst/status", timeout=8)
    for h in st.get("hacks", []):
        if h.get("name", "").lower() == "killaura" and h.get("enabled"):
            _http("POST", f"{base}/wurst/hack", {"name": h["name"], "enabled": False})
            return False  # was on, now off
    return True  # already off


def _fresh_cow(base: str, relay: str, xyz: tuple[int, int, int]) -> str | None:
    """Kill all cows, summon one stationary NoAI cow, return its UUID."""
    _relay(relay, "kill @e[type=cow]")
    time.sleep(0.4)
    x, y, z = xyz
    _relay(relay, f"summon minecraft:cow {x} {y} {z} "
                  "{NoAI:1b,Silent:1b,PersistenceRequired:1b}")
    time.sleep(0.7)
    r = _http("GET", f"{base}/scan_entities?type=cow&radius=20", timeout=8)
    ents = r.get("entities", [])
    return ents[0]["uuid"] if ents else None


def _attack(base: str, uuid: str) -> dict:
    return _http("POST", f"{base}/attack_entity", {"uuid": uuid}, timeout=10)


def _run_cell(base, relay, codec_url, cow_xyz, attacks, label, cfg):
    """Summon a fresh cow, set the sidecar cfg, arm substitution, attack K times,
    judge each by health-drop (aim landed), disarm, return aggregate."""
    sidecar = _sidecar_base(codec_url)
    _config(sidecar, cfg)
    uuid = _fresh_cow(base, relay, cow_xyz)
    if not uuid:
        return {"cell": label, "error": "no cow summoned", **cfg}

    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)  # let a few move packets flow under the new wire config

    landed = 0
    attempted = 0
    out_of_reach = 0
    total_damage = 0.0
    per_attack = []
    for _ in range(attacks):
        r = _attack(base, uuid)
        attempted += 1
        ok = bool(r.get("success"))
        hb, ha = r.get("health_before"), r.get("health_after")
        killed = bool(r.get("killed"))
        hit = killed or (ok and hb is not None and ha is not None and ha < hb)
        if hit:
            landed += 1
            if hb is not None and ha is not None:
                total_damage += (hb - ha)
        if r.get("reason") == "out_of_reach":
            out_of_reach += 1
        per_attack.append({"ok": ok, "hb": hb, "ha": ha, "killed": killed,
                           "hit": hit, "reason": r.get("reason"),
                           "dist": r.get("distance")})
        if killed:
            uuid = _fresh_cow(base, relay, cow_xyz) or uuid  # respawn to keep going
        time.sleep(0.6)

    status = _http("GET", f"{base}/codec/passthrough/status", timeout=8)
    _http("POST", f"{base}/codec/passthrough/disarm")

    return {
        "cell": label,
        "config": cfg,
        "attempted": attempted,
        "landed": landed,
        "landed_rate": round(landed / attempted, 3) if attempted else 0.0,
        "out_of_reach": out_of_reach,
        "total_damage": round(total_damage, 2),
        "substituted": status.get("substituted"),
        "substitute_errors": status.get("substitute_errors"),
        "drift": status.get("drift"),
        "per_attack": per_attack,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§17.0 aim-carrier live probe")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--bits", default="2,3,4",
                    help="obs-rel rotation bit levels (b2=180° steps = max loss)")
    ap.add_argument("--pos-bits", type=int, default=16,
                    help="position bits — kept HIGH to isolate rotation")
    ap.add_argument("--attacks", type=int, default=6, help="attacks per cell")
    ap.add_argument("--cow-dx", type=int, default=2, help="cow offset +x from player")
    ap.add_argument("--out", default="results/sprint17/aim_carrier.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    codec_url = args.codec_url
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]

    print(f"[aim_carrier] base={base} codec={codec_url} relay={args.relay}")
    if not _wait_in_world(base):
        print("[aim_carrier] FATAL: no player in world")
        return 2

    if not _killaura_off(base):
        print("[aim_carrier] Killaura was ON — disabled it (was confounding).")
    else:
        print("[aim_carrier] Killaura confirmed OFF.")

    p0 = _pos(base)
    px, py, pz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    cow_xyz = (px + args.cow_dx, py, pz)
    print(f"[aim_carrier] player=({px},{py},{pz}) cow={cow_xyz} attacks={args.attacks}")

    cells = []

    print("\n[aim_carrier] === CONTROL (lossless identity) ===")
    agg = _run_cell(base, args.relay, codec_url, cow_xyz, args.attacks, "control",
                    {"quant_bits": None, "obsrel": False})
    print(f"  landed={agg['landed_rate']} ({agg['landed']}/{agg['attempted']}) "
          f"dmg={agg['total_damage']} subst={agg['substituted']} "
          f"subErr={agg['substitute_errors']} oor={agg['out_of_reach']}")
    cells.append({"rot_bits": None, "rot_step_deg": 0.0, **agg})

    for b in bit_levels:
        step = _rot_step_deg(b)
        print(f"\n[aim_carrier] === obsrel rotation b{b} (step {step:.1f}°), "
              f"pos b{args.pos_bits} ===")
        agg = _run_cell(base, args.relay, codec_url, cow_xyz, args.attacks, f"b{b}",
                        {"obsrel": True, "pos_bits": args.pos_bits,
                         "yaw_bits": b, "pitch_bits": b, "pos_mode": "zero_preserving"})
        print(f"  landed={agg['landed_rate']} ({agg['landed']}/{agg['attempted']}) "
              f"dmg={agg['total_damage']} subst={agg['substituted']} "
              f"subErr={agg['substitute_errors']} oor={agg['out_of_reach']}")
        cells.append({"rot_bits": b, "rot_step_deg": round(step, 2), **agg})

    # reset sidecar to lossless
    _config(_sidecar_base(codec_url), {"quant_bits": None, "obsrel": False})

    out = {"base": base, "codec_url": codec_url, "pos_bits": args.pos_bits,
           "attacks": args.attacks, "cow_xyz": list(cow_xyz), "cells": cells}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("§17.0 AIM CARRIER — move-rotation arm")
    print("=" * 60)
    print(f"{'cell':>8} {'rot_step':>9} {'landed':>7} {'dmg':>6} {'subst':>6} {'subErr':>7}")
    for c in cells:
        print(f"{c['cell']:>8} {c['rot_step_deg']:>8.1f}° {c['landed_rate']:>7.2f} "
              f"{c.get('total_damage', 0):>6} {str(c.get('substituted')):>6} "
              f"{str(c.get('substitute_errors')):>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
