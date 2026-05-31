#!/usr/bin/env python3
"""§17.2.2 — live ENTITY-target codec: the decoy attack harness.

Two questions, paralleling §17.2.1 but on the entity channel (interact.entity_id):

  PHASE A — INDEX KNEE (confirmatory pointer reparam).
    Summon a ROW of N cows; attack the MIDDLE one (a non-endpoint index, so a
    coarse quantizer actually rounds to a neighbour). Sweep:
      control         entity_id lossless identity.
      index@bN        entity_id coded as its INDEX into obs.entity_set
                      (nearest-first), N bits. Predicted: lossless to
                      ceil(log2(N)) bits, then rounds to a NEIGHBOUR cow (a real,
                      observable miss). The §17.2.1 block-pointer reparam on the
                      entity channel.
      absolute@bN     entity_id quantized as the raw network int over ±abs_range.
                      Predicted: a BOGUS id below ~22 bits → reconstructor null →
                      substitute FALLS BACK to the original packet (substitute_
                      errors>0, substituted==0). It can't even produce a wrong-but-
                      real hit. The foil: the raw handle has no cheap coding.

  PHASE B — COLLAPSE / DECOY (the discovery, "predict the decision not the packet").
    Two cows T (intended) + D (decoy) at distinguishable close range. collapse
    DROPS the pointer entirely and names entity_set[0] (nearest = the §13.1
    geometric argmax, ~0 index bits). Two configs:
      intent==geom    T nearest. Attack T. collapse → T. Dropping the pointer is
                      FREE (the ~98.5% case, §13.1).
      intent!=geom    D nearest. Attack T. control/index → T (pointer PRESERVES
                      intent); collapse → D (honors geometry = the ~1.5% tail).
    This is the headroom block_pos did NOT have (§17.2.1 bottomed at the lossless
    pointer; here the target is reconstructable from obs geometry already in hand).

Metric = WHOSE HP dropped. /attack_entity reports the intended target's health
(reads T after the swing → a diverted hit shows success=false for T); we ALSO
scan all cows before/after and attribute the damage by position, so a divert to
the decoy is observed directly. Cows: NoAI/Silent/Persistent, stationary, so the
obs nearest-first order == scan nearest-first order == our geometric "nearest"
(all three go through the same Entities.query highway).

Killaura MUST be OFF (it would land its own un-substituted hits). Run PEACEFUL.

Usage:
    .venv/bin/python -m experiments.codec_loop.entity_decoy --port 25570 \
        --out results/sprint17/entity_decoy.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import _config, _killaura_off, _relay, _sidecar_base
from experiments.codec_loop.run_rungs import _http, _pos, _resolve_base, _wait_in_world

_LOSSLESS = {"quant_bits": None, "obsrel": False, "perturb": None,
             "blockpos": None, "entityid": None}

# Match radius (blocks): cows render at block-center (+0.5), so a cow summoned at
# integer (x,z) reports ~0.71 off; layouts keep cows >=2 blocks apart, so 1.2 is
# unambiguous while absorbing the center offset.
_MATCH_TOL = 1.2


# ----------------------------------------------------------------------------- cows
def _kill_cows(relay: str) -> None:
    _relay(relay, "kill @e[type=cow]")
    time.sleep(0.4)


def _summon_cow(relay: str, x: float, y: float, z: float) -> None:
    _relay(relay, f"summon minecraft:cow {x} {y} {z} "
                  "{NoAI:1b,Silent:1b,PersistenceRequired:1b}")


def _scan_cows(base: str) -> list[dict]:
    r = _http("GET", f"{base}/scan_entities?type=cow&radius=12", timeout=8)
    return r.get("entities", [])


def _match(cows: list[dict], x: float, y: float, z: float) -> dict | None:
    """The cow nearest the expected (x,y,z), within _MATCH_TOL. None if absent."""
    best, bestd = None, _MATCH_TOL
    for c in cows:
        cx, cy, cz = c["position"]
        d = ((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2) ** 0.5
        if d <= bestd:
            best, bestd = c, d
    return best


def _health_by_uuid(cows: list[dict]) -> dict[str, float]:
    return {c["uuid"]: float(c.get("health") or 0.0) for c in cows}


def _damaged_uuids(before: list[dict], after: list[dict]) -> list[str]:
    """uuids whose HP dropped or that vanished (killed) between two scans."""
    hb = _health_by_uuid(before)
    ha = _health_by_uuid(after)
    out = []
    for u, h0 in hb.items():
        if u not in ha:            # despawned == killed
            out.append(u)
        elif ha[u] < h0 - 1e-3:
            out.append(u)
    return out


# ----------------------------------------------------------------------------- attack
def _attack(base: str, uuid: str) -> dict:
    return _http("POST", f"{base}/attack_entity", {"uuid": uuid}, timeout=10)


def _arm(base: str, codec_url: str) -> None:
    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)


def _disarm(base: str) -> dict:
    st = _http("GET", f"{base}/codec/passthrough/status", timeout=8)
    _http("POST", f"{base}/codec/passthrough/disarm")
    return st


# ----------------------------------------------------------------------------- cells
def _attack_cell(base, relay, codec_url, cow_specs, intended_xyz, trials, label, eid_cfg):
    """Summon cows at cow_specs (list of (x,y,z)), set the codec, arm substitute,
    attack the cow at intended_xyz trials times, attribute each hit by HP-drop.

    Returns aggregate: how often the INTENDED cow was hit vs the NEAREST cow (the
    geometric argmax) vs neither, plus substitution counters."""
    sidecar = _sidecar_base(codec_url)
    _config(sidecar, {**_LOSSLESS, "entityid": eid_cfg})

    _kill_cows(relay)
    for (x, y, z) in cow_specs:
        _summon_cow(relay, x, y, z)
    time.sleep(0.8)

    cows0 = _scan_cows(base)
    intended = _match(cows0, *intended_xyz)
    nearest = cows0[0] if cows0 else None  # scan is nearest-first
    if intended is None or nearest is None:
        return {"cell": label, "error": "cow setup failed",
                "n_cows": len(cows0), "config": eid_cfg}
    intended_uuid = intended["uuid"]
    nearest_uuid = nearest["uuid"]

    _arm(base, codec_url)

    hit_intended = hit_nearest = hit_other = no_hit = 0
    per_trial = []
    for _ in range(trials):
        before = _scan_cows(base)
        r = _attack(base, intended_uuid)
        time.sleep(0.5)  # let the server→client HP echo arrive
        after = _scan_cows(base)
        dmg = _damaged_uuids(before, after)
        cls = "no_hit"
        if intended_uuid in dmg:
            hit_intended += 1; cls = "intended"
        elif nearest_uuid in dmg:
            hit_nearest += 1; cls = "nearest_decoy"
        elif dmg:
            hit_other += 1; cls = "other"
        else:
            no_hit += 1
        per_trial.append({"class": cls, "damaged": dmg,
                          "attack_success": bool(r.get("success")),
                          "reason": r.get("reason")})
        # keep cows alive/topped for a clean next trial
        if any(u not in _health_by_uuid(after) for u in (intended_uuid, nearest_uuid)):
            _kill_cows(relay)
            for (x, y, z) in cow_specs:
                _summon_cow(relay, x, y, z)
            time.sleep(0.8)
            cows0 = _scan_cows(base)
            im, nm = _match(cows0, *intended_xyz), (cows0[0] if cows0 else None)
            if im and nm:
                intended_uuid, nearest_uuid = im["uuid"], nm["uuid"]

    st = _disarm(base)
    return {
        "cell": label, "config": eid_cfg, "trials": trials,
        "n_cows": len(cows0),
        "intended_is_nearest": intended_uuid == nearest_uuid,
        "hit_intended": hit_intended, "hit_nearest_decoy": hit_nearest,
        "hit_other": hit_other, "no_hit": no_hit,
        "hit_intended_rate": round(hit_intended / trials, 3) if trials else 0.0,
        "substituted": st.get("substituted"),
        "substitute_errors": st.get("substitute_errors"),
        "substitute_fallbacks": st.get("substitute_fallbacks"),
        "drift": st.get("drift"),
        "per_trial": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§17.2.2 entity-target decoy harness")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--row-n", type=int, default=5, help="cows in the Phase-A row")
    ap.add_argument("--index-bits", default="4,3,2,1")
    ap.add_argument("--abs-bits", default="24,12")
    ap.add_argument("--out", default="results/sprint17/entity_decoy.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    codec_url = args.codec_url
    index_bits = [int(b) for b in args.index_bits.split(",") if b.strip()]
    abs_bits = [int(b) for b in args.abs_bits.split(",") if b.strip()]

    print(f"[entity_decoy] base={base} codec={codec_url} relay={args.relay}")
    if not _wait_in_world(base):
        print("[entity_decoy] FATAL: no player in world")
        return 2
    if not _killaura_off(base):
        print("[entity_decoy] Killaura was ON — disabled it (was confounding).")
    else:
        print("[entity_decoy] Killaura confirmed OFF.")

    p0 = _pos(base)
    px, py, pz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    py = py  # same plane
    print(f"[entity_decoy] player=({px},{py},{pz}) trials={args.trials}")

    cells = []

    # ---- PHASE A: index knee. Row of N cows along +x at dx=2..2+N-1, attack the
    # middle (idx = N//2, a non-endpoint so the quantizer rounds to a neighbour).
    n = args.row_n
    row = [(px + 2 + i, py, pz) for i in range(n)]
    mid_i = n // 2
    intended_xyz = row[mid_i]
    print(f"\n[entity_decoy] PHASE A: {n}-cow row, attack idx={mid_i} at {intended_xyz}")
    import math
    print(f"  predicted index-lossless bits = ceil(log2({n})) = {math.ceil(math.log2(n))}")

    agg = _attack_cell(base, args.relay, codec_url, row, intended_xyz, args.trials,
                       "A:control", None)
    _print_cell(agg)
    cells.append({"phase": "A", "mode": "control", "bits": None, **agg})

    for b in index_bits:
        agg = _attack_cell(base, args.relay, codec_url, row, intended_xyz, args.trials,
                           f"A:index_b{b}", {"mode": "index", "bits": b})
        _print_cell(agg)
        cells.append({"phase": "A", "mode": "index", "bits": b, **agg})

    for b in abs_bits:
        agg = _attack_cell(base, args.relay, codec_url, row, intended_xyz, args.trials,
                           f"A:abs_b{b}", {"mode": "absolute", "bits": b})
        _print_cell(agg)
        cells.append({"phase": "A", "mode": "absolute", "bits": b, **agg})

    # ---- PHASE B: collapse / decoy. T + D at distinguishable close range.
    # cfgB1 intent==geom (T nearest, dist 2.0; D farther, dist ~2.83)
    # cfgB2 intent!=geom (D nearest, dist 2.0; T farther, dist ~2.83)
    near = (px + 2, py, pz)        # dist 2.0
    far = (px + 2, py, pz + 2)     # dist ~2.83
    print("\n[entity_decoy] PHASE B: collapse/decoy (T=intended, D=decoy)")

    # B1: T nearest -> dropping the pointer is free
    print("\n  -- cfgB1 intent==geom (T nearest) --")
    for label, cfg in [("B1:control", None),
                        ("B1:collapse", {"mode": "collapse"})]:
        agg = _attack_cell(base, args.relay, codec_url, [near, far], near,
                           args.trials, label, cfg)
        _print_cell(agg)
        cells.append({"phase": "B1", "mode": label.split(":")[1], **agg})

    # B2: D nearest, T farther -> pointer preserves intent; collapse diverts to D
    print("\n  -- cfgB2 intent!=geom (D nearest, attack farther T) --")
    for label, cfg in [("B2:control", None),
                        ("B2:index_b4", {"mode": "index", "bits": 4}),
                        ("B2:collapse", {"mode": "collapse"})]:
        agg = _attack_cell(base, args.relay, codec_url, [near, far], far,
                           args.trials, label, cfg)
        _print_cell(agg)
        cells.append({"phase": "B2", "mode": label.split(":")[1], **agg})

    _config(_sidecar_base(codec_url), dict(_LOSSLESS))
    _kill_cows(args.relay)

    out = {"base": base, "codec_url": codec_url, "trials": args.trials,
           "row_n": n, "cells": cells}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("§17.2.2 ENTITY-TARGET codec — decoy harness")
    print("=" * 72)
    print(f"{'cell':>14} {'intended':>9} {'decoy':>6} {'other':>6} {'none':>5} "
          f"{'subst':>6} {'subErr':>7} {'subFB':>6}")
    for c in cells:
        if "error" in c:
            print(f"{c['cell']:>14}  ERROR: {c['error']}")
            continue
        print(f"{c['cell']:>14} {c['hit_intended']:>9} {c['hit_nearest_decoy']:>6} "
              f"{c['hit_other']:>6} {c['no_hit']:>5} {str(c.get('substituted')):>6} "
              f"{str(c.get('substitute_errors')):>7} {str(c.get('substitute_fallbacks')):>6}")
    print(f"\nwrote {args.out}")
    return 0


def _print_cell(agg: dict) -> None:
    if "error" in agg:
        print(f"  {agg['cell']}: ERROR {agg['error']} (n_cows={agg.get('n_cows')})")
        return
    print(f"  {agg['cell']:>14}: intended={agg['hit_intended']} decoy={agg['hit_nearest_decoy']} "
          f"other={agg['hit_other']} none={agg['no_hit']} "
          f"subst={agg['substituted']} subErr={agg['substitute_errors']} "
          f"intended_is_nearest={agg['intended_is_nearest']}")


if __name__ == "__main__":
    raise SystemExit(main())
