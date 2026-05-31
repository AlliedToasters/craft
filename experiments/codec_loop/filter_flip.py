#!/usr/bin/env python3
"""§18.1 grounding probe — does the KillAura passive filter FLIP the attack target?

The §18 premise: the interact/attack target is a function of (geometry, type, g_t)
where g_t = the operator's policy = the Wurst KillAura filter stack. shear_sheep is
the only mode exercised today (it flips "Filter passive mobs" on for the duration).
Before plumbing g_t into obs and building a mixed-mode dataset, confirm the BEHAVIOR
the whole sprint rests on:

  Scene (fixed): a sheep (passive, NEAREST) + a zombie (hostile, farther), both
  NoAI/Silent/Persistent so the geometry is frozen. KillAura ON.

  Mode A  filter_passive=false (attack passives, the ambient default):
            KillAura attacks the NEAREST attackable = the sheep.
  Mode B  filter_passive=true  (protect passives, the shear_sheep mode):
            sheep is excluded → KillAura attacks the zombie instead.

Same geometry, target flips by mode. Metric = whose HP dropped / who died over a
short KillAura window (reuse the entity_decoy HP-drop attribution). If this flip
reproduces, the sprint premise holds; if not, we learn it cheaply here.

Not a codec test — no sidecar, no substitution. Pure substrate behavior.

Usage:
    .venv/bin/python -m experiments.codec_loop.filter_flip --port 25570 \
        --out results/sprint18/filter_flip.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import _relay
from experiments.codec_loop.run_rungs import _http, _pos, _resolve_base, _wait_in_world

_KA = "KillAura"
_FILTER = "Filter passive mobs"  # Wurst exclusion semantics: true = DON'T attack passives


def _scan(base: str, species: str, radius: int = 12) -> list[dict]:
    r = _http("GET", f"{base}/scan_entities?type={species}&radius={radius}", timeout=8)
    return r.get("entities", [])


def _hp(ents: list[dict]) -> dict[str, float]:
    return {e["uuid"]: float(e.get("health") or 0.0) for e in ents}


def _damaged(before: list[dict], after: list[dict]) -> list[str]:
    hb, ha = _hp(before), _hp(after)
    out = []
    for u, h0 in hb.items():
        if u not in ha:           # despawned == killed
            out.append(u)
        elif ha[u] < h0 - 1e-3:
            out.append(u)
    return out


def _set_filter(base: str, on: bool) -> dict:
    return _http("POST", f"{base}/wurst/setting",
                 {"hack": _KA, "setting": _FILTER, "value": bool(on)}, timeout=8)


def _killaura(base: str, enabled: bool) -> dict:
    return _http("POST", f"{base}/wurst/hack", {"name": _KA, "enabled": enabled}, timeout=8)


def _set_priority(base: str, value: str) -> dict:
    return _http("POST", f"{base}/wurst/setting",
                 {"hack": _KA, "setting": "Priority", "value": value}, timeout=8)


def _clear(relay: str) -> None:
    _relay(relay, "kill @e[type=sheep]")
    _relay(relay, "kill @e[type=zombie]")
    time.sleep(0.4)


def _summon_scene(relay: str, px: int, py: int, pz: int, *, with_zombie: bool,
                  sheep_dx: int = 2, zombie_dx: int = 4) -> None:
    """Sheep at sheep_dx (NEAREST); optional zombie at zombie_dx (farther). Frozen."""
    tags = "{NoAI:1b,Silent:1b,PersistenceRequired:1b}"
    _relay(relay, f"summon minecraft:sheep {px+sheep_dx} {py} {pz} {tags}")
    if with_zombie:
        _relay(relay, f"summon minecraft:zombie {px+zombie_dx} {py} {pz} {tags}")
    time.sleep(0.9)


def _run_mode(base: str, relay: str, px: int, py: int, pz: int, *,
              filter_passive: bool, with_zombie: bool, window_s: float, label: str) -> dict:
    _clear(relay)
    _summon_scene(relay, px, py, pz, with_zombie=with_zombie)
    sheep0, zomb0 = _scan(base, "sheep"), _scan(base, "zombie")
    _set_filter(base, filter_passive)
    _killaura(base, True)
    time.sleep(window_s)
    _killaura(base, False)
    time.sleep(0.4)
    sheep1, zomb1 = _scan(base, "sheep"), _scan(base, "zombie")
    sheep_hit = bool(_damaged(sheep0, sheep1))
    zomb_hit = bool(_damaged(zomb0, zomb1))
    target = ("sheep" if sheep_hit and not zomb_hit else
              "zombie" if zomb_hit and not sheep_hit else
              "both" if sheep_hit and zomb_hit else "neither")
    return {
        "label": label, "filter_passive": filter_passive, "with_zombie": with_zombie,
        "n_sheep0": len(sheep0), "n_zombie0": len(zomb0),
        "sheep_hit": sheep_hit, "zombie_hit": zomb_hit,
        "sheep_hp": (_hp(sheep0), _hp(sheep1)),
        "zombie_hp": (_hp(zomb0), _hp(zomb1)),
        "target": target,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§18.1 filter-flip grounding probe")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--window", type=float, default=2.5, help="KillAura ON window (s)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="results/sprint18/filter_flip.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    print(f"[filter_flip] base={base} relay={args.relay}")
    if not _wait_in_world(base):
        print("[filter_flip] FATAL: no player in world")
        return 2

    # zombie must persist (easy ≥ peaceful) and not burn (night).
    _relay(args.relay, "difficulty easy")
    _relay(args.relay, "time set 18000")
    # no inter-agent / self PvP confound; KillAura should target mobs only.
    _http("POST", f"{base}/wurst/setting",
          {"hack": _KA, "setting": "Filter players", "value": True}, timeout=8)

    p0 = _pos(base)
    px, py, pz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    print(f"[filter_flip] player=({px},{py},{pz}) window={args.window}s trials={args.trials}")

    # The first probe showed KillAura's PRIORITY preferred the farther zombie over
    # the nearer sheep when both were eligible. Pin Priority=Distance so the duel
    # scene's mode-A pick is the nearest (sheep) — isolating the filter's effect.
    pr = _set_priority(base, "Distance")
    print(f"  set KillAura Priority=Distance: success={pr.get('success')} "
          f"changed={pr.get('changed')} reason={pr.get('reason')}")

    # ---- SCENE 1: solo sheep. The crispest flip — same single entity, filter
    # toggles whether KillAura acts on it at all. A->sheep hit, B->sheep safe.
    solo = []
    for t in range(args.trials):
        a = _run_mode(base, args.relay, px, py, pz, filter_passive=False,
                      with_zombie=False, window_s=args.window, label=f"solo.A.t{t}")
        b = _run_mode(base, args.relay, px, py, pz, filter_passive=True,
                      with_zombie=False, window_s=args.window, label=f"solo.B.t{t}")
        print(f"  [solo t{t}] A(attack)={a['target']}  B(protect)={b['target']}")
        solo += [a, b]

    # ---- SCENE 2: duel (sheep near + zombie far, Priority=Distance). The
    # two-candidate flip the dataset needs: A->near sheep, B->far zombie.
    duel = []
    for t in range(args.trials):
        a = _run_mode(base, args.relay, px, py, pz, filter_passive=False,
                      with_zombie=True, window_s=args.window, label=f"duel.A.t{t}")
        b = _run_mode(base, args.relay, px, py, pz, filter_passive=True,
                      with_zombie=True, window_s=args.window, label=f"duel.B.t{t}")
        print(f"  [duel t{t}] A(attack)={a['target']}  B(protect)={b['target']}")
        duel += [a, b]

    # restore ambient default (attack passives) + clear scene
    _set_filter(base, False)
    _killaura(base, False)
    _clear(args.relay)

    solo_A = sum(1 for r in solo if not r["filter_passive"] and r["target"] == "sheep")
    solo_B = sum(1 for r in solo if r["filter_passive"] and r["target"] == "neither")
    solo_n = args.trials
    duel_A = sum(1 for r in duel if not r["filter_passive"] and r["target"] == "sheep")
    duel_B = sum(1 for r in duel if r["filter_passive"] and r["target"] == "zombie")
    duel_n = args.trials
    solo_ok = solo_A >= 1 and solo_B >= 1
    duel_ok = duel_A >= 1 and duel_B >= 1

    out = {"base": base, "window_s": args.window, "trials": args.trials,
           "priority_set": pr,
           "solo": {"A_sheep": f"{solo_A}/{solo_n}", "B_neither": f"{solo_B}/{solo_n}",
                    "flip": solo_ok},
           "duel": {"A_sheep": f"{duel_A}/{duel_n}", "B_zombie": f"{duel_B}/{duel_n}",
                    "flip": duel_ok},
           "rows": solo + duel}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 64)
    print("§18.1 FILTER-FLIP grounding probe")
    print("=" * 64)
    print("  SCENE 1 solo-sheep (filter gates whether the sheep is hit):")
    print(f"    A attack-passives -> sheep hit:   {solo_A}/{solo_n}")
    print(f"    B protect-passives -> sheep safe:  {solo_B}/{solo_n}   flip={solo_ok}")
    print("  SCENE 2 duel near-sheep/far-zombie (Priority=Distance):")
    print(f"    A attack-passives -> sheep:        {duel_A}/{duel_n}")
    print(f"    B protect-passives -> zombie:      {duel_B}/{duel_n}   flip={duel_ok}")
    print(f"\n  PREMISE HOLDS: {solo_ok or duel_ok}")
    print(f"wrote {args.out}")
    return 0 if (solo_ok or duel_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
