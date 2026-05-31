#!/usr/bin/env python3
"""§18.1 — mixed-filter target capture: the g_t-conditioned dataset.

Premise (grounded by filter_flip.py): KillAura's attack target is a function of
(geometry, entity type, g_t) where g_t = the Wurst KillAura filter policy. The
§18.0 prior P(target | geom, type) is mode-BLIND; frozen_combat is single-mode so
it never paid for that. Here we build a MULTI-mode dataset where the SAME scene
maps to DIFFERENT targets under different g_t, so the bits g_t buys is measurable.

Each base scene = a random mix of passives + hostiles at varied in-reach offsets,
frozen (NoAI/Silent/Persistent). We capture every scene under TWO g_t modes:
  attack_all       Filter passive mobs = False  (attack passives; Priority=Distance)
                   -> KillAura picks the NEAREST entity regardless of type.
  protect_passive  Filter passive mobs = True   (shear_sheep mode)
                   -> passives excluded; KillAura picks the nearest HOSTILE.
When the nearest entity is passive but a hostile is present, the label FLIPS
between the matched rows -- the mode-blind ambiguity, built in by construction.

Label = KillAura's actual pick, observed by HP-drop (the §17.2.2 attribution),
NOT "nearest". Row shape mirrors rung_a_target.load_attacks (cands sorted by
distance + integer label) so the §13.1 feature/training pipeline reuses directly.
Each row also carries the g_t policy dict and a per-candidate `attackable` oracle
(computed offline from policy x type, logged not fed -- the user-approved design).

Not a codec test. KillAura ON is the executor under test; no sidecar/substitution.

Usage:
    .venv/bin/python -m experiments.codec_loop.filter_capture --port 25570 \
        --scenes 60 --seed 0 --out results/sprint18/filter_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from experiments.codec_loop.aim_carrier import _relay
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world

_KA = "KillAura"

# Species pools. attackable-by-passive-filter is the load-bearing class split.
_PASSIVE = ["sheep", "cow", "pig", "chicken"]
_HOSTILE = ["zombie", "skeleton"]
# Frozen so the geometry == what the obs entity_set would report. NoAI alone
# does NOT stop knockback (physics, not AI): KillAura's first hit shoves the
# nearest mob away, retargets, and sprays the whole scene over a multi-tick
# window — corrupting the "policy pick" label. knockback_resistance=1 pins them,
# so the priority target absorbs all damage and max-drop recovers the pick.
_TAGS = ('{NoAI:1b,Silent:1b,PersistenceRequired:1b,'
         'Attributes:[{id:"minecraft:knockback_resistance",base:1.0}]}')
# KillAura reach ~4.25 (filter_flip finding); keep every mob strictly inside.
_MIN_D, _MAX_D = 1.6, 3.8

# g_t modes swept per scene. Priority pinned Distance so geometry is the tiebreak
# and the only thing varying is the passive filter.
_MODES = {
    "attack_all":      {"Filter passive mobs": False, "Priority": "Distance"},
    "protect_passive": {"Filter passive mobs": True,  "Priority": "Distance"},
}


def _set(base: str, setting: str, value) -> dict:
    return _http("POST", f"{base}/wurst/setting",
                 {"hack": _KA, "setting": setting, "value": value}, timeout=8)


def _killaura(base: str, enabled: bool) -> dict:
    return _http("POST", f"{base}/wurst/hack", {"name": _KA, "enabled": enabled}, timeout=8)


def _apply_mode(base: str, mode: dict) -> bool:
    """Apply every setting; return False if any POST didn't report success (a
    silent failure would leave the PREVIOUS mode's filter in effect → a passive
    attacked in protect mode)."""
    ok = True
    for k, v in mode.items():
        r = _set(base, k, v)
        ok = ok and bool(r.get("success"))
    return ok


def _position(base: str) -> dict:
    return _http("GET", f"{base}/position", timeout=8)


def _scan_all(base: str, species: list[str], radius: int = 10) -> list[dict]:
    out = []
    for sp in species:
        r = _http("GET", f"{base}/scan_entities?type={sp}&radius={radius}", timeout=8)
        for e in r.get("entities", []):
            e["_species"] = sp
            out.append(e)
    return out


def _hp(ents: list[dict]) -> dict[str, float]:
    return {e["uuid"]: float(e.get("health") or 0.0) for e in ents}


def _drops(before: list[dict], after: list[dict]) -> dict[str, float]:
    """Per-uuid HP decrease (vanished == killed == its full prior HP)."""
    hb, ha = _hp(before), _hp(after)
    out = {}
    for u, h0 in hb.items():
        if u not in ha:
            out[u] = h0
        elif ha[u] < h0 - 1e-3:
            out[u] = h0 - ha[u]
    return out


def _clear(relay: str, species: list[str]) -> None:
    for sp in species:
        _relay(relay, f"kill @e[type={sp}]")
    time.sleep(0.35)


def _gen_scene(rng: random.Random) -> list[dict]:
    """A random mixed scene: >=1 passive and >=1 hostile (so the flip can fire),
    plus 0-2 extra mobs. Distinct distances so the dist-sort label is unambiguous."""
    k_extra = rng.randint(0, 2)
    specs = [(rng.choice(_PASSIVE), "passive"), (rng.choice(_HOSTILE), "hostile")]
    for _ in range(k_extra):
        cls = rng.choice(["passive", "hostile"])
        specs.append((rng.choice(_PASSIVE if cls == "passive" else _HOSTILE), cls))
    rng.shuffle(specs)
    # place at distinct distances + spread angles (horizontal ring at player y)
    dists = rng.sample([round(d, 2) for d in _frange(_MIN_D, _MAX_D, 0.2)], len(specs))
    out = []
    for (sp, cls), d in zip(specs, dists):
        ang = rng.uniform(0, 2 * math.pi)
        out.append({"species": sp, "cls": cls, "dist": d,
                    "dx": d * math.cos(ang), "dz": d * math.sin(ang)})
    return out


def _frange(lo, hi, step):
    n = int((hi - lo) / step)
    return [lo + i * step for i in range(n + 1)]


def _summon(relay: str, px: float, py: float, pz: float, scene: list[dict]) -> None:
    for s in scene:
        x, z = px + s["dx"], pz + s["dz"]
        _relay(relay, f"summon minecraft:{s['species']} {x:.2f} {py} {z:.2f} {_TAGS}")
    time.sleep(0.9)


def _attackable(species_cls: str, mode: dict) -> bool:
    """Offline oracle: would KillAura attack this class under `mode`? (logged, not fed)."""
    if species_cls == "passive":
        return not bool(mode.get("Filter passive mobs", False))
    return True  # hostiles always attackable under these modes


def _candidates(base: str, px: float, py: float, pz: float,
                yaw: float, pitch: float) -> list[dict] | None:
    """Scan the live scene, build dist-sorted candidates with the same geom features
    rung_a_target.load_attacks uses. Returns None if the scene didn't materialize."""
    ents = _scan_all(base, _PASSIVE + _HOSTILE)
    if len(ents) < 2:
        return None
    cands = []
    for e in ents:
        ex, ey, ez = e["position"]
        dx, dy, dz = ex - px, ey - py, ez - pz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        horiz = math.sqrt(dx * dx + dz * dz)
        bear_yaw = math.degrees(math.atan2(-dx, dz))
        bear_pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-6 else 0.0
        off_yaw = math.radians((bear_yaw - yaw + 180.0) % 360.0 - 180.0)
        off_pitch = math.radians((bear_pitch - pitch + 180.0) % 360.0 - 180.0)
        cls = "passive" if e["_species"] in _PASSIVE else "hostile"
        cands.append({"uuid": e["uuid"], "type": f"minecraft:{e['_species']}", "cls": cls,
                      "dx": dx, "dy": dy, "dz": dz, "dist": dist,
                      "off_yaw": off_yaw, "off_pitch": off_pitch})
    cands.sort(key=lambda c: c["dist"])
    return cands


def _capture_pick(base: str, before: list[dict], burst_s: float, tries: int,
                  attackable_uuids: set[str]) -> tuple[str | None, dict]:
    """KillAura's priority pick, via short fixed BURSTS (not a polling loop — a
    poll's multi-GET scan keeps KillAura on ~300ms, long enough to retarget and
    spray the scene). Each burst: ON for burst_s (~1 swing), OFF, then ONE scan.
    With knockback-pinned mobs the priority target stays put, so repeated bursts
    hammer the SAME entity → max-drop is its clean pick.

    Stale-filter guard: the first swing after a mode change can fire under the
    PREVIOUS filter (a 1-tick lag), hitting a now-excluded entity. We accept the
    pick only once an ATTACKABLE entity has been damaged; bursts that hit only
    excluded entities are treated as stale and retried (the filter has settled by
    the next burst). Returns the accumulated drops for the audit trail."""
    drops: dict = {}
    for _ in range(tries):
        _killaura(base, True)
        time.sleep(burst_s)
        _killaura(base, False)
        time.sleep(0.15)  # let the server->client HP echo land before scanning
        after = _scan_all(base, _PASSIVE + _HOSTILE)
        d = _drops(before, after)
        if any(u in attackable_uuids for u in d):
            return None, d  # caller picks among attackable by max-drop/nearest
        drops = d  # only-excluded hits → stale; keep for audit, retry
    return None, drops


def main() -> int:
    ap = argparse.ArgumentParser(description="§18.1 mixed-filter target capture")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burst", type=float, default=0.15, help="KillAura on-burst per swing (s)")
    ap.add_argument("--tries", type=int, default=6, help="bursts to retry until a hit lands")
    ap.add_argument("--out", default="results/sprint18/filter_dataset.jsonl")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    print(f"[filter_capture] base={base} relay={args.relay} scenes={args.scenes} seed={args.seed}")
    if not _wait_in_world(base):
        print("[filter_capture] FATAL: no player in world")
        return 2

    _relay(args.relay, "difficulty easy")
    _relay(args.relay, "time set 18000")           # night: hostiles don't burn
    _set(base, "Filter players", True)             # no PvP confound
    rng = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w")
    n_rows = n_flip = n_skip = n_artifact = 0
    species_all = _PASSIVE + _HOSTILE

    for si in range(args.scenes):
        scene = _gen_scene(rng)
        _clear(args.relay, species_all)
        p = _position(base)
        px, py, pz = float(p["x"]), float(p["y"]), float(p["z"])
        yaw, pitch = float(p.get("yaw", 0.0)), float(p.get("pitch", 0.0))
        _summon(args.relay, px, py, pz, scene)

        picks = {}
        rows_this = {}
        for mode_name, mode in _MODES.items():
            # fresh HP for a clean single-pick label
            _clear(args.relay, species_all)
            _summon(args.relay, px, py, pz, scene)
            cands = _candidates(base, px, py, pz, yaw, pitch)
            if cands is None:
                n_skip += 1
                continue
            if not _apply_mode(base, mode):
                n_skip += 1
                continue
            time.sleep(0.3)  # settle: let the filter take effect before KillAura runs
            # attach the per-candidate attackable oracle for this mode (policy x type)
            for c in cands:
                c["attackable"] = _attackable(c["cls"], mode)
            attackable_uuids = {c["uuid"] for c in cands if c["attackable"]}
            before = _scan_all(base, species_all)
            _, drops = _capture_pick(base, before, args.burst, args.tries, attackable_uuids)
            # keep only attackable drops (stale-filter swings on excluded entities
            # are not the policy's pick); skip if none landed on an attackable mob.
            adrops = {u: v for u, v in drops.items() if u in attackable_uuids}
            if not adrops:
                n_skip += 1
                continue
            # Label = max HP-drop among attackable, ties broken by NEAREST (matching
            # the pinned Priority=Distance policy, so scan-order can't steal the tie).
            dist_by_uuid = {c["uuid"]: c["dist"] for c in cands}
            pick_uuid = max(adrops, key=lambda u: (round(adrops[u], 1),
                                                   -dist_by_uuid.get(u, 1e9)))
            label = next((i for i, c in enumerate(cands) if c["uuid"] == pick_uuid), None)
            if label is None:
                n_skip += 1
                continue
            picks[mode_name] = label
            n_damaged = len(drops)
            row = {
                "scene_id": si, "mode": mode_name,
                "gt": {"filter_passive": bool(mode["Filter passive mobs"]),
                       "priority": mode["Priority"]},
                "cands": [{k: c[k] for k in
                           ("type", "cls", "dx", "dy", "dz", "dist",
                            "off_yaw", "off_pitch", "attackable")} for c in cands],
                "label": label,
                "label_type": cands[label]["type"],
                "n_damaged": n_damaged,
                "label_attackable": cands[label]["attackable"],
            }
            rows_this[mode_name] = row

        # only emit matched pairs (both modes captured) so the ablation is clean.
        # A label that is non-attackable in its own mode is a capture artifact (the
        # policy cannot legally pick a filtered entity) → discard the whole scene to
        # preserve the matched-pair structure. Counted, never silently dropped.
        if len(rows_this) == len(_MODES):
            if any(not r["label_attackable"] for r in rows_this.values()):
                n_artifact += 1
                continue
            flipped = len(set(picks.values())) > 1
            n_flip += int(flipped)
            for r in rows_this.values():
                r["scene_flipped"] = flipped
                f.write(json.dumps(r) + "\n")
                n_rows += 1
        if (si + 1) % 10 == 0:
            print(f"  scene {si+1}/{args.scenes}  rows={n_rows} flips={n_flip} "
                  f"skip={n_skip} artifact={n_artifact}")

    f.close()
    # restore ambient
    _set(base, "Filter passive mobs", False)
    _set(base, "Priority", "Angle")
    _killaura(base, False)
    _clear(args.relay, species_all)

    print("\n" + "=" * 64)
    print("§18.1 mixed-filter capture")
    print("=" * 64)
    print(f"  rows={n_rows}  matched scenes={n_rows//2}  flipped scenes={n_flip}")
    print(f"  skipped={n_skip}  discarded-artifact scenes={n_artifact}")
    print(f"  wrote {args.out}")
    return 0 if n_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
