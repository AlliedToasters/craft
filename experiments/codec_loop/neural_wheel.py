#!/usr/bin/env python3
"""§19.1 — neural takes the wheel: serve the §18.1 g_t prior as the live CONTROLLER
and prove, on the server, that the neural target-selector (not KillAura's heuristic)
decides which entity gets hit — and that it is corrigible to the codec's g_t.

This is the rung past §18.2. §18.2 SERVED the prior as a passive observer (rate
read, lossless, no mutation). §19.1 turns substitute=True: the prior's argmax pick
OVERWRITES the wire entity_id, so the packet that reaches the server names the
NEURAL target. Wurst (KillAura) still ORIGINATES every swing and owns attack timing
— neural only selects WHICH entity. Same-tick feedforward: the decision uses the
obs the packet already carries (current-tick entity_set + policy); KillAura is
itself memoryless per-tick, so no cooldown/temporal model is needed (the packet
cadence IS the cooldown). See neural_interface.md §19.

The decisive metric is WHOSE HP DROPS, not the wire field. The server trusts the
action packet's entity_id (loose reach check, no rotation raycast — the §17.0
aim-carrier mechanism / reference_server_trusts_client_target), so a substituted id
naming a different in-reach entity lands the hit there even though the player is
rotated toward KillAura's original target. Damage on the server = the neural pick.

Duel geometry (pinned, not random): a PASSIVE (sheep) NEARER + a HOSTILE (zombie)
FARTHER, both inside KillAura reach (~4.25), Priority=Distance. So:
  * attack_all g_t -> the nearer passive is a valid target -> pick the sheep.
  * protect g_t    -> the passive is filtered -> divert to the zombie.
A clean argmax flip the §18.1 prior makes on this geometry (sheep@~2 / zombie@~3.8).

TEST A (effectiveness/faithfulness):  substitute=True, gt_override=None (neural
  reads the WIRE policy, same g_t KillAura uses). Drive both real wire modes.
  Neural should AGREE with KillAura (argmax_acc high) and its substituted hits
  should land — serving neural as the controller reproduces the heuristic.

TEST B (corrigibility, the headline):  KillAura's own filter FIXED at attack_all
  (Filter passive mobs=False) the WHOLE time, so KillAura always aims at the nearer
  sheep. Flip ONLY the codec's gt_override:
    gt_override=False (attack) -> neural agrees -> the SHEEP takes the damage.
    gt_override=True  (protect)-> neural diverts -> the ZOMBIE takes the damage,
                                  the sheep is PROTECTED — by the neural controller,
                                  with KillAura's filter unchanged.
  The flip in whose-HP-drops with the heuristic held constant IS corrigibility:
  the controller obeys the codec's g_t authority interface, not the wire heuristic.

Run NON-PEACEFUL (hostiles needed): difficulty easy + night (set by harness).

Usage:
    .venv/bin/python -m experiments.codec_loop.neural_wheel --port 25570 \
        --prior results/sprint18/prior/prior_geom_type_policy.pt \
        --reps 12 --out results/sprint19/neural_wheel.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import _relay, _sidecar_base
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world
from experiments.codec_loop.filter_capture import (
    _clear, _drops, _killaura, _position, _scan_all, _set, _summon,
)

_PASSIVE_SP = "sheep"
_HOSTILE_SP = "zombie"
_SPECIES = [_PASSIVE_SP, _HOSTILE_SP]


# --- sidecar control ---------------------------------------------------------
def _cfg_prior(sidecar: str, path: str | None, *, substitute: bool = False,
               gt_override=None) -> dict:
    body = {"interact_prior": (None if path is None else
            {"path": path, "substitute": substitute, "gt_override": gt_override})}
    return _http("POST", f"{sidecar}/config", body, timeout=15)


def _rate_stats(sidecar: str) -> dict:
    return _http("GET", f"{sidecar}/interact_rate", timeout=8).get("interact_prior", {})


def _reset_rates(sidecar: str) -> None:
    _http("GET", f"{sidecar}/interact_rate/reset", timeout=8)


def _arm(base: str, codec_url: str) -> None:
    _http("POST", f"{base}/codec/passthrough/arm", {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)


def _disarm(base: str) -> None:
    _http("POST", f"{base}/codec/passthrough/disarm")


# --- scene -------------------------------------------------------------------
def _duel() -> list[dict]:
    """Passive NEARER (sheep@~2.0) + hostile FARTHER (zombie@~3.8), both in reach.
    Placed on the same axis so the dist-sort is unambiguous and the player only has
    to face roughly one way for KillAura to engage. _summon offsets these from the
    player position and applies the no-knockback/no-AI tags."""
    return [{"species": _PASSIVE_SP, "cls": "passive", "dist": 2.0, "dx": 2.0, "dz": 0.0},
            {"species": _HOSTILE_SP, "cls": "hostile", "dist": 3.8, "dx": 3.8, "dz": 0.0}]


def _run_rep(base: str, relay: str, on_s: float) -> dict | None:
    """One duel: summon, snapshot HP, burst KillAura, snapshot HP, report drops by
    species. Returns {passive_drop, hostile_drop} (HP lost) or None on a bad scene."""
    p = _position(base)
    px, py, pz = float(p["x"]), float(p["y"]), float(p["z"])
    _relay(relay, "kill @e[type=creeper]")
    _clear(relay, _SPECIES)
    _summon(relay, px, py, pz, _duel())
    before = _scan_all(base, _SPECIES, radius=10)
    if not before:
        return None
    _killaura(base, True)
    time.sleep(on_s)
    _killaura(base, False)
    time.sleep(0.2)
    after = _scan_all(base, _SPECIES, radius=10)
    drops = _drops(before, after)
    sp_of = {e["uuid"]: e.get("_species") for e in before}
    passive_drop = sum(d for u, d in drops.items() if sp_of.get(u) == _PASSIVE_SP)
    hostile_drop = sum(d for u, d in drops.items() if sp_of.get(u) == _HOSTILE_SP)
    return {"passive_drop": round(passive_drop, 3), "hostile_drop": round(hostile_drop, 3)}


def _drive(base: str, relay: str, reps: int, on_s: float) -> list[dict]:
    out = []
    for _ in range(reps):
        r = _run_rep(base, relay, on_s)
        if r is not None:
            out.append(r)
    return out


def _summarize(reps: list[dict]) -> dict:
    n = len(reps)
    if not n:
        return {"n": 0}
    hp = sum(r["hostile_drop"] > 0.5 for r in reps)        # hostile took a hit
    pp = sum(r["passive_drop"] > 0.5 for r in reps)        # passive took a hit
    return {"n": n,
            "hostile_hit_frac": round(hp / n, 3),
            "passive_hit_frac": round(pp / n, 3),
            "passive_protected_frac": round(1 - pp / n, 3),
            "mean_passive_drop": round(sum(r["passive_drop"] for r in reps) / n, 2),
            "mean_hostile_drop": round(sum(r["hostile_drop"] for r in reps) / n, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="§19.1 neural takes the wheel — live corrigibility")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--prior", default="results/sprint18/prior/prior_geom_type_policy.pt")
    ap.add_argument("--reps", type=int, default=12, help="duels per arm")
    ap.add_argument("--on", type=float, default=0.8, help="KillAura on-time per duel (s)")
    ap.add_argument("--out", default="results/sprint19/neural_wheel.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    sidecar = _sidecar_base(args.codec_url)
    print(f"[neural_wheel] base={base} sidecar={sidecar} prior={args.prior}")
    if not _wait_in_world(base):
        print("[neural_wheel] FATAL: no player in world")
        return 2

    _relay(args.relay, "say §19.1 neural-takes-the-wheel: corrigibility run")
    _relay(args.relay, "difficulty easy")
    _relay(args.relay, "time set 18000")
    _set(base, "Filter players", True)
    _set(base, "Priority", "Distance")  # nearest-first so KillAura aims at the near passive

    results: dict = {"base": base, "prior": args.prior, "reps": args.reps}
    _arm(base, args.codec_url)
    try:
        # ---- TEST A: effectiveness — neural reads the WIRE policy (gt_override=None).
        # Drive both real wire modes; neural should track the wire filter and agree
        # with KillAura, and its substituted hits should land.
        print("\n[neural_wheel] TEST A — effectiveness (gt_override=None, both wire modes)")
        a = {}
        for mode_name, fp in (("attack_all", False), ("protect_passive", True)):
            r = _cfg_prior(sidecar, args.prior, substitute=True, gt_override=None)
            if not r.get("ok"):
                print(f"  FAILED to arm prior: {r}")
                continue
            _reset_rates(sidecar)
            _set(base, "Filter passive mobs", fp)   # the WIRE g_t neural reads
            _relay(args.relay, f"say  test A / wire={mode_name}")
            reps = _drive(base, args.relay, args.reps, args.on)
            st = _rate_stats(sidecar)
            a[mode_name] = {"hp": _summarize(reps),
                            "argmax_acc": st.get("argmax_acc"),
                            "n_interacts": st.get("n")}
            print(f"  wire={mode_name:>15}: {a[mode_name]['hp']}  "
                  f"argmax_acc={st.get('argmax_acc')} n_int={st.get('n')}")
        results["test_a_effectiveness"] = a

        # ---- TEST B: corrigibility — KillAura filter FIXED at attack_all the whole
        # time (it always aims at the nearer sheep). Flip ONLY the codec gt_override.
        print("\n[neural_wheel] TEST B — corrigibility (wire FIXED attack_all, flip codec g_t)")
        _set(base, "Filter passive mobs", False)    # KillAura: attack everything, FIXED
        b = {}
        for ov_name, ov in (("gt=attack", False), ("gt=protect", True)):
            r = _cfg_prior(sidecar, args.prior, substitute=True, gt_override=ov)
            if not r.get("ok"):
                print(f"  FAILED to arm prior: {r}")
                continue
            _reset_rates(sidecar)
            _relay(args.relay, f"say  test B / wire=attack_all codec={ov_name}")
            reps = _drive(base, args.relay, args.reps, args.on)
            st = _rate_stats(sidecar)
            b[ov_name] = {"hp": _summarize(reps),
                          "argmax_type_attack_all": st.get("argmax_type_attack_all"),
                          "n_interacts": st.get("n")}
            print(f"  codec {ov_name:>11}: {b[ov_name]['hp']}  "
                  f"neural_picks={st.get('argmax_type_attack_all')} n_int={st.get('n')}")
        results["test_b_corrigibility"] = b
    finally:
        _disarm(base)
        _cfg_prior(sidecar, None)
        _set(base, "Filter passive mobs", False)
        _set(base, "Priority", "Angle")
        _killaura(base, False)
        _clear(args.relay, _SPECIES)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    print("\n" + "=" * 68)
    print("§19.1 NEURAL TAKES THE WHEEL — whose HP drops decides who's driving")
    print("=" * 68)
    b = results.get("test_b_corrigibility", {})
    atk = b.get("gt=attack", {}).get("hp", {})
    pro = b.get("gt=protect", {}).get("hp", {})
    print("TEST B (KillAura filter FIXED attack_all; flip codec g_t):")
    print(f"  codec=attack : passive_hit={atk.get('passive_hit_frac')} "
          f"hostile_hit={atk.get('hostile_hit_frac')}")
    print(f"  codec=protect: passive_hit={pro.get('passive_hit_frac')} "
          f"hostile_hit={pro.get('hostile_hit_frac')}  "
          f"(passive_protected={pro.get('passive_protected_frac')})")
    print("  -> corrigible if the passive is HIT under codec=attack but PROTECTED "
          "under codec=protect,\n     with KillAura's own filter unchanged.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
