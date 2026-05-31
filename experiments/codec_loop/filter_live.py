#!/usr/bin/env python3
"""§18.2 — the live g_t codec: serve the 18.1 prior in the sidecar, measure the
rate on REAL wire interacts under both filter modes.

The codec sidecar (craft.codec.server, /interact_rate) serves a saved prior and,
per outbound interact ATTACK, scores the obs.entity_set candidates under
P(idx | geom, type, obs.policy) — reading g_t straight off the wire (the obs.policy
plumbed in homunculus fa4466c) — and accumulates -log2 P(true target idx). The
index pointer reconstructs the target exactly, so this is LOSSLESS; the result is
the live RATE, auto-bucketed by the live filter mode.

Harness: arm the passthrough (sidecar sees every interact), then for each served
prior run KillAura over varied mixed scenes under both modes (attack_all /
protect_passive) so real interacts flow. Compare:
  geom+type+policy  (mode-aware) — should rate low in BOTH modes.
  geom+type         (mode-blind) — higher; can't resolve the flip.
The mode-blind − mode-aware gap should reproduce the §18.1 offline +~0.44 bits
on live wire data — proof the live codec carries g_t.

Run PEACEFUL? No — hostiles needed; difficulty easy + night (set by harness).

Usage:
    .venv/bin/python -m experiments.codec_loop.filter_live --port 25570 \
        --prior-dir results/sprint18/prior --scenes 40 --out results/sprint18/filter_live.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import _relay, _sidecar_base
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world
from experiments.codec_loop.filter_capture import (
    _PASSIVE, _HOSTILE, _clear, _gen_scene, _killaura, _position, _set, _summon, _KA,
)

import random


def _cfg_prior(sidecar: str, path: str | None) -> dict:
    body = {"interact_prior": {"path": path} if path else None}
    return _http("POST", f"{sidecar}/config", body, timeout=15)


def _rate_stats(sidecar: str) -> dict:
    return _http("GET", f"{sidecar}/interact_rate", timeout=8).get("interact_prior", {})


def _reset_rates(sidecar: str) -> None:
    _http("GET", f"{sidecar}/interact_rate/reset", timeout=8)


def _arm(base: str, codec_url: str) -> None:
    _http("POST", f"{base}/codec/passthrough/arm", {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)


def _disarm(base: str) -> dict:
    st = _http("GET", f"{base}/codec/passthrough/status", timeout=8)
    _http("POST", f"{base}/codec/passthrough/disarm")
    return st


_MODES = {"attack_all": False, "protect_passive": True}


def _drive(base: str, relay: str, scenes: list, on_s: float) -> None:
    """Run KillAura over each scene under both filter modes so real interacts flow
    to the sidecar. (No HP bookkeeping — the codec records the wire targets.)"""
    species = _PASSIVE + _HOSTILE
    for scene in scenes:
        p = _position(base)
        px, py, pz = float(p["x"]), float(p["y"]), float(p["z"])
        # clear wild hostiles that would draw KillAura off the scripted scene
        # (night/easy spawns creepers; OOV interacts skip but waste swings).
        _relay(relay, "kill @e[type=creeper]")
        for _name, fp in _MODES.items():
            _clear(relay, species)
            _summon(relay, px, py, pz, scene)
            _set(base, "Filter passive mobs", fp)
            _set(base, "Priority", "Distance")
            _killaura(base, True)
            time.sleep(on_s)
            _killaura(base, False)
            time.sleep(0.15)
    _clear(relay, species)


def main() -> int:
    ap = argparse.ArgumentParser(description="§18.2 live g_t codec rate")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--prior-dir", default="results/sprint18/prior")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--on", type=float, default=0.6, help="KillAura on-time per mode (s)")
    ap.add_argument("--out", default="results/sprint18/filter_live.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    sidecar = _sidecar_base(args.codec_url)
    print(f"[filter_live] base={base} sidecar={sidecar}")
    if not _wait_in_world(base):
        print("[filter_live] FATAL: no player in world")
        return 2

    _relay(args.relay, "difficulty easy")
    _relay(args.relay, "time set 18000")
    _set(base, "Filter players", True)
    rng = random.Random(args.seed)
    scenes = [_gen_scene(rng) for _ in range(args.scenes)]

    arms = {
        "geom+type+policy": os.path.join(args.prior_dir, "prior_geom_type_policy.pt"),
        "geom+type": os.path.join(args.prior_dir, "prior_geom_type.pt"),
    }
    results = {}
    _arm(base, args.codec_url)
    try:
        for arm, path in arms.items():
            r = _cfg_prior(sidecar, path)
            if not r.get("ok"):
                print(f"[filter_live] FAILED to load {arm}: {r}")
                continue
            _reset_rates(sidecar)
            print(f"\n[filter_live] serving {arm} — driving {args.scenes} scenes x 2 modes")
            _drive(base, args.relay, scenes, args.on)
            st = _rate_stats(sidecar)
            results[arm] = st
            print(f"  n={st.get('n')} mean_bits={st.get('mean_bits')} "
                  f"attack_all={st.get('mean_bits_attack_all')} "
                  f"protect={st.get('mean_bits_protect_passive')} "
                  f"argmax_acc={st.get('argmax_acc')}")
    finally:
        _disarm(base)
        _cfg_prior(sidecar, None)
        _set(base, "Filter passive mobs", False)
        _set(base, "Priority", "Angle")
        _killaura(base, False)
        _clear(args.relay, _PASSIVE + _HOSTILE)

    aware = results.get("geom+type+policy", {})
    blind = results.get("geom+type", {})
    gap = (blind.get("mean_bits") - aware.get("mean_bits")) \
        if (aware.get("mean_bits") is not None and blind.get("mean_bits") is not None) else None
    out = {"base": base, "scenes": args.scenes, "arms": results, "live_gap_bits": gap}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n" + "=" * 64)
    print("§18.2 LIVE g_t codec — rate on real wire interacts")
    print("=" * 64)
    for arm in ("geom+type", "geom+type+policy"):
        s = results.get(arm, {})
        print(f"  {arm:>18}: mean={s.get('mean_bits')}  "
              f"attack_all={s.get('mean_bits_attack_all')}  "
              f"protect={s.get('mean_bits_protect_passive')}  n={s.get('n')}")
    print(f"\n  LIVE bits g_t buys (mode-blind − mode-aware): "
          f"{gap:+.3f}" if gap is not None else "  gap: n/a")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
