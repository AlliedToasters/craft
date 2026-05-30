"""§13.1.4 — the online-vs-offline gap, measured by capture-replay.

Run stock Wurst KillAura LIVE against summoned waves while recording the SAME
(packet, entity_set) capture the offline work used, then replay the rung-A
attack-target head over KillAura's ACTUAL live picks. The replay accuracy IS the
online target-agreement, directly comparable to the 0.985 offline number (§11a /
§13.1.1) because it reuses `rung_a_target.load_attacks` verbatim.

Why this is faithful: KillAura's live ATTACK packets are visible in the stream
(verified: 47 ATTACK packets in a probe capture). Each ATTACK names the entity
KillAura chose (the label); the joined entity_set is the candidate pool. So we are
measuring whether the decoded decision transfers to the LIVE distribution
(entity_set jitter, multi-mob churn, real-time aim) — not a proxy.

Heads compared on the identical capture:
  geom gaze-on   — features exactly as trained
  geom gaze-off  — off_yaw/off_pitch neutralized → pure-position decision
  + analytic baselines (nearest, nearest-hostile)

KillAura stays ON (it is what we record). agent0 is buffed (resistance+regen) so it
survives the waves.

    HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 \
        .venv/bin/python -m experiments.next_packet.harness_capture \
            --out results/online_arena/cap0 --rounds 30
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import requests
import torch

from craft.config import HOMUNCULUS_BASE as H, SERVER_CMD_BASE as CMD
from .neural_selector import build_model
from . import rung_a_target as rat

# deterministic summon ring (no RNG), all within KillAura reach
RING = [(2, 0), (-2, 1), (0, 2), (1, -2), (3, 1), (-3, -1), (1, 3), (-1, -3)]


def cmd(s):
    return requests.post(f"{CMD}/cmd", json={"cmd": s}, timeout=5).json()


def killaura(on):
    return requests.post(f"{H}/wurst/hack",
                         json={"name": "KillAura", "enabled": on}, timeout=5).json()


def arm_both(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pk = os.path.join(out_dir, "packets.jsonl")
    sc = os.path.join(out_dir, "sidecar.jsonl.gz")
    rp = requests.post(f"{H}/packets/recording/arm",
                       json={"path": pk, "append": False}, timeout=5).json()
    rs = requests.post(f"{H}/obs/sidecar/arm",
                       json={"path": sc, "append": False}, timeout=5).json()
    return rp, rs


def disarm_both():
    requests.post(f"{H}/packets/recording/disarm", json={}, timeout=5)
    requests.post(f"{H}/obs/sidecar/disarm", json={}, timeout=5)


def n_zombies(radius=20):
    r = requests.get(f"{H}/scan_entities",
                     params={"type": "minecraft:zombie", "radius": radius, "limit": 16},
                     timeout=5).json()
    return len(r.get("entities", []))


def safety(player="agent0"):
    cmd(f"effect give {player} minecraft:resistance 120 4 true")
    cmd(f"effect give {player} minecraft:regeneration 120 3 true")
    cmd(f"effect give {player} minecraft:fire_resistance 120 0 true")


def carve(sx, sy, sz):
    cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy+2} {sz+1} minecraft:air")
    cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy-1} {sz+1} minecraft:stone")


def capture(out_dir, rounds, per_round, settle):
    print(f"[capture] -> {out_dir}  rounds={rounds} per_round={per_round}")
    print("killaura on:", killaura(True))
    cmd("difficulty easy")
    safety()
    cmd("kill @e[type=minecraft:zombie,distance=..40]")
    time.sleep(1.0)
    print("arm:", arm_both(out_dir))

    p = requests.get(f"{H}/position", timeout=5).json()
    px, py, pz = round(p["x"]), round(p["y"]), round(p["z"])
    for rnd in range(rounds):
        # keep ~per_round AI zombies alive — AI-enabled so they aggro and KillAura
        # engages (NoAI zombies don't aggro; KillAura ignores them).
        need = max(0, per_round - n_zombies())
        for k in range(need):
            ox, oz = RING[(rnd + k) % len(RING)]
            sx, sy, sz = px + ox, py, pz + oz
            carve(sx, sy, sz)
            cmd(f"summon minecraft:zombie {sx} {sy} {sz} "
                "{PersistenceRequired:1b,Silent:1b}")
        if rnd % 4 == 0:
            safety()
        time.sleep(settle)
        if rnd % 5 == 0:
            print(f"  round {rnd}: live≈{n_zombies()}")

    time.sleep(1.0)
    disarm_both()
    cmd("kill @e[type=minecraft:zombie,distance=..40]")
    print("[capture] disarmed + cleaned")


def replay_eval(out_dir):
    rows = rat.load_attacks([out_dir])
    if not rows:
        print("[replay] NO ATTACK events captured — check the capture dir / join.")
        return None
    types = sorted({c["type"] for r in rows for c in r["cands"] if c["type"]})
    evocab = rat.EntityVocab(types)
    avg_cand = float(np.mean([len(r["cands"]) for r in rows]))
    multi = [r for r in rows if len(r["cands"]) >= 2]
    print(f"[replay] {len(rows)} ATTACK events  avg_candidates={avg_cand:.2f}  "
          f"multi-candidate={len(multi)}")

    ck = torch.load("results/rung_a_target_ckpt/model.pt", map_location="cpu",
                    weights_only=False)
    model = build_model(ck["dim"], ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean = np.asarray(ck["geom_mean"], dtype=np.float32)
    std = np.asarray(ck["geom_std"], dtype=np.float32)

    def vec(c, gaze):
        f = rat.cand_features(c, evocab, use_type=False)  # geom-only, dim 8
        if not gaze:
            f[4], f[5], f[6], f[7] = 0.0, 1.0, 0.0, 1.0  # neutralize off_yaw/pitch
        for i in range(4):
            f[i] = (f[i] - float(mean[i])) / float(std[i])
        return f

    def acc(gaze, subset):
        if not subset:
            return float("nan")
        ok = 0
        for r in subset:
            X = torch.tensor([vec(c, gaze) for c in r["cands"]], dtype=torch.float32)
            with torch.no_grad():
                pred = int(model(X).squeeze(-1).argmax().item())
            ok += int(pred == r["label"])
        return ok / len(subset)

    res = {
        "n_events": len(rows), "n_multi": len(multi), "avg_cand": avg_cand,
        "geom_gaze_on_all": acc(True, rows),
        "geom_gaze_off_all": acc(False, rows),
        "geom_gaze_on_multi": acc(True, multi),
        "geom_gaze_off_multi": acc(False, multi),
        "baseline_nearest_all": rat.baseline_nearest(rows),
        "baseline_nearest_hostile_all": rat.baseline_nearest_hostile(rows),
        "baseline_nearest_multi": rat.baseline_nearest(multi) if multi else float("nan"),
    }
    print("\n=== ONLINE target-agreement (vs live KillAura picks) ===")
    print(f"  geom gaze-ON   all={res['geom_gaze_on_all']:.3f}  "
          f"multi-cand={res['geom_gaze_on_multi']:.3f}")
    print(f"  geom gaze-OFF  all={res['geom_gaze_off_all']:.3f}  "
          f"multi-cand={res['geom_gaze_off_multi']:.3f}")
    print(f"  baseline nearest         all={res['baseline_nearest_all']:.3f}  "
          f"multi={res['baseline_nearest_multi']:.3f}")
    print(f"  baseline nearest-hostile all={res['baseline_nearest_hostile_all']:.3f}")
    print("  (offline reference §13.1.1: geom val_best 0.954 / geom+type 0.985)")
    print("  NOTE: single-candidate frames are trivially correct — multi-cand is "
          "the real test of the decision.")
    with open(os.path.join(out_dir, "agreement.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/online_arena/cap0")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--per-round", type=int, default=4)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()
    if not args.replay_only:
        capture(args.out, args.rounds, args.per_round, args.settle)
    replay_eval(args.out)


if __name__ == "__main__":
    main()
