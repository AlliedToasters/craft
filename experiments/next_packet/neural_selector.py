"""§13.1.3 — the live neural target-selector. Loads the rung-A attack-target
checkpoint (results/rung_a_target_ckpt/model.pt + feature_spec.json), reads the live
entity set from homunculus, scores each candidate with the exact feature contract the
head was trained on, and drives /attack_entity at the argmax — a 2-4 Hz selection loop
(target selection changes slowly; no 20 Hz needed). KillAura's auto-aim servo is NOT
used; Attacker.java rotates + attacks the chosen target itself (the decision is
gaze-independent, neural_interface.md §13.0).

Feature parity is load-bearing: the per-candidate vector here MUST match
rung_a_target.cand_features byte-for-byte (same dx,dy,dz,dist + off_yaw/off_pitch
sin/cos, same geom z-score from the checkpoint) or the offline head scores garbage live.

    from experiments.next_packet.neural_selector import NeuralSelector
    sel = NeuralSelector()                       # loads ckpt + spec
    target, scores, cands = sel.pick()           # one decode
    sel.attack_loop(hz=3, duration_s=20)         # live selection loop

Env: HOMUNCULUS_PORT=25570 (agent0).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import requests
import torch
import torch.nn as nn

from craft.config import HOMUNCULUS_BASE

CKPT_DIR = "results/rung_a_target_ckpt"

# KillAura's hostile candidate pool. /scan_entities is per-type, so we query each
# and pool. At deploy we restrict to hostiles so the selector competes with KillAura
# on the same candidate set.
HOSTILE_TYPES = [
    "minecraft:zombie", "minecraft:skeleton", "minecraft:creeper",
    "minecraft:spider", "minecraft:cave_spider", "minecraft:witch",
    "minecraft:drowned", "minecraft:husk", "minecraft:zombie_villager",
    "minecraft:enderman", "minecraft:pillager", "minecraft:vindicator",
]


def build_model(dim, hidden):
    """MUST match rung_a_target_train.build_model exactly."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class NeuralSelector:
    def __init__(self, ckpt_dir=CKPT_DIR, base=None, gaze=True):
        self.base = base or HOMUNCULUS_BASE
        # gaze=False neutralizes off_yaw/off_pitch (constant across candidates) so the
        # pick is a PURE position decision (dx,dy,dz,dist) — the gaze-free A/B arm
        # (§13.1.4). The attack target should be gaze-INDEPENDENT (the original combat
        # data had KillAura auto-aiming), so a small gaze-on/off gap is the evidence.
        self.gaze = gaze
        with open(os.path.join(ckpt_dir, "feature_spec.json")) as f:
            self.spec = json.load(f)
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu",
                          weights_only=False)
        self.dim = ckpt["dim"]
        self.use_type = ckpt["use_type"]
        self.geom_mean = np.asarray(ckpt["geom_mean"], dtype=np.float32)
        self.geom_std = np.asarray(ckpt["geom_std"], dtype=np.float32)
        self.model = build_model(self.dim, ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.type_index = {t: i for i, t in
                           enumerate(self.spec.get("type_onehot_vocab") or [])}

    # ----- live obs -----
    def player(self):
        return requests.get(f"{self.base}/position", timeout=5).json()

    def candidates(self, radius=16, limit=16):
        """Pooled hostile candidates from /scan_entities, nearest-first overall."""
        pool = []
        for t in HOSTILE_TYPES:
            try:
                r = requests.get(f"{self.base}/scan_entities",
                                 params={"type": t, "radius": radius, "limit": limit},
                                 timeout=5).json()
            except requests.RequestException:
                continue
            pool.extend(r.get("entities", []))
        pool.sort(key=lambda e: e.get("distance", 1e9))
        return pool

    # ----- features (must mirror rung_a_target.cand_features) -----
    def _cand_vec(self, e, px, py, pz, cur_yaw, cur_pitch):
        ex, ey, ez = e["position"]
        dx, dy, dz = ex - px, ey - py, ez - pz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        horiz = math.sqrt(dx * dx + dz * dz)
        bear_yaw = math.degrees(math.atan2(-dx, dz))
        bear_pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-6 else 0.0
        off_yaw = math.radians((bear_yaw - cur_yaw + 180.0) % 360.0 - 180.0)
        off_pitch = math.radians((bear_pitch - cur_pitch + 180.0) % 360.0 - 180.0)
        if not self.gaze:
            off_yaw = off_pitch = 0.0  # constant → no influence on the argmax
        f = [dx, dy, dz, dist,
             math.sin(off_yaw), math.cos(off_yaw),
             math.sin(off_pitch), math.cos(off_pitch)]
        if self.use_type:
            oh = [0.0] * len(self.type_index)
            ti = self.type_index.get(e.get("type"))
            if ti is not None:
                oh[ti] = 1.0
            f += oh
        for i in range(4):  # z-score the geom dims with the checkpoint's stats
            f[i] = (f[i] - float(self.geom_mean[i])) / float(self.geom_std[i])
        return f

    def score(self, cands, player=None):
        p = player or self.player()
        px, py, pz = p["x"], p["y"], p["z"]
        cy, cp = p.get("yaw", 0.0), p.get("pitch", 0.0)
        X = np.asarray([self._cand_vec(e, px, py, pz, cy, cp) for e in cands],
                       dtype=np.float32)
        with torch.no_grad():
            s = self.model(torch.from_numpy(X)).squeeze(-1).numpy()
        return s, p

    def pick(self, radius=16):
        """Argmax hostile candidate. Returns (entity|None, scores|None, cands)."""
        cands = self.candidates(radius=radius)
        if not cands:
            return None, None, []
        scores, _ = self.score(cands)
        return cands[int(np.argmax(scores))], scores, cands

    # ----- attack -----
    def attack(self, uuid):
        return requests.post(f"{self.base}/attack_entity", json={"uuid": uuid},
                             timeout=8).json()

    def attack_loop(self, hz=3, duration_s=20, radius=16, verbose=True):
        period = 1.0 / hz
        t_end = time.time() + duration_s
        ticks = []
        while time.time() < t_end:
            t0 = time.time()
            target, scores, cands = self.pick(radius=radius)
            if target is None:
                if verbose:
                    print("  (no candidates)")
                ticks.append({"t": t0, "n_cands": 0, "target": None})
            else:
                res = self.attack(target["uuid"])
                ticks.append({"t": t0, "n_cands": len(cands),
                              "target_uuid": target["uuid"],
                              "target_type": target["type"],
                              "target_dist": target["distance"], "result": res})
                if verbose:
                    print(f"  pick {target['type']} d={target['distance']:.1f} "
                          f"hp {res.get('health_before')}->{res.get('health_after')} "
                          f"killed={res.get('killed')} reason={res.get('reason')}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        return ticks


def _selftest():
    """Smoke: load ckpt, summon 3 zombies in cleared pockets, score, print the pick."""
    import requests as _r
    from craft.config import SERVER_CMD_BASE as CMD

    def cmd(s):
        return _r.post(f"{CMD}/cmd", json={"cmd": s}, timeout=5).json()

    sel = NeuralSelector()
    print(f"loaded: dim={sel.dim} use_type={sel.use_type} vocab={len(sel.type_index)}")
    cmd("difficulty easy")
    p = sel.player()
    px, py, pz = round(p["x"]), round(p["y"]), round(p["z"])
    cmd(f"kill @e[type=minecraft:zombie,distance=..40]")
    for ox, oz in ((3, 0), (-3, 2), (0, 4)):
        sx, sy, sz = px + ox, py, pz + oz
        cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy+2} {sz+1} minecraft:air")
        cmd(f"fill {sx-1} {sy-1} {sz-1} {sx+1} {sy-1} {sz+1} minecraft:stone")
        cmd(f"summon minecraft:zombie {sx} {sy} {sz} "
            "{NoAI:1b,PersistenceRequired:1b,Silent:1b}")
    time.sleep(1.3)
    target, scores, cands = sel.pick()
    print(f"candidates={len(cands)}")
    if cands is not None:
        for e, s in zip(cands, scores):
            print(f"  {e['type']:22s} d={e['distance']:5.2f} score={s:+.3f}"
                  f"{'  <-- PICK' if e is target else ''}")
    cmd("kill @e[type=minecraft:zombie,distance=..40]")


if __name__ == "__main__":
    _selftest()
