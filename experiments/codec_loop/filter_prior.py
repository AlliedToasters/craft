"""§18.2 runtime — serve the saved g_t-conditioned interact-target prior.

The codec sidecar loads a bundle (filter_prior_train.py) and, per outbound
interact, scores the obs.entity_set candidates under P(idx | geom, type,
obs.policy) and reports the entropy-coding rate of the TRUE target index
(-log2 P(true idx)) — the live codec rate. Lossless: the index pointer
reconstructs the entity exactly; only the RATE depends on the prior.

The feature vector here MUST byte-match filter_bits._feat / the saved z-score:
  [dx, dy, dz, dist (z-scored), sin/cos off_yaw, sin/cos off_pitch]
  + species one-hot           (arm has 'type')
  + [float(filter_passive)]   (arm has 'policy')   <- the g_t bit, broadcast

torch is imported here (not at sidecar module load); the sidecar imports this
module only when a prior is configured.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_ZDIMS = (0, 1, 2, 3)  # dx,dy,dz,dist z-scored; angles pass through


def _build_model(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def load_prior(path: str) -> dict:
    """Load a saved bundle into a ready-to-score prior (CPU; per-packet inference
    is a handful of candidates — CPU is plenty and avoids GPU contention)."""
    b = torch.load(path, map_location="cpu")
    m = _build_model(b["dim"], b["hidden"])
    m.load_state_dict(b["state_dict"])
    m.eval()
    return {"model": m, "arm": b["arm"], "dim": b["dim"],
            "mean": b["zscore_mean"], "std": b["zscore_std"],
            "temp": float(b.get("temp", 1.0)),   # calibration T (overconfidence fix)
            "vocab": {s: i for i, s in enumerate(b["vocab"])},
            "vocab_size": len(b["vocab"]), "path": path}


def _candidates(obs: dict):
    """Dist-sorted candidates from obs.entity_set, with the same geom features
    filter_capture used. Returns (cands, id->index map)."""
    es = obs.get("entity_set") or []
    px, py, pz = float(obs.get("x", 0)), float(obs.get("y", 0)), float(obs.get("z", 0))
    yaw, pitch = float(obs.get("yaw", 0.0)), float(obs.get("pitch", 0.0))
    cands = []
    for e in es:
        pos = e.get("position") or [0, 0, 0]
        ex, ey, ez = float(pos[0]), float(pos[1]), float(pos[2])
        dx, dy, dz = ex - px, ey - py, ez - pz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        horiz = math.sqrt(dx * dx + dz * dz)
        bear_yaw = math.degrees(math.atan2(-dx, dz))
        bear_pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-6 else 0.0
        off_yaw = math.radians((bear_yaw - yaw + 180.0) % 360.0 - 180.0)
        off_pitch = math.radians((bear_pitch - pitch + 180.0) % 360.0 - 180.0)
        cands.append({"id": e.get("id"), "type": e.get("type", ""),
                      "dx": dx, "dy": dy, "dz": dz, "dist": dist,
                      "off_yaw": off_yaw, "off_pitch": off_pitch})
    cands.sort(key=lambda c: c["dist"])
    return cands


def _feat(c, filter_passive, prior):
    arm = prior["arm"]
    f = [c["dx"], c["dy"], c["dz"], c["dist"],
         math.sin(c["off_yaw"]), math.cos(c["off_yaw"]),
         math.sin(c["off_pitch"]), math.cos(c["off_pitch"])]
    mean, std = prior["mean"], prior["std"]
    for i, d in enumerate(_ZDIMS):
        f[d] = (f[d] - mean[i]) / std[i]
    if "type" in arm:
        oh = [0.0] * prior["vocab_size"]
        ti = prior["vocab"].get(c["type"])
        if ti is not None:
            oh[ti] = 1.0
        f += oh
    if "policy" in arm:
        f += [1.0 if filter_passive else 0.0]
    return f


def interact_rate(prior: dict, obs: dict, entity_id) -> dict | None:
    """Rate (bits) to entropy-code the TRUE target index under the prior, for one
    interact. None if not applicable (no obs/entity_set, target not in set, or a
    feature-dim mismatch). filter_passive is read from obs.policy (g_t)."""
    if not isinstance(obs, dict) or entity_id is None:
        return None
    cands = _candidates(obs)
    # Restrict to the species the prior was trained over — the codec's prior is
    # defined on its vocab; out-of-vocab entities (other players, wild mobs that
    # wandered into the obs radius) are not candidates it models. Dropping them
    # re-indexes the dist-sorted list to the modeled scene; an interact whose
    # target is OOV (e.g. KillAura hit a wild creeper) then skips cleanly.
    vocab = prior["vocab"]
    cands = [c for c in cands if c["type"] in vocab]
    if len(cands) < 1:
        return None
    idx = next((i for i, c in enumerate(cands) if c["id"] == entity_id), None)
    if idx is None:
        return None  # target not in the (vocab-filtered) entity_set — not codeable here
    policy = obs.get("policy") or {}
    filter_passive = bool(policy.get("Filter passive mobs", False))
    X = [_feat(c, filter_passive, prior) for c in cands]
    if any(len(row) != prior["dim"] for row in X):
        return None
    with torch.no_grad():
        logits = prior["model"](torch.tensor(X, dtype=torch.float32)).squeeze(-1)
        logp = torch.log_softmax(logits / prior["temp"], dim=0)  # calibrated rate
        rate = -logp[idx].item() / math.log(2.0)
        pred = int(logits.argmax().item())
    return {"rate_bits": rate, "idx": idx, "n_cands": len(cands),
            "filter_passive": filter_passive, "argmax_idx": pred,
            "argmax_correct": pred == idx}
