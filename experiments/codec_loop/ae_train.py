#!/usr/bin/env python3
"""§16.1 — conditional β-VAE for the move family + RD curve vs the obs-rel baseline.

THE TEST (pre-registered §16.2 null): can a LEARNED conditional codec compress the
move fields BELOW the deterministic obs-relative baseline (obsrel_baseline.py) at
matched reconstruction fidelity? The §16.1 headroom preflight (ae_headroom.py) says
NO for a per-timestep model — cross-field correlation is ~0.27 b/pkt (mostly a
boolean), rotation residuals are ~independent, and the only big temporal structure
is a position still/moving gate (trivially RLE-able), not rotation. This trains the
AE to CONFIRM that analytically-predicted null empirically.

Design (frozen objective, §16.0 RESULTS):
  * INPUT = obs-relative reparam (the dominant free win baked in, NOT relearned):
    pos = (dx,dy,dz) delta vs obs (already so); rot = (wrap180(yaw-obs.yaw),
    pitch-obs.pitch). 5 floats + presence masks.
  * CONDITIONING c (decoder-known, reconstructable) = [pos_mask, rot_mask,
    on_ground, horiz_collision]. NO rollout-id, NO Baritone path (§B leakage #2).
  * β-VAE: encoder->(μ,logσ); z~N(μ,σ); decoder(z,c)->5 floats. KL (in BITS) is the
    RATE; sweep β to trace the rate-distortion curve. Distortion reported in
    PHYSICAL units (pos blocks, yaw/pitch degrees) on HELD-OUT rollouts.
  * LOSS is NOT plain MSE (§15: L2 rediscovers zero_biased -> rubberband). Asymmetric
    weighting encodes the drift-fatal/dropout-benign prior: AT-REST truth (|delta|<eps)
    is penalised hard toward exact 0 (no drift); MOVING truth dropout is cheap.
  * by-rollout split (held-out rollouts; §B random-split leakage lesson).

The RD curve overlays obsrel_baseline.json: if the AE points sit ON (not below-left
of) the baseline frontier, learning bought nothing -> the §16.2 null is confirmed.

Usage:
    .venv/bin/python -m experiments.codec_loop.ae_train \
        --betas 0.003,0.01,0.03,0.1,0.3,1.0 --epochs 60 \
        --out results/sprint16/ae_rd.json --ckpt results/sprint16/ae_ckpt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch
import torch.nn as nn

from experiments.codec_loop.offline_fidelity import MOVE_TYPES, TP_THRESHOLD
from experiments.codec_loop.obsrel import wrap180

STATIONARY = 0.02       # |delta| below this (blocks) = at rest (pos)
STAT_ROT = 0.5          # |residual| below this (deg) = holding heading (rot)
LN2 = math.log(2.0)


# ---- data ----------------------------------------------------------------
def _rollout_packets(root: str):
    """Yield (rollout_index, list-of-packet-dicts) per rollout dir, time-ordered."""
    for pf in sorted(glob.glob(f"{root}/rollout-*/packets.jsonl")):
        idx = int(pf.split("rollout-")[1].split("/")[0])
        rows = []
        with open(pf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("id") in MOVE_TYPES:
                    rows.append(d)
        yield root, idx, rows


def _featurize(d):
    """Packet dict -> (x[5], mask[5], context[4], stat[5]) using obs-relative rot.
    Returns None for TP/spawn artifacts. x is RAW physical units (scaled later)."""
    f = d.get("fields", {}) or {}
    o = d.get("obs", {}) or {}
    has_pos = bool(f.get("has_pos"))
    has_rot = bool(f.get("has_rot"))
    x = [0.0] * 5
    mask = [0.0] * 5
    stat = [0.0] * 5
    if has_pos:
        try:
            dx = float(f["x"]) - float(o["x"])
            dy = float(f["y"]) - float(o["y"])
            dz = float(f["z"]) - float(o["z"])
        except (KeyError, TypeError, ValueError):
            return None
        if max(abs(dx), abs(dy), abs(dz)) >= TP_THRESHOLD:
            return None
        x[0], x[1], x[2] = dx, dy, dz
        mask[0] = mask[1] = mask[2] = 1.0
        stat[0] = 1.0 if abs(dx) < STATIONARY else 0.0
        stat[1] = 1.0 if abs(dy) < STATIONARY else 0.0
        stat[2] = 1.0 if abs(dz) < STATIONARY else 0.0
    if has_rot:
        try:
            yr = wrap180(float(f["yaw"]) - float(o["yaw"]))
            pr = float(f["pitch"]) - float(o["pitch"])
        except (KeyError, TypeError, ValueError):
            return None
        x[3], x[4] = yr, pr
        mask[3] = mask[4] = 1.0
        stat[3] = 1.0 if abs(yr) < STAT_ROT else 0.0
        stat[4] = 1.0 if abs(pr) < STAT_ROT else 0.0
    ctx = [1.0 if has_pos else 0.0, 1.0 if has_rot else 0.0,
           1.0 if f.get("on_ground") else 0.0,
           1.0 if f.get("horizontal_collision") else 0.0]
    return x, mask, ctx, stat


def load_split(roots, holdout_each=1):
    """Build train/val tensors. Holds out the first `holdout_each` rollout indices
    of EACH set (by-rollout split). Returns dict of tensors + scale."""
    train, val = [], []
    for root in roots:
        for _r, idx, rows in _rollout_packets(root):
            bucket = val if idx < holdout_each else train
            for d in rows:
                feat = _featurize(d)
                if feat is not None:
                    bucket.append(feat)
    return train, val


def _to_tensors(rows, scale, device):
    X = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)
    M = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=device)
    C = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=device)
    S = torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device)
    Xs = X / scale  # scaled inputs/targets
    return X, Xs, M, C, S


# ---- model ---------------------------------------------------------------
class CondVAE(nn.Module):
    def __init__(self, d_latent, x_dim=5, c_dim=4, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, d_latent)
        self.lv = nn.Linear(hidden, d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent + c_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim))

    def forward(self, xs, c):
        h = self.enc(torch.cat([xs, c], -1))
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        xhat = self.dec(torch.cat([z, c], -1))
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(-1)  # nats, per-sample
        return xhat, kl


def asym_recon(xhat_s, xs, mask, stat, scale, w_rest=8.0, w_move=1.0):
    """Asymmetric masked recon in SCALED space. At-rest truth pulled hard to 0
    (drift fatal); moving truth dropout cheap. Returns mean over present fields."""
    # at-rest target is 0 (hold/standstill); moving target is the true value
    err = xhat_s - xs                       # deviation from true
    rest_err = xhat_s - 0.0                 # deviation from exact-zero hold
    w = mask * (stat * w_rest + (1 - stat) * w_move)
    # for at-rest fields penalise distance-from-zero; for moving, distance-from-true
    per = stat * rest_err.pow(2) + (1 - stat) * err.pow(2)
    return (w * per).sum() / w.sum().clamp_min(1.0)


# ---- eval: physical-unit distortion + rate (bits) ------------------------
@torch.no_grad()
def evaluate(model, val, scale):
    model.eval()
    X, Xs, M, C, S = val
    xhat_s, kl = model(Xs, C)
    xhat = xhat_s * scale
    kl_bits = (kl.mean().item()) / LN2
    out = {}
    # pos RMSE (blocks) over present pos axes
    posm = M[:, :3].reshape(-1).bool()
    pe = (xhat[:, :3] - X[:, :3]).reshape(-1)[posm]
    out["pos_rmse_blocks"] = float(torch.sqrt((pe**2).mean())) if pe.numel() else float("nan")
    # yaw/pitch RMSE (deg) over present rot
    rm = M[:, 3].bool()
    ye = (xhat[rm, 3] - X[rm, 3])
    ye = (ye + 180) % 360 - 180
    out["yaw_rmse_deg"] = float(torch.sqrt((ye**2).mean())) if ye.numel() else float("nan")
    pe2 = (xhat[rm, 4] - X[rm, 4])
    out["pitch_rmse_deg"] = float(torch.sqrt((pe2**2).mean())) if pe2.numel() else float("nan")
    # zero-mean-at-rest gate: reconstruction magnitude where truth is at rest
    sp = (S[:, :3].reshape(-1).bool()) & posm
    rp = xhat[:, :3].reshape(-1)[sp]
    out["at_rest_pos_rmse_blocks"] = float(torch.sqrt((rp**2).mean())) if rp.numel() else 0.0
    sy = S[:, 3].bool() & rm
    ry = xhat[sy, 3] - X[sy, 3]          # how far reconstructed heading drifts at rest
    ry = (ry + 180) % 360 - 180
    out["at_rest_yaw_rmse_deg"] = float(torch.sqrt((ry**2).mean())) if ry.numel() else 0.0
    out["kl_bits"] = kl_bits
    return out


def train_one(beta, d_latent, train_t, val_t, scale, *, epochs, lr, seed, device,
              w_rest, batch):
    torch.manual_seed(seed)
    model = CondVAE(d_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, Xs, M, C, S = train_t
    n = Xs.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xhat_s, kl = model(Xs[idx], C[idx])
            recon = asym_recon(xhat_s, Xs[idx], M[idx], S[idx], scale, w_rest=w_rest)
            loss = recon + beta * (kl.mean() / LN2)   # rate in bits
            opt.zero_grad(); loss.backward(); opt.step()
    metrics = evaluate(model, val_t, scale)
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="narrated,combat")
    ap.add_argument("--betas", default="0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--w-rest", type=float, default=8.0,
                    help="at-rest drift penalty weight (drift fatal / dropout benign)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-each", type=int, default=1)
    ap.add_argument("--out", default="results/sprint16/ae_rd.json")
    ap.add_argument("--ckpt", default="results/sprint16/ae_ckpt")
    args = ap.parse_args()

    setmap = {"narrated": "results/frozen_narrated", "combat": "results/frozen_combat"}
    roots = [setmap[s.strip()] for s in args.sets.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, val_rows = load_split(roots, args.holdout_each)
    # scale = per-field std on TRAIN (z-score-ish; robust enough, tails handled by net)
    Xtr = torch.tensor([r[0] for r in train_rows], dtype=torch.float32)
    Mtr = torch.tensor([r[1] for r in train_rows], dtype=torch.float32)
    scale = []
    for j in range(5):
        col = Xtr[:, j][Mtr[:, j].bool()]
        s = float(col.std()) if col.numel() > 1 else 1.0
        scale.append(s if s > 1e-6 else 1.0)
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)
    print(f"[ae_train] device={device} train={len(train_rows)} val={len(val_rows)} "
          f"holdout_each={args.holdout_each} scale={[round(s,3) for s in scale]}")

    train_t = _to_tensors(train_rows, scale_t, device)
    val_t = _to_tensors(val_rows, scale_t, device)

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    os.makedirs(args.ckpt, exist_ok=True)
    rd = []
    for beta in betas:
        model, m = train_one(beta, args.latent, train_t, val_t, scale_t,
                              epochs=args.epochs, lr=args.lr, seed=args.seed,
                              device=device, w_rest=args.w_rest, batch=args.batch)
        row = {"beta": beta, "latent": args.latent, **{k: round(v, 4) for k, v in m.items()}}
        rd.append(row)
        torch.save({"state_dict": model.state_dict(), "beta": beta,
                    "latent": args.latent, "scale": scale},
                   f"{args.ckpt}/vae_beta{beta}.pt")
        print(f"  beta={beta:<6} kl_bits={m['kl_bits']:.3f}  "
              f"posRMSE={m['pos_rmse_blocks']:.4f}b  yawRMSE={m['yaw_rmse_deg']:.3f}°  "
              f"pitchRMSE={m['pitch_rmse_deg']:.3f}°  atRest(pos {m['at_rest_pos_rmse_blocks']:.4f}b "
              f"yaw {m['at_rest_yaw_rmse_deg']:.3f}°)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"sets": roots, "latent": args.latent, "w_rest": args.w_rest,
                   "n_train": len(train_rows), "n_val": len(val_rows),
                   "scale": scale, "rd": rd}, f, indent=2)
    print(f"\n[ae_train] wrote {args.out}")
    print("RD curve = (kl_bits rate) vs (per-field RMSE distortion) on held-out rollouts.")
    print("Compare to obsrel_baseline.json: AE on/right-of that frontier => null confirmed.")


if __name__ == "__main__":
    main()
