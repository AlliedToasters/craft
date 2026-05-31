#!/usr/bin/env python3
"""§21.2 analysis — VISUAL window-exit-subgoal distillation (neural_interface.md §21).

The §21 design invariant: FIX the target (Baritone's window-exit subgoal as a
deviation-from-goal-bearing sector + Δy class), MIGRATE the conditioning oracle→
perception. §21.0/§21.1 fed STRUCTURED terrain (a bearing-aligned floor/block/water
map). §21.2 feeds PIXELS — the agent's first-person frame — and asks the §21.1-
sharpened question: do pixels recover the DETOUR subset (where local planning earns
its keep) via terrain information the structured r=6 map lacked?

Anchoring the frame (why yaw is load-bearing): the PNG is a first-person view at the
player's CAMERA yaw, but the target is a WORLD-frame bearing deviation. So we feed the
goal direction RELATIVE TO THE CAMERA ([sin,cos] of the signed angle from facing→goal)
+ the bearing-free gvec, and let the CNN supply terrain. Without the per-tick yaw
(added to the sidecar for this rung) the frame can't be related to the goal at all.

THE CONTROL that makes it honest — `cam_only` (no pixels): the camera yaw already
encodes Baritone's CURRENT steering (≈ where it's about to go), so [sin,cos(rel),gvec]
alone is a strong detour cue (on a detour you're already heading off-goal). cam_only is
the visual analog of §21.1's `bearing_only` baseline. The headline is the PIXEL GAIN =
full_visual − cam_only on the detour subset = terrain the frame adds beyond the camera
angle. We also print the §21.0 structured detour number (~0.12) for the cross-rung read.

Held out by ROLLOUT (same convention as §21.0/§21.1). One sample per frame (~the
4.5k captured), nearest sidecar row by captured_at_ms.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_visual \
        --capture results/sprint21_visual/capture --target-r 5 --img-size 96 \
        --seeds 0 1 2 --out results/sprint21_visual/visual.json
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Pillow ≥10 moved resampling enums under Image.Resampling; keep a stable handle.
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

from experiments.codec_loop.nav_horizon import (
    N_SECTORS,
    _circ_sector_correct,
    _exit_dev,
    _global_vec,
    _goal_angle,
)


# --- frame ↔ tick join + features -------------------------------------------
def _cam_rel_angle(yaw_deg: float, goal_ang: float) -> float:
    """Signed angle (rad) from the camera facing to the goal direction, in the xz
    plane. MC yaw: horizontal look vector = (-sin(yaw), cos(yaw)) over (x, z). goal_ang
    = atan2(dz, dx) (nav_horizon convention) → goal vector (cos, sin) over (x, z)."""
    yr = math.radians(yaw_deg)
    fx, fz = -math.sin(yr), math.cos(yr)
    gx, gz = math.cos(goal_ang), math.sin(goal_ang)
    return math.atan2(fx * gz - fz * gx, fx * gx + fz * gz)


def _row_yaw(d):
    """Player camera yaw for the tick. Prefer a top-level `yaw` (post the sidecar
    add), else recover it from entity_set's SELF player entry — §17.2.2 already
    records each entity's yaw, and self is the minecraft:player nearest origin (the
    capture box spawns agents far apart, so the only nearby player is self). Returns
    None if no anchorable yaw. MC yaw is unwrapped (can exceed 360); sin/cos handle it."""
    if "yaw" in d:
        return d["yaw"]
    origin = d.get("origin")
    if not origin:
        return None
    cx, cz = origin[0] + 0.5, origin[2] + 0.5
    best, bestd = None, None
    for e in d.get("entity_set") or []:
        if "player" not in str(e.get("type", "")).lower() or "yaw" not in e:
            continue
        dd = (e.get("x", 1e9) - cx) ** 2 + (e.get("z", 1e9) - cz) ** 2
        if bestd is None or dd < bestd:
            bestd, best = dd, e["yaw"]
    return best if (bestd is not None and bestd <= 4.0) else None


def _load_frame(path: Path, size: int) -> np.ndarray:
    """RGB frame → (3, size, size) float32 in [0,1]."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), _BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(a, (2, 0, 1))


def load_visual_samples(capture_dir: Path, target_r: int, size: int, ms_tol: int = 400):
    """Per rollout: one sample per FRAME, paired to the nearest sidecar row by
    captured_at_ms (within ms_tol). Returns [(name, [sample,...])] where a sample is
    (img(3,S,S), vec[sin,cos,*gvec], dev_class, dy_class)."""
    rollouts = []
    for rdir in sorted(capture_dir.glob("rollout-*")):
        sc = rdir / "sidecar.jsonl.gz"
        fidx = rdir / "frames" / "frame_index.json"
        if not sc.exists() or not fidx.exists():
            continue
        rows = []
        try:
            with gzip.open(sc, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if "captured_at_ms" in d and "origin" in d:
                        rows.append(d)
        except (EOFError, OSError, gzip.BadGzipFile):
            pass
        if not rows:
            continue
        rows.sort(key=lambda d: d["captured_at_ms"])
        row_ms = [d["captured_at_ms"] for d in rows]

        frames = json.loads(fidx.read_text()).get("frames", [])
        samples = []
        for fr in frames:
            ms = fr["ms"]
            j = bisect.bisect_left(row_ms, ms)
            cand = [k for k in (j - 1, j) if 0 <= k < len(rows)]
            if not cand:
                continue
            k = min(cand, key=lambda i: abs(row_ms[i] - ms))
            if abs(row_ms[k] - ms) > ms_tol:
                continue
            d = rows[k]
            yaw = _row_yaw(d)
            if yaw is None:
                continue                       # no anchorable camera yaw → skip
            bs = d.get("baritone_state") or {}
            fwd, dest = bs.get("path_fwd"), bs.get("path_dest")
            if not fwd or len(fwd) < 2 or not dest:
                continue
            origin = d["origin"]
            goal_ang = _goal_angle(dest, origin)
            if goal_ang is None:
                continue
            tgt = _exit_dev(fwd, origin, target_r, goal_ang)
            if tgt is None:
                continue
            devc, dyc = tgt
            rel = _cam_rel_angle(yaw, goal_ang)
            gvec = _global_vec(dest, origin, target_r)
            vec = np.asarray([math.sin(rel), math.cos(rel), *gvec], dtype=np.float32)
            fp = rdir / "frames" / fr["file"]
            if not fp.exists():
                continue
            samples.append((_load_frame(fp, size), vec, devc, dyc))
        if samples:
            rollouts.append((rdir.name, samples))
    return rollouts


# --- model -------------------------------------------------------------------
class VisualHead(nn.Module):
    """Small CNN over the frame ⊕ the camera-relative goal vector → sector + Δy. With
    use_img=False it's the cam_only control (zeros the conv branch, vec-only MLP)."""
    def __init__(self, vec_dim: int, use_img: bool = True, hidden: int = 128):
        super().__init__()
        self.use_img = use_img
        feat = 0
        if use_img:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4))
            feat = 32 * 4 * 4
        self.body = nn.Sequential(nn.Linear(feat + vec_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.sector = nn.Linear(hidden, N_SECTORS)
        self.dyh = nn.Linear(hidden, 3)

    def forward(self, img, vec):
        if self.use_img:
            h = self.conv(img).flatten(1)
            h = torch.cat([h, vec], 1)
        else:
            h = vec
        h = self.body(h)
        return self.sector(h), self.dyh(h)


def _stack(samples, device):
    imgs = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
    vecs = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32, device=device)
    dev = torch.tensor([s[2] for s in samples], dtype=torch.long, device=device)
    dy = torch.tensor([s[3] for s in samples], dtype=torch.long, device=device)
    return imgs, vecs, dev, dy


def train_eval(rollouts, *, use_img, epochs, lr, seed, device, bs=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(rollouts)
    n_test = max(1, n // 3)
    test_names = {rollouts[i][0] for i in range(n - n_test, n)}
    tr = [s for name, ss in rollouts if name not in test_names for s in ss]
    te = [s for name, ss in rollouts if name in test_names for s in ss]
    if not tr or not te:
        return None
    Itr, Vtr, Dtr, Ytr = _stack(tr, device)
    Ite, Vte, Dte, Yte = _stack(te, device)

    straight_class = N_SECTORS // 2
    Bte = torch.full((len(te),), straight_class, dtype=torch.long, device=device)
    detour_mask = (_circ_sector_correct(Bte, Dte) == False)  # noqa: E712

    model = VisualHead(Vtr.shape[1], use_img=use_img).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    tail = max(1, epochs // 6)
    acc = {k: 0.0 for k in ("sector_within1", "dy_acc", "sector_ce_bits", "detour_within1")}
    seen = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Itr.shape[0], device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            ps, pd = model(Itr[idx], Vtr[idx])
            (ce(ps, Dtr[idx]) + ce(pd, Ytr[idx])).backward()
            opt.step()
        if ep < epochs - tail:
            continue
        model.eval()
        with torch.no_grad():
            ps, pd = model(Ite, Vte)
            hit1 = _circ_sector_correct(ps.argmax(1), Dte)
            acc["sector_within1"] += hit1.float().mean().item()
            acc["dy_acc"] += (pd.argmax(1) == Yte).float().mean().item()
            acc["sector_ce_bits"] += ce(ps, Dte).item() / math.log(2)
            acc["detour_within1"] += (hit1[detour_mask].float().mean().item()
                                      if detour_mask.any() else float("nan"))
        seen += 1
    out = {k: v / seen for k, v in acc.items()}
    out["straight_within1"] = _circ_sector_correct(Bte, Dte).float().mean().item()
    out["detour_frac"] = detour_mask.float().mean().item()
    out["n_train"] = len(tr)
    out["n_test"] = len(te)
    out["n_detour"] = int(detour_mask.sum().item())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="§21.2 visual subgoal distillation")
    ap.add_argument("--capture", default="results/sprint21_visual/capture")
    ap.add_argument("--target-r", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/sprint21_visual/visual.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = Path(args.capture)
    print(f"[nav_visual] loading frames+sidecar from {cap} (img={args.img_size}) ...", flush=True)
    rollouts = load_visual_samples(cap, args.target_r, args.img_size)
    total = sum(len(ss) for _, ss in rollouts)
    print(f"[nav_visual] {len(rollouts)} rollouts, {total} paired samples, device={device}")
    if len(rollouts) < 3 or total < 200:
        print("[nav_visual] too few samples / rollouts (need yaw-augmented recapture?)")
        return 2

    def arm(label, use_img):
        runs = [train_eval(rollouts, use_img=use_img, epochs=args.epochs, lr=args.lr,
                           seed=s, device=device) for s in args.seeds]
        runs = [r for r in runs if r]
        if not runs:
            return None
        a = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0] if isinstance(runs[0][k], float)}
        for k in ("n_train", "n_test", "n_detour"):
            a[k] = runs[0][k]
        det = [r["detour_within1"] for r in runs if not math.isnan(r["detour_within1"])]
        a["detour_std"] = (np.std(det).item() if len(det) > 1 else 0.0)
        print(f"  {label:12s} sec_±1={a['sector_within1']:.3f} dy={a['dy_acc']:.3f} "
              f"ce={a['sector_ce_bits']:.2f}b | DETOUR(frac={a['detour_frac']:.2f},"
              f"n={a['n_detour']}): {a['detour_within1']:.3f}±{a['detour_std']:.3f} "
              f"(straight={a['straight_within1']:.3f})", flush=True)
        return a

    print(f"[nav_visual] target_r={args.target_r}, seeds={args.seeds}\n")
    arms = {}
    arms["full_visual"] = arm("full_visual", True)
    arms["cam_only"] = arm("cam_only", False)

    fv = arms.get("full_visual") or {}
    co = arms.get("cam_only") or {}
    gain = fv.get("detour_within1", float("nan")) - co.get("detour_within1", float("nan"))
    print("\n=== §21.2 VISUAL READING ===")
    print(f"  full_visual detour±1 = {fv.get('detour_within1', float('nan')):.3f}")
    print(f"  cam_only    detour±1 = {co.get('detour_within1', float('nan')):.3f}  "
          f"(camera angle = Baritone's current heading, no pixels)")
    print(f"  PIXEL GAIN (full − cam_only) = {gain:+.3f}  = terrain the FRAME adds")
    print(f"  cross-rung: §21.0 STRUCTURED detour±1 ≈ 0.12 (floor/block/water @ r6)")

    out = {"capture": str(cap), "target_r": args.target_r, "img_size": args.img_size,
           "device": device, "seeds": list(args.seeds), "n_samples": total,
           "n_rollouts": len(rollouts), "arms": arms, "pixel_gain": gain}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
