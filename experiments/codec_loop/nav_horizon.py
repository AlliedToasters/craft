#!/usr/bin/env python3
"""§21.0 analysis — the navigation HORIZON CURVE (neural_interface.md §21).

One experiment, three readings. From the §21.0 capture (TickSidecarRecorder rows
carrying block_grid + baritone_state.path_fwd + path_dest), for each receptive
radius r in a sweep we ask: can a small head predict WHERE BARITONE'S PLANNED PATH
EXITS RADIUS r (the window-exit subgoal) from only the LOCAL terrain within r plus
a goal bearing?

  * TARGET (frozen across the whole §21 arc): the window-exit subgoal = the first
    forward path node beyond L∞-horizontal radius r. Represented as a 16-way bearing
    SECTOR + a 3-way Δy class (down/level/up) — clean classification, accuracy reads
    directly as a horizon curve (the design-locked representation).
  * FEATURES windowed to r: a local heightmap (relative surface height per column)
    + a water channel, from block_grid; plus the goal BEARING (sin/cos + log-distance
    bucket + goal-beyond-window flag) — the minimal global signal, so we don't
    penalise the model for not knowing an arbitrary far goal's direction.
  * Held-out by ROLLOUT (terrain generalisation, not memorisation).

Three readings of accuracy(r):
  (a) NAVIGATION HORIZON  — at what r does prediction saturate? (adding receptive
      field stops helping = the local-planning horizon).
  (b) DISTILLATION ACCURACY — how well a tiny head reproduces Baritone's local plan.
  (c) PATH-CODEC RESIDUAL — 1−acc (and the sector cross-entropy in bits) = the part
      of the subgoal NOT explained by local terrain+bearing = the irreducibly-global
      remainder. THIS RESOLVES §20.0's open caveat: the 437× "stream→goal" figure
      conflated cheap move→path compression with the expensive path→goal A* inversion;
      the residual at r is exactly the size of the inversion job that local prediction
      can't do and perception (§21.2+) must later supply.

Baselines that make the curve legible:
  * straight-line  — predict sector = the bearing-to-goal sector ("just head at the
    goal"). The gap (model − straight) = the value of LOCAL PLANNING (detours).
  * bearing-only   — the same head with terrain ablated. The gap (full − bearing) =
    the TERRAIN contribution at radius r.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_horizon \
        --capture results/sprint21/capture --r-min 1 --r-max 10 \
        --out results/sprint21/horizon.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_SECTORS = 16
GRID_R = 10                # CAPTURE_GRID_RADIUS in TickSidecarRecorder
WATER_KEYS = ("water", "kelp", "seagrass", "bubble_column")


# --- sample extraction -------------------------------------------------------
def _is_water(name: str) -> bool:
    return any(k in name for k in WATER_KEYS)


def _column_maps(row: dict):
    """Per (dx,dz) column within ±GRID_R: the SET of dy with a non-water solid block,
    and whether the column holds water. We keep the full solid set (not just a max
    height) because the navigation-relevant quantity is the WALKABLE FLOOR — the
    nearest level with solid-below + 2-air-above — not the topmost block. Taking the
    max height made tree CANOPY the 'surface' (leaves/logs are solid), which corrupts
    every forested biome (half this capture); Baritone walks UNDER the canopy weaving
    between trunks, so the floor + a blocked-column flag is what it actually plans on."""
    palette = row["block_palette"]
    is_water_idx = [_is_water(str(p)) for p in palette]
    solid: dict[tuple[int, int], set] = {}
    water: dict[tuple[int, int], bool] = {}
    for cell in row["block_grid"]:
        pi, dx, dy, dz = cell
        key = (dx, dz)
        if is_water_idx[pi]:
            water[key] = True
            continue
        solid.setdefault(key, set()).add(dy)
    return solid, water


def _floor_rel(solid_dys):
    """The standable feet-level nearest the player's own (dy=0), searched in a small
    band: a level d with solid at d-1 and air at d and d+1. Returns d (negative=step
    down, positive=step up) or None if the column is BLOCKED (a wall/trunk with no
    standable gap near the traversal level, or a deep void)."""
    best = None
    for d in range(-4, 3):
        if (d - 1) in solid_dys and d not in solid_dys and (d + 1) not in solid_dys:
            if best is None or abs(d) < abs(best):
                best = d
    return best


def _goal_angle(path_dest, origin):
    ox, _oy, oz = origin
    dx, dz = path_dest[0] - ox, path_dest[2] - oz
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(dz, dx)


def _exit_dev(path_fwd, origin, r, goal_ang):
    """The window-exit subgoal as a RELATIVE DEVIATION from straight-line bearing.

    First forward node beyond L∞-horizontal radius r → its bearing, minus the
    goal bearing, wrapped to a 16-way deviation class centred on 8 (=straight).
    Predicting the DEVIATION (not the absolute sector) makes the task bearing-
    invariant: one "given terrain ahead, deviate Δ" policy instead of 16 absolute
    ones — the inductive bias that lets a small head generalise across biomes.
    Returns (dev_class, dy_class) or None if the path never exits radius r."""
    ox, oy, oz = origin
    for node in path_fwd:
        nx, ny, nz = node
        dx, dz = nx - ox, nz - oz
        if max(abs(dx), abs(dz)) > r:
            ang = math.atan2(dz, dx)
            rel = (ang - goal_ang) % (2 * math.pi)               # [0, 2pi)
            dev = int(round(rel / (2 * math.pi) * N_SECTORS)) % N_SECTORS  # 0..15, 0=straight
            dev_class = (dev + N_SECTORS // 2) % N_SECTORS        # centre straight at 8
            ddy = ny - oy
            dy_class = 1 + (1 if ddy > 0 else (-1 if ddy < 0 else 0))
            return dev_class, dy_class
    return None


def _global_vec(path_dest, origin, target_r):
    """The minimal GLOBAL signal kept after the frame is rotated goal-forward:
    log-distance to goal + a goal-beyond-the-action-window flag. (The bearing
    itself is absorbed by the rotation, so no sin/cos here.)"""
    ox, _oy, oz = origin
    dist = math.hypot(path_dest[0] - ox, path_dest[2] - oz)
    dist_norm = math.log1p(dist) / math.log1p(256.0)
    beyond = 1.0 if dist > target_r else 0.0
    return [dist_norm, beyond]


def _aligned_terrain(solid, water, feat_r, goal_ang):
    """Local terrain sampled in a BEARING-ALIGNED frame: axis +forward points at the
    goal, +lateral to its left. Rotation removes the absolute-bearing nuisance
    variable, so the head sees terrain in the only frame the deviation decision
    depends on. Three channels per cell, nearest-column sampled:
      floor  — walkable feet-level relative to the player (0 when blocked)
      block  — 1 if the column is blocked (wall/trunk/void: no standable level)
      water  — 1 if the column holds water
    The block channel is what carries trees/walls — the obstacles a detour routes
    around — which a bare height map folded into the floor value."""
    span = 2 * feat_r + 1
    floor = np.zeros((span, span), dtype=np.float32)
    block = np.ones((span, span), dtype=np.float32)    # default blocked (unloaded/void)
    wmap = np.zeros((span, span), dtype=np.float32)
    cos, sin = math.cos(goal_ang), math.sin(goal_ang)
    for fi in range(-feat_r, feat_r + 1):          # forward (toward goal)
        for li in range(-feat_r, feat_r + 1):      # lateral (left)
            dx = fi * cos - li * sin
            dz = fi * sin + li * cos
            key = (int(round(dx)), int(round(dz)))
            i, j = fi + feat_r, li + feat_r
            fr = _floor_rel(solid.get(key, set()))
            if fr is not None:
                floor[i, j] = max(-1.0, min(1.0, fr / 4.0))
                block[i, j] = 0.0
            if water.get(key):
                wmap[i, j] = 1.0
    return np.concatenate([floor.ravel(), block.ravel(), wmap.ravel()])


def load_samples(capture_dir: Path):
    """Per rollout, the list of (row, origin, path_fwd, path_dest) usable rows."""
    rollouts = []
    for rdir in sorted(capture_dir.glob("rollout-*")):
        sc = rdir / "sidecar.jsonl.gz"
        if not sc.exists():
            continue
        rows = []
        try:
            with gzip.open(sc, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    bs = d.get("baritone_state") or {}
                    fwd = bs.get("path_fwd")
                    dest = bs.get("path_dest")
                    if not fwd or len(fwd) < 2 or not dest or "block_grid" not in d:
                        continue
                    rows.append((d, d["origin"], fwd, dest))
        except (EOFError, OSError, gzip.BadGzipFile):
            # a sidecar still being written by an in-flight rollout — keep what
            # decoded cleanly, skip the truncated tail.
            pass
        if rows:
            rollouts.append((rdir.name, rows))
    return rollouts


# --- model -------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.sector = nn.Linear(hidden, N_SECTORS)
        self.dyh = nn.Linear(hidden, 3)

    def forward(self, x):
        h = self.body(x)
        return self.sector(h), self.dyh(h)


def _circ_sector_correct(pred, true):
    """Sector hit within ±1 bin (a 22.5° tolerance) — the action envelope doesn't
    resolve finer, so adjacent-sector counts as correct. Reported alongside exact."""
    diff = torch.abs(pred - true)
    circ = torch.minimum(diff, N_SECTORS - diff)
    return circ <= 1


def train_eval_r(rollouts, feat_r, target_r, *, terrain=True, epochs=60, lr=1e-3,
                 seed=0, device):
    """The horizon-curve cell: predict the subgoal at the FIXED action radius
    `target_r` from a terrain window of (swept) radius `feat_r`. Decoupling the two
    is the point — the target (the local decision) is held constant; only how much
    the model SEES varies, so accuracy(feat_r) is a clean 'how far must you look'
    curve, not confounded by the decision itself moving with r."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # held-out terrain: last third of rollouts as test
    n = len(rollouts)
    n_test = max(1, n // 3)
    test_names = {rollouts[i][0] for i in range(n - n_test, n)}

    straight_class = N_SECTORS // 2     # dev_class 8 == zero deviation (head at goal)

    def build(split_rows):
        X, Ydev, Ydy = [], [], []
        for d, origin, fwd, dest in split_rows:
            goal_ang = _goal_angle(dest, origin)
            if goal_ang is None:
                continue
            tgt = _exit_dev(fwd, origin, target_r, goal_ang)
            if tgt is None:
                continue
            devc, dyc = tgt
            gvec = _global_vec(dest, origin, target_r)
            if terrain:
                solid, water = _column_maps(d)
                tvec = _aligned_terrain(solid, water, feat_r, goal_ang)
                feat = np.concatenate([tvec, np.asarray(gvec, np.float32)])
            else:
                feat = np.asarray(gvec, np.float32)
            X.append(feat)
            Ydev.append(devc)
            Ydy.append(dyc)
        if not X:
            return None
        # straight-line baseline = always predict zero deviation (class 8).
        Bsec = torch.full((len(X),), straight_class, dtype=torch.long)
        return (torch.tensor(np.stack(X), dtype=torch.float32),
                torch.tensor(Ydev), torch.tensor(Ydy), Bsec)

    train_rows = [row for name, rows in rollouts if name not in test_names for row in rows]
    test_rows = [row for name, rows in rollouts if name in test_names for row in rows]
    tr = build(train_rows)
    te = build(test_rows)
    if tr is None or te is None:
        return None
    Xtr, Ytr_s, Ytr_d, _ = tr
    Xte, Yte_s, Yte_d, Bte_s = te
    Xtr, Ytr_s, Ytr_d = Xtr.to(device), Ytr_s.to(device), Ytr_d.to(device)
    Xte, Yte_s, Yte_d = Xte.to(device), Yte_s.to(device), Yte_d.to(device)

    # DETOUR subset: test ticks where the true exit sector diverges from the
    # straight-line bearing sector by >1 bin — i.e. where local planning departs
    # from "head at the goal". On this subset straight-line is 0% by construction,
    # so it isolates exactly the signal terrain must supply (the local-planning
    # horizon hides in the aggregate, which is dominated by benign straight ticks).
    Bte_dev = Bte_s.to(device)
    detour_mask = (_circ_sector_correct(Bte_dev, Yte_s) == False)  # noqa: E712

    model = Head(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bs = 4096
    # Report the AVERAGE of the last `tail` epochs rather than the test-argmax-best:
    # the detour subset is small (~25% of test) so per-epoch detour accuracy is
    # noisy, and selecting the best epoch by test accuracy is mild test leakage.
    # Averaging the converged tail is honest and smooths the noise.
    tail = max(1, epochs // 6)
    acc = {k: 0.0 for k in ("sector_exact", "sector_within1", "dy_acc",
                            "both_exact", "sector_ce_bits", "detour_within1")}
    seen = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            ps, pd = model(Xtr[idx])
            loss = ce(ps, Ytr_s[idx]) + ce(pd, Ytr_d[idx])
            loss.backward()
            opt.step()
        if ep < epochs - tail:
            continue
        model.eval()
        with torch.no_grad():
            ps, pd = model(Xte)
            sec_pred = ps.argmax(1)
            dy_pred = pd.argmax(1)
            hit1 = _circ_sector_correct(sec_pred, Yte_s)
            acc["sector_exact"] += (sec_pred == Yte_s).float().mean().item()
            acc["sector_within1"] += hit1.float().mean().item()
            acc["dy_acc"] += (dy_pred == Yte_d).float().mean().item()
            acc["both_exact"] += ((sec_pred == Yte_s) & (dy_pred == Yte_d)).float().mean().item()
            acc["sector_ce_bits"] += ce(ps, Yte_s).item() / math.log(2)
            acc["detour_within1"] += (hit1[detour_mask].float().mean().item()
                                      if detour_mask.any() else float("nan"))
        seen += 1
    best = {k: v / seen for k, v in acc.items()}
    # straight-line baseline on test (deterministic, no training)
    straight_exact = (Bte_s == Yte_s.cpu()).float().mean().item()
    straight_within1 = _circ_sector_correct(Bte_s, Yte_s.cpu()).float().mean().item()
    best["straight_exact"] = straight_exact
    best["straight_within1"] = straight_within1
    best["detour_frac"] = detour_mask.float().mean().item()
    best["n_train"] = int(Xtr.shape[0])
    best["n_test"] = int(Xte.shape[0])
    best["n_detour"] = int(detour_mask.sum().item())
    best["feat_dim"] = int(Xtr.shape[1])
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="§21.0 navigation horizon curve")
    ap.add_argument("--capture", default="results/sprint21/capture")
    ap.add_argument("--r-min", type=int, default=1, help="min FEATURE-window radius")
    ap.add_argument("--r-max", type=int, default=10, help="max FEATURE-window radius")
    ap.add_argument("--target-r", type=int, default=5,
                    help="FIXED action radius the subgoal is defined at (strike/use envelope)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sprint21/horizon.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = Path(args.capture)
    rollouts = load_samples(cap)
    total = sum(len(r) for _, r in rollouts)
    print(f"[nav_horizon] {len(rollouts)} rollouts, {total} usable rows, device={device}")
    if len(rollouts) < 3:
        print("[nav_horizon] need >=3 rollouts for a held-out terrain split")
        return 2

    print(f"[nav_horizon] target_r={args.target_r} (fixed action radius); "
          f"sweeping feature window r={args.r_min}..{args.r_max}")
    curve = []
    # bearing-only is independent of the feature window → compute once.
    bear = train_eval_r(rollouts, args.target_r, args.target_r, terrain=False,
                        epochs=args.epochs, seed=args.seed, device=device)
    for r in range(args.r_min, args.r_max + 1):
        full = train_eval_r(rollouts, r, args.target_r, terrain=True,
                            epochs=args.epochs, seed=args.seed, device=device)
        if full is None:
            print(f"  feat_r={r}: no usable samples")
            continue
        row = {"r": r, "full": full, "bearing_only": bear}
        curve.append(row)
        bdet = bear["detour_within1"] if bear else float("nan")
        print(f"  feat_r={r:2d}  sec_±1={full['sector_within1']:.3f} dy={full['dy_acc']:.3f} "
              f"ce={full['sector_ce_bits']:.2f}b | straight_±1={full['straight_within1']:.3f} "
              f"bearing_±1={bear['sector_within1']:.3f} | "
              f"DETOUR(frac={full['detour_frac']:.2f},n={full['n_detour']}): "
              f"full={full['detour_within1']:.3f} bearing={bdet:.3f} "
              f"(straight=0) | n_te={full['n_test']}", flush=True)

    out = {"capture": str(cap), "n_rollouts": len(rollouts), "n_rows": total,
           "target_r": args.target_r, "device": device,
           "bearing_only": bear, "curve": curve}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # headline reading: where does terrain stop adding (saturation) + residual
    if curve:
        print("\n=== HORIZON READING ===")
        print("  (detour subset = ticks where the true subgoal diverges from "
              "straight-line bearing; straight-line scores 0 there by construction)")
        for row in curve:
            r = row["r"]
            full = row["full"]
            bo = row["bearing_only"] or {"detour_within1": float("nan")}
            # On the detour subset, terrain's contribution = full − bearing-only:
            # the part of a genuine local detour recoverable from terrain alone.
            terr_det = full["detour_within1"] - bo["detour_within1"]
            resid_det = 1.0 - full["detour_within1"]
            print(f"  feat_r={r:2d}  detour_frac={full['detour_frac']:.2f}  "
                  f"full_detour±1={full['detour_within1']:.3f}  "
                  f"terrain_gain_on_detour={terr_det:+.3f}  "
                  f"detour_residual={resid_det:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
