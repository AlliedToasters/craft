"""Rung A — the block-target pointer head (neural_interface.md §6/§8c/§11/§12.1).

The other half of the §6 pointer gap. `rung_a_target.py` closed the *entity*
pointer (which mob does KillAura strike? 0.985). This asks the block version: which
block does the executor target with `START_DESTROY_BLOCK` (mine) / `use_item_on`
(place against)? A pointer INTO the block_grid, not a fixed-width field.

Setup: for each destroy/place packet, join the sidecar by tick to get the
player-centered `block_grid` (list of [palette_idx, dx, dy, dz] relative to
`origin` = floor of the player block). Candidates = the *targetable* cells:
non-air grid cells with >=1 air neighbour (an exposed face) within `--reach` of
the eye. The label is the cell whose rel-coord == (block_pos - origin), 100%
recoverable in both captures. A shared per-candidate MLP scores each candidate;
segment-softmax over the event's candidates; cross-entropy to the true index.

The contrast with the entity head is the point. For entities, gaze does NOT track
the target (KillAura auto-aims server-side), so `nearest` was a real 0.48 baseline.
For blocks you must *look at* a block to mine it, so the strong baseline is the
**crosshair raycast** (DDA from the eye along yaw/pitch → first occupied voxel). If
the learned head merely re-derives the crosshair, the block "decision" lives in the
servo/aim channel (gaze), not in a separate discrete head — a sharper read on the
moat than a high accuracy alone.

Baselines (analytic, no training):
  nearest   pick the closest targetable cell        (weak: ~0.05-0.21 — target is
                                                      NOT the nearest exposed block)
  crosshair voxel-raycast from eye along look dir    (the gaze ceiling)

Arms (learned scorer):
  geom        per-candidate rel pos + off-axis angle (is it gaze geometry?)
  geom+type   + block-type one-hot                   (does block identity matter?)

Usage:
  .venv/bin/python -m experiments.next_packet.rung_a_block \
      --rollouts-glob "results/frozen_dryrun/rollout-*" --epochs 200
  # combine both regimes (more events):
  .venv/bin/python -m experiments.next_packet.rung_a_block \
      --rollouts-glob "results/frozen_*/rollout-*" --epochs 200
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

from .ablation_r0_r1 import _open_text
from .ablation_r1_r3 import EntityVocab  # reused as a generic str->index vocab

EYE_HEIGHT = 1.62  # MC player eye above feet (block origin = floor(feet))
_NEI = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# block_pos-bearing packets we treat as a "target a block" decision.
_DESTROY = ("minecraft:player_action", "START_DESTROY_BLOCK")
_PLACE = "minecraft:use_item_on"


def _look_dir(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    ry, rp = math.radians(yaw_deg), math.radians(pitch_deg)
    cp = math.cos(rp)
    return (-math.sin(ry) * cp, -math.sin(rp), math.cos(ry) * cp)


def _bearing(vx: float, vy: float, vz: float) -> tuple[float, float]:
    """MC (yaw, pitch) in degrees that points along (vx,vy,vz)."""
    horiz = math.sqrt(vx * vx + vz * vz)
    yaw = math.degrees(math.atan2(-vx, vz))
    pitch = math.degrees(math.atan2(-vy, horiz)) if horiz > 1e-6 else 0.0
    return yaw, pitch


def load_blocks(rollout_dirs: list[str], reach: float) -> list[dict]:
    """One row per destroy/place event: candidate targetable cells + label index."""
    rows: list[dict] = []
    for d in rollout_dirs:
        dp = Path(d)
        packets = dp / "packets.jsonl"
        sidecar = next(iter(glob.glob(str(dp / "sidecar.jsonl*"))), None)
        if not packets.exists() or not sidecar:
            continue
        side: dict[int, dict] = {}
        with _open_text(sidecar) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("tick")
                if isinstance(t, int):
                    side[t] = r
        with _open_text(str(packets)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                pid = rec.get("id")
                fld = rec.get("fields") or {}
                if pid == _DESTROY[0]:
                    if fld.get("action") != _DESTROY[1]:
                        continue
                    kind = "destroy"
                elif pid == _PLACE:
                    kind = "place"
                else:
                    continue
                bp = fld.get("block_pos")
                if not bp:
                    continue
                obs = rec.get("obs") or {}
                tick = obs.get("tick")
                s = side.get(tick) if isinstance(tick, int) else None
                if s is None:
                    continue
                ox, oy, oz = s["origin"]
                rel = (bp[0] - ox, bp[1] - oy, bp[2] - oz)
                pal = s["block_palette"]
                grid = s["block_grid"]
                occ = {(c[1], c[2], c[3]) for c in grid}
                # eye position relative to origin (origin = floor of player feet)
                ex = float(obs.get("x", ox)) - ox
                ey = float(obs.get("y", oy)) - oy + EYE_HEIGHT
                ez = float(obs.get("z", oz)) - oz
                cur_yaw = float(obs.get("yaw", 0.0))
                cur_pitch = float(obs.get("pitch", 0.0))

                cands = []
                for c in grid:
                    pi, cx, cy, cz = c[0], c[1], c[2], c[3]
                    if not any((cx + nx, cy + ny, cz + nz) not in occ for nx, ny, nz in _NEI):
                        continue  # buried: not a face we can target
                    # vector eye -> cell centre
                    vx, vy, vz = (cx + 0.5) - ex, (cy + 0.5) - ey, (cz + 0.5) - ez
                    dist = math.sqrt(vx * vx + vy * vy + vz * vz)
                    if dist > reach:
                        continue
                    byaw, bpitch = _bearing(vx, vy, vz)
                    off_yaw = math.radians((byaw - cur_yaw + 180.0) % 360.0 - 180.0)
                    off_pitch = math.radians((bpitch - cur_pitch + 180.0) % 360.0 - 180.0)
                    cands.append({
                        "cell": (cx, cy, cz),
                        "block": pal[pi] if 0 <= pi < len(pal) else "",
                        "dx": vx, "dy": vy, "dz": vz, "dist": dist,
                        "off_yaw": off_yaw, "off_pitch": off_pitch,
                    })
                if len(cands) < 2:
                    continue
                cands.sort(key=lambda c: c["dist"])
                cells = [c["cell"] for c in cands]
                if rel not in cells:
                    continue  # target outside reach/exposed set — dropped, reported
                rows.append({
                    "cands": cands, "label": cells.index(rel), "kind": kind,
                    "eye": (ex, ey, ez), "look": _look_dir(cur_yaw, cur_pitch),
                    "occ": occ, "reach": reach,
                })
    return rows


def cand_features(c: dict, bvocab: EntityVocab, use_type: bool) -> list[float]:
    f = [c["dx"], c["dy"], c["dz"], c["dist"],
         math.sin(c["off_yaw"]), math.cos(c["off_yaw"]),
         math.sin(c["off_pitch"]), math.cos(c["off_pitch"])]
    if use_type:
        oh = [0.0] * bvocab.size
        ti = bvocab.index.get(c["block"])
        if ti is not None:
            oh[ti] = 1.0
        f += oh
    return f


def baseline_nearest(rows) -> float:
    # candidates dist-sorted, so nearest targetable cell is index 0.
    return sum(1 for r in rows if r["label"] == 0) / len(rows)


def baseline_crosshair(rows) -> float:
    """Voxel-march from the eye along the look dir; first occupied cell is the hit."""
    hit = 0
    for r in rows:
        ex, ey, ez = r["eye"]
        dx, dy, dz = r["look"]
        occ = r["occ"]
        eye_voxel = (math.floor(ex), math.floor(ey), math.floor(ez))
        found = None
        step, t = 0.02, 0.0
        while t <= r["reach"]:
            vx = (math.floor(ex + dx * t), math.floor(ey + dy * t), math.floor(ez + dz * t))
            if vx != eye_voxel and vx in occ:
                found = vx
                break
            t += step
        if found is not None and found == r["cands"][r["label"]]["cell"]:
            hit += 1
    return hit / len(rows)


def train_arm(train, val, bvocab, use_type, *, hidden, epochs, lr, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    dim = len(cand_features(train[0]["cands"][0], bvocab, use_type))
    geom = [[c[k] for c in (cc for r in train for cc in r["cands"])]
            for k in ("dx", "dy", "dz", "dist")]
    mean = [sum(col) / len(col) for col in geom]
    std = [max(math.sqrt(sum(x * x for x in col) / len(col) - (sum(col) / len(col)) ** 2), 1e-6)
           for col in geom]

    def feat(c):
        f = cand_features(c, bvocab, use_type)
        for i in range(4):
            f[i] = (f[i] - mean[i]) / std[i]
        return f

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    tr = [(torch.tensor([feat(c) for c in r["cands"]], dtype=torch.float32, device=device),
           torch.tensor(r["label"], device=device)) for r in train]

    rng = random.Random(seed)
    best = 0.0
    for _e in range(epochs):
        model.train()
        rng.shuffle(tr)
        for X, y in tr:
            opt.zero_grad()
            scores = model(X).squeeze(-1).unsqueeze(0)   # [1, n_cand]
            ce(scores, y.unsqueeze(0)).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for r in val:
                X = torch.tensor([feat(c) for c in r["cands"]], dtype=torch.float32, device=device)
                if int(model(X).squeeze(-1).argmax().item()) == r["label"]:
                    correct += 1
        best = max(best, correct / len(val))
    return best, dim


def main() -> None:
    ap = argparse.ArgumentParser(description="Rung A block-target pointer head (§6/§12.1)")
    ap.add_argument("--rollouts-glob", default="results/frozen_dryrun/rollout-*")
    ap.add_argument("--reach", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(glob.glob(args.rollouts_glob))
    rows = load_blocks(dirs, args.reach)
    if not rows:
        print(f"No destroy/place events from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)
    blocks = sorted({c["block"] for r in rows for c in r["cands"] if c["block"]})
    bvocab = EntityVocab(blocks)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    avg_cand = sum(len(r["cands"]) for r in rows) / len(rows)
    n_destroy = sum(1 for r in rows if r["kind"] == "destroy")

    print(f"block events={len(rows)} (destroy={n_destroy} place={len(rows) - n_destroy})  "
          f"train={len(train)}  val={n_val}  avg_candidates={avg_cand:.1f}  "
          f"block_types={bvocab.size}  reach={args.reach}")
    print("\n  baseline (no training)            val_acc")
    print(f"    {'nearest':<30}{baseline_nearest(val):>8.3f}")
    print(f"    {'crosshair (raycast)':<30}{baseline_crosshair(val):>8.3f}")

    print("\n  learned pointer head              val_acc   dim")
    for label, use_type in [("geom", False), ("geom+type", True)]:
        acc, dim = train_arm(train, val, bvocab, use_type,
                             hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed)
        print(f"    {label:<30}{acc:>8.3f}{dim:>6}")

    print("\n  read: crosshair is the gaze ceiling. If the head only matches it, the")
    print("        block 'decision' is gaze (servo channel), not a separate pointer.")


if __name__ == "__main__":
    main()
