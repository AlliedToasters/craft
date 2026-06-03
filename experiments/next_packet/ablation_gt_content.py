"""Sprint B: g_t-CONTENT ablation for the next-packet discriminator.

The README's "g_t falsified (§8c-bis)" result used a one-hot GoalVocab on the
tool-name-only sets (combat/dryrun: 7 and 3 distinct g_t). It has NEVER been
tested with a content encoding on the narrated set (125 distinct full-sentence
intents). This runner closes that: it adds a FROZEN sentence-embedding arm
(gt_embed.GtEmbedder) and measures the brief's pre-registered (type x rung)
cells with per-type NLL.

Arms (rungs):
  R0               pose only (9 dims)
  R1_temporal      + ticks_since_g_t_issued, delta_tick
  R1_goal_onehot   + g_t one-hot  (the leakage-prone categorical, kept as the
                   contrast the brief's anti-pattern #2 warns about)
  R1_goal_content  + frozen g_t sentence embedding (768-dim, L2-normalized)
  R1_full_content  + temporal + frozen content

LEAKAGE GUARD (load-bearing): the default split holds out WHOLE ROLLOUTS. g_t
is constant across long segments, so a random split puts near-duplicate
(obs, g_t) pairs in train and val -- even a frozen embedding then acts as a
segment-id (anti-pattern #2). ``--split random`` is provided ONLY as the
contrast: a +g_t gain that appears under random but vanishes by-rollout is
leakage, not intent signal. Report both.

Metrics: per-type accuracy AND per-type NLL (mean -log p(true type)). The
aggregate is move-dominated and banned as a headline (brief).

Usage:
  .venv/bin/python -m experiments.next_packet.ablation_gt_content \
      --rollouts-glob "results/frozen_narrated/rollout-*" --epochs 30
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

from .features import PACKET_TYPES, PACKET_TYPE_INDEX
from .gt_embed import GtEmbedder, content_hash

_DIM = {"minecraft:overworld": 0.0, "minecraft:the_nether": 1.0, "minecraft:the_end": 2.0}


def load_tagged(rollout_dirs: list[str]) -> list[tuple[dict, str, int]]:
    """(obs, packet_type, rollout_idx). delta_tick computed per-rollout."""
    out: list[tuple[dict, str, int]] = []
    for ri, d in enumerate(rollout_dirs):
        pf = Path(d) / "packets.jsonl"
        if not pf.exists():
            continue
        prev: int | None = None
        with open(pf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                pt = rec.get("id")
                if pt not in PACKET_TYPE_INDEX:
                    continue
                obs = dict(rec.get("obs") or {})
                tick = obs.get("tick")
                obs["delta_tick"] = (max(0, tick - prev)
                                     if isinstance(tick, int) and prev is not None else 0)
                if isinstance(tick, int):
                    prev = tick
                out.append((obs, pt, ri))
    return out


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# group flags per arm
ARMS: dict[str, set[str]] = {
    "R0": set(),
    "R1_temporal": {"temporal"},
    "R1_goal_onehot": {"onehot"},
    "R1_goal_content": {"content"},
    "R1_full_content": {"temporal", "content"},
}


def featurize(obs: dict, groups: set[str], *, goal_index: dict[str, int],
              n_goals: int, emb: GtEmbedder | None) -> list[float]:
    yaw = math.radians(float(obs.get("yaw", 0.0)))
    pitch = math.radians(float(obs.get("pitch", 0.0)))
    f = [float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("z", 0.0)),
         math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
         1.0 if obs.get("on_ground") else 0.0,
         _DIM.get(str(obs.get("dim", "")), -1.0)]
    if "temporal" in groups:
        f += [float(obs.get("ticks_since_g_t_issued") or 0), float(obs.get("delta_tick") or 0)]
    if "onehot" in groups:
        oh = [0.0] * (n_goals + 1)
        g = obs.get("g_t")
        oh[goal_index.get(g, 0) if g is not None else 0] = 1.0
        f += oh
    if "content" in groups:
        assert emb is not None
        f += _l2(emb.vector(obs.get("g_t")))
    return f


# normalize only the raw-pose continuous cols (first 3) + temporal; sin/cos,
# one-hot, and (already L2-normed) embedding pass through.
def _norm_flags(groups: set[str], n_goals: int, emb_dim: int) -> list[bool]:
    flags = [True, True, True, False, False, False, False, False, False]
    if "temporal" in groups:
        flags += [True, True]
    if "onehot" in groups:
        flags += [False] * (n_goals + 1)
    if "content" in groups:
        flags += [False] * emb_dim
    return flags


def train_arm(examples, groups, label, *, val_idx, goal_index, n_goals, emb,
              hidden, epochs, lr, batch_size, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    emb_dim = emb.dim if (emb is not None and "content" in groups) else 0
    flags = _norm_flags(groups, n_goals, emb_dim)
    feats = [(featurize(o, groups, goal_index=goal_index, n_goals=n_goals, emb=emb),
              PACKET_TYPE_INDEX[p], p) for o, p, _ in examples]
    tr = [feats[i] for i in range(len(feats)) if i not in val_idx]
    va = [feats[i] for i in range(len(feats)) if i in val_idx]

    # z-score flagged cols on train only
    dim = len(flags)
    mean = [0.0] * dim
    std = [1.0] * dim
    if tr:
        for i in range(dim):
            if not flags[i]:
                continue
            col = [r[0][i] for r in tr]
            m = sum(col) / len(col)
            var = sum(x * x for x in col) / len(col) - m * m
            mean[i] = m
            std[i] = max(math.sqrt(max(var, 0.0)), 1e-6)

    def xf(v):
        return [(v[i] - mean[i]) / std[i] if flags[i] else v[i] for i in range(dim)]

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, len(PACKET_TYPES))).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor([xf(r[0]) for r in tr], dtype=torch.float32, device=dev)
    Ytr = torch.tensor([r[1] for r in tr], dtype=torch.long, device=dev)
    Xva = torch.tensor([xf(r[0]) for r in va], dtype=torch.float32, device=dev)
    Yva = torch.tensor([r[1] for r in va], dtype=torch.long, device=dev)
    va_types = [r[2] for r in va]

    rng = random.Random(seed)
    order = list(range(len(tr)))
    best = {"acc": -1.0, "per_type": {}}
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            bx = Xtr[idx]
            by = Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logp = F.log_softmax(model(Xva), dim=-1)
            pred = logp.argmax(dim=-1)
            acc = (pred == Yva).float().mean().item()
            if acc >= best["acc"]:
                # per-type acc + NLL
                nll_true = -logp[range(len(va_types)), Yva]
                per: dict[str, dict] = {}
                for t in set(va_types):
                    sel = [j for j, tt in enumerate(va_types) if tt == t]
                    pt_acc = (pred[sel] == Yva[sel]).float().mean().item()
                    pt_nll = nll_true[sel].mean().item()
                    per[t] = {"n": len(sel), "acc": pt_acc, "nll": pt_nll}
                best = {"acc": acc, "per_type": per, "dim": dim}
    return best


def make_val_idx(examples, split: str, seed: int, val_frac: float, holdout_rollout: int):
    n = len(examples)
    if split == "random":
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        return set(idx[:max(1, int(n * val_frac))])
    # by-rollout: hold out one rollout entirely
    return {i for i, (_, _, ri) in enumerate(examples) if ri == holdout_rollout}


def main() -> None:
    ap = argparse.ArgumentParser(description="g_t-content ablation (Sprint B)")
    ap.add_argument("--rollouts-glob", default="results/frozen_narrated/rollout-*")
    ap.add_argument("--split", choices=["by_rollout", "random"], default="by_rollout")
    ap.add_argument("--holdout-rollout", type=int, default=0,
                    help="rollout index held out for val (by_rollout split)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(glob.glob(args.rollouts_glob))
    examples = load_tagged(dirs)
    if not examples:
        print(f"no examples from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)

    goals = sorted({o.get("g_t") for o, _, _ in examples if o.get("g_t") is not None})
    goal_index = {g: i + 1 for i, g in enumerate(goals)}
    emb = GtEmbedder()
    n_new = emb.warm(goals)

    val_idx = make_val_idx(examples, args.split, args.seed, args.val_frac, args.holdout_rollout)
    n_val = len(val_idx)
    print(f"rollouts={len(dirs)} examples={len(examples)} distinct_g_t={len(goals)} "
          f"emb_dim={emb.dim} emb_new={n_new} gt_hash={content_hash(goals)}")
    print(f"split={args.split} holdout_rollout={args.holdout_rollout} "
          f"val={n_val} train={len(examples) - n_val}")
    if n_val == 0 or n_val == len(examples):
        print("ERROR: degenerate split (val empty or all)", file=sys.stderr)
        sys.exit(1)

    results = {}
    for label, groups in ARMS.items():
        use_emb = emb if "content" in groups else None
        results[label] = train_arm(
            examples, groups, label, val_idx=val_idx, goal_index=goal_index,
            n_goals=len(goals), emb=use_emb, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size, seed=args.seed)
        print(f"  {label:<18} dim={results[label]['dim']:<5} val_acc={results[label]['acc']:.4f}")

    # per-type table: the brief's (type x rung) heatmap, ACC and NLL.
    arm_order = list(ARMS.keys())
    focus = ["minecraft:player_command", "minecraft:move_player_pos_rot",
             "minecraft:move_player_rot", "minecraft:swing", "minecraft:interact",
             "minecraft:use_item_on", "minecraft:player_action", "minecraft:player_input"]
    print("\n=== per-type NLL (lower=better) by arm — Sprint B heatmap ===")
    print(f"  {'type':<34}{'n':>6}" + "".join(f"{a.replace('R1_',''):>12}" for a in arm_order))
    for t in focus:
        ns = [results[a]['per_type'].get(t, {}).get('n', 0) for a in arm_order]
        n = max(ns)
        if n == 0:
            continue
        cells = "".join(f"{results[a]['per_type'].get(t, {}).get('nll', float('nan')):>12.3f}"
                        for a in arm_order)
        print(f"  {t:<34}{n:>6}{cells}")
    print("\n=== per-type ACC by arm ===")
    print(f"  {'type':<34}{'n':>6}" + "".join(f"{a.replace('R1_',''):>12}" for a in arm_order))
    for t in focus:
        ns = [results[a]['per_type'].get(t, {}).get('n', 0) for a in arm_order]
        n = max(ns)
        if n == 0:
            continue
        cells = "".join(f"{results[a]['per_type'].get(t, {}).get('acc', float('nan')):>12.3f}"
                        for a in arm_order)
        print(f"  {t:<34}{n:>6}{cells}")

    if args.json_out:
        json.dump({"args": vars(args), "gt_hash": content_hash(goals),
                   "n_examples": len(examples), "n_val": n_val, "results": results},
                  open(args.json_out, "w"), indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
