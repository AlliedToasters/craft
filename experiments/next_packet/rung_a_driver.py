"""Rung A — the neural Baritone+Wurst *executor* (neural_interface.md §8d / §11).

Bottom rung of the control-hierarchy replacement ladder. The aggregate craft
controller is meta-driver → LLM driver → Baritone planner → Baritone/Wurst
*executor* → packets. Rung A asks the floor question:

    Given the command from above (the active tool) and the executor's own
    state (is-pathing, mob-proximity, timing), is the emitted packet
    determined — *without* any higher-layer (goal/LLM) signal?

i.e. predict movement packets given a path; swing given a mining goal; swing
given mob-proximity (KillAura always-on). If the executor's behavior factorizes
as packet ≈ f(command, executor_state), a small net *is* a drop-in Baritone+Wurst
driver, and we can climb to rung B (predict baritone_state from tool-calls).

Why these arms (not R0→R1's goal-vs-temporal): in this Qwen capture g_t collapses
to the tool name (§8a), so `g_t` and `current_tool` are the *same column* — there
is no clean layer-C signal to ablate. The genuinely *lower-than-the-command*
signals are the executor's: `baritone.pathing`, entity proximity, and exec
timing. So the cut is command (cmd) vs executor-state (exec+prox):

  R0       pose only                                  (no command, no exec)
  cmd      pose + current_tool one-hot                (the command from above)
  exec     pose + pathing/goal_active/waiting + dt + proximity   (executor only)
  driverA  pose + cmd + exec + proximity              (full neural executor)

The claim is rung-A-confirmed if (a) driverA ≈ ceiling on the executor-emitted
types (swing, move_*), and (b) exec carries large *independent* lift over cmd —
meaning within a fixed command the executor state, not anything higher, picks the
packet.

CAVEAT (honest): baritone_state here is thin — only `pathing` (bool) is reliable;
`goal`/`goal_active` are ~null because Baritone mining runs MineProcess, not the
CustomGoalProcess our snapshot reads. So this tests "given the executor is
pathing", not "given the path target". Enriching BaritoneState.snapshot() to
expose the MineProcess/GoalProcess target is the rung-A capture upgrade (§11).

Usage:
  .venv/bin/python -m experiments.next_packet.rung_a_driver \
      --rollouts-glob "results/frozen_dryrun/rollout-*" --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

from .ablation_r0_r1 import _R0, Normalizer, _open_text
from .ablation_r1_r3 import ENTITY_RADIUS, EntityVocab, entity_features
from .features import PACKET_TYPE_INDEX, PACKET_TYPES
from .metrics import TypeMetrics

_DIM_ORDER = {"minecraft:overworld": 0.0, "minecraft:the_nether": 1.0, "minecraft:the_end": 2.0}


def load_examples(rollout_dirs: list[str]) -> list[tuple[dict, list | None, str]]:
    """(obs, entity_set, packet_type). The sidecar is joined by tick for BOTH
    baritone_state (→ obs['pathing'], obs['goal_active']) and entity_set.
    delta_tick computed per-file."""
    out: list[tuple[dict, list | None, str]] = []
    for d in rollout_dirs:
        dp = Path(d)
        packets = dp / "packets.jsonl"
        sidecar = next(iter(glob.glob(str(dp / "sidecar.jsonl*"))), None)
        if not packets.exists():
            continue
        ent_by_tick: dict[int, list] = {}
        bs_by_tick: dict[int, dict] = {}
        if sidecar:
            with _open_text(sidecar) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    t = row.get("tick")
                    if isinstance(t, int):
                        ent_by_tick[t] = row.get("entity_set")
                        bs_by_tick[t] = row.get("baritone_state") or {}
        prev_tick: int | None = None
        with _open_text(str(packets)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                ptype = rec.get("id")
                if ptype not in PACKET_TYPE_INDEX:
                    continue
                obs = dict(rec.get("obs") or {})
                tick = obs.get("tick")
                obs["delta_tick"] = max(0, tick - prev_tick) if (isinstance(tick, int) and prev_tick is not None) else 0
                if isinstance(tick, int):
                    prev_tick = tick
                bs = bs_by_tick.get(tick) if isinstance(tick, int) else None
                obs["pathing"] = bool(bs.get("pathing")) if bs else False
                obs["goal_active"] = bool(bs.get("goal_active")) if bs else False
                ent = ent_by_tick.get(tick) if isinstance(tick, int) else None
                out.append((obs, ent, ptype))
    return out


class CommandVocab:
    """Active tool (`current_tool`) string → one-hot. Index 0 = none/idle."""

    def __init__(self, tools: list[str]) -> None:
        self.tools = list(tools)
        self.index = {t: i + 1 for i, t in enumerate(self.tools)}

    @classmethod
    def fit(cls, examples) -> "CommandVocab":
        s = {o.get("current_tool") for o, _e, _p in examples if o.get("current_tool")}
        return cls(sorted(s))

    @property
    def size(self) -> int:
        return len(self.tools) + 1

    def names(self) -> list[str]:
        return ["cmd=<idle>"] + [f"cmd={t}" for t in self.tools]

    def onehot(self, tool) -> list[float]:
        v = [0.0] * self.size
        v[self.index.get(tool, 0) if tool else 0] = 1.0
        return v


# (name, normalize?) — bools / one-hots pass through; continuous get z-scored.
# Executor state is split into three probeable sub-channels so an "exec" lift can
# be attributed to pathing-flags vs emission-timing vs mob-proximity (otherwise
# the result collapses into "delta_tick again", §8c-bis).
_PATHING = [("pathing", False), ("goal_active", False), ("waiting_on_llm", False)]
_TIMING = [("delta_tick", True), ("ticks_since_g_t_issued", True)]


def layout(cvocab: CommandVocab, evocab: EntityVocab, groups: set[str]) -> list[tuple[str, bool]]:
    cols = list(_R0)
    if "cmd" in groups:
        cols += [(n, False) for n in cvocab.names()]
    if "pathing" in groups:
        cols += _PATHING
    if "timing" in groups:
        cols += _TIMING
    if "prox" in groups:
        cols += [(n, False) for n in evocab.names()]
        cols += [("nearest_entity_dist", True), ("n_entities_within", True)]
    return cols


def featurize(obs: dict, ent: list | None, cvocab: CommandVocab, evocab: EntityVocab,
              groups: set[str]) -> list[float]:
    yaw = math.radians(float(obs.get("yaw", 0.0)))
    pitch = math.radians(float(obs.get("pitch", 0.0)))
    feats = [
        float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("z", 0.0)),
        math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
        1.0 if obs.get("on_ground") else 0.0,
        _DIM_ORDER.get(str(obs.get("dim", "")), -1.0),
    ]
    if "cmd" in groups:
        feats += cvocab.onehot(obs.get("current_tool"))
    if "pathing" in groups:
        feats += [
            1.0 if obs.get("pathing") else 0.0,
            1.0 if obs.get("goal_active") else 0.0,
            1.0 if obs.get("waiting_on_llm") else 0.0,
        ]
    if "timing" in groups:
        feats += [float(obs.get("delta_tick") or 0), float(obs.get("ticks_since_g_t_issued") or 0)]
    if "prox" in groups:
        feats += entity_features(ent, evocab)
    return feats


ARMS: dict[str, set[str]] = {
    "R0":       set(),                                    # pose only
    "cmd":      {"cmd"},                                  # command from above (tool)
    "timing":   {"timing"},                               # executor emission cadence
    "pathing":  {"pathing"},                              # baritone pathing flags
    "prox":     {"prox"},                                 # mob proximity (Wurst trigger)
    "exec":     {"pathing", "timing", "prox"},            # all executor state, no command
    "driverA":  {"cmd", "pathing", "timing", "prox"},     # full neural Baritone+Wurst driver
}


def train_arm(examples, cvocab, evocab, label, groups, *, val_idx, hidden, epochs, lr, batch_size, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    cols = layout(cvocab, evocab, groups)
    raw = [(featurize(o, ent, cvocab, evocab, groups), PACKET_TYPE_INDEX[p], p) for o, ent, p in examples]
    train_raw = [raw[i] for i in range(len(raw)) if i not in val_idx]
    val_raw = [raw[i] for i in range(len(raw)) if i in val_idx]

    norm = Normalizer(cols).fit([f for f, _, _ in train_raw])
    train = [(norm.transform(f), y, p) for f, y, p in train_raw]
    val = [(norm.transform(f), y, p) for f, y, p in val_raw]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(len(cols), hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, len(PACKET_TYPES)),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    def tens(batch):
        xs = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        return xs, ys

    rng = random.Random(seed)
    best_acc, best_m = 0.0, TypeMetrics()
    for _e in range(epochs):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), batch_size):
            xs, ys = tens(train[i:i + batch_size])
            opt.zero_grad()
            crit(model(xs), ys).backward()
            opt.step()
        model.eval()
        m = TypeMetrics()
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch = val[i:i + batch_size]
                xs, _ = tens(batch)
                preds = model(xs).argmax(dim=-1).tolist()
                for (_, _, pt), pi in zip(batch, preds):
                    m.update(pt, PACKET_TYPES[pi])
        acc = m.overall_accuracy()
        if acc >= best_acc:
            best_acc, best_m = acc, m
    return best_acc, best_m, len(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rung A — neural Baritone+Wurst executor discriminator (§11)")
    ap.add_argument("--rollouts-glob", default="results/frozen_dryrun/rollout-*")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(glob.glob(args.rollouts_glob))
    examples = load_examples(dirs)
    if not examples:
        print(f"No examples from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)

    cvocab = CommandVocab.fit(examples)
    evocab = EntityVocab.fit(examples)
    n_pathing = sum(1 for o, _e, _p in examples if o.get("pathing"))
    n_with_ent = sum(1 for _o, e, _p in examples if e)

    rng = random.Random(args.seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_idx = set(idx[:n_val])

    print(f"examples={len(examples)}  val={n_val}  pathing={n_pathing}  entity-joined={n_with_ent}")
    print(f"command_vocab: {cvocab.names()}")
    print(f"entity_vocab ({evocab.size}): {evocab.types}  (radius={ENTITY_RADIUS})")

    results = {}
    for label in ARMS:
        acc, m, dim = train_arm(
            examples, cvocab, evocab, label, ARMS[label], val_idx=val_idx,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
        )
        results[label] = (acc, m, dim)
        print(f"  {label:<9} dim={dim:<4} overall_val_acc={acc:.4f}")

    print("\n" + "=" * 76)
    print("Rung A — does packet ≈ f(command, executor_state)?  (val top-1 accuracy)")
    print("=" * 76)
    r0 = results["R0"][0]
    print(f"  {'arm':<10}{'dim':>5}{'overall':>10}{'Δ vs R0':>10}")
    for label in ARMS:
        acc, _m, dim = results[label]
        print(f"  {label:<10}{dim:>5}{acc:>10.4f}{acc - r0:>+10.4f}")

    print(f"\n  per-type accuracy by arm (executor-emitted types in focus):")
    ref_m = results["driverA"][1]
    print("  " + f"{'type':<34}{'n':>5}" + "".join(f"{a:>10}" for a in ARMS))
    for t in PACKET_TYPES:
        n = ref_m._total[t]
        if n == 0:
            continue
        cells = "".join(
            f"{(results[a][1].accuracy(t) if results[a][1].accuracy(t) is not None else float('nan')):>10.3f}"
            for a in ARMS
        )
        print(f"  {t:<34}{n:>5}{cells}")


if __name__ == "__main__":
    main()
