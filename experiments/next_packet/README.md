# next_packet — next-packet prediction baseline

Predicts `encode(packet, obs)` from `obs`. The first learning experiment
in the neural interface ladder (neural_interface.md §8). Uses packets
captured by `PacketRecorder` (homunculus) as training data.

## What this is

The experiment is structured as a §7-resolution device, not just a
baseline. It answers:

- **What is the R0 ceiling?** How predictable is the heuristic stack
  from the minimal 9-feature obs? The per-type accuracy table is the
  primary artifact; the aggregate number is secondary (move-dominated,
  uninformative on its own).
- **Does `g_t` matter for the discriminator?** The R0→R1 step on
  `interact` and `player_command` is pre-registered as a step-change
  (§8c). A flat result would be a finding.
- **Does the pointer gap matter for parameter heads?** The R2→R3 step
  on `use_item_on` and `interact.entity_id` is pre-registered as the
  pointer-gap closure test.

## Files

| File | What it does |
|---|---|
| `dataset.py` | JSONL reader → `(obs_dict, Action)` pairs. Pure Python, no ML deps. |
| `features.py` | `obs_dict` → flat float vector + `packet_type_label`. Pure Python. |
| `metrics.py` | Per-type accuracy tracker + pre-registered `(type × rung)` comparison table. |
| `train.py` | MLP trunk + 11 discriminator/parameter heads. Requires `torch`. |

## Quick start

**Dry run (no torch needed)** — exercises the full data pipeline and
prints the dataset distribution:

```bash
source .venv/bin/activate
python -m experiments.next_packet.train --dry-run
```

**Train (R0, discriminator only)**:

```bash
pip install torch   # once
python -m experiments.next_packet.train \
    --recordings "~/.homunculus/recordings/*.jsonl" \
    --rung R0 \
    --epochs 30 \
    --hidden 256 \
    --lr 1e-3
```

Per-type accuracy is printed after every epoch. The pre-registered
`(type × rung)` table (§8c) is populated in `metrics.preregistered_report`
once you have results from multiple rungs.

## Recording more data

The dry run will tell you how many examples each type has. To collect
more, arm the PacketRecorder while running a rollout:

```bash
# Arm (homunculus must be running):
curl -s -X POST http://127.0.0.1:25566/packets/recording/arm \
  -H "Content-Type: application/json" -d '{}'

# Run a rollout (any model/goal):
python -m craft.agent 50 minimal --model "$QWEN" ...

# Disarm:
curl -s -X POST http://127.0.0.1:25566/packets/recording/disarm
```

The recording lands at `~/.homunculus/recordings/recording-<ts>-<port>.jsonl`.
Pass it via `--recordings` to `train.py`.

## Rung progression

| Rung | Obs features added | Infra needed |
|---|---|---|
| R0 | §2a minimal (9 floats) | ready today |
| R1 | + `g_t`, `ticks_since_g_t_issued`, `delta_tick` | plumb goal context into recorder |
| R2 | + health, hunger, saturation, air | serialize stats into recorder obs |
| R3 | + `local_block_grid`, `entity_set` | new homunculus channels (§2b) |
| R4 | + vision frame | Xvfb frame capture integration |

R1 requires recording from LLM-driven rollouts only (`g_t` is null in
heuristic-only data). See neural_interface.md §8a.

## Pre-registered predictions (§8c)

| Type | Rung | Kind | Prediction |
|---|---|---|---|
| `interact` | R0→R1 | step | `g_t` disambiguates ATTACK vs INTERACT vs INTERACT_AT |
| `player_command` | R0→R1 | step | sprint/sneak edges are goal-driven |
| `use_item_on` | R2→R3 | step | block_pos pointer gap closes |
| `interact` | R2→R3 | step | entity_id pointer gap closes |
| `move_*` | R0→R1 | flat | movement is Baritone path-following, not LLM intent |
| `swing` | R0→R3 | flat | hand is near-deterministic from inventory state |

A flat result where a step was predicted is a finding, not a failure.
