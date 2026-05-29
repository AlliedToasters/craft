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

> **Status (2026-05-28): rung-A reframe — predict the decision, not the packet.**
> The program is bottom-up replacement of the control hierarchy (§11). Three
> offline heads on the frozen set: type-discriminator (faked by `delta_tick`
> cadence), aim-regression (faked by positional inertia), and the **attack-target
> pointer head — 0.985**, closing the §6 entity pointer gap. The packet stream is
> ~99% autocorrelation; the control signal is in the sparse discrete events.
> See `neural_interface.md` §11 (results + where to pick up: block_pos pointer,
> recapture, closed-loop swap).
>
> Prior (superseded as the *objective*, still valid as measurement): discriminator
> ablation R0/R1/R3 — lever ranking for packet *type* **temporal ≫ entity > goal**,
> `g_t` falsified (§8c-bis).

## Files

| File | What it does |
|---|---|
| `capture.py` | **Frozen-capture runner** (§8e). N rollouts, arms packet + sidecar streams, verifies tick-join, writes manifest (commits + sha256 + content hash). |
| `ablation_r0_r1.py` | R0→R1 discriminator ablation; 4 arms disentangle **goal vs temporal**. |
| `ablation_r1_r3.py` | R1→R3 ablation adding `entity_set` (tick-joined from the sidecar); focus on `interact`. |
| `rung_a_driver.py` | **Rung A** type discriminator: command vs executor-state (decomposed). Shows `delta_tick` is a cadence crutch (§11a). |
| `rung_a_aim.py` | **Rung A** aim head: yaw/pitch regression + re-target subset. Persistence wins per-packet; world-state helps only on combat re-targets. |
| `rung_a_target.py` | **Rung A** attack-target pointer over `entity_set` — **0.985**, closes the §6 entity pointer gap. |
| `dataset.py` | JSONL reader → `(obs_dict, Action)` pairs (used by `train.py`; the ablations read packet JSONL directly). |
| `features.py` | `obs_dict` → flat vector + `PACKET_TYPES`/`packet_type_label`. |
| `metrics.py` | Per-type accuracy tracker + pre-registered `(type × rung)` table. |
| `train.py` | Original MLP scaffold (fixed-width). The ablation scripts supersede it for rung work; **parameter heads go here or a successor**. |

## Reproduce the sprint results

```bash
# captured data is on disk (gitignored): results/frozen_dryrun (mining),
# results/frozen_combat (midnight survive). Re-run the ablations on it:
.venv/bin/python -m experiments.next_packet.ablation_r0_r1 \
    --recordings "results/frozen_combat/rollout-*/packets.jsonl" --epochs 50
.venv/bin/python -m experiments.next_packet.ablation_r1_r3 \
    --rollouts-glob "results/frozen_combat/rollout-*" --epochs 50

# capture fresh data (homunculus client on :2557N must be up — see CLAUDE.md):
export QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
export CRAFT_SCOUT_FANOUT_MODEL="$QWEN" CRAFT_SCOUT_UNIFY_MODEL="$QWEN" CRAFT_LOOK_AROUND_MAX_RADIUS=1
.venv/bin/python -m experiments.next_packet.capture \
    --rollouts 4 --turns 10 --goal survive --start-phase midnight \
    --model "$QWEN" --port 25570 --player agent0 --out results/frozen_combat
```

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
