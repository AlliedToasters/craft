# craft

An LLM-powered Minecraft agent. The interesting claim:

> Modestly capable LLMs *can* execute long-horizon planning when paired with
> a sufficiently automated execution layer. **The substrate is the
> load-bearing object** — the same model produces wildly different ceilings
> depending on how much execution the harness absorbs.

The agent thinks; Baritone and Wurst (vanilla MC-side mods) handle motor
control. craft is the glue: tool schemas, a closed planning loop, biome-aware
respawn, shelter-build primitives, async smelting, and so on.

## Architecture

Three boxes in the general case (collapse to one for a single-machine setup):

```
┌────────────────────┐    HTTP /cmd, /log    ┌──────────────────────┐
│  agent box         │ ────────────────────► │  MC server box       │
│  craft/ (Python)   │                       │  Purpur 1.21.4 jar   │
│  llm.py → Ollama   │ ◄──── world events ── │  + mc_api.py wrapper │
│              ↓     │                       │  + screen session    │
│   /turn /goto /…   │ ────────────────────► │                      │
│   HTTP via         │                       └──────────────────────┘
│   homunculus       │
│   inside           │    ┌────────────────────────────────────────┐
│   PrismLauncher    │ ──►│  Minecraft client (MC + Fabric)        │
│                    │    │  + Baritone (pathfinding/automine)     │
│                    │    │  + Wurst (combat/QoL)                  │
│                    │    │  + homunculus (HTTP bridge)            │
└────────────────────┘    └────────────────────────────────────────┘
```

- **`craft/`** — this repo. The LLM-side planner.
- **`server/mc_api.py`** — tiny HTTP wrapper that lives next to the MC server
  and exposes `/cmd` (inject console commands) and `/log` (tail
  `latest.log`). See [`server/README.md`](server/README.md).
- **[homunculus](https://github.com/AlliedToasters/homunculus)** — Fabric mod
  that runs inside the MC client and exposes HTTP for inventory, position,
  craft, place, mine, etc. Cloned and built separately.

## Prerequisites

- Python 3.10+
- An Ollama install (any host) with at least one model pulled, OR an
  Anthropic API key for `claude-*` models.
- A Minecraft 1.21.4 Purpur/Paper/vanilla server. See
  [`server/README.md`](server/README.md) for the wrapper setup.
- A Minecraft 1.21.4 Fabric client with **Baritone**, **Wurst**, and
  **homunculus** installed. PrismLauncher is the path of least resistance;
  any launcher that supports Fabric works.

## Install

```bash
git clone https://github.com/<your-fork>/craft.git
cd craft
./setup.sh
```

`setup.sh` creates `.venv/`, installs craft editable, and copies
`.env.example` → `.env`. Then edit `.env`:

| Var | Default | When you need it |
|-----|---------|------------------|
| `ANTHROPIC_API_KEY` | *(empty)* | Running rollouts with `--model claude-*` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Override if Ollama isn't on the same box |
| `MC_SERVER_CMD_BASE` | `http://127.0.0.1:4747` | URL of the `mc_api.py` wrapper |
| `MC_PLAYER_NAME` | `Player` | Your in-game username (case-sensitive) |
| `HOMUNCULUS_HOST` | `127.0.0.1` | Host running the homunculus mod |
| `HOMUNCULUS_PORT` | `25566` | Port homunculus binds (set via the mod's config / JvmArgs) |
| `PRISMLAUNCHER_BIN` | `prismlauncher` | Path to the AppImage if not on PATH |
| `CRAFT_MC_HOST` | `127.0.0.1` | IP the `launch_agent.sh` fleet auto-joins |

All env vars are optional; defaults assume a single-box install with the MC
server, agent, and client all on `localhost`.

## First rollout

Once your MC server is running with `mc_api.py` alongside it, and your
client is in-game with homunculus loaded:

```bash
source .venv/bin/activate

# A 50-turn survive-goal rollout with the minimal prompt and Qwen3-4B:
python -m craft.agent 50 minimal \
    --permadeath \
    --start-phase dawn \
    --random-spawn-range 20000 \
    --model "hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
```

What the flags do:

- `50` — max turns.
- `minimal` — goal prompt key (`diamond` / `survive` / `survive_first` /
  `survive_shelter` / `minimal` are the built-ins; see
  [`craft/agent.py`](craft/agent.py)).
- `--permadeath` — terminate the rollout on death (default behavior is to
  respawn and keep going).
- `--start-phase dawn` — set the in-game time before the rollout starts.
- `--random-spawn-range 20000` — TP the player to a random `(x, z)` within
  ±20k, with biome rejection (oceans, deserts, badlands, etc.) and a
  ground/health check. Drop to `0` to roll out from the current spawn.
- `--model <name>` — defaults to the gemma model; swap in any Ollama model
  tag or a Claude model id.

Output goes to `results/rollout-<goal>-<timestamp>.jsonl` (one line per
turn, full inventory + position + tool call + outcome).

Want to drive the substrate yourself instead of through an LLM? Pass
`--model human` — same loop, you type tool calls at a prompt. See
[`CHEATSHEET.md`](CHEATSHEET.md) for the keyboard interface.

## Tests

```bash
python -m run_tests                 # sequential, against the canonical client
python -m run_tests --concurrent    # phased fan-out across the agent fleet
```

The test suite is per-tool integration tests (mine, craft, smelt, place,
shelter, travel, evasion, …) that hit a real MC server and judge by
pass-rate against a threshold. Total wall time under 5 minutes at
`--iters 1`.

The `--concurrent` mode requires the PrismLauncher fleet (instances
`1.21.4.agent0..9` with offline accounts of the same name). See
[`CHEATSHEET.md`](CHEATSHEET.md) for fleet provisioning notes.

## Project shape

```
craft/                # the agent (Python)
  agent.py            # closed-loop tool-calling driver
  config.py           # env-var single source of truth
  llm.py              # Ollama + Anthropic dispatch
  tools.py            # tool schemas + dispatch into homunculus
  mine.py spawn.py world.py wurst.py …
  testkit.py          # shared test substrate
server/               # the MC-side HTTP wrapper
  mc_api.py
  README.md
results/              # rollout JSONLs (gitignored)
test_*.py             # per-tool integration tests
scripts/              # rollout fans, capacity probes, latency probes
CHEATSHEET.md         # developer-side notes: rollout flags, human driver, fleet ops
CLAUDE.md             # project context for Claude Code sessions
```

## Codec / neural output (ml.MD §4a)

Forward-looking, not wired into the current LLM agent. The Phase 2 experiment
replaces Mojang's byte `StreamCodec` with a structured tagged-union action
representation that doubles as the neural head's output shape — same type by
construction in training (heuristic packets → labels) and inference (neural
prediction → packet). Round-trip invariant:
`decode(encode(packet, obs), obs) ≈ packet`. See [`ml.MD`](ml.MD) §4a for the
full design.

The output hierarchy:

```
              ┌────────────────────────────────────────┐
              │  packet_type   categorical · 1-of-11   │   tagged-union
              │                (always fires)          │   discriminator
              └─────────────────────┬──────────────────┘
                                    │
                                    ▼
                  conditioned on packet_type, a per-type
                  parameter-head bundle fires. The four
                  most structurally interesting codecs:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ move  ·  4 wire types: pos / pos_rot / rot / status_only             │
  │   pos          Δ3f vs obs       fires per wire-type variant          │
  │   rot          2f absolute      fires per wire-type variant          │
  │   on_ground    ▫                                                     │
  │   h_coll       ▫                                                     │
  └──────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────┐
  │ use_item_on  ·  right-click on a block (place / activate)            │
  │   hand          ▪×2                                                  │
  │   block_pos     ☆   pointer gap                                      │
  │   face          ▪×6                                                  │
  │   cursor        3f  ∈ [0,1]³  block-relative                         │
  │   inside        ▫                                                    │
  │   world_border  ▫                                                    │
  │   ⟦sequence⟧                                                         │
  └──────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────┐
  │ interact  ·  attack / interact / interact-at on an entity            │
  │   entity_id          ☆    pointer gap                                │
  │   action             ▪×3  ATTACK / INTERACT / INTERACT_AT            │
  │   using_2nd_action   ▫                                               │
  │   hand        opt    ▪×2  fires only on INTERACT / INTERACT_AT       │
  │   at          opt    3f   fires only on INTERACT_AT                  │
  └──────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────┐
  │ player_action  ·  dig-lifecycle (spatial) + inventory edges          │
  │   action        ▪×7   3 dig-lifecycle (spatial)                      │
  │                       + 4 inventory edges (non-spatial)              │
  │   block_pos     ☆     spatial only — non-spatial: sentinel zero      │
  │   face          ▪×6   spatial only — non-spatial: sentinel DOWN      │
  │   ⟦sequence⟧                                                         │
  └──────────────────────────────────────────────────────────────────────┘

  Remaining 4 codecs (categoricals + booleans, no pointer gap):
    swing            hand                          ▪×2
    player_input     7 movement booleans           ▫ × 7
    player_command   action + entity_id + data     ▪×9 + ☆ + scalar int
    use_item         hand + look angles            ▪×2 + 2f + ⟦sequence⟧
```

Legend:

- **▪×N** — categorical head over N labels (hand=2, face=6, action enums vary).
- **▫** — boolean head.
- **Nf** — continuous N-vector.
- **ΔNf** — continuous delta against an observation channel ("pointer in vec
  form" — the structured-action argument is the delta, never the absolute
  coord; same physical motion → same representation across different starts).
- **☆** — pointer head: attention over observation tokens (entity set / local
  block grid). Currently absolute on the wire (raw `entity_id`, integer
  `block_pos`); swaps to a learned pointer when the corresponding observation
  channels land. Documented per-codec as the *pointer gap*.
- **opt** — fires only on certain action-enum values; presence enforced by the
  codec's `__post_init__`.
- **⟦x⟧** — plumbing: round-trips but is NOT predicted; filled mechanically at
  packet construction (sequence numbers).

Cross-codec contract: every `Action` exposes
`semantic_fields: frozenset[str]` — the exact set of heads that fired for
that instance. The head masks losses against this set; plumbing never enters
the loss.

Implementation: `craft/codec/<wire-type>.py` per-codec; `craft/codec/server.py`
is the HTTP shim that exercises encode→decode on live agent traffic
(homunculus `CodecPassthrough` ships fields + obs across, counts drift).

## Where to read next

- [`server/README.md`](server/README.md) — get the MC server side running.
- [homunculus README](https://github.com/AlliedToasters/homunculus) — set up
  the in-client HTTP bridge.
- [`CHEATSHEET.md`](CHEATSHEET.md) — quick-reference for everyday rollout
  flags and the human driver.
- [`CLAUDE.md`](CLAUDE.md) — project thesis, architecture, milestones, and
  the substrate-iteration workflow.

## Status

Experimental research code. The substrate evolves rollout-to-rollout —
features are added, removed, or reworked when something breaks under load.
Don't be surprised if interfaces shift between commits; the `git log`
messages are the authoritative changelog.
