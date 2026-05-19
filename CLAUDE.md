# craft

LLM-powered Minecraft agent. Mysterious by nature.

## Thesis

Modestly capable LLMs *can* execute long-horizon planning when paired with a sufficiently automated execution layer. **The substrate is the load-bearing object** — the same model produces wildly different ceilings depending on how much execution the harness absorbs. Experiments should ablate substrate features with the model held constant.

## Architecture

**Brain** — LLM (local Ollama at `$OLLAMA_BASE_URL`, currently `gemma-4-vanilla-q4-32k`; or Anthropic Haiku/Sonnet as benchmark). Plans, decides, replans. Never touches motor control. One tool call per turn (`max_tokens` ~256 — tight return loop, high turn count beats batched plans).

**Body** — Minecraft 1.21.4 client with Baritone (pathfinding, automining) + Wurst (combat, QoL). Atomic ops (craft, place, smelt, equip, mine, goto, stop, scans) driven through [homunculus](https://github.com/AlliedToasters/homunculus), a Fabric bridge exposing HTTP at `127.0.0.1:25566`.

**Channel** — `requests` against homunculus. Synchronous, structured-response, no window focus needed.

The split is the point: LLM reasons over goals; Baritone/Wurst execute. When designing a capability, ask "can Baritone or Wurst already do this if told?" before reaching for an LLM-level solution.

## Layout

- `craft/config.py` — **env-driven root**: `HOMUNCULUS_HOST/PORT/BASE`, `PLAYER_NAME`, `SERVER_CMD_BASE`. Reads `HOMUNCULUS_PORT` + `MC_PLAYER_NAME` env vars. **Every module pulls bases from here** — never hardcode `127.0.0.1:25566` or a specific player name. Concurrent retargeting depends on this.
- `craft/world.py` — `set_difficulty`, `set_time`, `set_gamemode`, `PHASE_TICKS`. No raw `cmd("difficulty ...")` anywhere else.
- `craft/spawn.py` — `random_spawn(range_blocks, ...)` biome-aware (8x retry, rejects water/lava/bad_biome/HP-drop/stuck). `BAD_BIOMES` grows only from observed failures (oceans, desert, badlands variants, ice_spikes, frozen_*, windswept_*).
- `craft/testkit.py` — `cmd/pos/stats/inventory/preflight` wrappers, `setup_clean`, `build_arena`, `TestLogger`. `preflight` enables Wurst hacks by default. `setup_clean` skips its difficulty set when `SUITE_COORDINATOR_MANAGES_DIFFICULTY=1`.
- `craft/wurst.py` — `ensure_hacks_on(REQUIRED_HACKS)`. Fresh clients boot with all hacks off; preflight pre-enables. Required set includes KillAura/AutoEat/AutoTool/AutoRespawn/AntiKnockback/AutoSword/Fullbright/AutoReconnect. (AutoSwim was tried and removed 2026-05-16 — thrashed against Baritone, caused drownings.)
- `craft/mine.py` — candidate-cycling miner (`mine_any_log/stone/iron/diamond`). `fair=True` switches to blind 1×2 tunneling (`tunnel_for`) for "no x-ray" mode.
- `craft/llm.py` — Ollama / Anthropic chat (`chat()`, `chat_with_tools()`).
- `craft/subagent.py` — generic `synthesize(prompt, payload, model)` for single-shot non-tool LLM calls. Ollama + Anthropic paths; Qwen3 routes plain output through `message.reasoning` (chat-template quirk), so we read `reasoning` when `content` is empty. Pattern scaffold for orthogonal synthesis tasks; first user is scout.
- `craft/scout.py` — `scan_chunk(dx, dz)` + `describe_chunk` + `describe_neighborhood(radius, fanout_model, unify_model)`. L3-compaction (heightmap + interesting-blocks) shrinks ~90KB raw block payload to ~5KB; below this threshold Qwen3-4B stops "going meta" and produces actual scout reports. TTL chunk-description cache. Env knobs: `CRAFT_SCOUT_FANOUT_MODEL`, `CRAFT_SCOUT_UNIFY_MODEL`, `CRAFT_SCOUT_CACHE_TTL_S`.
- `craft/tools.py` — tool schemas + dispatch (mine_*, craft, smelt, place, surface, descend, travel, build_shelter, evasion, scan_nearest, collect_smelt, look_around). Env knob: `CRAFT_LOOK_AROUND_MAX_RADIUS` silently clamps requested radius.
- `craft/agent.py` — closed-loop tool-calling agent. One tool call per turn, `max_turns` cap, per-turn stats injection (`pos=(x,y,z) facing=<cardinal>` ambient).
- `craft/ambush.py` + `e2e/stress_test_shelter.py` — shelter stress harness (block-occupancy breach detector, ≥2 consecutive polls).

## Substrate primitives (operational)

- **build_shelter** — surface-intended; warns on underground. Pre-flight: floor footprint + buildable block budget (53→63→73→...). At NIGHT, threshold drops 70→35 and refuses mine_* advice.
- **evasion** — autonomous `Evasion.java` in homunculus + `/evasion/{arm,disarm,status}` + agent chokepoint. **Always-on, never per-handler arm/disarm** (maintenance trap). Validated: 17 zombies, fires in <1s, flee in ~3s, player ~0.05 blocks from anchor.
- **water_aversion** — sibling reflex (`WaterAversion.java` + `/water_aversion/{arm,disarm,status}`). Fires on eye-submergence (`player.isUnderWater()`); cancels Baritone and paths to nearest dry standing spot via in-process BFS (radius 12 horiz, ±6 vert). Per-turn armed at the same chokepoint as evasion; arm body is empty (target computed at fire-time). Reflex is a substrate **pattern** — expect more (lava, suffocation, fall) sharing this shape.
- **DoorCourtesy** — homunculus tick handler auto-closes doors/gates behind player. No API.
- **scan_nearest** — pre-flight probe; drops absent species, sorts by distance. Saves ~6.75 min/call in absent-species biome. Probe-distance ≠ reachability (use as hint, not guarantee).
- **look_around(radius)** — agent-callable scout. Wraps `scout.describe_neighborhood`: fans out a per-chunk subagent over (2r-1)² chunks then a synth-of-synths into a unified report with cardinal-direction hints. Daily-driver caps `radius` at 1 via `CRAFT_LOOK_AROUND_MAX_RADIUS=1` (silent clamp) — pure-qwen r=2 saturates the GPU and kills agents via planning latency. TTL chunk-description cache (30s) at r≥2 → 8-9× speedup on warm calls; ~7% hit rate at r=1.
- **travel-scout interlock** — `handle_travel` pre-scans the corridor (player Y±1, ±3 perp, chunk-aligned per 4-block sample) for hazards (lava). On hit, clamps distance to stop ~2 blocks short and surfaces a `(clamped from N — lava at (x,y,z))` postscript. Honors no-dispatch-guards: action proceeds, no refusal.
- **collect_smelt** — fire-and-forget furnace + later collect; smelt is async in MC reality. Empty-arg has a fallback to find active furnace.
- **Shelter guards** — pre-shelter dusk guard (T-N before nightfall), shelter-stay guard (won't leave armed cavity), night-craft lockout (no `goto_home` walking agent out at night). Pre-dispatch death check at every tool boundary.
- **Bed + sleep** — **GAP**. 3 wool + 3 planks skips night entirely. Agents won't idle 8 min inside shelter; next high-leverage substrate fix.

## Survival rules (load-bearing)

- **MC damage is burst** — HP-threshold reactive rules are too slow. Only proactive (dusk-approach) shelter triggers work.
- **Wood-species recipe substitution** — `_resolve_wood_substitute()` handles all 9 species via `#planks` tag. Java-side `Recipes.canonicalItem` is now tag-aware (prefers in-inventory species).
- **Baritone consumes throwaway inventory during craft-triggered goto** — fix: substrate-level recipe-aware `allow_place` flag.
- **Shelter doorway pathing** — Baritone routes through 1×2 opening when `allow_break=true`, instead of breaking new cells. Load-bearing for closed-structure work.
- **Village spawn confound** — random TP per rollout; allow_break demolishes village houses during mine_wood (free doors/planks).

## Agent fleet (concurrency)

Ten PrismLauncher instances `1.21.4.agent0..9` at `$XDG_DATA_HOME/PrismLauncher/instances/` (e.g. `~/.local/share/PrismLauncher/instances/`). Each:
- Pinned to homunculus port `2557N` via `JvmArgs=-Dhomunculus.port=2557N` + `OverrideJavaArgs=true`.
- Pinned to offline account `agentN` via `InstanceAccountId=<offline-uuid>` + `UseAccountForInstance=true`. UUIDs = `MD5("OfflinePlayer:agentN")` (version-3).
- Same MC server (`$CRAFT_MC_HOST`); 10 distinct players.

**Launch**: `./launch_agent.sh N` — wraps `$PRISMLAUNCHER_BIN` with `-l 1.21.4.agentN -s $CRAFT_MC_HOST -a agentN`. The `-a` flag is load-bearing: it bypasses the GUI's cached `accounts.json`. Always pass it.

**Capacity**: ~1.1-1.2 GB RSS idle, 4GB max each → 40GB worst-case. The author's box has 32GB. **Don't run all 10**. 3 is comfortable; 8 borderline.

**Diagnostic when account feels wrong**: check `instances/1.21.4.agentN/minecraft/logs/latest.log` for `Setting user: agent<N>`. If it says a different user (your cached login), the GUI is stale — restart or use `-a`.

## Test suite

Two homes:
- `tests/` — unit tests (pure-Python, ~213 specs, run via `pytest`). No live MC.
- `e2e/` — integration tests (`test_*.py`, `stress_test_shelter.py`). Live MC + homunculus. Orchestrated by `run_tests.py` as subprocesses (`python -m e2e.<name>`); never imported into the runner. Excluded from pytest via `testpaths` so `pytest` stays offline-safe.

`python -m run_tests` (sequential against canonical `:25566`) or `--concurrent` (phased fan-out across agent0..N).

**Phase-grouped concurrent model**:
- Tests declare `world_state`: `peaceful` / `non_peaceful` / `mixed`.
- Phases run sequentially (`peaceful` → `non_peaceful` → `mixed`); coordinator owns difficulty per phase except `mixed` (tests self-manage, accept racing).
- Within a phase, tests fan out across `concurrent_agents: list[int]`.
- Subprocess env: `HOMUNCULUS_PORT=2557N`, `MC_PLAYER_NAME=agentN`, `SUITE_COORDINATOR_MANAGES_DIFFICULTY=1` (non-mixed only).
- Per-agent JSONLs (`results/suite-<name>-agent<N>.jsonl`), `_judge_combined` merges for pass-rate.

**Contract for a new test**:
- Imports from `craft.testkit` only (no duplicate HTTP / setup / arena helpers).
- Has `--iters --spawn-range --pass-rate --out --seed --quiet`.
- One JSONL line per iter with `passed: bool` + `fail_reason` + coords + biome.
- Does NOT mutate global state in coordinator-managed phases (read `SUITE_COORDINATOR_MANAGES_DIFFICULTY`).
- **Pass-rate vs threshold, never pass/fail.** Exit code is informational; JSONL is truth.

**Per-agent serialization (2026-05-16)**: multiple tests targeting the same agent within a phase queue and run sequentially. Across agents they advance in parallel. `_launch_phase` builds the queues; `_launch_one` is the spawn helper.

**Wired (16 specs)**:
- **Peaceful (14)**: mine_wood (agent0), mine_stone/iron/diamond/coal (agent2), surface/travel/place/craft/smelt (agent1), descend/collect_smelt/burrow/doorway_placement (agent0).
- **Non-peaceful (1)**: evasion (agent1).
- **Mixed (1)**: shelter × 5 fan-out (agents 0..4).
- **Fixed-arena (no fan-out)**: burrow, doorway_placement — both use coord (5000,100,5000), single-agent only.

**Smoke 2026-05-16**: 13/13 PASS (peaceful + non_peaceful) in 365s. shelter runs separately.

**Don't**: long-running stress in the default suite (keep total <5min @ iters=1); import test modules into `run_tests.py` (subprocess isolation matters for Baritone crashes); add tests to `mixed` casually.

## Workflow

- **Close-watch rollouts → surface substrate gaps → fix substrate → repeat.** Beats batched offline sweeps. User-confirmed working loop.
- **Don't block on the user.** Smoke tests, healing, difficulty changes, /give all go through `POST $MC_SERVER_CMD_BASE/cmd {"cmd":"..."}` (any MC server command; player=`$MC_PLAYER_NAME`).
- **Bug-fixing in Java side requires MC restart** (e.g., tag-aware `canonicalItem`).
- **MC world seed is fixed across wipes** (`mc_wipe.sh` regenerates same world). Same (x,y,z) → same terrain. Load-bearing for reproducible stress-test replays.

## Daily driver config (2026-05-17 baseline)

Validated at N=25 = 92% survival; both deaths underground-mob, no substrate-caused deaths. Run via `./scripts/run_bigN_pureqwen.sh` (5 waves × 5 agents). Manual invocation env:

```
QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
CRAFT_SCOUT_FANOUT_MODEL="$QWEN"           # scout fan-out subagent
CRAFT_SCOUT_UNIFY_MODEL="$QWEN"            # scout synth-of-synths
CRAFT_LOOK_AROUND_MAX_RADIUS=1             # critical: 1 chunk per call
HOMUNCULUS_PORT=2557N  MC_PLAYER_NAME=agentN
python -m craft.agent 30 minimal --model "$QWEN" --start-phase dawn --random-spawn-range 20000 ...
```

`CRAFT_LOOK_AROUND_MAX_RADIUS=1` is load-bearing for pure-qwen at fleet scale. Without it, 5 concurrent agents queue 9-25 scout fan-outs each → 40-80s look_around latency → agents drown / get mob-killed while idle in tool execution. With it, fan-out averages 1.65s. Same model, one substrate parameter, capability flips. Dominant remaining death mode: underground mobs (qwen barely uses build_shelter — 1× across N=25).

## Model-specific notes

- **gemma**: `reasoning_effort="low"` (not `"none"` — breaks tool calls); `stop=["<|channel|>","<|tool_response>"]`; `max_tokens=1024` for clean turns. Native tool calling works; multi-tool batching is default — keep prompts narrow to force one-at-a-time.
- **Haiku 4.5**: high-capability control; plan_s mean ~1.5s (17× gemma). First T50 survival run (R1, 2026-05-14).
- **Gemma intended deployment target** per thesis; Haiku is benchmark / capability ceiling.

## Milestones reached

- 2026-05-10: warm-start diamond, 8 turns (Haiku).
- 2026-05-11: cold-start diamond, 30 turns (Haiku); homunculus replaces xdotool.
- 2026-05-11: survive-goal rollout — 27 diamonds, first emergent `mine_coal`, strongest substrate-thesis evidence.
- 2026-05-14: first iron-tier reach (Haiku R5), 9 rollouts + 12 substrate fixes that session.
- 2026-05-15: r16 — gemma full T50 + diamond + 6 advancements, day 4 alive, zero deaths. r17 reproducibility fail (creeper T15) — r16 was an outlier; n=3+ before claiming baseline.
- 2026-05-15: agent fleet + phase-grouped concurrent test runner shipped.
- 2026-05-17: look_around scout subagent + travel-scout interlock + chunk-description TTL cache shipped. Pure-qwen daily-driver N=25 baseline = 92% survival, 0 substrate-caused deaths.

## Roadmap

1. **Channel.** Done. xdotool → homunculus.
2. **First closed loop.** Done.
3. **Robust channel.** Done.
4. **Observability.** Inventory + position landed; nearby entities + recent chat outstanding.
5. **Concurrency / substrate-as-instrument.** Phased test runner done; rollout-side concurrency (`agent.py` still hardcodes `_PLAYER_NAME`) outstanding.

## Constraints

- Phase 1's interface was brittle by design; phase 3 retired it. **Harden new surfaces when friction warrants — not before.**
- Wurst hacks are NOT auto-on for fresh instances; tests/rollouts must enable per-process via `craft.wurst.ensure_hacks_on()` or `testkit.preflight()`.
