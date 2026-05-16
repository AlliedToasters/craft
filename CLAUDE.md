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
- `craft/tools.py` — tool schemas + dispatch (mine_*, craft, smelt, place, surface, descend, travel, build_shelter, evasion, scan_nearest, collect_smelt).
- `craft/agent.py` — closed-loop tool-calling agent. One tool call per turn, `max_turns` cap, per-turn stats injection (`pos=(x,y,z) facing=<cardinal>` ambient).
- `craft/ambush.py` + `stress_test_shelter.py` — shelter stress harness (block-occupancy breach detector, ≥2 consecutive polls).

## Substrate primitives (operational)

- **build_shelter** — surface-intended; warns on underground. Pre-flight: floor footprint + buildable block budget (53→63→73→...). At NIGHT, threshold drops 70→35 and refuses mine_* advice.
- **evasion** — autonomous `Evasion.java` in homunculus + `/evasion/{arm,disarm,status}` + agent chokepoint. **Always-on, never per-handler arm/disarm** (maintenance trap). Validated: 17 zombies, fires in <1s, flee in ~3s, player ~0.05 blocks from anchor.
- **DoorCourtesy** — homunculus tick handler auto-closes doors/gates behind player. No API.
- **scan_nearest** — pre-flight probe; drops absent species, sorts by distance. Saves ~6.75 min/call in absent-species biome. Probe-distance ≠ reachability (use as hint, not guarantee).
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

`python -m run_tests` (sequential against canonical `:25566`) or `--concurrent` (phased fan-out across agent0..2).

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

**Wired (14 specs, one per agent tool)**:
- **Peaceful (12)**: mine_wood (agent0), mine_stone/iron/diamond/coal (agent2), surface/travel/place/craft/smelt (agent1), descend/collect_smelt (agent0).
- **Non-peaceful (1)**: evasion (agent1).
- **Mixed (1)**: shelter × 3 fan-out (agents 0/1/2).

`goto_corpse` deferred — permadeath rollouts make corpse retrieval irrelevant.

**Smoke 2026-05-16**: 13/13 PASS (peaceful + non_peaceful) in 365s. shelter runs separately.

**Don't**: long-running stress in the default suite (keep total <5min @ iters=1); import test modules into `run_tests.py` (subprocess isolation matters for Baritone crashes); add tests to `mixed` casually.

## Workflow

- **Close-watch rollouts → surface substrate gaps → fix substrate → repeat.** Beats batched offline sweeps. User-confirmed working loop.
- **Don't block on the user.** Smoke tests, healing, difficulty changes, /give all go through `POST $MC_SERVER_CMD_BASE/cmd {"cmd":"..."}` (any MC server command; player=`$MC_PLAYER_NAME`).
- **Bug-fixing in Java side requires MC restart** (e.g., tag-aware `canonicalItem`).
- **MC world seed is fixed across wipes** (`mc_wipe.sh` regenerates same world). Same (x,y,z) → same terrain. Load-bearing for reproducible stress-test replays.

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

## Roadmap

1. **Channel.** Done. xdotool → homunculus.
2. **First closed loop.** Done.
3. **Robust channel.** Done.
4. **Observability.** Inventory + position landed; nearby entities + recent chat outstanding.
5. **Concurrency / substrate-as-instrument.** Phased test runner done; rollout-side concurrency (`agent.py` still hardcodes `_PLAYER_NAME`) outstanding.

## Constraints

- Phase 1's interface was brittle by design; phase 3 retired it. **Harden new surfaces when friction warrants — not before.**
- Wurst hacks are NOT auto-on for fresh instances; tests/rollouts must enable per-process via `craft.wurst.ensure_hacks_on()` or `testkit.preflight()`.
