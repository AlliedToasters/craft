# CRAFT

here are some simple commands for general interfacing with the local game server:

### Enter a command in the server console
```bash
curl -X POST $MC_SERVER_CMD_BASE/cmd -H 'Content-Type: application/json' -d '{"cmd":"<YOUR COMMAND HERE>"}'
```

### Read console last n lines
```bash
curl "$MC_SERVER_CMD_BASE/log?n=5"
```

# Do a rollout
All the preflight (random TP + biome-accept loop, gamemode bounce, heal/clear, difficulty bounce, time set, Wurst hacks
  check) lives in _apply_setup() at agent.py:648 and fires automatically whenever you pass --start-phase and/or
  --random-spawn-range. So the manual rollout is just the agent CLI with those flags set — same line we use in the loop:

  cd <craft-checkout>
  .venv/bin/python -m craft.agent 50 survive_shelter --permadeath --start-phase dawn --random-spawn-range 20000

  What that gets you (in order, all internal):
  1. difficulty peaceful + effect clear — wipe mobs/lingering effects
  2. Read current position as anchor, random xz offset within ±20000
  3. gamemode creative → tp → wait for on_ground → reject in_water/in_lava/bad biome (ocean, desert, badlands, ice_spikes,
  frozen_river, frozen_peaks, windswept_*); up to 8 retries
  4. Switch to survival, wait 1.2s, verify HP=20 (catches "TP'd inside a wall")
  5. time set dawn (or whatever phase)
  6. difficulty easy
  7. ensure_wurst_hacks_on() — confirms KillAura/AutoEat/AutoTool/AntiKnockback/AntiSpam/AutoSwim etc., logs the report into
   the JSONL header
  8. Loads GOAL_PROMPTS["survive_shelter"], opens results/rollout-<goal>-<ts>.jsonl, starts turn loop

  Useful variants:
  - Want it backgrounded with logs like r17 — nohup .venv/bin/python -m craft.agent 50 survive_shelter --permadeath
  --start-phase dawn --random-spawn-range 20000 > results/manual-r1.log 2>&1 &
  - Want Haiku instead — append --model claude-haiku-4-5-20251001
  - Pin the JSONL name — --jsonl-out results/manual-r1.jsonl
  - Skip the TP / use current spawn — --random-spawn-range 0 (preflight still does time + Wurst check, but no teleport/biome
   retry)
  - Skip everything pre-rollout — omit both --start-phase and --random-spawn-range (defaults to none + 0, which
  short-circuits _apply_setup)

# Drive the substrate by hand

A human can take the LLM's seat in the agent loop via `--model human`. Same invocation surface as a real rollout — only the planner is different:

```bash
cd <craft-checkout>
.venv/bin/python -m craft.agent 50 survive_shelter --permadeath \
    --start-phase dawn --random-spawn-range 20000 --model human
```

This routes the per-turn planning step through `_chat_with_tools_human` in `craft/llm.py`. Everything else (equip step, shelter-watch, message-trim, pre-dispatch death checks, JSONL logging, _apply_setup TP) runs identically to an LLM rollout, so there's nothing for the driver to drift from. "Plan time" is whatever the human spent thinking and accumulates into the same `plan_s_total` metric.

At each turn you see the context the LLM would see, then a prompt:

```
--- context (what the model would see this turn) ---
[shelter_watch] (armed | breach) ...
Stats: HP=20.0/20.0 food=20 sat=5.0 air=300 pos=(...) facing=north time=DAY ...
Current inventory: ...
--- end context ---
> mine_stone quantity=8
```

Tool calls are shell-style `tool key=value` lines. Values are parsed as JSON literals when possible (so `quantity=8` becomes int, `fair=true` becomes bool, bare strings stay strings). Arrow-key history works.

In-prompt commands:
- `help` — list all tools with their param signatures (`?` marks optional)
- `schema <tool>` — dump the full JSON schema (same one the LLM sees), including the description
- `system` — print the system prompt (game rules) currently in effect for this rollout
- `quit` / `exit` / Ctrl-D — end the rollout (returns no tool call, agent loop's "no tool call returned; stopping" path fires and the JSONL flushes cleanly)
