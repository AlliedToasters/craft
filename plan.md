# Sprint A + B live tracks — handoff plan (2026-05-29)

> Written as an amnesia-proof handoff. Assume you remember nothing. This file
> + the memory index (`memory/MEMORY.md`) + `results/sprint_b/RESULTS.md` are
> enough to fully reconstruct state and resume.

## 0. TL;DR — where we are

The **offline** halves of both Sprint A and Sprint B are DONE and verified
(persisted in `results/sprint_b/`). We are now standing up the live MC fleet to
run the two **live** tracks that were blocked on the fleet being down:

1. **Sprint A live wire sweep** — lossy movement codec on the wire, sweep
   bits/field, measure behavioral parity (does the controller still reach its
   goto targets) vs a codec-off control. Find the knee.
2. **Narrated non-peaceful recapture** — capture combat rollouts with
   free-text g_t narration, to un-starve the `interact × +g_t` and
   `use_item_on × +g_t` cells in Sprint B (currently n=26/52, by-rollout =
   one-directional evidence only).

A third, purely-offline refinement is also queued:
3. **Sprint B k-fold** — leave-one-rollout-out (5 folds) to put error bars on
   the `player_command × +g_t` step (currently only holdout-rollout-0).

**FLEET IS CURRENTLY UP** (as of this session): agent0/1/2 launched fresh on
homunculus ports 25570/25571/25572 with a freshly-built+distributed homunculus
jar. See §3 for verified health. If you're resuming much later, re-verify —
clients may have died.

## 1. The brief (colleague's, do not drift)

Two complementary, parallel, **non-competing** post-§14 experiments. "The gap
between them is the science — report it explicitly. Do NOT pick a winner."

- **Sprint A — Loss-tolerance.** Deliverable = parity-vs-compression curve
  (X = bits/packet, Y = behavioral parity vs codec-off control), find the knee.
  Method = lossy codec on the wire via the §14 Rung-2 harness, sweep lossiness
  on **movement packets only** (volume-dominant; pick ONE target type).
  "Controller-robustness probe, NOT a baseline for learned codecs."
  Out of scope: training; comparing to learned codecs; multi-packet-type codecs.

- **Sprint B — Next-packet predictor.** Deliverable = NLL heatmap over
  (packet type × obs rung). Pre-registered STEP cells: `interact` discriminator
  × +g_t; `player_command` × +g_t; `use_item_on` params × +§2b;
  `interact.entity_id` × +§2b. Pre-registered FLAT cells: `move_*` × +g_t;
  `swing` × any rung. Method = offline, multi-headed across 11 wire types,
  obs-ablation ladder. "Per-type metrics mandatory; aggregate accuracy banned."
  "Baritone task goes in obs; Baritone path goes in auxiliary loss as a target,
  NEVER in input." Out of scope: wire integration; live substitution.

### Five anti-patterns to REFUSE explicitly (user asked for pushback)
1. Treating the quantizer as a baseline the predictor must beat.
2. Putting Baritone path / rollout-id in predictor INPUTS (label leakage).
3. Conflating "lossy codec for the wire" with "neural action emitter".
4. Architecture decisions before obs-ablation reveals what's learnable.
5. Pre-committing to a winner between A and B.

### Execution directive (verbatim intent)
"yaw check → kick off three things (Sprint A, Sprint B on unblocked cells,
narrated combat recapture) → backfill the starved cells when the recapture
lands. The recon script stays as the §15.0 characterization artifact."

## 2. What's DONE and verified (offline)

All persisted under `results/sprint_b/`: `RESULTS.md` (full writeup + tables),
`byrollout_holdout0.json`, `random_split.json`, `offline_fidelity.json`.

### Sprint A offline fidelity gate — DONE
- Built `experiments/codec_loop/quantize.py` (lossy movement quantizer) and
  `experiments/codec_loop/offline_fidelity.py` (the gate).
- **Yaw resolved:** codec carries yaw ABSOLUTE/unnormalized
  (`craft/codec/move.py:19-23`); quantizer does `mod 360` BEFORE quantizing
  (`_wrap360`). Pitch already physically bounded ±90. This is baked in.
- Knee for the LIVE sweep brackets **b=4–5**: pos RMSE crosses the walk-step
  scale (0.28 blk) between b=5 (0.21) and b=4 (0.47 > one full step).
  Fidelity table (narrated, n=36315, 22 TP excluded):
  | bits | float bits/pkt | pos RMSE | yaw RMSE° | pitch RMSE° |
  |---|---|---|---|---|
  | 8 | 26.3 | 0.025 | 0.41 | 0.22 |
  | 6 | 19.7 | 0.106 | 1.65 | 0.76 |
  | 5 | 16.5 | 0.212 | 3.36 | 1.52 |
  | 4 | 13.2 | 0.47 | 6.9 | 3.76 |
  | 3 | 9.9 | 1.07 | 15.0 | 7.3 |
- → Live sweep should densely sample **b ∈ {8,6,5,4,3}** vs a codec-off control.

### Sprint B first run — DONE (both splits)
- Built `experiments/next_packet/gt_embed.py` (frozen `nomic-embed-text`,
  768-dim, stores raw string alongside vector per user's "store both" call;
  cache at `experiments/next_packet/.gt_embed_cache/`, 125 entries,
  gt_hash=43fde20b2ee463cb) and `experiments/next_packet/ablation_gt_content.py`
  (5 arms: R0, R1_temporal, R1_goal_onehot, R1_goal_content, R1_full_content;
  per-type NLL + acc; by-rollout AND random splits).
- **Headline findings (by-rollout = the honest number):**
  1. **NLL was essential, accuracy lies.** `player_command` top-1 acc=0.000 at
     EVERY rung, yet NLL shows a clean +g_t step (R0 3.518 → content 2.931 →
     full 2.876). Vindicates per-type-NLL mandate.
  2. `player_command × +g_t` = STEP (pre-registered ✓, ~0.6 nat).
  3. `move_player_pos_rot × +g_t` = FLAT/negative (pre-registered ✓; R0 NLL
     0.316 best — Baritone path-following, not LLM intent).
  4. Frozen content BEATS one-hot on held-out rollouts (2.931 < 3.072), and the
     comparison FLIPS under random split (one-hot wins via memorization) — that
     flip empirically demonstrates anti-pattern #2's rollout-id leakage.
  5. `interact`/`use_item_on` data-starved by-rollout (n=26/52) → confirms the
     narrated non-peaceful recapture need.

## 3. Fleet state (this session) — VERIFY before trusting

Standup performed this session (in order — the order matters):
1. `./kill_client.sh` — killed the leftover zombie agent0 (it was hp 1.17/20,
   idle at spawn origin, day 38). kill_client.sh = `pkill -f
   'org.prismlauncher.EntryPoint'` then `-9` after 3s.
2. `cd ../homunculus && ./move_to_instance.sh` — gradle build + distribute jar
   to 42 mods dirs. (NEVER deploy over a running client — corrupts lazy class
   loading; that's why kill came first. See memory
   `feedback_jar_deploy_over_running`.) homunculus HEAD = `2817f1b §14
   codec-in-the-loop ...`; the build included §13.1 `/attack_entity` +
   §14 CodecPassthrough handlers (needed by both live tracks).
3. `nohup ./launch_agent.sh N` for N=0,1,2 (staggered ~8s). Each spawns its own
   Xvfb (:200+N), llvmpipe software GL (GPU reserved for Qwen), offline account
   agentN, homunculus on 2557N.

**Verified health (this session, ~20:42):**
- 25570 agent0: hp20 food20 survival, plains, spawn (9.5,60,-9.5) — CLEAN, best
  candidate for Sprint A (it does out-and-back gotos from spawn).
- 25571 agent1: hp17 dark_forest, far out (-7308,56,8674).
- 25572 agent2: hp10 plains, near spawn (-0.5,63,-6.5).
- Relay (MC server console) UP on :4747 (`POST /cmd {"cmd":"..."}` → {"ok":true}).
- ollama UP (:11434). MC server java proc up. codec sidecar UP on :25600
  (see §4 — 404 on GET root is fine, it only serves POST /codec/roundtrip +
  GET /healthz).

**Endpoint gotcha:** homunculus position route is **`/position`** NOT `/pos`
(404). `/stats`, `/inventory` are GET 200.

## 4. The live-track architecture (how the pieces connect)

### Sprint A wire path
- `experiments/codec_loop/run_rungs.py` is the §14 IDENTITY harness. It:
  - tells homunculus `POST {base}/codec/passthrough/arm {endpoint, substitute}`
  - homunculus POSTs each allowlisted outbound packet's decoded fields to the
    external **codec sidecar** (`craft/codec/server.py`, default
    `http://127.0.0.1:25600/codec/roundtrip`), which runs `craft.codec.roundtrip`
    and returns reconstructed fields that get substituted on the wire.
  - drives traffic with two Baritone gotos (out-and-back, `--delta 28` from
    spawn), reads counters, judges PASS by **final position within tol** (NOT
    Baritone's reason string — it spuriously reports CANCELED).
  - Rung 0 = observe (substitute:false), Rung 1 = byte-identity roundtrip,
    Rung 2 = THE TEST (substitute:true).
- `craft/codec/server.py` is currently a **LOSSLESS** sidecar
  (`--port 25600`). It is a thin shell around `craft.codec.roundtrip`.

### THE GAP / NEXT BUILD STEP for Sprint A (not yet done)
The lossy sweep needs the sidecar to apply `quantize.quantize_move` at a
configurable bit level on `move_*` packets, passing everything else through
lossless. Two clean options:
  - **(a) Add a `--quant-bits N` (and maybe `--quant-types`) flag to
    `craft/codec/server.py`**, so when set it quantizes the move family before
    re-encoding. Then the sweep = run the sidecar at each bit level (restart per
    level, OR add a `/config` POST to retune live) and run `run_rungs.py
    --rungs 2 --port 25570 --out results/sprintA/b<N>.json` against it, plus a
    codec-off control (`--rungs 2` with the lossless server = b=∞ control, and
    a no-arm baseline). RECOMMENDED — keeps the wire path identical, only the
    sidecar math changes.
  - (b) A separate lossy sidecar script that imports quantize. More code, same
    effect. Prefer (a).
- `quantize.py` already has the pieces: `quant_scalar(v, lo, hi, bits)` (mid-
  tread uniform, clamps, bits<=0→midpoint), `_wrap360`, `quantize_move(action,
  *, pos_bits, yaw_bits, pitch_bits)` (returns replace(action, pos=quantized,
  rot=(mod360 yaw, pitch))), `float_bits(...)`. Constants: POS_RANGE=8.0,
  YAW 0..360, PITCH -90..90.
  ⚠️ I was mid-read confirming quantize.py's body was fully fleshed out (not a
  stub) when the channel died — RE-READ `experiments/codec_loop/quantize.py`
  first thing and confirm `quantize_move` has a real body (the summary says it
  does; one truncated read this session showed a `pass` stub which was almost
  certainly a channel-truncation artifact, but VERIFY).

### Sprint A run recipe (once sidecar has lossy mode)
```
# control (lossless) — already-working identity harness:
.venv/bin/python -m experiments.codec_loop.run_rungs --port 25570 --rungs 0,1,2 \
    --out results/sprintA/control.json
# per bit level (restart sidecar at b, or retune), rung 2 only:
for b in 8 6 5 4 3; do
  # (re)start craft.codec.server with --quant-bits $b on :25600
  .venv/bin/python -m experiments.codec_loop.run_rungs --port 25570 --rungs 2 \
      --out results/sprintA/b$b.json
done
```
Parity metric = targets_reached/2 per run + drift/substitute_errors + p99
substitute latency (budget 10ms). Plot reached-rate (and any drift) vs b → knee.
Predicted from offline gate: clean through b≈5, degrades b≈4 and below.

### Sprint B recapture path
- `experiments/next_packet/capture.py` drives a rollout and records packets.
  Args: `--rollouts --turns --goal --model (required) --port --player
  --spawn-range --start-phase --difficulty --out --purpose --narrate`.
  `--narrate` is load-bearing: makes g_t free-text intent (not tool name) —
  §12.2. Without it the +g_t rung is degenerate.
- Need a **NON-PEACEFUL** (difficulty hard/normal) narrated capture so combat
  packets (interact / use_item_on / attack) flow with content-rich g_t.

### Sprint B recapture recipe
```
# difficulty is GLOBAL on the server — set via relay:
curl -s -X POST http://127.0.0.1:4747/cmd -H 'Content-Type: application/json' \
    -d '{"cmd":"difficulty hard"}'
QWEN="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
.venv/bin/python -m experiments.next_packet.capture --narrate --goal survive \
    --start-phase midnight --difficulty hard --model "$QWEN" \
    --rollouts 4 --turns 10 --port 25571 --player agent1 --out results/frozen_combat
# (can run a second on --port 25572 --player agent2 for more volume)
```
Then backfill Sprint B: re-warm gt_embed cache on the new set, re-run
`ablation_gt_content.py` with the combat rollouts included (it globs rollout
dirs), check `interact × +g_t` and `use_item_on × +g_t` now have enough n to
distinguish step-vs-flat.

⚠️ **Sequencing of the two live tracks (difficulty is global):** Sprint A's
parity is cleanest under PEACEFUL (no mobs perturbing the gotos / no combat
packets confounding move substitution). The recapture NEEDS hard. So run
**Sprint A first under peaceful**, finish it, THEN flip the server to hard for
the recapture. Don't run them concurrently on the same server difficulty.
(Sprint A uses agent0/25570; recapture can use agent1+2.)

## 5. Immediate next steps (resume here)

1. **Re-verify fleet** (§3): `curl :2557{0,1,2}/stats`; if down, redo §3
   standup (kill → build/distribute → launch 0,1,2). Confirm codec sidecar on
   :25600 (`curl :25600/healthz`); if down: `.venv/bin/python -m
   craft.codec.server --port 25600 &`.
2. **Re-read `experiments/codec_loop/quantize.py`** end to end; confirm
   `quantize_move` body is real (not a stub). Fix if truncated.
3. **Add lossy mode to `craft/codec/server.py`** (option (a), §4): `--quant-bits`
   + quantize move family, pass-through else. Smoke: POST a move packet at
   b=4, confirm fields come back quantized; POST a non-move, confirm identity.
4. **Run Sprint A sweep** under PEACEFUL (`difficulty peaceful` via relay):
   control + b∈{8,6,5,4,3} on agent0/25570 → `results/sprintA/*.json`. Tabulate
   reached-rate + drift + latency vs b. Confirm the offline-predicted knee.
5. **Flip to hard**, run **narrated non-peaceful recapture** on agent1/2 →
   `results/frozen_combat`.
6. **Backfill Sprint B**: re-warm gt_embed, re-run ablation with combat set,
   report interact/use_item_on cells.
7. **Sprint B k-fold** (offline, can do anytime): leave-one-rollout-out ×5 for
   error bars on the player_command step.
8. **Write up the GAP** (the central ask): movement needs ~5–6 bits/field to
   stay faithful (A) yet is nearly R0-predictable (B, pos_rot NLL 0.316 — highly
   compressible *because* predictable). Report A and B side by side; do NOT
   pick a winner. The A side isn't complete until the live parity sweep runs
   (offline fidelity ≠ behavioral parity).

## 6. Standing constraints (load-bearing — do not violate)

- Use **`.venv/bin/python`** for all experiments.
- **Do NOT delete/overwrite user-created untracked files**: `.vscode/`,
  `scratch.txt`, `glossary.md`, `scripts/bigN20_*.sh`. (This `plan.md` is new;
  fine to overwrite it.)
- **jar deploy over running client corrupts it** → always kill → deploy →
  relaunch. Never split an A/B into fresh-vs-stale jars.
- **pkill self-match trap**: use the `[x]` bracket trick AND keep kill patterns
  out of any command that also contains the relaunch string.
- **Verify every number via file readback** — there is an intermittent tool-
  channel bug that garbles/truncates stdout and file reads (the user says it's
  unrelated to our work and will clear). Persist results to JSON; corroborate
  across ≥2 reads before trusting a figure.
- **Don't block on the user** for smoke/heal/difficulty/give — all via
  `POST $MC_SERVER_CMD_BASE/cmd` (:4747).
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  (User said earlier "no need to commit — let me run the design questions by my
  colleague." Nothing committed this session.)
- **Config is env-driven** (`craft/config.py`): `HOMUNCULUS_PORT=2557N`,
  `MC_PLAYER_NAME=agentN` retarget the whole stack at one client. Never
  hardcode ports/players.

## 7. Key file map

- `experiments/codec_loop/quantize.py` — lossy move quantizer (Sprint A).
- `experiments/codec_loop/offline_fidelity.py` — Sprint A offline gate (DONE).
- `experiments/codec_loop/run_rungs.py` — §14 identity harness / live driver.
- `experiments/codec_loop/recon.py` — §15.0 characterization artifact
  (`--json` → /tmp/recon_summary.json; reads frozen_narrated/combat/dryrun).
- `craft/codec/server.py` — codec sidecar (:25600); NEEDS lossy mode added.
- `craft/codec/move.py` — MoveAction; yaw absolute/unnormalized (lines 19-23).
- `craft/codec/base.py` — registry; `encode/decode/roundtrip/fields_close`.
- `experiments/next_packet/gt_embed.py` — frozen g_t embedder (DONE).
- `experiments/next_packet/ablation_gt_content.py` — Sprint B ablation (DONE).
- `experiments/next_packet/capture.py` — rollout recorder (`--narrate`).
- `experiments/next_packet/README.md` — pipeline docs (per-type table, aggregate
  banned, R1-content named as next addition).
- `results/sprint_b/` — all verified offline artifacts + RESULTS.md.
- `launch_agent.sh N` / `kill_client.sh` — fleet lifecycle.
- `../homunculus/move_to_instance.sh` — build + distribute jar (relaunch after).
- memory: `memory/MEMORY.md` index → `project_obs_ablation_sprint.md` (§15
  section has the full sprint state).
