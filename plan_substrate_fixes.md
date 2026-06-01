# Substrate fixes — next iteration plan (2026-06-01)

> Handoff-grade. Context: the mining no-progress watchdog + two-tier placement
> make-room shipped & measured this session (homunculus `d64a8a7`, craft
> `36a9eee`, analysis tooling `2d26541`). The peaceful goal=diamond brain A/B
> (Haiku 3/20=15% vs Qwen 1/18=6%) then **exposed two depth-residuals** the
> surface-heavy minimal wave had hidden. Memory: `project_peaceful_diamond_ab`,
> `project_mining_watchdog_shipped`, `project_placement_slope_blind`.

## State / numbers to beat

Peaceful, dawn-pinned, 45 turns, fixed-substrate jar:
- `no_placeable_spot`: **0** in both diamond waves (Tier-1 vertical search holds at depth). ✅
- Mining watchdog catches SURFACE oscillation (`mine_wood` FAIL 2–6s, `no_progress` fired 30–36×) but NOT deep-mine (`mine_stone/iron/diamond` FAIL 30–58s, `timeout` fired 20–24×). Failed-mine = **14% (Haiku) / 25% (Qwen)** of wall at depth.
- Underground `no_space`: **66 (Haiku) / 114 (Qwen) = 16.4% of Qwen wall**. All placement-spot failures were `no_space`, zero `no_placeable_spot`.

Analyzers: `scripts/wallclock_by_tool.py` (total_s by tool×outcome, with think/exec/ctx split), `scripts/diamond_tally.py` (diamonds / peak-tier / mine_diamond hit-rate).

---

## Fix B — Underground `no_space` escape-check (READY; high-confidence)

**Root cause:** `Placer.searchAndPrecheck`'s `RING_1_OPEN_MIN=3` anti-encasement
gate is mis-specified for tunnels. In a 1-wide × 2-tall tunnel the player has
~2/8 ring tiles open → guard trips *before the candidate search runs*. But
placing a block in the open tunnel-floor cell ahead does NOT wall the player in.

**Fix:** replace the blanket pre-search gate with a per-candidate **escape
check**:
- Drop the `openCount < RING_1_OPEN_MIN → no_space` early return.
- A candidate is valid iff, after placing there, the player still has **≥1 open
  ring-1 escape tile** (placement doesn't seal the last exit).
- Tunnel: forward/back candidates leave the opposite end open → valid. True
  pocket (one open tile, placing it seals the player) → still rejected.
- Isolated to `/place`; env kill-switch `HOMUNCULUS_PLACE_ESCAPE_CHECK` (default
  on); `placeAt`/shelter path untouched.

**Regression coverage (load-bearing — same Placer the make-room fix touched):**
- Add a `tunnel` scenario to `e2e/test_place_constrained.py`: 1-wide 2-tall
  tunnel, give crafting_table, assert places (pre-fix returns `no_space`).
- Re-run the **shelter + doorway_placement** e2e as the encasement guardrail —
  the risk is re-introducing the very encasement the guard prevented. These
  must stay green.

**Cost:** one homunculus rebuild + `fleet.sh down→deploy→up`.
**Expected:** Qwen placement-spot failures drop from 16% of wall toward ~0;
crafting-table/furnace placement underground stops looping.

---

## Fix A — Deep-mine `timeout` — VERDICT IN (2026-06-01)

**Diagnostic shipped** (`MineHandler.java`, env `HOMUNCULUS_MINE_DIAG`, default off):
appends `[diag active=Ns invGain=N net=N dy=N path=N pathEvents={...}]` to the
`timeout`/`no_progress` message. Captured on a peaceful goal=diamond 8-agent
wave (`results/bigN20-easy-qwen-20260601-133310`), 7 timeout events:

| tool | inv | net | dy | path | p/n | class |
|------|----:|----:|---:|-----:|----:|-------|
| mine_wood    | 9 | 19 | -4 | 20 | 1.1 | (b) productive |
| mine_diamond | 0 | 28 | 16 | 67 | 2.4 | (b) directed travel |
| mine_diamond | 0 | 26 | -21 | 33 | 1.3 | (b) deep descent |
| mine_diamond | 0 | 20 | -16 | 29 | 1.4 | (b) deep descent |
| mine_diamond | 0 | 12 | -6 | 35 | 2.9 | (a) wandering |
| mine_diamond | 0 | 76 | 18 | 99 | 1.3 | (b) far travel |
| mine_iron    | 2 | 22 | -21 | 35 | 1.6 | (b) productive |

**VERDICT: (b) dominates 6/7.** The targets are reachable — Baritone computes a
valid path and is *executing* it (CALC_STARTED + CALC_FINISHED_NOW_EXECUTING,
**zero CALC_FAILED** during any timeout). Net displacement 12–76 blocks with
path≈net (directed, not orbiting); `invGain>0` in 2/7 confirms productive mining.
The 45s per-candidate guillotine cuts a mine that is *still making progress*. The
movement watchdog correctly abstains (player is moving), and fired `no_progress`
14× separately on the genuinely-stuck case. So Fix A is a **budget/decomposition**
problem, NOT a reachability bail.

**RESULT (2026-06-01) — budget raise BUILT, then DROPPED.** Implemented option 1
(deep-ore floor 45→120s, client+server). Before/after qwen peaceful diamond N=8
waves (`...-133310` 45s vs `...-144002` 120s): the mechanism worked — deep-ore
`timeout` 6→0 — but bought **zero tier/diamond progress** (both waves topped at
STONE pickaxe, 0 iron, 0 diamonds) and made failed deep-mines ~2.5× costlier
(mine_iron FAIL 19.6→50.5s). Root cause: **0/8 rollouts ever held an iron
pickaxe, yet `mine_diamond` was called 26× — every call doomed** (diamond ore
needs iron+ to drop). The deep-mine residual for qwen is a BRAIN/tier-ordering
problem, not a substrate-budget one. Per user call: **dropped the budget
mechanism; kept only the `HOMUNCULUS_MINE_DIAG` diagnostic** (pure observability,
default off). Budget could only ever help an iron-reaching brain (Haiku) —
untested, deferred.

**Mechanism options (the (b) branch) — for reference, not pursued:**
- **Raise per-ore budget** (mine_diamond/mine_iron 45s → ~90–120s). Smallest,
  isolated, reversible. Cost: the rare genuinely-far-unreachable target (≈1–2/7)
  burns up to 2× before failing; watchdog still bails stuck mines in 12s.
- **Progress-aware deadline** (extend while net-progressing toward target, cap at
  HARD_CAP=300s). Elegant — reuses the watchdog's move signal — but the
  "moving-but-never-arriving" orbit (the 1/7 (a) case) could run to the 300s cap,
  worse than a clean 45s fail.
- **Split descend-from-mine** (descend to ore layer as its own budgeted op). Biggest
  change, new agent-facing surface; likely overkill for a 7-event residual.

---

## Fix A — (superseded; original diagnose-first scoping kept for reference)

**Why not just build a watchdog:** the movement test can't fire (the agent IS
digging); an inventory-stall test **false-trips legit approaches** (descending
to y≈-58 + tunneling toward a diamond clump legitimately gains no diamond for
>30s while productively moving); and a time-cap kills real mining (successful
`mine_stone` runs 47–63s). A naive fix would REGRESS deep diamond mining.

**Step 1 — diagnostic (do this before any code):** instrument/tape ~5
`mine_diamond`/`mine_iron` `timeout` failures and classify each:
- **(a) no valid path** — Baritone repeatedly path-fails to the target
  (genuinely unreachable: lava/gap/walled).
- **(b) valid path, 45s insufficient** — legit slow descent/tunnel that ran out
  of clock.

Capture handle: `CRAFT_MINE_FORCE_XRAY=1` targets a specific clump; log
Baritone path events + per-candidate timing; `--record-video` on a couple of
agents for tape ([[feedback_video_debugging]]).

**Step 2 — mechanism by verdict:**
- If **(a)** dominates → surface Baritone `MineProcess` target-reachability (a
  "no path to any target candidate" signal, distinct from body movement) and
  bail on it. This is the real watchdog the movement proxy missed.
- If **(b)** dominates → NOT a watchdog problem: split **descend-to-ore-layer**
  from **mine** (descend as its own budgeted op, then mine with full clock), or
  raise the per-candidate budget for diamond specifically.

---

## Fix C — Tier-gate at mine_diamond/mine_iron — DONE & validated (2026-06-01)

**SHIPPED.** `craft/tools.py`: `_mine_tier_gate` pre-check on `handle_mine_iron`
(needs stone+ pickaxe) and `handle_mine_diamond` (needs iron+ pickaxe). When the
required pickaxe is absent it returns `SKIPPED <tool>: <redirect>` instead of
dispatching a doomed mine; fail-open on an inventory-read blip. env
`CRAFT_MINE_TIER_GATE` (default on; 0/false/no disables). Unit tests:
`tests/test_mine_tier_gate.py` (8/8, incl. the stone-satisfies-iron-not-diamond
boundary + kill-switch + fail-open). Live agent0: wooden→both SKIP, stone→diamond
still SKIP, gate-off→mine_diamond dispatches & fails no_progress (the doomed mine
the gate prevents). 395 mine/tool specs green.

(original scoping below)

## Fix C — Tier-gate at mine_diamond/mine_iron (scoping)

**The real qwen deep-mine residual.** The Fix-A wave exposed that qwen calls
`mine_diamond` (26×) and `mine_iron` without ever holding the required pickaxe
(0/8 reached iron tier). Those mines are doomed — diamond ore needs an iron+
pickaxe to drop, iron ore needs stone+ — so they burn wall-clock on
`no_progress`/`unreachable` with no possible payoff. The M1-iron milestone gate
is inert ([[project_milestone_gates_inert]]) so it doesn't steer.

**Fix:** a tier-gate at the `mine_diamond`/`mine_iron` tool boundary in
`craft/tools.py` — before dispatching the miner, check inventory for the
required pickaxe tier; if absent, refuse + redirect with a substrate message
("need an iron pickaxe to mine diamond — mine_iron then craft an iron pickaxe
first") instead of dispatching a doomed mine. Mirrors the existing pre-dispatch
guards (dusk shelter, night-craft lockout, death check). Kill-switch env.
Validate: a peaceful diamond wave should show the doomed-call wall (26 calls,
~big fraction) collapse, and ideally nudge qwen up the tier ladder.

Higher-leverage than the budget raise because it removes the doomed work
entirely rather than making it cheaper-but-still-doomed.

---

## Sequencing

1. Build **Fix B** + tunnel test + shelter/doorway regression — one rebuild.
2. In the same fleet session, capture the **Fix A diagnostic** tape (fleet is
   up anyway). Classify (a) vs (b).
3. Iterate Fix A's mechanism from the verdict; rebuild again.
4. Re-run the peaceful goal=diamond A/B (N≥20 each) to measure tier/diamond-rate
   lift and confirm friction stays low at depth. Consider larger N to tighten
   the 15%-vs-6% small-sample numbers.
