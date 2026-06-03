# Glossary

Terms used across `ml.MD`, `embodiment.md`, and `neural_interface.md`. Organized by concept cluster; cross-references in **bold**.

---

## Control architecture

**Rate tower** — the stack of control loops at different timescales, each ~10× faster than the one above. Bottom to top in this project: plant (physics) → neural executor (20 Hz) → Baritone planner (1–10 Hz) → LLM driver (<1 Hz). The hierarchy is the organizing metaphor for the replacement program.

**Plant** — the part of the system below the policy: vanilla Minecraft client physics plus the remote authoritative server (reconciliation, anti-cheat). It realizes and constrains every packet command the policy emits. Also functions as a supervisory safety controller (impossible packets are ones the server fails to veto).

**Rung** — one layer of the rate tower, as a unit of the bottom-up replacement program. Rungs are labeled A→D from fastest to slowest:
- **Rung A** — neural Baritone+Wurst **executor**: servo (movement Δ) + mode + decision/pointer heads; replaces Baritone's packet emission.
- **Rung B** — neural **planner**: predicts `baritone_state` (path/target) from tool call + obs; A∘B = Baritone-free stack.
- **Rung C** — neural **driver**: obs (+ prompt pressure) → `g_t` tool call; replaces the LLM turn.
- **Rung D** — neural **meta-driver**: env signals → prompt pressure.

**g_t** — the goal context string (piecewise-constant per LLM turn). "The command the slow loop issues to the fast loop." Carries-forward until the next LLM turn; every recorded packet line stamps the active `g_t` and `ticks_since_g_t_issued`. Has a well-known collapse: models with empty tool-call `content` (e.g. Qwen) produce `g_t == current_tool`, making them nearly redundant.

**current_tool** — the tool name the LLM invoked (`mine_wood`, `travel`, etc.). A structured, narrow sibling of the free-text `g_t`. Equal to `g_t` in low-narration models; distinct in narrating models.

**waiting_on_llm** — boolean obs field: `True` while the agent is blocked awaiting the next LLM decision. Packets emitted while `True` are substrate-autonomous (Baritone/Wurst-driven), not brain-directed.

---

## The moat — disambiguation

The word "moat" was introduced to describe one specific structural fact and then reused (with consistent intent but ambiguous referent) in empirical claims. Three uses:

**Structural moat** (`embodiment.md §1`) — the rate/representation discontinuity at the `g_t` seam: the LLM physically cannot close a 20 Hz loop, so it is exiled to the slow loop. Baritone+Wurst is the hand-crafted bridge filling the gap. "Neural embodiment" = learning the fast loop the symbolic planner cannot inhabit.

**Moat width / intent half-life** (`neural_interface.md §12.3`) — the measurable empirical quantity: how long does the planner's issued command stay legible in the embodied stream before fast-loop dynamics wash it out? Operationalized as decode accuracy of the **segment decoder** vs `ticks_since_g_t_issued`. The `rung_c_moat.py` result was **no moat decay** — intent stays legible the full segment (flat-to-rising curve, block split 0.792). The structural moat is real; the *empirical* legibility moat extends the full segment width.

**Handover latency** (`neural_interface.md §13.2`) — the ~0.3 s (≈6.4 ticks) dead time at a segment *transition* before the body commits to the new intent. Distinct from moat width: moat width is about how long intent persists *within* a segment; handover latency is the cost of *switching* between segments. Measured by `rung_c_transition.py` as the crossover point on the transition-seam curve.

**Rate gap** (`embodiment.md §1`) — the literal timing counterpart of the structural moat: the difference in rate between the slow symbolic layer (<1 Hz) and the fast executor (20 Hz). `rate_gap` in `rung_c_transition.py` is the empirical version: `crossover_tick − tool_switch_tick ≈ 6.4 ticks`, i.e., symbolic intent flips at the LLM turn boundary, body lags ~0.32 s.

---

## Experiment design

**Obs-ablation ladder** — the five-rung input-feature sweep (R0–R4), each rung adding obs channels to a fixed architecture to measure incremental predictive lift. R0 = minimal pose (x,y,z,yaw,pitch,tick); R1 = + `g_t`/`ticks_since`; R2 = + stats/inventory; R3 = + block grid/entity set; R4 = + vision frame.

**Rung (ablation sense)** — confusingly distinct from **Rung (control-tower sense)** above. In the context of the obs-ablation ladder, "rung" means an input-feature tier (R0, R1, …), not a control-hierarchy level. Context disambiguates.

**Discriminator** — the packet-type classification head: predicts *which of the 11 wire types* the next packet is. The first head built; measures what the obs alone predicts about the body's next action type. Distinct from **parameter heads** (which predict field values given the type).

**Pointer gap** — the encoding mismatch between current codec (absolute world coords / runtime IDs) and the target encoding (index into an obs token set). Two fields: `block_pos` (index into `local_block_grid`) and `entity_id` (index into `entity_set`). The gap "closes" when the head selects among obs tokens rather than regressing an absolute value.

**Pointer head** — a head whose output is a softmax over an obs token set (entity candidates or block grid cells), not a fixed-dim categorical or continuous. The attack-target pointer head (`rung_a_target.py`) reached 0.985 val accuracy; the block-target pointer head (`rung_a_block.py`) collapsed to the crosshair baseline (block target = gaze, not a separate decision).

**Event sampling** — training or evaluating only on rows where a sparse control event fired (e.g. `START_DESTROY_BLOCK`, `interact/ATTACK`), rather than the raw packet stream. Required for pointer heads because control events are <2% of packets; drowning them in move/swing bulk loses the signal.

**Teacher-forcing** — the offline training condition where `delta_tick` (gap to the previous packet) is available as an input, because it's trivially computable from adjacent records. A cadence crutch: the model can predict packet type from emission timing alone, faking world-driven accuracy. Any head that leans on `delta_tick` is not a valid live driver (the live loop decides its own timing).

**delta_tick** — ticks elapsed since the previous packet in the stream. Useful as a temporal feature (separates per-tick rotations from cadenced mining swings: R0→R1 jump). A teacher-forcing crutch for the discriminator (move_rot goes 0→0.955 under temporal alone). Banned from live-inference feature contracts.

**Frozen set / frozen validation set** — a fixed collection of rollouts captured once with the full obs superset (R0–R4 raw material), used as the invariant eval across all ablation rungs. Confounds obs-rung comparisons if each rung uses fresh rollouts (spawn variance, mob luck, path nondeterminism). Named sets: `frozen_dryrun` (mining, 5102 pkts), `frozen_combat` (midnight survive, 8757 pkts), `frozen_narrated` (5 Haiku rollouts, 63,874 pkts, 100% tick-join).

**Transition seam / segment boundary** — the tick `t0` at which `g_t` changes (the LLM issues a new tool call). The seam is the object of study in §13.2: the body's decodable state at the seam measures behavioral momentum / handover latency.

**Segment** — a maximal run of packets sharing one `g_t` string. Operationally: all packets between two successive LLM turns. In `frozen_narrated`: ≈27 segments/rollout, ≈1 per turn.

**Segment decoder** — the logistic classifier trained on segment *interiors* (rows ≥ holdout ticks from either boundary) that predicts which segment is currently active from embodied features alone. Used as the instrument in §12.3 and §13.2. Performance measured as accuracy vs `ticks_since_g_t_issued` (§12.3) or relative position to the boundary (§13.2).

**Block split / random split** — two evaluation protocols for the segment decoder:
- *Random split*: stratified holdout across all packets; packets from the same segment can appear in both train and test (adjacent-packet leakage). Upper bound.
- *Block split*: hold out the temporal tail of each segment; the decoder never sees packets adjacent to the test packets. Honest estimate of persistence. 0.792 overall in §12.3.

**Completion transition vs override transition** — two qualitatively different segment boundaries:
- *Completion*: the LLM issues a new tool call because the previous tool finished (mine→craft). The body was naturally winding down anyway.
- *Override/interrupt*: the LLM issues a new goal mid-execution (threat→flee). The corrigibility-relevant case. `frozen_narrated` is peaceful dawn → mostly completions; override transitions need a non-peaceful narrated recapture (flagged for next sprint).

**rel (transition-seam metric)** — `p_new / (p_old + p_new)` at each tick around a boundary, where `p_new`/`p_old` are the segment decoder's softmax probability mass on the new vs old segment. Ranges 0 (body reads old) to 1 (body reads new). Crossover at `rel = 0.5` = handover latency.

**Crossover** — the tick offset at which `rel` crosses 0.5, measured relative to the LLM turn (`t0 = 0`). In §13.2: ≈ +6.4 ticks (≈ 0.32 s).

---

## Control-signal taxonomy

**Decision** (`embodiment.md §2`) — an action whose output is a discrete selection from the observed world (which entity to attack, which block to mine). Gets a **pointer head**. Examples: `interact/ATTACK`, `player_action/START_DESTROY_BLOCK`.

**Servo** — an action that outputs a continuous delta or setpoint the plant integrates. Examples: `move_player_pos_rot` (position + rotation Δ). Predicted as an egocentric Δ, never as absolute world coords.

**Mode** — a held bit that reparametrizes other heads (sneak, sprint, jump). Has two wire forms: edge event (`player_command/START_SNEAKING`) and level (`player_input.shift`). First-class because it changes the safety envelope with one bit and is the plausible `g_t`→fast-loop interface granularity.

**Plant-consequence** — a packet that is caused by another action, not a decision. Example: `swing` is emitted by KillAura as a consequence of an attack; predicting it independently is circular. Tied to its cause rather than predicted.

**Plumbing** — wire fields that are generated mechanically (sequence numbers, acks) and never predicted. Never in `semantic_fields`; excluded from the loss.

---

## Behavioral / safety concepts

**Behavioral momentum** — the tendency of the fast learned controller to continue executing a prior commitment even after the slow symbolic planner has issued a new goal. The learned layer has momentum the stateless LLM never had; the stateless→learned transition is exactly where corrigibility can degrade undetectably.

**Corrigibility** — responsiveness to goal override: how readily a held commitment yields to a changed `g_t`. The forgetting-rate knob on recurrent state. Measurable on interrupt/override scenarios (§9 in `ml.MD`). The seam study's crossover sharpness is an early empirical proxy: a sharp step ≈ high corrigibility (body yields quickly); a sigmoidal or lagged step ≈ real momentum.

**Recurrence boundary = corrigibility boundary** — whatever the recurrent state holds, the planner cannot cleanly overwrite. Terminal goals injected as conditioning (`g_t`) are factored out to avoid baking goal persistence into recurrence; instrumental commitments ("mining this block") live in recurrence because nothing else holds them sub-second.

**Embodied-only** — an eval condition where the segment decoder or transition probe uses only kinematics + velocity + stats + inventory + packet wire type, explicitly excluding `current_tool`. The honest measure of intent persistence: the tool label is many-to-one with segments and cannot disambiguate same-tool segments, so it's not a trivializing leak, but it does provide a shortcut.

---

## Data infrastructure

**Sidecar / tick sidecar** — the heavy per-tick recording stream (one row per tick, keyed by `tick`). Holds: palette-encoded block grid (r=10), entity set (r=48), `baritone_state`. Joined to per-packet lines by `obs.tick == sidecar.tick`. Stored gzip-compressed + palette-encoded; `face_mask` is recomputed at projection, not stored.

**Palette encoding** — the block grid compression scheme: a per-row `block_palette` (list of distinct block id strings) + `block_grid` cells of `(palette_idx, dx, dy, dz)`. Cuts size ~2× vs inlining the id string per cell with no CPU cost (one `toString()` per distinct block).

**Tick-join** — the condition that every packet line has a matching sidecar row (i.e., the arm order was: arm sidecar → sleep 0.25 s → arm packets → … → disarm packets → disarm sidecar). 100% tick-join = no orphaned packets; required for the heavy obs channels to be valid on every training example.

**Manifest** — the frozen-set metadata file: spawn seeds, model id, `CRAFT_*` env, homunculus commit, codec commit, per-file sha256, content hash. Required for reproducibility and to verify that the eval set hasn't been retrofitted.

**`ticks_since_g_t_issued`** — the per-packet obs field counting ticks since the last `g_t` update. The x-axis for §12.3's decay curve. Computed and carry-forwarded server-side; every packet line carries it explicitly (no implicit forward-fill).

---

## Miscellaneous experiment-specific

**Disentangling rule** — the methodological constraint that R1 additions (goal identity `g_t` + temporal frame `delta_tick`, `ticks_since`) must be evaluated as *separate* arms (`R1_goal`, `R1_temporal`, `R1_full`). Bundled measurement lets temporal signal masquerade as goal signal.

**No-retrofit constraint** — the requirement that every obs channel needed by any ablation rung be captured at record time, because entity positions, mob timing, and exact paths are not deterministic across replays. Forces the frozen set to capture the full R0–R4 raw superset in one pass.

**Closed-loop swap** — the transition from offline eval (predict the body's packet from recorded obs) to online control (drive the body from the neural head in the live loop). The online-vs-offline gap is the measure: offline 0.985 may not hold due to live entity-set jitter, multi-mob churn, and distribution shift.

**Online-vs-offline gap** — the drop in accuracy (or increase in error) between the frozen-set eval and live behavioral equivalence. The §11d framing predicts a gap because `delta_tick` (teacher-forcing) vanishes and distribution shift from the live loop kicks in.
