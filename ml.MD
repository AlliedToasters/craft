# Minecraft Agentic Harness → World Model: Design Notes

*Distilled from a brainstorm. Reference doc, not a spec. Captures decisions, rationale, hypotheses, and sequencing.*

> **See also `embodiment.md`** — the fast-loop design layer (control-rate tower, action taxonomy, modes, recurrence=corrigibility boundary, `g_t` as authority interface, multi-goal). The control-theoretic vocabulary built on top of this doc.

---

## 0. One-line thesis

A rich-representation, bottom-up embodied learning problem that ships with its own competent teacher **and** its own world — the teacher (heuristic + LLM stack) and the environment (Minecraft) are the same system, so we get environment richness and clean supervision simultaneously. Minecraft as a tractable analog for the "hacked embodiment" question: how much of a hand-crafted robotic control stack can a neural system absorb, generalize past, and fuse with a language-model planner.

## 1. Current system (the teacher)

- High-level **LLM driver** (small Qwen instruct works) issues high-level goals / tool calls. Behaves like an executable walkthrough — correct on average (it's pattern-matching wiki/Reddit/guide knowledge from pretraining), brittle at the margins.
- **Heuristic execution layer**: forked Baritone (pathfinding/navigation) + kill aura + AutoTool + AutoEat + custom inventory manager + AutoDrop. These are the *exception handlers* for the cases the average-case walkthrough doesn't cover.
- Reaches **diamond tier from random spawns** at full survival difficulty (filtering hard biomes / open water). Robust; smarter LLMs do better.
- Architecture: Java mod arm + Python agentic harness, IPC between them (gRPC). Bespoke client (not SoulFire).
- Telemetry: screen capture + arbitrary client data across **20 parallel agents** → large-scale labeled rollout capture.

**Three-layer mental model:**

| Layer | Property |
|---|---|
| LLM planner | correct, general, sample-efficient (zero-shot), brittle at margins |
| Heuristic layer | correct within scope, hand-specified margins, no generalization |
| Learned layer (the goal) | open question — can it learn the residual the other two can't? |

The learned system targets the **residual**: as heuristic coverage grows, the residual shrinks and sharpens toward genuine edge cases. Ongoing heuristic work is therefore *also* sharpening the learning target.

## 2. Baritone fork — version decision

- Target: **1.21.4** — the last version Baritone supports. The harness, homunculus mod, Wurst integration, and the entire substrate are already built against 1.21.4 and that is the supported floor for the pathfinding layer the experiment depends on.
- Newer MC (1.21.9 "Copper Age" / 1.21.11 / 26.1+) is off the table for now: porting Baritone is the load-bearing cost, and without Baritone the heuristic execution layer this whole design assumes (§1) doesn't exist. Stay on 1.21.4 until either Baritone is officially ported forward or we accept the porting work as a discrete project.
- Practical consequence: copper armor/tools/golems aren't in scope yet. The agent's tech ladder caps at the 1.21.4 feature set (wood → stone → iron → diamond → netherite; no copper tier). Revisit if/when Baritone moves.

## 3. How Baritone actually controls the player

Baritone does **not** simulate input. It short-circuits, writing directly to the same fields/methods keyboard/mouse input would:

- Movement: sets `player.input` flags + `setYRot()`/`setXRot()` each tick.
- Block breaking / placing / interacting: calls `interactionManager` methods directly (`attackBlock`, `continueDestroyBlock`, `useItemOn`, etc.).
- Works synchronously on the game thread via a tick event listener (`TickHandler`) → integrates cleanly, no desync.

**Key downstream fact:** all these client-side methods ultimately emit **serverbound packets**. The player methods are just a human-analog convenience layer that generates a well-behaved *subset* of possible packets. "Impossible" mod behaviors (kill aura attacking through walls, placing blocks at range) bypass the player-analog layer and send packets directly. So the *true* action space is the serverbound packet set, not the player methods.

## 4. Action space / control surface (OUTPUT)

**Decision: capture and act at the serverbound packet layer.**

- One mixin on the outbound packet path (`Connection.send` / outbound channel) captures *everything*, mod-agnostic by construction. Every mod, however exotic, sends packets → captured for free.
- Philosophically consistent with the safety thesis: a human *could* act at the packet layer but in practice never would (their loop is built around look/reach/click). A neural system has no such attachment and can learn to exploit the packet layer — achieving the "impossible." The packet layer is where out-of-human-envelope behavior lives.
- The clientbound packet stream is a *bad* observation (sparse, delta-encoded, bursty) — don't learn from it; the client already reconstructs world state from it. (See §5.)

**Arbitration caveat:** today the mods implicitly arbitrate (last-writer-wins / load order). A neural policy writing final resolved control *replaces the whole arbitrated bundle*, not any single mod. The learning target is "emergent behavior of the entire mod ensemble."

### 4a. The codec / intervention layer

The neural output is **not** raw packet bytes. An inference scaffolding (codec) maps between packet space and a structured neural-action representation, both directions:

- **Type-then-parameters, autoregressive head** (à la AlphaStar's SC2 action space). Packet = tagged union: pick packet *type* (small categorical), then type-conditioned parameter heads.
- **Arguments are pointers into the observation**, not raw values:
  - Entity targets → attention/selection over the observed entity set (pointer net), not raw entity-ID regression.
  - Block positions → selection over observed local block grid / player-relative offset, not absolute coords.
  - Faces/hands/booleans → small categoricals.
  - **Sequence numbers / acks → generated mechanically at packet-construction; never predicted** (plumbing, not decisions).
- This couples action and observation through shared structure — the inductive bias that makes it learnable. "Attack nearest hostile" = attention over entity set, not "what does entity 4823 mean."
- The "impossible" wallhack attack is *the same head* selecting a target a fair-play raycast would have rejected — no special machinery. The affordance for emergence is built in; whether it's *used* is the result.

**Round-trip invariant:** `decode(encode(packet, obs_t), obs_t) ≈ packet` (modulo plumbing). Encode/decode are observation-conditioned (that's the formal statement of "actions are pointers into observations"). The encoder does double duty: it builds the **training target** (heuristic packets → structured action labels) and its inverse is the **inference output**. Same type by construction → zero train/inference representation skew.

**Codec is the contract between all parts and is independently testable with NO ML.** Test ladder:

1. **Static round-trip unit test** — recorded (packet, obs) pairs reproduce. *(Cheap, runnable today against existing diamond-tier rollouts — high info per effort.)*
2. **Live passthrough** — insert encode→decode into the live control path; confirm Baritone+killaura+autotool still play identically.
3. **Encodability audit** — every heuristic packet must encode against its tick observation. Failures localize **missing observation channels** → doubles as an observation-completeness checker.
4. **Then** train: predict `encode(packet, obs)` from `obs`; any failure now is a *learning* failure, not plumbing.

### 4b. Inventory scoping (first pass)

Defer learned inventory/menu manipulation; keep it fully automated. Rationale: spatial actions (move/place/break/interact) are pointers into spatial observation (well-conditioned). Inventory/crafting actions are abstract slot indices into transient type-dependent UIs, with long combinatorial sequences and sparse terminal reward — categorically harder, and precedent (AlphaStar UI combinatorics; MineRL/VPT scripted crafting) supports scoping it out.

- Cut the action space at the **real joint**: spatially-grounded manifold (neural) vs. abstract-index manifold (deferred to automation).
- **Tag every captured packet by origin stack** (spatial-control vs inventory-automation). Train codec only on spatial; let automation packets pass through as part of observed *environment dynamics*. Keeps the encodability audit clean and leaves an upgrade path (packets already captured + labeled for a future second action manifold).
- Still proves every core hypothesis — the emergence/safety story is a movement-and-interaction story.

## 5. Observation space (INPUT)

**Decision: structured world state + vision, with a principled two-sense decomposition.**

Two modalities with distinct *physical* roles (not arbitrary redundancy):

- **Structured local field ("supersense" / lidar-like):** dense, complete, geometrically precise, but radius-limited. No occlusion problem within radius; perfect near-field fidelity. Serves placement, mining, immediate collision, near-field interaction. Can "see" occluded adjacent blocks → superhuman in the near field.
- **Vision:** long range but projective, occluded, lossy, directional. Its *point* is sensing-at-distance (terrain, distant mobs, ridges). Human-like occludability at range.
- Overlap zone is fine/useful for learning correspondence; they earn their keep in non-overlapping regions.

**Consequences:**

- **Look angle becomes an interpretable active-sensing action.** Where the agent looks = where it queries distal info. Gaze trajectory is a readable, externally-visible record of attention/information-seeking — loggable as behavior (it's in the packet stream), no internal probing needed. Lets us ask: anticipatory vs reactive sensing? Does `g_t` change gaze ("hunt" vs "flee")? Emergent visual search strategies?
- **Superhuman info use becomes detectable by contrast:** acting on a block/entity never looked toward ⇒ provably using the non-visual channel.

### 5a. Radius — the superhuman-sensing knob

- Real tradeoff is **information vs SNR**, not "small easy / large noisy." Larger radius adds mostly low-value blocks → SNR falls; dense voxel grid cost scales ~r³ while relevant content scales slower.
- **Foveation:** high-res at reach horizon, coarsening outward — softens the cliff, mirrors vision's projective falloff (both senses degrade with distance alike).
- **Encoding determines whether radius is baked into params:**
  - Voxel grid → baked in (fixed shape, conv assumes it).
  - **Token-set (blocks/entities as tokens w/ relative position, attention-consumed) → radius is a RUNTIME FILTER.** Train at one radius, ablate others at inference; randomize radius during training so the model can't commit to one.
  - Token-set also matches the pointer action head (it already selects over entity/block tokens) → one move buys a runtime radius knob *and* action/observation consistency. **Lean hard this direction.**
- Principled radius ≈ interaction/reach horizon + small margin. Too large → subsumes vision, gaze interpretability dies. Too small → starves near-field tasks.

### 5b. Entity resolution asymmetry

- Near entities explicit in supersense; distant ones vision-only.
- Predicted emergent behaviors: "watching enemies" (gaze-tracking distant threats not in supersense); evading a threat group *just outside* the radius.
- **Deep version:** evading an out-of-sense, out-of-view threat requires **maintaining a belief** about unsensed entities in recurrent state (object permanence, seeded by an earlier glance). Not reactive — belief-state maintenance. Hunt for: does the policy act on threats acquired visually seconds ago and no longer visible? That's the world-model property surfacing as behavior.
- **CAVEAT (same family as overlay-leak):** do NOT bake threat-salience into the sensor. If the supersense preferentially resolves enemies / drops passive mobs, "watching enemies" isn't emergent — you built it in. Represent entities uniformly; let the *policy* learn salience. Don't put the answer in the input if the answer is what you want to claim emerged.

## 6. World model / teacher framing

Conditional dynamics model: `h_{t+1}, o_{t+1} = f(h_t, o_t, a_t, g_t)` where `g_t` = LLM goal context.

- Goal conditioning is the special sauce: tightens the action distribution ("move left while mining coal" ≠ "move left while fleeing creeper"), and provides free, collection-time intent labels (VPT had to *infer* intent via an IDM; we don't).
- **Better than VPT** on: ground-truth actions (from heuristics, no IDM), explicit goal labels, structured state alongside pixels. Worse on: raw behavioral scale/diversity → make rollout-generation diversity a deliberate concern.
- Frame fusion as a JEPA-adjacent, world-model-first approach (vs LLM-first agent work) — a position LeCun-camp critique would find more credible.

**Suggested first experiments (cheap → expensive):**

1. **Inverse dynamics model validation** — can a small net predict the heuristic action from two consecutive frames + `g_t`? Cheap; tells you the representation is rich enough for the forward model.
2. **Structured-state-delta prediction** before pixel prediction — tells you faster whether anything meaningful is being learned; pixels are expensive.
3. Full forward/world model; then latent-space planning conditioned on goal strings = a neural Baritone that can generalize past Baritone's hand-coded cost model.

### 6a. Baritone path as auxiliary TARGET (not input)

- Baritone's path overlay (green line + highlighted target blocks) encodes its A* reasoning in a learning-friendly form. Tempting to feed as input — **don't.**
- As **input** it's label leakage (the line *is* the next actions → model learns "follow the green line," absorbs no world knowledge; the "decode and see the path artifact" bonus becomes circular), AND it has a fatal **train/inference availability** problem: at inference Baritone's planner is replaced → no overlay exists to feed (keeping Baritone to generate it defeats the experiment).
- As **auxiliary prediction target** (predict path/next-waypoints from a *clean* observation) the asymmetry flips in your favor: available exactly when needed (training loss), not needed at inference (model produces its own). Forces internalizing the planning rather than copying. The emergent-path-in-representation result becomes *genuine*.
- **Mechanism / why it aids convergence:** dense structured target shapes the shared trunk toward traversability/geometry features the main tasks also need → representation learning via auxiliary task (UNREAL-style).
- **Honest nuance:** predicting Baritone's path teaches *Baritone's cost model/policy*, not ground-truth traversability → model inherits Baritone's blind spots; a model that fully internalizes it IS a neural Baritone (a good *floor* to exceed, not a ceiling). → weight the path head heavily early, **anneal** it down so main objectives pull the policy past Baritone.
- **Capture as structure, not pixels:** pull the planned path as `BetterBlockPos` waypoints from the Baritone API (we already have an API-hooked sidecar mod). Richer than the overlay, better-conditioned, matches the pointer action representation. Pixels already dominate disk/compute across 20 agents → don't render twin frames at collection scale.
- **Visual route, kept for interpretability/human intuition:** render the overlay **on-demand, post-hoc** from stored waypoints (keep the screen-space projection code). Lazy generation → near-zero collection cost, full debug-time visual access. *(If ever rendering paired frames: render scene once, overlay as a separate pass on a copy → provably identical base, no spurious nondeterminism; better, render the path as its own mask/distance-field channel, not green-on-world — Minecraft is full of green.)*

## 7. Safety framing

- Don't artificially constrain the agent to human-analog I/O — that constraint is arbitrary and may make alignment *harder*. The alignment-relevant action space is the *true* interface (packets), not its human-analog projection (look/reach/click). What matters is whether goals are pursued in aligned ways, not whether the sensor suite matches a human's.
- The architecture is a tight real-world analog already deployed in pieces: LLM planner = task decomposition; Baritone+mods = the existing robotic control stack (motion planners, sensor fusion); Minecraft = environment that doesn't care about your plans; learned layer = the next step that could subsume the hand-crafted layer.
- **The seam between LLM goal-spec and execution** is where the interesting failures live: goal misgeneralization, task-level reward hacking, corrigibility. Especially: a *stateless* LLM planner has no persistent task-preference; a *learned policy* has behavioral momentum → the stateless→learned transition is exactly where corrigibility can degrade undetectably.
- **What this setup can study that theory-only work can't:** run the full hybrid, vary how much the learned component controls, and *measure empirically* whether behavioral properties (decision quality, corrigibility, out-of-envelope exploitation) change as the learned share grows.
- **Strongest emergence result:** constrain the *teacher* to fair play (same anti-cheat plugins we know from running servers) so the dataset is human-envelope, but give the *policy* the full packet action space. Then any "impossible" behavior is *genuinely emergent* (discovered, not cloned). Fair-play plugins constrain the teacher, not the student's action space. Bonus: keeps the dataset interpretable.

## 8. Interpretability leads

The two-sense split creates a rare, tractable interp program:

- **Explicit channel as a labeling oracle for the latent channel.** Near-field entities have free ground-truth labels (type + relative position are literally in the structured input). Train a probe to decode "enemy at relative (dx,dy,dz)" from activations using only explicit near cases, then **test transfer to distant enemies present only in pixels.** Transfer ⇒ modality-invariant entity code (vision-derived and supersense-derived enemies share a subspace). Same epistemics as cross-distribution probe-transfer (Liars'-Bench-style: train one regime, measure generalization; transfer AUC = representational universality) — applied to perception instead of deception.
- **Boundary-crossing experiment (the one to run):** track a single entity's activation trajectory as it walks from vision-only range into the supersense radius. Smooth convergence onto the explicit-encoding manifold ⇒ unified object representation both sensors feed. Discontinuity at the boundary ⇒ two codes reconciled only downstream (maybe at the action head). Clean figure either way; negative result equally informative.
- **Methodological wrinkle:** a distant enemy only *has* a latent code if the model actually looked at it → experiment entangled with gaze; probe far-field code only in trajectories where gaze history made the entity visually available. Loops back to active sensing: gaze is both the interpretable action *and* the precondition for the representation existing.

## 9. Contrived scenarios (from existing integration tests) as curriculum + contrast factory

Existing integration tests seed contrived starts (e.g. given iron tools → does it seek diamond?; under hunger pressure → does it hunt vs build tools?). These are simultaneously **three things**:

- A curriculum of seeded initial states (low-variance, single-decision-isolating, short-horizon).
- Dense samples of specific transitions for the dynamics model (seed into the rare event directly).
- A **labeled decision set** (each scenario encodes the expected branch = ground-truth-correct behavior at a known decision point).

**Cautions & uses:**

- The labeled-decision property is gold and fragile: training on a scenario kills it as a test. Treat the **generator** as the asset; sample train and held-out eval from disjoint draws.
- **Decision-correctness = cheap high-signal eval** (e.g. fraction of hunger-pressure scenarios where it hunts), no reward model needed. Track this curve as the learned policy takes over from heuristics → the empirical handle on behavior-drift / corrigibility (§7).
- **Contrast-pair factory (interp):** minimal pairs in *scenario space* — identical scene, hunger high vs low; identical observation, `g_t` = "hunt" vs "mine." Any activation difference is attributable to the single varied factor (Arditi-style contrast, built in scenario space). Isolate "hunger" / "threat present" / "goal-intent" directions cleanly.
- **Keep the curriculum honest:** contrived scenarios are clean by design → training only on them converges to that manifold then eats distribution shift on real rollouts. Bootstrap/probe with contrived; generalize with open-ended. "Amount of contrivance" is itself a continuous curriculum axis (start maximally seeded/single-decision → randomize surroundings → add distractors → lengthen horizon → full distribution). Reverse-curriculum in spirit; sits alongside the other knobs (radius, fair-play, aux-loss weight).

## 10. Roadmap / sequencing

**Now (close the open survival threads — nearly done):**

- Finish shelter (working), hunting (in flight; sheep→wool reuses pathing+combat), cooking (≈ smelting, already solved), bedding.
- These aren't busywork: they roughly *double* behavioral coverage by enabling a **conservative strategic mode** (settle/sustain/gather low-tech) distinct from the **aggressive rush-to-diamond** mode. Different time-horizons, priorities, survival cycles → richer dynamics for the world model, and a genuinely informative `g_t` (same low-level action under different strategic intent → the variance that makes goal-conditioning interesting and probeable).
- Confirm the driver can actually express both modes. That's the natural transition point.

**In parallel, starting now (almost free, high info):**

- Codec **static round-trip unit test** (§4a step 1) against existing diamond-tier rollouts. Earns the pivot with evidence rather than novelty.

**Then pivot to the interface layer:**

- Codec + test ladder steps 1→4 (round-trip → live passthrough → encodability audit → train). Do this *before* nether/end — the encodability audit is far easier to debug against simple survival behavior than against nether navigation.

**Defer:**

- Nether/End. Recast later as deliberate **dataset-distribution expansion** for an already-validated pipeline — purpose-driven, not open-ended.

**Pivot discipline (re: the comfortable-endless-thing risk):** the actual risk isn't this pivot (it's justified independently by: literature stops at "get diamond," we have a robust teacher to that point, the interface is the novel load-bearing unproven piece). The risk is the *next* pivot — leaving the codec ~80% done when something newer looks shiny mid-hard-part. Defense: the test ladder gives discrete, finishable milestones with unambiguous completion criteria. Use them as commitment markers.

---

*Standing reminder of the recurring principle that showed up everywhere: don't put the answer in the input if the answer is the thing you want to claim emerged (overlay-as-input, threat-salience-in-sensor, training-on-eval-scenarios are all the same mistake). Controlled/privileged signal bootstraps; rich/clean signal generalizes; anneal between them.*   