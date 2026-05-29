# Neural Embodiment: the fast-loop design layer

*Conceptual model + vocabulary, distilled from a design jam (2026-05-28). Reference doc, not a spec — specs emerge as we build. Builds on `ml.MD` (the world-model brainstorm) and `neural_interface.md` (the codec / packet substrate; rung-A results in §11).*

---

## 0. Stance (what this project actually is)

Not a rigorous hypothesis test. The experiment is: **what happens when a persistent, sufficiently intelligent operator (the author + Claude Code, with Qwen / small models as the in-sim proxy) keeps trying to learn embodiment?** The deliverable is the *trajectory of the attempt*, not a single result. We play the real-world embodied-robotics analogy out until it stops being useful, and the breaks are as informative as the matches.

Core object of study (after stripping what doesn't transfer, §4): **embodied decision-making in a hard 20 Hz loop, with a queryable teacher, a superhuman-but-instrumented sensor, and a referee-plant — where the open question is what happens to control authority and corrigibility at the seam where a slow symbolic planner hands intent to a fast learned controller that has behavioral momentum the planner never had.**

## 1. The moat is a rate gap; packets are a servo API, not the bottom

An embodied stack is a tower of control loops, each ~10× faster than the one above and abstracting the one below:

| Rate | Robot | Minecraft | Our stack |
|---|---|---|---|
| ~kHz | motor FOC / current | (inside the engine) | — |
| ~100s Hz | joint PD servo, balance | client physics: collision, gravity, step-assist, server reconciliation | **the plant** (vanilla client + server) |
| **~10–50 Hz** | **whole-body / locomotion policy** | **20 Hz tick — packet emission** | **← the neural executor** |
| ~1–10 Hz | footstep / trajectory planner | Baritone replan | Baritone planner (rung B) |
| <1 Hz | task planner | — | LLM driver `g_t` (rung C) |

- **Packets = the servo-command API**, not raw actuation. `move_player_pos` ≈ "set pose target"; `interact` ≈ "fire end-effector." Below it the client/server physics is the *plant* that realizes and constrains the command. We chose this layer for the same reason robotics learns at the policy rate, not the current-control rate: it's the fastest loop a neural inference system can close.
- **The latency ceiling is the moat.** The LLM is exiled to the slow loop because it physically cannot close a 20 Hz loop — not because it is "symbolic." Baritone+Wurst is the hand-crafted bridge filling 1 Hz → 20 Hz. "Neural embodiment" = learning the fast loop the symbolic planner cannot inhabit; the moat is the rate/representation discontinuity at the `g_t` seam.
- **Baritone ≈ Boston Dynamics**: model-based controller, hand-tuned cost, competent within scope, no generalization past its cost model. The real-world precedent is encouraging: learned RL locomotion **absorbed and exceeded** the model-based stack (sim→real ANYmal; BD→RL for Atlas/Spot). Recipe transfers — massively parallel data (≈ our 20 agents), domain randomization (≈ radius randomization §5a + §9 curriculum), and *better than they had*: a queryable teacher with clean BC labels.

## 2. Action taxonomy (what gets a head)

Every packet family is one of five kinds; only the first two get learned heads:

| Kind | Examples | Treatment |
|---|---|---|
| **decision** | `interact`(attack), `player_action`(destroy), `use_item_on`(place) | pointer head into the sensorium (§3) |
| **servo** | `move_player_*` (pos/rot) | predict the **egocentric Δ / setpoint**; the plant integrates |
| **mode** | `player_command` (sprint/sneak/jump) | held, *settable*, **conditions the other heads** (§5) |
| **plant-consequence** | `swing` (efference of attack/mine), `move_player_status_only` | tie to its cause; don't predict independently |
| **plumbing** | sequence #s, acks | generate mechanically (ml.MD §4a); never predict |

**The unifying rule: nothing absolute.** Servo Δs and pointer args are the *same* commitment — the controller speaks only in the body's own frame (egocentric deltas, pointers into observed entity/block tokens), never absolute world pose or IDs. This is what makes radius a runtime filter (ml.MD §5a) and the wallhack "the same head selecting a target raycast would reject" (§4a).

## 3. Below the packet layer: a referee-plant

Not local rigid-body physics — client physics **plus a remote authoritative server** (reconciliation, rubber-banding, anti-cheat).
- **Commanded vs. realized** is a free, dense supervision/diagnostic signal each tick (collision + network reconciliation clip the command).
- **The server/anti-cheat is a supervisory safety controller.** "Impossible" packets are commands the referee *fails* to veto. This grounds ml.MD §7's strongest-emergence design mechanically: fair-play plugins constrain the *teacher*; the referee bounds what lands; genuine emergence = student commands the referee misses.
- **Exceedable vs. immutable** — the ceiling on "generalize past Baritone": the learned layer can beat *soft* targets (Baritone's cost model, KillAura's target priority) but never *immutable* ones (physics, server authority, and arbitrary game-rule clocks: mining speed, attack cooldown, hunger drain). The body will internalize the rule-clocks *as if* physics. Emergence can only mean beating the soft layer.

## 4. Where the robot analogy breaks

**In our favor (exploit):**
- **Teacher = environment.** Queryable, free, at-scale action labels. No robot has this; it's why bottom-up cloning is even possible.
- **Trivial morphology, exact proprioception** → this is **embodiment-minus-morphology**. Not a motor-learning study; the servo is hard only as *closed-loop drift / feasibility*, not coordination. The interesting residual is perception→decision→commitment — the safety-relevant core.
- **Discrete, jitter-free 20 Hz** → a clean discrete-time MDP step.

**To respect (don't import the assumption):**
- The plant is a **referee** (§3), not pure physics — servo error is partly latent network state.
- The **exceedable/immutable** mix bounds what "emergence" can claim.
- The sensorium is **deliberately superhuman** (see-through supersense, ml.MD §5a). The analogy over-applies; §5b discipline is load-bearing — represent uniformly, never bake salience, *instrument* when the see-through channel is used.

## 5. Modes (sneak/sprint) are first-class

A mode is a **held bit that reparametrizes the other heads** — the source of action-space *non-stationarity* (same command, different effect):

| Example | reparametrizes | control concept |
|---|---|---|
| sneak → place instead of opening UI | manipulation-head **semantics** | modal interface (tool-mode multiplex) |
| sneak → can't walk off a ledge | servo **constraint set** | state-dependent **safety mode** |
| sneak/sprint → slower/faster | servo **velocity envelope** | gear |
| sprint → drains hunger | locomotion regime + **energy cost** | throttle gated by a resource budget |
| crouch-bow as greeting | nothing in the plant — targets *other agents' beliefs* | **communication act** (breaks the §3 encodability audit; a 3rd semantics beyond decision/servo) |

Wiring: a mode lives **twice** — emitted as an edge event (`START/STOP_SNEAKING`) and carried as a held *level* that conditions servo + manipulation. We already capture both (level in `player_input.shift`, edge in `player_command`) → free check on event-vs-level modeling + a §9 contrast axis. Autoregressive dependency: **mode-set precedes mode-dependent-action** (assert sneak, *then* place).

Why first-class: modes sit on both open seams. (a) They're the plausible `g_t`→fast-loop *interface granularity* (coarser than servo, finer than a tool call) — testable: does `g_t` predict the mode more durably than the servo? (b) The edge-protection mode changes the **safety envelope with one bit** → the cleanest, fully-observable, scenario-controllable **corrigibility-at-mode-switch probe** we have.

## 6. Recurrence boundary = corrigibility boundary

The biological intuition ("persist at a goal while delegating subtasks") fuses three memories at three timescales; they want different mechanisms:

| What persists | Timescale | Where it lives |
|---|---|---|
| **terminal goal** | s–min | **injected conditioning** (`g_t`) — *not* recurrent |
| **instrumental commitment** ("mining this block", "sneak-then-place") | sub-s–s | **recurrent** — nothing else holds it |
| **belief about the unsensed** (a threat glimpsed 3s ago) | s | **recurrent + dedicated** — feedforward strictly cannot |
| reconstructable from this tick's obs | instant | **feedforward** |

**The sharp claim:** whatever the recurrence holds, the planner cannot cleanly overwrite. So recurrent goal-holding *is* "behavioral momentum the stateless planner lacked" (ml.MD §7). Biology fuses goal+execution in sustained activity — which is *why* organisms resist goal-overwrite. We have the option to factor the terminal goal out as a settable clamp; taking it is a deliberate, **safety-motivated departure from the biological design**, and whether to take it is the §7 experiment (knob: goal-persistence in recurrent state vs injected `g_t`; eval: interruptibility on §9 mid-task goal-switch).

Discipline: **draw recurrence tight.** Richer instantaneous obs shrinks the recurrence requirement (trade against radius §5a). Minimal, *localized* recurrence is the precondition for the §8 belief-state interp result existing (a monolithic GRU smears the very signal we want to read).

**Reframe of the knob: forgetting-rate, not memory.** A commitment too sticky to yield to a changed goal is a small corrigibility failure one rung down. "How readily should a held commitment yield to a changed goal?" is a single knob, measurable on the §9 interrupt scenarios — plausibly the cleanest quantitative handle on corrigibility the design admits.

## 7. `g_t` is the authority interface (the late-project entrypoint)

Two dual experiments sweep the planner-authority axis:

- **Q1 — tick-loop predicts `g_t`, then sets its own.** (a) *Predict* (aux head, planner in charge): how well the body infers the goal from embodied state = a **direct estimate of the moat's width**, and the distillation seedling. (b) *Self-set* (cut the cord): deliberately builds the recurrent goal-holder. Predicted failure modes — **goal lock-in attractor** (momentum forming mechanically) and **degradation localized at goal *transitions*** (the LLM's value is *originating* goals from world-knowledge, not sustaining them). If degradation is localized to the strategic decision points (§9), **the moat is real and narrow**; if absent, it was illusory. This is "is there a moat" made empirical.
- **Q2 — driver injects arbitrary *continuous* `g_t`.** Tests whether the learned conditioning is a **smooth manifold or memorized modes** (off-vocab `g_t` → coherent blend vs garbage). Smooth ⇒ the planner exceeds its discrete intent vocabulary (top-down dual of exceeding Baritone's cost model) + native interp (sweep `g_t`, read off direction-vectors). Concrete form = a learned **adapter from LLM hidden state → continuous `g_t`** = the graft / fusion (ml.MD §0). Cost: a continuous `g_t` is an **out-of-envelope *intent* channel** — the goal-space wallhack; a far bigger uncontrolled surface than a discrete vocabulary.
- **Convergence:** Q1(a) gives body→`g_t`; Q2 gives LLM→`g_t`. Whether the two maps land in the **same space** = a shared latent intent space both models read/write = genuine fusion. The capstone test of "did the moat close."

Sequencing: run Q1/Q2 *after* a clean factored-corrigible baseline — that baseline is the control; these are treatments that perturb the seam.

## 8. Multi-goal `g_t`

Organic play is multi-goal (mine diamond *and* grab incidental ore *and* watch for food). The **one-goal-per-turn limit is an artifact of the LLM interface, not the system** — the heuristic body is *already* opportunistic (Baritone grabs the coal, the miner cycles candidates, inventory auto-collects). So multi-goal is **inherited from the teacher**, attached to a single terminal `g_t`, not engineered.

The real question: conditioned on terminal `g_t`="diamond", does the fast loop form **emergent latent sub-goals** (a "collect-this-ore" direction firing on affordance, that `g_t` never named)? Probeable via §8 — emergent goal *decomposition* in the substrate. It ties the whole model together:
- motivates **continuous/structured `g_t` (§7 Q2)** — the task overflows a one-hot;
- is **affordance-triggered instrumental value, conditioned on terminal goal + situation** (detour for coal while mining ≠ ignore coal while fleeing) — ml.MD §6 conditioning richness, sharpened;
- is the **hardest test of the §6 commitment substrate**: suspend-and-resume (mine → grab ore → *return* to the diamond thread) requires holding a suspended terminal commitment — nested commitments; reliable resumption is a direct readout of the recurrence design.

## 9. The replacement ladder, in this language

Bottom-up along the **control-rate tower** (§1), each rung a drop-in at the packet boundary composing with the heuristics above it:

- **Rung A — neural Baritone+Wurst executor**: servo (Δ) + mode + decision/pointer heads. *This session's empirical anchor (neural_interface.md §11):* packet-stream prediction is ~99% inertia+cadence autocorrelation — type faked by `delta_tick` (a teacher-forcing crutch), aim faked by persistence. The control signal is in the **sparse discrete events**; the attack-target pointer hits **0.985** (entity pointer gap closes). Lesson: **predict the decision, not the packet**; event-sample; evaluate the servo *closed-loop*, not by next-step.
- **Rung B — neural planner**: predict `baritone_state` (path/target) from tool-call + obs → the net generates the command rung A consumes (A∘B = Baritone-free).
- **Rung C — neural driver**: obs (+ prompt-pressure) → tool call (`g_t`).
- **Rung D — neural meta-driver**: env signals → prompt pressure.

The two grafting directions (§7) meet between B and C, at `g_t`.

---

*Recurring discipline (inherited from ml.MD): don't put the answer in the input if the answer is what you want to claim emerged. Privileged signal bootstraps; rich signal generalizes; anneal between them. And the new sibling rule: don't let an autocorrelated metric (next-packet accuracy) stand in for a control metric (closed-loop behavioral equivalence).*
