# Neural Interface Spec

*The contract between observation and action for the learned policy. Living
design substrate — iterate in place. Sits between [`ml.MD`](ml.MD) (rationale,
hypotheses) and the codec implementation in [`craft/codec/`](craft/codec/)
(the executable bits).*

## 0. What this doc covers

- The **invariants** every codec must satisfy (the round-trip contract).
- The **observation space** the codec reads today + the channels needed to
  close the pointer gap.
- The **action space** as a tagged union over 11 wire types, with each
  parameter head's kind, cardinality, and `semantic_fields` rule.
- The **loss contract** that turns an `Action` instance into a per-head loss
  mask.
- The **training tuple** — how a line from `recorded.jsonl` becomes
  `(obs, action_label)`.
- The **pointer gap** — current absolute encoding, target pointer-over-channel
  encoding, and the migration path.

Out of scope: network architecture, optimizer choice, dataset assembly,
inventory/UI action space (deferred per ml.MD §4b).

## 1. Invariants (the contract)

The codec is the contract between the recorder, the policy, and the live
control path. Every per-type codec **must** satisfy:

1. **Round-trip identity.** `fields_close(decode(encode(p, obs), obs), p)`
   for every recorded packet `p` and its captured `obs` snapshot.
   Implementation: [`craft/codec/base.py:fields_close`](craft/codec/base.py).
   FP tolerance `atol=1e-9` (default); the live passthrough uses `atol=1e-6`
   for delta-cancelled position channels.

2. **`semantic_fields` exhaustiveness.** For an `Action` instance `a`, the
   set `a.semantic_fields` exactly enumerates the fields whose values the
   neural head predicts. The loss masks against this set; downstream
   consumers treat anything else as either plumbing or convention-zero
   filler.

3. **Plumbing exclusion.** Fields named in `Action._is_plumbing` are present
   for round-trip parity but are **never** in `semantic_fields`, **never**
   in the loss, and **never** predicted. They are filled mechanically at
   packet construction (sequence numbers, primarily).

4. **`__post_init__` enforces wire-shape consistency.** Constructing an
   `Action` with an action enum that requires a parameter, but with that
   parameter `None` (or vice versa), raises immediately. The policy cannot
   emit an inconsistent action; an inconsistent `Action` cannot exist.

5. **Observation as reference frame.** Encode and decode are
   observation-conditioned. The formal statement of "actions are pointers
   into observations" is: for a delta-encoded field `f` and obs `o`, encode
   produces `f_action = f_packet - o` and decode produces
   `f_packet = f_action + o`. The action carries the *relative* quantity;
   the absolute reconstruction needs the obs. Pointer fields (target state)
   will satisfy a stronger version: the action carries an *index into the
   obs token set*, and decode resolves the index back to the value the obs
   exposes.

6. **Strict registry.** `encode` and `decode` raise `KeyError` for
   unregistered packet types. An incomplete registry is loud, not silent.

## 2. Observation space

The obs is a `Mapping[str, Any]` passed to every encode/decode call. Two
states: what the codec reads today (minimal) and what it will read once
the pointer gap closes (minimal + 2 spatial channels).

### 2a. Current obs — minimal (codec already uses)

| Key | Type | Range / semantics |
|---|---|---|
| `tick` | `int` | Monotonic client tick at capture time. |
| `captured_at_ms` | `int` | Wall-clock ms (best-effort). |
| `x`, `y`, `z` | `float` | Player position, world coords. **Reference frame for all delta-encoded `move` pos fields.** |
| `yaw`, `pitch` | `float` | Player look, degrees. Yaw `[-180, 180]`, pitch `[-90, 90]`. |
| `on_ground` | `bool` | Standing-on-block flag at capture. |
| `dim` | `str` | Dimension id (`minecraft:overworld`, …). |

Nothing else is read by encode/decode today. The minimal contract is small
on purpose — every codec field needs an obs justification or it's not a
real pointer.

### 2b. Target obs — pointer-gap closure (2 channels added)

When these channels land, four codec fields swap from absolute encodings
to pointer encodings. They are described as obs channels here, not as code
shapes — the encoder/decoder API changes at the same time (see §6).

#### `local_block_grid` — pointer target for `block_pos`

Used by: `use_item_on.block_pos`, `player_action.block_pos`
(spatial-action variants only).

| Property | Value |
|---|---|
| Shape | `(K, 5)` — K tokens, each `(block_id, dx, dy, dz, face_mask)`. |
| Origin | Player feet (`obs.x`, `obs.y`, `obs.z`), integer-floored. |
| Selection | Every non-air block within an L∞ box of side `R_grid` (TBD; `R_grid = 8` is a reasonable first cut → up to ~512 tokens). |
| `block_id` | Categorical int over the block registry (or a learned embedding). |
| `dx, dy, dz` | Int offsets from origin; `[-R_grid, R_grid]`. |
| `face_mask` | 6-bit mask: which faces are exposed (non-occluded). Surfaces a `face` pointer head too — see §3. |
| Ordering | Stable per-tick; encoder and decoder must agree. Proposed: row-major `(dy, dz, dx)`. |

The codec's `block_pos` pointer is then an integer index `[0, K)`. Decode
recovers `(bx, by, bz) = (origin + grid[i][1:4])`.

#### `entity_set` — pointer target for `entity_id`

Used by: `interact.entity_id`, `player_command.entity_id`.

| Property | Value |
|---|---|
| Shape | `(M, F_ent)` — M nearby entities, each a feature vector. |
| Selection | Entities within `R_ent` blocks of the player (TBD; `R_ent = 32` first cut). Plus the player's own entity (index 0, pinned) — `player_command` often targets self. |
| Features (`F_ent`) | `(type_id, relative_x, relative_y, relative_z, yaw, pitch, on_ground_flag, hostile_flag, …)`. Concrete schema TBD. **No threat-salience field** — per ml.MD §5b, don't bake the answer into the sensor. |
| Stable id mapping | Encoder builds `entity_runtime_id → index` at obs construction; decoder reuses the same mapping. Captured in obs as `entity_index_by_id: dict[int, int]`. |
| Ordering | Stable per-tick. Proposed: player first, then by `(type_id, runtime_id)` ascending. |

The codec's `entity_id` pointer is then an integer index `[0, M)`. Decode
recovers `runtime_id = obs.entity_set[i].runtime_id`.

### 2c. Wider obs (policy-facing, not codec-facing)

The policy will read more channels than the codec does — health, hunger,
inventory, light level, vision frame, goal context `g_t`, etc. These do
not appear in the codec's obs argument because no codec field needs them
for round-trip. They live in the policy's input feature dict, alongside
the codec channels, but they are not part of *this* spec — they belong to
the policy spec when it gets written.

## 3. Action space

A neural action is one of 11 dataclasses, each a frozen, validated tagged
record. The discriminator is `packet_type: str` — the on-wire packet id,
checked against a closed set in `__post_init__`. The policy emits a
discriminator distribution + type-conditioned parameter bundles
(autoregressive: discriminator first, then heads for the picked type).

### 3a. Field kinds

Every parameter head falls into one of five kinds:

| Kind | Notation | Loss (§4) | Examples |
|---|---|---|---|
| **Categorical (N-way)** | `▪×N` | cross-entropy over N | `hand` (2), `face` (6), `action` enums (3 / 7 / 9). |
| **Boolean** | `▫` | binary cross-entropy | `on_ground`, `inside`, every `player_input` bit. |
| **Continuous (N-vec)** | `Nf` / `ΔNf` | MSE (delta-encoded for `Δ`) | `pos` (Δ3f), `cursor` (3f in [0,1]³), `rot` (2f abs, sin/cos features). |
| **Pointer** | `☆` | cross-entropy over obs tokens (target); MSE on absolute (current) | `block_pos`, `entity_id`. |
| **Plumbing** | `⟦x⟧` | none (excluded from loss) | `sequence`. |

### 3b. Per-type parameter bundles

The compact form. Each row: field name · kind · `semantic_fields`
participation rule (always / conditional on action / conditional on wire
type). Full validation rules live in the codec's `__post_init__`.

**`minecraft:move_player_pos_rot`** — `MoveAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `pos` | Δ3f against obs `(x,y,z)` | always (this wire type) |
| `rot` | 2f absolute, sin/cos at the head | always |
| `on_ground` | ▫ | always |
| `horizontal_collision` | ▫ | always |

**`minecraft:move_player_pos`** — `MoveAction` · same as above minus `rot`.

**`minecraft:move_player_rot`** — `MoveAction` · same as above minus `pos`.

**`minecraft:move_player_status_only`** — `MoveAction` · only
`on_ground` + `horizontal_collision`.

**`minecraft:swing`** — `SwingAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `hand` | ▪×2 (MAIN_HAND, OFF_HAND) | always |

**`minecraft:player_input`** — `PlayerInputAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `forward` `backward` `left` `right` `jump` `shift` `sprint` | ▫ × 7 | always |

**`minecraft:player_command`** — `PlayerCommandAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `entity_id` | ☆ (pointer gap — abs int today) | always |
| `action` | ▪×9 (PRESS_SHIFT_KEY, RELEASE_SHIFT_KEY, STOP_SLEEPING, START/STOP_SPRINTING, START/STOP_RIDING_JUMP, OPEN_INVENTORY, START_FALL_FLYING) | always |
| `data` | scalar int | always |

**`minecraft:use_item`** — `UseItemAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `hand` | ▪×2 | always |
| `yaw` | 1f abs | always |
| `pitch` | 1f abs | always |
| `sequence` | ⟦plumbing⟧ | never |

**`minecraft:use_item_on`** — `UseItemOnAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `hand` | ▪×2 | always |
| `block_pos` | ☆ (pointer gap — abs (int,int,int) today) | always |
| `face` | ▪×6 (DOWN, UP, NORTH, SOUTH, WEST, EAST) | always |
| `cursor` | 3f in [0,1]³ block-relative | always |
| `inside` | ▫ | always |
| `world_border_hit` | ▫ | always |
| `sequence` | ⟦plumbing⟧ | never |

**`minecraft:player_action`** — `PlayerActionAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `action` | ▪×7 (3 spatial: START_DESTROY_BLOCK, ABORT_DESTROY_BLOCK, STOP_DESTROY_BLOCK; 4 non-spatial: DROP_ALL_ITEMS, DROP_ITEM, RELEASE_USE_ITEM, SWAP_ITEM_WITH_OFFHAND) | always |
| `block_pos` | ☆ (pointer gap — abs today) | iff `is_spatial` (3 enum values) |
| `face` | ▪×6 | iff `is_spatial` |
| `sequence` | ⟦plumbing⟧ | never |

**`minecraft:interact`** — `InteractAction`

| Field | Kind | In `semantic_fields` |
|---|---|---|
| `entity_id` | ☆ (pointer gap — abs int today) | always |
| `action` | ▪×3 (ATTACK, INTERACT, INTERACT_AT) | always |
| `using_secondary_action` | ▫ | always |
| `hand` | ▪×2, optional | iff `action ∈ {INTERACT, INTERACT_AT}` |
| `at` | 3f entity-relative, optional | iff `action == INTERACT_AT` |

### 3c. `semantic_fields` — three conditional-fields patterns

Every conditional rule above resolves to one of three substrate patterns;
the head treats them uniformly via `semantic_fields`, the codec
representation differs:

| Wire behavior | Codec representation | Example |
|---|---|---|
| Field literally absent for some action values | `Optional[T]`; `__post_init__` matches presence to action | `interact.hand`, `interact.at` |
| Field always on wire; sentinel-zero for some action values | Non-optional; meaningfulness exposed via `semantic_fields` only | `player_action.block_pos` / `face` for non-spatial |
| Field always on wire; presence is a function of wire type | Wire-type discriminator + optional fields, both validated against type | `move.pos` / `move.rot` across the 4 wire types |

The first two are within-codec; the third is across-codec (the discriminator
is itself the conditioning). The loss spec (§4) only needs the
`semantic_fields` set — the carrying detail is substrate.

### 3d. Temporal frame

§3 defines what *one* action looks like. The temporal structure — how
actions sequence over time — is constrained but underdetermined at this
layer. Pin the baseline frame so the spec is testable; defer the
multi-packet-per-tick decision until live control is on the table.

**Frame: flat packet stream.** Each training example is a single captured
packet plus its tick obs snapshot. Sequence position is carried as a
`Δticks_since_last: int` feature on the obs (≥0; `0` means same-tick as
the previous packet). Tick boundaries are recoverable from
`Δticks_since_last > 0`. This is the form the next-packet-prediction
baseline (§7-resolution device) consumes.

**Multi-packet-per-tick: deferred, not precluded.** Real client traffic
emits ≥1 packet per tick (e.g. `swing` + `interact` for an attack); the
flat-stream frame encodes this as adjacent records with
`Δticks_since_last = 0`. The frame neither commits to nor rules out a
tick-grouped representation; the decision belongs to the policy spec when
live control is in scope.

**No-packet decision: out of scope today.** The 11-way discriminator
covers wire types only; there is no `NO_PACKET` class for the "this tick
emitted nothing" case. The next-packet baseline **conditions on packet
existence** — the dataset is packet-only and the question being answered
is "given an emit happens, what is it?" When the policy must decide
*whether* to emit on a given tick — live-control mode — a 12th
`NO_PACKET` discriminator class is the right shape, and the training
corpus must include tick-aligned negatives (one example per tick that
emitted nothing). Flag for §7 / closure when live control is on the
roadmap.

## 4. Loss contract

The loss is built per-instance from `Action.semantic_fields`. **Plumbing
fields never enter; non-fired conditional fields don't enter.** Per-head
losses, then a masked sum:

```
L(instance, prediction) =
    L_disc(p̂_type, instance.packet_type)              # always
  + Σ over f ∈ instance.semantic_fields:
        L_kind(f) (p̂_f, instance.f)
```

Per-kind loss:

| Kind | Loss |
|---|---|
| Discriminator (packet_type) | Cross-entropy over the 11 wire types. |
| Categorical N-way | Cross-entropy over N. |
| Boolean | BCE with logits. |
| Continuous (delta or absolute, non-angular) | MSE on the raw value. Delta encoding is invisible to the loss — it acts on the value the head emits, which is already in the policy's frame of reference. |
| Continuous angular (`yaw`, `pitch`, `rot`) | The head emits `(sin θ, cos θ)`; loss is MSE on the 2-vec, then `atan2` at decode. Avoids the `+359° == -1°` wrap pathology a direct-degree MSE would suffer. |
| Pointer (target state) | Cross-entropy over the obs token set (`local_block_grid` or `entity_set`). One head, masked-softmax over the actual K (or M) tokens at this tick. |
| Pointer (current — absolute encoding) | Treat as a continuous head: MSE on the integer `block_pos` triple, MSE on the integer `entity_id`. **Stopgap.** The encoding swap to true pointer is a loss-spec change, see §6. |
| Plumbing | None. Excluded by construction (not in `semantic_fields`). |

Discriminator-conditioned head masking: at **training** time the policy
emits all 11 type-bundles in parallel; only the heads for the picked type
contribute to the loss for this instance. Every head gets gradient on
every batch and the masking decides which contribute to *this instance's*
loss. At **inference** the 11× parameter-head forward cost is
unjustified: sample the discriminator first, then run heads for only the
picked bundle. The architecture is unchanged; the dispatch is lazy.
Different regime, same model.

`semantic_fields`-conditioned head masking: within a chosen type bundle,
some heads are gated by action-enum value (the conditional rules in §3b).
The gating function is `semantic_fields(instance)`; a head's loss
contribution is zeroed when its name is absent from the set. The head
**still emits a prediction** — it's the loss that zeroes — so the head's
parameters get no gradient on instances where its output didn't matter.
This is the same recipe as masked language modeling, applied per-head.

## 5. Training tuple

A training example is `(obs, action_label)`. The label is a concrete
`Action` subclass instance — *not* a flat tensor. The instance is the
source of truth; tensorization happens at the head boundary, where each
head reads the field it predicts.

### 5a. Derivation from a recorded JSONL line

`recorded.jsonl` (produced by `PacketRecorder` in homunculus, see
[`HomunculusHttpServer.java:69`](../homunculus/src/client/java/dev/toast/homunculus/HomunculusHttpServer.java))
has one line per captured outbound packet:

```json
{
  "packet_id": "minecraft:use_item_on",
  "captured_at_ms": 1716869341000,
  "tick": 4823,
  "fields": { "hand": "MAIN_HAND", "block_pos": [100, 63, -200], … },
  "obs": { "x": 100.5, "y": 64.0, "z": -200.25, "yaw": 45.0, … }
}
```

Derivation:

```python
from craft.codec import encode

action_label = encode(line["packet_id"], line["fields"], line["obs"])
# action_label is a concrete Action subclass instance.
# action_label.packet_type == line["packet_id"]
# action_label.semantic_fields is the mask for this instance.
```

`obs` for the forward pass is the same `line["obs"]` dict, possibly
augmented with the wider policy-facing channels (§2c). The
**codec-facing keys must match exactly** — the same `obs` that encoded the
label must be available at decode/inference time, or the round-trip
invariant breaks.

### 5b. Batching

Variable-shape labels (one of 11 dataclasses, sub-fields conditional)
don't batch into a single dense tensor. Per-head batching is the natural
shape: each head receives the *instances whose `semantic_fields` includes
its name* as a single batched call. The discriminator head receives
everything.

This is the same shape as masked-multi-task loss in image-text models —
the substrate detail (which head saw which instance) lives in a mask
tensor per head, derived once per batch from
`{i: instances[i].semantic_fields}`.

## 6. Pointer gap

Two codec fields are encoded absolute today and will swap to pointer-over-
obs-channel encoding when §2b lands. **The dataclass shape is designed so
the swap is a localized change.**

### 6a. Current state

| Field | Codec | Current encoding | obs channel needed |
|---|---|---|---|
| `block_pos` | `use_item_on`, `player_action` (spatial) | `tuple[int, int, int]` — absolute world coords | `local_block_grid` (§2b) |
| `entity_id` | `interact`, `player_command` | `int` — client-side runtime id | `entity_set` (§2b) |

Round-trip is satisfied today by carrying the absolute value through
encode → action → decode. The dataset is captured correctly; the gap is
purely on the *encoding* side.

### 6b. Target state

| Field | Target encoding | What decode does |
|---|---|---|
| `block_pos` | `block_idx: int` in `[0, K)` indexing `obs.local_block_grid` | `(bx, by, bz) = origin + grid[block_idx][1:4]` |
| `entity_id` | `entity_idx: int` in `[0, M)` indexing `obs.entity_set` | `runtime_id = obs.entity_set[entity_idx].runtime_id` |

The pointer is a softmax-over-tokens head at the same head boundary the
categorical heads sit at; the only difference is the head's output size
is *runtime-variable* (K or M for this tick) rather than fixed.

### 6c. Migration path

1. **Land `local_block_grid` and `entity_set` in obs** — homunculus
   produces them at the same time as the current minimal obs; the
   recorder serializes them into `obs` on each JSONL line. Both old
   (without channels) and new (with) JSONL lines exist for a window.
2. **Add an `encoding_mode` discriminator to the codec** — `"absolute"`
   (current) or `"pointer"` (target). Per-call. Default: pick the mode
   the obs supports. This keeps the round-trip invariant during the
   migration: lines captured without the channels still round-trip in
   absolute mode. **Inference-time invariant**: the mode is fixed per
   policy — whichever mode the policy was trained against is the only
   mode its decoder accepts. Mixed-mode training is fine (one model can
   learn either, given enough of each); mixed-mode inference per-call is
   a footgun (the pointer head's output shape depends on mode).
3. **Add `block_idx` / `entity_idx` to the dataclasses as
   `Optional[int]`** alongside the absolute fields. Encoder fills exactly
   one of the two per field; decoder reads the one that's set. The
   `_post_init__` check enforces "exactly one of `block_pos` or
   `block_idx`."
4. **Swap the loss head** for pointer fields from MSE (absolute) to
   masked cross-entropy (pointer). The `semantic_fields` set is
   unchanged — it still says "this field participates" — only the head's
   *kind* changed.
5. **Drop the absolute fields** once every active dataset has the obs
   channels. The optional becomes required; the dataclass shape becomes
   exactly the target.

Step 5 is a future cleanup; steps 1–4 are non-breaking.

### 6d. `face` is not a pointer gap

Six categorical labels are not a pointer — but the obs channel that
exposes them (the `face_mask` bits in `local_block_grid[i]`) is the same
channel the `block_pos` pointer attends over. The face head reads its
prediction off the *picked* block's face mask, which gives the policy a
single information source for "where is this block and which faces are
hittable." Documented here because it's the same channel; loss-wise
`face` stays a 6-way categorical.

## 7. Open questions / iteration substrate

The bin for things this doc names but hasn't resolved. Move items out as
they decide.

- **`R_grid`** (local block grid radius). Proposed `R_grid = 8` (≈ 4k
  tokens worst case; usually far fewer after air filtering). Tradeoff:
  pointer-target completeness vs token budget. Validate against the
  reach distances Baritone+Wurst actually use in captured rollouts. The
  frozen set captures raw at `R_capture_grid = 10` (§8e), so this can be
  swept in `[0, 10]` without re-recording.
- **`R_ent`** (entity set radius). Proposed `R_ent = 32`. Should it
  match supersense radius (ml.MD §5a)? Probably yes — the policy reads
  the same set the codec points into. Decide jointly with the supersense
  spec when that lands. The frozen set captures raw at
  `R_capture_ent = 48` (§8e), so this can be swept in `[0, 48]` without
  re-recording.
- **Entity feature schema** (`F_ent`). Type id encoding (raw → embed vs
  one-hot vs taxonomy), relative-pos quantization, what to do with
  player-vs-mob asymmetry. **No threat-salience field** (ml.MD §5b).
- **Block id encoding.** Raw registry int → embed table, or some
  factored scheme (material × variant). Affects pointer-head
  expressivity since the head conditions on the picked token's features.
- **Angular loss form.** `(sin, cos)` head + MSE is the proposed
  recipe; alternative is direct angular distance loss. Decide after the
  first IDM run on `move` packets, where the angular field dominates.
- **`data` field on `player_command`.** Currently a scalar int. For
  `START_RIDING_JUMP` it's a charge level; for most enum values it's
  zero. Should the head be a scalar regression, or an enum-conditioned
  multi-task head (regress for one enum value, ignore for the rest)?
  Same conditional-fields shape as elsewhere — almost certainly the
  latter.
- **Multi-packet ticks.** A single tick can emit multiple packets
  (e.g. `swing` + `interact` for an attack). The policy must be able
  to emit ≥1 packet per tick. Out of scope here; lives in the policy
  spec.
- **Where does `g_t` (goal context) enter?** Conditioning the
  discriminator, the parameter heads, or both. Likely both, but the
  spec for *how* belongs to the policy spec, not this one.

## 8. Next-packet-prediction baseline

The first training experiment is the §7-resolution device: predict
`encode(packet, obs)` from `obs`, report per-type metrics, sweep the
obs-ablation ladder. This section pins the experiment design decisions
that affect recorder schema and pre-registered hypotheses. Architecture
and optimizer choices are outside scope here.

### 8a. Goal context (`g_t`) — recorder schema

**Source: LLM-driven rollouts only.** `g_t` is the goal string the LLM
emits in its tool call (e.g. `"survive: build shelter before nightfall"`).
Heuristic-only data has no analogue. Rationale: deployment-coherent
definition — the policy will see goal strings at inference because the
LLM will issue them; training on Baritone task-state (a structurally
different signal the policy won't have at inference) contaminates the
ablation and produces a model with a train/inference representation skew.

**`g_t` is piecewise-constant over tick windows.** The LLM issues a tool
call, then 30+ ticks pass before the next. At most ticks `g_t` is "the
goal issued k ticks ago, still active." Duration is itself predictive —
packets right after a goal switch look different from packets deep into a
stable goal.

**Schema consequences for the recorder.** Two fields must be added to the
`obs` dict when recording LLM-driven rollouts:

| Field | Type | Semantics |
|---|---|---|
| `g_t` | `str` | The current goal string, carry-forwarded since the last LLM tool call. |
| `ticks_since_g_t_issued` | `int` | Ticks elapsed since `g_t` was last updated. `0` on the tick of a goal switch; increments each tick. |

The carry-forward is **explicit in serialization** — every recorded line
carries the active `g_t` and `ticks_since_g_t_issued`, not just lines
where `g_t` changed. Implicit forward-fill by the consumer is a footgun
(incomplete recordings, replay misalignment).

Lines from heuristic-only rollouts carry `g_t: null` and
`ticks_since_g_t_issued: null`. The obs-ablation's "`+ g_t` rung"
filters to non-null rows.

**Caveat (model-dependent `g_t` richness).** In practice `g_t` is the LLM's
free-text turn intent (the tool-call `content`) when present, else the tool
name. Measured on the first live Qwen-4B rollout: Qwen emits tool calls with
empty `content`, so `g_t` collapses to the tool name and is near-collinear
with the `current_tool` meta-observable (§8f). R1 still carries signal — tool
identity strongly predicts the packet mix (`mine_wood` → swings + look-rot;
`craft` → almost none) — but for low-`content` models `g_t` and `current_tool`
are mutually redundant, so an R0→R1 step cannot be attributed to *goal* intent
over *tool* identity. Disambiguating the two needs a model that narrates
intent in `content`, or a separate goal channel distinct from the tool. Report
the `content`-population rate per rollout set so this is visible, not assumed.

### 8b. Obs-ablation ladder

Five rungs. Each rung adds to the previous; the model's architecture is
held constant (MLP trunk + 11 type-conditioned heads); only the input
width changes.

| Rung | Obs channels | Infra status |
|---|---|---|
| R0 | §2a minimal (8 keys) | ready today |
| R1 | R0 + `g_t`, `ticks_since_g_t_issued` | ~1 day (schema plumbing into recorder) |
| R2 | R1 + stats (health, hunger, saturation, air) + inventory (slot → item type) | ~1 day (endpoint data exists) |
| R3 | R2 + `local_block_grid` + `entity_set` (§2b) | ~1 week (homunculus infra) |
| R4 | R3 + vision frame (RGB at capture tick) | ~1 week (Xvfb → JSONL integration) |

**Rung gating:** each rung's data collection can start when its infra is
ready; earlier rungs do not block later ones. Train on whatever rungs
have data. The baseline (R0 → R1) is the first gate to clear.

**Capture vs. train are decoupled.** The "Infra status" column above is the
gate on *training* each rung, not on *capturing* its raw material. Per §8e,
the frozen validation set captures the raw superset for all five rungs at
record time (the no-retrofit constraint forces this); a rung becomes
*trainable* when its projection + model-input code lands, which can lag the
capture by weeks. The infra estimates here are for the projection/training
side.

### 8c. Pre-registered (type × rung) predictions

The experiment is a 2D grid: `(wire_type, obs_rung)` cells that are
predicted to move vs stay flat. **Flat predictions matter as much as
step-change predictions** — an unexpected step where a flat was predicted
is a finding.

**Predicted step changes:**

| Wire type | Field(s) | Rung | Predicted signal |
|---|---|---|---|
| `interact` | discriminator (ATTACK vs INTERACT vs INTERACT_AT) | R0→R1 (+`g_t`) | Intent disambiguates action enum; LLM goal string should be strongly predictive of whether an interact is offensive or activating |
| `player_command` | discriminator (START/STOP_SPRINTING, PRESS_SHIFT_KEY, …) | R0→R1 (+`g_t`) | Sprint and sneak edges are goal-driven; goal switch → sprint change is a tight coupling |
| `use_item_on` | `block_pos`, `face`, `cursor` | R2→R3 (+§2b) | Block target is a literal pointer into `local_block_grid`; the pointer-gap closure is the step change. Expect near-zero improvement from R0–R2, then a large jump at R3 |
| `interact` | `entity_id` | R2→R3 (+§2b) | Same pointer-gap argument for entity targets |

**Predicted flat:**

| Wire type | Rung range | Reasoning |
|---|---|---|
| `move_*` | R0→R1 | Movement deltas are dictated by Baritone path-following, not LLM intent. Adding `g_t` should not move per-type accuracy. A measured step here would be a finding: goal context leaking into low-level control more than the architecture expects |
| `swing` | R0→R3 | Single-field packet; hand choice is nearly deterministic from inventory state. Should be high-accuracy at R0 and flat thereafter |

**Anchors (not hypotheses — calibration reference):**

| Wire type | Remark |
|---|---|
| `move_*` | Will dominate dataset by count; aggregate accuracy is mostly this. Report separately, don't let it swamp the per-type table |
| `player_input` | 7 independent booleans; expected high accuracy at R0 (keyboard state is near-deterministic from movement intent). Useful as a sanity check that the model trains at all |

### 8c-bis. First measurement (dry-run, mining regime) + the disentangling rule

The first R0→R1 discriminator run (3-goal Qwen capture, 5102 packets,
`experiments/next_packet/ablation_r0_r1.py`) surfaced a methodological trap and
a result.

**Disentangling rule (load-bearing).** R1 as specified (§8b) bundles *two*
additions: goal identity (`g_t`) **and** the temporal frame (`ticks_since`,
`delta_tick`). A bundled R0→R1 gain cannot be attributed to either. Always run
the arms separately — `R1_goal`, `R1_temporal`, `R1_full` — or the temporal
signal masquerades as a goal signal.

**Result (this regime only).** Overall val top-1: R0 0.465 → R1_goal 0.480
(**+0.014**) → R1_temporal 0.837 (**+0.371**) → R1_full 0.842. The temporal
frame drives essentially the entire gain; `g_t` adds ~nothing. Mechanism:
`move_player_rot` goes 0.000→0.955 under temporal alone — `delta_tick`
separates per-tick Baritone path rotation from cadenced mining swings. R0
collapses to the majority class (swing).

**Combat-regime test (the §8c prediction, now exercised).** A second capture —
`survive` goal, spawned at `midnight`, easy difficulty — generated the missing
types (interact 260, player_command 156, 7 distinct goals; 8757 packets,
`results/frozen_combat`). Discriminator ablation: R0 0.522 → R1_goal 0.545
(**+0.023**) → R1_temporal 0.775 (**+0.252**) → R1_full 0.781. **Same pattern:
`g_t` adds ~nothing; temporal dominates.** Critically, on `interact` itself:
R0 0.000 → R1_goal **0.000** → R1_temporal 0.116. **The goal channel does not
help predict interact at all.**

Mechanism — and the redirect: interact fires *during* `mine_wood`/`craft` (the
goal active when a mob wanders into KillAura range), so it is **mob-proximity-
driven, not goal-driven**. The substrate (Wurst KillAura) decouples combat
packets from LLM intent. So the predictive signal for interact is `entity_set`
(R3), **not** `g_t` (R1) — a concrete pre-registration flip for the R3 rung.

**Precision on scope.** This is the *discriminator* (predict that the next
packet is an interact). §8c's literal claim is about the interact *action-enum*
(ATTACK vs INTERACT vs INTERACT_AT) — a parameter head, not yet built. The
discriminator null + the KillAura mechanism make the enum claim unlikely to
hold, but it remains formally untested until heads land. Net: §8c's
goal-helps-discriminator prediction is **falsified in this substrate**; the
goal-helps-enum prediction is downgraded to unlikely. The plumbing — capture →
100% tick-join → rung-gated features → goal vocab → per-type metrics — is
end-to-end validated across two regimes.

**R1→R3 follow-up (the redirect, tested).** The mechanism predicted the
interact signal lives in `entity_set` (R3), not `g_t`. Tested on the combat
capture (`ablation_r1_r3.py`): entity features are threat-agnostic — a
histogram of entity *types* within 8 blocks + nearest distance + count (ml.MD
§5b, don't bake "hostile" into the sensor; the 26-type vocab is raw). On
interact: R0 0.000 → R1 0.140 → **R3 0.233 (+0.093 over R1)**. **Confirmed in
direction** — entity_set helps interact where the goal gave nothing. Two
nuances: (a) **modest magnitude** — interact co-emits with `swing` (KillAura
fires both on the same mob-contact), so "mob nearby" predicts the swing+interact
*cluster*, not interact-vs-swing cleanly; (b) **entity-alone is null**
(`ent_only` 0.000 on interact) — the entity channel is only usable layered on
the temporal frame. Overall R3 (0.786) barely beats R1 (0.781): entity helps
only the rare combat types, not the move/swing bulk the temporal frame already
owns. So the obs-lever ranking for this substrate's packet *types* is
**temporal ≫ entity > goal**.

### 8d. Per-type metrics

Report for every rung × type cell: top-1 accuracy on the discriminator,
per-head cross-entropy for categoricals/booleans, MSE for continuous
fields. Aggregate numbers are secondary — the per-type breakdown is the
primary artifact.

**Dataset balance:** `move_*` will represent >80% of packets. Do not
resample or re-weight for the baseline — measure the imbalance
empirically first, then decide whether per-type re-weighting changes the
results worth caring about. That decision is itself a §7 resolution.

### 8e. Frozen validation set + raw recording superset

The ablation ladder (§8b) is only an ablation if every rung is measured on
**identical eval packets**. The temptation is to add an obs channel, record
fresh rollouts, retrain, compare — but that confounds the rung delta with
spawn variance, mob encounters, and path luck. Instead: **freeze one
validation set, recorded once with a raw superset, and project it into each
rung's feature space later.**

**No-retrofit constraint (load-bearing).** The MC world seed is fixed
(`mc_wipe.sh` replays terrain), but entity positions, mob timing, and the
agent's exact path are *not* deterministic across replays. So heavy channels
(`local_block_grid`, `entity_set`, vision) cannot be added to a frozen set
after the fact — they must be captured at record time **or never exist for
that set**. Decision: capture **all five rungs (R0–R4) now**, so every
`(wire_type, obs_rung)` cell in §8c is measured on the same eval packets,
including the marquee R2→R3 pointer-gap steps.

**Set definition.** N LLM-driven Qwen rollouts. LLM-driven is required —
`g_t` and the meta-observables (§8f) have no heuristic analogue. Frozen
artifact carries a **manifest** (spawn seeds, model id, substrate config /
`CRAFT_*` env, homunculus commit, codec commit) and a **content hash** over
the recorded files. Stored **disjoint from training rollouts** (no
spawn-seed overlap) so the eval can't leak into training.

**Schema split by weight.** A 4900-cell block cube on every packet line
would duplicate across same-tick packets and stall the recorder (it runs on
the network thread). Split:

| Tier | Cadence | Holds | Join key |
|---|---|---|---|
| **Per-packet line** (light) | one per allowlisted packet | R0 pose + R1 `g_t`/`ticks_since` + R2 stats/inventory + meta-observables (§8f) | — |
| **Tick sidecar** (heavy) | one per tick | R3 raw (block cube + entity list) + R4 raw (vision frame ref) | `tick` |

Same-tick packets share one sidecar row. The per-packet line stays
network-thread-cheap and ships today; the sidecar is the new heavy capture.

**Capture raw, not encoded.** The sidecar stores raw material; the encoding
(`R_grid`, `R_ent`, block-id scheme, `F_ent`) is chosen at *projection*
time, not capture time. Capture radii deliberately **exceed** the proposed
encoding radii so they can be swept *downward* without re-recording:

| Sidecar field | Raw capture | Proposed encoding (§2b/§7) |
|---|---|---|
| block cube | L∞ radius `R_capture_grid = 10`, air-filtered, **palette-encoded**: a per-row `block_palette` (distinct ids) + `block_grid` of `(palette_idx, dx, dy, dz)` | `R_grid = 8` |
| entity list | radius `R_capture_ent = 48`, each `(runtime_id, type_id, abs x/y/z, vel x/y/z, yaw, pitch, on_ground, health, raw flags)` | `R_ent = 32` |

The block cube is palette-encoded rather than inlining the id string per cell:
a per-row `block_palette` of distinct ids + `block_grid` cells of
`(palette_idx, dx, dy, dz)`. On a CPU-bound host (GPU reserved for the policy
LLM, MC clients render on CPU) this was chosen over stream-gzip because it
cuts size with *negative* compute cost — one `getKey().toString()` per
distinct block instead of per cell — whereas gzip trades scarce CPU for
abundant disk.

**Palette factor: ~2×.** A real row: 3332 non-air cells, ~58 KB (~17.6 B/cell)
vs ~32 B/cell inline. The id string was only ~half the per-cell bytes; the
three coordinate ints + JSON punctuation are the rest and the palette can't
touch them.

**On top of palette: stream-gzip, opt-in at arm time.** The Phase-4 dry run
showed the real footprint (~64 KB/sidecar-row → ~383 MB per 5-min rollout
uncompressed → ~11 GB for a 30-rollout set), which crossed the "is it a
problem" line. The CPU objection that picked palette over gzip applies to
*fleet* rollouts (C~20); frozen capture runs only 1–3 agents, so gzip's
~3–5%/core is affordable there. The sidecar writer takes a `gzip` arm flag
(default off, so the channel stays cheap if armed mid-fleet); the
frozen-capture runner turns it on. **Measured 4.7×** (13.6 KB/row → ~82 MB
per 5-min rollout, ~2.4 GB for a 30-rollout set), `.gz` rows gunzip-inspectable.
A dense row-major positional array remains an unused lever (drops per-cell
offsets; wins when the cube is dense, loses when mostly air) — not needed at
the post-gzip footprint.

`face_mask` is **not** captured — it's a function of the block cube's
neighbors, recomputed at projection. Entity records carry **no
threat-salience field** (ml.MD §5b) — only raw kinematics + type.

**Vision (R4).** Frame grabbed off the agent's Xvfb display at tick cadence
(≤20fps), written as a **file-path reference** in the sidecar row, not
inlined. Storage is ~an order of magnitude over the block grid — acceptable
given the no-retrofit constraint and the full-capture decision. Pipeline:
the existing Xvfb frame grab (headless observability).

### 8f. Meta-observables (control-stack state)

Not world observations — the control stack's *internal* state. Captured
because (a) they may become first-class world-mod observables, and (b) they
sharply partition the packet stream for the ablation. Stamped onto the
per-packet line.

| Field | Type | Source | Semantics |
|---|---|---|---|
| `current_tool` | `str \| null` | agent.py | Tool the LLM is executing (`mine_wood`, `travel`, …). Structured sibling of the free-text `g_t`. |
| `current_tool_args` | `dict \| null` | agent.py | Args of the active tool call. |
| `waiting_on_llm` | `bool` | agent.py | True while the agent is blocked awaiting the next LLM decision. Packets emitted while true are **substrate-autonomous** (Baritone/Wurst), not brain-directed. |
| `baritone_state` | `obj \| null` | Baritone via homunculus | Current pathing goal/target + activity (pathing / mining / idle). The execution-layer intent behind most `move_*` packets. |

**Analytical value.** `waiting_on_llm` partitions every packet into
autonomous vs directed. This is the *mechanism* behind §8c's "`move_*` flat
across R0→R1" prediction: if move packets are emitted while `waiting_on_llm`
and explained by `baritone_state`, then `g_t` *shouldn't* help — and we can
show it directly instead of inferring it from a flat accuracy curve.
Conversely, a measured `g_t` step on packets emitted *while*
`waiting_on_llm` would be a finding (intent leaking into autonomous
control).

**Plumbing.** `current_tool` / `current_tool_args` / `waiting_on_llm` are
pushed from agent.py via `POST /obs/meta` (the same endpoint that carries
`g_t`); `PlayerObsSnapshot` stamps each tick's snapshot, carry-forward
automatic. `baritone_state` needs homunculus to read Baritone's
`PathingBehavior` — homunculus already drives Baritone, so the handle
exists; capture is best-effort (null when unavailable). These are
meta-observables **under evaluation**, not committed codec-facing channels
(§2c-adjacent) — they are not part of the round-trip contract.

## 9. Where to read next

- [`ml.MD`](ml.MD) — full design rationale, hypotheses, the world-model
  framing this spec serves.
- [`README.md`](README.md) §"Codec / neural output" — the at-a-glance
  diagram of the same hierarchy.
- [`craft/codec/`](craft/codec/) — per-codec implementations
  (`base.py`, `move.py`, `use_item_on.py`, …) and the HTTP shim
  (`server.py`).
- [`tests/test_codec_server.py`](tests/test_codec_server.py),
  per-codec round-trip tests — the executable form of §1's invariants.

## 10. Sprint state + where to pick up (2026-05-28)

The obs-ablation went from spec to a validated end-to-end pipeline with first
results across R0/R1/R3. This section is the handoff.

### 10a. What shipped

- **Capture infra** (homunculus): per-packet recording carries the obs
  superset (R0 pose + R2 stats/inventory + §8f meta); `/obs/meta` (agent-pushed
  `g_t`/`current_tool`/`waiting_on_llm`); `/obs/sidecar` heavy per-tick channel
  (palette block grid r=10, entity_set r=48, `baritone_state`), opt-in gzip.
- **Frozen-capture runner** (`experiments/next_packet/capture.py`): N rollouts,
  both streams armed (sidecar first / disarm last → 100% tick-join), manifest
  with commits + per-file sha256 + content hash + spawn biome.
- **Ablations**: `ablation_r0_r1.py` (goal vs temporal, disentangled),
  `ablation_r1_r3.py` (entity_set on interact). Discriminator only.
- **Findings** (§8c-bis): lever ranking **temporal ≫ entity > goal** for packet
  *type*; `g_t` falsified as a discriminator signal; entity_set confirmed for
  interact but modest (KillAura co-emits swing+interact).
- **Data on disk** (gitignored, kept for re-runs): `results/frozen_dryrun`
  (mining, 5102 pkts), `results/frozen_combat` (midnight survive, 8757 pkts).

### 10b. Pick up here (priority order)

1. **Parameter heads** — the model is discriminator-only, so §8c's *literal*
   claims are untested (they are all head-level). Build type-conditioned heads
   with `semantic_fields`-masked loss (§4). This is the gate for everything
   below. **Wiring note:** the ablations use `line["id"]` as the label and skip
   the codec entirely; heads need real labels — wire `craft.codec.encode(id,
   fields, obs)` into the dataset to get the `Action` and its `semantic_fields`.
   Start with two heads: `interact.action` (enum) and `use_item_on.block_pos`.
2. **block_pos pointer head over `block_grid`** — the **marquee unverified §8c
   prediction**: `use_item_on.block_pos` near-flat R0–R2 then a big jump at R3
   when it becomes a pointer into `block_grid`. The sidecar already captures the
   grid; `ablation_r1_r3.py` only wired `entity_set` — add a block-grid
   projection + a masked-softmax pointer head (§6b). This is the cleanest test
   of the pointer-gap thesis.
3. **Scale the frozen set** — pipeline is validated on 2–4 rollout dry runs.
   Capture a real N≥30 disjoint-seed set spanning regimes (dawn survival +
   midnight combat + mining), freeze it, hash it, keep it train-disjoint (§8e).
4. **Phase 3 — vision (R4)** — the last capture channel: Xvfb frame grab at tick
   cadence → file-path ref in the sidecar row (headless observability pipeline
   already exists).
5. **Give `g_t` a fair test** — Qwen's `g_t` collapses to the tool name (§8a
   caveat), so the goal channel was never richly exercised. Needs a model that
   narrates intent in tool-call `content`, or a distinct goal signal.

### 10c. Load-bearing gotchas for whoever picks up

- **Deploy = kill→cp→relaunch.** Never `cp` the homunculus jar over a running
  instance (corrupts lazy class loading). Kill agent N (`pkill -f
  "[h]omunculus.port=2557N"`), cp, relaunch via `./launch_agent.sh N`.
- **`TICK_COUNTER` is cumulative** over the client's life, not per-rollout — so
  sidecar/packet ticks keep climbing across rollouts on the same client.
- **Capture arm order**: sidecar first + 0.25 s sleep + packets, disarm packets
  first; otherwise the first packet of a rollout has no sidecar row.
- **Sidecar is gzipped + palette-encoded**; read with gzip + resolve
  `block_palette[idx]`. Entity records are raw (no threat field) — keep it that
  way (ml.MD §5b).
- **Disentangle obs groups** in every ablation (§8c-bis) — bundled rungs let
  temporal masquerade as goal.
- **Run with `.venv/bin/python`** (project venv; has `dotenv`, torch cu128).

## 11. Bottom-up replacement — the control hierarchy (2026-05-28)

Reframe (supersedes the §8 framing of next-packet prediction as the goal). The
craft controller is one aggregate heuristic system that *happens to have an LLM in
the loop*: **meta-driver** (prompt pressure from milestones / hunger / dusk) →
**LLM driver** (`g_t`, tool call) → **Baritone planner** (path / goal) →
**Baritone+Wurst executor** (path + mining-goal + mob-proximity → packets) →
**packets**. The research program is to replace this system with a neural one
*bottom-up along the control hierarchy* — not by swapping abstract single-tasks
(evade, mine_stone). Each learned rung is a drop-in at the same interface and
composes with the heuristic layers still above it.

Rung A = the neural **Baritone+Wurst executor**: given the command from above + the
executor's own state, emit the body output. Why the bottom first: data
availability *inverts with height* — the bottom layers are deterministic functions
we own (queryable for free, DAgger-able), the top (LLM) is stochastic and
non-queryable. And the sprint's sidecar already records the hierarchy's
intermediate state (`baritone_state`, `current_tool`, `entity_set`), so rung A is
already well-posed on the frozen data.

### 11a. The central finding: predict the *decision*, not the *packet*

Three offline heads on the frozen set (`results/frozen_dryrun` mining,
`results/frozen_combat` combat). The arc is the result:

1. **Type discriminator** (`rung_a_driver.py`): predict packet *type* from
   command vs executor-state. The **command/`g_t` adds ~nothing** (+0.01–0.02) —
   the hierarchy cut is real, the executor doesn't need the plan. But the headline
   0.86 is **faked by `delta_tick`** (the executor's emission *cadence*, a
   teacher-forcing crutch that vanishes when a net must decide *when* to act).
   World-state-only (pathing+proximity+pose, no timing) → ~0.52, barely above the
   pose baseline (0.47/0.52).
2. **Aim field head** (`rung_a_aim.py`): predict the yaw/pitch of each `*_rot`
   packet. **Persistence (echo current look) wins outright** — per-packet
   rotations are ~3–8°, so the stream is positional inertia. World-state makes it
   *worse* on the full stream. On the rare **re-target events** (|Δyaw|≥15°, ~1.3%
   of packets), world-state *does* help in combat (pose 81° → +prox 78° → +dt 73°;
   aim-at-nearest-mob oracle 57° vs persistence 86°) — but the signal is sparse
   and modest, and "aim at nearest" is a poor model (most rots are path-following).
3. **Attack-target pointer head** (`rung_a_target.py`): which entity does the
   executor (KillAura) strike? A per-candidate MLP scoring the `entity_set`,
   segment-softmax pointer. **0.985** (geom+type; geom-alone 0.954) vs baselines
   *nearest* 0.48, *nearest-hostile* 0.72. The **§6 entity pointer gap closes
   decisively**, threat-agnostic (the net learns which types matter).

**Conclusion.** The packet stream is ~99% inertia + cadence autocorrelation, ~1%
world-driven events. Next-packet prediction (type *or* continuous field) mostly
measures the stream's self-similarity, not control — type is faked by cadence,
aim by inertia. The genuinely world-driven, non-fakeable signal lives in the
**sparse discrete control events** (which block to break, which entity to attack)
— exactly the §6 pointer gap. **Rung A's correct objective is discrete
control-event prediction (pointer/enum heads), event-sampled, not the dense
move/rot/swing stream.** And at that granularity it *works*: a tiny net is a
near-perfect neural KillAura target-selector.

### 11b. Scripts (offline, on the frozen set)

| Script | Head | Result |
|---|---|---|
| `rung_a_driver.py` | packet-type discriminator (cmd vs exec, decomposed) | cmd ~0; lift is `delta_tick` cadence crutch |
| `rung_a_aim.py` | yaw/pitch regression (+ re-target subset) | persistence wins; world-state helps only on combat re-targets |
| `rung_a_target.py` | attack-target pointer over `entity_set` | **0.985** — entity pointer gap closes |
| `rung_a_block.py` | block-target pointer over `block_grid` (+ crosshair baseline) | **= crosshair (~0.93–1.0)** — block pointer *is* gaze, not a separate decision |

### 11c. Pick up here (priority order)

1. **block_pos pointer head** — the other half of §6: which block does
   `START_DESTROY_BLOCK` / `use_item_on` target, as a pointer into the sidecar
   `block_grid`. Same shape as `rung_a_target.py`, candidates = grid cells. Data is
   sparse in the current set (mining ~65 destroy events, place ~7) → **this is the
   recapture motivation**: a mining-heavy frozen set sampled at *event* granularity.
2. **Recapture for power + the path target.** (a) More combat for the entity
   pointer (n=260, val=65 is thin); (b) mining-heavy for block_pos; (c) enrich
   `BaritoneState.snapshot()` to expose the MineProcess/GoalProcess **path target**
   (currently `goal` is ~null because mining ≠ CustomGoalProcess) so movement heads
   get their proper input rather than the inertia/cadence proxy.
3. **Closed-loop swap** — the real predict→replace test. Drive the body from
   `rung_a_target` (a neural target-selector) and measure behavioral equivalence vs
   Wurst KillAura. The 0.86→0.52 (cadence-stripped) and persistence-wins gaps are
   the *offline* shadow of the distribution-shift the swap will expose.
4. **Climb to rung B** — predict `baritone_state` from tool-call + obs, so the net
   generates the command rung A consumes (A∘B = Baritone-free executor).

### 11d. Gotchas specific to rung A

- **`delta_tick` is teacher-forcing-only.** It encodes the gap to the *previous*
  packet — a closed-loop driver decides that gap, so it can't be an input. Any
  head that leans on it is not a driver. Prefer fields/pointers that need
  world-state (aim *value*, target *identity*).
- **Event-sample, don't packet-sample.** Destroy/attack events are <2% of packets;
  training on the raw stream drowns them. Filter to the event type first.
- **`interact.entity_id` ↔ `entity_set.runtime_id`** join is 100% in
  `frozen_combat` (tick-aligned). The label is the dist-sorted *index*; nearest is
  index 0, so the `nearest` baseline = P(label==0) = 0.48.

## 12. Next sprint — Measure the moat (2026-05-28 handoff)

Organizing question (per `embodiment.md` §0/§7): *where in the rate tower does the
symbolic layer actually reach?* This sprint produces the first number for it and
completes the §6 pointer story along the way. Three ordered tracks, each with an
unambiguous completion marker (ml.MD §10 pivot discipline). **By design this
sprint's scope ends at "plan the next sprint" — observe results, then update
approach; don't pre-plan further.**

### 12.0. Housekeeping (first)
Commit the rung-A session bundle (this §11 + `experiments/next_packet/rung_a_{driver,aim,target}.py`
+ the README update). Decide on `.vscode/`, `scratch.txt`, `scripts/bigN20_easy_*.sh`
(stage or gitignore). **Done:** working tree clean; `embodiment.md` §9's ref to
this §11 resolves.

### 12.1. `block_pos` pointer head — primary quick win (offline)
Build `rung_a_block.py`, mirror of `rung_a_target.py`: candidates = `block_grid`
cells (resolve the sidecar palette), label = the `START_DESTROY_BLOCK` /
`use_item_on` `block_pos` mapped to a grid index; per-candidate MLP →
segment-softmax pointer. Baselines: "block in the crosshair" (raycast from pose),
"nearest targetable face."
- *Why:* the entity pointer closed at 0.985 (§11a); this is the missing half of
  the §6 pointer gap and the marquee unverified §8c prediction.
- *Risk:* sparse (~290 destroy events across both frozen regimes).
- **Done (2026-05-28):** `rung_a_block.py` shipped. Candidates = *targetable* grid
  cells (exposed face, within `--reach 6` of the eye); label = `block_pos−origin`
  cell (100% join in both regimes); per-candidate MLP → segment-softmax. Result on
  the combined frozen set (172 events, val 43, seed-stable): `nearest` **0.15**,
  **crosshair raycast 0.93–1.0**, learned `geom` **= crosshair**, `geom+type` adds
  ~0. **Verdict — the block pointer is NOT a separate decision; it collapses to
  gaze.** This is the *mirror image* of the entity pointer: there gaze does NOT
  track the target (KillAura auto-aims server-side, so `nearest`=0.48 and a real
  head was needed to hit 0.985); here you must *look at* a block to mine it, so the
  crosshair perfectly determines `block_pos` and the learned head merely re-derives
  it. **The asymmetry is the finding:** block-target selection lives in the
  *servo/aim* channel, entity-target selection in the *discrete-decision* channel.
  Not data-starved — the verdict is conclusive at this n. The real block control
  signal is therefore *upstream* (what made Baritone aim there = the MineProcess
  path target), which is exactly what Track 2 exposes → 12.1 hands off cleanly to
  12.2.

### 12.2. Targeted recapture — the enabler (homunculus change)
One capture run that unblocks the rest. Three changes:
1. **Enrich `BaritoneState.snapshot()`** to expose the `MineProcess`/`GoalProcess`
   **path target** (`goal` is ~null today because mining ≠ CustomGoalProcess) →
   the servo head's proper input instead of the `pathing` bool.
2. **Mining-heavy, event-dense** rollouts → enough `block_pos` events to make 12.1
   conclusive.
3. **A content-narrating-intent arm** — Haiku, or Qwen prompted to narrate intent
   in `content` separate from the tool call — so `g_t ≠ current_tool` (breaks the
   §8a collapse). *Prerequisite for 12.3.*
- *Gotchas (§10c):* deploy = kill→cp→relaunch (never over a running jar); arm
  sidecar→sleep 0.25s→packets; `TICK_COUNTER` cumulative; `.venv/bin/python`.
- **Code landed (2026-05-29):**
  - *Change 1* — `BaritoneState.snapshot()` enriched (homunculus, compiles clean
    vs baritone-api-1.13.1). `goal` now reads `pb.getGoal()` (non-null during
    mining; the old `getCustomGoalProcess().getGoal()` was the null source) and new
    fields `path_dest` / `path_next` (the immediate waypoint = `positions[idx+1]`,
    the servo setpoint) / `path_len` / `mine_active` / `ticks_to_goal`. Old
    `pathing`/`goal_active` kept for back-compat; every read individually guarded.
    *Effect requires build → deploy (kill→cp→relaunch).*
  - *Change 3* — `--narrate` flag on `craft.agent` (+ threaded through
    `capture.py`, recorded in the manifest). Appends `NARRATE_SUFFIX` (overrides
    "leave content empty"); since the loop already stamps `g_t = (content or name)`
    (agent.py ~L1483), narrated content *is* `g_t` → distinct from `current_tool`.
    No recorder/obs-schema change was needed. Pair with Haiku/Sonnet (reliable
    text+tool-call); Qwen risky.
  - *Change 2* is pure capture config (`--goal diamond` + more turns + `--narrate`).
- **Done — live run (2026-05-29):** rebuilt jar deployed (kill→cp→relaunch on
  agent0, llvmpipe), then a frozen set captured →
  `results/frozen_narrated` (gitignored): **5 rollouts × 25 turns, Haiku,
  `--goal diamond --narrate`, peaceful/dawn**, all full 25 turns, zero deaths.
  **63,874 packets · 51,739 sidecar rows · 100% tick-join (all 5).** All three
  markers green:
  - *Change 1 (path target):* `baritone_state` present on every sidecar row;
    `goal` is now `GoalComposite[...]` during mining (was null);
    **`path_next` non-null on 25,523 ticks (49%)** = the servo setpoint as
    absolute `[x,y,z]` (downstream → egocentric Δ). `ticks_to_goal` stays null
    under `MineProcess` (`estimatedTicksToGoal` empty for composite goals) — not
    blocking.
  - *Change 3 (narration):* **125 distinct `g_t` intent clauses · `g_t ≠
    current_tool` on 97%** of g_t-bearing packets (the 3% equal = turns Haiku left
    content empty → fell back to tool name). `g_t` and `current_tool` are separate
    obs fields — intent/tool divergence recorded per packet, ready for 12.3.
- **Capture-dir reuse bug (found + fixed 2026-05-29):** the first 5-run reused the
  verify run's `rollout-0/1` dirs and their `packets.jsonl` came back at 49–57%
  join — `PacketRecorder` armed in `APPEND` while the gzip sidecar truncates, so
  stale prior-run packets were *prepended* (invisible to a tick-sort: TICK_COUNTER
  is cumulative, so the stale prefix is monotonic). Salvaged losslessly by dropping
  the sub-`sidecar_min` tick prefix (dropped counts matched the verify disarms
  exactly: 14467 + 8172) → all 5 now 100% join; originals kept as
  `packets.jsonl.contaminated`. **Durable fix:** both `PacketRecorder` and the
  non-gzip `TickSidecarRecorder` path now arm in `TRUNCATE_EXISTING` — "arm" always
  means a fresh file. (`manifest.json` `content_hash` predates the salvage — stale.)

### 12.3. Intent half-life / moat-width — the headline
On the narrated recapture: train a decoder `g_t ← (obs window)` and plot **decode
accuracy vs ticks-since-last-tool-call**. The decay is the moat width — how long
the planner's command stays legible in the embodied stream before the fast loop's
dynamics wash it out.
- *Why it needs 12.2:* on current data `g_t == current_tool`, so decoding is
  trivially "what is the body doing" — it measures tool *duration*, not intent
  *persistence*. Narrated intent separates them.
- **Done (2026-05-29) — `rung_c_moat.py`, framing (a) within-rollout segment
  recovery.** Per rollout, a *segment* = a maximal run of packets sharing one `g_t`
  string (27 segments/rollout, ≈1 per LLM turn). A torch multinomial-logistic head
  (linear softmax, embodied features only — kinematics + velocity + stats +
  inventory + wire packet-type; `current_tool` *excluded* by default) recovers WHICH
  segment is active; accuracy is binned by `ticks_since_g_t_issued`. Two splits:
  - *random* (stratified holdout, leaky upper bound): **overall 0.960** (chance
    0.037); curve **flat-high**, lift +0.86→+0.96 across all 12 tick bins, freshest
    bin only slightly lower (0.898). Shape: no decay, slight rise.
  - *block* (hold out each segment's temporal tail — honest, no adjacent-packet
    leakage): **overall 0.792, seed-stable** (0.794 @ seed1). Curve climbs 0.44 at
    the freshest bin (0–116 ticks) to a ~0.75–0.91 plateau through the bulk (out to
    ~1400 ticks ≈ 70 s); `READ: RISES`, pearson(lift, ticks)=+0.27.
  - *ablation* (block, `--with-tool`): **0.894**, +0.10 over no-tool — and
    `current_tool` is **many-to-one** with segment (rollout-0: 5/11 tools span ≥2
    segments; `craft` alone spans 10), so the tool label is *not* a trivializing
    leak — it cannot disambiguate same-tool segments. The embodied state already
    carries ~the intent on its own (0.79 without it). NB the with-tool curve reads
    `DECAYS` (fresh +0.89 → late +0.68): the tool one-hot pins the segment hardest
    right after issuance and blurs as the same tool recurs later — the *opposite* of
    the no-tool curve, and further evidence the embodied-only signal is the honest
    measure of persistence.
- **The headline — NO MOAT DECAY (the pre-registered "flat-high = finding"
  branch).** Intent legibility does not wash out over the inter-turn interval; it is
  flat (random) to rising (block) across the *entire* segment lifetime. The dip at
  the freshest block-split ticks is a **transition/length artifact**, not decay: the
  freshest held-out packets come from the shortest, most ambiguous transitional
  segments *and* from the post-issuance motor transient. Mechanistically: between
  LLM turns the substrate (Baritone/Wurst) deterministically executes exactly the
  issued command, so the body is a near-stationary readout of the active intent for
  the whole segment. **The symbolic layer reaches the full depth of the rate tower
  within an intent's lifetime — the moat is the entire segment width.** Extends
  rung A: there the *decision* (tool, attack-target) is decodable; here it stays
  decodable for as long as it is the active command. The planner's authority is not
  eroded by fast-loop dynamics within a turn.
- *Refinement flagged:* the block split's fresh-tick bins are segment-length-
  confounded (a test packet at tick-bin t came from a segment ≥ ~t long). A
  per-segment *within-segment* decay curve (decode at start→end of each individual
  segment, a different split that can see each segment's head) would isolate pure
  persistence from the length confound. The current two splits already bound the
  answer (flat-to-rising, no decay); this would sharpen the fresh-tick shape.
- *Run:* `.venv/bin/python -m experiments.next_packet.rung_c_moat --split block
  --bins 12 --out <csv>` (add `--split random` / `--with-tool` / `--seed N`).

**▶ §12.3 DONE (2026-05-29) — `rung_c_moat.py` shipped, headline = no moat decay
(see the "Done"/"headline" bullets above). The reference notes below are retained
as the dataset/codec guide for the flagged within-segment refinement and any future
work on the narrated set.**
- *Data:* `results/frozen_narrated/` (gitignored, ~1 GB), 5 rollouts, 100% join.
  Per rollout: `packets.jsonl` (light per-tick obs) + `sidecar.jsonl.gz` (heavy:
  `block_grid`, `entity_set`, `baritone_state`). Join packets↔sidecar by
  `obs.tick == sidecar.tick`. **NB rollout-0/1 also have a `.contaminated` backup**
  (the pre-salvage file) — ignore it; the live `packets.jsonl` is clean.
- *Obs fields you need* (in `packets.jsonl` `obs`): `g_t` (free-text intent),
  `current_tool`, **`ticks_since_g_t_issued`** (← this IS the x-axis, already
  computed server-side), `x/y/z/yaw/pitch`, `inventory`, `stats`. From the sidecar
  `baritone_state`: `path_next`/`path_dest` (servo setpoint), `goal`, `mine_active`.
- *The non-obvious fork — `g_t` target representation.* `g_t` is natural language
  with **125 distinct clauses over ~125 LLM turns (≈1 example per unique string
  globally)** → a global 125-way string-classify is degenerate. Don't do that.
  Options, cleanest first: **(a)** frame as *within-rollout segment recovery* — at
  each tick the active intent = the clause issued at the last turn; decode WHICH of
  that rollout's ~25 segments is active from the obs window, accuracy vs
  `ticks_since_g_t_issued`. **(b)** embed each `g_t` (sentence-transformer or a
  cheap local embed), decode the embedding from obs and score by
  cosine/nearest-clause — measures *semantic* legibility, smoother target. **(c)**
  cluster the 125 clauses into a handful of intent categories (mine-stone /
  descend-for-iron / craft / travel-relocate …) and classify those. Decide this
  first; it sets everything downstream.
- *Template:* mirror `experiments/next_packet/rung_a_target.py` / `rung_a_block.py`
  (per-event tensors, z-score geom dims, raw one-hot type, small MLP). The x-axis
  binning (ticks-since-issued) is the new structural piece.
- *Hypothesis (from rung A):* decode-acc starts high right after a tool call and
  decays as the fast loop's inertia/cadence washes the command out; the half-life
  is the moat width. A *flat-high* curve would mean intent stays legible
  indefinitely (symbolic layer reaches deep) — a finding, not a failure.

### Deferred (presuppose a trained executor we don't have)
Closed-loop swap (needs the packet-injection path + a servo); emergent sub-goal
probe (`embodiment.md` §8 — needs a recurrent executor to probe); continuous-`g_t`
graft (`embodiment.md` §7 Q2).

**If time is tight:** 12.1 + the narrated arm of 12.2 alone still moves the ball.

## 13. Next sprint — Close the loop (2026-05-29 plan)

Organizing question (the pivot of the whole program): *§11–§12 proved the control
signal is decodable offline and that planner intent stays legible the full segment
width (no moat decay). Everything so far is **read-only** — we have decoded that the
signals are there. This sprint crosses from **decoding to generating**: put a decoded
decision back into the live 20 Hz loop and measure whether 0.985-offline survives
contact with the body.* The closed-loop swap is the gate to every rung above A; it is
also the first demonstration that *feels* like embodiment rather than analysis, so it
is the **headliner**. The transition-seam study (the last high-value offline wedge)
runs warm and parallel as a lower-stakes thread.

**Scope discipline (ml.MD §10):** this sprint ends at "the attack-target selector
runs in the live loop and we have its online-vs-offline gap number." Block-mining
swap, rung B, and the seam-study refinements are explicitly *out* — observe the gap,
then plan the next sprint. Do not pre-plan past the first closed loop.

### 13.0. Why the attack-target decision is the cheapest *real* swap
This falls straight out of the rung-A asymmetry (§12.1). **The attack-target decision
is gaze-independent** — KillAura auto-aims server-side, so selecting *which* entity to
strike is a pure discrete decision that does **not** require owning the servo. The
block-target decision is the opposite (it *collapses to gaze*, §12.1), so a mining
swap would need a learned servo to aim. **Therefore: swap the decision that doesn't
need a servo first.** rung_a_target (0.985, §11a) is exactly that head. This is the
minimal predict→replace test — it isolates *one decision head in the live loop* with
no servo, no planner, no recurrence in the way. Block-mining (servo-coupled) is
deliberately deferred to a later sprint.

### 13.1. The closed-loop swap — HEADLINER
Replace Wurst KillAura's **target selection** with the neural pointer head; KillAura
(or homunculus) still does the aim+attack servo. Measure behavioral equivalence vs
stock KillAura in a controlled mob arena. Subtasks, cheapest-first:

1. **Train → checkpoint the head (offline, prerequisite).** rung_a_target.py is
   eval-only today (no `torch.save`). Add a train-and-persist path that freezes a
   feature spec using **only fields readable at live inference time** (no
   `delta_tick`, no teacher-forced cadence — §11d). Checkpoint = weights + the exact
   `EntityVocab` + the candidate-feature contract. *Done:* a `.pt` + feature-spec
   JSON that a separate process can load and score a live `entity_set` with.
2. **Spike: the injection path (decides the architecture).** Two options, pick by a
   <1 h probe:
   - *Option A (selector-only — preferred if it exists):* constrain KillAura's
     candidate set to `{our pick}` via a Wurst target filter (we already drive
     KillAura filters — `set_killaura_no_pvp`, the PvP-filter memory). The neural
     head selects; KillAura still aims+fires. **No servo, no new attack primitive.**
   - *Option B (full attack injection — fallback / more general):* a homunculus
     `/attack_entity {runtime_id}` primitive (aim at the entity + `interact`,
     respecting attack-cooldown). Bypasses KillAura target+execute. More work, but
     it's the reusable servo+injection path the block-mining swap will later need.
   - *Done:* one of the two demonstrably lands an attack on a chosen runtime_id.
3. **Live inference cadence.** Target *selection* changes slowly — we do **not** need
   20 Hz. A 2–4 Hz selection loop (fast obs read of `entity_set` → pointer → set the
   filter/fire) feeding KillAura's fast aim is the bar. Confirm the obs read is fast
   enough at that rate (the sidecar was built for *recording*; live needs a light
   `entity_set` query). *Done:* selection loop sustains ≥2 Hz against a live arena.
4. **Behavioral-equivalence harness (the deliverable).** Reuse `build_arena` + the
   ambush harness. A/B, N controlled encounters: **(i)** stock KillAura, **(ii)**
   neural-selector. Metrics: target-agreement rate (does it pick the same entity
   KillAura would?), time-to-clear, damage taken, and *failure* modes (no-target
   stalls, thrash between targets, wrong-type strikes). *Done:* **the online-vs-offline
   gap number** — does 0.985 offline hold in the loop? The §11c framing is the
   pre-registered expectation: the cadence-stripped (0.86→0.52) and persistence-wins
   gaps are the *offline shadow* of the distribution shift this will expose (live
   `entity_set` jitter, multi-mob churn, timing). A large gap is as informative as a
   small one — it's the first measurement of decode→control transfer.

*Risks / unknowns to retire early:* (a) Wurst KillAura may not expose a single-
candidate filter → Option B. (b) live `entity_set` latency at the selection rate. (c)
the head must retrain under the live-readable feature contract (any field we can't
read at 20–4 Hz is banned from training). (d) arena reproducibility — fixed seed +
fixed spawn (the world seed is stable across wipes; same coords → same terrain).

### 13.2. Transition-seam study — warm parallel thread (offline, lower-stakes)
§12.3 measured the flat *interior* of a segment (no decay). §7's sharp claim is that
the LLM's value is *originating* goals, not sustaining them → degradation localized at
goal **transitions**, which §12.3 averaged over. This thread measures the seam itself
on the existing `frozen_narrated` set — no new capture, no infra, reuses the §12.3
decoder as a readout instrument.

1. **Transition-aligned crossover.** For each segment boundary at tick `t0` (the LLM
   turn that changed `g_t`), evaluate the segment decoder over `[t0−W, t0+W]` and plot
   `P(predict old segment)` and `P(predict new segment)` vs `(tick − t0)`. The
   **crossover midpoint = handover latency = the moat width, properly measured.** A
   sharp step ⇒ no behavioral momentum (body yields instantly, fully corrigible). A
   lagged/sigmoidal crossover ⇒ real (narrow) momentum — the §6 forgetting-rate /
   commitment-stickiness knob made empirical.
2. **Honest instrument.** Train the decoder excluding a margin of ±M ticks around each
   boundary, then evaluate *on* those held-out boundary regions → the transition
   behavior is genuinely unseen. Run **embodied-only** (the real signal) with the
   tool-label one-hot version as a reference line (it switches ~instantly at the LLM
   turn, so `tool-switch − embodied-switch` = the literal rate-gap of §1).
3. **Resolve the §12.3 artifact.** This directly disentangles the fresh-tick block dip
   (0.44): is it genuine handover latency (body hasn't committed yet) or just short-
   segment ambiguity? The crossover separates them. Folds in the flagged within-
   segment refinement (decode start→end per individual segment).
4. **Flag the data limitation, don't fix it here.** `frozen_narrated` is
   peaceful/dawn diamond runs → mostly *completion* transitions (mine→craft), few
   *interrupt/override* transitions (threat→flee) — and the override-mid-commitment
   transition is the corrigibility-relevant one (§6). If the completion-crossover is
   interesting, a small non-peaceful recapture is a *next-sprint* candidate, noted not
   built. *Done:* `rung_c_transition.py` + the crossover curve + the handover-latency
   number.

### 13.3. Sequencing
13.1.1 (train→checkpoint) is the only hard prerequisite and is offline, so it kicks
off immediately alongside 13.2 (also offline). 13.1.2 (injection spike) retires the
biggest architecture unknown next — do it before building the harness. 13.2 fills the
gaps while the live-loop unknowns are being spiked. **Completion marker for the
sprint:** the neural target-selector runs live and we have read the online-vs-offline
gap; the seam study has produced a handover-latency number. Then re-plan.

---

### 13 — RESULTS (offline tranche, 2026-05-29)

Both no-MC items shipped and verified. The live items (13.1.2 spike → 13.1.3/4
harness) remain open. Run as package modules
(`python -m experiments.next_packet.<script>`).

**13.1.1 train→checkpoint — DONE.** `experiments/next_packet/rung_a_target_train.py`
reuses the validated `rung_a_target` pipeline (`load_attacks` / `cand_features` /
baselines) and mirrors `train_arm`'s arch + geom z-score, adding model capture +
persistence. On the 4 combat rollouts (260 ATTACK events, train 195 / val 65, 24
entity types):

| arm | dim | val_acc final | val_acc best |
|-----|-----|--------------|--------------|
| geom | 8 | 0.892 | 0.954 |
| geom+type | 32 | 0.923 | **0.985** |

Baselines (all events): nearest **0.431**, nearest-hostile **0.738**. geom+type
**reproduces the §11a 0.985 headline exactly**; geom-only reaches 0.954. **geom-only
is frozen as primary** (fewer live fields, no entity-type OOV; the +0.03 from type is
flagged for the 13.1.2 spike to revisit). Artifacts (gitignored)
`results/rung_a_target_ckpt/`: `model.pt`, `model_geom.pt`, `model_geomtype.pt`,
`feature_spec.json`, `metrics.json`. The contract is **live-readable-only**
(`no_delta_tick=True`): the 8 features are
[dx, dy, dz, dist, sin/cos(off_yaw), sin/cos(off_pitch)] from one live `entity_set`
snapshot + player pos/yaw/pitch, with the geom z-score stats baked into the spec.

**13.2 transition-seam study — DONE. Headline: the handover latency is ~6 ticks
(≈ 0.32 s) — sharp, not gradual.** `experiments/next_packet/rung_c_transition.py`,
reusing the §12.3 pipeline (`rung_c_moat.load_rollout` / `segments` / `featurize`
with `tool_vocab=None` → 22-dim embodied feats, no current_tool, no delta_tick).
Instrument: a per-rollout **multiclass** segment decoder (the §12.3 classifier)
trained on segment *interiors* (rows ≥ holdout=20 ticks from either of their own
boundaries) and evaluated *on* the held-out seam; at each seam row we read
`rel = p_new / (p_old + p_new)` and bin by signed offset (tick − t0). Offsets need no
absolute tick — within the new segment offset = `ticks_since_g_t_issued`, within the
old offset = tsi − len(old). 5 rollouts, 135 segments, 130 boundaries (all used);
decoder interior train acc **0.975** (the decoder is strongly real). The curve is a
clean, sharp step:

```
offset:  -60   -15     0    +5   +10   +15   +30   +60   (ticks rel. to g_t boundary)
p_new:  0.29  0.37  0.36  0.39  0.64  0.77  0.86  0.92
```

- **Crossover (rel=0.5) = 6.4 ticks ≈ 0.32 s.** Through the old segment *and across
  the boundary itself* the body still reads firmly OLD (p_new ≈ 0.29–0.39, flat); it
  does not begin to commit until ~+5 ticks, crosses 0.5 by ~+6, and saturates ~0.92
  by +30. So there is a real, short *dead time* (~0.3 s) before the body yields,
  then a fast flip — narrow behavioral momentum, highly corrigible (§6).
- **13.2.3 resolved:** long-new (3.4 ticks) and short-new (11.4 ticks) segments both
  show finite latency — controlling for new-segment length does NOT erase it (long
  segments, with more/cleaner interior signal, actually flip *sooner*). So the §12.3
  fresh-tick dip is **real handover latency**, not a short-segment artifact; at
  offset 0 the freshest new-segment ticks still decode as old, which is exactly that
  dip seen from the seam side.
- **Reference line:** the current_tool label flips at the boundary (114/130
  boundaries; the rest = g_t changed, tool didn't, consistent with g_t≠tool 97 %).
  Tool switch is at offset 0 by construction (set per LLM turn), so **rate_gap =
  crossover − 0 ≈ 6.4 ticks (≈ 0.32 s)** — the literal §1 rate gap: symbolic intent
  switches instantly, the body's decodable state lags by ~0.3 s.
- Output (gitignored) `results/rung_c_transition/crossover.json`.

**Combined picture (§12.3 + §13.2): the moat at a completion transition is ~0.3 s of
dead time, then a sharp flip; once crossed, intent stays legible the full segment.**
That ~0.3 s is the latency of the *existing* LLM→Baritone path through the rate
tower — which is what the 13.1 closed-loop swap is meant to bypass (a direct 2–4 Hz
neural target-selector at KillAura's filter does not route through that planner
handover). So the seam number *motivates* the swap rather than bounding its cadence.
**Data caveat (13.2.4) stands:** peaceful `frozen_narrated` → these are mostly
*completion* transitions, not *override*; the corrigibility-relevant override seam
needs a non-peaceful narrated recapture (next-sprint input).

*(Process note: a tool-channel fault earlier this session fabricated a first round of
plausible-but-fake numbers; every figure above is from a verified re-run — read back
out of `metrics.json` / `crossover.json`.)*

---

## 14. Next sprint — Close the codec loop (2026-05-29 plan)

Organizing question: *before we ever train a neural codec, prove the **substrate it
must run on** can carry a full intervention end-to-end — i.e. that an identity codec,
intercepting every action packet, encoding to the structured action space and
decoding back, can drive a **working controller through a full rollout** with no
behavioral regression.* If the loop can't carry a lossless codec live, training
against any encoding is wasted — we'd be optimizing for a representation the wire path
can't actually deliver. This sprint de-risks that: it closes the codec loop with the
**identity** codec as the payload, so the only thing under test is the *loop*, not the
*encoding*.

**Why now / how this relates to §13.** §13 swapped one *decision* (attack-target) via
a bespoke primitive (`/attack_entity`) — the decision→packets actuator, hand-written.
§14 is orthogonal and more fundamental: it exercises the **general** packet-codec path
(`OutboundPacketMixin` → `PacketFieldExtractor` → Python `encode/decode` →
`PacketReconstructor` → substitute-on-wire) that already exists in the tree but has
**never been run end-to-end live**. §13 proved *a* decision can drive the body; §14
proves *the codec substrate* can carry *all* the body's actions losslessly under load.
Block-mining / neural-codec training / rung B remain out of scope (ml.MD §10): this
sprint ends at "full-rollout identity substitution runs with no behavioral
regression, and we have the latency budget."

### 14.0. What already exists (read before building)
The machinery is **built and offline-green** — the gap is purely the live run.
- **Intercept:** `OutboundPacketMixin` HEAD-injects the private `Connection#sendPacket`
  funnel (all `send()` overloads converge there), SERVERBOUND-filtered, with the
  recursive-substitute trap already solved (`ROUNDTRIPPING` ThreadLocal).
- **Allowlist:** `PacketAllowlist.SPATIAL_PLAY` = 11 serverbound play packets (4 move
  subtypes, player_input, player_command, use_item, use_item_on, player_action,
  interact, swing). Inventory / containers / handshake deliberately excluded.
- **Encode/decode:** `PacketFieldExtractor` (all 11) → `craft/codec` Python
  `encode()→Action→decode()` → `PacketReconstructor.build` (all 11, incl. interact
  ATTACK/INTERACT/INTERACT_AT). Codec unit suite **29/29 green**; all 11 types
  registered.
- **Substitute on wire:** `CodecPassthrough.trySubstitute` (sync, `substitute:true`)
  reconstructs the decoded fields into a packet, sends the clone, cancels the
  original; **falls back to the original on any failure** (unsupported type, drift,
  transport error) — a coverage gap can never break the wire.
- **Two identity codecs at two altitudes** (the layering from the design discussion):
  - **Phase 1 `PacketRoundtrip`** — byte-identity through *Mojang's own* StreamCodec,
    fully in-process, no Python, no network. The **plumbing control**.
  - **Phase 2 `CodecPassthrough` substitute** — round-trips through the *real Python
    semantic codec* (delta-encoding, pointers-into-obs, `fields_close` atol 1e-6).
    **Not byte-identity** — reconstructs from decoded floats — so its correctness
    metric is **behavioral parity, not byte-equality.** This is the actual test.

### 14.1. The rungs (control → test)
**Byte-precise equality is explicitly NOT the bar.** The bar is a **working
end-to-end controller with the full identity-codec intervention active** across a real
rollout. Byte-equality (Rung 1) is only a control that isolates plumbing faults from
encoding faults.

- **Rung 0 — observer dry-run (no wire mutation).** Stand up `craft.codec.server`; arm
  `/codec/passthrough` in observer mode (`substitute:false`) over one rollout.
  **Pass:** `drift == 0` and `transport_errors == 0` across the full rollout — the
  semantic codec is identity-in-practice at live data rate, without touching the wire.
  Safe precondition for any substitution.
- **Rung 1 — plumbing control (Phase 1 byte-identity).** Arm `/packets/roundtrip` over
  full rollouts. **Pass:** `byte_mismatch == 0`, `encode_failed == 0`, zero
  disconnects, and rollout outcomes statistically indistinguishable from codec-off. If
  behavior moves *here*, the substitution machinery itself is the bug — independent of
  any encoding.
- **Rung 2 — THE TEST (Phase 2 semantic substitution, full rollout).** Arm
  `/codec/passthrough {substitute:true}` over full rollouts; the agent plays normally
  with every allowlisted action round-tripped through the Python codec and
  reconstructed onto the wire. **Pass:** the controller completes rollouts with
  behavioral parity to baseline (survival / milestones / distance), `substitute_errors`
  and `drift` near-zero, and latency within a no-desync budget. A delta here is
  attributable to the encoding-or-its-latency — which is exactly the feasibility
  verdict, and the template every future *candidate* (lossy/neural) encoding runs
  through.

### 14.2. What to build (all small — the machinery exists)
1. **Latency instrumentation** in `CodecPassthrough.trySubstitute` — per-packet
   round-trip wall-time → mean / p99 / max in `snapshot()`. Currently it counts
   *outcomes* but not *time*, and **time is the feasibility signal** (sync HTTP on the
   send thread, see gotcha). ~15 lines Java; ships in the build that does Rung 2.
2. **Driver script** (`experiments/codec_loop/run_rungs.py` or similar) — sequences
   Rungs 0→1→2 over the existing rollout harness, arms/disarms the homunculus routes,
   reads `/status` counters + the new latency fields, emits a comparison table. Reuses
   the rollout runner; do NOT reimplement spawn/agent.
3. **Behavioral-parity metric** — codec-off vs Rung-1 vs Rung-2 on existing rollout
   outcome JSONLs (survival, milestones, distance). Piggyback on the suite JSONL shape;
   parity = within-noise, not identical.
4. **Doc-drift fix** (do early, it actively misleads): `CodecPassthrough`'s class
   docstring still says *"does NOT substitute the codec's output"* and
   `OutboundPacketMixin` / `trySubstitute` comments say *"only ServerboundMovePlayerPacket
   is reconstructable"* — both false now (substitute exists; all 11 reconstruct). Fix
   the comments to match the code.

### 14.3. Load-bearing gotchas
- **Sync HTTP on the netty send thread is the make-or-break.** `trySubstitute` does a
  blocking POST to Python *per allowlisted packet* (1000 ms timeout) inline on the
  wire. Movement is 20 Hz. Even a *perfect* identity codec can rubberband / desync /
  trip server timeouts purely from added latency — and that would make the loop
  "practically infeasible" regardless of encoding quality. **This is the single most
  likely Rung-2 failure.** Measure first (instrumentation above); only if it desyncs
  do we consider mitigations (in-process Java codec — breaks single-source-of-truth;
  batching; or accept it as a measured property). Rung 1 (in-process, zero network) is
  the control that proves a desync is *latency*, not *logic*.
- **`fields_close` atol 1e-6 ≠ byte-identity.** Rung 2 reconstructs from decoded
  floats, so even an "ok" round-trip puts subtly different bytes on the wire than the
  client would have. Behaviorally harmless for movement; means **byte-equality is the
  wrong Rung-2 metric — behavioral parity is.** (Byte-equality belongs to Rung 1 only.)
- **Interact/player_command entity resolution** calls `mc.level.getEntity` on the
  network thread (latent threading hazard) and returns `null`→fallback if the entity
  unloaded. Acceptable for this sprint; flagged.
- **Sequence numbers** (`use_item` / `use_item_on` / `player_action`) round-trip fine
  under identity, but the *neural* codec convention is drop-and-regenerate from the
  local counter. Reconstruction will eventually need the live sequence counter, not the
  obs — **deferred to the neural-codec sprint, noted now while context is fresh.**
- **Deploy discipline** ([[feedback_jar_deploy_over_running]]): the latency-instrumentation
  build means a jar redeploy — stop the agent, deploy, relaunch; never split fresh-vs-stale.
- **Codec server is fleet-shared + stateless** — one `craft.codec.server` backs all
  agents (pure-function, no per-agent state). Start it before any rung; it's the
  `transport_errors` canary if it's down.

### 14.4. Sequencing
Doc-drift fix (14.2.4) first — it's misleading anyone reading the code. Then **Rung 0
immediately** (no new code — just start the codec server + arm observer over one
rollout) to confirm the codec still round-trips clean against *live* traffic before
investing in the harness. Then build latency instrumentation + driver (14.2.1–2),
deploy, run Rung 1 (control) and Rung 2 (the test) back-to-back so they share a
baseline. **Completion marker:** a full rollout completes with `substitute:true`
active across all 11 packet types, behavioral parity to baseline, and a latency budget
in hand. Then re-plan toward the first *lossy* codec.

### 14.5. RESULTS (live, verified)
- **14.2.4 doc-drift — DONE (2026-05-29).** Fixed stale comments in
  `CodecPassthrough` (class docstring + substitute-field comment),
  `CodecPassthroughHandler` (arm doc), all of which falsely claimed substitute
  doesn't exist / only the move packet reconstructs. Now describe the two-mode
  (observer/substitute) design + all-11 SPATIAL_PLAY reconstruction. Doc-only,
  no redeploy. (`OutboundPacketMixin` was already accurate.)
- **Rung 0 — PASS (2026-05-29, live on agent0).** Codec server already up
  (`craft.codec.server --port 25600`); deployed jar is today's substitute build
  (commit 56fad62). Armed `/codec/passthrough {substitute:false}` on agent0
  (homunculus 25570), drove one Baritone goto (9.5,62,6.5) → arrived (30,63,30),
  polled + disarmed. **Counters: attempted=221, ok=221, drift=0,
  transport_errors=0, queue_drops=0, no_obs=0.** 221 live movement packets
  round-tripped through the Python semantic codec at live data rate, perfectly
  clean — the codec is identity-in-practice on real traffic. Safe precondition
  for substitution met. (Verified from `/tmp/rung0_result.txt`; PASS criterion
  `drift==0 && transport_errors==0` satisfied.)
- **14.2.1 latency instrumentation — DONE (2026-05-29, built & verified).** Added
  per-send wall-time around the synchronous POST in `CodecPassthrough.trySubstitute`
  (timed in the `finally`, so transport errors are timed too). Lock-free:
  count + sum (mean) + CAS-max + a coarse fixed histogram (1/2/5/10/20/50/100/200/500ms
  buckets + open top) for p50/p99. Exposed in both `snapshot()` (status) and
  `snapshotCounters()` (disarm) as `subst_latency_{count,mean_ms,max_ms,p50_ms,p99_ms}`;
  reset in `arm()`. `./gradlew build` SUCCESSFUL; `javap` confirms `recordSubstLatency`,
  `percentileMs`, and the four latency fields in the jar (build/libs/homunculus-0.1.0.jar,
  15:52). NOT yet deployed to the fleet (running jar is the 12:28 substitute build
  without timing) — deploy + relaunch needed before any substitute-mode (Rung 2) run.
- **Rung 1 — PASS (2026-05-29, live on agent0).** Plumbing control: in-process
  Phase-1 byte-identity via `POST /packets/roundtrip {enabled:true}` over one
  Baritone goto (no Python, no network). **Counters: roundtripped=75,
  passed_through=0, encode_failed=0, decode_failed=0, byte_mismatch=0.** Mojang's
  StreamCodec round-trips every movement packet bit-for-bit — the substitution
  machinery itself is sound, independent of any encoding. (Verified from
  `/tmp/rung1_result.txt`; PASS criterion `byte_mismatch==0 && encode_failed==0`
  satisfied, no disconnect.) NOTE: this path needs no new jar (the running 12:28
  build already has it); only Rung 2's `subst_latency_*` fields need the redeploy.
- **14.2.1 latency build BUG + fix (2026-05-29).** The first latency-instrumented
  jar (15:46) was DOA: `LAT_BUCKETS_US` was declared *after* `INSTANCE` in the class,
  so `new CodecPassthrough()` (static init) read it as null when sizing `substLatHist`
  → `ExceptionInInitializerError` (NPE in `<clinit>`) → every `/codec/passthrough`
  call 500'd with `NoClassDefFoundError: Could not initialize class CodecPassthrough`
  (seen in agent0's latest.log after deploy+relaunch). FIX: moved `LAT_BUCKETS_US`
  above `INSTANCE`. `./gradlew build` SUCCESSFUL. **Rung 2 has NOT yet run** — the
  broken jar never substituted a single packet (player was also still joining world
  during the attempt). Need: deploy fixed jar → relaunch agent0 → confirm class
  inits (arm substitute, GET status returns 200 not 500) → THEN run Rung 2.
  Lesson logged: a green `gradlew build` + `javap` symbol check does NOT prove the
  class *initializes* — only a live arm/status does.
- **Rung 2 — THE TEST — PASS (2026-05-29, live on agent0, fixed jar 16:26).**
  After the init-order fix: confirmed class inits (arm substitute → status 200, not
  500), player in world at (6.5,62,6.5). Armed `/codec/passthrough {substitute:true}`
  — Python semantic codec DRIVES THE WIRE — drove goto (35,62,35) then back (9,62,6).
  **Packet substitution flawless: attempted=560, ok=560, substituted=560, drift=0,
  substitute_errors=0, substitute_fallbacks=0, transport_errors=0, no_obs=0.**
  Player navigated 6.5→35.7→9.7 — controller moved THROUGH the codec to both targets.
  **Latency (the make-or-break sync-POST-on-netty-send-thread signal @20Hz):
  mean=1.27ms, p50=2.0ms, p99=5.0ms, max=9.39ms** — p99 is ~10% of a 50ms tick, no
  desync budget concern.
  **Honest caveat + control:** both gotos returned Baritone `reason:"canceled"`
  (PathEvent.CANCELED) despite reaching the targets within tolerance. I ran a
  CODEC-OFF control (same two gotos, substitute disarmed): goto#2→(9,62,6) ALSO
  canceled at the same ~(10,62,7) endpoint, and goto#1 "arrived" — i.e. the cancel
  is the awkward second target's pre-existing pathing quirk, NOT codec-attributable,
  and the codec-on vs codec-off arms are behaviorally indistinguishable (same targets
  reached, same spurious cancel). Byte-equality is not the bar (§14.3); behavioral
  parity is, and it holds. (Verified from `/tmp/rung2_result.txt` + `/tmp/rung2_ctrl.txt`.)
  **VERDICT: the full identity codec carries a working controller end-to-end on the
  live wire. The path is feasible; training a lossy/neural codec is de-risked.**

**§14 COMPLETE — all three rungs PASS (live, verified, no fabrication):** Rung 0
observer (221/221 clean, drift=0) → Rung 1 byte-identity (75 roundtripped, 0 mismatch)
→ Rung 2 semantic substitution (560/560, 0 drift/errors, p99 5ms, controller reached
both targets; cancel-reason shown codec-independent by control). The codec-in-the-loop
is live and lossless. Next sprint: first *lossy* codec through this same Rung-2 template.

- **14.2.2 rung driver — DONE (2026-05-29, run live & green).**
  `experiments/codec_loop/run_rungs.py` sequences Rungs 0→1→2 over the live homunculus
  routes (base from `craft.config`/`HOMUNCULUS_PORT`/`--port`; codec from
  `--codec-url`/`CODEC_URL`). Drives an out-and-back goto pair per rung, reads counters,
  prints a comparison table, `--out`s JSON. **Key design choice from the Rung-2 finding:
  arrival is POSITION-BASED (`final_position` within tol), not `reason=="arrived"`** —
  Baritone's spurious PathEvent.CANCELED would otherwise false-fail a controller that
  reached its goal. `--rungs` subsets; `--latency-budget-ms` (default 10ms) gates Rung-2
  p99. Ran `--port 25570`: **OVERALL ALL PASS (3/3)** — Rung 0 attempted=504 drift=0
  transport_errors=0; Rung 1 roundtripped=386 byte_mismatch=0; Rung 2 substituted=389
  drift=0 substitute_errors=0 mean=1.07ms p99=2.0ms max=4.49ms, 2/2 targets reached.
  Independent driver reproduction of the manual rungs — the §14 verdict is
  driver-reproducible, not a one-off. (§14.2.3 behavioural-parity-over-rollout-JSONLs
  metric deferred to the lossy-codec sprint, where a real delta can appear; under the
  identity codec the targets-reached + codec-off control already establish parity.)

## 15. Next sprint — First *lossy* codec (2026-05-30 plan)

Organizing question: *§14 proved the wire path can carry a **lossless** identity
codec through a working controller. Now: how much can we throw away before the
controller notices?* This is the first codec that **loses information on purpose** —
the gate to the neural codec, because it answers whether the action stream has slack
to compress at all, and gives a baseline a learned codec must beat.

**Design-as-scope (this sprint's defining choice).** We are NOT pre-committing to a
codec family (quantizer vs learned AE vs codebook). **Rung 0 below is a
characterization pass whose *deliverable is the codec-family decision*** — measure
the action stream's structure, then pick the codec the data argues for. This keeps us
from training against the wrong target (the §14 lesson, one level up: don't build the
encoding before you've measured what needs encoding).

### 15.0. Characterize the action stream → DECIDE the codec (rung 0, no wire changes)
Pure offline analysis over the frozen captures (`results/frozen_{narrated,combat,dryrun}`,
~72.6k packet+obs records already on disk — no new capture needed to *characterize*).
Questions to answer, each with a number:
- **Volume by type** (measured): swing 33%, move_player_rot 31%, move_player_pos_rot
  24% → movement ~55%, swing 33%, all discrete actions (player_input/action/command,
  interact, use_item*) <10% combined. **Implication: compress movement first; it's
  where the bits are.** Swing carries ~1 bit of payload (hand) at huge volume — a
  separate, trivial story.
- **Per-field entropy / dynamic range** of the move `semantic_fields`
  (`pos`=Δ vs obs, `rot`=abs yaw/pitch, `on_ground`/`horizontal_collision` bools).
  How many *effective* bits does each field actually carry? (e.g. is Δpos already
  near-quantized by Baritone's step cadence? is yaw multimodal?)
- **Redundancy vs obs** — how predictable is each field from the obs the decoder
  already has (pos, last yaw/pitch, g_t, current_tool)? A field the decoder can
  *reconstruct* from obs is free to drop (the delta-coding `MoveAction` already does
  for `pos` is exactly this; the question is how much further it goes).
- **DECISION OUTPUT:** a short written verdict in §15 RESULTS picking the rung-1
  codec family + which fields/packet-type it targets, justified by the three numbers
  above. Candidate families to weigh: (a) **fixed-point quantization** (round each
  field to b bits — zero training, pure dimensionality probe); (b) **learned codebook**
  (k-means a vocab from data — natural for discrete-ish / multimodal fields);
  (c) **conditional autoencoder** (5 floats + obs → latent d → 5 floats — tests
  whether learning buys compression a quantizer can't). Deliverable is the pick, not
  all three.

### 15.1. Build the chosen lossy codec + the parity harness (rung 1)
- Implement the rung-0-chosen codec as a registered `encode/decode` pair behind a
  flag (so the identity codec stays the default; lossy is opt-in per the §14 seam).
  It must satisfy the same `Action` protocol — the only change is that `decode(encode())`
  is no longer `fields_close` to input.
- **Parity metric (this is the real §14.2.3, now that a delta can exist).** Reuse the
  Rung-2 driver (`experiments/codec_loop/run_rungs.py`, position-based arrival +
  codec-off control). Lossy-specific additions: (i) per-leg **path error** (not just
  targets-reached binary) — RMS deviation of the codec-on trajectory from the
  codec-off trajectory over the same goto; (ii) **bits/packet** actually shipped;
  (iii) latency (should stay ~§14 levels — codec compute is small).
- **Reconstruction-fidelity offline check FIRST** (cheap gate before any live run):
  decode(encode(x)) vs x over the frozen set, report per-field error distribution.
  Only go live once offline error is in a sane band.

### 15.2. Sweep + the headline plot (rung 2 = THE TEST)
- Sweep the codec's lossiness knob (quantizer bits b = 16,8,6,4,2; or codebook size
  K; or latent dim d) — each setting is one Rung-2 live run via the driver.
- **Headline deliverable: parity-vs-compression curve.** X = bits/packet (or
  effective compression ratio over identity); Y = behavioral parity (targets-reached
  AND path-error-within-noise-of-the-codec-off control). **PASS for the sprint =
  there exists a setting with compression > 1× where the controller still reaches
  targets within control-off noise.** The knee of that curve is the science.
- If a learned codec was chosen, the quantizer at equal bits is the baseline it must
  beat (else learning bought nothing here — itself a finding).

### 15.3. Sequencing & scope discipline (ml.MD §10)
Rung 0 (characterize → decide) **first and alone** — do not write codec code until
the data has picked the family. Then rung 1 (codec + offline fidelity gate + parity
harness), then rung 2 (sweep → curve). **Sprint ends at the parity-vs-compression
curve with a knee identified.** Out of scope: multi-packet-type codecs (pick ONE
target from rung 0), neural codec architecture search beyond the single chosen
family, and any rung-B/meta-controller work. Observe the knee, then plan §16.
**Load-bearing reuse:** the §14 Rung-2 driver + codec-off control + position-based
arrival are the harness; §15 adds the lossy codec and the path-error/bits axes only.

## 16. Next sprint — The first *learned* codec (2026-05-30 plan)

Organizing question: *§15's scalar quantizer is a memoryless, obs-blind floor. The
controller's loss-tolerance is structured (drift-fatal, dropout-benign) and the move
stream is highly predictable from obs (§B: `move_player_pos_rot` R0 NLL **0.316**).
Can a codec that **conditions on the obs the decoder already has** exploit that
predictability to go below the quantizer's bit floor — without reintroducing the
stationary drift that kills the controller?*

This is the **A↔B convergence sprint**. Sprint A measured *what the controller
tolerates* (drift fatal, dropout benign — `results/sprintA/RESULTS_zero_mode_ab.md`).
Sprint B measured *how predictable the move stream is from obs* (NLL 0.316, "compressible
*because* predictable" — `results/sprint_b/RESULTS.md`). A conditional learned codec sits
exactly at that intersection: it is the artifact that unifies them. Building it **is** the
GAP synthesis the original brief asked for — without picking a winner.

**Scope (locked 2026-05-30, user + colleague):** ONE family — a **conditional
autoencoder** (`5 floats + obs → latent d → 5 floats`); the latent dim `d` is the
lossiness knob. ONE target type — **move** family only. The final family pick is still
the 16.0 deliverable, but 16.0 is biased to characterize *for* the conditional AE.
NOT a codebook, NOT a family bake-off, NOT multi-packet-type.

### 16.0. Characterize the *conditional* residual → spec the objective (offline, no wire changes)
Pure offline analysis over the frozen corpus (`results/frozen_{narrated,combat}`,
~36k narrated move+obs packets on disk — no new capture). Deliverables, each a number:
- **The compression ceiling.** How many *effective* bits does each move field carry
  *given* obs (pos, last yaw/pitch, on_ground, horizontal_collision, g_t)? §B already
  says pos_rot is near-R0-predictable; quantify the residual entropy of `pos` (Δ vs obs),
  `rot` (abs yaw/pitch), bools — *conditioned* on obs. This residual is the floor a
  conditional AE can reach; it bounds how far below `zero_preserving@b5` learning *could*
  buy.
- **The objective spec — the load-bearing artifact.** §15 proved an MSE/L2 objective
  will "rediscover" `zero_biased` (trade a tiny stationary bias for lower average
  distortion → walk into the rubberband). So the loss is NOT plain reconstruction. Write
  it down here: **(i) hard-zero the at-rest reconstruction** (stationary input → exactly-0
  pos delta out, structurally if possible — e.g. a learned still/moving gate — not just
  penalized); **(ii) asymmetric penalty** that treats stationary drift as expensive and
  moving-delta dropout as cheap (the A finding as a loss term); **(iii)** a rate term on
  the latent (the d sweep, or a KL/entropy penalty) so compression is an explicit
  objective, not an accident of bottleneck width.
- **DECISION OUTPUT:** confirm conditional AE (or, if the residual analysis argues
  otherwise, justify the deviation), fix the obs feature set fed to encoder+decoder
  (must be **decoder-reconstructable** — no rollout-id, no Baritone path; §B anti-pattern
  #2 carries over verbatim), and freeze the objective.

**16.0 RESULT (2026-05-30) — DONE.** `experiments/codec_loop/cond_residual.py`
(offline, no training); `results/sprint16/RESULTS_cond_residual.md` +
`cond_residual_{narrated,combat}.json`. Three-level bits/packet ladder at the §15
parity-safe grid: fixed-point alloc **18.45** → marginal entropy (free arith-coder)
**9.14** → conditional entropy **1.81** b/pkt (10.2× vs alloc; combat 7.9× agrees).
**Headline: the conditional prize is ROTATION, not position.** Pos is already at its
floor (`zero_preserving@b5` marginal 0.46 b/axis; conditioning adds 0.07–0.09 — no
learnable pos prize). yaw/pitch carry ~4 bits ABSOLUTE but only **0.2–0.5 bits coded
RELATIVE to `obs.{yaw,pitch}`** (per-tick turn median 0.6°, p99 ≈ 154° snap tail;
obs confirmed pre-packet per-tick → live-faithful). **VERDICT: most of the 8–10×
is a FREE deterministic reparameterization, not learning.** The honest baseline a
learned codec must beat is therefore **≈1.8 b/pkt** (obs-relative rotation + pos on
zero_preserving + per-field arithmetic coding), NOT the 18.45 alloc. Family confirmed
(**conditional AE**); frozen objective (see RESULTS §"FAMILY + OBJECTIVE SPEC"):
(1) **input rotation MUST be obs-relative** — itself the dominant win, and a concrete
`craft/codec/move.py` change (it carries rotation absolute today, move.py:22);
(2) structural at-rest gate (hard-zero pos at rest; gate temporally coherent, run≈13–26
ticks → ~free); (3) latent rate term; (4) NOT MSE/L2 (rediscovers zero_biased); (5)
decoder-reconstructable inputs only, by-rollout split. Sharpened §16.2 null: if the AE
can't beat obs-relative-reparam, the stream's compressibility is a deterministic
reparameterization, not learnable structure — a crisp falsifiable outcome.

### 16.1. Train + offline fidelity gate (offline, before any live run)

**16.1 BASELINE-FIRST RESULT (2026-05-30) — DONE (offline leg).**
`experiments/codec_loop/obsrel.py` (`quantize_move_obsrel`: every move field a
zero_preserving delta vs obs — pos already is, yaw→`wrap180(yaw−obs.yaw)`,
pitch→`pitch−obs.pitch`) + `obsrel_baseline.py` (RD measurement).
`results/sprint16/RESULTS_obsrel_baseline.md` + `obsrel_baseline_{narrated,combat}.json`
(sha `0606d1c6…`/`42bafb72…`); codec tests 105 pass. **(1) obs-relative DOMINATES the
rate-distortion frontier:** the §15 parity point absolute@b5 = yaw RMSE 3.36° @ 8.0 bits;
obs-rel matches that fidelity at b3 = **0.37 bits (~22× rotation-rate cut), zero learning**
— total ≈ the §16.0-predicted 1.8 b/pkt. **(2) zero-mean-at-rest holds for the CAMERA:**
obs-rel at-rest RMSE = 0.25° FLAT across b8→b2 (still player → residual-0 → obs.yaw
exactly), while absolute injects a static heading offset growing 2.6°@b5 → 28°@b2 — the
§15 pos zero-bias on the rotation channel. So the principled codec MUST code rotation
obs-relative. **The honest baseline-to-beat (~1.8 b/pkt) is already excellent with zero
learning → §16.2's null is live.** **16.1 LIVE PARITY DONE (2026-05-30).** `obsrel_live.py` + `server.py` `obsrel` mode;
`results/sprint16/RESULTS_obsrel_live.md` + `obsrel_live_{d8,combined}.json`
(sha `12bca6e9…`/`9a782799…`). Swept obs-relative rotation b6→b2 (pos near-lossless) AND
the full baseline codec (pos zp@b5 + obs-rel rot b5→b3) on agent0/peaceful. **HEADLINE:
rotation deadband is BENIGN for navigation — reach=1.0 at EVERY level down to b2 (180°
steps), control=1.0**; codec machinery clean throughout (drift=0, subst_err=0,
transport_err=0, p99 5–10 ms). §15 dropout-tolerance extends to the rotation channel
(Baritone re-issues heading each tick → deadbanded micro-turns re-sent free). (v1 sweep's
flat reach=0.5 was an arena-platform-edge confound at delta=28, NOT codec — fixed
delta→8.) **HONEST CAVEAT: goto-reach is rotation-INSENSITIVE** (position carries
navigation; server-seen yaw near-cosmetic for a goto) → necessary-not-sufficient. §11a's
**block-target = gaze** says rotation IS load-bearing for mine/attack/place — those
aim-dependent behaviors are where a rotation knee could still live and are UNTESTED here.
→ **§16.2 scoping signal:** the baseline holds navigation parity ~for free with huge
rotation headroom, so the AE has ~nothing to prove on navigation; to give the learned
codec a parity test it can fail, the live arm needs an AIM-dependent task, not just gotos.

- Implement the conditional AE as a registered `encode/decode` pair behind a flag (the
  identity codec stays default; lossy/learned is opt-in per the §14 seam — same seam §15's
  quantizer uses). Reuse the §13.1 train harness pattern (`rung_a_target_train.py`):
  by-rollout split (held-out rollouts, never random — §B leakage lesson), PyTorch cu128.
- **Offline fidelity gate FIRST (cheap, before live):** `decode(encode(x))` vs `x` over the
  held-out frozen set. Two checks, both must pass: (a) per-field error distribution in a
  sane band; (b) **the zero-mean-at-rest check is the gate** — stationary RMSE ≈ 0 (this is
  the §15 prior made into a pass/fail; an AE that fails it is pre-disqualified from going
  live, no fleet time wasted). Sweep latent dim `d` ∈ {2,4,8,16} and report the
  fidelity/effective-bits frontier offline.
- **Effective bits** = the latent's actual coded size (d × per-dim bits, or measured
  entropy of the quantized latent), reported on the same axis as §15's `float_bits` so the
  curves overlay.

**16.1 LEARNED-CODEC RESULT (2026-05-30) — §16.2 NULL CONFIRMED (offline).**
`ae_headroom.py` (preflight) + `ae_train.py` (conditional β-VAE);
`results/sprint16/RESULTS_ae.md` + `ae_headroom_narrated.json` (sha `21a58442…`) +
`ae_rd.json` (sha `7e0c457d…`). The obs-relative baseline is already a per-field lag-1
conditional coder, so a learned codec's only headroom is cross-field/temporal structure.
**Measured directly:** cross-field MI ≤ **0.27 b/pkt** and almost all of it a *boolean*
(on_ground×moving 0.268; I(yaw;pitch)=0.032) — **rotation residuals are independent,
~0 learnable**; the only big temporal structure is a position still/moving GATE
(0.96→0.23 given prev = deterministic RLE, on the saturated pos channel), rotation has
~none. The β-VAE confirms: it traces an RD curve and the **zero-mean-at-rest gate PASSES
across the whole curve** (at-rest pos RMSE ~0.005 b = 30× under the §15 fatal threshold;
at-rest yaw ~0.4–0.5°; degrades safely to "hold" at zero rate) — the §16.0 objective
works, but there is no cross-field structure to exploit. **CONFOUND flagged:** the VAE's
continuous-Gaussian rate is NOT directly comparable to the baseline's uniform-quantize
rate (~0.25 b/dim scheme overhead would overstate a learned win) — the verdict rests on
the scheme-independent MI, not raw bits. **VERDICT: on this stream the move-codec's
compressibility IS the deterministic obs-relative reparam; learning buys nothing beyond
it (≤0.27 b/pkt, ~0 on rotation).** A live AE-on-wire sweep (below) is **moot for
compression** — the AE doesn't beat the baseline and the baseline already saturates
navigation parity (§16.1-live). The genuine open thread is **aim-dependent parity of the
deterministic baseline** (mine/attack/place; §11a block=gaze), untested here → §17 with an
aim harness, NOT a goto.

### 16.2. Live parity sweep — THE TEST (rung 2) — SUPERSEDED (see 16.1 learned-codec result: null reached offline)
- Sweep `d` (each setting = one Rung-2 live run via `experiments/codec_loop/run_rungs.py`,
  position-based arrival + codec-off control — the §14/§15 harness unchanged; only the
  sidecar's codec swaps from quantizer to the learned net).
- **Headline deliverable: parity-vs-effective-bits curve, learned AE vs the
  `zero_preserving@b5` quantizer at matched bits.** Y = behavioral parity (targets-reached
  AND per-leg path-error within codec-off noise); X = effective bits/packet.
- **Pre-registered PASS:** there exists a `d` with effective bits **< b5** that holds
  **lossless behavioral parity** AND **zero-mean-at-rest**, *beating the quantizer at
  matched bits*. **If the AE does NOT beat the quantizer → learning bought nothing on this
  stream, and that is the clean reportable finding** (consistent with §B: a stream this
  R0-predictable may already be near its memoryless floor).

### 16.3. Sequencing & scope discipline
16.0 (characterize → spec objective) **first and alone** — no net code until the residual
analysis sizes the prize and the objective is frozen. Then 16.1 (train + the
zero-mean-at-rest fidelity gate), then 16.2 (live sweep → curve). **Sprint ends at the
learned-vs-quantizer parity-vs-bits curve with a verdict on whether conditioning beats the
memoryless floor.** Live runs stay **peaceful** (no mobs perturbing the gotos). Out of
scope: codebook/other families beyond the AE, multi-packet-type codecs, swing/discrete
action codecs, any rung-B/meta-controller work, and the Sprint B combat backfill (deferred,
not this sprint). **Load-bearing reuse:** §14 Rung-2 driver + codec-off control +
position-based arrival + §15's sidecar lossy seam + §13.1 train harness — §16 adds only the
conditional AE and the zero-mean-at-rest objective/gate.

## 17. Next sprint — Aim-dependent parity: where does the wire carry the *decision*? (2026-05-30 plan)

Organizing question: *§16 proved the move-packet's rotation is near-cosmetic for
**navigation** — the obs-relative codec deadbands turns to 180° steps and the controller
still reaches every goto, because position carries navigation and Baritone re-aims the
camera client-side each tick. But goto-reach is **rotation-insensitive by construction**.
§11a found the opposite for **aim**: the block target collapses to **gaze** (you must look
at a block to mine it; attack/place are the same servo channel). So: for aim-dependent
behaviors, **which wire field actually carries the target — and how lossy can it be before
aim breaks?*** This completes the parity characterization of the codec we'd ship (it carries
the *whole* action stream, not just movement), and it sets up the real learned-codec target:
the **discrete decision channel** (§11a/§13 attack-target 0.985, block-pointer) which —
unlike the mechanical, Baritone-path-following move stream (§16 null) — carries genuine
intent and may actually be learnable-compressible.

**The hypothesis to test first (and why diagnosis comes before any lossy aim codec).**
In Minecraft the *targets* of aim actions are carried **explicitly in their own packets**,
not via the move-packet yaw: block-break = `ServerboundPlayerAction{blockpos, face}`,
attack = `ServerboundInteract{entity_id}`, place/use = `ServerboundUseItemOn{BlockHitResult}`.
The move-packet yaw is largely render/anti-cheat. **So the likely finding is that move-
rotation loss is benign even for aim** (the client aims correctly regardless of the wire;
the action packet names its target) — which would EXTEND §16 ("rotation near-free") from
navigation to all behaviors. The *real* aim knee would then live in lossy-compressing the
**discrete action packets' target fields** (block_pos / entity_id), which is the §11a
pointer channel + the deferred Sprint B `interact`/`use_item_on` cells. Don't build a lossy
aim codec until 17.0 says which field carries aim (the §14/§16 lesson: measure the carrier
before encoding it).

### 17.0. Diagnose the aim carrier (live probe, no new codec)
For a fixed aim task (mine one known block; attack one spawned passive), run the §16
obsrel sidecar at MAX move-rotation lossiness (b2, 180° steps) vs lossless, and judge an
**aim-sensitive** outcome (block broken? entity hit? break-time?). Separately, perturb the
*action packet's* target field. **DELIVERABLE = the verdict on the load-bearing carrier:**
(a) if move-rotation@b2 breaks aim → rotation is load-bearing for aim and 17.1 finds the
move-rotation aim knee; (b) if move-rotation@b2 is benign (the predicted case) → the target
lives in the action packet → 17.2 is the real test (lossy discrete-target codec). Either
outcome is a clean result; (b) is the more interesting one (it isolates the decision channel).

### 17.1. The aim parity harness + the move-rotation arm
- **New build (the one piece §16 doesn't give):** an aim driver that places/locates a known
  target and drives a mine/attack with an aim-sensitive metric (blocks-broken / hits-landed
  / break-time-vs-control), the aim analog of `obsrel_live.py`'s goto-reach. Reuse the §14
  Rung-2 substitution + §16 obsrel sidecar mode wholesale.
- Sweep move-rotation bits (the §16 `obsrel` knob) → an **aim**-parity curve (not a goto
  curve). Confirms/refutes that move-rotation loss is benign for aim; if benign, it extends
  the §16 "rotation near-free" result to the full behavior set (the deterministic
  obs-relative codec is free for everything tested) — the binding operating point is then
  set by aim, and we report it.

### 17.2. (conditional on 17.0=b) The discrete-target codec — where learning might finally matter
- If aim lives in the action packets, lossy-compress their target fields: quantize
  `use_item_on.block_pos` (a pointer into the block grid — §11a), bucket/embed
  `interact.entity_id` (the §13.1 attack-target pointer, decodable at 0.985). Find the knee
  where mining/attack breaks.
- **This is the natural home for a LEARNED codec** in a way the move stream was not: §16
  showed the continuous move stream is all-reparam/no-structure because it is Baritone-path-
  following (mechanical); the discrete decision channel carries real intent (§11a/§13) and
  is highly *predictable from obs* (attack-target 0.985 from `entity_set` geometry), so a
  conditional codec there could drop the explicit target and **reconstruct it from obs** —
  the genuine "predict the decision, not the packet" codec (§11a), now on the wire. Whether
  it holds aim parity is the §18 question; §17 only finds the knee + the headroom.

### 17.3. Sequencing & scope discipline
17.0 (diagnose the carrier) **first and alone** — no lossy aim codec until the probe says
which field carries aim. Then 17.1 (move-rotation aim arm — cheap, pure §16 reuse), then
17.2 only if 17.0 points to the action packets. **Sprint ends at the aim-parity verdict:
which wire field carries aim + the knee where aim breaks.** Out of scope: training the
learned discrete codec (that's §18 if 17.2 shows headroom), combat-AI/hunting improvements,
and the non-peaceful narrated recapture (separate deferred Sprint B item — though 17 may
reuse `results/frozen_combat`). Runs are **non-peaceful only where attack needs a mob**
(spawn a single passive; otherwise peaceful + a placed block target). **Load-bearing
reuse:** §14 Rung-2 substitution + §16 `obsrel` sidecar mode + the codec passthrough; §17
adds only the aim driver (new metric) and, in 17.2, lossy quantization of the discrete
target fields. **This closes the codec's parity story (move=§16, aim=§17) and tees up §18 =
the learned discrete-decision codec — the actual neural interface the doc is named for.**

### 17.0 RESULTS (2026-05-30) — VERDICT (b): the action packet carries aim; move-rotation is render

Live probe on agent0 (homunculus :25570), peaceful, codec sidecar :25600 + the §14
passthrough in `substitute:true`. Two channels, both judged by an AIM-SENSITIVE outcome
(not a sent packet). Drivers: `experiments/codec_loop/aim_carrier.py` (attack) and
`aim_carrier_block.py` (block). Artifacts: `results/sprint17/aim_carrier_moverot.json`,
`results/sprint17/aim_carrier_block.json`.

**Attack channel** (target = `interact.entity_id`; metric = cow HEALTH drop per
`/attack_entity` on a stationary NoAI cow at point-blank, Killaura OFF):

| cell | rot step | hits landed |
|------|---------:|------------:|
| control (lossless) | — | 6/6 |
| obsrel @b2 | 180° | 6/6 |
| obsrel @b3 | 60° | 6/6 |

**Block channel** (target = `use_item_on.block_pos`; metric = block landed at requested T,
read from `level.getBlockState` via `/scan_blocks` AFTER the server round-trip — placement
is client-PREDICTED, so the post-sync scan is the truth, not `/place_at`'s own flag):

| cell | place @T | place @T+Δ |
|------|---------:|-----------:|
| control (lossless) | 4/4 | 0 |
| obsrel @b2 (180°) | 4/4 | 0 |
| obsrel @b3 (60°) | 4/4 | 0 |
| **perturb block_pos +1x** | **0/4** | **4/4** |

**Verdict = the predicted (b).** Move-packet rotation is **near-cosmetic for aim too** —
deadbanding yaw/pitch to 180° steps (b2, near-total rotation loss) breaks NEITHER attack nor
place. The server does not gate either action on the wire rotation; it trusts the action
packet's own target field (`entity_id` resolved server-side; `block_pos` placed directly).
The **block_pos perturbation is the positive carrier control**: offsetting it by +1 on the
wire moves the placed block exactly one cell over (lands at T+1, never T) — deterministic,
n=4/4, proving block_pos is authoritative AND that the harness can detect a broken aim (so
the obsrel "benign" cells are a real null, not a blind metric). This **extends the §16
"rotation near-free" result from navigation to the full behavior set** — the deterministic
obs-relative move codec is parity-free for everything tested.

**Routing:** §17.1 (the move-rotation aim knee) is therefore **moot/confirmed-null** — there
is no rotation knee for aim because rotation isn't the carrier. The binding operating point
lives in the **discrete target fields**, so **§17.2 is the real test**: lossy-compress
`use_item_on`/`player_action.block_pos` (the §11a block pointer) and `interact.entity_id`
(the §13.1 attack-target pointer, decodable at 0.985 from `entity_set` geometry) and find the
knee where mining/attack/placing breaks. This is the natural home for a LEARNED codec — the
discrete decision channel carries real intent and is highly obs-predictable, so a conditional
codec there could drop the explicit target and reconstruct it from obs (the genuine "predict
the decision, not the packet" codec, now on the wire). New substrate this added: a `perturb`
diagnostic mode in the codec sidecar (`block_pos_delta`; entity_id deliberately NOT
perturbable — the Java reconstructor resolves it to an Entity, so a bogus id silently falls
back to the original packet, which is why the carrier proof runs on the clean block channel).

## 17.2. The lossy discrete-target codec — where does aim break, and where might learning finally win? (2026-05-30 plan)

Organizing question: §17.0 found aim rides the action packet's own target field (`block_pos`,
`entity_id`), and its perturbation already showed **+1 block = a deterministic miss**. So the
discrete target has no sub-unit scalar tolerance — the unit IS the floor. The real questions:
(1) how few bits does the target need *given obs* (the headroom), and (2) does coding it
obs-relative / as a pointer hold aim parity at that floor? This is the discrete-channel analog
of §16 — and unlike the mechanical, Baritone-path-following move stream that produced the §16
null, this channel carries genuine intent, so it is where a LEARNED codec might actually win.

**Inherited headroom (do NOT re-measure — the §16.0 decide-then-code discipline).** Rung-A
already sized the obs-predictability of both targets: §11a/§12.1 block pointer = crosshair
raycast **0.93–1.0**; §13.1 attack-target = **0.985** from `entity_set` geometry. §17.2
inherits these as the headroom and goes straight to the live knee; the new offline work is only
to convert those accuracies into bits-to-address (block: reach-6 volume ≈ 4 bits/axis when
coded obs-relative; entity: ≈ log2(N) for the candidate index) and the residual-given-obs
(≈0 because of the 0.93–1.0 / 0.985 predictability).

### The two channels differ sharply in cost — and that asymmetry is the scoping finding

- **A. `block_pos`** (place via `use_item_on`, mine via `player_action`): plain ints, clean
  reconstruction (proven §17.0). The obs-relative coding (`block_pos − player_block`) needs
  **only player position, which is already in the passthrough's obs** → feasible with no
  substrate change.
- **B. `entity_id`** (attack via `interact`): the raw id is an arbitrary handle — it can't be
  quantized or perturbed directly (a bogus id makes the Java reconstructor fall back to the
  original packet, §17.0). It must be reparameterized to an **`entity_set` index** (the §13.1
  candidate order). **But the obs the passthrough sends the codec is pose-only**
  (`PlayerObsSnapshot` is explicitly "minimal pose; the heavy R3/R4 channels — block grid,
  entity set — are NOT here"). So channel B requires **NEW SUBSTRATE: plumb the R3 `entity_set`
  into the passthrough's obs payload.** That plumbing is the gate on the interesting result.

### Sub-rungs (deliberately LIGHT on A, INVEST in B)

- **17.2.1 — `block_pos` live knee (one confirmatory sweep, mostly §17.0 reuse).** Add a
  `block_pos` quantizer to the sidecar (parallels `quantize_move`), two codings: *absolute
  scalar* (predict a CLIFF at 1-block resolution — the §17.0 +1 perturbation is already its
  first data point) and *obs-relative* (offset from player block, bounded by reach-6 → ~4
  bits/axis lossless). Metric = the §17.0 block harness (place-@T rate, post-server-sync scan).
  **Predicted headline:** no graded lossy tolerance — the only compression is the absolute→±6
  pointer reparam (lossless), exactly mirroring §16 "the reparam is the win." This is *largely
  confirmatory* and the writeup must say so. **Mining covered by symmetry:** `player_action`
  (dig) carries the identical `block_pos` field, so the place result transfers; no separate
  Baritone dig-aim driver (out of scope — place is the clean atomic block action).

- **17.2.2 — `entity_id` live knee (the discovery; gated on the obs plumbing).** Build the
  `entity_set`-index reparam in the codec: map `entity_id ↔ idx` (distance/geometry sorted, the
  §13.1 candidate order), quantize/collapse the index, reconstruct via `entity_set[idx] → its
  real entity_id → rebuild Interact` (always a VALID entity, possibly the wrong one). Lossy knob
  = index bits kept, down to "always the geometric prediction." Metric = the §17.0 attack
  harness **+ a decoy**: summon target T and decoy D, attack T, measure whose health drops
  (correct vs wrong target). **Predicted headline:** collapsing to the geom-predicted entity
  holds parity at ~0.985 (§13.1) → the entity target is genuinely obs-reconstructable LIVE — the
  first channel with real "predict the decision, not the packet" headroom on the wire, which is
  what hands to §18.

### Pre-registered outcomes
- `block_pos`: scalar cliff at 1 block; lossless obs-relative reparam to ~4 bits/axis; **no
  lossy sub-unit headroom** → a learned codec's only play is dropping the pointer and
  reconstructing from obs (§18, which needs the block grid in obs).
- `entity_id`: collapse-to-geom-prediction holds ~0.985 parity → **real live headroom** → §18
  has somewhere to win HERE, unlike the §16 move null. If instead it breaks parity at low index
  bits, the entity target is less obs-determined than §13.1's offline 0.985 implied (an
  online↔offline gap, itself a clean finding).

### Scope discipline (mirrors §17.3)
§17.2 finds the **knee + sizes the headroom**. Training the learned obs-reconstruction codec is
**§18**. **Reuse:** §17.0 block + attack harnesses, §14 Rung-2 passthrough, §16 sidecar config
pattern, the §13.1 candidate ordering, the inherited §11a/§13.1 accuracies. **New build:** (a) a
`block_pos` quantizer (small, parallels `quantize_move`); (b) **the R3 `entity_set` →
passthrough-obs plumbing** + the index-reparam interact codec (the real cost, and channel B's
gate); (c) the decoy attack harness (intended-vs-wrong-target metric). Runs are **non-peaceful
only for the attack/decoy arm** (spawn target + decoy passive); the block arm stays peaceful.
**Out of scope:** the §18 learned codec, a Baritone mining-aim driver (place covers `block_pos`),
combat-AI. **Sequencing:** 17.2.1 (block, cheap, confirmatory) → the obs plumbing → 17.2.2
(entity, the discovery). **Sprint ends at:** the two knees + the headroom verdict = "where a
learned obs-reconstruction codec has something to prove" → tees up §18, the learned
discrete-decision codec (the neural interface this doc is named for).

### 17.2.1 RESULTS — `block_pos` knee (2026-05-30, live, agent0 @ (6001,100,6000))

Confirmatory, as pre-registered. New codec: `experiments/codec_loop/blockpos.py` (`quantize_block_pos`,
the discrete analog of `quantize_move`) wired into the sidecar as a `blockpos` config mode
(`craft/codec/server.py`), driven live by `experiments/codec_loop/blockpos_knee.py`. Metric =
place-`@T` rate via post-server-sync world scan (placement is client-predicted; wait the
round-trip, then read truth). Fixed place task: one stone at T = player+3x, 5 trials/cell, PEACEFUL.
Wire integrity clean throughout: `substitute_errors=0`, `drift=0`, 35/35 substituted per cell.

| cell | coding | place_ok | landed `@T` |
|---|---|---:|---:|
| control | lossless identity | 1.00 | 1.00 |
| obsrel b6 | `block_pos − round(player)`, ±6 | 1.00 | 1.00 |
| obsrel b5 | ″ | 1.00 | 1.00 |
| **obsrel b4** | ″ | **1.00** | **1.00** |
| obsrel b3 | ″ | 0.00 | 0.00 |
| obsrel b2 | ″ | 0.00 | 0.00 |
| absolute b14 | raw world coord, ±8192 | 1.00 | 1.00 |
| absolute b8 | ″ | 0.00 | 0.00 |

**Headline (matches the pre-reg exactly): no graded sub-unit tolerance — a hard cliff at the bit
count where the dequant step crosses one block.** The block target has no scalar slack (the §17.0
+1 perturbation = a deterministic miss was its first data point; this is the full knee). The only
compression is the **obs-relative pointer reparam, which is lossless**: coding the offset vs the
player (bounded by the server's ~±6 reach check) reaches the floor at **~4 bits/axis**; the
absolute foil needs **~14 bits/axis** (3.5×) for identical parity, because it must resolve the
world-coordinate magnitude. This is the §16 result on the discrete channel — *the reparam is the
win, learning adds nothing on the pointer itself.* **Mining covered by symmetry:** `player_action`
(dig) carries the identical `block_pos` field via the same quantizer branch; no separate driver.

**Implication for §18 (and the gate it inherits):** a learned codec's only remaining play on
`block_pos` is **dropping the pointer entirely and reconstructing the target from obs** — i.e. the
"predict the decision, not the packet" move. For blocks that requires the **local block grid in
the passthrough obs** (which is pose-only today, the same plumbing gap that gates 17.2.2). So
17.2.1 confirms there is *no lossy headroom to chase here*, and points the genuine headroom at the
obs-reconstruction codec — exactly where 17.2.2 (`entity_id`, already obs-reconstructable to 0.985
offline) is predicted to be the first channel that pays. NEXT = the `entity_set`→obs plumbing.

### 17.2.2 RESULTS — `entity_id` knee + the live "predict-the-decision" headroom (2026-05-30, live, agent0 @ (6001,100,6000))

The gate first: the bounded R3 `entity_set` (nearest-first int network ids, the §13.1 candidate
order) is now plumbed into the codec-facing obs (homunculus `PlayerObsSnapshot`, commit `d60a9f7`;
captured tick-thread, serialized send-thread, radius 16 / limit 16). New codec:
`experiments/codec_loop/entity.py` (`quantize_entity_id`) wired into the sidecar as an `entityid`
config mode (`craft/codec/server.py`), three codings; driven live by
`experiments/codec_loop/entity_decoy.py`. Metric = **whose HP drops**: attack the intended cow,
scan all cows before/after, attribute the damage by position (NoAI/Silent cows → obs nearest-first
== scan nearest-first). Killaura OFF. 5 trials/cell, PEACEFUL. Path: `/attack_entity` →
`gameMode.attack` → `ServerboundInteract` → outbound mixin → passthrough → sidecar reparam → the
server damages the **reconstructed** entity.

**Phase A — index knee (confirmatory pointer reparam).** Row of 5 cows; attack the MIDDLE (idx 2,
a non-endpoint so a coarse index actually rounds to a neighbour).

| cell | coding | hit intended | hit other-real | subst | subErr |
|---|---|---:|---:|---:|---:|
| control | lossless identity | 5/5 | 0 | 21 | 0 |
| index b4 | idx into `entity_set` | 5/5 | 0 | 20 | 0 |
| **index b3** | ″ | **5/5** | 0 | 20 | 0 |
| index b2 | ″ | 0/5 | 5 (→ neighbour cow) | 21 | 0 |
| index b1 | ″ | 0/5 | 5 (→ nearest) | 20 | 0 |
| absolute b24 | raw network int, ±2²¹ | 5/5 | 0 | 20 | 0 |
| absolute b12 | ″ | 5/5 *(fallback)* | 0 | 16 | **5** |

Index pointer is **lossless to b3 = ⌈log2 5⌉**, then cliffs to a *different real entity* below
(b2 → the neighbour idx, b1 → nearest) — `subErr=0`, so these are clean substitutions onto a
wrong-but-real target, the entity analog of the §17.2.1 block neighbour-cell miss. The absolute
foil needs **~24 bits** (raw, unbounded, non-local int); below that the quantized id is BOGUS →
`level.getEntity` null → reconstructor null → the substitute **falls back to the original**
(b12: `subErr=5`, the 5 interact attempts errored and passed through — it can't even produce a
wrong-but-real hit). So the *only* compression is the **index-into-obs pointer: ~3 bits vs ~24**
— the §16 move-null / §17.2.1 block-pointer result carried onto the entity channel.

**Phase B — collapse / decoy (THE DISCOVERY).** `collapse` drops the pointer entirely and names
`entity_set[0]` (nearest = the §13.1 geometric argmax, ~0 index bits). T = intended, D = decoy at
distinguishable close range.

| cfg | geometry | cell | hit T | hit D (decoy) |
|---|---|---|---:|---:|
| B1 | T nearest | control | 5/5 | 0 |
| B1 | T nearest | **collapse** | **5/5** | 0 |
| B2 | D nearest, attack farther T | control | 5/5 | 0 |
| B2 | D nearest, attack farther T | index b4 | 5/5 | 0 |
| B2 | D nearest, attack farther T | **collapse** | 0 | **5/5** |

**Headline (matches the pre-reg): the `entity_id` channel has the live headroom `block_pos` did
not.** When intent coincides with geometry (B1, the ~98.5% case of §13.1) dropping the entire
entity-id pointer is **free** — collapse hits T 5/5 at ~0 bits. When intent *diverges* from
geometry (B2) the lossless pointer (control / index b4) **preserves intent** → hits T, while
collapse **honors geometry** → diverts to the nearer decoy D 5/5. That divergence IS the §13.1
~1.5% tail, made mechanical and visible on the wire: collapse reconstructs the attack target from
obs alone. This is **"predict the decision, not the packet" demonstrated LIVE** — the first wire
channel where reconstruct-from-obs genuinely pays (the block target needed a grid that isn't in
obs; the entity target's geometry already is). Wire integrity clean throughout (`subErr=0` except
the absolute-foil fallback, `drift=0`).

**Implication for §18:** the learned discrete-decision codec has a real target HERE — a head that
predicts the attack/interact target from the `entity_set` geometry (the §13.1 0.985 predictor)
and transmits only the **residual** when the operator's choice departs from the argmax (the B2
tail). The index pointer (~3 bits) is the lossless fallback; the geom-collapse (~0 bits) is the
compressed common case; §18 learns the gate between them. §17.2 closes: two knees mapped
(`block_pos` no headroom, `entity_id` real headroom), headroom located = where §18 wins.

## §18 — The learned discrete-decision codec (the neural interface)

This is the chapter the doc is named for. §16 found the move stream's compressibility IS the
deterministic obs-relative reparam (learning null, ≤0.27 b/pkt). §17.2.1 found `block_pos` has no
lossy headroom (the obs-pointer is lossless, nothing below it without a block grid not in obs).
§17.2.2 found `entity_id` DOES have headroom: the attack target is reconstructable from the obs
`entity_set` geometry — a learned predictor hits it where the trivial "nearest" prior does not.
§18 cashes that headroom as **a lossless predictive codec**: entropy-code the interact-target index
under a learned prior `P(idx | obs geometry)`. The achieved rate of such a coder is exactly the
prior's **cross-entropy** on the true index, `mean -log2 P(true_idx)`; behavioral parity is 100%
by construction (the coder always recovers the true index), so the entire result is the RATE. This
is the first channel where **learning pays** — the dual of §16's null.

- **18.0 — the prior's cross-entropy = the codec rate (offline baseline).** DONE (below). Read the
  frozen §13.1 target head (`results/rung_a_target_ckpt`, trained with `CrossEntropyLoss` so its
  softmax is a calibrated index prior) and measure `mean -log2 P(true_idx)` on the held-out split,
  raw + temperature-calibrated, vs uniform-pointer / nearest-bet / raw-int references.
- **18.1 — modeling in-world context: the bits g_t buys (DONE, below).** Reframed from "richer
  predictor" to the deeper point: §18.0's prior `P(target | geom, type)` is mode-BLIND. The true
  interact/attack target is `f(geom, type, g_t)` where `g_t` = the operator's *policy* — the Wurst
  KillAura **filter stack** (the embodiment-ladder authority interface: the neural model must faithfully
  EXECUTE against Wurst/Baritone settings, not invent targets). frozen_combat is single-mode so the
  prior never paid for that. 18.1 builds a MULTI-mode dataset (the same scene captured under
  filter-on / filter-off) and measures how many bits the `g_t` policy block buys a mode-aware prior
  over a mode-blind one. (The original "recover the best-epoch peak / recurrent context" sub-thread is
  deferred — the in-world-context question is the load-bearing one.)
- **18.2 — live predictive codec in the loop.** Wire the entropy-coding interact codec into the
  passthrough (reuse §17.2.2 sidecar + decoy harness): confirm lossless behavioral parity (100% by
  construction) and measure the live rate on real rollout interacts (the offline CE should transfer).

### Pre-registered outcomes
- The learned prior's calibrated CE ≪ uniform `log2(n)` ≪ raw `~24` → the learned predictive codec
  compresses the interact channel by ≫10× over the §17.2.2 lossless pointer, losslessly.
- It also ≪ the nearest-bet baseline → the compression is from LEARNING the non-trivial target
  function, not geometric triviality (nearest acc ~0.48). If instead CE ≈ nearest-bet, the head adds
  nothing over "guess nearest" and §18 collapses back to §17.2.2's trivial prior (a clean null).
- Entity type buys bits (geom+type CE < geom CE) → type is load-bearing for who you attack.

### 18.0 RESULTS — learned interact-target prior as a codec (2026-05-30, offline, frozen §13.1 ckpt)

`experiments/next_packet/rung_a_target_entropy.py`. Reuses the §13.1 feature pipeline
(`load_attacks`/`cand_features`) and reproduces the checkpoint's split EXACTLY (seed 42, val_frac
0.25 → the same held-out 65 of 260 combat ATTACK events). Rate = `mean -log2 softmax(scores)[true]`;
temperature fit on val (the 200-epoch/195-sample head is overconfident, so calibration is the honest
codec rate). Avg candidates/event = **49.7** (dense combat `entity_set`).

| prior | val acc | bits/interact (val) | note |
|---|---:|---:|---|
| raw network id | — | **24.00** | the §17.2.2 absolute foil (no obs) |
| uniform pointer `log2(n)` | — | **5.49** | the §17.2.2 lossless pointer floor |
| nearest-bet (lossless) | 0.48 | **3.85** | "guess nearest, send residual" — the trivial geom prior |
| §13.1 **geom** (CE) | 0.892 | 0.458 raw → **0.376** calibrated (T=1.81) | learned, geometry only |
| §13.1 **geom+type** (CE) | 0.923 | 0.286 raw → **0.218** calibrated (T=2.11) | learned, geometry + type |

**Headline: the learned prior compresses the interact target to ~0.22 bits/interact (calibrated,
held-out) — ~25× under the lossless `log2(n)`≈5.5 pointer and ~110× under the 24-bit raw id, while
staying LOSSLESS.** 77% of events cost <0.1 bits (the confident-correct common case is ~free); the
rate lives almost entirely in the rare-miss tail (p95 ≈ 1.9 bits). It beats the trivial nearest-bet
(3.85 bits) by ~18× — **the compression is LEARNING the non-trivial target function**, not geometric
triviality (nearest is right only 48%). Type features buy ~0.16 bits over geom-only (0.218 vs 0.376)
→ entity type is load-bearing for the attack decision. This is the result §16 (move null) and
§17.2.1 (block no-headroom) could not produce: **on the discrete-entity channel, learning is the
compression.**

**Honesty / 18.1 hook:** the FROZEN checkpoint is the *final-epoch* weights (val 0.892/0.923), below
§13.1's reported best-epoch peak (0.954/0.985) — the deployed artifact undershot its own accuracy, so
0.218 bits is a conservative (achievable-today) rate; recovering the peak should push it lower. The
estimate is also data-starved (65 val events, wide CIs).

### 18.1 RESULTS — modeling in-world context: the bits g_t buys (2026-05-30, live agent0)

§18.0 measured the rate of a mode-blind prior. But the executor (Wurst KillAura) selects its target
under a *policy* — a stack of ~26 filter toggles + `Priority`, the embodiment-ladder authority
interface `g_t`. The same scene maps to a DIFFERENT target under a different filter; a prior that
can't see `g_t` must pay for that ambiguity. 18.1 quantifies the payment.

**Premise grounded** (`experiments/codec_loop/filter_flip.py`, live HP-drop attribution, frozen
NoAI/knockback-resistant mobs): toggling `Filter passive mobs` flips KillAura's attack target —
SOLO sheep → mode A(attack-passives) hits it 3/3, mode B(protect-passives) leaves it 3/3; DUEL
sheep@dx2 + zombie@dx4 (Priority=Distance pinned) → A hits the near **sheep** 3/3, B the far
**zombie** 3/3. Same geometry, target flips by `g_t` alone. (Findings: KillAura reach ≈ 4.25 blocks;
default `Priority` preferred the farther hostile over the nearer passive → Priority is *also* `g_t`;
the label is "KillAura's pick under the active policy," not "nearest.")

**Dataset** (`experiments/codec_loop/filter_capture.py`): 159 matched scenes (318 rows), each a random
passive+hostile mix at varied in-reach offsets, captured under TWO modes — `attack_all`
(filter_passive off) and `protect_passive` (filter_passive on), `Priority=Distance` pinned. Label =
KillAura's observed pick. **51.6% of scenes flip** (label differs across modes). Two substrate quirks
debugged en route, both real: (1) **knockback** — `NoAI` doesn't stop it, so KillAura's first hit
shoves the priority mob and it sprays the scene; pinned with `knockback_resistance=1`. (2) a
**stale-filter first swing** — KillAura gets one swing under the *previous* filter before the new one
settles (~1 tick), which was biasing the discard toward exactly the flip scenarios; fixed by a settle
delay + rejecting non-attackable first-swings. Final dataset: 0 artifacts, 0 forced discards.

**Measurement** (`experiments/codec_loop/filter_bits.py`): per-candidate scorer (the §13.1
architecture), four feature arms, cross-entropy = codec rate (bits/interact), scene-split with a
held-out temperature-calibration split, seed-averaged (5). The `policy` feature is the broadcast
`filter_passive` bit — it helps ONLY via a learned `type×filter` interaction (an additive constant
cancels in the candidate softmax), so `+policy` tests whether the codec *discovers* the policy;
`+attackable` (= `1 − is_passive·filter`) hands it directly = the ceiling.

| arm | val_bits | val_acc | flip_bits | flip_acc |
|---|---|---|---|---|
| geom | 1.636 | 0.423 | 1.857 | 0.342 |
| geom+type (mode-blind) | 1.484 | 0.572 | 2.009 | 0.458 |
| **geom+type+policy** (mode-aware) | **1.049** | **0.692** | 1.082 | 0.711 |
| geom+type+attackable (oracle) | 1.031 | 0.708 | 1.093 | 0.753 |

**HEADLINE — STRUCTURAL (training-free, exact): `I(target ; g_t | scene) = 0.516 bits/interact`**
(flip_rate 0.52). The executor is deterministic, so `g_t` carries exactly the target-information that
identical `geom+type` features cannot — = mean per-scene target entropy across modes = flip_rate × 1
bit. This is a property of the matched-pair data, robust to any model.

**Learned-codec corroboration:** the trained `+policy` prior (1.049 bits) lands within **0.018 bits of
the attackable oracle** (1.031) — the MLP *discovered* the `type×filter` interaction from the raw
broadcast bit, without being handed `attackable`. The learned mode-blind→mode-aware gap is **+0.435
bits** on all val (recovering ~84% of the 0.516 structural MI; the shortfall is geometry-generalization
noise common to both arms) and **+0.927 bits on flipped scenes** (≈1 bit: the mode-blind prior must
coin-flip the contested pick; flip_acc 0.46→0.71). Type alone (mode-blind) only moves the rate
1.64→1.48 — it cannot resolve the flip, because the flip is about *policy*, not type.

**So:** on the discrete-target channel, modeling in-world context — the operator's filter policy —
buys ~0.52 bits/interact, and a learned codec recovers it to the oracle ceiling. This is §15's
"`g_t`-content is load-bearing" finding rendered as compression on §18's channel, and the dual of
§18.0 on a new axis: there *learning* was the compression; here *context* is. The codec MUST carry
`g_t` — which is what 18.2 needs live.

### 18.2 RESULTS — the live g_t codec: serving the prior on the wire (2026-05-30, live agent0)

18.1 stamped `g_t` harness-side. 18.2 reads it off the WIRE: the codec sidecar serves the learned
prior and entropy-codes real interacts under `P(idx | entity_set geom, type, obs.policy)`.

**Substrate (homunculus `fa4466c`):** `PlayerObsSnapshot.toJson` now carries `obs.policy` = KillAura's
target-selection policy (Priority + Range + all ~23 `Filter*` toggles), read tick-thread via the new
`Wurst.getSettingsMap` + `settingToJson` reflection path (the consistent homunculus↔Wurst API, no
`settings.json` parsing). Built, deployed to agent0 (both per-agent root + template), kill-by-PID →
deploy → relaunch. Verified live: 26-key policy on every packet, tri-state preserved
(`Filter neutral mobs='Off'`).

**Serving (`filter_prior_train.py` → `filter_prior.py` in `craft/codec/server.py`):** train+save the two
arms with a baked calibration temperature (geom+type T=3.37, geom+type+policy T=4.61 — these small
models are overconfident; the SERVED prior MUST carry T or its live raw CE blows up, esp. the sharper
policy arm). The sidecar gained an `interact_prior` config + `/interact_rate` readout: per outbound
interact ATTACK it scores the obs candidates and accumulates `-log2 P(true idx)`, auto-bucketed by the
live `obs.policy`. LOSSLESS — read-only rate, the index pointer reconstructs the target exactly
(entityid codec off → identity passthrough; **measured drift=0** over the run).

**Harness (`filter_live.py`):** arm the passthrough, then drive KillAura over 40 mixed scenes × both
filter modes under each served prior so real interacts flow. (Live obs is contaminated by other fleet
players + wild creepers — out-of-vocab; the codec restricts candidates to its trained species, so OOV
interacts skip and the modeled scene re-indexes cleanly.) 422 real interacts:

| arm | mean bits | attack_all | protect_passive | n |
|---|---|---|---|---|
| geom+type (mode-blind) | 1.485 | 1.792 | 1.179 | 200 |
| geom+type+policy (mode-aware) | **1.187** | 1.840 | **0.557** | 222 |

**HEADLINE: the live `g_t` codec compresses real interacts +0.299 bits/interact** over the mode-blind
prior — the §18.1 offline gain transfers to live wire data. The structure is exactly the policy: the
win is entirely in **protect mode** (0.557 vs 1.179 — knowing passives are excluded, the prior points
confidently at the hostile), while in attack_all (all attackable) the bit adds nothing (1.84≈1.79).
Lossless (drift=0). So the learned discrete-decision codec runs live, reads the operator's policy off
the wire, and pays exactly where the policy changes the decision — closing §18 end-to-end: a learned,
g_t-conditioned interact codec carries a real controller's target losslessly at a learned rate.

(Fleet caveat: only agent0 has the obs.policy jar; agents 1-9 run the stale pre-policy jar — don't mix
in obs-dependent work.)

## §19 — Neural takes the wheel: the codec as decision-MAKER, and its corrigibility

Through §18 the codec only ever *compressed* the executor's decision — KillAura picks the target, the
codec transmits it (losslessly, reading g_t). §19 takes the "predict the decision, not the packet"
thread to its end: the learned g_t-prior **makes** the decision. The codec substitutes the outbound
interact's `entity_id` with `entity_set[argmax P(idx | geom, type, g_t)].id` — the §17.2.2 collapse
path, but the index is the LEARNED g_t-conditioned argmax, not geom-nearest. The prediction *is* the
decision; KillAura's own pick is discarded on the wire. The question the embodiment frame
(g_t as authority interface, recurrence=corrigibility boundary) has been building toward: once a
neural model makes the call, does the operator still hold the wheel?

### Time dependency: same-tick, feedforward (and why that's correct here)
§19 makes the decision from the SAME-TICK obs the interact packet already carries (`entity_set`
geometry + `obs.policy`) — `P(target | same-tick obs)`, no history. This is not a shortcut: it is the
correct shape because the heuristic being replaced is itself feedforward — KillAura re-picks its
target every tick from the current candidates + filter settings, carrying no cross-tick state — and
§18.0 confirms the deciding information is fully present same-tick (0.985 accuracy; the ~1.5% residual
is geometry near-ties, not a memory gap). §19 is the "take the wheel" rung of the larger program:
bottom-up neural replacement of the Wurst/Baritone heuristic stack, starting with the cleanest
heuristic (discrete, same-tick, g_t-parameterized target selection). Time re-enters only at a LATER
rung — when we climb from a memoryless heuristic (KillAura) to a STATEFUL one (Baritone commits to a
path/goal and does not re-derive it each tick; the §12.3 intent-half-life / §13.2 handover seam). That
is where same-tick obs becomes insufficient and recurrence earns its keep — the temporal codec, not
this rung.

### Mechanism (reuse 18.2 sidecar + 17.2.2 substitution)
- Sidecar `entityid` gains a **`neural` substitution mode**: the served prior's argmax index →
  `entity_set[idx].id` becomes the decoded `entity_id`, `ok=true` forced (lossy substitution, like
  collapse). The Java `PacketReconstructor` resolves the id → Entity → the server damages the NEURAL
  pick `T'`. KillAura's intended `T` (original `fields.entity_id`) is logged for agreement.
- **g_t OVERRIDE** (the corrigibility control): a sidecar config field that forces the `filter_passive`
  value fed to the prior, DECOUPLED from `obs.policy` / KillAura's actual filter. Lets us hold the
  executor fixed while steering only the neural codec.

### Tests
- **A — effectiveness / agreement.** Combat scenes; measure (a) the agent lands hits on valid targets
  (whose-HP-drops, the §17.2.2 attribution) and (b) agreement `T'==T` with KillAura's own pick.
  Agreement ≈ the prior's live accuracy; divergence = the **controller substitution error** (the lossy
  cost of letting neural drive).
- **B — corrigibility (the headline).** Decoupled duel: KillAura's filter held FIXED (its pick `T`
  constant — passive nearest), while the codec's g_t override is flipped attack_all↔protect. If the
  neural pick `T'` flips passive↔hostile *with the override* while `T` stays put, the operator retains
  authority over the neural controller through the g_t interface — **corrigible**. If `T'` ignores the
  override, the neural decision has slipped the authority interface — a concrete corrigibility gap.

### Pre-registered outcomes
- **(Pass)** neural-driven attacks hit valid targets at ~the prior's accuracy; `T'==T` agreement high
  (>~0.9 live); AND `T'` tracks the g_t override (flips passive↔hostile) — the neural controller is
  both effective and steerable. The thesis culmination: a learned model owns the executor-level
  decision and remains corrigible via g_t.
- **(Informative failure)** if `T'` does NOT track the g_t override live — the prior under-weights the
  policy bit at decision-time, or live obs-policy lag — that is a real authority-interface gap worth
  surfacing, not a bug to hide.
- Report the divergence rate `T'≠T` = the lossy controller-substitution error.

### Substrate needs
1. `entity.py` / `server.py`: `entityid` `neural` substitution mode (argmax of the served prior →
   reconstruct entity_set[idx].id, force ok=true) + a `gt_override` config on the served prior.
2. `neural_wheel.py` harness: decoy scenes + whose-HP attribution (17.2.2) + the decoupled-g_t
   corrigibility sweep + KillAura-agreement logging; run live on agent0.
3. Fleet caveat persists (agent0 obs.policy jar only).

### §19 RESULTS — neural takes the wheel, and it's corrigible (2026-05-30, live agent0)

Commits: §19.1 sidecar `a679d17`, §19.2 harness `eb3bbde` (+ pick-bucket fix folded in).
Driver `experiments/codec_loop/neural_wheel.py`; artifact `results/sprint19/neural_wheel.json`.
Prior = the §18.1 `geom+type+policy` bundle (`results/sprint18/prior/prior_geom_type_policy.pt`,
val 0.985, calibration T=4.61). Live on agent0 @ (1.5,64,-2.5), difficulty easy + night,
`doMobSpawning false` (wild-spawn contamination control — see "Caught" below).

**Implementation note (deviation from the pre-registered mechanism, functionally identical).** The
substitution rides the **`interact_prior` config** (`substitute=true`), NOT a new `entityid` `neural`
mode — the served prior is already loaded + scored in that block, so the argmax pick overwrites the
decoded `entity_id` there (force `ok=true`, lossy). `gt_override` (bool|null) forces the
`filter_passive` fed to the prior, decoupled from `obs.policy`. null = the §18.2 passive observer
(byte-identity), so §18.2 is unchanged.

**TEST A — effectiveness / agreement (gt_override=None; neural reads the WIRE g_t).** 12 duels/mode,
sheep@1.5 (passive, nearer) + zombie@3.5 (hostile, farther), Priority=Distance.
- wire=attack_all:      `argmax_acc=1.0` (35/35) — neural's pick == KillAura's actual target on every
  interact; both mobs take damage (KillAura cycles both, neural mirrors it). Substitution = identity.
- wire=protect_passive: `argmax_acc=1.0` (34/34); sheep PROTECTED, zombie hit.
- VERDICT (Pass): serving neural as the controller **reproduces** KillAura when fed the same g_t —
  faithful + effective, hits land (whose-HP attribution). Divergence (controller substitution error)
  = 0 at this geometry. (Off the calibrated argmax, §18.1 already bounded the tail ~1.5%.)

**TEST B — corrigibility, THE HEADLINE (KillAura filter FIXED attack_all; flip only the codec g_t).**
KillAura's own filter never changes — it keeps aiming at the nearest sheep all run. Flipping the
CODEC's `gt_override` flips whose HP drops, 36/36 each, perfectly clean:

| codec g_t (override) | neural picks | passive (sheep) | hostile (zombie) |
|---|---|---|---|
| attack  (`False`) | sheep ×36 | **HIT** (Δ2.0)        | hit (Δ1.88) |
| protect (`True`)  | zombie ×36| **PROTECTED** (Δ0.0)  | hit (Δ2.19) |

The neural controller re-points every swing off the sheep KillAura is attacking and onto the zombie —
**protecting a passive the heuristic would have killed**, purely on the codec's g_t, with the
heuristic's filter held constant. `passive_hit_frac` flips 1.0→0.0. The operator retains authority
over the neural controller through the g_t interface: **corrigible**, proven on the server by
whose-HP-drops (not the wire field — the server trusts the substituted `entity_id`, the §17.0
aim-carrier / `reference_server_trusts_client_target` mechanism, so the hit lands on `T'` even though
the player is rotated toward KillAura's `T`).

This is the §18→§19 thesis culmination: a learned model owns the executor-level **decision** (which
entity dies), reproduces the heuristic it replaces, and **remains steerable through the same g_t
authority interface the heuristic exposes**. Same-tick feedforward throughout — no cooldown/temporal
model (Wurst still originates every swing + owns attack timing; neural only selects the target).

**Caught (live, fixed before the headline).**
1. **Stale sidecar.** The first live run measured *pure KillAura* (sheep protected only in protect mode
   = KillAura's OWN filter, no substitution; `neural_picks` stat absent). Cause: the `:25600` sidecar
   process was started in a PRIOR session and ran pre-§19.1 code — it silently ignored
   `substitute`/`gt_override`. Same class as "jar deploy over running" but for the Python sidecar:
   **a green import on disk ≠ the running process has it.** Fix: kill by PID (NOT pkill -f near the
   relaunch literal — the self-match trap), restart, verify the new stat keys are present, re-run.
2. **Saturated geometry.** sheep@2.0/zombie@3.8 → the prior picks the hostile in BOTH modes (no argmax
   flip; offline probe had flagged exactly this). The clean flip needs the passive clearly nearer:
   sheep@1.5/zombie@3.5 (attack p~0.65 sheep, protect p~0.95 zombie).
3. **Wild-spawn contamination.** At night a wild *skeleton* wandered into the obs, entered the prior's
   vocab+entity_set, and BECAME the argmax (`neural_picks={skeleton:36}`). Fix: `doMobSpawning false`
   + clear wild hostiles at setup (summoned mobs unaffected; keep night so the zombie doesn't burn).

§19 done & verified. NEXT (later rung): time re-enters when the target is a STATEFUL heuristic
(Baritone path/goal commitment, the §12.3/§13.2 seam) — the temporal codec, not this one.

## §20 — The stateful rung: predict the PLAN, not the packet stream (§20.0 + §20.1a DONE)

§18/§19 conquered the **memoryless** heuristic (KillAura): a same-tick, g_t-parameterized discrete
decision. The codec learned the decision (§18), then MADE it and stayed corrigible (§19). But §19's
corrigibility was *trivially* free: a feedforward controller re-reads g_t every tick, so it has no
state to diverge — it cannot slip authority. §20 climbs to the **stateful** heuristic (Baritone), where
the controller COMMITS to a goal and does not re-derive it each tick. Two things become non-trivial at
once, and they are the same thing:

1. **Compression (the codec).** §16 closed the per-tick MOVE packet (no learned headroom — a
   deterministic obs-relative reparam; a β-VAE buys ≤0.27b, §16.2 NULL). But §16 compressed each packet
   *in isolation*. Baritone emits thousands of move packets that are a deterministic function of ONE
   committed goal + current pos + terrain. The move **stream** therefore compresses to its generating
   **goal** — the temporal analog of §18's "predict the decision, not the packet," now at the PLAN
   level: **predict the plan, not the packet stream.** Headline = stream-bits / goal-bits (thousands of
   §16-obsrel move packets → one `GoalBlock` ≈ a few ints, or an index into a small waypoint set).
2. **Corrigibility (now a real result).** A committed controller CAN ride its old plan when the operator
   changes g_t mid-commitment. The latency before its goal updates = the corrigibility moat. §12.3/§13.2
   already built this instrument (per-rollout segment decoder on embodied/move features; the
   `rel = p_new/(p_old+p_new)` handover-latency crossover) and measured the **completion** seam (goal
   done → next) at ~6.4 ticks / 0.32s with NO interior moat decay. But §13.2.4 EXPLICITLY DEFERRED the
   corrigibility-relevant seam: peaceful data has only completions, "Override is the corrigibility-
   relevant seam (§6); a non-peaceful recapture is a next-sprint input." **That deferred override seam IS
   §20's headline test.** The moat = override-handover-latency − completion-handover-latency = how much
   longer a committed plan resists an operator INTERRUPT than it takes to roll over a natural completion.

### Why now / time dependency
This is where same-tick feedforward becomes insufficient and recurrence earns its keep. The move packet
at tick t is a function of (committed goal, current pos, terrain) — the goal is LATENT and PERSISTENT
across ticks, so a per-tick reader can't recover it from one packet; it needs the stream. KillAura was
memoryless (the §19 rung); Baritone carries state (this rung). The "take the wheel" verb is identical
to §19, only the decision is now a GOAL, not a target, and it persists.

### Substrate — already present, no new homunculus code for the offline tranche
- **Ground-truth goal is recorded per-tick.** `BaritoneState` (`/obs/sidecar`, `TickSidecarRecorder`)
  emits `goal` (`pathingBehavior.getGoal().toString()`), `pathing`, `goal_active`, and the driving
  process, one row/tick joined to the packet recording. So goal LABELS are free — read, don't infer.
- **Override is drivable.** `/baritone/goto` (`customGoalProcess.setGoalAndPath(GoalBlock)`) + re-issue
  mid-path = a clean, fully-controlled FORCED override seam (no waiting on organic hostile interrupts).
  `/baritone/stop` cancels. (Organic §6 interrupt — evasion/water_aversion cancelling Baritone — is a
  secondary, optional capture.)
- **Decoder + latency instrument exist.** `rung_c_moat.{load_rollout,segments,featurize}` +
  `rung_c_transition` (rel-crossover). Reuse verbatim so inputs match the validated NO-moat-decay result.

### §20.0 (offline) — the goal codec + the OVERRIDE-seam moat  [PRIMARY DELIVERABLE]
1. **Capture** move-streams with goal transitions INCLUDING forced overrides: drive `/baritone/goto A`,
   then mid-path `/baritone/goto B`; arm packet recording + obs sidecar; per-tick rows = {MoveAction
   fields, `baritone_state.goal` ground truth, pos, obs}. Mix completion seams (let A finish → goto C)
   and override seams (interrupt A with B) so the two are A/B-comparable on one capture.
2. **Compression headline:** stream-bits (Σ §16-obsrel MoveAction bits over a segment) vs goal-bits
   (code the goal as a `GoalBlock` delta or an index into the rollout's waypoint set). The goal-level
   headroom §16 never measured.
3. **Override-seam moat (THE HEADLINE):** run the §13.2 rel-crossover on OVERRIDE transitions; compare
   handover latency to the §13.2 COMPLETION latency (~6.4t). The gap = the corrigibility moat.

### §20.1 (live) — neural takes the wheel on NAVIGATION  [SCOPED, gated on §20.0 (DONE)]
The §19 analog one rung up: §19 served the g_t-prior as the live CONTROLLER of a MEMORYLESS decision
(attack target) and proved corrigibility by whose-HP-drops; §20.1 does it for the STATEFUL decision
(the Baritone goal) and proves it by **whose-WAYPOINT-the-body-converges-on**.

**Settled architecture:**
1. **The decision is a LATENT GOAL → the wheel is at the CONTROL layer, not a wire rewrite.** §19 could
   overwrite `entity_id` because the target was a FIELD on the packet the server trusts (§17.0,
   [[reference_server_trusts_client_target]]). A nav goal lives in Baritone's `customGoalProcess`, on NO
   move packet. So §20.1's "take the wheel" = the served codec ISSUES the goal via the §20.0
   stop+repath override — a control-loop policy, NOT a per-packet substitution. This is SIMPLER than §19
   in plumbing: no netty send-thread, no per-packet latency budget. A driver polls obs, the prior
   decides, the driver commands `/baritone/goto`. No packet sidecar needed for the override.
2. **Decisive metric = whose-waypoint-the-body-converges-on** (final position scan) — the navigation
   twin of whose-HP-drops. The body physically arriving at the NEURAL goal (not the operator's) = neural
   took the navigation wheel.
3. **g_t = the authority interface; `gt_override` forces it.** Scene = two typed beacons (the §19/§20.0
   duel geometry); the served prior selects WHICH beacon is the goal conditioned on g_t; `gt_override`
   forces that g_t decoupled from the operator's command. Default nav g_t = **seek/avoid** — the
   structural twin of §18.1's attack/protect: the g_t bit flips the goal between beacon-A and beacon-B.
4. **The LIVE moat = override the goal mid-path, measure ticks-until-the-body-changes-course.** §20.0
   (offline) found NO body-level moat (stop+repath redirects in ~2t/0.1s; Baritone commits in PATH STATE,
   not body momentum). §20.1 measures the moat of the LEARNED CONTROLLER itself: a live neural loop must
   NOTICE the g_t change before it can act, so the live moat = body-redirect (~2t) + the controller's
   **decision-cadence lag** (poll interval + inference). This is the corrigibility property §20.0's
   offline decoder could NOT see (it had instantaneous g_t). Headline: **the corrigibility moat of a
   stateful neural controller is set by its decision cadence, not body inertia.**

**Staging (fork RESOLVED 2026-05-30 → MVP authority-loop first):**
- **§20.1a — MVP authority-loop [FIRST].** Prove the live wheel + whose-waypoint flip + live-moat with the
  goal served directly through `gt_override` (the "decision" is the served g_t→beacon map,
  geometry-trivial), de-risking the hard substrate part (the live control loop + the corrigibility
  measurement) ahead of the model. Faster to a decisive live corrigibility number.
- **§20.1b — trained nav goal-prior [SECOND, grafted onto 20.1a].** Capture (scene-geometry, g_t,
  chosen-goal) over the beacon duel, train a §13.1/§18.1-style index-over-candidates head conditioned on
  the nav g_t, serve its argmax in place of the trivial map → the genuinely-LEARNED decision (full §19
  analog). The 20.1a harness is the test rig; only the policy block swaps.

**Substrate — NO new homunculus code (like §20.0).** Reuses: `_summon` typed beacons (filter_capture),
poll obs (`/position` + sidecar obs), the stop+repath override (§20.0), position scan for whose-waypoint.
Path-A reuses `filter_prior_train`/`filter_prior` (§18.1) + the §13.1 head. The served codec is a
control-loop driver (the `neural_wheel.py` pattern — own the prior + `gt_override` + the wheel — but
reading obs and commanding Baritone instead of configuring the packet sidecar).

**Pre-registered outcomes:**
- **(Effectiveness)** `gt_override` = the operator's command → the body converges on the operator's
  intended beacon (the neural controller REPRODUCES the navigation heuristic) — the §19 TEST-A analog.
- **(Corrigibility, the headline)** operator's command held FIXED, flip `gt_override` → the body converges
  on the OTHER beacon (whose-waypoint FLIPS): the body obeys the codec's g_t authority, not the operator's
  command — the §19 TEST-B analog at the plan level.
- **(Live moat)** override mid-path → the body changes course within ~2t + the controller's poll/inference
  cadence. ≈ body-redirect = corrigible-by-default live (the §20.0 null transfers); ≫ = the cadence IS the
  moat — either way a quantified live corrigibility latency for a stateful learned controller.

#### §20.1a RESULTS — neural takes the navigation wheel, live; corrigible; moat = decision cadence

Built `experiments/codec_loop/nav_wheel.py`, live on agent0 (peaceful window, no new homunculus
code). A `NavWheel` control loop polls the authority command (`gt_override`, else the operator
command), maps it to a beacon, and ISSUES the goal via the §20.0 stop+repath override (a worker-thread
`/baritone/goto` + lock-bypassing `/baritone/stop`). Scene = two typed beacons (cow A at +12X, pig B at
−12X); whose-waypoint = the nearer beacon at rest (final `/position` scan). All three landed:

**TEST A — effectiveness (gt_override=None, codec honors the operator):** operator=A → body at A
(dA=3.85); operator=B → body at B (dB=1.46). **2/2.** The wheel faithfully executes the commanded goal.

**TEST B — corrigibility (the HEADLINE; operator FIXED=A, flip codec gt_override):**
**codec_gt=A → body→A 5/5; codec_gt=B → body→B 5/5.** With the operator's command unchanged, the
codec's `gt_override` decides whose-waypoint the body converges on — the navigation analog of §19's
passive_hit 1.0→0.0, proven by where the body physically ends up. The body obeys the codec's g_t
authority interface, not the operator's command. (First pass hit a 2/4 inter-arm
`SESSION_LOCK`-contention stranding — a prior arm's background goto held the lock when the next arm's
goto fired → `busy` → stranded `near=None`, NEVER `near=A`; corrigibility never lost to the operator.
Fixed by draining the lock between arms — `/baritone/stop` returns `acked=False` once Baritone is idle =
lock free — → clean 5/5.)

**LIVE MOAT — override mid-path, latency vs controller cadence:**

| cadence | moat | decision lag | body redirect |
|--------:|-----:|-------------:|--------------:|
| 0.05 s | 1.006 s | **0.043 s** | 0.962 s |
| 0.50 s | 1.357 s | **0.265 s** | 1.092 s |

The **decision lag scales with the poll cadence** (10× cadence → ~6× lag, ≈ the expected cadence/2 for a
flip landing at a random point in the poll interval); the body redirect is ~cadence-independent (~1.0 s,
the stop+repath + 180°-reversal execution). **The live corrigibility moat is set by the controller's
DECISION CADENCE, not body inertia** — exactly what §20.0 predicted (Baritone commits in path state, not
momentum; there is no irreducible body-momentum moat, so the moat is the rate at which the controller
re-reads g_t, which you can shrink by polling faster). This is the corrigibility property §20.0's offline
decoder — having instantaneous g_t — structurally could not see. (Note: the moat measures course-change
onset; at the slow cadence the body did not always complete the full ~24-block return inside the 9 s
window — `final_near=None` there is mid-trip, not a failed override; fast-cadence runs landed B 4/4.)

Honest asymmetry from §19, restated: Baritone does NOT autonomously navigate (it only goes where
commanded), unlike KillAura which autonomously swings — so §20.1 is less "wrest the wheel from an active
competing heuristic" and more "the codec IS the goal source AND it is corrigible (obeys a g_t change
mid-commitment)." The gt_override=None baseline (codec honors the operator) is the competing-command
stand-in; setting gt_override is the override.

§20.1a = MVP authority-loop (the decision is the trivial g_t→beacon map). **NEXT = §20.1b:** capture
(scene-geometry, g_t, chosen-goal) over the beacon duel + train a §13.1/§18.1-style index-over-candidates
head conditioned on the nav g_t, and serve its argmax in place of the trivial map — the genuinely-LEARNED
decision (full §19 analog). The 20.1a harness IS the test rig; only the policy block swaps. Driver
`experiments/codec_loop/nav_wheel.py`, artifact `results/sprint20/nav_wheel.json` (gitignored).

### Pre-registered outcomes
- **(Compression)** the move stream compresses to its goal at a large ratio (stream ≫ goal bits) — the
  plan-level headroom exists, parallel to §18's discrete headroom and unlike §16's per-tick null.
- **(Moat / corrigibility, the headline)** override-handover-latency ≥ completion-latency. STRICTLY
  greater = a committed plan carries real interruption inertia (a measurable corrigibility moat, the
  recurrence/corrigibility boundary made quantitative). ≈ completion = no extra commitment inertia
  (corrigible-by-default even when stateful) — an equally publishable, thesis-relevant null.
- **(Informative failure)** if the goal is NOT recoverable from the move stream (decoder at chance), the
  plan is not legible from the body alone — surface it; do not paper over.

### Substrate needs
1. `experiments/.../goto_override_capture.py`: forced-override + completion capture via `/baritone/goto`
   + obs sidecar; per-tick {MoveAction, baritone_state.goal, pos, obs}. (No new homunculus code.)
2. Goal-codec measurement reusing `rung_c_moat`/`rung_c_transition` (decoder + rel-crossover) + a
   stream-vs-goal bits computation (§16 obsrel for the stream side).
3. §20.1 live take-the-wheel: a served goal-codec + a `gt_override` goal + a live override-latency
   harness (gated on §20.0).

Related: [[project_embodiment_design]] (recurrence = corrigibility boundary; g_t authority interface),
the §12.3 NO-moat-decay + §13.2 ~6.4t completion-handover results (this rung's direct ancestors),
§16 move-codec null (the per-tick floor this rung clears at the plan level), §18/§19 (the discrete
"predict the decision" the navigation channel now mirrors at the plan level).

### §20.0 RESULTS — goal codec offline: STRONG compression headroom + a clean corrigibility NULL

Built `experiments/codec_loop/goto_override_capture.py` (capture) +
`goto_codec_measure.py` (measure). Live on agent0 (port 25570), peaceful window,
no new homunculus code. Capture: 5 rollouts × 6 nav legs per mode, 100% packet↔
sidecar tick-join everywhere; **completion** = blocking `/baritone/goto` to arrival
then stamp next g_t (body AT REST at the flip); **override** = fire the goto on a
worker thread, ~2.5s in (`speed_at_flip≈0.15`, body MID-PATH) stamp the new g_t then
`/baritone/stop` (lock-bypassing) + re-path (body AT SPEED at the flip). g_t stamped
via `/obs/meta` = the nav-goal segment label; `baritone_state.goal` is the ground
truth (verified to track the stamp). Totals: completion 30 seams / 6460 moves,
override 25 seams / 1629 moves. Measure: §16 obs-relative per-packet bits (b=5,
the zero_preserving baseline-to-beat) for the stream side; rung_c_transition VERBATIM
(margin 40, holdout 10, bin 4, seed 0) for the rel-crossover.

**PART A — Compression: STRONG positive (the move stream compresses to its goal).**
A move segment is the run of move packets under one committed goal. Coding the goal
as an index into the controller's waypoint set (the nav analog of §17.2.2's pointer):

| mode | b/pkt | seg≈moves | stream bits | goal(index) | **stream/index** | goal(delta) | stream/delta |
|------|------:|----------:|------------:|------------:|-----------------:|------------:|-------------:|
| completion | 4.35 | 208 | 28068 | 64 | **437×** | 558 | 50× |
| override   | 1.83 | 47  | 2985  | 81 | **37×**  | 630 | 4.7× |

The ratio scales ~linearly with commitment length (per-segment: 66 moves→112×,
115→260×, …), because goal_bits is fixed (~2 b/segment) while stream_bits ∝ n_move.
This is the plan-level headroom §16 never measured: §16 coded each packet in
ISOLATION and found a per-tick null (~2 b/pkt, no learnable structure beyond the
reparam); here the stream pays that ~2-4 b/pkt **per packet** while the goal that
generates the whole stream pays it **once**. Parallel to §18's discrete-decision
headroom; unlike §16's per-tick floor. **"Predict the plan, not the packet stream"
is real — a committed Baritone move stream is ~40-440× redundant against its goal.**

**PART B — The override moat: a clean NULL (corrigible-by-default even when stateful).**
The §13.2 rel-crossover (`rel=p_new/(p_old+p_new)`), crossover offset measured from
the g_t-issue tick (= offset 0 by construction: the capture stamps g_t AT the seam):

| seam type | crossover | long-new | decoder acc | seams |
|-----------|----------:|---------:|------------:|------:|
| completion | 2.22 t (0.111s) | 1.97 t | 0.965 | 26 |
| override   | 2.40 t (0.120s) | 3.09 t | 0.996 | 30 |

**MOAT = override − completion = +0.18 t (+0.01s)** pooled; +1.1 t (+0.055s)
length-controlled (long-new) — at most ~1 tick of extra inertia, well inside the
~2-3 tick handover itself. Decoder interior accuracy 0.965/0.996 → the goal is
STRONGLY legible from embodied features (this is the NULL branch, NOT the
informative-failure "decoder at chance" branch). The mechanism: a forced override
redirects the body's decodable intent essentially as fast as a natural completion,
because **Baritone carries its commitment in PATH STATE, not body momentum** — MC
movement has negligible inertia, and `/baritone/stop`+re-path (the substrate's REAL
override, identical to what the evasion/water_aversion reflexes do) replaces the
path within ~0.1s. The pre-registered "≈ completion" null: the stateful executor is
corrigible-by-default at the move-stream level. (Caveat: this measures the
stop+repath override — the only override the lock-holding `/baritone/goto` substrate
exposes; a hypothetical soft in-place goal-swap is not available to test.)

**Secondary cross-check — the LLM origination latency, isolated.** Completion
handover WITHOUT an LLM in the loop = 2.2 t (0.11s) here; §13.2's completion handover
WITH the LLM = 6.4 t (0.32s). The ~4 t (~0.21s) difference is the LLM→Baritone
origination latency — quantifying exactly what the §13.1 neural-swap bypasses (the
body handover alone is ~2 t; the LLM turn is the other ~4 t).

Verdict: §20.0 lands BOTH pre-registered outcomes — the compression positive (large
stream/goal ratio, plan-level headroom exists) AND the corrigibility null (override ≈
completion, no commitment-inertia moat at the body level, with a fully-legible
decoder). The codec headroom for the move channel is at the PLAN level (predict the
goal), and the stateful executor surrenders authority on demand. §20.1 (live: serve
the goal-codec + gt_override goal, drive Baritone to a neural-inferred waypoint, the
whose-waypoint-the-body-converges-on analog of §19's whose-HP-drops) is the gated
follow-on. Drivers: `goto_override_capture.py`, `goto_codec_measure.py`;
data results/sprint20/{completion,override}/, results/sprint20/measure.json.

---

## §21 — Local-r navigation distillation: bootstrap toward a world-model navigator (§21.0, §21.1, §21.2 DONE)

§20 cut Baritone into a three-layer rate tower (goal → A* planner → path → follower → move
stream) and located the neural real estate: NOT the planner (don't relearn A* — it's the
substrate's expensive gift over an intractable world-graph search), NOT the follower (§16's
per-tick null), but the **goal/intent layer**. §20.1a served the goal as a control-layer wheel
and proved live corrigibility (whose-waypoint flips 5/5; moat = decision cadence, not body
inertia). §21 climbs from *selecting* a goal to *producing* navigation behavior — but only the
**local, tractable piece** of the planner, distilled from Baritone within the agent's action
envelope.

### North star (so the bootstrap can be checked against it)
A **temporal world model** that navigates against rich (eventually visual) input, where what
Baritone supplies explicitly today becomes EMERGENT PERCEPTION:
- "look out → that's the tree, head there" — distant-goal inference (destination unnamed, perceived).
- "see water ahead → bias toward shore" — obstacle perception + global re-bias, learned not searched.
- target-ID (which entity to go to / hit) is the SAME perceptual act at range — so this thread and
  the §18 entity-target thread CONVERGE in the world model.

The deep substitution: **perception replaces search.** Baritone's A* *is* a perfect structured
world-model-plus-planner; the mature model *looks* and infers what A* computes. So Baritone is not
replaced piecemeal — it is the TEACHER, and the arc weans the model off its explicit signals.

### The design invariant that keeps every rung in service of the north star
**Fix the prediction target; migrate the input source from oracle → perception.**
- **Target (frozen across the whole arc):** Baritone's optimal **window-exit subgoal** — where the
  planned path crosses radius r — as a distribution over local boundary cells. Same head, rung 1
  through the mature model. (Local action = move-to-subgoal; break/place fold in later as in-radius
  actions. r≈5–6 is doubly natural: the local-planning horizon AND the strike/use action-affordance
  radius — the affordance radius IS the horizon over which local planning is actionable.)
- **Conditioning, factored so global signals are removable + replaceable:**
  - `local terrain` — rung 1: structured `block_grid` (r-windowed); later: rendered frames (same target).
  - `goal signal` — rung 1: explicit bearing (oracle); later: ablated, then INFERRED from perception
    ("see the tree" = recover the bearing from the scene).
- **Capture once, ablate many.** The rung-1 capture records EVERYTHING later rungs need — full planned
  path, bearing, block_grid, AND frames, AND the time sequence — though rung 1 consumes only
  terrain+bearing. Rungs 2–4 become pure re-analysis, no re-capture.

The trick in one line: the bootstrap is "heavy-handed" *because* it hands the model the oracle bearing
and clean voxels; each later rung REMOVES one oracle crutch while holding the target fixed, until the
model reproduces A*'s local decision from raw input alone — at which point Baritone is pure fallback.

### The rung arc (migration of the conditioning source)
| Rung | Terrain input | Goal input | Proves | Status |
|------|---------------|------------|--------|--------|
| **§21.0** (next) | structured `block_grid` (r-windowed) | explicit bearing | the HORIZON CURVE — local-r distillation works; how far must you see? | bootstrap |
| §21.1 | structured | **ablated** | which goals are scene-inferable vs need the oracle (salient vs arbitrary) | sketch |
| §21.2 | **frames (visual)** | inferred | predict the SAME subgoal from pixels — first real world-model step ("see water → shore") | sketch |
| §21.3+ | visual + **temporal** | perceived | extended horizon via lookahead, distant-goal inference, anticipation | north star |

Orthogonal **serve-live** thread (the "take the wheel" lineage): once a rung predicts well offline,
serve it live with Baritone as the GLOBAL-REPLAN FALLBACK — the corrigible-local (re-reads the bearing
every window, like §19's memoryless feedforward) + inert-global (Baritone replans only when local gets
stuck, the §20 path-state inertia) hybrid. Gated per rung; never the first deliverable.

### §21.0 — tight scope for next session
1. **Substrate add (one piece):** expose Baritone's planned path per tick —
   `pathingBehavior.getCurrent().getPath().positions()` → `baritone_state`. Bounded, exactly analogous
   to the §17.2.2 `entity_set` plumbing. `block_grid` + `goal` already recorded.
2. **Capture:** drive Baritone to long random goals over VARIED terrain (hills/water/caves — flat makes
   r=1 trivial and teaches nothing; we WANT the non-locality). Record path + bearing + block_grid +
   **frames + the sequence** (the forward-investment for §21.1–3). Reuse the `goto_override_capture`
   driver pattern.
3. **Deliverable — the HORIZON CURVE (one experiment, three readings):** window-exit-subgoal prediction
   accuracy vs receptive radius r (sweep r=1…10), held-out terrain. It reads simultaneously as
   (a) the NAVIGATION HORIZON (at what r does local prediction saturate?), (b) the local-policy
   DISTILLATION accuracy, (c) the PATH-CODEC RESIDUAL — which RESOLVES §20.0's open caveat: it splits
   the overclaimed 437× into "compressible without A*" (where local prediction succeeds) vs
   "irreducibly global" (the residual = what perception must later supply).
4. **Pre-registered question:** does local r≈5 cover the common case (residual small; A* earns its keep
   only on rare long-horizon detours)? The residual-at-r≈5 IS the size of the job handed to perception —
   it sets the agenda for §21.1–3.

### Scope guards (explicitly NOT §21.0)
Visual modality (§21.2), bearing ablation (§21.1), serve-live, break/place actions (start move-only),
distant-goal inference, temporal prediction. All deferred — the capture is built so none need a
re-capture.

### Open design questions to settle at the TOP of next session (so it doesn't stall)
1. **Subgoal representation** — boundary-cell classification vs (heading + Δy + a "goal-inside-window"
   flag)? Lean boundary-cell; pin before capturing.
2. **Bearing encoding** — unit vector / sin-cos + distance bucket + "goal beyond window" flag (the
   minimal global signal we MUST feed, else we penalize the model for not knowing which way an arbitrary
   far goal is — not a local-planning failure).
3. **Terrain-variety capture recipe** — random long gotos seeded into mixed biomes; guarantee
   water/cliff/cave coverage.
4. **Frame capture in §21.0: yes/no** — lean yes (cheap insurance for §21.2); confirm.
5. **r-sweep range** — block_grid is r=10 today; sweep 1…10 or widen the grid first.

Supersedes the old §20.1b (train a nav goal-prior): §21 reframes the neural object from *selecting*
a substrate-provided goal to *producing* the local plan — the richer rung. Related: §20 (the rate-tower
cut + path-state corrigibility this builds on), §16 (the follower null below this), §18/§19 (the
"predict the decision" head reused for the subgoal; entity-target-ID converges with nav-target-ID in
the world model), [[project_embodiment_design]] (recurrence/corrigibility boundary; perception replacing
search is the maturation of the substrate-as-load-bearing thesis), [[reference_headless_observability]]
(the frame-grab the visual rungs need).

### §21.0 RESULTS — the horizon curve (DONE, verified)

**Substrate add (one piece, as scoped).** `BaritoneState.snapshot()` already read `positions()`
internally but emitted only `path_next`/`path_dest`; now it also emits `path_fwd` (the forward path
slice from the executor's current node, bounded to `PATH_FWD_MAX=96`) + `path_idx`. So the window-exit
subgoal at ANY radius is computed offline from one capture — the whole r-sweep replays without
recapture, exactly the §17.2.2 `entity_set` plumbing pattern. Built, deployed to agent0's per-agent
root, bounced agent0 only ([[feedback_jar_deploy_over_running]]); verified live (`path_fwd` of 26 nodes
on a test goto). The other agents run the stale jar — agent0 only, as ever.

**Capture (`nav_distill_capture.py`).** The TickSidecarRecorder line IS the dataset — after the add it
carries `block_grid` (r=10 terrain) + `path_fwd` (target) + `path_dest` (bearing), joined by tick. No
packet recording (the move stream is §16's follower null; the neural object lives at the path level).
12 rollouts, random_spawn across biomes (dark_forest, savanna, forest×3, plains×3, taiga,
sunflower_plains, snowy_plains, grove), 3 long random-heading gotos each — terrain VARIETY is
load-bearing (flat → straight path → trivial horizon by construction; the many `unreachable`/`timeout`
legs are the forced detours we want). **23 871 usable rows.** Throttled Xvfb frames captured in parallel
(0.5 s cadence, ~2 000 PNGs) — intended as forward-investment for §21.2, used by nothing in §21.0. **NB
(§21.1 follow-up): these frames have Baritone's path overlay ON (it ships on; the capture never toggled it),
so they are CONTAMINATED for §21.2 — the planned path is drawn at the goal = the answer on the input. §21.2
must recapture with `/baritone/render {visible:false}` (now wired into the driver). See §21.1's NEXT note.**

**Analysis (`nav_horizon.py`).** Predict Baritone's window-exit subgoal at a FIXED action radius
`target_r` from a terrain window of SWEPT radius `feat_r` — decoupling the two is the experiment (the
target/decision is held constant; only how far the head SEES varies, so accuracy(feat_r) is a clean
"how far must you look" curve). Held out by ROLLOUT (terrain generalisation). Target = the subgoal's
DEVIATION from straight-line bearing (16-way, centred on straight), features = a BEARING-ALIGNED local
map (rotate so goal-forward is canonical) of three channels — **walkable floor** (nearest standable
level), **blocked** (wall/trunk/void), **water**. Reported as the tail-averaged held-out accuracy (not
test-argmax — mild leakage on a small subset).

Two methodology corrections were load-bearing and are the lesson of this rung (an absolute-frame raw
heightmap gave an unreadable, overfit curve):
- **Bearing-aligned + relative-deviation** killed the overfit. Absolute-frame aggregate accuracy
  *declined* 0.54→0.32 with feat_r and CE *exploded* 4.9→12.8 b (the MLP relearning the bearing→sector
  map 16× with no weight sharing); the aligned/relative reframe flattened aggregate to ~0.7–0.77 stable
  and CE to ~3 b. The navigation-correct symmetry (one "given terrain ahead, deviate Δ" policy) is what
  generalises across biomes.
- **Walkable floor, not max-height.** Leaves/logs are solid, so a max-height "surface" made the tree
  CANOPY the terrain in every forested biome (half the capture) — Baritone walks UNDER the canopy
  weaving between trunks. Switching to floor + a blocked-column channel turned the detour-subset signal
  from erratic noise (0.0↔0.4) into a clean monotone rise.

**Three readings (one experiment).** Driver: `nav_horizon.py`; sweep `results/sprint21/sweeps/`; plot
`results/sprint21/horizon.png` (`nav_horizon_plot.py`).

1. **Most local navigation is bearing-trivial.** At the action radius the window-exit subgoal equals
   "head straight at the goal" (within ±1 sector) **76 %** of the time (target_r=5; detour_frac=0.24).
   And the detour fraction FALLS as the action radius grows — 26 % (r=3) → 24 % (r=5) → **13 % (r=8)**:
   a farther subgoal sees past local wiggles back toward the goal, so navigation gets MORE bearing-trivial
   the farther out you place the subgoal. The straight-line baseline is the right policy for the bulk of
   local nav; the residual is a minority of genuine detours.

2. **The navigation HORIZON tracks the ACTION RADIUS.** On the detour subset (where straight-line scores
   0 by construction), terrain's recovery of the subgoal rises monotonically with the feature window and
   then plateaus — and *the rate of rise and the radius at which it plateaus scale with `target_r`*:
   target_r=3 plateaus by feat_r≈3–4 (peak 0.146@5), target_r=5 by ≈5–6 (0.135/0.128@5, seed-stable),
   target_r=8 only reaches its plateau by feat_r≈8 (0.002→0.125@8). You must see out to roughly where the
   subgoal is — **no farther** (seeing past the action envelope does not help). The affordance radius IS
   the planning horizon, as the §21 design predicted. (Soft knee, not razor-sharp; the *ordering* across
   three independent action radii + seed-stability is the robust signal.)

3. **The detour residual is large (~0.87) — the size of the job handed to §21.1/§21.2.** Even at the
   horizon, the local floor/block/water map recovers only ~12–15 % of detour directions; ~87 % of
   detours are NOT predictable from local terrain. The signal is real (monotone rise, tracks `target_r`,
   seed-stable) but WEAK — local geometry within the action envelope barely dents the detour. This
   **resolves §20.0's open caveat**: the 437× "move-stream→goal compresses" figure presumed a decoder
   that re-runs Baritone (A* in the reconstructor), conflating the cheap move→path compression (no A*)
   with the expensive path→goal inversion (= A*). The horizon residual is exactly that inversion job: the
   local window explains a thin slice; the rest is global (the A* the substrate gifts) or needs richer
   perception. **That residual is the §21.1 (bearing ablation — which detours are scene-inferable) and
   §21.2 (visual — can pixels supply what the floor map can't) agenda**, and it is why the bootstrap is
   in-service of the world-model north star rather than a dead end.

**Artifacts.** `experiments/codec_loop/nav_distill_capture.py` (capture),
`experiments/codec_loop/nav_horizon.py` (analysis), `experiments/codec_loop/nav_horizon_plot.py` (plot);
`results/sprint21/capture/` (12 rollouts, sidecars + frames), `results/sprint21/sweeps/` (target_r×seed
JSONs), `results/sprint21/horizon.png`. Homunculus: `BaritoneState.java` `path_fwd`/`path_idx`.

### §21.1 RESULTS — the bearing-precision knee is FLAT (DONE, verified)

**Question (set by §21.0 finding #3).** The ~0.87 detour residual — is it a *missing-goal-signal* problem
(the head can't recover detours because it doesn't know the goal direction precisely enough) or a
*missing-terrain-information* problem (local structured terrain genuinely lacks the detour cause)? §21.1
is the bearing ablation that decides it, and it's PURE RE-ANALYSIS of the §21.0 capture — no recapture
(the capture-once-ablate-many design paying off as promised).

**Mechanism (`nav_bearing_ablation.py`).** §21.0's head sees terrain in a frame ROTATED so +forward
points at the goal, and predicts the subgoal as a DEVIATION from the true bearing. We ablate by aligning
that frame to the bearing QUANTIZED to k sectors while the TARGET stays the deviation from the TRUE
bearing — so the quantization error θ_true−θ_q ∈ [−π/k, π/k] is exactly the irreducible noise of "knowing
the goal direction only to resolution 2π/k". Sweep exact → 45° (k=8) → 90° (k=4) → 180° (k=2) → none
(k=1, world frame = the full ablation). The bearing-free gvec [log-dist, beyond-window] stays constant
across all arms, so only direction precision varies. Held out by rollout; 5 seeds; feat_r=6, target_r=5.

**Result — the knee is FLAT (`results/sprint21/bearing_knee.png`).** Detour-subset recovery is invariant
to bearing precision across the entire sweep:

| precision | detour ±1 | aggregate ±1 | CE (bits) |
|---|---|---|---|
| exact | 0.122 ± 0.010 | 0.686 | 5.23 |
| 45° | 0.149 ± 0.005 | 0.693 | 4.69 |
| 90° | 0.109 ± 0.010 | 0.672 | 5.04 |
| 180° | 0.109 ± 0.016 | 0.646 | 5.37 |
| none (k=1) | 0.122 ± 0.021 | 0.669 | 4.88 |
| bearing-only (terrain ablated) | **0.000** | 0.762 | 3.12 |

The **bearing-dependent gap (exact − none) = 0.000 ± ~0.02** — coarsening the goal direction to *nothing*
leaves detour recovery unchanged (the 0.109–0.149 spread is seed-noise with no monotone trend). So:

1. **The bearing's entire job is the bearing-trivial majority.** `bearing_only` (exact bearing, terrain
   ablated) scores **0.000** on detours and exactly the straight-line floor (0.762) on aggregate — it
   learns the constant "aim at the goal" and nothing else. A detour is BY DEFINITION a departure from the
   bearing, so the bearing carries zero detour information. Confirmed by construction and empirically.

2. **The §21.0 residual is NOT a goal-signal problem.** Detour recovery (~0.12) comes entirely from
   terrain and is the same whether the goal direction is known exactly or not at all. Knowing the goal
   more precisely does not unlock detours; the structured local terrain (floor/block/water within the
   action envelope) simply lacks the detour cause. **This redirects the §21.2 agenda**: vision's value is
   NOT bearing recovery ("see the tree → recover the heading" is cheap — even a 180°-coarse heading
   suffices), it is richer/farther TERRAIN perception — seeing the obstacle or global routing cue that the
   r=6 floor map cannot contain. The "look out, see the water ahead, bias toward shore" half of the north
   star is the load-bearing one; the "recover the bearing" half is nearly free.

3. **The local head barely pays for itself on aggregate.** Every terrain arm (~0.65–0.69 aggregate) sits
   *below* the always-straight baseline (0.762): the head trades straight-tick accuracy for marginal
   detour recovery (~12% of 24% = ~3% of ticks) that doesn't net out positive on the bulk metric. The
   detour subset is the only place local terrain earns its keep, and even there weakly — reinforcing
   §21.0 finding #1 (most local nav is bearing-trivial) and #3 (the residual is large and global).

**A methodological note.** This ablation keeps the relative-deviation TARGET (which presupposes the true
bearing to *define* what counts as a detour) and ablates only the INPUT frame's bearing alignment — so it
measures "does goal-direction precision help the head recover detours", not "could a bearing-blind agent
navigate at all" (it could not — without a goal it doesn't know which way to go). That is the right
question for the residual: the residual is detours, and detours are unmovable by goal precision.

**Artifacts.** `experiments/codec_loop/nav_bearing_ablation.py` (analysis, reuses `nav_horizon`'s feature
builders verbatim), `experiments/codec_loop/nav_bearing_plot.py` (plot); `results/sprint21/bearing_ablation.json`,
`results/sprint21/bearing_knee.png`. No substrate change, no recapture.

**NEXT (§21.2).** Visual modality — predict the SAME window-exit subgoal from frames instead of the
structured floor map. §21.1 sharpens the hypothesis: pixels should help on the DETOUR subset specifically
(richer obstacle perception at range), not on aggregate (the bearing already nails that), and the win is
terrain-information, not heading.

**§21.2 REQUIRES A RECAPTURE — the banked §21.0 frames are CONTAMINATED (caught by the user 2026-05-31).**
Baritone ships its in-world overlay ON (`renderPath`/`renderGoal`/`renderGoalXZBeacon`/`renderSelectionBoxes`/
`renderSelection` all true), and the §21.0 capture never toggled it — so every banked frame has the planned
PATH drawn as a line straight at the goal (visually confirmed: a red path line on the ground). That line IS
the window-exit subgoal painted on the input; a vision head would OCR the answer, not perceive terrain — a
false-positive generator for the whole "perception replaces search" thesis. The structured §21.0/§21.1
results are UNAFFECTED (they use block_grid, not pixels) — the contamination is confined to the visual
channel. Fix shipped: homunculus `/baritone/render {visible:false}` (BaritoneRenderHandler; pathfinder
still runs, only visuals change) is now wired into `nav_distill_capture.py` — forced OFF whenever `--frames`,
restored on exit, override with `--keep-overlay`. So §21.2 is a fresh JOINT capture (overlay-off frames +
the structured sidecar, tick-aligned in one run), NOT pure re-analysis. Secondary frame-hygiene TODO: the
Xvfb grab also caught the PrismLauncher log window in the corner — want a maximized/focused MC window (and
optionally the `/hud` clean-recording toggle) for the recapture. The banked frames remain useful as a
pipeline smoke-test only.

### §21.2 RESULTS — pixels don't crack the horizontal detour (DONE, first-pass)

The visual rung, finally on clean frames. **Capture (concurrent fleet):** `./fleet.sh cycle 20` +
`scripts/sprint21_visual_capture.sh` fanned `nav_distill_capture` across 19 in-world agents → 38 rollouts,
**17 biomes**, 66k with-path ticks, 4 497 frames, ~8 min wall-clock (vs ~70 min sequential — concurrent is
the permanent mode). All clean-frame switches held per agent: baritone/render off, `/hud` off, chat hidden
(`chatVisibility:2` per-instance), launcher console cropped out (grab cropped to the MC window rect),
Fullbright pinned (lighting as a controlled variable). Two capture bugs caught live: a shared `--seed` put
18/19 agents in ONE dark_forest (fixed: per-agent seed → 17 biomes); and the visual rung needs per-tick
camera **yaw** to anchor a first-person frame to the world-frame target — which turned out to already be in
the sidecar inside `entity_set`'s self-player entry (§17.2.2 records every entity's rotation), so
`nav_visual` reads yaw from there (a redundant top-level `yaw` field was also added; harmless).

**Analysis (`nav_visual.py`).** One sample per frame (nearest sidecar row by `captured_at_ms`), **3 709**
paired samples, 38 rollouts, by-rollout split (same as §21.0/§21.1). A small CNN over the 96px frame ⊕ the
goal direction relative to the camera yaw → the SAME deviation-from-bearing sector + Δy target. THE CONTROL
is `cam_only` (no pixels, the goal-relative-to-camera vector only) — the visual analog of §21.1's
`bearing_only`: the camera yaw already encodes Baritone's CURRENT heading (≈ "the subgoal is where I'm
already pointing"), so it's a near-free, *privileged* detour predictor (a plan readout, not perception).

| arm | detour ±1 | aggregate ±1 | Δy acc |
|---|---|---|---|
| structured (§21.0, cross-rung) | ~0.12 | — | — |
| **cam_only** (heading, no pixels) | **0.264 ± 0.021** | 0.866 | 0.564 |
| **full_visual** (CNN + cam vec) | 0.237 ± 0.017 | 0.848 | **0.646** |

**Three readings (`results/sprint21_visual/visual.png`):**

1. **Pixels add nothing to the horizontal detour.** full_visual − cam_only = **−0.03 ± ~0.02** (≈ 0, the CNN
   slightly *overfits* the horizontal head on 3.7k samples). The frame can't anticipate the detour beyond
   what the current heading already implies. This is the §21.2 form of §21.0/§21.1's verdict: the horizontal
   detour residual stays hard — **perception does not replace the horizontal search at this scale.**

2. **The camera heading dominates, and beats structured terrain (0.26 vs 0.12).** Because the window-exit
   subgoal at r=5 is ≈ where Baritone is already steering, simply knowing the current heading (the camera
   yaw) predicts the detour twice as well as a static r=6 floor/block/water map. The strongest signal for
   "the local plan" is a *readout of the plan in progress*, not perception of terrain.

3. **Pixels DO help on Δy — elevation is the one terrain signal the frame carries (0.65 vs 0.56).** The CNN
   is not inert: it reads vertical structure (cliffs/steps) that the horizontal heading can't encode. Vision's
   marginal value at this scale is *elevation*, not horizontal routing. (Suggestive — needs a Δy
   majority-class baseline to fully nail, since Δy skews to "level".)

**Honest scope (first-pass, locked for now):** small CNN, 3 709 samples, 96px, and a frame that looks down
the *current* heading (the obstacle forcing the NEXT bend may be peripheral or out of view); a larger model /
more data / higher res could move the horizontal story, though the Δy gain shows the CNN extracts terrain
when the signal is present. `cam_only` is a privileged plan-readout baseline — the kinder framing is
full_visual (0.24) > structured (0.12): pixels beat a static local map, but via the heading-correlated view,
not new terrain information.

**Arc takeaway across §21.0–21.2.** The window-exit subgoal is overwhelmingly "head where you're already
heading"; the genuine detours are a small, hard residual (~0.87 unrecovered) that neither finer goal
precision (§21.1), nor structured local terrain (§21.0, 0.12), nor pixels (§21.2, ~0 gain over heading)
crack. Baritone's global A* inversion is doing real work that local perception — at this scale — does not
reproduce. The "perception replaces search" thesis holds for the easy 76% (aim at goal) and for *elevation*,
but the horizontal-routing residual is where search still earns its keep.

**Artifacts.** `experiments/codec_loop/nav_visual.py` (analysis), `nav_visual_plot.py` (plot),
`scripts/sprint21_visual_capture.sh` (concurrent capture); `results/sprint21_visual/visual.json` + `.png`;
homunculus `TickSidecarRecorder` yaw/pitch add (redundant — `entity_set` already had it).

## §22 — The path-state navigation codec: predict the PLAN's residual, not perception (§22 Rung 1 DONE)

§21 closed a clean negative *from the perception side*: local terrain (§21.0), bearing precision
(§21.1), and pixels (§21.2) cannot predict the navigation detour — the only learnable local signal
was a **leak of Baritone's own heading** (a plan readout). §22 turns that around and asks the dual
question on the PLAN side, back in the spine's frame ("predict the decision, transmit the residual" —
§18/§19/§20.0): how cheaply does the controller's plan-state encode its near-future, and **where do
the irreducible bits live?** This is a pivot off the §21 rung-wall and back onto the codec/override
spine, carrying the one insight §21 surfaced: *the plan-state is the sufficient statistic for the
agent's near-future decisions* (the §20.0 "Baritone commits in path state" result, re-derived from
the perception side).

The structural fact (verified on the §21 capture, no recapture): Baritone paths in bounded SEGMENTS.
`path_dest` is a segment endpoint ~16-30 blocks ahead of a far `goal`; the agent walks the committed
`path_fwd` node list and re-invokes A* for a NEW segment only near the segment's end. So between
recomputes the stream is index-coded ≈ free (the §20.0 mechanism); **all the residual bits live at the
recomputes** — the navigation analog of §18's "operator departs the argmax".

### §22 Rung 1 RESULTS — locate + size the residual (DONE, verified)

Pure re-analysis on the `results/sprint21_visual/capture` (38 rollouts, **66 055** pathing ticks, 17
biomes). `path_codec.py` segments each rollout's plan-state stream into commit-runs by `path_dest`
change (goal-changes counted separately as operator re-commands), then measures the codec residual.

| metric | value | reading |
|---|---|---|
| **mean commit-run** | **167 ticks** (pooled), 191 (rollout-balanced), median 110, max 1202 | one transmitted event reproduces ~167 ticks = §20.0 commit-length factor, real nav data |
| **recompute rate** | 284 recomputes = **0.43 %/tick** (+ 74 operator goal-changes) | the residual is sparse |
| **within-run free-ness** | **0.996** committed-future coverage (n=65 659) | within a run, later `path_fwd` nodes ⊆ run-start set → pure consumption, ≈ 0 bits (index coding *empirically validated*, not asserted) |
| **detour fraction** | **5.6 %** of recomputes are >1 sector off goal; mean 0.34 sectors off; mean Δlen +10.3 nodes | 94 % of recomputes are benign extensions straight at the goal |
| **residual bits** | 1.33 b/recompute (marginal) × 0.0043 = **0.0057 b/tick** vs 4 b/tick raw = **697×** | marginal UPPER bound; Rung 2 conditions on plan-state to lower it |

**Three readings (`results/sprint22/rung1.png`):**

1. **The stream compresses ~167× and the within-run cost is ≈ 0 — measured, not assumed.** 0.996
   committed-future coverage means once a segment is transmitted, every subsequent tick in the run is the
   player deterministically consuming already-sent nodes. This is the §20.0 "37–437× commit-length"
   compression landing at ~167× on real multi-biome navigation, *with* the index-coding premise verified
   directly.

2. **The residual is sparse AND mostly benign.** 99.57 % of ticks carry no new plan information; of the
   0.43 % that are recomputes, 94 % are aligned extensions (the segment just grows +10 nodes straight
   toward the goal). Genuine detour *origination* is **5.6 % of recomputes ≈ 0.024 % of ticks** — a
   handful of events (16 in the whole capture).

3. **This relocates §21's detour and explains why perception leaked via heading.** §21's per-tick
   "detour" (≈15 % of ticks, uncrackable from perception) is the *consumption* of curvature that was
   **committed at a sparse recompute and then walked for free** (the 0.996 freeness). Perception
   "predicted" detours only via the heading because the heading reads a committed curve whose decision
   was made earlier, at a recompute, using global A*. So the detour *cause* localizes to a tiny set of
   recompute events — exactly Rung 2's target.

**What Rung 1 settles, and the Rung 2 hinge.** The codec's amortised cost is bounded by the recompute
residual (marginal 0.0057 b/tick). The whole question of whether the path-state codec is near-lossless
cheap — or whether the recompute carries irreducible global-search information — reduces to: **is the
recompute (especially the 5.6 % detour-origination) predictable from plan-state at the preceding tick?**
If yes → the residual collapses below the marginal bound, the plan is its own sufficient statistic, and
the live wheel (Phase B) is well-founded. If no → the recompute *is* the new information, and that
quantifies the global-A*-inversion bound at the plan layer (the §21 negative, now priced in bits).

**Artifacts.** `experiments/codec_loop/path_codec.py` (analysis), `path_codec_plot.py` (plot);
`results/sprint22/rung1.json` + `rung1.png`. No new capture, no homunculus change.
