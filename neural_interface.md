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
- **Done:** a val-acc number + verdict — pointer closes, **or** it's data-starved,
  which *quantifies* the Track-2 recapture need. Both are results.

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
- **Done:** frozen set on disk, manifest written, tick-join verified,
  `baritone_state.goal` non-null, narrated `g_t` distinct from tool.

### 12.3. Intent half-life / moat-width — the headline
On the narrated recapture: train a decoder `g_t ← (obs window)` and plot **decode
accuracy vs ticks-since-last-tool-call**. The decay is the moat width — how long
the planner's command stays legible in the embodied stream before the fast loop's
dynamics wash it out.
- *Why it needs 12.2:* on current data `g_t == current_tool`, so decoding is
  trivially "what is the body doing" — it measures tool *duration*, not intent
  *persistence*. Narrated intent separates them.
- **Done:** the curve + a one-line read ("intent legible ~N ticks → the symbolic
  layer reaches ~X of the rate tower").

### Deferred (presuppose a trained executor we don't have)
Closed-loop swap (needs the packet-injection path + a servo); emergent sub-goal
probe (`embodiment.md` §8 — needs a recurrent executor to probe); continuous-`g_t`
graft (`embodiment.md` §7 Q2).

**If time is tight:** 12.1 + the narrated arm of 12.2 alone still moves the ball.
