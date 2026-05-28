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

Discriminator-conditioned head masking: the policy emits all 11 type-bundles
in parallel; only the heads for the picked type contribute to the loss for
this instance. At inference, the head bundle for `argmax(p̂_type)` (or a
sample from `p̂_type`) is the one that fires.

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
   absolute mode.
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
  reach distances Baritone+Wurst actually use in captured rollouts.
- **`R_ent`** (entity set radius). Proposed `R_ent = 32`. Should it
  match supersense radius (ml.MD §5a)? Probably yes — the policy reads
  the same set the codec points into. Decide jointly with the supersense
  spec when that lands.
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

## 8. Where to read next

- [`ml.MD`](ml.MD) — full design rationale, hypotheses, the world-model
  framing this spec serves.
- [`README.md`](README.md) §"Codec / neural output" — the at-a-glance
  diagram of the same hierarchy.
- [`craft/codec/`](craft/codec/) — per-codec implementations
  (`base.py`, `move.py`, `use_item_on.py`, …) and the HTTP shim
  (`server.py`).
- [`tests/test_codec_server.py`](tests/test_codec_server.py),
  per-codec round-trip tests — the executable form of §1's invariants.
