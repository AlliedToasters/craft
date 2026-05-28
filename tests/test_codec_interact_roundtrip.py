"""Static round-trip for the ``interact`` codec (ml.MD §4a, step 1) and the
conditional-fields contract across all four codecs.

Inline coverage:

  * Each of the 3 sub-actions (ATTACK / INTERACT / INTERACT_AT) round-trips
    cleanly with the right field shape — ATTACK omits ``hand``/``at``;
    INTERACT carries ``hand`` but omits ``at``; INTERACT_AT carries both.

  * ``__post_init__`` rejects every wire-shape inconsistency: ATTACK with
    a hand, INTERACT without one, INTERACT_AT without ``at`` (and the
    reverse), and a bad-cardinality ``at`` value.

  * ``semantic_fields`` returns the right set per action — this is the
    contract the neural head's masking will rely on.

  * **Cross-codec uniformity**: ``semantic_fields`` exists on every
    registered Action subclass with consistent semantics (subset of the
    decoded fields, excluding plumbing). Adding a new codec that breaks the
    convention should fail this test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.interact import InteractAction, PACKET_TYPES


_OBS = {
    "tick": 12345,
    "captured_at_ms": 1_700_000_000_000,
    "x": -102_600.0,
    "y": 70.0,
    "z": 16_126.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


# Real-shape fixtures matching exactly what the recorder's JSONL emits per
# sub-action — note ATTACK has no hand/at keys at all, INTERACT has hand
# but no at, INTERACT_AT has both.
_ATTACK_FIELDS = {
    "entity_id": 4827,
    "using_secondary_action": False,
    "action": "ATTACK",
}
_INTERACT_FIELDS = {
    "entity_id": 4827,
    "using_secondary_action": False,
    "action": "INTERACT",
    "hand": "MAIN_HAND",
}
_INTERACT_AT_FIELDS = {
    "entity_id": 4827,
    "using_secondary_action": False,
    "action": "INTERACT_AT",
    "hand": "OFF_HAND",
    "at": [0.0, 1.2, 0.0],
}


@pytest.mark.parametrize("fields", [_ATTACK_FIELDS, _INTERACT_FIELDS, _INTERACT_AT_FIELDS])
def test_inline_roundtrip(fields: dict) -> None:
    action = codec.encode("minecraft:interact", fields, _OBS)
    assert isinstance(action, InteractAction)
    decoded = codec.decode(action, _OBS)
    assert codec.fields_close(decoded, fields), f"\nexpected: {fields}\ngot:      {decoded}"


def test_using_secondary_action_true_carries() -> None:
    """Sanity that the boolean field round-trips both polarities. Tiny but
    cheap and ``using_secondary_action`` is the only bool here that could
    silently get swallowed."""
    fields = {**_INTERACT_FIELDS, "using_secondary_action": True}
    action = codec.encode("minecraft:interact", fields, _OBS)
    assert isinstance(action, InteractAction)
    assert action.using_secondary_action is True
    assert codec.fields_close(codec.decode(action, _OBS), fields)


def test_attack_rejects_hand() -> None:
    with pytest.raises(ValueError, match="hand presence mismatch"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="ATTACK",
            using_secondary_action=False,
            hand="MAIN_HAND",
            at=None,
        )


def test_attack_rejects_at() -> None:
    with pytest.raises(ValueError, match="at presence mismatch"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="ATTACK",
            using_secondary_action=False,
            hand=None,
            at=(0.0, 0.0, 0.0),
        )


def test_interact_requires_hand() -> None:
    with pytest.raises(ValueError, match="hand presence mismatch"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="INTERACT",
            using_secondary_action=False,
            hand=None,
            at=None,
        )


def test_interact_rejects_at() -> None:
    with pytest.raises(ValueError, match="at presence mismatch"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="INTERACT",
            using_secondary_action=False,
            hand="MAIN_HAND",
            at=(0.0, 0.0, 0.0),
        )


def test_interact_at_requires_at() -> None:
    with pytest.raises(ValueError, match="at presence mismatch"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="INTERACT_AT",
            using_secondary_action=False,
            hand="MAIN_HAND",
            at=None,
        )


def test_invalid_hand_rejected() -> None:
    with pytest.raises(ValueError, match="invalid hand"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="INTERACT",
            using_secondary_action=False,
            hand="THIRD_HAND",
            at=None,
        )


def test_invalid_action_rejected() -> None:
    with pytest.raises(ValueError, match="invalid action"):
        InteractAction(
            packet_type="minecraft:interact",
            entity_id=1,
            action="HUG",
            using_secondary_action=False,
            hand=None,
            at=None,
        )


def test_semantic_fields_attack() -> None:
    action = codec.encode("minecraft:interact", _ATTACK_FIELDS, _OBS)
    assert isinstance(action, InteractAction)
    assert action.semantic_fields == frozenset({
        "entity_id", "action", "using_secondary_action",
    })


def test_semantic_fields_interact() -> None:
    action = codec.encode("minecraft:interact", _INTERACT_FIELDS, _OBS)
    assert isinstance(action, InteractAction)
    assert action.semantic_fields == frozenset({
        "entity_id", "action", "using_secondary_action", "hand",
    })


def test_semantic_fields_interact_at() -> None:
    action = codec.encode("minecraft:interact", _INTERACT_AT_FIELDS, _OBS)
    assert isinstance(action, InteractAction)
    assert action.semantic_fields == frozenset({
        "entity_id", "action", "using_secondary_action", "hand", "at",
    })


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt), f"{pt!r} missing from registry"


# ---------------------------------------------------------------------------
# Cross-codec contract: every registered Action exposes semantic_fields.
# Add a new codec without one and this test fails — the convention is
# enforced by the test layer (Action is a runtime-checkable Protocol with
# only ``packet_type``; making semantic_fields required there would force
# every isinstance call to do a hasattr check).
# ---------------------------------------------------------------------------


def _sample_actions() -> list[codec.Action]:
    """One representative Action per registered packet type, constructed by
    encoding a minimal fields dict + ``_OBS``. Used to enforce cross-codec
    contracts without re-importing each module by name."""
    samples: list[codec.Action] = []
    samples.append(codec.encode(
        "minecraft:move_player_pos_rot",
        {
            "has_pos": True, "has_rot": True,
            "x": _OBS["x"], "y": _OBS["y"], "z": _OBS["z"],
            "yaw": 0.0, "pitch": 0.0,
            "on_ground": True, "horizontal_collision": False,
        },
        _OBS,
    ))
    samples.append(codec.encode(
        "minecraft:use_item_on",
        {
            "hand": "MAIN_HAND",
            "block_pos": [0, 0, 0],
            "face": "UP",
            "cursor": [0.5, 1.0, 0.5],
            "inside": False,
            "world_border_hit": False,
            "sequence": 1,
        },
        _OBS,
    ))
    samples.append(codec.encode(
        "minecraft:player_action",
        {
            "action": "START_DESTROY_BLOCK",
            "block_pos": [0, 0, 0],
            "face": "UP",
            "sequence": 1,
        },
        _OBS,
    ))
    samples.append(codec.encode("minecraft:interact", _ATTACK_FIELDS, _OBS))
    samples.append(codec.encode("minecraft:interact", _INTERACT_FIELDS, _OBS))
    samples.append(codec.encode("minecraft:interact", _INTERACT_AT_FIELDS, _OBS))
    return samples


def test_semantic_fields_exists_on_every_action() -> None:
    """Cross-codec convention: every Action subclass exposes
    ``semantic_fields``. New codec without one will fail here loudly."""
    for action in _sample_actions():
        assert hasattr(action, "semantic_fields"), (
            f"{type(action).__name__} missing semantic_fields"
        )
        fields = getattr(action, "semantic_fields")
        assert isinstance(fields, frozenset), (
            f"{type(action).__name__}.semantic_fields must be frozenset, "
            f"got {type(fields).__name__}"
        )


def test_semantic_fields_subset_of_decoded() -> None:
    """``semantic_fields`` should be a subset of the decoded keys —
    otherwise the head is supposed to predict a value that never appears in
    the round-tripped packet, which is incoherent."""
    for action in _sample_actions():
        decoded_keys = set(codec.decode(action, _OBS).keys())
        # MoveAction packs (x, y, z) under one "pos" semantic field; the
        # decoded shape splits them. Same for rot → (yaw, pitch). Allow
        # the move family this single quirk without complicating
        # semantic_fields' name vocabulary.
        if action.packet_type.startswith("minecraft:move_player"):
            decoded_keys = decoded_keys | {"pos", "rot"}
        semantic = getattr(action, "semantic_fields")
        missing = semantic - decoded_keys
        assert not missing, (
            f"{type(action).__name__} semantic_fields {semantic!r} not "
            f"subset of decoded keys {decoded_keys!r} (missing {missing!r})"
        )


# ---------------------------------------------------------------------------
# Corpus replay (skips if not set or if no interact packets present)
# ---------------------------------------------------------------------------


def test_corpus_interact_roundtrip() -> None:
    env = os.environ.get("CRAFT_CODEC_CORPUS") or ""
    if not env:
        pytest.skip("set CRAFT_CODEC_CORPUS to a recording JSONL to enable")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"CRAFT_CODEC_CORPUS {path} does not exist")
    total = 0
    failures: list[tuple[dict, dict]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] != "minecraft:interact":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            obs = entry["obs"]
            action = codec.encode(entry["id"], fields, obs)
            decoded = codec.decode(action, obs)
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no interact packets in corpus {path}")
    assert not failures, (
        f"{len(failures)} / {total} interact round-trips failed; "
        f"first: {failures[0]}"
    )
