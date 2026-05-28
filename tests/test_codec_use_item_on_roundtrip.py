"""Static round-trip for the ``use_item_on`` codec (ml.MD §4a, step 1).

Inline fixtures cover the six faces (UP/DOWN/N/S/E/W) and both hands, plus
the edge case where the cursor lands inside the block (rare but legitimate
on the wire). Corpus replay (``CRAFT_CODEC_CORPUS``) runs against any
recorder JSONL and exits cleanly when use_item_on is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.use_item_on import PACKET_TYPES, UseItemOnAction


# Use_item_on doesn't currently consume obs (the block_pos is encoded as
# absolute pending the local-block-grid observation channel), so a tiny
# placeholder is fine. When the pointer-into-obs scheme lands, this fixture
# will need a populated block-grid channel — see the docstring in
# craft/codec/use_item_on.py.
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


def _fields(
    *,
    hand: str = "MAIN_HAND",
    block_pos: tuple[int, int, int] = (-102_601, 68, 16_126),
    face: str = "UP",
    cursor: tuple[float, float, float] = (-102_600.5, 69.0, 16_126.5),
    inside: bool = False,
    world_border_hit: bool = False,
    sequence: int = 3,
) -> dict:
    return {
        "hand": hand,
        "block_pos": list(block_pos),
        "face": face,
        "cursor": list(cursor),
        "inside": inside,
        "world_border_hit": world_border_hit,
        "sequence": sequence,
    }


@pytest.mark.parametrize(
    "fields",
    [
        # Real captured packet (place cobble on top of (-102601, 68, 16126))
        _fields(),
        # Each of the 6 faces with cursor pinned on the matching axis
        _fields(face="DOWN", cursor=(-102_600.5, 68.0, 16_126.5)),
        _fields(face="NORTH", cursor=(-102_600.5, 68.7, 16_126.0)),
        _fields(face="SOUTH", cursor=(-102_600.5, 68.7, 16_127.0)),
        _fields(face="WEST", cursor=(-102_601.0, 68.7, 16_126.5)),
        _fields(face="EAST", cursor=(-102_600.0, 68.7, 16_126.5)),
        # Off-hand
        _fields(hand="OFF_HAND"),
        # inside=True edge case (raycast started inside the targeted block)
        _fields(inside=True, cursor=(-102_600.5, 68.5, 16_126.5)),
        # High sequence number — verify int round-trip is exact
        _fields(sequence=987_654),
    ],
)
def test_inline_roundtrip(fields: dict) -> None:
    action = codec.encode("minecraft:use_item_on", fields, _OBS)
    assert isinstance(action, UseItemOnAction)
    decoded = codec.decode(action, _OBS)
    assert codec.fields_close(decoded, fields), f"\nexpected: {fields}\ngot:      {decoded}"


def test_block_relative_cursor_in_unit_cube() -> None:
    """Sanity that the encoder produces cursor offsets in ``[0, 1]`` for an
    in-block hit — the design invariant the neural head will assume."""
    fields = _fields(face="UP")
    action = codec.encode("minecraft:use_item_on", fields, _OBS)
    assert isinstance(action, UseItemOnAction)
    for axis in action.cursor:
        assert -1e-6 <= axis <= 1 + 1e-6, f"cursor axis {axis} out of [0,1]"


def test_invalid_hand_rejected() -> None:
    with pytest.raises(ValueError, match="invalid hand"):
        UseItemOnAction(
            packet_type="minecraft:use_item_on",
            hand="THIRD_HAND",
            block_pos=(0, 0, 0),
            face="UP",
            cursor=(0.5, 1.0, 0.5),
            inside=False,
            world_border_hit=False,
            sequence=0,
        )


def test_invalid_face_rejected() -> None:
    with pytest.raises(ValueError, match="invalid face"):
        UseItemOnAction(
            packet_type="minecraft:use_item_on",
            hand="MAIN_HAND",
            block_pos=(0, 0, 0),
            face="SIDEWAYS",
            cursor=(0.5, 1.0, 0.5),
            inside=False,
            world_border_hit=False,
            sequence=0,
        )


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt), f"{pt!r} missing from registry"


# ---------------------------------------------------------------------------
# Corpus replay
# ---------------------------------------------------------------------------


def test_corpus_use_item_on_roundtrip() -> None:
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
            if entry["id"] != "minecraft:use_item_on":
                continue
            fields = entry["fields"]
            # Recorder captures of older versions stub unimplemented packet
            # types with {"_unimplemented": true}. Skip those — they aren't
            # codec-level failures, they're data from before the extractor
            # landed.
            if fields.get("_unimplemented"):
                continue
            total += 1
            obs = entry["obs"]
            action = codec.encode(entry["id"], fields, obs)
            decoded = codec.decode(action, obs)
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no use_item_on packets in corpus {path}")
    assert not failures, (
        f"{len(failures)} / {total} use_item_on round-trips failed; "
        f"first: {failures[0]}"
    )
