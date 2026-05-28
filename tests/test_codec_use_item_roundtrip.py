"""Static round-trip for ``use_item`` (ml.MD §4a, step 1). Right-click in
empty space: hand + look angles + sequence (plumbing)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.use_item import PACKET_TYPES, UseItemAction


_OBS = {
    "tick": 1,
    "captured_at_ms": 0,
    "x": 0.0, "y": 64.0, "z": 0.0,
    "yaw": 0.0, "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


@pytest.mark.parametrize(
    "fields",
    [
        {"hand": "MAIN_HAND", "sequence": 1, "yaw": 0.0, "pitch": 0.0},
        {"hand": "OFF_HAND", "sequence": 12_345, "yaw": -135.5, "pitch": 60.25},
        # Wraparound yaw — codec must round-trip whatever the client puts on
        # the wire; the neural head normalizes.
        {"hand": "MAIN_HAND", "sequence": 0, "yaw": 359.999, "pitch": -89.999},
    ],
)
def test_inline_roundtrip(fields: dict) -> None:
    a = codec.encode("minecraft:use_item", fields, _OBS)
    assert isinstance(a, UseItemAction)
    # sequence is plumbing — excluded from semantic_fields by design.
    assert a.semantic_fields == frozenset({"hand", "yaw", "pitch"})
    assert a._is_plumbing == ("sequence",)
    assert codec.fields_close(codec.decode(a, _OBS), fields)


def test_invalid_hand_rejected() -> None:
    with pytest.raises(ValueError, match="invalid hand"):
        UseItemAction(
            packet_type="minecraft:use_item",
            hand="THIRD_HAND",
            yaw=0.0,
            pitch=0.0,
            sequence=0,
        )


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt)


def test_corpus_use_item_roundtrip() -> None:
    env = os.environ.get("CRAFT_CODEC_CORPUS") or ""
    if not env:
        pytest.skip("set CRAFT_CODEC_CORPUS to a recording JSONL")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    total = 0
    failures: list[tuple[dict, dict]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] != "minecraft:use_item":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            decoded = codec.decode(codec.encode(entry["id"], fields, entry["obs"]), entry["obs"])
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no use_item packets in corpus {path}")
    assert not failures, f"{len(failures)}/{total} use_item round-trips failed; first: {failures[0]}"
