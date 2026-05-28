"""Codec dispatch + Action protocol (ml.MD §4a).

Each per-type codec module registers a (encode_fn, decode_fn) pair under one
or more packet-id strings (e.g. ``"minecraft:move_player_pos_rot"``). The
public ``encode/decode`` functions look up the registered codec by packet id
on the input side, and by ``action.packet_type`` on the output side.

Why a registry rather than a switch: it lets each per-type codec live in its
own file with the per-type ``Action`` subclass, and lets us add types one at
a time without touching a central dispatcher. The registry is populated at
import time (per-module side-effect via ``register``).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Protocol, cast, runtime_checkable


@runtime_checkable
class Action(Protocol):
    """Structured neural action — a tagged union of per-type subclasses.

    Subclasses are concrete dataclasses defined per packet family. The
    ``packet_type`` attribute is the discriminator (the on-wire packet id,
    e.g. ``"minecraft:move_player_pos_rot"``). All other fields are
    pointers/deltas into the observation, type-conditioned.

    **Conventions every Action subclass should follow** (not enforced at the
    Protocol level so the runtime-checkable ``isinstance`` stays cheap):

      * ``__post_init__`` validates the action's invariants — enum membership,
        cardinality, wire-shape contracts. Constructing an inconsistent action
        raises; nothing downstream gets to wrap a broken value.

      * ``semantic_fields: frozenset[str]`` (property) returns the names of
        the fields whose values the neural head predicts and downstream
        consumers treat as meaningful. The set may depend on the action's
        own field values (e.g. ``hand`` participates only when present on the
        wire). Plumbing fields (sequence numbers, etc) are excluded.

      * ``_is_plumbing: ClassVar[tuple[str, ...]]`` (optional) lists field
        names that round-trip but are mechanically generated at packet
        construction (the canonical example is ``sequence`` on the use-item
        and dig packets). Convention only — codecs can omit if there's no
        plumbing.
    """

    packet_type: str


EncodeFn = Callable[[str, Mapping[str, Any], Mapping[str, Any]], Action]
DecodeFn = Callable[[Action, Mapping[str, Any]], dict[str, Any]]

_REGISTRY: dict[str, tuple[EncodeFn, DecodeFn]] = {}


def register(packet_types: list[str], encode_fn: EncodeFn, decode_fn: DecodeFn) -> None:
    """Bind a codec to one or more packet ids.

    A single codec module may handle multiple wire types — e.g. the move
    family covers four (pos, pos_rot, rot, status_only) with one (encode_fn,
    decode_fn) pair that disambiguates internally based on the packet id.
    Duplicate registration raises so accidental shadowing is caught at import.
    """
    for pt in packet_types:
        if pt in _REGISTRY:
            raise ValueError(f"codec for {pt!r} already registered")
        _REGISTRY[pt] = (encode_fn, decode_fn)


def is_registered(packet_type: str) -> bool:
    return packet_type in _REGISTRY


def registered_types() -> list[str]:
    return sorted(_REGISTRY.keys())


def encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    """Strict encode — raises ``KeyError`` for unregistered packet types.

    The encodability audit (test-ladder step 3 per §4a) will use a separate
    ``try_encode`` if we need a "skip unknown types" mode; the round-trip
    test depends on strict failure so an incomplete registry is loud.
    """
    try:
        enc, _ = _REGISTRY[packet_type]
    except KeyError as e:
        raise KeyError(f"no codec registered for {packet_type!r}") from e
    return enc(packet_type, fields, obs)


def decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    """Strict decode — inverse of ``encode``. Raises ``KeyError`` for unknown
    ``action.packet_type``."""
    try:
        _, dec = _REGISTRY[action.packet_type]
    except KeyError as e:
        raise KeyError(f"no codec registered for {action.packet_type!r}") from e
    return dec(action, obs)


def fields_close(a: Any, b: Any, *, atol: float = 1e-9) -> bool:
    """Recursive structural comparison with float tolerance.

    Generic over the shapes that show up in packet ``fields`` dicts:
    nested mappings, lists/tuples of numbers, scalar floats/ints/bools/str.
    Two floats are equal if ``math.isclose(a, b, abs_tol=atol, rel_tol=0)``;
    everything else falls back to ``==``.

    Used by every codec's round-trip test — delta-encoded position channels
    incur one subtract+add per axis, FP-stable but not bit-exact when the
    obs and packet differ in magnitude.
    """
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return False
        return all(fields_close(a[k], b[k], atol=atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(fields_close(av, bv, atol=atol) for av, bv in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        # bool is a subclass of int — guard against True == 1 false positives.
        return a is b or a == b
    if isinstance(a, float) or isinstance(b, float):
        # Branches above already excluded Mapping / list / tuple / bool, so
        # both sides are numerics by here. The cast silences the type checker
        # without weakening the runtime contract.
        return math.isclose(
            float(cast(float, a)), float(cast(float, b)), abs_tol=atol, rel_tol=0
        )
    return a == b
