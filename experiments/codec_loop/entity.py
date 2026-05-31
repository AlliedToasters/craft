"""§17.2.2 — lossy ``entity_id`` reparam codec (the discrete-ENTITY analog).

§17.0 found aim rides the action packet's own target field. For block actions
that field is ``block_pos`` (covered by §17.2.1 / ``blockpos.py``); for entity
actions (``interact`` ATTACK / INTERACT) it is ``entity_id`` — a raw client-side
network int (``Entity.getId()``), resolved server-side by ``level.getEntity(int)``.

This is the §16/§17.2.1 reparam pattern carried onto the entity channel, but with
a CRUCIAL difference that makes §17.2.2 the first channel with real "predict the
decision not the packet" headroom LIVE:

  * ABSOLUTE coding (the foil) — quantize the raw network int over a fixed span.
    Entity ids are large, unbounded, non-local integers, so resolving one exactly
    costs ~21+ bits. Below that the quantized id is BOGUS: ``getEntity`` returns
    null, the reconstructor returns null, and the substitute path falls back to
    the ORIGINAL packet (counted as a substitute_error). So absolute can't even
    produce a wrong-but-real hit — it just degrades to pass-through. The point it
    makes: the raw handle has no cheap coding.

  * INDEX coding (the pointer reparam) — code ``entity_id`` as its INDEX into the
    obs ``entity_set`` (nearest-first, the §13.1 candidate order), quantize that
    index over ``[0, limit)``, reconstruct via ``entity_set[idx].id``. The index
    ranges over a BOUNDED, small set (limit 16 → ~4 bits lossless), and every
    reconstructed id is a REAL entity from obs — so substitution always succeeds,
    and a too-coarse index resolves to a *different real entity* (a clean,
    observable miss, unlike the absolute foil's fallback). This is the §17.2.1
    block-pointer reparam on the entity channel: ~4 bits lossless, cliff below.

  * COLLAPSE coding (the discovery) — drop the pointer ENTIRELY and reconstruct
    the index from geometry: always ``entity_set[0].id``, the nearest entity =
    the §13.1 geometric argmax. ~0 index bits. This is the headroom block_pos did
    NOT have: §17.2.1 bottomed out at the lossless pointer because the block
    target is not predictable from pose (needs the block grid, §18). Here the
    entity target IS predictable from the obs geometry (§13.1 = 0.985), so the
    pointer is droppable — "predict the decision, not the packet." The decoy
    harness measures the failure mode: collapse honors geometry, so it diverts to
    a nearer decoy when intent diverges from the nearest entity (the 1.5% tail).

Only ``InteractAction`` is touched; any other action passes through. If the
``entity_id`` is not present in the obs ``entity_set`` (out of radius/limit), the
action passes through unchanged for index/collapse (no pointer to express it) —
counted by the caller so a starved obs can't masquerade as a codec result.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from craft.codec.interact import InteractAction

# Absolute-coding span (the foil): entity network ids are arbitrary unbounded
# ints; a representative live id is ~2e6 (≈2^21), so resolving one exactly costs
# ~22 bits/id. The point is the ~22-vs-4 gap = the pointer reparam, same shape as
# the §17.2.1 block_pos 14-vs-4 absolute foil.
ABS_ID_RANGE = 2_097_152.0  # 2^21


def _entity_ids(obs: Mapping[str, Any]) -> list[int]:
    """Nearest-first int network ids from the obs entity_set (§17.2.2 gate)."""
    eset: Sequence[Mapping[str, Any]] = obs.get("entity_set") or []
    return [int(e["id"]) for e in eset if "id" in e]


def _quant_index(idx: int, *, bits: int, n: int) -> int:
    """Uniform-quantize an index over ``[0, n-1]`` to ``bits`` levels, round back.

    n = number of entities in obs. At ``bits >= ceil(log2(n))`` the step is <1 →
    lossless (every index reconstructs exactly). Below that the step exceeds 1 →
    the index rounds to a NEIGHBOUR → a different (real) entity. bits<=0 collapses
    to index 0 (the nearest entity = the geometric argmax)."""
    if n <= 1 or bits <= 0:
        return 0
    hi = n - 1
    levels = (1 << bits) - 1  # 2^bits - 1 spacing endpoints
    if levels <= 0:
        return 0
    step = hi / levels
    q = round(round(idx / step) * step) if step > 0 else 0
    return max(0, min(hi, int(q)))


def _quant_id_absolute(entity_id: int, *, bits: int, span: float) -> int:
    """Quantize the raw network int over a fixed ±span (midtread), round to int.
    Below ~ceil(log2(span)) bits the step exceeds 1 → a bogus id that resolves to
    no entity → the reconstructor falls back to the original packet."""
    levels = (1 << bits) - 1
    if levels <= 0:
        return 0
    step = (2.0 * span) / levels
    q = round((entity_id + span) / step) * step - span
    return int(round(q))


def quantize_entity_id(action: Any, obs: Mapping[str, Any], *, bits: int,
                       mode: str = "index", limit: int = 16,
                       abs_range: float = ABS_ID_RANGE) -> tuple[Any, str]:
    """Lossy copy of an ``InteractAction`` with ``entity_id`` reparam'd.

    Returns ``(action_or_copy, status)``. status ∈ {"applied", "passthrough_type",
    "passthrough_no_obs", "passthrough_not_in_set"} so the caller can tell a real
    codec result from an obs-starvation pass-through.

      * mode="index"    — pointer into entity_set, quantized over [0,limit) to
                          ``bits`` levels. Lossless at bits>=ceil(log2(n)).
      * mode="collapse" — always entity_set[0] (nearest = §13.1 argmax). bits
                          ignored. The "drop the pointer" extreme.
      * mode="absolute" — quantize the raw id over ±abs_range (the foil).
    """
    if not isinstance(action, InteractAction):
        return action, "passthrough_type"

    if mode == "absolute":
        new_id = _quant_id_absolute(action.entity_id, bits=bits, span=abs_range)
        return replace(action, entity_id=new_id), "applied"

    ids = _entity_ids(obs)
    if not ids:
        return action, "passthrough_no_obs"
    n = min(len(ids), limit)
    ids = ids[:n]

    if mode == "collapse":
        return replace(action, entity_id=ids[0]), "applied"

    if mode == "index":
        if action.entity_id not in ids:
            # Target outside the observed set — no pointer can express it. Pass
            # through so a starved obs is not scored as a codec hit/miss.
            return action, "passthrough_not_in_set"
        idx = ids.index(action.entity_id)
        q_idx = _quant_index(idx, bits=bits, n=n)
        return replace(action, entity_id=ids[q_idx]), "applied"

    raise ValueError(f"unknown entity_id mode {mode!r}; "
                     "expected 'index' | 'collapse' | 'absolute'")


def bits_to_resolve_id(span: float) -> int:
    """Analytic: bits for the ABSOLUTE coding to reach a <1 step over ±span."""
    return int(math.ceil(math.log2(2.0 * span + 1.0)))


def index_bits_lossless(limit: int) -> int:
    """Bits for the INDEX pointer to be lossless over an entity_set of `limit`."""
    return int(math.ceil(math.log2(limit))) if limit > 1 else 0


__all__ = [
    "quantize_entity_id", "ABS_ID_RANGE",
    "bits_to_resolve_id", "index_bits_lossless",
]
