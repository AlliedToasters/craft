"""§17.2.1 — lossy ``block_pos`` quantizer (the discrete-target analog of §16).

§17.0 found aim rides the action packet's own target field: for block actions
that field is ``block_pos`` (``use_item_on`` = place, ``player_action`` = dig),
and a +1-block perturbation was a deterministic miss. So the block target has NO
sub-unit scalar tolerance — the unit IS the floor. The only question §17.2.1 asks
is how few bits address the target *given obs*, and that turns entirely on the
reparam, exactly like the move stream (§16):

  * ABSOLUTE coding — quantize the raw world coordinate over a fixed window wide
    enough to contain wherever the player roams (±``abs_range``). To resolve a
    single block you need ``ceil(log2(2*abs_range))`` bits/axis (~14 at the 8192
    default) — and BELOW that the step exceeds one block, so the reconstructed
    coordinate rounds to the WRONG cell and the action targets the wrong block.
    There is no graded tolerance: it is lossless above the 1-block-step bit count
    and a cliff below. This is the foil.

  * OBS-RELATIVE coding — code the offset ``block_pos − round(player_pos)``, which
    the server's reach check bounds to ≈±6. Over a tight ±``reach`` window that is
    ~4 bits/axis lossless (covers ±6 with step <1). The window FOLLOWS the player
    (it is read from obs), so the magnitude of the world coordinate never enters
    the bit budget. This is the §16 reparam — "every field a zero_preserving delta
    vs obs" — applied to the discrete target.

Both reconstruct to integer ``block_pos`` (round after dequant) so the Java
reconstructor rebuilds a valid packet. The reparam is internal to coding; the
downstream ``decode`` is unchanged. Predicted headline (§17.2 pre-reg): the only
compression is the absolute→±6 pointer reparam (lossless); no lossy sub-unit
headroom. Largely CONFIRMATORY — it converts the §17.0 +1 perturbation into a knee
and sizes the obs-relative floor.

Non-spatial ``player_action`` actions carry convention-zero ``block_pos`` (no
target) and pass through untouched — quantizing filler would be meaningless and
could only inject noise.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from craft.codec.player_action import PlayerActionAction, _SPATIAL_ACTIONS
from craft.codec.use_item_on import UseItemOnAction
from experiments.codec_loop.quantize import quant_scalar_zero_preserving

# Obs-relative window: the server's reach check bounds |block_pos - player| to
# ~5-6 on each axis, so ±6 never clips a legal target. At 4 bits the
# zero_preserving step is 6/7 = 0.857 < 1 -> every integer offset in [-6,6]
# reconstructs exactly. This is the ~4-bit lossless floor the spec predicts.
BLOCK_REACH = 6.0

# Absolute window: a FIXED ±span that must contain wherever the player roams.
# 8192 covers a large overworld neighborhood; resolving a single block then costs
# ceil(log2(2*8192)) = 14 bits/axis. The point is the 14-vs-4 gap = the reparam.
ABS_RANGE = 8192.0


def _quant_axis_obsrel(target: int, ref_f: float, *, bits: int,
                       reach: float) -> int:
    """Code ``target - round(ref)`` over ±reach (zero_preserving), reconstruct,
    round back to an integer absolute block coordinate. ref cancels exactly
    (same value used both ways), so error is purely the dequant step."""
    ref = int(round(ref_f))
    delta = float(target - ref)
    rec, _ = quant_scalar_zero_preserving(delta, -reach, reach, bits)
    return ref + int(round(rec))


def _quant_axis_absolute(target: int, *, bits: int, span: float) -> int:
    """Quantize the raw world coordinate over a fixed ±span, round to int. Below
    ceil(log2(2*span)) bits the step exceeds 1 block -> wrong cell."""
    rec, _ = quant_scalar_zero_preserving(float(target), -span, span, bits)
    return int(round(rec))


def _quant_block_pos(bp: tuple[int, int, int], obs: Mapping[str, Any], *,
                     bits: int, mode: str, reach: float,
                     abs_range: float) -> tuple[int, int, int]:
    if mode == "obsrel":
        ox, oy, oz = float(obs["x"]), float(obs["y"]), float(obs["z"])
        return (
            _quant_axis_obsrel(bp[0], ox, bits=bits, reach=reach),
            _quant_axis_obsrel(bp[1], oy, bits=bits, reach=reach),
            _quant_axis_obsrel(bp[2], oz, bits=bits, reach=reach),
        )
    if mode == "absolute":
        return (
            _quant_axis_absolute(bp[0], bits=bits, span=abs_range),
            _quant_axis_absolute(bp[1], bits=bits, span=abs_range),
            _quant_axis_absolute(bp[2], bits=bits, span=abs_range),
        )
    raise ValueError(f"unknown block_pos mode {mode!r}; expected 'obsrel' or 'absolute'")


def quantize_block_pos(action: Any, obs: Mapping[str, Any], *, bits: int,
                       mode: str = "obsrel", reach: float = BLOCK_REACH,
                       abs_range: float = ABS_RANGE) -> Any:
    """Lossy copy of a block-targeted action with ``block_pos`` quantized.

    Applies to ``UseItemOnAction`` and SPATIAL ``PlayerActionAction``; any other
    action (incl. non-spatial player_action with convention-zero block_pos) is
    returned unchanged. The reconstructed block_pos stays integer-valued."""
    if isinstance(action, UseItemOnAction):
        bp = _quant_block_pos(action.block_pos, obs, bits=bits, mode=mode,
                              reach=reach, abs_range=abs_range)
        return replace(action, block_pos=bp)
    if isinstance(action, PlayerActionAction) and action.action in _SPATIAL_ACTIONS:
        bp = _quant_block_pos(action.block_pos, obs, bits=bits, mode=mode,
                              reach=reach, abs_range=abs_range)
        return replace(action, block_pos=bp)
    return action


def bits_to_resolve_block(span: float) -> int:
    """Analytic: bits/axis for the ABSOLUTE coding to reach a <1-block step
    (zero_preserving step = span/(2^(b-1)-1) <= 1 => the foil's lossless point)."""
    return int(math.ceil(math.log2(span + 1.0))) + 1


__all__ = [
    "quantize_block_pos", "BLOCK_REACH", "ABS_RANGE", "bits_to_resolve_block",
]
