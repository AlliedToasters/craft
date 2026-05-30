"""Lossy movement quantizer for Sprint A (loss-tolerance probe).

This wraps the LOSSLESS structured ``MoveAction`` (craft/codec/move.py) with
fixed-point quantization of its semantic float fields, parameterized by
bits-per-field. It is the wire-side lossy layer the §14 Rung-2 harness will
substitute, and the object the offline fidelity gate measures.

WHAT THIS IS NOT (brief, anti-pattern #1): a baseline a learned codec must
beat. Quantization is appropriate here *precisely because* it baselines
nothing — it is a controller-robustness probe. The deliverable is a
parity-vs-bits curve and its knee, not a compression number to defend.

Field handling (ranges from recon over frozen_narrated, TP-jumps excluded):

  * pos dx/dy/dz : per-tick locomotion delta vs obs.{x,y,z}. Tightly clustered
                   (p99 ~ +/-0.28 walk) with rare fall/jump excursions to a few
                   blocks. Quantized over a symmetric +/-``POS_RANGE``; the
                   range is chosen to NOT clip observed locomotion+falls so the
                   only error mode is quantization resolution (clipping would
                   add a confounding systematic bias on falls).

  * yaw          : the codec carries yaw ABSOLUTE and does NOT normalize
                   (craft/codec/move.py:19-23); raw MC yaw accumulates and is
                   unbounded (-2700..+2087 observed). The wire-receiving server
                   wraps degrees, so we ``mod 360`` to [0,360) BEFORE
                   quantizing -- this is behavior-preserving and REQUIRED:
                   naive fixed-point over the raw range would alias wrap-points
                   into legitimate angles.

  * pitch        : physically bounded [-90, 90].

  * on_ground / horizontal_collision : booleans, never quantized (1 bit each).
"""

from __future__ import annotations

from dataclasses import replace

from craft.codec.move import MoveAction

# --- quantization ranges (documented constants, fixed across the sweep) ------
# +/-8 covers every observed locomotion delta and fall (max |dz|=8, min dy=-8)
# with no clipping, so the sole error mode is resolution, not range clipping.
POS_RANGE = 8.0
YAW_LO, YAW_HI = 0.0, 360.0      # after mod-360 normalization
PITCH_LO, PITCH_HI = -90.0, 90.0


def quant_scalar(v: float, lo: float, hi: float, bits: int) -> tuple[float, int]:
    """ZERO-BIASED uniform quantizer over [lo, hi] with ``bits`` resolution.

    Returns ``(reconstructed_value, integer_code)``. ``bits<=0`` collapses to
    the range midpoint (0 bits of information). Values outside [lo, hi] are
    clamped — for the move ranges above this never fires on observed data.

    "Zero-biased": there are ``2**bits`` reconstruction points (an EVEN count)
    spaced evenly across [lo, hi], so for a symmetric range [-R, R] the midpoint
    0.0 falls BETWEEN two codes and is NOT representable — a true-zero input
    reconstructs to ``±R/(2**bits-1)`` (see ``quant_scalar_zero_preserving`` for
    the fix, and ``recon_hist.py`` for why this DC offset rubberbands the
    controller on stationary packets). Textbook name: mid-rise. (NOTE: prior
    RESULTS/memory docs in this project labeled THIS one "mid-tread" — that was
    backwards; trust the property, not the old label.)
    """
    if bits <= 0:
        return (lo + hi) / 2.0, 0
    levels = (1 << bits) - 1
    span = hi - lo
    t = (min(hi, max(lo, v)) - lo) / span        # 0..1
    code = round(t * levels)
    return lo + (code / levels) * span, code


def quant_scalar_zero_preserving(v: float, lo: float, hi: float,
                                 bits: int) -> tuple[float, int]:
    """ZERO-PRESERVING uniform quantizer: the range MIDPOINT is a code level.

    Returns ``(reconstructed_value, integer_code)``. Places ``K`` steps on each
    side of the midpoint (code 0), for ``2K+1`` reconstruction points (an ODD
    count, one fewer level than ``quant_scalar`` for the same bit budget). For a
    symmetric range [-R, R] the midpoint is 0.0, so a TRUE-ZERO input
    reconstructs to EXACTLY 0.0 — no DC drift on stationary packets. This is the
    principled fix for the b5 cliff (recon_hist.py): stationary deltas (56% of
    move traffic) stop injecting phantom motion. Textbook name: mid-tread.

    Cost vs zero_biased: one step coarser (``step = half/K`` with K = 2^(b-1)-1,
    so 0.533 vs 0.516 at b5/R8) — but moving deltas already survive that
    resolution, so the trade is unambiguous. ``bits<=1`` collapses to the
    midpoint (K=0). Values outside [lo, hi] are clamped.
    """
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    k = (1 << (bits - 1)) - 1 if bits >= 1 else 0    # steps per side
    if k <= 0:
        return mid, 0
    step = half / k
    code = round((min(hi, max(lo, v)) - mid) / step)
    code = max(-k, min(k, code))
    return mid + code * step, code


# Dispatch table so callers (server, recon_hist) select a quantizer by mode name
# without an if-ladder. "zero_biased" is the DEFAULT (preserves all prior runs).
POS_MODES = {
    "zero_biased": quant_scalar,
    "zero_preserving": quant_scalar_zero_preserving,
}


def _wrap360(deg: float) -> float:
    """Normalize an angle to [0, 360)."""
    return deg % 360.0


def quantize_move(action: MoveAction, *, pos_bits: int, yaw_bits: int,
                  pitch_bits: int, pos_range: float = POS_RANGE,
                  pos_mode: str = "zero_biased") -> MoveAction:
    """Return a lossy copy of ``action`` with pos/rot fields quantized.

    Booleans and packet_type pass through untouched. Absent fields (per wire
    type) stay None. Yaw is mod-360 normalized before quantization; on decode
    the receiving server wraps degrees, so the normalized value is behaviorally
    equivalent to the original.

    ``pos_mode`` selects the POS quantizer (POS_MODES): "zero_biased" (default,
    the original mid-rise grid) or "zero_preserving" (midpoint=code, so a
    stationary delta reconstructs to exactly 0 — the b5-cliff fix). Yaw/pitch
    always use the zero_biased grid: they carry ABSOLUTE angles (not per-tick
    deltas), so a stationary player yields a constant value with no DC-drift
    failure mode — keeping them fixed isolates the A/B to the identified
    mechanism.

    ``pos_range`` is the symmetric span ±R the pos delta is quantized over. The
    min representable step is ``2*pos_range/(2**pos_bits - 1)``, so pos_range is
    the CONTINUOUS lever on quantization resolution: at fixed bits, shrinking it
    slides the parity knee in sub-bit increments (the bit grid alone is too
    coarse to land inside the b5-b6 transition). CAUTION: a delta exceeding
    ±pos_range CLIPS — for flat-arena locomotion (walk delta ~0.21, no falls)
    this never fires down to small ranges, but it WOULD bias falls (|dy| up to
    8), so a range sweep is clean only where no large excursions occur.
    """
    pos_quant = POS_MODES.get(pos_mode)
    if pos_quant is None:
        raise ValueError(f"unknown pos_mode {pos_mode!r}; expected one of {sorted(POS_MODES)}")
    pos = action.pos
    if pos is not None:
        pos = tuple(pos_quant(c, -pos_range, pos_range, pos_bits)[0]
                    for c in pos)  # type: ignore[assignment]
    rot = action.rot
    if rot is not None:
        yaw, pitch = rot
        qyaw, _ = quant_scalar(_wrap360(yaw), YAW_LO, YAW_HI, yaw_bits)
        qpitch, _ = quant_scalar(pitch, PITCH_LO, PITCH_HI, pitch_bits)
        rot = (qyaw, qpitch)
    return replace(action, pos=pos, rot=rot)


def float_bits(action: MoveAction, *, pos_bits: int, yaw_bits: int,
               pitch_bits: int) -> int:
    """Bits spent on the quantized float fields of this wire type.

    The swept quantity (X-axis of the parity curve). Excludes the constant
    overhead (packet-type tag + 2 boolean flags) which does not vary with the
    sweep; report that separately if an absolute bits/packet is wanted.
    """
    b = 0
    if action.pos is not None:
        b += 3 * pos_bits
    if action.rot is not None:
        b += yaw_bits + pitch_bits
    return b
