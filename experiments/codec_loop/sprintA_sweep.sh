#!/bin/bash
# Sprint A live parity-vs-bits sweep (loss-tolerance probe).
#
# Drives the §14 Rung-2 substitution harness (run_rungs.py) against the codec
# sidecar (craft.codec.server on :25600) at successive lossy bit levels, plus a
# lossless control. The sidecar's lossy level is retuned LIVE via POST /config
# so the wire path is byte-identical across levels — only the sidecar math
# changes. Parity = did the controller still reach its goto targets.
#
# Brief: movement-only quantization, ONE target type, NOT a learned-codec
# baseline. Deliverable = the knee of reached-rate vs bits/field.
#
# Usage: experiments/codec_loop/sprintA_sweep.sh [PORT] [OUTDIR] [BITS...]
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PORT="${1:-25570}"
OUTDIR="${2:-results/sprintA}"
shift || true; shift || true
BITS=("$@"); [ ${#BITS[@]} -eq 0 ] && BITS=(8 6 5 4 3)

PY=".venv/bin/python"
CODEC="http://127.0.0.1:25600"
mkdir -p "$OUTDIR"

cfg() { curl -s -m5 -X POST "$CODEC/config" -H 'Content-Type: application/json' -d "$1"; echo; }

echo "[sweep] port=$PORT out=$OUTDIR bits=${BITS[*]}"
echo "[sweep] codec health: $(curl -s -m5 $CODEC/healthz)"

# Control: full ladder (rungs 0,1,2) with the codec LOSSLESS (b=inf).
echo "[sweep] === CONTROL (lossless, rungs 0,1,2) ==="
cfg '{"quant_bits": null}'
$PY -m experiments.codec_loop.run_rungs --port "$PORT" --rungs 0,1,2 \
    --out "$OUTDIR/control.json" || echo "[sweep] control returned $?"

# Each lossy level: rung 2 only (THE TEST — substitution on the wire).
for b in "${BITS[@]}"; do
  echo "[sweep] === b=$b (rung 2) ==="
  cfg "{\"quant_bits\": $b}"
  $PY -m experiments.codec_loop.run_rungs --port "$PORT" --rungs 2 \
      --out "$OUTDIR/b$b.json" || echo "[sweep] b=$b returned $?"
done

# Restore lossless so a stray later harness run isn't silently lossy.
cfg '{"quant_bits": null}'
echo "[sweep] DONE -> $OUTDIR"
