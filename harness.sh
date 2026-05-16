#!/usr/bin/env bash
# Sweep survive prompt variants across N reps each, wiping the server between runs.
#
# Usage:
#   ./harness.sh                                # 5 reps × all default variants
#   ./harness.sh 3                              # 3 reps × all default variants
#   ./harness.sh 5 survive survive_first        # 5 reps × explicit variants
#
# Outputs:
#   /tmp/survive-sweep-<timestamp>/<variant>-r<N>.log
#   /tmp/survive-sweep-<timestamp>/results.jsonl

set -euo pipefail

cd "$(dirname "$0")"

REPS="${1:-5}"
shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=("survive" "survive_first")
fi

MAX_TURNS=30
HOMUNCULUS_BASE="http://127.0.0.1:25566"
SWEEP_DIR="/tmp/survive-sweep-$(date +%Y-%m-%d-%H%M%S)"
PY="$(pwd)/.venv/bin/python"
EXTRACT="$(pwd)/extract_rollout.py"

mkdir -p "$SWEEP_DIR"
RESULTS="$SWEEP_DIR/results.jsonl"

echo "[harness] sweep dir: $SWEEP_DIR"
echo "[harness] variants: ${VARIANTS[*]}; reps: $REPS; max_turns: $MAX_TURNS"

wait_for_homunculus() {
    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if /usr/bin/curl --silent --fail --max-time 3 "$HOMUNCULUS_BASE/position" > /dev/null 2>&1; then
            return 0
        fi
        sleep 3
    done
    echo "[harness] FAILED: homunculus did not come up within 180s" >&2
    return 1
}

for variant in "${VARIANTS[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        echo
        echo "===== [harness] $variant rep $rep/$REPS ====="
        log="$SWEEP_DIR/${variant}-r${rep}.log"

        echo "[harness] wiping server..."
        ./wipe_server.sh

        echo "[harness] waiting for homunculus..."
        wait_for_homunculus || exit 1

        echo "[harness] running agent → $log"
        "$PY" -m craft.agent "$MAX_TURNS" "$variant" permadeath > "$log" 2>&1 || true

        echo "[harness] extracting metrics..."
        "$PY" "$EXTRACT" "$log" --variant "$variant" --rep "$rep" >> "$RESULTS"
        tail -n 1 "$RESULTS"
    done
done

echo
echo "[harness] sweep complete → $RESULTS"
"$PY" "$EXTRACT" "$RESULTS" --summary
