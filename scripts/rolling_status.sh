#!/bin/bash
# Quick status report on rolling-rollouts output. Survival, spawn-time
# distribution, tech tier, death modes, in-flight heroes.
#
# Usage: ./scripts/rolling_status.sh [results-dir]
# Default: results/rolling-$(date +%Y%m%d)

set -e
DIR="${1:-results/rolling-$(date +%Y%m%d)}"
if [ ! -d "$DIR" ]; then
    echo "no such dir: $DIR"; exit 1
fi

echo "===== $DIR ====="

echo ""
echo "--- rollouts ended ---"
n_ended=$(grep -c "ended" "$DIR/_orchestrator.log" 2>/dev/null || echo 0)
echo "count: $n_ended"
grep "ended" "$DIR/_orchestrator.log" 2>/dev/null | sort -t= -k2 -nr | head -10

echo ""
echo "--- in-flight rollouts ---"
for f in "$DIR"/agent*.log; do
    name=$(basename "$f" .log)
    if ! grep -q "rollout complete" "$f" 2>/dev/null; then
        turns=$(grep -c "=== turn [0-9]*/9999: planning ===" "$f")
        last=$(grep -oE "time=(DAY|NIGHT) [0-9.]+min until [a-z]+|day [0-9]+" "$f" | tail -2 | tr '\n' ' ')
        printf "  %-30s turns=%-4d  %s\n" "$name" "$turns" "$last"
    fi
done

echo ""
echo "--- spawn-time distribution (from JSONL headers) ---"
for f in "$DIR"/agent*.jsonl; do
    head -1 "$f" 2>/dev/null
done | python3 -c "
import sys, json, collections
buckets = collections.Counter()
for line in sys.stdin:
    try: d = json.loads(line)
    except: continue
    s = d.get('spawn') or {}
    dt = s.get('day_ticks')
    if dt is None: continue
    phase = ('DAWN' if dt<2000 else 'MORN' if dt<6000 else 'NOON' if dt<10000 else 'DUSK' if dt<14000 else 'NIGHT' if dt<22000 else 'PRE-DAWN')
    buckets[phase] += 1
for k in ('DAWN','MORN','NOON','DUSK','NIGHT','PRE-DAWN'):
    print(f'  {k:9s} {buckets[k]:3d}')
"

echo ""
echo "--- tech tier per rollout ---"
for f in "$DIR"/agent*.log; do
    name=$(basename "$f" .log)
    if grep -q "minecraft:diamond_pickaxe\|minecraft:diamond_sword" "$f" 2>/dev/null; then tier="DIAMOND"
    elif grep -q "minecraft:iron_pickaxe\|minecraft:iron_sword" "$f" 2>/dev/null; then tier="IRON"
    elif grep -q "minecraft:stone_pickaxe\|minecraft:stone_sword" "$f" 2>/dev/null; then tier="STONE"
    elif grep -q "minecraft:wooden_pickaxe\|minecraft:wooden_sword" "$f" 2>/dev/null; then tier="WOOD"
    else tier="NONE"; fi
    turns=$(grep -c "=== turn [0-9]*/9999: planning ===" "$f")
    printf "%-30s turns=%3d tier=%s\n" "$name" "$turns" "$tier"
done | sort -k3 -t= -nr

echo ""
echo "--- tier ceiling histogram ---"
for f in "$DIR"/agent*.log; do
    if grep -q "minecraft:diamond_pickaxe\|minecraft:diamond_sword" "$f" 2>/dev/null; then echo "DIAMOND"
    elif grep -q "minecraft:iron_pickaxe\|minecraft:iron_sword" "$f" 2>/dev/null; then echo "IRON"
    elif grep -q "minecraft:stone_pickaxe\|minecraft:stone_sword" "$f" 2>/dev/null; then echo "STONE"
    elif grep -q "minecraft:wooden_pickaxe\|minecraft:wooden_sword" "$f" 2>/dev/null; then echo "WOOD"
    else echo "NONE"; fi
done | sort | uniq -c

echo ""
echo "--- death modes ---"
grep -h "YOU DIED" "$DIR"/agent*.log 2>/dev/null | sed 's/.*cause: //;s/).*//' | sort | uniq -c | sort -rn
