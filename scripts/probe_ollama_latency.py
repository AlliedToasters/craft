#!/usr/bin/env python3
"""
Probe Ollama KV prefix-cache and concurrency scaling.

Experiment 1 — Sequential cache warmup
  N sequential requests with the SAME system prompt (SURVIVE_SHELTER_PROMPT,
  the actual rollout prompt) and DISTINCT user turns. If Ollama's prefix cache
  is active, TTFT on req 2+ should drop vs req 1 — the system-prompt tokens
  are already in the KV cache and don't need to be re-processed.

  Hypothesis: TTFT[0] >> TTFT[1..N-1] if caching works.

Experiment 2 — Concurrent scaling
  Fan out N=1,2,3,5 requests simultaneously (threads → same Ollama endpoint).
  - OLLAMA_NUM_PARALLEL=1 (default serial): wall ≈ N × single, per-request
    latency is flat (each waits in queue).
  - OLLAMA_NUM_PARALLEL≥N (true batching): wall ≈ single, per-request latency
    may rise slightly due to KV memory split across batch.

  The wall/baseline ratio is the key number: ~1 = parallel, ~N = serial.

Usage:
  uv run python scripts/probe_ollama_latency.py
  uv run python scripts/probe_ollama_latency.py --seq-reps 6 --conc-sizes 1 2 3 5
  uv run python scripts/probe_ollama_latency.py --model gemma-4-vanilla-q4-32k:latest
"""
from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time

from openai import OpenAI

# Use the actual rollout prompt so results are directly applicable.
# craft.agent only has module-level constants; import is side-effect-free.
from craft.agent import SURVIVE_SHELTER_PROMPT
from craft.llm import OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL

# Realistic per-turn user messages: same structure agent.py injects each turn
# (stats line + inventory). Distinct so each request has different non-cached tokens.
_USER_MSGS = [
    (
        "Stats: HP=20/20 food=20 sat=5.0 air=300 pos=(142,68,-307) facing=north "
        "time=DAY 8.3min until dusk (day 1) biome=forest\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:wooden_pickaxe\n  slot 1: 32x minecraft:oak_log"
    ),
    (
        "Stats: HP=16/20 food=18 sat=3.0 air=300 pos=(89,62,-412) facing=east "
        "time=DAY 4.1min until dusk (day 1) biome=plains\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:stone_pickaxe\n"
        "  slot 1: 8x minecraft:cobblestone\n  slot 2: 6x minecraft:oak_planks"
    ),
    (
        "Stats: HP=20/20 food=20 sat=5.0 air=300 pos=(-203,71,88) facing=south "
        "time=NIGHT 6.2min until dawn (day 2) biome=forest\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:wooden_pickaxe\n"
        "  slot 1: 1x minecraft:crafting_table\n  slot 2: 24x minecraft:oak_log"
    ),
    (
        "Stats: HP=8/20 food=14 sat=1.0 air=300 pos=(334,44,-178) facing=west "
        "time=NIGHT 3.7min until dawn (day 1) biome=birch_forest\n\n"
        "Current inventory:\n  (empty)"
    ),
    (
        "Stats: HP=20/20 food=20 sat=5.0 air=300 pos=(-91,63,205) facing=north "
        "time=DAY 9.8min until dusk (day 3) biome=taiga\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:iron_pickaxe\n"
        "  slot 1: 4x minecraft:iron_ingot\n  slot 2: 16x minecraft:cobblestone\n"
        "  slot 3: 1x minecraft:furnace"
    ),
    (
        "Stats: HP=12/20 food=10 sat=0.0 air=300 pos=(67,38,-550) facing=south "
        "time=DAY 2.0min until dusk (day 2) biome=snowy_plains\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:stone_pickaxe\n"
        "  slot 1: 3x minecraft:coal\n  slot 2: 22x minecraft:cobblestone"
    ),
]


def _make_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def _request_timed(
    client: OpenAI,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 64,
) -> dict:
    """One streaming request. Returns {ttft_s, total_s, tokens_out}.

    No stop tokens or reasoning_effort here — we want the model to actually
    emit content tokens so TTFT is measurable. The probe is about cache and
    concurrency timing, not tool-call fidelity.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    ttft: float | None = None
    tokens_out = 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if ttft is None and delta:
            ttft = time.perf_counter() - t0
        if delta:
            tokens_out += 1
    total = time.perf_counter() - t0
    return {"ttft_s": ttft, "total_s": total, "tokens_out": tokens_out}


# ── Experiment 1 ────────────────────────────────────────────────────────────

def run_sequential(client: OpenAI, model: str, reps: int) -> list[dict]:
    print(f"\n{'='*64}")
    print(f"EXPERIMENT 1 — Sequential cache warmup  (reps={reps})")
    print(f"{'='*64}")
    print(f"  System prompt: SURVIVE_SHELTER_PROMPT ({len(SURVIVE_SHELTER_PROMPT)} chars)")
    print(f"  User prompt:   distinct per request (realistic agent turn context)")
    print()
    print(f"  {'req':>4}  {'TTFT (s)':>10}  {'total (s)':>10}  {'tok_out':>8}")
    print("  " + "-" * 42)

    results = []
    for i in range(reps):
        user = _USER_MSGS[i % len(_USER_MSGS)]
        r = _request_timed(client, SURVIVE_SHELTER_PROMPT, user, model)
        results.append(r)
        ttft_str = f"{r['ttft_s']:.3f}" if r["ttft_s"] is not None else " N/A "
        print(f"  {i+1:>4}  {ttft_str:>10}  {r['total_s']:>10.3f}  {r['tokens_out']:>8}")

    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    if len(ttfts) >= 2:
        miss = ttfts[0]
        hits = ttfts[1:]
        hit_mean = statistics.mean(hits)
        speedup = (miss - hit_mean) / miss * 100
        print()
        print(f"  req-1 TTFT  (cold / cache miss):  {miss:.3f}s")
        print(f"  req 2+ TTFT (warm / cache hit?):  {hit_mean:.3f}s  (mean of {len(hits)})")
        print(f"  Speedup: {speedup:+.1f}%  {'← prefix cache likely active' if speedup > 10 else '← no significant speedup (cache may be off or already warm)'}")
    return results


# ── Experiment 2 ────────────────────────────────────────────────────────────

def _worker(
    client: OpenAI,
    system: str,
    user: str,
    model: str,
    out: list,
    idx: int,
) -> None:
    r = _request_timed(client, system, user, model, max_tokens=48)
    out[idx] = r


def run_concurrent(client: OpenAI, model: str, sizes: list[int]) -> None:
    print(f"\n{'='*64}")
    print(f"EXPERIMENT 2 — Concurrent scaling  (sizes={sizes})")
    print(f"{'='*64}")
    print("  All requests share the same system prompt; distinct user turns.")
    print()

    # Single-request baseline (fresh, after exp-1 warmed the cache)
    baseline = _request_timed(
        client, SURVIVE_SHELTER_PROMPT, _USER_MSGS[0], model, max_tokens=48
    )
    ttft_b = f"{baseline['ttft_s']:.3f}s" if baseline["ttft_s"] is not None else "N/A"
    print(f"  Baseline N=1:  total={baseline['total_s']:.2f}s  ttft={ttft_b}")
    print()
    print(f"  {'N':>3}  {'wall (s)':>10}  {'per-req mean':>14}  {'per-req p95':>13}  {'wall/base':>10}  interpretation")
    print("  " + "-" * 80)

    for n in sizes:
        if n < 2:
            continue
        out: list[dict | None] = [None] * n
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futs = [
                ex.submit(
                    _worker,
                    client,
                    SURVIVE_SHELTER_PROMPT,
                    _USER_MSGS[i % len(_USER_MSGS)],
                    model,
                    out,
                    i,
                )
                for i in range(n)
            ]
            concurrent.futures.wait(futs)
        wall = time.perf_counter() - t0
        totals = [r["total_s"] for r in out if r is not None]
        mean_t = statistics.mean(totals) if totals else float("nan")
        p95_t = sorted(totals)[max(0, int(len(totals) * 0.95) - 1)] if totals else float("nan")
        ratio = wall / baseline["total_s"]

        if ratio < 1.4:
            interp = "≈ parallel batching"
        elif ratio < n * 0.6:
            interp = "≈ partial parallelism"
        else:
            interp = "≈ serial queue"

        print(f"  {n:>3}  {wall:>10.2f}  {mean_t:>14.2f}  {p95_t:>13.2f}  {ratio:>10.2f}x  {interp}")

    print()
    print("  wall/base ≈ 1  → true batching (OLLAMA_NUM_PARALLEL≥N)")
    print("  wall/base ≈ N  → serial queue  (OLLAMA_NUM_PARALLEL=1, default)")
    print()
    print("  To enable parallel mode on the Ollama host:")
    print("    OLLAMA_NUM_PARALLEL=5 ollama serve   # or set in systemd override")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=OLLAMA_DEFAULT_MODEL)
    ap.add_argument("--seq-reps", type=int, default=5, metavar="N",
                    help="sequential requests in experiment 1 (default 5)")
    ap.add_argument("--conc-sizes", type=int, nargs="+", default=[1, 2, 3, 5], metavar="N",
                    help="fan-out sizes for experiment 2 (default: 1 2 3 5)")
    args = ap.parse_args()

    client = _make_client()
    print(f"Model:    {args.model}")
    print(f"Endpoint: {OLLAMA_BASE_URL}")
    print(f"System prompt length: {len(SURVIVE_SHELTER_PROMPT)} chars")

    print("\nWarm-up ping (connection establishment, not cached)...")
    ping = _request_timed(
        client,
        "You are a helpful assistant.",
        "Reply with exactly one word: Ready",
        args.model,
        max_tokens=4,
    )
    ttft_str = f"{ping['ttft_s']:.2f}s" if ping["ttft_s"] is not None else "N/A"
    print(f"  ping: ttft={ttft_str}  total={ping['total_s']:.2f}s")

    run_sequential(client, args.model, args.seq_reps)
    run_concurrent(client, args.model, args.conc_sizes)

    print("\nDone.")


if __name__ == "__main__":
    main()
