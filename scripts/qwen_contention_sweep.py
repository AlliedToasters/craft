#!/usr/bin/env python3
"""Qwen inference contention sweep.

Hypothesis under test: the bottleneck for running many agents is *qwen
inference latency under concurrent load* — agents queueing for their next
action while game time ticks — not MC client overhead. This script isolates
the LLM serving layer: it fires C concurrent `chat_with_tools` calls (the
EXACT call the agent loop makes, same model / temperature / max_tokens /
reasoning_effort / stop tokens) at increasing concurrency levels and measures
per-call latency. The curve of latency-vs-concurrency is the answer.

Per-call latency is measured from a shared barrier release (all C calls start
together) to completion, so it INCLUDES any time spent queued behind other
calls inside Ollama — which is precisely the "agent standing around waiting"
cost we care about.

System load (GPU util/mem/power, CPU%, RAM%, loadavg) is sampled throughout
each batch, because whether qwen is GPU- or CPU-bound changes the shape of the
curve, and on this box the GPU is shared only with the model while the MC
clients hammer the CPU.

Usage:
  .venv/bin/python -m scripts.qwen_contention_sweep \
      --levels 1,2,3,4,5,6,8 --repeats 3 \
      --out results/qwen_contention_$(date +%Y%m%d-%H%M%S)

Outputs (when --out PREFIX given):
  PREFIX.calls.jsonl   one line per individual call
  PREFIX.summary.csv   one row per concurrency level
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# craft.* import triggers .env load (OLLAMA_BASE_URL etc.).
from craft.agent import SYSTEM_PROMPT, _build_state_chunk
from craft.llm import chat_with_tools
from craft.tools import TOOLS

QWEN = "hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"


# ─────────────────────────── representative prompt ──────────────────────────
# Mid-rollout shape: system + a couple of prior turn exchanges (so prefill
# token count is realistic, not turn-1 tiny) + the live STATE block. We do NOT
# need it to be a *good* plan — only to be a representative-sized input that
# makes qwen generate a representative-length tool call.

def _build_messages() -> list[dict]:
    stats = ("Stats: HP=18/20 food=15/20 sat=4.0 air=300/300 "
             "pos=(123,42,-87) facing=south time=DAY(day3) status=[on_ground]")
    inv = ("Current inventory: oak_log x4, cobblestone x31, stick x6, "
           "crafting_table x1, wooden_pickaxe x1, stone_pickaxe x1, "
           "torch x8, coal x3, raw_iron x2, dirt x12")
    smelts = "Active smelts: furnace@(120,41,-85): raw_iron x2 -> COOKING (~6s left)"
    state = _build_state_chunk(stats, inv, smelts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_state_chunk(
            "Stats: HP=20/20 food=20/20 sat=5.0 air=300/300 pos=(118,64,-90) "
            "facing=east time=DAY(day3) status=[on_ground]",
            "Current inventory: oak_log x6, stick x2",
            None)},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "mine_stone", "arguments": json.dumps({"quantity": 16})},
        }]},
        {"role": "tool", "tool_call_id": "call_0",
         "content": "mine_stone: OK — mined 16 cobblestone. pos=(123,42,-87)."},
        {"role": "user", "content": state},
    ]


# ──────────────────────────── system load sampler ───────────────────────────

def _read_cpu_times() -> tuple[int, int]:
    """(idle, total) jiffies from /proc/stat aggregate line."""
    with open("/proc/stat") as fh:
        parts = fh.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return idle, sum(vals)


def _read_ram_used_pct() -> float:
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])  # kB
    total = info.get("MemTotal", 1)
    avail = info.get("MemAvailable", total)
    return 100.0 * (total - avail) / total


def _read_loadavg() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.readline().split()[0])


def _read_gpu() -> tuple[float, float, float]:
    """(util%, mem_used_MB, power_W); zeros if nvidia-smi unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        ).stdout.strip().splitlines()[0]
        u, m, p = (x.strip() for x in out.split(","))
        return float(u), float(m), float(p)
    except Exception:
        return 0.0, 0.0, 0.0


class LoadSampler:
    """Polls system load in a background thread; aggregates mean/max."""

    def __init__(self, interval: float = 0.4):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self.gpu_pow: list[float] = []
        self.cpu: list[float] = []
        self.ram: list[float] = []
        self.load1: list[float] = []

    def _sample_point(self):
        """Point-in-time metrics (no delta needed) — so even a sub-interval
        batch captures GPU/RAM/load."""
        self.ram.append(_read_ram_used_pct())
        self.load1.append(_read_loadavg())
        u, m, p = _read_gpu()
        self.gpu_util.append(u)
        self.gpu_mem.append(m)
        self.gpu_pow.append(p)

    def _run(self):
        prev_idle, prev_total = _read_cpu_times()
        self._sample_point()  # immediate, so fast batches aren't all-zero
        while not self._stop.wait(self.interval):
            idle, total = _read_cpu_times()
            d_total = total - prev_total
            d_idle = idle - prev_idle
            prev_idle, prev_total = idle, total
            if d_total > 0:
                self.cpu.append(100.0 * (1.0 - d_idle / d_total))
            self._sample_point()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @staticmethod
    def _agg(xs: list[float]) -> tuple[float, float]:
        return (statistics.mean(xs), max(xs)) if xs else (0.0, 0.0)

    def summary(self) -> dict:
        gu_m, gu_x = self._agg(self.gpu_util)
        gm_m, gm_x = self._agg(self.gpu_mem)
        gp_m, gp_x = self._agg(self.gpu_pow)
        cpu_m, cpu_x = self._agg(self.cpu)
        ram_m, ram_x = self._agg(self.ram)
        ld_m, ld_x = self._agg(self.load1)
        return {
            "gpu_util_mean": round(gu_m, 1), "gpu_util_max": round(gu_x, 1),
            "gpu_mem_mb_mean": round(gm_m), "gpu_mem_mb_max": round(gm_x),
            "gpu_pow_w_mean": round(gp_m, 1), "gpu_pow_w_max": round(gp_x, 1),
            "cpu_pct_mean": round(cpu_m, 1), "cpu_pct_max": round(cpu_x, 1),
            "ram_pct_mean": round(ram_m, 1), "ram_pct_max": round(ram_x, 1),
            "load1_mean": round(ld_m, 1), "load1_max": round(ld_x, 1),
            "samples": len(self.cpu),
        }


# ─────────────────────────────── one call ───────────────────────────────────

def _one_call(messages, model, barrier: threading.Barrier) -> dict:
    barrier.wait()  # all C threads release together
    t0 = time.perf_counter()
    err = None
    out_chars = 0
    tool_name = None
    try:
        tool_calls, content, reasoning, _ = chat_with_tools(messages, TOOLS, model=model)
        out_chars = len(content or "") + len(reasoning or "")
        if tool_calls:
            tool_name = tool_calls[0].function.name
    except Exception as e:  # noqa: BLE001 — capture, don't crash the sweep
        err = f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    return {"latency_s": round(dt, 3), "ok": err is None, "err": err,
            "out_chars": out_chars, "tool": tool_name}


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


# ───────────────────────────────── sweep ────────────────────────────────────

def run_sweep(levels, repeats, model, warmup, calls_fh):
    messages = _build_messages()

    if warmup:
        print(f"[warmup] loading model {model} into VRAM ...", flush=True)
        b = threading.Barrier(1)
        w = _one_call(messages, model, b)
        print(f"[warmup] first call: {w['latency_s']}s ok={w['ok']} "
              f"tool={w['tool']} out_chars={w['out_chars']}"
              + (f" ERR={w['err']}" if w['err'] else ""), flush=True)

    rows = []
    for C in levels:
        all_lat: list[float] = []
        all_ok = 0
        all_n = 0
        out_chars: list[int] = []
        batch_walls: list[float] = []
        with LoadSampler() as sampler:
            for r in range(repeats):
                barrier = threading.Barrier(C)
                bw0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=C) as ex:
                    futs = [ex.submit(_one_call, messages, model, barrier)
                            for _ in range(C)]
                    results = [f.result() for f in futs]
                batch_walls.append(time.perf_counter() - bw0)
                for ci, res in enumerate(results):
                    all_n += 1
                    all_ok += int(res["ok"])
                    if res["ok"]:
                        all_lat.append(res["latency_s"])
                        out_chars.append(res["out_chars"])
                    if calls_fh:
                        calls_fh.write(json.dumps(
                            {"level": C, "repeat": r, "call": ci, **res}) + "\n")
                if calls_fh:
                    calls_fh.flush()
        load = sampler.summary()

        mean_lat = statistics.mean(all_lat) if all_lat else 0.0
        # Throughput: completed calls per second of batch wall, averaged across
        # repeats. With serialized serving this plateaus; with true parallelism
        # it scales with C until a resource saturates.
        mean_batch_wall = statistics.mean(batch_walls) if batch_walls else 0.0
        throughput = (C / mean_batch_wall) if mean_batch_wall > 0 else 0.0
        row = {
            "concurrency": C,
            "n": all_n,
            "ok": all_ok,
            "lat_mean_s": round(mean_lat, 2),
            "lat_p50_s": round(_pctl(all_lat, 0.50), 2),
            "lat_p95_s": round(_pctl(all_lat, 0.95), 2),
            "lat_max_s": round(max(all_lat), 2) if all_lat else 0.0,
            "batch_wall_s": round(mean_batch_wall, 2),
            "throughput_rps": round(throughput, 3),
            "out_chars_mean": round(statistics.mean(out_chars)) if out_chars else 0,
            **load,
        }
        rows.append(row)
        print(
            f"[level {C:>2}] n={all_n} ok={all_ok} | "
            f"lat mean={row['lat_mean_s']}s p50={row['lat_p50_s']}s "
            f"p95={row['lat_p95_s']}s max={row['lat_max_s']}s | "
            f"thru={row['throughput_rps']} req/s | "
            f"gpu={load['gpu_util_mean']}%/{load['gpu_util_max']}%max "
            f"{load['gpu_pow_w_mean']}W mem={load['gpu_mem_mb_mean']}MB | "
            f"cpu={load['cpu_pct_mean']}% ram={load['ram_pct_mean']}% "
            f"load1={load['load1_mean']}",
            flush=True)
    return rows


def _print_curve(rows):
    print("\n=== latency-vs-concurrency curve (mean per-call) ===")
    if not rows:
        return
    mx = max(r["lat_mean_s"] for r in rows) or 1.0
    width = 48
    for r in rows:
        bar = "█" * max(1, round(width * r["lat_mean_s"] / mx))
        print(f"  C={r['concurrency']:>2} {r['lat_mean_s']:>6.2f}s |{bar}")
    print("\n  Ideal (no contention): flat line — per-call latency independent of C.")
    print("  Contention: rising line — each extra agent slows everyone's turn.")
    base = rows[0]["lat_mean_s"] or 1.0
    last = rows[-1]
    print(f"\n  Solo latency:        {rows[0]['lat_mean_s']}s  (C={rows[0]['concurrency']})")
    print(f"  Latency @ C={last['concurrency']:<2}:       {last['lat_mean_s']}s  "
          f"({last['lat_mean_s']/base:.1f}x solo)")
    print(f"  Peak throughput:     {max(r['throughput_rps'] for r in rows)} req/s")


def _write_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="1,2,3,4,5,6,8",
                    help="comma-separated concurrency levels (default 1,2,3,4,5,6,8)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="batches per level (default 3)")
    ap.add_argument("--model", default=QWEN, help=f"model tag (default {QWEN})")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the model-load warmup call")
    ap.add_argument("--out", default=None,
                    help="output path prefix (writes PREFIX.calls.jsonl + PREFIX.summary.csv)")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"model={args.model}")
    print(f"OLLAMA_BASE_URL={os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}")
    print(f"levels={levels} repeats={args.repeats}")
    print("NOTE: per-call latency includes in-Ollama queue wait (faithful to "
          "'agent waiting for next action').\n")

    calls_fh = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        calls_fh = open(f"{args.out}.calls.jsonl", "w")

    try:
        rows = run_sweep(levels, args.repeats, args.model,
                         warmup=not args.no_warmup, calls_fh=calls_fh)
    finally:
        if calls_fh:
            calls_fh.close()

    _print_curve(rows)

    if args.out:
        _write_csv(rows, f"{args.out}.summary.csv")
        print(f"\nwrote {args.out}.calls.jsonl + {args.out}.summary.csv")


if __name__ == "__main__":
    main()
