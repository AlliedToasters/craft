"""Probe: vary system prompt / sampling params and measure gemma's output health.

For each (prompt_variant, scenario, params) tuple, hit gemma directly through
the OpenAI-compat layer and record:
  - tool_call_count   (want exactly 1)
  - leak_count        (count of Harmony-style template tokens in content; want 0)
  - completion_tokens (want lower — less bloat)
  - latency           (correlate with token count)

Writes JSONL to results/<timestamp>.jsonl so runs are cumulative across the night.
Use --quick to run one rep per cell; --reps N for averaging.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from craft.agent import SYSTEM_PROMPT as BASELINE_PROMPT
from craft.tools import TOOLS

import os

MODEL = "gemma-4-vanilla-q4-32k:latest"
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

client = OpenAI(base_url=BASE_URL, api_key="ollama")

# Matches Harmony-style chat-template tokens in content. Catches both proper
# (<|channel|>, <|tool_response|>) and corrupted variants (<channel|>, <|tool_response>).
LEAK_RE = re.compile(r"<\|[a-zA-Z_/]+\|?>|<[a-zA-Z_/]+\|>")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---- Scenarios (varied inventory states; minimal user-message form) ----
# Format mirrors how craft.agent presents inventory each turn.
def _inv_msg(items: list[tuple[int, str]]) -> str:
    lines = ["Begin. Read the inventory carefully before deciding the next action.", "", "Current inventory:"]
    if not items:
        lines.append("  (empty)")
    else:
        for i, (count, item) in enumerate(items):
            lines.append(f"  slot {i}: {count}x minecraft:{item}")
    return "\n".join(lines)


SCENARIOS: dict[str, str] = {
    "cold_start": _inv_msg([]),
    "got_logs": _inv_msg([(3, "oak_log")]),
    "wood_pick": _inv_msg([(1, "wooden_pickaxe"), (3, "oak_log"), (1, "crafting_table")]),
    "stone_pick": _inv_msg([(1, "stone_pickaxe"), (1, "wooden_axe"), (1, "wooden_shovel"), (12, "cobblestone")]),
    "iron_ingot": _inv_msg([(1, "stone_pickaxe"), (1, "stone_sword"), (5, "raw_iron"), (8, "cobblestone"), (3, "oak_planks")]),
    "iron_armored": _inv_msg([(1, "iron_pickaxe"), (1, "iron_sword"), (1, "iron_helmet"), (1, "iron_chestplate"), (1, "iron_leggings"), (1, "iron_boots"), (4, "iron_ingot")]),
    # Goal-recognition test: model should ideally emit NO tool call here.
    "diamond_owned": _inv_msg([(3, "diamond"), (1, "iron_pickaxe"), (1, "iron_sword"), (4, "iron_ingot"), (64, "cobblestone")]),
    # PARTIAL-recovery tests — simulate having just gotten a stuck response.
    "stuck_descend": (
        "Begin. Read the inventory carefully before deciding the next action.\n\n"
        "Recent action outcome: PARTIAL: at y=49 (started y=60, target y=-58) — Baritone unreachable; try descend(-58) again, or surface() if stuck.\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:iron_pickaxe\n  slot 1: 1x minecraft:iron_sword\n  slot 2: 1x minecraft:wooden_shovel\n  slot 3: 12x minecraft:dirt\n"
    ),
    "stuck_repeat": (
        "Begin. Read the inventory carefully before deciding the next action.\n\n"
        "Recent action outcomes (oldest → newest):\n"
        "- descend(-58) returned PARTIAL: stuck at y=53\n"
        "- travel east 20 → moved 20 blocks\n"
        "- descend(-58) returned PARTIAL: stuck at y=48\n"
        "- travel west 30 → moved 27 blocks\n"
        "- descend(-58) returned PARTIAL: stuck at y=56\n\n"
        "Current inventory:\n  slot 0: 1x minecraft:iron_pickaxe\n  slot 1: 1x minecraft:iron_sword\n  slot 2: 11x minecraft:dirt\n"
    ),
    # Edge cases — does v11 handle uncommon states gracefully?
    "lost_pickaxe": _inv_msg([(1, "iron_sword"), (1, "iron_helmet"), (1, "iron_chestplate"), (8, "cobblestone"), (32, "dirt")]),
    "complex_inv": _inv_msg([
        (4, "oak_log"), (1, "wooden_pickaxe"), (1, "stone_pickaxe"), (1, "iron_pickaxe"),
        (8, "stick"), (3, "iron_ingot"), (1, "raw_iron"), (5, "coal"), (12, "cobblestone"),
        (3, "iron_helmet"), (1, "wooden_sword"), (2, "stone_sword"), (1, "bread"),
        (2, "torch"), (5, "wheat_seeds"), (8, "gravel"), (4, "sand"), (1, "chicken"),
        (1, "crafting_table"), (1, "furnace"), (3, "oak_planks"), (1, "porkchop"),
    ]),
    "smelt_no_fuel": (
        "Begin. Read the inventory carefully before deciding the next action.\n\n"
        "Recent action outcome: FAILED: missing fuel (3x minecraft:oak_planks) — get logs/planks/coal first\n\n"
        "Current inventory:\n  slot 0: 5x minecraft:raw_iron\n  slot 1: 1x minecraft:stone_pickaxe\n  slot 2: 1x minecraft:wooden_axe\n  slot 3: 12x minecraft:cobblestone\n"
    ),
}


# ---- Prompt variants ----
# Each variant is a function that returns a system prompt string. Allows easy
# A/B-ing of sections of the base prompt.

def v_baseline() -> str:
    return BASELINE_PROMPT


def v_no_survival() -> str:
    """Strip the SURVIVAL section + the suggested-cadence paragraph."""
    text = BASELINE_PROMPT
    # cut from "SURVIVAL:" through the end (everything after navigation)
    cut = text.find("SURVIVAL:")
    if cut < 0:
        return text
    head = text[:cut].rstrip()
    tail = "\n\nAfter each action, the current inventory is listed so you can plan the next step accurately."
    return head + tail


def v_strict_one() -> str:
    """Baseline + extra-emphatic single-tool-call rule, top of prompt."""
    extra = (
        "CRITICAL OUTPUT RULES:\n"
        "1. Emit EXACTLY ONE tool call. Never two. Never a list of planned tool calls. ONE.\n"
        "2. Do NOT emit any text in the content field. The tool call is the entire output.\n"
        "3. Do NOT emit '<|channel|>', '<|tool_response|>', '<|return|>', or any '<|...|>'-style tokens. These are forbidden.\n\n"
    )
    return extra + BASELINE_PROMPT


def v_anti_leak() -> str:
    """Baseline + explicit anti-template-token instruction (no batching rule change)."""
    extra = (
        # "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
        # "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens — "
        # "they are not part of your output schema.\n\n"
    )
    return extra + BASELINE_PROMPT


def v_minimal() -> str:
    """Tool-list-only minimum: no SURVIVAL, no HOME, no NAVIGATION prose."""
    return (
        "You are a Minecraft agent. Goal: acquire a diamond. "
        "Emit ONE tool call per response. Do not emit text content. "
        "Inventory will be shown each turn — read it, then call the next tool.\n\n"
        "Tools (use exactly one per turn):\n"
        "- mine_wood(quantity), mine_stone(quantity), mine_iron(quantity), mine_diamond(quantity)\n"
        "- craft(item, quantity, location?) — recursive: handles sub-recipes and table placement\n"
        "- smelt(input, count, location?) — auto-places furnace, auto-picks fuel\n"
        "- place(item) — rarely needed; craft/smelt place tables/furnaces\n"
        "- surface() — go up to sky\n"
        "- descend(target_y) — dig down to Y\n"
        "- travel(direction, distance) — walk N blocks N/S/E/W (cap 64)\n"
    )


def v_minimal_armed() -> str:
    """Minimal + the survival cadence as a single short list (not paragraphs)."""
    return v_minimal() + (
        "\nSafety cadence (gate progress on gear):\n"
        "- After wooden_pickaxe: craft wooden_axe AND wooden_shovel.\n"
        "- After stone_pickaxe: craft stone_sword before mining iron.\n"
        "- After iron_pickaxe: craft iron_sword + iron armor before descending past Y<32.\n"
        "Auto-equip happens after every turn; you don't manage hotbar slots.\n"
    )


def v_minimal_antileak() -> str:
    """Minimal prompt + the v3 anti-leak preamble. Smallest plausible config."""
    extra = (
        "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
        "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens — "
        "they are not part of your output schema.\n\n"
    )
    return extra + v_minimal()


def v_fewshot() -> str:
    """Minimal + a single demonstration in the system prompt."""
    body = v_minimal()
    demo = (
        "\nExample.\n"
        'User: "Current inventory:\\n  slot 0: 2x oak_log"\n'
        'Assistant: <tool_call: craft({"item": "wooden_pickaxe", "quantity": 1})>\n'
        "(Note: the assistant emits ONLY the tool call. No text content. No <|channel|> tokens.)\n"
    )
    return body + demo


def v_blunt() -> str:
    """Maximally blunt: just rules, no flavor, no SURVIVAL prose."""
    return (
        "Minecraft agent. Goal: acquire a diamond.\n"
        "RULE 1: Emit exactly ONE tool call per response. Never multiple.\n"
        "RULE 2: Content field MUST be empty. No text. No <|channel|>. No <|tool_response|>. No special tokens.\n"
        "RULE 3: Read the inventory shown each turn. Pick the next single action.\n\n"
        "Tools: mine_wood, mine_stone, mine_iron, mine_diamond, craft, smelt, place, surface, descend, travel.\n"
        "Auto-equip runs after each turn. Cadence: wooden_pickaxe → wooden_axe + wooden_shovel → "
        "mine_stone → stone_pickaxe + stone_sword → mine_iron → smelt → iron_pickaxe + iron_sword + "
        "iron armor → descend → mine_diamond.\n"
    )


def v_goal_aware() -> str:
    """v6 + explicit goal-completion rule (stop emitting tool calls when goal is met)."""
    return v_minimal_antileak() + (
        "\nGoal completion: if your inventory already shows the goal item (a diamond), "
        "the run is OVER. Emit NO tool call. The harness ends the run when you stop "
        "emitting tool calls. Don't loop on surface() or other busywork.\n"
    )


def v_goal_aware_explicit() -> str:
    """v6 + a more emphatic goal-stop rule."""
    return v_minimal_antileak() + (
        "\nIMPORTANT — KNOW WHEN TO STOP:\n"
        "Your task is to acquire a DIAMOND. If the inventory shows minecraft:diamond, "
        "you have succeeded. DO NOT call any further tools. The harness terminates the run "
        "when no tool call is emitted. Calling surface(), travel(), or anything else after "
        "the goal is met is a BUG — just stop.\n"
    )


def v_recovery_aware() -> str:
    """v9 + PARTIAL recovery rules."""
    return v_goal_aware() + (
        "\nPARTIAL-recovery rules:\n"
        "- A 'PARTIAL' outcome means Baritone didn't reach the target. The most useful response is rarely to retry the exact same action.\n"
        "- If descend(N) returns PARTIAL, the target was too far. Pick a smaller Δy — current_y - 30 is a good first try, not -58.\n"
        "- If you get 2 PARTIALs on the same approach, switch strategy: mine_stone(8) to dig by hand, or travel to a different column.\n"
        "- Never call surface() repeatedly when already at surface (Δy < 2). Pick a new action.\n"
    )


def v_combined() -> str:
    """v6 anti-leak + goal-aware + recovery — full kitchen sink."""
    return v_recovery_aware()


VARIANTS = {
    "v0_baseline": v_baseline,
    "v1_no_survival": v_no_survival,
    "v2_strict_one": v_strict_one,
    "v3_anti_leak": v_anti_leak,
    "v4_minimal": v_minimal,
    "v5_minimal_armed": v_minimal_armed,
    "v6_minimal_antileak": v_minimal_antileak,
    "v7_fewshot": v_fewshot,
    "v8_blunt": v_blunt,
    "v9_goal_aware": v_goal_aware,
    "v10_goal_explicit": v_goal_aware_explicit,
    "v11_recovery_aware": v_recovery_aware,
    "v12_combined": v_combined,
}


def measure(
    variant_name: str,
    scenario_name: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.3,
    reasoning_effort: str | None = "low",
    stop: list[str] | None = None,
    tool_choice: str = "auto",
    rep: int = 0,
) -> dict[str, Any]:
    """Single sampling call. Returns a flat dict for JSONL logging."""
    extra: dict[str, Any] = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        kwargs["stop"] = stop
    if extra:
        kwargs["extra_body"] = extra

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        return {
            "variant": variant_name,
            "scenario": scenario_name,
            "rep": rep,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "reasoning_effort": reasoning_effort,
            "error": repr(e),
            "latency": time.monotonic() - t0,
        }
    elapsed = time.monotonic() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    tcs = msg.tool_calls or []
    leak_count = len(LEAK_RE.findall(content))
    return {
        "variant": variant_name,
        "scenario": scenario_name,
        "rep": rep,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "reasoning_effort": reasoning_effort,
        "tool_choice": tool_choice,
        "tool_calls": len(tcs),
        "tool_names": [tc.function.name for tc in tcs],
        "first_tool": tcs[0].function.name if tcs else None,
        "first_args": tcs[0].function.arguments if tcs else None,
        "leak_count": leak_count,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "latency": elapsed,
        "finish_reason": resp.choices[0].finish_reason,
        "content_snippet": content[:300],
    }


def run_sweep(
    variants: list[str],
    scenarios: list[str],
    *,
    reps: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.3,
    reasoning_effort: str | None = "low",
    stop: list[str] | None = None,
    tool_choice: str = "auto",
    log_path: Path | None = None,
) -> list[dict]:
    """Run all (variant × scenario × rep) cells. Writes JSONL as we go."""
    if log_path is None:
        log_path = RESULTS_DIR / f"{int(time.time())}.jsonl"
    results = []
    n_cells = len(variants) * len(scenarios) * reps
    i = 0
    print(
        f"sweep: variants={variants} scenarios={scenarios} reps={reps} "
        f"max_tokens={max_tokens} temp={temperature} stop={stop} tool_choice={tool_choice} → {n_cells} cells, logging to {log_path}"
    )
    with log_path.open("a") as f:
        for v in variants:
            system = VARIANTS[v]()
            for s in scenarios:
                user = SCENARIOS[s]
                for r in range(reps):
                    i += 1
                    rec = measure(
                        v, s, system, user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        stop=stop,
                        tool_choice=tool_choice,
                        rep=r,
                    )
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    err = rec.get("error")
                    if err:
                        print(f"  [{i}/{n_cells}] {v}/{s}#{r} ERROR: {err[:100]}")
                    else:
                        ft = rec.get("first_tool") or "<none>"
                        print(
                            f"  [{i}/{n_cells}] {v}/{s}#{r} "
                            f"tools={rec['tool_calls']} leak={rec['leak_count']} "
                            f"tok={rec['completion_tokens']} t={rec['latency']:.1f}s → {ft}"
                        )
                    results.append(rec)
    return results


def summarize(results: list[dict]) -> str:
    """Aggregate by variant: mean tools, mean leaks, mean tokens, mean latency."""
    by_v: dict[str, list[dict]] = {}
    for r in results:
        if "error" in r:
            continue
        by_v.setdefault(r["variant"], []).append(r)

    lines = [
        f"{'variant':<22} {'n':>4} {'tools':>7} {'leaks':>7} {'tokens':>8} {'lat':>6} {'1tc%':>6} {'noleak%':>8}"
    ]
    for v, rs in sorted(by_v.items()):
        n = len(rs)
        if n == 0:
            continue
        avg_tools = sum(r["tool_calls"] for r in rs) / n
        avg_leaks = sum(r["leak_count"] for r in rs) / n
        avg_tokens = sum((r["completion_tokens"] or 0) for r in rs) / n
        avg_lat = sum(r["latency"] for r in rs) / n
        pct_one_tool = 100 * sum(1 for r in rs if r["tool_calls"] == 1) / n
        pct_no_leak = 100 * sum(1 for r in rs if r["leak_count"] == 0) / n
        lines.append(
            f"{v:<22} {n:>4} {avg_tools:>7.2f} {avg_leaks:>7.2f} "
            f"{avg_tokens:>8.1f} {avg_lat:>6.2f} {pct_one_tool:>5.0f}% {pct_no_leak:>7.0f}%"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--reasoning-effort", default="low",
                    help="Set to 'none' to omit the reasoning_effort kwarg entirely.")
    ap.add_argument("--stop", nargs="*", default=None,
                    help="Stop sequences (e.g. --stop '<|channel|>' '<|return|>')")
    ap.add_argument("--tool-choice", default="auto", choices=["auto", "required", "none"])
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant: {v} (have {list(VARIANTS)})")
    for s in args.scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario: {s} (have {list(SCENARIOS)})")

    results = run_sweep(
        args.variants,
        args.scenarios,
        reps=args.reps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=reasoning_effort,
        stop=args.stop,
        tool_choice=args.tool_choice,
        log_path=args.log,
    )

    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
