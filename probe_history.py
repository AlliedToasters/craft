"""Multi-turn probe: feed v11 a full message history and check output health.

The single-turn probe_prompt.py validates the prompt on a single (system, user)
pair. Real rollouts have growing message histories — assistant tool calls and
tool outcomes accumulate. This probe stress-tests the prompt under realistic
long context.

For each history scenario, runs N reps with the v11 config and records:
- tool_calls (should be 1, or 0 if goal already met)
- leak_count (should be 0)
- first_tool + first_args (does the model pick a reasonable action?)
- completion_tokens, latency

Histories are hand-built to represent interesting realistic states: mid-craft
recovery, post-PARTIAL, post-goal, etc.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from craft.tools import TOOLS
from rollout_tuned import SYSTEM_PROMPT, STOP_SEQUENCES, MAX_TOKENS, TEMPERATURE, REASONING_EFFORT, MODEL, BASE_URL

client = OpenAI(base_url=BASE_URL, api_key="ollama")
LEAK_RE = re.compile(r"<\|[a-zA-Z_/]+\|?>|<[a-zA-Z_/]+\|>")

RESULTS_DIR = Path(__file__).parent / "results"


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _asst(tool_name: str, args: str, tc_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tool_name, "arguments": args}}],
    }


def _tool(tc_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


# ---- Hand-built histories representing interesting realistic states ----

# Scenario 1: Mid-progression, just smelted iron, ready for iron_pickaxe craft.
H_POST_SMELT = [
    _user("Begin. Current inventory:\n  slot 0: 3x minecraft:oak_log\n  slot 1: 1x minecraft:wooden_pickaxe\n  slot 6: 12x minecraft:cobblestone"),
    _asst("mine_stone", '{"quantity":10}', "c1"),
    _tool("c1", "acquired 10 more (now have 22 mine_stone-drops; last type mined: stone)\n\nCurrent inventory:\n  slot 0: 3x minecraft:oak_log\n  slot 1: 1x minecraft:wooden_pickaxe\n  slot 6: 22x minecraft:cobblestone"),
    _asst("craft", '{"item":"stone_pickaxe","quantity":1}', "c2"),
    _tool("c2", "crafted 1x minecraft:stone_pickaxe\n\nCurrent inventory:\n  slot 1: 1x minecraft:stone_pickaxe\n  slot 6: 19x minecraft:cobblestone\n  slot 7: 1x minecraft:wooden_pickaxe"),
    _asst("mine_iron", '{"quantity":5}', "c3"),
    _tool("c3", "acquired 5 more (now have 5 mine_iron-drops; last type mined: iron_ore)\n\nCurrent inventory:\n  slot 0: 5x minecraft:raw_iron\n  slot 1: 1x minecraft:stone_pickaxe\n  slot 6: 19x minecraft:cobblestone"),
    _asst("smelt", '{"input":"raw_iron","count":5}', "c4"),
    _tool("c4", "smelted 5x minecraft:iron_ingot (fuel: 2x minecraft:stick, 3x minecraft:oak_planks)\n\nCurrent inventory:\n  slot 0: 5x minecraft:iron_ingot\n  slot 1: 1x minecraft:stone_pickaxe\n  slot 6: 19x minecraft:cobblestone"),
    # Next turn: model should pick craft(iron_pickaxe) or mine_wood (for sticks). Either is reasonable.
]

# Scenario 2: After 3 PARTIAL descend responses — should trigger recovery rule.
H_TRIPLE_PARTIAL = [
    _user("Begin. Current inventory:\n  slot 0: 1x minecraft:iron_sword\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 8x minecraft:dirt"),
    _asst("descend", '{"target_y":-58}', "c1"),
    _tool("c1", "PARTIAL: at y=53 (started y=64, target y=-58) — Baritone unreachable; try descend(-58) again, or surface() if stuck\n\nCurrent inventory:\n  slot 0: 1x minecraft:iron_sword\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 8x minecraft:dirt"),
    _asst("travel", '{"direction":"east","distance":20}', "c2"),
    _tool("c2", "traveled east: moved 20 blocks (target 20); now at (53,53,-15)\n\nCurrent inventory:\n  slot 0: 1x minecraft:iron_sword\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 12x minecraft:dirt"),
    _asst("descend", '{"target_y":-58}', "c3"),
    _tool("c3", "PARTIAL: at y=48 (started y=53, target y=-58) — Baritone unreachable; try descend(-58) again, or surface() if stuck\n\nCurrent inventory:\n  slot 0: 1x minecraft:iron_sword\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 12x minecraft:dirt"),
    # Next turn: model should ideally pick descend(18) per recovery rule (current_y - 30 = 48 - 30 = 18) or mine_stone/travel.
]

# Scenario 3: Post-goal — has 3 diamonds. Should emit no tool call.
H_POST_GOAL = [
    _user("Begin. Current inventory:\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 32x minecraft:cobblestone"),
    _asst("descend", '{"target_y":8}', "c1"),
    _tool("c1", "descended to y=8 (target was 8)\n\nCurrent inventory:\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 32x minecraft:cobblestone"),
    _asst("mine_diamond", '{"quantity":3}', "c2"),
    _tool("c2", "acquired 3 more (now have 3 mine_diamond-drops; last type mined: deepslate_diamond_ore)\n\nCurrent inventory:\n  slot 1: 1x minecraft:iron_pickaxe\n  slot 6: 32x minecraft:cobblestone\n  slot 13: 3x minecraft:diamond"),
    # Next turn: model should emit NO tool call (goal met).
]

# Scenario 4: Just crafted wooden_pickaxe — should pick mine_stone or craft axe/shovel.
H_WOODEN_PICK = [
    _user("Begin. Current inventory:\n  (empty)"),
    _asst("mine_wood", '{"quantity":3}', "c1"),
    _tool("c1", "acquired 3 more (now have 3 mine_wood-drops; last type mined: oak_log)\n\nCurrent inventory:\n  slot 0: 3x minecraft:oak_log"),
    _asst("craft", '{"item":"wooden_pickaxe","quantity":1}', "c2"),
    _tool("c2", "crafted 1x minecraft:wooden_pickaxe\n\nCurrent inventory:\n  slot 0: 1x minecraft:wooden_pickaxe\n  slot 1: 1x minecraft:oak_log\n  slot 2: 1x minecraft:crafting_table"),
    # Next turn: model should pick mine_stone or craft wooden_axe/shovel
]


HISTORIES = {
    "post_smelt": (H_POST_SMELT, {"craft", "mine_wood"}),
    "triple_partial": (H_TRIPLE_PARTIAL, {"descend", "mine_stone", "travel", "surface"}),
    "post_goal": (H_POST_GOAL, set()),  # expect no tool call
    "wooden_pick": (H_WOODEN_PICK, {"mine_stone", "craft"}),
}


def measure(history: list[dict], expected_tools: set[str], rep: int) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=STOP_SEQUENCES,
            extra_body={"reasoning_effort": REASONING_EFFORT},
        )
    except Exception as e:
        return {"error": repr(e), "rep": rep, "latency": time.monotonic() - t0}
    elapsed = time.monotonic() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    tcs = msg.tool_calls or []
    leak_count = len(LEAK_RE.findall(content))
    first_tool = tcs[0].function.name if tcs else None
    first_args = tcs[0].function.arguments if tcs else None
    # Correct = matches expected behavior
    if not expected_tools:  # post-goal: expect zero tool calls
        correct = len(tcs) == 0 and leak_count == 0
    else:
        correct = len(tcs) == 1 and leak_count == 0 and first_tool in expected_tools
    return {
        "rep": rep,
        "tool_calls": len(tcs),
        "first_tool": first_tool,
        "first_args": first_args,
        "leak_count": leak_count,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "latency": elapsed,
        "finish_reason": resp.choices[0].finish_reason,
        "correct": correct,
        "content_snippet": content[:300],
    }


def run(reps: int = 10, log_path: Path | None = None) -> None:
    if log_path is None:
        log_path = RESULTS_DIR / f"history_{int(time.time())}.jsonl"
    print(f"multi-turn history probe: {len(HISTORIES)} scenarios × {reps} reps → {log_path}")
    with log_path.open("a") as f:
        for name, (history, expected) in HISTORIES.items():
            print(f"\n--- {name} (prompt_msgs={len(history)}, expected={expected or 'no-tool-call'}) ---")
            for r in range(reps):
                rec = measure(history, expected, r)
                rec["scenario"] = name
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if "error" in rec:
                    print(f"  rep{r}: ERROR {rec['error'][:80]}")
                else:
                    mark = "✓" if rec["correct"] else "✗"
                    pt = rec.get("prompt_tokens", "?")
                    ct = rec.get("completion_tokens", "?")
                    args_snip = (rec['first_args'] or '')[:40]
                    print(f"  rep{r} {mark} tools={rec['tool_calls']} leak={rec['leak_count']} pt={pt} ct={ct} t={rec['latency']:.1f}s → {rec['first_tool']}({args_snip})")


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
