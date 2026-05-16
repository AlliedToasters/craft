"""Rollout using the tuned prompt + sampling config found via probe_prompt.py.

Same control flow as craft.agent.run() but with:
  - v6_minimal_antileak system prompt (shorter, anti-leak preamble)
  - max_tokens=512 (gives reasoning headroom on complex states like iron_armored)
  - stop sequences ['<|channel|>', '<|tool_response>'] (Harmony leak truncation)
  - temperature=0.3, reasoning_effort='low' (unchanged from baseline)

Pass5 stability: 98% clean (59/60) on v6_minimal_antileak across 6 scenarios × 10 reps.
Pass3 baseline (v0 + default sampling): 33% 1tc rate, 0% clean.
"""

from __future__ import annotations

import sys
import time

import requests
from openai import OpenAI

from craft.tools import TOOLS, dispatch, HOMUNCULUS_BASE
from craft.agent import _fetch_inventory

import os

MODEL = "gemma-4-vanilla-q4-32k:latest"
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

SYSTEM_PROMPT = (
    "OUTPUT FORMAT: Respond with a single tool call. Leave the content field empty. "
    "Do NOT emit <|channel|>, <|tool_response|>, <|message|>, or any other '<|...|>' tokens — "
    "they are not part of your output schema.\n\n"
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
    "\nGoal completion: if your inventory already shows the goal item (a diamond), "
    "the run is OVER. Emit NO tool call. The harness ends the run when you stop "
    "emitting tool calls. Don't loop on surface() or other busywork.\n"
    "\nPARTIAL-recovery rules:\n"
    "- A 'PARTIAL' outcome means Baritone didn't reach the target. The most useful response is rarely to retry the exact same action.\n"
    "- If descend(N) returns PARTIAL, the target was too far. Pick a smaller Δy — current_y - 30 is a good first try, not -58.\n"
    "- If you get 2 PARTIALs on the same approach, switch strategy: mine_stone(8) to dig by hand, or travel to a different column.\n"
    "- Never call surface() repeatedly when already at surface (Δy < 2). Pick a new action.\n"
)

STOP_SEQUENCES = ["<|channel|>", "<|tool_response>"]
MAX_TOKENS = 1024
TEMPERATURE = 0.3
REASONING_EFFORT = "low"

_client = OpenAI(base_url=BASE_URL, api_key="ollama")


def chat_with_tools_tuned(messages, tools):
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=STOP_SEQUENCES,
        extra_body={"reasoning_effort": REASONING_EFFORT},
    )
    msg = resp.choices[0].message
    return msg.tool_calls or [], msg.content or ""


def run(max_turns: int = 30) -> None:
    initial_inv = _fetch_inventory()
    if initial_inv:
        opening = f"Begin. Read the inventory carefully before deciding the next action.\n\n{initial_inv}"
    else:
        opening = "Begin. (Inventory unavailable — homunculus may be offline.)"

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening},
    ]

    print(f"=== TUNED rollout — max_turns={max_turns}, max_tokens={MAX_TOKENS}, stop={STOP_SEQUENCES} ===")
    print("starting in 3s...")
    time.sleep(3)

    for turn in range(1, max_turns + 1):
        print(f"\n=== turn {turn}/{max_turns}: planning ===")
        tool_calls, content = chat_with_tools_tuned(messages, TOOLS)
        if content:
            print(f"[content] {content!r}")
        if not tool_calls:
            print("=== no tool call returned; stopping ===")
            break

        if len(tool_calls) > 1:
            print(f"!! WARNING: planner emitted {len(tool_calls)} tool calls; executing only the first")
            for extra in tool_calls[1:]:
                print(f"   discarded: {extra.function.name}({extra.function.arguments})")

        tc = tool_calls[0]
        name = tc.function.name
        args = tc.function.arguments
        print(f"=== turn {turn}: executing {name}({args}) ===")

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": args}}],
        })

        outcome = dispatch(name, args)
        print(f"=== turn {turn} outcome: {outcome} ===")

        try:
            equip_resp = requests.post(f"{HOMUNCULUS_BASE}/equip", timeout=5.0)
            equip_data = equip_resp.json() if equip_resp.ok else {}
            if equip_data.get("success") and equip_data.get("changes"):
                print(f"[equip] {equip_data.get('message', '')}")
        except requests.RequestException as e:
            print(f"[equip] failed (non-fatal): {e}")

        inv_str = _fetch_inventory()
        if inv_str:
            print(f"[inventory]\n{inv_str}")
            full_outcome = f"{outcome}\n\n{inv_str}"
        else:
            full_outcome = outcome

        messages.append({"role": "tool", "tool_call_id": tc.id, "content": full_outcome})

    print("\n=== rollout complete ===")


if __name__ == "__main__":
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(max_turns=turns)
