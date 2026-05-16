"""Probe: does gemma reliably emit tool calls via Ollama's OpenAI-compatible API?

Throwaway script. Tests four things:
1. Does Ollama+gemma respect the `tools` field at all?
2. When it emits a tool call, are the args structured correctly?
3. With multiple tools, does it pick the right one?
4. Does it hallucinate tool calls when none apply?
"""

from __future__ import annotations

import json
import time

from openai import OpenAI

import os

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = "gemma-4-vanilla-q4-32k:latest"

client = OpenAI(base_url=BASE_URL, api_key="ollama")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mine_wood",
            "description": "Mine wood logs from any nearby tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of logs to acquire.",
                    }
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "craft",
            "description": "Craft an item from materials in the inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Item id, e.g. 'wooden_pickaxe'.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to craft.",
                    },
                },
                "required": ["item", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_stone",
            "description": "Mine cobblestone with a pickaxe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of cobblestone to acquire.",
                    }
                },
                "required": ["quantity"],
            },
        },
    },
]

SYSTEM = (
    "You are an autonomous Minecraft agent. Your long-term goal is to acquire a diamond. "
    "Each turn, call exactly one tool to make progress. Be decisive."
)


def probe(
    label: str,
    user_msg: str,
    *,
    think: bool | None = None,
    reasoning_effort: str | None = None,
) -> None:
    suffix_parts = []
    if think is not None:
        suffix_parts.append(f"think={think}")
    if reasoning_effort is not None:
        suffix_parts.append(f"effort={reasoning_effort}")
    suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
    print(f"\n=== {label}{suffix} ===")
    print(f"user: {user_msg!r}")
    extra: dict = {}
    if think is not None:
        extra["think"] = think
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0.3,
        extra_body=extra or None,
    )
    elapsed = time.monotonic() - t0
    msg = resp.choices[0].message
    print(f"content: {msg.content!r}")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = tc.function.arguments
            print(f"tool_call: {tc.function.name}({args})")
    else:
        print("tool_call: <none>")
    print(f"finish_reason: {resp.choices[0].finish_reason}")
    u = resp.usage
    if u is not None:
        toks_per_sec = u.completion_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"timing: {elapsed:.2f}s · prompt={u.prompt_tokens} "
            f"completion={u.completion_tokens} · {toks_per_sec:.1f} tok/s"
        )
    else:
        print(f"timing: {elapsed:.2f}s · usage=<none>")


CASES = [
    ("baseline / first step", "What is the first step toward the goal?"),
    ("explicit quantity, mine wood", "Mine 5 logs."),
    ("implicit quantity, mine wood", "I need wood."),
    ("explicit, mine stone", "Get 8 cobblestone."),
    ("explicit, craft", "Craft a wooden pickaxe."),
    ("multi-step phrasing (should pick one)", "Make a wooden pickaxe."),
    ("no-tool case", "What's the weather like?"),
    ("free-form goal", "Acquire enough wood to make a crafting table."),
]


if __name__ == "__main__":
    for label, msg in CASES:
        try:
            probe(label, msg)
        except Exception as e:
            print(f"\n=== {label} ===\nERROR: {e!r}")

    print("\n\n########  reasoning_effort=low sweep  ########")
    for label, msg in CASES[:4]:
        try:
            probe(label, msg, reasoning_effort="low")
        except Exception as e:
            print(f"\n=== {label} [effort=low] ===\nERROR: {e!r}")
