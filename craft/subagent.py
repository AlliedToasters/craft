"""Subagent scaffolding for orthogonal synthesis tasks.

A subagent is a single-shot, non-tool-using LLM call that condenses a raw
data payload into natural language for the planning agent. The pilot use
case is situational synthesis (look_around): raw block + entity scans of
the surroundings → 2-3 sentences of terrain/hazards/resources.

Design:
- One generic `synthesize()` entry point; callers build prompt + payload.
- Defaults to Qwen3-4B local (fast, free). The model knob stays exposed so
  the ablation harness can sweep qwen / gemma / haiku side-by-side.
- No conversation history, no tool calls — single completion. The point of
  a subagent is exactly that it does NOT carry agent state.

Pattern reuse: drop in new tools (`describe_chunk`, `summarize_damage`,
`describe_cave`, ...) that build their own prompt + payload and call
synthesize(). Three or four callers is when extracting a base class
starts paying for itself; until then, callers compose freely.
"""

from __future__ import annotations

import json

from craft.llm import _anthropic, _is_anthropic, _ollama


QWEN_MODEL = "hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:F16"
DEFAULT_SUBAGENT_MODEL = QWEN_MODEL


def synthesize(
    task_prompt: str,
    payload: dict | list | str,
    *,
    model: str = DEFAULT_SUBAGENT_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """Run a single-shot LLM call over a raw payload. Returns text.

    task_prompt is the system message (the role/instructions for the
    synthesizer). payload is JSON-encoded as the user turn unless it's
    already a string. Falls through to Anthropic if `model` looks like a
    claude id, otherwise Ollama.
    """
    if isinstance(payload, (dict, list)):
        payload_str = json.dumps(payload, separators=(",", ":"))
    else:
        payload_str = str(payload)

    if _is_anthropic(model):
        return _synthesize_anthropic(
            task_prompt, payload_str,
            model=model, max_tokens=max_tokens, temperature=temperature,
        )
    return _synthesize_ollama(
        task_prompt, payload_str,
        model=model, max_tokens=max_tokens, temperature=temperature,
    )


def _synthesize_ollama(
    task_prompt: str, payload_str: str,
    *, model: str, max_tokens: int, temperature: float,
) -> str:
    resp = _ollama().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": payload_str},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|channel|>", "<|tool_response>"],
        extra_body={"reasoning_effort": "low"},
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    # Qwen3 (and similar thinking models) via Ollama route plain text into
    # message.reasoning when the chat template doesn't emit a separate answer
    # channel — same upstream quirk as the tool-call reasoning-fallback in
    # llm.py. For synthesis there's no separate "answer" turn to wait for,
    # so the reasoning IS the answer. Read it back.
    reasoning = (resp.choices[0].model_dump().get("message") or {}).get("reasoning") or ""
    return reasoning.strip()


def _synthesize_anthropic(
    task_prompt: str, payload_str: str,
    *, model: str, max_tokens: int, temperature: float,
) -> str:
    resp = _anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=task_prompt,
        messages=[{"role": "user", "content": payload_str}],
    )
    return "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", None) == "text"
    ).strip()


if __name__ == "__main__":
    # Smoke test: synthesize a trivial fake payload across configured models.
    fake = {
        "player_pos": [100, 70, 200],
        "biome": "plains",
        "blocks": [
            {"x": 100, "y": 69, "z": 200, "id": "minecraft:grass_block"},
            {"x": 101, "y": 69, "z": 200, "id": "minecraft:sand"},
            {"x": 100, "y": 70, "z": 201, "id": "minecraft:water"},
        ],
    }
    prompt = (
        "You are a scout for a Minecraft agent. Describe the surroundings in "
        "2-3 sentences. Mention terrain, hazards, and cardinal hints "
        "(+x=east, -x=west, +z=south, -z=north). Be concise."
    )
    print("--- qwen3:4b ---")
    try:
        print(synthesize(prompt, fake))
    except Exception as e:
        print(f"qwen error: {e}")
