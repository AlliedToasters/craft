"""LLM client with provider dispatch.

Two providers, selected via the `model` argument:

  * non-``claude-`` models → local Ollama (OpenAI-SDK compatible). URL is
    read from the ``OLLAMA_BASE_URL`` env var (default
    ``http://localhost:11434/v1``); point it at a remote Ollama box by
    exporting e.g. ``OLLAMA_BASE_URL=http://my-gpu-box.local:11434/v1``.
  * ``claude-…`` → Anthropic API. Used as a high-capability control. Fast,
    deterministic-ish, useful for A/B latency probes and for unblocking
    substrate iteration when local-model plan_s tails dominate.

Returns a normalized ``(tool_calls, content)`` pair regardless of provider
so the agent loop doesn't care which backend produced it. Tool calls are
exposed via simple namespace objects mimicking the OpenAI SDK shape
(``tc.id``, ``tc.function.name``, ``tc.function.arguments``) — the agent
loop already speaks that shape, no caller changes needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_DEFAULT_MODEL = "gemma-4-vanilla-q4-32k:latest"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5"

DEFAULT_MODEL = OLLAMA_DEFAULT_MODEL

_ollama_client: OpenAI | None = None
_anthropic_client = None  # lazy — only import anthropic if used


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude-") or model.startswith("anthropic/")


def _is_human(model: str) -> bool:
    return model == "human"


def _ollama() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _ollama_client


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # noqa: WPS433
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def chat(messages: list[dict], *, model: str = DEFAULT_MODEL) -> str:
    """Plain chat (no tools). Returns the assistant text. Ollama path only —
    Anthropic chat without tools is rarely needed for this project."""
    resp = _ollama().chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str = DEFAULT_MODEL,
):
    """Tool-aware chat completion. Dispatches by model name.

    Returns (tool_calls, content). tool_calls is a list of SimpleNamespace
    objects with ``.id``, ``.function.name``, ``.function.arguments`` (JSON
    string) — matches the OpenAI SDK shape the agent loop already consumes.
    """
    if _is_human(model):
        return _chat_with_tools_human(messages, tools, model=model)
    if _is_anthropic(model):
        return _chat_with_tools_anthropic(messages, tools, model=model)
    return _chat_with_tools_ollama(messages, tools, model=model)


# ─────────────────────── Ollama / gemma path ────────────────────────────────


def _extract_from_reasoning(reasoning: str) -> list:
    """Parse a tool call out of the reasoning field.

    Qwen3 (and possibly other thinking models) via Ollama route the entire
    model output — including the tool call JSON — into message.reasoning
    instead of message.tool_calls. The tool call arrives in the format:
        {"name": "mine_wood", "arguments": {"quantity": 3}}
        </tool_call>

    Tech debt (Option B): the root fix is correcting the Ollama modelfile
    chat template for Qwen3 so tool call responses are cleanly separated
    from thinking tokens and surfaced in tool_calls. Until then this parses
    the JSON from reasoning as a fallback.
    """
    cleaned = (
        reasoning.strip()
        .replace("<tool_call>", "")
        .replace("</tool_call>", "")
        .strip()
    )
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            args = obj["arguments"] if isinstance(obj["arguments"], dict) else {}
            return [SimpleNamespace(
                id="reasoning-fallback-0",
                type="function",
                function=SimpleNamespace(name=obj["name"], arguments=json.dumps(args)),
            )]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _chat_with_tools_ollama(messages, tools, *, model):
    resp = _ollama().chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=1024,
        stop=["<|channel|>", "<|tool_response>"],
        extra_body={"reasoning_effort": "low"},
    )
    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []
    content = msg.content or ""
    if not tool_calls and not content:
        reasoning = (resp.choices[0].model_dump().get("message") or {}).get("reasoning") or ""
        if reasoning:
            tool_calls = _extract_from_reasoning(reasoning)
            if tool_calls:
                print(f"[llm] reasoning-fallback: extracted {tool_calls[0].function.name} from reasoning field", flush=True)
    return tool_calls, content


# ─────────────────────── Anthropic / Haiku path ─────────────────────────────


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI function-tool schemas to Anthropic tool schemas."""
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _openai_messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI message format to (system_text, anthropic_messages).

    OpenAI:
      system  → {"role": "system", "content": "…"}
      asst    → {"role": "assistant", "content": "", "tool_calls": [{id, function:{name,arguments}}]}
      tool    → {"role": "tool", "tool_call_id": id, "content": "…"}

    Anthropic:
      system  → separate `system` argument (not in messages)
      asst    → {"role": "assistant", "content": [{type:"tool_use", id, name, input}]}
      tool    → {"role": "user", "content": [{type:"tool_result", tool_use_id, content}]}
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content") or "")
            continue
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
            continue
        if role == "assistant":
            blocks: list[dict] = []
            text = m.get("content") or ""
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in (m.get("tool_calls") or []):
                # tc is dict-shaped per agent.py's serialization.
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except (TypeError, json.JSONDecodeError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or "tc",
                    "name": tc.get("function", {}).get("name", ""),
                    "input": args,
                })
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            continue
        if role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": m.get("content") or "",
                }],
            })
            continue
    return ("\n\n".join(p for p in system_parts if p), out)


def _chat_with_tools_anthropic(messages, tools, *, model):
    client = _anthropic()
    system_text, anth_messages = _openai_messages_to_anthropic(messages)
    anth_tools = _openai_tools_to_anthropic(tools)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_text,
        messages=anth_messages,
        tools=anth_tools,
    )
    # Pull tool_use + text out of the content blocks.
    tool_calls = []
    text_parts: list[str] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            args_json = json.dumps(getattr(block, "input", {}) or {})
            tc = SimpleNamespace(
                id=getattr(block, "id", "tc"),
                type="function",
                function=SimpleNamespace(
                    name=getattr(block, "name", ""),
                    arguments=args_json,
                ),
            )
            tool_calls.append(tc)
        elif btype == "text":
            text_parts.append(getattr(block, "text", ""))
    return tool_calls, "".join(text_parts)


# ─────────────────────── Human / driver path ────────────────────────────────
# A REPL that hands the turn-N planning step to a person typing at the
# terminal. Plumbed through `chat_with_tools` so /equip, shelter-watch,
# message-trim, pre-dispatch death checks, and every other per-turn substrate
# event runs identically to an LLM run. Same invocation shape as the others:
#   python -m craft.agent 50 survive_shelter --permadeath \
#       --start-phase dawn --random-spawn-range 20000 --model human
#
# In the human path "planning latency" is the time the person spent thinking
# — accumulates into plan_s_total just like the LLM does, which is a feature
# (lets you compare your own decision cadence against a model's).


def _chat_with_tools_human(messages, tools, *, model):  # noqa: ARG001 (model unused)
    import shlex
    import readline  # noqa: F401 — enables arrow-key history at the prompt

    schemas = {t["function"]["name"]: t["function"] for t in tools}
    tool_names = list(schemas)

    # Show the LLM-equivalent context: the last user/tool message is what
    # a model would be planning on top of *this* turn. On turn 1 this is the
    # opening message (stats+inventory). On later turns the agent loop has
    # already echoed [stats]/[inventory]/[smelts] to the terminal in the
    # previous turn's tail, so the content here is the tool result + state
    # block being re-injected.
    last = messages[-1] if messages else None
    if last and last.get("role") in ("user", "tool"):
        print("\n--- context (what the model would see this turn) ---")
        print(last.get("content") or "")
        print("--- end context ---")

    def _parse_value(raw: str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def _parse_args(tokens):
        args = {}
        for tok in tokens:
            if "=" not in tok:
                raise ValueError(f"expected key=value, got {tok!r}")
            k, _, v = tok.partition("=")
            args[k] = _parse_value(v)
        return args

    def _print_help():
        print("Tools:")
        for name in tool_names:
            params = schemas[name].get("parameters", {}).get("properties", {})
            req = set(schemas[name].get("parameters", {}).get("required", []))
            sig = " ".join(
                f"{k}=<{v.get('type','?')}>" + ("" if k in req else "?")
                for k, v in params.items()
            )
            print(f"  {name} {sig}")
        print("Commands: help | schema <tool> | system | quit")

    def _print_schema(name):
        fn = schemas.get(name)
        if fn is None:
            print(f"unknown tool: {name}")
        else:
            print(json.dumps(fn, indent=2))

    def _print_system():
        # First message is always the system prompt in this codebase.
        sys = next((m for m in messages if m.get("role") == "system"), None)
        if not sys:
            print("(no system message in transcript)")
        else:
            print("--- system ---")
            print(sys.get("content") or "")
            print("--- end ---")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            # Empty tool_calls → agent loop's "no tool call returned; stopping"
            # path fires and the rollout ends gracefully. JSONL still flushes.
            return [], ""
        if not line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"parse error: {e}")
            continue
        head, rest = tokens[0], tokens[1:]
        if head in ("quit", "exit", "q"):
            return [], ""
        if head in ("help", "?"):
            _print_help()
            continue
        if head == "schema":
            if not rest:
                print("usage: schema <tool>")
                continue
            _print_schema(rest[0])
            continue
        if head == "system":
            _print_system()
            continue
        if head not in schemas:
            print(f"unknown tool: {head} (try 'help')")
            continue
        try:
            args = _parse_args(rest)
        except ValueError as e:
            print(f"arg parse error: {e}")
            continue
        tc = SimpleNamespace(
            id=f"human-{id(args):x}",
            type="function",
            function=SimpleNamespace(name=head, arguments=json.dumps(args)),
        )
        return [tc], ""


if __name__ == "__main__":
    # Smoke test both providers if their backends are reachable.
    print("--- ollama ---")
    try:
        print(chat([{"role": "user", "content": "hello"}]))
    except Exception as e:
        print(f"ollama error: {e}")
    print()
    print("--- anthropic (claude-haiku-4-5) ---")
    try:
        tcs, content = chat_with_tools(
            [
                {"role": "system", "content": "You are a calculator."},
                {"role": "user", "content": "Use the add tool to compute 2+3."},
            ],
            [{
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                },
            }],
            model="claude-haiku-4-5",
        )
        print(f"tool_calls={tcs}, content={content!r}")
    except Exception as e:
        print(f"anthropic error: {e}")
