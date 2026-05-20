"""Replay a single turn from a rollout transcript against an LLM.

Pulls the verbatim ``prompt_messages`` from a ``_type:"llm"`` record and
re-sends it. Useful for: (a) sanity-checking a model is still deterministic
on a given context, (b) re-rolling a non-deterministic sample to compare
responses, (c) testing targeted prompt tweaks — dump the prompt, edit any
message (system instructions, opening text, STATE block, even a tool result),
reload, and resend.

Usage:

    # Replay turn 3 with the same model the transcript used.
    python -m craft.replay results/foo.jsonl --turn 3

    # Same prompt, swap the model.
    python -m craft.replay results/foo.jsonl --turn 3 --model claude-sonnet-4-6

    # Sample N times to eyeball response variance.
    python -m craft.replay results/foo.jsonl --turn 3 --n 5

    # Prompt-tweak workflow.
    python -m craft.replay results/foo.jsonl --turn 3 --dump /tmp/p3.json
    $EDITOR /tmp/p3.json   # rewrite system prompt, STATE, whatever
    python -m craft.replay --prompt-file /tmp/p3.json --model "$QWEN"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from craft.llm import DEFAULT_MODEL, chat_with_tools
from craft.tools import TOOLS


def load_turn_from_transcript(path: Path, turn: int) -> tuple[list[dict], str]:
    """Find the _type:"llm" record for the given turn; return (messages, model)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_type") == "llm" and rec.get("turn") == turn:
                return rec["prompt_messages"], rec.get("model", DEFAULT_MODEL)
    raise SystemExit(f"no _type:'llm' record with turn={turn} in {path}")


def load_prompt_file(path: Path) -> list[dict]:
    """Load prompt_messages from a JSON file (bare list or wrapped)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("prompt_messages"), list):
        return data["prompt_messages"]
    raise SystemExit(
        f"expected JSON list or {{'prompt_messages': [...]}} in {path}, got {type(data).__name__}"
    )


def format_response(content: str, reasoning: str, tool_calls) -> str:
    """Render the parsed model response for human reading."""
    parts: list[str] = []
    if content:
        parts.append(f"--- content ---\n{content}")
    if reasoning:
        parts.append(f"--- reasoning ---\n{reasoning}")
    if tool_calls:
        parts.append("--- tool_calls ---")
        for tc in tool_calls:
            parts.append(f"  {tc.function.name}({tc.function.arguments})")
    if not parts:
        parts.append("(empty response — no content, no reasoning, no tool_calls)")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "transcript", nargs="?", type=Path,
        help="path to JSONL transcript (omit if using --prompt-file)",
    )
    ap.add_argument(
        "--turn", type=int,
        help="turn number to pull from the transcript",
    )
    ap.add_argument(
        "--model", default=None,
        help=f"override the model (default: transcript's recorded model, or {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--prompt-file", type=Path,
        help="load prompt_messages from a JSON file instead of a transcript+turn",
    )
    ap.add_argument(
        "--dump", type=Path,
        help="dump prompt_messages to this JSON file (pretty-printed) and exit — no LLM call",
    )
    ap.add_argument(
        "--n", type=int, default=1,
        help="number of samples to draw (default 1; >1 useful for variance check)",
    )
    args = ap.parse_args()

    if args.prompt_file:
        prompt_messages = load_prompt_file(args.prompt_file)
        model = args.model or DEFAULT_MODEL
        source = f"prompt file {args.prompt_file}"
    else:
        if args.transcript is None or args.turn is None:
            ap.error("provide either --prompt-file OR (transcript + --turn)")
        prompt_messages, recorded_model = load_turn_from_transcript(args.transcript, args.turn)
        model = args.model or recorded_model
        source = f"{args.transcript} turn {args.turn}"

    if args.dump:
        args.dump.write_text(json.dumps(prompt_messages, indent=2, ensure_ascii=False))
        print(f"wrote {len(prompt_messages)} messages from {source} to {args.dump}")
        return

    print(f"=== replaying {len(prompt_messages)} messages from {source} against {model} ===")
    for i in range(args.n):
        if args.n > 1:
            print(f"\n=== sample {i + 1}/{args.n} ===")
        tool_calls, content, reasoning, _raw = chat_with_tools(prompt_messages, TOOLS, model=model)
        print(format_response(content, reasoning, tool_calls))


if __name__ == "__main__":
    main()
