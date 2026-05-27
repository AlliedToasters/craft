#!/usr/bin/env python3
"""Overlay a rollout transcript onto its recorded video as a soft subtitle track.

Closes the perceptual gap when reviewing tapes: at any moment you can see which
tool call is active (or that the agent is "thinking" — a null tool call while the
model decides) and the result that call returned. The result is an "oracle": the
overlay is post-hoc, so it shows the outcome the instant the call begins, before
the on-screen action finishes resolving.

Per turn the transcript already decomposes into the two states we render:
  - plan window  -> "thinking…" + a snippet of the model's reasoning/content
  - exec window  -> the active tool(args) + its returned outcome

The .ass track is soft-muxed into an .mkv beside the source video (video stream
copied, no re-encode), so the overlay is toggleable in any player (mpv/VLC).

Alignment is exact: the JSONL header carries `video_started_at` (ffmpeg launch ≈
video t=0) and each turn carries a wall-clock `t`, so a turn maps to video time
via (t - video_started_at). Tapes predating that instrumentation fall back to
cumulative per-turn durations anchored at the header `started_at` (best-effort;
may drift).

Usage:
  python scripts/overlay_transcript.py results/foo-agent4.jsonl
  python scripts/overlay_transcript.py results/foo-agent4.jsonl --out /tmp/foo.mkv
  python scripts/overlay_transcript.py results/foo-agent4.jsonl --ass-only
  python scripts/overlay_transcript.py results/foo-agent4.jsonl --burn  # hardsub .mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys


# ---- ASS colours are &HBBGGRR (BGR, low byte = red). ----
C_TOOL = r"{\c&H00FFFF&}"   # yellow  — the active tool call
C_ORACLE = r"{\c&HFFFFFF&}"  # white   — the returned outcome
C_DIM = r"{\c&HAAAAAA&}"    # grey    — reasoning snippet
RESET = r"{\c&HFFFFFF&}"


def _ass_time(s: float) -> str:
    """Seconds -> ASS H:MM:SS.cc (centiseconds)."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    sec = int(s)
    cs = int(round((s - sec) * 100))
    if cs == 100:
        cs = 0
        sec += 1
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _sanitize(text: str) -> str:
    """Make a string safe for an ASS Text field: collapse whitespace and
    neutralize chars libass would interpret as override syntax."""
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _truncate(text: str, n: int) -> str:
    text = _sanitize(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_args(raw: str | dict) -> str:
    """Tool arguments -> compact `k=v, k=v`. `raw` is the JSON string the model
    emitted (or already a dict)."""
    obj = raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return _truncate(raw, 60)
    if isinstance(obj, dict):
        if not obj:
            return ""
        return ", ".join(f"{k}={_truncate(str(v), 40)}" for k, v in obj.items())
    return _truncate(str(obj), 60)


def _probe(video: str) -> tuple[int, int, float | None]:
    """(width, height, duration_seconds) via ffprobe; safe defaults on failure.

    Resolution sets ASS PlayResX/Y so font sizes are sane; duration clamps the
    final caption so it doesn't outrun the footage."""
    w, h, dur = 854, 480, None
    if shutil.which("ffprobe") is None:
        return w, h, dur
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        m = re.match(r"(\d+)x(\d+)", out)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
    except (subprocess.SubprocessError, OSError):
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dur = float(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return w, h, dur


def _read_transcript(path: str) -> tuple[dict, list[dict], dict[int, dict]]:
    """-> (header, turns sorted by turn, llm-records keyed by turn)."""
    header: dict = {}
    turns: list[dict] = []
    llms: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            t = rec.get("_type")
            if t == "header":
                header = rec
            elif t == "turn":
                turns.append(rec)
            elif t == "llm":
                tn = rec.get("turn")
                if tn is not None:
                    llms[tn] = rec
    turns.sort(key=lambda r: r.get("turn", 0))
    return header, turns, llms


def _num(v) -> str:
    """Render whole-valued floats as ints (hp 20.0 -> 20)."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _thinking_text(llm: dict | None) -> str:
    """Short snippet of what the model 'thought' during the plan window.

    Some models (qwen via the chat-template quirk) emit the tool call itself as
    text in `content`/`reasoning`. That's the serialized call, not reasoning, and
    duplicates the tool line below it — drop it so 'thinking…' stays clean.
    """
    if not llm:
        return ""
    resp = llm.get("response") or {}
    raw = (resp.get("reasoning") or resp.get("content") or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"</?tool_call>", " ", raw)
    raw = re.sub(r"<\|[^>]*\|>", " ", raw)
    low = raw.lower()
    if '"name"' in low and ('"arguments"' in low or '"parameters"' in low):
        return ""
    snippet = re.sub(r"\s+", " ", raw).strip()
    return _truncate(snippet, 160) if snippet else ""


def build_events(header: dict, turns: list[dict], llms: dict[int, dict],
                 duration: float | None) -> list[tuple[float, float, str, str]]:
    """-> list of (start_s, end_s, style, text) ASS dialogue events.

    Each turn yields a top-left HUD event (whole-turn span) plus a bottom caption
    that switches from 'thinking…' to 'tool -> outcome' at the plan/exec boundary.
    """
    if not turns:
        return []

    video_t0 = header.get("video_started_at")
    # Offset of each turn's start in video-time.
    offsets: list[float] = []
    if video_t0 is not None and all("t" in r for r in turns):
        offsets = [float(r["t"]) - float(video_t0) for r in turns]
    else:
        # Fallback: anchor at header started_at (if it predates ffmpeg, offsets
        # are relative to that) and accumulate per-turn wall time. Drifts, but
        # keeps the tool ordering legible on un-instrumented tapes.
        anchor = header.get("started_at")
        base = (float(turns[0].get("t", anchor or 0)) - float(anchor)) if anchor else 0.0
        acc = max(0.0, base)
        for r in turns:
            offsets.append(acc)
            acc += float(r.get("total_s", 0.0) or 0.0)

    events: list[tuple[float, float, str, str]] = []
    n = len(turns)
    for i, r in enumerate(turns):
        off = max(0.0, offsets[i])
        # Caption persists until the next turn begins (the on-screen action
        # continues through the exec + ctx tail). Last turn runs to EOF.
        if i + 1 < n:
            turn_end = max(off, offsets[i + 1])
        elif duration is not None:
            turn_end = max(off, duration)
        else:
            turn_end = off + float(r.get("total_s", 0.0) or 0.0)
        if duration is not None:
            turn_end = min(turn_end, duration)
        if turn_end <= off:
            turn_end = off + 0.5

        plan_s = float(r.get("plan_s", 0.0) or 0.0)
        plan_end = min(off + plan_s, turn_end)
        tn = r.get("turn")
        tool = r.get("tool", "?")
        args = _fmt_args(r.get("args", ""))
        call = f"{tool}({args})" if args else f"{tool}()"
        outcome = _truncate(r.get("outcome", ""), 220)

        # HUD (top-left), whole-turn span.
        hud_bits = [f"turn {tn}"]
        if "food" in r:
            hud_bits.append(f"food {_num(r['food'])}")
        if "health" in r:
            hud_bits.append(f"hp {_num(r['health'])}")
        if "day_count" in r:
            hud_bits.append(f"day {_num(r['day_count'])}")
        events.append((off, turn_end, "HUD", _sanitize(" | ".join(hud_bits))))

        # Bottom caption — thinking window, then tool/oracle window.
        think = _thinking_text(llms.get(tn))
        if plan_end > off:
            txt = r"{\i1}● thinking…" + r"{\i0}"
            if think:
                txt += r"\N" + C_DIM + think + RESET
            events.append((off, plan_end, "Caption", txt))

        exec_txt = C_TOOL + "▶ " + _sanitize(call) + RESET
        if outcome:
            exec_txt += r"\N" + C_ORACLE + "→ " + outcome + RESET
        events.append((plan_end, turn_end, "Caption", exec_txt))

    return events


def write_ass(path: str, events: list[tuple[float, float, str, str]],
              width: int, height: int) -> None:
    base_fs = max(14, round(height / 22))   # ~22 at 480p
    hud_fs = max(11, round(height / 30))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,{base_fs},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,3,2,1,2,20,20,18,1
Style: HUD,DejaVu Sans,{hud_fs},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,3,2,0,7,12,12,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for start, end, style, text in events:
        if end <= start:
            continue
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{text}"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def soft_mux(video: str, ass: str, out: str) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video, "-i", ass,
        "-map", "0:v:0", "-map", "1:0",
        "-c:v", "copy", "-c:s", "ass",
        "-metadata:s:s:0", "title=transcript",
        "-disposition:s:0", "default",
        out,
    ]
    return subprocess.run(cmd).returncode == 0


def burn_in(video: str, ass: str, out: str) -> bool:
    # libass filter; escape the path's special chars for the filtergraph.
    esc = ass.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video, "-vf", f"ass='{esc}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        out,
    ]
    return subprocess.run(cmd).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="rollout transcript (.jsonl)")
    ap.add_argument("--video", help="source video (default: header 'video' or <stem>.mp4)")
    ap.add_argument("--out", help="output path (default: <video-stem>.subbed.mkv, or .mp4 with --burn)")
    ap.add_argument("--ass-only", action="store_true", help="write the .ass sidecar and stop (no mux)")
    ap.add_argument("--burn", action="store_true", help="hardsub into an .mp4 (re-encode) instead of soft-muxing")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"transcript not found: {args.jsonl}", file=sys.stderr)
        return 2

    header, turns, llms = _read_transcript(args.jsonl)
    if not turns:
        print("no turn records in transcript — nothing to overlay", file=sys.stderr)
        return 1

    stem = re.sub(r"\.jsonl$", "", args.jsonl)
    video = args.video or header.get("video") or (stem + ".mp4")

    ass_path = stem + ".transcript.ass"

    if args.ass_only or not os.path.exists(video):
        # Still need a resolution for sane font sizing; probe if the video is
        # present, else assume the default MC window.
        w, h, dur = _probe(video) if os.path.exists(video) else (854, 480, None)
        events = build_events(header, turns, llms, dur)
        write_ass(ass_path, events, w, h)
        if not os.path.exists(video):
            print(f"video not found ({video}); wrote subtitle sidecar only: {ass_path}",
                  file=sys.stderr)
            return 0 if args.ass_only else 1
        print(f"wrote {ass_path} ({len(turns)} turns)")
        return 0

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found — cannot mux", file=sys.stderr)
        return 1

    w, h, dur = _probe(video)
    events = build_events(header, turns, llms, dur)
    write_ass(ass_path, events, w, h)

    if header.get("video_started_at") is None:
        print("[warn] tape predates video_started_at — alignment reconstructed "
              "from per-turn durations and may drift", file=sys.stderr)

    if args.burn:
        out = args.out or (stem + ".subbed.mp4")
        ok = burn_in(video, ass_path, out)
    else:
        out = args.out or (stem + ".subbed.mkv")
        ok = soft_mux(video, ass_path, out)

    if not ok:
        print("ffmpeg failed; the .ass sidecar is still at " + ass_path, file=sys.stderr)
        return 1
    mode = "hardsubbed" if args.burn else "soft-muxed (toggle in player)"
    print(f"{mode}: {out}\nsidecar: {ass_path} ({len(turns)} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
