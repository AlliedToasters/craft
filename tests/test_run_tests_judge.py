"""Unit tests for the JSONL judge functions in run_tests.py.

`_judge_pass_rate(path, threshold)` — single-file judge used by sequential
runs and by concurrent runs after a phase completes (per-spec call).

`_judge_combined(name, threshold)` — multi-file glob merge used by the
concurrent phased runner; combines `results/suite-<name>-agent*.jsonl`
into one pass-rate.

**Why this matters**: these two functions are the suite-wide trust surface.
Every test in `run_tests.py` reports `passed_bool` based on what these
return. If they silently miscount, the suite says "13/13 PASS" while
something's broken; if they bomb on a malformed line, the whole concurrent
phase fails and no signal escapes.

Pinned asymmetry to be aware of:
    - `_judge_pass_rate` FAILS LOUDLY on malformed JSONL (returns False +
      diagnostic).
    - `_judge_combined` SILENTLY SKIPS malformed lines (`except
      json.JSONDecodeError: pass`).
The asymmetry is intentional: in concurrent mode we'd rather salvage
partial signal across agents than mark the whole phase failed. These
tests pin both behaviors so any future "let's unify" refactor surfaces
the trade-off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_tests import _judge_combined, _judge_pass_rate


# ---------------------------------------------- helpers


def _write_jsonl(path: Path, records: list[dict | str]) -> None:
    """Write records to a JSONL file. Strings are written as-is (use for
    injecting malformed lines / blanks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in records:
        if isinstance(r, str):
            lines.append(r)
        else:
            lines.append(json.dumps(r))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------- _judge_pass_rate: missing / empty


class TestJudgePassRateMissingOrEmpty:
    def test_missing_file_returns_false(self, tmp_path):
        path = tmp_path / "ghost.jsonl"
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert "no output JSONL" in summary
        assert details == {"iters": 0}

    def test_empty_file_returns_false(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert "empty" in summary
        assert details == {"iters": 0}

    def test_blank_lines_only_returns_empty(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text("\n\n   \n\n")
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert "empty" in summary


# ---------------------------------------------- _judge_pass_rate: pass/fail


class TestJudgePassRatePassFail:
    def test_single_pass(self, tmp_path):
        path = tmp_path / "p.jsonl"
        _write_jsonl(path, [{"iter": 1, "passed": True}])
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is True
        assert details["iters"] == 1
        assert details["passed"] == 1
        assert details["rate"] == 1.0
        assert "1/1 passed" in summary

    def test_single_fail_with_fail_reason(self, tmp_path):
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [
            {"iter": 1, "passed": False, "fail_reason": "spawn rejected"}
        ])
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert details["fail_reasons"] == ["iter 1: spawn rejected"]
        assert "spawn rejected" in summary

    def test_fail_uses_fatal_error_when_no_fail_reason(self, tmp_path):
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [
            {"iter": 2, "passed": False, "fatal_error": "crash X"}
        ])
        _ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["fail_reasons"] == ["iter 2: crash X"]

    def test_fail_reason_takes_precedence_over_fatal_error(self, tmp_path):
        """Per code: `r.get("fail_reason") or r.get("fatal_error") or "unknown"`.
        fail_reason wins."""
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [{
            "iter": 1, "passed": False,
            "fail_reason": "specific", "fatal_error": "generic",
        }])
        _ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["fail_reasons"] == ["iter 1: specific"]

    def test_fail_unknown_when_neither_field_present(self, tmp_path):
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [{"iter": 3, "passed": False}])
        _ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["fail_reasons"] == ["iter 3: unknown"]

    def test_missing_iter_renders_question_mark(self, tmp_path):
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [{"passed": False, "fail_reason": "huh"}])
        _ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["fail_reasons"] == ["iter ?: huh"]

    def test_missing_passed_field_counts_as_fail(self, tmp_path):
        """`r.get("passed")` is None → falsy → fail. Defensive."""
        path = tmp_path / "f.jsonl"
        _write_jsonl(path, [{"iter": 1}])
        ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert details["passed"] == 0

    def test_mixed_pass_fail_counts(self, tmp_path):
        path = tmp_path / "m.jsonl"
        _write_jsonl(path, [
            {"iter": 1, "passed": True},
            {"iter": 2, "passed": False, "fail_reason": "x"},
            {"iter": 3, "passed": True},
            {"iter": 4, "passed": True},
        ])
        _ok, _summary, details = _judge_pass_rate(path, threshold=0.5)
        assert details["iters"] == 4
        assert details["passed"] == 3
        assert details["rate"] == 0.75


# ---------------------------------------------- _judge_pass_rate: threshold


class TestJudgePassRateThreshold:
    """Pass-rate vs threshold uses `>=` — pin the boundary explicitly."""

    def test_exactly_at_threshold_passes(self, tmp_path):
        path = tmp_path / "t.jsonl"
        records = [{"iter": i, "passed": True} for i in range(9)]
        records.append({"iter": 10, "passed": False, "fail_reason": "x"})
        _write_jsonl(path, records)
        ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["rate"] == 0.9
        assert ok is True, "rate == threshold must pass (>=, not >)"

    def test_just_below_threshold_fails(self, tmp_path):
        path = tmp_path / "t.jsonl"
        records = [{"iter": i, "passed": True} for i in range(8)]
        records.extend([
            {"iter": 9, "passed": False, "fail_reason": "a"},
            {"iter": 10, "passed": False, "fail_reason": "b"},
        ])
        _write_jsonl(path, records)
        ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert details["rate"] == 0.8
        assert ok is False

    def test_zero_threshold_always_passes(self, tmp_path):
        """threshold=0.0 → any non-empty file passes (rate >= 0)."""
        path = tmp_path / "z.jsonl"
        _write_jsonl(path, [{"iter": 1, "passed": False, "fail_reason": "x"}])
        ok, _summary, _details = _judge_pass_rate(path, threshold=0.0)
        assert ok is True


# ---------------------------------------------- _judge_pass_rate: summary format


class TestJudgePassRateSummaryFormat:
    """Summary string format is consumed by the human-readable suite output —
    pin the truncation and reason-joining behavior."""

    def test_summary_shape_all_pass(self, tmp_path):
        path = tmp_path / "p.jsonl"
        _write_jsonl(path, [{"iter": 1, "passed": True}] * 5)
        _ok, summary, _details = _judge_pass_rate(path, threshold=0.9)
        # No fail reasons → no trailing "—" section
        assert "5/5 passed" in summary
        assert "rate=1.00" in summary
        assert "threshold=0.90" in summary
        assert "—" not in summary

    def test_summary_truncates_at_three_fail_reasons(self, tmp_path):
        path = tmp_path / "f.jsonl"
        records = [
            {"iter": i, "passed": False, "fail_reason": f"reason_{i}"}
            for i in range(1, 6)  # 5 failures
        ]
        _write_jsonl(path, records)
        _ok, summary, _details = _judge_pass_rate(path, threshold=0.9)
        assert "reason_1" in summary
        assert "reason_2" in summary
        assert "reason_3" in summary
        assert "reason_4" not in summary
        assert "reason_5" not in summary
        assert "(+2 more)" in summary

    def test_summary_no_more_suffix_at_exactly_three_fails(self, tmp_path):
        path = tmp_path / "f.jsonl"
        records = [
            {"iter": i, "passed": False, "fail_reason": f"r{i}"}
            for i in range(1, 4)
        ]
        _write_jsonl(path, records)
        _ok, summary, _details = _judge_pass_rate(path, threshold=0.9)
        # All three shown, no "+N more"
        assert "r1" in summary and "r2" in summary and "r3" in summary
        assert "more" not in summary


# ---------------------------------------------- _judge_pass_rate: malformed


class TestJudgePassRateMalformed:
    """Single-file judge FAILS LOUDLY on bad JSON. This is by design —
    sequential runs benefit from immediate visibility into a corrupted
    log rather than silent under-counting. (Contrast with _judge_combined
    which silently skips bad lines.)"""

    def test_malformed_line_fails_with_diagnostic(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            '{"iter": 1, "passed": true}\n'
            'not-json-at-all\n'
            '{"iter": 3, "passed": true}\n'
        )
        ok, summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is False
        assert "malformed JSONL line" in summary
        assert details == {"iters": 0}

    def test_skips_blank_lines_mid_file(self, tmp_path):
        """Blank lines in the middle don't trip the JSON parser — they're
        stripped before json.loads."""
        path = tmp_path / "b.jsonl"
        path.write_text(
            '{"iter": 1, "passed": true}\n'
            '\n'
            '   \n'
            '{"iter": 2, "passed": true}\n'
        )
        ok, _summary, details = _judge_pass_rate(path, threshold=0.9)
        assert ok is True
        assert details["iters"] == 2


# ---------------------------------------------- _judge_combined: missing


class TestJudgeCombinedMissing:
    """`_judge_combined` reads `results/` relative to cwd. Use
    monkeypatch.chdir(tmp_path) to redirect."""

    def test_no_files_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ok, summary, details, paths = _judge_combined("ghost", threshold=0.9)
        assert ok is False
        assert "no output JSONL" in summary
        assert "ghost" in summary
        assert details == {"iters": 0}
        assert paths == []


# ---------------------------------------------- _judge_combined: fallback


class TestJudgeCombinedFallback:
    """If no `*-agent*.jsonl` files exist, fall back to the bare
    `suite-<name>.jsonl` (sequential-mode output)."""

    def test_falls_back_to_non_agent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        single = tmp_path / "results" / "suite-mytest.jsonl"
        _write_jsonl(single, [
            {"iter": 1, "passed": True},
            {"iter": 2, "passed": True},
        ])
        ok, summary, details, paths = _judge_combined("mytest", threshold=0.9)
        assert ok is True
        assert details["iters"] == 2
        assert details["agent_count"] == 1
        assert "across 1 agent(s)" in summary
        # Fallback path is constructed relative to cwd (not via the glob), so
        # compare by name rather than absolute path.
        assert len(paths) == 1
        assert paths[0].name == "suite-mytest.jsonl"

    def test_agent_files_take_precedence_over_single(self, tmp_path, monkeypatch):
        """If any *-agent*.jsonl exists, the bare file is IGNORED — the code
        flow is `if not candidates and single.exists(): candidates = [single]`.
        This pins that the bare file doesn't get merged in alongside."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        # Agent file: 2 iters
        _write_jsonl(tmp_path / "results" / "suite-x-agent0.jsonl", [
            {"iter": 1, "passed": True},
            {"iter": 2, "passed": True},
        ])
        # Bare file with FAIL — should be ignored
        _write_jsonl(tmp_path / "results" / "suite-x.jsonl", [
            {"iter": 1, "passed": False, "fail_reason": "stale"},
        ])
        ok, _summary, details, _paths = _judge_combined("x", threshold=0.9)
        assert ok is True
        assert details["iters"] == 2
        assert details["passed"] == 2
        assert details["agent_count"] == 1


# ---------------------------------------------- _judge_combined: merge


class TestJudgeCombinedMerge:
    def test_merges_three_agents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i in range(3):
            _write_jsonl(
                tmp_path / "results" / f"suite-shelter-agent{i}.jsonl",
                [{"iter": 1, "passed": True}, {"iter": 2, "passed": True}],
            )
        ok, summary, details, paths = _judge_combined("shelter", threshold=0.9)
        assert ok is True
        assert details["iters"] == 6
        assert details["passed"] == 6
        assert details["agent_count"] == 3
        assert "across 3 agent(s)" in summary
        assert len(paths) == 3

    def test_mixed_pass_fail_across_agents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # agent0: 1 pass / 1 fail; agent1: 2 fail
        _write_jsonl(tmp_path / "results" / "suite-x-agent0.jsonl", [
            {"iter": 1, "passed": True},
            {"iter": 2, "passed": False, "fail_reason": "a"},
        ])
        _write_jsonl(tmp_path / "results" / "suite-x-agent1.jsonl", [
            {"iter": 1, "passed": False, "fail_reason": "b"},
            {"iter": 2, "passed": False, "fail_reason": "c"},
        ])
        ok, _summary, details, _paths = _judge_combined("x", threshold=0.9)
        assert ok is False
        assert details["iters"] == 4
        assert details["passed"] == 1
        assert details["rate"] == 0.25

    def test_empty_jsonls_after_glob(self, tmp_path, monkeypatch):
        """Globbed files exist but contain no iters → "empty for '<name>'"."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results" / "suite-x-agent0.jsonl").parent.mkdir(parents=True)
        (tmp_path / "results" / "suite-x-agent0.jsonl").write_text("")
        ok, summary, details, paths = _judge_combined("x", threshold=0.9)
        assert ok is False
        assert "empty for 'x'" in summary
        assert details == {"iters": 0}
        assert len(paths) == 1


# ---------------------------------------------- _judge_combined: malformed


class TestJudgeCombinedMalformed:
    """Critical asymmetry: combined judge SILENTLY SKIPS malformed JSON
    (unlike _judge_pass_rate which fails). This is by design — in
    concurrent mode one corrupted agent log shouldn't blow away the
    pass-rate signal from the other agents.

    If anyone "unifies" the two judges, this test fails — surfacing the
    deliberate divergence so the trade-off is reconsidered explicitly."""

    def test_malformed_line_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "results" / "suite-x-agent0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"iter": 1, "passed": true}\n'
            'garbage-not-json\n'
            '{"iter": 3, "passed": true}\n'
        )
        ok, _summary, details, _paths = _judge_combined("x", threshold=0.9)
        # 2 valid lines counted, garbage skipped without affecting outcome
        assert ok is True
        assert details["iters"] == 2
        assert details["passed"] == 2

    def test_all_malformed_treated_as_empty(self, tmp_path, monkeypatch):
        """If every line is malformed → behaves like an empty file."""
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "results" / "suite-x-agent0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("garbage1\ngarbage2\n")
        ok, summary, details, _paths = _judge_combined("x", threshold=0.9)
        assert ok is False
        assert "empty" in summary
        assert details == {"iters": 0}


# ---------------------------------------------- _judge_combined: summary format


class TestJudgeCombinedSummaryFormat:
    def test_summary_includes_agent_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i in (0, 1, 2):
            _write_jsonl(
                tmp_path / "results" / f"suite-x-agent{i}.jsonl",
                [{"iter": 1, "passed": True}],
            )
        _ok, summary, _details, _paths = _judge_combined("x", threshold=0.9)
        assert "across 3 agent(s)" in summary

    def test_summary_truncates_fail_reasons_at_three(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        records = [
            {"iter": i, "passed": False, "fail_reason": f"r{i}"}
            for i in range(1, 6)
        ]
        _write_jsonl(tmp_path / "results" / "suite-x-agent0.jsonl", records)
        _ok, summary, _details, _paths = _judge_combined("x", threshold=0.9)
        assert "r1" in summary and "r2" in summary and "r3" in summary
        assert "r4" not in summary
        assert "(+2 more)" in summary
