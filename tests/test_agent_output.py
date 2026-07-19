"""Tests for agent rollout fallback output readers."""

import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claude_teams import agent_output as ao
from claude_teams import server_simple
from claude_teams.agent_output import (
    _claude_session_id,
    _claude_started_at,
    _codex_candidate_dirs,
    _content_text,
    _first_json_object,
    _iter_lines_reverse,
    _last_claude_message,
    _last_codex_message,
    _normalize_path,
    _parse_timestamp,
    _resolve_path_text,
    _rollout_contains_token,
    _started_after,
    _truncate_tail,
    claude_correlation_token,
    codex_correlation_token,
    read_claude_output,
    read_codex_output,
)
from claude_teams.backends import codex as codex_module
from claude_teams.backends import process_base
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import SpawnRequest


def _write_jsonl(path: Path, rows: list[Any], mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _codex_path(home: Path, spawned_at: float, name: str) -> Path:
    day = datetime.fromtimestamp(spawned_at, tz=UTC)
    return (
        home
        / ".codex"
        / "sessions"
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.day:02d}"
        / name
    )


def _timestamp_at(epoch: float, offset: float = 0.0) -> str:
    return (
        datetime.fromtimestamp(epoch + offset, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _codex_meta(
    cwd: Path, *, session_id: str = "session-id", timestamp: str | None = None
) -> dict:
    return {
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "timestamp": timestamp or _timestamp_at(1_762_969_000.0),
            "cwd": str(cwd),
            "originator": "codex-tui",
        },
    }


def _codex_message(text: str, *, phase: str = "final_answer") -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "phase": phase,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _claude_message(
    content: object, *, session_id: str = "session-id", timestamp: str | None = None
) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp or _timestamp_at(1_762_969_000.0),
        "sessionId": session_id,
        "message": {
            "role": "assistant",
            "content": content,
        },
    }


def _claude_project_dir(home: Path, cwd: Path) -> Path:
    encoded = re.sub(r"[\\/:]", "-", str(cwd.resolve()))
    return home / ".claude" / "projects" / encoded


def _make_request(tmp_path: Path, **overrides: object) -> SpawnRequest:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="sonnet",
        agent_type="worker",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="lead",
        permission_mode="bypass",
    )
    return replace(default, **overrides)


def test_read_codex_output_returns_latest_matching_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    other_cwd = tmp_path / "other"
    spawned_at = 1_762_969_000.0

    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-old.jsonl"),
        [_codex_meta(cwd), _codex_message("old")],
        spawned_at + 10,
    )
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-wrong-cwd.jsonl"),
        [_codex_meta(other_cwd), _codex_message("wrong")],
        spawned_at + 30,
    )
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-new.jsonl"),
        [
            _codex_meta(cwd),
            _codex_message("working", phase="commentary"),
            _codex_message("latest", phase="commentary"),
            '{"type": "response_item"',
        ],
        spawned_at + 20,
    )

    output = read_codex_output(spawned_at, str(cwd))

    assert output is not None
    assert output.last_activity_at == spawned_at + 20
    assert output.last_message == "latest"
    assert output.backend_session_id == "session-id"
    assert output.rollout_path.endswith("rollout-new.jsonl")


def test_read_codex_output_returns_session_without_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-empty.jsonl"),
        [_codex_meta(cwd), {"type": "event_msg", "payload": {"msg": "tool"}}],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd))

    assert output is not None
    assert output.backend_session_id == "session-id"
    assert output.last_message is None


def test_read_codex_output_ignores_old_session_with_newer_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    old_meta = _codex_meta(
        cwd, session_id="old-session", timestamp=_timestamp_at(spawned_at, -10)
    )
    new_meta = _codex_meta(cwd, session_id="new-session")

    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-old-active.jsonl"),
        [old_meta, _codex_message("wrong")],
        spawned_at + 30,
    )
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-new-target.jsonl"),
        [new_meta, _codex_message("right")],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd))

    assert output is not None
    assert output.backend_session_id == "new-session"
    assert output.last_message == "right"


def test_read_codex_output_can_match_known_session_started_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    resume_spawned_at = spawned_at + 2 * 86_400

    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-resumed.jsonl"),
        [
            _codex_meta(
                cwd,
                session_id="known-session",
                timestamp=_timestamp_at(spawned_at),
            ),
            _codex_message("follow-up answer"),
        ],
        resume_spawned_at + 10,
    )

    output = read_codex_output(
        resume_spawned_at, str(cwd), backend_session_id="known-session"
    )

    assert output is not None
    assert output.backend_session_id == "known-session"
    assert output.last_message == "follow-up answer"


def test_read_codex_output_truncates_keeping_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    text = "".join(str(i % 10) for i in range(5000))
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-long.jsonl"),
        [_codex_meta(cwd), _codex_message(text)],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd), max_bytes=1000)

    assert output is not None
    assert output.last_message is not None
    assert len(output.last_message) <= 1000
    assert output.last_message.startswith("[truncated: showing last ")
    # The tail of the original text is retained.
    assert output.last_message.endswith(text[-100:])


def test_read_codex_output_does_not_truncate_short_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-short.jsonl"),
        [_codex_meta(cwd), _codex_message("all done")],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd))

    assert output is not None
    assert output.last_message == "all done"
    assert "truncated" not in output.last_message


def test_read_codex_output_default_budget_truncates_to_one_thousand_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    text = "x" * 2500
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-default-budget.jsonl"),
        [_codex_meta(cwd), _codex_message(text)],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd))

    assert output is not None
    assert output.last_message is not None
    assert len(output.last_message) <= 1000
    assert output.last_message.startswith("[truncated: showing last ")
    assert output.last_message.endswith(text[-100:])


def test_read_codex_output_small_budget_returns_raw_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-long.jsonl"),
        [_codex_meta(cwd), _codex_message("abcde")],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd), max_bytes=3)

    assert output is not None
    assert output.last_message == "cde"
    assert len(output.last_message) <= 3


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 10, 40, 41, 60])
def test_read_codex_output_tiny_budget_within_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    # Unique, non-repeating tail so a wrong-length slice would be observable.
    text = "".join(chr(ord("a") + (i % 26)) for i in range(200))
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-tiny.jsonl"),
        [_codex_meta(cwd), _codex_message(text)],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd), max_bytes=budget)

    assert output is not None
    assert output.last_message is not None
    assert len(output.last_message) <= budget
    # Whatever is returned must end with the genuine tail of the source text.
    assert output.last_message.endswith(text[-1])


def test_read_codex_output_truncates_multibyte_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    text = "åäö" * 1000
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-multibyte.jsonl"),
        [_codex_meta(cwd), _codex_message(text)],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd), max_bytes=1000)

    assert output is not None
    assert output.last_message is not None
    assert len(output.last_message) <= 1000
    assert output.last_message.endswith(text[-50:])


def test_read_claude_output_truncates_keeping_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    text = "".join(str(i % 10) for i in range(5000))
    _write_jsonl(
        _claude_project_dir(tmp_path, cwd) / "session.jsonl",
        [_claude_message([{"type": "text", "text": text}])],
        spawned_at + 10,
    )

    output = read_claude_output(spawned_at, str(cwd), max_bytes=1000)

    assert output is not None
    assert output.last_message is not None
    assert len(output.last_message) <= 1000
    assert output.last_message.startswith("[truncated: showing last ")
    assert output.last_message.endswith(text[-100:])


def _codex_user_prompt(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def test_read_codex_output_disambiguates_concurrent_agents_by_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    token_a = codex_correlation_token("smoke-a@sess")
    token_b = codex_correlation_token("smoke-b@sess")

    # Both spawned in the same cwd at ~the same time. rollout-b has the newer
    # mtime, so plain max(mtime) would wrongly bind agent-a to session-b.
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-a.jsonl"),
        [
            _codex_meta(cwd, session_id="session-a"),
            _codex_user_prompt(f"do work\n\n{token_a}"),
            _codex_message("from a"),
        ],
        spawned_at + 10,
    )
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-b.jsonl"),
        [
            _codex_meta(cwd, session_id="session-b"),
            _codex_user_prompt(f"do work\n\n{token_b}"),
            _codex_message("from b"),
        ],
        spawned_at + 30,
    )

    output = read_codex_output(spawned_at, str(cwd), correlation_token=token_a)

    assert output is not None
    assert output.backend_session_id == "session-a"
    assert output.last_message == "from a"
    assert output.rollout_path.endswith("rollout-a.jsonl")


def test_read_codex_output_falls_back_when_token_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    # No rollout carries the token (e.g. agent spawned before the marker
    # existed, or Codex has not flushed the prompt yet) -> latest mtime.
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-latest.jsonl"),
        [_codex_meta(cwd, session_id="session-x"), _codex_message("latest")],
        spawned_at + 20,
    )

    output = read_codex_output(
        spawned_at, str(cwd), correlation_token=codex_correlation_token("ghost@sess")
    )

    assert output is not None
    assert output.backend_session_id == "session-x"
    assert output.last_message == "latest"


def test_codex_build_command_embeds_correlation_token(tmp_path: Path) -> None:
    backend = CodexBackend()
    request = _make_request(
        tmp_path, agent_id="worker@sess-uuid", prompt="single line task"
    )

    cmd = backend.build_command(request)

    assert codex_correlation_token("worker@sess-uuid") in cmd[-1]


def test_read_claude_output_returns_latest_project_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    project_dir = _claude_project_dir(tmp_path, cwd)

    _write_jsonl(
        project_dir / "old.jsonl",
        [_claude_message([{"type": "text", "text": "old"}])],
        spawned_at + 10,
    )
    _write_jsonl(
        project_dir / "new.jsonl",
        [
            {"type": "ai-title", "sessionId": "session-id", "aiTitle": "Title"},
            _claude_message([{"type": "text", "text": "latest"}]),
            '{"type": "assistant"',
        ],
        spawned_at + 20,
    )

    output = read_claude_output(spawned_at, str(cwd))

    assert output is not None
    assert output.last_activity_at == spawned_at + 20
    assert output.last_message == "latest"
    assert output.backend_session_id == "session-id"
    assert output.rollout_path.endswith("new.jsonl")


def test_read_claude_output_accepts_string_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    _write_jsonl(
        _claude_project_dir(tmp_path, cwd) / "session.jsonl",
        [_claude_message("plain text")],
        spawned_at + 10,
    )

    output = read_claude_output(spawned_at, str(cwd))

    assert output is not None
    assert output.last_message == "plain text"


def test_read_claude_output_ignores_old_session_with_newer_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    project_dir = _claude_project_dir(tmp_path, cwd)

    _write_jsonl(
        project_dir / "old-active.jsonl",
        [
            _claude_message(
                [{"type": "text", "text": "wrong"}],
                session_id="old-session",
                timestamp=_timestamp_at(spawned_at, -10),
            )
        ],
        spawned_at + 30,
    )
    _write_jsonl(
        project_dir / "new-target.jsonl",
        [
            _claude_message(
                [{"type": "text", "text": "right"}],
                session_id="new-session",
            )
        ],
        spawned_at + 10,
    )

    output = read_claude_output(spawned_at, str(cwd))

    assert output is not None
    assert output.backend_session_id == "new-session"
    assert output.last_message == "right"


def test_read_claude_output_can_match_known_session_started_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    resume_spawned_at = spawned_at + 1_000
    project_dir = _claude_project_dir(tmp_path, cwd)

    _write_jsonl(
        project_dir / "resumed.jsonl",
        [
            _claude_message(
                [{"type": "text", "text": "follow-up answer"}],
                session_id="known-session",
                timestamp=_timestamp_at(spawned_at),
            )
        ],
        resume_spawned_at + 10,
    )

    output = read_claude_output(
        resume_spawned_at, str(cwd), backend_session_id="known-session"
    )

    assert output is not None
    assert output.backend_session_id == "known-session"
    assert output.last_message == "follow-up answer"


def _claude_user_prompt(text: str, *, session_id: str, timestamp: str) -> dict:
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": session_id,
        "message": {"role": "user", "content": text},
    }


def test_read_claude_output_disambiguates_concurrent_agents_by_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    project_dir = _claude_project_dir(tmp_path, cwd)
    token_a = claude_correlation_token("worker-a@sess")
    token_b = claude_correlation_token("worker-b@sess")

    # Both agents spawned in the same cwd at ~the same time. transcript-b has
    # the newer mtime, so plain max(mtime) would wrongly bind agent-a to
    # session-b. The correlation token makes the binding deterministic.
    _write_jsonl(
        project_dir / "transcript-a.jsonl",
        [
            _claude_user_prompt(
                f"do work\n\n{token_a}",
                session_id="session-a",
                timestamp=_timestamp_at(spawned_at),
            ),
            _claude_message(
                [{"type": "text", "text": "from a"}], session_id="session-a"
            ),
        ],
        spawned_at + 10,
    )
    _write_jsonl(
        project_dir / "transcript-b.jsonl",
        [
            _claude_user_prompt(
                f"do work\n\n{token_b}",
                session_id="session-b",
                timestamp=_timestamp_at(spawned_at),
            ),
            _claude_message(
                [{"type": "text", "text": "from b"}], session_id="session-b"
            ),
        ],
        spawned_at + 30,
    )

    output = read_claude_output(spawned_at, str(cwd), correlation_token=token_a)

    assert output is not None
    assert output.backend_session_id == "session-a"
    assert output.last_message == "from a"
    assert output.rollout_path.endswith("transcript-a.jsonl")


def test_read_claude_output_falls_back_when_token_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    project_dir = _claude_project_dir(tmp_path, cwd)

    # No transcript carries the token (e.g. Claude has not flushed the prompt
    # yet, or the agent was spawned before this marker existed) -> newest mtime.
    _write_jsonl(
        project_dir / "older.jsonl",
        [_claude_message([{"type": "text", "text": "older"}], session_id="session-x")],
        spawned_at + 10,
    )
    _write_jsonl(
        project_dir / "latest.jsonl",
        [_claude_message([{"type": "text", "text": "latest"}], session_id="session-y")],
        spawned_at + 20,
    )

    output = read_claude_output(
        spawned_at,
        str(cwd),
        correlation_token=claude_correlation_token("ghost@sess"),
    )

    assert output is not None
    assert output.backend_session_id == "session-y"
    assert output.last_message == "latest"


def test_read_claude_output_ignores_token_when_session_id_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    project_dir = _claude_project_dir(tmp_path, cwd)
    other_token = claude_correlation_token("worker-other@sess")

    # A newer transcript carries a *different* agent's token, but the exact
    # backend_session_id is known -> match by session id, ignore the token.
    _write_jsonl(
        project_dir / "known.jsonl",
        [
            _claude_message(
                [{"type": "text", "text": "known answer"}],
                session_id="known-session",
            )
        ],
        spawned_at + 10,
    )
    _write_jsonl(
        project_dir / "decoy.jsonl",
        [
            _claude_user_prompt(
                f"do work\n\n{other_token}",
                session_id="decoy-session",
                timestamp=_timestamp_at(spawned_at),
            ),
            _claude_message(
                [{"type": "text", "text": "decoy"}], session_id="decoy-session"
            ),
        ],
        spawned_at + 30,
    )

    output = read_claude_output(
        spawned_at,
        str(cwd),
        backend_session_id="known-session",
        correlation_token=other_token,
    )

    assert output is not None
    assert output.backend_session_id == "known-session"
    assert output.last_message == "known answer"


def test_claude_build_command_embeds_correlation_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = ClaudeCodeBackend()
    request = _make_request(
        tmp_path, agent_id="worker@sess-uuid", prompt="single line task"
    )

    cmd = backend.build_command(request)

    assert cmd[-1].startswith("single line task")
    assert claude_correlation_token("worker@sess-uuid") in cmd[-1]

    # Resume already knows the backend session id, so it must NOT inject the
    # marker (it would pollute the resumed transcript on every follow-up).
    resume_cmd = backend.build_resume_command(request, "claude-session-id")
    assert claude_correlation_token("worker@sess-uuid") not in resume_cmd[-1]


def test_claude_build_command_embeds_token_with_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = ClaudeCodeBackend()
    request = _make_request(
        tmp_path,
        agent_id="worker@sess-uuid",
        prompt="ignored when a prompt file is used",
        extra={"prompt_file_path": "C:\\tmp\\worker.prompt.txt"},
    )

    cmd = backend.build_command(request)

    # Even when the real prompt travels via a file, the marker must land in the
    # file-read instruction so it reaches the first recorded user message.
    assert "C:\\tmp\\worker.prompt.txt" in cmd[-1]
    assert claude_correlation_token("worker@sess-uuid") in cmd[-1]


def test_codex_resume_command_preserves_permissions_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        codex_module.process_manager,
        "provides_tty",
        lambda backend_type, *, is_interactive=False: True,
    )
    backend = CodexBackend()
    request = _make_request(
        tmp_path,
        model="gpt-5.3-codex",
        reasoning_effort="high",
        prompt="first line\nsecond line",
    )

    cmd = backend.build_resume_command(request, "codex-session-id")

    assert cmd[0] == "/usr/bin/codex"
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert cmd[cmd.index("-C") + 1] == str(tmp_path)
    assert "model_reasoning_effort=high" in cmd
    assert any(arg.startswith("mcp_servers.win-agent-teams.env=") for arg in cmd)
    # Interactive (TTY) resume keeps the visible TUI session.
    assert "exec" not in cmd
    assert cmd[-3] == "resume"
    assert cmd[-2] == "codex-session-id"
    # /usr/bin/codex is the native binary (not the cmd.exe shim), so the
    # multi-line prompt passes verbatim; the JSON-wrap fallback only applies
    # when launching through the npm codex.cmd shim.
    assert cmd[-1] == "first line\nsecond line"


def test_codex_resume_command_uses_exec_resume_without_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        codex_module.process_manager,
        "provides_tty",
        lambda backend_type, *, is_interactive=False: False,
    )
    backend = CodexBackend()
    request = _make_request(tmp_path, prompt="follow up")

    cmd = backend.build_resume_command(request, "codex-session-id")

    # Non-interactive resume entrypoint: ``codex exec resume <session-id>``
    # (the TUI aborts with "stdin is not a terminal" without a real TTY).
    assert cmd[0] == "/usr/bin/codex"
    assert cmd[1] == "exec"
    assert cmd[2] == "resume"
    assert cmd[3] == "codex-session-id"
    assert cmd[-1] == "follow up"


def test_claude_resume_command_preserves_permissions_and_mcp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = ClaudeCodeBackend()
    request = _make_request(
        tmp_path,
        extra={"mcp_config_path": "C:\\tmp\\worker.mcp.json"},
        prompt="follow up",
    )

    cmd = backend.build_resume_command(request, "claude-session-id")

    assert cmd[0] == "/usr/bin/claude"
    assert cmd[cmd.index("--resume") + 1] == "claude-session-id"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert cmd[cmd.index("--mcp-config") + 1] == "C:\\tmp\\worker.mcp.json"
    assert cmd[-2:] == ["--", "follow up"]


def test_read_claude_output_returns_none_without_matching_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert read_claude_output(1_762_969_000.0, str(tmp_path / "missing")) is None


@pytest.mark.asyncio
async def test_spawn_agent_persists_output_lookup_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBackend:
        def default_model(self) -> str:
            return "model"

        def resolve_model(self, model: str) -> str:
            return model

        def resolve_launch(
            self, model: str, reasoning_effort: str | None
        ) -> tuple[str, str | None]:
            return (model if model.strip() else self.default_model()), reasoning_effort

        def spawn(self, request: object) -> SimpleNamespace:
            return SimpleNamespace(process_handle="456")

    class FakeRegistry:
        def default_backend(self) -> str:
            return "codex"

        def get(self, backend: str) -> FakeBackend:
            assert backend == "codex"
            return FakeBackend()

    cwd = tmp_path / "work"
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", FakeRegistry())
    before = 1_762_969_000.0
    monkeypatch.setattr(server_simple.time, "time", lambda: before)

    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend="codex", cwd=str(cwd)
    )

    agents = server_simple._load_agents(result["session_id"])
    assert agents == [
        {
            "name": "worker",
            "pid": 456,
            "backend": "codex",
            "session_id": result["session_id"],
            "status": "running",
            "spawned_at": before,
            "cwd": str(cwd),
            "model": "model",
            "permission_mode": "bypass",
            "reasoning_effort": None,
            "create_token": None,
        }
    ]


@pytest.mark.asyncio
async def test_spawn_agent_deduplicates_name_within_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.next_pid = 456

        def default_model(self) -> str:
            return "model"

        def resolve_model(self, model: str) -> str:
            return model

        def resolve_launch(
            self, model: str, reasoning_effort: str | None
        ) -> tuple[str, str | None]:
            return (model if model.strip() else self.default_model()), reasoning_effort

        def spawn(self, request: object) -> SimpleNamespace:
            self.next_pid += 1
            return SimpleNamespace(process_handle=str(self.next_pid))

    class FakeRegistry:
        def __init__(self) -> None:
            self.backend = FakeBackend()

        def default_backend(self) -> str:
            return "codex"

        def get(self, backend: str) -> FakeBackend:
            assert backend == "codex"
            return self.backend

    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", FakeRegistry())

    first = await server_simple.spawn_agent("prompt", name="worker", backend="codex")
    second = await server_simple.spawn_agent("prompt", name="worker", backend="codex")

    assert first["name"] == "worker"
    assert second["name"] == "worker-2"
    agents = server_simple._load_agents(first["session_id"])
    assert [agent["name"] for agent in agents] == ["worker", "worker-2"]


@pytest.mark.asyncio
async def test_agent_send_message_to_lead_routes_to_parent_in_nested_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    result = await server_simple.send_message(to="lead", text="hello parent")

    assert result == {"success": True, "to": "parent"}
    inbox = session_base / "session-id" / "inbox-parent.jsonl"
    rows = [json.loads(line) for line in inbox.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "from": "child",
            "text": "hello parent",
            "ts": rows[0]["ts"],
        }
    ]


@pytest.mark.asyncio
async def test_send_message_defaults_recipient_to_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    # No `to` given: the lazy-but-correct call still reaches the parent.
    result = await server_simple.send_message(text="hi")

    assert result == {"success": True, "to": "parent"}
    inbox = session_base / "session-id" / "inbox-parent.jsonl"
    assert inbox.exists()


@pytest.mark.asyncio
async def test_send_message_unknown_recipient_routes_to_lead_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    # A typo'd / unknown recipient must not be silently written to a dead inbox.
    result = await server_simple.send_message(to="leed", text="hello")

    assert result["success"] is True
    assert result["to"] == "parent"
    assert "warning" in result
    assert "leed" in result["warning"]
    # Routed to the parent inbox; no stray inbox-leed.jsonl created.
    assert (session_base / "session-id" / "inbox-parent.jsonl").exists()
    assert not (session_base / "session-id" / "inbox-leed.jsonl").exists()


@pytest.mark.asyncio
async def test_send_message_orchestrator_alias_routes_to_lead_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    # "orchestrator" is a recognized synonym for the lead: route cleanly, no warning.
    result = await server_simple.send_message(to="orchestrator", text="hello")

    assert result == {"success": True, "to": "parent"}


@pytest.mark.asyncio
async def test_send_message_team_lead_alias_routes_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    # Claude Code's native teams convention: a subagent addresses "team-lead".
    result = await server_simple.send_message(to="team-lead", text="hello parent")

    assert result == {"success": True, "to": "parent"}


@pytest.mark.asyncio
async def test_root_lead_identity_defaults_to_team_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    # Root lead: IDENTITY is the root name, no parent above it.
    monkeypatch.setattr(server_simple, "IDENTITY", server_simple.ROOT_LEAD_NAME)
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "")
    (session_base / "session-id").mkdir(parents=True)

    assert server_simple.ROOT_LEAD_NAME == "team-lead"
    # The root lead's aliases resolve to its own inbox (team-lead), not a parent.
    result = await server_simple.send_message(to="lead", text="note to self")

    assert result == {"success": True, "to": "team-lead"}
    assert (session_base / "session-id" / "inbox-team-lead.jsonl").exists()


@pytest.mark.asyncio
async def test_list_agents_recovers_lead_session_after_mcp_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("WIN_AGENT_TEAMS_PARENT_ID", "parent-process")
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "")

    session_id = server_simple._create_session()
    server_simple._save_agents(
        session_id,
        [
            {
                "name": "worker",
                "pid": 123,
                "backend": "codex",
                "session_id": session_id,
                "status": "running",
                "spawned_at": 100.0,
                "cwd": str(work),
                "model": "model",
                "permission_mode": "bypass",
                "reasoning_effort": None,
            }
        ],
    )
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "running"),
    )

    result = await server_simple.list_agents()

    assert server_simple._session_id == session_id
    assert result[0]["name"] == "worker"
    assert result[0]["alive"] is True


@pytest.mark.asyncio
async def test_check_agent_returns_stable_empty_fallback_for_unknown_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    (tmp_path / "sessions" / "session-id").mkdir(parents=True)
    server_simple._save_agents("session-id", [])

    result = await server_simple.check_agent("missing")

    assert result == {
        "name": "missing",
        "state": "dead",
        "alive": False,
        "pid": None,
        "backend": None,
        "last_activity_at": None,
        "unread_count": 0,
        "last_line": "",
        "seq": 0,
        "truncated": False,
        "full_len": 0,
        "heartbeat_age_s": None,
        "stalled": False,
    }

    full_result = await server_simple.check_agent("missing", full=True)
    assert full_result["last_message"] is None
    assert full_result["backend_session_id"] is None


@pytest.mark.asyncio
async def test_check_agent_skips_rollout_for_legacy_agent_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    (tmp_path / "sessions" / "session-id").mkdir(parents=True)
    server_simple._save_agents(
        "session-id",
        [
            {
                "name": "worker",
                "pid": 123,
                "backend": "codex",
                "session_id": "session-id",
                "status": "running",
            }
        ],
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, f"{handle} exited"),
    )

    def fail_read(*args: object, **kwargs: object) -> None:
        pytest.fail("legacy agent records must not scan rollout logs")

    monkeypatch.setattr(server_simple, "read_codex_output", fail_read)

    result = await server_simple.check_agent("worker")

    assert result == {
        "name": "worker",
        "state": "dead",
        "alive": False,
        "pid": 123,
        "backend": "codex",
        "last_activity_at": None,
        "unread_count": 0,
        "last_line": "",
        "seq": 0,
        "truncated": False,
        "full_len": 0,
        "heartbeat_age_s": None,
        "stalled": False,
    }


@pytest.mark.asyncio
async def test_check_agent_persists_backend_session_id_from_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    (tmp_path / "sessions" / "session-id").mkdir(parents=True)
    server_simple._save_agents(
        "session-id",
        [
            {
                "name": "worker",
                "pid": 123,
                "backend": "codex",
                "session_id": "session-id",
                "status": "running",
                "spawned_at": 1.0,
                "cwd": str(tmp_path),
            }
        ],
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, f"{handle} exited"),
    )
    monkeypatch.setattr(
        server_simple,
        "read_codex_output",
        lambda spawned_at, cwd, **kwargs: SimpleNamespace(
            last_activity_at=10.0,
            last_message="done",
            backend_session_id="backend-session-id",
            busy_hint=False,
        ),
    )

    result = await server_simple.check_agent("worker", full=True)

    assert result["backend_session_id"] == "backend-session-id"
    agents = server_simple._load_agents("session-id")
    assert agents[0]["backend_session_id"] == "backend-session-id"


class _FakeRegistry:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def get(self, backend: str) -> object:
        assert backend == "codex"
        return self.backend


class _FakeResumeBackend:
    def __init__(self, *, supports_resume: bool = True) -> None:
        self.supports_resume_value = supports_resume
        self.resume_calls: list[tuple[SpawnRequest, str]] = []

    def supports_resume(self) -> bool:
        return self.supports_resume_value

    def default_model(self) -> str:
        return "model"

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        self.resume_calls.append((request, backend_session_id))
        return SimpleNamespace(process_handle="789")


def _write_agent_for_follow_up(tmp_path: Path, **overrides: object) -> None:
    record: dict[str, object] = {
        "name": "worker",
        "pid": 123,
        "backend": "codex",
        "session_id": "session-id",
        "status": "running",
        "spawned_at": 100.0,
        "cwd": str(tmp_path / "work"),
        "backend_session_id": "backend-session-id",
        "model": "model",
        "permission_mode": "bypass",
        "reasoning_effort": None,
    }
    record.update(overrides)
    server_simple._save_agents("session-id", [record])


def _setup_follow_up_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: object
) -> None:
    session_dir = tmp_path / "sessions" / "session-id"
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))


@pytest.mark.asyncio
async def test_follow_up_agent_resumes_dead_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)

    result = await server_simple.follow_up_agent("worker", "next prompt")

    assert result["success"] is True
    assert result["pid"] == 789
    assert result["replaced_existing"] is False
    request, backend_session_id = backend.resume_calls[0]
    assert backend_session_id == "backend-session-id"
    assert request.prompt == "next prompt"
    assert request.permission_mode == "bypass"
    agents = server_simple._load_agents("session-id")
    assert len(agents) == 1
    assert agents[0]["pid"] == 789
    assert agents[0]["spawned_at"] == 1_000.0


@pytest.mark.asyncio
async def test_follow_up_agent_refuses_busy_live_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )

    result = await server_simple.follow_up_agent("worker", "next prompt")

    assert result["success"] is False
    assert result["reason"] == "agent_busy"
    assert backend.resume_calls == []


@pytest.mark.asyncio
async def test_follow_up_agent_refuses_idle_live_agent_without_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(
        server_simple,
        "read_codex_output",
        lambda spawned_at, cwd, **kwargs: SimpleNamespace(
            last_activity_at=900.0,
            last_message="done",
            backend_session_id="backend-session-id",
            busy_hint=False,
        ),
    )

    result = await server_simple.follow_up_agent(
        "worker", "next prompt", replace_if_idle=False
    )

    assert result["success"] is False
    assert result["reason"] == "agent_idle_but_alive"
    assert backend.resume_calls == []


@pytest.mark.asyncio
async def test_follow_up_agent_replaces_idle_live_agent_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(
        server_simple,
        "read_codex_output",
        lambda spawned_at, cwd, **kwargs: SimpleNamespace(
            last_activity_at=900.0,
            last_message="done",
            backend_session_id="backend-session-id",
            busy_hint=False,
        ),
    )
    # The idle-but-alive agent is genuinely ours, so ownership holds and the
    # graceful shutdown proceeds (fail-closed gate returns True here).
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda handle, token: True
    )
    graceful_calls = []
    monkeypatch.setattr(
        server_simple.process_manager,
        "graceful_shutdown",
        lambda handle, timeout_s: graceful_calls.append((handle, timeout_s)) or True,
    )

    result = await server_simple.follow_up_agent(
        "worker", "next prompt", replace_if_idle=True
    )

    assert result["success"] is True
    assert result["replaced_existing"] is True
    assert graceful_calls == [("123", 5.0)]
    assert backend.resume_calls[0][1] == "backend-session-id"


@pytest.mark.asyncio
async def test_follow_up_agent_rejects_backend_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend(supports_resume=False)
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)

    result = await server_simple.follow_up_agent("worker", "next prompt")

    assert result["success"] is False
    assert result["reason"] == "backend_not_supported"


@pytest.mark.asyncio
async def test_follow_up_agent_recovers_session_after_mcp_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    session_base = tmp_path / "sessions"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("WIN_AGENT_TEAMS_PARENT_ID", "parent-process")
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    monkeypatch.setattr(server_simple, "_session_id", "")

    session_id = server_simple._create_session()
    server_simple._save_agents(
        session_id,
        [
            {
                "name": "worker",
                "pid": 123,
                "backend": "codex",
                "session_id": session_id,
                "status": "running",
                "spawned_at": 100.0,
                "cwd": str(work),
                "backend_session_id": "backend-session-id",
                "model": "model",
                "permission_mode": "bypass",
                "reasoning_effort": None,
            }
        ],
    )
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)

    result = await server_simple.follow_up_agent("worker", "next prompt")

    assert result["success"] is True
    assert result["session_id"] == session_id
    assert backend.resume_calls[0][1] == "backend-session-id"


# ==========================================================================
# Guard / error-branch coverage for private helpers.
#
# These exercise the defensive branches (early-return guards, malformed-input
# skips, and file-IO error handlers) that the public happy-path tests above do
# not reach. Fault injection uses monkeypatching of ``Path.open``/``stat``
# rather than filesystem permissions so it is portable and deterministic.
# ==========================================================================

# ---- read_codex_output / read_claude_output entry guards ------------------


def test_read_codex_output_rejects_bad_inputs(tmp_path, monkeypatch):
    # Isolate home so that, should the guard under test regress, execution
    # cannot reach the developer's real ~/.codex/sessions tree.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert read_codex_output(0.0, "/some/cwd") is None
    assert read_codex_output(-1.0, "/some/cwd") is None
    assert read_codex_output(100.0, "") is None


def test_read_codex_output_none_when_cwd_normalizes_empty(monkeypatch):
    # Contract-isolation: a non-empty cwd never *naturally* normalizes to "",
    # so force the helper to model the defensive branch at line 58-59.
    monkeypatch.setattr(ao, "_normalize_path", lambda value: "")
    assert read_codex_output(100.0, "/real/cwd") is None


def test_read_claude_output_rejects_bad_inputs(tmp_path, monkeypatch):
    # Isolate home so a regressed guard cannot reach ~/.claude/projects.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert read_claude_output(0.0, "/some/cwd") is None
    assert read_claude_output(100.0, "") is None


def test_read_claude_output_none_when_cwd_resolves_empty(monkeypatch):
    # Contract-isolation: mirrors the Codex normalization test above — a
    # non-empty cwd never naturally resolves to "", so the helper is forced to
    # model the defensive branch at lines 90-92 (not a real filesystem edge).
    monkeypatch.setattr(ao, "_resolve_path_text", lambda value: "")
    assert read_claude_output(100.0, "/real/cwd") is None


def test_read_codex_output_none_when_no_message_and_no_session_id(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    spawned_at = 1_762_969_000.0
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # session_meta with a non-string id (-> session_id None) and no assistant
    # response_item (-> last message None) hits the "return None" at line 70.
    meta = {
        "type": "session_meta",
        "payload": {
            "id": 12345,  # non-string -> session_id resolves to None
            "timestamp": _timestamp_at(spawned_at),
            "cwd": str(cwd.resolve()),
        },
    }
    path = _codex_path(home, spawned_at, "rollout-none.jsonl")
    _write_jsonl(path, [meta], spawned_at + 1)

    assert read_codex_output(spawned_at, str(cwd)) is None


def test_read_claude_output_none_when_no_message_and_no_session_id(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    spawned_at = 1_762_969_000.0
    cwd = tmp_path / "proj"
    cwd.mkdir()
    project_dir = _claude_project_dir(home, cwd)
    # A record with a timestamp (so _started_after passes) but no sessionId and
    # no assistant message -> last message None, session id None -> line 112.
    path = project_dir / "sess.jsonl"
    _write_jsonl(
        path,
        [{"type": "user", "timestamp": _timestamp_at(spawned_at)}],
        spawned_at + 1,
    )

    assert read_claude_output(spawned_at, str(cwd)) is None


# ---- _matching_codex_rollouts malformed-meta skips ------------------------


def test_read_codex_output_skips_non_session_meta_and_bad_payload(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    spawned_at = 1_762_969_000.0
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # Valid OLDER rollout with a matching cwd + an assistant message. It must be
    # the selection, proving both newer malformed rollouts were excluded.
    _write_jsonl(
        _codex_path(home, spawned_at, "rollout-good.jsonl"),
        [_codex_meta(cwd, timestamp=_timestamp_at(spawned_at)), _codex_message("GOOD")],
        spawned_at + 1,
    )
    # NEWER non-session_meta rollout whose payload DOES carry a matching cwd, so
    # only the ``type != session_meta`` skip (line 136) keeps it out. If that
    # skip regressed it would win on mtime and return "WRONG".
    _write_jsonl(
        _codex_path(home, spawned_at, "rollout-a.jsonl"),
        [
            {
                "type": "response_item",
                "payload": {
                    "cwd": str(cwd),
                    "timestamp": _timestamp_at(spawned_at),
                },
            },
            _codex_message("WRONG"),
        ],
        spawned_at + 5,
    )
    # NEWER session_meta whose payload is not a dict -> skipped at line 139
    # (without the guard, ``payload.get`` on a str would raise).
    _write_jsonl(
        _codex_path(home, spawned_at, "rollout-b.jsonl"),
        [{"type": "session_meta", "payload": "not-a-dict"}],
        spawned_at + 6,
    )

    result = read_codex_output(spawned_at, str(cwd))
    assert result is not None
    assert result.last_message == "GOOD"


# ---- _rollout_contains_token ---------------------------------------------


def test_rollout_contains_token_found(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("first line has wat-corr:abc\nsecond\n", encoding="utf-8")
    assert _rollout_contains_token(path, "wat-corr:abc") is True


def test_rollout_contains_token_beyond_max_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("l0\nl1\nl2\ntoken-here\n", encoding="utf-8")
    # Token is on line index 3 but scan stops after 2 lines.
    assert _rollout_contains_token(path, "token-here", max_lines=2) is False


def test_rollout_contains_token_oserror(tmp_path, monkeypatch):
    path = tmp_path / "missing.jsonl"

    def boom(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "open", boom)
    assert _rollout_contains_token(path, "x") is False


# ---- _matching_jsonl_files: per-file stat failure -------------------------


def test_matching_jsonl_files_skips_unstattable_file(tmp_path, monkeypatch):
    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / "rollout-x.jsonl").write_text("{}\n", encoding="utf-8")

    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "rollout-x.jsonl":
            raise OSError
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert ao._matching_jsonl_files(directory, 0.0, pattern="rollout-*.jsonl") == []


# ---- _codex_candidate_dirs: bad timestamp ---------------------------------


def test_codex_candidate_dirs_bad_timestamp_returns_empty():
    # A wildly out-of-range epoch makes datetime.fromtimestamp raise.
    assert _codex_candidate_dirs(1e20) == []


# ---- _first_json_object ---------------------------------------------------


def test_first_json_object_skips_blank_and_invalid(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('\n   \nnot json\n{"a": 1}\n', encoding="utf-8")
    assert _first_json_object(path) == {"a": 1}


def test_first_json_object_none_when_all_bad(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("\nnope\n{bad\n", encoding="utf-8")
    assert _first_json_object(path) is None


def test_first_json_object_oserror(tmp_path, monkeypatch):
    path = tmp_path / "f.jsonl"

    def boom(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "open", boom)
    assert _first_json_object(path) is None


# ---- _last_codex_message / _last_claude_message skip logic ----------------


def test_last_codex_message_skips_malformed_returns_older(tmp_path):
    path = tmp_path / "r.jsonl"
    older = _codex_message("older answer")
    # File order: valid older record first, malformed record last. Reverse read
    # sees the malformed line first (JSONDecodeError -> skip) then the valid one.
    path.write_text(json.dumps(older) + "\n{bad json\n", encoding="utf-8")
    assert _last_codex_message(path) == "older answer"


def test_last_codex_message_none_without_assistant(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"type": "response_item", "payload": {"role": "user"}}) + "\n",
        encoding="utf-8",
    )
    assert _last_codex_message(path) is None


def test_last_codex_message_skips_non_assistant_returns_older(tmp_path):
    path = tmp_path / "r.jsonl"
    older = _codex_message("older answer")
    # NEWER record is a non-assistant (role=user) that nonetheless carries
    # output_text — only the role check at line 261 keeps it out. Reverse read
    # sees it first; if that skip regressed it would return "newer user text".
    newer_user = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "output_text", "text": "newer user text"}],
        },
    }
    path.write_text(
        json.dumps(older) + "\n" + json.dumps(newer_user) + "\n", encoding="utf-8"
    )
    assert _last_codex_message(path) == "older answer"


def test_last_claude_message_skips_bad_records_returns_older(tmp_path):
    path = tmp_path / "c.jsonl"
    older = _claude_message([{"type": "text", "text": "older text"}])
    lines = [
        json.dumps(older),
        "{bad json",  # JSONDecodeError -> skip (line 276)
        json.dumps({"type": "assistant", "message": "not-a-dict"}),  # skip (279)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _last_claude_message(path) == "older text"


def test_last_claude_message_none_without_assistant(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n",
        encoding="utf-8",
    )
    assert _last_claude_message(path) is None


# ---- _claude_session_id / _claude_started_at ------------------------------


def test_claude_session_id_none_when_absent(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('\nnot json\n[]\n{"type": "user"}\n', encoding="utf-8")
    assert _claude_session_id(path) is None


def test_claude_session_id_oserror(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert _claude_session_id(path) is None


def test_claude_started_at_none_when_absent(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('\nnot json\n[]\n{"foo": 1}\n', encoding="utf-8")
    assert _claude_started_at(path) is None


def test_claude_started_at_oserror(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert _claude_started_at(path) is None


# ---- _parse_timestamp / _started_after ------------------------------------


def test_parse_timestamp_variants():
    assert _parse_timestamp(None) is None
    assert _parse_timestamp("") is None
    assert _parse_timestamp("not-a-timestamp") is None
    # Trailing-Z form is normalized and parsed.
    assert _parse_timestamp("2025-01-01T00:00:00Z") is not None


def test_started_after_none_is_true():
    assert _started_after(None, 100.0) is True
    assert _started_after(100.0, 100.0) is True
    assert _started_after(10.0, 100.0) is False


# ---- _content_text --------------------------------------------------------


def test_content_text_non_list_returns_none():
    # Use non-iterable inputs: without the ``isinstance(content, list)`` guard
    # the subsequent ``for item in content`` would raise TypeError, so a clean
    # ``None`` proves the guard (not merely the later empty-parts return).
    assert _content_text(None, "text") is None
    assert _content_text(123, "text") is None


def test_content_text_no_matching_parts_returns_none():
    assert _content_text([{"type": "image"}], "text") is None


# ---- _iter_lines_reverse --------------------------------------------------


def test_iter_lines_reverse_oserror_yields_nothing(tmp_path, monkeypatch):
    path = tmp_path / "r.jsonl"

    def boom(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "open", boom)
    assert list(_iter_lines_reverse(path)) == []


def test_iter_lines_reverse_multichunk(tmp_path):
    # Exceed the 64 KiB reverse-read chunk so the multi-iteration buffer join
    # path is exercised, and lines come back newest-first.
    path = tmp_path / "big.jsonl"
    lines = [f"line-{i}" for i in range(6000)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = list(_iter_lines_reverse(path))
    assert result[0] == "line-5999"
    assert result[-1] == "line-0"
    assert len(result) == 6000


# ---- _truncate_tail / _normalize_path / _resolve_path_text ----------------


def test_truncate_tail_non_positive_budget():
    assert _truncate_tail("abcdef", 0) == ""
    assert _truncate_tail("abcdef", -5) == ""


def test_normalize_path_empty_input():
    assert _normalize_path("") == ""


def test_resolve_path_text_empty_input():
    assert _resolve_path_text("") == ""


def test_resolve_path_text_falls_back_on_resolve_error(monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "resolve", boom)
    # Falls back to the expanduser'd path instead of raising.
    result = _resolve_path_text("~/somewhere")
    assert result == str(Path("~/somewhere").expanduser())
