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
from claude_teams import delivery, server_simple
from claude_teams.agent_output import (
    BINDING_BOUND,
    AgentOutput,
    BindingResult,
    _claude_session_id,
    _claude_started_at,
    _codex_candidate_dirs,
    _content_text,
    _file_contains_token,
    _first_json_object,
    _iter_lines_reverse,
    _last_claude_message,
    _last_codex_message,
    _normalize_path,
    _parse_timestamp,
    _resolve_path_text,
    _started_after,
    _truncate_tail,
    correlation_marker_token,
    read_claude_output,
    read_codex_output,
)
from claude_teams.backends import codex as codex_module
from claude_teams.backends import process_base
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.delivery import DeliveryOutcome


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
    token_a = correlation_marker_token("corr-a")
    token_b = correlation_marker_token("corr-b")

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
        spawned_at, str(cwd), correlation_token=correlation_marker_token("corr-ghost")
    )

    assert output is not None
    assert output.backend_session_id == "session-x"
    assert output.last_message == "latest"


def test_codex_build_command_embeds_correlation_token(tmp_path: Path) -> None:
    backend = CodexBackend()
    request = _make_request(
        tmp_path,
        prompt="single line task",
        extra={"correlation_id": "corr-spawn"},
    )

    cmd = backend.build_command(request)

    assert correlation_marker_token("corr-spawn") in cmd[-1]


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
    # Ported from main's per-agent derived token to this branch's per-spawn
    # random correlation id: a derived token collides when a killed agent's
    # name is reused, so the id is minted once per spawn instead.
    token_a = correlation_marker_token("corr-a")
    token_b = correlation_marker_token("corr-b")

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
    #
    # This newest-mtime fallback survives ONLY in ``read_claude_output``, which
    # on this branch is reached exclusively through ``_ClaudeBinder.legacy_read``
    # for records that predate correlation. The A2 binding ladder deliberately
    # does NOT fall back this way: zero token matches is ``unverified`` (or
    # ``pending`` for an unread sidecar), never a guess at the newest file. That
    # fallback is the original defect — it pins a wrong backend_session_id once
    # and can never self-correct. See
    # ``test_binding_zero_matches_is_unverified_not_max_mtime``.
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
        correlation_token=correlation_marker_token("corr-ghost"),
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
    other_token = correlation_marker_token("corr-other")

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


# Main's `test_claude_build_command_embeds_correlation_token` and
# `test_claude_build_command_embeds_token_with_prompt_file` lived here. Both
# asserted that `ClaudeCodeBackend.build_command` injects the marker into argv.
# That injection is removed on this branch — the server materializes the prompt
# for both transports, so a backend-side injection would double-mark the argv
# path and could not reach the sidecar path at all.
#
# Replacements, by name:
#   - marker present exactly once, both transports:
#       test_correlation_transport.test_claude_spawn_prompt_carries_exactly_one_marker
#   - marker reaches the recorded user turn, both transports:
#       test_correlation_transport.test_marker_is_visible_in_claude_transcript_context
#   - resume assertion, deliberately inverted (see R8/A4):
#       test_correlation_transport.test_claude_resume_prompt_is_correlated
#   - argv stays marker-free under the sidecar transport:
#       test_backends/test_claude_code.py's
#       test_uses_prompt_file_instruction_when_provided


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
    # The correlation id is random per spawn, so it is asserted separately.
    correlation_id = agents[0].pop("correlation_id")
    assert isinstance(correlation_id, str)
    assert correlation_id
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
            "prompt_transport": "argv",
            "spawned_by": "team-lead",
            "spawned_by_source": "spawn",
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
async def test_send_message_unknown_recipient_is_refused_not_rerouted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_base = tmp_path / "sessions"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "IDENTITY", "child")
    monkeypatch.setattr(server_simple, "_AGENT_PARENT_NAME", "parent")
    (session_base / "session-id").mkdir(parents=True)

    # C3/R5: a typo'd recipient is refused. It used to be re-routed to the lead
    # with a warning, which made every typo a real-looking upstream message.
    result = await server_simple.send_message(to="leed", text="hello")

    assert result["success"] is False
    assert result["reason"] == "recipient_not_addressable"
    assert result["recipient_class"] == server_simple.RECIPIENT_UNKNOWN
    assert "leed" in result["detail"]
    # Nothing written anywhere: not to the lead, not to a dead inbox.
    assert not (session_base / "session-id" / "inbox-parent.jsonl").exists()
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

    monkeypatch.setattr(ao, "read_codex_output", fail_read)

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
        "binding": "legacy",
        "binding_retriable": False,
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
        "_resolve_agent_binding",
        lambda agent, **_: BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=10.0,
                last_message="done",
                rollout_path="t.jsonl",
                backend_session_id="backend-session-id",
            ),
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


def _default_follow_up_record(tmp_path: Path, **overrides: object) -> dict:
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
        "correlation_id": "corr-followup",
        # R2: follow-up is downstream-only, and the default test IDENTITY is
        # the root lead. The direction guard itself is covered in
        # tests/test_direction_guard.py.
        "spawned_by": "team-lead",
        "spawned_by_source": "spawn",
    }
    record.update(overrides)
    return record


def _write_agent_for_follow_up(tmp_path: Path, **overrides: object) -> None:
    server_simple._save_agents(
        "session-id", [_default_follow_up_record(tmp_path, **overrides)]
    )


def _pin_binding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    last_activity_at: float | None = None,
    last_message: str | None = None,
    backend_session_id: str | None = "backend-session-id",
    outcome: str = BINDING_BOUND,
) -> None:
    """Pin ``_resolve_agent_binding`` to a fixed outcome and transcript view."""
    output = AgentOutput(
        last_activity_at=last_activity_at or 0.0,
        last_message=last_message,
        rollout_path="t.jsonl",
        backend_session_id=backend_session_id,
    )
    monkeypatch.setattr(
        server_simple,
        "_resolve_agent_binding",
        lambda agent, **_: BindingResult(outcome, output),
    )


def _pin_delivery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "delivered",
    reason: str = "",
) -> None:
    """Pin A4 confirmation to a fixed outcome.

    The sibling of :func:`_pin_binding`, and used for the same reason: these
    tests are about busy/idle refusal, model selection, correlation
    preservation and so on, not about whether a nonce turned up in a
    transcript. The confirmation behaviour itself is covered end to end in
    ``tests/test_delivery_confirmation.py`` and ``tests/test_follow_up_delivery.py``
    against real transcripts and real poll/exit transitions — never a mock.
    """
    monkeypatch.setattr(
        server_simple,
        "confirm_delivery",
        lambda *a, **k: DeliveryOutcome(status, reason),
    )


def _setup_follow_up_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: object
) -> None:
    session_dir = tmp_path / "sessions" / "session-id"
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "session-id")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    # There is no real transcript on disk here, so pin the A2 ladder to a
    # bound binding; the outcome-specific consumer behaviour is covered by the
    # dedicated A6 tests below. Individual tests re-patch this when they need
    # a particular transcript-derived signal.
    _pin_binding(monkeypatch)
    _pin_delivery(monkeypatch)


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

    result = await server_simple.follow_up_agent("worker", "next prompt", "k1")

    assert result["success"] is True
    assert result["pid"] == 789
    assert result["replaced_existing"] is False
    request, backend_session_id = backend.resume_calls[0]
    assert backend_session_id == "backend-session-id"
    # A4: the resume prompt carries this attempt's delivery marker, which is
    # the only thing that later lets the receipt be attributed to this call.
    assert request.prompt.startswith("next prompt")
    assert request.prompt.count(delivery.DELIVERY_MARKER_PREFIX) == 1
    assert request.permission_mode == "bypass"
    agents = server_simple._load_agents("session-id")
    assert len(agents) == 1
    assert agents[0]["pid"] == 789
    assert agents[0]["spawned_at"] == 1_000.0


@pytest.mark.asyncio
async def test_follow_up_agent_waits_for_a_busy_live_agent_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 — ``agent_busy`` is no longer a refusal; it is a bounded wait.

    Here the budget is zero, so the wait ends immediately in R1's cooperative
    tail: ``queued(phase=pending)`` with an obligation on the sender. What it
    must NOT be is the old dead-end refusal, and it must not be ``failed``
    either — nothing was sent, so nothing definitely failed.
    """
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(server_simple, "_DELIVERY_CALL_BUDGET_SECONDS", 0.0)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k2")

    assert result["success"] is False
    assert result["status"] == "queued"
    assert result["phase"] == "pending"
    assert "deliver_pending" in result["sender_obligation"]
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
    _pin_binding(
        monkeypatch,
        last_activity_at=900.0,
        last_message="done",
    )

    result = await server_simple.follow_up_agent(
        "worker", "next prompt", "k3", replace_if_idle=False
    )

    assert result["success"] is False
    assert result["reason"] == "agent_idle_but_alive"
    assert backend.resume_calls == []


@pytest.mark.asyncio
async def test_follow_up_agent_waiting_marker_overrides_quiet_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``waiting`` marker beats an absent ``last_message``.

    The marker is the authoritative idle signal, but the ``last_message is
    None`` check used to run ahead of it, so an agent parked at a Stop hook
    before producing any assistant text was reported ``agent_busy`` — the one
    case where we know for certain it is idle.
    """
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path)
    (tmp_path / "sessions" / "session-id" / "state-worker.json").write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 950.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda handle, token: True
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "graceful_shutdown",
        lambda handle, timeout_s=5.0: True,
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)
    _pin_binding(
        monkeypatch,
        last_activity_at=990.0,
        last_message=None,
    )

    result = await server_simple.follow_up_agent("worker", "next prompt", "k4")

    assert result["success"] is True, (
        f"waiting marker must override a quiet transcript, got {result}"
    )
    assert backend.resume_calls, "expected a resume, not a busy refusal"


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
    _pin_binding(
        monkeypatch,
        last_activity_at=900.0,
        last_message="done",
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
        "worker", "next prompt", "k5", replace_if_idle=True
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

    result = await server_simple.follow_up_agent("worker", "next prompt", "k6")

    assert result["success"] is False
    assert result["reason"] == "backend_not_supported"


@pytest.mark.asyncio
async def test_follow_up_agent_preserves_correlation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    _write_agent_for_follow_up(tmp_path, correlation_id="corr-abc")
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k7")

    assert result["success"] is True
    request, _ = backend.resume_calls[0]
    assert (request.extra or {})["correlation_id"] == "corr-abc"
    # spawn -> resume -> read: the id must still be on the record afterwards.
    agents = server_simple._load_agents("session-id")
    assert agents[0]["correlation_id"] == "corr-abc"
    assert ao.classify_correlation(agents[0]) == (ao.CORRELATION_VALID, "corr-abc")


@pytest.mark.asyncio
async def test_follow_up_agent_refuses_legacy_record_and_invents_no_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: a record predating correlation is unresumable, not best-effort.

    Its stored session id may be exactly the wrong pinned id this feature
    exists to fix, so resuming on it could confirm a nonce in someone else's
    conversation and report ``delivered``. The refusal names the only recovery.
    """
    backend = _FakeResumeBackend()
    _setup_follow_up_session(tmp_path, monkeypatch, backend)
    record = _default_follow_up_record(tmp_path)
    del record["correlation_id"]
    server_simple._save_agents("session-id", [record])
    _pin_binding(monkeypatch, outcome=ao.BINDING_LEGACY)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k8")

    assert result["success"] is False
    assert result["reason"] == "binding_legacy"
    assert result["retriable"] is False
    # The refusal must name the ONLY recovery, not just say "no": an agent
    # predating correlation can never be made resumable.
    detail = str(result["detail"]).lower()
    assert "kill" in detail
    assert "respawn" in detail
    assert backend.resume_calls == []
    # No id is minted to paper over the gap: a fresh one would not appear
    # anywhere in the conversation that already exists.
    assert "correlation_id" not in server_simple._load_agents("session-id")[0]


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
                "correlation_id": "corr-restart",
                "spawned_by": "team-lead",
                "spawned_by_source": "spawn",
            }
        ],
    )
    _pin_binding(monkeypatch)
    _pin_delivery(monkeypatch)
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k9")

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


# ---- _file_contains_token ---------------------------------------------


def test_file_contains_token_found(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("first line has wat-corr:abc\nsecond\n", encoding="utf-8")
    assert _file_contains_token(path, "wat-corr:abc") is True


def test_file_contains_token_beyond_max_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("l0\nl1\nl2\ntoken-here\n", encoding="utf-8")
    # Token is on line index 3 but scan stops after 2 lines.
    assert _file_contains_token(path, "token-here", max_lines=2) is False


def test_file_contains_token_oserror(tmp_path, monkeypatch):
    path = tmp_path / "missing.jsonl"

    def boom(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "open", boom)
    assert _file_contains_token(path, "x") is False


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


# ---- A2: explicit validation ladder ---------------------------------------

_LADDER_SPAWNED_AT = 1_762_969_000.0
_ABSENT = object()


@pytest.fixture(autouse=True)
def _clear_binding_cache():
    ao.clear_binding_cache()
    yield
    ao.clear_binding_cache()


def _claude_user(text: str, *, session_id: str = "session-id") -> dict:
    return {
        "type": "user",
        "timestamp": _timestamp_at(_LADDER_SPAWNED_AT),
        "sessionId": session_id,
        "message": {"role": "user", "content": text},
    }


def _write_claude_transcript(
    project_dir: Path,
    filename: str,
    *,
    session_id: str,
    mtime: float,
    correlation_id: str | None = None,
    text: str = "hello",
    include_session_id: bool = True,
    drop_timestamp: bool = False,
) -> Path:
    prompt = "do the thing"
    if correlation_id is not None:
        prompt = f"{prompt} {correlation_marker_token(correlation_id)}"
    user_row = _claude_user(prompt, session_id=session_id)
    assistant_row = _claude_message(
        [{"type": "text", "text": text}],
        session_id=session_id,
    )
    if not include_session_id:
        user_row.pop("sessionId")
        assistant_row.pop("sessionId")
    if drop_timestamp:
        user_row.pop("timestamp", None)
        assistant_row.pop("timestamp", None)
    path = project_dir / filename
    _write_jsonl(path, [user_row, assistant_row], mtime)
    return path


def _claude_record(cwd: Path, *, correlation_id: object = "corr-own", **overrides):
    record: dict[str, object] = {
        "name": "worker",
        "pid": 123,
        "backend": "claude-code",
        "session_id": "team-session",
        "spawned_at": _LADDER_SPAWNED_AT,
        "cwd": str(cwd),
    }
    if correlation_id is not _ABSENT:
        record["correlation_id"] = correlation_id
    record.update(overrides)
    return record


def _bind(record, *, alive: bool = True, now: float | None = None, **kwargs):
    return ao.resolve_agent_binding(
        record,
        child_alive=lambda: alive,
        now=_LADDER_SPAWNED_AT + 5 if now is None else now,
        **kwargs,
    )


def test_binding_prefers_own_transcript_over_newer_foreign_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
        text="mine",
    )
    _write_claude_transcript(
        project_dir,
        "foreign.jsonl",
        session_id="foreign",
        mtime=_LADDER_SPAWNED_AT + 900,
        correlation_id=None,
        text="foreign",
    )

    result = _bind(_claude_record(cwd))

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "mine"
    assert result.output.last_message == "mine"


def test_binding_repins_wrong_stored_id_to_token_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "wrong.jsonl",
        session_id="wrong-session",
        mtime=_LADDER_SPAWNED_AT + 900,
        correlation_id=None,
        text="not mine",
    )
    _write_claude_transcript(
        project_dir,
        "right.jsonl",
        session_id="right-session",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
        text="mine",
    )

    result = _bind(_claude_record(cwd, backend_session_id="wrong-session"))

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "right-session"


def test_binding_stable_across_repeated_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
    )
    record = _claude_record(cwd, backend_session_id="mine")

    first = _bind(record)
    second = _bind(record)
    third = _bind(record)

    assert first.outcome == second.outcome == third.outcome == ao.BINDING_BOUND
    assert first.output is not None
    assert third.output is not None
    assert first.output.rollout_path == third.output.rollout_path
    assert third.output.backend_session_id == "mine"


def test_binding_two_token_matches_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "a.jsonl",
        session_id="stored",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
    )
    _write_claude_transcript(
        project_dir,
        "b.jsonl",
        session_id="other",
        mtime=_LADDER_SPAWNED_AT + 20,
        correlation_id="corr-own",
    )

    result = _bind(_claude_record(cwd, backend_session_id="stored"))

    assert result.outcome == ao.BINDING_AMBIGUOUS
    assert result.output is None


def test_binding_zero_matches_is_unverified_not_max_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "foreign.jsonl",
        session_id="foreign",
        mtime=_LADDER_SPAWNED_AT + 900,
        correlation_id=None,
    )

    result = _bind(_claude_record(cwd))

    assert result.outcome == ao.BINDING_UNVERIFIED
    assert result.output is None


def test_binding_gate_zero_reports_pending_for_live_sidecar_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    _claude_project_dir(tmp_path, cwd).mkdir(parents=True)
    record = _claude_record(cwd, prompt_transport="sidecar")

    result = _bind(record, alive=True, sidecar_pending_window_s=60.0)

    assert result.outcome == ao.BINDING_PENDING
    assert result.retriable is True
    assert result.output is None
    assert ao.binding_cache_size() == 0
    assert "backend_session_id" not in record


def test_binding_gate_zero_exits_when_receipt_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    project_dir.mkdir(parents=True)
    record = _claude_record(cwd, prompt_transport="sidecar")

    assert _bind(record, sidecar_pending_window_s=60.0).outcome == ao.BINDING_PENDING

    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 3,
        correlation_id="corr-own",
    )

    result = _bind(record, sidecar_pending_window_s=60.0)

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "mine"


def test_binding_gate_zero_exits_when_child_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    _claude_project_dir(tmp_path, cwd).mkdir(parents=True)
    record = _claude_record(cwd, prompt_transport="sidecar")

    result = _bind(record, alive=False, sidecar_pending_window_s=60.0)

    assert result.outcome == ao.BINDING_UNVERIFIED


def test_binding_gate_zero_exits_when_window_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    _claude_project_dir(tmp_path, cwd).mkdir(parents=True)
    record = _claude_record(cwd, prompt_transport="sidecar")

    result = _bind(
        record,
        alive=True,
        now=_LADDER_SPAWNED_AT + 1_000,
        sidecar_pending_window_s=60.0,
    )

    assert result.outcome == ao.BINDING_UNVERIFIED


def test_binding_gate_zero_not_entered_for_argv_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    _claude_project_dir(tmp_path, cwd).mkdir(parents=True)

    result = _bind(_claude_record(cwd, prompt_transport="argv"))

    assert result.outcome == ao.BINDING_UNVERIFIED


@pytest.mark.parametrize("value", ["", "   ", 17, None, ["x"]])
def test_binding_malformed_correlation_is_unverified_not_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    _claude_project_dir(tmp_path, cwd).mkdir(parents=True)

    result = _bind(_claude_record(cwd, correlation_id=value))

    assert result.outcome == ao.BINDING_UNVERIFIED


def test_binding_single_match_without_session_id_is_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
        include_session_id=False,
    )

    result = _bind(_claude_record(cwd))

    assert result.outcome == ao.BINDING_UNVERIFIED


def test_binding_scan_oserror_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
    )

    def boom(self, *args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "open", boom)

    result = _bind(_claude_record(cwd))

    assert result.outcome == ao.BINDING_INDETERMINATE
    assert result.retriable is True
    assert result.output is None


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_binding_candidate_enumeration_oserror_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Gate 2 is about ENUMERATION as much as reading.

    A directory listing that fails (permissions, a disconnected network share,
    a racing rotation) tells us nothing about whether a matching transcript
    exists. Collapsing it to an empty candidate list turns "we could not look"
    into "there is nothing there", which the count gate then reports as
    ``unverified`` — a terminal, non-retriable outcome derived from a scan that
    never happened.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
    )

    def boom(self, *args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr(Path, "iterdir", boom)

    result = _bind(_claude_record(cwd, backend=backend))

    assert result.outcome == ao.BINDING_INDETERMINATE
    assert result.retriable is True

    # Tier 1 enumerates too: the name-as-session-id convention is not a
    # contract, so a miss falls back to a directory walk.
    tier_one = _bind(
        _claude_record(cwd, backend=backend, backend_session_id="stored-sess")
    )
    assert tier_one.outcome == ao.BINDING_INDETERMINATE


def test_binding_legacy_record_reports_legacy_and_still_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "legacy.jsonl",
        session_id="legacy-session",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id=None,
        text="legacy output",
    )

    result = _bind(_claude_record(cwd, correlation_id=_ABSENT))

    assert result.outcome == ao.BINDING_LEGACY
    assert result.retriable is False
    assert result.output is not None
    assert result.output.last_message == "legacy output"
    assert ao.binding_cache_size() == 0


def test_binding_no_parseable_timestamp_not_accepted_on_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "untimed.jsonl",
        session_id="untimed",
        mtime=_LADDER_SPAWNED_AT + 900,
        correlation_id=None,
        drop_timestamp=True,
    )

    result = _bind(_claude_record(cwd))

    assert result.outcome == ao.BINDING_UNVERIFIED


def test_binding_tier_one_ignores_mtime_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "old.jsonl",
        session_id="stored",
        mtime=_LADDER_SPAWNED_AT - 10_000,
        correlation_id="corr-own",
        text="still mine",
    )

    result = _bind(_claude_record(cwd, backend_session_id="stored"))

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.last_message == "still mine"


def test_binding_reused_agent_name_kept_apart_by_correlation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    _write_claude_transcript(
        project_dir,
        "first.jsonl",
        session_id="first",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-first",
        text="first run",
    )
    _write_claude_transcript(
        project_dir,
        "second.jsonl",
        session_id="second",
        mtime=_LADDER_SPAWNED_AT + 20,
        correlation_id="corr-second",
        text="second run",
    )

    first = _bind(_claude_record(cwd, correlation_id="corr-first"))
    second = _bind(_claude_record(cwd, correlation_id="corr-second"))

    assert first.output is not None
    assert first.output.last_message == "first run"
    assert second.output is not None
    assert second.output.last_message == "second run"


# ---- A2: validated-binding cache ------------------------------------------


def _cached_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    path = _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 10,
        correlation_id="corr-own",
    )
    record = _claude_record(cwd, backend_session_id="mine")
    assert _bind(record).outcome == ao.BINDING_BOUND
    assert ao.binding_cache_size() == 1
    return cwd, project_dir, path, record


def test_binding_cache_hit_is_reused_without_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, _path, record = _cached_setup(tmp_path, monkeypatch)
    calls: list[Path] = []
    original = ao._scan_token

    def counting(path, token, **kwargs):
        calls.append(path)
        return original(path, token, **kwargs)

    monkeypatch.setattr(ao, "_scan_token", counting)

    result = _bind(record)

    assert result.outcome == ao.BINDING_BOUND
    assert calls == []


def test_binding_cache_append_does_not_invalidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, path, record = _cached_setup(tmp_path, monkeypatch)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_claude_message("appended", session_id="mine")) + "\n")

    result = _bind(record)

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.last_message == "appended"
    assert ao.binding_cache_size() == 1


def test_binding_cache_invalidated_by_path_disappearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, path, record = _cached_setup(tmp_path, monkeypatch)
    path.unlink()

    assert _bind(record).outcome == ao.BINDING_UNVERIFIED


def test_binding_cache_invalidated_by_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, path, record = _cached_setup(tmp_path, monkeypatch)
    path.write_text("", encoding="utf-8")

    assert _bind(record).outcome == ao.BINDING_UNVERIFIED


def test_binding_cache_invalidated_by_file_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, project_dir, path, record = _cached_setup(tmp_path, monkeypatch)
    path.unlink()
    _write_claude_transcript(
        project_dir,
        "mine.jsonl",
        session_id="mine",
        mtime=_LADDER_SPAWNED_AT + 30,
        correlation_id="corr-other",
        text="replaced",
    )

    assert _bind(record).outcome == ao.BINDING_UNVERIFIED


def test_binding_cache_invalidated_by_parsed_session_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, path, record = _cached_setup(tmp_path, monkeypatch)
    entry_key = next(iter(ao._BINDING_CACHE))
    ao._BINDING_CACHE[entry_key] = replace(
        ao._BINDING_CACHE[entry_key], session_id="something-else"
    )

    result = _bind(record)

    # Revalidation rejects the stale entry, then the scan re-binds correctly.
    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "mine"
    assert str(path) == result.output.rollout_path


def test_binding_cache_invalidated_by_grammar_version_bump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cwd, _project_dir, _path, record = _cached_setup(tmp_path, monkeypatch)
    entry_key = next(iter(ao._BINDING_CACHE))
    ao._BINDING_CACHE[entry_key] = replace(
        ao._BINDING_CACHE[entry_key],
        grammar_version=ao.BINDING_GRAMMAR_VERSION - 1,
    )
    calls: list[Path] = []
    original = ao._scan_token

    def counting(path, token, **kwargs):
        calls.append(path)
        return original(path, token, **kwargs)

    monkeypatch.setattr(ao, "_scan_token", counting)

    result = _bind(record)

    assert result.outcome == ao.BINDING_BOUND
    assert calls, "an entry from an older grammar version must not be trusted"


def test_binding_cache_key_separates_correlation_session_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd, _project_dir, _path, _record = _cached_setup(tmp_path, monkeypatch)

    # A different correlation id must not reuse the cached binding.
    assert (
        _bind(_claude_record(cwd, correlation_id="corr-other")).outcome
        == ao.BINDING_UNVERIFIED
    )
    # A different stored session id re-binds by token rather than by cache.
    assert (
        _bind(_claude_record(cwd, backend_session_id="elsewhere")).outcome
        == ao.BINDING_BOUND
    )
    # A different cwd has no transcripts at all.
    assert _bind(_claude_record(tmp_path / "other")).outcome == ao.BINDING_UNVERIFIED


@pytest.mark.parametrize(
    "outcome_setup",
    ["pending", "unverified", "ambiguous", "indeterminate"],
)
def test_binding_non_success_outcomes_are_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome_setup: str
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    project_dir = _claude_project_dir(tmp_path, cwd)
    project_dir.mkdir(parents=True)
    record = _claude_record(cwd)
    if outcome_setup == "pending":
        record["prompt_transport"] = "sidecar"
    elif outcome_setup == "ambiguous":
        for name, sid in (("a.jsonl", "a"), ("b.jsonl", "b")):
            _write_claude_transcript(
                project_dir,
                name,
                session_id=sid,
                mtime=_LADDER_SPAWNED_AT + 10,
                correlation_id="corr-own",
            )
    elif outcome_setup == "indeterminate":
        _write_claude_transcript(
            project_dir,
            "a.jsonl",
            session_id="a",
            mtime=_LADDER_SPAWNED_AT + 10,
            correlation_id="corr-own",
        )

        def boom(self, *args, **kwargs):
            raise OSError

        monkeypatch.setattr(Path, "open", boom)

    result = _bind(record, sidecar_pending_window_s=60.0)

    assert result.outcome != ao.BINDING_BOUND
    assert ao.binding_cache_size() == 0


def test_binding_retriable_flags_match_the_five_outcomes() -> None:
    retriable = {ao.BINDING_PENDING, ao.BINDING_INDETERMINATE}
    terminal = {ao.BINDING_UNVERIFIED, ao.BINDING_AMBIGUOUS, ao.BINDING_LEGACY}
    for outcome in retriable:
        assert ao.BindingResult(outcome).retriable is True
    for outcome in terminal:
        assert ao.BindingResult(outcome).retriable is False
    assert ao.BindingResult(ao.BINDING_BOUND).retriable is False
    assert retriable == ao.RETRIABLE_BINDING_OUTCOMES


def test_binding_codex_rollout_bound_by_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_jsonl(
        _codex_path(tmp_path, _LADDER_SPAWNED_AT, "rollout-a.jsonl"),
        [
            _codex_meta(cwd, session_id="codex-mine"),
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": correlation_marker_token("corr-own"),
                        }
                    ],
                },
            },
            _codex_message("codex answer"),
        ],
        _LADDER_SPAWNED_AT + 10,
    )
    _write_jsonl(
        _codex_path(tmp_path, _LADDER_SPAWNED_AT, "rollout-b.jsonl"),
        [
            _codex_meta(cwd, session_id="codex-foreign"),
            _codex_message("foreign answer"),
        ],
        _LADDER_SPAWNED_AT + 900,
    )

    result = _bind(_claude_record(cwd, backend="codex"))

    assert result.outcome == ao.BINDING_BOUND
    assert result.output is not None
    assert result.output.backend_session_id == "codex-mine"
    assert result.output.last_message == "codex answer"


# ---- A6: consumer decisions for the five outcomes -------------------------


_NON_SUCCESS_OUTCOMES = [
    ao.BINDING_PENDING,
    ao.BINDING_UNVERIFIED,
    ao.BINDING_AMBIGUOUS,
    ao.BINDING_LEGACY,
    ao.BINDING_INDETERMINATE,
]


def _force_binding(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    """Pin ``_resolve_agent_binding`` to one outcome for consumer tests."""
    output = (
        ao.AgentOutput(
            last_activity_at=900.0,
            last_message="legacy text",
            rollout_path="legacy.jsonl",
            backend_session_id="discovered-session-id",
        )
        if outcome == ao.BINDING_LEGACY
        else None
    )
    monkeypatch.setattr(
        server_simple,
        "_resolve_agent_binding",
        lambda record, **_: ao.BindingResult(outcome, output),
    )


def _setup_consumer_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> None:
    _setup_follow_up_session(tmp_path, monkeypatch, _FakeResumeBackend())
    record: dict[str, object] = {"correlation_id": "corr-own"}
    record.update(overrides)
    _write_agent_for_follow_up(tmp_path, **record)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", _NON_SUCCESS_OUTCOMES)
async def test_check_agent_reports_binding_and_never_persists_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    _setup_consumer_session(tmp_path, monkeypatch, backend_session_id="")
    _force_binding(monkeypatch, outcome)

    result = await server_simple.check_agent("worker", full=True)

    assert result["binding"] == outcome
    assert result["binding_retriable"] is (
        outcome in (ao.BINDING_PENDING, ao.BINDING_INDETERMINATE)
    )
    stored = server_simple._load_agents("session-id")[0]
    assert not stored.get("backend_session_id")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", _NON_SUCCESS_OUTCOMES)
async def test_follow_up_refuses_every_non_success_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    _setup_consumer_session(tmp_path, monkeypatch)
    _force_binding(monkeypatch, outcome)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k10")

    assert result["success"] is False
    assert result["reason"] == f"binding_{outcome}"
    assert result["retriable"] is (
        outcome in (ao.BINDING_PENDING, ao.BINDING_INDETERMINATE)
    )


@pytest.mark.asyncio
async def test_follow_up_legacy_refusal_names_kill_and_respawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_consumer_session(tmp_path, monkeypatch)
    _force_binding(monkeypatch, ao.BINDING_LEGACY)

    result = await server_simple.follow_up_agent("worker", "next prompt", "k11")

    assert result["reason"] == "binding_legacy"
    assert result["retriable"] is False
    detail = str(result["detail"]).lower()
    assert "kill" in detail
    assert "respawn" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", _NON_SUCCESS_OUTCOMES)
async def test_list_agents_reports_binding_in_both_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    _setup_consumer_session(tmp_path, monkeypatch)
    _force_binding(monkeypatch, outcome)

    compact = await server_simple.list_agents()
    full = await server_simple.list_agents(full=True)

    assert compact[0]["binding"] == outcome
    assert "backend_session_id" not in compact[0]
    assert full[0]["binding"] == outcome
    # A stored id is only presented as verified when the binding is bound.
    assert full[0]["backend_session_id_verified"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", _NON_SUCCESS_OUTCOMES)
async def test_agent_status_fallback_stays_cheap_on_non_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    _setup_consumer_session(tmp_path, monkeypatch)
    calls: list[object] = []

    def counting(record, *, bounded_only: bool = False):
        # Call COUNT alone is not "cheap": the resolver's own zero-match path
        # can escalate to an all-history scan inside a single call. The mode is
        # asserted here, and the resolver's honouring of it is proved without a
        # mock in tests/test_agent_status.py.
        assert bounded_only is True, "the cheap fallback must ask for bounded work"
        calls.append(record)
        output = (
            ao.AgentOutput(900.0, "legacy text", "x.jsonl", "sid")
            if outcome == ao.BINDING_LEGACY
            else None
        )
        return ao.BindingResult(outcome, output)

    monkeypatch.setattr(server_simple, "_resolve_agent_binding", counting)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "alive"),
    )

    rows = await server_simple.agent_status()

    assert len(calls) == 1, "agent_status must not add a second scan"
    if outcome == ao.BINDING_LEGACY:
        assert rows[0]["last_activity_ts"] == 900.0
    else:
        assert rows[0]["state"] == "unknown"
        assert rows[0]["last_activity_ts"] is None
