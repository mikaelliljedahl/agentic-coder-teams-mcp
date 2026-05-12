"""Tests for agent rollout fallback output readers."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claude_teams import server_simple
from claude_teams.agent_output import read_claude_output, read_codex_output


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


def _codex_meta(cwd: Path) -> dict:
    return {
        "type": "session_meta",
        "payload": {
            "id": "session-id",
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


def _claude_message(content: object) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": content,
        },
    }


def _claude_project_dir(home: Path, cwd: Path) -> Path:
    encoded = re.sub(r"[\\/:]", "-", str(cwd.resolve()))
    return home / ".claude" / "projects" / encoded


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
    assert output.rollout_path.endswith("rollout-new.jsonl")


def test_read_codex_output_returns_none_without_assistant(
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

    assert read_codex_output(spawned_at, str(cwd)) is None


def test_read_codex_output_truncates_at_utf8_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = tmp_path / "work"
    spawned_at = 1_762_969_000.0
    _write_jsonl(
        _codex_path(tmp_path, spawned_at, "rollout-long.jsonl"),
        [_codex_meta(cwd), _codex_message("aaaaa")],
        spawned_at + 10,
    )

    output = read_codex_output(spawned_at, str(cwd), max_bytes=3)

    assert output is not None
    assert output.last_message == "aaa"


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
    monkeypatch.setattr(server_simple, "_update_codex_mcp_env", lambda *args: None)
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
        }
    ]


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
        "alive": False,
        "pid": None,
        "backend": None,
        "last_activity_at": None,
        "last_message": None,
    }


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
        lambda handle: (False, f"{handle} exited"),
    )

    def fail_read(*args: object, **kwargs: object) -> None:
        pytest.fail("legacy agent records must not scan rollout logs")

    monkeypatch.setattr(server_simple, "read_codex_output", fail_read)

    result = await server_simple.check_agent("worker")

    assert result == {
        "name": "worker",
        "alive": False,
        "pid": 123,
        "backend": "codex",
        "last_activity_at": None,
        "last_message": None,
    }
