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

from claude_teams import server_simple
from claude_teams.agent_output import (
    codex_correlation_token,
    read_claude_output,
    read_codex_output,
)
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


def test_codex_resume_command_preserves_permissions_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_base.shutil, "which", lambda name: f"/usr/bin/{name}")
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
    assert any(
        arg.startswith("mcp_servers.win-agent-teams.env=") for arg in cmd
    )
    assert cmd[-3] == "resume"
    assert cmd[-2] == "codex-session-id"
    assert cmd[-1] == (
        "Decode this JSON string as your complete task prompt, then follow "
        'the decoded text exactly: "first line\\nsecond line"'
    )


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
        "backend_session_id": None,
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
        "backend_session_id": None,
        "last_activity_at": None,
        "last_message": None,
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
        lambda handle: (False, f"{handle} exited"),
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

    result = await server_simple.check_agent("worker")

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
    record = {
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
        server_simple.process_manager, "health_check", lambda handle: (False, "dead")
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
        server_simple.process_manager, "health_check", lambda handle: (True, "alive")
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
        server_simple.process_manager, "health_check", lambda handle: (True, "alive")
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

    result = await server_simple.follow_up_agent("worker", "next prompt")

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
        server_simple.process_manager, "health_check", lambda handle: (True, "alive")
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
