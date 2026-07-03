"""Tests for the disk-derived stall/heartbeat signal (item 4)."""

import asyncio
import json
from pathlib import Path

import pytest

from claude_teams import server_simple


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the server at a temp session directory and return its id."""
    session_id = "test-session"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
    monkeypatch.setattr(server_simple, "_session_id", session_id)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})
    (tmp_path / session_id).mkdir(parents=True, exist_ok=True)
    return session_id


def _add_agent(session_id: str, **overrides: object) -> dict:
    agent = {
        "name": "worker",
        "pid": 4242,
        "backend": "claude-code",
        "session_id": session_id,
        "status": "running",
        "spawned_at": 1000.0,
        "cwd": "C:\\project",
    }
    agent.update(overrides)
    server_simple._save_agents(session_id, [agent])
    return agent


def _write_marker(session_id: str, name: str, state: str, ts: float) -> None:
    path = server_simple._state_marker_file(session_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"state": state, "event": "Stop", "ts": ts}), encoding="utf-8"
    )


class TestStallSecondsConstant:
    def test_default_is_300(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STALL_SECONDS", raising=False)
        assert server_simple._stall_seconds() == 300.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STALL_SECONDS", "42")
        assert server_simple._stall_seconds() == 42.0

    def test_invalid_env_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STALL_SECONDS", "not-a-number")
        assert server_simple._stall_seconds() == 300.0


class TestAgentStatusStallFields:
    def test_recent_activity_running_not_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 5.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["heartbeat_age_s"] == pytest.approx(5.0)
        assert row["stalled"] is False

    def test_old_activity_running_is_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 400.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["heartbeat_age_s"] == pytest.approx(400.0)
        assert row["stalled"] is True

    def test_waiting_state_never_stalled_even_if_old(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "waiting", ts=now - 400.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["stalled"] is False

    def test_dead_agent_never_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 400.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (False, "")
        )

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["stalled"] is False

    def test_stall_seconds_env_override_respected(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 50.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )
        monkeypatch.setenv("WIN_AGENT_TEAMS_STALL_SECONDS", "10")

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        assert result[0]["stalled"] is True

    def test_no_activity_signal_heartbeat_age_none_not_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["heartbeat_age_s"] is None
        assert row["stalled"] is False


class TestCheckAgentStallFields:
    def test_recent_activity_not_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 1.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["heartbeat_age_s"] == pytest.approx(1.0)
        assert result["stalled"] is False

    def test_old_activity_running_is_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "running", ts=now - 500.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["heartbeat_age_s"] == pytest.approx(500.0)
        assert result["stalled"] is True

    def test_waiting_never_stalled(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        now = 100_000.0
        _write_marker(session, "worker", "waiting", ts=now - 500.0)
        monkeypatch.setattr(server_simple.time, "time", lambda: now)
        monkeypatch.setattr(
            server_simple.process_manager, "health_check", lambda pid: (True, "")
        )

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["stalled"] is False


def test_agent_status_docstring_documents_stall_fields() -> None:
    description = server_simple.agent_status.__doc__ or ""
    assert "heartbeat_age_s" in description
    assert "stalled" in description
    assert "STALL_SECONDS" in description or "stall" in description.lower()


def test_check_agent_docstring_documents_stall_fields() -> None:
    description = server_simple.check_agent.__doc__ or ""
    assert "heartbeat_age_s" in description
    assert "stalled" in description
