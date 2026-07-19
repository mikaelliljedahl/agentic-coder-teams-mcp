"""Tests for R2 check_agent, R3 list_agents, and R5 agent_status compact shapes."""

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
    agent: dict[str, object] = {
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


def _write_marker(session_id: str, name: str, state: str, ts: float = 1000.0) -> None:
    path = server_simple._state_marker_file(session_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"state": state, "event": "Stop", "ts": ts}), encoding="utf-8"
    )


def _append_message(session_id: str, reader: str, sender: str, text: str) -> None:
    inbox = server_simple._inbox_file(session_id, reader)
    inbox.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"from": sender, "text": text, "ts": "now"})
    with inbox.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class TestCheckAgentCompactShape:
    def test_default_returns_compact_shape(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _write_marker(session, "worker", "running")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)

        result = asyncio.run(server_simple.check_agent("worker"))

        assert set(result) == {
            "name",
            "state",
            "alive",
            "pid",
            "backend",
            "last_activity_at",
            "unread_count",
            "last_line",
            "seq",
            "truncated",
            "full_len",
            "heartbeat_age_s",
            "stalled",
        }
        assert "last_message" not in result
        assert "backend_session_id" not in result
        assert result["state"] == "running"

    def test_last_line_truncated_to_max_chars(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        long_message = "x" * 500
        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=1000.0,
                last_message=long_message,
                backend_session_id=None,
            ),
        )

        result = asyncio.run(server_simple.check_agent("worker", max_chars=50))

        assert len(result["last_line"]) == 50
        assert result["truncated"] is True
        assert result["full_len"] == 500

    def test_full_len_present_and_equal_when_not_truncated(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=1000.0,
                last_message="short line",
                backend_session_id=None,
            ),
        )

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["truncated"] is False
        assert result["full_len"] == len("short line")

    def test_full_true_restores_last_message_and_session_id(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=1000.0,
                last_message="hello world",
                backend_session_id="backend-sess",
            ),
        )

        result = asyncio.run(server_simple.check_agent("worker", full=True))

        assert result["last_message"] == "hello world"
        assert result["backend_session_id"] == "backend-sess"

    def test_empty_agent_check_state_dead(self, session: str) -> None:
        result = asyncio.run(server_simple.check_agent("ghost"))
        assert result["state"] == "dead"
        assert result["alive"] is False
        assert result["full_len"] == 0
        assert result["truncated"] is False

    def test_unread_count_counts_messages_from_named_agent(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)
        _append_message(session, server_simple.IDENTITY, "worker", "hi")
        _append_message(session, server_simple.IDENTITY, "worker", "there")
        _append_message(session, server_simple.IDENTITY, "someone-else", "noise")

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["unread_count"] == 2
        assert result["seq"] == 2


class TestFollowUpAgentInternalDict:
    """Regression: follow_up_agent must keep consuming the rich internal dict."""

    def test_follow_up_refuses_busy_agent_with_recent_activity(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, spawned_at=1000.0, cwd="C:\\project")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=server_simple.time.time(),
                last_message="working...",
                backend_session_id="backend-sess",
            ),
        )

        result = asyncio.run(server_simple.follow_up_agent("worker", "next task"))

        assert result["success"] is False
        assert result["reason"] == "agent_busy"
        assert "backend_session_id" in result
        assert "last_activity_at" in result

    def test_follow_up_reports_backend_session_missing(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, spawned_at=1000.0, cwd="C:\\project")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)

        result = asyncio.run(server_simple.follow_up_agent("worker", "next task"))

        assert result["success"] is False
        assert result["reason"] == "backend_session_missing"


class TestListAgentsCompactRows:
    def test_compact_rows_no_leaked_internal_fields(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(
            session,
            model="sonnet",
            permission_mode="bypass",
        )
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)

        result = asyncio.run(server_simple.list_agents())

        assert len(result) == 1
        row = result[0]
        assert set(row) == {
            "name",
            "state",
            "alive",
            "pid",
            "backend",
            "last_activity_at",
            "unread_count",
        }
        assert "model" not in row
        assert "permission_mode" not in row

    def test_returns_list_not_dict(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        monkeypatch.setattr(server_simple, "_read_agent_output", lambda agent: None)

        result = asyncio.run(server_simple.list_agents())

        assert isinstance(result, list)

    def test_marker_present_uses_marker_ts_no_transcript_scan(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _write_marker(session, "worker", "waiting", ts=1234.5)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        def fail_read(agent: dict) -> None:
            pytest.fail("list_agents must not scan the transcript when a marker exists")

        monkeypatch.setattr(server_simple, "_read_agent_output", fail_read)

        result = asyncio.run(server_simple.list_agents())

        assert result[0]["last_activity_at"] == 1234.5

    def test_full_true_includes_raw_record_and_last_line(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, model="sonnet")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=1000.0,
                last_message="line one\nline two",
                backend_session_id=None,
            ),
        )

        result = asyncio.run(server_simple.list_agents(full=True))

        row = result[0]
        assert row["model"] == "sonnet"
        assert row["last_line"] == "line two"
        assert row["truncated"] is False
        assert row["full_len"] == len("line two")

    def test_full_true_last_line_truncation_metadata(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, model="sonnet")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        from types import SimpleNamespace

        long_line = "y" * 500
        monkeypatch.setattr(
            server_simple,
            "_read_agent_output",
            lambda agent: SimpleNamespace(
                last_activity_at=1000.0,
                last_message=long_line,
                backend_session_id=None,
            ),
        )

        result = asyncio.run(server_simple.list_agents(full=True))

        row = result[0]
        assert len(row["last_line"]) == server_simple._DEFAULT_LAST_LINE_MAX_CHARS
        assert row["truncated"] is True
        assert row["full_len"] == 500


class TestAgentStatus:
    def test_minimal_fields_only(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _write_marker(session, "worker", "waiting", ts=1234.5)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        result = asyncio.run(server_simple.agent_status())

        assert len(result) == 1
        row = result[0]
        assert set(row) == {
            "name",
            "state",
            "last_activity_ts",
            "unread_count",
            "seq",
            "heartbeat_age_s",
            "stalled",
        }
        assert row["name"] == "worker"
        assert row["state"] == "waiting"
        assert row["last_activity_ts"] == 1234.5

    def test_names_filter_subsets(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agents = [
            {
                "name": "worker-1",
                "pid": 1,
                "backend": "claude-code",
                "session_id": session,
                "status": "running",
                "spawned_at": 1000.0,
                "cwd": "C:\\project",
            },
            {
                "name": "worker-2",
                "pid": 2,
                "backend": "claude-code",
                "session_id": session,
                "status": "running",
                "spawned_at": 1000.0,
                "cwd": "C:\\project",
            },
        ]
        server_simple._save_agents(session, agents)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        result = asyncio.run(server_simple.agent_status(names=["worker-1"]))

        assert len(result) == 1
        assert result[0]["name"] == "worker-1"

        all_result = asyncio.run(server_simple.agent_status(names=None))
        assert len(all_result) == 2

    def test_seq_and_unread_count_are_per_sender_count(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )
        _append_message(session, server_simple.IDENTITY, "worker", "hi")
        _append_message(session, server_simple.IDENTITY, "worker", "there")

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        row = result[0]
        assert row["seq"] == 2
        assert row["unread_count"] == 2

    def test_uses_marker_no_transcript_scan(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _write_marker(session, "worker", "running")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        def fail_read(agent: dict) -> None:
            pytest.fail(
                "agent_status must not scan the transcript when a marker exists"
            )

        monkeypatch.setattr(server_simple, "_read_agent_output", fail_read)

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        assert result[0]["state"] == "running"


class TestKillAgentDeletesMarker:
    def test_kill_agent_removes_state_marker(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _write_marker(session, "worker", "running")
        marker_path = server_simple._state_marker_file(session, "worker")
        assert marker_path.exists()
        monkeypatch.setattr(
            server_simple.process_manager, "kill_process", lambda pid: None
        )

        result = asyncio.run(server_simple.kill_agent("worker"))

        assert result["success"] is True
        assert not marker_path.exists()
