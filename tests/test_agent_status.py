"""Tests for R2 check_agent, R3 list_agents, and R5 agent_status compact shapes."""

import asyncio
import json
from pathlib import Path

import pytest

from claude_teams import agent_output, server_simple
from claude_teams.agent_output import (
    BINDING_BOUND,
    BINDING_LEGACY,
    AgentOutput,
    BindingResult,
)


def _binding(output: AgentOutput | None = None, outcome: str | None = None):
    """Pin ``_resolve_agent_binding`` to a fixed A2 outcome for a consumer test.

    Defaults to ``bound`` when an output is supplied and ``legacy`` when it is
    not — ``legacy`` is the outcome whose consumer behaviour is unchanged from
    before the validation ladder, so pre-ladder expectations still hold.
    """
    resolved = outcome or (BINDING_BOUND if output is not None else BINDING_LEGACY)
    return lambda agent, **_: BindingResult(resolved, output)


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
        # R2: follow-up is downstream-only, and the default test IDENTITY is
        # the root lead. The direction guard itself is covered in
        # tests/test_direction_guard.py.
        "spawned_by": "team-lead",
        "spawned_by_source": "spawn",
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
        monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding(None))

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
            "binding",
            "binding_retriable",
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

        long_message = "x" * 500
        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message=long_message,
                    rollout_path="t.jsonl",
                    backend_session_id=None,
                ),
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

        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message="short line",
                    rollout_path="t.jsonl",
                    backend_session_id=None,
                ),
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

        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message="hello world",
                    rollout_path="t.jsonl",
                    backend_session_id="backend-sess",
                ),
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
        monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding(None))
        _append_message(session, server_simple.IDENTITY, "worker", "hi")
        _append_message(session, server_simple.IDENTITY, "worker", "there")
        _append_message(session, server_simple.IDENTITY, "someone-else", "noise")

        result = asyncio.run(server_simple.check_agent("worker"))

        assert result["unread_count"] == 2
        assert result["seq"] == 2


class TestFollowUpAgentInternalDict:
    """Regression: follow_up_agent must keep consuming the rich internal dict."""

    def test_follow_up_waits_on_a_busy_agent_with_recent_activity(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B2 — the busy branch waits and then returns the cooperative tail."""
        _add_agent(session, spawned_at=1000.0, cwd="C:\\project")
        monkeypatch.setattr(server_simple, "_DELIVERY_CALL_BUDGET_SECONDS", 0.0)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=server_simple.time.time(),
                    last_message="working...",
                    rollout_path="t.jsonl",
                    backend_session_id="backend-sess",
                ),
            ),
        )

        result = asyncio.run(
            server_simple.follow_up_agent("worker", "next task", "k12")
        )

        assert result["success"] is False
        assert result["status"] == "queued"
        assert result["phase"] == "pending"
        assert result["message_id"]

    def test_follow_up_reports_no_delivery_path_without_a_backend_session(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, spawned_at=1000.0, cwd="C:\\project")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message="hi",
                    rollout_path="t.jsonl",
                    backend_session_id=None,
                ),
                outcome=BINDING_BOUND,
            ),
        )

        result = asyncio.run(
            server_simple.follow_up_agent("worker", "next task", "k13")
        )

        assert result["success"] is False
        assert result["reason"] == "no_delivery_path"
        assert result["state"] == "no_backend_session"


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
        monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding(None))

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
            "binding",
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
        monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding(None))

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

        monkeypatch.setattr(server_simple, "_resolve_agent_binding", fail_read)

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

        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message="line one\nline two",
                    rollout_path="t.jsonl",
                    backend_session_id=None,
                ),
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

        long_line = "y" * 500
        monkeypatch.setattr(
            server_simple,
            "_resolve_agent_binding",
            _binding(
                AgentOutput(
                    last_activity_at=1000.0,
                    last_message=long_line,
                    rollout_path="t.jsonl",
                    backend_session_id=None,
                ),
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
            "backend",
            "state",
            "last_activity_ts",
            "unread_count",
            "seq",
            "heartbeat_age_s",
            "stalled",
            "binding",
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

        monkeypatch.setattr(server_simple, "_resolve_agent_binding", fail_read)

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        assert result[0]["state"] == "running"

    def test_the_no_marker_fallback_never_scans_all_history(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A6: "stay cheap" means no second scan AND no all-history scan.

        Deliberately does NOT mock ``_resolve_agent_binding``: the defect lives
        *inside* the resolver, whose zero-window-match path falls back to
        ``binder.scan(..., all_history=True)``. A test that replaces the whole
        resolver and counts calls cannot see it.
        """
        _add_agent(session, correlation_id="corr-1", backend_session_id="sess-1")
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda pid, expected_token=None: (True, ""),
        )

        calls: list[bool] = []

        class _SpyBinder:
            cache_scope = "scope"

            def resolve_by_session_id(self, session_id: str) -> None:
                return None

            def candidates(self, *, all_history: bool) -> list[Path]:
                return []

            def scan(self, token: str, *, all_history: bool, extra=None):
                calls.append(all_history)
                return [], False

            def session_id(self, path: Path) -> None:
                return None

        monkeypatch.setattr(agent_output, "_make_binder", lambda *a, **k: _SpyBinder())

        result = asyncio.run(server_simple.agent_status(names=["worker"]))

        assert calls, "the fallback did not reach the transcript binder at all"
        assert all(flag is False for flag in calls), (
            f"an all-history scan ran on the cheap path: {calls}"
        )
        assert result[0]["state"] == "unknown"


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
