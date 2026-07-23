"""R5 — kill_agent removes the record entirely and cleans up its artifacts.

kill is terminal: the agent disappears from list_agents, follow_up returns
agent_not_found, and inbox/cursor state is cleaned so a same-name respawn
starts clean. A naturally-dead (but un-killed) agent stays listed/resumable.
The OS kill is fail-closed: only signalled when we own the PID.
"""

import asyncio
import json
from pathlib import Path

import pytest

from claude_teams import server_simple


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
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


def _stub_owns(monkeypatch: pytest.MonkeyPatch, *, owned: bool) -> list[str]:
    calls: list[str] = []

    def fake_kill(handle: str, *a: object, **k: object) -> None:
        calls.append(handle)

    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: owned
    )
    monkeypatch.setattr(server_simple.process_manager, "kill_process", fake_kill)
    return calls


class TestKillRemovesRecord:
    def test_kill_removes_record_from_agents_json(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)

        result = asyncio.run(server_simple.kill_agent("worker"))

        assert result["success"] is True
        assert server_simple._load_agents(session) == []

    def test_killed_agent_not_in_list_agents(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))
        rows = asyncio.run(server_simple.list_agents())

        assert rows == []

    def test_follow_up_after_kill_names_the_unreachable_state(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, backend_session_id="abc")
        _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))
        result = asyncio.run(server_simple.follow_up_agent("worker", "continue", "k38"))

        # R7: a killed agent is unreachable by design, and the refusal names
        # that state rather than reading as a lookup miss.
        assert result["success"] is False
        assert result["reason"] == "no_delivery_path"
        assert result["state"] == "record_removed"

    def test_check_agent_after_kill_returns_empty_dead(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))
        status = asyncio.run(server_simple.check_agent("worker"))

        assert status["state"] == "dead"
        assert status["alive"] is False


class TestKillFailClosed:
    def test_skips_process_kill_on_token_mismatch_but_removes_record(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, create_token="stale-token")
        calls = _stub_owns(monkeypatch, owned=False)

        result = asyncio.run(server_simple.kill_agent("worker"))

        assert result["success"] is True
        assert calls == []  # foreign/reused PID never signalled
        assert server_simple._load_agents(session) == []

    def test_kills_process_when_owned(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session, pid=4242)
        calls = _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))

        assert calls == ["4242"]


class TestKillCleansArtifacts:
    def test_unlinks_state_marker(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        marker = server_simple._state_marker_file(session, "worker")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"state": "running"}), encoding="utf-8")
        _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))

        assert not marker.exists()

    def test_unlinks_prompt_sidecar(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        prompt_file = server_simple._prompt_file(session, "worker")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("prompt", encoding="utf-8")
        _stub_owns(monkeypatch, owned=True)

        asyncio.run(server_simple.kill_agent("worker"))

        assert not prompt_file.exists()

    def test_kill_then_respawn_same_name_does_not_inherit_inbox_or_cursor(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)
        # The killed agent's own inbox + read cursor.
        inbox = server_simple._inbox_file(session, "worker")
        inbox.write_text(
            json.dumps({"from": "lead", "text": "old", "ts": "t"}) + "\n",
            encoding="utf-8",
        )
        worker_cursor = server_simple._inbox_cursor_file(session, "worker")
        worker_cursor.write_text(json.dumps({"lead": 1}), encoding="utf-8")
        # The lead's cursor entry FOR this sender (stale high-water mark).
        lead_cursor = server_simple._inbox_cursor_file(session, server_simple.IDENTITY)
        lead_cursor.write_text(json.dumps({"worker": 5}), encoding="utf-8")

        asyncio.run(server_simple.kill_agent("worker"))

        assert not inbox.exists()
        assert not worker_cursor.exists()
        remaining = json.loads(lead_cursor.read_text(encoding="utf-8"))
        assert "worker" not in remaining

    def test_kill_wipes_killed_agents_messages_from_reader_inbox(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killing an agent removes its already-delivered messages from the
        lead's inbox and its cursor entry, while preserving other senders."""
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)
        reader_inbox = server_simple._inbox_file(session, server_simple.IDENTITY)
        reader_inbox.write_text(
            "\n".join(
                [
                    json.dumps({"from": "worker", "text": "w1", "ts": "t"}),
                    json.dumps({"from": "other", "text": "o1", "ts": "t"}),
                    json.dumps({"from": "worker", "text": "w2", "ts": "t"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        reader_cursor = server_simple._inbox_cursor_file(
            session, server_simple.IDENTITY
        )
        reader_cursor.write_text(
            json.dumps({"worker": 2, "other": 0}), encoding="utf-8"
        )

        asyncio.run(server_simple.kill_agent("worker"))

        by_sender = server_simple.read_inbox_by_sender(reader_inbox)
        assert "worker" not in by_sender
        assert [m["text"] for _, m in by_sender["other"]] == ["o1"]
        cursors = json.loads(reader_cursor.read_text(encoding="utf-8"))
        assert "worker" not in cursors
        assert cursors.get("other") == 0

    def test_kill_does_not_resurface_read_messages_as_unread(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported quirk: after kill, the killed agent's already-read
        messages must not reappear as unread on the next read."""
        _add_agent(session)
        _stub_owns(monkeypatch, owned=True)
        reader_inbox = server_simple._inbox_file(session, server_simple.IDENTITY)
        reader_inbox.write_text(
            "\n".join(
                [
                    json.dumps({"from": "worker", "text": "w1", "ts": "t"}),
                    json.dumps({"from": "worker", "text": "w2", "ts": "t"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        reader_cursor = server_simple._inbox_cursor_file(
            session, server_simple.IDENTITY
        )
        reader_cursor.write_text(json.dumps({"worker": 2}), encoding="utf-8")

        asyncio.run(server_simple.kill_agent("worker"))
        result = asyncio.run(server_simple.read_messages(""))

        assert result["unread_count"] == 0
        assert result["messages"] == []


class TestNaturalDeathKeepsRecord:
    def test_dead_agent_stays_listed_and_not_removed(
        self, session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _add_agent(session)
        monkeypatch.setattr(
            server_simple.process_manager,
            "health_check",
            lambda h, expected_token=None: (False, "dead"),
        )

        rows = asyncio.run(server_simple.list_agents())

        assert len(rows) == 1
        assert rows[0]["name"] == "worker"
        assert rows[0]["state"] == "dead"
        # Still on disk — a naturally-dead agent is resumable until killed.
        assert len(server_simple._load_agents(session)) == 1
