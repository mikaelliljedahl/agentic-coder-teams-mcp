"""Tests for the agent_watch_paths tool (item 3)."""

import asyncio
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


def _add_agents(session_id: str, names: list[str]) -> None:
    agents = [
        {
            "name": name,
            "pid": idx + 1,
            "backend": "claude-code",
            "session_id": session_id,
            "status": "running",
            "spawned_at": 1000.0,
            "cwd": "C:\\project",
        }
        for idx, name in enumerate(names)
    ]
    server_simple._save_agents(session_id, agents)


class TestAgentWatchPaths:
    def test_names_none_returns_all_agents(self, session: str) -> None:
        _add_agents(session, ["worker-1", "worker-2"])

        result = asyncio.run(server_simple.agent_watch_paths())

        assert result["has_session"] is True
        assert Path(result["session_dir"]) == server_simple._session_dir(session)
        names = {row["name"] for row in result["agents"]}
        assert names == {"worker-1", "worker-2"}

    def test_row_shape_is_minimal(self, session: str) -> None:
        _add_agents(session, ["worker"])

        result = asyncio.run(server_simple.agent_watch_paths())

        assert len(result["agents"]) == 1
        row = result["agents"][0]
        assert set(row) == {"name", "state_marker_path"}
        expected = server_simple._state_marker_file(session, "worker")
        assert Path(row["state_marker_path"]) == expected
        assert Path(row["state_marker_path"]).is_absolute()

    def test_names_filter_subsets(self, session: str) -> None:
        _add_agents(session, ["worker-1", "worker-2"])

        result = asyncio.run(server_simple.agent_watch_paths(names=["worker-1"]))

        assert len(result["agents"]) == 1
        assert result["agents"][0]["name"] == "worker-1"

    def test_unknown_names_are_skipped(self, session: str) -> None:
        _add_agents(session, ["worker-1"])

        result = asyncio.run(
            server_simple.agent_watch_paths(names=["worker-1", "ghost"])
        )

        assert len(result["agents"]) == 1
        assert result["agents"][0]["name"] == "worker-1"

    def test_returns_empty_envelope_without_session_and_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_base = tmp_path / "sessions"
        monkeypatch.setattr(server_simple, "_SESSION_BASE", session_base)
        monkeypatch.setattr(server_simple, "_session_id", "")

        result = asyncio.run(server_simple.agent_watch_paths())

        assert result == {
            "has_session": False,
            "session_dir": "",
            "watch_argv": [],
            "watch_command_bash": "",
            "watch_command_powershell": "",
            "agents": [],
        }
        assert not session_base.exists()

    def test_live_session_with_zero_agents_is_distinct(self, session: str) -> None:
        result = asyncio.run(server_simple.agent_watch_paths())

        assert result["has_session"] is True
        assert result["agents"] == []
        assert result["session_dir"]
        assert result["watch_argv"]


def test_agent_watch_paths_docstring_is_canonical_watch_recipe() -> None:
    description = server_simple.agent_watch_paths.__doc__ or ""

    assert "state_marker_path" in description
    assert "watch_argv" in description
    assert "agent_status" in description
