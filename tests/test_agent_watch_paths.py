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

        assert len(result) == 2
        names = {row["name"] for row in result}
        assert names == {"worker-1", "worker-2"}

    def test_row_shape_is_minimal(self, session: str) -> None:
        _add_agents(session, ["worker"])

        result = asyncio.run(server_simple.agent_watch_paths())

        assert len(result) == 1
        row = result[0]
        assert set(row) == {"name", "state_marker_path"}
        expected = server_simple._state_marker_file(session, "worker")
        assert Path(row["state_marker_path"]) == expected
        assert Path(row["state_marker_path"]).is_absolute()

    def test_names_filter_subsets(self, session: str) -> None:
        _add_agents(session, ["worker-1", "worker-2"])

        result = asyncio.run(server_simple.agent_watch_paths(names=["worker-1"]))

        assert len(result) == 1
        assert result[0]["name"] == "worker-1"

    def test_unknown_names_are_skipped(self, session: str) -> None:
        _add_agents(session, ["worker-1"])

        result = asyncio.run(
            server_simple.agent_watch_paths(names=["worker-1", "ghost"])
        )

        assert len(result) == 1
        assert result[0]["name"] == "worker-1"

    def test_returns_empty_list_without_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
        monkeypatch.setattr(server_simple, "_session_id", "")

        result = asyncio.run(server_simple.agent_watch_paths())

        assert result == []


def test_agent_watch_paths_docstring_is_canonical_watch_recipe() -> None:
    description = server_simple.agent_watch_paths.__doc__ or ""

    assert "state_marker_path" in description
    assert "watch" in description.lower()
    assert "agent_status" in description
