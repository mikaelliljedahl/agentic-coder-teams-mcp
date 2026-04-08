"""Shared helpers and fixtures for server tests."""

import json
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from mcp.types import TextContent

from claude_teams import messaging, tasks, teams
from claude_teams.backends import registry
from claude_teams.backends.base import HealthStatus, SpawnResult as BackendSpawnResult
from claude_teams.models import TeammateMember
from claude_teams.server import mcp


def _make_teammate(name: str, team_name: str, pane_id: str = "%1") -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="teammate",
        model="claude-sonnet-4-20250514",
        prompt="Do stuff",
        color="blue",
        plan_mode_required=False,
        joined_at=int(time.time() * 1000),
        tmux_pane_id=pane_id,
        cwd="/tmp",
    )


def _make_mock_backend(name: str = "claude-code") -> MagicMock:
    """Create a mock backend that satisfies the Backend protocol."""
    mock = MagicMock()
    mock.name = name
    mock.binary_name = "claude"
    mock.is_interactive = name == "claude-code"
    mock.is_available.return_value = True
    mock.discover_binary.return_value = "/usr/bin/echo"
    mock.supported_models.return_value = ["haiku", "sonnet", "opus"]
    mock.default_model.return_value = "sonnet"
    mock.resolve_model.side_effect = lambda model: {
        "fast": "haiku",
        "balanced": "sonnet",
        "powerful": "opus",
        "haiku": "haiku",
        "sonnet": "sonnet",
        "opus": "opus",
    }.get(model, model)
    mock.spawn.return_value = BackendSpawnResult(
        process_handle="%mock",
        backend_type=name,
    )
    mock.health_check.return_value = HealthStatus(alive=True, detail="mock check")
    mock.capture.return_value = ""
    mock.retain_pane_after_exit.return_value = None
    mock.kill.return_value = None
    mock.supports_permission_bypass.return_value = True
    return mock


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Client, None]:
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")

    mock_backend = _make_mock_backend("claude-code")
    registry._loaded = True
    registry._backends = {"claude-code": mock_backend}

    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()
    async with Client(mcp) as test_client:
        yield test_client

    registry._loaded = False
    registry._backends = {}


@pytest.fixture
async def team_client(client: Client) -> AsyncGenerator[Client, None]:
    """Client with a team created and a teammate spawned."""
    await client.call_tool("team_create", {"team_name": "test-team"})
    await client.call_tool(
        "spawn_teammate",
        {
            "team_name": "test-team",
            "name": "worker",
            "prompt": "help out",
        },
    )
    yield client


def _text(result) -> str:
    """Extract text from the first content item of a tool result."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _data(result):
    """Extract raw Python data from a successful CallToolResult."""
    if result.content:
        return json.loads(_text(result))
    return result.data


def _items(result):
    """Extract paginated result items from a successful CallToolResult."""
    data = _data(result)
    assert isinstance(data, dict)
    return data["items"]


def _extract_capability(text: str) -> str:
    match = re.search(r'capability="([^"]+)"', text)
    assert match is not None
    return match.group(1)
