"""Tests for the Typer CLI."""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest_asyncio
from typer.testing import CliRunner

from claude_teams import capabilities, messaging, tasks, teams
from claude_teams.backends.registry import registry as reg
from claude_teams.cli import app
from claude_teams.models import TeammateMember

runner = CliRunner()


async def _invoke(arguments: list[str], env: dict[str, str] | None = None):
    """Run the sync CLI in a worker thread from async tests."""
    return await asyncio.to_thread(runner.invoke, app, arguments, env=env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def team(tmp_path):
    """Create a team and return (team_name, base_dir)."""
    name = "test-team"
    await teams.create_team(
        name,
        session_id="sess-1",
        description="A test team",
        base_dir=tmp_path,
    )
    lead_capability = await capabilities.initialize_team_capabilities(
        name,
        base_dir=tmp_path,
    )
    return name, tmp_path, lead_capability


async def _add_teammate(
    team_name: str, base_dir, name: str = "alice"
) -> TeammateMember:
    """Add a dummy teammate to the team config."""

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="general-purpose",
        model="sonnet",
        prompt="Do work",
        color="blue",
        joined_at=int(time.time() * 1000),
        tmux_pane_id="%42",
        cwd="/tmp",
        backend_type="claude-code",
        process_handle="%42",
    )
    await teams.add_member(team_name, member, base_dir)
    await messaging.ensure_inbox(team_name, name, base_dir)
    return member


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the MCP server" in result.output


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def test_backends_runs():
    """backends command runs without error (may find 0 backends)."""
    result = runner.invoke(app, ["backends"])
    # exit code 0 if backends found, 1 if none — both are valid
    assert result.exit_code in (0, 1)


def test_backends_json():
    result = runner.invoke(app, ["backends", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_not_found():
    result = runner.invoke(app, ["config", "nonexistent-team"])
    assert result.exit_code == 1
    assert "not found" in result.output


async def test_config_requires_capability(team, monkeypatch):
    name, base_dir, _lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(["config", name])
    assert result.exit_code == 1
    assert "requires a valid team capability" in result.output


async def test_config_table(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(
        ["config", name], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 0
    assert name in result.output
    assert "team-lead" in result.output


async def test_config_json(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(
        ["config", name, "--json"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == name


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_not_found():
    result = runner.invoke(app, ["status", "nonexistent-team"])
    assert result.exit_code == 1


async def test_status_no_tasks(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    result = await _invoke(
        ["status", name], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 0
    assert "No tasks" in result.output


async def test_status_with_tasks(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    await tasks.create_task(name, "Fix bug", "Fix the login bug", base_dir=base_dir)
    result = await _invoke(
        ["status", name], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 0
    assert "Fix bug" in result.output


async def test_status_json(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    await tasks.create_task(name, "Fix bug", "Fix the login bug", base_dir=base_dir)
    result = await _invoke(
        ["status", name, "--json"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["team"] == name
    assert len(data["tasks"]) == 1


# ---------------------------------------------------------------------------
# inbox
# ---------------------------------------------------------------------------


async def test_inbox_empty(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(
        ["inbox", name, "team-lead"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    assert "empty" in result.output.lower()


async def test_inbox_with_messages(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "bob")
    await messaging.send_plain_message(
        name, "team-lead", "bob", "Hello Bob", summary="greeting", base_dir=base_dir
    )
    result = await _invoke(
        ["inbox", name, "bob"], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 0
    assert "greeting" in result.output


async def test_inbox_json(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "bob")
    await messaging.send_plain_message(
        name, "team-lead", "bob", "Hello", summary="hi", base_dir=base_dir
    )
    result = await _invoke(
        ["inbox", name, "bob", "--json"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1


async def test_inbox_order_newest(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "bob")
    for text in ("first", "second", "third"):
        await messaging.send_plain_message(
            name, "team-lead", "bob", text, summary=text, base_dir=base_dir
        )
    result = await _invoke(
        ["inbox", name, "bob", "--json", "--order", "newest"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert [msg["text"] for msg in data] == ["third", "second", "first"]


async def test_inbox_agent_capability_can_read_own_inbox(team, monkeypatch):
    name, base_dir, _lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "bob")
    agent_capability = await capabilities.issue_agent_capability(
        name, "bob", base_dir=base_dir
    )
    await messaging.send_plain_message(
        name, "team-lead", "bob", "Hello", summary="hi", base_dir=base_dir
    )
    result = await _invoke(
        ["inbox", name, "bob"], env={"CLAUDE_TEAMS_CAPABILITY": agent_capability}
    )
    assert result.exit_code == 0
    assert "hi" in result.output


async def test_inbox_agent_capability_cannot_read_other_inbox(team, monkeypatch):
    name, base_dir, _lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")
    await _add_teammate(name, base_dir, "bob")
    agent_capability = await capabilities.issue_agent_capability(
        name, "bob", base_dir=base_dir
    )
    result = await _invoke(
        ["inbox", name, "alice"],
        env={"CLAUDE_TEAMS_CAPABILITY": agent_capability},
    )
    assert result.exit_code == 1
    assert "cannot access inbox" in result.output


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_team_not_found():
    result = runner.invoke(app, ["health", "nonexistent", "alice"])
    assert result.exit_code == 1
    assert "not found" in result.output


async def test_health_agent_not_found(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(
        ["health", name, "ghost"], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 1
    assert "not found" in result.output


# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------


def test_kill_team_not_found():
    result = runner.invoke(app, ["kill", "nonexistent", "alice"])
    assert result.exit_code == 1
    assert "not found" in result.output


async def test_kill_agent_not_found(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    result = await _invoke(
        ["kill", name, "ghost"], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 1
    assert "not found" in result.output


async def test_kill_removes_member(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    # Mock the registry.get to avoid needing a real backend
    mock_backend = MagicMock()
    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            return mock_backend
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["kill", name, "alice"], env={"CLAUDE_TEAMS_CAPABILITY": lead_capability}
    )
    assert result.exit_code == 0
    assert "stopped" in result.output

    # Verify member removed
    cfg = await teams.read_config(name, base_dir)
    member_names = {member.name for member in cfg.members}
    assert "alice" not in member_names


async def test_kill_json(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    mock_backend = MagicMock()
    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            return mock_backend
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["kill", name, "alice", "--json"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["success"] is True
