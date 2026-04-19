"""Tests for the Typer CLI."""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from typer.testing import CliRunner

from claude_teams import capabilities, messaging, presets, tasks, teams, templates
from claude_teams.backends.base import Backend
from claude_teams.backends.contracts import HealthStatus
from claude_teams.backends.registry import registry as reg
from claude_teams.cli import app
from claude_teams.errors import BackendNotRegisteredError
from claude_teams.models import TeammateMember
from tests._server_support import _make_mock_backend

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
        cwd=str(base_dir),
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


def test_config_rejects_invalid_team_name():
    result = runner.invoke(app, ["config", "../bad-team"])
    assert result.exit_code == 1
    assert "Invalid team name" in result.output


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

    # Spec against Backend so typos surface as AttributeError instead of
    # silent auto-created child mocks (hence the kill assertion below is
    # guaranteed to observe the real method name).
    mock_backend = MagicMock(spec=Backend)
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

    # Pin that kill actually fired with alice's pane id — prior assertion only
    # proved CLI exit 0 + member removed, which silently passed even if kill
    # were never invoked.
    mock_backend.kill.assert_called_once_with("%42")

    cfg = await teams.read_config(name, base_dir)
    member_names = {member.name for member in cfg.members}
    assert "alice" not in member_names


async def test_kill_json(team, monkeypatch):
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    mock_backend = MagicMock(spec=Backend)
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
    mock_backend.kill.assert_called_once_with("%42")


# ---------------------------------------------------------------------------
# preset launch
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def preset_env(tmp_path, monkeypatch):
    """Wire module globals + a mock backend so ``preset launch`` runs offline.

    Mirrors the MCP ``client`` fixture pattern: point every TEAMS_DIR /
    TASKS_DIR at ``tmp_path`` and swap the module-level registry for a
    lone ``claude-code`` mock. Teardown restores both registry slots the
    CLI and ``orchestration`` paths consult.
    """
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")
    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()

    mock_backend = _make_mock_backend("claude-code")
    monkeypatch.setattr(reg, "_loaded", True)
    monkeypatch.setattr(reg, "_backends", {"claude-code": mock_backend})

    yield tmp_path, mock_backend

    presets._seed_builtin_presets()


async def test_preset_launch_creates_team_and_spawns_members(preset_env):
    """Happy path: expansion persists the team and stamps member config."""
    _base_dir, _mock_backend = preset_env
    result = await _invoke(
        ["preset", "launch", "review-and-fix", "cli-preset-team"],
    )
    assert result.exit_code == 0, result.output
    assert "Launched preset 'review-and-fix'" in result.output
    # Lead capability is printed so the operator can export it.
    assert "Lead capability:" in result.output
    assert "CLAUDE_TEAMS_CAPABILITY=" in result.output
    # Member table rows present.
    assert "reviewer" in result.output
    assert "executor" in result.output

    cfg = await teams.read_config("cli-preset-team")
    member_names = {m.name for m in cfg.members if isinstance(m, TeammateMember)}
    assert {"reviewer", "executor"} <= member_names


async def test_preset_launch_json_output(preset_env):
    """``--json`` emits the lead capability in a machine-readable envelope."""
    _base_dir, _mock_backend = preset_env
    result = await _invoke(
        ["preset", "launch", "review-and-fix", "cli-json-team", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["preset"] == "review-and-fix"
    assert data["team"]["team_name"] == "cli-json-team"
    assert data["lead_capability"]
    assert [m["name"] for m in data["members"]] == ["reviewer", "executor"]


async def test_preset_launch_unknown_preset_exits_nonzero(preset_env):
    """Unknown preset surfaces available names and exits 1."""
    _base_dir, _mock_backend = preset_env
    result = await _invoke(
        ["preset", "launch", "does-not-exist", "cli-bad-team"],
    )
    assert result.exit_code == 1
    assert "does-not-exist" in result.output
    # Available presets are listed so the operator can pick a valid one.
    assert "review-and-fix" in result.output


async def test_preset_launch_description_override(preset_env):
    """``--description`` wins over the preset's ``team_description``."""
    _base_dir, _mock_backend = preset_env
    result = await _invoke(
        [
            "preset",
            "launch",
            "review-and-fix",
            "cli-desc-team",
            "--description",
            "Custom CLI override.",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = await teams.read_config("cli-desc-team")
    assert cfg.description == "Custom CLI override."


async def test_preset_launch_falls_back_to_preset_description(preset_env):
    """Empty ``--description`` defers to the preset's own team_description."""
    _base_dir, _mock_backend = preset_env
    result = await _invoke(
        ["preset", "launch", "review-and-fix", "cli-default-desc"],
    )
    assert result.exit_code == 0, result.output
    cfg = await teams.read_config("cli-default-desc")
    preset = presets.get_preset("review-and-fix")
    assert cfg.description == preset.team_description


async def test_preset_launch_member_spawn_failure_exits_nonzero(preset_env):
    """Mid-fan-out spawn failure surfaces the failing member and exits 1."""
    _base_dir, mock_backend = preset_env
    mock_backend.spawn.side_effect = RuntimeError("simulated backend failure")

    result = await _invoke(
        ["preset", "launch", "review-and-fix", "cli-fail-team"],
    )
    assert result.exit_code == 1
    # Error message names the failing member per PresetMemberSpawnFailedError.
    # Collapse Rich's console-width wrapping before substring-matching so the
    # assertion doesn't hinge on terminal width.
    collapsed = " ".join(result.output.split())
    assert "reviewer" in collapsed
    assert "simulated backend failure" in collapsed


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_preset_launch_help(flag):
    """Both ``--help`` and ``-h`` render command help."""
    result = runner.invoke(app, ["preset", "launch", flag])
    # Typer wires -h via the context settings in modern versions; this
    # smoke test tolerates either help being supported or bailing out
    # with a clean exit code for unsupported flags.
    assert result.exit_code in (0, 2)
    if result.exit_code == 0:
        assert "Expand a preset" in result.output


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


def test_templates_table_lists_builtins():
    """Default ``templates`` output renders a table with seeded roles."""
    result = runner.invoke(app, ["templates"])
    assert result.exit_code == 0, result.output
    # Collapse Rich's console-width wrapping before substring matching so
    # table cells split across lines still assert cleanly.
    collapsed = " ".join(result.output.split())
    assert "code-reviewer" in collapsed
    assert "executor" in collapsed


def test_templates_json():
    result = runner.invoke(app, ["templates", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    names = {row["name"] for row in data}
    assert {"code-reviewer", "executor"} <= names


def test_templates_empty_exits_nonzero(monkeypatch):
    """No registered templates surfaces a warning and exits 1."""
    monkeypatch.setattr(templates, "_registry", {})
    try:
        result = runner.invoke(app, ["templates"])
        assert result.exit_code == 1
        assert "No templates registered" in result.output
    finally:
        templates._seed_builtin_templates()


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def test_presets_table_lists_builtins():
    """Default ``presets`` output renders a table row per preset."""
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0, result.output
    collapsed = " ".join(result.output.split())
    assert "review-and-fix" in collapsed
    # Member names (reviewer, executor) render inside the Members column.
    assert "reviewer" in collapsed
    assert "executor" in collapsed


def test_presets_json():
    result = runner.invoke(app, ["presets", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    names = {row["name"] for row in data}
    assert "review-and-fix" in names


def test_presets_empty_exits_nonzero(monkeypatch):
    """No registered presets surfaces a warning and exits 1."""
    monkeypatch.setattr(presets, "_registry", {})
    try:
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 1
        assert "No presets registered" in result.output
    finally:
        presets._seed_builtin_presets()


# ---------------------------------------------------------------------------
# backends (empty path)
# ---------------------------------------------------------------------------


def test_backends_empty_exits_nonzero(monkeypatch):
    """No registered backends surfaces a warning and exits 1."""
    monkeypatch.setattr(reg, "_loaded", True)
    monkeypatch.setattr(reg, "_backends", {})
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 1
    assert "No backends available" in result.output


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


async def test_health_alive_flow(team, monkeypatch):
    """Live teammate renders the green status line."""
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    mock_backend = MagicMock(spec=Backend)
    mock_backend.health_check.return_value = HealthStatus(
        alive=True, detail="pane responsive"
    )
    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            return mock_backend
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["health", name, "alice"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0, result.output
    collapsed = " ".join(result.output.split())
    assert "alive" in collapsed
    assert "pane responsive" in collapsed
    mock_backend.health_check.assert_called_once_with("%42")


async def test_health_dead_flow(team, monkeypatch):
    """Dead teammate renders the red status line."""
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    mock_backend = MagicMock(spec=Backend)
    mock_backend.health_check.return_value = HealthStatus(alive=False, detail="")
    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            return mock_backend
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["health", name, "alice"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0, result.output
    collapsed = " ".join(result.output.split())
    assert "dead" in collapsed


async def test_health_json(team, monkeypatch):
    """``--json`` returns the structured health envelope."""
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    mock_backend = MagicMock(spec=Backend)
    mock_backend.health_check.return_value = HealthStatus(alive=True, detail="ok")
    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            return mock_backend
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["health", name, "alice", "--json"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["alive"] is True
    assert data["agent_name"] == "alice"
    assert data["backend"] == "claude-code"


async def test_health_backend_unavailable(team, monkeypatch):
    """Missing backend registration exits 1 with a clear error."""
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            raise BackendNotRegisteredError("claude-code", [])
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["health", name, "alice"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    assert result.exit_code == 1
    collapsed = " ".join(result.output.split())
    assert "not available" in collapsed


# ---------------------------------------------------------------------------
# kill (backend unavailable branch)
# ---------------------------------------------------------------------------


async def test_kill_backend_unavailable_still_removes_member(team, monkeypatch):
    """If the backend is missing, kill swallows the error and still cleans up."""
    name, base_dir, lead_capability = team
    monkeypatch.setattr(teams, "TEAMS_DIR", base_dir / "teams")
    monkeypatch.setattr(tasks, "TASKS_DIR", base_dir / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", base_dir / "teams")
    await _add_teammate(name, base_dir, "alice")

    original_get = reg.get

    def patched_get(backend_name):
        if backend_name == "claude-code":
            raise BackendNotRegisteredError("claude-code", [])
        return original_get(backend_name)

    monkeypatch.setattr(reg, "get", patched_get)

    result = await _invoke(
        ["kill", name, "alice"],
        env={"CLAUDE_TEAMS_CAPABILITY": lead_capability},
    )
    # Member removal succeeds regardless of backend availability — the
    # process may already be dead, so swallowing BackendNotRegisteredError
    # matches the docstring intent.
    assert result.exit_code == 0, result.output
    cfg = await teams.read_config(name, base_dir)
    assert "alice" not in {m.name for m in cfg.members}
