"""Native record fields: interactive, codex_home, dispatch_epoch (plan v3 §2.1, §2.3.1).

They are written only with the master flag on, so flag-off records stay
byte-identical to ``main``.
"""

import json
from pathlib import Path

import pytest

from claude_teams import server_simple
from tests.test_spawn_agent_watch_contract import _FakeBackend, _FakeRegistry


class _InteractiveBackend(_FakeBackend):
    is_interactive = True


async def _spawn(tmp_path, monkeypatch, backend_name="claude-code"):
    backend = _InteractiveBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, backend_name))
    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend=backend_name, cwd=str(tmp_path)
    )
    agents = json.loads(
        (server_simple._session_dir(result["session_id"]) / "agents.json").read_text(
            encoding="utf-8"
        )
    )
    return agents[0] if isinstance(agents, list) else agents["agents"][0]


@pytest.mark.asyncio
async def test_flag_off_record_has_no_native_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    record = await _spawn(tmp_path, monkeypatch)
    for field in ("interactive", "codex_home", "dispatch_epoch"):
        assert field not in record


@pytest.mark.asyncio
async def test_flag_on_claude_record(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: True
    )
    record = await _spawn(tmp_path, monkeypatch)
    assert record["interactive"] is True
    assert record["dispatch_epoch"] == 1
    assert "codex_home" not in record


@pytest.mark.asyncio
async def test_flag_on_codex_record_pins_effective_home(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: False
    )
    record = await _spawn(tmp_path, monkeypatch, "codex")
    assert record["interactive"] is False
    assert record["codex_home"] == str(tmp_path / "codex-home")


def test_default_codex_home(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert server_simple._effective_codex_home() == str(Path.home() / ".codex")


def test_dispatch_epoch_is_monotonic_across_name_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    server_simple._session_dir("s1").mkdir(parents=True)
    assert server_simple._dispatch_extra("s1", "worker", {}) == {"dispatch_epoch": "1"}
    assert server_simple._dispatch_extra("s1", "worker", {"dispatch_epoch": 1}) == {
        "dispatch_epoch": "2"
    }
    # Record removed by kill; a same-name successor must not reuse 1 or 2.
    assert server_simple._dispatch_extra("s1", "worker", {}) == {"dispatch_epoch": "3"}
    # A record ahead of the file (e.g. restored) still wins.
    assert server_simple._dispatch_extra("s1", "worker", {"dispatch_epoch": 9}) == {
        "dispatch_epoch": "10"
    }
    assert server_simple._dispatch_extra("s1", "other", {}) == {"dispatch_epoch": "1"}


def test_dispatch_extra_flag_off_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    assert server_simple._dispatch_extra("s1", "worker", {}) == {}


def test_record_fields_take_the_minted_epoch(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: True
    )
    fields = server_simple._native_record_fields(
        "claude-code", _InteractiveBackend(), {"dispatch_epoch": "5"}
    )
    assert fields == {"interactive": True, "dispatch_epoch": 5}
    assert server_simple._native_record_fields("claude-code", _FakeBackend(), {}) == {}


def test_epoch_is_exported_to_the_child_environment(monkeypatch):
    from claude_teams.backends import process_base
    from claude_teams.backends.claude_code import ClaudeCodeBackend
    from claude_teams.backends.contracts import SpawnRequest

    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    observed = []
    monkeypatch.setattr(
        process_base.process_manager,
        "spawn_process",
        lambda request, argv, env_vars, *a, **k: observed.append(env_vars),
    )
    backend = ClaudeCodeBackend()
    request = SpawnRequest(
        agent_id="id",
        name="agent",
        team_name="team",
        prompt="task",
        model="",
        agent_type="",
        color="",
        cwd=".",
        lead_session_id="team-lead",
        extra={"dispatch_epoch": "7"},
    )
    backend._spawn_with_command(request, ["fake"], {})
    assert observed[0]["WIN_AGENT_TEAMS_DISPATCH_EPOCH"] == "7"


def test_recovery_metadata_is_recorded_with_the_flag_off(monkeypatch):
    """A minted epoch (native recovery metadata) carries current launch facts."""
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    monkeypatch.setenv("CODEX_HOME", "/new-home")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: False
    )
    fields = server_simple._native_record_fields(
        "codex", _FakeBackend(), {"dispatch_epoch": "6"}
    )
    assert fields == {
        "interactive": False,
        "dispatch_epoch": 6,
        "codex_home": "/new-home",
    }


def test_dispatch_extra_flag_off_mints_for_a_record_with_native_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    server_simple._session_dir("s1").mkdir(parents=True)
    assert server_simple._dispatch_extra("s1", "worker", {"dispatch_epoch": 5}) == {
        "dispatch_epoch": "6"
    }


def test_epoch_is_exported_with_the_flag_off(monkeypatch):
    from claude_teams.backends import process_base
    from claude_teams.backends.claude_code import ClaudeCodeBackend
    from claude_teams.backends.contracts import SpawnRequest

    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    observed = []
    monkeypatch.setattr(
        process_base.process_manager,
        "spawn_process",
        lambda request, argv, env_vars, *a, **k: observed.append(env_vars),
    )
    request = SpawnRequest(
        agent_id="id",
        name="agent",
        team_name="team",
        prompt="task",
        model="",
        agent_type="",
        color="",
        cwd=".",
        lead_session_id="team-lead",
        extra={"dispatch_epoch": "7"},
    )
    ClaudeCodeBackend()._spawn_with_command(request, ["fake"], {})
    assert observed[0]["WIN_AGENT_TEAMS_DISPATCH_EPOCH"] == "7"
    assert "CLAUDE_CODE_MESSAGING_SOCKET" not in observed[0]
