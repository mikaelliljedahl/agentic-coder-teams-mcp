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


def test_resume_bumps_dispatch_epoch(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.setattr(
        server_simple.process_manager, "provides_tty", lambda *a, **k: True
    )
    backend = _InteractiveBackend()
    fields = server_simple._native_record_fields(
        {"dispatch_epoch": 4}, "claude-code", backend
    )
    assert fields["dispatch_epoch"] == 5
    assert (
        server_simple._native_record_fields(
            {"dispatch_epoch": "x"}, "claude-code", backend
        )["dispatch_epoch"]
        == 1
    )


def test_resume_flag_off_adds_nothing(monkeypatch):
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    assert (
        server_simple._native_record_fields(
            {"dispatch_epoch": 4}, "codex", _FakeBackend()
        )
        == {}
    )
