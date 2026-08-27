"""Integration tests: hook materialisation at spawn time and marker-driven state."""

import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import hooks, server_simple
from claude_teams.agent_output import (
    BINDING_BOUND,
    BINDING_LEGACY,
    AgentOutput,
    BindingResult,
)
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.registry import canonical_backend_name


def _binding(output: AgentOutput | None = None, outcome: str | None = None):
    """Pin ``_resolve_agent_binding`` to a fixed A2 outcome for a consumer test.

    Defaults to ``bound`` when an output is supplied and ``legacy`` when it is
    not — ``legacy`` is the outcome whose consumer behaviour is unchanged from
    before the validation ladder, so pre-ladder expectations still hold.
    """
    resolved = outcome or (BINDING_BOUND if output is not None else BINDING_LEGACY)
    return lambda agent, **_: BindingResult(resolved, output)


class _FakeClaudeBackend:
    def __init__(self) -> None:
        self.last_request: SpawnRequest | None = None

    def default_model(self) -> str:
        return "sonnet"

    def resolve_model(self, model: str) -> str:
        return model

    def resolve_launch(
        self, model: str, reasoning_effort: str | None
    ) -> tuple[str, str | None]:
        return (model if model.strip() else self.default_model()), reasoning_effort

    def spawn(self, request: SpawnRequest) -> SimpleNamespace:
        self.last_request = request
        return SimpleNamespace(process_handle="789")


class _FakeRegistry:
    def __init__(self, backend: object, name: str = "claude-code") -> None:
        self._backend = backend
        self._name = name

    def resolve_name(self, name: str) -> str:
        return canonical_backend_name(name)

    def default_backend(self) -> str:
        return self._name

    def get(self, backend: str) -> object:
        assert backend == self._name
        return self._backend


@pytest.mark.asyncio
async def test_spawn_agent_writes_claude_settings_file_and_threads_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
    backend = _FakeClaudeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))

    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    assert request.extra is not None
    extra = request.extra
    assert "hooks_settings_path" in extra
    settings_path_str = extra["hooks_settings_path"]
    assert result["name"] == "worker"
    _assert_settings_file_has_hooks_block(settings_path_str)


def _assert_settings_file_has_hooks_block(settings_path_str: str) -> None:
    settings_path = Path(settings_path_str)
    assert settings_path.exists()
    config = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "hooks" in config


@pytest.mark.asyncio
async def test_spawn_agent_writes_codex_hook_overrides_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCodexBackend(_FakeClaudeBackend):
        pass

    backend = _FakeCodexBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend, "codex"))

    await server_simple.spawn_agent(
        "prompt", name="worker", backend="codex", cwd=str(tmp_path)
    )

    request = backend.last_request
    assert request is not None
    assert request.extra is not None
    extra = request.extra
    assert "hook_overrides" in extra
    overrides = json.loads(extra["hook_overrides"])
    assert overrides  # non-empty argv list
    assert overrides[0] == "-c"


def test_marker_roundtrip_drives_check_agent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "sess"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
    monkeypatch.setattr(server_simple, "_session_id", session_id)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    server_simple._save_agents(
        session_id,
        [
            {
                "name": "worker",
                "pid": 999,
                "backend": "claude-code",
                "session_id": session_id,
                "status": "running",
                "spawned_at": 1000.0,
                "cwd": str(tmp_path),
            }
        ],
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda pid, expected_token=None: (True, ""),
    )
    monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding(None))

    # Drive the real emit() entrypoint as the hook would.
    payload = json.dumps({"hook_event_name": "Stop", "session_id": session_id})
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    hooks.emit(session_dir, "worker")

    result = asyncio.run(server_simple.check_agent("worker"))
    assert result["state"] == "waiting"

    status_rows = asyncio.run(server_simple.agent_status(names=["worker"]))
    assert status_rows[0]["state"] == "waiting"


def test_kill_agent_deletes_marker_so_reused_name_starts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "sess"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
    monkeypatch.setattr(server_simple, "_session_id", session_id)
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    server_simple._save_agents(
        session_id,
        [
            {
                "name": "worker",
                "pid": 999,
                "backend": "claude-code",
                "session_id": session_id,
                "status": "running",
            }
        ],
    )
    marker_path = server_simple._state_marker_file(session_id, "worker")
    marker_path.write_text(
        json.dumps({"state": "running", "event": "Stop", "ts": 1.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(server_simple.process_manager, "kill_process", lambda pid: None)

    asyncio.run(server_simple.kill_agent("worker"))

    assert not marker_path.exists()
