"""Tests for spawn_agent's watch-contract return payload (item 1)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple
from claude_teams.backends.contracts import SpawnRequest


class _FakeBackend:
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

    def default_backend(self) -> str:
        return self._name

    def get(self, backend: str) -> object:
        assert backend == self._name
        return self._backend


@pytest.mark.asyncio
async def test_spawn_agent_returns_state_marker_path_and_session_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))

    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    assert result["name"] == "worker"
    assert result["pid"] == 789
    assert result["backend"] == "claude-code"
    session_id = result["session_id"]

    expected_marker = server_simple._state_marker_file(session_id, "worker")
    assert Path(result["state_marker_path"]) == expected_marker
    assert Path(result["state_marker_path"]).is_absolute()

    expected_session_dir = server_simple._session_dir(session_id)
    assert Path(result["session_dir"]) == expected_session_dir
    assert Path(result["session_dir"]).is_absolute()

    assert result["expected_outputs"] == []


@pytest.mark.asyncio
async def test_spawn_agent_echoes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))

    outputs = ["C:/out/report.md", "C:/out/data.json"]
    result = await server_simple.spawn_agent(
        "prompt",
        name="worker",
        backend="claude-code",
        cwd=str(tmp_path),
        expected_outputs=outputs,
    )

    assert result["expected_outputs"] == outputs


@pytest.mark.asyncio
async def test_spawn_agent_writes_claude_prompt_file_for_sensitive_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    prompt = "first 'line'\nsecond \"line\""

    result = await server_simple.spawn_agent(
        prompt, name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    prompt_path = (
        server_simple._session_dir(result["session_id"])
        / "prompts"
        / "worker.prompt.txt"
    )
    assert prompt_path.read_text(encoding="utf-8") == prompt
    request = backend.last_request
    assert request is not None
    assert request.extra is not None
    assert request.prompt == prompt
    assert request.extra["prompt_file_path"] == str(prompt_path)


@pytest.mark.asyncio
async def test_spawn_agent_expected_outputs_defaults_to_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))

    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    assert result["expected_outputs"] == []


def test_spawn_agent_docstring_documents_watch_contract() -> None:
    description = server_simple.spawn_agent.__doc__ or ""

    assert "state_marker_path" in description
    assert "session_dir" in description
    assert "expected_outputs" in description
    assert '"state": "running" | "waiting"' in description or (
        '"running"' in description and '"waiting"' in description
    )
    assert '"event"' in description
    assert '"ts"' in description
    assert "survives" in description
    assert "restart" in description
    assert "watch" in description.lower()
