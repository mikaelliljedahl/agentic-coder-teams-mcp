"""Tests for spawn_agent's watch-contract return payload (item 1)."""

import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import server_simple
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.registry import canonical_backend_name


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

    def resolve_name(self, name: str) -> str:
        return canonical_backend_name(name)

    def default_backend(self) -> str:
        return self._name

    def get(self, backend: str) -> object:
        assert backend == self._name
        return self._backend


@pytest.mark.asyncio
async def test_spawn_agent_returns_watch_metadata(
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

    assert isinstance(result["watch_argv"], list)
    assert result["watch_argv"][4] == result["session_dir"]
    assert shlex.split(result["watch_command_bash"]) == result["watch_argv"]
    powershell_session = "'" + result["session_dir"].replace("'", "''") + "'"
    assert powershell_session in result["watch_command_powershell"]
    assert "watch_command" not in result

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

    # A5: one prompt file per call, keyed on a per-call token, so two
    # concurrent calls for the same agent can never overwrite each other.
    prompts = sorted(
        (server_simple._session_dir(result["session_id"]) / "prompts").glob(
            "worker.*.prompt.txt"
        )
    )
    assert len(prompts) == 1, f"expected exactly one per-call sidecar, got {prompts}"
    # The user prompt is preserved losslessly at the head of the sidecar; the
    # correlation marker it is followed by is asserted in
    # tests/test_correlation_transport.py.
    assert prompts[0].read_text(encoding="utf-8").startswith(prompt)
    request = backend.last_request
    assert request is not None
    assert request.extra is not None
    assert request.prompt.startswith(prompt)
    assert request.extra["prompt_file_path"] == str(prompts[0])


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
    assert "watch_argv" in description


@pytest.mark.asyncio
async def test_spawn_agent_and_watch_paths_return_equal_watch_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    spawned = await server_simple.spawn_agent(
        "prompt", name="worker", backend="claude-code", cwd=str(tmp_path)
    )

    watched = await server_simple.agent_watch_paths()

    keys = {
        "session_dir",
        "watch_argv",
        "watch_command_bash",
        "watch_command_powershell",
    }
    assert {key: spawned[key] for key in keys} == {key: watched[key] for key in keys}


@pytest.mark.asyncio
async def test_spawn_agent_records_canonical_backend_for_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backend="claude"`` must spawn and persist as ``claude-code``.

    The stored name keys every later lookup (resume, follow-up, output
    reader), so an alias may never survive into the agent record.
    """
    backend = _FakeBackend()
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", "")
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))

    result = await server_simple.spawn_agent(
        "prompt", name="worker", backend="claude", cwd=str(tmp_path)
    )

    assert result["backend"] == "claude-code"
    agents = server_simple._load_agents(result["session_id"])
    assert [a["backend"] for a in agents] == ["claude-code"]


def test_spawn_agent_docstring_documents_backend_values() -> None:
    description = server_simple.spawn_agent.__doc__ or ""

    assert "backend:" in description
    for name in ("claude-code", "codex", "pi"):
        assert name in description
