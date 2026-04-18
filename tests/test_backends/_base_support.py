"""Shared helpers for BaseBackend tests."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from claude_teams.backends.base import BaseBackend, SpawnRequest


class _StubBackend(BaseBackend):
    """Minimal concrete backend for testing BaseBackend methods."""

    _name = "stub"
    _binary_name = "stub-cli"

    def build_command(self, request: SpawnRequest) -> list[str]:
        binary = self.discover_binary()
        return [binary, "--prompt", request.prompt]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"STUB_MODE": "1"}

    def supported_models(self) -> list[str]:
        return ["default"]

    def default_model(self) -> str:
        return "default"

    def resolve_model(self, generic_name: str) -> str:
        return generic_name


class _InvalidEnvBackend(_StubBackend):
    """Backend that returns an invalid environment key for validation tests."""

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"INVALID-KEY": "val"}


@pytest.fixture
def _make_spawn_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    """Factory yielding ``SpawnRequest`` instances rooted at ``tmp_path``.

    Exposed as a fixture (not a module-level function) so each test body
    that takes it gets its own per-test-isolated ``cwd`` — no shared
    ``/tmp`` placeholder leaks across tests.
    """
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="default",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


def _make_backend_with_mock_controller() -> tuple[_StubBackend, MagicMock]:
    """Create a stub backend with a mocked tmux controller."""
    backend = _StubBackend()
    mock_ctrl = MagicMock()
    backend._controller = mock_ctrl
    return backend, mock_ctrl
