from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.rovodev import RovoDevBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="balanced",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestRovoDevProperties:
    def test_name_is_rovodev(self):
        backend = RovoDevBackend()
        assert backend.name == "rovodev"

    def test_binary_name_is_acli(self):
        backend = RovoDevBackend()
        assert backend.binary_name == "acli"


class TestRovoDevSupportedModels:
    def test_returns_expected_models(self):
        backend = RovoDevBackend()
        models = backend.supported_models()
        assert "gpt-5-2025-08-07" in models
        assert "claude-opus-4-20250918" in models
        assert "claude-sonnet-4-20250514" in models


class TestRovoDevDefaultModel:
    def test_returns_gpt5(self):
        backend = RovoDevBackend()
        assert backend.default_model() == "gpt-5-2025-08-07"


class TestRovoDevResolveModel:
    def test_resolves_fast(self):
        backend = RovoDevBackend()
        assert backend.resolve_model("fast") == "gpt-5-mini-2025-08-07"

    def test_resolves_balanced(self):
        backend = RovoDevBackend()
        assert backend.resolve_model("balanced") == "gpt-5-2025-08-07"

    def test_resolves_powerful(self):
        backend = RovoDevBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4-20250918"

    def test_passes_through_unknown_name(self):
        backend = RovoDevBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestRovoDevBuildCommand:
    def test_produces_rovodev_run_command(self, _make_request):
        backend = RovoDevBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/acli"
        assert cmd[1] == "rovodev"
        assert cmd[2] == "run"
        assert "--yolo" in cmd

    def test_includes_prompt_as_positional(self, _make_request):
        backend = RovoDevBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_does_not_include_model_flag(self, _make_request):
        backend = RovoDevBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--model" not in cmd
        assert "-m" not in cmd


class TestRovoDevBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = RovoDevBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestRovoDevAvailability:
    def test_available_when_binary_found(self):
        backend = RovoDevBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = RovoDevBackend()
        assert backend.is_available() is False
