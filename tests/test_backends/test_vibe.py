from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.vibe import VibeBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="devstral-2",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestVibeProperties:
    def test_name_is_vibe(self):
        backend = VibeBackend()
        assert backend.name == "vibe"

    def test_binary_name_is_vibe(self):
        backend = VibeBackend()
        assert backend.binary_name == "vibe"


class TestVibeSupportedModels:
    def test_returns_expected_models(self):
        backend = VibeBackend()
        models = backend.supported_models()
        assert "devstral-2" in models
        assert "devstral-small" in models

    def test_returns_list(self):
        backend = VibeBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestVibeDefaultModel:
    def test_returns_devstral_2(self):
        backend = VibeBackend()
        assert backend.default_model() == "devstral-2"


class TestVibeResolveModel:
    def test_resolves_fast_to_devstral_small(self):
        backend = VibeBackend()
        assert backend.resolve_model("fast") == "devstral-small"

    def test_resolves_balanced_to_devstral_2(self):
        backend = VibeBackend()
        assert backend.resolve_model("balanced") == "devstral-2"

    def test_resolves_powerful_to_devstral_2(self):
        backend = VibeBackend()
        assert backend.resolve_model("powerful") == "devstral-2"

    def test_resolves_direct_model_name(self):
        backend = VibeBackend()
        assert backend.resolve_model("devstral-2") == "devstral-2"

    def test_passes_through_unknown_model_name(self):
        backend = VibeBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestVibeBuildCommand:
    def test_produces_prompt_command(self, _make_request):
        backend = VibeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/vibe"
        assert "-p" in cmd
        assert "--output" in cmd
        assert "text" in cmd

    def test_includes_prompt_after_p_flag(self, _make_request):
        backend = VibeBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-p")
        assert cmd[idx + 1] == "fix the bug"

    def test_does_not_include_model_flag(self, _make_request):
        backend = VibeBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        assert "--model" not in cmd
        assert "-m" not in cmd

    def test_output_format_is_text(self, _make_request):
        backend = VibeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        idx = cmd.index("--output")
        assert cmd[idx + 1] == "text"


class TestVibeBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = VibeBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestVibeAvailability:
    def test_available_when_binary_found(self):
        backend = VibeBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = VibeBackend()
        assert backend.is_available() is False
