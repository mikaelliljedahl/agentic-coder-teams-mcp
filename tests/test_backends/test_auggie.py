from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.auggie import AuggieBackend
from claude_teams.backends.base import SpawnRequest


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="claude-sonnet-4.5",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestAuggieProperties:
    def test_name_is_auggie(self):
        backend = AuggieBackend()
        assert backend.name == "auggie"

    def test_binary_name_is_auggie(self):
        backend = AuggieBackend()
        assert backend.binary_name == "auggie"


class TestAuggieSupportedModels:
    def test_returns_expected_models(self):
        backend = AuggieBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models
        assert "claude-haiku-4.5" in models


class TestAuggieDefaultModel:
    def test_returns_claude_sonnet(self):
        backend = AuggieBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestAuggieResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = AuggieBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = AuggieBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = AuggieBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = AuggieBackend()
        assert backend.resolve_model("claude-sonnet-4.5") == "claude-sonnet-4.5"

    def test_passes_through_unknown_model_name(self):
        backend = AuggieBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestAuggieBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/auggie")
    def test_produces_print_command(self, _mock_which, _make_request):
        backend = AuggieBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/auggie"
        assert "-i" in cmd
        assert "--model" in cmd
        assert "--print" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/auggie")
    def test_includes_prompt_after_i_flag(self, _mock_which, _make_request):
        backend = AuggieBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-i")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/auggie")
    def test_includes_model_flag(self, _mock_which, _make_request):
        backend = AuggieBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4.6"


class TestAuggieBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = AuggieBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestAuggieAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/auggie")
    def test_available_when_binary_found(self, _mock_which):
        backend = AuggieBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = AuggieBackend()
        assert backend.is_available() is False
