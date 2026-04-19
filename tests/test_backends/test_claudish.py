from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.claudish import ClaudishBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="oai@gpt-5.2",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestClaudishProperties:
    def test_name_is_claudish(self):
        backend = ClaudishBackend()
        assert backend.name == "claudish"

    def test_binary_name_is_claudish(self):
        backend = ClaudishBackend()
        assert backend.binary_name == "claudish"


class TestClaudishSupportedModels:
    def test_returns_expected_models(self):
        backend = ClaudishBackend()
        models = backend.supported_models()
        assert "google@gemini-2.5-flash" in models
        assert "oai@gpt-5.2" in models
        assert "google@gemini-3-pro" in models

    def test_returns_list(self):
        backend = ClaudishBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestClaudishDefaultModel:
    def test_returns_oai_gpt52(self):
        backend = ClaudishBackend()
        assert backend.default_model() == "oai@gpt-5.2"


class TestClaudishResolveModel:
    def test_resolves_fast_to_gemini_flash(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("fast") == "google@gemini-2.5-flash"

    def test_resolves_balanced_to_gpt52(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("balanced") == "oai@gpt-5.2"

    def test_resolves_powerful_to_gemini_pro(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("powerful") == "google@gemini-3-pro"

    def test_resolves_direct_model_name(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("ollama@llama3.2") == "ollama@llama3.2"

    def test_passes_through_unknown_model_name(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("custom@model") == "custom@model"


class TestClaudishBuildCommand:
    def test_produces_single_shot_command(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/claudish"
        assert "--model" in cmd
        assert "-y" in cmd

    def test_includes_prompt_as_last_arg(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_includes_model_with_provider_syntax(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request(model="google@gemini-3-pro")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "google@gemini-3-pro"

    def test_resolves_generic_model(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "google@gemini-2.5-flash"


class TestClaudishBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestClaudishAvailability:
    def test_available_when_binary_found(self):
        backend = ClaudishBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = ClaudishBackend()
        assert backend.is_available() is False


class TestClaudishAgentSelect:
    def test_spec_advertises_agent_flag(self):
        backend = ClaudishBackend()
        spec = backend.agent_select_spec()
        assert spec is not None
        assert spec.flag == "--agent"
        assert spec.value_template == "{name}"

    def test_discover_finds_project_agents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "planner.md").write_text("planner")
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = ClaudishBackend()
        profiles = backend.discover_agents(str(tmp_path))

        names = [p.name for p in profiles]
        assert "planner" in names

    def test_build_command_appends_agent_flag_when_discovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "planner.md").write_text("planner")
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = ClaudishBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="planner", prompt="go")

        cmd = backend.build_command(request)

        assert "--agent" in cmd
        idx = cmd.index("--agent")
        assert cmd[idx + 1] == "planner"
        assert cmd[-1] == "go"

    def test_build_command_omits_agent_flag_when_profile_none(self, _make_request):
        backend = ClaudishBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--agent" not in cmd

    def test_build_command_omits_agent_flag_when_profile_undiscovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = ClaudishBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="ghost")

        cmd = backend.build_command(request)

        assert "--agent" not in cmd
