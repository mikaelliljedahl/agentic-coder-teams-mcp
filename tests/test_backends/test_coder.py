from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.coder import CoderBackend


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


class TestCoderProperties:
    def test_name_is_coder(self):
        backend = CoderBackend()
        assert backend.name == "coder"

    def test_binary_name_is_coder(self):
        backend = CoderBackend()
        assert backend.binary_name == "coder"


class TestCoderSupportedModels:
    def test_returns_expected_models(self):
        backend = CoderBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models
        assert "gpt-5.2-codex" in models
        assert "o3" in models

    def test_returns_list(self):
        backend = CoderBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestCoderDefaultModel:
    def test_returns_claude_sonnet(self):
        backend = CoderBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestCoderResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = CoderBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = CoderBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = CoderBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = CoderBackend()
        assert backend.resolve_model("gpt-5.2-codex") == "gpt-5.2-codex"

    def test_passes_through_unknown_model_name(self):
        backend = CoderBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestCoderBuildCommand:
    def test_produces_exec_command(self, _make_request):
        backend = CoderBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/coder"
        assert "exec" in cmd
        assert "-m" in cmd
        assert "--full-auto" in cmd

    def test_includes_prompt_as_last_arg(self, _make_request):
        backend = CoderBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_resolves_generic_model(self, _make_request):
        backend = CoderBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "claude-opus-4.6"


class TestCoderBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = CoderBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestCoderAvailability:
    def test_available_when_binary_found(self):
        backend = CoderBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = CoderBackend()
        assert backend.is_available() is False


class TestCoderReasoningEffort:
    def test_spec_advertises_c_flag_and_options(self):
        backend = CoderBackend()
        spec = backend.reasoning_effort_spec()
        assert spec is not None
        assert spec.flag == "-c"
        assert spec.value_template == "model_reasoning_effort={value}"
        assert spec.options == frozenset({"low", "medium", "high", "xhigh"})

    def test_build_command_appends_c_override_when_set(self, _make_request):
        backend = CoderBackend()
        request = _make_request(reasoning_effort="high")

        cmd = backend.build_command(request)

        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "model_reasoning_effort=high"

    def test_build_command_keeps_prompt_last_with_effort(self, _make_request):
        backend = CoderBackend()
        request = _make_request(reasoning_effort="medium", prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_build_command_omits_c_override_when_none(self, _make_request):
        backend = CoderBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "-c" not in cmd


class TestCoderAgentSelect:
    def test_spec_advertises_c_agents_template(self):
        backend = CoderBackend()
        spec = backend.agent_select_spec()
        assert spec is not None
        assert spec.flag == "-c"
        assert spec.value_template == 'agents.{name}.config_file="{path}"'

    def test_discover_reads_coder_config_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        coder_dir = tmp_path / ".coder"
        coder_dir.mkdir(parents=True)
        (coder_dir / "config.toml").write_text(
            '[agents.planner]\nconfig_file = "/abs/planner.md"\n'
        )
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = CoderBackend()
        profiles = backend.discover_agents(str(tmp_path))

        names = [p.name for p in profiles]
        assert "planner" in names
        planner = next(p for p in profiles if p.name == "planner")
        assert planner.path == "/abs/planner.md"

    def test_build_command_appends_c_override_when_discovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        coder_dir = tmp_path / ".coder"
        coder_dir.mkdir(parents=True)
        (coder_dir / "config.toml").write_text(
            '[agents.planner]\nconfig_file = "/abs/planner.md"\n'
        )
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = CoderBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="planner", prompt="go")

        cmd = backend.build_command(request)

        assert 'agents.planner.config_file="/abs/planner.md"' in cmd
        assert cmd[-1] == "go"

    def test_build_command_omits_agents_override_when_profile_none(self, _make_request):
        backend = CoderBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert not any("agents." in arg for arg in cmd)

    def test_build_command_omits_agents_override_when_profile_undiscovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = CoderBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="ghost")

        cmd = backend.build_command(request)

        assert not any("agents." in arg for arg in cmd)
