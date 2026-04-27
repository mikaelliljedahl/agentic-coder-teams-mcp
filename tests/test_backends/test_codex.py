from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.codex import CodexBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="gpt-5.3-codex",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestCodexProperties:
    def test_name_is_codex(self):
        backend = CodexBackend()
        assert backend.name == "codex"

    def test_binary_name_is_codex(self):
        backend = CodexBackend()
        assert backend.binary_name == "codex"

    def test_is_interactive(self):
        backend = CodexBackend()
        assert backend.is_interactive is True


class TestCodexSupportedModels:
    def test_returns_expected_models(self):
        backend = CodexBackend()
        models = backend.supported_models()
        assert "gpt-5.3-codex" in models
        assert "gpt-5.1-codex-max" in models
        assert "gpt-5.1-codex-mini" in models


class TestCodexDefaultModel:
    def test_returns_gpt_5_3_codex(self):
        backend = CodexBackend()
        assert backend.default_model() == "gpt-5.3-codex"


class TestCodexResolveModel:
    def test_resolves_fast_to_mini(self):
        backend = CodexBackend()
        assert backend.resolve_model("fast") == "gpt-5.1-codex-mini"

    def test_resolves_balanced_to_codex(self):
        backend = CodexBackend()
        assert backend.resolve_model("balanced") == "gpt-5.3-codex"

    def test_resolves_powerful_to_max(self):
        backend = CodexBackend()
        assert backend.resolve_model("powerful") == "gpt-5.1-codex-max"

    def test_resolves_direct_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("gpt-5.3-codex") == "gpt-5.3-codex"

    def test_passes_through_unknown_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("custom-model") == "custom-model"

    def test_passes_through_empty_string(self):
        backend = CodexBackend()
        assert backend.resolve_model("") == ""


class TestCodexBuildCommand:
    def test_produces_interactive_command(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" not in cmd
        assert "--model" in cmd
        assert "--full-auto" in cmd
        assert "-C" in cmd

    def test_omits_full_auto_when_require_approval(self, _make_request):
        backend = CodexBackend()
        request = _make_request(permission_mode="require_approval")

        cmd = backend.build_command(request)

        assert "--full-auto" not in cmd

    def test_includes_prompt_as_last_arg(self, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_includes_cwd_flag(self, _make_request, tmp_path: Path):
        backend = CodexBackend()
        project_dir = str(tmp_path / "my" / "project")
        request = _make_request(cwd=project_dir)

        cmd = backend.build_command(request)

        idx = cmd.index("-C")
        assert cmd[idx + 1] == project_dir

    def test_excludes_output_file_flag_in_interactive_mode(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--output-last-message" not in cmd


class TestCodexBuildEnv:
    def test_returns_empty_dict(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}


class TestCodexPermissionSupport:
    def test_supports_permission_bypass(self):
        backend = CodexBackend()
        assert backend.supports_permission_bypass() is True


class TestCodexReasoningEffort:
    def test_spec_advertises_c_flag_and_options(self):
        backend = CodexBackend()
        spec = backend.reasoning_effort_spec()
        assert spec is not None
        assert spec.flag == "-c"
        assert spec.value_template == "model_reasoning_effort={value}"
        assert spec.options == frozenset({"low", "medium", "high", "xhigh"})

    def test_build_command_appends_c_override_when_set(self, _make_request):
        backend = CodexBackend()
        request = _make_request(reasoning_effort="xhigh")

        cmd = backend.build_command(request)

        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "model_reasoning_effort=xhigh"

    def test_build_command_keeps_prompt_last_with_effort(self, _make_request):
        backend = CodexBackend()
        request = _make_request(reasoning_effort="low", prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    def test_build_command_omits_c_override_when_none(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "-c" not in cmd


class TestCodexAgentSelect:
    def test_spec_advertises_c_agents_template(self):
        backend = CodexBackend()
        spec = backend.agent_select_spec()
        assert spec is not None
        assert spec.flag == "-c"
        assert spec.value_template == 'agents.{name}.config_file="{path}"'

    def test_discover_reads_codex_config_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir(parents=True)
        (codex_dir / "config.toml").write_text(
            '[agents.reviewer]\nconfig_file = "/abs/reviewer.md"\n'
        )
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = CodexBackend()
        profiles = backend.discover_agents(str(tmp_path))

        names = [p.name for p in profiles]
        assert "reviewer" in names
        reviewer = next(p for p in profiles if p.name == "reviewer")
        assert reviewer.path == "/abs/reviewer.md"

    def test_build_command_appends_c_override_when_discovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir(parents=True)
        (codex_dir / "config.toml").write_text(
            '[agents.reviewer]\nconfig_file = "/abs/reviewer.md"\n'
        )
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = CodexBackend()
        request = _make_request(
            cwd=str(tmp_path), agent_profile="reviewer", prompt="go"
        )

        cmd = backend.build_command(request)

        assert 'agents.reviewer.config_file="/abs/reviewer.md"' in cmd
        assert cmd[-1] == "go"

    def test_build_command_omits_agents_override_when_profile_none(self, _make_request):
        backend = CodexBackend()
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

        backend = CodexBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="ghost")

        cmd = backend.build_command(request)

        assert not any("agents." in arg for arg in cmd)
