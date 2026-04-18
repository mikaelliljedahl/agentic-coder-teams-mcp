from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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

    def test_is_not_interactive(self):
        backend = CodexBackend()
        assert backend.is_interactive is False


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
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_produces_exec_command(self, mock_which, _make_request):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "--model" in cmd
        assert "--full-auto" in cmd
        assert "-C" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_omits_full_auto_when_require_approval(self, mock_which, _make_request):
        backend = CodexBackend()
        request = _make_request(permission_mode="require_approval")

        cmd = backend.build_command(request)

        assert "--full-auto" not in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_includes_prompt_as_last_arg(self, mock_which, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_includes_cwd_flag(self, mock_which, _make_request, tmp_path: Path):
        backend = CodexBackend()
        project_dir = str(tmp_path / "my" / "project")
        request = _make_request(cwd=project_dir)

        cmd = backend.build_command(request)

        idx = cmd.index("-C")
        assert cmd[idx + 1] == project_dir

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_includes_output_file_flag_when_extra_path_provided(
        self, mock_which, _make_request, tmp_path: Path
    ):
        backend = CodexBackend()
        output_path = str(tmp_path / "codex-last-message.txt")
        request = _make_request(extra={"output_last_message_path": output_path})

        cmd = backend.build_command(request)

        assert "--output-last-message" in cmd
        idx = cmd.index("--output-last-message")
        assert cmd[idx + 1] == output_path

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_excludes_output_file_flag_when_no_extra(self, mock_which, _make_request):
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
