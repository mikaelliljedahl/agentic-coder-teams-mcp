from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.agent_output import codex_correlation_token
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.codex import CodexBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="gpt-5.5",
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
        assert models == [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
        ]


class TestCodexDefaultModel:
    def test_returns_gpt_5_5(self):
        backend = CodexBackend()
        assert backend.default_model() == "gpt-5.5"


class TestCodexResolveModel:
    def test_resolves_fast_to_mini(self):
        backend = CodexBackend()
        assert backend.resolve_model("fast") == "gpt-5.4-mini"

    def test_resolves_balanced_to_codex(self):
        backend = CodexBackend()
        assert backend.resolve_model("balanced") == "gpt-5.4"

    def test_resolves_powerful_to_max(self):
        backend = CodexBackend()
        assert backend.resolve_model("powerful") == "gpt-5.5"

    def test_resolves_gpt_5_4_mini_direct(self):
        backend = CodexBackend()
        assert backend.resolve_model("gpt-5.4-mini") == "gpt-5.4-mini"

    def test_resolves_direct_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("gpt-5.5") == "gpt-5.5"

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
        assert "--model" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "-C" in cmd

    def test_omits_bypass_when_require_approval(self, _make_request):
        backend = CodexBackend()
        request = _make_request(permission_mode="require_approval")

        cmd = backend.build_command(request)

        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd

    def test_includes_prompt_as_last_arg(self, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        # The prompt (plus correlation marker) is a single verbatim argv token;
        # the native binary is launched directly so no escaping/wrapping.
        assert "fix the bug" in cmd[-1]
        assert codex_correlation_token("worker@team") in cmd[-1]

    def test_passes_multiline_prompt_verbatim_as_single_arg(self, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="first line\nsecond line")

        cmd = backend.build_command(request)

        assert cmd[-1].startswith("first line\nsecond line")
        assert "\n" in cmd[-1]
        assert codex_correlation_token("worker@team") in cmd[-1]

    def test_passes_cmd_metachars_verbatim(self, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="write <!-- DONE --> & exit | done (x)")

        cmd = backend.build_command(request)

        assert "write <!-- DONE --> & exit | done (x)" in cmd[-1]

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
    def test_passes_agent_identity(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {
            "AGENT_NAME": request.name,
            "AGENT_SESSION_ID": request.team_name,
            "AGENT_PARENT_NAME": request.lead_session_id,
        }


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

        assert "model_reasoning_effort=xhigh" in cmd

    def test_build_command_keeps_prompt_last_with_effort(self, _make_request):
        backend = CodexBackend()
        request = _make_request(reasoning_effort="low", prompt="fix the bug")

        cmd = backend.build_command(request)

        assert "fix the bug" in cmd[-1]
        assert codex_correlation_token("worker@team") in cmd[-1]

    def test_build_command_omits_c_override_when_none(self, _make_request):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert not any(arg.startswith("model_reasoning_effort=") for arg in cmd)


class TestCodexMcpIdentity:
    def _identity_token(self, cmd: list[str]) -> str:
        return next(
            arg for arg in cmd if arg.startswith("mcp_servers.win-agent-teams.env=")
        )

    def test_build_command_injects_identity_env_override(self, _make_request):
        backend = CodexBackend()
        request = _make_request(name="worker", team_name="sess-uuid")

        cmd = backend.build_command(request)

        assert "-c" in cmd
        token = self._identity_token(cmd)
        _, _, value = token.partition("=")
        # value portion must parse as TOML and carry this agent's identity
        import tomllib

        parsed = tomllib.loads("x=" + value)["x"]
        assert parsed == {
            "CLAUDE_TEAMS_PERMISSION_MODE": "bypass",
            "AGENT_NAME": "worker",
            "AGENT_SESSION_ID": "sess-uuid",
            "AGENT_PARENT_NAME": request.lead_session_id,
        }

    def test_build_resume_command_injects_identity_env_override(self, _make_request):
        backend = CodexBackend()
        request = _make_request(name="worker", team_name="sess-uuid")

        cmd = backend.build_resume_command(request, "codex-session-123")

        token = self._identity_token(cmd)
        assert "AGENT_SESSION_ID = 'sess-uuid'" in token
        assert "codex-session-123" in cmd

    def test_rejects_single_quote_in_identity(self, _make_request):
        backend = CodexBackend()
        request = _make_request(name="bad'name")

        with pytest.raises(ValueError, match="TOML literal"):
            backend.build_command(request)


class TestCodexHookOverrides:
    def test_build_command_omits_hook_overrides_by_default(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", raising=False)
        backend = CodexBackend()
        request = _make_request(
            extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'}
        )

        cmd = backend.build_command(request)

        assert not any("hooks." in arg for arg in cmd)

    def test_build_command_includes_hook_overrides_when_enabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "1")
        backend = CodexBackend()
        request = _make_request(
            extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'}
        )

        cmd = backend.build_command(request)

        assert "hooks.Stop=[]" in cmd

    def test_build_resume_command_includes_hook_overrides_when_enabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "1")
        backend = CodexBackend()
        request = _make_request(
            extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'}
        )

        cmd = backend.build_resume_command(request, "resume-session-id")

        assert "hooks.Stop=[]" in cmd

    def test_build_command_omits_hook_overrides_when_extra_missing(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "1")
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert not any("hooks." in arg for arg in cmd)


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
        assert "go" in cmd[-1]
        assert codex_correlation_token("worker@team") in cmd[-1]

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


_TRIPLE = "x86_64-pc-windows-msvc"


def _make_codex_npm_tree(
    tmp_path: Path, exe_subdir: str | None, tools_subdir: str | None = None
) -> tuple[Path, Path, Path]:
    """Build a fake npm-global Codex install under ``tmp_path``.

    Returns ``(shim_path, exe_path, vendor_dir)``. ``exe_subdir=None`` omits the
    native exe entirely (forces the shim fallback).
    """
    npm = tmp_path / "npm"
    npm.mkdir(exist_ok=True)
    shim = npm / "codex.CMD"
    shim.write_text("@echo shim")
    vendor = (
        npm
        / "node_modules"
        / "@openai"
        / "codex"
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / _TRIPLE
    )
    vendor.mkdir(parents=True)
    exe_path = vendor / (exe_subdir or "bin") / "codex.exe"
    if exe_subdir is not None:
        exe_path.parent.mkdir(parents=True, exist_ok=True)
        exe_path.write_text("MZ")
    if tools_subdir is not None:
        (vendor / tools_subdir).mkdir(parents=True, exist_ok=True)
    return shim, exe_path, vendor


def _patch_windows(monkeypatch: pytest.MonkeyPatch, shim: Path) -> None:
    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr("shutil.which", lambda name: str(shim))


class TestCodexDiscoverBinary:
    def test_resolves_new_bin_layout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, exe, _ = _make_codex_npm_tree(tmp_path, "bin")
        _patch_windows(monkeypatch, shim)

        assert CodexBackend().discover_binary() == str(exe)

    def test_resolves_old_codex_layout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, exe, _ = _make_codex_npm_tree(tmp_path, "codex")
        _patch_windows(monkeypatch, shim)

        assert CodexBackend().discover_binary() == str(exe)

    def test_prefers_bin_over_codex_when_both_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, bin_exe, vendor = _make_codex_npm_tree(tmp_path, "bin")
        old_exe = vendor / "codex" / "codex.exe"
        old_exe.parent.mkdir(parents=True, exist_ok=True)
        old_exe.write_text("MZ")
        _patch_windows(monkeypatch, shim)

        assert CodexBackend().discover_binary() == str(bin_exe)

    def test_falls_back_to_shim_when_native_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, _ = _make_codex_npm_tree(tmp_path, exe_subdir=None)
        _patch_windows(monkeypatch, shim)

        assert CodexBackend().discover_binary() == str(shim)


class TestCodexBuildEnvNativePath:
    def test_prepends_codex_path_tools_dir(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, vendor = _make_codex_npm_tree(
            tmp_path, "bin", tools_subdir="codex-path"
        )
        _patch_windows(monkeypatch, shim)

        env = CodexBackend().build_env(_make_request())

        assert env["CODEX_MANAGED_BY_NPM"] == "1"
        assert env["PATH"].startswith(str(vendor / "codex-path"))

    def test_prepends_legacy_path_tools_dir(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, vendor = _make_codex_npm_tree(tmp_path, "codex", tools_subdir="path")
        _patch_windows(monkeypatch, shim)

        env = CodexBackend().build_env(_make_request())

        assert env["PATH"].startswith(str(vendor / "path"))


class TestCodexPromptCmdShimFallback:
    """When native exe is missing we fall back to the cmd.exe shim; a
    multi-line prompt must then be JSON-wrapped so cmd.exe can't truncate it."""

    def test_initial_prompt_json_wrapped_via_cmd_shim(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, _ = _make_codex_npm_tree(tmp_path, exe_subdir=None)
        _patch_windows(monkeypatch, shim)
        request = _make_request(prompt="first line\nsecond line")

        cmd = CodexBackend().build_command(request)

        assert cmd[0] == str(shim)
        assert cmd[-1].startswith(
            "Decode this JSON string as your complete task prompt,"
        )
        # carried as a single-line JSON token -> no real newline reaches cmd.exe
        assert "\n" not in cmd[-1]
        assert "first line" in cmd[-1]

    def test_resume_prompt_json_wrapped_via_cmd_shim(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, _ = _make_codex_npm_tree(tmp_path, exe_subdir=None)
        _patch_windows(monkeypatch, shim)
        request = _make_request(prompt="first line\nsecond line")

        cmd = CodexBackend().build_resume_command(request, "sess-id")

        assert cmd[-1] == (
            "Decode this JSON string as your complete task prompt, then follow "
            'the decoded text exactly: "first line\\nsecond line"'
        )

    def test_single_line_prompt_not_wrapped_via_cmd_shim(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, _ = _make_codex_npm_tree(tmp_path, exe_subdir=None)
        _patch_windows(monkeypatch, shim)
        request = _make_request(prompt="single line")

        cmd = CodexBackend().build_resume_command(request, "sess-id")

        assert cmd[-1] == "single line"

    def test_native_exe_keeps_multiline_verbatim(
        self, _make_request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        shim, _, _ = _make_codex_npm_tree(tmp_path, "bin")
        _patch_windows(monkeypatch, shim)
        request = _make_request(prompt="first line\nsecond line")

        cmd = CodexBackend().build_resume_command(request, "sess-id")

        assert cmd[-1] == "first line\nsecond line"
