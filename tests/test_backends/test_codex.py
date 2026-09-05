from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.agent_output import CORRELATION_FIELD, correlation_marker_token
from claude_teams.backends import codex as codex_module
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.codex import CodexBackend
from claude_teams.backends.contracts import BackendModelUnavailableError


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
        request = replace(default, **overrides)
        # Every real spawn carries the server-issued correlation id in ``extra``;
        # the codex backend reads it from there instead of deriving one.
        return replace(
            request,
            extra={CORRELATION_FIELD: "corr-test", **(request.extra or {})},
        )

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


@pytest.fixture
def _stub_discovery(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that stubs live model discovery with a fixed slug list."""

    def apply(slugs: list[str]) -> None:
        monkeypatch.setattr(
            CodexBackend, "discover_binary", lambda self: "/usr/bin/codex"
        )
        monkeypatch.setattr(
            codex_module, "_discover_codex_model_slugs", lambda binary: list(slugs)
        )

    return apply


class TestCodexSupportedModels:
    def test_returns_capability_tiers_not_slugs(self):
        backend = CodexBackend()
        assert backend.supported_models() == [
            "cheapest",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]


class TestCodexDefaultModel:
    def test_returns_medium_tier(self):
        backend = CodexBackend()
        assert backend.default_model() == "medium"


class TestCodexResolveModel:
    def test_resolves_tier_to_slug(self):
        backend = CodexBackend()
        assert backend.resolve_model("cheapest") == "gpt-5.6-luna"
        assert backend.resolve_model("low") == "gpt-5.6-luna"
        assert backend.resolve_model("medium") == "gpt-5.6-luna"
        assert backend.resolve_model("high") == "gpt-5.6-sol"
        assert backend.resolve_model("xhigh") == "gpt-6-astra"
        assert backend.resolve_model("max") == "gpt-6-astra"

    def test_passes_through_direct_slug(self):
        backend = CodexBackend()
        assert backend.resolve_model("gpt-5.6-terra") == "gpt-5.6-terra"

    def test_passes_through_unknown_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("custom-model") == "custom-model"

    def test_passes_through_empty_string(self):
        backend = CodexBackend()
        assert backend.resolve_model("") == ""


class TestCodexResolveLaunch:
    def test_tiers_map_to_model_and_effort(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"])
        backend = CodexBackend()
        assert backend.resolve_launch("cheapest", None) == (
            "gpt-5.6-luna",
            "medium",
        )
        assert backend.resolve_launch("low", None) == ("gpt-5.6-luna", "high")
        assert backend.resolve_launch("medium", None) == ("gpt-5.6-luna", "xhigh")
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("xhigh", None) == ("gpt-6-astra", "low")
        assert backend.resolve_launch("max", None) == ("gpt-6-astra", "medium")

    def test_legacy_luna_env_has_no_effect(self, _stub_discovery, monkeypatch):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"])
        monkeypatch.setenv("WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS", "1")
        backend = CodexBackend()
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("xhigh", None) == ("gpt-6-astra", "low")
        assert backend.resolve_launch("max", None) == ("gpt-6-astra", "medium")

    def test_old_ultra_name_uses_raw_slug_passthrough(self, _stub_discovery):
        _stub_discovery(["ultra"])
        assert CodexBackend().resolve_launch("ultra", None) == ("ultra", None)

    def test_explicit_effort_ignored_for_tier(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"])
        backend = CodexBackend()
        # A tier owns its effort; a caller-supplied reasoning_effort is
        # silently ignored and the bundled tier effort is used.
        assert backend.resolve_launch("low", "max") == ("gpt-5.6-luna", "high")
        assert backend.resolve_launch("high", "xhigh") == ("gpt-5.6-sol", "medium")
        assert backend.resolve_launch("xhigh", "max") == ("gpt-6-astra", "low")

    def test_blank_model_defers_to_codex_config(self):
        backend = CodexBackend()
        assert backend.resolve_launch("", None) == ("", None)
        # an explicit effort still applies to codex's own default model
        assert backend.resolve_launch("", "high") == ("", "high")

    def test_raw_slug_passthrough_when_available(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-terra", "gpt-5.6-sol"])
        backend = CodexBackend()
        assert backend.resolve_launch("gpt-5.6-terra", "xhigh") == (
            "gpt-5.6-terra",
            "xhigh",
        )

    def test_errors_when_sol_tier_unavailable(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra"])
        backend = CodexBackend()
        with pytest.raises(BackendModelUnavailableError):
            backend.resolve_launch("high", None)

    def test_luna_and_sol_tiers_ok_without_astra(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol"])
        backend = CodexBackend()
        assert backend.resolve_launch("cheapest", None) == (
            "gpt-5.6-luna",
            "medium",
        )
        assert backend.resolve_launch("low", None) == ("gpt-5.6-luna", "high")
        assert backend.resolve_launch("medium", None) == ("gpt-5.6-luna", "xhigh")
        assert backend.resolve_launch("high", None) == ("gpt-5.6-sol", "medium")

    def test_errors_when_luna_tier_unavailable(self, _stub_discovery):
        _stub_discovery(["gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"])
        backend = CodexBackend()
        with pytest.raises(BackendModelUnavailableError):
            backend.resolve_launch("medium", None)

    @pytest.mark.parametrize("tier", ["xhigh", "max"])
    def test_missing_astra_tier_includes_upgrade_hint(self, _stub_discovery, tier):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol"])
        backend = CodexBackend()
        with pytest.raises(
            BackendModelUnavailableError,
            match=r"npm install -g @openai/codex@latest",
        ):
            backend.resolve_launch(tier, None)

    def test_errors_when_no_5_6_at_all(self, _stub_discovery):
        _stub_discovery(["gpt-5.5"])
        backend = CodexBackend()
        with pytest.raises(BackendModelUnavailableError):
            backend.resolve_launch("low", None)

    def test_skips_validation_when_discovery_unavailable(self, _stub_discovery):
        _stub_discovery([])  # cannot determine -> no false error
        backend = CodexBackend()
        assert backend.resolve_launch("max", None) == ("gpt-6-astra", "medium")


class TestCodexBuildCommand:
    def test_produces_interactive_command_when_tty(self, _make_request, monkeypatch):
        # With a real TTY (WT tab / new console / tmux / terminal window) the
        # interactive TUI is used so the session stays visible to the user.
        monkeypatch.setattr(
            codex_module.process_manager,
            "provides_tty",
            lambda backend_type, *, is_interactive=False: True,
        )
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" not in cmd
        assert "--model" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "-C" in cmd

    def test_produces_exec_command_when_no_tty(self, _make_request, monkeypatch):
        # codex-cli 0.142.5's interactive TUI aborts with "stdin is not a
        # terminal" when spawned without a TTY, so the backend falls back to
        # the non-interactive ``codex exec`` entrypoint (which still runs
        # hooks and MCP servers).
        monkeypatch.setattr(
            codex_module.process_manager,
            "provides_tty",
            lambda backend_type, *, is_interactive=False: False,
        )
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1] == "exec"
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
        assert correlation_marker_token("corr-test") in cmd[-1]

    def test_passes_multiline_prompt_verbatim_as_single_arg(self, _make_request):
        backend = CodexBackend()
        request = _make_request(prompt="first line\nsecond line")

        cmd = backend.build_command(request)

        assert cmd[-1].startswith("first line\nsecond line")
        assert "\n" in cmd[-1]
        assert correlation_marker_token("corr-test") in cmd[-1]

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


class TestCodexModelArg:
    def test_build_command_emits_c_model_override_when_set(self, _make_request):
        backend = CodexBackend()
        request = _make_request(model="gpt-5.6-terra")

        cmd = backend.build_command(request)

        assert "model='gpt-5.6-terra'" in cmd
        # rendered right after its own -c flag
        idx = cmd.index("model='gpt-5.6-terra'")
        assert cmd[idx - 1] == "-c"

    def test_build_command_emits_default_tier_launch(
        self, _make_request, _stub_discovery
    ):
        # The default tier's resolved (slug, effort) must survive into argv:
        # tuple-level resolve_launch assertions alone don't prove that.
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"])
        backend = CodexBackend()
        model, effort = backend.resolve_launch("medium", None)

        cmd = backend.build_command(_make_request(model=model, reasoning_effort=effort))

        assert "model='gpt-5.6-luna'" in cmd
        assert "model_reasoning_effort=xhigh" in cmd

    def test_build_command_emits_max_tier_launch(self, _make_request, _stub_discovery):
        _stub_discovery(["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"])
        backend = CodexBackend()
        model, effort = backend.resolve_launch("max", None)

        cmd = backend.build_command(_make_request(model=model, reasoning_effort=effort))

        assert "model='gpt-6-astra'" in cmd
        assert "model_reasoning_effort=medium" in cmd

    def test_build_command_omits_c_model_override_when_blank(self, _make_request):
        backend = CodexBackend()
        request = _make_request(model="")

        cmd = backend.build_command(request)

        assert not any(arg.startswith("model=") for arg in cmd)

    def test_build_resume_command_emits_c_model_override_when_set(self, _make_request):
        backend = CodexBackend()
        request = _make_request(model="gpt-5.6-luna")

        cmd = backend.build_resume_command(request, "codex-session-123")

        assert "model='gpt-5.6-luna'" in cmd


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
        assert spec.options == frozenset(
            {"low", "medium", "high", "xhigh", "max", "ultra"}
        )

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
        assert correlation_marker_token("corr-test") in cmd[-1]

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
    def test_build_command_includes_hook_overrides_by_default(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", raising=False)
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_command(request)

        assert "hooks.Stop=[]" in cmd
        assert "--dangerously-bypass-hook-trust" in cmd

    def test_build_command_includes_hook_overrides_when_explicitly_enabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "1")
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_command(request)

        assert "hooks.Stop=[]" in cmd
        assert "--dangerously-bypass-hook-trust" in cmd

    def test_build_resume_command_includes_hook_overrides_by_default(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", raising=False)
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_resume_command(request, "resume-session-id")

        assert "hooks.Stop=[]" in cmd
        assert "--dangerously-bypass-hook-trust" in cmd

    def test_build_command_omits_hook_overrides_when_extra_missing(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", raising=False)
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert not any("hooks." in arg for arg in cmd)
        # No hook overrides to trust, but hooks are conceptually enabled;
        # the trust-bypass flag only accompanies actual -c hook args.
        assert "--dangerously-bypass-hook-trust" not in cmd

    def test_build_command_omits_hook_overrides_when_codex_specific_disabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "0")
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_command(request)

        assert not any("hooks." in arg for arg in cmd)
        assert "--dangerously-bypass-hook-trust" not in cmd

    def test_build_command_omits_hook_overrides_when_global_hooks_disabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", raising=False)
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_command(request)

        assert not any("hooks." in arg for arg in cmd)
        assert "--dangerously-bypass-hook-trust" not in cmd

    def test_build_resume_command_omits_hook_overrides_when_codex_specific_disabled(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS_CODEX", "0")
        backend = CodexBackend()
        request = _make_request(extra={"hook_overrides": '["-c", "hooks.Stop=[]"]'})

        cmd = backend.build_resume_command(request, "resume-session-id")

        assert not any("hooks." in arg for arg in cmd)
        assert "--dangerously-bypass-hook-trust" not in cmd


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
        assert correlation_marker_token("corr-test") in cmd[-1]

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
