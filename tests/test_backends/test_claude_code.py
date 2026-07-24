from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from claude_teams.agent_output import _CODEX_CORRELATION_PREFIX
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.claude_code import ClaudeCodeBackend


@pytest.fixture
def _make_request(tmp_path: Path) -> Callable[..., SpawnRequest]:
    default = SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="sonnet",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )

    def factory(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
        return replace(default, **overrides)

    return factory


class TestClaudeCodeProperties:
    def test_name_is_claude_code(self):
        backend = ClaudeCodeBackend()
        assert backend.name == "claude-code"

    def test_binary_name_is_claude(self):
        backend = ClaudeCodeBackend()
        assert backend.binary_name == "claude"

    def test_is_interactive(self):
        backend = ClaudeCodeBackend()
        assert backend.is_interactive is True


class TestClaudeCodeDiscoverBinary:
    def test_resolves_native_windows_binary_from_npm_shim(
        self, monkeypatch, tmp_path: Path
    ):
        shim = tmp_path / "npm" / "claude.CMD"
        exe = (
            tmp_path
            / "npm"
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        exe.parent.mkdir(parents=True)
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text("@ECHO off\n", encoding="utf-8")
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setattr(
            "claude_teams.backends.claude_code.shutil.which",
            lambda name: str(shim) if name == "claude" else None,
        )

        assert ClaudeCodeBackend().discover_binary() == str(exe)

    def test_falls_back_to_shim_when_native_binary_missing(
        self, monkeypatch, tmp_path: Path
    ):
        shim = tmp_path / "npm" / "claude.CMD"
        shim.parent.mkdir(parents=True)
        shim.write_text("@ECHO off\n", encoding="utf-8")
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setattr(
            "claude_teams.backends.claude_code.shutil.which",
            lambda name: str(shim) if name == "claude" else None,
        )

        assert ClaudeCodeBackend().discover_binary() == str(shim)


class TestClaudeCodeSupportedModels:
    def test_returns_expected_models(self):
        backend = ClaudeCodeBackend()
        models = backend.supported_models()
        assert "haiku" in models
        assert "sonnet" in models
        assert "opus" in models
        assert "fable" in models
        assert len(models) == 4


class TestClaudeCodeDefaultModel:
    def test_returns_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.default_model() == "sonnet"


class TestClaudeCodeResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("fast") == "haiku"

    def test_resolves_balanced_to_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("balanced") == "sonnet"

    def test_resolves_powerful_to_opus(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("powerful") == "opus"

    def test_resolves_direct_name_haiku(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("haiku") == "haiku"

    def test_resolves_direct_name_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("sonnet") == "sonnet"

    def test_resolves_direct_name_opus(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("opus") == "opus"

    def test_resolves_direct_name_fable(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("fable") == "fable"

    def test_raises_for_unsupported_model(self):
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError, match="Unsupported model"):
            backend.resolve_model("gpt-4")

    def test_raises_for_empty_string(self):
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError, match="Unsupported model"):
            backend.resolve_model("")


class TestClaudeCodeBuildCommand:
    def test_produces_correct_flags(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/claude"
        assert "--agent-id" in cmd
        assert "--agent-name" in cmd
        assert "--team-name" in cmd
        assert "--agent-color" in cmd
        assert "--parent-session-id" in cmd
        assert "--agent-type" in cmd
        assert "--model" in cmd
        # Values match request
        idx = cmd.index("--agent-id")
        assert cmd[idx + 1] == "worker@team"
        idx = cmd.index("--agent-name")
        assert cmd[idx + 1] == "worker"
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"
        # The backend no longer injects a correlation marker: the server owns
        # prompt materialization for both transports, so build_command must pass
        # the already-materialized prompt through untouched. That the marker
        # does reach the transcript is covered by test_correlation_transport's
        # test_marker_is_visible_in_claude_transcript_context.
        assert cmd[-1] == "do stuff"
        assert _CODEX_CORRELATION_PREFIX not in cmd[-1]

    def test_includes_plan_mode_required_when_set(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(plan_mode_required=True)

        cmd = backend.build_command(request)

        assert "--plan-mode-required" in cmd

    def test_excludes_plan_mode_required_when_false(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(plan_mode_required=False)

        cmd = backend.build_command(request)

        assert "--plan-mode-required" not in cmd

    def test_includes_bypass_permission_mode_when_requested(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(permission_mode="bypass")

        cmd = backend.build_command(request)

        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "bypassPermissions"

    def test_omits_permission_mode_flag_when_require_approval(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(permission_mode="require_approval")

        cmd = backend.build_command(request)

        assert "--permission-mode" not in cmd

    def test_includes_mcp_config_when_provided(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(extra={"mcp_config_path": "C:\\tmp\\worker.mcp.json"})

        cmd = backend.build_command(request)

        idx = cmd.index("--mcp-config")
        assert cmd[idx + 1] == "C:\\tmp\\worker.mcp.json"
        assert cmd[-2] == "--"
        assert cmd[-1] == "do stuff"
        assert _CODEX_CORRELATION_PREFIX not in cmd[-1]

    def test_terminates_options_before_prompt(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[-2] == "--"
        assert cmd[-1].startswith("do stuff")

    def test_preserves_multiline_prompt_as_single_arg(self, _make_request):
        backend = ClaudeCodeBackend()
        prompt = "first line\nsecond 'line'\nname: \u00c5sa"
        request = _make_request(prompt=prompt)

        cmd = backend.build_command(request)

        assert cmd[-1].startswith(prompt)
        assert "Decode this JSON string" not in cmd[-1]

    def test_uses_prompt_file_instruction_when_provided(self, _make_request):
        backend = ClaudeCodeBackend()
        prompt = "first 'line' and \"second\""
        request = _make_request(
            prompt=prompt,
            extra={"prompt_file_path": "C:\\sessions\\worker.prompt.txt"},
        )

        cmd = backend.build_command(request)

        assert cmd[-1].startswith(
            "Read your complete task prompt from UTF-8 file path "
            "C:\\sessions\\worker.prompt.txt then follow the file contents exactly."
        )
        # Inverted from main's assertion. Main rode the marker on the file-read
        # instruction because the *backend* injected it. The server now writes
        # the marker into the sidecar file itself, so argv carries only the read
        # instruction and must stay free of the marker — otherwise the sidecar
        # transport would double-mark (once in argv, once in the file the agent
        # reads). That the marker reaches the transcript via the file's
        # tool_result is covered by test_correlation_transport's
        # test_marker_is_visible_in_claude_transcript_context[sidecar].
        assert _CODEX_CORRELATION_PREFIX not in cmd[-1]
        assert prompt not in cmd[-1]
        assert "'" not in cmd[-1]
        assert '"' not in cmd[-1]


class TestClaudeCodeBuildEnv:
    def test_returns_claude_env_vars(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env["CLAUDECODE"] == "1"
        assert env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert len(env) == 5

    def test_supplies_child_identity_for_nested_orchestration(self, _make_request):
        request = _make_request(
            name="child",
            team_name="team-123",
            lead_session_id="parent-agent",
        )

        env = ClaudeCodeBackend().build_env(request)

        assert env["AGENT_NAME"] == "child"
        assert env["AGENT_SESSION_ID"] == "team-123"
        assert env["AGENT_PARENT_NAME"] == "parent-agent"


class TestClaudeCodePermissionSupport:
    def test_supports_permission_bypass(self):
        backend = ClaudeCodeBackend()
        assert backend.supports_permission_bypass() is True


class TestClaudeCodeReasoningEffort:
    def test_spec_advertises_effort_flag_and_options(self):
        backend = ClaudeCodeBackend()
        spec = backend.reasoning_effort_spec()
        assert spec is not None
        assert spec.flag == "--effort"
        assert spec.options == frozenset({"low", "medium", "high", "xhigh", "max"})

    def test_build_command_appends_effort_when_set(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request(reasoning_effort="high")

        cmd = backend.build_command(request)

        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"

    def test_build_command_omits_effort_flag_when_none(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--effort" not in cmd


class TestClaudeCodeHooksSettings:
    def test_build_command_appends_settings_flag_when_hooks_enabled(
        self, _make_request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        settings_path = tmp_path / "hooks-worker.settings.json"
        settings_path.write_text("{}", encoding="utf-8")
        backend = ClaudeCodeBackend()
        request = _make_request(extra={"hooks_settings_path": str(settings_path)})

        cmd = backend.build_command(request)

        idx = cmd.index("--settings")
        assert cmd[idx + 1] == str(settings_path)

    def test_build_command_omits_settings_flag_when_hooks_disabled(
        self, _make_request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")
        settings_path = tmp_path / "hooks-worker.settings.json"
        settings_path.write_text("{}", encoding="utf-8")
        backend = ClaudeCodeBackend()
        request = _make_request(extra={"hooks_settings_path": str(settings_path)})

        cmd = backend.build_command(request)

        assert "--settings" not in cmd

    def test_build_command_omits_settings_flag_when_path_absent(
        self, _make_request, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--settings" not in cmd

    def test_build_resume_command_appends_settings_flag_when_hooks_enabled(
        self, _make_request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_STATE_HOOKS", raising=False)
        settings_path = tmp_path / "hooks-worker.settings.json"
        settings_path.write_text("{}", encoding="utf-8")
        backend = ClaudeCodeBackend()
        request = _make_request(extra={"hooks_settings_path": str(settings_path)})

        cmd = backend.build_resume_command(request, "resume-session-id")

        idx = cmd.index("--settings")
        assert cmd[idx + 1] == str(settings_path)

    def test_build_resume_command_omits_settings_flag_when_hooks_disabled(
        self, _make_request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_STATE_HOOKS", "0")
        settings_path = tmp_path / "hooks-worker.settings.json"
        settings_path.write_text("{}", encoding="utf-8")
        backend = ClaudeCodeBackend()
        request = _make_request(extra={"hooks_settings_path": str(settings_path)})

        cmd = backend.build_resume_command(request, "resume-session-id")

        assert "--settings" not in cmd


class TestClaudeCodeDisallowedTools:
    """Spawned team agents are autonomous workers with no interactive user.

    Claude Code's native agent-teams mode (enabled here via
    ``CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`` + ``--parent-session-id``) routes
    ``AskUserQuestion`` as a permission request to the *team leader* session,
    which in this system is the MCP orchestrator that never runs a native
    approval queue -- so the child hangs forever on "Waiting for team lead
    approval". The tool is therefore disabled for spawned agents; they escalate
    decisions via ``send_message`` to ``lead`` instead.
    """

    def test_build_command_disallows_askuserquestion(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert "--disallowed-tools" in cmd
        idx = cmd.index("--disallowed-tools")
        assert cmd[idx + 1] == "AskUserQuestion"

    def test_build_resume_command_disallows_askuserquestion(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_resume_command(request, "resume-session-id")

        assert "--disallowed-tools" in cmd
        idx = cmd.index("--disallowed-tools")
        assert cmd[idx + 1] == "AskUserQuestion"

    def test_disallowed_tools_precede_prompt_terminator(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd.index("--disallowed-tools") < cmd.index("--")


class TestClaudeCodeAgentSelect:
    def test_spec_advertises_agent_flag(self):
        backend = ClaudeCodeBackend()
        spec = backend.agent_select_spec()
        assert spec is not None
        assert spec.flag == "--agent"
        assert spec.value_template == "{name}"

    def test_discover_finds_project_agents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "reviewer.md").write_text("reviewer")
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = ClaudeCodeBackend()
        profiles = backend.discover_agents(str(tmp_path))

        names = [p.name for p in profiles]
        assert "reviewer" in names

    def test_build_command_appends_agent_flag_when_discovered(
        self,
        _make_request,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "reviewer.md").write_text("reviewer")
        monkeypatch.setenv("HOME", str(tmp_path / "no-home"))

        backend = ClaudeCodeBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="reviewer")

        cmd = backend.build_command(request)

        assert "--agent" in cmd
        idx = cmd.index("--agent")
        assert cmd[idx + 1] == "reviewer"

    def test_build_command_omits_agent_flag_when_profile_none(self, _make_request):
        backend = ClaudeCodeBackend()
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

        backend = ClaudeCodeBackend()
        request = _make_request(cwd=str(tmp_path), agent_profile="ghost")

        cmd = backend.build_command(request)

        assert "--agent" not in cmd
