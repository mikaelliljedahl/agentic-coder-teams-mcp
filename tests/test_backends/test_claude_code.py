from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

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


class TestClaudeCodeSupportedModels:
    def test_returns_expected_models(self):
        backend = ClaudeCodeBackend()
        models = backend.supported_models()
        assert "haiku" in models
        assert "sonnet" in models
        assert "opus" in models
        assert len(models) == 3


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


class TestClaudeCodeBuildEnv:
    def test_returns_claude_env_vars(self, _make_request):
        backend = ClaudeCodeBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env["CLAUDECODE"] == "1"
        assert env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert len(env) == 2


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
        assert spec.options == frozenset({"low", "medium", "high", "max"})

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
