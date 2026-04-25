"""Backend-oriented server operation tests."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

from claude_teams import teams, templates
from claude_teams.backends import registry
from claude_teams.backends.base import (
    AgentProfile,
    AgentSelectSpec,
    HealthStatus,
    ReasoningEffortSpec,
)
from claude_teams.backends.base import (
    SpawnResult as BackendSpawnResult,
)
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.models import TeammateMember
from claude_teams.templates import AgentTemplate
from tests._server_support import (
    _data,
    _make_mock_backend,
    _make_teammate,
    _text,
)


class TestProcessShutdownGuard:
    async def test_should_reject_shutdown_of_team_lead(self, team_client: Client):
        result = await team_client.call_tool(
            "process_shutdown_approved",
            {"team_name": "test-team", "agent_name": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in _text(result)


class TestProcessShutdownCleanup:
    async def test_kills_via_correct_backend(self, team_client: Client, tmp_path: Path):
        mate = _make_teammate("graceful", "test-team", tmp_path, pane_id="5555")
        mate.backend_type = "claude-code"
        mate.process_handle = "5555"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        result = _data(
            await team_client.call_tool(
                "process_shutdown_approved",
                {"team_name": "test-team", "agent_name": "graceful"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("5555")

    async def test_legacy_tmux_backend_type_maps_to_claude_code(
        self, team_client: Client, tmp_path: Path
    ):
        mate = _make_teammate("legacy", "test-team", tmp_path, pane_id="%56")
        mate.backend_type = "tmux"
        mate.process_handle = "%56"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        result = _data(
            await team_client.call_tool(
                "process_shutdown_approved",
                {"team_name": "test-team", "agent_name": "legacy"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%56")

    async def test_skips_cleanup_when_backend_unavailable(
        self, team_client: Client, tmp_path: Path
    ):
        mate = _make_teammate("orphaned", "test-team", tmp_path, pane_id="5757")
        mate.backend_type = "nonexistent"
        mate.process_handle = "5757"
        await teams.add_member("test-team", mate)

        result = _data(
            await team_client.call_tool(
                "process_shutdown_approved",
                {"team_name": "test-team", "agent_name": "orphaned"},
            )
        )

        assert result["success"] is True


class TestHealthCheck:
    async def test_returns_alive_for_running_teammate(self, team_client: Client):
        result = _data(
            await team_client.call_tool(
                "health_check", {"team_name": "test-team", "agent_name": "worker"}
            )
        )

        assert result["alive"] is True
        assert result["agent_name"] == "worker"
        assert "backend" in result
        assert "detail" in result

    async def test_returns_dead_when_backend_says_dead(self, team_client: Client):
        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.health_check.return_value = HealthStatus(
            alive=False, detail="pane gone"
        )

        result = _data(
            await team_client.call_tool(
                "health_check", {"team_name": "test-team", "agent_name": "worker"}
            )
        )

        assert result["alive"] is False
        # Pin the call: if routing ever short-circuits to a cached alive=True
        # branch, the backend would never be queried and the prior assertion
        # would silently drift out of sync.
        mock_backend.health_check.assert_called_once_with("%mock")

    async def test_rejects_nonexistent_teammate(self, team_client: Client):
        result = await team_client.call_tool(
            "health_check",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_rejects_nonexistent_team(self, team_client: Client):
        result = await team_client.call_tool(
            "health_check",
            {"team_name": "no-such-team", "agent_name": "worker"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()


class TestAgentLogs:
    async def test_returns_agent_log_tail(
        self,
        team_client: Client,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path / "logs"))
        log_dir = tmp_path / "logs" / "test-team"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "worker.log"
        log_path.write_text("one\ntwo\nthree\n")

        result = _data(
            await team_client.call_tool(
                "get_agent_logs",
                {"team_name": "test-team", "agent_name": "worker", "tail": 2},
            )
        )

        assert result["agent_name"] == "worker"
        assert result["tail"] == 2
        assert result["content"] == "two\nthree"
        assert result["log_path"] == str(log_path)


class TestSpawnWithBackend:
    async def test_spawns_with_explicit_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb1"})
        result = _data(
            await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "sb1",
                    "name": "coder",
                    "prompt": "write code",
                    "options": {"backend": "claude-code"},
                },
            )
        )
        assert result["name"] == "coder"
        assert result["team_name"] == "sb1"

    async def test_spawns_with_default_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb2"})
        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.spawn.return_value = BackendSpawnResult(
            process_handle="2002",
            backend_type="claude-code",
        )
        result = _data(
            await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "sb2",
                    "name": "coder",
                    "prompt": "write code",
                },
            )
        )
        assert result["name"] == "coder"
        cfg = await teams.read_config("sb2")
        teammate = next(
            member
            for member in cfg.members
            if isinstance(member, TeammateMember) and member.name == "coder"
        )
        assert teammate.process_handle == "2002"
        assert teammate.process_handle.isdecimal()
        assert teammate.tmux_pane_id == "2002"

    async def test_assigns_distinct_colors_in_join_order(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb-colors"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-colors",
                "name": "alice",
                "prompt": "write code",
            },
        )
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-colors",
                "name": "bob",
                "prompt": "review code",
            },
        )

        cfg = await teams.read_config("sb-colors")
        teammates = {
            member.name: member
            for member in cfg.members
            if isinstance(member, TeammateMember)
        }

        assert teammates["alice"].color == "blue"
        assert teammates["bob"].color == "green"

    async def test_rejects_invalid_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb3"})
        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb3",
                "name": "coder",
                "prompt": "write code",
                "options": {"backend": "nonexistent-backend"},
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "nonexistent-backend" in _text(result)

    async def test_uses_explicit_cwd_when_provided(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "sb-cwd"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-cwd",
                "name": "coder",
                "prompt": "write code",
                "options": {"cwd": str(tmp_path)},
            },
        )

        request = mock_backend.spawn.call_args.args[0]
        assert request.cwd == str(tmp_path)

        cfg = await teams.read_config("sb-cwd")
        teammate = next(
            member
            for member in cfg.members
            if isinstance(member, TeammateMember) and member.name == "coder"
        )
        assert teammate.cwd == str(tmp_path)

    async def test_rejects_relative_cwd(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb-cwd-relative"})

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-cwd-relative",
                "name": "coder",
                "prompt": "write code",
                "options": {"cwd": "relative/path"},
            },
            raise_on_error=False,
        )

        assert result.is_error is True
        assert "absolute path" in _text(result)

    async def test_rejects_missing_cwd(self, client: Client, tmp_path: Path):
        await client.call_tool("team_create", {"team_name": "sb-cwd-missing"})

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-cwd-missing",
                "name": "coder",
                "prompt": "write code",
                "options": {"cwd": str(tmp_path / "missing")},
            },
            raise_on_error=False,
        )

        assert result.is_error is True
        assert "does not exist" in _text(result)

    async def test_resolves_generic_model_name(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb4"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb4",
                "name": "coder",
                "prompt": "write code",
                "options": {"model": "fast"},
            },
        )
        mock_backend.resolve_model.assert_called_with("fast")

    async def test_rejects_invalid_model_for_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb5"})

        # Swap in the REAL ClaudeCodeBackend so the rejection branch runs
        # against live model logic — mocking resolve_model's side_effect only
        # tested that errors propagate, not that "bogus" would actually be
        # rejected by production code. resolve_model is pure and runs long
        # before any spawn I/O, so no process ever starts.
        original = registry._backends["claude-code"]
        registry._backends["claude-code"] = ClaudeCodeBackend()
        try:
            result = await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "sb5",
                    "name": "coder",
                    "prompt": "write code",
                    "options": {"model": "bogus"},
                },
                raise_on_error=False,
            )
            assert result.is_error is True
            assert "bogus" in _text(result)
        finally:
            registry._backends["claude-code"] = original

    async def test_passes_permission_mode_through_to_backend_request(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "sb6"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.supports_permission_bypass.return_value = True

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb6",
                "name": "coder",
                "prompt": "write code",
                "options": {"permission_mode": "bypass"},
            },
        )

        request = mock_backend.spawn.call_args.args[0]
        assert request.permission_mode == "bypass"

    async def test_rejects_permission_bypass_for_unsupported_backend(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "sb7"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.supports_permission_bypass.return_value = False

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb7",
                "name": "coder",
                "prompt": "write code",
                "options": {"permission_mode": "bypass"},
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "permission_mode='bypass'" in _text(result)


class TestForceKillWithBackend:
    async def test_kills_via_correct_backend(self, team_client: Client, tmp_path: Path):
        mate = _make_teammate("victim", "test-team", tmp_path, pane_id="7777")
        mate.backend_type = "claude-code"
        mate.process_handle = "7777"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "victim"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("7777")

    async def test_legacy_tmux_backend_type_maps_to_claude_code(
        self, team_client: Client, tmp_path: Path
    ):
        mate = _make_teammate("oldmate", "test-team", tmp_path, pane_id="%88")
        mate.backend_type = "tmux"
        mate.process_handle = "%88"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "oldmate"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%88")

    async def test_rejects_nonexistent_teammate(self, team_client: Client):
        result = await team_client.call_tool(
            "force_kill_teammate",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_skips_kill_when_backend_unavailable(
        self, team_client: Client, tmp_path: Path
    ):
        mate = _make_teammate("orphan", "test-team", tmp_path, pane_id="9999")
        mate.backend_type = "nonexistent"
        mate.process_handle = "9999"
        await teams.add_member("test-team", mate)

        # Should not raise even if backend is unavailable
        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "orphan"},
            )
        )
        assert result["success"] is True


class TestSpawnTemplate:
    """Template application through the spawn pipeline.

    Covers the three observable guarantees of ``apply_template``:
    role-prompt composition, default-field merging, and error taxonomy.
    Reaches into the mock backend's captured ``SpawnRequest`` to confirm
    values traverse the full tool → backend path, not just the top of
    ``spawn_teammate_tool``.
    """

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        templates._seed_builtin_templates()

    async def test_applies_role_prompt_and_default_subagent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tpl-role"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-role",
                "name": "reviewer",
                "prompt": "review PR #42",
                "options": {"template": "code-reviewer"},
            },
        )

        request = mock_backend.spawn.call_args.args[0]
        # Role prompt prepended with blank-line separator.
        assert request.prompt.startswith("You are acting as a code reviewer.")
        assert "\n\nreview PR #42" in request.prompt
        # Default subagent_type from the template reached the backend.
        assert request.agent_type == "code-reviewer"

        cfg = await teams.read_config("tpl-role")
        teammate = next(
            m
            for m in cfg.members
            if isinstance(m, TeammateMember) and m.name == "reviewer"
        )
        assert teammate.agent_type == "code-reviewer"

    async def test_explicit_option_overrides_template_default(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tpl-override"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-override",
                "name": "reviewer",
                "prompt": "review",
                "options": {
                    "template": "code-reviewer",
                    "subagent_type": "executor",
                },
            },
        )

        request = mock_backend.spawn.call_args.args[0]
        # Explicit subagent_type wins over the template's default.
        assert request.agent_type == "executor"
        # But the role_prompt still composes — it has no "explicit" equivalent.
        assert request.prompt.startswith("You are acting as a code reviewer.")

    async def test_empty_role_prompt_does_not_compose(self, client: Client):
        templates.register_template(
            AgentTemplate(
                name="no-prefix",
                description="Defaults only, no role header.",
                default_subagent_type="executor",
            )
        )
        await client.call_tool("team_create", {"team_name": "tpl-bare"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-bare",
                "name": "worker",
                "prompt": "do the task",
                "options": {"template": "no-prefix"},
            },
        )

        request = mock_backend.spawn.call_args.args[0]
        # No leading blank line when role_prompt is empty.
        assert request.prompt == "do the task"
        assert request.agent_type == "executor"

    async def test_unknown_template_raises_structured_error(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tpl-missing"})

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-missing",
                "name": "ghost",
                "prompt": "task",
                "options": {"template": "does-not-exist"},
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        text = _text(result)
        assert "does-not-exist" in text
        # Error surface should enumerate available templates so the caller
        # can correct the typo without guessing.
        assert "code-reviewer" in text

    @staticmethod
    def _configure_full_mock_backend() -> MagicMock:
        """Mount effort/profile specs on the mock so template defaults pass validation.

        ``apply_template`` defers validation of ``reasoning_effort`` and
        ``agent_profile`` to ``_validate_reasoning_effort`` /
        ``_validate_agent_profile``, which call the backend's spec methods.
        ``_make_mock_backend`` leaves those returning ``None`` (matching the
        no-op default most backends ship), so tests that exercise those two
        defaults must opt in to real specs here.
        """
        mock = cast(MagicMock, registry._backends["claude-code"])
        mock.reasoning_effort_spec.return_value = ReasoningEffortSpec(
            flag="--effort",
            value_template="{value}",
            options=frozenset({"low", "medium", "high"}),
        )
        mock.agent_select_spec.return_value = AgentSelectSpec(
            flag="--agent", value_template="{name}"
        )
        mock.discover_agents.return_value = [
            AgentProfile(name="senior", path="/abs/senior.md"),
            AgentProfile(name="junior", path="/abs/junior.md"),
        ]
        return mock

    @staticmethod
    def _register_full_defaults_template() -> None:
        """Register a template setting all 7 ``apply_template`` precedence fields."""
        templates.register_template(
            AgentTemplate(
                name="full-defaults",
                description="All 7 defaults set for precedence tests.",
                # Empty role_prompt isolates precedence assertions from
                # prompt-composition behaviour already covered above.
                role_prompt="",
                default_backend="claude-code",
                default_model="sonnet",
                default_subagent_type="tpl-sub",
                default_reasoning_effort="medium",
                default_agent_profile="senior",
                default_permission_mode="require_approval",
                default_plan_mode_required=True,
            )
        )

    async def test_template_defaults_fill_every_unset_field(self, client: Client):
        """All 7 template defaults reach the SpawnRequest when caller sets none.

        Covers the "template fills every gap" half of the precedence contract:
        ``backend``, ``model``, ``subagent_type``, ``reasoning_effort``,
        ``agent_profile``, ``permission_mode``, ``plan_mode_required``. The
        companion parametrized test locks the "explicit wins" half.
        """
        mock = self._configure_full_mock_backend()
        self._register_full_defaults_template()

        await client.call_tool("team_create", {"team_name": "tpl-fill-all"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-fill-all",
                "name": "target",
                "prompt": "do work",
                "options": {"template": "full-defaults"},
            },
        )

        request = mock.spawn.call_args.args[0]
        assert request.model == "sonnet"
        assert request.agent_type == "tpl-sub"
        assert request.reasoning_effort == "medium"
        assert request.agent_profile == "senior"
        assert request.permission_mode == "require_approval"
        assert request.plan_mode_required is True
        # ``default_backend="claude-code"`` routed the spawn to this mock.
        # Only the claude-code mock is registered, so a single spawn call
        # proves the default reached ``_resolve_backend``.
        assert mock.spawn.call_count == 1

    @pytest.mark.parametrize(
        ("field", "override_value", "request_attr", "expected"),
        [
            pytest.param("model", "haiku", "model", "haiku", id="model"),
            pytest.param(
                "subagent_type",
                "override-sub",
                "agent_type",
                "override-sub",
                id="subagent_type",
            ),
            pytest.param(
                "reasoning_effort",
                "high",
                "reasoning_effort",
                "high",
                id="reasoning_effort",
            ),
            pytest.param(
                "agent_profile",
                "junior",
                "agent_profile",
                "junior",
                id="agent_profile",
            ),
            pytest.param(
                "permission_mode",
                "bypass",
                "permission_mode",
                "bypass",
                id="permission_mode",
            ),
            pytest.param(
                "plan_mode_required",
                False,
                "plan_mode_required",
                False,
                id="plan_mode_required",
            ),
        ],
    )
    async def test_explicit_option_beats_template_default_per_field(
        self,
        client: Client,
        field: str,
        override_value: object,
        request_attr: str,
        expected: object,
    ):
        """Explicit option wins over the template default for each field.

        Backend precedence is exercised in its own test because it needs a
        second registered backend — the other 6 fields are deliberately kept
        in one parametrize block so the shared template + mock setup does not
        drift between near-identical assertions.
        """
        mock = self._configure_full_mock_backend()
        self._register_full_defaults_template()

        team_name = f"tpl-ov-{field.replace('_', '-')}"
        await client.call_tool("team_create", {"team_name": team_name})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": team_name,
                "name": "target",
                "prompt": "do work",
                "options": {"template": "full-defaults", field: override_value},
            },
        )

        request = mock.spawn.call_args.args[0]
        assert getattr(request, request_attr) == expected

    async def test_explicit_backend_beats_template_default(self, client: Client):
        """Explicit ``options.backend`` wins over the template's ``default_backend``.

        A second mock backend is registered so the template has somewhere
        other than ``claude-code`` to point at; the explicit override then
        must force selection back, proving the precedence without coupling
        to a specific non-default backend identifier.
        """
        alt_mock = _make_mock_backend("aider")
        # Default is_interactive=False for non-claude-code names, which would
        # schedule a one-shot relay task we do not need here.
        alt_mock.is_interactive = True
        registry._backends["aider"] = alt_mock

        templates.register_template(
            AgentTemplate(
                name="aider-default",
                description="Template routes to aider; override must win.",
                default_backend="aider",
                default_subagent_type="tpl-sub",
            )
        )

        await client.call_tool("team_create", {"team_name": "tpl-backend-override"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tpl-backend-override",
                "name": "target",
                "prompt": "do work",
                "options": {"template": "aider-default", "backend": "claude-code"},
            },
        )

        claude_mock = cast(MagicMock, registry._backends["claude-code"])
        assert claude_mock.spawn.call_count == 1
        assert alt_mock.spawn.call_count == 0
