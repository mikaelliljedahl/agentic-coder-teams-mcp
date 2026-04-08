"""Backend-oriented server operation tests."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from fastmcp import Client

from claude_teams import teams
from claude_teams.backends import registry
from claude_teams.backends.base import HealthStatus
from claude_teams.models import TeammateMember
from tests._server_support import (
    _data,
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
    async def test_kills_via_correct_backend(self, team_client: Client):
        mate = _make_teammate("graceful", "test-team", pane_id="%55")
        mate.backend_type = "claude-code"
        mate.process_handle = "%55"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

        result = _data(
            await team_client.call_tool(
                "process_shutdown_approved",
                {"team_name": "test-team", "agent_name": "graceful"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%55")

    async def test_legacy_tmux_backend_type_maps_to_claude_code(
        self, team_client: Client
    ):
        mate = _make_teammate("legacy", "test-team", pane_id="%56")
        mate.backend_type = "tmux"
        mate.process_handle = "%56"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

        result = _data(
            await team_client.call_tool(
                "process_shutdown_approved",
                {"team_name": "test-team", "agent_name": "legacy"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%56")

    async def test_skips_cleanup_when_backend_unavailable(self, team_client: Client):
        mate = _make_teammate("orphaned", "test-team", pane_id="%57")
        mate.backend_type = "nonexistent"
        mate.process_handle = "%57"
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
        # Override mock to return dead
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
        # Restore original behavior
        mock_backend.health_check.return_value = HealthStatus(
            alive=True, detail="mock check"
        )

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
                    "backend": "claude-code",
                },
            )
        )
        assert result["name"] == "coder"
        assert result["team_name"] == "sb1"

    async def test_spawns_with_default_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb2"})
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
                "backend": "nonexistent-backend",
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
        mock_backend.spawn.reset_mock()

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb-cwd",
                "name": "coder",
                "prompt": "write code",
                "cwd": str(tmp_path),
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
                "cwd": "relative/path",
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
                "cwd": str(tmp_path / "missing"),
            },
            raise_on_error=False,
        )

        assert result.is_error is True
        assert "does not exist" in _text(result)

    async def test_resolves_generic_model_name(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb4"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.resolve_model.reset_mock()

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb4",
                "name": "coder",
                "prompt": "write code",
                "model": "fast",
            },
        )
        mock_backend.resolve_model.assert_called_with("fast")

    async def test_rejects_invalid_model_for_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb5"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        original_side_effect = mock_backend.resolve_model.side_effect
        mock_backend.resolve_model.side_effect = ValueError("Unsupported model 'bogus'")

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb5",
                "name": "coder",
                "prompt": "write code",
                "model": "bogus",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "bogus" in _text(result)

        mock_backend.resolve_model.side_effect = original_side_effect

    async def test_passes_permission_mode_through_to_backend_request(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "sb6"})

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.supports_permission_bypass.return_value = True
        mock_backend.spawn.reset_mock()

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb6",
                "name": "coder",
                "prompt": "write code",
                "permission_mode": "bypass",
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
                "permission_mode": "bypass",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "permission_mode='bypass'" in _text(result)


class TestForceKillWithBackend:
    async def test_kills_via_correct_backend(self, team_client: Client):
        mate = _make_teammate("victim", "test-team", pane_id="%77")
        mate.backend_type = "claude-code"
        mate.process_handle = "%77"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "victim"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%77")

    async def test_legacy_tmux_backend_type_maps_to_claude_code(
        self, team_client: Client
    ):
        mate = _make_teammate("oldmate", "test-team", pane_id="%88")
        mate.backend_type = "tmux"
        mate.process_handle = "%88"
        await teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

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

    async def test_skips_kill_when_backend_unavailable(self, team_client: Client):
        mate = _make_teammate("orphan", "test-team", pane_id="%99")
        mate.backend_type = "nonexistent"
        mate.process_handle = "%99"
        await teams.add_member("test-team", mate)

        # Should not raise even if backend is unavailable
        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "orphan"},
            )
        )
        assert result["success"] is True
