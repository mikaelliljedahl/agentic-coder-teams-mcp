"""Bootstrap and capability server tests."""

from pathlib import Path
from unittest.mock import patch

from fastmcp import Client

from claude_teams import teams
from claude_teams.backends import registry
from claude_teams.backends.base import AgentProfile, AgentSelectSpec
from claude_teams.server import mcp
from tests._server_support import (
    _data,
    _extract_capability,
    _items,
    _make_mock_backend,
    _text,
)


class TestProgressiveDisclosure:
    async def test_only_bootstrap_tools_at_startup(self, client: Client):
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Bootstrap tools should be visible
        assert "team_create" in names
        assert "team_attach" in names
        assert "team_delete" in names
        assert "list_backends" in names
        assert "list_agents" in names
        assert "read_config" in names
        # Team-tier tools should NOT be visible
        assert "spawn_teammate" not in names
        assert "send_message" not in names
        assert "task_create" not in names
        # Teammate-tier tools should NOT be visible
        assert "force_kill_teammate" not in names
        assert "poll_inbox" not in names
        assert "check_teammate" not in names
        assert "health_check" not in names

    async def test_team_tools_visible_after_create(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-test"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Team-tier tools should now be visible
        assert "spawn_teammate" in names
        assert "send_message" in names
        assert "task_create" in names
        assert "task_update" in names
        assert "task_list" in names
        assert "task_get" in names
        assert "read_inbox" in names
        # Teammate-tier tools should still NOT be visible
        assert "force_kill_teammate" not in names
        assert "poll_inbox" not in names
        assert "check_teammate" not in names

    async def test_teammate_tools_visible_after_spawn(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-test2"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "vis-test2",
                "name": "coder",
                "prompt": "write code",
            },
        )
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # All tiers should be visible
        assert "force_kill_teammate" in names
        assert "poll_inbox" in names
        assert "check_teammate" in names
        assert "process_shutdown_approved" in names
        assert "health_check" in names

    async def test_tools_hidden_after_delete(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-del"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "vis-del",
                "name": "temp",
                "prompt": "temporary",
            },
        )
        # Remove member so delete succeeds
        await teams.remove_member("vis-del", "temp")
        await client.call_tool("team_delete", {"team_name": "vis-del"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Only bootstrap should remain
        assert "team_create" in names
        assert "list_backends" in names
        assert "spawn_teammate" not in names
        assert "force_kill_teammate" not in names

    async def test_re_enable_cycle(self, client: Client):
        # Create -> delete -> re-create cycle
        await client.call_tool("team_create", {"team_name": "cycle1"})
        await client.call_tool("team_delete", {"team_name": "cycle1"})
        # After delete, team tools should be gone
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "spawn_teammate" not in names
        # Re-create should bring them back
        await client.call_tool("team_create", {"team_name": "cycle2"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "spawn_teammate" in names

    async def test_tool_annotations_are_exposed(self, client: Client):
        tool_list = await client.list_tools()
        read_config = next(t for t in tool_list if t.name == "read_config")
        team_delete = next(t for t in tool_list if t.name == "team_delete")

        assert read_config.annotations is not None
        assert read_config.annotations.readOnlyHint is True
        assert read_config.annotations.idempotentHint is True
        assert team_delete.annotations is not None
        assert team_delete.annotations.destructiveHint is True

        await client.call_tool("team_create", {"team_name": "annot-team"})
        tool_list = await client.list_tools()
        read_inbox = next(t for t in tool_list if t.name == "read_inbox")
        spawn_teammate = next(t for t in tool_list if t.name == "spawn_teammate")

        assert read_inbox.annotations is not None
        assert read_inbox.annotations.readOnlyHint is False
        assert spawn_teammate.annotations is not None
        assert spawn_teammate.annotations.destructiveHint is False


class TestCapabilities:
    async def test_worker_can_attach_from_separate_session_and_read_own_inbox(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "auth-team"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "auth-team",
                "name": "worker",
                "prompt": "help out",
            },
        )
        worker_inbox = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "auth-team",
                    "agent_name": "worker",
                    "mark_as_read": False,
                },
            )
        )
        capability = _extract_capability(worker_inbox[0]["text"])

        async with Client(mcp) as worker_client:
            attach_result = _data(
                await worker_client.call_tool(
                    "team_attach",
                    {"team_name": "auth-team", "capability": capability},
                )
            )
            assert attach_result["principal_name"] == "worker"
            assert attach_result["principal_role"] == "agent"

            inbox = _items(
                await worker_client.call_tool(
                    "read_inbox",
                    {"team_name": "auth-team", "agent_name": "worker"},
                )
            )
            assert len(inbox) == 1

    async def test_agent_attach_cannot_use_lead_only_tools(self, client: Client):
        await client.call_tool("team_create", {"team_name": "auth-team-2"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "auth-team-2",
                "name": "worker",
                "prompt": "help out",
            },
        )
        worker_inbox = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "auth-team-2",
                    "agent_name": "worker",
                    "mark_as_read": False,
                },
            )
        )
        capability = _extract_capability(worker_inbox[0]["text"])

        async with Client(mcp) as worker_client:
            await worker_client.call_tool(
                "team_attach",
                {"team_name": "auth-team-2", "capability": capability},
            )
            result = await worker_client.call_tool(
                "check_teammate",
                {"team_name": "auth-team-2", "agent_name": "worker"},
                raise_on_error=False,
            )
            assert result.is_error is True
            assert "team-lead capability" in _text(result)

    async def test_read_config_rejects_invalid_team_name(self, client: Client):
        result = await client.call_tool(
            "read_config",
            {"team_name": "../bad-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        text = _text(result)
        # ``strict_input_validation=True`` routes inputs through jsonschema, which
        # surfaces the offending value and failing pattern. The test pins the
        # schema-level guarantee: path-traversal inputs never reach the tool body.
        assert "'../bad-team'" in text
        assert "^[A-Za-z0-9_-]+$" in text


class TestListBackends:
    async def test_returns_registered_backends(self, client: Client):
        result = _data(await client.call_tool("list_backends", {}))
        assert isinstance(result, list)
        assert len(result) >= 1
        backend_info = result[0]
        assert "name" in backend_info
        assert "binary" in backend_info
        assert "available" in backend_info
        assert "defaultModel" in backend_info
        assert "supportedModels" in backend_info

    async def test_returns_correct_backend_name(self, client: Client):
        result = _data(await client.call_tool("list_backends", {}))
        names = [backend["name"] for backend in result]
        assert "claude-code" in names

    async def test_returns_empty_when_no_backends(self, client: Client):
        # Exercise the real discovery path: reset the loaded flag, stub out
        # both binary-on-PATH lookup and entry-point plugin discovery, then
        # let the registry run its actual code. Previously the dict-mutation
        # short-circuited all of this — a regression that broke discovery
        # while leaving _backends empty in tests would still have passed.
        registry._loaded = False
        registry._backends = {}
        with (
            patch("claude_teams.backends.base.shutil.which", return_value=None),
            patch("importlib.metadata.entry_points", return_value=[]),
        ):
            result = _data(await client.call_tool("list_backends", {}))
        assert result == []

    async def test_reports_backend_availability_from_backend_check(
        self, client: Client
    ):
        mock_backend = _make_mock_backend("claude-code")
        mock_backend.is_available.return_value = False
        registry._backends = {"claude-code": mock_backend}

        result = _data(await client.call_tool("list_backends", {}))

        assert result[0]["available"] is False


class TestListAgents:
    async def test_returns_unsupported_when_backend_has_no_spec(self, client: Client):
        result = _data(
            await client.call_tool("list_agents", {"backend_name": "claude-code"})
        )

        assert result["backend"] == "claude-code"
        assert result["supported"] is False
        assert result["profiles"] == []
        assert result["cwd"]

    async def test_returns_profiles_when_backend_supports_selection(
        self, client: Client
    ):
        mock = _make_mock_backend("claude-code")
        mock.agent_select_spec.return_value = AgentSelectSpec(
            flag="--agent", value_template="{name}"
        )
        mock.discover_agents.return_value = [
            AgentProfile(name="reviewer", path="/abs/reviewer.md")
        ]
        registry._backends = {"claude-code": mock}

        result = _data(
            await client.call_tool("list_agents", {"backend_name": "claude-code"})
        )

        assert result["supported"] is True
        assert result["profiles"] == [{"name": "reviewer", "path": "/abs/reviewer.md"}]

    async def test_returns_empty_profiles_when_supported_but_none_discovered(
        self, client: Client
    ):
        mock = _make_mock_backend("claude-code")
        mock.agent_select_spec.return_value = AgentSelectSpec(
            flag="--agent", value_template="{name}"
        )
        mock.discover_agents.return_value = []
        registry._backends = {"claude-code": mock}

        result = _data(
            await client.call_tool("list_agents", {"backend_name": "claude-code"})
        )

        assert result["supported"] is True
        assert result["profiles"] == []

    async def test_rejects_unknown_backend(self, client: Client):
        result = await client.call_tool(
            "list_agents",
            {"backend_name": "not-a-backend"},
            raise_on_error=False,
        )

        assert result.is_error is True

    async def test_passes_explicit_cwd_to_discover(
        self, client: Client, tmp_path: Path
    ):
        mock = _make_mock_backend("claude-code")
        mock.agent_select_spec.return_value = AgentSelectSpec(
            flag="--agent", value_template="{name}"
        )
        # Return a non-empty list so we can verify profiles flow through
        # the whole pipeline — prior version only checked call args, which
        # would pass even if the tool dropped the profiles on the floor.
        mock.discover_agents.return_value = [
            AgentProfile(name="scoped-a", path=str(tmp_path / "scoped-a.md")),
            AgentProfile(name="scoped-b", path=str(tmp_path / "scoped-b.md")),
        ]
        registry._backends = {"claude-code": mock}

        result = _data(
            await client.call_tool(
                "list_agents",
                {"backend_name": "claude-code", "cwd": str(tmp_path)},
            )
        )

        mock.discover_agents.assert_called_once_with(str(tmp_path))
        assert result["cwd"] == str(tmp_path)
        assert result["supported"] is True
        assert result["profiles"] == [
            {"name": "scoped-a", "path": str(tmp_path / "scoped-a.md")},
            {"name": "scoped-b", "path": str(tmp_path / "scoped-b.md")},
        ]
