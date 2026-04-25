"""Bootstrap and capability server tests."""

import logging
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client

from claude_teams import presets, teams, templates
from claude_teams.backends import registry
from claude_teams.backends.base import AgentProfile, AgentSelectSpec, HealthStatus
from claude_teams.backends.base import SpawnResult as BackendSpawnResult
from claude_teams.errors import (
    BackendSpawnFailedError,
    PresetMemberSpawnFailedError,
    TeamAlreadyExistsError,
)
from claude_teams.models import TeammateMember
from claude_teams.orchestration import expand_preset_core
from claude_teams.presets import PresetMemberSpec, TeamPreset
from claude_teams.server import mcp
from claude_teams.server_team_spawn import _build_spawn_dependencies
from claude_teams.templates import AgentTemplate
from tests._server_support import (
    _data,
    _extract_capability,
    _items,
    _make_mock_backend,
    _text,
)


class TestStaticToolDiscovery:
    async def test_all_tools_visible_at_startup(self, client: Client):
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "team_create" in names
        assert "team_attach" in names
        assert "team_delete" in names
        assert "list_backends" in names
        assert "list_agents" in names
        assert "read_config" in names
        assert "spawn_teammate" in names
        assert "send_message" in names
        assert "task_create" in names
        assert "task_update" in names
        assert "task_list" in names
        assert "task_get" in names
        assert "read_inbox" in names
        assert "force_kill_teammate" in names
        assert "poll_inbox" in names
        assert "check_teammate" in names
        assert "process_shutdown_approved" in names
        assert "health_check" in names
        assert "get_agent_logs" in names

    async def test_team_tools_reject_without_authenticated_session(
        self, client: Client
    ):
        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "missing-auth",
                "name": "coder",
                "prompt": "write code",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_tools_remain_visible_after_delete(self, client: Client):
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
        assert "team_create" in names
        assert "list_backends" in names
        assert "spawn_teammate" in names
        assert "force_kill_teammate" in names

    async def test_recreate_cycle_keeps_tools_visible(self, client: Client):
        # Create -> delete -> re-create cycle
        await client.call_tool("team_create", {"team_name": "cycle1"})
        await client.call_tool("team_delete", {"team_name": "cycle1"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "spawn_teammate" in names
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


class TestListTemplates:
    """Tier-0 ``list_templates`` visibility and payload shape.

    Tier-0 placement matters: callers need to see available templates
    before they have a team to attach to, so the tool must appear in the
    pre-create tool list. Payload shape matters for SDK clients that type
    their parsing against the documented fields.
    """

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        templates._seed_builtin_templates()

    async def test_list_templates_is_visible_at_tier_zero(self, client: Client):
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "list_templates" in names

    async def test_returns_seeded_builtins(self, client: Client):
        result = _data(await client.call_tool("list_templates", {}))
        assert isinstance(result, list)
        names = {entry["name"] for entry in result}
        # The five shipped built-ins must always be present.
        assert {
            "code-reviewer",
            "debugger",
            "executor",
            "test-engineer",
            "writer",
        } <= names

    async def test_entries_carry_expected_fields(self, client: Client):
        result = _data(await client.call_tool("list_templates", {}))
        reviewer = next(e for e in result if e["name"] == "code-reviewer")
        # camelCase alias — matches every other Tier-0 payload.
        assert "description" in reviewer
        assert reviewer["rolePrompt"].startswith("You are acting as a code reviewer.")
        assert reviewer["defaultSubagentType"] == "code-reviewer"
        # Optional defaults are null when unset, not missing — keeps the
        # field shape stable for typed clients.
        assert reviewer["defaultBackend"] is None

    async def test_returns_sorted_by_name(self, client: Client):
        result = _data(await client.call_tool("list_templates", {}))
        names = [entry["name"] for entry in result]
        assert names == sorted(names)

    async def test_custom_registered_template_is_visible(self, client: Client):
        templates.register_template(
            AgentTemplate(
                name="custom-role",
                description="Ad-hoc registration for tests.",
                role_prompt="Custom header.",
                default_backend="claude-code",
            )
        )
        result = _data(await client.call_tool("list_templates", {}))
        custom = next(e for e in result if e["name"] == "custom-role")
        assert custom["rolePrompt"] == "Custom header."
        assert custom["defaultBackend"] == "claude-code"


class TestListPresets:
    """Tier-0 ``list_presets`` visibility and payload shape."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        presets._seed_builtin_presets()

    async def test_list_presets_is_visible_at_tier_zero(self, client: Client):
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "list_presets" in names

    async def test_returns_seeded_builtins(self, client: Client):
        result = _data(await client.call_tool("list_presets", {}))
        names = {entry["name"] for entry in result}
        assert {"review-and-fix", "docs-pair"} <= names

    async def test_entries_carry_member_details(self, client: Client):
        result = _data(await client.call_tool("list_presets", {}))
        review = next(e for e in result if e["name"] == "review-and-fix")
        assert review["description"]
        assert review["teamDescription"]
        assert len(review["members"]) == 2
        reviewer = next(m for m in review["members"] if m["name"] == "reviewer")
        assert reviewer["template"] == "code-reviewer"
        assert reviewer["prompt"]


class TestCreateTeamFromPreset:
    """Preset expansion through the ``team_create`` + ``spawn_teammate`` path."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        presets._seed_builtin_presets()
        templates._seed_builtin_templates()

    async def test_expands_preset_into_team_and_members(self, client: Client):
        result = _data(
            await client.call_tool(
                "create_team_from_preset",
                {
                    "preset_name": "review-and-fix",
                    "team_name": "preset-team-1",
                },
            )
        )
        assert result["preset"] == "review-and-fix"
        # TeamCreateResult and SpawnResult don't use camelCase aliases,
        # so nested payloads stay snake_case.
        assert result["team"]["team_name"] == "preset-team-1"
        member_names = [m["name"] for m in result["members"]]
        assert member_names == ["reviewer", "executor"]

        cfg = await teams.read_config("preset-team-1")
        teammates = {m.name: m for m in cfg.members if isinstance(m, TeammateMember)}
        # Template default subagent_types came through the expansion.
        assert teammates["reviewer"].agent_type == "code-reviewer"
        assert teammates["executor"].agent_type == "executor"

    async def test_applies_template_role_prompt_through_expansion(self, client: Client):
        mock_backend = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "review-and-fix",
                "team_name": "preset-team-2",
            },
        )

        # Inspect the last spawn (executor) — should carry executor role
        # prompt via its template, composed with the preset's per-member
        # task prompt.
        spawn_calls = mock_backend.spawn.call_args_list
        assert len(spawn_calls) == 2
        executor_request = spawn_calls[1].args[0]
        assert executor_request.prompt.startswith("You are acting as an executor.")
        assert "Wait for review findings" in executor_request.prompt

    async def test_explicit_description_overrides_preset_default(self, client: Client):
        await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "review-and-fix",
                "team_name": "preset-team-3",
                "description": "Custom override description.",
            },
        )
        cfg = await teams.read_config("preset-team-3")
        assert cfg.description == "Custom override description."

    async def test_falls_back_to_preset_team_description(self, client: Client):
        await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "review-and-fix",
                "team_name": "preset-team-4",
            },
        )
        cfg = await teams.read_config("preset-team-4")
        assert "Review-and-fix" in cfg.description

    async def test_unknown_preset_raises_structured_error(self, client: Client):
        result = await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "does-not-exist",
                "team_name": "preset-bad",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        text = _text(result)
        assert "does-not-exist" in text
        # Error surface enumerates available presets.
        assert "review-and-fix" in text

    async def test_collision_with_existing_team_surfaces_tool_error(
        self, client: Client
    ):
        """MCP wrapper translates ``TeamAlreadyExistsError`` into the tool error.

        Pre-seeds a team on disk by calling ``teams.create_team`` directly
        (bypassing the MCP session's ``active_team`` bookkeeping), then
        drives ``create_team_from_preset`` with the same name. The domain
        error raised inside ``expand_preset_core`` must surface as the
        discoverable ``TeamAlreadyExistsToolError`` payload — the error
        message includes the colliding name and the remediation path so
        clients can branch on the typed surface rather than parsing a
        generic failure.
        """
        await teams.create_team(
            name="taken-name",
            session_id="other-session",
            description="pre-existing team",
        )

        result = await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "review-and-fix",
                "team_name": "taken-name",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        text = _text(result)
        assert "taken-name" in text
        assert "already exists" in text

        # Pre-existing team survives the failed expansion attempt — the
        # core docstring promises no rollback of a team we didn't create.
        cfg = await teams.read_config("taken-name")
        assert cfg.description == "pre-existing team"

    async def test_custom_preset_with_explicit_member_overrides(self, client: Client):
        presets.register_preset(
            TeamPreset(
                name="custom-duo",
                description="Custom duo for tests.",
                members=(
                    PresetMemberSpec(
                        name="alpha",
                        prompt="do alpha work",
                        template="code-reviewer",
                        # Explicit subagent_type wins over template default.
                        subagent_type="executor",
                    ),
                ),
            )
        )
        await client.call_tool(
            "create_team_from_preset",
            {
                "preset_name": "custom-duo",
                "team_name": "preset-custom",
            },
        )
        cfg = await teams.read_config("preset-custom")
        alpha = next(
            m
            for m in cfg.members
            if isinstance(m, TeammateMember) and m.name == "alpha"
        )
        # Explicit beats template default.
        assert alpha.agent_type == "executor"

    async def test_member_spawn_failure_persists_team_and_earlier_members(
        self, client: Client
    ):
        """Mid-fan-out failure leaves team and already-spawned members intact.

        Locks the non-transactional expansion contract at the MCP boundary:
        the call returns an error naming the failing member, and a follow-up
        ``read_config`` still sees the team plus every member spawned before
        the failure. Without this guarantee a partial expansion would either
        leak a hidden team or roll back successful work without surfacing it.
        """
        mock = cast(MagicMock, registry._backends["claude-code"])
        # side_effect as an iterable runs one entry per call. First member
        # spawns cleanly, second raises — exercising the mid-fan-out branch.
        mock.spawn.side_effect = [
            BackendSpawnResult(process_handle="%ok", backend_type="claude-code"),
            RuntimeError("synthetic spawn failure for second member"),
        ]

        presets.register_preset(
            TeamPreset(
                name="failure-duo",
                description="Two-member preset; second member crashes on spawn.",
                team_description="Non-transactional expansion contract.",
                members=(
                    PresetMemberSpec(
                        name="ok-one", prompt="succeeds", template="executor"
                    ),
                    PresetMemberSpec(name="boom", prompt="fails", template="executor"),
                ),
            )
        )

        result = await client.call_tool(
            "create_team_from_preset",
            {"preset_name": "failure-duo", "team_name": "fail-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        # Error names the failing member so the operator knows exactly which
        # one to retry via ``spawn_teammate``.
        assert "boom" in _text(result)

        # Non-transactional contract: team persists after the failure.
        cfg = await teams.read_config("fail-team")
        names = {m.name for m in cfg.members}
        assert "team-lead" in names
        # First member succeeded — stays persisted.
        assert "ok-one" in names
        # Failing member was rolled back inside ``spawn_teammate_core``'s
        # BackendSpawnFailedError branch; only the preset-level wrap surfaces
        # to the caller, but the partially-added member record is gone.
        assert "boom" not in names

    async def test_member_spawn_failure_keeps_session_lead_attached(
        self, client: Client
    ):
        """Partial failure still attaches the session as lead.

        Locks the ``on_capability_minted`` early-attach contract: the MCP
        wrapper attaches the session principal before member fan-out so the
        caller can still reach lead-gated tools on the partially-created
        team when a later member spawn fails mid-expansion. Without the
        early attach, any follow-up on the failed team (``read_config``,
        ``team_delete``, ``spawn_teammate``) would hit
        ``AuthenticationRequiredError`` and strand the caller with a team
        they can neither use nor clean up.

        ``read_config`` is the cleanest probe because it requires a lead
        principal but makes no state changes — a successful call proves
        session auth survived the failure.
        """
        mock = cast(MagicMock, registry._backends["claude-code"])
        mock.spawn.side_effect = [
            BackendSpawnResult(process_handle="%ok", backend_type="claude-code"),
            RuntimeError("synthetic spawn failure"),
        ]

        presets.register_preset(
            TeamPreset(
                name="attach-duo",
                description="Second member fails; exercises early-attach.",
                team_description="Covers on_capability_minted attach timing.",
                members=(
                    PresetMemberSpec(
                        name="ok-one", prompt="succeeds", template="executor"
                    ),
                    PresetMemberSpec(name="boom", prompt="fails", template="executor"),
                ),
            )
        )

        failure = await client.call_tool(
            "create_team_from_preset",
            {"preset_name": "attach-duo", "team_name": "attach-team"},
            raise_on_error=False,
        )
        assert failure.is_error is True

        # No explicit capability — ``_require_lead`` must resolve the
        # principal from session state or this call raises auth error.
        cfg = await client.call_tool(
            "read_config",
            {"team_name": "attach-team"},
            raise_on_error=False,
        )
        assert cfg.is_error is False


class TestExpandPresetCoreFailure:
    """Core-level expansion contract: failure path without the MCP wrapper.

    Exercises ``expand_preset_core`` directly so a future refactor that
    swaps in transactional rollback fails here explicitly rather than
    changing observable behaviour behind the MCP boundary. The companion
    MCP-level test in ``TestCreateTeamFromPreset`` pins the same contract
    from the client side.
    """

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        presets._seed_builtin_presets()
        templates._seed_builtin_templates()

    async def test_partial_failure_raises_named_error_and_persists_team(
        self, client: Client
    ):
        """Direct ``expand_preset_core`` call: typed error + team persistence."""
        # The ``client`` fixture is depended on for its side effects only
        # (monkeypatched ``TEAMS_DIR``/``TASKS_DIR`` and the mocked backend
        # registry). The test drives the orchestration core directly instead
        # of going through ``client.call_tool``, which is what makes this
        # the core-layer contract test.
        _ = client

        mock = cast(MagicMock, registry._backends["claude-code"])
        mock.spawn.side_effect = [
            BackendSpawnResult(process_handle="%ok", backend_type="claude-code"),
            RuntimeError("boom"),
        ]

        presets.register_preset(
            TeamPreset(
                name="core-failure-duo",
                description="Core-level partial failure.",
                team_description="Non-transactional core contract.",
                members=(
                    PresetMemberSpec(name="first", prompt="ok", template="executor"),
                    PresetMemberSpec(
                        name="second", prompt="crash", template="executor"
                    ),
                ),
            )
        )
        preset = presets.get_preset("core-failure-duo")

        with pytest.raises(PresetMemberSpawnFailedError) as exc_info:
            await expand_preset_core(
                registry=registry,
                preset=preset,
                team_name="core-fail-team",
                session_id="test-session",
                description="desc",
                deps=_build_spawn_dependencies(),
            )

        # Typed error names the failing member for targeted retry.
        assert "second" in str(exc_info.value)
        # Direct cause is the spawn-layer wrap; its cause is the backend's
        # original exception. Both links must survive so callers can branch
        # on either "preset fan-out failed" (outer) or the raw failure mode
        # (root) without losing the wrapper context.
        outer_cause = exc_info.value.__cause__
        assert isinstance(outer_cause, BackendSpawnFailedError)
        assert isinstance(outer_cause.__cause__, RuntimeError)
        assert "boom" in str(outer_cause.__cause__)

        # Team persists with the successful member; the failed one was
        # rolled back at the spawn-core layer before the preset wrap.
        cfg = await teams.read_config("core-fail-team")
        names = {m.name for m in cfg.members}
        assert "team-lead" in names
        assert "first" in names
        assert "second" not in names

    async def test_team_already_exists_surfaces_raw_domain_error(self, client: Client):
        """Core raises ``TeamAlreadyExistsError`` unwrapped on name collision.

        The docstring contract: the collision surfaces directly (no preset
        wrap) because expansion never started — no partial team state to
        report. This test also locks the no-rollback property: the
        pre-seeded team survives the failed expansion because the core
        never owned it.
        """
        # ``client`` fixture is depended on for its side effects only
        # (monkeypatched ``TEAMS_DIR``/``TASKS_DIR`` and the mocked backend
        # registry). The test drives ``expand_preset_core`` directly so a
        # future refactor that swaps in a wrapped surface fails here
        # explicitly rather than drifting the MCP contract silently.
        _ = client

        await teams.create_team(
            name="collision-team",
            session_id="other-session",
            description="pre-existing team",
        )

        preset = presets.get_preset("review-and-fix")

        with pytest.raises(TeamAlreadyExistsError) as exc_info:
            await expand_preset_core(
                registry=registry,
                preset=preset,
                team_name="collision-team",
                session_id="test-session",
                description="from preset",
                deps=_build_spawn_dependencies(),
            )
        assert "collision-team" in str(exc_info.value)

        # Pre-existing team persists — no rollback of a team we never
        # owned. Re-reading its description proves the original config is
        # still on disk untouched.
        cfg = await teams.read_config("collision-team")
        assert cfg.description == "pre-existing team"

    async def test_on_capability_minted_receives_lead_capability(self, client: Client):
        """Callback fires once, before fan-out, with the token in the result.

        Locks the hook's data contract: the MCP wrapper needs the lead
        capability minted during expansion to attach the session as lead
        before member fan-out starts. If the callback ever stops being
        called, or is called with a different token than what's returned
        on ``PresetExpansionResult``, the session-attach path silently
        desynchronises from the returned capability.
        """
        _ = client

        presets.register_preset(
            TeamPreset(
                name="hook-solo",
                description="Single-member preset to exercise the hook.",
                team_description="One member that spawns cleanly.",
                members=(
                    PresetMemberSpec(name="only", prompt="runs", template="executor"),
                ),
            )
        )
        preset = presets.get_preset("hook-solo")

        minted: list[str] = []

        async def spy(capability: str) -> None:
            minted.append(capability)

        result = await expand_preset_core(
            registry=registry,
            preset=preset,
            team_name="hook-team",
            session_id="test-session",
            description="desc",
            deps=_build_spawn_dependencies(),
            on_capability_minted=spy,
        )

        # Exactly one mint per expansion, and it matches the token the
        # caller receives — no drift between what's attached to the
        # session and what ``PresetExpansionResult`` advertises.
        assert minted == [result.lead_capability]

    async def test_on_capability_minted_failure_rolls_back_team(self, client: Client):
        """A raising hook rolls the team back so retries stay idempotent.

        Exercises the ``except Exception`` branch in the setup block of
        ``expand_preset_core``: the team is already on disk when the
        callback fires, so a hook failure without rollback would leave a
        dangling config that blocks ``preset launch`` retries with
        ``TeamAlreadyExistsError`` and no way to recover from the session
        that could not attach.
        """
        _ = client

        presets.register_preset(
            TeamPreset(
                name="hook-raise",
                description="Preset used only to verify rollback.",
                team_description="Rollback test; members never spawn.",
                members=(
                    PresetMemberSpec(
                        name="never-spawns", prompt="x", template="executor"
                    ),
                ),
            )
        )
        preset = presets.get_preset("hook-raise")

        class _HookFailureError(RuntimeError):
            pass

        async def boom(_capability: str) -> None:
            raise _HookFailureError

        with pytest.raises(_HookFailureError):
            await expand_preset_core(
                registry=registry,
                preset=preset,
                team_name="hook-raise-team",
                session_id="test-session",
                description="desc",
                deps=_build_spawn_dependencies(),
                on_capability_minted=boom,
            )

        # Team rolled back — retrying the same name must not hit
        # ``TeamAlreadyExistsError`` from leftover on-disk state.
        assert not await teams.team_exists("hook-raise-team")


class TestRetainPaneFailureWiring:
    """``retain_pane_after_exit`` errors log a warning without failing the spawn.

    The orchestration core catches exceptions from
    ``Backend.retain_pane_after_exit`` and hands them to
    ``log_retain_pane_failure`` via ``SpawnDependencies``. The spawn has
    already succeeded by that point — pane retention is an operational
    breadcrumb, not a reason to fail the user-facing call. This test
    pins the wiring end-to-end: a non-interactive backend whose
    ``retain_pane_after_exit`` raises still produces a successful
    ``spawn_teammate`` return, and the warning lands in the shared
    ``claude_teams.server_runtime`` logger.
    """

    async def test_retain_pane_failure_is_logged_without_failing_spawn(
        self, client: Client, caplog: pytest.LogCaptureFixture
    ):
        # A non-interactive backend is required: orchestration only calls
        # ``retain_pane_after_exit`` on the one-shot (non-interactive) path.
        # ``_make_mock_backend`` sets ``is_interactive`` from the name, so
        # any non-``claude-code`` name flips the flag off.
        mock_codex = _make_mock_backend("codex")
        mock_codex.retain_pane_after_exit.side_effect = RuntimeError("pane retain boom")
        # Background relay task needs a "pane ended" signal to exit
        # quickly; without this it would sit in its poll loop until
        # pytest-asyncio cancels it at teardown.
        mock_codex.health_check.return_value = HealthStatus(
            alive=False, detail="one-shot complete"
        )
        registry._backends["codex"] = mock_codex

        await client.call_tool("team_create", {"team_name": "retain-wire"})

        with caplog.at_level(logging.WARNING, logger="claude_teams.server_runtime"):
            result = await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "retain-wire",
                    "name": "codex-worker",
                    "prompt": "do something",
                    "options": {"backend": "codex", "model": "gpt-5.3-codex"},
                },
                raise_on_error=False,
            )

        # Spawn succeeds — a failure here would mean the exception bubbled
        # past the ``except Exception`` guard in orchestration, which is
        # the exact regression this test exists to prevent.
        assert result.is_error is False
        # The mock's retain hook was actually invoked by orchestration;
        # rules out the "callback logged for some other reason" failure
        # mode before asserting on the log contents.
        assert mock_codex.retain_pane_after_exit.call_count == 1
        # Injected ``log_retain_pane_failure`` ran with the raised
        # exception. Substring match is narrow enough to distinguish the
        # retain breadcrumb from any background relay-task error.
        assert "retain_pane_after_exit failed" in caplog.text
        assert "pane retain boom" in caplog.text
