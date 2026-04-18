import asyncio
import json
import time
import unittest.mock
from pathlib import Path

import pytest

from claude_teams import capabilities
from claude_teams.models import LeadMember, TeammateMember
from claude_teams.teams import (
    add_member,
    create_team,
    delete_team,
    read_config,
    remove_member,
    team_exists,
    write_config,
)


def _make_teammate(name: str, team_name: str, cwd: str | Path) -> TeammateMember:
    """Construct a ``TeammateMember`` fixture with an explicit ``cwd``.

    ``cwd`` is required so callers thread a pytest ``tmp_path``-rooted
    directory through and each test stays filesystem-isolated.
    """
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="teammate",
        model="claude-sonnet-4-20250514",
        prompt="Do stuff",
        color="blue",
        plan_mode_required=False,
        joined_at=int(time.time() * 1000),
        tmux_pane_id="%1",
        cwd=str(cwd),
    )


class TestCreateTeam:
    async def test_create_team_produces_correct_directory_structure(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("alpha", "sess-1", base_dir=tmp_claude_dir)

        assert await asyncio.to_thread((tmp_claude_dir / "teams" / "alpha").is_dir)
        assert await asyncio.to_thread((tmp_claude_dir / "tasks" / "alpha").is_dir)
        assert await asyncio.to_thread(
            (tmp_claude_dir / "tasks" / "alpha" / ".lock").exists
        )
        assert not await asyncio.to_thread(
            (tmp_claude_dir / "teams" / "alpha" / "inboxes").exists
        )

    async def test_create_team_config_has_correct_schema(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team(
            "beta",
            "sess-42",
            description="test team",
            base_dir=tmp_claude_dir,
        )

        raw = json.loads(
            await asyncio.to_thread(
                (tmp_claude_dir / "teams" / "beta" / "config.json").read_text
            )
        )

        assert raw["name"] == "beta"
        assert raw["description"] == "test team"
        assert raw["leadSessionId"] == "sess-42"
        assert raw["leadAgentId"] == "team-lead@beta"
        assert "createdAt" in raw
        assert isinstance(raw["createdAt"], int)
        assert isinstance(raw["members"], list)
        assert len(raw["members"]) == 1

    async def test_create_team_lead_member_shape(self, tmp_claude_dir: Path) -> None:
        await create_team("gamma", "sess-7", base_dir=tmp_claude_dir)

        raw = json.loads(
            await asyncio.to_thread(
                (tmp_claude_dir / "teams" / "gamma" / "config.json").read_text
            )
        )
        lead = raw["members"][0]

        assert lead["agentId"] == "team-lead@gamma"
        assert lead["name"] == "team-lead"
        assert lead["agentType"] == "team-lead"
        assert lead["tmuxPaneId"] == ""
        assert lead["subscriptions"] == []

    async def test_create_team_rejects_invalid_names(
        self, tmp_claude_dir: Path
    ) -> None:
        for bad_name in ["has space", "has.dot", "has/slash", "has\\back"]:
            with pytest.raises(ValueError, match="Invalid team name"):
                await create_team(bad_name, "sess-x", base_dir=tmp_claude_dir)

    async def test_should_reject_name_exceeding_max_length(
        self, tmp_claude_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="too long"):
            await create_team("a" * 65, "sess-x", base_dir=tmp_claude_dir)

    async def test_should_accept_name_at_max_length(self, tmp_claude_dir: Path) -> None:
        result = await create_team("a" * 64, "sess-x", base_dir=tmp_claude_dir)
        assert result.team_name == "a" * 64


class TestDeleteTeam:
    async def test_delete_team_removes_directories(self, tmp_claude_dir: Path) -> None:
        await create_team("doomed", "sess-1", base_dir=tmp_claude_dir)
        result = await delete_team("doomed", base_dir=tmp_claude_dir)

        assert result.success is True
        assert result.team_name == "doomed"
        assert not await asyncio.to_thread((tmp_claude_dir / "teams" / "doomed").exists)
        assert not await asyncio.to_thread((tmp_claude_dir / "tasks" / "doomed").exists)

    async def test_delete_team_fails_with_active_members(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("busy", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("worker", "busy", tmp_claude_dir)
        await add_member("busy", mate, base_dir=tmp_claude_dir)

        with pytest.raises(RuntimeError):
            await delete_team("busy", base_dir=tmp_claude_dir)


class TestMembers:
    async def test_add_member_appends_to_config(self, tmp_claude_dir: Path) -> None:
        await create_team("squad", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("coder", "squad", tmp_claude_dir)
        await add_member("squad", mate, base_dir=tmp_claude_dir)

        cfg = await read_config("squad", base_dir=tmp_claude_dir)
        assert len(cfg.members) == 2
        assert cfg.members[1].name == "coder"

    async def test_remove_member_filters_from_config(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("squad2", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("temp", "squad2", tmp_claude_dir)
        await add_member("squad2", mate, base_dir=tmp_claude_dir)
        await remove_member("squad2", "temp", base_dir=tmp_claude_dir)

        cfg = await read_config("squad2", base_dir=tmp_claude_dir)
        assert len(cfg.members) == 1
        assert cfg.members[0].name == "team-lead"


class TestDuplicateMember:
    async def test_should_reject_duplicate_member_name(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("dup", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("worker", "dup", tmp_claude_dir)
        await add_member("dup", mate, base_dir=tmp_claude_dir)
        mate2 = _make_teammate("worker", "dup", tmp_claude_dir)
        with pytest.raises(ValueError, match="already exists"):
            await add_member("dup", mate2, base_dir=tmp_claude_dir)

    async def test_should_allow_member_after_removal(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("reuse", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("worker", "reuse", tmp_claude_dir)
        await add_member("reuse", mate, base_dir=tmp_claude_dir)
        await remove_member("reuse", "worker", base_dir=tmp_claude_dir)
        mate2 = _make_teammate("worker", "reuse", tmp_claude_dir)
        await add_member("reuse", mate2, base_dir=tmp_claude_dir)
        cfg = await read_config("reuse", base_dir=tmp_claude_dir)
        assert any(member.name == "worker" for member in cfg.members)


class TestWriteConfig:
    async def test_should_cleanup_temp_file_when_replace_fails(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("atomic", "sess-1", base_dir=tmp_claude_dir)
        config = await read_config("atomic", base_dir=tmp_claude_dir)
        config.description = "updated"

        config_dir = tmp_claude_dir / "teams" / "atomic"

        with unittest.mock.patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                await write_config("atomic", config, base_dir=tmp_claude_dir)

        tmp_files = await asyncio.to_thread(lambda: list(config_dir.glob("*.tmp")))
        assert tmp_files == [], f"Leaked temp files: {tmp_files}"

    async def test_should_reject_invalid_team_name(self, tmp_claude_dir: Path) -> None:
        await create_team("atomic2", "sess-1", base_dir=tmp_claude_dir)
        config = await read_config("atomic2", base_dir=tmp_claude_dir)

        with pytest.raises(ValueError, match="Invalid team name"):
            await write_config("../bad", config, base_dir=tmp_claude_dir)


class TestTeamExists:
    async def test_should_return_true_for_existing_team(
        self, tmp_claude_dir: Path
    ) -> None:

        await create_team("exists", "sess-1", base_dir=tmp_claude_dir)
        assert await team_exists("exists", base_dir=tmp_claude_dir) is True

    async def test_should_return_false_for_nonexistent_team(
        self, tmp_claude_dir: Path
    ) -> None:

        assert await team_exists("ghost", base_dir=tmp_claude_dir) is False

    async def test_should_reject_invalid_team_name(self, tmp_claude_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid team name"):
            await team_exists("../ghost", base_dir=tmp_claude_dir)


class TestRemoveMemberGuard:
    async def test_should_reject_removing_team_lead(self, tmp_claude_dir: Path) -> None:
        await create_team("guarded", "sess-1", base_dir=tmp_claude_dir)
        with pytest.raises(ValueError, match="Cannot remove team-lead"):
            await remove_member("guarded", "team-lead", base_dir=tmp_claude_dir)

    async def test_should_allow_removing_non_lead_member(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("ok-rm", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("temp", "ok-rm", tmp_claude_dir)
        await add_member("ok-rm", mate, base_dir=tmp_claude_dir)
        await remove_member("ok-rm", "temp", base_dir=tmp_claude_dir)
        cfg = await read_config("ok-rm", base_dir=tmp_claude_dir)
        assert len(cfg.members) == 1


class TestReadConfig:
    async def test_read_config_round_trip(self, tmp_claude_dir: Path) -> None:
        await create_team(
            "roundtrip", "sess-99", description="rt test", base_dir=tmp_claude_dir
        )
        cfg = await read_config("roundtrip", base_dir=tmp_claude_dir)

        assert cfg.name == "roundtrip"
        assert cfg.description == "rt test"
        assert cfg.lead_session_id == "sess-99"
        assert cfg.lead_agent_id == "team-lead@roundtrip"
        assert len(cfg.members) == 1
        lead = cfg.members[0]
        assert isinstance(lead, LeadMember)
        assert lead.agent_id == "team-lead@roundtrip"

    async def test_read_config_rejects_invalid_team_name(
        self, tmp_claude_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="Invalid team name"):
            await read_config("../bad", base_dir=tmp_claude_dir)


class TestDeleteTeamValidation:
    async def test_delete_team_rejects_invalid_name(self, tmp_claude_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid team name"):
            await delete_team("../ghost", base_dir=tmp_claude_dir)


class TestCapabilitiesValidation:
    async def test_initialize_team_capabilities_rejects_invalid_team_name(
        self, tmp_claude_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="Invalid team name"):
            await capabilities.initialize_team_capabilities(
                "../ghost", base_dir=tmp_claude_dir
            )

    async def test_issue_agent_capability_rejects_invalid_agent_name(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("cap-team", "sess-1", base_dir=tmp_claude_dir)
        await capabilities.initialize_team_capabilities(
            "cap-team", base_dir=tmp_claude_dir
        )

        with pytest.raises(ValueError, match="Invalid agent name"):
            await capabilities.issue_agent_capability(
                "cap-team",
                "../bob",
                base_dir=tmp_claude_dir,
            )

    async def test_remove_agent_capability_rejects_invalid_agent_name(
        self, tmp_claude_dir: Path
    ) -> None:
        await create_team("cap-team-rm", "sess-1", base_dir=tmp_claude_dir)
        await capabilities.initialize_team_capabilities(
            "cap-team-rm", base_dir=tmp_claude_dir
        )

        with pytest.raises(ValueError, match="Invalid agent name"):
            await capabilities.remove_agent_capability(
                "cap-team-rm",
                "../bob",
                base_dir=tmp_claude_dir,
            )
