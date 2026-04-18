"""Task and team-lifecycle server tests."""

import time
from pathlib import Path

from fastmcp import Client

from claude_teams import teams
from tests._server_support import _data, _items, _make_teammate, _text


class TestWiring:
    async def test_should_round_trip_task_create_and_list(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t4"})
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "first", "description": "d1"},
        )
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "second", "description": "d2"},
        )
        result = _data(await client.call_tool("task_list", {"team_name": "t4"}))
        assert result["totalCount"] == 2
        assert len(result["items"]) == 2
        assert result["items"][0]["subject"] == "first"
        assert result["items"][1]["subject"] == "second"

    async def test_should_round_trip_send_message_and_read_inbox(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "t5"})
        await teams.add_member("t5", _make_teammate("bob", "t5", tmp_path))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t5",
                "message_type": "message",
                "recipient": "bob",
                "content": "hello bob",
                "summary": "greeting",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t5", "agent_name": "bob"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["text"] == "hello bob"
        assert inbox[0]["from"] == "team-lead"

    async def test_should_round_trip_teammate_message_to_team_lead(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "t5b"})
        await teams.add_member("t5b", _make_teammate("worker", "t5b", tmp_path))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t5b",
                "message_type": "message",
                "sender": "worker",
                "recipient": "team-lead",
                "content": "done",
                "summary": "status",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t5b", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["text"] == "done"
        assert inbox[0]["from"] == "worker"

    async def test_should_round_trip_teammate_message_to_teammate(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "t5c"})
        await teams.add_member("t5c", _make_teammate("alice", "t5c", tmp_path))
        await teams.add_member("t5c", _make_teammate("bob", "t5c", tmp_path))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t5c",
                "message_type": "message",
                "sender": "alice",
                "recipient": "bob",
                "content": "pair with me",
                "summary": "coordination",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t5c", "agent_name": "bob"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["text"] == "pair with me"
        assert inbox[0]["from"] == "alice"

    async def test_task_list_returns_pagination_metadata(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t5d"})
        for subject in ("one", "two", "three"):
            await client.call_tool(
                "task_create",
                {"team_name": "t5d", "subject": subject, "description": subject},
            )

        result = _data(
            await client.call_tool(
                "task_list",
                {"team_name": "t5d", "limit": 2, "offset": 0},
            )
        )

        assert result["totalCount"] == 3
        assert result["limit"] == 2
        assert result["offset"] == 0
        assert result["hasMore"] is True
        assert result["nextOffset"] == 2
        assert [item["subject"] for item in result["items"]] == ["one", "two"]

    async def test_read_inbox_returns_pagination_metadata(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "t5e"})
        await teams.add_member("t5e", _make_teammate("bob", "t5e", tmp_path))
        for text in ("msg-1", "msg-2", "msg-3"):
            await client.call_tool(
                "send_message",
                {
                    "team_name": "t5e",
                    "message_type": "message",
                    "recipient": "bob",
                    "content": text,
                    "summary": text,
                },
            )

        result = _data(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "t5e",
                    "agent_name": "bob",
                    "mark_as_read": False,
                    "limit": 2,
                    "offset": 0,
                },
            )
        )

        assert result["totalCount"] == 3
        assert result["limit"] == 2
        assert result["offset"] == 0
        assert result["hasMore"] is True
        assert result["nextOffset"] == 2
        assert [item["text"] for item in result["items"]] == ["msg-1", "msg-2"]

    async def test_read_inbox_supports_newest_order(
        self, client: Client, tmp_path: Path
    ):
        await client.call_tool("team_create", {"team_name": "t5f"})
        await teams.add_member("t5f", _make_teammate("bob", "t5f", tmp_path))
        for text in ("msg-1", "msg-2", "msg-3"):
            await client.call_tool(
                "send_message",
                {
                    "team_name": "t5f",
                    "message_type": "message",
                    "recipient": "bob",
                    "content": text,
                    "summary": text,
                },
            )

        result = _data(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "t5f",
                    "agent_name": "bob",
                    "mark_as_read": False,
                    "order": "newest",
                    "limit": 2,
                    "offset": 0,
                },
            )
        )

        assert [item["text"] for item in result["items"]] == ["msg-3", "msg-2"]

    async def test_read_inbox_rejects_invalid_agent_name(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t5g"})
        result = await client.call_tool(
            "read_inbox",
            {
                "team_name": "t5g",
                "agent_name": "../other-team/inboxes/bob",
            },
            raise_on_error=False,
        )

        assert result.is_error is True
        text = _text(result)
        # AgentName uses the same ``^[A-Za-z0-9_-]+$`` pattern as the storage
        # layer, so jsonschema rejects a path traversal attempt before reaching
        # any filesystem call.
        assert "'../other-team/inboxes/bob'" in text
        assert "^[A-Za-z0-9_-]+$" in text


class TestTeamDeleteClearsSession:
    async def test_should_allow_new_team_after_delete(self, client: Client):
        await client.call_tool("team_create", {"team_name": "first"})
        await client.call_tool("team_delete", {"team_name": "first"})
        result = await client.call_tool("team_create", {"team_name": "second"})
        data = _data(result)
        assert data["team_name"] == "second"

    async def test_team_create_returns_lead_capability(self, client: Client):
        result = _data(await client.call_tool("team_create", {"team_name": "cap-team"}))
        assert result["team_name"] == "cap-team"
        assert isinstance(result["lead_capability"], str)
        assert len(result["lead_capability"]) > 20


class TestErrorWrapping:
    async def test_read_config_wraps_file_not_found(self, client: Client):
        result = await client.call_tool(
            "read_config",
            {"team_name": "nonexistent"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_get_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew"})
        result = await client.call_tool(
            "task_get",
            {"team_name": "tew", "task_id": "999"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_update_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew2"})
        result = await client.call_tool(
            "task_update",
            {
                "team_name": "tew2",
                "task_id": "999",
                "fields": {"status": "completed"},
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_create_wraps_nonexistent_team(self, client: Client):
        # Create a team to unlock team-tier tools, then target a different team
        await client.call_tool("team_create", {"team_name": "real-team"})
        result = await client.call_tool(
            "task_create",
            {"team_name": "ghost-team", "subject": "x", "description": "y"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_update_wraps_validation_error(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew3"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "tew3", "subject": "S", "description": "d"},
            )
        )
        await client.call_tool(
            "task_update",
            {
                "team_name": "tew3",
                "task_id": created["id"],
                "fields": {"status": "in_progress"},
            },
        )
        result = await client.call_tool(
            "task_update",
            {
                "team_name": "tew3",
                "task_id": created["id"],
                "fields": {"status": "pending"},
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "cannot transition" in _text(result).lower()

    async def test_task_update_rejects_invalid_owner(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew4"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "tew4", "subject": "S", "description": "d"},
            )
        )

        result = await client.call_tool(
            "task_update",
            {
                "team_name": "tew4",
                "task_id": created["id"],
                "fields": {"owner": "../other-team/inboxes/bob"},
            },
            raise_on_error=False,
        )

        assert result.is_error is True
        text = _text(result)
        # ``owner`` lives inside ``TaskUpdateFields`` so wire-level jsonschema
        # pattern checks no longer fire; validation happens in the tasks layer
        # via ``validate_safe_name`` which surfaces as a ToolError.
        assert "'../other-team/inboxes/bob'" in text
        assert "Invalid owner" in text

    async def test_task_list_wraps_nonexistent_team(self, client: Client):
        # Create a team to unlock team-tier tools, then target a different team
        await client.call_tool("team_create", {"team_name": "real-team2"})
        result = await client.call_tool(
            "task_list",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()


class TestPollInbox:
    async def test_should_return_empty_on_timeout(self, team_client: Client):
        start = time.monotonic()
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "nobody", "timeout_ms": 100},
            )
        )
        elapsed = time.monotonic() - start
        assert result == []
        assert elapsed < 0.35

    async def test_should_return_messages_when_present(self, team_client: Client):
        await team_client.call_tool(
            "send_message",
            {
                "team_name": "test-team",
                "message_type": "message",
                "recipient": "worker",
                "content": "wake up",
                "summary": "nudge",
            },
        )
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "worker", "timeout_ms": 100},
            )
        )
        # worker already has the initial prompt message + new message
        assert any(msg["text"] == "wake up" for msg in result)

    async def test_should_return_existing_messages_with_zero_timeout(
        self, team_client: Client
    ):
        await team_client.call_tool(
            "send_message",
            {
                "team_name": "test-team",
                "message_type": "message",
                "recipient": "worker",
                "content": "instant",
                "summary": "fast",
            },
        )
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "worker", "timeout_ms": 0},
            )
        )
        assert any(msg["text"] == "instant" for msg in result)

    async def test_should_reject_invalid_agent_name(self, team_client: Client):
        result = await team_client.call_tool(
            "poll_inbox",
            {
                "team_name": "test-team",
                "agent_name": "../other-team/inboxes/bob",
                "timeout_ms": 100,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        text = _text(result)
        # jsonschema pattern validation short-circuits poll_inbox before it
        # touches the inbox filesystem.
        assert "'../other-team/inboxes/bob'" in text
        assert "^[A-Za-z0-9_-]+$" in text
