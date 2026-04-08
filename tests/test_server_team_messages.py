"""Team messaging and validation server tests."""

import json

from fastmcp import Client

from claude_teams import teams
from tests._server_support import _data, _items, _make_teammate, _text


class TestErrorPropagation:
    async def test_should_reject_second_team_in_same_session(self, client: Client):
        await client.call_tool("team_create", {"team_name": "alpha"})
        result = await client.call_tool(
            "team_create", {"team_name": "beta"}, raise_on_error=False
        )
        assert result.is_error is True
        assert "alpha" in _text(result)

    async def test_should_reject_unknown_agent_in_force_kill(self, team_client: Client):
        result = await team_client.call_tool(
            "force_kill_teammate",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_reject_invalid_message_type(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_msg"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "t_msg", "type": "bogus"},
            raise_on_error=False,
        )
        assert result.is_error is True


class TestDeletedTaskGuard:
    async def test_should_not_send_assignment_when_task_deleted(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t2"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2", "subject": "doomed", "description": "will delete"},
            )
        )
        await client.call_tool(
            "task_update",
            {
                "team_name": "t2",
                "task_id": created["id"],
                "status": "deleted",
                "owner": "worker",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t2", "agent_name": "worker"}
            )
        )
        assert inbox == []

    async def test_should_send_assignment_when_owner_set_on_live_task(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t2b"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2b", "subject": "live", "description": "stays"},
            )
        )
        await client.call_tool(
            "task_update",
            {"team_name": "t2b", "task_id": created["id"], "owner": "worker"},
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t2b", "agent_name": "worker"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "task_assignment"
        assert payload["taskId"] == created["id"]


class TestShutdownResponseSender:
    async def test_should_populate_correct_from_and_pane_id_on_approve(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t3"})
        await teams.add_member("t3", _make_teammate("worker", "t3", pane_id="%42"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3",
                "type": "shutdown_response",
                "sender": "worker",
                "request_id": "req-1",
                "approve": True,
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t3", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "shutdown_approved"
        assert payload["from"] == "worker"
        assert payload["paneId"] == "%42"
        assert payload["requestId"] == "req-1"

    async def test_should_attribute_rejection_to_sender(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t3b"})
        await teams.add_member("t3b", _make_teammate("rebel", "t3b"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3b",
                "type": "shutdown_response",
                "sender": "rebel",
                "request_id": "req-2",
                "approve": False,
                "content": "still busy",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t3b", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "rebel"
        assert inbox[0]["text"] == "still busy"


class TestPlanApprovalSender:
    async def test_should_use_sender_as_from_on_approve(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan"})
        await teams.add_member("t_plan", _make_teammate("dev", "t_plan"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev",
                "request_id": "plan-1",
                "approve": True,
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan", "agent_name": "dev"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "plan_approval"
        assert payload["approved"] is True

    async def test_should_use_sender_as_from_on_reject(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan2"})
        await teams.add_member("t_plan2", _make_teammate("dev2", "t_plan2"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan2",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev2",
                "approve": False,
                "content": "needs error handling",
            },
        )
        inbox = _items(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan2", "agent_name": "dev2"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        assert inbox[0]["text"] == "needs error handling"


class TestSendMessageValidation:
    async def test_should_reject_empty_content(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv1"})
        await teams.add_member("tv1", _make_teammate("bob", "tv1"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv1",
                "type": "message",
                "recipient": "bob",
                "content": "",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "content" in _text(result).lower()

    async def test_should_reject_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv2"})
        await teams.add_member("tv2", _make_teammate("bob", "tv2"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv2",
                "type": "message",
                "recipient": "bob",
                "content": "hi",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in _text(result).lower()

    async def test_should_reject_empty_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv3"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv3",
                "type": "message",
                "recipient": "",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in _text(result).lower()

    async def test_should_reject_nonexistent_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv4"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv4",
                "type": "message",
                "recipient": "ghost",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_reject_nonexistent_sender(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv4b"})
        await teams.add_member("tv4b", _make_teammate("bob", "tv4b"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv4b",
                "type": "message",
                "sender": "ghost",
                "recipient": "bob",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_pass_target_color(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv5"})
        await teams.add_member("tv5", _make_teammate("bob", "tv5"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv5",
                "type": "message",
                "recipient": "bob",
                "content": "hey",
                "summary": "greet",
            },
        )
        data = _data(result)
        assert data["routing"]["targetColor"] == "blue"

    async def test_should_reject_broadcast_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv6"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv6",
                "type": "broadcast",
                "content": "hello",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in _text(result).lower()

    async def test_should_reject_broadcast_from_non_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv6b"})
        await teams.add_member("tv6b", _make_teammate("worker", "tv6b"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv6b",
                "type": "broadcast",
                "sender": "worker",
                "content": "hello",
                "summary": "status",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in _text(result)

    async def test_should_reject_shutdown_request_to_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv7"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv7", "type": "shutdown_request", "recipient": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in _text(result)

    async def test_should_reject_shutdown_request_to_nonexistent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv8"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv8", "type": "shutdown_request", "recipient": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)


class TestTeamDeleteErrorWrapping:
    async def test_should_reject_delete_with_active_members(self, client: Client):
        await client.call_tool("team_create", {"team_name": "td1"})
        await teams.add_member("td1", _make_teammate("worker", "td1"))
        result = await client.call_tool(
            "team_delete",
            {"team_name": "td1"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "member" in _text(result).lower()

    async def test_should_reject_delete_nonexistent_team(self, client: Client):
        result = await client.call_tool(
            "team_delete",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "Traceback" not in _text(result)


class TestPlanApprovalValidation:
    async def test_should_reject_plan_approval_to_nonexistent_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp1"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp1",
                "type": "plan_approval_response",
                "recipient": "ghost",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_reject_plan_approval_with_empty_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp2"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp2",
                "type": "plan_approval_response",
                "recipient": "",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in _text(result).lower()


# ---------------------------------------------------------------------------
# New backend-aware tools
# ---------------------------------------------------------------------------
