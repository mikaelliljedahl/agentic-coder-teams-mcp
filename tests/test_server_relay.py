"""Relay and teammate-introspection server tests."""

import asyncio
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from fastmcp import Client

from claude_teams import messaging
from claude_teams.backends import registry
from claude_teams.backends.base import HealthStatus, SpawnResult as BackendSpawnResult
from claude_teams.server_team_relay import relay_one_shot_result
from tests._server_support import (
    _data,
    _items,
    _make_mock_backend,
    _text,
)


class TestOneShotBackendRelay:
    async def test_should_relay_codex_result_to_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "oneshot"})

        mock_codex = _make_mock_backend("codex")

        def _spawn_side_effect(request):
            output_path = Path(request.extra["output_last_message_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("codex teammate result")
            return BackendSpawnResult(process_handle="%codex", backend_type="codex")

        mock_codex.spawn.side_effect = _spawn_side_effect
        mock_codex.health_check.return_value = HealthStatus(
            alive=False,
            detail="one-shot complete",
        )
        registry._backends["codex"] = mock_codex

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot",
                "name": "codex-worker",
                "backend": "codex",
                "model": "gpt-5.3-codex",
                "prompt": "reply with result",
            },
        )

        inbox = await _wait_for_lead_inbox_message(client, "oneshot", "teammate_result")
        assert len(inbox) == 1
        assert inbox[0]["from"] == "codex-worker"
        assert inbox[0]["summary"] == "teammate_result"
        assert "codex teammate result" in inbox[0]["text"]

    async def test_should_relay_when_output_exists_even_if_pane_is_alive(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "oneshot-alive"})

        mock_codex = _make_mock_backend("codex")

        def _spawn_side_effect(request):
            output_path = Path(request.extra["output_last_message_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("codex pane-still-alive result")
            return BackendSpawnResult(
                process_handle="%codex-alive", backend_type="codex"
            )

        mock_codex.spawn.side_effect = _spawn_side_effect
        mock_codex.health_check.return_value = HealthStatus(
            alive=True,
            detail="tmux pane still open",
        )
        registry._backends["codex"] = mock_codex

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot-alive",
                "name": "codex-worker",
                "backend": "codex",
                "model": "gpt-5.3-codex",
                "prompt": "reply with result",
            },
        )

        inbox = await _wait_for_lead_inbox_message(
            client,
            "oneshot-alive",
            "teammate_result",
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "codex-worker"
        assert inbox[0]["summary"] == "teammate_result"
        assert "codex pane-still-alive result" in inbox[0]["text"]

    async def test_should_relay_generic_backend_via_pane_capture(self, client: Client):
        await client.call_tool("team_create", {"team_name": "oneshot-generic"})

        mock_gemini = _make_mock_backend("gemini")
        mock_gemini.spawn.return_value = BackendSpawnResult(
            process_handle="%gemini", backend_type="gemini"
        )
        mock_gemini.health_check.return_value = HealthStatus(
            alive=False, detail="one-shot complete"
        )
        mock_gemini.capture.return_value = "gemini pane output here"
        registry._backends["gemini"] = mock_gemini

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot-generic",
                "name": "gemini-worker",
                "backend": "gemini",
                "model": "gemini-2.5-pro",
                "prompt": "analyze code",
            },
        )

        inbox = await _wait_for_lead_inbox_message(
            client,
            "oneshot-generic",
            "teammate_result",
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "gemini-worker"
        assert inbox[0]["summary"] == "teammate_result"
        assert "gemini pane output here" in inbox[0]["text"]

    async def test_should_strip_ansi_from_pane_capture(self, client: Client):
        await client.call_tool("team_create", {"team_name": "oneshot-ansi"})

        mock_gemini = _make_mock_backend("gemini")
        mock_gemini.spawn.return_value = BackendSpawnResult(
            process_handle="%gemini-ansi", backend_type="gemini"
        )
        mock_gemini.health_check.return_value = HealthStatus(alive=False, detail="done")
        mock_gemini.capture.return_value = (
            "\x1b[32mgreen output\x1b[0m with \x1b[1mbold\x1b[0m"
        )
        registry._backends["gemini"] = mock_gemini

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot-ansi",
                "name": "gemini-worker",
                "backend": "gemini",
                "model": "gemini-2.5-pro",
                "prompt": "analyze",
            },
        )

        inbox = await _wait_for_lead_inbox_message(
            client,
            "oneshot-ansi",
            "teammate_result",
        )
        assert len(inbox) == 1
        assert "\x1b" not in inbox[0]["text"]
        assert "green output" in inbox[0]["text"]
        assert "bold" in inbox[0]["text"]

    async def test_should_not_relay_for_interactive_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "oneshot-norelay"})

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot-norelay",
                "name": "claude-worker",
                "backend": "claude-code",
                "prompt": "do stuff",
            },
        )

        await asyncio.sleep(0.1)

        inbox = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "oneshot-norelay",
                    "agent_name": "team-lead",
                    "unread_only": True,
                },
            )
        )
        # Interactive backends handle their own messaging — no relay message
        assert len(inbox) == 0

    async def test_should_fail_fast_when_backend_missing_and_no_result_file(
        self, client: Client
    ):
        start = time.monotonic()
        await client.call_tool("team_create", {"team_name": "relay-missing"})

        await relay_one_shot_result(
            "relay-missing",
            "worker",
            "missing-backend",
            "%missing",
            None,
            "blue",
        )

        inbox = await _wait_for_lead_inbox_message(
            client,
            "relay-missing",
            "teammate_result",
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert len(inbox) == 1
        assert "could not be resolved" in inbox[0]["text"]


async def _wait_for_lead_inbox_message(
    client: Client,
    team_name: str,
    expected_summary: str,
    timeout_seconds: float = 2.0,
) -> list[dict]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        inbox = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": team_name,
                    "agent_name": "team-lead",
                    "unread_only": True,
                    "mark_as_read": False,
                },
            )
        )
        if inbox and inbox[0]["summary"] == expected_summary:
            return inbox
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for lead inbox message {expected_summary!r} in {team_name!r}"
    )


class TestCheckTeammate:
    async def test_returns_backend_status_and_optional_output(self, client: Client):
        await client.call_tool("team_create", {"team_name": "check-team"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "check-team",
                "name": "worker",
                "prompt": "help out",
            },
        )
        await messaging.read_inbox("check-team", "worker", mark_as_read=True)

        mock_backend = cast(MagicMock, registry._backends["claude-code"])
        mock_backend.health_check.return_value = HealthStatus(
            alive=True, detail="pane healthy"
        )
        mock_backend.capture.return_value = "captured output"

        result = _data(
            await client.call_tool(
                "check_teammate",
                {
                    "team_name": "check-team",
                    "agent_name": "worker",
                    "include_output": True,
                },
            )
        )

        assert result["name"] == "worker"
        assert result["backend"] == "claude-code"
        assert result["alive"] is True
        assert result["detail"] == "pane healthy"
        assert result["output"] == "captured output"
        assert result["error"] is None
        assert result["pending_from"] == []
        assert result["their_unread_count"] == 0

    async def test_returns_pending_from_and_preserves_other_senders(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "check-msg"})
        for name in ("alice", "bob"):
            await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "check-msg",
                    "name": name,
                    "prompt": f"instructions for {name}",
                },
            )
            await messaging.read_inbox("check-msg", name, mark_as_read=True)

        await messaging.send_plain_message(
            "check-msg",
            "alice",
            "team-lead",
            "alice update",
            summary="update",
        )
        await messaging.send_plain_message(
            "check-msg",
            "bob",
            "team-lead",
            "bob update",
            summary="update",
        )

        result = _data(
            await client.call_tool(
                "check_teammate",
                {"team_name": "check-msg", "agent_name": "alice"},
            )
        )

        assert len(result["pending_from"]) == 1
        assert result["pending_from"][0]["from"] == "alice"
        assert result["pending_from"][0]["text"] == "alice update"

        unread = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "check-msg",
                    "agent_name": "team-lead",
                    "unread_only": True,
                    "mark_as_read": False,
                },
            )
        )
        assert len(unread) == 1
        assert unread[0]["from"] == "bob"

    async def test_include_messages_false_leaves_lead_inbox_unread(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "check-no-msg"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "check-no-msg",
                "name": "worker",
                "prompt": "help out",
            },
        )
        await messaging.read_inbox("check-no-msg", "worker", mark_as_read=True)

        await messaging.send_plain_message(
            "check-no-msg",
            "worker",
            "team-lead",
            "still unread",
            summary="status",
        )

        result = _data(
            await client.call_tool(
                "check_teammate",
                {
                    "team_name": "check-no-msg",
                    "agent_name": "worker",
                    "include_messages": False,
                },
            )
        )

        assert result["pending_from"] == []

        unread = _items(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "check-no-msg",
                    "agent_name": "team-lead",
                    "unread_only": True,
                    "mark_as_read": False,
                },
            )
        )
        assert len(unread) == 1
        assert unread[0]["from"] == "worker"

    async def test_reports_their_unread_count(self, client: Client):
        await client.call_tool("team_create", {"team_name": "check-unread"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "check-unread",
                "name": "worker",
                "prompt": "help out",
            },
        )
        await messaging.read_inbox("check-unread", "worker", mark_as_read=True)

        for text in ("task one", "task two"):
            await client.call_tool(
                "send_message",
                {
                    "team_name": "check-unread",
                    "type": "message",
                    "recipient": "worker",
                    "content": text,
                    "summary": "task",
                },
            )

        result = _data(
            await client.call_tool(
                "check_teammate",
                {"team_name": "check-unread", "agent_name": "worker"},
            )
        )

        assert result["their_unread_count"] == 2

    async def test_rejects_unknown_agent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "check-missing"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "check-missing",
                "name": "worker",
                "prompt": "help out",
            },
        )

        result = await client.call_tool(
            "check_teammate",
            {"team_name": "check-missing", "agent_name": "ghost"},
            raise_on_error=False,
        )

        assert result.is_error is True
        assert "ghost" in _text(result)


# ---------------------------------------------------------------------------
# Existing tests — updated to use team_client where needed
# ---------------------------------------------------------------------------
