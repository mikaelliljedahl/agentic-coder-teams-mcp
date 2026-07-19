"""Tests for MCP tool descriptions exposed from docstrings."""

import pytest

from claude_teams import server_simple


def test_send_message_description_points_to_follow_up_for_non_polling_agents() -> None:
    description = server_simple.send_message.__doc__ or ""

    assert "actively poll read_messages" in description
    assert "not a push/resume mechanism" in description
    assert "use follow_up_agent instead" in description


def test_follow_up_agent_description_explains_non_polling_use_case() -> None:
    description = server_simple.follow_up_agent.__doc__ or ""

    assert "not polling read_messages" in description
    assert "send_message only writes to an inbox" in description
    assert "continuing a spawned agent" in description


def test_follow_up_agent_description_states_the_bounded_wait_not_a_refusal() -> None:
    """B2 — the docstring is the contract calling agents actually read.

    It previously advertised ``reason="agent_busy"``, which is exactly the
    dead end R1 removes; leaving that text would keep leads picking wrong.
    """
    description = server_simple.follow_up_agent.__doc__ or ""

    assert 'reason="agent_busy"' not in description
    assert "A busy agent is NOT refused" in description
    assert "idempotency_key is REQUIRED" in description


def test_follow_up_agent_description_names_all_three_statuses() -> None:
    description = server_simple.follow_up_agent.__doc__ or ""

    for status in ("delivered", "failed", "queued"):
        assert f'"{status}"' in description


def test_delivery_status_description_says_it_reconciles() -> None:
    """A passive-lookup reading would make response-loss recovery useless."""
    description = server_simple.delivery_status.__doc__ or ""

    assert "ACTIVE reconciler" in description
    assert "idempotency_key" in description


def test_deliver_pending_description_names_the_drain_allow_list() -> None:
    description = server_simple.deliver_pending.__doc__ or ""

    assert "no background dispatcher" in description
    assert "agent_status" in description
    assert "stay cheap reads" in description


def test_check_agent_description_documents_full_len() -> None:
    description = server_simple.check_agent.__doc__ or ""

    assert "full_len" in description


def test_list_agents_description_documents_full_len() -> None:
    description = server_simple.list_agents.__doc__ or ""

    assert "full_len" in description


def _assert_disk_contract_note(description: str) -> None:
    """Shared assertions for the item-2 uniform disk-contract docstring note."""
    assert "state-{name}.json" in description
    assert '"state"' in description
    assert '"event"' in description
    assert '"ts"' in description
    assert "auto-restart" in description
    assert "tight-poll" in description
    assert "background" in description.lower()
    assert "foreground" in description.lower()
    assert "Claude Code" in description
    assert "Codex" in description
    assert 'reason="message"' in description
    assert "read_messages" in description
    assert 'reason="waiting"' in description
    assert "exit 2" in description
    assert "re-check" in description


def test_disk_contract_note_documents_discoverable_watch_argv() -> None:
    note = server_simple._DISK_CONTRACT_NOTE

    assert "may not be on PATH" in note
    assert "watch_argv" in note


def test_disk_contract_note_documents_one_shot_rearming() -> None:
    note = server_simple._DISK_CONTRACT_NOTE

    assert "one-shot" in note
    assert "re-arm" in note


async def _registered_description(tool_name: str) -> str:
    """Return the client-visible ``Tool.description`` FastMCP registered.

    This is the description FastMCP parsed from the docstring at
    ``@mcp.tool()`` decoration time, i.e. exactly what a client sees from
    ``list_tools``/``get_tool``. It is distinct from ``func.__doc__``, which
    can be mutated after registration without affecting the client-visible
    schema.
    """
    tool = await server_simple.mcp.get_tool(tool_name)
    assert tool is not None
    return tool.description or ""


@pytest.mark.asyncio
async def test_agent_status_description_documents_disk_contract_and_both_recipes() -> (
    None
):
    description = await _registered_description("agent_status")
    _assert_disk_contract_note(description)


@pytest.mark.asyncio
async def test_check_agent_description_documents_disk_contract_and_both_recipes() -> (
    None
):
    description = await _registered_description("check_agent")
    _assert_disk_contract_note(description)


@pytest.mark.asyncio
async def test_list_agents_description_documents_disk_contract_and_both_recipes() -> (
    None
):
    description = await _registered_description("list_agents")
    _assert_disk_contract_note(description)


@pytest.mark.asyncio
async def test_agent_watch_paths_registered_description_has_disk_contract() -> None:
    description = await _registered_description("agent_watch_paths")

    _assert_disk_contract_note(description)
