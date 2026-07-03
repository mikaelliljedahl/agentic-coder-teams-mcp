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
    assert 'reason="agent_busy"' in description


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


async def _registered_description(tool_name: str) -> str:
    """Return the client-visible ``Tool.description`` FastMCP registered.

    This is the description FastMCP parsed from the docstring at
    ``@mcp.tool()`` decoration time, i.e. exactly what a client sees from
    ``list_tools``/``get_tool``. It is distinct from ``func.__doc__``, which
    can be mutated after registration without affecting the client-visible
    schema.
    """
    tool = await server_simple.mcp.get_tool(tool_name)
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
