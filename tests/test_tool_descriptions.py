"""Tests for MCP tool descriptions exposed from docstrings."""

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
