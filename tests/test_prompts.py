"""Tests for team-lead MCP prompts and rendering."""

import pytest
from fastmcp import Client

from tests._server_support import _content_text

pytest_plugins = ["tests._server_support"]

_EXPECTED_PROMPTS = [
    "status_check",
    "health_sweep",
    "task_handoff",
    "wrap_up",
    "unblock_teammate",
]


# ---------------------------------------------------------------------------
# Static discovery — prompts are visible to clients at startup
# ---------------------------------------------------------------------------


async def test_prompts_visible_before_team(client: Client) -> None:
    """Prompts are visible before a team is created."""
    prompts = await client.list_prompts()
    names = [p.name for p in prompts]
    for expected in _EXPECTED_PROMPTS:
        assert expected in names, f"Missing prompt: {expected}"


async def test_prompts_visible_after_team(team_client: Client) -> None:
    """All team prompts appear after team_create."""
    prompts = await team_client.list_prompts()
    names = [p.name for p in prompts]
    for expected in _EXPECTED_PROMPTS:
        assert expected in names, f"Missing prompt: {expected}"


async def test_prompt_count(team_client: Client) -> None:
    """Exactly 5 team prompts are registered."""
    prompts = await team_client.list_prompts()
    assert len(prompts) == 5


# ---------------------------------------------------------------------------
# Prompt rendering — arguments populate the template
# ---------------------------------------------------------------------------


async def test_status_check_renders(team_client: Client) -> None:
    """status_check includes team and teammate in the user message."""
    result = await team_client.get_prompt(
        "status_check", arguments={"team": "alpha", "teammate": "bob"}
    )
    messages = result.messages
    assert len(messages) == 2
    user_text = _content_text(messages[0].content)
    assert "bob" in user_text
    assert "alpha" in user_text
    assert messages[1].role == "assistant"


async def test_health_sweep_renders(team_client: Client) -> None:
    """health_sweep includes team name in the user message."""
    result = await team_client.get_prompt("health_sweep", arguments={"team": "bravo"})
    messages = result.messages
    assert len(messages) == 2
    user_text = _content_text(messages[0].content)
    assert "bravo" in user_text
    assert "health_check" in user_text


async def test_task_handoff_renders(team_client: Client) -> None:
    """task_handoff includes source, target, and optional context."""
    result = await team_client.get_prompt(
        "task_handoff",
        arguments={
            "team": "charlie",
            "from_teammate": "alice",
            "to_teammate": "bob",
            "context": "parsing is done",
        },
    )
    messages = result.messages
    user_text = _content_text(messages[0].content)
    assert "alice" in user_text
    assert "bob" in user_text
    assert "parsing is done" in user_text


async def test_task_handoff_without_context(team_client: Client) -> None:
    """task_handoff works without optional context."""
    result = await team_client.get_prompt(
        "task_handoff",
        arguments={
            "team": "delta",
            "from_teammate": "x",
            "to_teammate": "y",
        },
    )
    user_text = _content_text(result.messages[0].content)
    assert "Additional context" not in user_text


async def test_wrap_up_renders(team_client: Client) -> None:
    """wrap_up includes team name and references task_list."""
    result = await team_client.get_prompt("wrap_up", arguments={"team": "echo"})
    messages = result.messages
    user_text = _content_text(messages[0].content)
    assert "echo" in user_text
    assert "task_list" in user_text
    assert messages[1].role == "assistant"


async def test_unblock_teammate_renders(team_client: Client) -> None:
    """unblock_teammate includes team, teammate, and optional hint."""
    result = await team_client.get_prompt(
        "unblock_teammate",
        arguments={
            "team": "foxtrot",
            "teammate": "stuck-agent",
            "hint": "check the auth token",
        },
    )
    user_text = _content_text(result.messages[0].content)
    assert "stuck-agent" in user_text
    assert "check the auth token" in user_text


async def test_unblock_without_hint(team_client: Client) -> None:
    """unblock_teammate works without optional hint."""
    result = await team_client.get_prompt(
        "unblock_teammate",
        arguments={"team": "golf", "teammate": "agent-x"},
    )
    user_text = _content_text(result.messages[0].content)
    assert "Hint from lead" not in user_text


# ---------------------------------------------------------------------------
# Assistant prefill — every prompt primes the model for action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt_name", "args"),
    [
        ("status_check", {"team": "t", "teammate": "m"}),
        ("health_sweep", {"team": "t"}),
        ("task_handoff", {"team": "t", "from_teammate": "a", "to_teammate": "b"}),
        ("wrap_up", {"team": "t"}),
        ("unblock_teammate", {"team": "t", "teammate": "m"}),
    ],
)
async def test_assistant_prefill_present(
    team_client: Client, prompt_name: str, args: dict[str, str]
) -> None:
    """Every prompt includes an assistant prefill message."""
    result = await team_client.get_prompt(prompt_name, arguments=args)
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
