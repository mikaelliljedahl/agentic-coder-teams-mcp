"""Tests for MCP tool descriptions exposed from docstrings."""

import json
import os
import subprocess
import sys

import pytest

from claude_teams import server_simple


def test_send_message_description_states_both_paths_and_the_refusal() -> None:
    """C3/R5 — the docstring is the contract a calling agent actually reads.

    It previously promised that an unknown recipient is re-routed to the lead
    with a warning. Leaving that text would have agents keep relying on the
    behaviour R5 removed.
    """
    description = server_simple.send_message.__doc__ or ""

    assert "guaranteed path" in description
    assert "does NOT enter the recipient's inbox" in description
    assert "REFUSED" in description
    assert "idempotency_key" in description
    assert "re-routed" not in description.replace("used to be re-routed", "")


def test_follow_up_agent_description_explains_non_polling_use_case() -> None:
    description = server_simple.follow_up_agent.__doc__ or ""

    assert "never read an inbox message" in description
    assert "routes through this same path" in description
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


def test_disk_contract_note_documents_lead_wake_hook() -> None:
    note = server_simple._DISK_CONTRACT_NOTE

    assert "background_tasks" in note
    assert "wake-progress-" in note
    assert "WIN_AGENT_TEAMS_LEAD_WAKE" in note
    assert "read_messages" in note


def test_install_lead_wake_description_documents_contract() -> None:
    description = server_simple.install_lead_wake.__doc__ or ""

    assert ".claude/settings.local.json" in description
    assert "migrated_from" in description
    assert "Idempotent" in description
    assert "remove=True" in description
    assert "WIN_AGENT_TEAMS_LEAD_WAKE=0" in description


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
async def test_spawn_agent_permission_mode_schema_is_constrained() -> None:
    tool = await server_simple.mcp.get_tool("spawn_agent")
    assert tool is not None

    permission_schema = tool.parameters["properties"]["permission_mode"]

    assert permission_schema["default"] == "bypass"
    assert permission_schema["enum"] == ["bypass", "default", "require_approval"]
    assert "backend-native" in (tool.description or "")
    assert "acceptEdits" in (tool.description or "")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["spawn_agent", "agent_watch_paths"])
async def test_watch_contract_documents_parked_marker_delivery(tool_name: str) -> None:
    """A coordinator that arms its watch late must learn from the description
    alone that a parked worker still wakes it, once per reader, at-least-once."""
    # Docstrings wrap at 79 columns; compare on collapsed whitespace.
    description = " ".join((await _registered_description(tool_name)).split())

    assert "ALREADY waiting when the watch started" in description
    assert ".watch/ack-<reader>.json" in description
    # The qualified sentence, not just the keyword: a consumer must not read
    # "once per park" as exactly-once.
    assert "Delivery is at-least-once" in description
    assert "duplicate is possible after an interrupted wake" in description
    assert "--no-parked" in description
    assert "message > output > waiting" in description
    assert "nested lead keeps its own ack file" in description


def _tier_line(description: str, tier: str) -> str | None:
    """Return the bullet line that defines ``tier`` in the description."""
    return next(
        (
            ln
            for ln in description.splitlines()
            if ln.strip().startswith(f"- ``{tier}``")
        ),
        None,
    )


@pytest.mark.asyncio
async def test_spawn_agent_description_documents_tier_ladder() -> None:
    description = await _registered_description("spawn_agent")

    # The consuming agent only ever reads the registered tool description, so
    # every tier's model AND effort must be pinned together on its own line,
    # not merely be present somewhere in the text, or a swapped effort would
    # still pass.
    for tier, slug, effort in (
        ("cheapest", "gpt-6-luna", "high"),
        ("low", "gpt-6-luna", "xhigh"),
        ("medium", "gpt-6-luna", "max"),
        ("high", "gpt-6-sol", "high"),
        ("xhigh", "gpt-6-astra", "low"),
        ("max", "gpt-6-astra", "medium"),
    ):
        line = _tier_line(description, tier)
        assert line is not None, f"{tier} missing from the registered description"
        assert f"(``{slug}``) @ {effort} " in line


@pytest.mark.asyncio
async def test_spawn_agent_description_documents_pi_fast_subtiers() -> None:
    description = await _registered_description("spawn_agent")

    # The pi-only subtier must be discoverable from the description alone;
    # the retired ``high-fast`` must not be offered as a tier, only named as
    # removed so a stale caller learns its replacement.
    assert "pi only" in description
    assert "faster" in description
    line = _tier_line(description, "medium-fast")
    assert line is not None, "medium-fast missing from the registered description"
    assert "(``gpt-6-sol``) @ medium " in line
    assert _tier_line(description, "high-fast") is None
    flat = " ".join(description.split())
    assert "``high-fast`` was removed" in flat
    assert "RetiredTierError" in flat


@pytest.mark.asyncio
async def test_spawn_agent_description_documents_gpt6_ladder() -> None:
    description = " ".join((await _registered_description("spawn_agent")).split())

    # The ladder runs on GPT-6 only; a stale 5.6 slug in the description would
    # tell the consuming agent the wrong model. The minimum pi release that
    # exposes GPT-6 Sol/Luna must be stated so a hard-fail is actionable.
    for slug in ("gpt-6-luna", "gpt-6-sol", "gpt-6-astra"):
        assert f"``{slug}``" in description
    assert "gpt-5.6" not in description
    # The fast-subtier speed multiplier was measured on GPT-5.6 Terra/Sol.
    assert "3-4x" not in description
    assert "0.87.1" in description


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["install_lead_wake", "install_member_wake"])
async def test_wake_install_descriptions_document_local_project_file(
    tool_name: str,
) -> None:
    description = await _registered_description(tool_name)

    assert ".claude/settings.local.json" in description
    assert "migrated_from" in description
    assert "legacy_cleanup" in description
    assert "git_ignore" in description
    assert "tracked" in description
    assert "~/.claude/settings.json" in description
    assert "settings_write_failed" in description


def test_flag_on_tool_contract_and_watch_retained():
    env = os.environ.copy()
    env["WIN_AGENT_TEAMS_NATIVE_WAKE"] = "1"
    code = (
        "import asyncio,json; "
        "from claude_teams import server_simple as s; "
        'print(json.dumps({"tools":{t.name:t.description '
        "for t in asyncio.run(s.mcp.list_tools())}, "
        '"note":s._DISK_CONTRACT_NOTE}))'
    )
    result = subprocess.run(  # noqa: S603 - fresh interpreter, no external CLI.
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    tools = data["tools"]
    for name in (
        "send_message",
        "read_messages",
        "external_read",
        "create_join_ticket",
        "session_info",
        "resume_session",
        "external_set_wake",
    ):
        assert "WIN_AGENT_TEAMS_NATIVE_WAKE=1" in tools[name]
        assert "best-effort" in tools[name]
        assert "watcher" in tools[name] or "watch" in tools[name]
        assert "stop arming" not in tools[name]
    for name in ("read_messages", "external_read", "session_info", "resume_session"):
        assert "Linux-only" in tools[name]
        assert "macOS" in tools[name]
    for name in ("session_info", "resume_session"):
        assert "backlog notice follows immediately" in tools[name]
    for name in ("read_messages", "external_read"):
        assert "no content" in tools[name]
    for status in (
        "queued",
        "coalesced",
        "backoff",
        "failed",
        "timeout",
        "unavailable",
        "unverified_thread",
        "stale_registration",
        "disabled",
    ):
        assert status in tools["send_message"]
    assert "or wake is involved" not in tools["send_message"]
    assert "or process resume is involved" in tools["send_message"]
    for name in ("external_read", "external_set_wake"):
        assert "call external_read once to re-arm notices" in tools[name]
        assert "no native notice" in tools[name]
    assert "member-supplied" in tools["external_set_wake"]
    assert "existing directory" in tools["external_set_wake"]
    assert "run the watch as a BACKGROUND command" in data["note"]
    for literal in (
        "Linux-only",
        "native-wake-",
        "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE",
        "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX",
        "keep arming",
    ):
        assert literal in data["note"]
