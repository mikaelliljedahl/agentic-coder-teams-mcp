"""Plan §2.8: flag-on tool text for native downstream delivery; flag-off golden."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/native_wake/flag_off.json").read_text()
)["tools"]
FLAG = "Only with WIN_AGENT_TEAMS_NATIVE_WAKE=1"


def _words(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def tools() -> dict[str, str]:
    env = os.environ.copy()
    env["WIN_AGENT_TEAMS_NATIVE_WAKE"] = "1"
    env.pop("WIN_AGENT_TEAMS_EXTERNAL_ONLY", None)
    code = (
        "import asyncio,json; "
        "from claude_teams.server_simple import mcp; "
        "print(json.dumps({t.name:t.description "
        "for t in asyncio.run(mcp.list_tools())}))"
    )
    result = subprocess.run(  # noqa: S603 - fresh interpreter, no external CLI.
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", ["follow_up_agent", "send_message", "delivery_status"])
def test_delivery_tools_describe_native_carriers(tools, name):
    text = tools[name]
    for literal in (
        "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1",
        "method",
        "resume",
        "codex_queue",
        "claude_mailbox",
        "native_unresolved",
        "prior_native_attempt_unresolved",
        "blocking_key",
        "do NOT resend",
        "may still run",
        "16 KiB",
        "deliveries release-native",
        "operator_released",
    ):
        assert literal in text, (name, literal)


@pytest.mark.parametrize(
    "name", ["follow_up_agent", "delivery_status", "kill_agent", "spawn_agent"]
)
def test_flag_on_text_extends_the_golden_text(tools, name):
    # Whitespace-normalised: a decorated docstring is not dedented.
    assert _words(tools[name]).startswith(_words(GOLDEN[name]))
    assert tools[name].count(FLAG) == 1


def test_send_message_keeps_external_note_and_adds_downstream(tools):
    text = tools["send_message"]
    assert "wake:{method:codex_queue" in text
    assert text.count(FLAG) == 2


def test_kill_agent_reports_native_unresolved(tools):
    text = tools["kill_agent"]
    for literal in (
        "native_unresolved",
        "may still run",
        "prior_native_attempt_unresolved",
        "deliveries release-native",
    ):
        assert literal in text, literal


def test_spawn_agent_has_one_merged_native_note(tools):
    text = tools["spawn_agent"]
    for literal in (
        "interactive",
        "codex_home",
        "dispatch_epoch",
        "list_agents",
        "enable_spawned_lead_wake",
        "provisional",
    ):
        assert literal in text, literal


def test_session_info_and_set_lead_wake_unchanged_contract(tools):
    assert "codex_lead" in tools["session_info"]
    assert "set_lead_wake" in tools
