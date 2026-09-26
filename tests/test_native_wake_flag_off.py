"""Golden compatibility contracts captured before native wake implementation."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from claude_teams import server_simple as ss
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_base import BaseBackend, process_manager
from tests import test_join_team

join_session = test_join_team.join_session

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/native_wake/flag_off.json").read_text(
        encoding="utf-8"
    )
)


def test_golden_is_ascii_for_windows_default_encoding() -> None:
    fixture = Path(__file__).parent / "fixtures/native_wake/flag_off.json"
    assert fixture.read_bytes().isascii()


@pytest.fixture(params=[None, "0", "true", ""])
def flag_off(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raising=False)
    else:
        monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", request.param)


def test_main_does_not_start_notifier(flag_off, monkeypatch):
    monkeypatch.setattr(type(ss.mcp), "run", lambda self: None)
    monkeypatch.setattr(
        "threading.Thread.start", lambda _: pytest.fail("thread started")
    )
    if hasattr(ss, "native_wake"):
        monkeypatch.setattr(
            ss.native_wake,
            "NativeWakeNotifier",
            lambda **kwargs: pytest.fail("notifier constructed"),
        )
    ss.main()


def test_join_prompt_golden(flag_off, monkeypatch):
    monkeypatch.setattr(ss, "_SESSION_BASE", PurePosixPath("/flag-off-sessions"))
    assert (
        ss._build_join_prompt(
            session_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            token="golden",
            name="member",
            parent="team-lead",
            note="Role",
        )
        == GOLDEN["prompt"]
    )


def test_send_read_cycle_identical(flag_off, join_session, monkeypatch):
    sid, directory = join_session
    ticket = asyncio.run(ss.create_join_ticket("member"))
    joined = asyncio.run(ss.join_team(sid, ticket["token"]))
    with ss._agents_transaction(sid) as agents:
        agents[0]["codex_wake"] = {
            "thread_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            "codex_home": str(directory),
            "generation": 1,
        }
    # No production CLI or real socket is permitted in this test.
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("subprocess ran")
    )
    if hasattr(ss, "_codex_member_wake"):
        monkeypatch.setattr(
            ss._codex_member_wake,
            "runner",
            lambda *a, **k: pytest.fail("queue ran"),
        )
    assert asyncio.run(ss.send_message("body", to="member")) == {
        "success": True,
        "to": "member",
        "delivery": "inbox",
        "note": (
            "External agent; pull-based and unconfirmed. It reads "
            "this inbox on its next external_read call."
        ),
    }
    assert asyncio.run(ss.external_read(joined["member_token"]))["success"]
    assert "native_wake" not in asyncio.run(ss.session_info())
    assert not list(directory.glob("native-wake-*"))


@pytest.mark.parametrize("backend", ["claude-code", "codex", "pi"])
@pytest.mark.parametrize("operation", ["spawn", "resume"])
def test_spawn_resume_environment_identical(flag_off, monkeypatch, backend, operation):
    from claude_teams.backends.claude_code import ClaudeCodeBackend
    from claude_teams.backends.codex import CodexBackend
    from claude_teams.backends.pi import PiBackend

    instance = {
        "claude-code": ClaudeCodeBackend,
        "codex": CodexBackend,
        "pi": PiBackend,
    }[backend]()
    env = {"EXAMPLE": "value"}
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_SOCKET", "/inherited.sock")
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "inherited")
    monkeypatch.setattr(instance, "build_env", lambda _: env.copy())
    monkeypatch.setattr(instance, "build_command", lambda _: ["fake"])
    monkeypatch.setattr(instance, "build_resume_command", lambda *a: ["fake"])
    observed = []
    monkeypatch.setattr(
        process_manager,
        "spawn_process",
        lambda request, argv, env_vars, *a, **k: observed.append(env_vars),
    )
    request = SpawnRequest(
        agent_id="id",
        name="agent",
        team_name="team",
        prompt="task",
        model="",
        agent_type="",
        color="",
        cwd=".",
        lead_session_id="team-lead",
    )
    if operation == "spawn":
        BaseBackend.spawn(instance, request)
    else:
        BaseBackend.resume(instance, request, "thread")
    assert observed == [env]


@pytest.mark.parametrize("external_only", [False, True])
def test_registered_tools_golden(flag_off, external_only):
    env = os.environ.copy()
    env["WIN_AGENT_TEAMS_EXTERNAL_ONLY"] = "1" if external_only else "0"
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
    expected = GOLDEN["tools"]
    if external_only:
        expected = {
            k: v
            for k, v in expected.items()
            if k
            in {
                "join_team",
                "external_send",
                "external_read",
                "leave_team",
                "list_backends",
            }
        }
    assert json.loads(result.stdout) == expected
