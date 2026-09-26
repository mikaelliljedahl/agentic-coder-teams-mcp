"""D: a Codex lead woken by ``codex queue`` on its own thread (plan §2.6, test 11).

Every transport, host lookup and thread verification is a fake: no codex CLI,
no process walk and no real socket runs here.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claude_teams import native_wake as nw
from claude_teams import server_simple as ss
from tests import test_join_team
from tests.test_spawn_agent_watch_contract import _FakeBackend, _FakeRegistry

join_session = test_join_team.join_session

THREAD_A = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
THREAD_B = "bbbbbbbb-bbbb-4ccc-8ddd-eeeeeeeeeeee"
HOST = (4242, "token-1")
NEW_HOST = (4343, "token-2")
READ_TOOL = "mcp__win_agent_teams__read_messages"
SET_TOOL = "mcp__win_agent_teams__set_lead_wake"


@pytest.fixture(autouse=True)
def wake_on(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", raising=False)
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", raising=False)
    nw._activation.clear()
    nw._members.clear()


def inbox(tmp_path, sid="s1", reader="team-lead", senders=("alice",)):
    directory = tmp_path / sid
    directory.mkdir(exist_ok=True)
    with (directory / f"inbox-{reader}.jsonl").open("a", encoding="utf-8") as handle:
        for sender in senders:
            handle.write(json.dumps({"from": sender, "text": "PRIVATE BODY"}) + "\n")
    return directory


def register(directory, *, thread=THREAD_A, host=HOST, spawned=False, bound=None):
    return nw.register_lead_wake(
        directory,
        "team-lead",
        thread_id=thread,
        codex_home="/codex-home" if thread else "",
        host=host,
        spawned=spawned,
        bound=bound,
    )


def stored(directory):
    registration = nw.read_lead_wake(directory, "team-lead")
    assert registration is not None
    return registration


class Fakes:
    """Injected host, binding, verification and queue for the Codex lead channel."""

    def __init__(self):
        self.host = HOST
        self.bound = None
        self.queued = []
        self.verified = []
        self.outcome = nw.QueueOutcome(True, 0, "01a0ddd0-2200-4000-8000-000000000000")
        self.before_verify = None

    def lead(self):
        def verify(home, thread):
            self.verified.append((home, thread))
            if self.before_verify is not None:
                self.before_verify()
            return True, ""

        return nw.CodexLeadWake(
            host=lambda: self.host,
            binding=lambda sid, identity: self.bound,
            queue=lambda thread, home, text: (
                self.queued.append((thread, home, text)) or self.outcome
            ),
            verify=verify,
        )


def notifier(tmp_path, active, fakes, **kwargs):
    kwargs.setdefault("channel", nw.ClaudeChannel("disabled"))
    kwargs.setdefault("post", Mock(side_effect=AssertionError("Claude posted")))
    return nw.NativeWakeNotifier(
        get_target=lambda: active[0],
        session_dir=lambda sid: tmp_path / sid,
        member_alive=lambda sid, name: True,
        codex_lead=fakes.lead(),
        **kwargs,
    )


# --- registration store --------------------------------------------------------


def test_human_lead_registers_active_with_host_incarnation(tmp_path):
    directory = inbox(tmp_path)
    stored = register(directory)
    assert stored["status"] == "active"
    assert stored["generation"] == 1
    assert (stored["host_pid"], stored["host_create_token"]) == HOST
    assert stored["thread_id"] == THREAD_A
    on_disk = json.loads((directory / "lead-wake-team-lead.json").read_text())
    assert on_disk == stored
    assert nw.read_lead_wake(directory, "team-lead") == stored


def test_spawned_lead_is_provisional_until_binding(tmp_path):
    directory = inbox(tmp_path)
    assert register(directory, spawned=True)["status"] == "provisional"
    assert register(directory, spawned=True, bound=THREAD_A)["status"] == "active"
    cleared = register(directory, spawned=True, bound=THREAD_B)
    assert cleared["status"] == "cleared"
    assert cleared["reason"] == "thread_mismatch"
    assert cleared["generation"] == 3


def test_clear_keeps_tombstone_and_bumps_generation(tmp_path):
    directory = inbox(tmp_path)
    register(directory)
    tombstone = register(directory, thread="")
    assert tombstone["status"] == "cleared"
    assert tombstone["thread_id"] is None
    assert tombstone["codex_home"] is None
    assert tombstone["generation"] == 2
    assert register(directory)["generation"] == 3


@pytest.mark.parametrize(
    "content", ["", "[]", "{", json.dumps({"generation": "1", "status": "active"})]
)
def test_corrupt_registration_reads_as_absent(tmp_path, content):
    directory = inbox(tmp_path)
    (directory / "lead-wake-team-lead.json").write_text(content)
    assert nw.read_lead_wake(directory, "team-lead") is None


# --- notifier: provisional, binding, incarnation, sessions ---------------------


def test_provisional_wrong_thread_gets_zero_queues(tmp_path):
    directory = inbox(tmp_path)
    register(directory, spawned=True)
    fakes = Fakes()
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        for _ in range(3):
            wake.tick()
        assert fakes.queued == []
        fakes.bound = THREAD_B
        wake.tick()
        wake.tick()
        assert fakes.queued == []
        current = stored(directory)
        assert (current["status"], current["reason"]) == ("cleared", "thread_mismatch")
    finally:
        wake.close()


def test_binding_after_first_ticks_activates_and_catches_up(tmp_path):
    directory = inbox(tmp_path)
    register(directory, spawned=True)
    fakes = Fakes()
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        wake.tick()
        wake.tick()
        assert fakes.queued == []
        fakes.bound = THREAD_A
        wake.tick()
        assert stored(directory)["status"] == "active"
        assert len(fakes.queued) == 1
        thread, home, text = fakes.queued[0]
        assert (thread, home) == (THREAD_A, "/codex-home")
        assert READ_TOOL in text
        assert "PRIVATE BODY" not in text
    finally:
        wake.close()


def test_session_switch_with_equal_generations_and_backlog(tmp_path):
    register(inbox(tmp_path, "s1"))
    register(inbox(tmp_path, "s2", senders=("bob",)))
    fakes = Fakes()
    active = [("s1", "team-lead")]
    wake = notifier(tmp_path, active, fakes)
    try:
        wake.tick()
        assert len(fakes.queued) == 1
        active[0] = ("s2", "team-lead")
        wake.tick()
        # S1's outstanding notice at generation 1 must not suppress S2's.
        assert len(fakes.queued) == 2
        assert "bob" in fakes.queued[1][2]
        assert list(wake.codex_targets) == [("s2", "team-lead", 1, *HOST)]
    finally:
        wake.close()


def test_mcp_restart_under_same_host_keeps_registration(tmp_path):
    register(inbox(tmp_path))
    first = Fakes()
    wake = notifier(tmp_path, [("s1", "team-lead")], first)
    wake.tick()
    wake.close()
    second = Fakes()
    restarted = notifier(tmp_path, [("s1", "team-lead")], second)
    try:
        restarted.tick()
        # The unread backlog is re-announced once after the restart.
        assert len(second.queued) == 1
    finally:
        restarted.close()


def test_new_host_incarnation_ignores_old_registration(tmp_path):
    directory = inbox(tmp_path)
    register(directory)
    fakes = Fakes()
    fakes.host = NEW_HOST
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        wake.tick()
        wake.tick()
        assert fakes.queued == []
        # A resumed nested lead under a new host re-registers, then is woken.
        register(directory, host=NEW_HOST, spawned=True, bound=THREAD_A)
        wake.tick()
        assert len(fakes.queued) == 1
    finally:
        wake.close()


def test_cleared_registration_never_queues(tmp_path):
    directory = inbox(tmp_path)
    register(directory)
    register(directory, thread="")
    fakes = Fakes()
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        wake.tick()
        assert fakes.queued == []
        assert fakes.verified == []
    finally:
        wake.close()


def test_revalidates_generation_immediately_before_queue(tmp_path):
    directory = inbox(tmp_path)
    register(directory)
    fakes = Fakes()
    fakes.before_verify = lambda: register(directory, thread="")
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        wake.tick()
        assert fakes.verified
        assert fakes.queued == []
    finally:
        wake.close()


def test_revalidates_target_immediately_before_queue(tmp_path):
    register(inbox(tmp_path))
    fakes = Fakes()
    active: list[tuple[str, str] | None] = [("s1", "team-lead")]

    def drop_target():
        active[0] = None

    fakes.before_verify = drop_target
    wake = notifier(tmp_path, active, fakes)
    try:
        wake.tick()
        assert fakes.queued == []
    finally:
        wake.close()


def test_uncertain_queue_backs_off_and_retries(tmp_path):
    register(inbox(tmp_path))
    fakes = Fakes()
    fakes.outcome = nw.QueueOutcome(True, 1)
    now = [0.0]
    wake = notifier(tmp_path, [("s1", "team-lead")], fakes, clock=lambda: now[0])
    try:
        wake.tick()
        wake.tick()
        assert len(fakes.queued) == 1
        fakes.outcome = nw.QueueOutcome(True, 0, "01a0ddd0-2200-4000-8000-00000000000a")
        now[0] = 3
        wake.tick()
        assert len(fakes.queued) == 2
        now[0] = 4
        wake.tick()
        assert len(fakes.queued) == 2
    finally:
        wake.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner locks")
def test_one_owner_per_codex_lead(tmp_path):
    register(inbox(tmp_path))
    fakes = Fakes()
    first = notifier(tmp_path, [("s1", "team-lead")], fakes)
    second = notifier(tmp_path, [("s1", "team-lead")], fakes)
    try:
        first.tick()
        second.tick()
        assert len(fakes.queued) == 1
        assert (tmp_path / "s1" / "native-wake-codex-lead.team-lead.lock").exists()
    finally:
        first.close()
        second.close()


def test_notice_text_is_body_free_and_names_the_full_tool():
    text = nw.codex_lead_notice({"al ice&|": 2, "bob": 1}, 3)
    assert READ_TOOL in text
    assert "#3" in text
    assert "3 unread" in text
    for forbidden in "&|<>^\"'()%\n":
        assert forbidden not in text


# --- independent channel gating ------------------------------------------------


def test_claude_off_codex_on_still_queues(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", "0")
    register(inbox(tmp_path))
    fakes = Fakes()
    post = Mock(side_effect=AssertionError("Claude posted"))
    wake = notifier(
        tmp_path,
        [("s1", "team-lead")],
        fakes,
        channel=nw.ClaudeChannel("available", "/fake", "secret"),
        post=post,
    )
    try:
        wake.tick()
        assert len(fakes.queued) == 1
        post.assert_not_called()
        assert not wake.targets
    finally:
        wake.close()


def test_codex_off_claude_still_posts(tmp_path, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", "0")
    register(inbox(tmp_path))
    fakes = Fakes()
    posted = []
    wake = notifier(
        tmp_path,
        [("s1", "team-lead")],
        fakes,
        channel=nw.ClaudeChannel("available", "/fake", "secret"),
        post=lambda channel, text: posted.append(text) or nw.PostResult(True),
    )
    try:
        wake.tick()
        assert len(posted) == 1
        assert fakes.queued == []
        assert not wake.codex_targets
    finally:
        wake.close()


def test_windows_codex_lead_is_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(nw, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(nw.winpipe, "reap_parked", lambda: None)
    register(inbox(tmp_path))
    fakes = Fakes()
    wake = notifier(
        tmp_path,
        [("s1", "team-lead")],
        fakes,
        channel=nw.ClaudeChannel("unsupported_platform"),
    )
    try:
        wake.tick()
        assert len(fakes.queued) == 1
    finally:
        wake.close()


def test_main_starts_notifier_for_codex_half_alone(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", "0")
    started = Mock()
    monkeypatch.setattr(type(ss.mcp), "run", lambda self: None)
    monkeypatch.setattr(nw.NativeWakeNotifier, "start", started)
    monkeypatch.setattr(nw.NativeWakeNotifier, "close", Mock())
    ss.main()
    started.assert_called_once()


def test_main_skips_notifier_when_both_halves_are_off(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", "0")
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", "0")
    monkeypatch.setattr(type(ss.mcp), "run", lambda self: None)
    monkeypatch.setattr(
        nw, "NativeWakeNotifier", Mock(side_effect=AssertionError("constructed"))
    )
    ss.main()


def test_external_member_notice_is_unchanged(tmp_path):
    directory = inbox(tmp_path, reader="member")
    fakes = Fakes()
    posted = []
    wake = notifier(
        tmp_path,
        [None],
        fakes,
        channel=nw.ClaudeChannel("available", "/fake", "secret"),
        post=lambda channel, text: posted.append(text) or nw.PostResult(True),
    )
    try:
        nw.watch_member("s1", "member")
        wake.tick()
        assert "call external_read with the member_token you saved" in posted[0]
        assert fakes.queued == []
        assert (directory / "native-wake-member.member.lock").exists()
    finally:
        wake.close()


# --- server: tool, status, prompts ---------------------------------------------


@pytest.fixture
def lead(join_session, monkeypatch, tmp_path):
    sid, directory = join_session
    monkeypatch.setattr(ss, "_codex_lead_host", lambda: HOST)
    monkeypatch.setattr(ss, "_AGENT_NAME", "")
    verified = []

    def verify(home, thread):
        verified.append((home, thread))
        return True, ""

    monkeypatch.setattr(nw, "verify_codex_thread", verify)
    return SimpleNamespace(
        sid=sid, directory=directory, verified=verified, home=str(tmp_path / "codex")
    )


def test_set_lead_wake_registers_human_lead(lead):
    result = asyncio.run(ss.set_lead_wake(THREAD_A, lead.home))
    assert result["success"] is True
    registration = result["lead_wake"]
    assert registration["status"] == "active"
    assert registration["generation"] == 1
    assert lead.verified == [(lead.home, THREAD_A)]
    assert nw.read_lead_wake(lead.directory, "team-lead") == registration


def test_set_lead_wake_default_home_and_clear(lead):
    result = asyncio.run(ss.set_lead_wake(THREAD_A))
    assert result["lead_wake"]["codex_home"] == str(Path.home() / ".codex")
    cleared = asyncio.run(ss.set_lead_wake(""))
    assert cleared["lead_wake"]["status"] == "cleared"
    assert cleared["lead_wake"]["generation"] == 2


@pytest.mark.parametrize(
    ("thread", "home", "reason"),
    [
        ("not-a-uuid", "/codex-home", "invalid_codex_thread_id"),
        (THREAD_A.upper(), "/codex-home", "invalid_codex_thread_id"),
        (THREAD_A, "relative/home", "invalid_codex_home"),
    ],
)
def test_set_lead_wake_rejects_invalid_input_without_writing(
    lead, thread, home, reason
):
    result = asyncio.run(ss.set_lead_wake(thread, home))
    assert result == {"success": False, "reason": reason}
    assert not (lead.directory / "lead-wake-team-lead.json").exists()


def test_set_lead_wake_refuses_non_codex_host(lead, monkeypatch):
    monkeypatch.setattr(ss, "_codex_lead_host", lambda: None)
    result = asyncio.run(ss.set_lead_wake(THREAD_A, lead.home))
    assert result == {"success": False, "reason": "host_not_codex"}
    assert not (lead.directory / "lead-wake-team-lead.json").exists()


def test_set_lead_wake_refuses_unverified_thread(lead, monkeypatch):
    monkeypatch.setattr(nw, "verify_codex_thread", lambda h, t: (False, "archived"))
    result = asyncio.run(ss.set_lead_wake(THREAD_A, lead.home))
    assert result == {
        "success": False,
        "reason": "unverified_thread",
        "detail": "archived",
    }
    assert not (lead.directory / "lead-wake-team-lead.json").exists()


def _add_self_record(directory, name, backend_session_id=None):
    record = {"name": name, "backend": "codex", "status": "running"}
    if backend_session_id:
        record["backend_session_id"] = backend_session_id
    (directory / "agents.json").write_text(json.dumps([record]), encoding="utf-8")


@pytest.mark.parametrize(
    ("bound", "status"),
    [(None, "provisional"), (THREAD_A, "active"), (THREAD_B, "cleared")],
)
def test_set_lead_wake_spawned_lead_needs_parent_binding(
    lead, monkeypatch, bound, status
):
    monkeypatch.setattr(ss, "_AGENT_NAME", "nested")
    monkeypatch.setattr(ss, "IDENTITY", "nested")
    _add_self_record(lead.directory, "nested", bound)
    result = asyncio.run(ss.set_lead_wake(THREAD_A, lead.home))
    assert result["lead_wake"]["status"] == status
    assert (lead.directory / "lead-wake-nested.json").exists()


def test_codex_lead_binding_reads_only_the_own_record(lead, monkeypatch):
    monkeypatch.setattr(ss, "_AGENT_NAME", "nested")
    _add_self_record(lead.directory, "nested", THREAD_A)
    assert ss._codex_lead_binding(lead.sid, "nested") == THREAD_A
    assert ss._codex_lead_binding(lead.sid, "other") is None
    monkeypatch.setattr(ss, "_AGENT_NAME", "")
    assert ss._codex_lead_binding(lead.sid, "nested") is None


def test_session_info_reports_codex_lead(lead, monkeypatch):
    monkeypatch.setattr(
        nw, "resolve_claude_channel", lambda env: nw.ClaudeChannel("no_socket")
    )
    monkeypatch.setattr(ss, "_native_notifier", None)
    before = asyncio.run(ss.session_info())["native_wake"]["codex_lead"]
    assert before == {
        "status": "unregistered",
        "generation": 0,
        "thread_verified": False,
    }
    asyncio.run(ss.set_lead_wake(THREAD_A, lead.home))
    after = asyncio.run(ss.session_info())["native_wake"]["codex_lead"]
    assert after == {"status": "active", "generation": 1, "thread_verified": True}
    monkeypatch.setattr(ss, "_codex_lead_host", lambda: NEW_HOST)
    stale = asyncio.run(ss.session_info())["native_wake"]["codex_lead"]
    assert stale == {"status": "stale_host", "generation": 1, "thread_verified": False}


def test_unregistered_status_needs_no_host_lookup(lead, monkeypatch):
    monkeypatch.setattr(
        ss, "_codex_lead_host", Mock(side_effect=AssertionError("host walked"))
    )
    status = nw.codex_lead_status(lead.directory, "team-lead", ss._codex_lead_host)
    assert status["status"] == "unregistered"


def test_instruction_uses_full_tool_names_and_both_shells():
    text = ss._lead_wake_instruction()
    assert SET_TOOL in text
    assert "$CODEX_THREAD_ID" in text
    assert "$env:CODEX_THREAD_ID" in text
    assert "${CODEX_HOME:-$HOME/.codex}" in text
    assert "Join-Path" in text
    assert "\n" not in text


@pytest.mark.parametrize(
    ("flag", "backend", "enabled", "expected"),
    [
        ("1", "codex", True, True),
        ("1", "codex", False, False),
        ("1", "claude-code", True, False),
        ("", "codex", True, False),
    ],
)
def test_instruction_only_for_flag_on_codex_leads(
    monkeypatch, flag, backend, enabled, expected
):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", flag)
    prompt = ss._with_lead_wake_instruction("task", backend, enabled)
    assert (prompt != "task") is expected
    if expected:
        assert prompt.startswith("task")
        assert SET_TOOL in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["1", ""])
async def test_spawn_prompt_carries_registration_for_codex_lead(
    tmp_path, monkeypatch, flag
):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", flag)
    monkeypatch.setattr(ss.process_manager, "provides_tty", lambda *a, **k: True)
    backend = _FakeBackend()
    monkeypatch.setattr(ss, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(ss, "_session_id", "")
    monkeypatch.setattr(ss, "registry", _FakeRegistry(backend, "codex"))
    await ss.spawn_agent(
        "task",
        name="nested",
        backend="codex",
        cwd=str(tmp_path),
        enable_spawned_lead_wake=True,
    )
    assert backend.last_request is not None
    assert (SET_TOOL in backend.last_request.prompt) is bool(flag)


@pytest.mark.parametrize("flag", ["1", ""])
def test_resume_prompt_carries_registration_for_nested_codex_lead(
    tmp_path, monkeypatch, flag
):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", flag)
    monkeypatch.setattr(ss, "_SESSION_BASE", tmp_path / "sessions")
    (ss._session_dir("s1") / "mcp").mkdir(parents=True)
    agent = {
        "name": "nested",
        "backend": "codex",
        "model": "",
        "enable_spawned_lead_wake": True,
    }
    request = ss._build_resume_request(
        "s1", agent, "nested", str(tmp_path), _FakeBackend(), "codex", "next", "n1"
    )[4]
    assert (SET_TOOL in request.prompt) is bool(flag)
    agent["enable_spawned_lead_wake"] = False
    request = ss._build_resume_request(
        "s1", agent, "nested", str(tmp_path), _FakeBackend(), "codex", "next", "n2"
    )[4]
    assert SET_TOOL not in request.prompt


def test_flag_on_tool_text_for_codex_lead_wake():
    env = os.environ.copy()
    env["WIN_AGENT_TEAMS_NATIVE_WAKE"] = "1"
    code = (
        "import asyncio,json; "
        "from claude_teams import server_simple as s; "
        "print(json.dumps({t.name:t.description "
        "for t in asyncio.run(s.mcp.list_tools())}))"
    )
    result = subprocess.run(  # noqa: S603 - fresh interpreter, no external CLI.
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    tools = json.loads(result.stdout)
    text = tools["set_lead_wake"]
    for literal in (
        "WIN_AGENT_TEAMS_NATIVE_WAKE=1",
        "best-effort",
        "watch",
        "provisional",
        "cleared",
        "generation",
        "host_not_codex",
        "unverified_thread",
        "$CODEX_THREAD_ID",
        READ_TOOL,
    ):
        assert literal in text
    for name in ("session_info", "resume_session"):
        assert "codex_lead" in tools[name] or "set_lead_wake" in tools[name]
        assert SET_TOOL in tools[name]
    assert "codex_lead" in tools["session_info"]
    assert "set_lead_wake" in tools["spawn_agent"]


def test_flag_off_has_no_set_lead_wake_tool(monkeypatch):
    env = os.environ.copy()
    env.pop("WIN_AGENT_TEAMS_NATIVE_WAKE", None)
    code = (
        "import asyncio,json; "
        "from claude_teams import server_simple as s; "
        "print(json.dumps([t.name for t in asyncio.run(s.mcp.list_tools())]))"
    )
    result = subprocess.run(  # noqa: S603 - fresh interpreter, no external CLI.
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "set_lead_wake" not in json.loads(result.stdout)
