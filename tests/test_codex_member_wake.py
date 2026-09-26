"""Fake-only Codex registration, queue, verification, and race contracts."""

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claude_teams import native_wake as nw
from claude_teams import server_simple as ss
from tests import test_join_team

join_session = test_join_team.join_session

THREAD_A = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
THREAD_B = "bbbbbbbb-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def run(coro):
    return asyncio.run(coro)


def rollout(home, tid=THREAD_A):
    directory = home / "sessions/2026/09/26"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rollout-2026-09-26-{tid}.jsonl"
    path.touch()
    return path


@pytest.fixture
def member(join_session, monkeypatch, tmp_path):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", raising=False)
    sid, directory = join_session
    ticket = run(ss.create_join_ticket("member"))
    joined = run(ss.join_team(sid, ticket["token"]))
    home = tmp_path / "codex-home"
    rollout(home)
    calls = []
    clock = [0.0]

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    wake = nw.CodexMemberWake(
        runner=runner, discover=lambda: "fake-codex", clock=lambda: clock[0]
    )
    monkeypatch.setattr(ss, "_codex_member_wake", wake)
    return SimpleNamespace(
        sid=sid,
        directory=directory,
        token=joined["member_token"],
        home=home,
        calls=calls,
        wake=wake,
        clock=clock,
    )


def register(member, tid=THREAD_A):
    return run(ss.external_set_wake(member.token, tid, str(member.home)))


def send():
    return run(ss.send_message("PRIVATE BODY", to="member"))


def test_registration_updates_clears_validation_left(member):
    first = register(member)
    assert first["success"]
    assert first["name"] == "member"
    assert first["codex_wake"]["generation"] == 1
    assert register(member, THREAD_B)["codex_wake"]["generation"] == 2
    cleared = register(member, "")
    assert cleared["codex_wake"]["thread_id"] is None
    assert cleared["codex_wake"]["generation"] == 3
    before = (member.directory / "agents.json").read_bytes()
    for tid, home, reason in [
        ("bad", str(member.home), "invalid_codex_thread_id"),
        (THREAD_A.upper(), str(member.home), "invalid_codex_thread_id"),
        (THREAD_A, "relative", "invalid_codex_home"),
    ]:
        result = run(ss.external_set_wake(member.token, tid, home))
        assert result == {"success": False, "reason": reason}
        assert (member.directory / "agents.json").read_bytes() == before
    assert run(ss.external_set_wake(member.token, THREAD_A, ""))["codex_wake"][
        "codex_home"
    ] == str(Path.home() / ".codex")
    run(ss.leave_team(member.token))
    assert not register(member)["success"]


def test_registration_import_gate_external_only(monkeypatch):
    env = os.environ.copy()
    env["WIN_AGENT_TEAMS_NATIVE_WAKE"] = "1"
    env["WIN_AGENT_TEAMS_EXTERNAL_ONLY"] = "1"
    code = (
        "import asyncio,json; "
        "from claude_teams.server_simple import mcp; "
        "print(json.dumps([t.name for t in asyncio.run(mcp.list_tools())]))"
    )
    result = subprocess.run(  # noqa: S603 - fresh interpreter, no external CLI.
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "external_set_wake" in json.loads(result.stdout)


def test_single_send_immediate_safe_shim_no_timer(member, monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_RENOTIFY_SECONDS", "0.01")
    register(member)
    member.wake.discover = lambda: "codex.cmd"
    result = send()
    assert result["success"]
    assert result["delivery"] == "inbox"
    assert result["wake"] == {"method": "codex_queue", "status": "queued"}
    argv, kwargs = member.calls[0]
    assert argv[:4] == ["codex.cmd", "queue", "--thread", THREAD_A]
    assert argv[4] == "--message"
    notice = argv[5]
    assert not re.search(r"""[()<>|&^%!"'\n]""", notice)
    assert "PRIVATE BODY" not in notice
    assert member.token not in notice
    assert "external_read" in notice
    assert kwargs["env"]["CODEX_HOME"] == str(member.home)
    assert kwargs["cwd"] == Path.home()
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] == 15
    assert kwargs["encoding"] == "utf-8"
    time.sleep(2.05)
    assert len(member.calls) == 1


def test_failed_then_retry_backoff_reset(member):
    register(member)
    runner = Mock(
        side_effect=[
            SimpleNamespace(returncode=1, stderr="failure"),
            SimpleNamespace(returncode=0, stderr=""),
        ]
    )
    member.wake.runner = runner
    assert send()["wake"]["status"] == "failed"
    assert send()["wake"]["status"] == "backoff"
    member.clock[0] = 2
    assert send()["wake"]["status"] == "queued"
    assert runner.call_count == 2


def test_concurrent_sends_coalesce(member):
    register(member)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: send(), range(2)))
    assert len(member.calls) == 1
    assert sorted(r["wake"]["status"] for r in results) == ["coalesced", "queued"]


def test_drain_then_send_again(member):
    register(member)
    assert send()["wake"]["status"] == "queued"
    assert run(ss.external_read(member.token))["messages"]
    assert send()["wake"]["status"] == "queued"
    assert len(member.calls) == 2


def test_generation_state_clear_and_failure(member):
    register(member)
    rollout(member.home, THREAD_B)
    assert send()["wake"]["status"] == "queued"
    assert register(member, THREAD_B)["codex_wake"]["generation"] == 2
    assert send()["wake"]["status"] == "queued"
    assert member.calls[-1][0][3] == THREAD_B
    register(member, "")
    verify = Mock(side_effect=AssertionError("verification after clear"))
    runner = Mock(side_effect=AssertionError("queue after clear"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nw, "verify_codex_thread", verify)
        patch.setattr(member.wake, "runner", runner)
        assert "wake" not in send()  # R4-A: truthy tombstones are not registrations.
    assert register(member)["codex_wake"]["generation"] == 4
    member.wake.runner = Mock(
        return_value=SimpleNamespace(returncode=1, stderr="failure")
    )
    assert send()["wake"]["status"] == "failed"
    register(member, THREAD_B)
    member.wake.runner = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    assert send()["wake"]["status"] == "queued"
    member.wake.runner.assert_called_once()


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        ("timeout", "timeout"),
        ("failed", "failed"),
        ("missing", "unavailable"),
        ("oserror", "failed"),
    ],
)
def test_queue_failures_preserve_send(member, outcome, status):
    register(member)
    if outcome == "missing":
        from claude_teams.backends.contracts import BackendBinaryNotFoundError

        member.wake.discover = Mock(
            side_effect=BackendBinaryNotFoundError("codex", "codex")
        )
    elif outcome == "timeout":
        member.wake.runner = Mock(side_effect=subprocess.TimeoutExpired(["fake"], 15))
    elif outcome == "oserror":
        member.wake.runner = Mock(side_effect=OSError("failure"))
    else:
        member.wake.runner = Mock(
            return_value=SimpleNamespace(returncode=1, stderr="x" * 500)
        )
    result = send()
    assert result["success"]
    assert result["wake"]["status"] == status
    assert len(result["wake"].get("detail", "")) <= 200


def test_queue_agents_lock_free_and_member_lock_order(member, monkeypatch):
    register(member)
    local = threading.local()
    original = ss._agents_file_lock

    @contextmanager
    def agents_lock(sid):
        with original(sid):
            local.held = True
            try:
                yield
            finally:
                local.held = False

    monkeypatch.setattr(ss, "_agents_file_lock", agents_lock)
    original_lock = member.wake.member_lock

    def member_lock(key):
        assert not getattr(local, "held", False)
        return original_lock(key)

    monkeypatch.setattr(member.wake, "member_lock", member_lock)

    def runner(argv, **kwargs):
        assert not getattr(local, "held", False)
        ready = threading.Event()

        def probe():
            with original(member.sid):
                ready.set()

        thread = threading.Thread(target=probe)
        thread.start()
        assert ready.wait(0.5)
        thread.join(1)
        return SimpleNamespace(returncode=0, stderr="")

    member.wake.runner = runner
    assert send()["wake"]["status"] == "queued"


def test_subswitch(member, monkeypatch):
    register(member)
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", "0")
    assert send()["wake"]["status"] == "disabled"
    assert not member.calls


def test_reregister_while_first_queue_running(member):
    register(member)
    rollout(member.home, THREAD_B)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return SimpleNamespace(returncode=0, stderr="")

    member.wake.runner = runner
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(send)
        assert entered.wait(2)
        register(member, THREAD_B)
        second = pool.submit(send)
        release.set()
        assert first.result(timeout=3)["wake"]["status"] == "queued"
        assert second.result(timeout=3)["wake"]["status"] == "queued"
    assert [argv[3] for argv in calls] == [THREAD_A, THREAD_B]


@pytest.mark.parametrize(
    ("operation", "status"), [("leave", "coalesced"), ("change", "stale_registration")]
)
def test_revalidate_before_subprocess(member, monkeypatch, operation, status):
    register(member)
    original = nw.verify_codex_thread

    def verify(home, tid):
        result = original(home, tid)
        if operation == "leave":
            run(ss.leave_team(member.token))
        else:
            register(member, THREAD_B)
        return result

    monkeypatch.setattr(nw, "verify_codex_thread", verify)
    assert send()["wake"]["status"] == status
    assert not member.calls
    if operation == "leave":
        assert send()["reason"] == "member_left"


def test_registration_send_stress_no_deadlock(member):
    register(member)
    rollout(member.home, THREAD_B)
    errors = []

    def updates():
        try:
            for i in range(200):
                register(member, THREAD_A if i % 2 else THREAD_B)
        except Exception as err:
            errors.append(err)

    def sends():
        try:
            for _ in range(200):
                assert send()["success"]
        except Exception as err:
            errors.append(err)

    threads = [
        threading.Thread(target=updates, daemon=True),
        threading.Thread(target=sends, daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive()
    assert not errors


def database(home, tid=THREAD_A, archived=0, schema=True, wal=False):
    home.mkdir(exist_ok=True)
    connection = sqlite3.connect(home / "state_5.sqlite")
    if wal:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
    columns = "id TEXT, rollout_path TEXT" + (", archived INTEGER" if schema else "")
    connection.execute(f"CREATE TABLE threads ({columns})")
    path = rollout(home, tid)
    if schema:
        connection.execute(
            "INSERT INTO threads VALUES (?, ?, ?)", (tid, str(path), archived)
        )
    else:
        connection.execute("INSERT INTO threads VALUES (?, ?)", (tid, str(path)))
    connection.commit()
    return connection


def files(home):
    return {
        str(p.relative_to(home)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in home.rglob("*")
    }


@pytest.mark.parametrize(
    "mode",
    [
        "db",
        "wal",
        "glob",
        "schema",
        "locked",
        "archived",
        "archived_path",
        "mismatch",
        "missing",
    ],
)
def test_verification_readonly_wal_fallback_archive(tmp_path, mode):
    home = tmp_path / "home"
    connection = None
    if mode in {"db", "wal", "schema", "locked", "archived", "archived_path"}:
        connection = database(
            home,
            archived=int(mode == "archived"),
            schema=mode != "schema",
            wal=mode == "wal",
        )
        if mode == "locked":
            connection.execute("BEGIN EXCLUSIVE")
        if mode == "archived_path":
            connection.execute(
                "UPDATE threads SET rollout_path=?",
                (str(home / "archived_sessions/a.jsonl"),),
            )
            connection.commit()
    elif mode != "missing":
        rollout(home, THREAD_B if mode == "mismatch" else THREAD_A)
    before = files(home) if home.exists() else {}
    try:
        ok, detail = nw.verify_codex_thread(str(home), THREAD_A)
        assert ok is (mode not in {"archived", "archived_path", "mismatch", "missing"})
        assert files(home) == before
        if mode in {"archived", "archived_path"}:
            assert detail == "archived"
    finally:
        if connection is not None:
            connection.close()
    assert (
        not (home / "state_5.sqlite").exists()
        if mode in {"glob", "mismatch", "missing"}
        else True
    )


def test_verification_closed_before_queue(member, monkeypatch):
    connection = database(member.home)
    connection.close()
    original = nw.sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        assert kwargs["timeout"] == 0.5
        assert kwargs["uri"]
        assert "mode=ro" in args[0]
        assert "immutable" not in args[0]
        result = original(*args, **kwargs)
        connections.append(result)
        return result

    monkeypatch.setattr(nw.sqlite3, "connect", connect)

    def runner(*args, **kwargs):
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
        return SimpleNamespace(returncode=0, stderr="")

    member.wake.runner = runner
    register(member)
    assert send()["wake"]["status"] == "queued"


def test_snapshot_failure_never_fails_successful_append(member, monkeypatch):
    register(member)
    monkeypatch.setattr(
        ss, "_native_member_snapshot", Mock(side_effect=OSError("snapshot unavailable"))
    )
    result = send()
    assert result["success"]
    assert result["wake"]["status"] == "failed"
    assert not member.calls
    assert "PRIVATE BODY" in (member.directory / "inbox-member.jsonl").read_text()


def test_removed_member_between_append_and_snapshot_coalesces(member, monkeypatch):
    register(member)
    monkeypatch.setattr(ss, "_native_member_snapshot", lambda sid, name: (None, {}))
    assert send()["wake"]["status"] == "coalesced"
    assert not member.calls


def test_slow_queue_backoff_starts_after_completion(member):
    register(member)
    calls = []

    def runner(argv, **kwargs):
        calls.append(member.clock[0])
        member.clock[0] += 15
        raise subprocess.TimeoutExpired(argv, 15)

    member.wake.runner = runner
    assert send()["wake"]["status"] == "timeout"
    assert send()["wake"]["status"] == "backoff"
    assert calls == [0]
    member.clock[0] = 17
    assert send()["wake"]["status"] == "timeout"
    assert calls == [0, 17]


def test_queue_environment_drops_lead_channel_and_identity(member, monkeypatch):
    inherited = {
        "CLAUDE_CODE_MESSAGING_SOCKET": "/inherited.sock",
        "CLAUDE_CODE_MESSAGING_TOKEN": "inherited-secret",
        "AGENT_NAME": "lead",
        "AGENT_SESSION_ID": member.sid,
        "AGENT_PARENT_NAME": "parent",
        "AGENT_CAPABILITY": "capability",
        "WIN_AGENT_TEAMS_SESSION_DIR": str(member.directory),
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("UNRELATED_ENV", "retained")
    register(member)
    assert send()["wake"]["status"] == "queued"
    environ = member.calls[0][1]["env"]
    assert not set(inherited).intersection(environ)
    assert environ["UNRELATED_ENV"] == "retained"
    assert environ["CODEX_HOME"] == str(member.home)
    for key, value in inherited.items():
        assert ss.os.environ[key] == value
