"""Native Claude channel, notice policy, and lifetime ownership contracts."""

import asyncio
import errno
import json
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claude_teams import filelock, procinfo
from claude_teams import native_wake as nw
from claude_teams import server_simple as ss
from tests import test_join_team

join_session = test_join_team.join_session


@pytest.fixture(autouse=True)
def wake_on(monkeypatch):
    monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", "1")
    # H1-H5 are Linux rules, independent of the runner's OS. Windows tests
    # override this module-local platform view to exercise H1b explicitly.
    monkeypatch.setattr(nw, "os", SimpleNamespace(name="posix", environ=ss.os.environ))
    monkeypatch.setattr(nw, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", raising=False)
    nw._activation.clear()
    nw._members.clear()


def host(kind="claude", pid=123):
    return lambda: procinfo.HostResolution((), procinfo.ProcessInfo(pid, 1, kind))


def env(path):
    return {
        "WIN_AGENT_TEAMS_NATIVE_WAKE": "1",
        "CLAUDE_CODE_MESSAGING_SOCKET": str(path),
        "CLAUDE_CODE_MESSAGING_TOKEN": "secret",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("true", False),
        ("0", False),
        ("1", True),
        (" 1 ", True),
    ],
)
def test_gate(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("WIN_AGENT_TEAMS_NATIVE_WAKE")
    else:
        monkeypatch.setenv("WIN_AGENT_TEAMS_NATIVE_WAKE", raw)
    assert nw.enabled() is expected
    assert nw.resolve_claude_channel(
        {"WIN_AGENT_TEAMS_NATIVE_WAKE": raw or ""},
        resolve_host=Mock(side_effect=AssertionError()),
    ).reason == ("no_socket" if expected else "disabled")


@pytest.mark.parametrize("half", ["CLAUDE", "CODEX"])
def test_subswitch(half, monkeypatch):
    monkeypatch.setenv(f"WIN_AGENT_TEAMS_NATIVE_WAKE_{half}", "0")
    assert not nw.enabled(half)


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "x", ""])
def test_seconds_parser(raw):
    assert nw.positive_seconds(raw, 2) == 2
    assert nw.positive_seconds("1.5", 2) == 1.5


@pytest.mark.parametrize(
    "missing", ["CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN"]
)
def test_no_socket(missing):
    values = env("/unused")
    values[missing] = ""
    assert (
        nw.resolve_claude_channel(
            values, resolve_host=Mock(side_effect=AssertionError())
        ).reason
        == "no_socket"
    )


@pytest.mark.parametrize("kind", ["codex", "pi"])
def test_nearest_host_guard(kind):
    assert (
        nw.resolve_claude_channel(env("/123.sock"), resolve_host=host(kind)).reason
        == "host_not_claude"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sockets")
def test_socket_ownership(tmp_path):
    assert (
        nw.resolve_claude_channel(
            env(tmp_path / "999.sock"), resolve_host=host()
        ).reason
        == "socket_not_owned"
    )
    assert (
        nw.resolve_claude_channel(
            env(tmp_path / "123.sock"), resolve_host=host()
        ).reason
        == "socket_missing"
    )
    for name, verified in [("123.sock", True), ("other.sock", False)]:
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(tmp_path / name))
            channel = nw.resolve_claude_channel(
                env(tmp_path / name), resolve_host=host()
            )
            assert channel.reason == "available"
            assert channel.owner_verified is verified


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sockets")
def test_wire_two_lines_then_eof(tmp_path):
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(tmp_path / "123.sock"))
        listener.listen()
        listener.settimeout(2)
        channel = nw.resolve_claude_channel(
            env(tmp_path / "123.sock"), resolve_host=host()
        )
        assert nw.post_claude_notice(channel, "notice").ok
        with listener.accept()[0] as connection:
            connection.settimeout(2)
            data = b""
            while chunk := connection.recv(4096):
                data += chunk
    assert [json.loads(line) for line in data.splitlines()] == [
        {"type": "auth", "token": "secret"},
        {"type": "user", "message": {"role": "user", "content": "notice"}},
    ]


@pytest.mark.parametrize("with_env", [False, True])
def test_windows_short_circuit(with_env, monkeypatch):
    resolver = Mock(side_effect=AssertionError())
    with monkeypatch.context() as patch:
        patch.setattr(nw, "os", SimpleNamespace(name="nt", environ=ss.os.environ))
        patch.setattr(nw.socket, "socket", Mock(side_effect=AssertionError()))
        values = env("/123.sock") if with_env else {"WIN_AGENT_TEAMS_NATIVE_WAKE": "1"}
        assert (
            nw.resolve_claude_channel(values, resolve_host=resolver).reason
            == "unsupported_platform"
        )
        patch.setattr(type(ss.mcp), "run", lambda self: None)
        patch.setattr(
            nw.NativeWakeNotifier, "start", Mock(side_effect=AssertionError())
        )
        ss.main()
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "error", [OSError("failed"), TimeoutError("late"), ValueError("bad")]
)
def test_post_errors_never_escape(error, monkeypatch):
    monkeypatch.setattr(nw.socket, "socket", Mock(side_effect=error))
    result = nw.post_claude_notice(
        nw.ClaudeChannel("available", "/fake", "secret"), "notice"
    )
    assert not result.ok
    assert "secret" not in result.reason


def test_policy_coalesce_outstanding_floor_drain_baseline():
    state = nw.NoticeState(notified={"alice": 0}, first_new=0)
    snap = {"alice": {"total": 2, "cursor": 0}}
    assert nw.plan_notice(state, snap, 0, nw.NoticeConfig()) is None
    notice = nw.plan_notice(state, snap, 2, nw.NoticeConfig())
    assert notice
    assert notice.counts == {"alice": 2}
    assert state.seq == 0  # A failed post does not commit the proposal.
    state.succeeded(snap, 2)
    snap["alice"]["total"] = 3
    assert nw.plan_notice(state, snap, 3, nw.NoticeConfig(coalesce=0)) is None
    assert nw.plan_notice(state, snap, 303, nw.NoticeConfig(coalesce=0))
    snap["alice"]["cursor"] = 2
    assert nw.plan_notice(state, snap, 304, nw.NoticeConfig(coalesce=0))
    fresh = nw.NoticeState(notified={"alice": 2})
    assert nw.plan_notice(fresh, snap, 304, nw.NoticeConfig(coalesce=0))


def test_outstanding_without_growth_renotifies():
    state = nw.NoticeState(notified={"alice": 2}, last_success=0)
    snap = {"alice": {"total": 2, "cursor": 0}}
    assert nw.plan_notice(state, snap, 300, nw.NoticeConfig(coalesce=0))


def test_notice_has_counts_sequence_no_content():
    state = nw.NoticeState()
    snap = {"alice": {"total": 2, "cursor": 0}}
    proposal = nw.plan_notice(state, snap, 0, nw.NoticeConfig(coalesce=0))
    assert proposal is not None
    first = proposal.text
    assert "alice (2)" in first
    assert "#1" in first
    assert "read_messages" in first
    state.succeeded(snap, 0)
    proposal = nw.plan_notice(state, snap, 300, nw.NoticeConfig(coalesce=0))
    assert proposal is not None
    second = proposal.text
    assert first != second
    assert "secret" not in first


def make_notifier(tmp_path, active, **kwargs):
    return nw.NativeWakeNotifier(
        get_target=lambda: active[0],
        session_dir=lambda sid: tmp_path / sid,
        member_alive=lambda sid, name: True,
        channel=nw.ClaudeChannel("available", "/fake", "secret"),
        **kwargs,
    )


def inbox(tmp_path, sid="s1", reader="team-lead"):
    directory = tmp_path / sid
    directory.mkdir(exist_ok=True)
    (directory / f"inbox-{reader}.jsonl").write_text(
        json.dumps({"from": "alice", "text": "PRIVATE BODY"}) + "\n"
    )
    return directory


def test_backoff_every_error_then_reset(tmp_path):
    inbox(tmp_path)
    active = [("s1", "team-lead")]
    calls = []
    now = [0.0]

    def post(channel, text):
        calls.append(now[0])
        return nw.PostResult(len(calls) == 5, "failed")

    notifier = make_notifier(tmp_path, active, post=post, clock=lambda: now[0])
    try:
        for tick in range(31):
            now[0] = tick
            notifier.tick()
        assert calls == [0, 2, 6, 14, 30]
        target = notifier.targets[("s1", "team-lead")]
        assert target.state.seq == 1
        assert target.backoff.delay == 0
    finally:
        notifier.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner locks")
def test_two_notifiers_switch_releases_and_stdout(tmp_path, capsys):
    inbox(tmp_path)
    inbox(tmp_path, "s2")
    active = [("s1", "team-lead")]
    calls = []
    first = make_notifier(
        tmp_path, active, post=lambda c, t: calls.append(t) or nw.PostResult(True)
    )
    second = make_notifier(
        tmp_path, active, post=lambda c, t: calls.append(t) or nw.PostResult(True)
    )
    try:
        first.tick()
        second.tick()
        assert len(calls) == 1
        old = first.targets[("s1", "team-lead")].handle
        active[0] = ("s2", "team-lead")
        first.tick()
        assert old.closed
        assert len(calls) == 2
        assert not capsys.readouterr().out
    finally:
        first.close()
        second.close()


def test_no_target_no_recovery_member_drop(tmp_path, monkeypatch):
    active = [None]
    monkeypatch.setattr(ss, "_recover_session_id", Mock(side_effect=AssertionError()))
    notifier = make_notifier(tmp_path, active)
    notifier.tick()
    assert not notifier.targets
    inbox(tmp_path, reader="member")
    nw.watch_member("s1", "member")
    notifier.member_alive = lambda sid, name: False
    notifier.tick()
    assert not notifier.targets
    notifier.close()


def test_filelock_free_held_and_error(tmp_path, monkeypatch):
    with (tmp_path / "lock").open("a+b") as a, (tmp_path / "lock").open("a+b") as b:
        assert filelock.try_lock_handle(a)
        assert not filelock.try_lock_handle(b)
        filelock.unlock_handle(a)
        assert filelock.try_lock_handle(b)
        filelock.unlock_handle(b)
    if sys.platform != "win32":
        monkeypatch.setattr(
            filelock.fcntl, "flock", Mock(side_effect=OSError(errno.EIO, "broken"))
        )
        with (
            (tmp_path / "lock").open("a+b") as handle,
            pytest.raises(OSError, match="broken"),
        ):
            filelock.try_lock_handle(handle)


def test_windows_lock_seek_contention_and_noncontention(monkeypatch):
    handle = Mock()
    locking = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(filelock, "os", SimpleNamespace(name="nt"))
        patch.setattr(
            filelock,
            "msvcrt",
            SimpleNamespace(locking=locking, LK_NBLCK=1),
            raising=False,
        )
        assert filelock.try_lock_handle(handle)
        handle.seek.assert_called_once_with(0)
        locking.side_effect = OSError(errno.EACCES, "busy")
        assert not filelock.try_lock_handle(handle)
        locking.side_effect = OSError(errno.EIO, "broken")
        with pytest.raises(OSError, match="broken"):
            filelock.try_lock_handle(handle)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocess lock")
def test_owner_death_releases_lock(tmp_path):
    path = tmp_path / "lock"
    code = (
        "import sys,time; "
        "from pathlib import Path; "
        "from claude_teams.filelock import lock_handle; "
        'h=Path(sys.argv[1]).open("a+b"); '
        "lock_handle(h); "
        'print("ready",flush=True); '
        "time.sleep(30)"
    )
    child = subprocess.Popen(  # noqa: S603 - fake lock holder.
        [sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with path.open("a+b") as handle:
            assert not filelock.try_lock_handle(handle)
            child.kill()
            child.wait(timeout=5)
            assert filelock.try_lock_handle(handle)
            filelock.unlock_handle(handle)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_activation_catchup_repeated_rapid_and_during_scan(tmp_path):
    inbox(tmp_path)
    inbox(tmp_path, "s2")
    active = [None]
    calls = []
    notifier = make_notifier(
        tmp_path, active, post=lambda c, t: calls.append(t) or nw.PostResult(True)
    )
    try:
        notifier.tick()
        active[0] = ("s1", "team-lead")
        nw.session_activated("s1")
        notifier.tick()
        target = notifier.targets[active[0]]
        target.backoff.failed(0)
        original = (target.state, target.backoff, target.handle)
        for _ in range(2):
            nw.session_activated("s1")
            notifier.tick()
        assert len(calls) == 1
        assert (target.state, target.backoff, target.handle) == original
        active[0] = ("s1", "team-lead")
        nw.session_activated("s1")
        active[0] = ("s2", "team-lead")
        nw.session_activated("s2")
        notifier.tick()
        assert len(calls) == 2
        assert target.handle is None
        assert ("s1", "team-lead") not in notifier.targets
    finally:
        notifier.close()


def test_event_during_scan_survives(tmp_path, monkeypatch):
    inbox(tmp_path)
    inbox(tmp_path, "s2")
    active = [("s1", "team-lead")]
    calls = []
    original = nw.scan_inbox

    def scan(directory, reader):
        if directory.name == "s1":
            active[0] = ("s2", "team-lead")
            nw.session_activated("s2")
        return original(directory, reader)

    monkeypatch.setattr(nw, "scan_inbox", scan)
    notifier = make_notifier(
        tmp_path, active, post=lambda c, t: calls.append(t) or nw.PostResult(True)
    )
    try:
        notifier.tick()
        assert nw._activation.is_set()
        assert not calls
        notifier.tick()
        assert len(calls) == 1
        assert ("s2", "team-lead") in notifier.targets
    finally:
        notifier.close()


def test_resume_auto_adopt_signals_and_wakes_before_tick(join_session, monkeypatch):
    sid, directory = join_session
    inbox(directory.parent, sid)
    monkeypatch.setattr(ss, "_persist_session_binding", lambda sid: None)
    idle = threading.Event()

    class ActivationEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            # run() has finished its first empty tick and reached the long wait.
            idle.set()
            return super().wait(timeout)

    for resume in (True, False):
        monkeypatch.setattr(ss, "_session_id", "")
        monkeypatch.setattr(ss, "_recover_session_id", lambda: sid)
        monkeypatch.setattr(nw, "_activation", ActivationEvent())
        idle.clear()
        posted = threading.Event()
        notices = []

        def post(channel, text, notices=notices, posted=posted):
            notices.append(text)
            posted.set()
            return nw.PostResult(True)

        notifier = nw.NativeWakeNotifier(
            get_target=ss._native_wake_target,
            session_dir=ss._session_dir,
            member_alive=lambda sid, name: True,
            channel=nw.ClaudeChannel("available", "/fake", "secret"),
            post=post,
            poll=30,
        )
        notifier.start()
        try:
            assert idle.wait(1)
            assert not notifier.targets
            if resume:
                assert asyncio.run(ss.resume_session(sid))["success"]
            else:
                assert ss._active_session_id() == sid
            assert posted.wait(1)
            assert len(notices) == 1
            assert "alice (1)" in notices[0]
        finally:
            notifier.close()

    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", True)
    assert ss._native_wake_target() is None
    monkeypatch.setattr(ss, "_IDENTITY_UNRESOLVED", False)
    monkeypatch.setenv("WIN_AGENT_TEAMS_EXTERNAL_ONLY", "1")
    assert ss._native_wake_target() is None


def test_unchanged_inbox_scans_only_on_deadline_or_activation(tmp_path, monkeypatch):
    inbox(tmp_path)
    active = [("s1", "team-lead")]
    scan = Mock(wraps=nw.scan_inbox)
    monkeypatch.setattr(nw, "scan_inbox", scan)
    now = [0.0]
    notifier = make_notifier(
        tmp_path, active, clock=lambda: now[0], post=lambda c, t: nw.PostResult(True)
    )
    try:
        notifier.tick()
        for tick in range(1, 10):
            now[0] = tick
            notifier.tick()
        assert scan.call_count == 1
        now[0] = 300
        notifier.tick()
        assert scan.call_count == 2
    finally:
        notifier.close()


def test_policy_is_pure():
    from dataclasses import asdict

    state = nw.NoticeState()
    snapshot = {"alice": {"total": 2, "cursor": 0}}
    before = asdict(state)
    assert nw.plan_notice(state, snapshot, 0, nw.NoticeConfig()) is None
    assert asdict(state) == before


def test_slow_post_backoff_starts_after_completion(tmp_path):
    inbox(tmp_path)
    active = [("s1", "team-lead")]
    now = [0.0]
    calls = []

    def post(channel, text):
        calls.append(now[0])
        now[0] += 5
        return nw.PostResult(False, "timeout")

    notifier = make_notifier(tmp_path, active, clock=lambda: now[0], post=post)
    try:
        notifier.tick()
        notifier.tick()
        assert len(calls) == 1
        now[0] = 7
        notifier.tick()
        assert calls == [0, 7]
    finally:
        notifier.close()


def test_macos_channel_unavailable_without_host_lookup(monkeypatch):
    monkeypatch.setattr(nw, "sys", SimpleNamespace(platform="darwin"), raising=False)
    resolver = Mock(side_effect=AssertionError("host lookup on macOS"))
    assert (
        nw.resolve_claude_channel(env("/123.sock"), resolve_host=resolver).reason
        == "unsupported_platform"
    )
    resolver.assert_not_called()
    monkeypatch.setattr(type(ss.mcp), "run", lambda self: None)
    monkeypatch.setattr(
        nw.NativeWakeNotifier,
        "start",
        Mock(side_effect=AssertionError("macOS notifier started")),
    )
    ss.main()


def test_missing_proc_has_no_claude_host(tmp_path):
    def resolve():
        return procinfo._walk(
            123,
            lambda pid: procinfo._read_linux_process(
                pid, proc_root=tmp_path / "missing-proc"
            ),
        )

    assert (
        nw.resolve_claude_channel(env("/123.sock"), resolve_host=resolve).reason
        == "host_not_claude"
    )


@pytest.mark.parametrize(
    "reason", ["disabled", "no_socket", "host_not_claude", "unsupported_platform"]
)
def test_unavailable_channel_drains_members_without_registry_reads(
    tmp_path, monkeypatch, reason
):
    nw.watch_member("s1", "member")
    registry_read = Mock(
        side_effect=AssertionError("unavailable channel read registry")
    )
    monkeypatch.setattr(ss, "_load_agents", registry_read)
    notifier = nw.NativeWakeNotifier(
        get_target=lambda: None,
        session_dir=lambda sid: tmp_path / sid,
        member_alive=ss._native_member_alive,
        channel=nw.ClaudeChannel(reason),
        post=Mock(side_effect=AssertionError("unavailable channel posted")),
    )
    try:
        notifier.tick()
        notifier.tick()
        registry_read.assert_not_called()
        assert not nw._members
        assert not notifier.targets
        assert not list(tmp_path.rglob("*.lock"))
    finally:
        notifier.close()


def test_member_target_posts_catchup_then_coalesced_growth(tmp_path):
    directory = inbox(tmp_path, reader="member")
    calls = []
    now = [0.0]
    notifier = make_notifier(
        tmp_path,
        [None],
        clock=lambda: now[0],
        post=lambda c, t: calls.append(t) or nw.PostResult(True),
    )
    try:
        nw.watch_member("s1", "member")
        notifier.tick()
        assert len(calls) == 1  # Baseline catch-up bypasses coalescing.
        assert "alice (1)" in calls[0]
        assert "call external_read with the member_token you saved" in calls[0]
        lock = directory / "native-wake-member.member.lock"
        assert lock.exists()
        assert notifier.owns(("s1", "member"))
        (directory / "inbox-member.pos.json").write_text(json.dumps({"alice": 1}))
        with (directory / "inbox-member.jsonl").open("a") as handle:
            handle.write(json.dumps({"from": "alice", "text": "next"}) + "\n")
        now[0] = 1
        notifier.tick()
        assert len(calls) == 1
        now[0] = 3
        notifier.tick()
        assert len(calls) == 2
        assert "#2" in calls[1]
    finally:
        notifier.close()


def test_lead_and_member_lock_names_cannot_collide(tmp_path):
    directory = inbox(tmp_path, reader="member-alice")
    inbox(tmp_path, reader="alice")
    calls = []
    notifier = make_notifier(
        tmp_path,
        [("s1", "member-alice")],
        post=lambda c, t: calls.append(t) or nw.PostResult(True),
    )
    try:
        nw.watch_member("s1", "alice")
        notifier.tick()
        assert len(calls) == 2
        assert (directory / "native-wake-lead.member-alice.lock").exists()
        assert (directory / "native-wake-member.alice.lock").exists()
    finally:
        notifier.close()
