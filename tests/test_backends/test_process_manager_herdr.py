"""Tests for the Herdr Linux launcher.

Herdr is a terminal workspace manager that gives each spawned agent its own
tab. These tests drive the two subprocess seams the manager defines —
``_run_herdr`` for finite JSON control commands and ``_popen_herdr_server``
for the long-running daemon — so nothing here needs a live Herdr server.

The ownership tests are the load-bearing ones: ``_PidOwnershipMixin``
short-circuits to "ours" the moment ``_tracked_alive`` is true, so this
manager's probe *is* the PID-reuse proof for everything that kills.
"""

import functools
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from claude_teams.backends import process_manager as pm
from claude_teams.backends.contracts import SpawnRequest, SpawnResult
from claude_teams.filelock import file_lock


@pytest.fixture
def _request(tmp_path: Path) -> SpawnRequest:
    return SpawnRequest(
        agent_id="worker@team",
        name="worker",
        team_name="team",
        prompt="do stuff",
        model="default",
        agent_type="general-purpose",
        color="blue",
        cwd=str(tmp_path),
        lead_session_id="sess-1",
    )


# --------------------------------------------------------------------------
# Selection and constructor purity (plan tests 1-6)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("herdr", pm.HerdrProcessManager),
        ("HERDR", pm.HerdrProcessManager),
        ("  herdr  ", pm.HerdrProcessManager),
        ("tmux", pm.TmuxProcessManager),
        ("terminal", pm.LinuxTerminalProcessManager),
        ("", pm.LinuxTerminalProcessManager),
        ("nonsense", pm.LinuxTerminalProcessManager),
    ],
)
def test_launcher_selection(value: str, expected: type) -> None:
    """The launcher is chosen purely from the configured value."""
    assert pm._select_linux_manager(value) is expected


def test_herdr_env_alone_does_not_select_herdr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running *inside* Herdr is context, not consent to switch launchers.

    Auto-detecting would silently change the launcher for anyone whose MCP
    server happens to start in a Herdr pane.
    """
    monkeypatch.setenv("HERDR_ENV", "1")
    assert pm._select_linux_manager("") is pm.LinuxTerminalProcessManager


def test_constructor_runs_no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction is pure: no ``which``, no probe, no daemon.

    The manager is built at import time, so a constructor that shelled out
    would make merely importing the module mutate state.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        msg = "constructor must not run a subprocess"
        raise AssertionError(msg)

    monkeypatch.setattr(pm.subprocess, "run", _boom)
    monkeypatch.setattr(pm.subprocess, "Popen", _boom)
    monkeypatch.setattr(pm.shutil, "which", _boom)

    manager = pm.HerdrProcessManager()

    assert manager.socket_endpoint is None  # resolved lazily, on first spawn


# --------------------------------------------------------------------------
# Session routing (plan tests 7-10)
# --------------------------------------------------------------------------


def test_argv_omits_session_when_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIN_AGENT_TEAMS_HERDR_SESSION", raising=False)
    manager = pm.HerdrProcessManager()

    assert manager._herdr_argv("tab", "list") == ["herdr", "tab", "list"]


def test_argv_prefixes_pinned_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--session`` is a top-level prefix, never a per-subcommand flag."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "agent-teams")
    manager = pm.HerdrProcessManager()

    assert manager._herdr_argv("pane", "read", "w1:p2") == [
        "herdr",
        "--session",
        "agent-teams",
        "pane",
        "read",
        "w1:p2",
    ]


def test_pinned_session_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the env later must not re-route a live manager's commands."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "agent-teams")
    manager = pm.HerdrProcessManager()
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "somebody-else")

    assert manager._herdr_argv("tab", "list")[:3] == [
        "herdr",
        "--session",
        "agent-teams",
    ]


@pytest.mark.parametrize("bad", ["has space", "semi;colon", "--dash", "a" * 65])
def test_invalid_session_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """A malformed name could inject argv or address a foreign session."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", bad)

    with pytest.raises(ValueError, match="session"):
        pm.HerdrProcessManager()


# --------------------------------------------------------------------------
# The _run_herdr protocol boundary (plan tests 26-31)
# --------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(monkeypatch: pytest.MonkeyPatch, result: object) -> list[list[str]]:
    """Record argv and answer with ``result`` (or raise it, if it is one)."""
    seen: list[list[str]] = []

    def _run(argv: list[str], **kwargs: object) -> object:
        seen.append(argv)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(pm.subprocess, "run", _run)
    return seen


def test_run_herdr_returns_result_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = pm.HerdrProcessManager()
    payload = {"id": "cli:tab:list", "result": {"type": "tab_list", "tabs": []}}
    _fake_run(monkeypatch, _FakeCompleted(stdout=json.dumps(payload)))

    assert manager._run_herdr("tab", "list", expect="tab_list") == payload["result"]


def test_run_herdr_is_bounded_by_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unbounded control call would hang the whole MCP server."""
    manager = pm.HerdrProcessManager()
    seen_kwargs: dict[str, object] = {}

    def _run(argv: list[str], **kwargs: object) -> object:
        seen_kwargs.update(kwargs)
        return _FakeCompleted(stdout=json.dumps({"result": {"type": "ok"}}))

    monkeypatch.setattr(pm.subprocess, "run", _run)
    manager._run_herdr("tab", "list", expect="ok")

    assert isinstance(seen_kwargs.get("timeout"), (int, float))


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_run_herdr_rejects_error_envelope_on_either_stream(
    monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    """Herdr puts success on stdout and server errors on stderr."""
    manager = pm.HerdrProcessManager()
    envelope = json.dumps(
        {"id": "cli:pane:get", "error": {"code": "not_found", "message": "no pane"}}
    )
    _fake_run(monkeypatch, _FakeCompleted(returncode=1, **{stream: envelope}))

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        manager._run_herdr("pane", "get", "w1:p9", expect="pane")

    assert excinfo.value.code == "not_found"


def test_run_herdr_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, _FakeCompleted(stdout="not json at all"))

    with pytest.raises(pm.HerdrCommandError):
        manager._run_herdr("tab", "list", expect="tab_list")


def test_run_herdr_rejects_unexpected_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validating ``type`` is what stops a CLI change silently changing meaning."""
    manager = pm.HerdrProcessManager()
    payload = {"result": {"type": "something_else"}}
    _fake_run(monkeypatch, _FakeCompleted(stdout=json.dumps(payload)))

    with pytest.raises(pm.HerdrCommandError):
        manager._run_herdr("tab", "list", expect="tab_list")


def test_run_herdr_reports_syntax_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 2 is a CLI syntax error: our argv is wrong, not Herdr's state."""
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, _FakeCompleted(returncode=2, stderr="bad usage"))

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        manager._run_herdr("pane", "nonsense", expect="ok")

    assert excinfo.value.code == "cli_usage"


def test_run_herdr_maps_timeout_to_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout is indeterminate, and must never read as 'the agent is gone'."""
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, pm.subprocess.TimeoutExpired(cmd="herdr", timeout=5))

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        manager._run_herdr("pane", "get", "w1:p1", expect="pane")

    assert excinfo.value.code == "timeout"


def test_run_herdr_routes_through_the_session_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "agent-teams")
    manager = pm.HerdrProcessManager()
    seen = _fake_run(
        monkeypatch, _FakeCompleted(stdout=json.dumps({"result": {"type": "ok"}}))
    )
    manager._run_herdr("tab", "close", "w1:t2", expect="ok")

    assert seen[0][:3] == ["herdr", "--session", "agent-teams"]


# --------------------------------------------------------------------------
# Spawn (plan tests 11-17, 20)
# --------------------------------------------------------------------------

_DISK_FULL = "disk full"
_NO_BINARY = "no such file: herdr"

_TAB_CREATED = {
    "type": "tab_created",
    "tab": {"tab_id": "w3:t2", "label": "worker@team"},
    "root_pane": {
        "pane_id": "w3:p2",
        "tab_id": "w3:t2",
        "workspace_id": "w3",
        "cwd": "/work",
    },
}
_PROCESS_INFO = {
    "type": "pane_process_info",
    "process_info": {
        "pane_id": "w3:p2",
        "shell_pid": 4242,
        "foreground_processes": [{"pid": 4242, "name": "bash"}],
    },
}


class _Herdr:
    """A scripted Herdr CLI: records every call, answers per result type."""

    def __init__(self, **replies: dict | BaseException) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.replies: dict[str, dict | BaseException] = {
            "tab_created": _TAB_CREATED,
            "pane_process_info": _PROCESS_INFO,
            "ok": {"type": "ok"},
            **replies,
        }

    def __call__(self, *args: str, expect: str, **kwargs: object) -> dict:
        self.calls.append(args)
        reply = self.replies[expect]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def argv_for(self, *prefix: str) -> tuple[str, ...] | None:
        for call in self.calls:
            if call[: len(prefix)] == prefix:
                return call
        return None


@pytest.fixture
def _manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pm.HerdrProcessManager:
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "log_path", lambda team, agent: tmp_path / "agent.log")
    monkeypatch.setattr(manager, "_ensure_server", lambda: "/run/herdr.sock")
    monkeypatch.setattr(pm, "creation_token", lambda handle: f"token-{handle}")
    return manager


def _spawn(
    manager: pm.HerdrProcessManager, request: SpawnRequest, herdr: _Herdr
) -> SpawnResult:
    return manager.spawn_process(
        request, ["claude", "--print"], {"AGENT_NAME": "worker"}, "claude-code"
    )


def test_spawn_creates_a_tab_with_cwd_label_and_env(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)

    _spawn(_manager, _request, herdr)
    argv = herdr.argv_for("tab", "create")

    assert argv is not None
    assert "--no-focus" in argv
    assert argv[argv.index("--cwd") + 1] == _request.cwd
    assert argv[argv.index("--label") + 1] == "worker@team"
    assert argv[argv.index("--env") + 1] == "AGENT_NAME=worker"


def test_spawn_returns_the_pane_shell_pid_as_handle(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)

    result = _spawn(_manager, _request, herdr)

    assert result.process_handle == "4242"
    assert result.backend_type == "claude-code"
    info = _manager._processes["4242"]
    assert (info.pane_id, info.tab_id, info.workspace_id) == ("w3:p2", "w3:t2", "w3")


def test_spawn_runs_the_posix_shell_command_in_the_pane(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend already built argv; we must hand it over untouched."""
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)

    _spawn(_manager, _request, herdr)
    argv = herdr.argv_for("pane", "run")

    assert argv is not None
    assert argv[2] == "w3:p2"
    assert "exec claude --print" in argv[3]
    assert "export AGENT_NAME=worker;" in argv[3]
    assert argv[3].startswith(f"cd {_request.cwd} &&")


def test_spawn_passes_awkward_env_values_as_single_tokens(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)
    value = 'a b "c" $d\nnewline'

    _manager.spawn_process(_request, ["claude"], {"WEIRD": value}, "claude-code")
    argv = herdr.argv_for("tab", "create")

    assert argv is not None
    assert f"WEIRD={value}" in argv


def test_spawn_fails_and_closes_the_tab_when_the_token_is_null(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null creation token must never be registered.

    ``_tracked_alive`` compares stored and live tokens, and ``None == None``
    is true -- so storing one would hand ``ownership_probe`` a forged "ours"
    for a PID we have proven nothing about.
    """
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)
    monkeypatch.setattr(pm, "creation_token", lambda handle: None)

    with pytest.raises(RuntimeError, match="creation token"):
        _spawn(_manager, _request, herdr)

    assert herdr.argv_for("tab", "close") is not None
    assert _manager._processes == {}


def test_spawn_closes_the_tab_when_pane_run_fails(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed spawn must not leave an orphan tab behind."""
    boom = pm.HerdrCommandError(["herdr", "pane", "run"], "error", "nope")
    herdr = _Herdr(ok=boom)

    def _run(*args: str, expect: str, **kw: object) -> dict:
        if args[:2] == ("tab", "close"):
            herdr.calls.append(args)
            return {"type": "ok"}
        return herdr(*args, expect=expect, **kw)

    monkeypatch.setattr(_manager, "_run_herdr", _run)

    with pytest.raises(pm.HerdrCommandError):
        _spawn(_manager, _request, herdr)

    assert herdr.argv_for("tab", "close") is not None
    assert _manager._processes == {}


# --------------------------------------------------------------------------
# Ownership probe and per-caller projections (plan tests 18-25, 32-39)
# --------------------------------------------------------------------------


def _track(
    manager: pm.HerdrProcessManager, tmp_path: Path, *, pid: int = 4242
) -> pm.HerdrProcessInfo:
    info = pm.HerdrProcessInfo(
        pid=pid,
        creation_token=f"token-{pid}",
        session_name=None,
        socket_endpoint="/run/herdr.sock",
        name="worker",
        agent_id="worker@team",
        team_name="team",
        backend="claude-code",
        tab_id="w3:t2",
        pane_id="w3:p2",
        workspace_id="w3",
        log_path=tmp_path / "agent.log",
        started_at=0.0,
    )
    manager._processes[str(pid)] = info
    return info


def _probe_setup(
    manager: pm.HerdrProcessManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    process_info: dict | BaseException = _PROCESS_INFO,
    token: str | None = "token-4242",
    pid_alive: bool = True,
) -> None:
    def _run(*args: str, expect: str, **kw: object) -> dict:
        if isinstance(process_info, BaseException):
            raise process_info
        return process_info

    monkeypatch.setattr(manager, "_run_herdr", _run)
    monkeypatch.setattr(pm, "creation_token", lambda handle: token)
    # Patch the boundary the code actually consults. Patching the manager's
    # _pid_alive instead left these tests quietly depending on host PID 4242
    # being absent.
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: pid_alive)
    monkeypatch.setattr(manager, "_ensure_server", lambda: "/run/herdr.sock")


def test_probe_owned_when_everything_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    _probe_setup(manager, monkeypatch)

    assert manager._probe(info) is pm._HerdrProbe.OWNED
    assert manager._tracked_alive(info) is True


def test_probe_pane_gone_when_the_pane_hosts_a_different_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Our PID/token still match, but the pane is someone else's now.

    That is unmanageable, not dead: reporting IDENTITY_MISMATCH here would
    let graceful_shutdown claim a still-running agent had exited.
    """
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    other = {
        "type": "pane_process_info",
        "process_info": {"pane_id": "w3:p2", "shell_pid": 9999},
    }
    _probe_setup(manager, monkeypatch, process_info=other)

    assert manager._probe(info) is pm._HerdrProbe.PANE_GONE
    assert manager._tracked_alive(info) is False


def test_probe_identity_mismatch_when_token_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same PID number, different process: the classic PID-reuse case."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    _probe_setup(manager, monkeypatch, token="token-recycled")

    assert manager._probe(info) is pm._HerdrProbe.IDENTITY_MISMATCH


def test_probe_never_accepts_two_null_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``None == None`` must never be allowed to read as proof of ownership."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    object.__setattr__(info, "creation_token", None)  # simulate a bad record
    _probe_setup(manager, monkeypatch, token=None)

    assert manager._probe(info) is not pm._HerdrProbe.OWNED
    assert manager._tracked_alive(info) is False


def test_probe_pane_gone_but_process_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A moved pane gets a NEW id while its process keeps running.

    Reporting that as death would let kill_agent drop the record while the
    real agent carried on.
    """
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    gone = pm.HerdrCommandError(["herdr", "pane"], "not_found", "no such pane")
    _probe_setup(manager, monkeypatch, process_info=gone)

    assert manager._probe(info) is pm._HerdrProbe.PANE_GONE


def test_probe_pid_gone_when_pane_absent_and_process_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    gone = pm.HerdrCommandError(["herdr", "pane"], "not_found", "no such pane")
    _probe_setup(manager, monkeypatch, process_info=gone, token=None, pid_alive=False)

    assert manager._probe(info) is pm._HerdrProbe.PID_GONE


def test_probe_indeterminate_on_control_plane_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timeout says nothing about the agent, only about the CLI."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    timeout = pm.HerdrCommandError(["herdr", "pane"], "timeout", "timed out")
    _probe_setup(manager, monkeypatch, process_info=timeout)

    assert manager._probe(info) is pm._HerdrProbe.INDETERMINATE


def test_health_check_stays_alive_when_only_the_control_plane_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One CLI hiccup must not be reported as a dead agent."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    timeout = pm.HerdrCommandError(["herdr", "pane"], "timeout", "timed out")
    _probe_setup(manager, monkeypatch, process_info=timeout)

    alive, detail = manager.health_check("4242")

    assert alive is True
    assert "degraded" in detail


def test_health_check_stays_alive_for_a_moved_pane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    gone = pm.HerdrCommandError(["herdr", "pane"], "not_found", "no such pane")
    _probe_setup(manager, monkeypatch, process_info=gone)

    alive, _ = manager.health_check("4242")

    assert alive is True


def test_health_check_reports_dead_when_the_process_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    gone = pm.HerdrCommandError(["herdr", "pane"], "not_found", "no such pane")
    _probe_setup(manager, monkeypatch, process_info=gone, token=None, pid_alive=False)

    alive, _ = manager.health_check("4242")

    assert alive is False


@pytest.mark.parametrize(
    "state",
    [
        pm._HerdrProbe.PANE_GONE,
        pm._HerdrProbe.PID_GONE,
        pm._HerdrProbe.IDENTITY_MISMATCH,
        pm._HerdrProbe.INDETERMINATE,
    ],
)
def test_send_issues_nothing_unless_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: pm._HerdrProbe
) -> None:
    """Input is a mutating pane operation: never write into an unproven pane."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "_probe", lambda info: state)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )

    manager.send("4242", "hello")

    assert calls == []


@pytest.mark.parametrize(
    "state",
    [
        pm._HerdrProbe.PANE_GONE,
        pm._HerdrProbe.IDENTITY_MISMATCH,
        pm._HerdrProbe.INDETERMINATE,
    ],
)
def test_capture_reads_nothing_unless_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: pm._HerdrProbe
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "_probe", lambda info: state)
    monkeypatch.setattr(
        manager, "_run_herdr_text", lambda *a, **k: (calls.append(a), "x")[1]
    )

    assert manager.capture("4242") == ""
    assert calls == []


def test_send_writes_text_then_enter_when_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )

    manager.send("4242", "hello")

    assert calls[0][:2] == ("pane", "send-text")
    assert calls[1][:2] == ("pane", "send-keys")
    assert calls[1][-1] == "enter"


def test_send_without_enter_sends_no_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )

    manager.send("4242", "hello", enter=False)

    assert [c[:2] for c in calls] == [("pane", "send-text")]


# --------------------------------------------------------------------------
# Kill and graceful shutdown (plan tests 36-37, 40-41)
# --------------------------------------------------------------------------


def _kill_setup(
    manager: pm.HerdrProcessManager,
    monkeypatch: pytest.MonkeyPatch,
    state: pm._HerdrProbe,
) -> tuple[list[tuple[str, ...]], list[str]]:
    calls: list[tuple[str, ...]] = []
    killed: list[str] = []
    monkeypatch.setattr(manager, "_probe", lambda info: state)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )
    monkeypatch.setattr(manager, "_kill_pid", killed.append)
    monkeypatch.setattr(manager, "_wait_pid_exit", lambda pid, timeout: True)
    return calls, killed


def test_kill_closes_the_tab_when_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls, _ = _kill_setup(manager, monkeypatch, pm._HerdrProbe.OWNED)

    manager.kill_process("4242")

    assert calls[0][:2] == ("tab", "close")
    assert "4242" not in manager._processes


def test_kill_of_a_moved_pane_signals_the_pid_instead_of_closing_a_tab(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A moved pane must still stop the agent, not orphan it.

    But the tab id we hold may now belong to someone else, so no tab
    operation is allowed -- only a token-checked signal.
    """
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls, killed = _kill_setup(manager, monkeypatch, pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-4242")

    manager.kill_process("4242")

    assert calls == []
    assert killed == ["4242"]


def test_kill_touches_nothing_on_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recycled PID belongs to a stranger: never signal it."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls, killed = _kill_setup(manager, monkeypatch, pm._HerdrProbe.IDENTITY_MISMATCH)
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-somebody-else")

    manager.kill_process("4242")

    assert calls == []
    assert killed == []


def test_kill_revalidates_the_token_before_the_pid_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Between close and force-kill the PID may have been recycled."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    _calls, killed = _kill_setup(manager, monkeypatch, pm._HerdrProbe.OWNED)
    monkeypatch.setattr(manager, "_wait_pid_exit", lambda pid, timeout: False)
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-recycled")

    manager.kill_process("4242")

    assert killed == []


def test_graceful_shutdown_interrupts_the_pane_when_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )
    # Alive at first, then the process really exits.
    alive = iter([True, False])
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: next(alive, False))
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-4242")

    assert manager.graceful_shutdown("4242", timeout_s=1.0) is True
    assert calls[0][:2] == ("pane", "send-keys")
    assert calls[0][-1] == "ctrl+c"


def test_graceful_shutdown_of_a_moved_pane_uses_a_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []
    signalled: list[str] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(
        manager, "_run_herdr", lambda *a, **k: calls.append(a) or {"type": "ok"}
    )
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-4242")
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: True)
    monkeypatch.setattr(manager, "_interrupt_pid", signalled.append)

    # The process never exits, so the honest answer is False -- a moved pane
    # is not evidence that our process stopped.
    assert manager.graceful_shutdown("4242", timeout_s=0.2) is False
    assert calls == []
    assert signalled == ["4242"]


def test_graceful_shutdown_reports_true_when_already_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: False)

    assert manager.graceful_shutdown("4242", timeout_s=0.2) is True


# --------------------------------------------------------------------------
# Server bootstrap (plan tests 44-49)
# --------------------------------------------------------------------------


class _FakeChild:
    """A Popen stand-in rich enough to express cleanup, not just liveness."""

    def __init__(self, exits_with: int | None = None) -> None:
        self.returncode = exits_with
        self._exits_with = exits_with
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._exits_with

    def terminate(self) -> None:
        self.terminated = True
        self._exits_with = -15
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self._exits_with = -9
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self._exits_with if self._exits_with is not None else 0


def _status(running: bool, socket: str = "/run/herdr.sock") -> dict:
    """A bare ``status server --json`` object, as 0.8.2 really answers."""
    return {
        "status": "running" if running else "not_running",
        "running": running,
        "socket": socket,
    }


def test_ensure_server_reuses_a_running_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user's own session is reused, never restarted."""
    manager = pm.HerdrProcessManager()
    started: list[str] = []
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(True))
    monkeypatch.setattr(
        manager,
        "_popen_herdr_server",
        lambda: (started.append("x"), _FakeChild())[1],
    )

    assert manager._ensure_server() == "/run/herdr.sock"
    assert started == []


def test_ensure_server_starts_one_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    answers = iter([_status(False), _status(False), _status(True)])
    monkeypatch.setattr(
        manager, "_run_herdr_raw", lambda *a, **k: next(answers, _status(True))
    )
    monkeypatch.setattr(manager, "_popen_herdr_server", _FakeChild)
    monkeypatch.setattr(pm.time, "sleep", lambda s: None)

    assert manager._ensure_server() == "/run/herdr.sock"


def test_ensure_server_rechecks_inside_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loser of a startup race must not launch a second daemon."""
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    started: list[str] = []
    answers = iter([_status(False), _status(True)])
    monkeypatch.setattr(
        manager, "_run_herdr_raw", lambda *a, **k: next(answers, _status(True))
    )
    monkeypatch.setattr(
        manager,
        "_popen_herdr_server",
        lambda: (started.append("x"), _FakeChild())[1],
    )

    manager._ensure_server()

    assert started == []


def test_ensure_server_reports_a_server_that_exits_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(False))
    monkeypatch.setattr(
        manager, "_popen_herdr_server", functools.partial(_FakeChild, exits_with=1)
    )

    with pytest.raises(pm.HerdrServerUnavailableError, match="exited immediately"):
        manager._ensure_server()


def test_ensure_server_error_names_the_command_to_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A never-ready server must tell the user how to start one themselves."""
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(False))
    monkeypatch.setattr(manager, "_popen_herdr_server", _FakeChild)
    monkeypatch.setattr(pm.time, "sleep", lambda s: None)
    monkeypatch.setattr(pm, "_HERDR_START_TIMEOUT_SECONDS", 0.0)

    with pytest.raises(pm.HerdrServerUnavailableError, match="herdr server"):
        manager._ensure_server()


def test_start_lock_is_shared_by_everyone_targeting_one_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unrelated team sessions must contend for the SAME lock."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "shared")
    first = pm.HerdrProcessManager()
    second = pm.HerdrProcessManager()

    assert first._start_lock_path() == second._start_lock_path()


def test_start_lock_differs_per_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "one")
    first = pm.HerdrProcessManager()
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "two")
    second = pm.HerdrProcessManager()

    assert first._start_lock_path() != second._start_lock_path()


# --------------------------------------------------------------------------
# Inherited public interface (plan tests 24-25)
# --------------------------------------------------------------------------


def test_provides_tty_is_true() -> None:
    """Herdr panes are real PTYs, so interactive TUIs are fine."""
    assert pm.HerdrProcessManager().provides_tty("codex", is_interactive=True) is True


def test_resolve_agent_pid_returns_the_handle(tmp_path: Path) -> None:
    """The shell command ends in ``exec``, so the shell PID IS the agent.

    Picking a foreground process instead would latch onto a transient hook
    or tool helper and make liveness flap.
    """
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)

    assert manager.resolve_agent_pid("4242", "team", "worker") == "4242"


def test_ownership_probe_is_not_fooled_by_a_recycled_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: the mixin's destructive gate must stay closed."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    _probe_setup(manager, monkeypatch, token="token-recycled")

    assert manager.owns_process("4242", "token-4242") is False


# --------------------------------------------------------------------------
# Endpoint rebinding (plan test 39) and capture semantics (plan test 42)
# --------------------------------------------------------------------------


def test_rebinds_when_the_endpoint_moved_but_the_agent_did_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``herdr --handoff`` replaces the endpoint while panes keep running.

    Refusing to rebind would leave a live agent permanently unmanageable.
    """
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    manager.socket_endpoint = "/run/herdr-new.sock"
    _probe_setup(manager, monkeypatch)

    assert manager._probe(info) is pm._HerdrProbe.OWNED
    assert info.socket_endpoint == "/run/herdr-new.sock"


def test_does_not_rebind_when_the_agent_identity_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new endpoint showing a DIFFERENT process is not our agent."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    manager.socket_endpoint = "/run/herdr-new.sock"
    _probe_setup(manager, monkeypatch, token="token-somebody-else")

    assert manager._probe(info) is not pm._HerdrProbe.OWNED
    assert info.socket_endpoint == "/run/herdr.sock"


def test_capture_prefers_recent_then_falls_back_to_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Observed on 0.8.2: recent-unwrapped can be empty while visible is not."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    seen: list[str] = []

    def _read(*args: str, **kw: object) -> str:
        source = args[args.index("--source") + 1]
        seen.append(source)
        return "" if source != "visible" else "hi"

    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    monkeypatch.setattr(manager, "_run_herdr_text", _read)

    assert manager.capture("4242") == "hi"
    assert seen == ["recent-unwrapped", "visible"]


def test_capture_omits_lines_when_none_and_passes_it_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``lines=None`` is defined mechanically: omit ``--lines``."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    calls: list[tuple[str, ...]] = []

    def _read(*args: str, **kw: object) -> str:
        calls.append(args)
        return "x"

    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    monkeypatch.setattr(manager, "_run_herdr_text", _read)

    manager.capture("4242")
    assert "--lines" not in calls[0]

    calls.clear()
    manager.capture("4242", lines=40)
    assert calls[0][calls[0].index("--lines") + 1] == "40"


def test_capture_returns_empty_for_non_positive_lines(tmp_path: Path) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)

    assert manager.capture("4242", lines=0) == ""


def test_run_herdr_treats_undecodable_output_as_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON is a machine protocol: never let U+FFFD through into a parse."""
    manager = pm.HerdrProcessManager()
    _fake_run(
        monkeypatch,
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    )

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        manager._run_herdr("tab", "list", expect="tab_list")

    assert excinfo.value.code == "malformed"


def test_status_json_is_read_as_a_bare_object_not_an_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``status --json`` answers with a plain object, unlike control commands.

    Real 0.8.2 output::

        {"status":"running","running":true,"socket":"/.../herdr.sock", ...}

    There is no ``result``/``type`` envelope, so validating it as one made
    every server probe look malformed and the launcher never found a server.
    """
    manager = pm.HerdrProcessManager()
    raw = json.dumps(
        {
            "status": "running",
            "running": True,
            "socket": "/home/u/.config/herdr/sessions/s/herdr.sock",
            "session": "s",
        }
    )
    _fake_run(monkeypatch, _FakeCompleted(stdout=raw))

    assert manager._server_socket() == "/home/u/.config/herdr/sessions/s/herdr.sock"


def test_server_socket_is_none_when_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pm.HerdrProcessManager()
    raw = json.dumps({"status": "not_running", "running": False, "socket": "/x.sock"})
    _fake_run(monkeypatch, _FakeCompleted(stdout=raw))

    assert manager._server_socket() is None


def test_spawn_creates_a_workspace_on_a_fresh_server(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-attached Herdr server has no workspace, so ``tab create`` fails.

    Observed on 0.8.2: a freshly started headless server reports
    ``workspaces: []`` and answers ``tab create`` with
    ``workspace_not_found: no active workspace``. The first agent must
    therefore create the workspace, which yields the same root pane.
    """
    calls: list[tuple[str, ...]] = []
    workspace_created = {
        "type": "workspace_created",
        "tab": {"tab_id": "w1:t1", "label": "1"},
        "root_pane": {"pane_id": "w1:p1", "tab_id": "w1:t1", "workspace_id": "w1"},
    }

    def _run(*args: str, expect: str, **kw: object) -> dict:
        calls.append(args)
        if args[:2] == ("tab", "create"):
            raise pm.HerdrCommandError(
                ["herdr", "tab", "create"], "workspace_not_found", "no active workspace"
            )
        if expect == "workspace_created":
            return workspace_created
        if expect == "pane_process_info":
            return _PROCESS_INFO
        return {"type": "ok"}

    monkeypatch.setattr(_manager, "_run_herdr", _run)

    result = _manager.spawn_process(
        _request, ["claude"], {"AGENT_NAME": "worker"}, "claude-code"
    )

    assert result.process_handle == "4242"
    assert _manager._processes["4242"].pane_id == "w1:p1"
    kinds = [c[:2] for c in calls]
    assert ("workspace", "create") in kinds
    # The workspace's tab is labelled "1", so the agent label is restored.
    assert ("tab", "rename") in kinds


def test_spawn_does_not_create_a_workspace_when_one_exists(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    herdr = _Herdr()
    monkeypatch.setattr(_manager, "_run_herdr", herdr)

    _spawn(_manager, _request, herdr)

    assert herdr.argv_for("workspace", "create") is None


def test_run_herdr_accepts_a_silent_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some mutating commands answer with exit 0 and no payload at all.

    Observed on 0.8.2: ``pane run`` prints nothing on success. Demanding an
    envelope turned every successful spawn into a ``malformed`` error.
    """
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, _FakeCompleted(returncode=0, stdout="", stderr=""))

    assert manager._run_herdr(
        "pane", "run", "w1:p2", "true", expect="ok", allow_empty_success=True
    ) == {"type": "ok"}


def test_run_herdr_still_rejects_silent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence is only success when the exit code says so."""
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, _FakeCompleted(returncode=1, stdout="", stderr=""))

    with pytest.raises(pm.HerdrCommandError):
        manager._run_herdr("pane", "run", "w1:p2", "true", expect="ok")


def test_capture_reads_plain_text_not_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``pane read`` prints terminal text on stdout; there is no JSON envelope."""
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.OWNED)
    _fake_run(monkeypatch, _FakeCompleted(stdout="line one\nline two\n"))

    assert manager.capture("4242") == "line one\nline two\n"


# --- local-identity-first cross-product (implementation-review-1 MAJOR 1) ---


def test_probe_is_indeterminate_when_the_token_is_unreadable_but_pid_lives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A transiently unreadable token must not read as a dead agent.

    This is the false-death case: the pane is healthy and reports our PID,
    but one unreadable token would previously classify it IDENTITY_MISMATCH
    and health_check would report a live agent dead.
    """
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    _probe_setup(manager, monkeypatch, token=None, pid_alive=True)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: True)

    assert manager._probe(info) is pm._HerdrProbe.INDETERMINATE
    assert manager.health_check("4242")[0] is True


def test_probe_is_pid_gone_when_the_pane_times_out_and_the_pid_is_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Herdr timeout must not rescue a process that is provably gone.

    Previously the pane error was consulted first, so a dead PID plus a slow
    CLI reported "alive, pid still ours".
    """
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    timeout = pm.HerdrCommandError(["herdr", "pane"], "timeout", "timed out")
    _probe_setup(manager, monkeypatch, process_info=timeout, token=None)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: False)

    assert manager._probe(info) is pm._HerdrProbe.PID_GONE
    assert manager.health_check("4242")[0] is False


def test_probe_is_identity_mismatch_when_the_token_changed_and_herdr_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recycled PID is settled locally; Herdr being unreachable cannot undo it."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    timeout = pm.HerdrCommandError(["herdr", "pane"], "timeout", "timed out")
    _probe_setup(manager, monkeypatch, process_info=timeout, token="token-recycled")

    assert manager._probe(info) is pm._HerdrProbe.IDENTITY_MISMATCH
    assert manager.health_check("4242")[0] is False


def test_probe_rejects_a_record_with_no_stored_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    object.__setattr__(info, "creation_token", None)
    _probe_setup(manager, monkeypatch, token="anything")

    assert manager._probe(info) is pm._HerdrProbe.IDENTITY_MISMATCH


# --- kill/graceful record lifecycle (implementation-review-1 MAJOR 7-8) ---


def test_graceful_shutdown_does_not_claim_success_for_a_live_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pane hosting someone else is not proof that OUR process exited.

    The follow-up path reads True as "no force kill needed" and resumes the
    agent, so a false True lets the old and resumed workers overlap.
    """
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: True)
    monkeypatch.setattr(pm, "creation_token", lambda handle: "token-4242")
    monkeypatch.setattr(manager, "_interrupt_pid", lambda handle: None)

    assert manager.graceful_shutdown("4242", timeout_s=0.2) is False


def test_kill_keeps_the_record_when_ownership_cannot_be_proven(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreadable token on a live PID must not look like a successful kill.

    server_simple.kill_agent deletes the durable record when kill_process
    returns normally, so quietly forgetting the agent here would strand a
    live worker with nothing left to manage it.
    """
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    killed: list[str] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(manager, "_kill_pid", killed.append)
    monkeypatch.setattr(pm, "creation_token", lambda handle: None)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: True)

    with pytest.raises(pm.HerdrOwnershipUnprovenError):
        manager.kill_process("4242")

    assert killed == []
    assert "4242" in manager._processes


def test_kill_drops_the_record_once_the_pid_is_proven_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = pm.HerdrProcessManager()
    _track(manager, tmp_path)
    killed: list[str] = []
    monkeypatch.setattr(manager, "_probe", lambda info: pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(manager, "_kill_pid", killed.append)
    monkeypatch.setattr(pm, "creation_token", lambda handle: None)
    monkeypatch.setattr(pm, "_pid_is_live", lambda pid: False)

    manager.kill_process("4242")

    assert killed == []
    assert "4242" not in manager._processes


def test_silent_success_is_not_generalised_to_other_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only commands observed to answer silently may be assumed successful.

    ``pane run`` prints nothing on success; ``tab close`` and friends answer.
    Treating any empty exit-0 response as the requested result would let a
    response-bearing command fabricate the semantics we asked for.
    """
    manager = pm.HerdrProcessManager()
    _fake_run(monkeypatch, _FakeCompleted(returncode=0, stdout="", stderr=""))

    with pytest.raises(pm.HerdrCommandError):
        manager._run_herdr("tab", "close", "w1:t1", expect="ok")


def test_an_error_on_stderr_wins_over_a_success_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning the first parseable object would hide a reported error."""
    manager = pm.HerdrProcessManager()
    _fake_run(
        monkeypatch,
        _FakeCompleted(
            returncode=0,
            stdout=json.dumps({"result": {"type": "ok"}}),
            stderr=json.dumps({"error": {"code": "not_found", "message": "gone"}}),
        ),
    )

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        manager._run_herdr("tab", "close", "w1:t1", expect="ok")

    assert excinfo.value.code == "not_found"


# --- server child lifecycle (implementation-review-1 MAJOR 5) ---


def test_a_server_that_never_becomes_ready_is_not_leaked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raising while our half-started daemon keeps running leaves a mess.

    The lock is released on the way out, so the next process would race a
    server nobody is tracking.
    """
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(False))
    child = _FakeChild()
    monkeypatch.setattr(manager, "_popen_herdr_server", lambda: child)
    monkeypatch.setattr(pm.time, "sleep", lambda s: None)
    monkeypatch.setattr(pm, "_HERDR_START_TIMEOUT_SECONDS", 0.0)

    with pytest.raises(pm.HerdrServerUnavailableError):
        manager._ensure_server()

    assert child.terminated is True
    assert manager._server_child is None


def test_a_failed_launch_is_reported_as_server_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing binary must not surface as a bare OSError from Popen."""
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(False))

    def _boom() -> object:
        raise OSError(_NO_BINARY)

    monkeypatch.setattr(manager, "_popen_herdr_server", _boom)

    with pytest.raises(pm.HerdrServerUnavailableError, match="could not launch"):
        manager._ensure_server()


def test_an_exited_server_child_is_reaped_on_the_next_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pm.HerdrProcessManager()
    manager._server_child = cast("subprocess.Popen[bytes]", _FakeChild(exits_with=0))
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: _status(True))

    manager._ensure_server()

    assert manager._server_child is None


def test_a_status_query_failure_never_authorises_starting_a_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ "Could not tell" is not "confirmed absent".

    A timeout against a busy live server would otherwise start a second
    daemon beside a perfectly healthy one.
    """
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_start_lock_path", lambda: tmp_path / "herdr.lock")
    started: list[str] = []

    def _raise(*args: str, **kw: object) -> dict:
        raise pm.HerdrCommandError(["herdr", "status"], "timeout", "timed out")

    monkeypatch.setattr(manager, "_run_herdr_raw", _raise)
    monkeypatch.setattr(
        manager, "_popen_herdr_server", lambda: (started.append("x"), _FakeChild())[1]
    )

    with pytest.raises(pm.HerdrCommandError):
        manager._ensure_server()

    assert started == []


def test_a_malformed_status_object_is_not_read_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pm.HerdrProcessManager()
    monkeypatch.setattr(manager, "_run_herdr_raw", lambda *a, **k: {})

    with pytest.raises(pm.HerdrCommandError):
        manager._server_socket()


def test_socket_path_comes_from_session_list_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real layout is <config>/sessions/<name>/herdr.sock, which a
    hand-built path got wrong -- and it moves with HERDR_CONFIG_PATH."""
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "nested")
    manager = pm.HerdrProcessManager()
    real = "/home/u/.config/herdr/sessions/nested/herdr.sock"

    def _raw(*args: str, **kw: object) -> dict:
        if args[0] == "status":
            return {"status": "running", "running": True}
        return {"sessions": [{"name": "nested", "socket_path": real}]}

    monkeypatch.setattr(manager, "_run_herdr_raw", _raw)

    assert manager._server_socket() == real


def test_a_cached_endpoint_is_revalidated_on_the_next_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that died since the last spawn must be noticed, not cached."""
    manager = pm.HerdrProcessManager()
    manager.socket_endpoint = "/run/old.sock"
    monkeypatch.setattr(
        manager, "_run_herdr_raw", lambda *a, **k: _status(True, "/run/new.sock")
    )

    assert manager._ensure_server() == "/run/new.sock"
    assert manager.socket_endpoint == "/run/new.sock"


# --- spawn rollback scope (implementation-review-1 MAJOR 6) ---


def test_a_partial_create_response_still_closes_the_created_tab(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real tab id with a malformed pane must not raise past cleanup."""
    calls: list[tuple[str, ...]] = []

    def _run(*args: str, expect: str, **kw: object) -> dict:
        calls.append(args)
        if args[:2] == ("tab", "create"):
            return {"type": "tab_created", "tab": {"tab_id": "w1:t7"}}  # no root_pane
        return {"type": "ok"}

    monkeypatch.setattr(_manager, "_run_herdr", _run)

    with pytest.raises(pm.HerdrCommandError):
        _manager.spawn_process(
            _request, ["claude"], {"AGENT_NAME": "worker"}, "claude-code"
        )

    assert ("tab", "close", "w1:t7") in calls
    assert _manager._processes == {}


def test_a_failing_provenance_write_rolls_the_spawn_back(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller sees a failed spawn, so no live agent may be left behind."""
    herdr = _Herdr()
    closed: list[tuple[str, ...]] = []

    def _run(*args: str, expect: str, **kw: object) -> dict:
        if args[:2] == ("tab", "close"):
            closed.append(args)
            return {"type": "ok"}
        return herdr(*args, expect=expect, **kw)

    monkeypatch.setattr(_manager, "_run_herdr", _run)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError(_DISK_FULL)

    monkeypatch.setattr(_manager, "_write_provenance", _boom)

    with pytest.raises(OSError, match="disk full"):
        _spawn(_manager, _request, herdr)

    assert closed
    assert _manager._processes == {}


def test_a_cleanup_failure_is_attached_not_masked(
    _manager: pm.HerdrProcessManager,
    _request: SpawnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original cause must survive; the cleanup problem rides along."""

    def _run(*args: str, expect: str, **kw: object) -> dict:
        if args[:2] == ("tab", "create"):
            return {"type": "tab_created", "tab": {"tab_id": "w1:t9"}}
        raise pm.HerdrCommandError(["herdr", "tab", "close"], "error", "cannot close")

    monkeypatch.setattr(_manager, "_run_herdr", _run)

    with pytest.raises(pm.HerdrCommandError) as excinfo:
        _manager.spawn_process(
            _request, ["claude"], {"AGENT_NAME": "worker"}, "claude-code"
        )

    assert "no pane" in str(excinfo.value)
    assert any("could not close herdr tab w1:t9" in n for n in excinfo.value.__notes__)


def test_a_refused_operation_records_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Silence with no explanation is indistinguishable from a bug."""
    manager = pm.HerdrProcessManager()
    info = _track(manager, tmp_path)
    info.log_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(manager, "_probe", lambda i: pm._HerdrProbe.PANE_GONE)
    monkeypatch.setattr(manager, "_run_herdr", lambda *a, **k: {"type": "ok"})

    manager.send("4242", "hello")

    assert "refused send on pane w3:p2: pane_gone" in info.log_path.read_text(
        encoding="utf-8"
    )


def test_the_start_lock_follows_herdr_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relocated Herdr config must not silently split the lock in two.

    HERDR_CONFIG_PATH names the config FILE, so the lock belongs beside it.
    Reading it as a directory would try to mkdir inside an existing file.
    """
    config_file = tmp_path / "cfg" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("HERDR_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("WIN_AGENT_TEAMS_HERDR_SESSION", "shared")

    lock_path = pm.HerdrProcessManager()._start_lock_path()

    assert lock_path.parent == config_file.parent
    # It must be usable: the real lock, on the real path.
    with file_lock(lock_path):
        assert lock_path.exists()


def test_a_zombie_is_not_a_live_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zombie answers kill(pid, 0) and keeps a readable /proc token.

    Without the zombie check a dead agent reads as owned-and-alive, and a
    graceful shutdown waits out its whole timeout on a corpse.
    """
    monkeypatch.setattr(pm.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(pm, "_pid_is_zombie", lambda pid: True)

    assert pm._pid_is_live(4242) is False


def test_a_live_process_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(pm, "_pid_is_zombie", lambda pid: False)

    assert pm._pid_is_live(4242) is True
