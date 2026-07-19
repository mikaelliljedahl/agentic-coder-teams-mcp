"""Tests for WindowsProcessManager job-object breakaway behavior.

``subprocess.CREATE_BREAKAWAY_FROM_JOB`` only exists on Windows; on other
platforms ``getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)`` evaluates
to ``0``. These tests patch the constant to a concrete sentinel value so the
assertions hold on both Windows and Linux CI.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from claude_teams.backends import process_manager as process_manager_mod
from claude_teams.backends.contracts import SpawnRequest

_BREAKAWAY = 0x01000000
_NEW_PROCESS_GROUP = 0x00000200
_NEW_CONSOLE = 0x00000010


@pytest.fixture
def _spawn_request(tmp_path: Path) -> SpawnRequest:
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


@pytest.fixture
def _manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> process_manager_mod.WindowsProcessManager:
    manager = process_manager_mod.WindowsProcessManager()
    monkeypatch.setattr(
        manager, "log_path", lambda team_name, agent_name: tmp_path / "agent.log"
    )
    monkeypatch.setattr(manager, "_open_windows_terminal_tail", lambda *a, **k: None)
    return manager


@pytest.fixture(autouse=True)
def _patch_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        process_manager_mod.subprocess,
        "CREATE_BREAKAWAY_FROM_JOB",
        _BREAKAWAY,
        raising=False,
    )
    monkeypatch.setattr(
        process_manager_mod.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        _NEW_PROCESS_GROUP,
        raising=False,
    )
    monkeypatch.setattr(
        process_manager_mod.subprocess,
        "CREATE_NEW_CONSOLE",
        _NEW_CONSOLE,
        raising=False,
    )


def _fake_process(pid: int = 1234) -> MagicMock:
    process = MagicMock()
    process.pid = pid
    process.poll.return_value = None
    return process


class TestBreakawayFlag:
    def test_default_spawn_includes_breakaway_bit(
        self, monkeypatch, _manager, _spawn_request
    ):
        captured = {}

        def fake_popen(cmd, creationflags=0, **kwargs):
            captured["creationflags"] = creationflags
            return _fake_process()

        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", fake_popen)

        _manager.spawn_process(_spawn_request, ["stub-cli"], {}, "stub")

        assert captured["creationflags"] & _BREAKAWAY

    def test_no_breakaway_env_disables_flag(
        self, monkeypatch, _manager, _spawn_request
    ):
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_BREAKAWAY", "1")
        captured = {}

        def fake_popen(cmd, creationflags=0, **kwargs):
            captured["creationflags"] = creationflags
            return _fake_process()

        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", fake_popen)

        _manager.spawn_process(_spawn_request, ["stub-cli"], {}, "stub")

        assert not captured["creationflags"] & _BREAKAWAY


class TestBreakawayFallback:
    def test_retries_once_without_breakaway_on_access_denied(
        self, monkeypatch, _manager, _spawn_request
    ):
        calls = []

        def fake_popen(cmd, creationflags=0, **kwargs):
            calls.append(creationflags)
            if len(calls) == 1:
                err = OSError("access denied")
                # ``winerror`` is a Windows-only OSError attribute; ty checks
                # against the Linux stdlib where it is absent. The test
                # deliberately simulates the Windows exception shape.
                err.winerror = 5  # ty: ignore[unresolved-attribute]
                raise err
            return _fake_process()

        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", fake_popen)

        result = _manager.spawn_process(_spawn_request, ["stub-cli"], {}, "stub")

        assert result.process_handle == "1234"
        assert len(calls) == 2
        assert calls[0] & _BREAKAWAY
        assert not calls[1] & _BREAKAWAY

    def test_non_breakaway_oserror_propagates_and_closes_log(
        self, monkeypatch, _manager, _spawn_request
    ):
        calls = []

        def fake_popen(cmd, creationflags=0, **kwargs):
            calls.append(creationflags)
            err = OSError("file not found")
            # Windows-only OSError attribute; see note above.
            err.winerror = 2  # ty: ignore[unresolved-attribute]
            raise err

        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", fake_popen)

        with pytest.raises(OSError, match="file not found"):
            _manager.spawn_process(_spawn_request, ["stub-cli"], {}, "stub")

        assert len(calls) == 1
        log_path = _manager.log_path(_spawn_request.team_name, _spawn_request.name)
        # The handle must be closed on failure: reopening for append should
        # not raise and the file should contain the "starting" line written
        # before the failed Popen call.
        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8")
        assert "starting stub-cli" in contents


class TestProvidesTty:
    """Regression: the real WindowsProcessManager must implement provides_tty.

    The Codex backend calls ``process_manager.provides_tty(...)`` before
    building its command to choose the interactive TUI vs the head-less
    ``codex exec`` entrypoint. When that method was missing on
    ``WindowsProcessManager``, ``spawn_agent`` failed at runtime with
    ``'WindowsProcessManager' object has no attribute 'provides_tty'``.

    The codex build-command tests monkeypatch ``provides_tty`` away, so they
    never exercised the concrete manager. These tests call the real method on
    a real instance for both the ``codex`` and ``claude-code`` backends. They
    force the Windows branch via ``os.name`` so the interactive assertions
    hold on Linux CI too.
    """

    def test_method_is_present_and_callable(self, _manager):
        # The exact regression: the attribute must exist on the instance.
        assert callable(_manager.provides_tty)

    @pytest.mark.parametrize("backend_type", ["codex", "claude-code"])
    def test_non_interactive_has_no_tty(self, _manager, backend_type):
        # The stdin-pipe + log-file stdout path never attaches a real TTY,
        # regardless of backend or platform.
        assert _manager.provides_tty(backend_type, is_interactive=False) is False

    @pytest.mark.parametrize("backend_type", ["codex", "claude-code"])
    def test_interactive_console_provides_tty(
        self, _manager, backend_type, monkeypatch
    ):
        # A WT tab / CREATE_NEW_CONSOLE gives the agent a real console TTY.
        monkeypatch.setattr(process_manager_mod.os, "name", "nt")
        monkeypatch.setenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "1")
        assert _manager.provides_tty(backend_type, is_interactive=True) is True

    @pytest.mark.parametrize("backend_type", ["codex", "claude-code"])
    def test_interactive_console_disabled_has_no_tty(
        self, _manager, backend_type, monkeypatch
    ):
        # Opting out of the interactive console falls back to the TTY-less
        # spawn path, so Codex must use ``codex exec``.
        monkeypatch.setattr(process_manager_mod.os, "name", "nt")
        monkeypatch.setenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "0")
        assert _manager.provides_tty(backend_type, is_interactive=True) is False


class TestTabSettle:
    """The settle check distinguishes a healthy tab start from an instant abort."""

    def test_immediate_exit_is_detected(self, _manager, monkeypatch):
        # A wrapper PID that is already dead means the agent aborted at once
        # (e.g. codex TUI: "stdin is not a terminal" in a degraded WT window).
        monkeypatch.setenv("WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS", "0.2")
        monkeypatch.setattr(_manager, "_pid_alive", lambda handle: False)
        assert _manager._tab_survived_settle("4321") is False

    def test_live_pid_survives_settle(self, _manager, monkeypatch):
        monkeypatch.setenv("WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS", "0.1")
        monkeypatch.setattr(_manager, "_pid_alive", lambda handle: True)
        assert _manager._tab_survived_settle("4321") is True

    def test_settle_disabled_skips_check(self, _manager, monkeypatch):
        # 0 disables the check: even a dead PID is reported as a healthy start.
        monkeypatch.setenv("WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS", "0")
        monkeypatch.setattr(_manager, "_pid_alive", lambda handle: False)
        assert _manager._tab_survived_settle("4321") is True

    def test_pid_reuse_during_settle_is_detected(self, _manager, monkeypatch):
        # The PID stays "alive" but its creation token diverges from the one
        # captured at launch: a foreign process recycled the PID, so the tab is
        # correctly treated as an abort rather than a false survival.
        monkeypatch.setenv("WIN_AGENT_TEAMS_WT_TAB_SETTLE_SECONDS", "0.2")
        monkeypatch.setattr(
            process_manager_mod, "creation_token", lambda handle: "tok-NEW"
        )
        assert _manager._tab_survived_settle("4321", "tok-ORIG") is False


class TestNewConsoleFallback:
    """A failed WT tab spawn falls back to a CREATE_NEW_CONSOLE launch."""

    def test_tab_failure_falls_back_to_new_console(
        self, monkeypatch, _manager, _spawn_request
    ):
        # Force the Windows interactive path with wt.exe available.
        monkeypatch.setattr(process_manager_mod.os, "name", "nt")
        monkeypatch.setenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "1")
        monkeypatch.setattr(
            process_manager_mod.shutil, "which", lambda name: "C:\\wt.exe"
        )

        # The tab agent aborts immediately in a degraded window.
        exc = process_manager_mod.WindowsTerminalTabImmediateExitError

        def boom(*_args, **_kwargs):
            raise exc("worker@team")

        monkeypatch.setattr(_manager, "_spawn_in_terminal_tab", boom)

        captured = {}

        def fake_popen(cmd, creationflags=0, **kwargs):
            captured["creationflags"] = creationflags
            return _fake_process()

        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", fake_popen)

        result = _manager.spawn_process(
            _spawn_request, ["codex"], {}, "codex", is_interactive=True
        )

        # The agent was relaunched in a real console (new-console bit set),
        # not left dead.
        assert result.process_handle == "1234"
        assert captured["creationflags"] & _NEW_CONSOLE
        # The reopened log records the fallback for the operator.
        log_path = _manager.log_path(_spawn_request.team_name, _spawn_request.name)
        assert "new console window" in log_path.read_text(encoding="utf-8")
