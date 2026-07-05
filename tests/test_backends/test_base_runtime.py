"""Runtime BaseBackend process-manager operation tests."""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from claude_teams.backends import process_base
from claude_teams.backends import process_manager as process_manager_mod
from claude_teams.backends.base import HealthStatus, SpawnRequest, SpawnResult
from tests.test_backends._base_support import (
    _InvalidProcessEnvBackend,
    _make_backend_with_mock_process_manager,
    _ProcessStubBackend,
)


class _DangerousEnvBackend(_ProcessStubBackend):
    """Stub backend returning a shell-metacharacter env value."""

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"DANGER": "$(whoami); :"}


class TestBaseBackendSpawn:
    def test_returns_spawn_result_on_success(self, _make_spawn_request, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.spawn_process.return_value = SpawnResult(
            process_handle="4242",
            backend_type="stub",
        )
        request = _make_spawn_request()

        result = backend.spawn(request)

        assert isinstance(result, SpawnResult)
        assert result.process_handle == "4242"
        assert result.process_handle.isdecimal()
        assert result.backend_type == "stub"

    def test_calls_process_manager_with_command_and_env(
        self, _make_spawn_request, monkeypatch
    ):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.spawn_process.return_value = SpawnResult(
            process_handle="4242",
            backend_type="stub",
        )
        request = _make_spawn_request()

        backend.spawn(request)

        manager.spawn_process.assert_called_once_with(
            request,
            ["/usr/bin/stub-cli", "--prompt", "do stuff"],
            {"STUB_MODE": "1"},
            "stub",
            is_interactive=False,
        )

    def test_propagates_process_manager_spawn_failure(
        self, _make_spawn_request, monkeypatch
    ):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.spawn_process.side_effect = OSError("spawn failed")
        request = _make_spawn_request()

        with pytest.raises(OSError, match="spawn failed"):
            backend.spawn(request)

    def test_rejects_invalid_env_var_name(self, _make_spawn_request):
        backend = _InvalidProcessEnvBackend()
        request = _make_spawn_request()

        with pytest.raises(ValueError, match="Invalid environment variable"):
            backend.spawn(request)

    def test_env_values_are_passed_unquoted_to_process_manager(
        self, _make_spawn_request, monkeypatch
    ):
        backend = _DangerousEnvBackend()
        manager = MagicMock(spec=process_base.process_manager)
        monkeypatch.setattr(process_base, "process_manager", manager)
        manager.spawn_process.return_value = SpawnResult(
            process_handle="4242",
            backend_type="stub",
        )
        request = _make_spawn_request()

        backend.spawn(request)

        assert manager.spawn_process.call_args.args[2] == {"DANGER": "$(whoami); :"}


class TestBaseBackendHealthCheck:
    def test_returns_alive_from_process_manager(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.health_check.return_value = (True, "process running")

        status = backend.health_check("4242")

        assert status == HealthStatus(alive=True, detail="process running")
        manager.health_check.assert_called_once_with("4242")

    def test_returns_dead_from_process_manager(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.health_check.return_value = (False, "process not found")

        status = backend.health_check("4242")

        assert status == HealthStatus(alive=False, detail="process not found")


class TestBaseBackendKill:
    def test_calls_process_manager_kill_process(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)

        backend.kill("4242")

        manager.kill_process.assert_called_once_with("4242")


class TestBaseBackendGracefulShutdown:
    def test_delegates_to_process_manager(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.graceful_shutdown.return_value = True

        result = backend.graceful_shutdown("4242", timeout_s=5.0)

        assert result is True
        manager.graceful_shutdown.assert_called_once_with("4242", timeout_s=5.0)


class TestBaseBackendCapture:
    def test_captures_full_log(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.capture.return_value = "line 1\nline 2\n"

        output = backend.capture("4242")

        assert output == "line 1\nline 2\n"
        manager.capture.assert_called_once_with("4242", lines=None)

    def test_captures_limited_lines(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.capture.return_value = "last line\n"

        output = backend.capture("4242", lines=1)

        assert output == "last line\n"
        manager.capture.assert_called_once_with("4242", lines=1)


class TestBaseBackendSend:
    def test_sends_text_with_enter(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)

        backend.send("4242", "hello world")

        manager.send.assert_called_once_with("4242", "hello world", enter=True)

    def test_sends_text_without_enter(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)

        backend.send("4242", "partial", enter=False)

        manager.send.assert_called_once_with("4242", "partial", enter=False)


class TestBaseBackendWaitIdle:
    def test_returns_true_when_process_is_already_dead(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.health_check.return_value = (False, "process exited (0)")

        result = backend.wait_idle("4242")

        assert result is True
        manager.health_check.assert_called_once_with("4242")

    def test_returns_false_for_running_process_without_timeout(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.health_check.return_value = (True, "process running")

        result = backend.wait_idle("4242", idle_time=5.0, timeout=None)

        assert result is False

    def test_polls_after_version_command_when_timeout_is_set(self, monkeypatch):
        backend, manager = _make_backend_with_mock_process_manager(monkeypatch)
        manager.health_check.side_effect = [
            (True, "process running"),
            (False, "process exited (0)"),
        ]
        run_mock = MagicMock()
        monkeypatch.setattr(process_base.subprocess, "run", run_mock)

        result = backend.wait_idle("4242", idle_time=5.0, timeout=30)

        assert result is True
        run_mock.assert_called_once_with(
            ["/usr/bin/stub-cli", "--version"],
            timeout=30,
            check=False,
            capture_output=True,
            text=True,
        )


class TestBaseBackendExecuteInPane:
    def test_executes_shell_command_and_returns_result(self, monkeypatch):
        backend, _manager = _make_backend_with_mock_process_manager(monkeypatch)
        completed = MagicMock(stdout="ok\n", stderr="", returncode=0)
        run_mock = MagicMock(return_value=completed)
        monkeypatch.setattr(process_base.subprocess, "run", run_mock)

        result = backend.execute_in_pane("4242", "echo hello")

        assert result == {"output": "ok\n", "exit_code": 0}
        run_mock.assert_called_once()
        expected_cmd = (
            ["cmd", "/c", "echo hello"]
            if process_base.os.name == "nt"
            else ["sh", "-lc", "echo hello"]
        )
        assert run_mock.call_args.args == (expected_cmd,)
        assert run_mock.call_args.kwargs["timeout"] == 30

    def test_respects_custom_timeout(self, monkeypatch):
        backend, _manager = _make_backend_with_mock_process_manager(monkeypatch)
        completed = MagicMock(stdout="", stderr="timed out", returncode=-1)
        run_mock = MagicMock(return_value=completed)
        monkeypatch.setattr(process_base.subprocess, "run", run_mock)

        result = backend.execute_in_pane("4242", "long cmd", timeout=120)

        assert result["exit_code"] == -1
        assert result["output"] == "timed out"
        assert run_mock.call_args.kwargs["timeout"] == 120


class TestInteractiveConsoleSpawn:
    def test_spawned_agents_are_detached_from_mcp_lifetime_by_default(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        assign_mock = MagicMock()
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_WT_TABS", "1")
        monkeypatch.delenv("WIN_AGENT_TEAMS_KILL_ON_EXIT", raising=False)
        monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
        monkeypatch.setattr(
            process_manager_mod.subprocess,
            "Popen",
            MagicMock(return_value=process),
        )
        monkeypatch.setattr(manager._job, "assign", assign_mock)

        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        assign_mock.assert_not_called()

    def test_env_can_attach_spawned_agents_to_mcp_lifetime(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("WIN_AGENT_TEAMS_KILL_ON_EXIT", "1")
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        assign_mock = MagicMock()
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_WT_TABS", "1")
        monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
        monkeypatch.setattr(
            process_manager_mod.subprocess,
            "Popen",
            MagicMock(return_value=process),
        )
        monkeypatch.setattr(manager._job, "assign", assign_mock)

        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        assign_mock.assert_called_once_with(process)

    def test_env_can_enable_claude_code_new_console_and_debug_file(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        popen_mock = MagicMock(return_value=process)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "1")
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_WT_TABS", "1")
        monkeypatch.setattr(
            process_manager_mod.subprocess,
            "CREATE_NEW_CONSOLE",
            0x10,
            raising=False,
        )
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(manager._job, "assign", lambda process: None)
        monkeypatch.setattr(
            manager,
            "_should_use_interactive_console",
            lambda backend_type, *, is_interactive=False: True,
        )
        request = _make_spawn_request()

        result = manager.spawn_process(
            request,
            ["claude", "--mcp-config", "worker.mcp.json", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        assert result.process_handle == "4242"
        command = popen_mock.call_args.args[0]
        debug_idx = command.index("--debug-file")
        prompt_sep_idx = command.index("--")
        assert debug_idx < prompt_sep_idx
        assert command[debug_idx + 1] == str(tmp_path / "team" / "worker.log")
        kwargs = popen_mock.call_args.kwargs
        assert kwargs["stdin"] is None
        assert kwargs["stdout"] is None
        assert kwargs["stderr"] is None
        assert kwargs["creationflags"] & getattr(
            process_manager_mod.subprocess, "CREATE_NEW_CONSOLE", 0
        )
        assert "[interactive console]" in (tmp_path / "team" / "worker.log").read_text(
            encoding="utf-8"
        )

    def test_env_can_disable_claude_code_interactive_console(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        popen_mock = MagicMock(return_value=process)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", "0")
        monkeypatch.setenv("USE_WINDOWS_TERMINAL", "0")
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(manager._job, "assign", lambda process: None)

        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        command = popen_mock.call_args.args[0]
        assert "--debug-file" not in command
        kwargs = popen_mock.call_args.kwargs
        assert kwargs["stdin"] == process_manager_mod.subprocess.DEVNULL
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] == process_manager_mod.subprocess.STDOUT

    def test_non_claude_interactive_backend_uses_console_by_default(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        popen_mock = MagicMock(return_value=process)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_WT_TABS", "1")
        monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
        monkeypatch.setattr(
            process_manager_mod.subprocess,
            "CREATE_NEW_CONSOLE",
            0x10,
            raising=False,
        )
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(manager._job, "assign", lambda process: None)
        monkeypatch.setattr(
            manager,
            "_should_use_interactive_console",
            lambda backend_type, *, is_interactive=False: True,
        )

        manager.spawn_process(
            _make_spawn_request(),
            ["codex", "exec", "do stuff"],
            {},
            "codex",
            is_interactive=True,
        )

        kwargs = popen_mock.call_args.kwargs
        assert kwargs["stdin"] is None
        assert kwargs["stdout"] is None
        assert kwargs["stderr"] is None
        assert kwargs["creationflags"] & getattr(
            process_manager_mod.subprocess, "CREATE_NEW_CONSOLE", 0
        )


class TestWindowsTerminalTabSpawn:
    def _prep_manager(self, monkeypatch, tmp_path, *, pid=4242):
        manager = process_manager_mod.WindowsProcessManager()
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
        monkeypatch.delenv("WIN_AGENT_TEAMS_NO_WT_TABS", raising=False)
        # Force the interactive decision on so the branch runs on Linux CI too.
        monkeypatch.setattr(
            manager,
            "_should_use_interactive_console",
            lambda backend_type, *, is_interactive=False: True,
        )
        monkeypatch.setattr(
            process_manager_mod.shutil,
            "which",
            lambda name: "C:\\wt.exe" if name == "wt.exe" else None,
        )
        popen_mock = MagicMock(return_value=MagicMock(pid=999))
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(manager, "_await_tab_pid", lambda sidecar: pid)
        monkeypatch.setattr(
            process_manager_mod, "creation_token", lambda handle: f"tok-{handle}"
        )
        return manager, popen_mock

    def test_interactive_spawn_launches_windows_terminal_tab(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, popen_mock = self._prep_manager(monkeypatch, tmp_path)

        result = manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {"CLAUDECODE": "1"},
            "claude-code",
            is_interactive=True,
        )

        assert result.process_handle == "4242"
        cmd = popen_mock.call_args.args[0]
        assert cmd[0] == "C:\\wt.exe"
        assert cmd[1:5] == ["-w", "wt-team-team", "nt", "--title"]
        assert cmd[5] == "worker@team"
        # The tab title must be pinned so the agent CLI can't overwrite it.
        assert "--suppressApplicationTitle" in cmd
        assert "powershell" in cmd
        assert cmd[-1].endswith("worker.launch.ps1")
        assert "4242" in manager._tabs

        wrapper = tmp_path / "team" / "worker.launch.ps1"
        text = wrapper.read_text(encoding="utf-8")
        assert "$env:CLAUDECODE = '1'" in text
        assert "& 'claude'" in text
        # claude-code still gets --debug-file so its log is captured.
        assert "--debug-file" in text

    def test_tab_lifecycle_dispatch(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, _ = self._prep_manager(monkeypatch, tmp_path)
        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        assert "windows terminal tab" in manager.capture("4242").lower()
        alive, _ = manager.health_check("4242")
        assert alive
        # send must be a no-op for a tab (no stdin pipe) and must not raise.
        manager.send("4242", "hi")

        manager.kill_process("4242")

        assert "4242" not in manager._tabs
        assert not (tmp_path / "team" / "worker.launch.ps1").exists()

    def test_kill_terminates_agent_subtree_not_the_shell(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, _ = self._prep_manager(monkeypatch, tmp_path)
        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )
        killed: list[str] = []
        # Shell (4242) is alive with one agent child (555); after the child is
        # killed the shell exits on its own (exit 0) so the tab closes.
        monkeypatch.setattr(manager, "_pid_alive", lambda handle: True)
        monkeypatch.setattr(manager, "_child_pids", lambda handle: [555])
        monkeypatch.setattr(manager, "_kill_pid", killed.append)
        monkeypatch.setattr(manager, "_win_wait_pid_exit", lambda handle, t: True)

        manager.kill_process("4242")

        # Only the agent child is killed, never the wrapper shell (4242).
        assert killed == ["555"]
        assert "4242" not in manager._tabs

    def test_tab_health_reports_exited_on_token_mismatch(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, _ = self._prep_manager(monkeypatch, tmp_path)
        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )
        # Simulate the launcher PID dying (token now unreadable).
        monkeypatch.setattr(process_manager_mod, "creation_token", lambda handle: None)

        alive, _ = manager.health_check("4242")

        assert not alive

    def test_spawn_raises_when_pid_never_reported(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, _ = self._prep_manager(monkeypatch, tmp_path)
        monkeypatch.setattr(manager, "_await_tab_pid", lambda sidecar: None)

        with pytest.raises(process_manager_mod.WindowsTerminalTabSpawnError):
            manager.spawn_process(
                _make_spawn_request(),
                ["claude", "--", "do stuff"],
                {},
                "claude-code",
                is_interactive=True,
            )

    def test_no_wt_tabs_env_forces_classic_console(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager, popen_mock = self._prep_manager(monkeypatch, tmp_path)
        monkeypatch.setenv("WIN_AGENT_TEAMS_NO_WT_TABS", "1")
        monkeypatch.setattr(
            process_manager_mod.subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False
        )

        manager.spawn_process(
            _make_spawn_request(),
            ["claude", "--", "do stuff"],
            {},
            "claude-code",
            is_interactive=True,
        )

        assert manager._tabs == {}
        assert popen_mock.call_args.args[0][0] == "claude"

    def test_window_id_prefixes_numeric_team(self):
        manager = process_manager_mod.WindowsProcessManager()
        assert manager._tab_window_id("123") == "wt-team-123"

    def test_wrapper_quotes_embedded_single_quotes(self, tmp_path):
        manager = process_manager_mod.WindowsProcessManager()
        wrapper = tmp_path / "w.launch.ps1"
        sidecar = tmp_path / "w.pid"

        manager._write_tab_wrapper(
            wrapper, "C:\\proj", ["claude", "it's"], {"K": "a'b"}, sidecar
        )

        text = wrapper.read_text(encoding="utf-8")
        assert "$env:K = 'a''b'" in text
        assert "'it''s'" in text
        # Always exit 0 so Windows Terminal closes the tab on completion/kill.
        assert "exit 0" in text


class TestProcessLivenessFallback:
    def test_pid_alive_detects_external_running_process(self):
        manager = process_manager_mod.WindowsProcessManager()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            assert manager._pid_alive(str(process.pid)) is True
        finally:
            process.terminate()
            process.wait(timeout=10)


class TestWindowsTerminalTail:
    def test_opens_windows_terminal_by_default_when_available(
        self, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        popen_mock = MagicMock()
        monkeypatch.delenv("USE_WINDOWS_TERMINAL", raising=False)
        monkeypatch.setattr(
            process_manager_mod.shutil,
            "which",
            lambda name: "C:\\WindowsApps\\wt.exe" if name == "wt.exe" else None,
        )
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        log_path = tmp_path / "worker.log"

        manager._open_windows_terminal_tail("team", "worker", log_path)

        popen_mock.assert_called_once()
        command = popen_mock.call_args.args[0]
        assert command[:6] == [
            "C:\\WindowsApps\\wt.exe",
            "-w",
            "0",
            "nt",
            "--title",
            "worker@team",
        ]
        assert f"Get-Content -LiteralPath '{log_path}' -Wait -Tail 80" in command

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_env_can_disable_windows_terminal_tail(self, monkeypatch, tmp_path, value):
        manager = process_manager_mod.WindowsProcessManager()
        popen_mock = MagicMock()
        monkeypatch.setenv("USE_WINDOWS_TERMINAL", value)
        monkeypatch.setattr(
            process_manager_mod.shutil,
            "which",
            lambda name: "C:\\WindowsApps\\wt.exe" if name == "wt.exe" else None,
        )
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)

        manager._open_windows_terminal_tail("team", "worker", tmp_path / "worker.log")

        popen_mock.assert_not_called()

    def test_skips_windows_terminal_when_wt_is_missing(self, monkeypatch, tmp_path):
        manager = process_manager_mod.WindowsProcessManager()
        popen_mock = MagicMock()
        monkeypatch.delenv("USE_WINDOWS_TERMINAL", raising=False)
        monkeypatch.setattr(process_manager_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)

        manager._open_windows_terminal_tail("team", "worker", tmp_path / "worker.log")

        popen_mock.assert_not_called()


class TestPlatformProcessManagerSelection:
    def test_defaults_to_terminal_on_posix_and_windows_manager_on_windows(self):
        if process_manager_mod.os.name == "nt":
            assert isinstance(
                process_manager_mod.process_manager,
                process_manager_mod.WindowsProcessManager,
            )
        elif (
            process_manager_mod.os.environ.get(
                process_manager_mod._LINUX_LAUNCHER_ENV, ""
            )
            .strip()
            .lower()
            == process_manager_mod._TMUX_LAUNCHER_VALUE
        ):
            assert isinstance(
                process_manager_mod.process_manager,
                process_manager_mod.TmuxProcessManager,
            )
        else:
            assert isinstance(
                process_manager_mod.process_manager,
                process_manager_mod.LinuxTerminalProcessManager,
            )


class TestTmuxProcessManager:
    def test_spawn_inside_tmux_uses_split_window_and_returns_pane_pid(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.TmuxProcessManager()
        run_mock = MagicMock(
            return_value=MagicMock(stdout="@7\t%42\t4242\n", returncode=0)
        )
        monkeypatch.setenv("TMUX", "tmux-session-token")
        monkeypatch.delenv("USE_TMUX_WINDOWS", raising=False)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(process_manager_mod.subprocess, "run", run_mock)

        result = manager.spawn_process(
            _make_spawn_request(),
            ["codex", "exec", "do stuff"],
            {"AGENT_NAME": "worker"},
            "codex",
            is_interactive=True,
        )

        assert result.process_handle == "4242"
        assert result.backend_type == "codex"
        command = run_mock.call_args.args[0]
        assert command[:5] == [
            "tmux",
            "split-window",
            "-dP",
            "-F",
            "#{window_id}\t#{pane_id}\t#{pane_pid}",
        ]
        assert "cd " in command[-1]
        assert "AGENT_NAME=worker" in command[-1]
        assert "env AGENT_NAME=worker exec" not in command[-1]
        assert "exec codex exec 'do stuff'" in command[-1]
        assert manager._processes["4242"].target_id == "%42"

    def test_spawn_inside_tmux_can_use_windows(self, _make_spawn_request, monkeypatch):
        manager = process_manager_mod.TmuxProcessManager()
        monkeypatch.setenv("TMUX", "tmux-session-token")
        monkeypatch.setenv("USE_TMUX_WINDOWS", "1")

        command, target_kind = manager._build_tmux_spawn_args(
            _make_spawn_request(),
            "exec worker",
        )

        assert target_kind == "window"
        assert command[:5] == [
            "tmux",
            "new-window",
            "-dP",
            "-F",
            "#{window_id}\t#{pane_id}\t#{pane_pid}",
        ]

    def test_explicit_tmux_target_uses_split_window_even_without_tmux_env(
        self, _make_spawn_request, monkeypatch
    ):
        manager = process_manager_mod.TmuxProcessManager()
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("WIN_AGENT_TEAMS_TMUX_TARGET", "codex-lead")

        command, target_kind = manager._build_tmux_spawn_args(
            _make_spawn_request(),
            "exec worker",
        )

        assert target_kind == "pane"
        assert command[:7] == [
            "tmux",
            "split-window",
            "-dP",
            "-t",
            "codex-lead",
            "-F",
            "#{window_id}\t#{pane_id}\t#{pane_pid}",
        ]

    def test_explicit_tmux_target_can_use_windows(
        self, _make_spawn_request, monkeypatch
    ):
        manager = process_manager_mod.TmuxProcessManager()
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("WIN_AGENT_TEAMS_TMUX_TARGET", "codex-lead")
        monkeypatch.setenv("USE_TMUX_WINDOWS", "1")

        command, target_kind = manager._build_tmux_spawn_args(
            _make_spawn_request(),
            "exec worker",
        )

        assert target_kind == "window"
        assert command[:7] == [
            "tmux",
            "new-window",
            "-dP",
            "-t",
            "codex-lead",
            "-F",
            "#{window_id}\t#{pane_id}\t#{pane_pid}",
        ]

    def test_spawn_outside_tmux_creates_detached_session(
        self, _make_spawn_request, monkeypatch
    ):
        manager = process_manager_mod.TmuxProcessManager()
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setattr(manager, "_tmux_session_exists", lambda session_name: False)

        command, target_kind = manager._build_tmux_spawn_args(
            _make_spawn_request(team_name="session-123"),
            "exec worker",
        )

        assert target_kind == "window"
        assert command[:5] == [
            "tmux",
            "new-session",
            "-dP",
            "-s",
            "win-agent-teams-session-123",
        ]

    def test_spawn_outside_tmux_reuses_detached_session(
        self, _make_spawn_request, monkeypatch
    ):
        manager = process_manager_mod.TmuxProcessManager()
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setattr(manager, "_tmux_session_exists", lambda session_name: True)

        command, target_kind = manager._build_tmux_spawn_args(
            _make_spawn_request(team_name="session-123"),
            "exec worker",
        )

        assert target_kind == "window"
        assert command[:5] == [
            "tmux",
            "new-window",
            "-dP",
            "-t",
            "win-agent-teams-session-123",
        ]

    def test_kill_process_uses_tmux_target_when_known(self, monkeypatch, tmp_path):
        manager = process_manager_mod.TmuxProcessManager()
        run_mock = MagicMock()
        monkeypatch.setattr(process_manager_mod.subprocess, "run", run_mock)
        manager._processes["4242"] = process_manager_mod.TmuxProcessInfo(
            pid=4242,
            name="worker",
            agent_id="worker@team",
            team_name="team",
            backend="codex",
            target_id="@7",
            pane_id="%42",
            log_path=tmp_path / "worker.log",
            started_at=1.0,
        )

        manager.kill_process("4242")

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        assert command[0].endswith("tmux")
        assert command[1:] == ["kill-window", "-t", "@7"]
        assert run_mock.call_args.kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
        }


class TestLinuxTerminalProcessManager:
    def test_spawn_opens_terminal_and_returns_terminal_pid(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        process = MagicMock(pid=4242)
        popen_mock = MagicMock(return_value=process)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(
            manager,
            "_discover_terminal",
            lambda: "/usr/bin/gnome-terminal",
        )
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)

        result = manager.spawn_process(
            _make_spawn_request(),
            ["codex", "exec", "do stuff"],
            {"AGENT_NAME": "worker"},
            "codex",
            is_interactive=True,
        )

        assert result.process_handle == "4242"
        command = popen_mock.call_args.args[0]
        assert command[:6] == [
            "/usr/bin/gnome-terminal",
            "--wait",
            "--title",
            "worker@team",
            "--",
            "bash",
        ]
        assert "printf '%s\\n' \"$$\"" in command[-1]
        assert str(tmp_path / "team" / "worker.pid") in command[-1]
        assert "AGENT_NAME=worker" in command[-1]
        assert "env AGENT_NAME=worker exec" not in command[-1]
        assert "exec codex exec 'do stuff'" in command[-1]
        kwargs = popen_mock.call_args.kwargs
        assert kwargs["stdin"] is None
        assert kwargs["stdout"].name == str(tmp_path / "team" / "worker.log")
        assert kwargs["stderr"] == process_manager_mod.subprocess.STDOUT
        assert kwargs["start_new_session"] is True

    def test_discovers_terminal_from_env_override(self, monkeypatch):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        monkeypatch.setenv("WIN_AGENT_TEAMS_LINUX_TERMINAL", "custom-terminal")
        monkeypatch.setattr(
            process_manager_mod.shutil,
            "which",
            lambda name: (
                "/opt/bin/custom-terminal" if name == "custom-terminal" else None
            ),
        )

        assert manager._discover_terminal() == "/opt/bin/custom-terminal"

    def test_skips_qterminal_when_an_instance_is_already_running(
        self, monkeypatch
    ):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        terminal_paths = {
            "qterminal": "/usr/bin/qterminal",
            "xterm": "/usr/bin/xterm",
        }

        def fake_which(name: str) -> str | None:
            return terminal_paths.get(name)

        monkeypatch.delenv("WIN_AGENT_TEAMS_LINUX_TERMINAL", raising=False)
        monkeypatch.setattr(process_manager_mod.shutil, "which", fake_which)
        monkeypatch.setattr(
            manager,
            "_process_name_running",
            lambda name: name == "qterminal",
        )

        assert manager._discover_terminal() == "/usr/bin/xterm"

    def test_health_follows_agent_pid_after_terminal_launcher_exits(
        self, tmp_path, monkeypatch
    ):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        process = MagicMock(pid=4242)
        process.poll.return_value = 0
        log_path = tmp_path / "worker.log"
        pid_path = tmp_path / "worker.pid"
        pid_path.write_text("5150\n", encoding="utf-8")
        manager._processes["4242"] = process_manager_mod.LinuxTerminalProcessInfo(
            pid=4242,
            name="worker",
            agent_id="worker@team",
            team_name="team",
            backend="codex",
            terminal_process=process,
            log_path=log_path,
            agent_pid_path=pid_path,
            started_at=0.0,
        )
        monkeypatch.setattr(manager, "_pid_alive", lambda handle: handle == "5150")

        assert manager.health_check("4242") == (True, "agent process running")

    def test_health_allows_short_pid_file_race_after_clean_launcher_exit(
        self, tmp_path
    ):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        process = MagicMock(pid=4242)
        process.poll.return_value = 0
        manager._processes["4242"] = process_manager_mod.LinuxTerminalProcessInfo(
            pid=4242,
            name="worker",
            agent_id="worker@team",
            team_name="team",
            backend="codex",
            terminal_process=process,
            log_path=tmp_path / "worker.log",
            agent_pid_path=tmp_path / "worker.pid",
            started_at=process_manager_mod.time.time(),
        )

        assert manager.health_check("4242") == (
            True,
            "terminal launcher exited; waiting for agent pid",
        )

    def test_kill_stops_agent_pid_and_terminal_process(
        self, tmp_path, monkeypatch
    ):
        manager = process_manager_mod.LinuxTerminalProcessManager()
        process = MagicMock(pid=4242)
        process.poll.return_value = None
        pid_path = tmp_path / "worker.pid"
        pid_path.write_text("5150\n", encoding="utf-8")
        manager._processes["4242"] = process_manager_mod.LinuxTerminalProcessInfo(
            pid=4242,
            name="worker",
            agent_id="worker@team",
            team_name="team",
            backend="codex",
            terminal_process=process,
            log_path=tmp_path / "worker.log",
            agent_pid_path=pid_path,
            started_at=1.0,
        )
        kill_mock = MagicMock()
        monkeypatch.setattr(manager, "_pid_alive", lambda handle: handle == "5150")
        monkeypatch.setattr(manager, "_wait_pid_exit", lambda pid, timeout_s: True)
        monkeypatch.setattr(process_manager_mod.os, "kill", kill_mock)

        manager.kill_process("4242")

        kill_mock.assert_called_once_with(5150, process_manager_mod.signal.SIGTERM)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10.0)

    def test_gnome_terminal_command_uses_title_and_shell(self):
        manager = process_manager_mod.LinuxTerminalProcessManager()

        command = manager._terminal_command(
            "/usr/bin/gnome-terminal",
            "worker@team",
            "exec worker",
        )

        assert command == [
            "/usr/bin/gnome-terminal",
            "--wait",
            "--title",
            "worker@team",
            "--",
            "bash",
            "-lc",
            "exec worker",
        ]

    def test_qterminal_command_omits_title(self):
        manager = process_manager_mod.LinuxTerminalProcessManager()

        command = manager._terminal_command(
            "/usr/bin/qterminal",
            "worker@team",
            "exec worker",
        )

        assert command == [
            "/usr/bin/qterminal",
            "-e",
            "bash",
            "-lc",
            "exec worker",
        ]
