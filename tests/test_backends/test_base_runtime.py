"""Runtime BaseBackend process-manager operation tests."""

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
        assert run_mock.call_args.args == (["cmd", "/c", "echo hello"],)
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
    def test_claude_code_uses_new_console_and_debug_file(
        self, _make_spawn_request, monkeypatch, tmp_path
    ):
        manager = process_manager_mod.WindowsProcessManager()
        process = MagicMock(pid=4242)
        popen_mock = MagicMock(return_value=process)
        monkeypatch.setenv("WIN_AGENT_TEAMS_LOG_DIR", str(tmp_path))
        monkeypatch.delenv("WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE", raising=False)
        monkeypatch.setattr(process_manager_mod.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(manager._job, "assign", lambda process: None)
        request = _make_spawn_request()

        result = manager.spawn_process(
            request,
            ["claude", "--mcp-config", "worker.mcp.json", "--", "do stuff"],
            {},
            "claude-code",
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

    def test_env_can_disable_interactive_console(
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
        )

        command = popen_mock.call_args.args[0]
        assert "--debug-file" not in command
        kwargs = popen_mock.call_args.kwargs
        assert kwargs["stdin"] == process_manager_mod.subprocess.PIPE
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] == process_manager_mod.subprocess.STDOUT


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
