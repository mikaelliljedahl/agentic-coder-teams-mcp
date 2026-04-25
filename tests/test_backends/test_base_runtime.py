"""Runtime BaseBackend process-manager operation tests."""

from unittest.mock import MagicMock

import pytest

from claude_teams.backends import process_base
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
