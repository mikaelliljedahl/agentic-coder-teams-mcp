"""Runtime BaseBackend controller-operation tests."""

import shlex
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from claude_code_tools.tmux_cli_controller import TmuxCLIController

from claude_teams.backends.base import SpawnRequest, SpawnResult
from tests.test_backends._base_support import (
    _InvalidEnvBackend,
    _make_backend_with_mock_controller,
    _StubBackend,
)


class _DangerousEnvBackend(_StubBackend):
    """Stub backend returning a shell-metacharacter env value.

    Used to verify ``build_env`` values are routed through ``shlex.quote``
    before being interpolated into the pane startup command.
    """

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"DANGER": "$(whoami); :"}


class TestBaseBackendSpawn:
    def test_returns_spawn_result_on_success(self, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.2"
        request = _make_spawn_request()

        result = backend.spawn(request)

        assert isinstance(result, SpawnResult)
        assert result.process_handle == "remote:1.2"
        assert result.backend_type == "stub"

    def test_calls_launch_cli_with_full_command(self, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        request = _make_spawn_request()

        backend.spawn(request)

        mock_ctrl.launch_cli.assert_called_once()
        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        assert "cd" in full_cmd
        assert "STUB_MODE=" in full_cmd
        assert "stub-cli" in full_cmd

    def test_raises_runtime_error_when_launch_fails(self, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = None
        request = _make_spawn_request()

        with pytest.raises(RuntimeError, match="Failed to create tmux pane"):
            backend.spawn(request)

    def test_includes_env_prefix_in_command(self, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        request = _make_spawn_request()

        backend.spawn(request)

        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        assert "STUB_MODE=" in full_cmd

    def test_rejects_invalid_env_var_name(self, _make_spawn_request):
        backend = _InvalidEnvBackend()
        request = _make_spawn_request()

        with pytest.raises(ValueError, match="Invalid environment variable"):
            backend.spawn(request)


# ---------------------------------------------------------------------------
# BaseBackend.health_check
# ---------------------------------------------------------------------------


class TestBaseBackendHealthCheck:
    def test_returns_alive_when_pane_exists_by_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
            {"id": "%42", "formatted_id": "remote:1.1"},
        ]
        mock_ctrl._run_tmux_command.return_value = ("0", 0)

        status = backend.health_check("%42")

        assert status.alive is True
        assert status.detail == "tmux pane check"

    def test_returns_alive_when_pane_exists_by_formatted_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
            {"id": "%42", "formatted_id": "remote:1.1"},
        ]
        mock_ctrl._run_tmux_command.return_value = ("0", 0)

        status = backend.health_check("remote:1.1")

        assert status.alive is True

    def test_returns_dead_when_process_exited_but_pane_retained(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%42", "formatted_id": "remote:1.1"},
        ]
        mock_ctrl._run_tmux_command.return_value = ("1", 0)

        status = backend.health_check("%42")

        assert status.alive is False
        assert status.detail == "process exited (pane retained)"

    def test_returns_dead_when_pane_missing(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
        ]

        status = backend.health_check("%42")

        assert status.alive is False
        assert status.detail == "tmux pane not found"

    def test_returns_dead_when_no_panes(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = []

        status = backend.health_check("%42")

        assert status.alive is False
        assert status.detail == "tmux pane not found"


# ---------------------------------------------------------------------------
# BaseBackend.kill
# ---------------------------------------------------------------------------


class TestBaseBackendKill:
    def test_calls_controller_kill_pane(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.kill("%42")

        mock_ctrl.kill_pane.assert_called_once_with(pane_id="%42")

    def test_accepts_formatted_pane_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.kill("remote:1.2")

        mock_ctrl.kill_pane.assert_called_once_with(pane_id="remote:1.2")


# ---------------------------------------------------------------------------
# BaseBackend.graceful_shutdown
# ---------------------------------------------------------------------------


class TestBaseBackendGracefulShutdown:
    def test_sends_interrupt_and_waits(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = True

        result = backend.graceful_shutdown("%42", timeout_s=5.0)

        assert result is True
        mock_ctrl.send_interrupt.assert_called_once_with(pane_id="%42")
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=1.0,
            timeout=5,
        )

    def test_returns_false_when_timeout_exceeded(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = False

        result = backend.graceful_shutdown("%42", timeout_s=2.0)

        assert result is False
        mock_ctrl.send_interrupt.assert_called_once()


# ---------------------------------------------------------------------------
# BaseBackend.capture
# ---------------------------------------------------------------------------


class TestBaseBackendCapture:
    def test_captures_full_buffer(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.capture_pane.return_value = "line 1\nline 2\n"

        output = backend.capture("%42")

        assert output == "line 1\nline 2\n"
        mock_ctrl.capture_pane.assert_called_once_with(pane_id="%42", lines=None)

    def test_captures_limited_lines(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.capture_pane.return_value = "last line\n"

        output = backend.capture("%42", lines=1)

        assert output == "last line\n"
        mock_ctrl.capture_pane.assert_called_once_with(pane_id="%42", lines=1)


# ---------------------------------------------------------------------------
# BaseBackend.send
# ---------------------------------------------------------------------------


class TestBaseBackendSend:
    def test_sends_text_with_enter(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.send("%42", "hello world")

        mock_ctrl.send_keys.assert_called_once_with(
            "hello world",
            pane_id="%42",
            enter=True,
        )

    def test_sends_text_without_enter(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.send("%42", "partial", enter=False)

        mock_ctrl.send_keys.assert_called_once_with(
            "partial",
            pane_id="%42",
            enter=False,
        )


# ---------------------------------------------------------------------------
# BaseBackend.wait_idle
# ---------------------------------------------------------------------------


class TestBaseBackendWaitIdle:
    def test_waits_with_defaults(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = True

        result = backend.wait_idle("%42")

        assert result is True
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=2.0,
            timeout=None,
        )

    def test_waits_with_custom_params(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = False

        result = backend.wait_idle("%42", idle_time=5.0, timeout=30)

        assert result is False
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=5.0,
            timeout=30,
        )


# ---------------------------------------------------------------------------
# BaseBackend.execute_in_pane
# ---------------------------------------------------------------------------


class TestBaseBackendExecuteInPane:
    def test_executes_command_and_returns_result(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.execute.return_value = {"output": "ok", "exit_code": 0}

        result = backend.execute_in_pane("%42", "echo hello")

        assert result == {"output": "ok", "exit_code": 0}
        mock_ctrl.execute.assert_called_once_with(
            "echo hello",
            pane_id="%42",
            timeout=30,
        )

    def test_respects_custom_timeout(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.execute.return_value = {"output": "", "exit_code": -1}

        result = backend.execute_in_pane("%42", "long cmd", timeout=120)

        assert result["exit_code"] == -1
        mock_ctrl.execute.assert_called_once_with(
            "long cmd",
            pane_id="%42",
            timeout=120,
        )


# ---------------------------------------------------------------------------
# BaseBackend.spawn — shlex.quote regression contract
# ---------------------------------------------------------------------------


class TestBaseBackendSpawnQuotingContract:
    """Lock the shell-quoting contract so prompt/cwd/env are never interpolated.

    ``BaseBackend.spawn`` concatenates a string command that tmux runs in a
    shell, so *every* user-controlled argument must be ``shlex.quote``d.
    Dropping any of these quotes silently turns a prompt containing
    ``$(rm -rf /)`` into live command substitution. These tests re-parse
    the launched command with ``shlex.split`` and assert the dangerous
    tokens reappear as a single argv element, proving the quoting held.
    """

    def test_prompt_metacharacters_are_shell_quoted(self, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        dangerous = "$(rm -rf /); echo `id`; && :"
        request = _make_spawn_request(prompt=dangerous)

        backend.spawn(request)

        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        parsed = shlex.split(full_cmd)
        assert dangerous in parsed, (
            f"prompt was not preserved as a single argv token: {parsed!r}"
        )

    def test_cwd_with_spaces_is_shell_quoted(self, tmp_path: Path, _make_spawn_request):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        spaced = str(tmp_path / "has space" / "dir")
        request = _make_spawn_request(cwd=spaced)

        backend.spawn(request)

        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        parsed = shlex.split(full_cmd)
        assert parsed[0] == "cd"
        assert parsed[1] == spaced

    def test_env_var_values_are_shell_quoted(self, _make_spawn_request):
        backend = _DangerousEnvBackend()
        mock_ctrl = MagicMock(spec=TmuxCLIController)
        backend._controller = mock_ctrl
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        request = _make_spawn_request()

        backend.spawn(request)

        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        parsed = shlex.split(full_cmd)
        assert "DANGER=$(whoami); :" in parsed, (
            f"env value was not preserved as a single argv token: {parsed!r}"
        )
