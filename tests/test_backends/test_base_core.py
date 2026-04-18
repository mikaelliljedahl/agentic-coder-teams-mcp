"""Core BaseBackend type and setup tests."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_teams.backends.base import Backend, HealthStatus, SpawnRequest, SpawnResult
from tests.test_backends._base_support import (
    _make_backend_with_mock_controller,
    _StubBackend,
)


class TestSpawnRequest:
    def test_creates_with_all_fields(self, tmp_path: Path):
        req = SpawnRequest(
            agent_id="a@t",
            name="a",
            team_name="t",
            prompt="do",
            model="sonnet",
            agent_type="general-purpose",
            color="blue",
            cwd=str(tmp_path),
            lead_session_id="sess-1",
        )
        assert req.agent_id == "a@t"
        assert req.name == "a"
        assert req.team_name == "t"
        assert req.prompt == "do"
        assert req.model == "sonnet"
        assert req.agent_type == "general-purpose"
        assert req.color == "blue"
        assert req.cwd == str(tmp_path)
        assert req.lead_session_id == "sess-1"
        assert req.permission_mode == "default"
        assert req.plan_mode_required is False
        assert req.extra is None

    def test_frozen_raises_on_mutation(self, _make_spawn_request):
        req = _make_spawn_request()
        with pytest.raises(FrozenInstanceError):
            # ``setattr`` still triggers ``__setattr__`` on a frozen dataclass,
            # so the runtime guard fires identically to direct assignment while
            # keeping static type checkers out of a known-bad-by-design path.
            setattr(req, "name", "other")  # noqa: B010

    def test_plan_mode_required_default_false(self, _make_spawn_request):
        req = _make_spawn_request()
        assert req.plan_mode_required is False

    def test_extra_accepts_dict(self, _make_spawn_request):
        req = _make_spawn_request(extra={"key": "val"})
        assert req.extra == {"key": "val"}

    def test_permission_mode_defaults_to_default(self, _make_spawn_request):
        req = _make_spawn_request()
        assert req.permission_mode == "default"


# ---------------------------------------------------------------------------
# SpawnResult dataclass
# ---------------------------------------------------------------------------


class TestSpawnResult:
    def test_creates_with_required_fields(self):
        result = SpawnResult(process_handle="%42", backend_type="claude-code")
        assert result.process_handle == "%42"
        assert result.backend_type == "claude-code"

    def test_frozen_raises_on_mutation(self):
        result = SpawnResult(process_handle="%1", backend_type="codex")
        with pytest.raises(FrozenInstanceError):
            # Use ``setattr`` so static checkers don't flag the intentional
            # frozen-mutation probe; runtime guard still fires.
            setattr(result, "process_handle", "%2")  # noqa: B010


# ---------------------------------------------------------------------------
# HealthStatus dataclass
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_creates_with_alive_flag(self):
        health = HealthStatus(alive=True)
        assert health.alive is True
        assert health.detail == ""

    def test_detail_defaults_to_empty_string(self):
        health = HealthStatus(alive=False)
        assert health.detail == ""

    def test_detail_accepts_custom_string(self):
        health = HealthStatus(alive=True, detail="tmux pane check")
        assert health.detail == "tmux pane check"

    def test_frozen_raises_on_mutation(self):
        health = HealthStatus(alive=True)
        with pytest.raises(FrozenInstanceError):
            # Use ``setattr`` so static checkers don't flag the intentional
            # frozen-mutation probe; runtime guard still fires.
            setattr(health, "alive", False)  # noqa: B010


# ---------------------------------------------------------------------------
# Backend Protocol runtime check
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    def test_stub_backend_satisfies_protocol(self):
        backend = _StubBackend()
        assert isinstance(backend, Backend)

    def test_plain_object_does_not_satisfy_protocol(self):
        assert not isinstance(object(), Backend)


# ---------------------------------------------------------------------------
# BaseBackend.controller property
# ---------------------------------------------------------------------------


class TestBaseBackendController:
    @patch("claude_teams.backends.tmux_base.TmuxCLIController")
    def test_creates_controller_lazily(self, mock_ctrl_cls: MagicMock):
        backend = _StubBackend()
        ctrl = backend.controller
        mock_ctrl_cls.assert_called_once()
        assert ctrl is mock_ctrl_cls.return_value

    @patch("claude_teams.backends.tmux_base.TmuxCLIController")
    def test_returns_same_controller_on_subsequent_calls(
        self, mock_ctrl_cls: MagicMock
    ):
        backend = _StubBackend()
        ctrl1 = backend.controller
        ctrl2 = backend.controller
        assert ctrl1 is ctrl2
        mock_ctrl_cls.assert_called_once()

    def test_accepts_injected_controller(self):
        backend = _StubBackend()
        mock_ctrl = MagicMock()
        backend._controller = mock_ctrl
        assert backend.controller is mock_ctrl


# ---------------------------------------------------------------------------
# BaseBackend.is_available
# ---------------------------------------------------------------------------


class TestBaseBackendIsInteractive:
    def test_defaults_to_false(self):
        backend = _StubBackend()
        assert backend.is_interactive is False


class TestBaseBackendPermissionArgs:
    def test_default_mode_uses_backend_defaults(self, _make_spawn_request):
        class _AutoApproveBackend(_StubBackend):
            def default_permission_args(self) -> list[str]:
                return ["--auto"]

        backend = _AutoApproveBackend()
        assert backend.permission_args(_make_spawn_request()) == ["--auto"]

    def test_require_approval_strips_backend_default_flags(self, _make_spawn_request):
        class _AutoApproveBackend(_StubBackend):
            def default_permission_args(self) -> list[str]:
                return ["--auto"]

        backend = _AutoApproveBackend()
        request = _make_spawn_request(permission_mode="require_approval")
        assert backend.permission_args(request) == []

    def test_bypass_raises_when_backend_does_not_support_it(self, _make_spawn_request):
        backend = _StubBackend()
        request = _make_spawn_request(permission_mode="bypass")
        with pytest.raises(ValueError, match="permission_mode='bypass'"):
            backend.permission_args(request)


class TestBaseBackendRetainPaneAfterExit:
    def test_sets_remain_on_exit_option(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.retain_pane_after_exit("%42")

        mock_ctrl._run_tmux_command.assert_called_once_with(
            ["set-option", "-p", "-t", "%42", "remain-on-exit", "on"]
        )


class TestBaseBackendIsAvailable:
    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_true_when_binary_found(self, mock_which: MagicMock):
        mock_which.return_value = "/usr/bin/stub-cli"
        backend = _StubBackend()
        assert backend.is_available() is True
        mock_which.assert_called_once_with("stub-cli")

    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_false_when_binary_not_found(self, mock_which: MagicMock):
        mock_which.return_value = None
        backend = _StubBackend()
        assert backend.is_available() is False


# ---------------------------------------------------------------------------
# BaseBackend.discover_binary
# ---------------------------------------------------------------------------


class TestBaseBackendDiscoverBinary:
    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_full_path_when_found(self, mock_which: MagicMock):
        mock_which.return_value = "/usr/local/bin/stub-cli"
        backend = _StubBackend()
        assert backend.discover_binary() == "/usr/local/bin/stub-cli"

    @patch("claude_teams.backends.base.shutil.which")
    def test_raises_file_not_found_when_missing(self, mock_which: MagicMock):
        mock_which.return_value = None
        backend = _StubBackend()
        with pytest.raises(FileNotFoundError, match="stub-cli"):
            backend.discover_binary()


# ---------------------------------------------------------------------------
# BaseBackend.spawn
# ---------------------------------------------------------------------------
