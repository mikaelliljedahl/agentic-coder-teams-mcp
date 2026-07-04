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
                err.winerror = 5
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
            err.winerror = 2
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
