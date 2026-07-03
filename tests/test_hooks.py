"""Tests for the state-marker hook emitter (``claude_teams.hooks``)."""

import io
import json
import sys
from pathlib import Path

import pytest

from claude_teams import hooks


def _marker_path(session_dir: Path, agent: str) -> Path:
    return session_dir / f"state-{agent}.json"


def _read_marker(session_dir: Path, agent: str) -> dict:
    return json.loads(_marker_path(session_dir, agent).read_text(encoding="utf-8"))


class TestEmit:
    def test_emit_writes_running_marker_for_sessionstart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"hook_event_name": "SessionStart", "session_id": "s1"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        marker = _read_marker(tmp_path, "worker")
        assert marker["state"] == "running"
        assert marker["event"] == "SessionStart"
        assert isinstance(marker["ts"], float)

    def test_emit_writes_waiting_marker_for_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"hook_event_name": "Stop", "session_id": "s1"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        marker = _read_marker(tmp_path, "worker")
        assert marker["state"] == "waiting"

    def test_emit_maps_subagentstop_to_waiting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"hook_event_name": "SubagentStop"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        marker = _read_marker(tmp_path, "worker")
        assert marker["state"] == "waiting"

    @pytest.mark.parametrize(
        "event_name", ["PreToolUse", "PostToolUse", "UserPromptSubmit"]
    )
    def test_emit_maps_tooluse_and_prompt_to_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event_name: str
    ) -> None:
        payload = json.dumps({"hook_event_name": event_name})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        marker = _read_marker(tmp_path, "worker")
        assert marker["state"] == "running"

    def test_emit_atomic_write_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"hook_event_name": "SessionStart"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_emit_corrupt_stdin_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("{not valid json"))

        hooks.emit(session_dir=tmp_path, agent="worker")

        assert not _marker_path(tmp_path, "worker").exists()

    def test_emit_empty_stdin_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))

        hooks.emit(session_dir=tmp_path, agent="worker")

        assert not _marker_path(tmp_path, "worker").exists()

    def test_emit_unknown_event_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"hook_event_name": "SomeUnknownEvent"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        hooks.emit(session_dir=tmp_path, agent="worker")

        assert not _marker_path(tmp_path, "worker").exists()


class TestResolveAgentState:
    def test_state_dead_when_not_alive(self) -> None:
        marker = {"state": "running", "ts": 1000.0}
        result = hooks_resolve_agent_state(
            alive=False, marker=marker, last_activity_at=1000.0, now=1000.0
        )
        assert result == "dead"

    def test_state_from_marker_running_when_alive(self) -> None:
        marker = {"state": "running", "ts": 1000.0}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=1000.0, now=1000.0
        )
        assert result == "running"

    def test_state_from_marker_waiting_when_alive(self) -> None:
        marker = {"state": "waiting", "ts": 1000.0}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=1000.0, now=1000.0
        )
        assert result == "waiting"

    def test_state_fallback_running_recent_activity(self) -> None:
        result = hooks_resolve_agent_state(
            alive=True, marker=None, last_activity_at=1000.0, now=1010.0
        )
        assert result == "running"

    def test_state_fallback_idle_stale_activity(self) -> None:
        result = hooks_resolve_agent_state(
            alive=True, marker=None, last_activity_at=1000.0, now=1100.0
        )
        assert result == "idle"

    def test_state_fallback_idle_when_no_activity_known(self) -> None:
        result = hooks_resolve_agent_state(
            alive=True, marker=None, last_activity_at=None, now=1100.0
        )
        assert result == "idle"

    def test_idle_threshold_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIN_AGENT_TEAMS_IDLE_SECONDS", "5")
        result = hooks_resolve_agent_state(
            alive=True, marker=None, last_activity_at=1000.0, now=1006.0
        )
        assert result == "idle"


def hooks_resolve_agent_state(
    *, alive: bool, marker: dict | None, last_activity_at: float | None, now: float
) -> str:
    """Call ``server_simple._resolve_agent_state`` with an injected ``now``."""
    from claude_teams import server_simple

    return server_simple._resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_at, now=now
    )


class TestWriteClaudeSettings:
    def test_writes_hooks_block_and_returns_path(self, tmp_path: Path) -> None:
        path = hooks.write_claude_settings(tmp_path, "worker")

        assert path.exists()
        config = json.loads(path.read_text(encoding="utf-8"))
        assert "hooks" in config
        for event in (
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "SubagentStop",
        ):
            assert event in config["hooks"]

    def test_hook_command_references_emit_and_agent(self, tmp_path: Path) -> None:
        path = hooks.write_claude_settings(tmp_path, "worker")

        config = json.loads(path.read_text(encoding="utf-8"))
        stop_entry = config["hooks"]["Stop"]
        command = stop_entry[0]["hooks"][0]["command"]
        assert "claude_teams.hooks" in command
        assert "emit" in command
        assert "worker" in command
        assert str(tmp_path) in command


class TestCodexHookOverrides:
    def test_returns_c_override_args(self, tmp_path: Path) -> None:
        args = hooks.codex_hook_overrides(tmp_path, "worker")

        assert args
        assert args[0] == "-c"
        assert any("hooks" in arg for arg in args)

    def test_overrides_reference_agent_and_session_dir(self, tmp_path: Path) -> None:
        args = hooks.codex_hook_overrides(tmp_path, "worker")

        joined = " ".join(args)
        assert "worker" in joined
        assert str(tmp_path).replace("\\", "\\\\") in joined or str(tmp_path) in joined
