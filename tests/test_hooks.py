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

    def test_empty_marker_dict_falls_back_to_heuristic(self) -> None:
        result = hooks_resolve_agent_state(
            alive=True, marker={}, last_activity_at=1000.0, now=1010.0
        )
        assert result == "running"

    def test_invalid_marker_state_falls_back_to_heuristic(self) -> None:
        marker = {"state": "paused", "ts": 1000.0}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=1000.0, now=1010.0
        )
        assert result == "running"

    def test_invalid_marker_state_falls_back_to_idle_when_stale(self) -> None:
        marker = {"state": "paused", "ts": 1000.0}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=1000.0, now=1100.0
        )
        assert result == "idle"

    def test_missing_ts_in_marker_still_uses_valid_state(self) -> None:
        marker = {"state": "waiting"}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=None, now=1100.0
        )
        assert result == "waiting"

    def test_non_numeric_ts_does_not_affect_valid_state(self) -> None:
        marker = {"state": "running", "ts": "not-a-number"}
        result = hooks_resolve_agent_state(
            alive=True, marker=marker, last_activity_at=1000.0, now=1010.0
        )
        assert result == "running"


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
        assert tmp_path.as_posix() in command


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
        assert tmp_path.as_posix() in joined

    def test_emits_one_c_arg_per_lifecycle_event_in_confirmed_shape(
        self, tmp_path: Path
    ) -> None:
        args = hooks.codex_hook_overrides(tmp_path, "worker")

        # args is a flat ["-c", value, "-c", value, ...] list, one pair per event.
        assert len(args) % 2 == 0
        values = args[1::2]
        assert all(a == "-c" for a in args[0::2])

        events = {
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "SubagentStop",
        }
        seen_events = set()
        for value in values:
            key, _, rest = value.partition("=")
            assert key.startswith("hooks.")
            event = key[len("hooks.") :]
            assert event in events
            seen_events.add(event)
            assert rest.startswith('[{hooks=[{type="command",command="')
            assert rest.endswith('"}]}]')
        assert seen_events == events

    def test_stop_event_command_string_is_toml_safe_no_raw_backslash(
        self, tmp_path: Path
    ) -> None:
        windows_session_dir = Path("C:\\Users\\mlilj\\sessions\\abc")
        args = hooks.codex_hook_overrides(windows_session_dir, "worker")

        stop_value = next(
            v
            for k, v in zip(args[0::2], args[1::2], strict=True)
            if k == "-c" and v.startswith("hooks.Stop=")
        )
        # Extract the command="..." payload.
        marker = 'command="'
        start = stop_value.index(marker) + len(marker)
        end = stop_value.rindex('"')
        command_str = stop_value[start:end]

        assert "\\" not in command_str
        assert "claude_teams.hooks" in command_str
        assert "emit" in command_str
        assert "worker" in command_str
        assert "C:/Users/mlilj/sessions/abc" in command_str

    def test_command_reads_event_from_stdin_and_takes_session_dir_and_agent_args(
        self,
    ) -> None:
        argv = hooks._emit_command(Path("C:/sessions/abc"), "worker")

        assert argv[0] == Path(sys.executable).as_posix()
        assert "-m" in argv
        assert "claude_teams.hooks" in argv
        assert "emit" in argv
        assert "--session-dir" in argv
        assert argv[argv.index("--session-dir") + 1] == "C:/sessions/abc"
        assert "--agent" in argv
        assert argv[argv.index("--agent") + 1] == "worker"
