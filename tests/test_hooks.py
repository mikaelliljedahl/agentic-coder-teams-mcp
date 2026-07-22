"""Tests for the state-marker hook emitter (``claude_teams.hooks``)."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claude_teams import hooks


def _toml_decode(value: str) -> str:
    r"""Decode a TOML basic string body (``\\`` and ``\"`` escapes)."""
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _override_value(args: list[str], event: str) -> str:
    """Return the single ``-c`` value for ``hooks.<event>=`` in ``args``."""
    prefix = f"hooks.{event}="
    return next(
        v
        for k, v in zip(args[0::2], args[1::2], strict=True)
        if k == "-c" and v.startswith(prefix)
    )


def _field_value(override: str, field: str) -> str:
    """Extract and TOML-decode ``<field>="..."`` from an override value.

    Walks the TOML basic string honouring escape pairs (``\\\\``/``\\"``) so the
    scan stops at the first UNescaped ``"`` — correct for ``command`` (whose
    body contains escaped quotes) as well as ``commandWindows``.
    """
    marker = f'{field}="'
    i = override.index(marker) + len(marker)
    out: list[str] = []
    while i < len(override):
        ch = override[i]
        if ch == "\\":
            out.append(override[i : i + 2])
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return _toml_decode("".join(out))


def _single_quoted_field(override: str, field: str) -> str:
    """Extract ``<field>='...'`` from a single-quoted (Windows) override value.

    TOML literal strings do no escaping, so the value ends at the next single
    quote.
    """
    marker = f"{field}='"
    i = override.index(marker) + len(marker)
    return override[i : override.index("'", i)]


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

    def test_idle_threshold_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        emit_group = next(g for g in stop_entry if "emit" in g["hooks"][0]["command"])
        command = emit_group["hooks"][0]["command"]
        assert "claude_teams.hooks" in command
        assert "emit" in command
        assert "worker" in command
        assert tmp_path.as_posix() in command

    def test_write_claude_settings_stop_has_two_matcher_groups(
        self, tmp_path: Path
    ) -> None:
        path = hooks.write_claude_settings(tmp_path, "worker")

        config = json.loads(path.read_text(encoding="utf-8"))
        stop_groups = config["hooks"]["Stop"]
        # Stop is the ONE event that carries a second (wake) matcher group.
        assert len(stop_groups) == 2
        commands = [g["hooks"][0]["command"] for g in stop_groups]
        assert any("claude_teams.hooks" in c and "emit" in c for c in commands), (
            commands
        )
        assert any("claude_teams.lead_wake" in c for c in commands), commands
        # Every other lifecycle event keeps exactly the single emit group.
        for event in (
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "SubagentStop",
        ):
            assert len(config["hooks"][event]) == 1, event
            assert (
                "lead_wake" not in config["hooks"][event][0]["hooks"][0]["command"]
            ), event

    def test_wake_command_references_lead_wake_module_and_reader(
        self, tmp_path: Path
    ) -> None:
        path = hooks.write_claude_settings(tmp_path, "worker")

        config = json.loads(path.read_text(encoding="utf-8"))
        wake_group = next(
            g
            for g in config["hooks"]["Stop"]
            if "claude_teams.lead_wake" in g["hooks"][0]["command"]
        )
        command = wake_group["hooks"][0]["command"]
        assert "claude_teams.lead_wake" in command
        assert "--reader" in command
        assert "worker" in command
        assert tmp_path.as_posix() in command

    def test_wake_command_argv_shape(self) -> None:
        argv = hooks._wake_command(Path("C:/sessions/abc"), "worker")

        assert argv[0] == Path(sys.executable).as_posix()
        assert "-m" in argv
        assert "claude_teams.lead_wake" in argv
        assert "--session-dir" in argv
        assert argv[argv.index("--session-dir") + 1] == "C:/sessions/abc"
        assert "--reader" in argv
        assert argv[argv.index("--reader") + 1] == "worker"


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

    def test_stop_event_command_string_is_toml_safe(self, tmp_path: Path) -> None:
        # A backslash-containing path must not corrupt the TOML basic string:
        # the invariant is that every backslash is ESCAPED (``\\``), never left
        # raw. This is platform-independent — on Windows ``as_posix()`` renders
        # forward slashes (no backslash at all), while on POSIX a literal
        # backslash survives ``as_posix()`` and must be escaped by the TOML
        # renderer. The earlier "no backslash at all" assertion only held on
        # Windows and failed on Linux.
        windows_session_dir = Path("C:\\Users\\mlilj\\sessions\\abc")
        args = hooks.codex_hook_overrides(windows_session_dir, "worker")

        stop_value = next(
            v
            for k, v in zip(args[0::2], args[1::2], strict=True)
            if k == "-c" and v.startswith("hooks.Stop=")
        )
        # Extract the command="..." payload (still TOML-escaped).
        marker = 'command="'
        start = stop_value.index(marker) + len(marker)
        end = stop_value.rindex('"')
        escaped = stop_value[start:end]

        # No UNescaped backslash: after removing valid escape pairs (\\ and \")
        # nothing containing a lone backslash may remain.
        stripped = escaped.replace("\\\\", "").replace('\\"', "")
        assert "\\" not in stripped, f"unescaped backslash in TOML value: {escaped!r}"
        # Backslash-free substrings appear verbatim regardless of platform.
        assert "claude_teams.hooks" in escaped
        assert "emit" in escaped
        assert "worker" in escaped
        assert "sessions" in escaped

    def test_command_tokens_are_double_quoted_not_single_quoted(
        self, tmp_path: Path
    ) -> None:
        # Regression: Codex runs a ``command`` hook through the platform shell,
        # which on Windows is ``cmd.exe``. There a single quote is a literal
        # character, so a single-quoted command (``'python' '-m' ...``) is not
        # recognized and the hook exits 1 on every event. The command tokens
        # must be DOUBLE-quoted (escaped to ``\"`` inside the TOML basic string).
        args = hooks.codex_hook_overrides(tmp_path, "worker")
        post_value = next(
            v
            for k, v in zip(args[0::2], args[1::2], strict=True)
            if k == "-c" and v.startswith("hooks.PostToolUse=")
        )
        marker = 'command="'
        start = post_value.index(marker) + len(marker)
        end = post_value.rindex('"')
        escaped = post_value[start:end]
        # The interpreter/args must be wrapped in escaped double quotes...
        assert '\\"' in escaped, f"expected escaped double quotes, got: {escaped!r}"
        # ...and never in single quotes (which cmd.exe treats literally).
        # Decode the TOML basic string to the real shell command Codex runs.
        decoded = escaped.replace('\\"', '"').replace("\\\\", "\\")
        assert "'" not in decoded, f"single-quoted tokens break cmd.exe: {decoded!r}"
        assert decoded.startswith('"'), f"command must start double-quoted: {decoded!r}"
        assert "claude_teams.hooks" in decoded
        assert "worker" in decoded

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


_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "SubagentStop",
)


class TestCodexWindowsLauncher:
    def test_overrides_omit_commandWindows_without_launcher(  # noqa: N802
        self, tmp_path: Path
    ) -> None:
        # Linux path: no launcher -> command only, no commandWindows.
        args = hooks.codex_hook_overrides(tmp_path, "worker")
        assert "commandWindows" not in " ".join(args)

    def test_overrides_include_commandWindows_for_every_event_with_launcher(  # noqa: N802
        self, tmp_path: Path
    ) -> None:
        launcher = r"C:\s\codex-hook-worker.cmd"
        args = hooks.codex_hook_overrides(tmp_path, "worker", windows_launcher=launcher)
        for event in _EVENTS:
            override = _override_value(args, event)
            assert "commandWindows=" in override, event
            # A placeholder `command` is still present alongside it.
            assert "command=" in override

    def test_commandWindows_is_bare_launcher_path_single_quoted(  # noqa: N802
        self, tmp_path: Path
    ) -> None:
        launcher = r"C:\Users\me\.claude\agent-sessions\s1\codex-hook-worker.cmd"
        args = hooks.codex_hook_overrides(tmp_path, "worker", windows_launcher=launcher)
        override = _override_value(args, "PostToolUse")
        value = _single_quoted_field(override, "commandWindows")
        # Decoded value is exactly the bare launcher path (Codex adds its own
        # single-level quoting; extra quote pairs would break cmd /C).
        assert value == launcher

    def test_windows_override_has_no_double_quotes(self, tmp_path: Path) -> None:
        # The wt.exe safety invariant: a codex agent launched directly in a
        # Windows Terminal tab (`wt -- codex ...`) receives corrupt argv if the
        # hook override contains double quotes (wt's parser mangles them), so the
        # Windows form must be entirely single-quoted TOML literals.
        args = hooks.codex_hook_overrides(
            tmp_path, "worker", windows_launcher=r"C:\s\l.cmd"
        )
        for event in _EVENTS:
            override = _override_value(args, event)
            assert '"' not in override, f"{event}: {override!r}"
            # command is the inert single-quoted placeholder, not the emit argv.
            assert "command='true'" in override

    def test_write_codex_launcher_invokes_emit_with_session_and_agent(
        self, tmp_path: Path
    ) -> None:
        path = hooks.write_codex_launcher(tmp_path, "worker")
        assert path.exists()
        assert path.suffix == ".cmd"
        assert path.name == "codex-hook-worker.cmd"
        content = path.read_text(encoding="utf-8")
        assert "claude_teams.hooks" in content
        assert "emit" in content
        assert "--agent" in content
        assert "worker" in content
        assert str(tmp_path) in content  # native session-dir path referenced
        # Interpreter path is quoted (may contain spaces on Windows).
        assert f'"{Path(sys.executable)!s}"' in content

    @pytest.mark.skipif(
        os.name != "nt", reason="cmd.exe arg-escaping behaviour is Windows-only"
    )
    def test_commandWindows_survives_cmd_arg_escaping(  # noqa: N802
        self, tmp_path: Path
    ) -> None:
        # Ground-truth fidelity model of how Codex runs the hook on Windows:
        # `subprocess.run(["cmd", "/C", value])` escapes argv exactly like Rust
        # std's Command::arg (which matched the live failure). The bare-path
        # commandWindows value must run under it; the multi-quote `command` does
        # not.
        marker = tmp_path / "ran.txt"
        launcher = tmp_path / "probe.cmd"
        launcher.write_text(
            f'@echo off\r\n> "{marker}" echo ok\r\nexit /b 0\r\n', encoding="utf-8"
        )
        args = hooks.codex_hook_overrides(
            tmp_path, "worker", windows_launcher=str(launcher)
        )
        win_value = _single_quoted_field(
            _override_value(args, "PostToolUse"), "commandWindows"
        )
        comspec = os.environ.get("COMSPEC", "cmd.exe")

        r = subprocess.run(  # noqa: S603
            [comspec, "/C", win_value],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        assert marker.exists(), "launcher did not run under cmd /C arg-escaping"

    def test_commandWindows_bare_path_with_spaces_is_unquoted(  # noqa: N802
        self, tmp_path: Path
    ) -> None:
        # The fix relies on Codex/cmd adding at most one quote pair around the
        # single path token, so we must emit the path WITHOUT our own quotes
        # even when it contains spaces. A single-quoted TOML literal preserves
        # the spaces without introducing double quotes (which wt.exe mangles).
        launcher = r"C:\Users\me\ag ent\codex-hook-worker.cmd"
        args = hooks.codex_hook_overrides(tmp_path, "worker", windows_launcher=launcher)
        value = _single_quoted_field(
            _override_value(args, "PostToolUse"), "commandWindows"
        )
        assert value == launcher
        assert '"' not in value

    @pytest.mark.parametrize(
        "bad",
        ["../evil", "a/b", "a\\b", 'a"b', "a b", "", "x" * 65, "a;b", "a.cmd"],
    )
    def test_write_codex_launcher_rejects_unsafe_agent_name(
        self, tmp_path: Path, bad: str
    ) -> None:
        # Safe-name invariant must be enforced BEFORE any path/content is
        # written, so an unsafe name can't influence the on-disk launcher.
        with pytest.raises(ValueError, match="unsafe agent name"):
            hooks.write_codex_launcher(tmp_path, bad)
        assert list(tmp_path.glob("*.cmd")) == []  # nothing written

    @pytest.mark.skipif(
        os.name != "nt", reason="cmd.exe launcher execution is Windows-only"
    )
    def test_launcher_run_writes_state_marker_end_to_end(self, tmp_path: Path) -> None:
        # End-to-end: run the ACTUAL write_codex_launcher output via cmd with a
        # real event JSON on stdin and assert the state marker is written — the
        # full path Codex exercises on Windows.
        launcher = hooks.write_codex_launcher(tmp_path, "worker")
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        r = subprocess.run(  # noqa: S603
            [comspec, "/C", str(launcher)],
            input='{"hook_event_name":"Stop","session_id":"s1"}',
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        marker = tmp_path / "state-worker.json"
        assert marker.exists(), "launcher did not write the state marker"
        assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "waiting"
