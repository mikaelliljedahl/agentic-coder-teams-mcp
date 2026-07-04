"""State-marker hook emitter and per-agent hook wiring for claude-code/codex.

A tiny per-agent JSON marker (``state-{agent}.json``) is written by a small
CLI entrypoint invoked as a Claude Code / Codex hook command. The marker
records the coarse-grained agent state derived from the hook event name so
``check_agent``/``list_agents``/``agent_status`` can report state cheaply
without scanning the agent's full transcript.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

_RUNNING_EVENTS: frozenset[str] = frozenset(
    {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse"}
)
_WAITING_EVENTS: frozenset[str] = frozenset({"Stop", "SubagentStop"})

_HOOK_MODULE = "claude_teams.hooks"


def _marker_file(session_dir: Path, agent: str) -> Path:
    return session_dir / f"state-{agent}.json"


def _map_event_to_state(event_name: str) -> str | None:
    """Map a hook event name to a marker state, or ``None`` if unrecognized."""
    if event_name in _RUNNING_EVENTS:
        return "running"
    if event_name in _WAITING_EVENTS:
        return "waiting"
    return None


def _write_marker_atomic(path: Path, marker: dict) -> None:
    """Atomically persist ``marker`` via a uniquely named temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(marker), encoding="utf-8")
    tmp.replace(path)


def emit(session_dir: Path, agent: str) -> None:
    """Read a hook JSON payload from stdin and write the agent's state marker.

    Tolerates corrupt/empty stdin and unrecognized event names without
    raising: in every such case, no marker is written (or the prior marker is
    left untouched).
    """
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return
    if not raw or not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return
    state = _map_event_to_state(event_name)
    if state is None:
        return

    marker = {"state": state, "event": event_name, "ts": time.time()}
    _write_marker_atomic(_marker_file(Path(session_dir), agent), marker)


def _emit_command(session_dir: Path, agent: str) -> list[str]:
    """Return the argv for the hook command invoking ``emit`` for ``agent``.

    Paths are rendered with forward slashes (``Path.as_posix()``) rather than
    the platform-native separator. This argv is also embedded verbatim into a
    Codex ``-c`` TOML basic string (see :func:`codex_hook_overrides`), where
    backslash is an escape character; a raw Windows backslash path would
    corrupt the TOML value. Forward-slash paths parse correctly on Windows
    too, so this is safe for both the Claude Code settings-file JSON use and
    the Codex TOML use.
    """
    return [
        Path(sys.executable).as_posix(),
        "-m",
        _HOOK_MODULE,
        "emit",
        "--session-dir",
        Path(session_dir).as_posix(),
        "--agent",
        agent,
    ]


def _hook_matcher(session_dir: Path, agent: str) -> dict:
    return {
        "hooks": [
            {
                "type": "command",
                "command": _shell_quote_command(_emit_command(session_dir, agent)),
            }
        ]
    }


def _shell_quote_command(argv: list[str]) -> str:
    """Render ``argv`` as a single shell command string.

    Claude Code settings hooks take a shell command string, not an argv list.
    Each token is double-quoted so paths containing spaces (common on
    Windows) survive shell parsing.
    """
    return " ".join(f'"{part}"' for part in argv)


def write_claude_settings(session_dir: Path, agent_name: str) -> Path:
    """Write a per-agent Claude Code settings file wiring the state hooks.

    Returns the path to the written settings JSON file, suitable for passing
    to ``claude --settings <path>``.
    """
    session_dir = Path(session_dir)
    events = _RUNNING_EVENTS | _WAITING_EVENTS
    config = {
        "hooks": {event: [_hook_matcher(session_dir, agent_name)] for event in events}
    }
    path = session_dir / f"hooks-{agent_name}.settings.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def write_codex_launcher(session_dir: Path, agent_name: str) -> Path:
    """Write a Windows ``.cmd`` launcher that runs the state-marker emit.

    Why a launcher (see ``docs/tickets/codex-hook-windows-cmd``): Codex runs a
    ``command`` hook by handing the string to ``cmd /C`` via argv-escaping
    (Rust ``Command::arg``). That corrupts our multi-token double-quoted
    command — the manually added inner quotes get backslash-escaped and
    ``cmd.exe`` rejects them (``'"…"' is not recognized`` → exit 1, marker
    never written). A launcher referenced as a SINGLE bare path in the
    ``commandWindows`` override sidesteps this: Codex quotes the lone path at
    most once (only if it contains spaces), which ``cmd /C`` handles fine. The
    launcher, being a batch file, quotes the interpreter/args with normal cmd
    rules; and stdin (the event JSON) is inherited cmd -> launcher -> python, so
    ``emit`` still reads it. Native backslash paths are used (cmd-friendly).
    """
    session_dir = Path(session_dir)
    python = str(Path(sys.executable))
    sdir = str(session_dir)
    lines = [
        "@echo off",
        f'"{python}" -m {_HOOK_MODULE} emit '
        f'--session-dir "{sdir}" --agent "{agent_name}"',
        "exit /b %errorlevel%",
    ]
    path = session_dir / f"codex-hook-{agent_name}.cmd"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return path


def codex_hook_overrides(
    session_dir: Path, agent_name: str, windows_launcher: str | None = None
) -> list[str]:
    """Return ``-c`` config-override argv expressing the state hooks for Codex.

    Mirrors the same event -> emit command wiring used for Claude Code,
    expressed as ``-c`` value overrides (one per event) for the Codex hooks
    table.

    Confirmed shape (codex-cli 0.142.5)::

        hooks.<Event>=[{hooks=[{type="command",command="<CMD>"}]}]

    ``command`` (``<CMD>``) is the POSIX form: the double-quoted emit argv as a
    TOML basic string. It works under ``sh -c`` on Unix. Backslash is a TOML
    escape char, so the paths baked into it use forward slashes only — see
    :func:`_emit_command`.

    ``windows_launcher`` (a :func:`write_codex_launcher` path) adds a
    ``commandWindows=<path>`` override to each event. On Windows Codex runs
    ``commandWindows`` instead of ``command``, via ``cmd /C``. The value is the
    BARE launcher path (no manually added quotes): Codex's argv-escaping quotes
    it at most once, which ``cmd.exe`` handles — a multi-token double-quoted
    string does not (its inner quotes get escaped and rejected). Pass ``None``
    (the Linux case) to omit ``commandWindows`` and rely on ``command``.
    """
    session_dir = Path(session_dir)
    command = _emit_command(session_dir, agent_name)
    command_str = _shell_quote_command(command)
    win_fragment = ""
    if windows_launcher:
        win_fragment = f",commandWindows={_toml_basic_string(windows_launcher)}"
    args: list[str] = []
    for event in sorted(_RUNNING_EVENTS | _WAITING_EVENTS):
        args.extend(
            [
                "-c",
                f'hooks.{event}=[{{hooks=[{{type="command",'
                f"command={_toml_basic_string(command_str)}{win_fragment}}}]}}]",
            ]
        )
    return args


def _toml_basic_string(value: str) -> str:
    """Render ``value`` as a TOML basic (double-quoted) string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m claude_teams.hooks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--session-dir", required=True)
    emit_parser.add_argument("--agent", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: ``emit --session-dir <dir> --agent <name>``.

    Reads the hook JSON payload from stdin.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "emit":
        emit(Path(args.session_dir), args.agent)


if __name__ == "__main__":
    main()
