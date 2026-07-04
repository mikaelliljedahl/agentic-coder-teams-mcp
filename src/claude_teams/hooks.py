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


def codex_hook_overrides(session_dir: Path, agent_name: str) -> list[str]:
    """Return ``-c`` config-override argv expressing the state hooks for Codex.

    Pure function: does not write any file. Mirrors the same event -> emit
    command wiring used for Claude Code, expressed as ``-c`` value overrides
    (one per event) for the Codex hooks table.

    Confirmed shape (spike on codex-cli 0.142.5, hooks feature stable)::

        hooks.<Event>=[{hooks=[{type="command",command="<CMD>"}]}]

    ``<CMD>`` is the shell-quoted emit command as a single TOML basic string.
    Basic strings treat backslash as an escape character, so any Windows path
    baked into the command (interpreter, session dir) must use forward
    slashes only — see :func:`_emit_command`.

    Tokens are DOUBLE-quoted (not single-quoted): Codex runs a ``command``
    hook through the platform shell, which on Windows is ``cmd.exe`` where a
    single quote is a literal character, not a quote. A single-quoted command
    therefore fails there with ``exit code 1`` on every hook event. Double
    quotes work in both ``cmd.exe`` and POSIX ``sh``; the ``"`` characters are
    escaped to ``\\"`` by :func:`_toml_basic_string` when nested in the TOML
    basic string.
    """
    session_dir = Path(session_dir)
    command = _emit_command(session_dir, agent_name)
    command_str = _shell_quote_command(command)
    args: list[str] = []
    for event in sorted(_RUNNING_EVENTS | _WAITING_EVENTS):
        args.extend(
            [
                "-c",
                f'hooks.{event}=[{{hooks=[{{type="command",command={_toml_basic_string(command_str)}}}]}}]',
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
