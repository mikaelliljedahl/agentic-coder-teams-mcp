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
import re
import sys
import time
import uuid
from pathlib import Path

_RUNNING_EVENTS: frozenset[str] = frozenset(
    {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse"}
)
_WAITING_EVENTS: frozenset[str] = frozenset({"Stop", "SubagentStop"})

_HOOK_MODULE = "claude_teams.hooks"

# Mirror of process_manager's safe-name invariant. Enforced here BEFORE an agent
# name is interpolated into an on-disk launcher path/content, rather than relying
# on the later (post-file-write) validation in the process manager.
_SAFE_AGENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _require_safe_agent(agent_name: str) -> str:
    """Return ``agent_name`` if it matches the safe-name invariant, else raise."""
    if not _SAFE_AGENT_RE.match(agent_name):
        msg = f"unsafe agent name: {agent_name!r}"
        raise ValueError(msg)
    return agent_name


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

    ``agent_name`` is validated against the safe-name invariant BEFORE it is
    interpolated into the launcher path or its batch content, so a name with
    path separators or quotes can never influence what/where we write (the
    process manager's later validation runs only after this file is created).
    """
    session_dir = Path(session_dir)
    _require_safe_agent(agent_name)
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

    ``windows_launcher`` (a :func:`write_codex_launcher` path) selects the
    Windows form. On Windows Codex runs ``commandWindows`` (a bare ``.cmd``
    path) for each event, never the POSIX ``command``. The Windows override is
    therefore built entirely from single-quoted TOML *literal* strings with **no
    double-quote characters anywhere** — see :func:`_codex_hook_overrides_windows`
    for why (Windows Terminal's command-line parser corrupts the double-quoted
    form). Pass ``None`` (the Linux case) for the POSIX ``command`` form.
    """
    session_dir = Path(session_dir)
    if windows_launcher is not None:
        return _codex_hook_overrides_windows(windows_launcher)
    command_str = _shell_quote_command(_emit_command(session_dir, agent_name))
    args: list[str] = []
    for event in sorted(_RUNNING_EVENTS | _WAITING_EVENTS):
        args.extend(
            [
                "-c",
                f'hooks.{event}=[{{hooks=[{{type="command",'
                f"command={_toml_basic_string(command_str)}}}]}}]",
            ]
        )
    return args


# ``command`` is required by Codex's hook schema but is never executed on
# Windows (``commandWindows`` runs instead), so it carries this inert, quote-free
# placeholder to keep the override entirely free of double-quote characters.
_WINDOWS_COMMAND_PLACEHOLDER = "true"


def _codex_hook_overrides_windows(windows_launcher: str) -> list[str]:
    r"""Return Windows ``-c`` hook overrides that survive wt.exe's arg parser.

    On Windows Codex runs ``commandWindows`` (the bare launcher path) for each
    event, so the POSIX ``command`` is unused. The earlier form rendered
    ``command`` as a double-quoted TOML basic string (escaped ``\\"``); when the
    agent is launched directly in a Windows Terminal tab (``wt … -- codex …``),
    wt.exe's own command-line parser mangles those embedded double quotes, Codex
    receives corrupt argv and exits instantly (no child ever appears). Building
    the whole override from single-quoted TOML *literal* strings — with **no
    double-quote characters at all** — passes through wt intact. This mirrors
    :meth:`CodexBackend._mcp_identity_args`, which is single-quoted for the same
    Windows-quoting reason and is known to survive wt.

    The launcher path already encodes the session dir and agent name, so those
    are not needed here. It must not contain a single quote —
    :func:`_toml_literal_string` rejects one rather than emit a corrupt override.
    """
    win = _toml_literal_string(windows_launcher)
    placeholder = _toml_literal_string(_WINDOWS_COMMAND_PLACEHOLDER)
    args: list[str] = []
    for event in sorted(_RUNNING_EVENTS | _WAITING_EVENTS):
        args.extend(
            [
                "-c",
                f"hooks.{event}=[{{hooks=[{{type='command',"
                f"command={placeholder},commandWindows={win}}}]}}]",
            ]
        )
    return args


def _toml_basic_string(value: str) -> str:
    """Render ``value`` as a TOML basic (double-quoted) string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_literal_string(value: str) -> str:
    """Render ``value`` as a TOML literal (single-quoted) string.

    Literal strings perform no escaping and cannot contain a single quote, so a
    value with one is rejected rather than silently corrupted. Backslashes and
    double quotes are preserved verbatim — ideal for a bare Windows path and for
    keeping the rendered override free of the double quotes wt.exe mangles.
    """
    if "'" in value:
        msg = f"cannot render as TOML literal (contains single quote): {value!r}"
        raise ValueError(msg)
    return f"'{value}'"


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
