"""Minimal CLI for win-agent-teams."""

import fnmatch
import json
import math
import os
import signal
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from claude_teams import server_simple as _ss
from claude_teams.backends.registry import registry
from claude_teams.hooks import _SAFE_AGENT_RE
from claude_teams.messaging import (
    load_inbox_cursors,
    read_inbox_by_sender,
    unread_sender_counts,
)
from claude_teams.server_simple import mcp

_WATCH_POLL_SECONDS = 0.5
_WATCH_DEFAULT_PATTERN = "state-*.json"

# Waiting markers written by these hook events are NOT coordinator-actionable:
# ``SubagentStop`` fires when one of an agent's OWN built-in Task subagents
# finishes, while the agent itself is still mid-task and will resume on its
# next tool call. Waking a coordinator for it is a false positive.
_NON_ACTIONABLE_WAITING_EVENTS: frozenset[str] = frozenset({"SubagentStop"})

_WATCH_SETTLE_DEFAULT_SECONDS = 15.0


def _settle_seconds_from_env() -> float:
    """Return the settle window from the environment, falling back to the default.

    A malformed override (non-numeric, ``NaN``, infinite, or negative) falls
    back to :data:`_WATCH_SETTLE_DEFAULT_SECONDS` rather than breaking the CLI or
    — in the ``NaN`` case — silently making ``now - since >= settle`` never true,
    which would suppress every genuine wake. ``0`` is a valid override (settle
    disabled).
    """
    raw = os.environ.get("WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS")
    if raw is None:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    if not math.isfinite(value) or value < 0:
        return _WATCH_SETTLE_DEFAULT_SECONDS
    return value


# Seconds an actionable ``waiting`` marker must persist before it wakes the
# coordinator. An agent that merely parks between operations (e.g. a
# backgrounded bash or a brief yield) flips waiting->running again within this
# window; requiring the marker to settle suppresses that churn. Env-overridable.
_WATCH_SETTLE_SECONDS = _settle_seconds_from_env()

app = typer.Typer(
    name="win-agent-teams",
    help="Spawn and communicate with Claude Code and Codex agents.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve() -> None:
    """Start the MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


@app.command()
def backends(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON."),
) -> None:
    """List available backends."""
    rows = []
    for name, backend in registry:
        rows.append(
            {
                "name": name,
                "binary": backend.binary_name,
                "default_model": backend.default_model(),
                "supported_models": backend.supported_models(),
            }
        )
    if output_json:
        console.print_json(json.dumps(rows))
        return
    if not rows:
        console.print("[yellow]No backends available.[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title="Available Backends")
    table.add_column("Name", style="bold cyan")
    table.add_column("Binary")
    table.add_column("Default Model", style="green")
    table.add_column("Supported Models")
    for row in rows:
        table.add_row(
            row["name"],
            row["binary"],
            row["default_model"],
            ", ".join(row["supported_models"]),
        )
    console.print(table)


def _snapshot_mtimes(session_dir: Path, pattern: str) -> dict[str, tuple[int, int]]:
    """Return ``{path: (mtime_ns, size)}`` for files matching ``pattern``.

    Using ``(mtime_ns, size)`` instead of a bare ``st_mtime`` float catches
    same-second (or same-tick) atomic-replace/rewrites that preserve the
    exposed mtime but change file size, not just mtime increases.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    if not session_dir.is_dir():
        return snapshot
    for entry in session_dir.iterdir():
        if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
            try:
                stat = entry.stat()
                snapshot[str(entry)] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
    return snapshot


def _changed_paths(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]
) -> list[str]:
    """Return paths that are new in ``after`` or whose ``(mtime_ns, size)`` differs."""
    changed = []
    for path, identity in after.items():
        prior = before.get(path)
        if prior is None or identity != prior:
            changed.append(path)
    return changed


def _path_identity(path: Path) -> tuple[int, int] | None:
    """Return one file's ``(mtime_ns, size)`` identity, or ``None`` if absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _waiting_agent(path: Path) -> str | None:
    """Return the agent name for a coordinator-actionable waiting marker.

    Returns ``None`` when ``path`` is not such a marker. A marker is actionable
    only when its state is ``waiting`` AND the hook event
    that produced it is not in :data:`_NON_ACTIONABLE_WAITING_EVENTS`. This
    filters out ``SubagentStop`` churn (an agent's own Task subagent finishing)
    the same way the caller already ignores ``running`` transitions. Markers with
    no recorded ``event`` are treated as actionable for backward compatibility.
    """
    if not (path.name.startswith("state-") and path.suffix == ".json"):
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or marker.get("state") != "waiting":
        return None
    event = marker.get("event")
    # ``event`` may be any JSON value; guard the membership test so a non-string
    # (unhashable list/dict) does not raise and regress marker tolerance. A
    # missing or non-string event is treated as actionable.
    if isinstance(event, str) and event in _NON_ACTIONABLE_WAITING_EVENTS:
        return None
    return path.stem.removeprefix("state-")


def _emit_wake(payload: dict) -> None:
    """Print one unwrapped JSONL wake record for a harness to inspect."""
    typer.echo(json.dumps(payload, separators=(",", ":")))


def _require_safe_reader(reader: str) -> str:
    """Return ``reader`` when it is safe to interpolate into inbox filenames.

    Reuses the repo's shared safe-agent-name invariant
    (``claude_teams.hooks._SAFE_AGENT_RE``) so a reader value can never carry
    path separators or control characters into ``inbox-<reader>.jsonl`` /
    ``inbox-<reader>.pos.json`` paths.
    """
    if not _SAFE_AGENT_RE.match(reader):
        msg = f"unsafe reader name: {reader!r}"
        raise ValueError(msg)
    return reader


@app.command()
def watch(
    session_dir: str = typer.Argument(..., help="Directory to watch."),
    timeout: float = typer.Option(
        60.0, "--timeout", "-t", help="Seconds to wait before giving up."
    ),
    pattern: str = typer.Option(
        _WATCH_DEFAULT_PATTERN,
        "--pattern",
        "-p",
        help="Glob pattern (relative to session_dir) to watch, e.g. state-*.json.",
    ),
    watch_inbox: bool = typer.Option(
        True,
        "--inbox/--no-inbox",
        help="Wake for unread messages to this orchestrator (enabled by default).",
    ),
    reader: str | None = typer.Option(
        None,
        "--reader",
        help=(
            "Inbox reader identity to watch. Overrides AGENT_NAME/team-lead; "
            "omit for the current env-based behavior."
        ),
    ),
) -> None:
    """Block until an agent is waiting, an inbox is unread, or output changes.

    State-marker changes are semantic: ``running`` lifecycle transitions and
    ``SubagentStop`` (a worker's own Task subagent finishing) are ignored, and a
    marker whose state is ``waiting`` exits 0 only after it *persists* as waiting
    for a short settle window (``WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS``, default
    15s) — a marker that resumes ``running`` inside the window is suppressed as
    a brief park. Other files selected by PATTERN wake on any creation/change.
    Unless ``--no-inbox`` is passed, unread messages for ``AGENT_NAME`` (or the
    root ``team-lead`` identity, or an explicit ``--reader NAME``) also exit 0
    without consuming them. When several
    signals are ready in one poll the priority is message > output > waiting.

    Success prints one JSON object with ``reason`` equal to ``message``,
    ``waiting``, or ``output``. Timeout prints nothing and exits 2; re-check
    status after exit 2 because a waiting transition may precede the initial
    marker snapshot, or a genuine waiting edge may still be inside its settle
    window at the deadline.
    """
    directory = Path(session_dir)
    deadline = time.monotonic() + timeout
    before = _snapshot_mtimes(directory, pattern)

    if reader is not None:
        try:
            _require_safe_reader(reader)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        reader_name = reader
    else:
        reader_name = os.environ.get("AGENT_NAME", "").strip() or "team-lead"
    inbox_path = directory / f"inbox-{reader_name}.jsonl"
    cursor_path = directory / f"inbox-{reader_name}.pos.json"
    inbox_before = _path_identity(inbox_path)

    if watch_inbox:
        unread = unread_sender_counts(inbox_path, cursor_path)
        if unread:
            _emit_wake(
                {
                    "reason": "message",
                    "from": list(unread),
                    "path": str(inbox_path),
                }
            )
            raise typer.Exit(code=0)

    # A waiting marker only wakes the coordinator once it has stayed waiting for
    # _WATCH_SETTLE_SECONDS. Every actionable candidate is tracked by its marker
    # path — a coordinator watches ALL agents' markers, so overlapping waits must
    # each settle independently; a single slot would drop one when another arrives
    # or resumes. Value is the first-seen monotonic time.
    pending_waits: dict[str, float] = {}

    while True:
        now = time.monotonic()
        after = _snapshot_mtimes(directory, pattern)
        changed = _changed_paths(before, after)
        inbox_after = _path_identity(inbox_path)

        # Explicit communication wins when message and state/output edges land
        # in the same polling interval.
        if watch_inbox and inbox_after != inbox_before:
            unread = unread_sender_counts(inbox_path, cursor_path)
            if unread:
                _emit_wake(
                    {
                        "reason": "message",
                        "from": list(unread),
                        "path": str(inbox_path),
                    }
                )
                raise typer.Exit(code=0)
        inbox_before = inbox_after

        waiting: list[tuple[str, str]] = []
        outputs: list[str] = []
        for changed_path in changed:
            path = Path(changed_path)
            if path.name.startswith("state-") and path.suffix == ".json":
                agent = _waiting_agent(path)
                if agent is not None:
                    waiting.append((agent, changed_path))
            else:
                outputs.append(changed_path)

        # Advance after every edge, including ignored running/corrupt markers,
        # so one non-ready write cannot be rediscovered forever.
        before = after

        # Register every actionable waiting edge seen this tick, keeping the
        # earliest first-seen time while a marker stays a candidate.
        for _agent, wpath in waiting:
            pending_waits.setdefault(wpath, now)

        # Emit outputs BEFORE any settled wait. A wait settles over several
        # polls, so an output edge can land in the same poll a wait matures;
        # since `before = after` has already consumed that output edge, waking
        # on the wait first would drop the output for good. Priority is
        # message > output > waiting (message is handled at the top of the loop).
        if outputs:
            _emit_wake({"reason": "output", "path": outputs[0]})
            raise typer.Exit(code=0)

        # Settle each candidate against its CURRENT state: drop any that flipped
        # back to running (or to a non-actionable SubagentStop) within the
        # window, and wake on the first that has stayed waiting long enough.
        # Iterate in insertion order so the earliest-seen settled wait wins.
        for wpath in list(pending_waits):
            current = _waiting_agent(Path(wpath))
            if current is None:
                del pending_waits[wpath]
            elif now - pending_waits[wpath] >= _WATCH_SETTLE_SECONDS:
                _emit_wake({"reason": "waiting", "agent": current, "path": wpath})
                raise typer.Exit(code=0)
        if time.monotonic() >= deadline:
            raise typer.Exit(code=2)
        time.sleep(_WATCH_POLL_SECONDS)


@app.command(name="session-dir")
def session_dir() -> None:
    """Print the current workspace session as ``id<TAB>dir<TAB>identity``.

    Discovery-only: resolves the active/recoverable session WITHOUT creating a
    session directory. On success exits 0 with exactly one tab-separated line on
    stdout and nothing on stderr. When no session exists it exits 3 with empty
    stdout. An internal error exits 1 with a message on stderr only.
    """
    try:
        session_id = _ss._active_session_id(create=False)
        line = (
            f"{session_id}\t{_ss._session_dir(session_id)}\t{_ss.IDENTITY}"
            if session_id
            else None
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if line is None:
        raise typer.Exit(code=3)
    typer.echo(line)


@app.command(name="inbox-status")
def inbox_status(
    session_dir: str = typer.Argument(..., help="Session directory to probe."),
    reader: str = typer.Option(
        "team-lead", "--reader", help="Inbox reader identity to probe."
    ),
) -> None:
    """Emit a non-consuming inbox generation snapshot as one JSON object.

    Prints ``{"schema":"inbox-status/1","reader":...,"senders":{<from>:
    {"total":N,"cursor":M,"unread":K}}}`` where ``unread = total -
    min(cursor, total)``; an empty inbox yields ``"senders":{}``. This never
    writes a cursor. A bad/nonexistent/outside-base ``session_dir`` exits 4 with
    a message on stderr and no stdout; an internal error exits 1 with stderr
    only.
    """
    try:
        _require_safe_reader(reader)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    directory = Path(session_dir)
    base = _ss._SESSION_BASE.resolve()
    if not directory.is_dir() or directory.resolve().parent != base:
        typer.echo(f"session_dir not under session base: {session_dir!r}", err=True)
        raise typer.Exit(code=4)

    try:
        inbox_path = directory / f"inbox-{reader}.jsonl"
        cursor_path = directory / f"inbox-{reader}.pos.json"
        by_sender = read_inbox_by_sender(inbox_path)
        cursors = load_inbox_cursors(cursor_path)
        senders: dict[str, dict[str, int]] = {}
        for sender, messages in by_sender.items():
            total = len(messages)
            cursor = cursors.get(sender, 0)
            unread = total - min(cursor, total)
            senders[sender] = {"total": total, "cursor": cursor, "unread": unread}
        payload = {"schema": "inbox-status/1", "reader": reader, "senders": senders}
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    app()
