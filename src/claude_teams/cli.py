"""Minimal CLI for win-agent-teams."""

import fnmatch
import json
import os
import signal
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from claude_teams.backends.registry import registry
from claude_teams.messaging import unread_sender_counts
from claude_teams.server_simple import mcp

_WATCH_POLL_SECONDS = 0.5
_WATCH_DEFAULT_PATTERN = "state-*.json"

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
    """Return the agent name when ``path`` is a valid waiting-state marker."""
    if not (path.name.startswith("state-") and path.suffix == ".json"):
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or marker.get("state") != "waiting":
        return None
    return path.stem.removeprefix("state-")


def _emit_wake(payload: dict) -> None:
    """Print one unwrapped JSONL wake record for a harness to inspect."""
    typer.echo(json.dumps(payload, separators=(",", ":")))


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
) -> None:
    """Block until an agent is waiting, an inbox is unread, or output changes.

    State-marker changes are semantic: ``running`` lifecycle transitions are
    ignored, while a changed marker whose state is ``waiting`` exits 0. Other
    files selected by PATTERN wake on any creation/change. Unless
    ``--no-inbox`` is passed, unread messages for ``AGENT_NAME`` (or the root
    ``team-lead`` identity) also exit 0 without consuming them.

    Success prints one JSON object with ``reason`` equal to ``message``,
    ``waiting``, or ``output``. Timeout prints nothing and exits 2; re-check
    status after exit 2 because a waiting transition may precede the initial
    marker snapshot.
    """
    directory = Path(session_dir)
    deadline = time.monotonic() + timeout
    before = _snapshot_mtimes(directory, pattern)

    reader = os.environ.get("AGENT_NAME", "").strip() or "team-lead"
    inbox_path = directory / f"inbox-{reader}.jsonl"
    cursor_path = directory / f"inbox-{reader}.pos.json"
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

    while True:
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

        if waiting:
            agent, path = waiting[0]
            _emit_wake({"reason": "waiting", "agent": agent, "path": path})
            raise typer.Exit(code=0)
        if outputs:
            _emit_wake({"reason": "output", "path": outputs[0]})
            raise typer.Exit(code=0)
        if time.monotonic() >= deadline:
            raise typer.Exit(code=2)
        time.sleep(_WATCH_POLL_SECONDS)


if __name__ == "__main__":
    app()
