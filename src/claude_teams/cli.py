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
        rows.append({
            "name": name,
            "binary": backend.binary_name,
            "default_model": backend.default_model(),
            "supported_models": backend.supported_models(),
        })
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
        table.add_row(row["name"], row["binary"], row["default_model"], ", ".join(row["supported_models"]))
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
) -> None:
    """Block until a file matching PATTERN in SESSION_DIR is created or changes.

    Pure-stdlib, poll-based (portable across Windows and Linux, no inotify
    dependency). Snapshots matching files' mtimes, then polls every ~0.5s
    until a matched file is newly created or its mtime advances, or until
    --timeout elapses.

    On change: prints each changed path (one per line) and exits 0. On
    timeout: prints nothing and exits 2, so a caller (e.g. a Codex
    coordinator looping a bounded foreground watch) can retry.

    Typical use: `win-agent-teams watch <session-dir> --timeout 60`, pointed
    at a `spawn_agent` result's `session_dir` (default pattern
    `state-*.json` catches any agent's state marker), then call
    `agent_status`/`check_agent` once this exits 0.
    """
    directory = Path(session_dir)
    deadline = time.monotonic() + timeout
    before = _snapshot_mtimes(directory, pattern)

    while True:
        after = _snapshot_mtimes(directory, pattern)
        changed = _changed_paths(before, after)
        if changed:
            for path in changed:
                console.print(path)
            raise typer.Exit(code=0)
        if time.monotonic() >= deadline:
            raise typer.Exit(code=2)
        time.sleep(_WATCH_POLL_SECONDS)


if __name__ == "__main__":
    app()
