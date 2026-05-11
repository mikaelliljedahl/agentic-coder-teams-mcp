"""Minimal CLI for win-agent-teams."""

import json
import os
import signal

import typer
from rich.console import Console
from rich.table import Table

from claude_teams.backends.registry import registry
from claude_teams.server_simple import mcp

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


if __name__ == "__main__":
    app()
