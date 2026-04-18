"""Typer CLI for claude-teams.

Provides human-friendly commands that operate on the same file-based state
as the MCP server.  Both the CLI and MCP server can run concurrently thanks
to ``fcntl.flock()`` guards in the core modules.
"""

import asyncio
import json
import os
import signal
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from claude_teams import capabilities, messaging, tasks, teams
from claude_teams.async_utils import run_blocking
from claude_teams.backends.registry import registry
from claude_teams.errors import (
    BackendNotRegisteredError,
    InvalidNameError,
    NameTooLongError,
    TeamNotFoundValueError,
)
from claude_teams.models import TeammateMember
from claude_teams.server import mcp

_INBOX_SUMMARY_TRUNC_LEN = 80

app = typer.Typer(
    name="claude-teams",
    help="CLI for orchestrating Claude Code agent teams.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as JSON instead of a table."),
]
CapabilityFlag = Annotated[
    str,
    typer.Option(
        "--capability",
        help="Lead or agent capability. Falls back to CLAUDE_TEAMS_CAPABILITY.",
    ),
]


def _resolved_capability(capability: str) -> str:
    return capability or os.environ.get("CLAUDE_TEAMS_CAPABILITY", "")


def _run(awaitable):
    """Run an awaitable from the synchronous CLI surface.

    Args:
        awaitable: Awaitable value to execute.

    Returns:
        object: Awaitable result.

    """
    return asyncio.run(awaitable)


def _ensure_team_exists(team_name: str) -> None:
    try:
        exists = _run(teams.team_exists(team_name))
    except (InvalidNameError, NameTooLongError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not exists:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)


def _require_cli_principal(team_name: str, capability: str) -> capabilities.Principal:
    _ensure_team_exists(team_name)
    principal = _run(
        capabilities.resolve_principal(team_name, _resolved_capability(capability))
    )
    if principal is None:
        err_console.print(
            "[red]This command requires a valid team capability. "
            "Pass --capability or set CLAUDE_TEAMS_CAPABILITY.[/red]"
        )
        raise typer.Exit(code=1)
    return principal


def _require_cli_lead(team_name: str, capability: str) -> capabilities.Principal:
    principal = _require_cli_principal(team_name, capability)
    if principal["role"] != "lead":
        err_console.print("[red]This command requires the team-lead capability.[/red]")
        raise typer.Exit(code=1)
    return principal


def _require_cli_self_or_lead(
    team_name: str, agent_name: str, capability: str
) -> capabilities.Principal:
    principal = _require_cli_principal(team_name, capability)
    if principal["role"] == "lead" or principal["name"] == agent_name:
        return principal
    err_console.print(
        f"[red]Authenticated principal {principal['name']!r} "
        f"cannot access inbox {agent_name!r}.[/red]"
    )
    raise typer.Exit(code=1)


@app.command()
def serve() -> None:
    """Start the MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


@app.command()
def backends(output_json: JsonFlag = False) -> None:
    """List available spawner backends.

    Args:
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If no backends are available (exit code 1).

    """
    rows: list[dict[str, str | list[str]]] = []
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
        err_console.print("[yellow]No backends available.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Available Backends")
    table.add_column("Name", style="bold cyan")
    table.add_column("Binary")
    table.add_column("Default Model", style="green")
    table.add_column("Supported Models")
    for row in rows:
        table.add_row(
            str(row["name"]),
            str(row["binary"]),
            str(row["default_model"]),
            ", ".join(row["supported_models"]),
        )
    console.print(table)
    console.print(
        "\n[dim]Note: Supported models shown are a curated set. Actual"
        " availability depends on authentication state, account tier,"
        " and configured providers.[/dim]"
    )


@app.command()
def config(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Show the team configuration.

    Args:
        team_name (str): Name of the team.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    if output_json:
        console.print_json(json.dumps(cfg.model_dump(by_alias=True)))
        return

    console.print(f"[bold]Team:[/bold] {cfg.name}")
    console.print(f"[bold]Description:[/bold] {cfg.description or '(none)'}")
    console.print(f"[bold]Lead:[/bold] {cfg.lead_agent_id}")
    console.print(f"[bold]Members:[/bold] {len(cfg.members)}")

    table = Table(title="Members")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Model", style="green")
    table.add_column("Backend")
    table.add_column("Active")
    for member in cfg.members:
        if isinstance(member, TeammateMember):
            table.add_row(
                member.name,
                member.agent_type,
                member.model,
                member.backend_type,
                "[green]yes[/green]" if member.is_active else "[dim]no[/dim]",
            )
        else:
            table.add_row(member.name, member.agent_type, member.model, "-", "-")
    console.print(table)


@app.command()
def status(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Show team tasks and member summary.

    Args:
        team_name (str): Name of the team.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or task listing fails (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    try:
        task_list = _run(tasks.list_tasks(team_name))
    except (InvalidNameError, NameTooLongError, TeamNotFoundValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if output_json:
        payload = {
            "team": team_name,
            "member_count": len(cfg.members),
            "tasks": [
                task.model_dump(by_alias=True, exclude_none=True) for task in task_list
            ],
        }
        console.print_json(json.dumps(payload))
        return

    teammates = [member for member in cfg.members if isinstance(member, TeammateMember)]
    console.print(
        f"[bold]Team:[/bold] {team_name}  ({len(teammates)} teammate(s) + lead)"
    )

    if task_list:
        table = Table(title="Tasks")
        table.add_column("ID", style="dim")
        table.add_column("Status")
        table.add_column("Owner")
        table.add_column("Subject")
        for task in task_list:
            status_style = {
                "pending": "[yellow]pending[/yellow]",
                "in_progress": "[blue]in_progress[/blue]",
                "completed": "[green]completed[/green]",
            }.get(task.status, task.status)
            table.add_row(
                task.id, status_style, task.owner or "[dim]-[/dim]", task.subject
            )
        console.print(table)
    else:
        console.print("[dim]No tasks.[/dim]")


@app.command()
def inbox(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    capability: CapabilityFlag = "",
    unread_only: Annotated[
        bool, typer.Option("--unread", "-u", help="Show only unread messages.")
    ] = False,
    order: Annotated[
        str,
        typer.Option(
            "--order",
            help="Inbox ordering: oldest or newest.",
            case_sensitive=False,
        ),
    ] = "oldest",
    output_json: JsonFlag = False,
) -> None:
    """Read an agent's inbox messages.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose inbox to read.
        capability (str): Lead or matching agent capability override.
        unread_only (bool): Whether to show only unread messages.
        order (str): Inbox ordering, oldest or newest.
        output_json (bool): Whether to output as JSON instead of a table.

    """
    _require_cli_self_or_lead(team_name, agent_name, capability)
    normalized_order = order.lower()
    if normalized_order not in {"oldest", "newest"}:
        err_console.print("[red]order must be 'oldest' or 'newest'.[/red]")
        raise typer.Exit(code=1)
    inbox_order: Literal["oldest", "newest"] = (
        "newest" if normalized_order == "newest" else "oldest"
    )
    msgs = _run(
        messaging.read_inbox(
            team_name,
            agent_name,
            unread_only=unread_only,
            mark_as_read=False,
            order=inbox_order,
        )
    )

    if output_json:
        console.print_json(
            json.dumps(
                [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
            )
        )
        return

    if not msgs:
        console.print("[dim]Inbox empty.[/dim]")
        return

    table = Table(title=f"Inbox: {agent_name}")
    table.add_column("From", style="bold")
    table.add_column("Read")
    table.add_column("Time", style="dim")
    table.add_column("Summary / Text")
    for msg in msgs:
        read_mark = "[green]yes[/green]" if msg.read else "[red]no[/red]"
        display = msg.summary or (
            msg.text[:_INBOX_SUMMARY_TRUNC_LEN] + "..."
            if len(msg.text) > _INBOX_SUMMARY_TRUNC_LEN
            else msg.text
        )
        table.add_row(msg.from_, read_mark, msg.timestamp, display)
    console.print(table)


@app.command()
def health(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Check if a teammate's process is alive.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to check.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found, teammate not found, or backend
            unavailable (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(f"[red]Teammate {agent_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    try:
        backend_obj = registry.get(backend_type)
    except BackendNotRegisteredError:
        err_console.print(f"[red]Backend {backend_type!r} not available.[/red]")
        raise typer.Exit(code=1) from None

    health_status = _run(run_blocking(backend_obj.health_check, process_handle))

    result = {
        "agent_name": agent_name,
        "alive": health_status.alive,
        "backend": backend_type,
        "detail": health_status.detail,
    }

    if output_json:
        console.print_json(json.dumps(result))
        return

    if health_status.alive:
        console.print(
            f"[green]{agent_name}[/green] is [bold green]alive[/bold green] "
            f"({backend_type})"
        )
    else:
        console.print(
            f"[red]{agent_name}[/red] is [bold red]dead[/bold red] ({backend_type})"
        )
    if health_status.detail:
        console.print(f"  [dim]{health_status.detail}[/dim]")


@app.command()
def kill(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name to kill.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Force-kill a teammate and remove from team.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to kill.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or teammate not found (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(
            f"[red]Teammate {agent_name!r} not found in team {team_name!r}.[/red]"
        )
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            _run(run_blocking(backend_obj.kill, process_handle))
        except BackendNotRegisteredError:
            pass  # backend unavailable; process may already be dead

    _run(teams.remove_member(team_name, agent_name))
    _run(capabilities.remove_agent_capability(team_name, agent_name))
    _run(tasks.reset_owner_tasks(team_name, agent_name))

    result = {"success": True, "message": f"{agent_name} has been stopped."}

    if output_json:
        console.print_json(json.dumps(result))
        return

    console.print(
        f"[green]{agent_name} has been stopped and removed from {team_name}.[/green]"
    )


def _find_teammate(cfg: teams.TeamConfig, agent_name: str) -> TeammateMember | None:
    """Find a TeammateMember by name in a TeamConfig.

    Args:
        cfg (teams.TeamConfig): Team configuration to search.
        agent_name (str): Name of the agent to find.

    Returns:
        TeammateMember | None: The teammate if found, None otherwise.

    """
    for member in cfg.members:
        if isinstance(member, TeammateMember) and member.name == agent_name:
            return member
    return None


if __name__ == "__main__":
    app()
