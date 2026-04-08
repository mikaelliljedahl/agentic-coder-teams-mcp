"""Teammate-tier MCP tools for runtime control and inspection."""

import asyncio
import time

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, messaging, tasks, teams
from claude_teams.async_utils import run_blocking
from claude_teams.backends import registry
from claude_teams.models import TeammateMember
from claude_teams.server_runtime import (
    _ANN_DESTRUCTIVE,
    _ANN_READ,
    _ANN_READ_WITH_SIDE_EFFECTS,
    _TAG_TEAMMATE,
    _require_authenticated_principal,
    _require_lead,
    _strip_ansi,
    logger,
)


async def _resolve_teammate(
    team_name: str, agent_name: str
) -> tuple[TeammateMember, str | None, str]:
    """Look up a teammate and resolve process handle and backend type."""
    try:
        config = await teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")

    member = None
    for config_member in config.members:
        if (
            isinstance(config_member, TeammateMember)
            and config_member.name == agent_name
        ):
            member = config_member
            break
    if member is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type
    if backend_type == "tmux":
        backend_type = "claude-code"

    return member, process_handle, backend_type


async def force_kill_teammate(
    team_name: str, agent_name: str, ctx: Context, capability: str = ""
) -> dict:
    """Forcibly kill a teammate and remove it from the team."""
    await _require_lead(ctx, team_name, capability)
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )

    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            await run_blocking(backend_obj.kill, process_handle)
        except KeyError:
            pass

    await teams.remove_member(team_name, agent_name)
    await capabilities.remove_agent_capability(team_name, agent_name)
    await tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} has been stopped."}


async def poll_inbox(
    team_name: str,
    agent_name: str,
    ctx: Context,
    timeout_ms: int = 30000,
    capability: str = "",
) -> list[dict]:
    """Poll an inbox for unread messages and mark returned messages as read."""
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead" and principal["name"] != agent_name:
        raise ToolError(
            f"Authenticated principal {principal['name']!r} cannot poll inbox {agent_name!r}."
        )
    msgs = await messaging.read_inbox(
        team_name, agent_name, unread_only=True, mark_as_read=True
    )
    if msgs:
        return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        msgs = await messaging.read_inbox(
            team_name, agent_name, unread_only=True, mark_as_read=True
        )
        if msgs:
            return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    return []


async def check_teammate(
    team_name: str,
    agent_name: str,
    ctx: Context,
    include_output: bool = False,
    output_lines: int = 20,
    include_messages: bool = True,
    max_messages: int = 5,
    capability: str = "",
) -> dict:
    """Check teammate status, lead-facing unread messages, and optional output."""
    await _require_lead(ctx, team_name, capability)
    member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )
    output_lines = max(1, min(output_lines, 120))
    max_messages = max(1, min(max_messages, 20))

    pending_from: list[dict] = []
    if include_messages:
        msgs = await messaging.read_inbox_filtered(
            team_name,
            "team-lead",
            sender_filter=agent_name,
            unread_only=True,
            mark_as_read=True,
            limit=max_messages,
        )
        pending_from = [
            msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs
        ]

    their_unread = await messaging.read_inbox(
        team_name, agent_name, unread_only=True, mark_as_read=False
    )
    their_unread_count = len(their_unread)

    alive = False
    detail = ""
    error: str | None = None
    output = ""

    if not process_handle:
        error = "no process handle recorded"
    else:
        try:
            backend_obj = registry.get(backend_type)
        except KeyError as exc:
            raise ToolError(str(exc))

        status = await run_blocking(backend_obj.health_check, process_handle)
        alive = status.alive
        detail = status.detail

        if include_output:
            try:
                captured = await run_blocking(
                    backend_obj.capture, process_handle, lines=output_lines
                )
                output = _strip_ansi(captured)
            except Exception as exc:  # pragma: no cover
                error = f"output capture failed: {exc}"

    result = {
        "name": member.name,
        "backend": backend_type,
        "alive": alive,
        "detail": detail,
        "pending_from": pending_from,
        "their_unread_count": their_unread_count,
        "error": error,
    }
    if include_output:
        result["output"] = output
    return result


async def process_shutdown_approved(
    team_name: str, agent_name: str, ctx: Context, capability: str = ""
) -> dict:
    """Remove a teammate after graceful shutdown approval."""
    await _require_lead(ctx, team_name, capability)
    if agent_name == "team-lead":
        raise ToolError("Cannot process shutdown for team-lead")
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )
    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            await run_blocking(backend_obj.kill, process_handle)
        except KeyError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to clean up %s process handle %s during shutdown approval: %s",
                backend_type,
                process_handle,
                exc,
            )
    await teams.remove_member(team_name, agent_name)
    await capabilities.remove_agent_capability(team_name, agent_name)
    await tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


async def health_check(
    team_name: str, agent_name: str, ctx: Context, capability: str = ""
) -> dict:
    """Check if a teammate's process is still running."""
    await _require_lead(ctx, team_name, capability)
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )

    if not process_handle:
        raise ToolError(f"No process handle for teammate {agent_name!r}")

    try:
        backend_obj = registry.get(backend_type)
    except KeyError as exc:
        raise ToolError(str(exc))

    status = await run_blocking(backend_obj.health_check, process_handle)
    return {
        "agent_name": agent_name,
        "alive": status.alive,
        "backend": backend_type,
        "detail": status.detail,
    }


def register_teammate_tools(mcp: FastMCP) -> None:
    """Register teammate-tier tools on the FastMCP app."""
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_DESTRUCTIVE)(force_kill_teammate)
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_READ_WITH_SIDE_EFFECTS)(poll_inbox)
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_READ_WITH_SIDE_EFFECTS)(
        check_teammate
    )
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_DESTRUCTIVE)(
        process_shutdown_approved
    )
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_READ)(health_check)
