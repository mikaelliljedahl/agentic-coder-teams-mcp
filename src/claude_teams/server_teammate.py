"""Teammate-tier MCP tools for runtime control and inspection."""

import asyncio
import logging
import time

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, messaging, tasks, teams
from claude_teams.async_utils import run_blocking
from claude_teams.backends import registry
from claude_teams.errors import (
    BackendNotRegisteredError,
    InboxAccessDeniedError,
    NoProcessHandleError,
    ShutdownLeadError,
    TeammateNotFoundToolError,
    TeamNotFoundToolError,
)
from claude_teams.models import TeammateMember
from claude_teams.server_runtime import (
    _ANN_DESTRUCTIVE,
    _ANN_READ,
    _ANN_READ_WITH_SIDE_EFFECTS,
    _TAG_TEAMMATE,
    _require_authenticated_principal,
    _require_lead,
    _strip_ansi,
)
from claude_teams.server_schema import (
    AgentName,
    Capability,
    MaxMessages,
    OutputLines,
    TeamName,
    TimeoutMs,
)

logger = logging.getLogger(__name__)


async def _resolve_teammate(
    team_name: str, agent_name: str
) -> tuple[TeammateMember, str | None, str]:
    """Look up a teammate and resolve process handle and backend type.

    Legacy configs persisted ``backend_type='tmux'`` before the backend
    registry split tmux-based CLIs by tool. Rewrite those to ``claude-code``
    so they dispatch to the modern registry entry; new writes always record
    the concrete backend id. Emit a debug log so operators can tell when a
    stale config is still being migrated on the fly.
    """
    try:
        config = await teams.read_config(team_name)
    except FileNotFoundError:
        raise TeamNotFoundToolError(team_name) from None

    member = None
    for config_member in config.members:
        if (
            isinstance(config_member, TeammateMember)
            and config_member.name == agent_name
        ):
            member = config_member
            break
    if member is None:
        raise TeammateNotFoundToolError(agent_name, team_name)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type
    if backend_type == "tmux":
        logger.debug(
            "Rewriting legacy backend_type='tmux' to 'claude-code' for %s/%s",
            team_name,
            agent_name,
        )
        backend_type = "claude-code"

    return member, process_handle, backend_type


async def force_kill_teammate(
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    capability: Capability = "",
) -> dict[str, object]:
    """Forcibly kill a teammate and remove it from the team."""
    await _require_lead(ctx, team_name, capability)
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )

    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            await run_blocking(backend_obj.kill, process_handle)
        except BackendNotRegisteredError:
            pass

    await teams.remove_member(team_name, agent_name)
    await capabilities.remove_agent_capability(team_name, agent_name)
    await tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} has been stopped."}


async def poll_inbox(
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    timeout_ms: TimeoutMs = 30000,
    capability: Capability = "",
) -> list[dict[str, object]]:
    """Poll an inbox for unread messages and mark returned messages as read."""
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead" and principal["name"] != agent_name:
        raise InboxAccessDeniedError("poll", principal["name"], agent_name)
    msgs = await messaging.read_inbox(
        team_name, agent_name, unread_only=True, mark_as_read=True
    )
    if msgs:
        return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        sleep_seconds = min(0.5, max(0.0, deadline - time.monotonic()))
        if sleep_seconds == 0.0:
            break
        await asyncio.sleep(sleep_seconds)
        msgs = await messaging.read_inbox(
            team_name, agent_name, unread_only=True, mark_as_read=True
        )
        if msgs:
            return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    return []


async def check_teammate(
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    include_output: bool = False,
    output_lines: OutputLines = 20,
    include_messages: bool = True,
    max_messages: MaxMessages = 5,
    capability: Capability = "",
) -> dict[str, object]:
    """Check teammate status, lead-facing unread messages, and optional output."""
    await _require_lead(ctx, team_name, capability)
    member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )

    pending_from: list[dict[str, object]] = []
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
        except BackendNotRegisteredError as exc:
            raise ToolError(str(exc)) from exc

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

    result: dict[str, object] = {
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
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    capability: Capability = "",
) -> dict[str, object]:
    """Remove a teammate after graceful shutdown approval."""
    await _require_lead(ctx, team_name, capability)
    if agent_name == "team-lead":
        raise ShutdownLeadError()
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )
    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            await run_blocking(backend_obj.kill, process_handle)
        except BackendNotRegisteredError:
            pass
        except Exception as exc:  # pragma: no cover
            await ctx.warning(
                f"Failed to clean up {backend_type} process handle "
                f"{process_handle!r} during shutdown approval of {agent_name!r}: "
                f"{exc}",
                extra={
                    "team": team_name,
                    "agent": agent_name,
                    "backend": backend_type,
                    "process_handle": process_handle,
                },
            )
    await teams.remove_member(team_name, agent_name)
    await capabilities.remove_agent_capability(team_name, agent_name)
    await tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


async def health_check(
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    capability: Capability = "",
) -> dict[str, object]:
    """Check if a teammate's process is still running."""
    await _require_lead(ctx, team_name, capability)
    _member, process_handle, backend_type = await _resolve_teammate(
        team_name, agent_name
    )

    if not process_handle:
        raise NoProcessHandleError(agent_name)

    try:
        backend_obj = registry.get(backend_type)
    except BackendNotRegisteredError as exc:
        raise ToolError(str(exc)) from exc

    status = await run_blocking(backend_obj.health_check, process_handle)
    return {
        "agent_name": agent_name,
        "alive": status.alive,
        "backend": backend_type,
        "detail": status.detail,
    }


def register_teammate_tools(mcp: FastMCP) -> None:
    """Register teammate-tier tools on the FastMCP app.

    ``poll_inbox`` carries a timeout that comfortably exceeds the maximum
    ``timeout_ms`` argument so the tool can honour the caller's deadline
    without the transport cutting it short.
    """
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_DESTRUCTIVE)(force_kill_teammate)
    mcp.tool(
        tags={_TAG_TEAMMATE},
        annotations=_ANN_READ_WITH_SIDE_EFFECTS,
        timeout=360.0,
    )(poll_inbox)
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_READ_WITH_SIDE_EFFECTS)(
        check_teammate
    )
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_DESTRUCTIVE)(
        process_shutdown_approved
    )
    mcp.tool(tags={_TAG_TEAMMATE}, annotations=_ANN_READ)(health_check)
