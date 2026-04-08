"""Team-tier spawn and messaging MCP tools."""

import asyncio
import time
from pathlib import Path
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, messaging, teams
from claude_teams.async_utils import run_blocking
from claude_teams.backends import SpawnRequest
from claude_teams.models import (
    COLOR_PALETTE,
    InboxMessage,
    MemberUnion,
    SendMessageResult,
    ShutdownApproved,
    SpawnResult,
    TeammateMember,
    MessageRouting,
)
from claude_teams.server_team_relay import (
    build_agent_auth_notice,
    create_one_shot_result_path,
    log_relay_task_exception,
    relay_one_shot_result,
)
from claude_teams.teams import _VALID_NAME_RE
from claude_teams.server_runtime import (
    _ANN_CREATE,
    _ANN_MUTATE,
    _TAG_TEAM,
    _TAG_TEAMMATE,
    _get_lifespan,
    _require_lead,
    _require_sender_or_lead,
    _resolve_permission_mode,
    _resolve_spawn_cwd,
    logger,
)


def _find_member(config: teams.TeamConfig, member_name: str) -> MemberUnion | None:
    """Find a team member by name.

    Args:
        config (teams.TeamConfig): Team configuration.
        member_name (str): Member name to find.

    Returns:
        teams.MemberUnion | None: Matching member or None.

    """
    for member in config.members:
        if member.name == member_name:
            return member
    return None


async def _assign_color(team_name: str) -> str:
    """Return the next teammate color for the team.

    Args:
        team_name (str): Team name.

    Returns:
        str: Next color from the shared palette.

    """
    config = await teams.read_config(team_name)
    teammate_count = sum(
        1 for member in config.members if isinstance(member, TeammateMember)
    )
    return COLOR_PALETTE[teammate_count % len(COLOR_PALETTE)]


async def spawn_teammate_tool(
    team_name: str,
    name: str,
    prompt: str,
    ctx: Context,
    cwd: str = "",
    model: str = "balanced",
    backend: str = "",
    subagent_type: str = "general-purpose",
    plan_mode_required: bool = False,
    permission_mode: Literal["default", "require_approval", "bypass"] | None = None,
    capability: str = "",
) -> dict[str, object]:
    """Spawn a teammate via the selected backend.

    Args:
        team_name (str): Team name.
        name (str): New teammate name.
        prompt (str): Initial prompt to send to the teammate.
        ctx (Context): FastMCP request context.
        cwd (str): Optional absolute working directory.
        model (str): Generic or backend-specific model name.
        backend (str): Explicit backend name or empty for default.
        subagent_type (str): Agent subtype to record in config.
        plan_mode_required (bool): Whether plan mode is required.
        permission_mode (Literal["default", "require_approval", "bypass"] | None):
            Permission behavior override.
        capability (str): Optional lead capability override.

    Returns:
        dict: Spawn result payload.

    """
    await _require_lead(ctx, team_name, capability)
    ls = _get_lifespan(ctx)
    reg = ls["registry"]

    if backend:
        try:
            backend_obj = reg.get(backend)
        except KeyError as exc:
            raise ToolError(str(exc))
    else:
        try:
            backend_obj = reg.get(reg.default_backend())
        except (RuntimeError, KeyError) as exc:
            raise ToolError(str(exc))

    try:
        resolved_model = await run_blocking(backend_obj.resolve_model, model)
    except ValueError as exc:
        raise ToolError(str(exc))

    resolved_permission_mode = _resolve_permission_mode(permission_mode)
    if resolved_permission_mode == "bypass" and not await run_blocking(
        backend_obj.supports_permission_bypass
    ):
        raise ToolError(
            f"Backend {backend_obj.name!r} does not support permission_mode='bypass'."
        )

    if not _VALID_NAME_RE.match(name):
        raise ToolError(
            f"Invalid agent name: {name!r}. Use only letters, numbers, hyphens, underscores."
        )
    if len(name) > 64:
        raise ToolError(f"Agent name too long ({len(name)} chars, max 64)")
    if name == "team-lead":
        raise ToolError("Agent name 'team-lead' is reserved")

    resolved_cwd = await run_blocking(_resolve_spawn_cwd, cwd)
    color = await _assign_color(team_name)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=subagent_type,
        model=resolved_model,
        prompt=prompt,
        color=color,
        plan_mode_required=plan_mode_required,
        joined_at=int(time.time() * 1000),
        tmux_pane_id="",
        cwd=resolved_cwd,
        backend_type=backend_obj.name,
        is_active=False,
        process_handle="",
    )

    try:
        await teams.add_member(team_name, member)
    except ValueError as exc:
        raise ToolError(str(exc))

    agent_capability = await capabilities.issue_agent_capability(team_name, name)
    await messaging.ensure_inbox(team_name, name)
    await messaging.append_message(
        team_name,
        name,
        InboxMessage(
            from_="team-lead",
            text=prompt + build_agent_auth_notice(team_name, agent_capability),
            timestamp=messaging.now_iso(),
            read=False,
        ),
    )

    one_shot_result_path: Path | None = None
    extra = {"agent_capability": agent_capability}
    if backend_obj.name == "codex":
        one_shot_result_path = create_one_shot_result_path(team_name, name)
        extra["output_last_message_path"] = str(one_shot_result_path)

    request = SpawnRequest(
        agent_id=member.agent_id,
        name=name,
        team_name=team_name,
        prompt=prompt,
        model=resolved_model,
        agent_type=subagent_type,
        color=color,
        cwd=resolved_cwd,
        lead_session_id=ls["session_id"],
        permission_mode=resolved_permission_mode,
        plan_mode_required=plan_mode_required,
        extra=extra,
    )
    try:
        spawn_result = await run_blocking(backend_obj.spawn, request)
    except Exception as exc:
        await teams.remove_member(team_name, name)
        await capabilities.remove_agent_capability(team_name, name)
        raise ToolError(f"Backend spawn failed: {exc}")

    config = await teams.read_config(team_name)
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == name:
            config_member.process_handle = spawn_result.process_handle
            config_member.tmux_pane_id = spawn_result.process_handle
            break
    await teams.write_config(team_name, config)

    if not ls["has_teammates"]:
        ls["has_teammates"] = True
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    if not backend_obj.is_interactive:
        try:
            await run_blocking(
                backend_obj.retain_pane_after_exit, spawn_result.process_handle
            )
        except Exception:
            logger.debug("Failed to set remain-on-exit for %s", name, exc_info=True)
        relay_task = asyncio.create_task(
            relay_one_shot_result(
                team_name=team_name,
                agent_name=name,
                backend_type=backend_obj.name,
                process_handle=spawn_result.process_handle,
                result_file=one_shot_result_path,
                color=color,
            )
        )
        relay_task.add_done_callback(log_relay_task_exception)

    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    ).model_dump()


async def send_message(
    team_name: str,
    ctx: Context,
    type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ],
    recipient: str = "",
    content: str = "",
    summary: str = "",
    request_id: str = "",
    approve: bool | None = None,
    sender: str = "team-lead",
    capability: str = "",
) -> dict[str, object]:
    """Send a team protocol message.

    Args:
        team_name (str): Team name.
        ctx (Context): FastMCP request context.
        type (Literal[...]): Message type.
        recipient (str): Target member when applicable.
        content (str): Message body or rejection reason.
        summary (str): Summary text for plain messages.
        request_id (str): Request identifier for response types.
        approve (bool | None): Approval flag for response types.
        sender (str): Logical sender identity.
        capability (str): Optional capability override.

    Returns:
        dict: Send result payload.

    """
    if type == "message":
        await _require_sender_or_lead(ctx, team_name, sender, capability)
        if not content:
            raise ToolError("Message content must not be empty")
        if not summary:
            raise ToolError("Message summary must not be empty")
        if not recipient:
            raise ToolError("Message recipient must not be empty")
        config = await teams.read_config(team_name)
        sender_member = _find_member(config, sender)
        if sender_member is None:
            raise ToolError(f"Sender {sender!r} is not a member of team {team_name!r}")
        recipient_member = _find_member(config, recipient)
        if recipient_member is None:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        target_color = (
            recipient_member.color
            if isinstance(recipient_member, TeammateMember)
            else None
        )
        sender_color = (
            sender_member.color if isinstance(sender_member, TeammateMember) else None
        )
        await messaging.send_plain_message(
            team_name,
            sender,
            recipient,
            content,
            summary=summary,
            color=sender_color,
        )
        routing: MessageRouting = {
            "sender": sender,
            "target": recipient,
            "targetColor": target_color,
            "summary": summary,
            "content": content,
        }
        return SendMessageResult(
            success=True,
            message=f"Message sent to {recipient}",
            routing=routing,
        ).model_dump(exclude_none=True)

    if type == "broadcast":
        await _require_lead(ctx, team_name, capability)
        if not summary:
            raise ToolError("Broadcast summary must not be empty")
        if sender != "team-lead":
            raise ToolError("Broadcast sender must be 'team-lead'")
        config = await teams.read_config(team_name)
        count = 0
        for member in config.members:
            if isinstance(member, TeammateMember):
                await messaging.send_plain_message(
                    team_name,
                    "team-lead",
                    member.name,
                    content,
                    summary=summary,
                    color=None,
                )
                count += 1
        return SendMessageResult(
            success=True,
            message=f"Broadcast sent to {count} teammate(s)",
        ).model_dump(exclude_none=True)

    if type == "shutdown_request":
        await _require_lead(ctx, team_name, capability)
        if not recipient:
            raise ToolError("Shutdown request recipient must not be empty")
        if recipient == "team-lead":
            raise ToolError("Cannot send shutdown request to team-lead")
        config = await teams.read_config(team_name)
        member_names = {member.name for member in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        req_id = await messaging.send_shutdown_request(
            team_name, recipient, reason=content
        )
        return SendMessageResult(
            success=True,
            message=f"Shutdown request sent to {recipient}",
            request_id=req_id,
            target=recipient,
        ).model_dump(exclude_none=True)

    if type == "shutdown_response":
        await _require_sender_or_lead(ctx, team_name, sender, capability)
        if approve:
            config = await teams.read_config(team_name)
            member = None
            for config_member in config.members:
                if (
                    isinstance(config_member, TeammateMember)
                    and config_member.name == sender
                ):
                    member = config_member
                    break
            pane_id = member.tmux_pane_id if member else ""
            process_handle = (
                (member.process_handle or member.tmux_pane_id) if member else ""
            )
            backend_type = member.backend_type if member else "tmux"
            payload = ShutdownApproved(
                request_id=request_id,
                from_=sender,
                timestamp=messaging.now_iso(),
                pane_id=pane_id,
                backend_type=backend_type,
                process_handle=process_handle,
            )
            await messaging.send_structured_message(
                team_name, sender, "team-lead", payload
            )
            return SendMessageResult(
                success=True,
                message=f"Shutdown approved for request {request_id}",
            ).model_dump(exclude_none=True)
        await messaging.send_plain_message(
            team_name,
            sender,
            "team-lead",
            content or "Shutdown rejected",
            summary="shutdown_rejected",
        )
        return SendMessageResult(
            success=True,
            message=f"Shutdown rejected for request {request_id}",
        ).model_dump(exclude_none=True)

    if type == "plan_approval_response":
        await _require_sender_or_lead(ctx, team_name, sender, capability)
        if not recipient:
            raise ToolError("Plan approval recipient must not be empty")
        config = await teams.read_config(team_name)
        member_names = {member.name for member in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        if approve:
            await messaging.send_plain_message(
                team_name,
                sender,
                recipient,
                '{"type":"plan_approval","approved":true}',
                summary="plan_approved",
            )
        else:
            await messaging.send_plain_message(
                team_name,
                sender,
                recipient,
                content or "Plan rejected",
                summary="plan_rejected",
            )
        return SendMessageResult(
            success=True,
            message=f"Plan {'approved' if approve else 'rejected'} for {recipient}",
        ).model_dump(exclude_none=True)

    raise ToolError(f"Unknown message type: {type}")


def register_team_spawn_tools(mcp: FastMCP) -> None:
    """Register team-tier spawn and messaging tools.

    Args:
        mcp (FastMCP): MCP server instance.

    """
    mcp.tool(name="spawn_teammate", tags={_TAG_TEAM}, annotations=_ANN_CREATE)(
        spawn_teammate_tool
    )
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_MUTATE)(send_message)
