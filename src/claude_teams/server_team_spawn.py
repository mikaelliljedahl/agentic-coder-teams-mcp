"""Team-tier spawn and messaging MCP tools."""

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, messaging, teams
from claude_teams.async_utils import run_blocking
from claude_teams.backends import SpawnRequest
from claude_teams.backends.contracts import AgentProfile
from claude_teams.errors import (
    AgentSelectUnsupportedToolError,
    BackendNotRegisteredError,
    BackendSpawnFailedError,
    BroadcastSenderError,
    BroadcastSummaryEmptyToolError,
    BroadcastTooManyRecipientsError,
    InvalidReasoningEffortToolError,
    MemberAlreadyExistsError,
    MessageContentEmptyToolError,
    MessageRecipientEmptyToolError,
    MessageSummaryEmptyToolError,
    NoBackendsAvailableError,
    NotTeamMemberError,
    PermissionBypassUnsupportedToolError,
    PlanRecipientEmptyToolError,
    ReasoningEffortUnsupportedToolError,
    ReservedAgentNameError,
    ShutdownRecipientEmptyToolError,
    ShutdownResponseApprovalRequiredError,
    ShutdownSelfError,
    UnknownAgentProfileToolError,
    UnknownMessageTypeError,
    UnsupportedBackendModelError,
)
from claude_teams.models import (
    COLOR_PALETTE,
    InboxMessage,
    MemberUnion,
    MessageRouting,
    SendMessageResult,
    ShutdownApproved,
    SpawnOptions,
    SpawnResult,
    TeammateMember,
)
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
)
from claude_teams.server_schema import (
    AgentName,
    Capability,
    MessageContent,
    MessageSummary,
    Prompt,
    RequestId,
    SenderName,
    TeamName,
)
from claude_teams.server_team_relay import (
    build_agent_auth_notice,
    create_one_shot_result_path,
    log_relay_task_exception,
    relay_one_shot_result,
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


async def _validate_reasoning_effort(backend_obj, effort: str | None) -> None:
    """Validate reasoning_effort against the backend's spec.

    Raises:
        ReasoningEffortUnsupportedToolError: Backend does not expose a spec.
        InvalidReasoningEffortToolError: Value is outside the spec's options.

    """
    if effort is None:
        return
    spec = await run_blocking(backend_obj.reasoning_effort_spec)
    if spec is None:
        raise ReasoningEffortUnsupportedToolError(backend_obj.name)
    if effort not in spec.options:
        raise InvalidReasoningEffortToolError(backend_obj.name, effort, spec.options)


async def _validate_agent_profile(
    backend_obj, profile: str | None, cwd: str
) -> AgentProfile | None:
    """Validate agent_profile against the backend's spec and current discovery.

    Returns the matched ``AgentProfile`` so the caller can thread its
    resolved path through ``SpawnRequest.extra`` — backends would
    otherwise re-run ``discover_agents`` at command-build time just to
    recover the path for the spec's ``value_template``.

    Returns:
        The discovered ``AgentProfile`` when ``profile`` was set and
        matched, otherwise ``None``.

    Raises:
        AgentSelectUnsupportedToolError: Backend lacks an agent_select_spec.
        UnknownAgentProfileToolError: Name not among discovered profiles.

    """
    if profile is None:
        return None
    spec = await run_blocking(backend_obj.agent_select_spec)
    if spec is None:
        raise AgentSelectUnsupportedToolError(backend_obj.name)
    profiles = await run_blocking(backend_obj.discover_agents, cwd)
    for discovered in profiles:
        if discovered.name == profile:
            return discovered
    names = [p.name for p in profiles]
    raise UnknownAgentProfileToolError(backend_obj.name, profile, names)


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
    team_name: TeamName,
    name: AgentName,
    prompt: Prompt,
    ctx: Context,
    options: SpawnOptions | None = None,
) -> dict[str, object]:
    """Spawn a teammate via the selected backend.

    Emits an indeterminate-progress heartbeat every 15 seconds while the
    backend spawn is in flight. One-shot backends (codex, qwen, gemini)
    can take several minutes to return; the heartbeat prevents the call
    from appearing silent. Quick spawns (interactive backends) finish
    before the first tick fires, so short-lived spawns emit no progress
    events at all — the request/response lifecycle is enough for those.

    Args:
        team_name: Team the new teammate joins.
        name: Member name for the new teammate (must be unique within the team
            and not equal to the reserved ``team-lead``).
        prompt: Initial prompt delivered to the teammate via its inbox.
        ctx: FastMCP request context (injected).
        options: Optional tuning knobs (backend, model, cwd, subagent_type,
            plan_mode_required, permission_mode, capability). Omit to accept
            all defaults.

    """
    opts = options or SpawnOptions()

    await _require_lead(ctx, team_name, opts.capability)
    ls = _get_lifespan(ctx)
    reg = ls["registry"]

    if opts.backend:
        try:
            backend_obj = reg.get(opts.backend)
        except BackendNotRegisteredError as exc:
            raise ToolError(str(exc)) from exc
    else:
        try:
            backend_obj = reg.get(reg.default_backend())
        except (NoBackendsAvailableError, BackendNotRegisteredError) as exc:
            raise ToolError(str(exc)) from exc

    try:
        resolved_model = await run_blocking(backend_obj.resolve_model, opts.model)
    except UnsupportedBackendModelError as exc:
        raise ToolError(str(exc)) from exc

    resolved_permission_mode = _resolve_permission_mode(opts.permission_mode)
    if resolved_permission_mode == "bypass" and not await run_blocking(
        backend_obj.supports_permission_bypass
    ):
        raise PermissionBypassUnsupportedToolError(backend_obj.name)

    await _validate_reasoning_effort(backend_obj, opts.reasoning_effort)

    if name == "team-lead":
        raise ReservedAgentNameError()

    resolved_cwd = await run_blocking(_resolve_spawn_cwd, opts.cwd)
    resolved_profile = await _validate_agent_profile(
        backend_obj, opts.agent_profile, resolved_cwd
    )
    color = await _assign_color(team_name)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=opts.subagent_type,
        model=resolved_model,
        prompt=prompt,
        color=color,
        plan_mode_required=opts.plan_mode_required,
        joined_at=int(time.time() * 1000),
        tmux_pane_id="",
        cwd=resolved_cwd,
        backend_type=backend_obj.name,
        is_active=False,
        process_handle="",
    )

    try:
        await teams.add_member(team_name, member)
    except MemberAlreadyExistsError as exc:
        raise ToolError(str(exc)) from exc

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
    if resolved_profile is not None:
        # Thread the resolved path through so ``_agent_args`` doesn't
        # re-scan the filesystem / re-parse TOML at command-build time.
        extra["agent_profile_path"] = resolved_profile.path

    request = SpawnRequest(
        agent_id=member.agent_id,
        name=name,
        team_name=team_name,
        prompt=prompt,
        model=resolved_model,
        agent_type=opts.subagent_type,
        color=color,
        cwd=resolved_cwd,
        lead_session_id=ctx.session_id,
        permission_mode=resolved_permission_mode,
        plan_mode_required=opts.plan_mode_required,
        reasoning_effort=opts.reasoning_effort,
        agent_profile=opts.agent_profile,
        extra=extra,
    )

    async def _spawn_heartbeat() -> None:
        """Emit an indeterminate-progress pulse every 15s until cancelled.

        Uses ``total=None`` so the client renders this as ongoing-but-
        unknown-duration work (typically a spinner, not a percentage
        bar). The elapsed-seconds count in the message gives the user
        a concrete "how long has this been going?" signal without
        implying a known completion target.
        """
        elapsed = 0
        while True:
            await asyncio.sleep(15)
            elapsed += 15
            await ctx.report_progress(
                progress=elapsed,
                total=None,
                message=(
                    f"Still spawning {backend_obj.name!r} for {name!r} "
                    f"({elapsed}s elapsed)..."
                ),
            )

    heartbeat_task = asyncio.create_task(_spawn_heartbeat())
    try:
        spawn_result = await run_blocking(backend_obj.spawn, request)
    except Exception as exc:
        await teams.remove_member(team_name, name)
        await capabilities.remove_agent_capability(team_name, name)
        raise BackendSpawnFailedError(exc) from exc
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    config = await teams.read_config(team_name)
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == name:
            config_member.process_handle = spawn_result.process_handle
            config_member.tmux_pane_id = spawn_result.process_handle
            break
    await teams.write_config(team_name, config)

    if not await ctx.get_state("has_teammates"):
        await ctx.set_state("has_teammates", True)
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    if not backend_obj.is_interactive:
        try:
            await run_blocking(
                backend_obj.retain_pane_after_exit, spawn_result.process_handle
            )
        except Exception as exc:
            await ctx.debug(
                f"Failed to set remain-on-exit for {name!r}: {exc}",
                extra={
                    "team": team_name,
                    "agent": name,
                    "backend": backend_obj.name,
                },
            )
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


async def _handle_plain_message(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    summary: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='message'`` — addressed plain message."""
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if not content:
        raise MessageContentEmptyToolError()
    if not summary:
        raise MessageSummaryEmptyToolError()
    if not recipient:
        raise MessageRecipientEmptyToolError()
    config = await teams.read_config(team_name)
    sender_member = _find_member(config, sender)
    if sender_member is None:
        raise NotTeamMemberError("Sender", sender, team_name)
    recipient_member = _find_member(config, recipient)
    if recipient_member is None:
        raise NotTeamMemberError("Recipient", recipient, team_name)
    target_color = (
        recipient_member.color if isinstance(recipient_member, TeammateMember) else None
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


_BROADCAST_RECIPIENT_LIMIT = 50


async def _handle_broadcast(
    team_name: str,
    ctx: Context,
    content: str,
    summary: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='broadcast'`` — lead-only fan-out to teammates.

    Recipient fan-out is capped at ``_BROADCAST_RECIPIENT_LIMIT``. Each
    delivery acquires a per-inbox file lock, so a broadcast to a large team
    could monopolise the messaging lane; the cap keeps a single call
    bounded. Teams that legitimately exceed the cap should send targeted
    messages instead.
    """
    await _require_lead(ctx, team_name, capability)
    if not summary:
        raise BroadcastSummaryEmptyToolError()
    if sender != "team-lead":
        raise BroadcastSenderError()
    config = await teams.read_config(team_name)
    teammates = [m for m in config.members if isinstance(m, TeammateMember)]
    if len(teammates) > _BROADCAST_RECIPIENT_LIMIT:
        raise BroadcastTooManyRecipientsError(
            len(teammates), _BROADCAST_RECIPIENT_LIMIT
        )
    for member in teammates:
        await messaging.send_plain_message(
            team_name,
            "team-lead",
            member.name,
            content,
            summary=summary,
            color=None,
        )
    return SendMessageResult(
        success=True,
        message=f"Broadcast sent to {len(teammates)} teammate(s)",
    ).model_dump(exclude_none=True)


async def _handle_shutdown_request(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='shutdown_request'`` — lead asks teammate to exit."""
    await _require_lead(ctx, team_name, capability)
    if not recipient:
        raise ShutdownRecipientEmptyToolError()
    if recipient == "team-lead":
        raise ShutdownSelfError()
    config = await teams.read_config(team_name)
    member_names = {member.name for member in config.members}
    if recipient not in member_names:
        raise NotTeamMemberError("Recipient", recipient, team_name)
    req_id = await messaging.send_shutdown_request(team_name, recipient, reason=content)
    return SendMessageResult(
        success=True,
        message=f"Shutdown request sent to {recipient}",
        request_id=req_id,
        target=recipient,
    ).model_dump(exclude_none=True)


async def _handle_shutdown_response(
    team_name: str,
    ctx: Context,
    request_id: str,
    approve: bool | None,
    content: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='shutdown_response'`` — teammate accepts or rejects.

    ``approve`` must be set to ``True`` or ``False`` explicitly; a missing
    value (``None``) is rejected so that a client omitting the flag cannot
    silently fall through to the rejection branch and leave the lead with a
    stale approval request. The sender must be a registered teammate — the
    approval path publishes backend/pane identifiers the lead uses to clean
    up the process, so an unknown sender would otherwise fabricate a
    ``ShutdownApproved`` payload with empty handles.
    """
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if approve is None:
        raise ShutdownResponseApprovalRequiredError()
    config = await teams.read_config(team_name)
    member = None
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == sender:
            member = config_member
            break
    if member is None:
        raise NotTeamMemberError("Sender", sender, team_name)
    if approve is True:
        payload = ShutdownApproved(
            request_id=request_id,
            from_=sender,
            timestamp=messaging.now_iso(),
            pane_id=member.tmux_pane_id,
            backend_type=member.backend_type,
            process_handle=member.process_handle or member.tmux_pane_id,
        )
        await messaging.send_structured_message(team_name, sender, "team-lead", payload)
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


async def _handle_plan_approval_response(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    approve: bool | None,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='plan_approval_response'`` — plan approval or rejection."""
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if not recipient:
        raise PlanRecipientEmptyToolError()
    config = await teams.read_config(team_name)
    member_names = {member.name for member in config.members}
    if recipient not in member_names:
        raise NotTeamMemberError("Recipient", recipient, team_name)
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


async def send_message(
    team_name: TeamName,
    ctx: Context,
    message_type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ],
    recipient: str = "",
    content: MessageContent = "",
    summary: MessageSummary = "",
    request_id: RequestId = "",
    approve: bool | None = None,
    sender: SenderName = "team-lead",
    capability: Capability = "",
) -> dict[str, object]:
    """Send a team protocol message.

    Dispatches to a per-type handler. Each handler owns its own validation,
    lookups, and response shape — ``send_message`` itself is a thin router
    to keep the branch count low and make each protocol message type
    independently testable.

    Args:
        team_name: Team name.
        ctx: FastMCP request context.
        message_type: Message type.
        recipient: Target member when applicable.
        content: Message body or rejection reason.
        summary: Summary text for plain messages.
        request_id: Request identifier for response types.
        approve: Approval flag for response types.
        sender: Logical sender identity.
        capability: Optional capability override.

    Returns:
        dict: Send result payload.

    """
    if message_type == "message":
        return await _handle_plain_message(
            team_name, ctx, recipient, content, summary, sender, capability
        )
    if message_type == "broadcast":
        return await _handle_broadcast(
            team_name, ctx, content, summary, sender, capability
        )
    if message_type == "shutdown_request":
        return await _handle_shutdown_request(
            team_name, ctx, recipient, content, capability
        )
    if message_type == "shutdown_response":
        return await _handle_shutdown_response(
            team_name, ctx, request_id, approve, content, sender, capability
        )
    if message_type == "plan_approval_response":
        return await _handle_plan_approval_response(
            team_name, ctx, recipient, content, approve, sender, capability
        )
    raise UnknownMessageTypeError(message_type)


def register_team_spawn_tools(mcp: FastMCP) -> None:
    """Register team-tier spawn and messaging tools.

    ``spawn_teammate`` carries a generous 15-minute timeout because one-shot
    backends can take several minutes to return; other tools inherit the
    default FastMCP request timeout.

    Args:
        mcp (FastMCP): MCP server instance.

    """
    mcp.tool(
        name="spawn_teammate",
        tags={_TAG_TEAM},
        annotations=_ANN_CREATE,
        timeout=900.0,
    )(spawn_teammate_tool)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_MUTATE)(send_message)
