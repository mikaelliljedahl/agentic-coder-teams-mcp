"""Team-tier task and inbox MCP tools."""

from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import messaging, tasks
from claude_teams.models import PaginatedInboxMessages, PaginatedTaskList
from claude_teams.server_runtime import (
    _ANN_CREATE,
    _ANN_MUTATE,
    _ANN_READ,
    _ANN_READ_WITH_SIDE_EFFECTS,
    _DEFAULT_PAGE_SIZE,
    _TAG_TEAM,
    _normalize_pagination,
    _page_items,
    _require_authenticated_principal,
    _resolve_authenticated_principal,
)


async def task_create(
    team_name: str,
    subject: str,
    description: str,
    ctx: Context,
    active_form: str = "",
    metadata: dict | None = None,
    capability: str = "",
) -> dict:
    """Create a new task for the team.

    Args:
        team_name (str): Team name.
        subject (str): Task subject.
        description (str): Task description.
        ctx (Context): FastMCP request context.
        active_form (str): Optional active-form text.
        metadata (dict | None): Optional metadata payload.
        capability (str): Optional capability override.

    Returns:
        dict: Task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    try:
        task = await tasks.create_task(
            team_name, subject, description, active_form, metadata
        )
    except ValueError as e:
        raise ToolError(str(e))
    return task.model_dump(by_alias=True, exclude_none=True)


async def task_update(
    team_name: str,
    task_id: str,
    ctx: Context,
    status: Literal["pending", "in_progress", "completed", "deleted"] | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
    capability: str = "",
) -> dict:
    """Update a task and optionally notify a new assignee.

    Args:
        team_name (str): Team name.
        task_id (str): Task identifier.
        ctx (Context): FastMCP request context.
        status (Literal[...] | None): Optional new status.
        owner (str | None): Optional new owner.
        subject (str | None): Optional new subject.
        description (str | None): Optional new description.
        active_form (str | None): Optional new active-form text.
        add_blocks (list[str] | None): Tasks this task blocks.
        add_blocked_by (list[str] | None): Tasks blocking this task.
        metadata (dict | None): Metadata merge payload.
        capability (str): Optional capability override.

    Returns:
        dict: Updated task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    try:
        task = await tasks.update_task(
            team_name,
            task_id,
            status=status,
            owner=owner,
            subject=subject,
            description=description,
            active_form=active_form,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
            metadata=metadata,
        )
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    except ValueError as e:
        raise ToolError(str(e))
    if owner is not None and task.owner is not None and task.status != "deleted":
        principal = await _resolve_authenticated_principal(ctx, team_name, capability)
        assigned_by = principal["name"] if principal is not None else "team-lead"
        await messaging.send_task_assignment(team_name, task, assigned_by=assigned_by)
    return task.model_dump(by_alias=True, exclude_none=True)


async def task_list(
    team_name: str,
    ctx: Context,
    capability: str = "",
    limit: int = _DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> dict:
    """List team tasks in canonical task-id order with pagination metadata.

    Args:
        team_name (str): Team name.
        ctx (Context): FastMCP request context.
        capability (str): Optional capability override.
        limit (int): Page size.
        offset (int): Page offset.

    Returns:
        dict: Paginated task envelope.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    limit, offset = _normalize_pagination(limit, offset)
    try:
        result = await tasks.list_tasks(team_name)
    except ValueError as e:
        raise ToolError(str(e))
    paged = _page_items(
        [task.model_dump(by_alias=True, exclude_none=True) for task in result],
        limit,
        offset,
    )
    return PaginatedTaskList.model_validate(paged).model_dump(
        by_alias=True, exclude_none=True
    )


async def task_get(
    team_name: str, task_id: str, ctx: Context, capability: str = ""
) -> dict:
    """Get full details of a specific task by ID.

    Args:
        team_name (str): Team name.
        task_id (str): Task identifier.
        ctx (Context): FastMCP request context.
        capability (str): Optional capability override.

    Returns:
        dict: Task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    try:
        task = await tasks.get_task(team_name, task_id)
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    return task.model_dump(by_alias=True, exclude_none=True)


async def read_inbox(
    team_name: str,
    agent_name: str,
    ctx: Context,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: int = _DEFAULT_PAGE_SIZE,
    offset: int = 0,
    order: Literal["oldest", "newest"] = "oldest",
    capability: str = "",
) -> dict:
    """Read inbox messages with pagination metadata and explicit ordering.

    Args:
        team_name (str): Team name.
        agent_name (str): Inbox owner.
        ctx (Context): FastMCP request context.
        unread_only (bool): Whether to return only unread messages.
        mark_as_read (bool): Whether returned messages should be marked read.
        limit (int): Page size.
        offset (int): Page offset.
        order (Literal["oldest", "newest"]): Inbox ordering mode.
        capability (str): Optional capability override.

    Returns:
        dict: Paginated inbox envelope.

    """
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead" and principal["name"] != agent_name:
        raise ToolError(
            f"Authenticated principal {principal['name']!r} cannot read inbox {agent_name!r}."
        )
    limit, offset = _normalize_pagination(limit, offset)
    total_msgs = await messaging.read_inbox(
        team_name,
        agent_name,
        unread_only=unread_only,
        mark_as_read=False,
        order=order,
    )
    msgs = await messaging.read_inbox(
        team_name,
        agent_name,
        unread_only=unread_only,
        mark_as_read=mark_as_read,
        limit=limit,
        offset=offset,
        order=order,
    )
    next_offset = offset + limit
    total_count = len(total_msgs)
    has_more = next_offset < total_count
    return PaginatedInboxMessages.model_validate(
        {
            "items": [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs],
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
        }
    ).model_dump(by_alias=True, exclude_none=True)


def register_team_task_tools(mcp: FastMCP) -> None:
    """Register team-tier task and inbox tools.

    Args:
        mcp (FastMCP): MCP server instance.

    """
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_CREATE)(task_create)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_MUTATE)(task_update)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_READ)(task_list)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_READ)(task_get)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_READ_WITH_SIDE_EFFECTS)(read_inbox)
