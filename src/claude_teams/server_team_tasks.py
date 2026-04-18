"""Team-tier task and inbox MCP tools."""

from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import messaging, tasks
from claude_teams.errors import (
    BlockedTaskStatusError,
    CyclicTaskBlockedByError,
    CyclicTaskBlockError,
    InboxAccessDeniedError,
    InvalidNameError,
    InvalidTaskStatusError,
    NameTooLongError,
    TaskNotFoundToolError,
    TaskReferenceNotFoundError,
    TaskSelfBlockedByError,
    TaskSelfBlockError,
    TaskStatusRegressionError,
    TaskSubjectEmptyError,
    TeamNotFoundValueError,
)
from claude_teams.models import (
    PaginatedInboxMessages,
    PaginatedTaskList,
    TaskMetadata,
    TaskUpdateFields,
)
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
from claude_teams.server_schema import (
    ActiveForm,
    AgentName,
    Capability,
    Description,
    Limit,
    Offset,
    Subject,
    TaskId,
    TeamName,
)


async def task_create(
    team_name: TeamName,
    subject: Subject,
    description: Description,
    ctx: Context,
    active_form: ActiveForm = "",
    metadata: TaskMetadata | None = None,
    capability: Capability = "",
) -> dict[str, object]:
    """Create a new task for the team.

    Args:
        team_name: Team name.
        subject: Task subject.
        description: Task description.
        ctx: FastMCP request context.
        active_form: Optional active-form text.
        metadata: Optional metadata payload.
        capability: Optional capability override.

    Returns:
        dict: Task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    try:
        task = await tasks.create_task(
            team_name, subject, description, active_form, metadata
        )
    except (
        InvalidNameError,
        NameTooLongError,
        TaskSubjectEmptyError,
        TeamNotFoundValueError,
    ) as e:
        raise ToolError(str(e)) from e
    return task.model_dump(by_alias=True, exclude_none=True)


async def task_update(
    team_name: TeamName,
    task_id: TaskId,
    ctx: Context,
    fields: TaskUpdateFields | None = None,
    capability: Capability = "",
) -> dict[str, object]:
    """Update a task and optionally notify a new assignee.

    Args:
        team_name: Team name.
        task_id: Task identifier.
        ctx: FastMCP request context.
        fields: Optional per-field update payload (status/owner/subject/etc.).
        capability: Optional capability override.

    Returns:
        dict: Updated task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    update_fields = fields or TaskUpdateFields()
    try:
        task = await tasks.update_task(team_name, task_id, update_fields)
    except TeamNotFoundValueError as e:
        raise ToolError(str(e)) from e
    except FileNotFoundError:
        raise TaskNotFoundToolError(task_id, team_name) from None
    except (
        BlockedTaskStatusError,
        CyclicTaskBlockedByError,
        CyclicTaskBlockError,
        InvalidNameError,
        InvalidTaskStatusError,
        NameTooLongError,
        TaskReferenceNotFoundError,
        TaskSelfBlockedByError,
        TaskSelfBlockError,
        TaskStatusRegressionError,
    ) as e:
        raise ToolError(str(e)) from e
    if (
        update_fields.owner is not None
        and task.owner is not None
        and task.status != "deleted"
    ):
        principal = await _resolve_authenticated_principal(ctx, team_name, capability)
        assigned_by = principal["name"] if principal is not None else "team-lead"
        await messaging.send_task_assignment(team_name, task, assigned_by=assigned_by)
    return task.model_dump(by_alias=True, exclude_none=True)


async def task_list(
    team_name: TeamName,
    ctx: Context,
    capability: Capability = "",
    limit: Limit = _DEFAULT_PAGE_SIZE,
    offset: Offset = 0,
) -> dict[str, object]:
    """List team tasks in canonical task-id order with pagination metadata.

    Args:
        team_name: Team name.
        ctx: FastMCP request context.
        capability: Optional capability override.
        limit: Page size.
        offset: Page offset.

    Returns:
        dict: Paginated task envelope.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    limit, offset = _normalize_pagination(limit, offset)
    try:
        result = await tasks.list_tasks(team_name)
    except (InvalidNameError, NameTooLongError, TeamNotFoundValueError) as e:
        raise ToolError(str(e)) from e
    paged = _page_items(
        [task.model_dump(by_alias=True, exclude_none=True) for task in result],
        limit,
        offset,
    )
    return PaginatedTaskList.model_validate(paged).model_dump(
        by_alias=True, exclude_none=True
    )


async def task_get(
    team_name: TeamName, task_id: TaskId, ctx: Context, capability: Capability = ""
) -> dict[str, object]:
    """Get full details of a specific task by ID.

    Args:
        team_name: Team name.
        task_id: Task identifier.
        ctx: FastMCP request context.
        capability: Optional capability override.

    Returns:
        dict: Task payload.

    """
    await _require_authenticated_principal(ctx, team_name, capability)
    try:
        task = await tasks.get_task(team_name, task_id)
    except TeamNotFoundValueError as e:
        raise ToolError(str(e)) from e
    except FileNotFoundError:
        raise TaskNotFoundToolError(task_id, team_name) from None
    return task.model_dump(by_alias=True, exclude_none=True)


async def read_inbox(
    team_name: TeamName,
    agent_name: AgentName,
    ctx: Context,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: Limit = _DEFAULT_PAGE_SIZE,
    offset: Offset = 0,
    order: Literal["oldest", "newest"] = "oldest",
    capability: Capability = "",
) -> dict[str, object]:
    """Read inbox messages with pagination metadata and explicit ordering.

    Args:
        team_name: Team name.
        agent_name: Inbox owner.
        ctx: FastMCP request context.
        unread_only: Whether to return only unread messages.
        mark_as_read: Whether returned messages should be marked read.
        limit: Page size.
        offset: Page offset.
        order: Inbox ordering mode.
        capability: Optional capability override.

    Returns:
        dict: Paginated inbox envelope.

    """
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead" and principal["name"] != agent_name:
        raise InboxAccessDeniedError("read", principal["name"], agent_name)
    limit, offset = _normalize_pagination(limit, offset)
    msgs, total_count = await messaging.read_inbox_page(
        team_name,
        agent_name,
        unread_only=unread_only,
        mark_as_read=mark_as_read,
        limit=limit,
        offset=offset,
        order=order,
    )
    next_offset = offset + limit
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
