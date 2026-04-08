"""Shared FastMCP runtime state and authorization helpers."""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Literal, TypedDict, cast

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from mcp.types import ToolAnnotations

from claude_teams import capabilities, teams
from claude_teams.backends import BackendRegistry, registry
from claude_teams.models import TeammateMember

_TAG_BOOTSTRAP = "bootstrap"
_TAG_TEAM = "team"
_TAG_TEAMMATE = "teammate"
_ONE_SHOT_RESULT_MAX_CHARS = 12000
_ONE_SHOT_TIMEOUT_S = 900
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 500
_PERMISSION_MODES = {"default", "require_approval", "bypass"}

logger = logging.getLogger(__name__)


def _ann(
    *,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
) -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


_ANN_CREATE = _ann(
    read_only=False, destructive=False, idempotent=False, open_world=False
)
_ANN_ATTACH = _ann(
    read_only=False, destructive=False, idempotent=False, open_world=False
)
_ANN_DELETE = _ann(
    read_only=False, destructive=True, idempotent=False, open_world=False
)
_ANN_MUTATE = _ann(
    read_only=False, destructive=False, idempotent=False, open_world=False
)
_ANN_DESTRUCTIVE = _ann(
    read_only=False, destructive=True, idempotent=False, open_world=False
)
_ANN_READ = _ann(read_only=True, destructive=False, idempotent=True, open_world=False)
_ANN_READ_WITH_SIDE_EFFECTS = _ann(
    read_only=False, destructive=False, idempotent=False, open_world=False
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage returns from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _resolve_spawn_cwd(cwd: str) -> str:
    """Return a validated working directory for a new teammate."""
    if not cwd:
        return str(Path.cwd())

    candidate = Path(cwd).expanduser()
    if not candidate.is_absolute():
        raise ToolError(f"cwd must be an absolute path, got: {cwd!r}")
    if not candidate.exists():
        raise ToolError(f"cwd does not exist: {candidate}")
    if not candidate.is_dir():
        raise ToolError(f"cwd is not a directory: {candidate}")
    return str(candidate)


def _resolve_permission_mode(
    permission_mode: Literal["default", "require_approval", "bypass"] | None,
) -> Literal["default", "require_approval", "bypass"]:
    raw = permission_mode or os.environ.get("CLAUDE_TEAMS_PERMISSION_MODE", "default")
    if raw not in _PERMISSION_MODES:
        supported = ", ".join(sorted(_PERMISSION_MODES))
        raise ToolError(
            f"Invalid permission mode {raw!r}. Supported values: {supported}."
        )
    return cast(Literal["default", "require_approval", "bypass"], raw)


def _validate_agent_name(name: str, label: str = "agent name") -> str:
    """Validate a safe agent-style identifier and raise ToolError on failure."""
    try:
        return teams.validate_safe_name(name, label)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _normalize_pagination(limit: int, offset: int) -> tuple[int, int]:
    if limit < 1:
        raise ToolError("limit must be >= 1")
    if limit > _MAX_PAGE_SIZE:
        raise ToolError(f"limit must be <= {_MAX_PAGE_SIZE}")
    if offset < 0:
        raise ToolError("offset must be >= 0")
    return limit, offset


def _page_items(
    items: list[dict[str, object]], limit: int, offset: int
) -> dict[str, object]:
    total_count = len(items)
    page_items = items[offset : offset + limit]
    next_offset = offset + limit
    has_more = next_offset < total_count
    return {
        "items": page_items,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
    }


class _LifespanState(TypedDict):
    registry: BackendRegistry
    session_id: str
    active_team: str | None
    has_teammates: bool
    principal_name: str | None
    principal_role: Literal["lead", "agent"] | None
    lead_capability: str | None


@lifespan
async def app_lifespan(server):
    """Initialize and manage the MCP server lifespan."""
    session_id = str(uuid.uuid4())
    yield {
        "registry": registry,
        "session_id": session_id,
        "active_team": None,
        "has_teammates": False,
        "principal_name": None,
        "principal_role": None,
        "lead_capability": None,
    }


mcp = FastMCP(
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
)
mcp.enable(tags={_TAG_BOOTSTRAP}, only=True, components={"tool"})


def _get_lifespan(ctx: Context) -> _LifespanState:
    """Extract and cast the lifespan state from the MCP context."""
    return cast(_LifespanState, ctx.lifespan_context)


async def _set_session_principal(
    ls: _LifespanState,
    team_name: str,
    principal_name: str,
    principal_role: Literal["lead", "agent"],
    *,
    lead_capability: str | None = None,
) -> None:
    ls["active_team"] = team_name
    ls["principal_name"] = principal_name
    ls["principal_role"] = principal_role
    ls["lead_capability"] = lead_capability if principal_role == "lead" else None

    config = await teams.read_config(team_name)
    ls["has_teammates"] = any(
        isinstance(member, TeammateMember) for member in config.members
    )


def _resolve_session_principal(
    ctx: Context, team_name: str
) -> capabilities.Principal | None:
    ls = _get_lifespan(ctx)
    if ls.get("active_team") != team_name:
        return None
    if ls.get("principal_name") is None or ls.get("principal_role") is None:
        return None
    return {
        "name": cast(str, ls["principal_name"]),
        "role": cast(Literal["lead", "agent"], ls["principal_role"]),
    }


async def _resolve_authenticated_principal(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal | None:
    session_principal = _resolve_session_principal(ctx, team_name)
    if session_principal is not None:
        return session_principal
    if capability:
        return await capabilities.resolve_principal(team_name, capability)
    return None


async def _ensure_team_exists(team_name: str) -> None:
    try:
        safe_team_name = teams.validate_safe_name(team_name, "team name")
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if not await teams.team_exists(safe_team_name):
        raise ToolError(f"Team {team_name!r} not found")


async def _require_authenticated_principal(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal:
    await _ensure_team_exists(team_name)
    principal = await _resolve_authenticated_principal(ctx, team_name, capability)
    if principal is None:
        raise ToolError(
            "This action requires an attached team session or a valid capability."
        )
    return principal


async def _require_lead(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal:
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead":
        raise ToolError("This action requires the team-lead capability.")
    return principal


async def _require_sender_or_lead(
    ctx: Context, team_name: str, sender: str, capability: str = ""
) -> capabilities.Principal:
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] == "lead":
        return principal
    if principal["name"] != sender:
        raise ToolError(
            f"Authenticated principal {principal['name']!r} cannot act as {sender!r}."
        )
    return principal
