"""Shared FastMCP runtime state and authorization helpers."""

import logging
import os
import re
from pathlib import Path
from typing import Literal, TypedDict, cast

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from mcp.types import ToolAnnotations

from claude_teams import capabilities, teams
from claude_teams.backends import BackendRegistry, registry
from claude_teams.errors import (
    AuthenticationRequiredError,
    CwdMissingError,
    CwdNotAbsoluteError,
    CwdNotDirectoryError,
    InvalidNameError,
    InvalidPermissionModeError,
    LeadCapabilityRequiredError,
    NameTooLongError,
    PaginationLimitTooLargeError,
    PaginationLimitTooSmallError,
    PaginationOffsetNegativeError,
    PrincipalActingAsOtherError,
    TeamNotFoundToolError,
)
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
        raise CwdNotAbsoluteError(cwd)
    if not candidate.exists():
        raise CwdMissingError(candidate)
    if not candidate.is_dir():
        raise CwdNotDirectoryError(candidate)
    return str(candidate)


def _resolve_permission_mode(
    permission_mode: Literal["default", "require_approval", "bypass"] | None,
) -> Literal["default", "require_approval", "bypass"]:
    raw = permission_mode or os.environ.get("CLAUDE_TEAMS_PERMISSION_MODE", "default")
    if raw not in _PERMISSION_MODES:
        raise InvalidPermissionModeError(raw, _PERMISSION_MODES)
    return cast(Literal["default", "require_approval", "bypass"], raw)


def _normalize_pagination(limit: int, offset: int) -> tuple[int, int]:
    if limit < 1:
        raise PaginationLimitTooSmallError()
    if limit > _MAX_PAGE_SIZE:
        raise PaginationLimitTooLargeError(_MAX_PAGE_SIZE)
    if offset < 0:
        raise PaginationOffsetNegativeError()
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
    """Server-scoped (per-process) state shared across all client sessions.

    Per-session state (active team, principal identity, teammate presence
    flag) lives in ``ctx.set_state``/``get_state``/``delete_state`` which
    FastMCP auto-namespaces by ``ctx.session_id``. Storing per-session
    concepts here would clobber across concurrent HTTP/SSE sessions.
    """

    registry: BackendRegistry


@lifespan
async def app_lifespan(_server):
    """Initialize and manage the MCP server lifespan."""
    yield {"registry": registry}


mcp = FastMCP(
    name="win-agent-teams",
    instructions=(
        "Windows-native MCP server for orchestrating Claude Code and Codex "
        "agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
    # Only ``ToolError``/``ResourceError``/``PromptError`` strings reach the
    # client; unexpected exceptions surface as a generic "internal error"
    # message while the full traceback still lands in server logs. Prevents
    # filesystem paths, stack frames, and credential fragments from leaking
    # into LLM-visible tool results.
    mask_error_details=True,
    # Fail fast on type-mismatched inputs instead of silently coercing
    # (e.g. ``"10"`` → ``10``). Surfaces client bugs immediately instead of
    # letting them propagate as subtly-wrong values into team state.
    strict_input_validation=True,
)


def _get_lifespan(ctx: Context) -> _LifespanState:
    """Extract and cast the lifespan state from the MCP context."""
    return cast(_LifespanState, ctx.lifespan_context)


async def _set_session_principal(
    ctx: Context,
    team_name: str,
    principal_name: str,
    principal_role: Literal["lead", "agent"],
    *,
    lead_capability: str | None = None,
) -> None:
    """Persist the authenticated principal on the per-session context store."""
    await ctx.set_state("active_team", team_name)
    await ctx.set_state("principal_name", principal_name)
    await ctx.set_state("principal_role", principal_role)
    await ctx.set_state(
        "lead_capability",
        lead_capability if principal_role == "lead" else None,
    )

    config = await teams.read_config(team_name)
    await ctx.set_state(
        "has_teammates",
        any(isinstance(member, TeammateMember) for member in config.members),
    )


async def _clear_session_principal(ctx: Context) -> None:
    """Drop all per-session principal keys, used on team deletion/detach."""
    for key in (
        "active_team",
        "principal_name",
        "principal_role",
        "lead_capability",
        "has_teammates",
    ):
        await ctx.delete_state(key)


async def _resolve_session_principal(
    ctx: Context, team_name: str
) -> capabilities.Principal | None:
    active_team = await ctx.get_state("active_team")
    if active_team != team_name:
        return None
    principal_name = await ctx.get_state("principal_name")
    principal_role = await ctx.get_state("principal_role")
    if principal_name is None or principal_role is None:
        return None
    return {
        "name": cast(str, principal_name),
        "role": cast(Literal["lead", "agent"], principal_role),
    }


async def _resolve_authenticated_principal(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal | None:
    session_principal = await _resolve_session_principal(ctx, team_name)
    if session_principal is not None:
        return session_principal
    if capability:
        return await capabilities.resolve_principal(team_name, capability)
    return None


async def _ensure_team_exists(team_name: str) -> None:
    try:
        safe_team_name = teams.validate_safe_name(team_name, "team name")
    except (InvalidNameError, NameTooLongError) as exc:
        raise ToolError(str(exc)) from exc
    if not await teams.team_exists(safe_team_name):
        raise TeamNotFoundToolError(team_name)


async def _require_authenticated_principal(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal:
    await _ensure_team_exists(team_name)
    principal = await _resolve_authenticated_principal(ctx, team_name, capability)
    if principal is None:
        raise AuthenticationRequiredError()
    return principal


async def _require_lead(
    ctx: Context, team_name: str, capability: str = ""
) -> capabilities.Principal:
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] != "lead":
        raise LeadCapabilityRequiredError()
    return principal


async def _require_sender_or_lead(
    ctx: Context, team_name: str, sender: str, capability: str = ""
) -> capabilities.Principal:
    principal = await _require_authenticated_principal(ctx, team_name, capability)
    if principal["role"] == "lead":
        return principal
    if principal["name"] != sender:
        raise PrincipalActingAsOtherError(principal["name"], sender)
    return principal
