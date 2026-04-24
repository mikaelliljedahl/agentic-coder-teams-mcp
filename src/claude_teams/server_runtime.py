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
    InvalidEnvVarValueError,
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
from claude_teams.models import SpawnOptions, TeammateMember

_TAG_BOOTSTRAP = "bootstrap"
_TAG_TEAM = "team"
_TAG_TEAMMATE = "teammate"
_ONE_SHOT_RESULT_MAX_CHARS = 12000
_ONE_SHOT_TIMEOUT_S = 900
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 500
_PERMISSION_MODES = {"default", "require_approval", "bypass"}

# Feature H / Slice H.2 — env-var precedence surface.
#
# Every configurable spawn/lifecycle dimension has an env-var fallback so
# operators can set a baseline without wiring an explicit arg on every call.
# The chain is ``direct → env → pydantic default`` for SpawnOptions fields
# (via ``apply_spawn_env_defaults``) and ``direct → env → ""`` for the
# lifecycle scalars (description, capability, backend_name, cwd).
#
# Naming: ``CLAUDE_TEAMS_DEFAULT_*`` for the new fallbacks; existing
# ``CLAUDE_TEAMS_PERMISSION_MODE`` and ``CLAUDE_TEAMS_CAPABILITY`` keep
# their pre-rename names to preserve the current public surface. All keys
# in this namespace move together when the ``ac-teams`` rename lands.
_ENV_CAPABILITY = "CLAUDE_TEAMS_CAPABILITY"
_ENV_PERMISSION_MODE = "CLAUDE_TEAMS_PERMISSION_MODE"
_ENV_DEFAULT_CWD = "CLAUDE_TEAMS_DEFAULT_CWD"
_ENV_DEFAULT_MODEL = "CLAUDE_TEAMS_DEFAULT_MODEL"
_ENV_DEFAULT_BACKEND = "CLAUDE_TEAMS_DEFAULT_BACKEND"
_ENV_DEFAULT_SUBAGENT_TYPE = "CLAUDE_TEAMS_DEFAULT_SUBAGENT_TYPE"
_ENV_DEFAULT_REASONING_EFFORT = "CLAUDE_TEAMS_DEFAULT_REASONING_EFFORT"
_ENV_DEFAULT_AGENT_PROFILE = "CLAUDE_TEAMS_DEFAULT_AGENT_PROFILE"
_ENV_DEFAULT_TEMPLATE = "CLAUDE_TEAMS_DEFAULT_TEMPLATE"
_ENV_DEFAULT_PLAN_MODE_REQUIRED = "CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED"
_ENV_DEFAULT_DESCRIPTION = "CLAUDE_TEAMS_DEFAULT_DESCRIPTION"

_TRUTHY_BOOL_ENV = frozenset({"true", "1", "yes", "on"})
_FALSY_BOOL_ENV = frozenset({"false", "0", "no", "off"})

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
    """Return a validated working directory for a new teammate.

    Precedence chain: ``direct → CLAUDE_TEAMS_DEFAULT_CWD → Path.cwd()``.
    The env fallback lets operators pin a project root without threading
    ``cwd`` through every MCP call; callers that pass a non-empty value
    always win over the env.
    """
    raw = cwd or os.environ.get(_ENV_DEFAULT_CWD, "")
    if not raw:
        return str(Path.cwd())

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise CwdNotAbsoluteError(raw)
    if not candidate.exists():
        raise CwdMissingError(candidate)
    if not candidate.is_dir():
        raise CwdNotDirectoryError(candidate)
    return str(candidate)


def _resolve_permission_mode(
    permission_mode: Literal["default", "require_approval", "bypass"] | None,
) -> Literal["default", "require_approval", "bypass"]:
    raw = permission_mode or os.environ.get(_ENV_PERMISSION_MODE, "default")
    if raw not in _PERMISSION_MODES:
        raise InvalidPermissionModeError(raw, _PERMISSION_MODES)
    return cast(Literal["default", "require_approval", "bypass"], raw)


def _resolve_capability(capability: str) -> str:
    """Return a capability token, falling back to ``CLAUDE_TEAMS_CAPABILITY``.

    Shared by MCP tool handlers and the Typer CLI so both surfaces agree on
    the precedence chain. An empty string after resolution means "no
    capability was supplied" — downstream helpers already treat that as
    absence and attempt session-state auth instead.
    """
    return capability or os.environ.get(_ENV_CAPABILITY, "")


def _resolve_description(description: str) -> str:
    """Return a team description, falling back to ``CLAUDE_TEAMS_DEFAULT_DESCRIPTION``.

    Descriptions are advisory free-form text, so the resolver never raises —
    an absent direct arg and absent env both yield the empty string.
    """
    return description or os.environ.get(_ENV_DEFAULT_DESCRIPTION, "")


def _resolve_backend_name(backend_name: str) -> str:
    """Return a backend name, falling back to ``CLAUDE_TEAMS_DEFAULT_BACKEND``.

    Used by ``list_agents`` where ``backend_name`` is positional but schema-
    wise accepts empty strings. The SpawnOptions ``backend`` field has a
    parallel fallback via :func:`apply_spawn_env_defaults`.
    """
    return backend_name or os.environ.get(_ENV_DEFAULT_BACKEND, "")


def _parse_bool_env(env_var: str, raw: str) -> bool:
    """Coerce an env-var string to bool using the standard truthy/falsy set.

    Accepted (case-insensitive): ``true/1/yes/on`` → True,
    ``false/0/no/off`` → False. Any other value raises
    ``InvalidEnvVarValueError`` so operators see a typed complaint at tool
    entry rather than a silently dropped override.
    """
    lowered = raw.strip().lower()
    if lowered in _TRUTHY_BOOL_ENV:
        return True
    if lowered in _FALSY_BOOL_ENV:
        return False
    raise InvalidEnvVarValueError(
        env_var,
        raw,
        "expected one of true/false/1/0/yes/no/on/off (case-insensitive)",
    )


def apply_spawn_env_defaults(opts: SpawnOptions) -> SpawnOptions:
    """Fill unset ``SpawnOptions`` fields from ``CLAUDE_TEAMS_DEFAULT_*`` env vars.

    Runs before :func:`claude_teams.orchestration.apply_template` so the
    precedence chain is ``direct → env → template default → pydantic
    default``. "Direct" is tracked via ``opts.model_fields_set`` — fields
    the caller explicitly set retain their values even when an env var
    exists for the same dimension.

    The bool field ``plan_mode_required`` is coerced via
    :func:`_parse_bool_env`; a malformed value raises
    :class:`claude_teams.errors.InvalidEnvVarValueError` at resolve time.
    The literal field ``permission_mode`` is validated against the known
    mode set; an unknown value raises
    :class:`claude_teams.errors.InvalidPermissionModeError` (same surface
    as a bad direct arg).

    Args:
        opts: Spawn options payload received from a caller.

    Returns:
        A new ``SpawnOptions`` with env-driven defaults applied where the
        caller did not explicitly set a field, or the original object when
        no env overrides were active (avoids a needless ``model_copy``).

    Raises:
        InvalidPermissionModeError: ``CLAUDE_TEAMS_PERMISSION_MODE`` is
            set to a value outside the known permission-mode set.
        InvalidEnvVarValueError: ``CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED``
            is set to a value the bool parser does not recognize.

    """
    explicit = opts.model_fields_set
    updates: dict[str, object] = {}

    # Plain string fields — empty env value is equivalent to absence.
    string_fields = (
        ("cwd", _ENV_DEFAULT_CWD),
        ("model", _ENV_DEFAULT_MODEL),
        ("backend", _ENV_DEFAULT_BACKEND),
        ("subagent_type", _ENV_DEFAULT_SUBAGENT_TYPE),
        ("capability", _ENV_CAPABILITY),
        ("reasoning_effort", _ENV_DEFAULT_REASONING_EFFORT),
        ("agent_profile", _ENV_DEFAULT_AGENT_PROFILE),
        ("template", _ENV_DEFAULT_TEMPLATE),
    )
    for field, env_var in string_fields:
        if field in explicit:
            continue
        value = os.environ.get(env_var, "")
        if value:
            updates[field] = value

    # Permission mode — validate against the known literal set before stuffing
    # because ``model_copy(update=...)`` bypasses pydantic re-validation.
    if "permission_mode" not in explicit:
        raw_mode = os.environ.get(_ENV_PERMISSION_MODE, "")
        if raw_mode:
            if raw_mode not in _PERMISSION_MODES:
                raise InvalidPermissionModeError(raw_mode, _PERMISSION_MODES)
            updates["permission_mode"] = raw_mode

    # Bool field — coerce via typed parser so operators see a typed error
    # instead of silent truthiness.
    if "plan_mode_required" not in explicit:
        raw_bool = os.environ.get(_ENV_DEFAULT_PLAN_MODE_REQUIRED, "")
        if raw_bool:
            updates["plan_mode_required"] = _parse_bool_env(
                _ENV_DEFAULT_PLAN_MODE_REQUIRED, raw_bool
            )

    return opts.model_copy(update=updates) if updates else opts


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
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
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
mcp.enable(tags={_TAG_BOOTSTRAP}, only=True, components={"tool", "prompt"})


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
    """Resolve the authenticated principal for a request.

    Precedence: attached session > direct capability > env-var capability.
    Session state wins because an attached MCP session carries a principal
    the caller chose deliberately; falling through to env only happens when
    neither the session nor a direct arg supplied a token. Env fallback is
    applied here so every lead-gated and sender-gated tool inherits it
    without each call site resolving explicitly.
    """
    session_principal = await _resolve_session_principal(ctx, team_name)
    if session_principal is not None:
        return session_principal
    resolved = _resolve_capability(capability)
    if resolved:
        return await capabilities.resolve_principal(team_name, resolved)
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
