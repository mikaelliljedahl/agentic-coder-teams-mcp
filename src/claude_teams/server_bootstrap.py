"""Bootstrap-tier MCP tools for team lifecycle management."""

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, teams
from claude_teams.errors import (
    BackendNotRegisteredError,
    InvalidCapabilityError,
    SessionActiveTeamError,
    TeamNotFoundToolError,
)
from claude_teams.models import (
    AgentListResult,
    AgentProfileInfo,
    BackendInfo,
    TeamAttachResult,
    TeamCreateResult,
)
from claude_teams.server_runtime import (
    _ANN_ATTACH,
    _ANN_CREATE,
    _ANN_DELETE,
    _ANN_READ,
    _TAG_BOOTSTRAP,
    _TAG_TEAM,
    _TAG_TEAMMATE,
    _clear_session_principal,
    _get_lifespan,
    _require_lead,
    _resolve_spawn_cwd,
    _set_session_principal,
)
from claude_teams.server_schema import (
    BackendName,
    Capability,
    Cwd,
    Description,
    TeamName,
)


async def team_create(
    team_name: TeamName,
    ctx: Context,
    description: Description = "",
) -> dict[str, object]:
    """Create a team and return the lead capability for this session."""
    active_team = await ctx.get_state("active_team")
    if active_team:
        raise SessionActiveTeamError(active_team)
    result = await teams.create_team(
        name=team_name, session_id=ctx.session_id, description=description
    )
    lead_capability = await capabilities.initialize_team_capabilities(team_name)
    await _set_session_principal(
        ctx,
        team_name,
        "team-lead",
        "lead",
        lead_capability=lead_capability,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool", "prompt"})
    return TeamCreateResult(
        team_name=result.team_name,
        team_file_path=result.team_file_path,
        lead_agent_id=result.lead_agent_id,
        lead_capability=lead_capability,
    ).model_dump()


async def team_attach(
    team_name: TeamName,
    capability: Capability,
    ctx: Context,
) -> dict[str, object]:
    """Attach this MCP session to an existing team as lead or agent."""
    principal = await capabilities.resolve_principal(team_name, capability)
    if principal is None:
        raise InvalidCapabilityError()

    active_team = await ctx.get_state("active_team")
    if active_team and active_team != team_name:
        raise SessionActiveTeamError(active_team)

    await _set_session_principal(
        ctx,
        team_name,
        principal["name"],
        principal["role"],
        lead_capability=capability if principal["role"] == "lead" else None,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool", "prompt"})
    if await ctx.get_state("has_teammates"):
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    return TeamAttachResult(
        team_name=team_name,
        principal_name=principal["name"],
        principal_role=principal["role"],
    ).model_dump()


async def team_delete(
    team_name: TeamName, ctx: Context, capability: Capability = ""
) -> dict[str, object]:
    """Delete a team and its files. Fails if teammates are still present."""
    await _require_lead(ctx, team_name, capability)
    try:
        result = await teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e
    await _clear_session_principal(ctx)
    await ctx.disable_components(
        tags={_TAG_TEAM, _TAG_TEAMMATE}, components={"tool", "prompt"}
    )
    return result.model_dump()


async def read_config(
    team_name: TeamName, ctx: Context, capability: Capability = ""
) -> dict[str, object]:
    """Read the current team configuration including all members."""
    await _require_lead(ctx, team_name, capability)
    try:
        config = await teams.read_config(team_name)
    except FileNotFoundError:
        raise TeamNotFoundToolError(team_name) from None
    return config.model_dump(by_alias=True)


def list_backends(ctx: Context) -> list[dict[str, object]]:
    """List all available spawner backends with their supported models."""
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    result = []
    for name, backend_obj in reg:
        info = BackendInfo(
            name=name,
            binary=backend_obj.binary_name,
            available=backend_obj.is_available(),
            default_model=backend_obj.default_model(),
            supported_models=backend_obj.supported_models(),
        )
        result.append(info.model_dump(by_alias=True))
    return result


def list_agents(
    backend_name: BackendName,
    ctx: Context,
    cwd: Cwd = "",
) -> dict[str, object]:
    """List discoverable agent/persona profiles for a backend.

    Returns ``supported=False`` when the backend has no agent-selection
    mechanism; otherwise enumerates profiles visible from ``cwd`` (empty
    string resolves to the server's working directory).
    """
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    try:
        backend_obj = reg.get(backend_name)
    except BackendNotRegisteredError as exc:
        raise ToolError(str(exc)) from exc

    resolved_cwd = str(_resolve_spawn_cwd(cwd))

    spec = backend_obj.agent_select_spec()
    if spec is None:
        return AgentListResult(
            backend=backend_name,
            supported=False,
            cwd=resolved_cwd,
            profiles=[],
        ).model_dump(by_alias=True)

    profiles = backend_obj.discover_agents(resolved_cwd)
    return AgentListResult(
        backend=backend_name,
        supported=True,
        cwd=resolved_cwd,
        profiles=[AgentProfileInfo(name=p.name, path=p.path) for p in profiles],
    ).model_dump(by_alias=True)


def register_bootstrap_tools(mcp: FastMCP) -> None:
    """Register bootstrap-tier tools on the FastMCP app."""
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_CREATE)(team_create)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_ATTACH)(team_attach)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_DELETE)(team_delete)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(read_config)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_backends)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_agents)
