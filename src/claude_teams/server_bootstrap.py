"""Bootstrap-tier MCP tools for team lifecycle management."""

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, teams
from claude_teams.models import BackendInfo, TeamAttachResult, TeamCreateResult
from claude_teams.server_runtime import (
    _ANN_ATTACH,
    _ANN_CREATE,
    _ANN_DELETE,
    _ANN_READ,
    _TAG_BOOTSTRAP,
    _TAG_TEAM,
    _TAG_TEAMMATE,
    _get_lifespan,
    _require_lead,
    _set_session_principal,
)


async def team_create(
    team_name: str,
    ctx: Context,
    description: str = "",
) -> dict:
    """Create a team and return the lead capability for this session."""
    ls = _get_lifespan(ctx)
    if ls.get("active_team"):
        raise ToolError(
            f"Session already has active team: {ls['active_team']}. One team per session."
        )
    result = await teams.create_team(
        name=team_name, session_id=ls["session_id"], description=description
    )
    lead_capability = await capabilities.initialize_team_capabilities(team_name)
    await _set_session_principal(
        ls,
        team_name,
        "team-lead",
        "lead",
        lead_capability=lead_capability,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool"})
    return TeamCreateResult(
        team_name=result.team_name,
        team_file_path=result.team_file_path,
        lead_agent_id=result.lead_agent_id,
        lead_capability=lead_capability,
    ).model_dump()


async def team_attach(team_name: str, capability: str, ctx: Context) -> dict:
    """Attach this MCP session to an existing team as lead or agent."""
    principal = await capabilities.resolve_principal(team_name, capability)
    if principal is None:
        raise ToolError("Invalid capability for team attachment.")

    ls = _get_lifespan(ctx)
    if ls.get("active_team") and ls["active_team"] != team_name:
        raise ToolError(
            f"Session already has active team: {ls['active_team']}. One team per session."
        )

    await _set_session_principal(
        ls,
        team_name,
        principal["name"],
        principal["role"],
        lead_capability=capability if principal["role"] == "lead" else None,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool"})
    if ls["has_teammates"]:
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    return TeamAttachResult(
        team_name=team_name,
        principal_name=principal["name"],
        principal_role=principal["role"],
    ).model_dump()


async def team_delete(team_name: str, ctx: Context, capability: str = "") -> dict:
    """Delete a team and its files. Fails if teammates are still present."""
    await _require_lead(ctx, team_name, capability)
    try:
        result = await teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e))
    ls = _get_lifespan(ctx)
    ls["active_team"] = None
    ls["has_teammates"] = False
    ls["principal_name"] = None
    ls["principal_role"] = None
    ls["lead_capability"] = None
    await ctx.disable_components(tags={_TAG_TEAM, _TAG_TEAMMATE}, components={"tool"})
    return result.model_dump()


async def read_config(team_name: str, ctx: Context, capability: str = "") -> dict:
    """Read the current team configuration including all members."""
    await _require_lead(ctx, team_name, capability)
    try:
        config = await teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")
    return config.model_dump(by_alias=True)


def list_backends(ctx: Context) -> list[dict]:
    """List all available spawner backends with their supported models."""
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    result = []
    for name, backend_obj in reg:
        info = BackendInfo(
            name=name,
            binary=backend_obj.binary_name,
            available=True,
            default_model=backend_obj.default_model(),
            supported_models=backend_obj.supported_models(),
        )
        result.append(info.model_dump(by_alias=True))
    return result


def register_bootstrap_tools(mcp: FastMCP) -> None:
    """Register bootstrap-tier tools on the FastMCP app."""
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_CREATE)(team_create)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_ATTACH)(team_attach)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_DELETE)(team_delete)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(read_config)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_backends)
