"""Server assembly entrypoint for the FastMCP app."""

import os
import signal

from claude_teams.server_bootstrap import register_bootstrap_tools
from claude_teams.server_runtime import mcp
from claude_teams.server_team_spawn import register_team_spawn_tools
from claude_teams.server_team_tasks import register_team_task_tools
from claude_teams.server_teammate import register_teammate_tools

register_bootstrap_tools(mcp)
register_team_spawn_tools(mcp)
register_team_task_tools(mcp)
register_teammate_tools(mcp)


def main():
    """Entry point for the claude-teams MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


if __name__ == "__main__":
    main()
