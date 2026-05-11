"""Server entry point — delegates to the simplified server."""

from claude_teams.server_simple import main, mcp

__all__ = ["main", "mcp"]

if __name__ == "__main__":
    main()
