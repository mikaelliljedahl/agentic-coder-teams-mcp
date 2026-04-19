"""Server assembly entrypoint for the FastMCP app."""

import os
import signal
from pathlib import Path

from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.providers.skills import (
    ClaudeSkillsProvider,
    CodexSkillsProvider,
    CopilotSkillsProvider,
    GeminiSkillsProvider,
    OpenCodeSkillsProvider,
    SkillsDirectoryProvider,
)

from claude_teams.server_bootstrap import register_bootstrap_tools
from claude_teams.server_prompts import register_prompts
from claude_teams.server_runtime import mcp
from claude_teams.server_team_spawn import register_team_spawn_tools
from claude_teams.server_team_tasks import register_team_task_tools
from claude_teams.server_teammate import register_teammate_tools
from claude_teams.skill_providers import build_custom_skills_providers
from claude_teams.telemetry import configure_tracing

register_bootstrap_tools(mcp)
register_team_spawn_tools(mcp)
register_team_task_tools(mcp)
register_teammate_tools(mcp)
register_prompts(mcp)

# Expose bundled skills as MCP resources (skill:// URIs). Any connected
# backend can discover and read these without filesystem installation.
_skills_dir = Path(__file__).parent / "skills"
mcp.add_provider(SkillsDirectoryProvider(roots=_skills_dir))

# Surface each backend's home-dir skill directories so user-installed
# ``SKILL.md`` files also appear as ``skill://`` resources inside this
# MCP session. Five FastMCP built-ins cover the backends whose path
# convention matches the SDK defaults (ClaudeSkillsProvider also serves
# the Claude-compatible ``claudish`` and ``happy`` backends since they
# read from ``~/.claude/skills``). The remaining nine come from the
# custom table in ``claude_teams.skill_providers``. The only
# claude-teams backend without a provider today is ``aider``, which has
# no canonical ``SKILL.md`` convention.
for _builtin_cls, _namespace in (
    (ClaudeSkillsProvider, "claude"),
    (CodexSkillsProvider, "codex"),
    (CopilotSkillsProvider, "copilot"),
    (GeminiSkillsProvider, "gemini"),
    (OpenCodeSkillsProvider, "opencode"),
):
    mcp.add_provider(_builtin_cls(), namespace=_namespace)

for _backend, _provider in build_custom_skills_providers().items():
    mcp.add_provider(_provider, namespace=_backend)

mcp.enable(components={"resource"})

# Middleware chain (outer → inner). Registration order is outer-first, so
# TimingMiddleware wraps everything (so its timings include everything the
# logging layer records) and LoggingMiddleware sits closer to tool handlers.
#
# ``ResponseCachingMiddleware`` is intentionally **not** registered. An
# earlier iteration scoped it to ``list_backends`` under the assumption that
# the backend registry is stable within a process, but ``is_available``
# consults the filesystem/subprocess state live — a user installing a new
# backend binary mid-session would be invisible until the 5-minute TTL
# expired. Discovery tools must return current truth, so freshness wins over
# the microseconds saved on a non-hot path.
mcp.add_middleware(TimingMiddleware())
mcp.add_middleware(LoggingMiddleware())


def main():
    """Entry point for the claude-teams MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    configure_tracing()
    mcp.run()


if __name__ == "__main__":
    main()
