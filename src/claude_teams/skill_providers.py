"""Custom skills providers for supported Windows-native backends.

This fork supports Claude Code and Codex as first-class backends. FastMCP
already provides built-in skill providers for those ecosystems, so no
additional vendor-specific roots are registered here.
"""

from pathlib import Path
from typing import Literal

from fastmcp.server.providers.skills import SkillsDirectoryProvider

_CUSTOM_PROVIDER_ROOTS: dict[str, list[Path]] = {}


def build_custom_skills_providers(
    *,
    reload: bool = False,
    supporting_files: Literal["template", "resources"] = "template",
) -> dict[str, SkillsDirectoryProvider]:
    """Construct one ``SkillsDirectoryProvider`` per backend in the table.

    Callers typically iterate the returned mapping and register each
    provider with the running FastMCP server using a matching namespace::

        for name, provider in build_custom_skills_providers().items():
            mcp.add_provider(provider, namespace=name)

    Args:
        reload: Forwarded to ``SkillsDirectoryProvider``; when ``True``,
            skills are re-scanned on every request.
        supporting_files: Whether supporting files next to a ``SKILL.md``
            are surfaced as MCP ``template`` or ``resources`` components.

    Returns:
        Ordered dict mapping backend name to a ready-to-register provider.
    """
    return {
        name: SkillsDirectoryProvider(
            roots=roots,
            reload=reload,
            main_file_name="SKILL.md",
            supporting_files=supporting_files,
        )
        for name, roots in _CUSTOM_PROVIDER_ROOTS.items()
    }
