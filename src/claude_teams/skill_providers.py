"""Custom skills providers for backends not covered by FastMCP built-ins.

FastMCP ships vendor providers for Claude, Codex, Copilot, Cursor, Gemini,
Goose, OpenCode, and VSCode. The table below covers the remaining
backends in the claude-teams registry that follow the ``agentskills.io``
``SKILL.md`` convention but need a non-default root.

``_CUSTOM_PROVIDER_ROOTS`` maps backend name -> ordered list of candidate
skill root directories. ``SkillsDirectoryProvider`` consults roots in
order; earlier entries win on name collision. The built-in
``GooseSkillsProvider`` ships with ``~/.config/agents/skills/``, which is
not the convention Block Goose actually uses, so this module overrides
it with the ``goose`` entry below.
"""

from pathlib import Path
from typing import Literal

from fastmcp.server.providers.skills import SkillsDirectoryProvider

_CUSTOM_PROVIDER_ROOTS: dict[str, list[Path]] = {
    "amp": [
        Path.home() / ".config" / "amp" / "skills",
        Path.home() / ".config" / "agents" / "skills",
        Path.home() / ".claude" / "skills",
    ],
    "auggie": [
        Path.home() / ".augment" / "skills",
        Path.home() / ".claude" / "skills",
        Path.home() / ".agents" / "skills",
    ],
    "coder": [
        Path.home() / ".code" / "skills",
        Path.home() / ".codex" / "skills",
    ],
    "goose": [
        Path.home() / ".agents" / "skills",
        Path.home() / ".goose" / "skills",
        Path.home() / ".claude" / "skills",
    ],
    "kimi": [
        Path.home() / ".kimi" / "skills",
        Path.home() / ".claude" / "skills",
        Path.home() / ".codex" / "skills",
        Path.home() / ".config" / "agents" / "skills",
        Path.home() / ".agents" / "skills",
    ],
    "llxprt": [Path.home() / ".llxprt" / "skills"],
    "qwen": [Path.home() / ".qwen" / "skills"],
    "rovodev": [Path.home() / ".rovodev" / "skills"],
    "vibe": [Path.home() / ".vibe" / "skills"],
}


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
