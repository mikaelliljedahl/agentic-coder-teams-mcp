"""Shared agent/profile discovery helpers used by supporting backends.

Three mechanisms are covered:

* ``discover_claude_agents`` — filesystem scan of ``.claude/agents/*.md``
  under the project root and ``~/.claude/agents``. Used by ``claude-code``
  and ``claudish``.
* ``discover_codex_style_agents`` — parses the ``[agents.<name>]`` tables
  in a ``config.toml`` under a backend-specific dot-directory (``.codex``,
  ``.coder``). Used by ``codex`` and ``coder``.
* ``discover_goose_recipes`` — walks directories listed in the
  ``GOOSE_RECIPE_PATH`` environment variable for ``*.yaml`` / ``*.yml`` /
  ``*.json`` recipe files. Used by ``goose``.

In every case, name collisions resolve in favor of the first entry seen,
which lets project-local sources shadow user-global ones.
"""

import os
import tomllib
from pathlib import Path

from claude_teams.backends.contracts import AgentProfile


def discover_claude_agents(cwd: str) -> list[AgentProfile]:
    """Enumerate Claude-style agent markdown files.

    Walks ``<cwd>/.claude/agents`` first (project-local) then
    ``~/.claude/agents`` (user-global). Only ``*.md`` files are yielded;
    the stem becomes the profile name.

    Args:
        cwd: Absolute path anchoring project-local discovery.

    Returns:
        Discovered profiles, project entries shadowing home entries.

    """
    roots = [
        Path(cwd) / ".claude" / "agents",
        Path.home() / ".claude" / "agents",
    ]
    profiles: dict[str, AgentProfile] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            profiles.setdefault(path.stem, AgentProfile(name=path.stem, path=str(path)))
    return list(profiles.values())


def discover_codex_style_agents(cwd: str, config_dir: str) -> list[AgentProfile]:
    """Parse ``agents.<name>`` tables from a Codex-style ``config.toml``.

    Reads ``~/.{config_dir}/config.toml`` (user-global) and
    ``<cwd>/.{config_dir}/config.toml`` (project-local). Only entries with
    a non-empty ``config_file`` string attribute are yielded; agents that
    exist in TOML but lack a persona file cannot be selected through the
    ``-c 'agents.<name>.config_file="<path>"'`` override mechanism.

    Args:
        cwd: Absolute path anchoring project-local discovery.
        config_dir: Dot-directory name (``codex`` → ``.codex``).

    Returns:
        Discovered profiles. Later entries (project-local) overwrite
        earlier ones on name collision so project config wins.

    """
    config_paths = [
        Path.home() / f".{config_dir}" / "config.toml",
        Path(cwd) / f".{config_dir}" / "config.toml",
    ]
    profiles: dict[str, AgentProfile] = {}
    for config_path in config_paths:
        if not config_path.is_file():
            continue
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        raw_agents = data.get("agents")
        if not isinstance(raw_agents, dict):
            continue
        for name, entry in raw_agents.items():
            if not isinstance(entry, dict):
                continue
            config_file = entry.get("config_file")
            if not isinstance(config_file, str) or not config_file:
                continue
            profiles[name] = AgentProfile(name=name, path=config_file)
    return list(profiles.values())


def discover_goose_recipes(cwd: str) -> list[AgentProfile]:
    """Walk ``$GOOSE_RECIPE_PATH`` for recipe files.

    ``cwd`` is intentionally unused: Goose recipe discovery is env-var
    driven so that users opt in by pointing ``$GOOSE_RECIPE_PATH`` at one
    or more directories (separated by ``os.pathsep``) holding recipe
    ``*.yaml`` / ``*.yml`` / ``*.json`` files.

    Args:
        cwd: Ignored; kept to match the Backend discovery signature.

    Returns:
        Discovered recipes. Directories are scanned in the order listed in
        the env var; earlier entries win on name collision.

    """
    _ = cwd
    path_var = os.environ.get("GOOSE_RECIPE_PATH", "")
    if not path_var:
        return []
    profiles: dict[str, AgentProfile] = {}
    for dir_str in path_var.split(os.pathsep):
        if not dir_str:
            continue
        directory = Path(dir_str)
        if not directory.is_dir():
            continue
        for pattern in ("*.yaml", "*.yml", "*.json"):
            for path in sorted(directory.glob(pattern)):
                profiles.setdefault(
                    path.stem, AgentProfile(name=path.stem, path=str(path))
                )
    return list(profiles.values())
