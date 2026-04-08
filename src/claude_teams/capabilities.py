"""Capability token storage and resolution helpers."""

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from claude_teams.async_utils import run_blocking
from claude_teams import teams
from claude_teams.filelock import file_lock


class Principal(TypedDict):
    """Resolved authenticated identity for a lead or teammate."""

    role: Literal["lead", "agent"]
    name: str


class LeadCapabilityRecord(TypedDict):
    """Stored lead capability metadata."""

    name: str
    hash: str


class CapabilityStore(TypedDict):
    """Serialized capability store for a team."""

    lead: LeadCapabilityRecord
    agents: dict[str, str]


def _capabilities_path(team_name: str, base_dir: Path | None = None) -> Path:
    safe_team_name = teams.validate_safe_name(team_name, "team name")
    return teams._teams_dir(base_dir) / safe_team_name / ".capabilities.json"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _read_capabilities(team_name: str, base_dir: Path | None = None) -> CapabilityStore:
    path = _capabilities_path(team_name, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Capability store not found for team {team_name!r}")
    return json.loads(path.read_text())


def _write_capabilities(
    team_name: str, data: CapabilityStore, base_dir: Path | None = None
) -> None:
    path = _capabilities_path(team_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".capabilities.lock"

    with file_lock(lock_path):
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, json.dumps(data, indent=2).encode())
            os.close(fd)
            fd = -1
            os.replace(tmp_path, path)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def _initialize_team_capabilities(team_name: str, base_dir: Path | None = None) -> str:
    """Create the initial lead capability for a team.

    Args:
        team_name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Raw lead capability token.

    """
    lead_capability = secrets.token_urlsafe(32)
    data: CapabilityStore = {
        "lead": {"name": "team-lead", "hash": _hash_token(lead_capability)},
        "agents": {},
    }
    _write_capabilities(team_name, data, base_dir)
    return lead_capability


async def initialize_team_capabilities(
    team_name: str, base_dir: Path | None = None
) -> str:
    """Create the initial lead capability in a worker thread.

    Args:
        team_name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Raw lead capability token.

    """
    return await run_blocking(_initialize_team_capabilities, team_name, base_dir)


def _issue_agent_capability(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> str:
    """Issue a capability token for a teammate.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Raw agent capability token.

    """
    teams.validate_safe_name(agent_name, "agent name")
    capability = secrets.token_urlsafe(32)
    data = _read_capabilities(team_name, base_dir)
    data.setdefault("agents", {})[agent_name] = _hash_token(capability)
    _write_capabilities(team_name, data, base_dir)
    return capability


async def issue_agent_capability(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> str:
    """Issue an agent capability in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Raw agent capability token.

    """
    return await run_blocking(_issue_agent_capability, team_name, agent_name, base_dir)


def _remove_agent_capability(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Remove a teammate capability from the team store.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    """
    teams.validate_safe_name(agent_name, "agent name")
    data = _read_capabilities(team_name, base_dir)
    data.setdefault("agents", {}).pop(agent_name, None)
    _write_capabilities(team_name, data, base_dir)


async def remove_agent_capability(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Remove an agent capability in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_remove_agent_capability, team_name, agent_name, base_dir)


def _resolve_principal(
    team_name: str, capability: str, base_dir: Path | None = None
) -> Principal | None:
    """Resolve a raw capability token to a principal.

    Args:
        team_name (str): Team name.
        capability (str): Raw lead or agent capability token.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Principal | None: Resolved principal, if the token is valid.

    """
    if not capability:
        return None

    token_hash = _hash_token(capability)
    try:
        data = _read_capabilities(team_name, base_dir)
    except FileNotFoundError:
        return None

    lead = data.get("lead", {})
    if token_hash == lead.get("hash"):
        return {"role": "lead", "name": "team-lead"}

    for agent_name, stored_hash in data.get("agents", {}).items():
        if token_hash == stored_hash:
            return {"role": "agent", "name": agent_name}

    return None


async def resolve_principal(
    team_name: str, capability: str, base_dir: Path | None = None
) -> Principal | None:
    """Resolve a capability token to a principal in a worker thread.

    Args:
        team_name (str): Team name.
        capability (str): Raw lead or agent capability token.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Principal | None: Resolved principal, if the token is valid.

    """
    return await run_blocking(_resolve_principal, team_name, capability, base_dir)
