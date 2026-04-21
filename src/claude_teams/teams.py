"""Team configuration persistence helpers."""

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from claude_teams.async_utils import run_blocking
from claude_teams.errors import (
    CannotRemoveLeadError,
    InvalidNameError,
    MemberAlreadyExistsError,
    NameTooLongError,
    TeamAlreadyExistsError,
    TeamHasMembersError,
)
from claude_teams.models import (
    LeadMember,
    TeamConfig,
    TeamCreateResult,
    TeamDeleteResult,
    TeammateMember,
)

CLAUDE_DIR = Path.home() / ".claude"
TEAMS_DIR = CLAUDE_DIR / "teams"
TASKS_DIR = CLAUDE_DIR / "tasks"

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_NAME_LEN = 64


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


def _tasks_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "tasks") if base_dir else TASKS_DIR


def validate_safe_name(name: str, label: str = "name") -> str:
    """Validate a filesystem-safe team or agent identifier.

    Args:
        name (str): Identifier to validate.
        label (str): Human-readable field label for error messages.

    Returns:
        str: The validated name.

    Raises:
        ValueError: If the name is empty, too long, or contains unsafe characters.

    """
    if not _VALID_NAME_RE.match(name):
        raise InvalidNameError(label, name)
    if len(name) > _MAX_NAME_LEN:
        raise NameTooLongError(label, name, _MAX_NAME_LEN)
    return name


def _team_exists(name: str, base_dir: Path | None = None) -> bool:
    """Check if a team configuration file exists.

    Args:
        name (str): Name of the team.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        bool: True if the team's config.json exists.

    """
    safe_name = validate_safe_name(name, "team name")
    config_path = _teams_dir(base_dir) / safe_name / "config.json"
    return config_path.exists()


async def team_exists(name: str, base_dir: Path | None = None) -> bool:
    """Check whether a team exists.

    Args:
        name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        bool: True when the team config exists.

    """
    return await run_blocking(_team_exists, name, base_dir)


def _create_team(
    name: str,
    session_id: str,
    description: str = "",
    lead_model: str = "claude-opus-4-7",
    base_dir: Path | None = None,
) -> TeamCreateResult:
    """Create a new team directory structure and configuration file.

    Args:
        name (str): Name of the team (alphanumeric, hyphens, underscores only).
        session_id (str): Session ID for the team lead.
        description (str): Optional team description.
        lead_model (str): Model to use for the team lead agent.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamCreateResult: Information about the created team.

    Raises:
        ValueError: If team name is invalid or exceeds 64 characters.
        TeamAlreadyExistsError: If a team config already exists at the
            target name. Caught and converted to
            ``TeamAlreadyExistsToolError`` at the MCP boundary;
            propagated as ValueError to CLI callers.

    """
    validate_safe_name(name, "team name")

    teams_dir = _teams_dir(base_dir)
    tasks_dir = _tasks_dir(base_dir)

    team_dir = teams_dir / name
    config_path = team_dir / "config.json"
    team_dir.mkdir(parents=True, exist_ok=True)

    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / ".lock").touch()

    now_ms = int(time.time() * 1000)

    lead = LeadMember(
        agent_id=f"team-lead@{name}",
        name="team-lead",
        agent_type="team-lead",
        model=lead_model,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=str(Path.cwd()),
    )

    config = TeamConfig(
        name=name,
        description=description,
        created_at=now_ms,
        lead_agent_id=f"team-lead@{name}",
        lead_session_id=session_id,
        members=[lead],
    )

    # config.json exclusive-create is the commit point; task_dir setup above
    # is idempotent so a pre-commit failure leaves no "team exists" state.
    payload = json.dumps(config.model_dump(by_alias=True), indent=2)
    try:
        with config_path.open("x") as f:
            f.write(payload)
    except FileExistsError:
        raise TeamAlreadyExistsError(name) from None

    return TeamCreateResult(
        team_name=name,
        team_file_path=str(config_path),
        lead_agent_id=f"team-lead@{name}",
    )


async def create_team(
    name: str,
    session_id: str,
    description: str = "",
    lead_model: str = "claude-opus-4-7",
    base_dir: Path | None = None,
) -> TeamCreateResult:
    """Create a team in a worker thread.

    Args:
        name (str): Team name.
        session_id (str): Lead session identifier.
        description (str): Team description.
        lead_model (str): Lead model name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamCreateResult: Creation result payload.

    """
    return await run_blocking(
        _create_team,
        name,
        session_id,
        description,
        lead_model,
        base_dir,
    )


def _read_config(name: str, base_dir: Path | None = None) -> TeamConfig:
    """Read and parse a team's configuration file.

    Args:
        name (str): Name of the team.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamConfig: Parsed team configuration object.

    Raises:
        FileNotFoundError: If the team config does not exist.
        json.JSONDecodeError: If the config file is malformed.

    """
    safe_name = validate_safe_name(name, "team name")
    config_path = _teams_dir(base_dir) / safe_name / "config.json"
    raw = json.loads(config_path.read_text())
    return TeamConfig.model_validate(raw)


async def read_config(name: str, base_dir: Path | None = None) -> TeamConfig:
    """Read a team config in a worker thread.

    Args:
        name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamConfig: Parsed team configuration.

    """
    return await run_blocking(_read_config, name, base_dir)


def _write_config(name: str, config: TeamConfig, base_dir: Path | None = None) -> None:
    """Write team configuration to disk atomically.

    Args:
        name (str): Name of the team.
        config (TeamConfig): Configuration object to serialize and write.
        base_dir (Path | None): Override for the base config directory.

    Raises:
        OSError: If file creation or write fails.

    """
    safe_name = validate_safe_name(name, "team name")
    config_dir = _teams_dir(base_dir) / safe_name
    data = json.dumps(config.model_dump(by_alias=True), indent=2)

    # NOTE(victor): atomic write to avoid partial reads from concurrent agents
    fd, tmp_name = tempfile.mkstemp(dir=config_dir, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.write(fd, data.encode())
        os.close(fd)
        fd = -1
        tmp_path.replace(config_dir / "config.json")
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def write_config(
    name: str, config: TeamConfig, base_dir: Path | None = None
) -> None:
    """Write a team config in a worker thread.

    Args:
        name (str): Team name.
        config (TeamConfig): Configuration payload.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_write_config, name, config, base_dir)


def _delete_team(name: str, base_dir: Path | None = None) -> TeamDeleteResult:
    """Delete a team's directories and configuration after validation.

    Args:
        name (str): Name of the team to delete.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamDeleteResult: Result containing success status and cleanup message.

    Raises:
        RuntimeError: If non-lead members are still present in the team.
        FileNotFoundError: If the team does not exist.

    """
    safe_name = validate_safe_name(name, "team name")
    config = _read_config(safe_name, base_dir=base_dir)

    non_lead = [
        member for member in config.members if isinstance(member, TeammateMember)
    ]
    if non_lead:
        raise TeamHasMembersError(safe_name, len(non_lead))

    shutil.rmtree(_teams_dir(base_dir) / safe_name)
    shutil.rmtree(_tasks_dir(base_dir) / safe_name)

    return TeamDeleteResult(
        success=True,
        message=f'Cleaned up directories and worktrees for team "{safe_name}"',
        team_name=safe_name,
    )


async def delete_team(name: str, base_dir: Path | None = None) -> TeamDeleteResult:
    """Delete a team in a worker thread.

    Args:
        name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TeamDeleteResult: Deletion result payload.

    """
    return await run_blocking(_delete_team, name, base_dir)


def _add_member(
    name: str, member: TeammateMember, base_dir: Path | None = None
) -> None:
    """Add a new teammate member to an existing team.

    Args:
        name (str): Name of the team.
        member (TeammateMember): Teammate member object to add.
        base_dir (Path | None): Override for the base config directory.

    Raises:
        ValueError: If a member with the same name already exists in the team.
        FileNotFoundError: If the team does not exist.

    """
    config = _read_config(name, base_dir=base_dir)
    existing_names = {member_item.name for member_item in config.members}
    if member.name in existing_names:
        raise MemberAlreadyExistsError(member.name, name)
    config.members.append(member)
    _write_config(name, config, base_dir=base_dir)


async def add_member(
    name: str, member: TeammateMember, base_dir: Path | None = None
) -> None:
    """Add a teammate member in a worker thread.

    Args:
        name (str): Team name.
        member (TeammateMember): Member payload.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_add_member, name, member, base_dir)


def _remove_member(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Remove a teammate member from a team by agent name.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to remove from the team.
        base_dir (Path | None): Override for the base config directory.

    Raises:
        ValueError: If attempting to remove the team-lead agent.
        FileNotFoundError: If the team does not exist.

    """
    if agent_name == "team-lead":
        raise CannotRemoveLeadError()
    config = _read_config(team_name, base_dir=base_dir)
    config.members = [member for member in config.members if member.name != agent_name]
    _write_config(team_name, config, base_dir=base_dir)


async def remove_member(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Remove a teammate member in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_remove_member, team_name, agent_name, base_dir)
