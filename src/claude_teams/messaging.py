"""Mailbox persistence helpers for team messaging."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from claude_teams.async_utils import run_blocking
from claude_teams.filelock import file_lock
from claude_teams.inbox_crypto import decrypt_entry, encrypt_entry
from claude_teams.models import (
    InboxMessage,
    ShutdownRequest,
    TaskAssignment,
    TaskFile,
)

TEAMS_DIR = Path.home() / ".claude" / "teams"


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with millisecond precision.

    Returns:
        str: ISO timestamp string (e.g., "2024-01-15T14:30:45.123Z").

    """
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def inbox_path(team_name: str, agent_name: str, base_dir: Path | None = None) -> Path:
    """Return the file path to an agent's inbox JSON file.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Path: Full path to the agent's inbox file.

    """
    return _teams_dir(base_dir) / team_name / "inboxes" / f"{agent_name}.json"


def _ensure_inbox(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> Path:
    path = inbox_path(team_name, agent_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]")
    return path


async def ensure_inbox(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> Path:
    """Ensure an inbox exists in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Path: Inbox path.

    """
    return await run_blocking(_ensure_inbox, team_name, agent_name, base_dir)


def _load_inbox_messages(path: Path, team_name: str) -> list[InboxMessage]:
    raw_list = json.loads(path.read_text())
    return [
        InboxMessage.model_validate(decrypt_entry(team_name, entry))
        for entry in raw_list
    ]


def _serialize_inbox_messages(
    team_name: str, messages: list[InboxMessage]
) -> list[dict[str, object]]:
    return [
        encrypt_entry(team_name, msg.model_dump(by_alias=True, exclude_none=True))
        for msg in messages
    ]


def _read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: int | None = None,
    offset: int = 0,
    order: Literal["oldest", "newest"] = "oldest",
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return []
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if order not in {"oldest", "newest"}:
        raise ValueError("order must be 'oldest' or 'newest'")

    if mark_as_read:
        lock_path = path.parent / ".lock"
        with file_lock(lock_path):
            all_msgs = _load_inbox_messages(path, team_name)
            selected_indices = [
                index
                for index, msg in enumerate(all_msgs)
                if not unread_only or not msg.read
            ]
            if order == "newest":
                selected_indices.reverse()
            if offset:
                selected_indices = selected_indices[offset:]
            if limit is not None:
                selected_indices = selected_indices[:limit]
            result = [all_msgs[index] for index in selected_indices]

            if result:
                for index in selected_indices:
                    all_msgs[index].read = True
                path.write_text(
                    json.dumps(_serialize_inbox_messages(team_name, all_msgs))
                )

            return result
    else:
        all_msgs = _load_inbox_messages(path, team_name)
        if unread_only:
            all_msgs = [msg for msg in all_msgs if not msg.read]
        if order == "newest":
            all_msgs = list(reversed(all_msgs))
        if offset:
            all_msgs = all_msgs[offset:]
        if limit is not None:
            all_msgs = all_msgs[:limit]
        return list(all_msgs)


async def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: int | None = None,
    offset: int = 0,
    order: Literal["oldest", "newest"] = "oldest",
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    """Read inbox messages in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        unread_only (bool): Whether to return only unread messages.
        mark_as_read (bool): Whether returned messages should be marked read.
        limit (int | None): Maximum number of messages to return.
        offset (int): Number of messages to skip.
        order (Literal["oldest", "newest"]): Inbox ordering to use.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        list[InboxMessage]: Matching inbox messages.

    """
    return await run_blocking(
        _read_inbox,
        team_name,
        agent_name,
        unread_only,
        mark_as_read,
        limit,
        offset,
        order,
        base_dir,
    )


def _read_inbox_filtered(
    team_name: str,
    agent_name: str,
    sender_filter: str,
    unread_only: bool = True,
    mark_as_read: bool = True,
    limit: int | None = None,
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return []

    if mark_as_read:
        lock_path = path.parent / ".lock"
        with file_lock(lock_path):
            all_msgs = _load_inbox_messages(path, team_name)

            selected_indices = []
            for index, msg in enumerate(all_msgs):
                if msg.from_ != sender_filter:
                    continue
                if unread_only and msg.read:
                    continue
                selected_indices.append(index)

            if limit is not None and len(selected_indices) > limit:
                selected_indices = selected_indices[-limit:]

            result = [all_msgs[index] for index in selected_indices]
            if result:
                for index in selected_indices:
                    all_msgs[index].read = True
                path.write_text(
                    json.dumps(_serialize_inbox_messages(team_name, all_msgs))
                )

            return result

    all_msgs = _load_inbox_messages(path, team_name)
    filtered = [msg for msg in all_msgs if msg.from_ == sender_filter]
    if unread_only:
        filtered = [msg for msg in filtered if not msg.read]
    if limit is not None and len(filtered) > limit:
        filtered = filtered[-limit:]
    return filtered


async def read_inbox_filtered(
    team_name: str,
    agent_name: str,
    sender_filter: str,
    unread_only: bool = True,
    mark_as_read: bool = True,
    limit: int | None = None,
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    """Read sender-filtered inbox messages in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Inbox owner.
        sender_filter (str): Sender name to match.
        unread_only (bool): Whether to return only unread messages.
        mark_as_read (bool): Whether returned messages should be marked read.
        limit (int | None): Maximum number of messages to return.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        list[InboxMessage]: Matching inbox messages.

    """
    return await run_blocking(
        _read_inbox_filtered,
        team_name,
        agent_name,
        sender_filter,
        unread_only,
        mark_as_read,
        limit,
        base_dir,
    )


def _append_message(
    team_name: str,
    agent_name: str,
    message: InboxMessage,
    base_dir: Path | None = None,
) -> None:
    path = _ensure_inbox(team_name, agent_name, base_dir)
    lock_path = path.parent / ".lock"

    with file_lock(lock_path):
        all_msgs = _load_inbox_messages(path, team_name)
        all_msgs.append(message)
        path.write_text(json.dumps(_serialize_inbox_messages(team_name, all_msgs)))


async def append_message(
    team_name: str,
    agent_name: str,
    message: InboxMessage,
    base_dir: Path | None = None,
) -> None:
    """Append a message in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Message recipient.
        message (InboxMessage): Message payload.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_append_message, team_name, agent_name, message, base_dir)


def _send_plain_message(
    team_name: str,
    from_name: str,
    to_name: str,
    text: str,
    summary: str,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    msg = InboxMessage(
        from_=from_name,
        text=text,
        timestamp=now_iso(),
        read=False,
        summary=summary,
        color=color,
    )
    _append_message(team_name, to_name, msg, base_dir)


async def send_plain_message(
    team_name: str,
    from_name: str,
    to_name: str,
    text: str,
    summary: str,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    """Send a plain message in a worker thread.

    Args:
        team_name (str): Team name.
        from_name (str): Sender name.
        to_name (str): Recipient name.
        text (str): Message text.
        summary (str): Message summary.
        color (str | None): Optional color hint.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(
        _send_plain_message,
        team_name,
        from_name,
        to_name,
        text,
        summary,
        color,
        base_dir,
    )


def _send_structured_message(
    team_name: str,
    from_name: str,
    to_name: str,
    payload: BaseModel,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    serialized = payload.model_dump_json(by_alias=True)
    msg = InboxMessage(
        from_=from_name,
        text=serialized,
        timestamp=now_iso(),
        read=False,
        color=color,
    )
    _append_message(team_name, to_name, msg, base_dir)


async def send_structured_message(
    team_name: str,
    from_name: str,
    to_name: str,
    payload: BaseModel,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    """Send a structured message in a worker thread.

    Args:
        team_name (str): Team name.
        from_name (str): Sender name.
        to_name (str): Recipient name.
        payload (BaseModel): Structured payload.
        color (str | None): Optional color hint.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(
        _send_structured_message,
        team_name,
        from_name,
        to_name,
        payload,
        color,
        base_dir,
    )


def _send_task_assignment(
    team_name: str,
    task: TaskFile,
    assigned_by: str,
    base_dir: Path | None = None,
) -> None:
    if task.owner is None:
        raise ValueError("Cannot send task assignment: task has no owner")
    payload = TaskAssignment(
        task_id=task.id,
        subject=task.subject,
        description=task.description,
        assigned_by=assigned_by,
        timestamp=now_iso(),
    )
    _send_structured_message(
        team_name, assigned_by, task.owner, payload, base_dir=base_dir
    )


async def send_task_assignment(
    team_name: str,
    task: TaskFile,
    assigned_by: str,
    base_dir: Path | None = None,
) -> None:
    """Send a task assignment in a worker thread.

    Args:
        team_name (str): Team name.
        task (TaskFile): Task payload.
        assigned_by (str): Assigning principal name.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_send_task_assignment, team_name, task, assigned_by, base_dir)


def _send_shutdown_request(
    team_name: str,
    recipient: str,
    reason: str = "",
    base_dir: Path | None = None,
) -> str:
    request_id = f"shutdown-{int(time.time() * 1000)}@{recipient}"
    payload = ShutdownRequest(
        request_id=request_id,
        from_="team-lead",
        reason=reason,
        timestamp=now_iso(),
    )
    _send_structured_message(
        team_name, "team-lead", recipient, payload, base_dir=base_dir
    )
    return request_id


async def send_shutdown_request(
    team_name: str,
    recipient: str,
    reason: str = "",
    base_dir: Path | None = None,
) -> str:
    """Send a shutdown request in a worker thread.

    Args:
        team_name (str): Team name.
        recipient (str): Target agent.
        reason (str): Optional shutdown reason.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Shutdown request identifier.

    """
    return await run_blocking(
        _send_shutdown_request, team_name, recipient, reason, base_dir
    )
