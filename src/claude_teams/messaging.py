"""Mailbox persistence helpers for team messaging."""

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from claude_teams.async_utils import run_blocking
from claude_teams.errors import (
    InvalidInboxOffsetError,
    InvalidInboxOrderError,
    TaskAssignmentNoOwnerError,
)
from claude_teams.filelock import file_lock
from claude_teams.inbox_crypto import decrypt_entry, encrypt_entry
from claude_teams.models import (
    InboxMessage,
    ShutdownRequest,
    TaskAssignment,
    TaskFile,
)
from claude_teams.teams import validate_safe_name

TEAMS_DIR = Path.home() / ".claude" / "teams"

_INBOX_MAX_MESSAGES = 500


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


def _compact_messages(messages: list[InboxMessage]) -> list[InboxMessage]:
    """Drop oldest read messages when the inbox exceeds the retention cap.

    Unread messages are preserved in all cases — they represent work the
    recipient hasn't acknowledged, so dropping them would silently lose
    signal. Read messages are replayable history and are evicted FIFO to
    keep each inbox file bounded and avoid O(N) rewrite costs growing
    without limit.
    """
    if len(messages) <= _INBOX_MAX_MESSAGES:
        return messages
    unread_count = sum(1 for m in messages if not m.read)
    if unread_count >= _INBOX_MAX_MESSAGES:
        return [m for m in messages if not m.read]
    keep_read = _INBOX_MAX_MESSAGES - unread_count
    read_indices = [i for i, m in enumerate(messages) if m.read]
    kept_read = set(read_indices[-keep_read:]) if keep_read else set()
    return [m for i, m in enumerate(messages) if not m.read or i in kept_read]


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with millisecond precision.

    Returns:
        str: ISO timestamp string (e.g., "2024-01-15T14:30:45.123Z").

    """
    dt = datetime.now(UTC)
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
    safe_team_name = validate_safe_name(team_name, "team name")
    safe_agent_name = validate_safe_name(agent_name, "agent name")
    return _teams_dir(base_dir) / safe_team_name / "inboxes" / f"{safe_agent_name}.json"


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


def _select_inbox_indices(
    messages: Sequence[InboxMessage],
    unread_only: bool,
    order: Literal["oldest", "newest"],
) -> list[int]:
    selected_indices = [
        index
        for index, message in enumerate(messages)
        if not unread_only or not message.read
    ]
    if order == "newest":
        selected_indices.reverse()
    return selected_indices


def _apply_pagination(
    selected_indices: list[int], limit: int | None, offset: int
) -> list[int]:
    paged_indices = selected_indices[offset:] if offset else list(selected_indices)
    if limit is not None:
        paged_indices = paged_indices[:limit]
    return paged_indices


def _read_inbox_page(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: int | None = None,
    offset: int = 0,
    order: Literal["oldest", "newest"] = "oldest",
    base_dir: Path | None = None,
) -> tuple[list[InboxMessage], int]:
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return [], 0
    if offset < 0:
        raise InvalidInboxOffsetError()
    if order not in {"oldest", "newest"}:
        raise InvalidInboxOrderError()

    lock_path = path.parent / ".lock"
    with file_lock(lock_path):
        all_msgs = _load_inbox_messages(path, team_name)
        selected_indices = _select_inbox_indices(all_msgs, unread_only, order)
        total_count = len(selected_indices)
        paged_indices = _apply_pagination(selected_indices, limit, offset)
        result = [all_msgs[index] for index in paged_indices]

        if mark_as_read and result:
            for index in paged_indices:
                all_msgs[index].read = True
            path.write_text(json.dumps(_serialize_inbox_messages(team_name, all_msgs)))

        return result, total_count


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
    messages, _total_count = _read_inbox_page(
        team_name=team_name,
        agent_name=agent_name,
        unread_only=unread_only,
        mark_as_read=mark_as_read,
        limit=limit,
        offset=offset,
        order=order,
        base_dir=base_dir,
    )
    return messages


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


async def read_inbox_page(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    limit: int | None = None,
    offset: int = 0,
    order: Literal["oldest", "newest"] = "oldest",
    base_dir: Path | None = None,
) -> tuple[list[InboxMessage], int]:
    """Read inbox messages and total count in a single worker-thread pass.

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
        tuple[list[InboxMessage], int]: Page of messages and total count
            before pagination.

    """
    return await run_blocking(
        _read_inbox_page,
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
        if mark_as_read and result:
            for index in selected_indices:
                all_msgs[index].read = True
            path.write_text(json.dumps(_serialize_inbox_messages(team_name, all_msgs)))

        return result


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
        all_msgs = _compact_messages(all_msgs)
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
        raise TaskAssignmentNoOwnerError()
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
