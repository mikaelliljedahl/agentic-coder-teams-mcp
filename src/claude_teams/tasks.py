"""Task persistence helpers for file-backed team state."""

import json
from collections import deque
from pathlib import Path
from typing import Literal

from claude_teams.async_utils import run_blocking
from claude_teams.errors import (
    BlockedTaskStatusError,
    CyclicTaskBlockedByError,
    CyclicTaskBlockError,
    InvalidTaskStatusError,
    TaskReferenceNotFoundError,
    TaskSelfBlockedByError,
    TaskSelfBlockError,
    TaskStatusRegressionError,
    TaskSubjectEmptyError,
    TeamNotFoundValueError,
)
from claude_teams.filelock import file_lock
from claude_teams.models import TaskFile, TaskMetadata, TaskUpdateFields
from claude_teams.teams import _team_exists, validate_safe_name

_TaskStatus = Literal["pending", "in_progress", "completed", "deleted"]

TASKS_DIR = Path.home() / ".claude" / "tasks"


def _tasks_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "tasks") if base_dir else TASKS_DIR


_STATUS_ORDER = {"pending": 0, "in_progress": 1, "completed": 2}


def _flush_pending_writes(pending_writes: dict[Path, TaskFile]) -> None:
    for task_file, task_obj in pending_writes.items():
        task_file.write_text(
            json.dumps(task_obj.model_dump(by_alias=True, exclude_none=True))
        )


def _would_create_cycle(
    team_dir: Path, from_id: str, to_id: str, pending_edges: dict[str, set[str]]
) -> bool:
    """Return whether adding a dependency edge would create a cycle."""
    visited: set[str] = set()
    queue = deque([to_id])
    while queue:
        current = queue.popleft()
        if current == from_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        fpath = team_dir / f"{current}.json"
        if fpath.exists():
            task = TaskFile(**json.loads(fpath.read_text()))
            queue.extend(dep_id for dep_id in task.blocked_by if dep_id not in visited)
        queue.extend(
            dep_id
            for dep_id in pending_edges.get(current, set())
            if dep_id not in visited
        )
    return False


def next_task_id(team_name: str, base_dir: Path | None = None) -> str:
    """Return the next available integer task ID for a team."""
    safe_team_name = validate_safe_name(team_name, "team name")
    team_dir = _tasks_dir(base_dir) / safe_team_name
    ids: list[int] = []
    for task_file in team_dir.glob("*.json"):
        try:
            ids.append(int(task_file.stem))
        except ValueError:
            continue
    return str(max(ids) + 1) if ids else "1"


def _create_task(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: TaskMetadata | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    safe_team_name = validate_safe_name(team_name, "team name")
    if not subject or not subject.strip():
        raise TaskSubjectEmptyError()
    if not _team_exists(safe_team_name, base_dir):
        raise TeamNotFoundValueError(safe_team_name)
    team_dir = _tasks_dir(base_dir) / safe_team_name
    team_dir.mkdir(parents=True, exist_ok=True)
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        task_id = next_task_id(safe_team_name, base_dir)
        task = TaskFile(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            status="pending",
            metadata=metadata,
        )
        fpath = team_dir / f"{task_id}.json"
        fpath.write_text(json.dumps(task.model_dump(by_alias=True, exclude_none=True)))

    return task


async def create_task(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: TaskMetadata | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    """Create a task in a worker thread.

    Args:
        team_name (str): Team name.
        subject (str): Task subject.
        description (str): Task description.
        active_form (str): Optional active-form text.
        metadata (dict | None): Optional metadata payload.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: Newly created task.

    """
    return await run_blocking(
        _create_task,
        team_name,
        subject,
        description,
        active_form,
        metadata,
        base_dir,
    )


def _get_task(team_name: str, task_id: str, base_dir: Path | None = None) -> TaskFile:
    safe_team_name = validate_safe_name(team_name, "team name")
    team_dir = _tasks_dir(base_dir) / safe_team_name
    fpath = team_dir / f"{task_id}.json"
    raw = json.loads(fpath.read_text())
    return TaskFile(**raw)


async def get_task(
    team_name: str, task_id: str, base_dir: Path | None = None
) -> TaskFile:
    """Read a task in a worker thread.

    Args:
        team_name (str): Team name.
        task_id (str): Task identifier.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: Parsed task payload.

    """
    return await run_blocking(_get_task, team_name, task_id, base_dir)


def _link_dependency(
    task: TaskFile,
    task_id: str,
    dep_ids: list[str],
    forward_field: str,
    inverse_field: str,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
) -> None:
    """Link dependency fields on the task and the affected peer tasks."""
    forward_list: list[str] = getattr(task, forward_field)
    existing = set(forward_list)
    for dep_id in dep_ids:
        if dep_id not in existing:
            forward_list.append(dep_id)
            existing.add(dep_id)
        dep_path = team_dir / f"{dep_id}.json"
        if dep_path in pending_writes:
            other = pending_writes[dep_path]
        else:
            other = TaskFile(**json.loads(dep_path.read_text()))
        inverse_list: list[str] = getattr(other, inverse_field)
        if task_id not in inverse_list:
            inverse_list.append(task_id)
        pending_writes[dep_path] = other


def _remove_task_references(
    task_id: str,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
    fields: tuple[str, ...] = ("blocked_by",),
) -> None:
    """Remove a task ID from dependency fields across sibling tasks."""
    for task_file in team_dir.glob("*.json"):
        try:
            int(task_file.stem)
        except ValueError:
            continue
        if task_file.stem == task_id:
            continue
        if task_file in pending_writes:
            other = pending_writes[task_file]
        else:
            other = TaskFile(**json.loads(task_file.read_text()))
        changed = False
        for field in fields:
            dep_list: list[str] = getattr(other, field)
            if task_id in dep_list:
                dep_list.remove(task_id)
                changed = True
        if changed:
            pending_writes[task_file] = other


def _validate_dependency_additions(
    team_dir: Path,
    task_id: str,
    add_blocks: list[str] | None,
    add_blocked_by: list[str] | None,
    pending_edges: dict[str, set[str]],
) -> None:
    """Validate requested dependency additions and stage pending edges."""
    if add_blocks:
        for blocked_id in add_blocks:
            if blocked_id == task_id:
                raise TaskSelfBlockError(task_id)
            if not (team_dir / f"{blocked_id}.json").exists():
                raise TaskReferenceNotFoundError(blocked_id)
        for blocked_id in add_blocks:
            pending_edges.setdefault(blocked_id, set()).add(task_id)

    if add_blocked_by:
        for blocked_id in add_blocked_by:
            if blocked_id == task_id:
                raise TaskSelfBlockedByError(task_id)
            if not (team_dir / f"{blocked_id}.json").exists():
                raise TaskReferenceNotFoundError(blocked_id)
        for blocked_id in add_blocked_by:
            pending_edges.setdefault(task_id, set()).add(blocked_id)

    if add_blocks:
        for blocked_id in add_blocks:
            if _would_create_cycle(team_dir, blocked_id, task_id, pending_edges):
                raise CyclicTaskBlockError(task_id, blocked_id)

    if add_blocked_by:
        for blocked_id in add_blocked_by:
            if _would_create_cycle(team_dir, task_id, blocked_id, pending_edges):
                raise CyclicTaskBlockedByError(task_id, blocked_id)


def _validate_status_transition(
    task: TaskFile,
    new_status: _TaskStatus | None,
    team_dir: Path,
    extra_blocked_by: list[str] | None,
) -> None:
    """Validate a requested status transition against order and blockers."""
    if new_status is None or new_status == "deleted":
        return
    cur_order = _STATUS_ORDER[task.status]
    new_order = _STATUS_ORDER.get(new_status)
    if new_order is None:
        raise InvalidTaskStatusError(new_status)
    if new_order < cur_order:
        raise TaskStatusRegressionError(task.status, new_status)
    if new_status not in ("in_progress", "completed"):
        return
    effective_blocked_by = set(task.blocked_by)
    if extra_blocked_by:
        effective_blocked_by.update(extra_blocked_by)
    for blocker_id in effective_blocked_by:
        blocker_path = team_dir / f"{blocker_id}.json"
        if not blocker_path.exists():
            continue
        blocker = TaskFile(**json.loads(blocker_path.read_text()))
        if blocker.status != "completed":
            raise BlockedTaskStatusError(new_status, blocker_id, blocker.status)


def _apply_simple_fields(task: TaskFile, fields: TaskUpdateFields) -> None:
    """Apply subject/description/active_form/owner updates in place."""
    if fields.subject is not None:
        task.subject = fields.subject
    if fields.description is not None:
        task.description = fields.description
    if fields.active_form is not None:
        task.active_form = fields.active_form
    if fields.owner is not None:
        validate_safe_name(fields.owner, "owner")
        task.owner = fields.owner


def _apply_dependency_links(
    task: TaskFile,
    task_id: str,
    fields: TaskUpdateFields,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
) -> None:
    """Apply add_blocks / add_blocked_by edges to task and target records."""
    if fields.add_blocks:
        _link_dependency(
            task,
            task_id,
            fields.add_blocks,
            "blocks",
            "blocked_by",
            team_dir,
            pending_writes,
        )
    if fields.add_blocked_by:
        _link_dependency(
            task,
            task_id,
            fields.add_blocked_by,
            "blocked_by",
            "blocks",
            team_dir,
            pending_writes,
        )


def _apply_metadata(task: TaskFile, metadata: TaskMetadata | None) -> None:
    """Merge a metadata payload into the task (None values remove keys)."""
    if metadata is None:
        return
    current = task.metadata or {}
    for key, value in metadata.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    task.metadata = current if current else None


def _apply_status_mutation(
    task: TaskFile,
    new_status: _TaskStatus | None,
    task_id: str,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
) -> None:
    """Apply a status change and its referential side effects in place."""
    if new_status is None:
        return
    if new_status == "deleted":
        task.status = "deleted"
        _remove_task_references(
            task_id, team_dir, pending_writes, ("blocked_by", "blocks")
        )
        return
    task.status = new_status
    if new_status == "completed":
        _remove_task_references(task_id, team_dir, pending_writes, ("blocked_by",))


def _persist_task_file(
    fpath: Path,
    task: TaskFile,
    new_status: _TaskStatus | None,
    pending_writes: dict[Path, TaskFile],
) -> None:
    """Flush pending edge writes then persist or delete the primary task file."""
    if new_status == "deleted":
        _flush_pending_writes(pending_writes)
        fpath.unlink()
        return
    fpath.write_text(json.dumps(task.model_dump(by_alias=True, exclude_none=True)))
    _flush_pending_writes(pending_writes)


def _update_task(
    team_name: str,
    task_id: str,
    fields: TaskUpdateFields,
    *,
    base_dir: Path | None = None,
) -> TaskFile:
    safe_team_name = validate_safe_name(team_name, "team name")
    team_dir = _tasks_dir(base_dir) / safe_team_name
    lock_path = team_dir / ".lock"
    fpath = team_dir / f"{task_id}.json"

    with file_lock(lock_path):
        task = TaskFile(**json.loads(fpath.read_text()))
        pending_edges: dict[str, set[str]] = {}
        pending_writes: dict[Path, TaskFile] = {}

        _validate_dependency_additions(
            team_dir, task_id, fields.add_blocks, fields.add_blocked_by, pending_edges
        )
        _validate_status_transition(
            task, fields.status, team_dir, fields.add_blocked_by
        )
        _apply_simple_fields(task, fields)
        _apply_dependency_links(task, task_id, fields, team_dir, pending_writes)
        _apply_metadata(task, fields.metadata)
        _apply_status_mutation(task, fields.status, task_id, team_dir, pending_writes)
        _persist_task_file(fpath, task, fields.status, pending_writes)

    return task


async def update_task(
    team_name: str,
    task_id: str,
    fields: TaskUpdateFields,
    *,
    base_dir: Path | None = None,
) -> TaskFile:
    """Update a task in a worker thread.

    Args:
        team_name (str): Team name.
        task_id (str): Task identifier.
        fields (TaskUpdateFields): Field updates to apply.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: Updated task payload.

    """
    return await run_blocking(
        _update_task, team_name, task_id, fields, base_dir=base_dir
    )


def _list_tasks(team_name: str, base_dir: Path | None = None) -> list[TaskFile]:
    if not _team_exists(team_name, base_dir):
        raise TeamNotFoundValueError(team_name)
    team_dir = _tasks_dir(base_dir) / team_name
    tasks: list[TaskFile] = []
    for task_file in team_dir.glob("*.json"):
        try:
            int(task_file.stem)
        except ValueError:
            continue
        tasks.append(TaskFile(**json.loads(task_file.read_text())))
    tasks.sort(key=lambda task: int(task.id))
    return tasks


async def list_tasks(team_name: str, base_dir: Path | None = None) -> list[TaskFile]:
    """List tasks in a worker thread.

    Args:
        team_name (str): Team name.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        list[TaskFile]: Tasks sorted by canonical task ID order.

    """
    return await run_blocking(_list_tasks, team_name, base_dir)


def _reset_owner_tasks(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    safe_team_name = validate_safe_name(team_name, "team name")
    team_dir = _tasks_dir(base_dir) / safe_team_name
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        for task_file in team_dir.glob("*.json"):
            try:
                int(task_file.stem)
            except ValueError:
                continue
            task = TaskFile(**json.loads(task_file.read_text()))
            if task.owner == agent_name:
                if task.status != "completed":
                    task.status = "pending"
                task.owner = None
                task_file.write_text(
                    json.dumps(task.model_dump(by_alias=True, exclude_none=True))
                )


async def reset_owner_tasks(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Reset an agent's owned tasks in a worker thread.

    Args:
        team_name (str): Team name.
        agent_name (str): Agent name.
        base_dir (Path | None): Override for the base config directory.

    """
    await run_blocking(_reset_owner_tasks, team_name, agent_name, base_dir)
