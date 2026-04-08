"""Task persistence helpers for file-backed team state."""

import json
from collections import deque
from pathlib import Path
from typing import Literal

from claude_teams.async_utils import run_blocking
from claude_teams.filelock import file_lock
from claude_teams.models import TaskFile
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
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    safe_team_name = validate_safe_name(team_name, "team name")
    if not subject or not subject.strip():
        raise ValueError("Task subject must not be empty")
    if not _team_exists(safe_team_name, base_dir):
        raise ValueError(f"Team {safe_team_name!r} does not exist")
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
    metadata: dict | None = None,
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


def _update_task(
    team_name: str,
    task_id: str,
    *,
    status: _TaskStatus | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    safe_team_name = validate_safe_name(team_name, "team name")
    team_dir = _tasks_dir(base_dir) / safe_team_name
    lock_path = team_dir / ".lock"
    fpath = team_dir / f"{task_id}.json"

    with file_lock(lock_path):
        # --- Phase 1: Read ---
        task = TaskFile(**json.loads(fpath.read_text()))

        # --- Phase 2: Validate (no disk writes) ---
        pending_edges: dict[str, set[str]] = {}

        if add_blocks:
            for blocked_id in add_blocks:
                if blocked_id == task_id:
                    raise ValueError(f"Task {task_id} cannot block itself")
                if not (team_dir / f"{blocked_id}.json").exists():
                    raise ValueError(f"Referenced task {blocked_id!r} does not exist")
            for blocked_id in add_blocks:
                pending_edges.setdefault(blocked_id, set()).add(task_id)

        if add_blocked_by:
            for blocked_id in add_blocked_by:
                if blocked_id == task_id:
                    raise ValueError(f"Task {task_id} cannot be blocked by itself")
                if not (team_dir / f"{blocked_id}.json").exists():
                    raise ValueError(f"Referenced task {blocked_id!r} does not exist")
            for blocked_id in add_blocked_by:
                pending_edges.setdefault(task_id, set()).add(blocked_id)

        if add_blocks:
            for blocked_id in add_blocks:
                if _would_create_cycle(team_dir, blocked_id, task_id, pending_edges):
                    raise ValueError(
                        f"Adding block {task_id} -> {blocked_id} would create a circular dependency"
                    )

        if add_blocked_by:
            for blocked_id in add_blocked_by:
                if _would_create_cycle(team_dir, task_id, blocked_id, pending_edges):
                    raise ValueError(
                        f"Adding dependency {task_id} blocked_by {blocked_id} would create a circular dependency"
                    )

        if status is not None and status != "deleted":
            cur_order = _STATUS_ORDER[task.status]
            new_order = _STATUS_ORDER.get(status)
            if new_order is None:
                raise ValueError(f"Invalid status: {status!r}")
            if new_order < cur_order:
                raise ValueError(
                    f"Cannot transition from {task.status!r} to {status!r}"
                )
            effective_blocked_by = set(task.blocked_by)
            if add_blocked_by:
                effective_blocked_by.update(add_blocked_by)
            if status in ("in_progress", "completed") and effective_blocked_by:
                for blocker_id in effective_blocked_by:
                    blocker_path = team_dir / f"{blocker_id}.json"
                    if blocker_path.exists():
                        blocker = TaskFile(**json.loads(blocker_path.read_text()))
                        if blocker.status != "completed":
                            raise ValueError(
                                f"Cannot set status to {status!r}: "
                                f"blocked by task {blocker_id} (status: {blocker.status!r})"
                            )

        # --- Phase 3: Mutate (in-memory only) ---
        pending_writes: dict[Path, TaskFile] = {}

        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            validate_safe_name(owner, "owner")
            task.owner = owner

        if add_blocks:
            _link_dependency(
                task,
                task_id,
                add_blocks,
                "blocks",
                "blocked_by",
                team_dir,
                pending_writes,
            )

        if add_blocked_by:
            _link_dependency(
                task,
                task_id,
                add_blocked_by,
                "blocked_by",
                "blocks",
                team_dir,
                pending_writes,
            )

        if metadata is not None:
            current = task.metadata or {}
            for key, value in metadata.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            task.metadata = current if current else None

        if status is not None and status != "deleted":
            task.status = status
            if status == "completed":
                _remove_task_references(
                    task_id, team_dir, pending_writes, ("blocked_by",)
                )

        if status == "deleted":
            task.status = "deleted"
            _remove_task_references(
                task_id, team_dir, pending_writes, ("blocked_by", "blocks")
            )

        # --- Phase 4: Write ---
        if status == "deleted":
            _flush_pending_writes(pending_writes)
            fpath.unlink()
        else:
            fpath.write_text(
                json.dumps(task.model_dump(by_alias=True, exclude_none=True))
            )
            _flush_pending_writes(pending_writes)

    return task


async def update_task(
    team_name: str,
    task_id: str,
    *,
    status: _TaskStatus | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    """Update a task in a worker thread.

    Args:
        team_name (str): Team name.
        task_id (str): Task identifier.
        status (_TaskStatus | None): Optional new status.
        owner (str | None): Optional new owner.
        subject (str | None): Optional new subject.
        description (str | None): Optional new description.
        active_form (str | None): Optional new active-form text.
        add_blocks (list[str] | None): Task IDs this task blocks.
        add_blocked_by (list[str] | None): Task IDs blocking this task.
        metadata (dict | None): Metadata merge payload.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: Updated task payload.

    """
    return await run_blocking(
        _update_task,
        team_name,
        task_id,
        status=status,
        owner=owner,
        subject=subject,
        description=description,
        active_form=active_form,
        add_blocks=add_blocks,
        add_blocked_by=add_blocked_by,
        metadata=metadata,
        base_dir=base_dir,
    )


def _list_tasks(team_name: str, base_dir: Path | None = None) -> list[TaskFile]:
    if not _team_exists(team_name, base_dir):
        raise ValueError(f"Team {team_name!r} does not exist")
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
