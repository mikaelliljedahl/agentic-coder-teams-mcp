"""Core task lifecycle tests."""

import json
from pathlib import Path

import pytest
import pytest_asyncio

from claude_teams.tasks import (
    create_task,
    get_task,
    list_tasks,
    reset_owner_tasks,
    update_task,
)
from claude_teams.teams import create_team


@pytest_asyncio.fixture
async def team_context(tmp_claude_dir: Path) -> tuple[str, Path, Path]:
    team_name = "test-team"
    await create_team(team_name, "sess-test", base_dir=tmp_claude_dir)
    return team_name, tmp_claude_dir, tmp_claude_dir / "tasks" / team_name


async def test_create_task_assigns_id_1_first(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "First", "desc", base_dir=base_dir)
    assert task.id == "1"


async def test_create_task_auto_increments(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    await create_task(team_name, "First", "desc", base_dir=base_dir)
    task = await create_task(team_name, "Second", "desc2", base_dir=base_dir)
    assert task.id == "2"


async def test_create_task_excludes_none_owner(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    raw = json.loads((tasks_dir / f"{task.id}.json").read_text())
    assert "owner" not in raw


async def test_create_task_with_metadata(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, tasks_dir = team_context
    task = await create_task(
        team_name,
        "Sub",
        "desc",
        metadata={"key": "val"},
        base_dir=base_dir,
    )
    raw = json.loads((tasks_dir / f"{task.id}.json").read_text())
    assert raw["metadata"] == {"key": "val"}


async def test_get_task_round_trip(team_context: tuple[str, Path, Path]) -> None:
    team_name, base_dir, _tasks_dir = team_context
    created = await create_task(
        team_name,
        "Sub",
        "desc",
        active_form="do the thing",
        base_dir=base_dir,
    )
    fetched = await get_task(team_name, created.id, base_dir=base_dir)
    assert fetched.id == created.id
    assert fetched.subject == "Sub"
    assert fetched.description == "desc"
    assert fetched.active_form == "do the thing"
    assert fetched.status == "pending"


async def test_update_task_changes_status(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        status="in_progress",
        base_dir=base_dir,
    )
    assert updated.status == "in_progress"
    on_disk = await get_task(team_name, task.id, base_dir=base_dir)
    assert on_disk.status == "in_progress"


async def test_update_task_sets_owner(team_context: tuple[str, Path, Path]) -> None:
    team_name, base_dir, tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        owner="worker-1",
        base_dir=base_dir,
    )
    assert updated.owner == "worker-1"
    raw = json.loads((tasks_dir / f"{task.id}.json").read_text())
    assert raw["owner"] == "worker-1"


async def test_update_task_delete_removes_file(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    task_path = tasks_dir / f"{task.id}.json"
    assert task_path.exists()
    result = await update_task(
        team_name,
        task.id,
        status="deleted",
        base_dir=base_dir,
    )
    assert not task_path.exists()
    assert result.status == "deleted"


async def test_list_tasks_returns_sorted(team_context: tuple[str, Path, Path]) -> None:
    team_name, base_dir, _tasks_dir = team_context
    await create_task(team_name, "A", "d1", base_dir=base_dir)
    await create_task(team_name, "B", "d2", base_dir=base_dir)
    await create_task(team_name, "C", "d3", base_dir=base_dir)
    listed_tasks = await list_tasks(team_name, base_dir=base_dir)
    assert [task.id for task in listed_tasks] == ["1", "2", "3"]


async def test_list_tasks_empty(team_context: tuple[str, Path, Path]) -> None:
    team_name, base_dir, _tasks_dir = team_context
    listed_tasks = await list_tasks(team_name, base_dir=base_dir)
    assert listed_tasks == []


async def test_reset_owner_tasks_reverts_status(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        owner="w",
        status="in_progress",
        base_dir=base_dir,
    )
    await reset_owner_tasks(team_name, "w", base_dir=base_dir)
    after = await get_task(team_name, task.id, base_dir=base_dir)
    assert after.status == "pending"
    assert after.owner is None


async def test_reset_owner_tasks_only_affects_matching_owner(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_one.id,
        owner="w1",
        status="in_progress",
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_two.id,
        owner="w2",
        status="in_progress",
        base_dir=base_dir,
    )
    await reset_owner_tasks(team_name, "w1", base_dir=base_dir)
    after_one = await get_task(team_name, task_one.id, base_dir=base_dir)
    after_two = await get_task(team_name, task_two.id, base_dir=base_dir)
    assert after_one.status == "pending"
    assert after_one.owner is None
    assert after_two.status == "in_progress"
    assert after_two.owner == "w2"


async def test_create_task_rejects_empty_subject(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    with pytest.raises(ValueError, match="subject must not be empty"):
        await create_task(team_name, "", "desc", base_dir=base_dir)


async def test_create_task_rejects_whitespace_subject(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    with pytest.raises(ValueError, match="subject must not be empty"):
        await create_task(team_name, "   ", "desc", base_dir=base_dir)


async def test_create_task_rejects_nonexistent_team(tmp_claude_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        await create_task("no-such-team", "Sub", "desc", base_dir=tmp_claude_dir)


async def test_create_task_rejects_invalid_team_name(tmp_claude_dir: Path) -> None:
    with pytest.raises(ValueError, match="Invalid team name"):
        await create_task("../bad-team", "Sub", "desc", base_dir=tmp_claude_dir)


async def test_get_task_rejects_invalid_team_name(tmp_claude_dir: Path) -> None:
    with pytest.raises(ValueError, match="Invalid team name"):
        await get_task("../bad-team", "1", base_dir=tmp_claude_dir)


async def test_update_task_rejects_backward_status_transition(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    await update_task(team_name, task.id, status="in_progress", base_dir=base_dir)
    with pytest.raises(ValueError, match="Cannot transition"):
        await update_task(team_name, task.id, status="pending", base_dir=base_dir)


async def test_update_task_rejects_completed_to_in_progress(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    await update_task(team_name, task.id, status="in_progress", base_dir=base_dir)
    await update_task(team_name, task.id, status="completed", base_dir=base_dir)
    with pytest.raises(ValueError, match="Cannot transition"):
        await update_task(team_name, task.id, status="in_progress", base_dir=base_dir)


async def test_update_task_allows_forward_status_transition(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        status="in_progress",
        base_dir=base_dir,
    )
    assert updated.status == "in_progress"
    completed = await update_task(
        team_name,
        task.id,
        status="completed",
        base_dir=base_dir,
    )
    assert completed.status == "completed"


async def test_update_task_allows_pending_to_completed(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        status="completed",
        base_dir=base_dir,
    )
    assert updated.status == "completed"


async def test_list_tasks_rejects_nonexistent_team(tmp_claude_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        await list_tasks("no-such-team", base_dir=tmp_claude_dir)


async def test_reset_owner_tasks_preserves_completed_status(
    team_context: tuple[str, Path, Path],
) -> None:
    team_name, base_dir, _tasks_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        owner="w",
        status="in_progress",
        base_dir=base_dir,
    )
    await update_task(team_name, task.id, status="completed", base_dir=base_dir)
    await reset_owner_tasks(team_name, "w", base_dir=base_dir)
    after = await get_task(team_name, task.id, base_dir=base_dir)
    assert after.status == "completed"
    assert after.owner is None


async def test_reset_owner_tasks_rejects_invalid_team_name(
    tmp_claude_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="Invalid team name"):
        await reset_owner_tasks("../bad-team", "worker", base_dir=tmp_claude_dir)
