"""Task dependency and graph validation tests."""

from pathlib import Path

import pytest
import pytest_asyncio

from claude_teams.models import TaskUpdateFields
from claude_teams.tasks import create_task, get_task, update_task
from claude_teams.teams import create_team


@pytest_asyncio.fixture
async def team_context(tmp_claude_dir: Path) -> tuple[str, Path]:
    team_name = "test-team"
    await create_team(team_name, "sess-test", base_dir=tmp_claude_dir)
    return team_name, tmp_claude_dir


async def test_update_task_add_blocks(team_context: tuple[str, Path]) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    task_two = await create_task(team_name, "T2", "d", base_dir=base_dir)
    task_three = await create_task(team_name, "T3", "d", base_dir=base_dir)
    task_four = await create_task(team_name, "T4", "d", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocks=[task_two.id, task_three.id]),
        base_dir=base_dir,
    )
    assert updated.blocks == [task_two.id, task_three.id]
    updated_again = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocks=[task_three.id, task_four.id]),
        base_dir=base_dir,
    )
    assert updated_again.blocks == [task_two.id, task_three.id, task_four.id]


async def test_update_task_add_blocked_by(team_context: tuple[str, Path]) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    task_two = await create_task(team_name, "Dep1", "d", base_dir=base_dir)
    task_three = await create_task(team_name, "Dep2", "d", base_dir=base_dir)
    task_four = await create_task(team_name, "Dep3", "d", base_dir=base_dir)
    updated = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocked_by=[task_two.id, task_three.id]),
        base_dir=base_dir,
    )
    assert updated.blocked_by == [task_two.id, task_three.id]
    updated_again = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocked_by=[task_three.id, task_four.id]),
        base_dir=base_dir,
    )
    assert updated_again.blocked_by == [task_two.id, task_three.id, task_four.id]


async def test_update_task_rejects_self_reference_in_blocks(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    with pytest.raises(ValueError, match="cannot block itself"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocks=[task.id]),
            base_dir=base_dir,
        )


async def test_update_task_rejects_self_reference_in_blocked_by(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    with pytest.raises(ValueError, match="cannot be blocked by itself"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocked_by=[task.id]),
            base_dir=base_dir,
        )


async def test_update_task_rejects_nonexistent_dep_in_blocks(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    with pytest.raises(ValueError, match="does not exist"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocks=["999"]),
            base_dir=base_dir,
        )


async def test_update_task_rejects_nonexistent_dep_in_blocked_by(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Sub", "desc", base_dir=base_dir)
    with pytest.raises(ValueError, match="does not exist"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocked_by=["999"]),
            base_dir=base_dir,
        )


async def test_update_task_rejects_start_when_blocked(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    blocker = await create_task(team_name, "Blocker", "b", base_dir=base_dir)
    task = await create_task(team_name, "Blocked", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocked_by=[blocker.id]),
        base_dir=base_dir,
    )
    with pytest.raises(ValueError, match="blocked by task"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(status="in_progress"),
            base_dir=base_dir,
        )


async def test_update_task_allows_start_when_blockers_completed(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    blocker = await create_task(team_name, "Blocker", "b", base_dir=base_dir)
    task = await create_task(team_name, "Blocked", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocked_by=[blocker.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        blocker.id,
        TaskUpdateFields(status="in_progress"),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        blocker.id,
        TaskUpdateFields(status="completed"),
        base_dir=base_dir,
    )
    updated = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(status="in_progress"),
        base_dir=base_dir,
    )
    assert updated.status == "in_progress"


async def test_update_task_allows_start_when_blocker_deleted(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    blocker = await create_task(team_name, "Blocker", "b", base_dir=base_dir)
    task = await create_task(team_name, "Blocked", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        TaskUpdateFields(add_blocked_by=[blocker.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        blocker.id,
        TaskUpdateFields(status="deleted"),
        base_dir=base_dir,
    )
    updated = await update_task(
        team_name,
        task.id,
        TaskUpdateFields(status="in_progress"),
        base_dir=base_dir,
    )
    assert updated.status == "in_progress"


async def test_add_blocked_by_syncs_blocks_on_target(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_two.id,
        TaskUpdateFields(add_blocked_by=[task_one.id]),
        base_dir=base_dir,
    )
    task_one_after = await get_task(team_name, task_one.id, base_dir=base_dir)
    assert task_two.id in task_one_after.blocks


async def test_add_blocks_syncs_blocked_by_on_target(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(add_blocks=[task_two.id]),
        base_dir=base_dir,
    )
    task_two_after = await get_task(team_name, task_two.id, base_dir=base_dir)
    assert task_one.id in task_two_after.blocked_by


async def test_bidirectional_sync_is_idempotent(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(add_blocks=[task_two.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(add_blocks=[task_two.id]),
        base_dir=base_dir,
    )
    task_one_after = await get_task(team_name, task_one.id, base_dir=base_dir)
    task_two_after = await get_task(team_name, task_two.id, base_dir=base_dir)
    assert task_one_after.blocks == [task_two.id]
    assert task_two_after.blocked_by == [task_one.id]


async def test_completing_task_cleans_blocked_by_on_dependents(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_two.id,
        TaskUpdateFields(add_blocked_by=[task_one.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(status="completed"),
        base_dir=base_dir,
    )
    task_two_after = await get_task(team_name, task_two.id, base_dir=base_dir)
    assert task_one.id not in task_two_after.blocked_by


async def test_completing_task_preserves_blocks_on_self(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(add_blocks=[task_two.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(status="completed"),
        base_dir=base_dir,
    )
    task_one_after = await get_task(team_name, task_one.id, base_dir=base_dir)
    assert task_two.id in task_one_after.blocks


async def test_delete_task_cleans_up_stale_refs(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_one = await create_task(team_name, "A", "d1", base_dir=base_dir)
    task_two = await create_task(team_name, "B", "d2", base_dir=base_dir)
    await update_task(
        team_name,
        task_two.id,
        TaskUpdateFields(add_blocked_by=[task_one.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(add_blocks=[task_two.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_one.id,
        TaskUpdateFields(status="deleted"),
        base_dir=base_dir,
    )
    task_two_after = await get_task(team_name, task_two.id, base_dir=base_dir)
    assert task_one.id not in task_two_after.blocked_by
    assert task_one.id not in task_two_after.blocks


async def test_no_partial_write_when_status_validation_fails(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    blocker = await create_task(team_name, "Blocker", "b", base_dir=base_dir)
    task = await create_task(team_name, "Task", "t", base_dir=base_dir)
    blocker_before = await get_task(team_name, blocker.id, base_dir=base_dir)
    task_before = await get_task(team_name, task.id, base_dir=base_dir)
    with pytest.raises(ValueError, match="blocked by task"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocked_by=[blocker.id], status="in_progress"),
            base_dir=base_dir,
        )
    blocker_after = await get_task(team_name, blocker.id, base_dir=base_dir)
    task_after = await get_task(team_name, task.id, base_dir=base_dir)
    assert blocker_after.blocks == blocker_before.blocks
    assert task_after.blocked_by == task_before.blocked_by


async def test_no_partial_write_on_add_blocks_with_failed_status(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task = await create_task(team_name, "Task", "t", base_dir=base_dir)
    other = await create_task(team_name, "Other", "o", base_dir=base_dir)
    await update_task(
        team_name,
        task.id,
        TaskUpdateFields(status="in_progress"),
        base_dir=base_dir,
    )
    with pytest.raises(ValueError, match="Cannot transition"):
        await update_task(
            team_name,
            task.id,
            TaskUpdateFields(add_blocks=[other.id], status="pending"),
            base_dir=base_dir,
        )
    other_after = await get_task(team_name, other.id, base_dir=base_dir)
    task_after = await get_task(team_name, task.id, base_dir=base_dir)
    assert task_after.blocks == []
    assert other_after.blocked_by == []


async def test_rejects_simple_circular_dependency(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_a = await create_task(team_name, "A", "d", base_dir=base_dir)
    task_b = await create_task(team_name, "B", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task_a.id,
        TaskUpdateFields(add_blocked_by=[task_b.id]),
        base_dir=base_dir,
    )
    with pytest.raises(ValueError, match="circular dependency"):
        await update_task(
            team_name,
            task_b.id,
            TaskUpdateFields(add_blocked_by=[task_a.id]),
            base_dir=base_dir,
        )


async def test_rejects_transitive_circular_dependency(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_a = await create_task(team_name, "A", "d", base_dir=base_dir)
    task_b = await create_task(team_name, "B", "d", base_dir=base_dir)
    task_c = await create_task(team_name, "C", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task_a.id,
        TaskUpdateFields(add_blocked_by=[task_b.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_b.id,
        TaskUpdateFields(add_blocked_by=[task_c.id]),
        base_dir=base_dir,
    )
    with pytest.raises(ValueError, match="circular dependency"):
        await update_task(
            team_name,
            task_c.id,
            TaskUpdateFields(add_blocked_by=[task_a.id]),
            base_dir=base_dir,
        )


async def test_rejects_circular_via_add_blocks(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_a = await create_task(team_name, "A", "d", base_dir=base_dir)
    task_b = await create_task(team_name, "B", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task_a.id,
        TaskUpdateFields(add_blocked_by=[task_b.id]),
        base_dir=base_dir,
    )
    with pytest.raises(ValueError, match="circular dependency"):
        await update_task(
            team_name,
            task_a.id,
            TaskUpdateFields(add_blocks=[task_b.id]),
            base_dir=base_dir,
        )


async def test_allows_non_cyclic_diamond_dependency(
    team_context: tuple[str, Path],
) -> None:
    team_name, base_dir = team_context
    task_a = await create_task(team_name, "A", "d", base_dir=base_dir)
    task_b = await create_task(team_name, "B", "d", base_dir=base_dir)
    task_c = await create_task(team_name, "C", "d", base_dir=base_dir)
    task_d = await create_task(team_name, "D", "d", base_dir=base_dir)
    await update_task(
        team_name,
        task_d.id,
        TaskUpdateFields(add_blocked_by=[task_b.id, task_c.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_b.id,
        TaskUpdateFields(add_blocked_by=[task_a.id]),
        base_dir=base_dir,
    )
    await update_task(
        team_name,
        task_c.id,
        TaskUpdateFields(add_blocked_by=[task_a.id]),
        base_dir=base_dir,
    )
    task_d_after = await get_task(team_name, task_d.id, base_dir=base_dir)
    assert set(task_d_after.blocked_by) == {task_b.id, task_c.id}
