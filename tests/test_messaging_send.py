"""Message sending tests."""

import json
import re
from pathlib import Path

from claude_teams.messaging import (
    now_iso,
    read_inbox,
    send_plain_message,
    send_shutdown_request,
    send_structured_message,
    send_task_assignment,
)
from claude_teams.models import TaskAssignment, TaskFile


async def test_send_plain_message_appears_in_inbox(tmp_claude_dir: Path) -> None:
    await send_plain_message(
        "test-team",
        "lead",
        "frank",
        "hey there",
        summary="greeting",
        base_dir=tmp_claude_dir,
    )
    messages = await read_inbox(
        "test-team",
        "frank",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    assert messages[0].from_ == "lead"
    assert messages[0].text == "hey there"
    assert messages[0].summary == "greeting"
    assert messages[0].read is False


async def test_send_plain_message_with_color(tmp_claude_dir: Path) -> None:
    await send_plain_message(
        "test-team",
        "lead",
        "gina",
        "colorful",
        summary="c",
        color="blue",
        base_dir=tmp_claude_dir,
    )
    messages = await read_inbox(
        "test-team",
        "gina",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert messages[0].color == "blue"


async def test_send_structured_message_serializes_json_in_text(
    tmp_claude_dir: Path,
) -> None:
    payload = TaskAssignment(
        task_id="t-1",
        subject="Do thing",
        description="Details here",
        assigned_by="lead",
        timestamp=now_iso(),
    )
    await send_structured_message(
        "test-team",
        "lead",
        "hank",
        payload,
        base_dir=tmp_claude_dir,
    )
    messages = await read_inbox(
        "test-team",
        "hank",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    parsed = json.loads(messages[0].text)
    assert parsed["type"] == "task_assignment"
    assert parsed["taskId"] == "t-1"


async def test_send_task_assignment_format(tmp_claude_dir: Path) -> None:
    task = TaskFile(
        id="task-42",
        subject="Build feature",
        description="Build it well",
        owner="iris",
    )
    await send_task_assignment(
        "test-team",
        task,
        assigned_by="lead",
        base_dir=tmp_claude_dir,
    )
    messages = await read_inbox(
        "test-team",
        "iris",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    parsed = json.loads(messages[0].text)
    assert parsed["type"] == "task_assignment"
    assert parsed["taskId"] == "task-42"
    assert parsed["subject"] == "Build feature"
    assert parsed["description"] == "Build it well"
    assert parsed["assignedBy"] == "lead"


async def test_send_shutdown_request_returns_request_id(
    tmp_claude_dir: Path,
) -> None:
    request_id = await send_shutdown_request(
        "test-team",
        "jake",
        base_dir=tmp_claude_dir,
    )
    assert re.match(r"^shutdown-\d+@jake$", request_id)


async def test_send_shutdown_request_with_reason(tmp_claude_dir: Path) -> None:
    await send_shutdown_request(
        "test-team",
        "kate",
        reason="Done",
        base_dir=tmp_claude_dir,
    )
    messages = await read_inbox(
        "test-team",
        "kate",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    parsed = json.loads(messages[0].text)
    assert parsed["type"] == "shutdown_request"
    assert parsed["reason"] == "Done"
