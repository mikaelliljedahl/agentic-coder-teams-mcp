"""Inbox storage and retrieval tests."""

import asyncio
import fcntl
import json
import re
from pathlib import Path

import pytest

from claude_teams.messaging import (
    append_message,
    ensure_inbox,
    inbox_path,
    now_iso,
    read_inbox,
    read_inbox_filtered,
)
from claude_teams.models import InboxMessage


async def test_ensure_inbox_creates_directory_and_file(tmp_claude_dir: Path) -> None:
    path = await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert path.exists()
    assert path.parent.name == "inboxes"
    assert path.name == "alice.json"
    assert json.loads(path.read_text()) == []


async def test_ensure_inbox_idempotent(tmp_claude_dir: Path) -> None:
    await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    path = await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert path.exists()
    assert json.loads(path.read_text()) == []


async def test_append_message_accumulates(tmp_claude_dir: Path) -> None:
    msg1 = InboxMessage(
        from_="lead", text="hello", timestamp=now_iso(), read=False, summary="hi"
    )
    msg2 = InboxMessage(
        from_="lead", text="world", timestamp=now_iso(), read=False, summary="yo"
    )
    await append_message("test-team", "bob", msg1, base_dir=tmp_claude_dir)
    await append_message("test-team", "bob", msg2, base_dir=tmp_claude_dir)
    raw = json.loads(
        inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text()
    )
    assert len(raw) == 2


async def test_append_message_does_not_overwrite(tmp_claude_dir: Path) -> None:
    msg1 = InboxMessage(
        from_="lead", text="first", timestamp=now_iso(), read=False, summary="1"
    )
    msg2 = InboxMessage(
        from_="lead", text="second", timestamp=now_iso(), read=False, summary="2"
    )
    await append_message("test-team", "bob", msg1, base_dir=tmp_claude_dir)
    await append_message("test-team", "bob", msg2, base_dir=tmp_claude_dir)
    raw = json.loads(
        inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text()
    )
    texts = [message["text"] for message in raw]
    assert "first" in texts
    assert "second" in texts


async def test_read_inbox_returns_all_by_default(tmp_claude_dir: Path) -> None:
    msg1 = InboxMessage(
        from_="lead", text="a", timestamp=now_iso(), read=False, summary="s1"
    )
    msg2 = InboxMessage(
        from_="lead", text="b", timestamp=now_iso(), read=True, summary="s2"
    )
    await append_message("test-team", "carol", msg1, base_dir=tmp_claude_dir)
    await append_message("test-team", "carol", msg2, base_dir=tmp_claude_dir)

    messages = await read_inbox(
        "test-team",
        "carol",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert len(messages) == 2


async def test_read_inbox_unread_only(tmp_claude_dir: Path) -> None:
    msg1 = InboxMessage(
        from_="lead", text="a", timestamp=now_iso(), read=True, summary="s1"
    )
    msg2 = InboxMessage(
        from_="lead", text="b", timestamp=now_iso(), read=False, summary="s2"
    )
    await append_message("test-team", "dave", msg1, base_dir=tmp_claude_dir)
    await append_message("test-team", "dave", msg2, base_dir=tmp_claude_dir)

    messages = await read_inbox(
        "test-team",
        "dave",
        unread_only=True,
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert len(messages) == 1
    assert messages[0].text == "b"


async def test_read_inbox_marks_as_read(tmp_claude_dir: Path) -> None:
    message = InboxMessage(
        from_="lead", text="unread", timestamp=now_iso(), read=False, summary="s"
    )
    await append_message("test-team", "eve", message, base_dir=tmp_claude_dir)
    await read_inbox("test-team", "eve", mark_as_read=True, base_dir=tmp_claude_dir)

    remaining = await read_inbox(
        "test-team",
        "eve",
        unread_only=True,
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert len(remaining) == 0


async def test_read_inbox_applies_limit_and_offset(tmp_claude_dir: Path) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "paged",
            InboxMessage(from_="lead", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    messages = await read_inbox(
        "test-team",
        "paged",
        mark_as_read=False,
        limit=2,
        offset=1,
        base_dir=tmp_claude_dir,
    )

    assert [message.text for message in messages] == ["b", "c"]


async def test_read_inbox_supports_newest_order(tmp_claude_dir: Path) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "ordered",
            InboxMessage(from_="lead", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    messages = await read_inbox(
        "test-team",
        "ordered",
        mark_as_read=False,
        order="newest",
        base_dir=tmp_claude_dir,
    )

    assert [message.text for message in messages] == ["c", "b", "a"]


async def test_read_inbox_marks_only_paged_messages_as_read(
    tmp_claude_dir: Path,
) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "page-mark",
            InboxMessage(from_="lead", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    await read_inbox(
        "test-team",
        "page-mark",
        unread_only=True,
        mark_as_read=True,
        limit=1,
        offset=1,
        base_dir=tmp_claude_dir,
    )

    remaining = await read_inbox(
        "test-team",
        "page-mark",
        unread_only=True,
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert [message.text for message in remaining] == ["a", "c"]


async def test_read_inbox_marks_only_newest_paged_messages_as_read(
    tmp_claude_dir: Path,
) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "page-newest",
            InboxMessage(from_="lead", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    await read_inbox(
        "test-team",
        "page-newest",
        unread_only=True,
        mark_as_read=True,
        limit=1,
        offset=0,
        order="newest",
        base_dir=tmp_claude_dir,
    )

    remaining = await read_inbox(
        "test-team",
        "page-newest",
        unread_only=True,
        mark_as_read=False,
        order="oldest",
        base_dir=tmp_claude_dir,
    )

    assert [message.text for message in remaining] == ["a", "b"]


async def test_read_inbox_filtered_returns_only_matching_sender(
    tmp_claude_dir: Path,
) -> None:
    await append_message(
        "test-team",
        "lead",
        InboxMessage(from_="alice", text="a", timestamp=now_iso(), read=False),
        base_dir=tmp_claude_dir,
    )
    await append_message(
        "test-team",
        "lead",
        InboxMessage(from_="bob", text="b", timestamp=now_iso(), read=False),
        base_dir=tmp_claude_dir,
    )

    messages = await read_inbox_filtered(
        "test-team",
        "lead",
        sender_filter="alice",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert len(messages) == 1
    assert messages[0].from_ == "alice"
    assert messages[0].text == "a"


async def test_read_inbox_filtered_marks_only_matching_sender_as_read(
    tmp_claude_dir: Path,
) -> None:
    await append_message(
        "test-team",
        "lead",
        InboxMessage(from_="alice", text="a", timestamp=now_iso(), read=False),
        base_dir=tmp_claude_dir,
    )
    await append_message(
        "test-team",
        "lead",
        InboxMessage(from_="bob", text="b", timestamp=now_iso(), read=False),
        base_dir=tmp_claude_dir,
    )

    messages = await read_inbox_filtered(
        "test-team",
        "lead",
        sender_filter="alice",
        mark_as_read=True,
        base_dir=tmp_claude_dir,
    )

    assert len(messages) == 1

    remaining = await read_inbox(
        "test-team",
        "lead",
        unread_only=True,
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(remaining) == 1
    assert remaining[0].from_ == "bob"


async def test_read_inbox_filtered_applies_limit_to_newest_messages(
    tmp_claude_dir: Path,
) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "lead",
            InboxMessage(from_="alice", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    messages = await read_inbox_filtered(
        "test-team",
        "lead",
        sender_filter="alice",
        limit=2,
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )

    assert [message.text for message in messages] == ["b", "c"]


async def test_read_inbox_nonexistent_returns_empty(tmp_claude_dir: Path) -> None:
    messages = await read_inbox("test-team", "ghost", base_dir=tmp_claude_dir)
    assert messages == []


async def test_append_message_encrypts_when_master_key_set(
    tmp_claude_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY", "test-master-key")
    message = InboxMessage(
        from_="lead", text="super secret", timestamp=now_iso(), read=False, summary="s"
    )
    await append_message("test-team", "secure", message, base_dir=tmp_claude_dir)

    raw = inbox_path("test-team", "secure", base_dir=tmp_claude_dir).read_text()
    assert "super secret" not in raw
    assert '"enc"' in raw

    messages = await read_inbox(
        "test-team",
        "secure",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    assert messages[0].text == "super secret"


async def test_read_inbox_supports_plaintext_entries_when_encryption_enabled(
    tmp_claude_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = await ensure_inbox("test-team", "legacy", base_dir=tmp_claude_dir)
    path.write_text(
        json.dumps(
            [
                {
                    "from": "lead",
                    "text": "legacy plaintext",
                    "timestamp": now_iso(),
                    "read": False,
                }
            ]
        )
    )
    monkeypatch.setenv("CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY", "test-master-key")

    messages = await read_inbox(
        "test-team",
        "legacy",
        mark_as_read=False,
        base_dir=tmp_claude_dir,
    )
    assert len(messages) == 1
    assert messages[0].text == "legacy plaintext"


async def test_read_inbox_raises_without_master_key_for_encrypted_entries(
    tmp_claude_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY", "test-master-key")
    message = InboxMessage(
        from_="lead", text="top secret", timestamp=now_iso(), read=False, summary="s"
    )
    await append_message("test-team", "sealed", message, base_dir=tmp_claude_dir)
    monkeypatch.delenv("CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY")

    with pytest.raises(RuntimeError, match="encryption key"):
        await read_inbox(
            "test-team",
            "sealed",
            mark_as_read=False,
            base_dir=tmp_claude_dir,
        )


async def test_read_inbox_waits_for_lock_before_mark_as_read(
    tmp_claude_dir: Path,
) -> None:
    message = InboxMessage(
        from_="lead", text="A", timestamp=now_iso(), read=False, summary="a"
    )
    await append_message("test-team", "race", message, base_dir=tmp_claude_dir)

    path = inbox_path("test-team", "race", base_dir=tmp_claude_dir)
    lock_path = path.parent / ".lock"
    lock_path.touch(exist_ok=True)
    completed = asyncio.Event()

    async def do_read() -> None:
        await read_inbox(
            "test-team",
            "race",
            mark_as_read=True,
            base_dir=tmp_claude_dir,
        )
        completed.set()

    with lock_path.open() as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        read_task = asyncio.create_task(do_read())
        await asyncio.sleep(0.1)
        assert not completed.is_set(), (
            "read_inbox(mark_as_read=True) completed without acquiring the inbox lock"
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    await asyncio.wait_for(read_task, timeout=5)


def test_now_iso_format() -> None:
    timestamp = now_iso()
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        timestamp,
    )
