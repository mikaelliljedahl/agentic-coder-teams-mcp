"""Inbox storage and retrieval tests."""

import asyncio
import fcntl
import json
import re
from pathlib import Path
from typing import TextIO

import pytest

from claude_teams.errors import InboxMasterKeyTooShortError
from claude_teams.messaging import (
    _INBOX_MAX_MESSAGES,
    _compact_messages,
    append_message,
    ensure_inbox,
    inbox_path,
    now_iso,
    read_inbox,
    read_inbox_filtered,
    read_inbox_page,
)
from claude_teams.models import InboxMessage


async def test_ensure_inbox_creates_directory_and_file(tmp_claude_dir: Path) -> None:
    path = await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert await asyncio.to_thread(path.exists)
    assert path.parent.name == "inboxes"
    assert path.name == "alice.json"
    assert json.loads(await asyncio.to_thread(path.read_text)) == []


async def test_ensure_inbox_idempotent(tmp_claude_dir: Path) -> None:
    await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    path = await ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert await asyncio.to_thread(path.exists)
    assert json.loads(await asyncio.to_thread(path.read_text)) == []


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
        await asyncio.to_thread(
            inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text
        )
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
        await asyncio.to_thread(
            inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text
        )
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


async def test_read_inbox_page_returns_messages_and_total_count(
    tmp_claude_dir: Path,
) -> None:
    for text in ("a", "b", "c"):
        await append_message(
            "test-team",
            "paged-count",
            InboxMessage(from_="lead", text=text, timestamp=now_iso(), read=False),
            base_dir=tmp_claude_dir,
        )

    messages, total_count = await read_inbox_page(
        "test-team",
        "paged-count",
        mark_as_read=False,
        limit=2,
        offset=1,
        base_dir=tmp_claude_dir,
    )

    assert total_count == 3
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


def test_inbox_path_rejects_invalid_agent_name(tmp_claude_dir: Path) -> None:
    with pytest.raises(ValueError, match="Invalid agent name"):
        inbox_path("test-team", "../other-team/inboxes/bob", base_dir=tmp_claude_dir)


async def test_append_message_encrypts_when_master_key_set(
    tmp_claude_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY",
        "test-master-key-must-be-32-plus-chars",
    )
    message = InboxMessage(
        from_="lead", text="super secret", timestamp=now_iso(), read=False, summary="s"
    )
    await append_message("test-team", "secure", message, base_dir=tmp_claude_dir)

    raw = await asyncio.to_thread(
        inbox_path("test-team", "secure", base_dir=tmp_claude_dir).read_text
    )
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
    await asyncio.to_thread(
        path.write_text,
        json.dumps(
            [
                {
                    "from": "lead",
                    "text": "legacy plaintext",
                    "timestamp": now_iso(),
                    "read": False,
                }
            ]
        ),
    )
    monkeypatch.setenv(
        "CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY",
        "test-master-key-must-be-32-plus-chars",
    )

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
    monkeypatch.setenv(
        "CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY",
        "test-master-key-must-be-32-plus-chars",
    )
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
    await asyncio.to_thread(lock_path.touch, exist_ok=True)
    completed = asyncio.Event()

    async def do_read() -> None:
        await read_inbox(
            "test-team",
            "race",
            mark_as_read=True,
            base_dir=tmp_claude_dir,
        )
        completed.set()

    def _open_and_lock() -> TextIO:
        lock_file = lock_path.open()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return lock_file

    def _unlock_and_close(lock_file: TextIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    lock_file = await asyncio.to_thread(_open_and_lock)
    try:
        read_task = asyncio.create_task(do_read())
        await asyncio.sleep(0.1)
        assert not completed.is_set(), (
            "read_inbox(mark_as_read=True) completed without acquiring the inbox lock"
        )
    finally:
        await asyncio.to_thread(_unlock_and_close, lock_file)

    await asyncio.wait_for(read_task, timeout=5)


async def test_append_raises_when_master_key_below_min_length(
    tmp_claude_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A master key shorter than 32 chars must be rejected, not silently used.

    Short keys slide through HKDF without error but give the attacker a
    brute-forceable input space. The gate surfaces the misconfiguration
    loudly on the first encrypted write.
    """
    monkeypatch.setenv("CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY", "too-short-key")
    message = InboxMessage(
        from_="lead", text="payload", timestamp=now_iso(), read=False, summary="s"
    )
    with pytest.raises(InboxMasterKeyTooShortError, match="at least 32 characters"):
        await append_message("test-team", "short-key", message, base_dir=tmp_claude_dir)


def test_compact_messages_preserves_all_unread() -> None:
    """Compaction must never drop unread messages — they encode pending work.

    When the inbox overflows but unread already equals or exceeds the cap,
    the compactor returns unread-only; read messages are evicted wholesale.
    """
    unread_count = _INBOX_MAX_MESSAGES + 5
    unread = [
        InboxMessage(
            from_="lead", text=f"u{i}", timestamp=now_iso(), read=False, summary="s"
        )
        for i in range(unread_count)
    ]
    read = [
        InboxMessage(
            from_="lead", text=f"r{i}", timestamp=now_iso(), read=True, summary="s"
        )
        for i in range(10)
    ]
    result = _compact_messages(unread + read)
    assert len(result) == unread_count
    assert all(not m.read for m in result)


def test_compact_messages_drops_oldest_read_first() -> None:
    """Compaction keeps newest read messages and all unread, in their order.

    Verifies the FIFO eviction rule: when the inbox overflows but unread is
    sparse, the oldest read messages are evicted first and the newest
    ``keep_read`` read messages are retained alongside all unread.
    """
    unread = [
        InboxMessage(
            from_="lead", text=f"u{i}", timestamp=now_iso(), read=False, summary="s"
        )
        for i in range(5)
    ]
    read_excess = _INBOX_MAX_MESSAGES
    read = [
        InboxMessage(
            from_="lead", text=f"r{i}", timestamp=now_iso(), read=True, summary="s"
        )
        for i in range(read_excess)
    ]
    result = _compact_messages(unread + read)
    assert len(result) == _INBOX_MAX_MESSAGES
    assert sum(1 for m in result if not m.read) == 5
    retained_read_texts = {m.text for m in result if m.read}
    # The oldest read message (r0) must be evicted; the newest (r{N-1}) retained.
    assert "r0" not in retained_read_texts
    assert f"r{read_excess - 1}" in retained_read_texts


def test_compact_messages_below_cap_is_noop() -> None:
    """Under the cap, compaction returns the input list unchanged."""
    msgs = [
        InboxMessage(
            from_="lead",
            text=f"m{i}",
            timestamp=now_iso(),
            read=i % 2 == 0,
            summary="s",
        )
        for i in range(10)
    ]
    assert _compact_messages(msgs) is msgs


def test_now_iso_format() -> None:
    timestamp = now_iso()
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        timestamp,
    )
