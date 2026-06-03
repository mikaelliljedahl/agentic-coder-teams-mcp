"""Tests for unread-only read_messages with the per-sender counter sidecar."""

import asyncio
import json
from pathlib import Path

import pytest

from claude_teams import server_simple


def _read(from_agent: str = "") -> list[dict]:
    return asyncio.run(server_simple.read_messages(from_agent))


@pytest.fixture
def inbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server at a temp session and return this identity's inbox path."""
    session_id = "test-session"
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path)
    monkeypatch.setattr(server_simple, "_session_id", session_id)
    # Reset the per-inbox lock registry so tests do not share state.
    monkeypatch.setattr(server_simple, "_inbox_locks", {})
    path = server_simple._inbox_file(session_id, server_simple.IDENTITY)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append(inbox: Path, sender: str, text: str) -> None:
    line = json.dumps({"from": sender, "text": text, "ts": "now"})
    with inbox.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_raw(inbox: Path, raw: str) -> None:
    with inbox.open("a", encoding="utf-8") as handle:
        handle.write(raw + "\n")


def _texts(messages: list[dict]) -> list[str]:
    return [m["text"] for m in messages]


def test_first_read_returns_all_then_empty(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")

    assert _texts(_read()) == ["a1", "a2"]
    # Second read with no new messages -> empty.
    assert _read() == []


def test_filtered_read_does_not_advance_other_sender(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "B", "b1")

    assert _texts(_read(from_agent="A")) == ["a1"]
    # B was not advanced by the filtered read.
    assert _texts(_read(from_agent="B")) == ["b1"]


def test_unfiltered_read_advances_all_senders(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "B", "b1")

    assert _texts(_read()) == ["a1", "b1"]
    assert _read() == []


def test_malformed_json_line_is_skipped(inbox: Path) -> None:
    _append_raw(inbox, '{"from": "A", "text"')  # malformed JSON
    _append(inbox, "A", "a1")

    assert _texts(_read()) == ["a1"]
    # Counter only advanced for the one valid message.
    assert _read() == []
    cursors = server_simple._load_inbox_cursors(
        server_simple._inbox_cursor_file("test-session", server_simple.IDENTITY)
    )
    assert cursors == {"A": 1}


def test_missing_or_empty_from_is_skipped(inbox: Path) -> None:
    _append_raw(inbox, json.dumps({"text": "no sender"}))
    _append_raw(inbox, json.dumps({"from": "", "text": "empty sender"}))
    _append_raw(inbox, json.dumps(["not", "a", "dict"]))
    _append(inbox, "A", "a1")

    assert _texts(_read()) == ["a1"]
    cursors = server_simple._load_inbox_cursors(
        server_simple._inbox_cursor_file("test-session", server_simple.IDENTITY)
    )
    # No stray counter keys for the skipped records.
    assert cursors == {"A": 1}


def test_corrupt_cursor_file_treated_as_empty(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    cursor_file = server_simple._inbox_cursor_file(
        "test-session", server_simple.IDENTITY
    )
    cursor_file.write_text("{not valid json", encoding="utf-8")

    # Corrupt file -> treated as empty, re-reads history, no crash.
    assert _texts(_read()) == ["a1"]
    assert _read() == []


def test_missing_cursor_file_treated_as_empty(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    assert not server_simple._inbox_cursor_file(
        "test-session", server_simple.IDENTITY
    ).exists()
    assert _texts(_read()) == ["a1"]


def test_clamps_forward_cursor_value(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    cursor_file = server_simple._inbox_cursor_file(
        "test-session", server_simple.IDENTITY
    )
    cursor_file.write_text(json.dumps({"A": 999999}), encoding="utf-8")

    # Stored count beyond observed is clamped down; a1 is already delivered.
    assert _read() == []
    _append(inbox, "A", "a2")
    assert _texts(_read()) == ["a2"]


def test_forward_cursor_on_empty_inbox_does_not_skip_first_message(
    inbox: Path,
) -> None:
    # Regression: a forward cursor for a sender absent from the (empty) snapshot
    # must be clamped to 0 so the sender's first future message is delivered.
    cursor_file = server_simple._inbox_cursor_file(
        "test-session", server_simple.IDENTITY
    )
    cursor_file.write_text(json.dumps({"A": 999999}), encoding="utf-8")
    # Inbox does not exist at this point.
    assert not inbox.exists()

    assert _read() == []
    # The bad cursor was clamped and persisted; A's first message is delivered.
    _append(inbox, "A", "a1")
    assert _texts(_read()) == ["a1"]


def test_mixed_ordering_per_sender_cursor(inbox: Path) -> None:
    # Inbox order: A1, B1, A2.
    _append(inbox, "A", "A1")
    _append(inbox, "B", "B1")
    _append(inbox, "A", "A2")

    # Filtered A read returns A1, A2 and advances only A.
    assert _texts(_read(from_agent="A")) == ["A1", "A2"]
    # A later unfiltered read returns B1 (B was never advanced).
    assert _texts(_read()) == ["B1"]


def test_bool_cursor_values_rejected(tmp_path: Path) -> None:
    # isinstance(True, int) is True; bools must not be accepted as counts.
    cursor = tmp_path / "inbox-x.pos.json"
    cursor.write_text(json.dumps({"A": True, "B": 2}), encoding="utf-8")
    loaded = server_simple._load_inbox_cursors(cursor)
    assert loaded == {"B": 2}
