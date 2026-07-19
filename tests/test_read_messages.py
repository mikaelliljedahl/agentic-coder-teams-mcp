"""Tests for unread-only read_messages with the per-sender counter sidecar."""

import asyncio
import json
from pathlib import Path

import pytest

from claude_teams import server_simple


def _read(from_agent: str = "", **kwargs: object) -> dict:
    return asyncio.run(server_simple.read_messages(from_agent, **kwargs))


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


def _texts(result: dict) -> list[str]:
    return [m["text"] for m in result["messages"]]


def _empty(result: dict) -> bool:
    return result["messages"] == []


def test_first_read_returns_all_then_empty(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")

    assert _texts(_read()) == ["a1", "a2"]
    # Second read with no new messages -> empty.
    assert _empty(_read())


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
    assert _empty(_read())


def test_malformed_json_line_is_skipped(inbox: Path) -> None:
    _append_raw(inbox, '{"from": "A", "text"')  # malformed JSON
    _append(inbox, "A", "a1")

    assert _texts(_read()) == ["a1"]
    # Counter only advanced for the one valid message.
    assert _empty(_read())
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
    assert _empty(_read())


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
    assert _empty(_read())
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

    assert _empty(_read())
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


def test_returns_dict_shape_with_messages_cursors(inbox: Path) -> None:
    _append(inbox, "A", "a1")

    result = _read()

    assert set(result) == {"messages", "cursors", "seq", "unread_count", "has_more"}
    assert isinstance(result["messages"], list)
    assert result["cursors"] == {"A": 1}
    assert result["seq"] is None
    assert result["unread_count"] == 1
    assert result["has_more"] is False


def test_message_seq_is_index_plus_one(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")
    _append(inbox, "A", "a3")

    result = _read(from_agent="A")

    seqs = [m["seq"] for m in result["messages"]]
    assert seqs == [1, 2, 3]
    assert result["seq"] == 3
    assert result["cursors"] is None


def test_scalar_seq_only_when_from_agent(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "B", "b1")

    filtered = _read(from_agent="A")
    assert isinstance(filtered["seq"], int)
    assert filtered["cursors"] is None

    _append(inbox, "A", "a2")
    _append(inbox, "B", "b2")
    unfiltered = _read()
    assert unfiltered["seq"] is None
    assert unfiltered["cursors"] == {"A": 2, "B": 2}


def test_since_seq_filters_and_advances_no_reread(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")
    _append(inbox, "A", "a3")

    # Simulate the caller already having consumed up to seq=1 (a1) via some
    # other channel; ask only for what's after it.
    result = _read(from_agent="A", since_seq=1)
    assert _texts(result) == ["a2", "a3"]
    assert result["seq"] == 3

    # A following default read for A must not re-deliver a2 (already
    # consumed per the advanced cursor) nor skip anything new.
    _append(inbox, "A", "a4")
    follow_up = _read(from_agent="A")
    assert _texts(follow_up) == ["a4"]


def test_since_seq_does_not_skip_boundary_message(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")

    # since_seq=0 means "everything from the start" (0-based count floor).
    result = _read(from_agent="A", since_seq=0)
    assert _texts(result) == ["a1", "a2"]


def test_since_seq_idempotent_max_advance(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")
    _append(inbox, "A", "a3")

    _read(from_agent="A", since_seq=2)  # cursor -> 3
    # A stale/lower since_seq must not rewind the persisted cursor.
    result = _read(from_agent="A", since_seq=0)
    assert _texts(result) == []
    assert result["seq"] == 3


def test_since_seq_requires_from_agent(inbox: Path) -> None:
    _append(inbox, "A", "a1")

    with pytest.raises(ValueError, match="since_seq requires from_agent"):
        _read(since_seq=0)


def test_limit_sets_has_more(inbox: Path) -> None:
    for i in range(5):
        _append(inbox, "A", f"a{i}")

    result = _read(from_agent="A", limit=2)

    assert _texts(result) == ["a0", "a1"]
    assert result["has_more"] is True

    remainder = _read(from_agent="A")
    assert _texts(remainder) == ["a2", "a3", "a4"]


def test_full_true_ignores_limit(inbox: Path) -> None:
    for i in range(5):
        _append(inbox, "A", f"a{i}")

    result = _read(from_agent="A", full=True, limit=2)

    assert _texts(result) == ["a0", "a1", "a2", "a3", "a4"]
    assert result["has_more"] is False


def test_max_chars_truncates_message_text(inbox: Path) -> None:
    _append(inbox, "A", "x" * 100)

    result = _read(from_agent="A", max_chars=10)

    message = result["messages"][0]
    assert len(message["text"]) == 10
    assert message["truncated"] is True
    assert message["full_len"] == 100


def test_limit_advances_each_sender_only_by_what_was_delivered(inbox: Path) -> None:
    # Regression: a global limit clipping a multi-sender unfiltered batch
    # must only advance each sender's cursor by what was actually delivered
    # to them, not by everything that sender had pending.
    _append(inbox, "A", "a1")
    _append(inbox, "B", "b1")
    _append(inbox, "A", "a2")
    _append(inbox, "B", "b2")

    first = _read(limit=2)
    assert _texts(first) == ["a1", "b1"]
    assert first["cursors"] == {"A": 1, "B": 1}
    assert first["has_more"] is True

    second = _read(limit=10)
    assert _texts(second) == ["a2", "b2"]
    assert second["cursors"] == {"A": 2, "B": 2}
    assert second["has_more"] is False


def test_negative_limit_raises_value_error(inbox: Path) -> None:
    _append(inbox, "A", "a1")

    with pytest.raises(ValueError, match="limit"):
        _read(from_agent="A", limit=-1)


def test_limit_zero_returns_empty_batch_no_cursor_advance(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")

    result = _read(from_agent="A", limit=0)

    assert _texts(result) == []
    assert result["has_more"] is True
    # Cursor must not advance past what was actually read (nothing).
    assert result["seq"] == 0

    # A follow-up unfiltered read still sees both original messages.
    follow_up = _read(from_agent="A")
    assert _texts(follow_up) == ["a1", "a2"]


def test_limit_zero_has_more_false_when_nothing_pending(inbox: Path) -> None:
    result = _read(from_agent="A", limit=0)

    assert _texts(result) == []
    assert result["has_more"] is False


def test_limit_zero_reports_true_unread_count(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")

    result = _read(from_agent="A", limit=0)

    # A non-consuming peek must report the real backlog, not the empty batch:
    # a coordinator uses this to decide whether to act.
    assert result["messages"] == []
    assert result["has_more"] is True
    assert result["unread_count"] == 2


def test_clipped_batch_reports_total_unread_not_batch_size(inbox: Path) -> None:
    _append(inbox, "A", "a1")
    _append(inbox, "A", "a2")
    _append(inbox, "A", "a3")

    result = _read(limit=1)

    assert _texts(result) == ["a1"]
    assert result["has_more"] is True
    # unread_count is the pending backlog, not the size of this clipped batch.
    assert result["unread_count"] == 3


def test_max_chars_none_does_not_add_truncation_fields(inbox: Path) -> None:
    _append(inbox, "A", "hello")

    result = _read(from_agent="A")

    message = result["messages"][0]
    assert "truncated" not in message
    assert "full_len" not in message
