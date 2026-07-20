"""C4/R3 — the watcher's delivery contract, stated as tests.

R3 makes the watcher a protocol component rather than a convenience: upstream
messages reach a lead only via ``send_message`` + the lead's watcher, so the
watcher owes three guarantees. This file characterises the guarantees that
**already hold**; it deliberately changes nothing. Wake priority and the settle
window are out of scope (see the requirements' non-goals), so nothing here
touches the ``before = after`` output-edge coupling.

1. **Wake without consume.** A wake reports unread senders; it never advances a
   cursor. The message is still there for ``read_messages`` afterwards, and a
   second watcher would wake for the same message.
2. **Cursor clamping.** A cursor ahead of the message count yields zero unread,
   never a negative count and never a phantom wake, and later messages past the
   stale mark are still reported.
3. **Exit 2 does not strand.** A timeout is not a consumption: the message
   survives it and the next watch wakes for it.

Every test here is timing-free by construction: an already-unread inbox wakes in
the pre-loop check before any poll, and ``--timeout 0`` reaches the deadline
after exactly one non-sleeping pass. No wall-clock sleeps, no threads.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from claude_teams.cli import app
from claude_teams.messaging import unread_sender_counts

runner = CliRunner()

READER = "team-lead"


def _inbox(session_dir: Path) -> Path:
    return session_dir / f"inbox-{READER}.jsonl"


def _cursor(session_dir: Path) -> Path:
    return session_dir / f"inbox-{READER}.pos.json"


def _append(session_dir: Path, sender: str, text: str) -> None:
    line = json.dumps({"from": sender, "text": text, "ts": "2026-01-01T00:00:00+00:00"})
    with _inbox(session_dir).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _watch(session_dir: Path, *extra: str):
    return runner.invoke(app, ["watch", str(session_dir), "--timeout", "0", *extra])


# ==========================================================================
# 1. Wake without consume
# ==========================================================================


def test_wake_reports_the_sender_without_consuming_it(tmp_path: Path) -> None:
    _append(tmp_path, "worker-b", "the message")

    result = _watch(tmp_path)

    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["reason"] == "message"
    assert payload["from"] == ["worker-b"]
    # No cursor was written, so nothing was consumed.
    assert not _cursor(tmp_path).exists()
    assert unread_sender_counts(_inbox(tmp_path), _cursor(tmp_path)) == {"worker-b": 1}


def test_a_second_watch_wakes_for_the_same_unconsumed_message(tmp_path: Path) -> None:
    """A wake is a notification, not a delivery. Only the reader consumes."""
    _append(tmp_path, "worker-b", "the message")

    first = _watch(tmp_path)
    second = _watch(tmp_path)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert json.loads(second.stdout.strip().splitlines()[-1])["from"] == ["worker-b"]


def test_wake_does_not_wake_again_once_the_reader_has_consumed(tmp_path: Path) -> None:
    """The other half of the contract: a consumed message stops waking."""
    _append(tmp_path, "worker-b", "the message")
    assert _watch(tmp_path).exit_code == 0

    _cursor(tmp_path).write_text(json.dumps({"worker-b": 1}), encoding="utf-8")

    assert _watch(tmp_path).exit_code == 2


# ==========================================================================
# 2. Cursor clamping
# ==========================================================================


def test_cursor_beyond_the_count_clamps_to_zero_unread(tmp_path: Path) -> None:
    """Kill purges inbox lines but a stale cursor can outlive them (PR #30)."""
    _append(tmp_path, "worker-b", "one")
    _cursor(tmp_path).write_text(json.dumps({"worker-b": 5}), encoding="utf-8")

    assert unread_sender_counts(_inbox(tmp_path), _cursor(tmp_path)) == {}
    # And therefore no phantom wake.
    assert _watch(tmp_path).exit_code == 2


def test_messages_past_a_stale_cursor_are_still_reported(tmp_path: Path) -> None:
    """Clamping must not become a permanent mute for that sender."""
    _append(tmp_path, "worker-b", "one")
    _cursor(tmp_path).write_text(json.dumps({"worker-b": 2}), encoding="utf-8")
    for text in ("two", "three"):
        _append(tmp_path, "worker-b", text)

    assert unread_sender_counts(_inbox(tmp_path), _cursor(tmp_path)) == {"worker-b": 1}
    assert _watch(tmp_path).exit_code == 0


def test_clamping_is_per_sender(tmp_path: Path) -> None:
    """One sender's stale cursor must not suppress another's unread message."""
    _append(tmp_path, "worker-b", "one")
    _append(tmp_path, "worker-c", "one")
    _cursor(tmp_path).write_text(json.dumps({"worker-b": 9}), encoding="utf-8")

    assert unread_sender_counts(_inbox(tmp_path), _cursor(tmp_path)) == {"worker-c": 1}
    result = _watch(tmp_path)
    assert result.exit_code == 0
    assert json.loads(result.stdout.strip().splitlines()[-1])["from"] == ["worker-c"]


# ==========================================================================
# 3. Exit 2 does not strand
# ==========================================================================


def test_timeout_with_an_empty_inbox_then_a_message_wakes(tmp_path: Path) -> None:
    assert _watch(tmp_path).exit_code == 2

    _append(tmp_path, "worker-b", "arrived after the timeout")

    assert _watch(tmp_path).exit_code == 0


def test_a_timeout_that_ignored_the_inbox_does_not_consume_it(tmp_path: Path) -> None:
    """``--no-inbox`` is the cleanest way to force a timeout with mail waiting.

    Whatever the reason a watcher exits 2, the unread message must survive it —
    that is what makes exit 2 "re-check", not "lost".
    """
    _append(tmp_path, "worker-b", "the message")

    timed_out = _watch(tmp_path, "--no-inbox")

    assert timed_out.exit_code == 2
    assert not _cursor(tmp_path).exists()
    assert _watch(tmp_path).exit_code == 0


def test_timeout_writes_no_cursor_file(tmp_path: Path) -> None:
    assert _watch(tmp_path).exit_code == 2
    assert not _cursor(tmp_path).exists()
