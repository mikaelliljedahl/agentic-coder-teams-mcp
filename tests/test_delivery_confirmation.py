"""Tests for A3/A4: child-liveness early failure and nonce delivery confirmation.

Every timing case here uses an **injected clock and poll interval**. Nothing in
this file sleeps on the wall clock: the repo already carries one flaky
wall-clock watcher test and confirmation is far too timing-dense to add more.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from claude_teams.delivery import (
    DELIVERY_AMBIGUOUS,
    DELIVERY_DELIVERED,
    DELIVERY_FAILED,
    DELIVERY_UNCONFIRMED,
    SCAN_AMBIGUOUS,
    SCAN_FOUND,
    SCAN_PENDING,
    ReceiptScanner,
    confirm_delivery,
    delivered_prompt,
    delivery_marker,
    delivery_marker_token,
    new_delivery_nonce,
    receipt_nonces,
)

NONCE = "0123456789abcdef0123456789abcdef"
OTHER_NONCE = "fedcba9876543210fedcba9876543210"


class _Clock:
    """Injected monotonic clock; ``sleep`` advances it instead of blocking."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_raw(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _claude_user_text(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _claude_tool_result(text: str) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": text}],
        },
    }


def _claude_assistant(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _codex_user(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _codex_assistant(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _scanner(
    path: Path,
    backend: str = "claude-code",
    *,
    successors: Callable[[], list[Path]] | None = None,
) -> ReceiptScanner:
    return ReceiptScanner(
        path,
        backend=backend,
        backend_session_id="sess",
        successors=successors,
    )


# --------------------------------------------------------------------------
# Marker grammar
# --------------------------------------------------------------------------


def test_new_delivery_nonces_are_unique_and_high_entropy() -> None:
    nonces = {new_delivery_nonce() for _ in range(200)}

    assert len(nonces) == 200
    assert all(len(value) >= 32 for value in nonces)


def test_delivered_prompt_carries_exactly_one_marker() -> None:
    single = delivered_prompt("do stuff", NONCE, single_line=True)
    multi = delivered_prompt("do stuff", NONCE, single_line=False)

    assert single == f"do stuff {delivery_marker(NONCE)}"
    assert multi == f"do stuff\n\n{delivery_marker(NONCE)}"
    assert "\n" not in single
    assert single.count(delivery_marker_token(NONCE)) == 1


@pytest.mark.parametrize(
    "text",
    [
        # User text that LOOKS like a marker: the prefix alone must never match.
        "wat-deliver:",
        "wat-deliver:not-a-nonce",
        f"wat-deliver:{NONCE[:16]}",
        # Longer/adjacent hex run: a prefix of a longer id is not this id.
        f"wat-deliver:{NONCE}0",
        f"wat-deliver:0{NONCE}",
    ],
)
def test_only_the_full_random_id_matches(text: str) -> None:
    record = _claude_user_text(f"please note {text} in your reply")

    assert NONCE not in receipt_nonces(record, "claude-code")


def test_full_marker_in_user_text_matches() -> None:
    record = _claude_user_text(delivered_prompt("go", NONCE, single_line=True))

    assert receipt_nonces(record, "claude-code") == {NONCE}


# --------------------------------------------------------------------------
# Named receipt records — semantic, not substring
# --------------------------------------------------------------------------


def test_claude_tool_result_is_a_receipt_record() -> None:
    """A sidecar spawn's receipt is the tool result for the file read."""
    record = _claude_tool_result(f"task body\n\n{delivery_marker(NONCE)}")

    assert receipt_nonces(record, "claude-code") == {NONCE}


def test_claude_assistant_message_is_not_a_receipt_record() -> None:
    record = _claude_assistant(f"I see {delivery_marker(NONCE)}")

    assert receipt_nonces(record, "claude-code") == set()


def test_codex_user_input_is_a_receipt_record() -> None:
    record = _codex_user(f"task {delivery_marker(NONCE)}")

    assert receipt_nonces(record, "codex") == {NONCE}


def test_codex_assistant_output_is_not_a_receipt_record() -> None:
    record = _codex_assistant(f"echoing {delivery_marker(NONCE)}")

    assert receipt_nonces(record, "codex") == set()


@pytest.mark.parametrize(
    "record",
    [
        # A CLI diagnostic line.
        {"type": "system", "content": f"resume failed: {delivery_marker(NONCE)}"},
        # Serialized argv recorded as a tool invocation, not user input.
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": f"claude -p '{delivery_marker(NONCE)}'"},
                    }
                ],
            },
        },
    ],
)
def test_diagnostics_and_argv_do_not_confirm(record: dict) -> None:
    assert receipt_nonces(record, "claude-code") == set()


def test_scanner_ignores_a_nonce_that_only_appears_in_a_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("hello")])
    scanner = _scanner(path)
    scanner.snapshot()
    _append_raw(
        path,
        json.dumps({"type": "system", "content": delivery_marker(NONCE)}) + "\n",
    )

    assert scanner.poll(NONCE) == SCAN_PENDING


# --------------------------------------------------------------------------
# Record boundary, retained partial bytes
# --------------------------------------------------------------------------


def test_snapshot_starts_at_the_last_complete_record_not_raw_eof(
    tmp_path: Path,
) -> None:
    """A partial trailing record must be reconsidered once it completes.

    The readers skip malformed lines *permanently*, so a scan that started at
    raw EOF would consume an unparsable fragment and never revisit the
    completed record.
    """
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier turn")])
    # A half-written record: no trailing newline, invalid JSON on its own.
    partial = json.dumps(_claude_user_text(f"go {delivery_marker(NONCE)}"))
    head, tail = partial[:20], partial[20:]
    _append_raw(path, head)

    scanner = _scanner(path)
    scanner.snapshot()

    assert scanner.poll(NONCE) == SCAN_PENDING

    _append_raw(path, tail + "\n")

    assert scanner.poll(NONCE) == SCAN_FOUND


def test_partial_bytes_are_retained_between_polls(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()

    row = json.dumps(_claude_user_text(f"go {delivery_marker(NONCE)}"))
    for chunk_start in range(0, len(row), 7):
        _append_raw(path, row[chunk_start : chunk_start + 7])
        assert scanner.poll(NONCE) == SCAN_PENDING
    _append_raw(path, "\n")

    assert scanner.poll(NONCE) == SCAN_FOUND


# --------------------------------------------------------------------------
# Must NOT confirm
# --------------------------------------------------------------------------


def test_transcript_growth_alone_does_not_confirm(tmp_path: Path) -> None:
    """A surviving OLD process keeps writing; growth is not delivery."""
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()

    for index in range(5):
        _append_raw(
            path, json.dumps(_claude_assistant(f"still working {index}")) + "\n"
        )
        assert scanner.poll(NONCE) == SCAN_PENDING


def test_an_unrelated_user_turn_does_not_confirm(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    _append_raw(path, json.dumps(_claude_user_text("some other instruction")) + "\n")

    assert scanner.poll(NONCE) == SCAN_PENDING


def test_a_different_attempts_nonce_does_not_confirm(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    _append_raw(
        path,
        json.dumps(_claude_user_text(f"go {delivery_marker(OTHER_NONCE)}")) + "\n",
    )

    assert scanner.poll(NONCE) == SCAN_PENDING


def test_nonce_in_the_correct_transcript_confirms(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    _append_raw(
        path, json.dumps(_claude_user_text(f"go {delivery_marker(NONCE)}")) + "\n"
    )

    assert scanner.poll(NONCE) == SCAN_FOUND


# --------------------------------------------------------------------------
# Rotation / truncation / replacement
# --------------------------------------------------------------------------


def _successors(*paths: Path):
    return lambda: list(paths)


def test_rotation_is_followed_by_session_id_plus_file_identity(
    tmp_path: Path,
) -> None:
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    successor = tmp_path / "b.jsonl"
    scanner = _scanner(original, successors=_successors(successor))
    scanner.snapshot()

    original.unlink()
    _write(
        successor,
        [
            {"sessionId": "sess", **_claude_user_text("carried over")},
            {"sessionId": "sess", **_claude_user_text(f"go {delivery_marker(NONCE)}")},
        ],
    )

    assert scanner.poll(NONCE) == SCAN_FOUND


def test_a_successor_that_does_not_replay_the_initial_marker_is_still_followed(
    tmp_path: Path,
) -> None:
    """The correlation token corroborates; it is never a precondition.

    A successor may legitimately not replay the spawn marker, so requiring it
    would fail a delivery that genuinely landed.
    """
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    successor = tmp_path / "b.jsonl"
    scanner = _scanner(
        original,
        successors=_successors(successor),
    )
    scanner.snapshot()

    original.unlink()
    _write(
        successor,
        # Note: no "wat-corr:deadbeef" anywhere in this file.
        [{"sessionId": "sess", **_claude_user_text(f"go {delivery_marker(NONCE)}")}],
    )

    assert scanner.poll(NONCE) == SCAN_FOUND


def test_truncation_is_detected_and_rescanned_from_the_start(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    _write(
        path, [{"sessionId": "sess", **_claude_user_text("earlier")} for _ in range(6)]
    )
    scanner = _scanner(path, successors=_successors(path))
    scanner.snapshot()

    # Replaced in place with a shorter file: size regression at the same path.
    _write(
        path,
        [{"sessionId": "sess", **_claude_user_text(f"go {delivery_marker(NONCE)}")}],
    )

    assert scanner.poll(NONCE) == SCAN_FOUND


def test_two_candidate_successors_are_ambiguous_not_a_guess(tmp_path: Path) -> None:
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    first = tmp_path / "b.jsonl"
    second = tmp_path / "c.jsonl"
    for path in (first, second):
        _write(
            path,
            [
                {
                    "sessionId": "sess",
                    **_claude_user_text(f"go {delivery_marker(NONCE)}"),
                }
            ],
        )
    scanner = _scanner(original, successors=_successors(first, second))
    scanner.snapshot()
    original.unlink()

    assert scanner.poll(NONCE) == SCAN_AMBIGUOUS


def test_the_correlation_token_never_selects_between_two_successors(
    tmp_path: Path,
) -> None:
    """Corroboration, never selection.

    This test previously asserted the opposite — that a token carried by
    exactly one of two candidates picks that one. That is a guess: the token
    is written at spawn and a successor may or may not replay it, so its
    presence in one file is not evidence that the OTHER file is not the live
    conversation. Attributing a delivery on that basis is precisely the class
    of false receipt the whole feature exists to eliminate, so more than one
    candidate successor is unconditionally ``ambiguous``.
    """
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    marked = tmp_path / "b.jsonl"
    unmarked = tmp_path / "c.jsonl"
    _write(
        marked,
        [
            {"sessionId": "sess", **_claude_user_text("wat-corr:deadbeef")},
            {"sessionId": "sess", **_claude_user_text(f"go {delivery_marker(NONCE)}")},
        ],
    )
    _write(unmarked, [{"sessionId": "sess", **_claude_user_text("unrelated")}])
    scanner = _scanner(
        original,
        successors=_successors(marked, unmarked),
    )
    scanner.snapshot()
    original.unlink()

    assert scanner.poll(NONCE) == SCAN_AMBIGUOUS


def test_a_successor_with_a_different_session_id_is_not_a_candidate(
    tmp_path: Path,
) -> None:
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    foreign = tmp_path / "b.jsonl"
    _write(
        foreign,
        [{"sessionId": "other", **_claude_user_text(f"go {delivery_marker(NONCE)}")}],
    )
    scanner = _scanner(original, successors=_successors(foreign))
    scanner.snapshot()
    original.unlink()

    assert scanner.poll(NONCE) == SCAN_PENDING


# --------------------------------------------------------------------------
# confirm_delivery — the bound, and the two non-delivery cases (R6)
# --------------------------------------------------------------------------


def _confirm(scanner: ReceiptScanner, clock: _Clock, alive, *, bound: float = 5.0):
    return confirm_delivery(
        scanner,
        NONCE,
        child_alive=alive,
        bound_s=bound,
        poll_interval_s=1.0,
        clock=clock,
        sleep=clock.sleep,
    )


def test_child_that_exits_immediately_fails_fast(tmp_path: Path) -> None:
    """A3: an immediately-dead child is an early-failure signal, not evidence."""
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()

    outcome = _confirm(scanner, clock, lambda: False)

    assert outcome.status == DELIVERY_FAILED
    assert outcome.reason == "resume_not_confirmed"
    # Failed fast: it did not burn the whole bound waiting on a dead child.
    assert clock.now < 5.0


def test_child_dead_with_no_receipt_is_definite_non_delivery(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()
    # Alive for the settle window, then dies without ever writing a receipt.
    alive_calls = {"n": 0}

    def alive() -> bool:
        alive_calls["n"] += 1
        return alive_calls["n"] <= 3

    outcome = _confirm(scanner, clock, alive)

    assert outcome.status == DELIVERY_FAILED
    assert outcome.reason == "not_delivered"


def test_bound_expiry_with_a_live_child_is_not_terminal(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()

    outcome = _confirm(scanner, clock, lambda: True)

    assert outcome.status == DELIVERY_UNCONFIRMED
    assert outcome.reason == "scan_expired"
    assert clock.now >= 5.0


def test_a_later_flush_reconciles_an_unconfirmed_attempt(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()

    assert _confirm(scanner, clock, lambda: True).status == DELIVERY_UNCONFIRMED

    # The buffered transcript write lands after the call returned.
    _append_raw(
        path, json.dumps(_claude_user_text(f"go {delivery_marker(NONCE)}")) + "\n"
    )

    assert scanner.poll(NONCE) == SCAN_FOUND


def test_nonce_found_inside_the_bound_is_delivered(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()
    polls = {"n": 0}

    def alive() -> bool:
        polls["n"] += 1
        if polls["n"] == 2:
            _append_raw(
                path,
                json.dumps(_claude_user_text(f"go {delivery_marker(NONCE)}")) + "\n",
            )
        return True

    outcome = _confirm(scanner, clock, alive)

    assert outcome.status == DELIVERY_DELIVERED
    assert outcome.reason == ""


def test_ambiguous_successor_short_circuits_the_bound(tmp_path: Path) -> None:
    original = tmp_path / "a.jsonl"
    _write(original, [{"sessionId": "sess", **_claude_user_text("earlier")}])
    first = tmp_path / "b.jsonl"
    second = tmp_path / "c.jsonl"
    for path in (first, second):
        _write(path, [{"sessionId": "sess", **_claude_user_text("carried")}])
    scanner = _scanner(original, successors=_successors(first, second))
    scanner.snapshot()
    original.unlink()
    clock = _Clock()

    outcome = _confirm(scanner, clock, lambda: True)

    assert outcome.status == DELIVERY_AMBIGUOUS


def test_confirm_delivery_uses_the_injected_clock_only(tmp_path: Path) -> None:
    """Guards against a real ``time.sleep`` creeping back into the poll loop."""
    path = tmp_path / "t.jsonl"
    _write(path, [_claude_user_text("earlier")])
    scanner = _scanner(path)
    scanner.snapshot()
    clock = _Clock()

    _confirm(scanner, clock, lambda: True)

    assert clock.slept, "expected the injected sleep to be used"
    assert all(interval == 1.0 for interval in clock.slept)
