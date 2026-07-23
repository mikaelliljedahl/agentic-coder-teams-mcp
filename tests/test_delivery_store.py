"""B1 — the delivery status store: schema, namespacing, and atomicity.

The store is the sender's audit trail. Three properties here are load-bearing
and each is asserted directly rather than through the server:

- **Cross-process, not merely cross-thread.** Several per-agent MCP servers
  share one session dir, so the store takes the same file-lock transaction
  model the registry uses. ``_inbox_lock`` is a ``threading.Lock`` and would
  not have been enough.
- **Records are never deleted.** Kill purges inbox lines; it must not purge
  these, or the settled outcome R4 requires would vanish exactly when the
  target does.
- **Keys are namespaced per sender.** Two senders may legitimately reuse one
  textual key.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from claude_teams import delivery_store as ds

SENDER = "team-lead"
OTHER = "worker-a"


def _record(**overrides: object) -> dict:
    base = ds.new_record(
        sender=SENDER,
        idempotency_key="k1",
        to="worker",
        fingerprint="fp",
        created_at=100.0,
    )
    base.update(overrides)
    return base


# ==========================================================================
# Idempotency key validation (before any waiting)
# ==========================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ds.KEY_REQUIRED),
        ("", ds.KEY_REQUIRED),
        ("   ", ds.KEY_REQUIRED),
        (42, ds.KEY_MALFORMED),
        ("has space", ds.KEY_MALFORMED),
        ("has/slash", ds.KEY_MALFORMED),
        ("tab\tchar", ds.KEY_MALFORMED),
        ("new\nline", ds.KEY_MALFORMED),
        ("-leading-punctuation", ds.KEY_MALFORMED),
        ("x" * (ds.MAX_IDEMPOTENCY_KEY_LENGTH + 1), ds.KEY_TOO_LONG),
    ],
)
def test_invalid_idempotency_keys_are_named_precisely(
    value: object, expected: str
) -> None:
    """Each rejection reason is distinct: the caller must know what to fix."""
    assert ds.validate_idempotency_key(value) == expected


@pytest.mark.parametrize(
    "value",
    ["k1", "deploy-2026-07-19", "a.b:c_d-1", "x" * ds.MAX_IDEMPOTENCY_KEY_LENGTH],
)
def test_valid_idempotency_keys_are_accepted(value: str) -> None:
    assert ds.validate_idempotency_key(value) is None


# ==========================================================================
# Fingerprinting — same key + differing field must be detectable
# ==========================================================================


def test_fingerprint_is_stable_across_option_ordering() -> None:
    """Option order is not a difference; the caller did not change anything."""
    left = ds.request_fingerprint(
        to="worker", prompt="go", options={"a": 1, "b": False}
    )
    right = ds.request_fingerprint(
        to="worker", prompt="go", options={"b": False, "a": 1}
    )
    assert left == right


@pytest.mark.parametrize(
    ("to", "prompt", "options"),
    [
        ("other", "go", {"replace_if_idle": True}),
        ("worker", "go elsewhere", {"replace_if_idle": True}),
        ("worker", "go", {"replace_if_idle": False}),
    ],
)
def test_fingerprint_changes_when_any_field_differs(
    to: str, prompt: str, options: dict
) -> None:
    baseline = ds.request_fingerprint(
        to="worker", prompt="go", options={"replace_if_idle": True}
    )
    assert ds.request_fingerprint(to=to, prompt=prompt, options=options) != baseline


# ==========================================================================
# Namespacing and persistence
# ==========================================================================


def test_two_senders_may_reuse_one_textual_key(tmp_path: Path) -> None:
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record())
        txn.put(_record(sender=OTHER, message_id="m2"))

    with ds.delivery_transaction(path) as txn:
        mine = txn.get(SENDER, "k1")
        theirs = txn.get(OTHER, "k1")
    assert mine is not None
    assert theirs is not None
    assert mine["message_id"] != theirs["message_id"]


def test_a_sender_cannot_read_another_senders_record(tmp_path: Path) -> None:
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record())

    with ds.delivery_transaction(path) as txn:
        assert txn.get(OTHER, "k1") is None


def test_records_survive_a_reload(tmp_path: Path) -> None:
    """Restart persistence: the store is a file, not process memory."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record(status=ds.STATUS_DELIVERED, phase=ds.PHASE_SETTLED))

    assert ds.load_records(path)[ds.record_key(SENDER, "k1")]["status"] == (
        ds.STATUS_DELIVERED
    )


def test_a_missing_store_reads_as_empty(tmp_path: Path) -> None:
    """Absence is the one legitimately-empty case, and stays empty."""
    assert ds.load_records(tmp_path / ds.DELIVERIES_FILE_NAME) == {}


def test_a_corrupt_store_is_unknown_state_and_not_an_empty_one(
    tmp_path: Path,
) -> None:
    """An error is not an absence.

    Reading corruption as ``{}`` let the next dirty transaction atomically
    REPLACE the store — erasing every prior audit row — and let a key whose row
    could not be read look unused, authorizing a second delivery under an
    idempotency key that already had an attempt.
    """
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ds.DeliveryStoreUnreadableError):
        ds.load_records(path)


def test_a_corrupt_store_is_not_silently_overwritten_by_the_next_write(
    tmp_path: Path,
) -> None:
    """The consequence the loader fix exists to prevent, asserted end to end."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    original = "{not json but precious"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ds.DeliveryStoreError), ds.delivery_transaction(path) as txn:
        txn.put(_record())

    assert path.read_text(encoding="utf-8") == original, (
        "a transaction over an unreadable store replaced the audit trail"
    )


def test_a_malformed_row_does_not_vanish_from_an_otherwise_readable_store(
    tmp_path: Path,
) -> None:
    """Dropping the bad entry hides exactly the row a caller is deciding on."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    path.write_text(json.dumps({"lead|k-1": "not-a-record"}), encoding="utf-8")

    with pytest.raises(ds.DeliveryStoreUnreadableError):
        ds.load_records(path)


def test_an_unreadable_store_is_distinguished_from_an_absent_one(
    tmp_path: Path,
) -> None:
    """Induced ``OSError`` on read, not a patched ``load_records``."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    path.mkdir()  # a directory where the store should be: read raises OSError

    with pytest.raises(ds.DeliveryStoreUnreadableError):
        ds.load_records(path)


def test_transaction_does_not_rewrite_the_file_when_nothing_changed(
    tmp_path: Path,
) -> None:
    """A pure read must not churn the store — ``deliver_pending`` polls it."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record())
    before = path.read_bytes()

    with ds.delivery_transaction(path) as txn:
        txn.get(SENDER, "k1")

    assert path.read_bytes() == before


# ==========================================================================
# The public query contract
# ==========================================================================


def test_public_view_exposes_exactly_the_query_contract_fields(
    tmp_path: Path,
) -> None:
    """No internals leak: ``fingerprint`` in particular is a comparison key,
    not something a caller should learn or depend on."""
    view = ds.public_view(_record(nonce="abc", attempts=2))
    assert set(view) == {
        "message_id",
        "idempotency_key",
        "to",
        "status",
        "phase",
        "reason",
        "attempts",
        "nonce",
        "created_at",
        "settled_at",
    }


def test_a_new_record_starts_queued_pending_and_unsettled() -> None:
    record = _record()
    assert record["status"] == ds.STATUS_QUEUED
    assert record["phase"] == ds.PHASE_PENDING
    assert record["settled_at"] is None
    assert record["attempts"] == 0


def test_list_for_sender_is_scoped_and_ordered(tmp_path: Path) -> None:
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record(idempotency_key="b", created_at=2.0, message_id="mb"))
        txn.put(_record(idempotency_key="a", created_at=1.0, message_id="ma"))
        txn.put(_record(sender=OTHER, idempotency_key="c", message_id="mc"))

    rows = ds.list_for_sender(path, SENDER)
    assert [row["idempotency_key"] for row in rows] == ["a", "b"]


def test_list_for_sender_can_filter_by_target(tmp_path: Path) -> None:
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    with ds.delivery_transaction(path) as txn:
        txn.put(_record(idempotency_key="a", to="worker", message_id="ma"))
        txn.put(_record(idempotency_key="b", to="other", message_id="mb"))

    rows = ds.list_for_sender(path, SENDER, to="other")
    assert [row["idempotency_key"] for row in rows] == ["b"]


# ==========================================================================
# Terminal transitions
# ==========================================================================


def test_settling_stamps_the_terminal_status_phase_and_time() -> None:
    record = _record(phase=ds.PHASE_SENT)
    ds.settle(record, ds.STATUS_FAILED, reason="not_delivered", now=500.0)
    assert record["status"] == ds.STATUS_FAILED
    assert record["phase"] == ds.PHASE_SETTLED
    assert record["reason"] == "not_delivered"
    assert record["settled_at"] == 500.0


def test_a_settled_record_is_terminal_and_reports_itself_so() -> None:
    record = _record()
    assert ds.is_terminal(record) is False
    ds.settle(record, ds.STATUS_DELIVERED, reason="", now=1.0)
    assert ds.is_terminal(record) is True


def test_settle_never_overwrites_an_already_terminal_record() -> None:
    """A late reconciliation must not turn a delivered message into failed.

    This is the false-status bug the whole feature exists to remove, in
    miniature: once ``delivered`` is recorded it is the truth.
    """
    record = _record()
    ds.settle(record, ds.STATUS_DELIVERED, reason="", now=1.0)
    ds.settle(record, ds.STATUS_FAILED, reason="not_delivered", now=2.0)
    assert record["status"] == ds.STATUS_DELIVERED
    assert record["settled_at"] == 1.0


# ==========================================================================
# Cross-PROCESS atomicity
# ==========================================================================


_WRITER = """
import sys
from pathlib import Path
from claude_teams import delivery_store as ds

path = Path(sys.argv[1])
sender = sys.argv[2]
for index in range(40):
    with ds.delivery_transaction(path) as txn:
        txn.put(
            ds.new_record(
                sender=sender,
                idempotency_key=f"k{index}",
                to="worker",
                fingerprint="fp",
                created_at=float(index),
            )
        )
"""


def test_two_processes_writing_the_store_lose_nothing(tmp_path: Path) -> None:
    """Not two threads — two OS processes, which is the real deployment.

    Every per-agent MCP server is its own process against one session dir, so
    an in-process lock would pass a threaded test and still corrupt this.
    """
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    script = tmp_path / "writer.py"
    script.write_text(_WRITER, encoding="utf-8")

    # S603 is suppressed below: a fixed interpreter running a script this test just
    # wrote, with paths it controls. There is no untrusted input here.
    procs = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, str(script), str(path), sender],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        for sender in ("alpha", "beta")
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    records = ds.load_records(path)
    assert len(records) == 80, "a lost update means the file lock is not doing its job"
    assert json.loads(path.read_text(encoding="utf-8")), "the file is still valid JSON"
