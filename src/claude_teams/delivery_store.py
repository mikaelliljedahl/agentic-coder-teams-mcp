"""B1 — the sender's durable delivery/audit store (R4).

This is deliberately **not** the inbox, and the difference is the whole point.

The inbox is live actionable state: since PR #30, killing an agent purges that
agent's lines from every reader's inbox so a same-named successor does not
inherit them. The delivery store is the *sender's* audit trail, and its entire
value is that a settled outcome stays queryable after the target is gone. So:

- **Records here are never deleted.** Not by kill, not by cleanup of the
  agent record. Extending the inbox's delete-on-kill to this store would
  destroy exactly the evidence R4 requires.
- **Guaranteed-path messages live here and only here.** They never enter the
  actionable inbox, because a resumed prompt plus an unread inbox copy lets a
  polling worker act on one instruction twice (B4).

Three structural rules:

1. **The key is caller-supplied, and required.** Generating it server-side
   would recreate the problem it exists to solve: a sender whose response was
   lost would have no id to ask about, because the id would only ever have
   arrived in the lost response. So the sender chooses it *before* the call.
2. **The namespace is (sender, key), not key.** Two senders may reuse one
   textual key without colliding, and no sender can read another's record.
3. **Terminal is terminal.** :func:`settle` refuses to overwrite an already
   settled record. A late reconciliation must never turn ``delivered`` into
   ``failed`` — that is the false-status bug this feature exists to remove.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast

from claude_teams.filelock import file_lock

#: Per-session store file, beside ``agents.json`` and ``operation-leases.json``.
DELIVERIES_FILE_NAME = "deliveries.json"
#: Its lock sidecar. Separate from the data file so an atomic replace of the
#: data cannot invalidate a lock another process is holding.
DELIVERIES_LOCK_NAME = "deliveries.lock"

#: R4's three public statuses. There is no fourth: transport progress is a
#: ``phase`` beneath ``queued`` so "sent" can never be read as "arrived".
STATUS_QUEUED = "queued"
STATUS_DELIVERED = "delivered"
STATUS_FAILED = "failed"

#: Phases beneath ``queued``.
PHASE_PENDING = "pending"
PHASE_SENT = "sent"
PHASE_UNCONFIRMED = "unconfirmed"
#: The phase of a terminal record. Named rather than blank so a caller reading
#: ``phase`` never has to special-case an empty string.
PHASE_SETTLED = "settled"

#: Terminal failure reason for definite non-delivery (dead child, no receipt).
REASON_NOT_DELIVERED = "not_delivered"

#: Optional provenance on a row settled from ANOTHER key's receipt: the same
#: request retried under a new key, aliased onto the attempt whose nonce is the
#: evidence. Absent on every ordinary row.
RECONCILED_FROM_FIELD = "reconciled_from_key"

#: Refusal when a key is reused with any differing request field.
IDEMPOTENCY_CONFLICT = "idempotency_conflict"

KEY_REQUIRED = "idempotency_key_required"
KEY_MALFORMED = "idempotency_key_malformed"
KEY_TOO_LONG = "idempotency_key_too_long"

MAX_IDEMPOTENCY_KEY_LENGTH = 128

#: Conservative on purpose. The key ends up in a JSON object key and in log
#: lines, so control characters, separators and whitespace are excluded rather
#: than escaped.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@=+-]*$")

#: Record-key separator. ``|`` cannot appear in a validated key, so the join is
#: unambiguous without escaping.
_KEY_SEPARATOR = "|"


class DeliveryStoreError(RuntimeError):
    """The store could not be persisted, so nothing may rely on it.

    Raised — never swallowed — because the durable row is the whole recovery
    story. A caller that also loses its response has no status and no
    recoverable key when this write is lost, which is precisely the hole the
    caller-supplied idempotency key exists to close (R4/B0). Returning
    normally here would report a delivery as underway while the only handle on
    it was thrown away.
    """

    def __init__(self, path: object = "") -> None:
        """Name the store that could not be written."""
        super().__init__(
            f"delivery store {path} could not be persisted; "
            "the in-memory changes were discarded"
        )


class DeliveryStoreUnreadableError(DeliveryStoreError):
    """The store exists but its contents are unknown, so nothing may proceed.

    A subclass of :class:`DeliveryStoreError` on purpose: every call site that
    already fails closed on a lost *write* must fail closed on an unreadable
    *read* for the same reason. Treating either as "the store is empty" lets a
    duplicate delivery through and lets the next write erase the audit trail.
    """

    def __init__(self, path: object = "") -> None:
        """Name the store that could not be read."""
        RuntimeError.__init__(
            self,
            f"delivery store {path} exists but could not be read or parsed; "
            "its contents are unknown and must not be treated as empty",
        )


def validate_idempotency_key(value: object) -> str | None:
    """Return the error code for ``value``, or ``None`` when it is usable.

    Called **before** anything is created and before any waiting, so a caller
    that got the key wrong learns immediately instead of after the budget.
    """
    if value is None:
        return KEY_REQUIRED
    if not isinstance(value, str):
        return KEY_MALFORMED
    if not value.strip():
        return KEY_REQUIRED
    if len(value) > MAX_IDEMPOTENCY_KEY_LENGTH:
        return KEY_TOO_LONG
    if not _KEY_RE.match(value):
        return KEY_MALFORMED
    return None


def record_key(sender: str, idempotency_key: str) -> str:
    """Return the store key for one sender's idempotency key."""
    return f"{sender}{_KEY_SEPARATOR}{idempotency_key}"


def request_fingerprint(*, to: str, prompt: str, options: dict[str, Any]) -> str:
    """Return a stable digest of everything a same-key retry must match.

    Sorted keys, so option ordering is not mistaken for a changed request; any
    genuine difference in recipient, prompt or options changes the digest and
    therefore yields ``idempotency_conflict`` rather than a second delivery.
    """
    payload = json.dumps(
        {"to": to, "prompt": prompt, "options": options},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_record(
    *,
    sender: str,
    idempotency_key: str,
    to: str,
    fingerprint: str,
    created_at: float,
) -> dict[str, Any]:
    """Build the durable record that is written **before** any waiting starts.

    Creating it first is what makes response loss survivable: if the client
    times out, cancels, or this process dies after creation, the sender can
    still ask ``delivery_status(key)`` what happened.
    """
    return {
        "message_id": uuid.uuid4().hex,
        "idempotency_key": idempotency_key,
        "sender": sender,
        "to": to,
        "status": STATUS_QUEUED,
        "phase": PHASE_PENDING,
        "reason": "",
        "attempts": 0,
        "nonce": "",
        "created_at": created_at,
        "settled_at": None,
        "fingerprint": fingerprint,
        "operation_id": "",
        "prompt_file": "",
    }


def is_terminal(record: dict[str, Any]) -> bool:
    """Whether this record has settled and may never change again."""
    return record.get("status") in {STATUS_DELIVERED, STATUS_FAILED}


def settle(
    record: dict[str, Any], status: str, *, reason: str, now: float
) -> dict[str, Any]:
    """Move ``record`` to a terminal status, unless it is already terminal.

    The no-overwrite rule is load-bearing. Kill-time cleanup, a late
    reconciliation and a retry can all reach a record that another path has
    already settled; whichever of them ran second must not contradict the
    first, and ``delivered`` in particular is proof that cannot be withdrawn.
    """
    if is_terminal(record):
        return record
    record["status"] = status
    record["phase"] = PHASE_SETTLED
    record["reason"] = reason
    record["settled_at"] = now
    return record


def mark_phase(record: dict[str, Any], phase: str, *, reason: str = "") -> None:
    """Set a non-terminal phase beneath ``queued``, leaving terminals alone."""
    if is_terminal(record):
        return
    record["status"] = STATUS_QUEUED
    record["phase"] = phase
    record["reason"] = reason


def public_view(record: dict[str, Any]) -> dict[str, Any]:
    """Project a record onto exactly the documented query contract.

    ``fingerprint``, ``sender``, ``operation_id`` and ``prompt_file`` are
    internal: the first is a comparison key callers must not depend on, and
    the rest are transport bookkeeping.

    ``reconciled_from_key`` is the one optional member of the contract. It
    appears only on a row settled from ANOTHER key's receipt — a retry of the
    same request under a new key — and names the key whose nonce is the
    evidence. Without it such a row would read as ``delivered`` with an empty
    nonce and no way to see why.
    """
    view = {
        "message_id": record.get("message_id", ""),
        "idempotency_key": record.get("idempotency_key", ""),
        "to": record.get("to", ""),
        "status": record.get("status", STATUS_QUEUED),
        "phase": record.get("phase", PHASE_PENDING),
        "reason": record.get("reason", ""),
        "attempts": record.get("attempts", 0),
        "nonce": record.get("nonce", ""),
        "created_at": record.get("created_at"),
        "settled_at": record.get("settled_at"),
    }
    provenance = record.get(RECONCILED_FROM_FIELD)
    if provenance:
        view[RECONCILED_FROM_FIELD] = provenance
    return view


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load the store. Absence is empty; unreadable or malformed is an error.

    The distinction is the whole contract. "No file yet" is a legitimately empty
    store and a first write may create it. "The file is there but I could not
    read or parse it" is *unknown* state, and returning ``{}`` for it — which is
    what this used to do — makes every caller act on a lie:

    - the next dirty transaction atomically **replaces** the store, erasing
      every prior audit row;
    - a key whose row cannot be read looks unused, so a duplicate delivery is
      authorized under an idempotency key that already has an attempt;
    - a settled outcome the sender is entitled to recover simply disappears.

    So an unreadable or malformed store raises, and every caller inherits the
    fail-closed behaviour it already has for a failed *write*. An error is not
    an absence.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise DeliveryStoreUnreadableError(path) from exc
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DeliveryStoreUnreadableError(path) from exc
    if not isinstance(raw, dict):
        raise DeliveryStoreUnreadableError(path)
    rows: dict[str, dict[str, Any]] = {}
    for key, value in cast("dict[str, Any]", raw).items():
        if not isinstance(key, str) or not isinstance(value, dict):
            # Dropping the malformed entry would hide exactly one row — most
            # likely the one a caller is about to decide something about.
            raise DeliveryStoreUnreadableError(path)
        rows[key] = cast("dict[str, Any]", value)
    return rows


def save_records(path: Path, data: dict[str, dict[str, Any]]) -> bool:
    """Atomically persist the store (temp file + replace).

    Returns whether the store actually reached disk. Swallowing the failure
    here was the defect: the row written *before* any waiting is the sender's
    only handle on an in-flight message, so a lost write plus a lost response
    leaves neither a reliable status nor a recoverable key. Mirrors
    ``leases.save_leases``, for the same reason and with the same obligation on
    callers: fail closed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False
    return True


class DeliveryTransaction:
    """The mutable view handed out inside :func:`delivery_transaction`."""

    def __init__(self, data: dict[str, dict[str, Any]]) -> None:
        """Wrap the loaded store; ``dirty`` decides whether it is rewritten."""
        self.data = data
        self.dirty = False

    def get(self, sender: str, idempotency_key: str) -> dict[str, Any] | None:
        """Return this sender's record for ``idempotency_key``, if any."""
        return self.data.get(record_key(sender, idempotency_key))

    def put(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace ``record`` and mark the transaction dirty."""
        key = record_key(
            str(record.get("sender") or ""), str(record["idempotency_key"])
        )
        self.data[key] = record
        self.dirty = True
        return record

    def touch(self) -> None:
        """Mark the transaction dirty after mutating a record in place."""
        self.dirty = True

    def for_sender(self, sender: str, to: str | None = None) -> list[dict[str, Any]]:
        """Return this sender's records, oldest first."""
        rows = [
            record
            for record in self.data.values()
            if record.get("sender") == sender and (to is None or record.get("to") == to)
        ]
        rows.sort(
            key=lambda record: (
                _as_float(record.get("created_at")),
                str(record.get("message_id")),
            )
        )
        return rows


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


@contextmanager
def delivery_transaction(path: Path) -> Iterator[DeliveryTransaction]:
    """Read-modify-write the store under the cross-process file lock.

    The whole body runs under the lock, so a check-then-act sequence (does this
    key exist? if not, create it) is atomic against another MCP server doing
    the same thing in another process. The file is rewritten only when
    something actually changed, so a pure read — which ``deliver_pending`` and
    ``delivery_status`` both perform — does not churn the store.

    Raises :class:`DeliveryStoreError` when a dirty transaction could not be
    written. Callers must treat that as "this did not happen": the row a
    delivery is about to rely on is not on disk, so no resume may begin.
    """
    lock_path = path.with_name(DELIVERIES_LOCK_NAME)
    with file_lock(lock_path):
        txn = DeliveryTransaction(load_records(path))
        yield txn
        if txn.dirty and not save_records(path, txn.data):
            raise DeliveryStoreError(path)


def list_for_sender(
    path: Path, sender: str, to: str | None = None
) -> list[dict[str, Any]]:
    """Return a sender's records as public views, oldest first.

    A convenience list only. It deliberately cannot serve response-loss
    recovery: with several messages targeting one agent there is no way to say
    which row is the one whose response was lost. That is what the idempotency
    key is for.
    """
    with delivery_transaction(path) as txn:
        return [public_view(record) for record in txn.for_sender(sender, to)]


__all__ = [
    "DELIVERIES_FILE_NAME",
    "DELIVERIES_LOCK_NAME",
    "IDEMPOTENCY_CONFLICT",
    "KEY_MALFORMED",
    "KEY_REQUIRED",
    "KEY_TOO_LONG",
    "MAX_IDEMPOTENCY_KEY_LENGTH",
    "PHASE_PENDING",
    "PHASE_SENT",
    "PHASE_SETTLED",
    "PHASE_UNCONFIRMED",
    "REASON_NOT_DELIVERED",
    "RECONCILED_FROM_FIELD",
    "STATUS_DELIVERED",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "DeliveryStoreError",
    "DeliveryStoreUnreadableError",
    "DeliveryTransaction",
    "delivery_transaction",
    "is_terminal",
    "list_for_sender",
    "load_records",
    "mark_phase",
    "new_record",
    "public_view",
    "record_key",
    "request_fingerprint",
    "save_records",
    "settle",
    "validate_idempotency_key",
]
