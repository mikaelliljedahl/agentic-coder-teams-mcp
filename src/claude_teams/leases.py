"""A4b — the per-target operation lease.

Confirmation (A4) polls a transcript for up to tens of seconds. It cannot run
inside ``_agents_transaction``: that holds a **cross-process** file lock for its
whole body, so a confirmation poll there would block every registry reader in
every MCP server on the machine. The lock therefore has to be released around
the resume-and-confirm window.

Compare-and-swap *after the fact* is not enough to make that safe. Two callers
can snapshot the same generation, both resume the same conversation, and both
deliver distinct nonces; the losing CAS cannot undo an irreversible side
effect. So a caller **atomically reserves a per-agent lease before releasing
the registry lock**. The lease does not itself resume — it only reserves the
right to. Finalization then CASes on generation **and** ``operation_id``.

Four properties are deliberate and each guards a specific failure:

- **A second valid caller QUEUES; it is never refused.** Per-target FIFO.
  Refusing a valid caller would hand back exactly the dead end R1 forbids.
  Refusal is reserved for an invalid caller/request or a genuine no-path state.
- **Crash-atomic storage, outside the registry.** ``_save_agents_unlocked``
  overwrites ``agents.json`` with ``write_text``, so a crash mid-write can
  destroy the registry *and* the lease. The file lock serializes writers but
  does not prevent a torn write. Leases live in their own file written with the
  temp-file + atomic-replace pattern ``messaging.save_inbox_cursors`` uses,
  which also keeps lease churn out of the registry's write path.
- **Expiry is not fencing, and neither is a bare PID.** A holder that is alive
  but slow *after spawning* would otherwise let a second caller observe expiry,
  fail to find a not-yet-flushed nonce, and retry into a delivery still in
  flight. Recovery therefore checks holder liveness; wall-clock expiry alone
  never justifies a resend. And because a dead holder's PID can be reused, the
  lease pairs ``holder_pid`` with ``holder_create_token`` and validates the
  pair fail-closed, exactly as ``process_manager.owns_process`` does.
- **``operation_id`` is in the CAS key, not generation alone.** A name reused
  after removal starts a fresh record that can legitimately be back at the same
  generation; only the per-attempt id keeps a stale finalize from updating it.

Every function here is pure with respect to concurrency: callers invoke them
while holding the registry lock, which is what makes reservation atomic.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from claude_teams.backends.process_manager import (
    OWNERSHIP_NOT_OURS,
)

#: Deliberately not ``agents.json``: see the module docstring on torn writes.
LEASES_FILE_NAME = "operation-leases.json"

#: The caller now holds the lease and may resume.
LEASE_GRANTED = "granted"
#: Another caller holds it. This caller is queued behind per-target FIFO and
#: must retry with its ticket — it is NOT refused.
LEASE_QUEUED = "queued"


@dataclass(frozen=True)
class Lease:
    """A reservation of the right to resume one agent."""

    agent: str
    generation: int
    operation_id: str
    backend_session_id: str
    nonce: str
    holder_pid: int
    holder_create_token: str | None
    deadline: float
    acquired_at: float


@dataclass(frozen=True)
class ReserveResult:
    """Outcome of a reservation attempt."""

    status: str
    lease: Lease | None = None
    ticket: str | None = None
    position: int = 0

    @property
    def granted(self) -> bool:
        """Whether this caller may proceed to resume."""
        return self.status == LEASE_GRANTED


class LeaseStoreError(RuntimeError):
    """The lease store could not be used, so no reservation may be trusted."""


class LeaseNotPersistedError(LeaseStoreError):
    """A lease mutation did not reach disk, so on-disk state is unchanged."""

    def __init__(self, path: object = "", agent: object = "") -> None:
        """Name the store and the agent whose lease is still held on disk."""
        super().__init__(
            f"lease store {path} could not be persisted while reclaiming "
            f"{agent!r}; the lease is still held on disk"
        )


class LeaseStoreUnreadableError(LeaseStoreError):
    """The lease store exists but its contents are unknown.

    Same rule as ``delivery_store.load_records``: reading an unreadable or
    corrupt lease file as an **empty** one reports every target as free, so a
    caller is granted a lease over a live holder and resumes a conversation
    somebody else is already resuming. That is the exact double delivery this
    module exists to prevent, arriving through the loader instead of the
    writer. Absence (no file yet) stays empty; anything else raises.
    """

    def __init__(self, path: object = "") -> None:
        """Name the store that could not be read."""
        super().__init__(
            f"lease store {path} exists but could not be read or parsed; "
            "its contents are unknown and must not be treated as empty"
        )


def load_leases(path: Path) -> dict[str, dict[str, Any]]:
    """Load the lease store. Absence is empty; unreadable/malformed raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise LeaseStoreUnreadableError(path) from exc
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LeaseStoreUnreadableError(path) from exc
    if not isinstance(raw, dict):
        raise LeaseStoreUnreadableError(path)
    entries: dict[str, dict[str, Any]] = {}
    for key, value in cast("dict[str, Any]", raw).items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise LeaseStoreUnreadableError(path)
        entries[key] = cast("dict[str, Any]", value)
    return entries


def save_leases(path: Path, data: dict[str, dict[str, Any]]) -> bool:
    """Atomically persist the lease store (temp file + replace).

    Returns whether the store actually reached disk. Callers **must** fail
    closed on ``False``: a lease nobody else can read is not a lease, and
    reporting ``granted`` after a failed write would let another process
    reserve the same target and resume the same conversation — the exact
    double delivery this module exists to prevent.
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


def _entry(data: dict[str, dict[str, Any]], agent: str) -> dict[str, Any]:
    entry = data.get(agent)
    if not isinstance(entry, dict):
        entry = {"lease": None, "waiters": []}
        data[agent] = entry
    entry.setdefault("lease", None)
    waiters = entry.get("waiters")
    if not isinstance(waiters, list):
        entry["waiters"] = []
    return entry


def _to_lease(agent: str, payload: object) -> Lease | None:
    """Parse a stored lease payload. ``None`` means *no lease*, never *unknown*.

    ``None``/absent is the legitimate "nothing held" case. A payload that is
    present but unparseable is corruption, and reading it as "no lease held"
    is the same fail-open the loader just closed, one level in: it would let a
    caller be granted a target that the unreadable bytes say is taken.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise LeaseStoreUnreadableError(agent)
    mapping = cast("dict[str, Any]", payload)
    try:
        return Lease(
            agent=agent,
            generation=int(mapping["generation"]),
            operation_id=str(mapping["operation_id"]),
            backend_session_id=str(mapping.get("backend_session_id") or ""),
            nonce=str(mapping.get("nonce") or ""),
            holder_pid=int(mapping["holder_pid"]),
            holder_create_token=(
                str(mapping["holder_create_token"])
                if mapping.get("holder_create_token")
                else None
            ),
            deadline=float(mapping.get("deadline") or 0.0),
            acquired_at=float(mapping.get("acquired_at") or 0.0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LeaseStoreUnreadableError(agent) from exc


def active_lease(path: Path, agent: str) -> Lease | None:
    """Return the lease currently recorded for ``agent``, if any.

    This is the *recorded* lease, with no liveness check: use
    :func:`reconcile_lease` when the answer must be "is it provably live".
    """
    entry = load_leases(path).get(agent)
    if not isinstance(entry, dict):
        return None
    return _to_lease(agent, entry.get("lease"))


def _holder_reclaimable(
    lease: Lease, holder_probe: Callable[[int, str | None], str]
) -> bool:
    """Whether ``lease``'s holder is **provably** gone, so it may be reclaimed.

    Three-valued on purpose (``process_manager.ownership_probe``). Only
    :data:`OWNERSHIP_NOT_OURS` — a dead PID, a reused PID, or a tokenless
    expectation — releases the lease. A holder whose creation token could not be
    read is alive-but-unprovable, and reclaiming on that authorizes a second
    caller to resume a conversation the first is still working. A probe that
    raises is likewise unknown, not gone.

    This is deliberately NOT the mirror of the destructive gate: killing must
    refuse on "unproven", and so must reclaiming — but they refuse in opposite
    directions, so they cannot share one bool.
    """
    try:
        outcome = holder_probe(lease.holder_pid, lease.holder_create_token)
    except OSError:
        return False
    # Whitelist, not blacklist: only a positive "provably not the holder"
    # reclaims. Anything else — ``ours``, ``indeterminate``, or an outcome this
    # version does not recognise — leaves the lease held.
    return outcome == OWNERSHIP_NOT_OURS


def reconcile_lease(
    path: Path,
    agent: str,
    *,
    holder_probe: Callable[[int, str | None], str],
    now: float | None = None,
) -> Lease | None:
    """Clear a lease whose holder is provably gone; return the surviving lease.

    ``now`` is accepted (and deliberately unused for the reclaim decision) to
    make the rule explicit at every call site: **wall-clock expiry alone never
    reclaims a lease.** Only a holder that is provably dead, or whose creation
    token no longer matches, releases it; one whose ownership merely could not
    be established stays held.

    Raises :class:`LeaseStoreError` when the reclaim could not be persisted. The
    caller asked "is a live lease held here", and after a failed write the
    answer on disk is still "yes" — returning ``None`` would let it kill or
    resume against a lease every other process can still see.
    """
    _ = now
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return None
    lease = _to_lease(agent, entry.get("lease"))
    if lease is None:
        return None
    if not _holder_reclaimable(lease, holder_probe):
        return lease
    entry["lease"] = None
    _drop_waiter(entry, lease.operation_id)
    if not save_leases(path, data):
        raise LeaseNotPersistedError(path, agent)
    return None


def _drop_waiter(entry: dict[str, Any], operation_id: str) -> None:
    waiters = entry.get("waiters")
    if isinstance(waiters, list):
        entry["waiters"] = [
            waiter
            for waiter in cast("list[Any]", waiters)
            if not (
                isinstance(waiter, dict)
                and cast("dict[str, Any]", waiter).get("operation_id") == operation_id
            )
        ]


def reserve_lease(  # noqa: PLR0913 - the lease's stored fields are its arguments.
    path: Path,
    agent: str,
    *,
    generation: int,
    operation_id: str,
    backend_session_id: str,
    nonce: str,
    holder_pid: int,
    holder_create_token: str | None,
    deadline: float,
    now: float,
    holder_probe: Callable[[int, str | None], str],
    ticket: str | None = None,
) -> ReserveResult:
    """Reserve the right to resume ``agent``, or take a FIFO place in line.

    Call this while holding the registry lock: that is what makes the
    read-decide-write sequence atomic against another server doing the same.

    A caller that is not granted the lease receives ``LEASE_QUEUED`` with a
    ``ticket``. Passing that ticket back on a later attempt keeps its place;
    a caller that jumps ahead of an earlier ticket is queued again rather than
    granted, so per-target ordering is genuinely FIFO.
    """
    data = load_leases(path)
    entry = _entry(data, agent)
    current = _to_lease(agent, entry.get("lease"))
    if current is not None and _holder_reclaimable(current, holder_probe):
        _drop_waiter(entry, current.operation_id)
        entry["lease"] = None
        current = None

    waiters = cast("list[Any]", entry["waiters"])
    if ticket is None:
        ticket = uuid.uuid4().hex
    existing = next(
        (
            cast("dict[str, Any]", waiter)
            for waiter in waiters
            if isinstance(waiter, dict)
            and cast("dict[str, Any]", waiter).get("ticket") == ticket
        ),
        None,
    )
    if existing is None:
        # Either a brand-new ticket, or one we no longer know about (store
        # reset, or already promoted and released): enter at the tail rather
        # than silently jumping the queue.
        waiters.append(
            {"ticket": ticket, "operation_id": operation_id, "enqueued_at": now}
        )
    elif current is None or existing.get("operation_id") != current.operation_id:
        # The ticket is the durable identity; ``operation_id`` is per-attempt
        # and a retrying caller mints a fresh one on every poll. Re-pointing
        # the waiter at the id this attempt will actually be granted (and later
        # finalized/released) under is what keeps ``_drop_waiter`` able to find
        # it. Leaving the original id here strands the promoted waiter at
        # position 0 forever, and every later valid caller queues behind an
        # orphan with no active lease.
        #
        # The guard above is the other half of that rule: a waiter entry whose
        # id is the ACTIVE HOLDER's is not a waiter at all, it is the holder
        # keeping its place at the head while it works. Two callers sharing a
        # ticket (same sender and idempotency key) would otherwise let the
        # second re-point the holder's own entry at an id the holder knows
        # nothing about, after which the holder's release/finalize can no
        # longer drop it — the queue is then permanently headed by an orphan
        # and the holder's serialization guarantee is gone.
        existing["operation_id"] = operation_id

    order = [
        cast("dict[str, Any]", waiter).get("ticket")
        for waiter in waiters
        if isinstance(waiter, dict)
    ]
    position = order.index(ticket)

    if current is not None or position > 0:
        save_leases(path, data)
        # A queue place that failed to persist is still not a grant, so the
        # caller's next attempt simply re-enters the queue. Nothing to fail.
        # ``position`` is how many callers are ahead of this one. The current
        # holder keeps its own ticket at the head of the queue while it works,
        # so a holder plus one waiter puts that waiter at position 1.
        return ReserveResult(
            LEASE_QUEUED, lease=current, ticket=ticket, position=position
        )

    lease = Lease(
        agent=agent,
        generation=generation,
        operation_id=operation_id,
        backend_session_id=backend_session_id,
        nonce=nonce,
        holder_pid=holder_pid,
        holder_create_token=holder_create_token,
        deadline=deadline,
        acquired_at=now,
    )
    stored = asdict(lease)
    stored.pop("agent")
    entry["lease"] = stored
    if not save_leases(path, data):
        # Fail closed. The on-disk store is unchanged, so the target still
        # reads as free to everyone including us; granting here would hand two
        # callers the same conversation. Queueing is not a dead end (R1): the
        # caller retries within its budget and gets the cooperative tail if the
        # disk problem persists.
        return ReserveResult(LEASE_QUEUED, lease=None, ticket=ticket, position=position)
    return ReserveResult(LEASE_GRANTED, lease=lease, ticket=ticket, position=0)


def release_lease(path: Path, agent: str, operation_id: str) -> bool:
    """Release a lease held under ``operation_id``. Returns whether it matched."""
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return False
    mapping = entry
    lease = _to_lease(agent, mapping.get("lease"))
    if lease is None or lease.operation_id != operation_id:
        return False
    mapping["lease"] = None
    _drop_waiter(mapping, operation_id)
    return save_leases(path, data)


def finalize_lease(path: Path, agent: str, operation_id: str, generation: int) -> bool:
    """CAS on generation **and** ``operation_id``, releasing on success.

    A mismatch leaves the stored lease untouched. That is deliberate: after the
    operator force path has bumped the fencing generation, the lease must
    survive for the operator to inspect and clear, rather than being dropped by
    the very finalize the fence just rejected.
    """
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return False
    mapping = entry
    lease = _to_lease(agent, mapping.get("lease"))
    if lease is None:
        return False
    if lease.operation_id != operation_id or lease.generation != generation:
        return False
    mapping["lease"] = None
    _drop_waiter(mapping, operation_id)
    # A finalize that did not reach disk did not win: the lease is still held
    # on disk, and reporting success would let this attempt write the agent
    # record while another caller could still be granted the same target.
    return save_leases(path, data)


def force_clear_lease(
    path: Path, agent: str, *, expect_operation_id: str | None = None
) -> tuple[Lease | None, bool]:
    """Operator escape: drop ``agent``'s lease. Returns ``(cleared, persisted)``.

    Reachable only from the CLI, behind the session recovery token. Ordinary
    ``kill_agent`` never bypasses the lease.

    ``expect_operation_id`` makes the clear a compare-and-swap. The operator
    path fences, then kills a child, then clears — and killing a child takes
    real time, during which a queued caller can legitimately be granted the
    target. Clearing unconditionally would drop *that* caller's lease, letting
    a third resume the same conversation underneath it. When the stored lease
    is a different operation the clear is refused and ``(lease, False)``
    reports what is actually held.

    ``persisted`` is the write result and callers must honour it: a clear that
    never reached disk left the lease exactly where it was.
    """
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return None, True
    mapping = entry
    lease = _to_lease(agent, mapping.get("lease"))
    if (
        expect_operation_id is not None
        and lease is not None
        and lease.operation_id != expect_operation_id
    ):
        return lease, False
    mapping["lease"] = None
    mapping["waiters"] = []
    return lease, save_leases(path, data)


def drop_agent(path: Path, agent: str) -> bool:
    """Remove an agent's whole lease entry (used when its record is removed).

    Returns whether the store reflects the removal. A discarded ``False`` here
    leaves a killed agent's entry on disk, which a same-named successor then
    inherits — so the caller is given the result rather than a silent ``None``.
    """
    data = load_leases(path)
    if agent not in data:
        return True
    del data[agent]
    return save_leases(path, data)


__all__ = [
    "LEASES_FILE_NAME",
    "LEASE_GRANTED",
    "LEASE_QUEUED",
    "Lease",
    "LeaseNotPersistedError",
    "LeaseStoreError",
    "LeaseStoreUnreadableError",
    "ReserveResult",
    "active_lease",
    "drop_agent",
    "finalize_lease",
    "force_clear_lease",
    "load_leases",
    "reconcile_lease",
    "release_lease",
    "reserve_lease",
    "save_leases",
]
