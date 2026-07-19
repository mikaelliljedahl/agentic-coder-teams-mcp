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


def load_leases(path: Path) -> dict[str, dict[str, Any]]:
    """Load the lease store, tolerating absence and corruption."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in cast("dict[str, Any]", raw).items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def save_leases(path: Path, data: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the lease store (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


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
    if not isinstance(payload, dict):
        return None
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
    except (KeyError, TypeError, ValueError):
        return None


def active_lease(path: Path, agent: str) -> Lease | None:
    """Return the lease currently recorded for ``agent``, if any.

    This is the *recorded* lease, with no liveness check: use
    :func:`reconcile_lease` when the answer must be "is it provably live".
    """
    entry = load_leases(path).get(agent)
    if not isinstance(entry, dict):
        return None
    return _to_lease(agent, entry.get("lease"))


def _holder_provably_live(
    lease: Lease, holder_live: Callable[[int, str | None], bool]
) -> bool:
    """Fail-closed liveness for a lease holder.

    Mirrors ``process_manager.owns_process``: an unreadable or mismatched
    creation token means "not ours", never "probably fine".
    """
    try:
        return bool(holder_live(lease.holder_pid, lease.holder_create_token))
    except OSError:
        return False


def reconcile_lease(
    path: Path,
    agent: str,
    *,
    holder_live: Callable[[int, str | None], bool],
    now: float | None = None,
) -> Lease | None:
    """Clear a lease whose holder is gone; return the surviving live lease.

    ``now`` is accepted (and deliberately unused for the reclaim decision) to
    make the rule explicit at every call site: **wall-clock expiry alone never
    reclaims a lease.** Only a holder that is dead, or whose creation token no
    longer matches, releases it.
    """
    _ = now
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return None
    lease = _to_lease(agent, entry.get("lease"))
    if lease is None:
        return None
    if _holder_provably_live(lease, holder_live):
        return lease
    entry["lease"] = None
    _drop_waiter(entry, lease.operation_id)
    save_leases(path, data)
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
    holder_live: Callable[[int, str | None], bool],
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
    if current is not None and not _holder_provably_live(current, holder_live):
        _drop_waiter(entry, current.operation_id)
        entry["lease"] = None
        current = None

    waiters = cast("list[Any]", entry["waiters"])
    if ticket is None:
        ticket = uuid.uuid4().hex
        waiters.append(
            {"ticket": ticket, "operation_id": operation_id, "enqueued_at": now}
        )
    elif not any(
        isinstance(waiter, dict)
        and cast("dict[str, Any]", waiter).get("ticket") == ticket
        for waiter in waiters
    ):
        # A ticket we no longer know about (store reset, or already promoted
        # and released): re-enter the queue rather than silently jumping it.
        waiters.append(
            {"ticket": ticket, "operation_id": operation_id, "enqueued_at": now}
        )

    order = [
        cast("dict[str, Any]", waiter).get("ticket")
        for waiter in waiters
        if isinstance(waiter, dict)
    ]
    position = order.index(ticket)

    if current is not None or position > 0:
        save_leases(path, data)
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
    save_leases(path, data)
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
    save_leases(path, data)
    return True


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
    save_leases(path, data)
    return True


def force_clear_lease(path: Path, agent: str) -> Lease | None:
    """Operator escape: drop any lease for ``agent`` and return what was there.

    Reachable only from the CLI, behind the session recovery token. Ordinary
    ``kill_agent`` never bypasses the lease.
    """
    data = load_leases(path)
    entry = data.get(agent)
    if not isinstance(entry, dict):
        return None
    mapping = entry
    lease = _to_lease(agent, mapping.get("lease"))
    mapping["lease"] = None
    mapping["waiters"] = []
    save_leases(path, data)
    return lease


def drop_agent(path: Path, agent: str) -> None:
    """Remove an agent's whole lease entry (used when its record is removed)."""
    data = load_leases(path)
    if agent in data:
        del data[agent]
        save_leases(path, data)


__all__ = [
    "LEASES_FILE_NAME",
    "LEASE_GRANTED",
    "LEASE_QUEUED",
    "Lease",
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
