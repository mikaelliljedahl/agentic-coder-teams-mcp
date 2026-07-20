"""Tests for A4b: the per-target operation lease.

The lease exists because CAS-after-the-fact is not sufficient. Two callers can
snapshot the same generation, both resume the same conversation, and both
deliver distinct nonces; the losing CAS cannot undo an irreversible side
effect. So the lease is reserved *atomically, before the registry lock is
released*, and finalization CASes on generation **and** ``operation_id``.
"""

import json
from pathlib import Path

import pytest

from claude_teams import leases
from claude_teams.backends.process_manager import (
    OWNERSHIP_INDETERMINATE,
    OWNERSHIP_NOT_OURS,
    OWNERSHIP_OURS,
)
from claude_teams.leases import (
    LEASE_GRANTED,
    LEASE_QUEUED,
    Lease,
    LeaseNotPersistedError,
    LeaseStoreUnreadableError,
    active_lease,
    drop_agent,
    finalize_lease,
    force_clear_lease,
    load_leases,
    reconcile_lease,
    release_lease,
    reserve_lease,
    save_leases,
)


def _store(tmp_path: Path) -> Path:
    return tmp_path / "operation-leases.json"


def _reserve(
    path: Path,
    agent: str = "worker",
    *,
    generation: int = 1,
    operation_id: str = "op-1",
    holder_pid: int = 111,
    holder_create_token: str | None = "tok-111",
    deadline: float = 100.0,
    now: float = 0.0,
    holder_probe=lambda pid, token: OWNERSHIP_OURS,
    ticket: str | None = None,
):
    return reserve_lease(
        path,
        agent,
        generation=generation,
        operation_id=operation_id,
        backend_session_id="sess",
        nonce="n" * 32,
        holder_pid=holder_pid,
        holder_create_token=holder_create_token,
        deadline=deadline,
        now=now,
        holder_probe=holder_probe,
        ticket=ticket,
    )


# --------------------------------------------------------------------------
# Reservation and FIFO queueing
# --------------------------------------------------------------------------


def test_first_caller_is_granted_the_lease(tmp_path: Path) -> None:
    result = _reserve(_store(tmp_path))

    assert result.status == LEASE_GRANTED
    assert result.lease is not None
    assert result.lease.operation_id == "op-1"
    assert result.lease.holder_create_token == "tok-111"


def test_a_second_valid_caller_queues_and_is_not_refused(tmp_path: Path) -> None:
    """R1 forbids handing a valid caller a dead end. Queue, never refuse."""
    path = _store(tmp_path)
    _reserve(path)

    second = _reserve(path, operation_id="op-2", holder_pid=222)

    assert second.status == LEASE_QUEUED
    assert second.ticket, "a queued caller needs a FIFO ticket"
    assert second.position == 1
    # It is emphatically NOT a refusal: no reason, and it is retriable.
    assert second.status != "refused"


def test_queued_callers_are_promoted_in_fifo_order(tmp_path: Path) -> None:
    path = _store(tmp_path)
    first = _reserve(path)
    second = _reserve(path, operation_id="op-2", holder_pid=222)
    third = _reserve(path, operation_id="op-3", holder_pid=333)
    assert (second.position, third.position) == (1, 2)

    assert first.lease is not None
    release_lease(path, "worker", first.lease.operation_id)

    # The later ticket must NOT jump the queue.
    jumped = _reserve(path, operation_id="op-3", holder_pid=333, ticket=third.ticket)
    assert jumped.status == LEASE_QUEUED
    assert jumped.position == 1

    promoted = _reserve(path, operation_id="op-2", holder_pid=222, ticket=second.ticket)
    assert promoted.status == LEASE_GRANTED
    assert promoted.lease is not None
    assert promoted.lease.operation_id == "op-2"


def test_a_promoted_waiter_leaves_no_stale_ticket_for_the_next_caller(
    tmp_path: Path,
) -> None:
    """Three callers, because the defect is invisible with two.

    ``_prepare`` mints a **fresh** ``operation_id`` on every polling iteration,
    so the retry that finally wins the lease never carries the id its ticket was
    created with. If the waiter record keeps the original id, the
    ``_drop_waiter`` inside ``finalize_lease`` misses it and the promoted
    caller stays at position 0 forever — every later valid caller then queues
    behind an orphan with no active lease, which is exactly the dead end R1
    forbids.
    """
    path = _store(tmp_path)
    _reserve(path, operation_id="op-1")
    second = _reserve(path, operation_id="op-2a", holder_pid=222)
    third = _reserve(path, operation_id="op-3a", holder_pid=333)
    assert (second.position, third.position) == (1, 2)

    release_lease(path, "worker", "op-1")

    promoted = _reserve(
        path, operation_id="op-2b", holder_pid=222, ticket=second.ticket
    )
    assert promoted.status == LEASE_GRANTED
    assert finalize_lease(path, "worker", "op-2b", 1) is True
    assert active_lease(path, "worker") is None

    after = _reserve(path, operation_id="op-3b", holder_pid=333, ticket=third.ticket)
    assert after.status == LEASE_GRANTED, (
        "caller 3 queued behind an orphaned head ticket with no active lease"
    )
    assert load_leases(path)["worker"]["waiters"] == [
        {"ticket": third.ticket, "operation_id": "op-3b", "enqueued_at": 0.0}
    ]


def test_a_released_waiter_leaves_no_stale_ticket_for_the_next_caller(
    tmp_path: Path,
) -> None:
    """Same defect through ``release_lease`` — the ``finally`` path."""
    path = _store(tmp_path)
    _reserve(path, operation_id="op-1")
    second = _reserve(path, operation_id="op-2a", holder_pid=222)
    third = _reserve(path, operation_id="op-3a", holder_pid=333)

    release_lease(path, "worker", "op-1")
    promoted = _reserve(
        path, operation_id="op-2b", holder_pid=222, ticket=second.ticket
    )
    assert promoted.status == LEASE_GRANTED
    assert release_lease(path, "worker", "op-2b") is True

    after = _reserve(path, operation_id="op-3b", holder_pid=333, ticket=third.ticket)
    assert after.status == LEASE_GRANTED


def test_a_waiter_sharing_the_active_holders_ticket_is_not_repointed(
    tmp_path: Path,
) -> None:
    """Two callers under one ``(sender, key)`` derive ONE ticket (A4b).

    The waiter entry at the head of the queue while a lease is held IS the
    holder keeping its place. Re-pointing it at the second caller's per-attempt
    id leaves the holder unable to drop its own entry on release, so the queue
    is permanently headed by an orphan and the per-target serialization the
    lease provides is gone.
    """
    path = _store(tmp_path)
    first = _reserve(path, operation_id="op-holder")
    assert first.granted
    ticket = first.ticket

    second = _reserve(path, operation_id="op-second", ticket=ticket)

    assert second.status == LEASE_QUEUED
    entry = load_leases(path)["worker"]
    assert [w["operation_id"] for w in entry["waiters"]] == ["op-holder"]
    # ...and the holder can therefore still drop its own entry.
    assert release_lease(path, "worker", "op-holder") is True
    assert load_leases(path)["worker"]["waiters"] == []


def test_releasing_drops_the_holders_ticket_from_the_queue(tmp_path: Path) -> None:
    path = _store(tmp_path)
    first = _reserve(path)
    second = _reserve(path, operation_id="op-2", holder_pid=222)
    assert first.lease is not None

    release_lease(path, "worker", first.lease.operation_id)
    promoted = _reserve(path, operation_id="op-2", holder_pid=222, ticket=second.ticket)
    assert promoted.status == LEASE_GRANTED
    assert promoted.lease is not None
    release_lease(path, "worker", promoted.lease.operation_id)

    assert active_lease(path, "worker") is None
    assert load_leases(path).get("worker", {}).get("waiters", []) == []


# --------------------------------------------------------------------------
# Finalization CAS
# --------------------------------------------------------------------------


def test_finalize_succeeds_on_matching_generation_and_operation_id(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    _reserve(path, generation=7, operation_id="op-1")

    assert finalize_lease(path, "worker", "op-1", 7) is True
    assert active_lease(path, "worker") is None


def test_finalize_is_rejected_when_the_generation_moved(tmp_path: Path) -> None:
    """The CLI force path bumps the fencing generation before terminating."""
    path = _store(tmp_path)
    _reserve(path, generation=7, operation_id="op-1")

    assert finalize_lease(path, "worker", "op-1", 8) is False
    # The lease is left for the operator path to clear, not silently dropped.
    assert active_lease(path, "worker") is not None


def test_finalize_is_rejected_for_a_different_operation_id(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path, generation=7, operation_id="op-1")

    assert finalize_lease(path, "worker", "op-other", 7) is False


def test_a_stale_finalize_does_not_update_a_reused_name(tmp_path: Path) -> None:
    """Name reuse after removal must not be hijacked by an old attempt.

    This is why ``operation_id`` is in the CAS key rather than generation
    alone: a replacement record can legitimately start at generation 0 again.
    """
    path = _store(tmp_path)
    _reserve(path, generation=0, operation_id="op-old")
    # The agent is killed and its record removed; the lease goes with it.
    force_clear_lease(path, "worker")
    # A replacement agent takes the same name and starts its own attempt.
    _reserve(path, generation=0, operation_id="op-new", holder_pid=999)

    assert finalize_lease(path, "worker", "op-old", 0) is False
    surviving = active_lease(path, "worker")
    assert surviving is not None
    assert surviving.operation_id == "op-new"


# --------------------------------------------------------------------------
# Liveness, tokens, and reconciliation
# --------------------------------------------------------------------------


def test_a_dead_holder_is_reconciled_automatically(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path)

    surviving = reconcile_lease(
        path, "worker", holder_probe=lambda pid, token: OWNERSHIP_NOT_OURS
    )

    assert surviving is None
    assert active_lease(path, "worker") is None


def test_a_token_mismatch_is_reconciled_automatically(tmp_path: Path) -> None:
    """``holder_pid`` alone is insufficient: a dead holder's PID gets reused."""
    path = _store(tmp_path)
    _reserve(path, holder_pid=111, holder_create_token="tok-111")

    def probe(pid: int, token: str | None) -> str:
        # The PID is live, but it is now a DIFFERENT process.
        ours = pid == 111 and token == "tok-222"
        return OWNERSHIP_OURS if ours else OWNERSHIP_NOT_OURS

    assert reconcile_lease(path, "worker", holder_probe=probe) is None
    assert active_lease(path, "worker") is None


def test_a_provably_live_holder_survives_reconciliation(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path)

    surviving = reconcile_lease(
        path, "worker", holder_probe=lambda pid, token: OWNERSHIP_OURS
    )

    assert surviving is not None
    assert surviving.operation_id == "op-1"


def test_wall_clock_expiry_alone_never_reclaims_a_live_holder(tmp_path: Path) -> None:
    """Expiry is not fencing.

    A holder that is alive but slow *after spawning* would otherwise let a
    second caller observe expiry, fail to find a not-yet-flushed nonce, and
    retry into a delivery still in progress.
    """
    path = _store(tmp_path)
    _reserve(path, deadline=10.0, now=0.0)

    # Long past the deadline, but the holder is provably still ours.
    surviving = reconcile_lease(
        path, "worker", holder_probe=lambda pid, token: OWNERSHIP_OURS, now=10_000.0
    )

    assert surviving is not None, "an overdue but LIVE holder must not be reclaimed"


def test_an_expired_lease_with_a_dead_holder_is_reclaimed(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path, deadline=10.0, now=0.0)

    surviving = reconcile_lease(
        path, "worker", holder_probe=lambda pid, token: OWNERSHIP_NOT_OURS, now=10_000.0
    )

    assert surviving is None


def test_a_tokenless_holder_is_treated_as_unprovable(tmp_path: Path) -> None:
    """Fail-closed, matching ``process_manager.owns_process``."""
    path = _store(tmp_path)
    _reserve(path, holder_create_token=None)

    assert (
        reconcile_lease(
            path,
            "worker",
            holder_probe=lambda pid, token: (
                OWNERSHIP_OURS if token else OWNERSHIP_NOT_OURS
            ),
        )
        is None
    )


# --------------------------------------------------------------------------
# Crash-atomic storage
# --------------------------------------------------------------------------


def test_leases_are_written_atomically_via_temp_and_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leases must NOT share the registry's ``write_text`` overwrite path.

    ``_save_agents_unlocked`` overwrites ``agents.json`` in place, so a crash
    mid-write can destroy the registry *and* the lease. The file lock
    serializes writers but does not prevent a torn write.
    """
    path = _store(tmp_path)
    replaced: list[tuple[str, str]] = []
    real_replace = Path.replace

    def spy(self: Path, target) -> Path:
        replaced.append((self.name, Path(target).name))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy)

    save_leases(path, {"worker": {"lease": None, "waiters": []}})

    assert replaced, "expected a temp-file + atomic replace, not a direct write"
    tmp_name, final_name = replaced[0]
    assert tmp_name.endswith(".tmp")
    assert final_name == path.name


def _break_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the atomic replace fail the way a full or read-only disk would."""

    def boom(self: Path, target) -> Path:
        raise OSError

    monkeypatch.setattr(Path, "replace", boom)


def test_a_reservation_that_cannot_be_persisted_is_not_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: a lease nobody can read is not a lease.

    ``save_leases`` swallowed the ``OSError``, so a caller could be told
    ``granted`` while the store on disk still showed the target free — and
    another process would then reserve the same target and resume the same
    conversation, which is the exact double-delivery the lease exists to
    prevent. Reporting ``queued`` instead costs a retry and loses nothing.
    """
    path = _store(tmp_path)
    _break_writes(monkeypatch)

    result = _reserve(path)

    assert result.status == LEASE_QUEUED
    assert result.lease is None
    assert active_lease(path, "worker") is None


def test_a_finalize_that_cannot_be_persisted_does_not_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _store(tmp_path)
    _reserve(path, generation=7, operation_id="op-1")
    _break_writes(monkeypatch)

    assert finalize_lease(path, "worker", "op-1", 7) is False
    assert release_lease(path, "worker", "op-1") is False


def test_save_leases_reports_whether_it_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _store(tmp_path)
    assert save_leases(path, {"worker": {"lease": None, "waiters": []}}) is True
    _break_writes(monkeypatch)
    assert save_leases(path, {"worker": {"lease": None, "waiters": []}}) is False


def test_lease_store_is_not_the_agents_registry_file() -> None:
    assert leases.LEASES_FILE_NAME != "agents.json"


def test_a_missing_lease_store_reads_as_empty(tmp_path: Path) -> None:
    """No file yet is the legitimately-empty case."""
    assert load_leases(_store(tmp_path)) == {}


def test_an_unreadable_lease_store_is_distinguished_from_an_absent_one(
    tmp_path: Path,
) -> None:
    """The ``OSError`` branch, which corrupt JSON does not reach.

    Induced at the OS boundary — a directory where the store should be — not by
    patching ``load_leases``. This mirrors the delivery store's equivalent test,
    which existed while this one did not: a review found that reverting
    ``load_leases``' ``OSError`` branch to ``return {}`` survived the **entire**
    suite, because the corrupt-JSON test below covers only the decode branch.

    The gap mattered most on the platform this runs on. Corrupt JSON needs a
    torn write; an ``OSError`` here is the everyday Windows case — a sharing
    violation from a concurrent MCP server, or an ACL denial — and failing open
    on it grants a lease over a live holder.
    """
    path = _store(tmp_path)
    path.mkdir(parents=True)  # read() raises OSError, not FileNotFoundError

    with pytest.raises(LeaseStoreUnreadableError):
        load_leases(path)
    with pytest.raises(LeaseStoreUnreadableError):
        active_lease(path, "worker")


def test_a_corrupt_lease_store_never_reads_as_no_lease_held(tmp_path: Path) -> None:
    """An error is not an absence.

    Returning ``{}`` here reported every target as free, so a caller was granted
    a lease over a live holder and resumed a conversation somebody else was
    already resuming — the exact double delivery this module exists to prevent,
    arriving through the loader rather than the writer.
    """
    path = _store(tmp_path)
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(LeaseStoreUnreadableError):
        load_leases(path)
    with pytest.raises(LeaseStoreUnreadableError):
        active_lease(path, "worker")
    with pytest.raises(LeaseStoreUnreadableError):
        _reserve(path)


def test_a_malformed_lease_payload_is_not_read_as_no_lease(tmp_path: Path) -> None:
    """The same fail-open, one level in: a corrupt lease body is not 'free'."""
    path = _store(tmp_path)
    path.write_text(
        json.dumps({"worker": {"lease": {"generation": "not-an-int"}, "waiters": []}}),
        encoding="utf-8",
    )

    with pytest.raises(LeaseStoreUnreadableError):
        active_lease(path, "worker")


def _break_atomic_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the atomic replace raise a real ``OSError``, as a full disk would.

    Deliberately at the OS boundary rather than on ``save_leases`` or the
    function under test, so the test exercises the real failure path.
    """
    original = Path.replace

    def _raise(self: Path, target):
        if str(target).endswith(".json"):
            raise OSError(28, "No space left on device")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", _raise)


def test_reconcile_reports_a_reclaim_that_did_not_reach_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed write leaves the lease held on disk; ``None`` would be a lie.

    Returning ``None`` after a lost write told ``kill_agent`` there was no live
    lease, so it killed the process and deleted the record while the on-disk
    lease still named a live holder mid-delivery.
    """
    path = _store(tmp_path)
    _reserve(path)
    _break_atomic_replace(monkeypatch)

    with pytest.raises(LeaseNotPersistedError):
        reconcile_lease(
            path, "worker", holder_probe=lambda pid, token: OWNERSHIP_NOT_OURS
        )


def test_a_lease_round_trips_through_disk(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path, generation=3, operation_id="op-x", holder_pid=42)

    reloaded = active_lease(path, "worker")

    assert reloaded == Lease(
        agent="worker",
        generation=3,
        operation_id="op-x",
        backend_session_id="sess",
        nonce="n" * 32,
        holder_pid=42,
        holder_create_token="tok-111",
        deadline=100.0,
        acquired_at=0.0,
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["worker"]["lease"]["operation_id"] == "op-x"


# --------------------------------------------------------------------------
# Operator force path
# --------------------------------------------------------------------------


def test_force_clear_returns_the_cleared_lease_for_inspection(tmp_path: Path) -> None:
    """The operator path must be able to see the attempt nonce before forcing."""
    path = _store(tmp_path)
    _reserve(path, operation_id="op-1")

    cleared, persisted = force_clear_lease(path, "worker")

    assert persisted is True
    assert cleared is not None
    assert cleared.nonce == "n" * 32
    assert cleared.operation_id == "op-1"
    assert active_lease(path, "worker") is None


def test_force_clear_on_an_absent_lease_is_a_no_op(tmp_path: Path) -> None:
    assert force_clear_lease(_store(tmp_path), "worker") == (None, True)


def test_force_clear_refuses_to_clobber_a_different_operation(tmp_path: Path) -> None:
    """The operator's fence/kill/clear window must not drop someone else's lease.

    Terminating a child takes real time, and a queued caller can legitimately be
    granted the target inside that window. Clearing unconditionally would drop
    THAT caller's lease and let a third resume the same conversation underneath
    it, which is the double delivery the lease exists to prevent.
    """
    path = _store(tmp_path)
    _reserve(path, operation_id="op-new")

    cleared, persisted = force_clear_lease(
        path, "worker", expect_operation_id="op-fenced"
    )

    assert persisted is False
    assert cleared is not None
    assert cleared.operation_id == "op-new"
    surviving = active_lease(path, "worker")
    assert surviving is not None
    assert surviving.operation_id == "op-new"


# ==========================================================================
# Critical 4 — the claim must not reclaim uncertainty as death
# ==========================================================================


def test_an_indeterminate_holder_is_never_reclaimed(tmp_path: Path) -> None:
    """ "I could not check" is not "the holder is gone".

    ``owns_process`` collapses "dead" and "token unreadable / access denied"
    into one ``False``, which is right for a destructive gate and wrong for a
    reclaim gate: acting on it lets a second caller take a live record and
    resume the same conversation.
    """
    path = _store(tmp_path)
    _reserve(path)

    surviving = reconcile_lease(
        path, "worker", holder_probe=lambda pid, token: OWNERSHIP_INDETERMINATE
    )

    assert surviving is not None, "an unprovable holder must not be reclaimed"
    assert surviving.operation_id == "op-1"


def test_an_indeterminate_holder_queues_a_second_caller_rather_than_granting(
    tmp_path: Path,
) -> None:
    """The R1-safe answer: queued, never granted, never refused."""
    path = _store(tmp_path)
    _reserve(path)

    second = _reserve(
        path,
        operation_id="op-2",
        holder_probe=lambda pid, token: OWNERSHIP_INDETERMINATE,
    )

    assert second.status == LEASE_QUEUED
    assert second.lease is not None
    assert second.lease.operation_id == "op-1"


def test_a_probe_that_raises_is_uncertainty_not_death(tmp_path: Path) -> None:
    path = _store(tmp_path)
    _reserve(path)

    def _raises(pid: int, token: str | None) -> str:
        raise OSError(13, "Access is denied")

    assert reconcile_lease(path, "worker", holder_probe=_raises) is not None


def test_drop_agent_reports_whether_the_entry_actually_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A surviving entry is inherited by a same-named successor."""
    path = _store(tmp_path)
    _reserve(path)
    assert drop_agent(path, "worker") is True

    _reserve(path)
    _break_atomic_replace(monkeypatch)
    assert drop_agent(path, "worker") is False
