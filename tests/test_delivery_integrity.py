"""The four end-to-end protocol failures the final review reproduced.

Every test here failed before its fix and passes after. They are grouped by the
review's numbering:

1. **Attempt idempotency.** One idempotency key must produce at most one resume,
   whether the two calls race each other or arrive one after the other while the
   first attempt's uncertainty is still unresolved. A duplicate resume is a
   duplicate prompt in a real conversation, not a duplicate row.
2. **Fail-closed persistence.** The pre-wait durable row is what makes response
   loss recoverable; a resume that starts without it has destroyed the only
   handle the caller has.
3. **Uncertainty is not absence.** An unreadable or ambiguous receipt scan may
   never become terminal ``failed``.
4. **A C2 refusal changes nothing** — including when parentage moves between the
   read-only pre-flight and the authoritative under-lock check.

Timing is on the injected clock throughout. The one place a real thread is used
(the concurrency test) blocks on an ``Event``, never on wall time.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import delivery_store as ds
from claude_teams import server_simple
from claude_teams.agent_output import (
    BINDING_BOUND,
    SPAWNED_BY_FIELD,
    SPAWNED_BY_SOURCE_FIELD,
    SPAWNED_BY_SOURCE_OPERATOR,
    AgentOutput,
    BindingResult,
)
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import (
    OWNERSHIP_INDETERMINATE,
    OWNERSHIP_NOT_OURS,
)
from claude_teams.delivery import (
    DELIVERY_MARKER_PREFIX,
    SCAN_ABSENT,
    SCAN_AMBIGUOUS,
    SCAN_FOUND,
    SCAN_INDETERMINATE,
    new_delivery_nonce,
)

SESSION = "session-id"
AGENT = "worker"
BACKEND_SESSION = "backend-session-id"
KEY = "k-1"


class _Clock:
    """Injected clock; ``sleep`` advances it rather than blocking."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0
        self.on_sleep = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self.sleeps)


class _FakeResumeBackend:
    def __init__(self, on_resume=None, *, handle: str = "789") -> None:
        self.resume_calls: list[tuple[SpawnRequest, str]] = []
        self.on_resume = on_resume
        self.handle = handle

    def supports_resume(self) -> bool:
        return True

    def default_model(self) -> str:
        return "model"

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        self.resume_calls.append((request, backend_session_id))
        if self.on_resume is not None:
            self.on_resume(_nonce_of(request.prompt))
        return SimpleNamespace(process_handle=self.handle)


class _FakeRegistry:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def get(self, backend: str) -> object:
        return self.backend


def _nonce_of(prompt: str) -> str:
    _, _, tail = prompt.partition(DELIVERY_MARKER_PREFIX)
    return tail.split()[0].strip("]")


def _claude_user_record(text: str) -> dict:
    return {
        "type": "user",
        "sessionId": BACKEND_SESSION,
        "message": {"role": "user", "content": text},
    }


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    session_dir = tmp_path / "sessions" / SESSION
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", SESSION)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})

    transcript = tmp_path / "transcript.jsonl"
    _append(transcript, _claude_user_record("the original task"))

    work = tmp_path / "work"
    work.mkdir()
    server_simple._save_agents(
        SESSION,
        [
            {
                "name": AGENT,
                "pid": 123,
                "backend": "claude-code",
                "session_id": SESSION,
                "status": "running",
                "spawned_at": 100.0,
                "cwd": str(work),
                "backend_session_id": BACKEND_SESSION,
                "model": "model",
                "permission_mode": "bypass",
                "reasoning_effort": None,
                "correlation_id": "corr-delivery",
                "spawned_by": "team-lead",
                "spawned_by_source": "spawn",
            }
        ],
    )
    server_simple._persist_session_binding(SESSION)

    state = SimpleNamespace(last_activity_at=900.0, last_message="done")

    def _binding(agent: dict, *, bounded_only: bool = False) -> BindingResult:
        return BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=state.last_activity_at,
                last_message=state.last_message,
                rollout_path=str(transcript),
                backend_session_id=BACKEND_SESSION,
            ),
        )

    monkeypatch.setattr(server_simple, "_resolve_agent_binding", _binding)
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000.0)
    clock = _Clock()
    monkeypatch.setattr(server_simple, "_delivery_clock", clock)
    monkeypatch.setattr(server_simple, "_delivery_sleep", clock.sleep)
    monkeypatch.setattr(server_simple, "_DELIVERY_CALL_BUDGET_SECONDS", 10.0)
    monkeypatch.setattr(server_simple, "_DELIVERY_POLL_SECONDS", 1.0)
    # A hook-written waiting marker: the target is parked, so nothing has to be
    # waited for and the delivery path is exercised rather than the busy wait.
    (session_dir / f"state-{AGENT}.json").write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 950.0}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        tmp_path=tmp_path,
        transcript=transcript,
        clock=clock,
        session_dir=session_dir,
        state=state,
        deliveries=session_dir / ds.DELIVERIES_FILE_NAME,
    )


def _install(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))


def _child_alive(monkeypatch: pytest.MonkeyPatch, alive: bool) -> None:
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (alive, "x"),
    )


def _maybe_record(key: str = KEY, sender: str = "team-lead") -> dict | None:
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        found = txn.get(sender, key)
        return dict(found) if found else None


def _record(key: str = KEY, sender: str = "team-lead") -> dict:
    found = _maybe_record(key, sender)
    assert found is not None, f"no delivery record for {sender}/{key}"
    return found


def _agent_record() -> dict:
    record = server_simple._find_agent(server_simple._load_agents(SESSION), AGENT)
    assert record is not None
    return record


# ==========================================================================
# Critical 1 — one key, at most one resume
# ==========================================================================


def test_two_concurrent_calls_under_one_key_resume_the_conversation_once(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's concurrent probe, deterministically.

    Both callers derive the same FIFO ticket from ``(sender, key)``. The second
    used to re-point the **active holder's** waiter at its own ``operation_id``,
    so the holder could no longer drop it, and both went on to resume — two real
    prompts in one conversation under one idempotency key.

    Both callers run in real threads and genuinely overlap: B is inside the
    delivery path — queued behind A's lease under their shared ticket — while A
    is parked inside ``resume``. Nothing here waits on wall time. The injected
    clock only advances once A has fully returned, so B cannot fall out on its
    budget before the window the defect lives in has actually opened.
    """
    a_entered_resume = threading.Event()
    a_finished = threading.Event()
    b_started = threading.Event()

    def _park(nonce: str) -> None:
        # Every resume writes its receipt, so confirmation needs no polling and
        # therefore never advances the shared clock out from under B.
        _append(
            env.transcript,
            _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
        )
        if a_entered_resume.is_set():
            # A second resume IS the defect; do not deadlock the test on it.
            return
        a_entered_resume.set()
        # Hold the lease until B is demonstrably inside the delivery path.
        b_started.wait(timeout=10.0)

    backend = _FakeResumeBackend(_park)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    main_thread = threading.current_thread()

    def _sleep(seconds: float) -> None:
        # Only caller A's own polling advances the injected clock. B runs on
        # the main thread, and burning its budget while it is queued behind A
        # would end the race before the window the defect lives in has opened —
        # the test would then pass for the wrong reason.
        if threading.current_thread() is main_thread:
            b_started.set()
        else:
            env.clock.now += seconds

    monkeypatch.setattr(server_simple, "_delivery_sleep", _sleep)

    results: dict[str, dict] = {}

    def _call_a() -> None:
        try:
            results["a"] = server_simple._guaranteed_send(
                SESSION, AGENT, "next prompt", KEY, True, tool="follow_up_agent"
            )
        finally:
            a_finished.set()

    thread = threading.Thread(target=_call_a, daemon=True)
    thread.start()
    assert a_entered_resume.wait(timeout=10.0), "caller A never reached the resume"

    results["b"] = server_simple._guaranteed_send(
        SESSION, AGENT, "next prompt", KEY, True, tool="follow_up_agent"
    )
    b_started.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    assert len(backend.resume_calls) == 1, (
        f"one key resumed the conversation {len(backend.resume_calls)} times"
    )
    assert _record()["attempts"] == 1
    assert results["b"].get("message_id") == _record()["message_id"]


def test_a_second_caller_is_blocked_before_the_row_says_anything_was_sent(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim, isolated from the reconcile gate that usually covers for it.

    The reconcile-before-resend rule only bites once the row has reached
    ``sent``. There is a real window before that: caller A has the lease but has
    not yet marked its attempt, so B reads a ``pending`` row, passes the gate,
    and queues behind A's lease under their shared ticket. When A finishes with
    no receipt — live uncertainty, the normal case — B is promoted and resumes
    the same conversation a second time.

    B is therefore released into the path at exactly that instant, and only the
    delivery-record claim can stop it.
    """
    b_opened = threading.Event()
    a_at_mark = threading.Event()
    main_thread = threading.current_thread()

    original_open = server_simple._open_delivery_record

    def _open(*args, **kwargs):
        try:
            return original_open(*args, **kwargs)
        finally:
            # Set for B whether it was let through or refused, so A is never
            # left waiting on a caller that already returned.
            if threading.current_thread() is main_thread:
                b_opened.set()

    original_mark = server_simple._mark_attempt_sent

    def _mark(session_id: str, record: dict, plan) -> None:
        # Hold the row at ``pending`` until B has read it.
        a_at_mark.set()
        b_opened.wait(timeout=10.0)
        original_mark(session_id, record, plan)

    monkeypatch.setattr(server_simple, "_open_delivery_record", _open)
    monkeypatch.setattr(server_simple, "_mark_attempt_sent", _mark)
    monkeypatch.setattr(
        server_simple,
        "_delivery_sleep",
        lambda seconds: (
            None
            if threading.current_thread() is main_thread
            else env.clock.__setattr__("now", env.clock.now + seconds)
        ),
    )

    backend = _FakeResumeBackend()  # deliberately writes no receipt
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    def _call_a() -> None:
        server_simple._guaranteed_send(
            SESSION, AGENT, "next prompt", KEY, True, tool="follow_up_agent"
        )

    thread = threading.Thread(target=_call_a, daemon=True)
    thread.start()
    assert a_at_mark.wait(timeout=10.0), "caller A never reached its attempt marker"

    result = server_simple._guaranteed_send(
        SESSION, AGENT, "next prompt", KEY, True, tool="follow_up_agent"
    )
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    assert result["reason"] == server_simple.REASON_DELIVERY_IN_PROGRESS
    assert result["retriable"] is True
    assert len(backend.resume_calls) == 1, (
        f"one key resumed the conversation {len(backend.resume_calls)} times"
    )
    assert _record()["attempts"] == 1


@pytest.mark.asyncio
async def test_same_key_retry_does_not_resend_while_the_prior_attempt_is_unresolved(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live ``unconfirmed`` attempt is uncertainty, not a free retry slot.

    The first call sends and gets no receipt inside the budget; the child is
    alive, so the outcome is ``queued(unconfirmed)``. Repeating the same key
    used to walk straight past ``_reconcile_pending_delivery`` (the nonce is
    genuinely not there yet) into a second lease, a second nonce and a second
    resume. The recipient is a backend conversation with no dedupe table, so
    that is the same instruction delivered twice.
    """
    backend = _FakeResumeBackend()  # never writes a receipt
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    assert first["status"] == ds.STATUS_QUEUED
    assert first["phase"] == ds.PHASE_UNCONFIRMED
    assert len(backend.resume_calls) == 1

    second = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert len(backend.resume_calls) == 1, "the unresolved attempt was sent again"
    assert _record()["attempts"] == 1
    assert second["status"] == ds.STATUS_QUEUED
    assert second["phase"] == ds.PHASE_UNCONFIRMED


@pytest.mark.asyncio
async def test_a_retry_after_the_receipt_lands_reconciles_instead_of_resending(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the rule: not resending must not mean not settling."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    assert first["phase"] == ds.PHASE_UNCONFIRMED

    # The buffered transcript write arrives after the call returned.
    _append(
        env.transcript,
        _claude_user_record(
            f"next prompt {DELIVERY_MARKER_PREFIX}{_record()['nonce']}"
        ),
    )

    second = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert len(backend.resume_calls) == 1
    assert second["status"] == ds.STATUS_DELIVERED


# ==========================================================================
# Critical 2 — no resume without a durable pre-wait row
# ==========================================================================


@pytest.mark.asyncio
async def test_a_store_write_that_never_reached_disk_fails_closed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistence failure must refuse, not proceed and report success.

    The row created before the wait is the caller's only handle on the outcome.
    Losing it while continuing to resume produces the exact hole the
    caller-supplied key exists to close: a lost response with neither a status
    nor a recoverable key.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    monkeypatch.setattr(ds, "save_records", lambda path, data: False)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["reason"] == "delivery_store_unavailable"
    assert result["retriable"] is True
    assert backend.resume_calls == [], "a resume began without a durable row"
    assert not env.deliveries.exists()


def test_save_records_reports_whether_the_store_reached_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primitive itself must return the fact, not swallow it.

    Both halves matter: a swallowed failure that always returns ``True`` reads
    exactly like a success to every caller, which is how a lost row became a
    delivery that proceeded anyway.
    """
    path = tmp_path / "sub" / ds.DELIVERIES_FILE_NAME
    assert ds.save_records(path, {"a|b": {"idempotency_key": "b"}}) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a|b": {"idempotency_key": "b"}
    }

    def _no_replace(self, target):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", _no_replace)
    assert ds.save_records(path, {"a|b": {"idempotency_key": "c"}}) is False
    # ...and the failed write left the previous contents intact.
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a|b": {"idempotency_key": "b"}
    }
    assert not list(path.parent.glob("*.tmp")), "the temp file was not cleaned up"


def test_delivery_transaction_raises_when_the_write_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``delivery_transaction`` must surface the failure to its caller."""
    path = tmp_path / ds.DELIVERIES_FILE_NAME
    monkeypatch.setattr(ds, "save_records", lambda p, d: False)
    with pytest.raises(ds.DeliveryStoreError):
        with ds.delivery_transaction(path) as txn:
            txn.put(
                ds.new_record(
                    sender="s",
                    idempotency_key="k",
                    to="t",
                    fingerprint="f",
                    created_at=1.0,
                )
            )


# ==========================================================================
# Critical 3 — uncertainty is not absence
# ==========================================================================


def _delivery_record(nonce: str, *, phase: str = ds.PHASE_UNCONFIRMED) -> dict:
    record = ds.new_record(
        sender="team-lead",
        idempotency_key=KEY,
        to=AGENT,
        fingerprint="f",
        created_at=0.0,
    )
    record["nonce"] = nonce
    record["phase"] = phase
    record["attempted_at"] = 0.0
    return record


def test_scan_for_nonce_reports_found_absent_indeterminate_and_ambiguous(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four outcomes must survive ``_scan_for_nonce``, not collapse to a bool."""
    agent = _agent_record()
    landed = new_delivery_nonce()
    missing = new_delivery_nonce()

    _append(env.transcript, _claude_user_record(f"x {DELIVERY_MARKER_PREFIX}{landed}"))
    assert server_simple._scan_for_nonce(SESSION, agent, landed) == SCAN_FOUND

    # Read to the end, nonce genuinely not present: a complete negative.
    assert server_simple._scan_for_nonce(SESSION, agent, missing) == SCAN_ABSENT

    # A corrupt record means the negative is not authoritative.
    with env.transcript.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert server_simple._scan_for_nonce(SESSION, agent, missing) == SCAN_INDETERMINATE

    # Two candidate successors: attribution is unprovable.
    monkeypatch.setattr(
        server_simple,
        "_delivery_scanner",
        lambda *a, **k: _AmbiguousScanner(),
    )
    assert server_simple._scan_for_nonce(SESSION, agent, "def456") == SCAN_AMBIGUOUS


class _AmbiguousScanner:
    def rewind(self) -> None:
        return None

    def full_scan(self, nonce: str) -> str:
        return SCAN_AMBIGUOUS


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_phase"),
    [
        (SCAN_FOUND, ds.STATUS_DELIVERED, ds.PHASE_SETTLED),
        (SCAN_ABSENT, ds.STATUS_FAILED, ds.PHASE_SETTLED),
        (SCAN_INDETERMINATE, ds.STATUS_QUEUED, ds.PHASE_UNCONFIRMED),
        (SCAN_AMBIGUOUS, ds.STATUS_QUEUED, ds.PHASE_UNCONFIRMED),
    ],
)
def test_only_a_complete_authoritative_negative_settles_failed(
    env,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_status: str,
    expected_phase: str,
) -> None:
    """R6's central distinction, one scan outcome at a time.

    The child is dead and the flush grace has passed in every case, so the only
    thing deciding the result is what the scan actually established.
    """
    monkeypatch.setattr(
        server_simple, "_scan_for_nonce", lambda session, agent, nonce: outcome
    )
    _child_alive(monkeypatch, False)
    agent = _agent_record()
    record = _delivery_record("nonce-1")

    server_simple._reconcile_delivery_record(SESSION, record, agent, now=10_000.0)

    assert record["status"] == expected_status
    assert record["phase"] == expected_phase


@pytest.mark.parametrize("outcome", [SCAN_INDETERMINATE, SCAN_AMBIGUOUS])
def test_kill_time_reconciliation_does_not_manufacture_a_terminal_failure(
    env, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """The kill path settles ``failed`` only on a definite negative too."""
    monkeypatch.setattr(
        server_simple, "_scan_for_nonce", lambda session, agent, nonce: outcome
    )
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(_delivery_record("nonce-1"))

    server_simple._reconcile_deliveries_for_target(SESSION, AGENT, _agent_record())

    stored = _record()
    assert stored["status"] == ds.STATUS_QUEUED
    assert stored["phase"] == ds.PHASE_UNCONFIRMED


def test_kill_time_reconciliation_still_settles_a_definite_negative(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server_simple, "_scan_for_nonce", lambda session, agent, nonce: SCAN_ABSENT
    )
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(_delivery_record("nonce-1"))

    server_simple._reconcile_deliveries_for_target(SESSION, AGENT, _agent_record())

    assert _record()["status"] == ds.STATUS_FAILED


# ==========================================================================
# Warning 1 — the `to` view reconciles too, as its contract says
# ==========================================================================


@pytest.mark.asyncio
async def test_the_to_view_reconciles_rather_than_publishing_a_stale_row(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One message must not have two published truths.

    ``delivery_status(to=...)`` used to call ``list_for_sender`` directly, so it
    could answer ``queued(unconfirmed)`` for a row the keyed lookup would call
    ``delivered`` in the very next call. The tool contract describes active
    reconciliation; either the code does it or the contract is false.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    assert first["phase"] == ds.PHASE_UNCONFIRMED

    _append(
        env.transcript,
        _claude_user_record(
            f"next prompt {DELIVERY_MARKER_PREFIX}{_record()['nonce']}"
        ),
    )

    listed = await server_simple.delivery_status(to=AGENT)

    rows = listed["deliveries"]
    assert len(rows) == 1
    assert rows[0]["status"] == ds.STATUS_DELIVERED

    keyed = await server_simple.delivery_status(idempotency_key=KEY)
    assert keyed["status"] == rows[0]["status"]


# ==========================================================================
# Critical 4 — an authoritative refusal changes nothing
# ==========================================================================


@pytest.mark.asyncio
async def test_parentage_moving_after_preflight_leaves_no_audit_row(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2's change-nothing rule survives the pre-flight/locked-check window.

    The read-only pre-flight passed, the durable row was created, and only then
    did the authoritative under-lock check refuse — leaving an audit row behind
    for a call that was refused. A refusal must leave the session
    byte-identical, so the authorization is snapshotted with the parentage it
    was granted against and the row does not survive a stale one.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    original = server_simple._open_delivery_record

    def _reparent_after_opening(*args, **kwargs):
        opened = original(*args, **kwargs)
        with server_simple._agents_transaction(SESSION) as agents:
            record = server_simple._find_agent(agents, AGENT)
            assert record is not None
            record[SPAWNED_BY_FIELD] = "someone-else"
            server_simple._save_agents_transaction(SESSION, agents)
        return opened

    monkeypatch.setattr(server_simple, "_open_delivery_record", _reparent_after_opening)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["reason"] in {"not_spawner", "stale_authorization"}
    assert backend.resume_calls == []
    assert _maybe_record() is None, "a refused call left an audit row behind"


@pytest.mark.asyncio
async def test_a_reparented_record_is_rejected_even_when_the_spawner_still_matches(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authorization is the snapshot, not just "is the spawner me now".

    An operator ``adopt`` between the pre-flight and the under-lock check
    rewrites the parentage to *operator-asserted* while leaving the spawner name
    unchanged, so the direction guard alone still says yes. The write was
    authorized against a record that no longer exists in that form, and the
    honest answer is to reject the stale authorization and leave nothing behind
    rather than proceed on an authorization that was never granted for this
    record.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    original = server_simple._open_delivery_record

    def _readopt_after_opening(*args, **kwargs):
        opened = original(*args, **kwargs)
        with server_simple._agents_transaction(SESSION) as agents:
            record = server_simple._find_agent(agents, AGENT)
            assert record is not None
            record[SPAWNED_BY_SOURCE_FIELD] = SPAWNED_BY_SOURCE_OPERATOR
            server_simple._save_agents_transaction(SESSION, agents)
        return opened

    monkeypatch.setattr(server_simple, "_open_delivery_record", _readopt_after_opening)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["reason"] == "stale_authorization"
    assert backend.resume_calls == []
    assert _maybe_record() is None, "a refused call left an audit row behind"


# ==========================================================================
# Round-2 critical 2 — a failed claim release must not wedge a valid delivery
# ==========================================================================


def _break_atomic_replace(monkeypatch: pytest.MonkeyPatch, suffix: str):
    """Induce a real ``OSError`` at the atomic replace, as a full disk would.

    Deliberately at the OS boundary and not on the function under test, so the
    production failure path runs for real. Returns a toggle so the disk can be
    "repaired" mid-test without ``monkeypatch.undo()``, which would also unwind
    the ``env`` fixture's session redirection.
    """
    original = Path.replace
    broken = {"on": True}

    def _raise(self: Path, target):
        if broken["on"] and str(target).endswith(suffix):
            raise OSError(28, "No space left on device")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", _raise)
    return broken


def test_a_claim_release_that_fails_to_persist_does_not_wedge_the_key(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live-process claim nobody can clear is an R1 dead end.

    Before the fix, ``_release_delivery_claim`` swallowed the failed write and
    left ``active_holder`` on disk naming THIS still-live server. Every later
    call under the key — and every ``deliver_pending`` drain — then refused
    ``delivery_in_progress`` against a claim that only this process could clear
    and that this process would never try to clear again.
    """
    record = ds.new_record(
        sender="team-lead",
        idempotency_key=KEY,
        to=AGENT,
        fingerprint="f",
        created_at=0.0,
    )
    record[server_simple.ACTIVE_HOLDER_FIELD] = server_simple._claim_holder()
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(dict(record))

    broken = _break_atomic_replace(monkeypatch, ds.DELIVERIES_FILE_NAME)
    assert server_simple._release_delivery_claim(SESSION, dict(record)) is False
    broken["on"] = False

    stranded = _record()[server_simple.ACTIVE_HOLDER_FIELD]
    assert stranded["pid"] == server_simple.os.getpid(), (
        "precondition: the stale claim still names this live process"
    )
    assert server_simple._claim_is_held(stranded) is False, (
        "a claim this process has finished must not block the next caller "
        "just because its release write was lost"
    )


def test_a_claim_this_process_is_still_working_does_block(env) -> None:
    """The reclaim must not become a free-for-all: a working claim still holds."""
    holder = server_simple._claim_holder()

    assert server_simple._claim_is_held(holder) is True


def test_a_claim_with_an_unprovable_foreign_holder_is_not_reclaimed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 critical 4, at the record claim: indeterminate is not death."""
    monkeypatch.setattr(
        server_simple.process_manager,
        "ownership_probe",
        lambda handle, token=None: OWNERSHIP_INDETERMINATE,
    )
    foreign = {"pid": 4242, "create_token": "tok-4242", "claim_id": "other"}

    assert server_simple._claim_is_held(foreign) is True


def test_a_claim_with_a_provably_gone_holder_is_reclaimed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server_simple.process_manager,
        "ownership_probe",
        lambda handle, token=None: OWNERSHIP_NOT_OURS,
    )
    foreign = {"pid": 4242, "create_token": "tok-4242", "claim_id": "other"}

    assert server_simple._claim_is_held(foreign) is False


# ==========================================================================
# Round-2 critical 3 — kill must leave the rescan possible
# ==========================================================================


def test_an_unsettled_row_can_still_be_rescanned_after_the_agent_is_gone(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill deletes the agent record; the evidence path must outlive it.

    ``_scan_for_nonce`` answers ``indeterminate`` for a missing agent, so once
    kill removed the record an attempt whose settlement write had been lost
    could never settle again — ``unconfirmed`` forever, with nothing able to
    move it. The transcript binding is now copied onto the durable row at
    attempt time, so a later ``delivery_status`` can still find the receipt.
    """
    nonce = new_delivery_nonce()
    record = _delivery_record(nonce)
    record[server_simple.TARGET_SNAPSHOT_FIELD] = _agent_record()
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(record)

    # The receipt landed, but the agent record is gone (killed).
    _append(env.transcript, _claude_user_record(f"x {DELIVERY_MARKER_PREFIX}{nonce}"))
    server_simple._save_agents(SESSION, [])
    assert server_simple._find_agent(server_simple._load_agents(SESSION), AGENT) is None

    stored = _record()
    moved = server_simple._reconcile_delivery_record(
        SESSION, stored, None, now=10_000.0
    )

    assert moved is True
    assert stored["status"] == ds.STATUS_DELIVERED, (
        "a receipt on disk must still be findable after the agent record is "
        "purged, or the row is stranded at unconfirmed forever"
    )


def test_without_a_snapshot_a_purged_agent_leaves_the_row_uncertain(env) -> None:
    """The complement: no snapshot, no scan — and still never a false failure."""
    record = _delivery_record(new_delivery_nonce())
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(record)
    server_simple._save_agents(SESSION, [])

    stored = _record()
    server_simple._reconcile_delivery_record(SESSION, stored, None, now=10_000.0)

    assert stored["status"] == ds.STATUS_QUEUED
    assert stored["phase"] == ds.PHASE_UNCONFIRMED


def test_the_scan_target_never_supplies_liveness(env) -> None:
    """A frozen snapshot's PID must not be mistaken for a live child."""
    record = _delivery_record(new_delivery_nonce())
    record[server_simple.TARGET_SNAPSHOT_FIELD] = _agent_record()

    assert server_simple._scan_target(record, None) is not None
    # ...but the caller passes the real (absent) record for liveness, so a
    # complete negative after the grace window still settles.
    server_simple._reconcile_delivery_record(SESSION, record, None, now=10_000.0)
    assert record["status"] == ds.STATUS_FAILED


# ==========================================================================
# Round-2 warning 1 — the C2 rollback must not lie on disk failure
# ==========================================================================


@pytest.mark.asyncio
async def test_a_refusal_that_cannot_remove_its_row_says_so(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 promises "nothing changed"; a swallowed rollback made that false.

    If row creation succeeds and the disk fails during refusal rollback, the
    API used to return ``stale_authorization`` and claim nothing changed while
    the row it promised to remove was still there — and the idempotency key was
    silently consumed. Nothing was sent either way, so the refusal stands; what
    changes is that the response no longer overstates what it knows.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    original = server_simple._open_delivery_record
    broken: dict[str, bool] = {}

    def _reparent_then_break_the_disk(*args, **kwargs):
        opened = original(*args, **kwargs)
        with server_simple._agents_transaction(SESSION) as agents:
            record = server_simple._find_agent(agents, AGENT)
            assert record is not None
            record[SPAWNED_BY_FIELD] = "someone-else"
            server_simple._save_agents_transaction(SESSION, agents)
        broken.update(_break_atomic_replace(monkeypatch, ds.DELIVERIES_FILE_NAME))
        return opened

    monkeypatch.setattr(
        server_simple, "_open_delivery_record", _reparent_then_break_the_disk
    )

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    broken["on"] = False

    assert result["success"] is False
    assert result["reason"] in {"not_spawner", "stale_authorization"}
    assert backend.resume_calls == [], "a refusal must still send nothing"
    assert result.get("record_discarded") is False, (
        "the refusal claimed nothing changed while its row survived on disk"
    )
    assert "could not be removed" in result["detail"]
    assert _maybe_record() is not None, "precondition: the row really did survive"
