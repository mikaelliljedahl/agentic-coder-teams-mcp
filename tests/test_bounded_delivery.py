"""Phase B — bounded in-call delivery, end to end through the MCP tools.

Nothing here mocks a delivery outcome. Receipts are real records appended to a
real transcript, the store is the real on-disk store, and the only injected
things are the clock and the sleep — the repo already carries one flaky
wall-clock test and this code is far more timing-dense than that one.

The single most important assertion in this file is the negative one: a busy
target no longer produces ``agent_busy``. That refusal *was* the defect.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import delivery_store as ds
from claude_teams import leases, server_simple
from claude_teams.agent_output import (
    BINDING_AMBIGUOUS,
    BINDING_BOUND,
    BINDING_UNVERIFIED,
    AgentOutput,
    BindingResult,
)
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.delivery import DELIVERY_MARKER_PREFIX

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

    def _binding(agent: dict) -> BindingResult:
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


def _busy(env: SimpleNamespace) -> None:
    """Recent activity and no ``waiting`` marker: the target is mid-task."""
    env.state.last_activity_at = 995.0


def _write_waiting_marker(env: SimpleNamespace) -> None:
    (env.session_dir / f"state-{AGENT}.json").write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 950.0}),
        encoding="utf-8",
    )


def _maybe_record(key: str = KEY, sender: str = "team-lead") -> dict | None:
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        found = txn.get(sender, key)
        return dict(found) if found else None


def _record(key: str = KEY, sender: str = "team-lead") -> dict:
    found = _maybe_record(key, sender)
    assert found is not None, f"no delivery record for {sender}/{key}"
    return found


# ==========================================================================
# B2 / B0 — a busy target is waited for, not refused
# ==========================================================================


@pytest.mark.asyncio
async def test_busy_target_that_becomes_resumable_is_delivered_from_this_call(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect, closed. ``agent_busy`` must not appear anywhere."""
    _busy(env)

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    # The target reaches its Stop hook two polls into the wait.
    env.clock.on_sleep = lambda n: _write_waiting_marker(env) if n == 2 else None

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["reason"] != "agent_busy" if "reason" in result else True
    assert result["success"] is True
    assert result["status"] == "delivered"
    assert backend.resume_calls, "the wait ended in a real resume"
    assert _record()["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_busy_target_that_never_becomes_resumable_returns_the_pending_tail(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Budget expiry with no lease ever acquired is the cooperative tail.

    It is ``queued(phase=pending)`` with a stated obligation — **never**
    ``failed``. There is one timeout, so the same instant cannot mean both
    "come back for it" and "it will never happen".
    """
    _busy(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["status"] == "queued"
    assert result["phase"] == "pending"
    assert result["reason"] != "failed"
    assert "deliver_pending" in result["sender_obligation"]
    assert backend.resume_calls == [], "nothing was sent"
    stored = _record()
    assert stored["status"] == ds.STATUS_QUEUED
    assert stored["phase"] == ds.PHASE_PENDING
    assert stored["settled_at"] is None


@pytest.mark.asyncio
async def test_the_call_budget_is_reported_so_a_caller_can_read_it(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One total budget, and a documented one — not a guess at the client's."""
    _busy(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["call_budget_s"] == 10.0


# ==========================================================================
# B1 — the state machine
# ==========================================================================


@pytest.mark.asyncio
async def test_live_child_without_a_receipt_ends_unconfirmed_then_reconciles(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound expiry with a live child is uncertainty, not failure (R6)."""
    _write_waiting_marker(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert first["status"] == "queued"
    assert first["phase"] == "unconfirmed"
    stored = _record()
    assert stored["phase"] == ds.PHASE_UNCONFIRMED
    assert stored["settled_at"] is None

    # The buffered transcript write finally lands.
    _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{stored['nonce']}"),
    )

    status = await server_simple.delivery_status(KEY)

    assert status["status"] == ds.STATUS_DELIVERED, (
        "delivery_status must actively reconcile, not echo the stale phase"
    )
    assert _record()["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_dead_child_with_no_receipt_is_terminally_failed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, False)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["status"] == "failed"
    stored = _record()
    assert stored["status"] == ds.STATUS_FAILED
    assert stored["reason"] in {"not_delivered", "resume_not_confirmed"}
    assert stored["settled_at"] is not None


@pytest.mark.asyncio
async def test_a_retry_rescans_for_the_prior_nonce_and_does_not_resend(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recipient is a conversation, not a consumer with a dedupe table.

    Duplicate suppression therefore has to happen here, before the resend.
    """
    _write_waiting_marker(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    nonce = _record()["nonce"]
    _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    again = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert again["status"] == "delivered"
    assert len(backend.resume_calls) == 1, "no duplicate prompt"


# ==========================================================================
# Idempotency
# ==========================================================================


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("", ds.KEY_REQUIRED),
        ("has space", ds.KEY_MALFORMED),
        ("x" * 500, ds.KEY_TOO_LONG),
    ],
)
@pytest.mark.asyncio
async def test_a_bad_idempotency_key_is_rejected_before_any_waiting(
    env, monkeypatch: pytest.MonkeyPatch, key: str, reason: str
) -> None:
    _busy(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", key)

    assert result["success"] is False
    assert result["reason"] == reason
    assert env.clock.sleeps == 0, "validation must precede the bounded wait"
    assert backend.resume_calls == []
    assert not env.deliveries.exists(), "a rejected call creates no record"


@pytest.mark.asyncio
async def test_same_key_and_identical_payload_creates_one_attempt_not_two(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    second = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert first["message_id"] == second["message_id"]
    assert len(backend.resume_calls) == 1
    assert len(ds.load_records(env.deliveries)) == 1


@pytest.mark.asyncio
async def test_same_key_with_a_differing_field_conflicts_and_mutates_nothing(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    before = env.deliveries.read_bytes()

    result = await server_simple.follow_up_agent(AGENT, "a DIFFERENT prompt", KEY)

    assert result["success"] is False
    assert result["reason"] == ds.IDEMPOTENCY_CONFLICT
    assert env.deliveries.read_bytes() == before, "a conflict mutates nothing"
    assert len(backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_two_senders_may_use_the_same_textual_key(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The namespace is (session, sender, key). Keys are not globally unique."""
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(
            ds.new_record(
                sender="someone-else",
                idempotency_key=KEY,
                to=AGENT,
                fingerprint="other",
                created_at=1.0,
            )
        )

    assert len(ds.load_records(env.deliveries)) == 2
    assert _record(sender="someone-else")["fingerprint"] == "other"


@pytest.mark.asyncio
async def test_response_loss_is_recoverable_through_the_key_alone(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drop the response on the floor; the sender still learns the outcome.

    This is why the record is created *before* the wait and why the key comes
    from the caller: an id minted server-side would only ever have arrived in
    the response that was lost.
    """

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)  # response lost

    recovered = await server_simple.delivery_status(KEY)

    assert recovered["success"] is True
    assert recovered["status"] == ds.STATUS_DELIVERED
    assert recovered["idempotency_key"] == KEY


@pytest.mark.asyncio
async def test_delivery_status_for_an_unknown_key_says_so(env) -> None:
    result = await server_simple.delivery_status("never-used")
    assert result["success"] is False
    assert result["reason"] == "delivery_not_found"


@pytest.mark.asyncio
async def test_delivery_status_never_returns_another_senders_record(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ds.delivery_transaction(env.deliveries) as txn:
        txn.put(
            ds.new_record(
                sender="someone-else",
                idempotency_key=KEY,
                to=AGENT,
                fingerprint="fp",
                created_at=1.0,
            )
        )

    result = await server_simple.delivery_status(KEY)

    assert result["success"] is False
    assert result["reason"] == "delivery_not_found"


# ==========================================================================
# B3 — no_delivery_path (R7)
# ==========================================================================


@pytest.mark.asyncio
async def test_a_removed_record_names_the_state(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _FakeResumeBackend())
    result = await server_simple.follow_up_agent("ghost", "next prompt", KEY)

    assert result["reason"] == "no_delivery_path"
    assert result["state"] == "record_removed"


@pytest.mark.asyncio
async def test_a_dead_agent_without_a_backend_session_names_the_state(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server_simple,
        "_resolve_agent_binding",
        lambda agent, **_: BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=900.0,
                last_message="done",
                rollout_path=str(env.transcript),
                backend_session_id=None,
            ),
        ),
    )
    monkeypatch.setattr(server_simple, "_stored_backend_session_id", lambda agent: None)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, False)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["reason"] == "no_delivery_path"
    assert result["state"] == "no_backend_session"


@pytest.mark.asyncio
async def test_a_dead_agent_with_a_valid_session_is_still_resumed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dead-but-resumable is intentionally supported and must keep working."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    calls = {"n": 0}

    def health(handle, expected_token=None):
        # The pre-existing child is dead; the resumed one is alive.
        calls["n"] += 1
        return (handle == "789", "x")

    monkeypatch.setattr(server_simple.process_manager, "health_check", health)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is True
    assert result["status"] == "delivered"


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (BINDING_UNVERIFIED, "binding_unverified"),
        (BINDING_AMBIGUOUS, "binding_ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_an_unprovable_binding_refuses_rather_than_waiting(
    env, monkeypatch: pytest.MonkeyPatch, outcome: str, reason: str
) -> None:
    """We cannot prove the target, so there is nothing safe to wait for."""
    monkeypatch.setattr(
        server_simple,
        "_resolve_agent_binding",
        lambda agent, **_: BindingResult(outcome, None),
    )
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert result["success"] is False
    assert result["reason"] == reason
    assert backend.resume_calls == []


# ==========================================================================
# B4 — the guaranteed path never touches the actionable inbox
# ==========================================================================


@pytest.mark.asyncio
async def test_a_guaranteed_message_never_appears_in_the_recipients_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"secret instruction {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "secret instruction", KEY)

    inbox = server_simple._inbox_file(SESSION, AGENT)
    text = inbox.read_text(encoding="utf-8") if inbox.exists() else ""
    assert "secret instruction" not in text
    assert _record()["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_the_guaranteed_path_does_not_disturb_an_unread_inbox_message(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rejected design advanced the recipient's cursor and ate this line.

    The cursor is a per-sender consumed *count*, so pre-advancing it past an
    audit record silently consumes every earlier unread message from the same
    sender. Nothing may touch cursors here.
    """
    inbox = server_simple._inbox_file(SESSION, AGENT)
    inbox.write_text(
        json.dumps({"from": "team-lead", "text": "earlier unread", "ts": "t"}) + "\n",
        encoding="utf-8",
    )

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert "earlier unread" in inbox.read_text(encoding="utf-8")
    assert not server_simple._inbox_cursor_file(SESSION, AGENT).exists()


# ==========================================================================
# Kill-time reconciliation, and survival of the audit trail
# ==========================================================================


@pytest.mark.asyncio
async def test_kill_reconciles_an_unread_receipt_to_delivered_not_failed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marking a delivered message failed at kill time is the original bug."""
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    nonce = _record()["nonce"]
    assert _record()["phase"] == ds.PHASE_UNCONFIRMED
    # The receipt was written but nobody has scanned it yet.
    _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    await server_simple.kill_agent(AGENT)

    assert _record()["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_kill_settles_a_genuinely_receiptless_attempt_as_failed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    _child_alive(monkeypatch, False)

    await server_simple.kill_agent(AGENT)

    stored = _record()
    assert stored["status"] == ds.STATUS_FAILED
    assert stored["reason"] == ds.REASON_NOT_DELIVERED


@pytest.mark.asyncio
async def test_delivery_records_survive_the_kill_that_purges_the_inbox(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two stores differ on purpose: audit survives, actionable state does not."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend(write_receipt))
    _child_alive(monkeypatch, True)
    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    lead_inbox = server_simple._inbox_file(SESSION, "team-lead")
    lead_inbox.write_text(
        json.dumps({"from": AGENT, "text": "from the worker", "ts": "t"}) + "\n",
        encoding="utf-8",
    )

    await server_simple.kill_agent(AGENT)

    assert "from the worker" not in lead_inbox.read_text(encoding="utf-8")
    survivor = await server_simple.delivery_status(KEY)
    assert survivor["success"] is True
    assert survivor["status"] == ds.STATUS_DELIVERED


# ==========================================================================
# The drain allow-list
# ==========================================================================


@pytest.mark.asyncio
async def test_cheap_reads_do_not_drain(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Draining on an advertised cheap read turns it into a slow mutator."""
    _write_waiting_marker(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    nonce = _record()["nonce"]
    _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    await server_simple.agent_status()
    await server_simple.check_agent(AGENT)
    await server_simple.list_agents()

    assert _record()["phase"] == ds.PHASE_UNCONFIRMED, (
        "agent_status/check_agent/list_agents must not reconcile"
    )


@pytest.mark.asyncio
async def test_deliver_pending_drains(env, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    nonce = _record()["nonce"]
    _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    result = await server_simple.deliver_pending()

    assert result["success"] is True
    assert _record()["status"] == ds.STATUS_DELIVERED
    assert any(row["idempotency_key"] == KEY for row in result["deliveries"])


@pytest.mark.asyncio
async def test_deliver_pending_completes_the_cooperative_tail(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tail must be genuinely usable, not a theoretical branch."""
    _busy(env)
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    tail = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    assert tail["phase"] == "pending"

    # The worker has since parked at its Stop hook.
    _write_waiting_marker(env)
    backend.on_resume = lambda nonce: _append(
        env.transcript,
        _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    result = await server_simple.deliver_pending()

    assert result["success"] is True
    assert _record()["status"] == ds.STATUS_DELIVERED
    assert len(backend.resume_calls) == 1


def _foreign_lease(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Park a live lease held by ANOTHER process on the target.

    That is the only way to reach the FIFO tail rather than the busy-wait tail:
    the queue only forms when a second valid caller wants a target somebody
    else has already reserved.
    """
    monkeypatch.setattr(
        server_simple.process_manager,
        "owns_process",
        lambda pid, token=None: str(pid) == "4242",
    )
    leases.save_leases(
        server_simple._leases_file(SESSION),
        {
            AGENT: {
                "lease": {
                    "generation": 0,
                    "operation_id": "foreign-op",
                    "backend_session_id": BACKEND_SESSION,
                    "nonce": "f" * 32,
                    "holder_pid": 4242,
                    "holder_create_token": "tok-4242",
                    "deadline": 1e12,
                    "acquired_at": 0.0,
                },
                "waiters": [
                    {
                        "ticket": "foreign-ticket",
                        "operation_id": "foreign-op",
                        "enqueued_at": 0.0,
                    }
                ],
            }
        },
    )


def _waiters() -> list[dict]:
    store = leases.load_leases(server_simple._leases_file(SESSION))
    return list(store.get(AGENT, {}).get("waiters", []))


@pytest.mark.asyncio
async def test_a_fresh_call_reclaims_its_place_in_the_per_target_queue(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cooperative tail must not orphan the ticket it is queued under.

    The tail tells the caller "your place in the per-target queue is
    preserved". A genuinely fresh MCP call starts with ``ticket = None``, so
    unless the ticket is derived from something the caller still holds, the
    retry appends a SECOND waiter behind an orphaned head it can never reclaim
    — the queue grows by one on every retry and the caller never advances.
    That is a dead end, which R1 forbids.
    """
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    _foreign_lease(env, monkeypatch)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    assert first["reason"] == "operation_in_progress"
    assert first["queue_position"] == 1
    assert len(_waiters()) == 2

    # A genuinely fresh call: nothing is carried over in memory, only the
    # idempotency key the caller chose and still has.
    env.clock.now = 0.0
    second = await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert second["queue_position"] == 1, (
        "a retry must reclaim its FIFO place, not queue behind its own orphan"
    )
    assert len(_waiters()) == 2, "the retry appended a second waiter for one caller"


@pytest.mark.asyncio
async def test_two_distinct_senders_keep_distinct_queue_places(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable ticket must be per-message, not per-target."""
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    _foreign_lease(env, monkeypatch)

    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)
    env.clock.now = 0.0
    other = await server_simple.follow_up_agent(AGENT, "other prompt", "k-2")

    assert other["queue_position"] == 2
    assert len(_waiters()) == 3


# ==========================================================================
# Cross-process safety of the store as the server uses it
# ==========================================================================


_WRITER = """
import sys
from pathlib import Path
from claude_teams import delivery_store as ds

path = Path(sys.argv[1])
for index in range(30):
    with ds.delivery_transaction(path) as txn:
        txn.put(
            ds.new_record(
                sender=sys.argv[2],
                idempotency_key=f"k{index}",
                to="worker",
                fingerprint="fp",
                created_at=float(index),
            )
        )
"""


def _run_second_server(tmp_path: Path, deliveries: Path) -> int:
    """Run the writer script in a real second process and return its exit code.

    Kept out of the async test body deliberately: blocking process calls inside
    a coroutine are a lint error and, more to the point, a bad habit.
    """
    script = tmp_path / "writer.py"
    script.write_text(_WRITER, encoding="utf-8")
    # S603 is suppressed below: a fixed interpreter running a script this test just
    # wrote, with paths it controls. There is no untrusted input here.
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(script), str(deliveries), "other-lead"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    return proc.wait(timeout=120)


@pytest.mark.asyncio
async def test_the_store_the_server_uses_is_safe_against_a_second_process(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second MCP *server process*, not a second thread in this one."""
    _write_waiting_marker(env)
    _install(monkeypatch, _FakeResumeBackend())
    _child_alive(monkeypatch, True)
    await server_simple.follow_up_agent(AGENT, "next prompt", KEY)

    assert _run_second_server(env.tmp_path, env.deliveries) == 0

    assert _maybe_record() is not None, "the concurrent process kept our record"
    assert len(ds.load_records(env.deliveries)) == 31
