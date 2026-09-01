"""End-to-end A3/A4/A4b/A5 behaviour through ``follow_up_agent``/``kill_agent``.

Nothing here mocks the confirmation outcome. The scanner reads real transcript
files, child liveness comes from a real process exit or an explicit liveness
stub driving a real poll loop, and the lease is the real on-disk store. The
only injected things are the clock and the sleep, so the timing cases are
deterministic — the repo already has one flaky wall-clock test and this code is
far more timing-dense than that one.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from claude_teams import cli, leases, server_simple
from claude_teams.agent_output import BINDING_BOUND, AgentOutput, BindingResult
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import (
    OWNERSHIP_OURS,
)
from claude_teams.backends.registry import canonical_backend_name
from claude_teams.delivery import DELIVERY_MARKER_PREFIX

SESSION = "session-id"
AGENT = "worker"
BACKEND_SESSION = "backend-session-id"


class _Clock:
    """Injected clock; ``sleep`` advances it rather than blocking."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeResumeBackend:
    """A resume backend that can also play the CLI's part in the transcript.

    ``on_resume`` receives the delivery nonce parsed out of the prompt the
    server actually built, which is exactly what a real CLI would receive and
    then record. Tests use it to write a genuine receipt record.
    """

    def __init__(self, on_resume=None, *, handle: str = "789") -> None:
        self.resume_calls: list[tuple[SpawnRequest, str]] = []
        self.on_resume = on_resume
        self.handle = handle
        self.overlapped = False
        self._in_flight = threading.Event()

    def supports_resume(self) -> bool:
        return True

    def default_model(self) -> str:
        return "model"

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        if self._in_flight.is_set():
            self.overlapped = True
        self._in_flight.set()
        try:
            self.resume_calls.append((request, backend_session_id))
            if self.on_resume is not None:
                self.on_resume(_nonce_of(request.prompt))
            return SimpleNamespace(process_handle=self.handle)
        finally:
            self._in_flight.clear()


class _FakeRegistry:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def resolve_name(self, name: str) -> str:
        return canonical_backend_name(name)

    def get(self, backend: str) -> object:
        return self.backend


def _nonce_of(prompt: str) -> str:
    """Extract the delivery nonce the server embedded in this attempt."""
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
def exited_pid() -> int:
    """PID of a real process that has genuinely exited.

    A real poll/exit-code transition, not a stubbed "dead" — the A3 signal is
    only worth anything if it is driven by an actual process ending.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A session with one agent and one real transcript file on disk."""
    session_dir = tmp_path / "sessions" / SESSION
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", SESSION)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})

    transcript = tmp_path / "transcript.jsonl"
    _append(transcript, _claude_user_record("the original task"))

    work = tmp_path / "work"
    work.mkdir()
    record: dict[str, object] = {
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
        # R2: follow-up is downstream-only, and the default test IDENTITY is
        # the root lead. The direction guard itself is covered in
        # tests/test_direction_guard.py.
        "spawned_by": "team-lead",
        "spawned_by_source": "spawn",
    }
    server_simple._save_agents(SESSION, [record])
    server_simple._persist_session_binding(SESSION)

    # A2 is covered elsewhere; pin the ladder to the real transcript so this
    # file exercises confirmation rather than binding.
    monkeypatch.setattr(
        server_simple,
        "_resolve_agent_binding",
        lambda agent, **_: BindingResult(
            BINDING_BOUND,
            AgentOutput(
                last_activity_at=900.0,
                last_message="done",
                rollout_path=str(transcript),
                backend_session_id=BACKEND_SESSION,
            ),
        ),
    )
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
    )


def _install(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))


def _dead_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing child is gone, so no shutdown decision is involved."""
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )


def _child_alive(monkeypatch: pytest.MonkeyPatch, alive: bool) -> None:
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (alive and handle == "789", "x"),
    )


def _write_waiting_marker(env: SimpleNamespace) -> None:
    server_simple._state_marker_file(SESSION, AGENT).write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 950.0}),
        encoding="utf-8",
    )


def _record() -> dict:
    return server_simple._load_agents(SESSION)[0]


# ==========================================================================
# A3 — child liveness as an early-failure signal only
# ==========================================================================


@pytest.mark.asyncio
async def test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record(
    env, exited_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real process exit drives this, not a mocked confirmation.

    ``claude --resume <bad-id>`` exits within a second; the returned PID is
    not evidence of anything, so the resume must fail fast and the agent
    record must be left exactly as it was.
    """
    backend = _FakeResumeBackend(handle=str(exited_pid))
    _install(monkeypatch, backend)
    before = dict(_record())

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k22")

    assert result["success"] is False
    assert result["reason"] == "resume_not_confirmed"
    assert backend.resume_calls, "the resume was attempted"
    after = _record()
    assert after["pid"] == before["pid"], "a failed resume must not repoint the record"
    assert after.get("create_token") == before.get("create_token")
    assert server_simple.PENDING_DELIVERY_FIELD not in after


# ==========================================================================
# A4 — nonce confirmation against a real transcript
# ==========================================================================


@pytest.mark.asyncio
async def test_nonce_in_the_correct_transcript_is_delivered(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k23")

    assert result["success"] is True
    assert result["status"] == "delivered"
    assert _record()["pid"] == 789


@pytest.mark.asyncio
async def test_a_surviving_old_process_growing_the_transcript_does_not_confirm(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Growth is not delivery: an old process writing proves only that it lives."""

    def noisy(nonce: str) -> None:
        for index in range(4):
            _append(
                env.transcript,
                {
                    "type": "assistant",
                    "sessionId": BACKEND_SESSION,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"still going {index}"}],
                    },
                },
            )

    backend = _FakeResumeBackend(noisy)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k24")

    assert result["success"] is False
    assert result["reason"] == "delivery_unconfirmed"


@pytest.mark.asyncio
async def test_the_old_process_writing_the_shared_state_marker_does_not_confirm(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Markers are keyed on agent NAME, so both processes write the same one.

    A marker transition therefore cannot distinguish a surviving old process
    from a freshly resumed one, and must not be treated as a receipt.
    """

    def write_marker(nonce: str) -> None:
        server_simple._state_marker_file(SESSION, AGENT).write_text(
            json.dumps({"state": "waiting", "event": "Stop", "ts": 1_000.0}),
            encoding="utf-8",
        )

    backend = _FakeResumeBackend(write_marker)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k25")

    assert result["success"] is False
    assert result["reason"] == "delivery_unconfirmed"


@pytest.mark.asyncio
async def test_a_nonce_echoed_only_in_a_diagnostic_does_not_confirm(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic, not substring: a diagnostic is not a receipt record."""

    def diagnostic(nonce: str) -> None:
        _append(
            env.transcript,
            {
                "type": "system",
                "sessionId": BACKEND_SESSION,
                "content": (
                    f"spawn argv: claude -p 'next prompt "
                    f"{DELIVERY_MARKER_PREFIX}{nonce}'"
                ),
            },
        )

    backend = _FakeResumeBackend(diagnostic)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k26")

    assert result["success"] is False
    assert result["reason"] == "delivery_unconfirmed"


@pytest.mark.asyncio
async def test_dead_child_with_no_receipt_is_definite_non_delivery(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def health(handle: str, expected_token: str | None = None):
        calls["n"] += 1
        # Survives the settle window, then dies without ever writing a receipt.
        return (calls["n"] <= 4 and handle == "789", "x")

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    monkeypatch.setattr(server_simple.process_manager, "health_check", health)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k27")

    assert result["success"] is False
    assert result["reason"] == "not_delivered"
    assert result["retriable"] is False
    assert server_simple.PENDING_DELIVERY_FIELD not in _record()


@pytest.mark.asyncio
async def test_bound_expiry_with_a_live_child_is_queued_not_terminal(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k28")

    # R6: neither delivered nor terminally failed.
    assert result["success"] is False
    assert result["status"] == "queued"
    assert result["phase"] == "unconfirmed"
    assert result["retriable"] is True
    pending = _record()[server_simple.PENDING_DELIVERY_FIELD]
    assert pending["nonce"] == _nonce_of(backend.resume_calls[0][0].prompt)


@pytest.mark.asyncio
async def test_a_later_flush_reconciles_and_does_not_resend(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry must reconcile the prior attempt before re-sending (R6)."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    first = await server_simple.follow_up_agent(AGENT, "next prompt", "k29")
    assert first["phase"] == "unconfirmed"
    nonce = _nonce_of(backend.resume_calls[0][0].prompt)

    # The buffered transcript write lands after that call returned.
    _append(
        env.transcript,
        _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
    )

    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k30")

    assert second["success"] is True
    assert second["status"] == "delivered"
    assert second["reconciled"] is True
    assert len(backend.resume_calls) == 1, "the prompt must not be delivered twice"
    assert server_simple.PENDING_DELIVERY_FIELD not in _record()
    # The response and the store must agree: a receipt the caller cannot look
    # up under its own key is the false receipt this reconciliation exists to
    # avoid producing.
    assert second["reconciled_from_key"] == "k29"
    assert (await server_simple.delivery_status("k29"))["status"] == "delivered"
    assert (await server_simple.delivery_status("k30"))["status"] == "delivered"


# ==========================================================================
# R6 — reconciliation is scoped to the request it reconciles
# ==========================================================================


def _unconfirmed_attempt(backend, env, key: str, prompt: str = "next prompt"):
    """Leave one unconfirmed attempt behind, then flush its receipt."""

    async def _run() -> str:
        result = await server_simple.follow_up_agent(AGENT, prompt, key)
        assert result["phase"] == "unconfirmed", result
        nonce = _nonce_of(backend.resume_calls[-1][0].prompt)
        _append(
            env.transcript,
            _claude_user_record(f"{prompt} {DELIVERY_MARKER_PREFIX}{nonce}"),
        )
        return nonce

    return _run()


def _rows() -> dict:
    """The durable delivery store, keyed by idempotency key."""
    from claude_teams import delivery_store

    path = server_simple._deliveries_file(SESSION)
    with delivery_store.delivery_transaction(path) as txn:
        return {str(row.get("idempotency_key")): dict(row) for row in txn.data.values()}


@pytest.mark.asyncio
async def test_a_new_key_for_the_same_request_settles_the_callers_own_row(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt must describe the key the caller actually holds.

    Answering ``delivered`` while ``delivery_status(that key)`` still says
    ``queued/pending, attempts: 0`` is a false receipt: the caller cannot
    recover the outcome from the only handle it was given.
    """
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k101")
    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k102")

    assert second["status"] == "delivered"
    assert second["reconciled_from_key"] == "k101"
    assert len(backend.resume_calls) == 1

    own = await server_simple.delivery_status("k102")
    assert own["status"] == "delivered"
    assert own["reconciled_from_key"] == "k101"
    assert own["nonce"] == "", "an aliased row must not claim the other's receipt"
    assert (await server_simple.delivery_status("k101"))["status"] == "delivered"


@pytest.mark.asyncio
async def test_a_new_key_for_a_different_request_is_really_sent(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciling a prior attempt must not swallow a different prompt."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k103")
    _write_waiting_marker(env)

    backend.on_resume = lambda nonce: _append(
        env.transcript,
        _claude_user_record(f"something else {DELIVERY_MARKER_PREFIX}{nonce}"),
    )
    second = await server_simple.follow_up_agent(AGENT, "something else", "k104")

    assert second["status"] == "delivered"
    assert second.get("reconciled") is not True
    assert len(backend.resume_calls) == 2, "the new prompt was never sent"
    assert _nonce_of(backend.resume_calls[1][0].prompt) in {
        row["nonce"] for row in _rows().values()
    }
    assert (await server_simple.delivery_status("k103"))["status"] == "delivered"


@pytest.mark.asyncio
async def test_an_option_only_difference_is_a_different_request(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``options`` are part of the fingerprint, so they are part of identity."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k105")
    _write_waiting_marker(env)

    second = await server_simple.follow_up_agent(
        AGENT, "next prompt", "k106", replace_if_idle=False
    )

    # Not aliased onto the prior attempt, and past the shortcut into the
    # ordinary send path — where this call's own option refuses a marked idle
    # child.
    assert second.get("reconciled") is not True
    assert second.get("status") != "delivered"
    assert second["reason"] == "agent_idle_but_alive"


@pytest.mark.asyncio
async def test_an_option_only_difference_is_actually_sent(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same arm, followed all the way to a real second attempt."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k118")

    # The child is gone, so ``replace_if_idle=False`` has nothing to refuse and
    # the fall-through reaches a genuine resume.
    _dead_agent(monkeypatch)
    backend.on_resume = lambda nonce: _append(
        env.transcript,
        _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
    )
    second = await server_simple.follow_up_agent(
        AGENT, "next prompt", "k119", replace_if_idle=False
    )

    assert second["status"] == "delivered"
    assert second.get("reconciled") is not True
    assert len(backend.resume_calls) == 2
    assert (await server_simple.delivery_status("k119"))["nonce"] == _nonce_of(
        backend.resume_calls[1][0].prompt
    )


@pytest.mark.asyncio
async def test_an_unresolvable_prior_attempt_is_refused_not_claimed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker whose row is gone is unknown state: refuse, keep the guard."""
    from claude_teams import delivery_store

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k107")

    # The store is damaged: the row carrying the marker's nonce is gone.
    path = server_simple._deliveries_file(SESSION)
    with delivery_store.delivery_transaction(path) as txn:
        for key in [k for k, row in txn.data.items() if row.get("nonce")]:
            del txn.data[key]
        txn.touch()

    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k108")

    assert second["success"] is False
    assert second["reason"] == "pending_attempt_unresolvable"
    assert second["status"] == "queued"
    assert second["retriable"] is True
    assert len(backend.resume_calls) == 1, "nothing may be sent under a barrier"
    assert server_simple.PENDING_DELIVERY_FIELD in _record(), (
        "the marker is the only remaining duplicate guard"
    )
    own = await server_simple.delivery_status("k108")
    assert own["status"] == "queued"
    assert own["reason"] == "pending_attempt_unresolvable"


@pytest.mark.asyncio
async def test_a_prior_attempt_settled_failed_is_not_resurrected(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row already settled ``failed`` must not become someone else's receipt.

    Reachable when kill-time cleanup settled the row on a complete negative
    scan and the transcript write landed afterwards.
    """
    from claude_teams import delivery_store

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k109")

    path = server_simple._deliveries_file(SESSION)
    with delivery_store.delivery_transaction(path) as txn:
        for row in txn.data.values():
            if row.get("nonce"):
                delivery_store.settle(
                    row, delivery_store.STATUS_FAILED, reason="not_delivered", now=1.0
                )
        txn.touch()

    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k110")

    assert second["success"] is False
    assert second["reason"] == "prior_attempt_settled_failed"
    assert second["retriable"] is True
    assert len(backend.resume_calls) == 1
    assert server_simple.PENDING_DELIVERY_FIELD in _record()
    assert (await server_simple.delivery_status("k109"))["status"] == "failed"


@pytest.mark.asyncio
async def test_two_rows_carrying_one_nonce_are_a_barrier_too(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonce the store cannot attribute to ONE request attributes to none."""
    from claude_teams import delivery_store

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k120")

    path = server_simple._deliveries_file(SESSION)
    with delivery_store.delivery_transaction(path) as txn:
        original = next(row for row in txn.data.values() if row.get("nonce"))
        twin = dict(original)
        twin["idempotency_key"] = "k121"
        twin["message_id"] = "twin"
        txn.put(twin)

    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k122")

    assert second["reason"] == "pending_attempt_unresolvable"
    assert len(backend.resume_calls) == 1
    assert server_simple.PENDING_DELIVERY_FIELD in _record()
    # Nothing was settled off an ambiguous nonce. (delivery_status is itself a
    # reconciler, so read the store directly rather than through it.)
    assert {row["status"] for row in _rows().values()} == {"queued"}


@pytest.mark.asyncio
async def test_a_late_receipt_after_a_failed_same_key_retry_does_not_resend(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settling ``failed`` must not throw away the duplicate guard.

    A complete negative scan against a dead child is definite non-delivery, but
    a transcript write can still land afterwards. If the marker went with the
    failure, the next call would have nothing to reconcile against and would
    send an instruction the target has already had.
    """
    alive = {"v": True}
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (alive["v"] and handle == "789", "x"),
    )

    first = await server_simple.follow_up_agent(AGENT, "next prompt", "k123")
    assert first["phase"] == "unconfirmed"
    nonce = _nonce_of(backend.resume_calls[0][0].prompt)

    # The child dies with no receipt, and the same-key retry settles it failed
    # once the flush grace has passed.
    alive["v"] = False
    monkeypatch.setattr(
        server_simple.time,
        "time",
        lambda: 1_000.0 + server_simple._UNCONFIRMED_FLUSH_GRACE_SECONDS + 1.0,
    )
    retry = await server_simple.follow_up_agent(AGENT, "next prompt", "k123")
    assert retry["status"] == "failed"

    # ...and only then does the buffered write arrive.
    _append(
        env.transcript,
        _claude_user_record(f"next prompt {DELIVERY_MARKER_PREFIX}{nonce}"),
    )
    third = await server_simple.follow_up_agent(AGENT, "next prompt", "k124")

    assert third["reason"] == "prior_attempt_settled_failed"
    assert len(backend.resume_calls) == 1, "the prompt must not be sent twice"


@pytest.mark.asyncio
async def test_a_same_key_retry_settles_and_clears_the_marker(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same-key half settles before ``_prepare``, so it clears too."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k111")
    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k111")

    assert second["status"] == "delivered"
    assert len(backend.resume_calls) == 1
    assert server_simple.PENDING_DELIVERY_FIELD not in _record()


@pytest.mark.asyncio
async def test_deliver_pending_aliases_an_identical_pending_row(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain shares ``_prepare``, so it must share the contract."""
    from claude_teams import delivery_store

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k112")

    # A second row for the same request, still pending — the shape the tail
    # exists to complete.
    path = server_simple._deliveries_file(SESSION)
    with delivery_store.delivery_transaction(path) as txn:
        txn.put(
            delivery_store.new_record(
                sender=server_simple.IDENTITY,
                idempotency_key="k113",
                to=AGENT,
                fingerprint=delivery_store.request_fingerprint(
                    to=AGENT,
                    prompt="next prompt",
                    options={"replace_if_idle": True},
                ),
                created_at=1_000.0,
            )
        )

    await server_simple.deliver_pending()

    assert len(backend.resume_calls) == 1, "an identical request is not resent"
    aliased = await server_simple.delivery_status("k113")
    assert aliased["status"] == "delivered"
    assert aliased["reconciled_from_key"] == "k112"


@pytest.mark.asyncio
async def test_send_message_to_a_child_uses_the_same_contract(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child ``send_message`` is the guaranteed path under another name."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k114", prompt="an instruction")
    result = await server_simple.send_message("an instruction", AGENT, "k115")

    assert result["status"] == "delivered"
    assert result["reconciled_from_key"] == "k114"
    assert len(backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_a_lost_settlement_write_sends_nothing_and_keeps_the_marker(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4/B0: an unwritable store is unknown state, so nothing may proceed."""
    from claude_teams import delivery_store

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    await _unconfirmed_attempt(backend, env, "k116")

    # The failure has to land on the SETTLEMENT write, not on the row creation
    # that precedes it — failing earlier would refuse before this code path is
    # ever reached, and the test would pass with the old bug still in place.
    real_save = delivery_store.save_records
    writes = {"n": 0}

    def flaky_save(path, data) -> bool:
        writes["n"] += 1
        return real_save(path, data) if writes["n"] == 1 else False

    monkeypatch.setattr(delivery_store, "save_records", flaky_save)
    second = await server_simple.follow_up_agent(AGENT, "next prompt", "k117")

    assert writes["n"] > 1, "the settlement write was never attempted"
    assert second["success"] is False
    assert second["reason"] == "delivery_store_unavailable"
    assert len(backend.resume_calls) == 1
    assert server_simple.PENDING_DELIVERY_FIELD in _record()
    rows = _rows()
    assert "k117" in rows, "the caller's row was created before the failure"
    assert rows["k117"]["status"] == "queued"
    assert rows["k116"]["status"] == "queued", "the old row must not be settled"


@pytest.mark.asyncio
async def test_confirmation_does_not_hold_the_registry_lock(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmation poll must not block every registry reader on the machine.

    ``_agents_transaction`` holds a cross-process file lock for its whole body,
    so the lock has to be released around resume-and-confirm.
    """
    read_ok = threading.Event()

    def probe_registry_during_poll(handle: str, expected_token: str | None = None):
        def read() -> None:
            with server_simple._agents_transaction(SESSION) as agents:
                if agents:
                    read_ok.set()

        thread = threading.Thread(target=read)
        thread.start()
        thread.join(timeout=5.0)
        return (True, "alive")

    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    monkeypatch.setattr(
        server_simple.process_manager, "health_check", probe_registry_during_poll
    )
    _write_waiting_marker(env)

    await server_simple.follow_up_agent(AGENT, "next prompt", "k31")

    assert read_ok.is_set(), (
        "a concurrent registry read blocked during confirmation, so the "
        "cross-process lock was still held"
    )


# ==========================================================================
# A4b — the per-target operation lease
# ==========================================================================


def _own_token() -> str | None:
    """The live creation token for THIS process.

    A lease held by us is identified by ``(pid, create_token)`` like any
    other: there is no shortcut for our own PID, because a lease left by an
    earlier incarnation whose PID was recycled onto us would otherwise be
    unreclaimable forever. Reservation records exactly this token.
    """
    return server_simple.process_manager.creation_token(str(os.getpid()))


def _hold_lease(env, *, holder_pid: int, token: str | None, operation_id="op-held"):
    return leases.reserve_lease(
        server_simple._leases_file(SESSION),
        AGENT,
        generation=0,
        operation_id=operation_id,
        backend_session_id=BACKEND_SESSION,
        nonce="a" * 32,
        holder_pid=holder_pid,
        holder_create_token=token,
        deadline=0.0,
        now=0.0,
        holder_probe=lambda pid, tok: OWNERSHIP_OURS,
    )


@pytest.mark.asyncio
async def test_a_second_valid_caller_queues_and_does_not_resume(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is queued, never refused — refusing would be R1's dead end."""
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k32")

    assert result["status"] == "queued"
    assert result["phase"] == "pending"
    assert result["retriable"] is True
    assert backend.resume_calls == [], "a queued caller must not resume"


@pytest.mark.asyncio
async def test_concurrent_callers_never_resume_at_the_same_time(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly one resume is in flight at a time; the other waits its turn."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"go {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    results: list[dict] = []
    lock = threading.Lock()

    def call() -> None:
        import asyncio

        outcome = asyncio.run(
            server_simple.follow_up_agent(AGENT, "next prompt", "k33")
        )
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)

    assert not backend.overlapped, "two resumes overlapped: the lease did not hold"
    assert len(results) == 2
    # Nobody was refused outright; every outcome is either delivered or a
    # retriable queued tail.
    for outcome in results:
        assert outcome.get("status") in {"delivered", "queued"}, outcome


@pytest.mark.asyncio
async def test_kill_agent_refuses_while_a_live_lease_exists(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())
    killed: list[str] = []
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: True
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "kill_process",
        lambda h, *a, **k: killed.append(h),
    )

    result = await server_simple.kill_agent(AGENT)

    assert result["success"] is False
    assert result["reason"] == "operation_in_progress"
    assert killed == [], "no process may be killed under a live lease"
    assert server_simple._load_agents(SESSION), "the record must survive the refusal"


@pytest.mark.asyncio
async def test_kill_agent_proceeds_when_the_lease_holder_is_dead(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hold_lease(env, holder_pid=999_999, token="tok-gone")
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: False
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )

    result = await server_simple.kill_agent(AGENT)

    assert result["success"] is True
    assert server_simple._load_agents(SESSION) == []
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


@pytest.mark.asyncio
async def test_kill_agent_proceeds_when_the_holder_token_no_longer_matches(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead holder's PID can be reused, so PID alone is not fencing."""
    _hold_lease(env, holder_pid=4242, token="tok-original")
    monkeypatch.setattr(
        server_simple.process_manager,
        "owns_process",
        # The PID is live but is now a different process.
        lambda handle, token: token == "tok-current",
    )
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "dead"),
    )

    result = await server_simple.kill_agent(AGENT)

    assert result["success"] is True
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


# ==========================================================================
# A4b — the CLI operator escape
# ==========================================================================


def _token() -> str:
    return server_simple._ensure_lead_token(SESSION)


def test_force_bumps_the_fencing_generation_before_anything_else(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late finalize by the original holder must be rejected afterwards."""
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token(), operation_id="op-hung")
    generation_before = server_simple._record_generation(_record())
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: False
    )

    result = CliRunner().invoke(
        cli.app,
        ["lease", "force", SESSION, AGENT, "--token", _token()],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["attempt_nonce"] == "a" * 32
    assert payload["fenced_generation"] == generation_before + 1
    assert server_simple._record_generation(_record()) == generation_before + 1
    # Deliberately NOT asserted here by calling ``finalize_lease`` directly:
    # force has already cleared the lease, so such a call is rejected for
    # absence rather than by the fence, which makes it a false positive. That
    # the bumped generation is what actually fences a *late holder whose lease
    # is still present* is proved end-to-end by
    # ``test_a_generation_bump_fences_finalization_while_the_lease_still_exists``.
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


@pytest.mark.asyncio
async def test_a_generation_bump_fences_finalization_while_the_lease_still_exists(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fence must be the CURRENT record generation, not the lease payload.

    ``lease force`` bumps the generation (step 1), then terminates the child
    and clears the lease (steps 2-3) — real work, with the registry lock
    released in between. A holder that finalizes inside that window still finds
    its own lease, at its own generation and operation id, so ``finalize_lease``
    alone says yes. Unless finalization also CASes the *record's* current
    generation, the fence loses its race and the holder rewrites a record
    describing a delivery the operator has already torn down.
    """
    _dead_agent(monkeypatch)

    def bump_generation_then_deliver(nonce: str) -> None:
        # Exactly step 1 of ``lease force``: fence, lease untouched.
        with server_simple._agents_transaction(SESSION) as agents:
            server_simple._bump_generation(agents[0])
            server_simple._save_agents_transaction(SESSION, agents)
        _append(
            env.transcript,
            _claude_user_record(f"next {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    _install(monkeypatch, _FakeResumeBackend(bump_generation_then_deliver))
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "next prompt", "k-fence")

    assert result["success"] is False
    assert result["reason"] == "operation_superseded"
    assert _record()["pid"] == 123, "a fenced holder rewrote the agent record"
    # The operator's bump is the last word on this record: a fenced attempt
    # neither writes it nor advances the generation past the fence.
    assert server_simple._record_generation(_record()) == 1
    # The lease is still released, by the caller-loop ``finally`` rather than by
    # the finalize CAS — a lease held by a process that has stopped working on
    # it would block every later caller for nothing.
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


def test_force_refuses_a_live_holder_that_is_not_yet_overdue(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Force" is documented for a live-but-**overdue** holder, and enforces it.

    Without the deadline comparison the operator escape can terminate a
    delivery that is still comfortably inside its lease — the child is killed
    and the generation fenced for an attempt that was doing nothing wrong.
    """
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())
    lease_path = server_simple._leases_file(SESSION)
    data = leases.load_leases(lease_path)
    data[AGENT]["lease"]["deadline"] = 10_000.0
    leases.save_leases(lease_path, data)
    monkeypatch.setattr(cli.time, "time", lambda: 100.0)
    generation_before = server_simple._record_generation(_record())

    result = CliRunner().invoke(
        cli.app, ["lease", "force", SESSION, AGENT, "--token", _token()]
    )

    assert result.exit_code == 3
    assert "not yet overdue" in result.output
    assert server_simple._record_generation(_record()) == generation_before
    assert leases.active_lease(lease_path, AGENT) is not None


def test_force_requires_the_session_recovery_token(env) -> None:
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())

    result = CliRunner().invoke(
        cli.app, ["lease", "force", SESSION, AGENT, "--token", "wrong"]
    )

    assert result.exit_code == 2
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is not None


def test_clear_refuses_a_provably_live_holder(env) -> None:
    """Clearing under a live holder would let a second caller resume into it."""
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())

    result = CliRunner().invoke(
        cli.app, ["lease", "clear", SESSION, AGENT, "--token", _token()]
    )

    assert result.exit_code == 3
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is not None


def test_clear_removes_a_dead_holders_lease(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hold_lease(env, holder_pid=999_999, token="tok-gone")
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: False
    )

    result = CliRunner().invoke(
        cli.app, ["lease", "clear", SESSION, AGENT, "--token", _token()]
    )

    assert result.exit_code == 0, result.output
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


def test_inspect_reports_the_attempt_nonce(env) -> None:
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token())

    result = CliRunner().invoke(
        cli.app, ["lease", "inspect", SESSION, AGENT, "--token", _token()]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["attempt_nonce"] == "a" * 32
    assert payload["holder_live"] is True


# ==========================================================================
# A5 — unique prompt files and their lifecycle
# ==========================================================================


@pytest.mark.asyncio
async def test_each_call_writes_a_distinct_prompt_file(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)
    sensitive = "line one\n'quoted'"

    await server_simple.follow_up_agent(AGENT, sensitive, "k34")
    first = sorted(server_simple._prompts_dir(SESSION).iterdir())
    _write_waiting_marker(env)
    await server_simple.follow_up_agent(AGENT, sensitive, "k35")
    second = sorted(server_simple._prompts_dir(SESSION).iterdir())

    assert len(first) == 1
    assert len(second) == 2, "a second call must not overwrite the first file"


@pytest.mark.asyncio
async def test_an_unconfirmed_attempt_keeps_its_prompt_file(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-running CLI may not have read it yet — timeout is not licence."""
    backend = _FakeResumeBackend()
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "line one\n'quoted'", "k36")

    assert result["phase"] == "unconfirmed"
    assert list(server_simple._prompts_dir(SESSION).iterdir())


@pytest.mark.asyncio
async def test_a_confirmed_attempt_removes_its_prompt_file(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt IS the read, so removal cannot race a pending read."""

    def write_receipt(nonce: str) -> None:
        _append(
            env.transcript,
            _claude_user_record(f"go {DELIVERY_MARKER_PREFIX}{nonce}"),
        )

    backend = _FakeResumeBackend(write_receipt)
    _install(monkeypatch, backend)
    _child_alive(monkeypatch, True)

    result = await server_simple.follow_up_agent(AGENT, "line one\n'quoted'", "k37")

    assert result["status"] == "delivered"
    assert list(server_simple._prompts_dir(SESSION).iterdir()) == []


@pytest.mark.asyncio
async def test_kill_removes_prompt_files_only_once_the_child_is_gone(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = server_simple._prompts_dir(SESSION)
    prompts.mkdir(parents=True, exist_ok=True)
    orphan = prompts / f"{AGENT}.{'b' * 32}.prompt.txt"
    orphan.write_text("body", encoding="utf-8")
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: True
    )
    monkeypatch.setattr(
        server_simple.process_manager, "kill_process", lambda h, *a, **k: None
    )

    await server_simple.kill_agent(AGENT)

    assert not orphan.exists()


def test_a_fresh_prompt_file_is_not_garbage_collected_while_the_child_lives(
    env,
) -> None:
    prompts = server_simple._prompts_dir(SESSION)
    prompts.mkdir(parents=True, exist_ok=True)
    fresh = prompts / f"{AGENT}.{'c' * 32}.prompt.txt"
    fresh.write_text("body", encoding="utf-8")

    server_simple._gc_prompt_files(SESSION, AGENT, child_exited=False)

    assert fresh.exists(), "age-based GC must not delete a file a CLI may still read"


def test_an_aged_prompt_file_is_garbage_collected(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = server_simple._prompts_dir(SESSION)
    prompts.mkdir(parents=True, exist_ok=True)
    old = prompts / f"{AGENT}.{'d' * 32}.prompt.txt"
    old.write_text("body", encoding="utf-8")
    ancient = 1.0
    os.utime(old, (ancient, ancient))
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000_000.0)

    server_simple._gc_prompt_files(SESSION, AGENT, child_exited=False)

    assert not old.exists()


@pytest.mark.asyncio
async def test_an_orphaned_sidecar_is_collected_by_an_ordinary_follow_up(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The age GC has to be REACHABLE, not merely correct.

    A sidecar written just before a spawn or resume raised is left with no
    agent record naming it. `_gc_prompt_files` was called only from
    `kill_agent`, so nothing could ever collect it: there is no agent left to
    kill. The tests that call the helper directly prove the predicate, not that
    any production path runs it.

    The orphan here belongs to an agent name that does not exist, so a
    per-agent sweep keyed on the agent being delivered to cannot reach it.
    """
    prompts = server_simple._prompts_dir(SESSION)
    prompts.mkdir(parents=True, exist_ok=True)
    orphan = prompts / f"vanished-agent.{'e' * 32}.prompt.txt"
    orphan.write_text("body", encoding="utf-8")
    ancient = 1.0
    os.utime(orphan, (ancient, ancient))
    monkeypatch.setattr(server_simple.time, "time", lambda: 1_000_000.0)

    _dead_agent(monkeypatch)
    _install(monkeypatch, _FakeResumeBackend())

    await server_simple.follow_up_agent(AGENT, "next prompt", "k-gc")

    assert not orphan.exists(), (
        "no production path collects an orphaned sidecar; it lives forever"
    )


def test_force_revalidates_the_operation_before_fencing_or_killing(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CAS must protect the fence and the kill, not only the final clear.

    ``lease force`` read the lease outside the registry lock, then fenced,
    killed and only afterwards compared operation ids. If the inspected
    operation finalized and a queued caller was legitimately granted the target
    in between, the command destroyed THAT caller's live delivery and reported
    the mismatch afterwards, when nothing could be undone.

    The handoff is simulated at the only point it can happen — between the
    out-of-lock read and the locked sequence — by replacing the stored lease.
    """
    _hold_lease(env, holder_pid=os.getpid(), token=_own_token(), operation_id="op-hung")
    lease_path = server_simple._leases_file(SESSION)
    generation_before = server_simple._record_generation(_record())
    killed: list[str] = []
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda h, t: True
    )
    monkeypatch.setattr(server_simple.process_manager, "kill_process", killed.append)

    original_transaction = server_simple._agents_transaction

    def _hand_off_then_enter(session_id: str):
        # A successor is granted the target in the window the old ordering
        # left unprotected.
        data = leases.load_leases(lease_path)
        data[AGENT]["lease"]["operation_id"] = "op-successor"
        leases.save_leases(lease_path, data)
        monkeypatch.setattr(server_simple, "_agents_transaction", original_transaction)
        return original_transaction(session_id)

    monkeypatch.setattr(server_simple, "_agents_transaction", _hand_off_then_enter)

    result = CliRunner().invoke(
        cli.app, ["lease", "force", SESSION, AGENT, "--token", _token()]
    )

    assert result.exit_code == 4, result.output
    assert killed == [], "the successor's child was terminated before the CAS ran"
    assert server_simple._record_generation(_record()) == generation_before, (
        "the successor's delivery was fenced before the CAS ran"
    )
    surviving = leases.active_lease(lease_path, AGENT)
    assert surviving is not None
    assert surviving.operation_id == "op-successor"
