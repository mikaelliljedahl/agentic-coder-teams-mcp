"""N5 barrier and two-stage native method selection (plan v3.1 §2.1, §4 tests 1-3).

The carriers themselves (``codex queue`` dispatch, the delivery mailbox) are
later steps. What is pinned here is everything around them:

- the N5 barrier, which reads the delivery store and holds regardless of any
  flag, so an unresolved native attempt can never be followed by a resume;
- E0-E6 eligibility, evaluated before the idle gate (stage 1) and again under
  the granted lease (stage 2);
- the ``method``/``carrier`` row fields and their flag-gated public view;
- that with no carrier implemented, every flag combination still resumes.

As in ``test_bounded_delivery``, nothing mocks a delivery outcome: receipts are
real transcript records and the store is the real on-disk store.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_teams import delivery_store as ds
from claude_teams import leases, native_wake, server_simple
from claude_teams.agent_output import BINDING_BOUND, AgentOutput, BindingResult
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.delivery import DELIVERY_MARKER_PREFIX

SESSION = "session-id"
AGENT = "worker"
BACKEND_SESSION = "backend-session-id"
KEY = "k-1"
LEAD = "team-lead"
FLAGS = (
    "WIN_AGENT_TEAMS_NATIVE_WAKE",
    "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX",
)
#: The public delivery view exactly as ``main`` ships it.
BASELINE_VIEW_KEYS = {
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


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


class _FakeResumeBackend:
    def __init__(self, transcript: Path) -> None:
        self.transcript = transcript
        self.resume_calls: list[tuple[SpawnRequest, str]] = []

    def supports_resume(self) -> bool:
        return True

    def default_model(self) -> str:
        return "model"

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        self.resume_calls.append((request, backend_session_id))
        _, _, tail = request.prompt.partition(DELIVERY_MARKER_PREFIX)
        nonce = tail.split()[0].strip("]")
        _append(self.transcript, _user_record(f"p {DELIVERY_MARKER_PREFIX}{nonce}"))
        return SimpleNamespace(process_handle="789")


class _FakeRegistry:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def get(self, backend: str) -> object:
        return self.backend


def _user_record(text: str) -> dict:
    return {
        "type": "user",
        "sessionId": BACKEND_SESSION,
        "message": {"role": "user", "content": text},
    }


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _agent_record(work: Path, backend: str = "claude-code", **extra: object) -> dict:
    return {
        "name": AGENT,
        "pid": 123,
        "backend": backend,
        "session_id": SESSION,
        "status": "running",
        "spawned_at": 100.0,
        "cwd": str(work),
        "backend_session_id": BACKEND_SESSION,
        "model": "model",
        "permission_mode": "bypass",
        "reasoning_effort": None,
        "correlation_id": "corr-native",
        "spawned_by": LEAD,
        "spawned_by_source": "spawn",
        "create_token": "tok-123",
        **extra,
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    session_dir = tmp_path / "sessions" / SESSION
    (session_dir / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server_simple, "_SESSION_BASE", tmp_path / "sessions")
    monkeypatch.setattr(server_simple, "_session_id", SESSION)
    monkeypatch.setattr(server_simple, "_inbox_locks", {})

    transcript = tmp_path / "transcript.jsonl"
    _append(transcript, _user_record("the original task"))
    work = tmp_path / "work"
    work.mkdir()
    server_simple._save_agents(SESSION, [_agent_record(work)])
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
    monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (True, "x"),
    )
    monkeypatch.setattr(
        server_simple.process_manager, "owns_process", lambda *a, **k: True
    )
    monkeypatch.setattr(
        server_simple.process_manager, "graceful_shutdown", lambda *a, **k: True
    )
    backend = _FakeResumeBackend(transcript)
    monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    return SimpleNamespace(
        tmp_path=tmp_path,
        work=work,
        transcript=transcript,
        clock=clock,
        session_dir=session_dir,
        state=state,
        backend=backend,
        monkeypatch=monkeypatch,
    )


def _flags(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for suffix, value in values.items():
        name = "WIN_AGENT_TEAMS_" + suffix
        monkeypatch.setenv(name, value)


def _all_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _flags(monkeypatch, NATIVE_WAKE="1", NATIVE_DOWNSTREAM="1")


def _idle(env: SimpleNamespace) -> None:
    (env.session_dir / f"state-{AGENT}.json").write_text(
        json.dumps({"state": "waiting", "event": "Stop", "ts": 950.0}),
        encoding="utf-8",
    )


def _busy(env: SimpleNamespace) -> None:
    env.state.last_activity_at = 995.0


def _set_agent(env: SimpleNamespace, **fields: object) -> None:
    agents = server_simple._load_agents(SESSION)
    agents[0].update(fields)
    server_simple._save_agents(SESSION, agents)


def _row(key: str = KEY, sender: str = LEAD) -> dict:
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        found = txn.get(sender, key)
    assert found is not None, f"no row for {sender}/{key}"
    return dict(found)


def _seed_native_row(
    *,
    key: str = "earlier-1",
    sender: str = LEAD,
    to: str = AGENT,
    method: str = ds.METHOD_CODEX_QUEUE,
    status: str = ds.STATUS_QUEUED,
    phase: str = ds.PHASE_SENT,
    reason: str = "",
    fingerprint: str = "f",
) -> None:
    """Persist a row as a native attempt would have left it (flags since off)."""
    record = ds.new_record(
        sender=sender,
        idempotency_key=key,
        to=to,
        fingerprint=fingerprint,
        created_at=500.0,
    )
    record.update(
        {
            "status": status,
            "phase": phase,
            "reason": reason,
            "nonce": "n" * 32,
            "operation_id": "op-native",
            "attempts": 1,
            ds.METHOD_FIELD: method,
            ds.CARRIER_FIELD: {"backend": "codex", "backend_session_id": "old"},
        }
    )
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        txn.put(record)


def _register_dispatcher(env: SimpleNamespace, method: str) -> list:
    """Register a stand-in carrier; the real ones are later steps."""
    calls: list = []

    def dispatch(session_id: str, record: dict, plan, deadline: float) -> dict:
        calls.append(plan)
        return server_simple._with_public_status(
            {"success": False, "name": plan.agent_name, "reason": "stand_in"},
            record,
        )

    env.monkeypatch.setitem(server_simple._NATIVE_DISPATCH, method, dispatch)
    return calls


def _claude_channel_proven(env: SimpleNamespace, proven=None) -> None:
    capability = {"channel": "available"}
    env.monkeypatch.setattr(
        server_simple,
        "_claude_delivery_capability",
        proven or (lambda *a, **k: capability),
    )


def _native_claude_target(env: SimpleNamespace) -> None:
    _all_on(env.monkeypatch)
    _set_agent(env, interactive=True, dispatch_epoch=3)


# ==========================================================================
# §4 test 1 — baselines
# ==========================================================================


@pytest.mark.asyncio
async def test_flag_off_golden_row_view_and_result(env) -> None:
    """Master off, no native state: nothing new on the row, view or result."""
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    status = await server_simple.delivery_status(KEY)

    assert result["status"] == "delivered"
    assert "method" not in result
    row = _row()
    assert ds.METHOD_FIELD not in row
    assert ds.CARRIER_FIELD not in row
    assert set(status) - {"success"} == BASELINE_VIEW_KEYS
    assert not (env.session_dir / "dispatch-epochs.json").exists()


@pytest.mark.asyncio
async def test_flag_off_with_persisted_native_row_is_still_blocked(env) -> None:
    """N5 is not an eligibility condition: no flag lifts it."""
    _idle(env)
    _seed_native_row()

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert env.backend.resume_calls == [], "N5 never falls through to resume"
    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_PENDING
    assert result["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert result["blocking_key"] == "earlier-1"
    assert result["retriable"] is True
    assert "sender_obligation" in result
    assert "method" not in result
    # The flag-off wording must not advertise the feature or its switches.
    assert "WIN_AGENT_TEAMS" not in result["detail"]
    assert "native" not in result["detail"].lower()
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["attempts"] == 0
    assert row["nonce"] == ""


@pytest.mark.asyncio
async def test_master_on_downstream_off_records_resume(env) -> None:
    _flags(env.monkeypatch, NATIVE_WAKE="1")
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    status = await server_simple.delivery_status(KEY)

    assert result["status"] == "delivered"
    assert result["method"] == ds.METHOD_RESUME
    assert _row()[ds.METHOD_FIELD] == ds.METHOD_RESUME
    assert ds.CARRIER_FIELD not in _row()
    assert status["method"] == ds.METHOD_RESUME
    assert len(env.backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_eligible_target_without_an_implemented_carrier_resumes(env) -> None:
    """Every flag on, every E true, but no carrier registered: plain resume."""
    _native_claude_target(env)
    _claude_channel_proven(env)
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == "delivered"
    assert result["method"] == ds.METHOD_RESUME
    assert len(env.backend.resume_calls) == 1


_GOLDEN_ROW_KEYS = {
    "attempted_at",
    "attempts",
    "created_at",
    "fingerprint",
    "idempotency_key",
    "message_id",
    "nonce",
    "operation_id",
    "options",
    "phase",
    "prompt",
    "prompt_file",
    "reason",
    "sender",
    "settled_at",
    "status",
    "target_snapshot",
    "to",
}


def _golden(payload: object) -> object:
    """Blank the per-run identifiers so a payload can be compared literally."""
    if isinstance(payload, list):
        return [_golden(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    return {
        key: "<id>" if key in {"message_id", "nonce"} else _golden(value)
        for key, value in payload.items()
    }


@pytest.mark.asyncio
async def test_flag_off_payloads_are_golden(env) -> None:
    """Master off: every payload is exactly what ``main`` returns.

    The literals were captured from ``main`` (pre-feature) with this scenario;
    any new key, even an empty one, fails here.
    """
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    by_key = await server_simple.delivery_status(KEY)
    by_to = await server_simple.delivery_status(to=AGENT)
    drained = await server_simple.deliver_pending()

    view = {
        "attempts": 1,
        "created_at": 1000.0,
        "idempotency_key": KEY,
        "message_id": "<id>",
        "nonce": "<id>",
        "phase": "settled",
        "reason": "",
        "settled_at": 1000.0,
        "status": "delivered",
        "to": AGENT,
    }
    assert _golden(result) == {
        "backend": "claude-code",
        "backend_session_id": BACKEND_SESSION,
        "call_budget_s": 10.0,
        "idempotency_key": KEY,
        "message_id": "<id>",
        "name": AGENT,
        "phase": "settled",
        "pid": 789,
        "replaced_existing": True,
        "session_id": SESSION,
        "status": "delivered",
        "success": True,
    }
    assert _golden(by_key) == {"success": True, **view}
    assert _golden(by_to) == {
        "success": True,
        "to": AGENT,
        "deliveries": [view],
        "note": (
            "Convenience list. Use delivery_status(idempotency_key) "
            "to identify one specific message."
        ),
    }
    assert _golden(drained) == {
        "success": True,
        "attempted": 0,
        "deliveries": [view],
        "refusals": [],
    }
    assert set(_row()) == _GOLDEN_ROW_KEYS


# ==========================================================================
# §4 test 2 — N5
# ==========================================================================


@pytest.mark.parametrize("method", sorted(ds.NATIVE_METHODS))
@pytest.mark.parametrize("phase", [ds.PHASE_SENT, ds.PHASE_UNCONFIRMED])
@pytest.mark.parametrize("sender", [LEAD, "other-lead"])
@pytest.mark.asyncio
async def test_n5_blocks_across_senders_methods_and_phases(
    env, method: str, phase: str, sender: str
) -> None:
    """``sent`` covers an in-flight queue call, ref write or finalisation."""
    _all_on(env.monkeypatch)
    _idle(env)
    _seed_native_row(method=method, phase=phase, sender=sender)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert result["blocking_key"] == "earlier-1"
    assert result["blocking_sender"] == sender
    assert env.backend.resume_calls == []


@pytest.mark.parametrize(
    "row",
    [
        {"status": ds.STATUS_DELIVERED, "phase": ds.PHASE_SETTLED},
        {
            "status": ds.STATUS_FAILED,
            "phase": ds.PHASE_SETTLED,
            "reason": "operator_released",
        },
        {"phase": ds.PHASE_PENDING},
        {"method": ds.METHOD_RESUME},
        {"to": "someone-else"},
    ],
    ids=["delivered", "operator-released", "pending", "resume-row", "other-target"],
)
@pytest.mark.asyncio
async def test_n5_ignores_resolved_resume_and_foreign_rows(env, row: dict) -> None:
    _idle(env)
    _seed_native_row(**row)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == "delivered"
    assert len(env.backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_n5_survives_same_name_replacement(env) -> None:
    """The store, not the registry record, is what N5 reads."""
    _idle(env)
    _seed_native_row()
    _set_agent(env, pid=456, create_token="tok-456", spawned_at=200.0)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert env.backend.resume_calls == []


@pytest.mark.asyncio
async def test_native_attempt_then_flags_off_then_retry_is_still_blocked(env) -> None:
    _native_claude_target(env)
    _claude_channel_proven(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    first = await server_simple.follow_up_agent(AGENT, "first", KEY)
    assert first["method"] == ds.METHOD_CLAUDE_MAILBOX
    assert len(dispatched) == 1

    _flags(env.monkeypatch)
    second = await server_simple.follow_up_agent(AGENT, "second", "k-2")

    assert second["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert second["blocking_key"] == KEY
    assert "method" not in second
    assert env.backend.resume_calls == []


@pytest.mark.asyncio
async def test_n5_appearing_between_stage_one_and_stage_two_blocks(env) -> None:
    """A native row committed after stage 1 is caught at the commit itself."""
    _flags(env.monkeypatch, NATIVE_WAKE="1")
    _idle(env)
    original = server_simple._build_resume_request

    def build_then_race(*args, **kwargs):
        _seed_native_row()
        return original(*args, **kwargs)

    env.monkeypatch.setattr(server_simple, "_build_resume_request", build_then_race)

    # A quote forces the Claude sidecar, so its cleanup is observable.
    result = await server_simple.follow_up_agent(AGENT, "say 'next'", KEY)

    assert result["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert env.backend.resume_calls == []
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["attempts"] == 0
    assert ds.METHOD_FIELD not in row
    leases_path = server_simple._leases_file(SESSION)
    assert leases.active_lease(leases_path, AGENT) is None
    assert not list(env.tmp_path.rglob(f"{AGENT}.*.prompt.txt")), "sidecar removed"


@pytest.mark.asyncio
async def test_stage_two_checks_n5_before_re_evaluating_eligibility(env) -> None:
    """Under the lease N5 comes first: a barrier needs no eligibility verdict."""
    _native_claude_target(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    probes: list = []

    def proven_then_raced(*args, **kwargs):
        probes.append(args)
        if len(probes) == 1:
            # Stage 1 has passed its N5 check; another sender's native
            # attempt lands before this call reaches stage 2.
            _seed_native_row(sender="other-lead")
        return {"channel": "available"}

    _claude_channel_proven(env, proven_then_raced)
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert result["blocking_sender"] == "other-lead"
    assert len(probes) == 1, "stage 2 stopped at N5, before E0-E6"
    assert dispatched == []
    assert env.backend.resume_calls == []
    row = _row()
    assert row["phase"] == ds.PHASE_PENDING
    assert row["attempts"] == 0
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


@pytest.mark.asyncio
async def test_same_key_retry_of_an_unresolved_native_attempt_is_not_resent(
    env,
) -> None:
    """The caller's own row is not a barrier: it reconciles, as today."""
    _idle(env)
    _seed_native_row(
        key=KEY,
        fingerprint=ds.request_fingerprint(
            to=AGENT, prompt="next", options={"replace_if_idle": True}
        ),
    )

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["reason"] == server_simple.REASON_ATTEMPT_UNRESOLVED
    assert env.backend.resume_calls == []
    row = _row()
    assert row["status"] == ds.STATUS_QUEUED
    assert row["phase"] in {ds.PHASE_SENT, ds.PHASE_UNCONFIRMED}
    assert row["nonce"] == "n" * 32, "the attempt was not replaced"


# ==========================================================================
# §4 test 3 — selection
# ==========================================================================


def _eligibility_case(
    env: SimpleNamespace, backend: str, *, prompt: str = "hello"
) -> dict:
    """Arguments under which every E0-E6 holds for ``backend``."""
    _all_on(env.monkeypatch)
    home = env.tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    agent = _agent_record(
        env.work,
        backend=backend,
        interactive=True,
        dispatch_epoch=3,
        **({"codex_home": str(home)} if backend == "codex" else {}),
    )
    _idle(env)
    env.monkeypatch.setattr(
        native_wake, "verify_codex_thread", lambda home, thread: (True, "")
    )
    env.monkeypatch.setattr(
        server_simple, "_codex_queue_binary", lambda: "C:/codex/codex.exe"
    )
    env.monkeypatch.setattr(
        server_simple, "_delivery_owner_lock_held", lambda *a, **k: True
    )
    (env.session_dir / f"native-delivery-{AGENT}.json").write_text(
        json.dumps(
            {
                "pid": 999,
                "create_token": "poster",
                "host_pid": 123,
                "host_create_token": "tok-123",
                "backend_session_id": BACKEND_SESSION,
                "dispatch_epoch": 3,
                "channel": "available",
                "heartbeat_ts": 999.5,
            }
        ),
        encoding="utf-8",
    )
    return {
        "agent": agent,
        "backend_name": backend,
        "alive": True,
        "binding": server_simple._resolve_agent_binding(agent),
        "backend_session_id": BACKEND_SESSION,
        "prompt": prompt,
    }


def _candidate(case: dict) -> tuple:
    return server_simple._native_candidate(SESSION, AGENT, **case)


@pytest.mark.parametrize(
    ("backend", "method"),
    [("codex", ds.METHOD_CODEX_QUEUE), ("claude-code", ds.METHOD_CLAUDE_MAILBOX)],
)
def test_all_conditions_hold_selects_the_native_candidate(
    env, backend: str, method: str
) -> None:
    assert _candidate(_eligibility_case(env, backend)) == (method, "")


def test_pi_never_has_a_native_candidate(env) -> None:
    method, _ = _candidate(_eligibility_case(env, "pi"))
    assert method is None


def _marker(env: SimpleNamespace, **changes: object) -> None:
    path = env.session_dir / f"native-delivery-{AGENT}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


_BREAKS = {
    "E0-master-off": ("E0", lambda env, case: _flags(env.monkeypatch)),
    "E0-downstream-off": (
        "E0",
        lambda env, case: _flags(env.monkeypatch, NATIVE_WAKE="1"),
    ),
    "E1-dead": ("E1", lambda env, case: case.update(alive=False)),
    "E1-unbound": (
        "E1",
        lambda env, case: case.update(binding=BindingResult("pending")),
    ),
    "E1-no-session": ("E1", lambda env, case: case.update(backend_session_id="")),
    "E2-missing": ("E2", lambda env, case: case["agent"].pop("interactive")),
    "E2-false": ("E2", lambda env, case: case["agent"].update(interactive=False)),
    "E5-too-large": (
        "E5",
        lambda env, case: case.update(prompt="x" * server_simple.NATIVE_INLINE_MAX),
    ),
}

_CODEX_BREAKS = {
    "E3-thread-unverified": (
        "E3",
        lambda env, case: env.monkeypatch.setattr(
            native_wake, "verify_codex_thread", lambda h, t: (False, "archived")
        ),
    ),
    "E3-no-home": ("E3", lambda env, case: case["agent"].pop("codex_home")),
    "E4-shim": (
        "E4",
        lambda env, case: env.monkeypatch.setattr(
            server_simple, "_codex_queue_binary", lambda: "C:/npm/codex.CMD"
        ),
    ),
    "E4-undiscoverable": (
        "E4",
        lambda env, case: env.monkeypatch.setattr(
            server_simple, "_codex_queue_binary", lambda: ""
        ),
    ),
    "E6-busy": (
        "E6",
        lambda env, case: (env.session_dir / f"state-{AGENT}.json").unlink(),
    ),
    "E0-half-off": (
        "E0",
        lambda env, case: env.monkeypatch.setenv(
            "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX", "0"
        ),
    ),
}

_CLAUDE_BREAKS = {
    "E3-no-marker": (
        "E3",
        lambda env, case: (env.session_dir / f"native-delivery-{AGENT}.json").unlink(),
    ),
    "E3-stale-heartbeat": ("E3", lambda env, case: _marker(env, heartbeat_ts=900.0)),
    "E3-other-epoch": ("E3", lambda env, case: _marker(env, dispatch_epoch=2)),
    "E3-other-session": (
        "E3",
        lambda env, case: _marker(env, backend_session_id="other"),
    ),
    "E3-other-host": ("E3", lambda env, case: _marker(env, host_pid=456)),
    "E3-other-host-token": (
        "E3",
        lambda env, case: _marker(env, host_create_token="tok-reused"),
    ),
    "E3-owner-lock-free": (
        "E3",
        lambda env, case: env.monkeypatch.setattr(
            server_simple, "_delivery_owner_lock_held", lambda *a, **k: False
        ),
    ),
    "E4-channel-unavailable": (
        "E4",
        lambda env, case: _marker(env, channel="socket_missing"),
    ),
    "E0-half-off": (
        "E0",
        lambda env, case: env.monkeypatch.setenv(
            "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE", "0"
        ),
    ),
}


@pytest.mark.parametrize("name", sorted({**_BREAKS, **_CODEX_BREAKS}))
def test_each_false_condition_rules_out_codex(env, name: str) -> None:
    condition, apply = {**_BREAKS, **_CODEX_BREAKS}[name]
    case = _eligibility_case(env, "codex")
    apply(env, case)
    method, reason = _candidate(case)
    assert method is None
    assert reason.startswith(condition)


@pytest.mark.parametrize("name", sorted({**_BREAKS, **_CLAUDE_BREAKS}))
def test_each_false_condition_rules_out_claude(env, name: str) -> None:
    condition, apply = {**_BREAKS, **_CLAUDE_BREAKS}[name]
    case = _eligibility_case(env, "claude-code")
    apply(env, case)
    method, reason = _candidate(case)
    assert method is None
    assert reason.startswith(condition)


def test_busy_claude_target_is_still_a_candidate(env) -> None:
    """Claude's poster enforces idleness itself; E6 is Codex-only."""
    case = _eligibility_case(env, "claude-code")
    (env.session_dir / f"state-{AGENT}.json").unlink()
    assert _candidate(case) == (ds.METHOD_CLAUDE_MAILBOX, "")


def test_inline_limit_counts_encoded_bytes_marker_included(env) -> None:
    """16 KiB of UTF-8 for the delivered text, not code points of the prompt."""
    marker_len = len(server_simple._native_delivered_text("", "0" * 32).encode())
    room = server_simple.NATIVE_INLINE_MAX - marker_len
    fits = _eligibility_case(env, "codex", prompt="a" * room)
    assert _candidate(fits)[0] == ds.METHOD_CODEX_QUEUE
    # One non-BMP character is 4 UTF-8 bytes: swapping it in overflows.
    over = _eligibility_case(env, "codex", prompt="a" * (room - 1) + "\U0001f600")
    assert _candidate(over) == (None, "E5_too_large")


@pytest.mark.parametrize(
    "combo",
    [
        {"NATIVE_WAKE": "1", "NATIVE_DOWNSTREAM": "1", "NATIVE_WAKE_CLAUDE": "0"},
        {"NATIVE_WAKE": "1", "NATIVE_DOWNSTREAM": "1", "NATIVE_WAKE_CODEX": "0"},
        {"NATIVE_WAKE": "1", "NATIVE_DOWNSTREAM": "1"},
        {"NATIVE_WAKE": "1", "NATIVE_DOWNSTREAM": "1", "NATIVE_WAKE_CLAUDE": "1"},
    ],
)
@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_half_switches_only_disable_their_own_half(
    env, combo: dict, backend: str
) -> None:
    case = _eligibility_case(env, backend)
    _flags(env.monkeypatch, **combo)
    half = "CODEX" if backend == "codex" else "CLAUDE"
    disabled = combo.get(f"NATIVE_WAKE_{half}") == "0"
    method, reason = _candidate(case)
    assert (method is None) is disabled
    if disabled:
        assert reason.startswith("E0")


@pytest.mark.asyncio
async def test_no_channel_is_probed_while_no_carrier_is_implemented(env) -> None:
    """Without a carrier to hand to, stage 1 costs nothing and cannot fail."""
    case = _eligibility_case(env, "codex")
    _set_agent(env, **case["agent"])
    probes: list = []
    env.monkeypatch.setattr(
        native_wake,
        "verify_codex_thread",
        lambda home, thread: probes.append("thread") or (True, ""),
    )
    env.monkeypatch.setattr(
        server_simple,
        "_codex_queue_binary",
        lambda: probes.append("binary") or "C:/codex/codex.exe",
    )

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    # The stand-in transcript is Claude-shaped, so the Codex receipt scan does
    # not settle it; what matters is the carrier that was used.
    assert result["method"] == ds.METHOD_RESUME
    assert len(env.backend.resume_calls) == 1
    assert probes == []


@pytest.mark.asyncio
async def test_busy_claude_reserves_the_lease_and_commits_the_method(env) -> None:
    """No wait loop for a Claude candidate, and no epoch bump or sidecar."""
    _native_claude_target(env)
    _claude_channel_proven(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _busy(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    row = _row()
    status = await server_simple.delivery_status(KEY)

    assert len(dispatched) == 1
    assert env.clock.sleeps == 0, "a busy Claude candidate does not wait"
    assert env.backend.resume_calls == []
    assert result["method"] == ds.METHOD_CLAUDE_MAILBOX
    assert status["method"] == ds.METHOD_CLAUDE_MAILBOX
    assert ds.CARRIER_FIELD not in status, "the carrier is internal"
    assert row["phase"] == ds.PHASE_SENT
    assert row["generation"] == 0
    assert row[ds.METHOD_FIELD] == ds.METHOD_CLAUDE_MAILBOX
    assert row[ds.CARRIER_FIELD] == {
        "backend": "claude-code",
        "backend_session_id": BACKEND_SESSION,
        "transcript_path": str(env.transcript),
        "dispatch_epoch": 3,
        "host_pid": 123,
        "host_create_token": "tok-123",
    }
    assert not (env.session_dir / "dispatch-epochs.json").exists()


@pytest.mark.asyncio
async def test_codex_carrier_is_frozen_with_home_and_rollout(env) -> None:
    case = _eligibility_case(env, "codex")
    _set_agent(env, **case["agent"])
    dispatched = _register_dispatcher(env, ds.METHOD_CODEX_QUEUE)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert len(dispatched) == 1
    assert result["method"] == ds.METHOD_CODEX_QUEUE
    assert _row()[ds.CARRIER_FIELD] == {
        "backend": "codex",
        "backend_session_id": BACKEND_SESSION,
        "codex_home": case["agent"]["codex_home"],
        "rollout_path": str(env.transcript),
        "dispatch_epoch": 3,
        "host_pid": 123,
        "host_create_token": "tok-123",
    }


@pytest.mark.asyncio
async def test_busy_codex_waits_under_e6(env) -> None:
    case = _eligibility_case(env, "codex")
    _set_agent(env, **case["agent"])
    dispatched = _register_dispatcher(env, ds.METHOD_CODEX_QUEUE)
    (env.session_dir / f"state-{AGENT}.json").unlink()
    _busy(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert dispatched == []
    assert env.backend.resume_calls == []
    assert result["phase"] == ds.PHASE_PENDING
    assert result["reason"] == "agent_busy"


@pytest.mark.asyncio
async def test_replace_if_idle_false_does_not_refuse_a_native_candidate(env) -> None:
    """A native carrier never replaces the process, so there is nothing to refuse."""
    _native_claude_target(env)
    _claude_channel_proven(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    result = await server_simple.follow_up_agent(
        AGENT, "next", KEY, replace_if_idle=False
    )

    assert result["reason"] == "stand_in"
    assert len(dispatched) == 1


def _flaky_channel(env: SimpleNamespace, on_loss=None) -> list:
    """Proven at stage 1, lost by stage 2 (and for the rest of the call)."""
    calls: list = []

    def proven(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return {"channel": "available"}
        if len(calls) == 2 and on_loss is not None:
            on_loss()
        return None

    _claude_channel_proven(env, proven)
    return calls


@pytest.mark.asyncio
async def test_eligibility_lost_under_the_lease_falls_back_through_the_idle_gate(
    env,
) -> None:
    _native_claude_target(env)
    _flaky_channel(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert dispatched == []
    assert result["status"] == "delivered"
    assert result["method"] == ds.METHOD_RESUME
    assert len(env.backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_eligibility_lost_with_replace_if_idle_false_meets_the_idle_gate(
    env,
) -> None:
    _native_claude_target(env)
    _flaky_channel(env)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    result = await server_simple.follow_up_agent(
        AGENT, "next", KEY, replace_if_idle=False
    )

    assert dispatched == []
    assert result["reason"] == "agent_idle_but_alive"
    assert env.backend.resume_calls == []


@pytest.mark.asyncio
async def test_eligibility_lost_under_the_lease_gives_up_the_fifo_place(env) -> None:
    """The holder drops its place: a caller that queued behind it goes first."""
    _native_claude_target(env)
    leases_path = server_simple._leases_file(SESSION)

    def other_caller_queues() -> None:
        result = leases.reserve_lease(
            leases_path,
            AGENT,
            generation=0,
            operation_id="op-b",
            backend_session_id=BACKEND_SESSION,
            nonce="nb",
            holder_pid=1,
            holder_create_token=None,
            deadline=time.time() + 60,
            now=time.time(),
            holder_probe=lambda pid, token: "ours",
            ticket="ticket-b",
        )
        assert result.position == 1, "queued behind the native holder"

    _flaky_channel(env, other_caller_queues)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert dispatched == []
    assert env.backend.resume_calls == []
    assert result["phase"] == ds.PHASE_PENDING
    assert result["reason"] == "operation_in_progress"
    assert result["queue_position"] == 1
    data = leases.load_leases(leases_path)[AGENT]
    assert [w["ticket"] for w in data["waiters"]] == [
        "ticket-b",
        server_simple._queue_ticket(_row()),
    ]


@pytest.mark.asyncio
async def test_eligibility_lost_after_the_budget_is_spent_returns_the_tail(
    env,
) -> None:
    _native_claude_target(env)

    def spend_budget() -> None:
        env.clock.now += 60.0

    _flaky_channel(env, spend_budget)
    dispatched = _register_dispatcher(env, ds.METHOD_CLAUDE_MAILBOX)
    _idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert dispatched == []
    assert env.backend.resume_calls == []
    assert result["phase"] == ds.PHASE_PENDING
    assert result["reason"] == "call_budget_expired"
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


def test_public_view_exposes_method_only_when_asked() -> None:
    record = ds.new_record(
        sender=LEAD, idempotency_key=KEY, to=AGENT, fingerprint="f", created_at=1.0
    )
    record[ds.METHOD_FIELD] = ds.METHOD_CODEX_QUEUE
    record[ds.CARRIER_FIELD] = {"backend": "codex"}
    assert set(ds.public_view(record)) == BASELINE_VIEW_KEYS
    shown = ds.public_view(record, include_method=True)
    assert shown["method"] == ds.METHOD_CODEX_QUEUE
    assert ds.CARRIER_FIELD not in shown


def test_unresolved_native_query_excludes_the_callers_own_row() -> None:
    txn = ds.DeliveryTransaction({})
    for key, method, phase in [
        ("a", ds.METHOD_CODEX_QUEUE, ds.PHASE_SENT),
        ("b", ds.METHOD_CLAUDE_MAILBOX, ds.PHASE_UNCONFIRMED),
        ("c", ds.METHOD_RESUME, ds.PHASE_SENT),
    ]:
        record = ds.new_record(
            sender=LEAD, idempotency_key=key, to=AGENT, fingerprint="f", created_at=1.0
        )
        record.update({"phase": phase, ds.METHOD_FIELD: method})
        txn.put(record)
    rows = txn.unresolved_native(AGENT, exclude=ds.record_key(LEAD, "a"))
    assert [row["idempotency_key"] for row in rows] == ["b"]
