"""A: a live Codex child via ``codex queue`` (plan v3.1 §2.2, §2.9 R3-5, §4 tests 4-5).

What is pinned here:

- the carrier plugged into ``_NATIVE_DISPATCH``: durable ``sent`` first, then
  the queue run, then the ``carrier_ref`` CAS on the attempt's identity;
- every row of the §2.2.3 outcome table, driven through the real
  ``native_wake.codex_queue`` runner with a fake ``Popen``;
- durable settlement: the frozen carrier is scanned, never a same-name
  successor, and absence is never terminal (``delivery_status``, a same-key
  retry and ``kill_agent`` alike);
- the operator escape ``deliveries release-native``;
- the R3-5 command budget, measured in UTF-16 units on Windows.

Receipts are real Codex rollout records and the store is the real store.
"""

import dataclasses
import functools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from claude_teams import cli, leases, native_wake, server_simple
from claude_teams import delivery_store as ds
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.delivery import DELIVERY_MARKER_PREFIX
from tests import test_native_selection as _sel
from tests.test_native_selection import (
    AGENT,
    BACKEND_SESSION,
    KEY,
    LEAD,
    SESSION,
    _append,
    _eligibility_case,
    _FakeRegistry,
    _row,
    _set_agent,
)

env = _sel.env

SUBMISSION = "01a0ddd0-2200-7373-a65f-8b58834c87bb"
BINARY = "C:/codex/codex.exe"
_REAL_QUEUE = native_wake.codex_queue


def _codex_user(text: str) -> dict:
    """The rollout record S-1 showed a queued turn produces."""
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _nonce_of(text: str) -> str:
    _, _, tail = text.partition(DELIVERY_MARKER_PREFIX)
    return tail.split()[0].strip("]")


class _Proc:
    def __init__(self, queue: "_Queue", argv: list[str]) -> None:
        self.queue = queue
        self.argv = argv
        self.returncode: int | None = None
        self.communicated = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicated += 1
        if self.communicated > 1:  # the reaper collecting a killed process
            return "", ""
        queue = self.queue
        if queue.on_communicate is not None:
            queue.on_communicate(self.argv)
        if queue.receipt:
            _append(queue.rollout, _codex_user(self.argv[-1]))
        if queue.mode == "timeout":
            raise subprocess.TimeoutExpired(self.argv, timeout or 0)
        if queue.mode == "oserror":
            raise OSError
        self.returncode = 1 if queue.mode == "exit1" else 0
        if queue.mode == "no_id":
            return "done\n", ""
        if queue.mode == "exit1":
            return "", "boom"
        return f"Queued message {SUBMISSION} for thread {BACKEND_SESSION}.\n", ""

    def kill(self) -> None:
        self.queue.killed += 1


class _Queue:
    """A fake ``Popen`` for the real runner; the codex child is simulated."""

    def __init__(self, rollout: Path, mode: str = "enqueue", *, receipt: bool = True):
        self.rollout = rollout
        self.mode = mode
        self.receipt = receipt
        self.on_communicate = None
        self.calls: list[tuple[list[str], dict]] = []
        self.killed = 0

    def popen(self, argv: list[str], **kwargs: object) -> _Proc:
        if self.mode == "ctor":
            raise FileNotFoundError("codex.exe")
        self.calls.append((list(argv), kwargs))
        return _Proc(self, list(argv))


class _CodexResumeBackend:
    """A resume that lands in the same rollout, Codex-shaped."""

    def __init__(self, rollout: Path, binary: str = BINARY) -> None:
        self.rollout = rollout
        self.binary = binary
        self.is_interactive = True
        self.resume_calls: list[tuple[SpawnRequest, str]] = []

    def supports_resume(self) -> bool:
        return True

    def default_model(self) -> str:
        return "model"

    def build_resume_command(
        self, request: SpawnRequest, backend_session_id: str
    ) -> list[str]:
        return [self.binary, "resume", backend_session_id, request.prompt]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"AGENT_NAME": request.name}

    def resume(self, request: SpawnRequest, backend_session_id: str) -> SimpleNamespace:
        self.resume_calls.append((request, backend_session_id))
        _append(self.rollout, _codex_user(request.prompt))
        return SimpleNamespace(process_handle="789")


def _codex_target(env, *, prompt: str = "hello") -> dict:
    case = _eligibility_case(env, "codex", prompt=prompt)
    _set_agent(env, **case["agent"])
    backend = _CodexResumeBackend(env.transcript)
    env.monkeypatch.setattr(server_simple, "registry", _FakeRegistry(backend))
    env.codex = backend
    _no_kill(env)
    return case


def _no_kill(env) -> None:
    """``owns_process`` is faked true: never signal a real PID from a test."""
    env.monkeypatch.setattr(
        server_simple.process_manager, "kill_process", lambda *a, **k: None
    )


def _install(env, queue: _Queue) -> _Queue:
    env.monkeypatch.setattr(
        native_wake, "codex_queue", functools.partial(_REAL_QUEUE, popen=queue.popen)
    )
    return queue


def _agent() -> dict:
    agent = server_simple._find_agent(server_simple._load_agents(SESSION), AGENT)
    assert agent is not None
    return agent


def _put_row(row: dict) -> None:
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        txn.put(row)


# ==========================================================================
# §4 test 4 — A: in place, same PID, receipt
# ==========================================================================


@pytest.mark.asyncio
async def test_idle_codex_child_gets_the_message_in_place(env) -> None:
    case = _codex_target(env)
    queue = _install(env, _Queue(env.transcript))
    before = _agent()

    result = await server_simple.follow_up_agent(AGENT, "next step", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    assert result["method"] == ds.METHOD_CODEX_QUEUE
    assert result["pid"] == 123, "the same process, never a respawn"
    assert env.codex.resume_calls == []
    ((argv, kwargs),) = queue.calls
    assert argv[:5] == [BINARY, "queue", "--thread", BACKEND_SESSION, "--message"]
    assert argv[5].startswith("next step\n\n")
    assert _nonce_of(argv[5]) == _row()["nonce"]
    assert kwargs["env"]["CODEX_HOME"] == case["agent"]["codex_home"]
    row = _row()
    assert row["status"] == ds.STATUS_DELIVERED
    assert row["carrier_ref"] == SUBMISSION
    after = _agent()
    for field in ("pid", "create_token", "spawned_at", "dispatch_epoch"):
        assert after[field] == before[field], field
    assert after.get("prompt_transport") == before.get("prompt_transport")
    assert after["generation"] == before.get("generation", 0) + 1
    assert server_simple.PENDING_DELIVERY_FIELD not in after
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


@pytest.mark.asyncio
async def test_enqueued_without_a_receipt_is_native_unresolved(env) -> None:
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript, receipt=False))

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_UNCONFIRMED
    assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert result["retriable"] is True
    assert result["pid"] == 123
    row = _row()
    assert row["phase"] == ds.PHASE_UNCONFIRMED
    assert row["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert row["carrier_ref"] == SUBMISSION
    pending = _agent()[server_simple.PENDING_DELIVERY_FIELD]
    assert pending["nonce"] == row["nonce"]
    assert pending["operation_id"] == row["operation_id"]
    assert pending["method"] == ds.METHOD_CODEX_QUEUE
    assert pending["carrier_ref"] == SUBMISSION
    assert env.codex.resume_calls == []

    # The same key reconciles and never sends again; another key meets N5.
    again = await server_simple.follow_up_agent(AGENT, "next", KEY)
    other = await server_simple.follow_up_agent(AGENT, "other", "k-2")
    assert again["reason"] == server_simple.REASON_ATTEMPT_UNRESOLVED
    assert other["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    assert len(queue.calls) == 1
    assert env.codex.resume_calls == []

    # The receipt lands later, in the frozen carrier: delivered.
    _append(env.transcript, _codex_user(queue.calls[0][0][-1]))
    status = await server_simple.delivery_status(KEY)
    assert status["status"] == ds.STATUS_DELIVERED


# ==========================================================================
# §2.2.3 — the outcome table
# ==========================================================================


@pytest.mark.parametrize("mode", ["no_id", "exit1", "timeout", "oserror"])
@pytest.mark.asyncio
async def test_any_uncertain_queue_outcome_is_native_unresolved(env, mode) -> None:
    """Exit 0 without an id, non-zero, timeout, an error after spawn."""
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript, mode, receipt=False))

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["phase"] == ds.PHASE_UNCONFIRMED
    assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert len(queue.calls) == 1
    assert env.codex.resume_calls == [], "never a second carrier"
    row = _row()
    assert ds.is_unresolved_native(row)
    assert "carrier_ref" not in row
    assert _agent()[server_simple.PENDING_DELIVERY_FIELD]["carrier_ref"] == ""


@pytest.mark.asyncio
async def test_uncertain_outcome_whose_receipt_lands_is_delivered(env) -> None:
    _codex_target(env)
    _install(env, _Queue(env.transcript, "exit1", receipt=True))

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    assert env.codex.resume_calls == []


def _fails_at_dispatch(env, target: str, bad: object) -> None:
    """Healthy at stages 1 and 2, broken by the time the carrier runs."""
    calls = {"n": 0}
    good = getattr(target_module(target), target)

    def flaky(*args: object) -> object:
        calls["n"] += 1
        return bad if calls["n"] >= 3 else good(*args)

    env.monkeypatch.setattr(target_module(target), target, flaky)


def target_module(target: str) -> object:
    return native_wake if target == "verify_codex_thread" else server_simple


@pytest.mark.parametrize(
    ("breaks", "reason"),
    [
        ("exec", "native_not_enqueued"),
        ("discovery", "native_ineligible_this_call"),
        ("shim", "native_ineligible_this_call"),
        ("thread", "native_ineligible_this_call"),
    ],
)
@pytest.mark.asyncio
async def test_provably_not_enqueued_falls_back_to_resume(env, breaks, reason) -> None:
    """Pre-spawn failures return the row to pending, then resume, once."""
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript, "ctor" if breaks == "exec" else ""))
    if breaks == "discovery":
        _fails_at_dispatch(env, "_codex_queue_binary", "")
    elif breaks == "shim":
        _fails_at_dispatch(env, "_codex_queue_binary", "C:/npm/codex.cmd")
    elif breaks == "thread":
        _fails_at_dispatch(env, "verify_codex_thread", (False, "archived"))
    reverted: list = []
    real_revert = server_simple._revert_native_attempt

    def spy(session_id, record, plan, why):
        reverted.append(why)
        return real_revert(session_id, record, plan, why)

    env.monkeypatch.setattr(server_simple, "_revert_native_attempt", spy)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert reverted == [reason]
    assert queue.calls == []
    assert len(env.codex.resume_calls) == 1
    assert result["status"] == ds.STATUS_DELIVERED
    assert result["method"] == ds.METHOD_RESUME
    row = _row()
    assert row["attempts"] == 2
    assert row[ds.METHOD_FIELD] == ds.METHOD_RESUME
    assert ds.CARRIER_FIELD not in row


@pytest.mark.asyncio
async def test_fallback_after_the_budget_is_spent_returns_the_tail(env) -> None:
    _codex_target(env)
    _install(env, _Queue(env.transcript, "ctor"))
    real = native_wake.codex_queue

    def spend(*args, **kwargs):
        env.clock.now += 60.0
        return real(*args, **kwargs)

    env.monkeypatch.setattr(native_wake, "codex_queue", spend)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["phase"] == ds.PHASE_PENDING
    assert result["reason"] == "call_budget_expired"
    assert env.codex.resume_calls == []
    assert not ds.is_unresolved_native(_row())
    assert leases.active_lease(server_simple._leases_file(SESSION), AGENT) is None


# ==========================================================================
# §2.2.2 — the carrier_ref CAS
# ==========================================================================


@pytest.mark.asyncio
async def test_receipt_settled_before_the_ref_is_never_reverted(env) -> None:
    """A concurrent reconcile settles the row first; the CAS leaves it alone."""
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript))

    def settle_first(argv: list[str]) -> None:
        _append(env.transcript, _codex_user(argv[-1]))
        row = _row()
        ds.settle(row, ds.STATUS_DELIVERED, reason="", now=1_000.0)
        _put_row(row)

    queue.receipt = False
    queue.on_communicate = settle_first

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    row = _row()
    assert row["status"] == ds.STATUS_DELIVERED
    assert "carrier_ref" not in row


@pytest.mark.parametrize("receipt", [True, False])
@pytest.mark.asyncio
async def test_lost_ref_write_leaves_the_row_unresolved_not_failed(
    env, receipt: bool
) -> None:
    _codex_target(env)
    _install(env, _Queue(env.transcript, receipt=receipt))

    def lost(*args, **kwargs):
        raise ds.DeliveryStoreError

    env.monkeypatch.setattr(server_simple, "_attach_carrier_ref", lost)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    row = _row()
    assert "carrier_ref" not in row
    assert env.codex.resume_calls == []
    if receipt:
        assert result["status"] == row["status"] == ds.STATUS_DELIVERED
    else:
        assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
        assert ds.is_unresolved_native(row)


@pytest.mark.asyncio
async def test_operator_release_racing_the_ref_write_wins(env) -> None:
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript))
    queue.on_communicate = lambda argv: server_simple._release_native_row(SESSION, KEY)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    row = _row()
    assert row["status"] == ds.STATUS_FAILED
    assert row["reason"] == server_simple.REASON_OPERATOR_RELEASED
    assert "carrier_ref" not in row
    assert result["status"] == ds.STATUS_FAILED, "the stored status is reported"


def test_ref_cas_never_attaches_to_another_attempt(env) -> None:
    row = ds.new_record(
        sender=LEAD, idempotency_key=KEY, to=AGENT, fingerprint="f", created_at=1.0
    )
    row.update(nonce="n-new", operation_id="op-new", phase=ds.PHASE_SENT)
    _put_row(row)
    stale = SimpleNamespace(nonce="n-old", operation_id="op-old")

    assert not server_simple._attach_carrier_ref(SESSION, dict(row), stale, "ref")
    assert "carrier_ref" not in _row()
    current = SimpleNamespace(nonce="n-new", operation_id="op-new")
    assert server_simple._attach_carrier_ref(SESSION, dict(row), current, "ref")
    assert _row()["carrier_ref"] == "ref"


# ==========================================================================
# §2.2.4 / §4 test 5 — durable settlement against the frozen carrier
# ==========================================================================


async def _unresolved(env) -> _Queue:
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript, receipt=False))
    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    return queue


def _dead_and_late(env) -> None:
    env.monkeypatch.setattr(
        server_simple.process_manager,
        "health_check",
        lambda handle, expected_token=None: (False, "gone"),
    )
    env.monkeypatch.setattr(server_simple.time, "time", lambda: 5_000.0)


@pytest.mark.asyncio
async def test_absence_from_a_dead_child_is_never_terminal(env) -> None:
    await _unresolved(env)
    _dead_and_late(env)

    status = await server_simple.delivery_status(KEY)
    retry = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert status["status"] == ds.STATUS_QUEUED
    assert retry["reason"] == server_simple.REASON_ATTEMPT_UNRESOLVED
    assert ds.is_unresolved_native(_row())
    assert env.codex.resume_calls == []


@pytest.mark.asyncio
async def test_kill_reports_and_keeps_native_rows_unresolved(env) -> None:
    await _unresolved(env)
    _dead_and_late(env)

    result = await server_simple.kill_agent(AGENT)

    assert result == {"success": True, "name": AGENT, "native_unresolved": [KEY]}
    row = _row()
    assert row["status"] == ds.STATUS_QUEUED
    assert ds.is_unresolved_native(row)


@pytest.mark.asyncio
async def test_kill_settles_a_native_row_whose_receipt_is_in_the_carrier(env) -> None:
    queue = await _unresolved(env)
    _append(env.transcript, _codex_user(queue.calls[0][0][-1]))

    result = await server_simple.kill_agent(AGENT)

    assert result["native_unresolved"] == []
    assert _row()["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_kill_output_is_unchanged_with_flags_off_and_no_native_rows(env) -> None:
    _no_kill(env)
    result = await server_simple.kill_agent(AGENT)
    assert result == {"success": True, "name": AGENT}


@pytest.mark.asyncio
async def test_frozen_carrier_settles_after_a_same_name_successor(env) -> None:
    """Kill, respawn under the same name: only the OLD thread settles the row."""
    queue = await _unresolved(env)
    nonce = _nonce_of(queue.calls[0][0][-1])
    await server_simple.kill_agent(AGENT)

    successor = env.tmp_path / "successor.jsonl"
    _append(successor, _codex_user(f"echo {DELIVERY_MARKER_PREFIX}{nonce}"))
    real_binding = server_simple._resolve_agent_binding

    def binding(agent: dict):
        found = real_binding(agent)
        if agent.get("pid") == 456 and found.output is not None:
            output = dataclasses.replace(found.output, rollout_path=str(successor))
            return dataclasses.replace(found, output=output)
        return found

    env.monkeypatch.setattr(server_simple, "_resolve_agent_binding", binding)
    agents = server_simple._load_agents(SESSION)
    agents.append({**_sel._agent_record(env.work, backend="codex"), "pid": 456})
    server_simple._save_agents(SESSION, agents)

    status = await server_simple.delivery_status(KEY)
    blocked = await server_simple.follow_up_agent(AGENT, "other", "k-2")
    assert status["status"] == ds.STATUS_QUEUED, "never the successor's transcript"
    assert blocked["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED

    _append(env.transcript, _codex_user(queue.calls[0][0][-1]))
    assert (await server_simple.delivery_status(KEY))["status"] == (ds.STATUS_DELIVERED)


# ==========================================================================
# Operator release
# ==========================================================================

runner = CliRunner()


def _release(*extra: str):
    token = server_simple._ensure_lead_token(SESSION)
    return runner.invoke(
        cli.app,
        ["deliveries", "release-native", SESSION, *extra, "--token", token],
    )


@pytest.mark.asyncio
async def test_release_native_settles_operator_released_and_lifts_n5(env) -> None:
    await _unresolved(env)

    result = _release(KEY)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["released"] is True
    assert payload["idempotency_key"] == KEY
    assert payload["carrier_ref"] == SUBMISSION
    assert "may still execute" in payload["warning"]
    row = _row()
    assert row["status"] == ds.STATUS_FAILED
    assert row["reason"] == server_simple.REASON_OPERATOR_RELEASED
    assert server_simple.PENDING_DELIVERY_FIELD not in _agent()
    assert not ds.DeliveryTransaction(
        {ds.record_key(LEAD, KEY): row}
    ).unresolved_native(AGENT)


def test_release_native_refuses_a_wrong_token(env) -> None:
    result = runner.invoke(
        cli.app,
        ["deliveries", "release-native", SESSION, KEY, "--token", "wrong"],
    )
    assert result.exit_code == 2


def test_release_native_refuses_an_unknown_key(env) -> None:
    assert _release("missing").exit_code == 1


@pytest.mark.asyncio
async def test_release_native_refuses_a_row_that_is_not_unresolved_native(
    env,
) -> None:
    _codex_target(env)
    _install(env, _Queue(env.transcript))
    await server_simple.follow_up_agent(AGENT, "next", KEY)

    result = _release(KEY)

    assert result.exit_code == 3
    assert _row()["status"] == ds.STATUS_DELIVERED


def test_release_native_needs_a_sender_when_the_key_is_ambiguous(env) -> None:
    for sender in (LEAD, "other-lead"):
        _sel._seed_native_row(key=KEY, sender=sender)

    assert _release(KEY).exit_code == 4
    assert _release(KEY, "--sender", "other-lead").exit_code == 0
    assert _row(KEY, "other-lead")["status"] == ds.STATUS_FAILED
    assert _row(KEY, LEAD)["status"] == ds.STATUS_QUEUED


# ==========================================================================
# R3-5 — the command budget
# ==========================================================================


def test_windows_budget_counts_utf16_units_after_quoting() -> None:
    fits = native_wake.WINDOWS_COMMAND_BUDGET - 1  # the terminating null
    assert native_wake.command_fits(["a" * fits], windows=True)
    assert not native_wake.command_fits(["a" * (fits + 1)], windows=True)
    # One non-BMP character is two UTF-16 units though one code point.
    assert not native_wake.command_fits(["a" * (fits - 1) + "\U0001f600"], windows=True)
    # A quote is escaped to two units, and a space forces surrounding quotes.
    quotes = '"' * (fits // 2)
    assert native_wake.command_fits([quotes], windows=True)
    assert not native_wake.command_fits([quotes + " "], windows=True)


def test_posix_budget_checks_each_argument_and_the_whole_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_arg = native_wake.POSIX_ARG_STRLEN_MAX
    roomy = {"windows": False, "arg_max": 4 * per_arg}
    assert native_wake.command_fits(["a" * (per_arg - 1)], {}, **roomy)
    assert not native_wake.command_fits(["a" * per_arg], {}, **roomy)
    # Multibyte text is counted in encoded bytes.
    assert not native_wake.command_fits(["é" * (per_arg // 2)], {}, **roomy)
    # Without sysconf the whole block is held to 128 KiB less the margin.
    monkeypatch.delattr(native_wake.os, "sysconf", raising=False)
    assert native_wake.command_fits(["a" * (per_arg - 8192)], {}, windows=False)
    assert not native_wake.command_fits(["a" * (per_arg - 2048)], {}, windows=False)
    env = {"K": "v" * 1000}
    assert native_wake.command_fits(["x"], env, windows=False, arg_max=8192)
    assert not native_wake.command_fits(["x" * 3500], env, windows=False, arg_max=8192)


@pytest.mark.asyncio
async def test_queue_over_the_command_budget_forces_resume(env) -> None:
    """Native pre-launch rejection, and resume's own argv still fits."""
    _codex_target(env)
    queue = _install(env, _Queue(env.transcript))
    env.monkeypatch.setattr(native_wake, "command_fits", _fits_unless("queue"))

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert queue.calls == []
    assert len(env.codex.resume_calls) == 1
    assert result["status"] == ds.STATUS_DELIVERED
    assert result["method"] == ds.METHOD_RESUME


def _fits_unless(word: str):
    real = native_wake.command_fits

    def check(argv, env=None, **kwargs):
        return word not in argv and real(argv, env, **kwargs)

    return check


@pytest.mark.asyncio
async def test_message_fitting_neither_path_fails_message_too_large(env) -> None:
    """Quote-heavy text inside 16 KiB of UTF-8 that no command line can carry."""
    marker = len(server_simple._native_delivered_text("", "0" * 32).encode())
    prompt = '"' * (server_simple.NATIVE_INLINE_MAX - marker)
    _codex_target(env, prompt=prompt)
    queue = _install(env, _Queue(env.transcript))
    env.monkeypatch.setattr(native_wake, "_WINDOWS", True)
    shutdowns: list = []
    env.monkeypatch.setattr(
        server_simple.process_manager,
        "graceful_shutdown",
        lambda *a, **k: shutdowns.append(a) or True,
    )

    result = await server_simple.follow_up_agent(AGENT, prompt, KEY)

    assert queue.calls == []
    assert env.codex.resume_calls == []
    assert shutdowns == [], "the live child is never touched"
    assert result["status"] == ds.STATUS_FAILED
    assert result["reason"] == server_simple.REASON_MESSAGE_TOO_LARGE
    assert result["retriable"] is False
    assert _row()["reason"] == server_simple.REASON_MESSAGE_TOO_LARGE
    assert _agent()["pid"] == 123


@pytest.mark.asyncio
async def test_flag_off_oversize_codex_resume_is_launched_as_on_main(env) -> None:
    """``message_too_large`` is a flag-on improvement only."""
    _codex_target(env)
    _sel._flags(env.monkeypatch)
    _sel._idle(env)
    env.monkeypatch.setattr(native_wake, "command_fits", lambda *a, **k: False)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert len(env.codex.resume_calls) == 1
    assert result["status"] == ds.STATUS_DELIVERED
    assert "method" not in result
