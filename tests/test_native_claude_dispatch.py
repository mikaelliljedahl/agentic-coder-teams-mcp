"""B — a live Claude child via the delivery mailbox, lead side (plan v3.1 §2.3.3,
§2.3.5, §2.9 R3-3, §4 tests 7 and 12).

What is pinned here:

- the carrier plugged into ``_NATIVE_DISPATCH``: the child's mailbox is
  initialised BEFORE the attempt is marked ``sent``, then ``publish`` by CAS,
  then confirmation against the frozen carrier;
- recovery of ``sent``/``unconfirmed`` mailbox rows: receipts first, then the
  table (absent ⇒ revoke ⇒ pending; offered ⇒ retract ⇒ pending; provably
  unsent ⇒ pending; taken and later ⇒ ``native_unresolved``), skipped only
  while a live call in this process holds the row;
- budget expiry in the live call (retract; ``done`` ⇒ pending);
- kill and CLI force: the dispatch epoch is bumped first, then offered AND
  taken entries are retracted; ``posting`` and later stay unresolved;
- retention: entries go only once their rows are terminal;
- the real E3 capability check (owner lock, host identity, freshness).

The store and the mailbox are real; the child's poster is simulated on the
lead's poll sleeps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from claude_teams import (
    cli,
    delivery_poster,
    filelock,
    leases,
    native_wake,
    server_simple,
)
from claude_teams import delivery_mailbox as mb
from claude_teams import delivery_store as ds
from claude_teams.backends.process_manager import OWNERSHIP_OURS
from tests import test_native_selection as _sel
from tests.test_native_selection import (
    AGENT,
    BACKEND_SESSION,
    KEY,
    LEAD,
    SESSION,
    _append,
    _eligibility_case,
    _row,
    _set_agent,
    _user_record,
)

env = _sel.env

EPOCH = 3
POSTER = mb.PosterIdentity(4242, "poster-token")
BINDING = mb.HostBinding(EPOCH, BACKEND_SESSION, 123, "tok-123")
_REAL_OWNER_LOCK_HELD = server_simple._delivery_owner_lock_held


def _claude_target(env, *, prompt: str = "hello") -> dict:
    case = _eligibility_case(env, "claude-code", prompt=prompt)
    _set_agent(env, **case["agent"])
    env.monkeypatch.setattr(
        server_simple.process_manager, "kill_process", lambda *a, **k: None
    )
    return case


def _entries(env) -> dict:
    doc = mb.read_mailbox(env.session_dir, AGENT)
    assert doc is not None
    return doc["entries"]


def _entry_state(env, nonce: str) -> str | None:
    return mb.read_entry(env.session_dir, AGENT, nonce).state


def _drive(
    env, nonce: str, upto: str, *, receipt: bool = False, text: str = ""
) -> None:
    """Move an offered entry through the poster's transitions up to ``upto``."""
    sd = env.session_dir
    if upto == mb.STATE_RETRACTED:
        assert mb.retract(sd, AGENT, nonce).done
        return
    if upto == mb.STATE_OFFERED:
        return
    assert mb.take(
        sd,
        AGENT,
        nonce,
        poster=POSTER,
        binding=BINDING,
        current_binding=lambda: BINDING,
    ).done
    if upto == mb.STATE_TAKEN:
        return
    assert mb.begin(
        sd,
        AGENT,
        nonce,
        poster=POSTER,
        binding=BINDING,
        current_binding=lambda: BINDING,
        idle=mb.IdleProof(EPOCH, BACKEND_SESSION, 1),
    ).done
    if upto == mb.STATE_POSTING:
        return
    ok = upto == mb.STATE_POSTED
    started = upto != mb.STATE_FAILED_BEFORE_WRITE
    if receipt:
        _append(env.transcript, _user_record(text))
    assert mb.finish(sd, AGENT, nonce, poster=POSTER, ok=ok, write_started=started).done


def _fake_poster(env, upto: str = mb.STATE_POSTED, *, receipt: bool = True) -> list:
    """Act as the child's poster whenever the lead sleeps between polls."""
    seen: list[str] = []
    sleep = env.clock.sleep

    def poll_sleep(seconds: float) -> None:
        sleep(seconds)
        doc = mb.read_mailbox(env.session_dir, AGENT) or {"entries": {}}
        for nonce, entry in doc["entries"].items():
            if entry["state"] == mb.STATE_OFFERED:
                seen.append(nonce)
                _drive(env, nonce, upto, receipt=receipt, text=entry["text"])

    env.monkeypatch.setattr(server_simple, "_delivery_sleep", poll_sleep)
    return seen


# ==========================================================================
# The carrier: init before sent, publish, confirm
# ==========================================================================


@pytest.mark.asyncio
async def test_idle_claude_child_gets_the_message_in_place(env) -> None:
    _claude_target(env)
    seen = _fake_poster(env)

    result = await server_simple.follow_up_agent(AGENT, "next step", KEY)

    assert result["status"] == ds.STATUS_DELIVERED, result
    assert result["method"] == ds.METHOD_CLAUDE_MAILBOX
    assert result["pid"] == 123, "the same process, never a respawn"
    assert env.backend.resume_calls == []
    row = _row()
    assert seen == [row["nonce"]]
    assert _entries(env) == {}, "a terminal row's entry is cleaned up"
    agent = server_simple._find_agent(server_simple._load_agents(SESSION), AGENT)
    assert agent is not None
    assert agent["pid"] == 123
    assert agent["dispatch_epoch"] == EPOCH, "a native finalisation keeps the epoch"
    assert server_simple.PENDING_DELIVERY_FIELD not in agent


@pytest.mark.asyncio
async def test_mailbox_is_initialised_before_the_attempt_is_marked_sent(env) -> None:
    _claude_target(env)
    _fake_poster(env)
    order: list[str] = []
    real_init = mb.ensure_initialised
    real_mark = server_simple._mark_attempt_sent

    def init(session_dir: Path, child: str) -> mb.MailboxResult:
        order.append("init")
        return real_init(session_dir, child)

    def mark(session_id: str, record: dict, plan) -> dict | None:
        order.append("sent")
        return real_mark(session_id, record, plan)

    env.monkeypatch.setattr(mb, "ensure_initialised", init)
    env.monkeypatch.setattr(server_simple, "_mark_attempt_sent", mark)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    assert order == ["init", "sent"]


@pytest.mark.asyncio
async def test_mailbox_init_failure_never_marks_sent_and_resumes(env) -> None:
    """Without a valid mailbox, absence could not be told from loss: no offer."""
    _claude_target(env)
    env.monkeypatch.setattr(
        mb, "ensure_initialised", lambda *a, **k: mb.MailboxResult(mb.OUTCOME_UNKNOWN)
    )

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    assert result["method"] == ds.METHOD_RESUME
    assert len(env.backend.resume_calls) == 1


@pytest.mark.asyncio
async def test_publish_checks_the_row_under_the_mailbox_lock(env) -> None:
    """The row must still be ``sent`` with THIS operation when published."""
    _claude_target(env)
    _fake_poster(env)
    checks: list[bool] = []
    real_publish = mb.publish

    def publish(*args, row_is_current, **kwargs) -> mb.MailboxResult:
        def spy() -> bool:
            checks.append(row_is_current())
            return checks[-1]

        return real_publish(*args, row_is_current=spy, **kwargs)

    env.monkeypatch.setattr(mb, "publish", publish)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert checks == [True]
    assert result["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_revoke_before_publish_makes_the_live_publish_abort(env) -> None:
    """Revoke wins: the late publish hits the tombstone and returns pending."""
    _claude_target(env)
    seen = _fake_poster(env)
    real_publish = mb.publish

    def publish(session_dir, child, nonce, *, operation_id, **kwargs):
        assert mb.revoke(
            session_dir,
            child,
            nonce,
            operation_id=operation_id,
            row_is_current=lambda: True,
        ).done
        return real_publish(
            session_dir, child, nonce, operation_id=operation_id, **kwargs
        )

    env.monkeypatch.setattr(mb, "publish", publish)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    row = _row()
    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_PENDING
    assert row["phase"] == ds.PHASE_PENDING
    assert _entry_state(env, row["nonce"]) == mb.STATE_REVOKED
    assert seen == [], "nothing was ever offered"
    assert env.backend.resume_calls == [], "a revoked attempt is not resumed here"


@pytest.mark.asyncio
async def test_publish_before_revoke_makes_the_revoke_lose(env) -> None:
    _claude_target(env)
    _fake_poster(env)
    real_publish = mb.publish
    revokes: list[mb.MailboxResult] = []

    def publish(session_dir, child, nonce, *, operation_id, **kwargs):
        result = real_publish(
            session_dir, child, nonce, operation_id=operation_id, **kwargs
        )
        revokes.append(
            mb.revoke(
                session_dir,
                child,
                nonce,
                operation_id=operation_id,
                row_is_current=lambda: True,
            )
        )
        return result

    env.monkeypatch.setattr(mb, "publish", publish)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert [r.state for r in revokes] == [mb.STATE_OFFERED]
    assert revokes[0].lost
    assert result["status"] == ds.STATUS_DELIVERED


@pytest.mark.asyncio
async def test_live_publisher_paused_between_sent_and_publish_is_not_recovered(
    env,
) -> None:
    """§4 test 7: delivery_status mid-call neither pends nor double-publishes."""
    _claude_target(env)
    _fake_poster(env)
    real_publish = mb.publish
    during: list[dict] = []

    def publish(*args, **kwargs):
        during.append(server_simple._delivery_status(SESSION, KEY, ""))
        return real_publish(*args, **kwargs)

    env.monkeypatch.setattr(mb, "publish", publish)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert during[0]["phase"] == ds.PHASE_SENT, "no pending while the call is live"
    assert result["status"] == ds.STATUS_DELIVERED
    assert _entries(env) == {}
    assert len(during) == 1, "published exactly once"


@pytest.mark.asyncio
async def test_first_publication_failing_is_unresolved_then_recovered(env) -> None:
    """R3-3: an unknown publish never pends in the call; recovery settles it."""
    _claude_target(env)
    env.monkeypatch.setattr(
        mb, "publish", lambda *a, **k: mb.MailboxResult(mb.OUTCOME_UNKNOWN)
    )

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    row = _row()
    assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert row["phase"] == ds.PHASE_UNCONFIRMED
    assert env.backend.resume_calls == []

    # The call is over; its claim is released, so recovery may act.
    status = await server_simple.delivery_status(KEY)

    assert status["phase"] == ds.PHASE_PENDING
    assert _entry_state(env, row["nonce"]) == mb.STATE_REVOKED


@pytest.mark.asyncio
async def test_publish_rejected_when_the_row_moved_reports_the_stored_row(
    env,
) -> None:
    """An operator released the row between ``sent`` and publish."""
    _claude_target(env)
    real_publish = mb.publish

    def publish(*args, **kwargs):
        released = server_simple._release_native_row(SESSION, KEY)
        assert released["released"] is True
        return real_publish(*args, **kwargs)

    env.monkeypatch.setattr(mb, "publish", publish)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_FAILED
    assert result["reason"] == server_simple.REASON_OPERATOR_RELEASED
    assert _entries(env) == {}, "nothing was offered"


# ==========================================================================
# Budget expiry in the live call
# ==========================================================================


@pytest.mark.asyncio
async def test_budget_expiry_with_the_offer_untouched_retracts_to_pending(env) -> None:
    _claude_target(env)  # no poster: the child is busy all budget long

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    row = _row()
    assert result["status"] == ds.STATUS_QUEUED
    assert result["phase"] == ds.PHASE_PENDING
    assert row["phase"] == ds.PHASE_PENDING
    assert _entry_state(env, row["nonce"]) == mb.STATE_RETRACTED
    assert env.backend.resume_calls == []
    # Provably unsent, so the next attempt is not held by N5.
    assert not ds.is_unresolved_native(row)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upto",
    [mb.STATE_TAKEN, mb.STATE_POSTING, mb.STATE_POSTED, mb.STATE_UNCERTAIN],
)
async def test_budget_expiry_after_the_poster_took_it_is_unresolved(
    env, upto: str
) -> None:
    _claude_target(env)
    _fake_poster(env, upto, receipt=False)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    blocked = await server_simple.follow_up_agent(AGENT, "other", "k-2")

    row = _row()
    assert result["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert row["phase"] == ds.PHASE_UNCONFIRMED
    assert row["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
    assert _entry_state(env, row["nonce"]) == upto, "never retracted past take"
    assert blocked["reason"] == server_simple.REASON_PRIOR_NATIVE_UNRESOLVED
    agent = server_simple._find_agent(server_simple._load_agents(SESSION), AGENT)
    assert agent is not None
    pending = agent[server_simple.PENDING_DELIVERY_FIELD]
    assert pending["method"] == ds.METHOD_CLAUDE_MAILBOX


@pytest.mark.asyncio
async def test_budget_expiry_after_a_pre_write_failure_is_pending(env) -> None:
    _claude_target(env)
    _fake_poster(env, mb.STATE_FAILED_BEFORE_WRITE, receipt=False)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["phase"] == ds.PHASE_PENDING
    assert _row()["phase"] == ds.PHASE_PENDING


@pytest.mark.asyncio
async def test_retry_after_a_retracted_attempt_offers_a_new_nonce(env) -> None:
    _claude_target(env)
    first = await server_simple.follow_up_agent(AGENT, "next", KEY)
    old_nonce = _row()["nonce"]
    assert first["phase"] == ds.PHASE_PENDING
    seen = _fake_poster(env)

    again = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert again["status"] == ds.STATUS_DELIVERED
    assert seen
    assert seen[0] != old_nonce
    assert _entries(env) == {}, "both the retracted and the posted entry are gone"


# ==========================================================================
# Recovery of rows left by another holder (§2.3.3, R3-3)
# ==========================================================================


def _seed_row(
    env,
    *,
    phase: str = ds.PHASE_SENT,
    holder: dict | None = None,
    nonce: str = "c" * 32,
    key: str = KEY,
    sender: str = LEAD,
) -> dict:
    record = ds.new_record(
        sender=sender,
        idempotency_key=key,
        to=AGENT,
        fingerprint=ds.request_fingerprint(
            to=AGENT, prompt="next", options={"replace_if_idle": True}
        ),
        created_at=500.0,
    )
    record.update(
        {
            "status": ds.STATUS_QUEUED,
            "phase": phase,
            "reason": "",
            "nonce": nonce,
            "operation_id": f"op-{nonce[:4]}",
            "attempts": 1,
            "prompt": "next",
            "options": {"replace_if_idle": True},
            ds.METHOD_FIELD: ds.METHOD_CLAUDE_MAILBOX,
            ds.CARRIER_FIELD: {
                "backend": "claude-code",
                "backend_session_id": BACKEND_SESSION,
                "transcript_path": str(env.transcript),
                "dispatch_epoch": EPOCH,
                "host_pid": 123,
                "host_create_token": "tok-123",
            },
        }
    )
    if holder is not None:
        record[server_simple.ACTIVE_HOLDER_FIELD] = holder
    with ds.delivery_transaction(server_simple._deliveries_file(SESSION)) as txn:
        txn.put(record)
    return record


def _seed_entry(env, record: dict, state: str) -> None:
    sd = env.session_dir
    assert mb.ensure_initialised(sd, AGENT).done
    nonce = record["nonce"]
    if state == mb.STATE_ABSENT:
        return
    if state == mb.STATE_REVOKED:
        assert mb.revoke(
            sd,
            AGENT,
            nonce,
            operation_id=record["operation_id"],
            row_is_current=lambda: True,
        ).done
        return
    assert mb.publish(
        sd,
        AGENT,
        nonce,
        operation_id=record["operation_id"],
        sender=record["sender"],
        key=record["idempotency_key"],
        dispatch_epoch=EPOCH,
        backend_session_id=BACKEND_SESSION,
        text=f"next [nonce {nonce}]",
        row_is_current=lambda: True,
    ).done
    _drive(env, nonce, state)


_OTHER_HOLDER = {"pid": 999_999, "create_token": "gone", "claim_id": "other"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "phase", "after"),
    [
        (mb.STATE_ABSENT, ds.PHASE_PENDING, mb.STATE_REVOKED),
        (mb.STATE_REVOKED, ds.PHASE_PENDING, mb.STATE_REVOKED),
        (mb.STATE_OFFERED, ds.PHASE_PENDING, mb.STATE_RETRACTED),
        (mb.STATE_FAILED_BEFORE_WRITE, ds.PHASE_PENDING, mb.STATE_FAILED_BEFORE_WRITE),
        (mb.STATE_RETRACTED, ds.PHASE_PENDING, mb.STATE_RETRACTED),
        (mb.STATE_TAKEN, ds.PHASE_UNCONFIRMED, mb.STATE_TAKEN),
        (mb.STATE_POSTING, ds.PHASE_UNCONFIRMED, mb.STATE_POSTING),
        (mb.STATE_POSTED, ds.PHASE_UNCONFIRMED, mb.STATE_POSTED),
        (mb.STATE_UNCERTAIN, ds.PHASE_UNCONFIRMED, mb.STATE_UNCERTAIN),
    ],
)
async def test_recovery_table(env, state: str, phase: str, after: str) -> None:
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, state)

    status = await server_simple.delivery_status(KEY)

    assert status["phase"] == phase
    row = _row()
    if phase == ds.PHASE_UNCONFIRMED:
        assert row["reason"] == server_simple.REASON_NATIVE_UNRESOLVED
        assert ds.is_unresolved_native(row)
    else:
        assert not ds.is_unresolved_native(row)
    assert _entry_state(env, record["nonce"]) == after


@pytest.mark.asyncio
async def test_recovered_absent_row_blocks_a_late_publish(env) -> None:
    """§4 test 7: tombstone, then pending; the old holder's publish aborts."""
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, mb.STATE_ABSENT)

    await server_simple.delivery_status(KEY)
    late = mb.publish(
        env.session_dir,
        AGENT,
        record["nonce"],
        operation_id=record["operation_id"],
        sender=LEAD,
        key=KEY,
        dispatch_epoch=EPOCH,
        backend_session_id=BACKEND_SESSION,
        text="late",
        row_is_current=lambda: True,
    )

    assert late.lost
    assert late.state == mb.STATE_REVOKED


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [mb.STATE_POSTED, mb.STATE_TAKEN, mb.STATE_ABSENT])
async def test_receipts_are_inspected_first(env, state: str) -> None:
    """A receipt without any observed running marker settles delivered."""
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, state)
    _append(env.transcript, _user_record(f"next [nonce {record['nonce']}]"))
    _append(
        env.transcript,
        _user_record(server_simple._native_delivered_text("next", record["nonce"])),
    )

    status = await server_simple.delivery_status(KEY)

    assert status["status"] == ds.STATUS_DELIVERED
    assert _entry_state(env, record["nonce"]) == mb.STATE_ABSENT, "cleaned up"


@pytest.mark.asyncio
async def test_unknown_mailbox_keeps_the_row_sent(env) -> None:
    _claude_target(env)
    _seed_row(env, holder=_OTHER_HOLDER)  # never initialised: absence is unknown

    status = await server_simple.delivery_status(KEY)

    assert status["phase"] == ds.PHASE_SENT
    assert not mb.mailbox_path(env.session_dir, AGENT).exists()


@pytest.mark.asyncio
async def test_a_live_claim_in_this_process_skips_recovery(env) -> None:
    _claude_target(env)
    claim = server_simple._claim_holder()
    try:
        record = _seed_row(env, holder=claim)
        _seed_entry(env, record, mb.STATE_ABSENT)

        status = await server_simple.delivery_status(KEY)

        assert status["phase"] == ds.PHASE_SENT
        assert _entry_state(env, record["nonce"]) == mb.STATE_ABSENT, "no tombstone"
    finally:
        server_simple._forget_claim({server_simple.ACTIVE_HOLDER_FIELD: claim})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "holder",
    [
        "completed",  # a finished call in this still-live server
        None,  # removed holder
        {"pid": 999_999},  # holder without a token
        "failed_release",  # the release write failed; the claim id is gone
    ],
)
async def test_recovery_does_not_depend_on_holder_liveness(env, holder) -> None:
    _claude_target(env)
    if holder in ("completed", "failed_release"):
        claim = server_simple._claim_holder()
        server_simple._forget_claim({server_simple.ACTIVE_HOLDER_FIELD: claim})
        holder = claim
    record = _seed_row(env, holder=holder)
    _seed_entry(env, record, mb.STATE_ABSENT)

    status = await server_simple.delivery_status(KEY)

    assert status["phase"] == ds.PHASE_PENDING
    assert _entry_state(env, record["nonce"]) == mb.STATE_REVOKED


@pytest.mark.asyncio
async def test_same_key_retry_recovers_then_sends_a_new_offer(env) -> None:
    """The retrying call's own claim is not a live publisher."""
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, mb.STATE_OFFERED)
    seen = _fake_poster(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["status"] == ds.STATUS_DELIVERED
    assert seen
    assert seen[0] != record["nonce"]


@pytest.mark.asyncio
async def test_posted_then_kill_then_delayed_receipt_is_delivered_without_retry(
    env,
) -> None:
    """§4 test 7: socket write → posted → kill cleanup → delayed receipt."""
    _claude_target(env)
    record = _seed_row(env, phase=ds.PHASE_UNCONFIRMED, holder=_OTHER_HOLDER)
    _seed_entry(env, record, mb.STATE_POSTED)

    killed = await server_simple.kill_agent(AGENT)
    retry = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert killed["native_unresolved"] == [KEY]
    assert retry["success"] is False
    assert env.backend.resume_calls == [], "no retry in between"
    assert _entry_state(env, record["nonce"]) == mb.STATE_POSTED, "retained"

    text = server_simple._native_delivered_text("next", record["nonce"])
    _append(env.transcript, _user_record(text))
    status = await server_simple.delivery_status(KEY)

    assert status["status"] == ds.STATUS_DELIVERED
    assert _entry_state(env, record["nonce"]) == mb.STATE_ABSENT


# ==========================================================================
# Kill and force (§2.3.5)
# ==========================================================================


def _seed_three(env) -> dict[str, dict]:
    rows = {}
    for key, nonce, state in (
        ("k-off", "a" * 32, mb.STATE_OFFERED),
        ("k-take", "b" * 32, mb.STATE_TAKEN),
        ("k-post", "d" * 32, mb.STATE_POSTING),
    ):
        rows[state] = _seed_row(
            env, key=key, nonce=nonce, sender=f"s-{key}", holder=_OTHER_HOLDER
        )
        _seed_entry(env, rows[state], state)
    return rows


def _row_of(record: dict) -> dict:
    return _row(record["idempotency_key"], record["sender"])


@pytest.mark.asyncio
async def test_kill_bumps_the_epoch_then_retracts_offered_and_taken(env) -> None:
    _claude_target(env)
    rows = _seed_three(env)
    epochs_at_retract: list[int] = []
    real_retract = mb.retract

    def retract(session_dir, child, nonce, **kwargs):
        stored = json.loads(
            (env.session_dir / "dispatch-epochs.json").read_text(encoding="utf-8")
        )
        epochs_at_retract.append(stored[AGENT])
        agents = json.loads(
            (env.session_dir / "agents.json").read_text(encoding="utf-8")
        )
        record = next((a for a in agents if a["name"] == AGENT), None)
        assert record is not None
        assert record["dispatch_epoch"] == EPOCH + 1, (
            "the bump is visible to posters before anything is retracted"
        )
        return real_retract(session_dir, child, nonce, **kwargs)

    env.monkeypatch.setattr(mb, "retract", retract)

    killed = await server_simple.kill_agent(AGENT)

    assert killed["success"] is True
    assert killed["native_unresolved"] == ["k-post"]
    assert set(epochs_at_retract) == {EPOCH + 1}
    assert _entry_state(env, rows[mb.STATE_OFFERED]["nonce"]) == mb.STATE_RETRACTED
    assert _entry_state(env, rows[mb.STATE_TAKEN]["nonce"]) == mb.STATE_RETRACTED
    assert _entry_state(env, rows[mb.STATE_POSTING]["nonce"]) == mb.STATE_POSTING
    assert _row_of(rows[mb.STATE_OFFERED])["phase"] == ds.PHASE_PENDING
    assert _row_of(rows[mb.STATE_TAKEN])["phase"] == ds.PHASE_PENDING
    posting = _row_of(rows[mb.STATE_POSTING])
    assert posting["phase"] == ds.PHASE_UNCONFIRMED
    assert posting["reason"] == server_simple.REASON_NATIVE_UNRESOLVED


@pytest.mark.asyncio
async def test_kill_prunes_consumption_of_older_epochs(env) -> None:
    _claude_target(env)
    _seed_three(env)
    before = mb.read_mailbox(env.session_dir, AGENT)
    assert before is not None
    assert before["consumed"]

    await server_simple.kill_agent(AGENT)

    doc = mb.read_mailbox(env.session_dir, AGENT)
    assert doc is not None
    assert doc["consumed"] == {}


@pytest.mark.asyncio
async def test_kill_with_an_unknown_retract_keeps_the_row_unresolved(env) -> None:
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, mb.STATE_OFFERED)
    env.monkeypatch.setattr(
        mb, "retract", lambda *a, **k: mb.MailboxResult(mb.OUTCOME_UNKNOWN)
    )

    killed = await server_simple.kill_agent(AGENT)

    assert killed["native_unresolved"] == [KEY]
    assert _row()["phase"] == ds.PHASE_SENT
    assert _entry_state(env, record["nonce"]) == mb.STATE_OFFERED


@pytest.mark.asyncio
async def test_kill_without_a_persisted_epoch_bump_leaves_taken_unresolved(
    env,
) -> None:
    """Without the fence a poster could still begin: taken is not retracted."""
    _claude_target(env)
    rows = _seed_three(env)

    def refuse(*args, **kwargs):
        raise OSError

    env.monkeypatch.setattr(server_simple, "_next_dispatch_epoch", refuse)

    killed = await server_simple.kill_agent(AGENT)

    assert killed["success"] is True
    assert sorted(killed["native_unresolved"]) == ["k-post", "k-take"]
    assert _entry_state(env, rows[mb.STATE_OFFERED]["nonce"]) == mb.STATE_RETRACTED
    assert _entry_state(env, rows[mb.STATE_TAKEN]["nonce"]) == mb.STATE_TAKEN


def test_epoch_bump_stops_a_poster_at_begin(env) -> None:
    """After force/kill a poster holding a taken entry can never begin it."""
    _claude_target(env)
    record = _seed_row(env, holder=_OTHER_HOLDER)
    _seed_entry(env, record, mb.STATE_OFFERED)
    assert mb.take(
        env.session_dir,
        AGENT,
        record["nonce"],
        poster=POSTER,
        binding=BINDING,
        current_binding=lambda: BINDING,
    ).done
    with server_simple._agents_transaction(SESSION) as agents:
        server_simple._revoke_native_offers(SESSION, AGENT, agents)

    begun = mb.begin(
        env.session_dir,
        AGENT,
        record["nonce"],
        poster=POSTER,
        binding=BINDING,
        current_binding=lambda: server_simple._poster_current_binding(
            SESSION, AGENT, (123, "tok-123")
        ),
        idle=mb.IdleProof(EPOCH, BACKEND_SESSION, 1),
    )

    assert not begun.done
    assert _entry_state(env, record["nonce"]) == mb.STATE_RETRACTED


def test_poster_binding_is_read_from_the_record(env) -> None:
    _claude_target(env)
    binding = server_simple._poster_current_binding(SESSION, AGENT, (123, "tok-123"))
    assert binding == BINDING
    _set_agent(env, dispatch_epoch=EPOCH + 1)
    bumped = server_simple._poster_current_binding(SESSION, AGENT, (123, "tok-123"))
    assert bumped is not None
    assert bumped.dispatch_epoch == EPOCH + 1


def _force(env) -> Result:
    leases.reserve_lease(
        server_simple._leases_file(SESSION),
        AGENT,
        generation=0,
        operation_id="op-hung",
        backend_session_id=BACKEND_SESSION,
        nonce="a" * 32,
        holder_pid=999_999,
        holder_create_token="gone",
        deadline=0.0,
        now=0.0,
        holder_probe=lambda pid, tok: OWNERSHIP_OURS,
    )
    token = server_simple._ensure_lead_token(SESSION)
    return CliRunner().invoke(
        cli.app, ["lease", "force", SESSION, AGENT, "--token", token]
    )


def test_cli_force_bumps_the_epoch_and_retracts(env) -> None:
    _claude_target(env)
    rows = _seed_three(env)

    result = _force(env)

    assert result.exit_code == 0, result.output
    agent = server_simple._find_agent(server_simple._load_agents(SESSION), AGENT)
    assert agent is not None
    assert agent["dispatch_epoch"] == EPOCH + 1
    assert _entry_state(env, rows[mb.STATE_OFFERED]["nonce"]) == mb.STATE_RETRACTED
    assert _entry_state(env, rows[mb.STATE_TAKEN]["nonce"]) == mb.STATE_RETRACTED
    assert _entry_state(env, rows[mb.STATE_POSTING]["nonce"]) == mb.STATE_POSTING


# ==========================================================================
# E3: the real capability check
# ==========================================================================


def _lock_file(env) -> Path:
    return env.session_dir / f"native-delivery-{AGENT}.lock"


def test_owner_lock_is_free_without_a_file(env) -> None:
    assert server_simple._delivery_owner_lock_held(SESSION, AGENT) is False


def test_owner_lock_held_only_while_someone_holds_it(env) -> None:
    path = _lock_file(env)
    path.touch()
    assert server_simple._delivery_owner_lock_held(SESSION, AGENT) is False
    with path.open("a+b") as handle:
        assert filelock.try_lock_handle(handle)
        try:
            assert server_simple._delivery_owner_lock_held(SESSION, AGENT) is True
        finally:
            filelock.unlock_handle(handle)
    assert server_simple._delivery_owner_lock_held(SESSION, AGENT) is False


def test_capability_accepts_the_resolved_host_of_a_launcher(env) -> None:
    case = _claude_target(env)
    env.monkeypatch.setattr(
        server_simple.process_manager,
        "resolve_agent_pid",
        lambda handle, team, name: "456",
    )
    env.monkeypatch.setattr(
        server_simple.process_manager,
        "creation_token",
        lambda pid: "tok-456" if str(pid) == "456" else None,
    )
    _sel._marker(env, host_pid=456, host_create_token="tok-456")
    assert server_simple._native_candidate(SESSION, AGENT, **case) == (
        ds.METHOD_CLAUDE_MAILBOX,
        "",
    )
    _sel._marker(env, host_pid=456, host_create_token="tok-other")
    assert server_simple._native_candidate(SESSION, AGENT, **case)[0] is None


def test_real_poster_capability_satisfies_e3_and_goes_stale(env) -> None:
    """A capability written by the poster, under its held lock, is E3."""
    case = _claude_target(env)
    env.monkeypatch.setattr(
        server_simple, "_delivery_owner_lock_held", _REAL_OWNER_LOCK_HELD
    )
    env.monkeypatch.setattr(delivery_poster.time, "time", lambda: 1_000.0)
    marker = {
        "state": "waiting",
        "idle_seq": 1,
        "backend_session_id": BACKEND_SESSION,
        "dispatch_epoch": EPOCH,
    }
    poster = delivery_poster.DeliveryPoster(
        delivery_poster.PosterFacts(
            host=lambda: (123, "tok-123"),
            current_binding=lambda s, n, h: BINDING,
            read_marker=lambda s, n: marker,
            identity=mb.PosterIdentity(os.getpid(), "me"),
            epoch=lambda: EPOCH,
        ),
        get_target=lambda: (SESSION, AGENT),
        session_dir=server_simple._session_dir,
        channel=native_wake.ClaudeChannel("available", "/x", "t", True, 123),
        post=lambda channel, text: native_wake.PostResult(True, "", True),
    )
    try:
        poster.tick()
        assert _candidate(case) == (ds.METHOD_CLAUDE_MAILBOX, "")
        _sel._marker(env, heartbeat_ts=1_000.0 - 3.5)  # older than 3 x poll
        assert _candidate(case)[1].startswith("E3")
        poster.tick()  # the heartbeat is refreshed
        assert _candidate(case) == (ds.METHOD_CLAUDE_MAILBOX, "")
    finally:
        poster.close()
    assert _candidate(case)[1].startswith("E3"), "the owner lock is free again"


def _candidate(case: dict) -> tuple:
    return server_simple._native_candidate(SESSION, AGENT, **case)


# ==========================================================================
# Flag-off: byte-identical, no mailbox
# ==========================================================================


@pytest.mark.asyncio
async def test_flag_off_claude_follow_up_touches_no_mailbox(env) -> None:
    _sel._idle(env)
    env.monkeypatch.setattr(
        server_simple.process_manager, "kill_process", lambda *a, **k: None
    )

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)
    killed = await server_simple.kill_agent(AGENT)

    assert result["status"] == ds.STATUS_DELIVERED
    assert "method" not in result
    assert killed == {"success": True, "name": AGENT}
    names = sorted(p.name for p in env.session_dir.iterdir())
    assert not [n for n in names if n.startswith("delivery-mailbox-")]
    assert "dispatch-epochs.json" not in names


@pytest.mark.asyncio
async def test_downstream_off_claude_follow_up_touches_no_mailbox(env) -> None:
    _sel._flags(env.monkeypatch, NATIVE_WAKE="1")
    _set_agent(env, interactive=True, dispatch_epoch=EPOCH)
    _sel._idle(env)

    result = await server_simple.follow_up_agent(AGENT, "next", KEY)

    assert result["method"] == ds.METHOD_RESUME
    assert not mb.mailbox_path(env.session_dir, AGENT).exists()
