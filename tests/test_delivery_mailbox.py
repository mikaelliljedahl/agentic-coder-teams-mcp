"""B — the delivery mailbox store: CAS transitions, tombstones, fail-closed reads.

Plan §2.3.2 as amended by §2.9 R3-2/R3-3 (test 6). The mailbox is the only
channel between a lead that publishes a Claude-bound delivery and the child's
poster that presents it, so every property here is asserted directly against
the store rather than through the server:

- **Every transition is one locked CAS with an authoritative result.** A
  caller never infers the outcome from a preliminary read.
- **Fail closed.** A missing, unreadable, non-object or malformed document is
  *unknown*, never empty, and nothing is written over it.
- **Tombstones win.** Whichever of ``publish`` / ``revoke`` runs first under
  the lock decides; the loser sees the winner's state.
- **Retention.** Entries in every state stay until their row is terminal;
  deleting one is never an acknowledgement.
"""

import json
import threading
from pathlib import Path

import pytest

from claude_teams import delivery_mailbox as mb
from claude_teams import filelock

CHILD = "worker"
NONCE = "n-1"
OP = "op-1"
EPOCH = 3
SESSION = "backend-sess-a"
POSTER = mb.PosterIdentity(pid=4242, create_token="tok-poster")
OTHER_POSTER = mb.PosterIdentity(pid=4343, create_token="tok-other")
BINDING = mb.HostBinding(
    dispatch_epoch=EPOCH,
    backend_session_id=SESSION,
    host_pid=1000,
    host_create_token="tok-host",
)
IDLE = mb.IdleProof(dispatch_epoch=EPOCH, backend_session_id=SESSION, idle_seq=5)


def _yes() -> bool:
    return True


def _no() -> bool:
    return False


def _current() -> mb.HostBinding:
    return BINDING


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """An initialised, empty mailbox for ``CHILD``."""
    assert mb.ensure_initialised(tmp_path, CHILD).done
    return tmp_path


def _publish(
    session: Path, nonce: str = NONCE, **overrides: object
) -> mb.MailboxResult:
    kwargs: dict = {
        "operation_id": OP,
        "sender": "team-lead",
        "key": "k1",
        "dispatch_epoch": EPOCH,
        "backend_session_id": SESSION,
        "text": "hello",
        "row_is_current": _yes,
        "now": 10.0,
    }
    kwargs.update(overrides)
    return mb.publish(session, CHILD, nonce, **kwargs)


def _take(session: Path, nonce: str = NONCE, **overrides: object) -> mb.MailboxResult:
    kwargs: dict = {
        "poster": POSTER,
        "binding": BINDING,
        "current_binding": _current,
        "now": 11.0,
    }
    kwargs.update(overrides)
    return mb.take(session, CHILD, nonce, **kwargs)


def _begin(session: Path, nonce: str = NONCE, **overrides: object) -> mb.MailboxResult:
    kwargs: dict = {
        "poster": POSTER,
        "binding": BINDING,
        "current_binding": _current,
        "idle": IDLE,
        "now": 12.0,
    }
    kwargs.update(overrides)
    return mb.begin(session, CHILD, nonce, **kwargs)


def _state(session: Path, nonce: str = NONCE) -> str | None:
    return mb.read_entry(session, CHILD, nonce).state


def _raw(session: Path) -> dict:
    return json.loads(mb.mailbox_path(session, CHILD).read_text(encoding="utf-8"))


def _write_raw(session: Path, payload: object) -> None:
    mb.mailbox_path(session, CHILD).write_text(json.dumps(payload), encoding="utf-8")


# ==========================================================================
# Paths, initialisation, fail-closed loading
# ==========================================================================


def test_paths_are_per_child_beside_the_session(tmp_path: Path) -> None:
    assert mb.mailbox_path(tmp_path, CHILD) == (
        tmp_path / "delivery-mailbox-worker.json"
    )
    assert mb.mailbox_lock_path(tmp_path, CHILD) == (
        tmp_path / "delivery-mailbox-worker.lock"
    )


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a\\b", "has space", "x" * 65])
def test_unsafe_child_names_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unsafe agent name"):
        mb.mailbox_path(tmp_path, name)


def test_ensure_initialised_creates_an_empty_valid_document(tmp_path: Path) -> None:
    result = mb.ensure_initialised(tmp_path, CHILD)
    assert result.done
    snap = mb.read_mailbox(tmp_path, CHILD)
    assert snap is not None
    assert snap["entries"] == {}
    assert snap["tombstones"] == {}
    assert snap["consumed"] == {}


def test_ensure_initialised_is_idempotent_and_keeps_entries(session: Path) -> None:
    assert _publish(session).done
    assert mb.ensure_initialised(session, CHILD).done
    assert _state(session) == mb.STATE_OFFERED


def test_ensure_initialised_never_overwrites_a_corrupt_document(tmp_path: Path) -> None:
    path = mb.mailbox_path(tmp_path, CHILD)
    path.write_text("{not json", encoding="utf-8")
    assert mb.ensure_initialised(tmp_path, CHILD).unknown
    assert path.read_text(encoding="utf-8") == "{not json"


def test_missing_mailbox_is_unknown_not_empty(tmp_path: Path) -> None:
    """R3-3: once initialised before ``sent``, absence can only mean unknown."""
    assert mb.read_mailbox(tmp_path, CHILD) is None
    assert mb.read_entry(tmp_path, CHILD, NONCE).unknown
    assert _publish(tmp_path).unknown
    assert mb.revoke(
        tmp_path, CHILD, NONCE, operation_id=OP, row_is_current=_yes
    ).unknown
    assert mb.retract(tmp_path, CHILD, NONCE).unknown
    # Nothing was created as a side effect: absence stays unknown.
    assert not mb.mailbox_path(tmp_path, CHILD).exists()


_ENTRY = {
    "operation_id": OP,
    "sender": "team-lead",
    "key": "k1",
    "dispatch_epoch": EPOCH,
    "backend_session_id": SESSION,
    "text": "hello",
    "state": "offered",
    "ts": 1.0,
}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "string",
        42,
        None,
        {},
        {"version": 1, "entries": {}, "tombstones": {}},
        {"version": 2, "entries": {}, "tombstones": {}, "consumed": {}},
        {"version": 1, "entries": [], "tombstones": {}, "consumed": {}},
        {"version": 1, "entries": {}, "tombstones": {}, "consumed": {}, "x": 1},
        {"version": 1, "entries": {"n": "x"}, "tombstones": {}, "consumed": {}},
        {
            "version": 1,
            "entries": {"n": {**_ENTRY, "state": "bogus"}},
            "tombstones": {},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {"n": {**_ENTRY, "dispatch_epoch": "3"}},
            "tombstones": {},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {"n": {**_ENTRY, "dispatch_epoch": True}},
            "tombstones": {},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {"n": {k: v for k, v in _ENTRY.items() if k != "text"}},
            "tombstones": {},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {"n": {**_ENTRY, "extra": 1}},
            "tombstones": {},
            "consumed": {},
        },
        {
            # A taken entry must name its poster.
            "version": 1,
            "entries": {"n": {**_ENTRY, "state": "taken"}},
            "tombstones": {},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {},
            "tombstones": {"n": {"operation_id": 5, "ts": 1.0}},
            "consumed": {},
        },
        {
            # A nonce cannot be both published and revoked.
            "version": 1,
            "entries": {"n": dict(_ENTRY)},
            "tombstones": {"n": {"operation_id": OP, "ts": 1.0}},
            "consumed": {},
        },
        {
            "version": 1,
            "entries": {},
            "tombstones": {},
            "consumed": {"x": {"seq": 1, "nonce": "n"}},
        },
        {
            "version": 1,
            "entries": {},
            "tombstones": {},
            "consumed": {"3": {"seq": "1", "nonce": "n"}},
        },
    ],
)
def test_malformed_documents_are_unknown_and_untouched(
    tmp_path: Path, payload: object
) -> None:
    _write_raw(tmp_path, payload)
    before = mb.mailbox_path(tmp_path, CHILD).read_bytes()
    assert mb.read_mailbox(tmp_path, CHILD) is None
    assert mb.read_entry(tmp_path, CHILD, "n").unknown
    assert _publish(tmp_path, "other").unknown
    assert mb.retract(tmp_path, CHILD, "n").unknown
    assert mb.cleanup(tmp_path, CHILD, {"n"}).unknown
    assert mb.mailbox_path(tmp_path, CHILD).read_bytes() == before


def test_unreadable_file_is_unknown(tmp_path: Path) -> None:
    # A directory where the file should be: exists, cannot be read as text.
    mb.mailbox_path(tmp_path, CHILD).mkdir()
    assert mb.read_mailbox(tmp_path, CHILD) is None
    assert _publish(tmp_path).unknown


def test_persistence_failure_is_unknown(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mb, "_save", lambda path, doc: False)
    assert _publish(session).unknown
    monkeypatch.undo()
    # The failed write left the prior document: still no entry.
    assert _state(session) == mb.STATE_ABSENT


def test_lock_failure_is_unknown(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise filelock.FileLockTimeoutError

    monkeypatch.setattr(mb, "file_lock", _boom)
    assert _publish(session).unknown
    assert mb.retract(session, CHILD, NONCE).unknown


# ==========================================================================
# publish
# ==========================================================================


def test_publish_offers_a_new_entry(session: Path) -> None:
    result = _publish(session)
    assert result.done
    assert result.state == mb.STATE_OFFERED
    entry = mb.read_entry(session, CHILD, NONCE).entry
    assert entry == {
        "operation_id": OP,
        "sender": "team-lead",
        "key": "k1",
        "dispatch_epoch": EPOCH,
        "backend_session_id": SESSION,
        "text": "hello",
        "state": "offered",
        "ts": 10.0,
    }


def test_publish_refuses_an_existing_entry(session: Path) -> None:
    assert _publish(session).done
    again = _publish(session, text="different")
    assert again.lost
    assert again.state == mb.STATE_OFFERED
    entry = mb.read_entry(session, CHILD, NONCE).entry
    assert entry is not None
    assert entry["text"] == "hello"


def test_publish_requires_the_row_to_be_current(session: Path) -> None:
    result = _publish(session, row_is_current=_no)
    assert result.rejected
    assert _state(session) == mb.STATE_ABSENT


def test_publish_callback_error_is_unknown_and_writes_nothing(session: Path) -> None:
    def _raise() -> bool:
        raise RuntimeError

    assert _publish(session, row_is_current=_raise).unknown
    assert _state(session) == mb.STATE_ABSENT


def test_publish_row_check_runs_under_the_mailbox_lock(session: Path) -> None:
    """The row check is the one place ``deliveries.lock`` nests inside."""
    seen: list[bool] = []

    def _probe() -> bool:
        lock = mb.mailbox_lock_path(session, CHILD)
        with lock.open("a+b") as handle:
            seen.append(filelock.try_lock_handle(handle))
            if seen[-1]:
                filelock.unlock_handle(handle)
        return True

    assert _publish(session, row_is_current=_probe).done
    assert seen == [False]


# ==========================================================================
# revoke vs publish (R3-3): whichever runs first wins
# ==========================================================================


def test_revoke_writes_a_tombstone_that_blocks_a_late_publish(session: Path) -> None:
    revoked = mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_yes)
    assert revoked.done
    assert _state(session) == mb.STATE_REVOKED
    late = _publish(session)
    assert late.lost
    assert late.state == mb.STATE_REVOKED
    assert NONCE not in _raw(session)["entries"]


def test_publish_first_makes_revoke_lose(session: Path) -> None:
    assert _publish(session).done
    revoked = mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_yes)
    assert revoked.lost
    assert revoked.state == mb.STATE_OFFERED
    assert NONCE not in _raw(session)["tombstones"]


def test_revoke_is_idempotent(session: Path) -> None:
    for _ in range(2):
        assert mb.revoke(
            session, CHILD, NONCE, operation_id=OP, row_is_current=_yes
        ).done


def test_revoke_requires_the_row_to_be_current(session: Path) -> None:
    result = mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_no)
    assert result.rejected
    assert _state(session) == mb.STATE_ABSENT


def _race(first, second) -> tuple[mb.MailboxResult, mb.MailboxResult]:
    """Run ``first`` holding the lock at a barrier while ``second`` contends.

    ``first(gate)`` must call ``gate()`` from inside its locked callback. The
    second call is started while the first is parked there, and must not
    complete until the first releases the lock.
    """
    entered = threading.Event()
    release = threading.Event()

    def gate() -> bool:
        entered.set()
        assert release.wait(10)
        return True

    results: dict[str, mb.MailboxResult] = {}
    t1 = threading.Thread(target=lambda: results.__setitem__("a", first(gate)))
    t1.start()
    assert entered.wait(10)
    t2 = threading.Thread(target=lambda: results.__setitem__("b", second()))
    t2.start()
    t2.join(0.3)
    assert t2.is_alive(), "second CAS ran while the first held the lock"
    release.set()
    t1.join(10)
    t2.join(10)
    return results["a"], results["b"]


def test_race_publish_holding_lock_then_revoke(session: Path) -> None:
    first, second = _race(
        lambda gate: _publish(session, row_is_current=gate),
        lambda: mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_yes),
    )
    assert first.done
    assert second.lost
    assert second.state == mb.STATE_OFFERED


def test_race_revoke_holding_lock_then_publish(session: Path) -> None:
    first, second = _race(
        lambda gate: mb.revoke(
            session, CHILD, NONCE, operation_id=OP, row_is_current=gate
        ),
        lambda: _publish(session),
    )
    assert first.done
    assert second.lost
    assert second.state == mb.STATE_REVOKED


# ==========================================================================
# take / begin: validation leaves the entry untouched
# ==========================================================================


def test_take_marks_taken_with_the_poster(session: Path) -> None:
    assert _publish(session).done
    result = _take(session)
    assert result.done
    assert result.state == mb.STATE_TAKEN
    assert result.entry is not None
    assert result.entry["poster"] == {"pid": 4242, "create_token": "tok-poster"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"binding": mb.HostBinding(EPOCH + 1, SESSION, 1000, "tok-host")},
        {"binding": mb.HostBinding(EPOCH, "other-session", 1000, "tok-host")},
        {
            "current_binding": lambda: mb.HostBinding(
                EPOCH + 1, SESSION, 1000, "tok-host"
            )
        },
        {"current_binding": lambda: mb.HostBinding(EPOCH, SESSION, 999, "tok-host")},
        {"current_binding": lambda: mb.HostBinding(EPOCH, SESSION, 1000, "other")},
        {"current_binding": lambda: None},
    ],
)
def test_take_validation_failure_leaves_entry_untouched(
    session: Path, overrides: dict
) -> None:
    assert _publish(session).done
    before = _raw(session)
    result = _take(session, **overrides)
    assert result.rejected
    assert _raw(session) == before


def test_take_callback_error_is_unknown(session: Path) -> None:
    assert _publish(session).done

    def _raise() -> mb.HostBinding:
        raise OSError

    assert _take(session, current_binding=_raise).unknown
    assert _state(session) == mb.STATE_OFFERED


def test_take_of_absent_or_revoked_nonce_is_lost(session: Path) -> None:
    assert _take(session).state == mb.STATE_ABSENT
    assert mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_yes).done
    result = _take(session)
    assert result.lost
    assert result.state == mb.STATE_REVOKED


def test_second_take_loses(session: Path) -> None:
    assert _publish(session).done
    assert _take(session).done
    again = _take(session, poster=OTHER_POSTER)
    assert again.lost
    assert again.state == mb.STATE_TAKEN


def test_begin_moves_to_posting_and_consumes_the_idle_seq(session: Path) -> None:
    assert _publish(session).done
    assert _take(session).done
    result = _begin(session)
    assert result.done
    assert result.state == mb.STATE_POSTING
    assert _raw(session)["consumed"] == {str(EPOCH): {"seq": 5, "nonce": NONCE}}


@pytest.mark.parametrize(
    "overrides",
    [
        {"poster": OTHER_POSTER},
        {"binding": mb.HostBinding(EPOCH + 1, SESSION, 1000, "tok-host")},
        {
            "current_binding": lambda: mb.HostBinding(
                EPOCH + 1, SESSION, 1000, "tok-host"
            )
        },
        {"current_binding": lambda: mb.HostBinding(EPOCH, SESSION, 1, "tok-host")},
        {"idle": mb.IdleProof(EPOCH + 1, SESSION, 5)},
        {"idle": mb.IdleProof(EPOCH, "other-session", 5)},
    ],
)
def test_begin_validation_failure_leaves_entry_untouched(
    session: Path, overrides: dict
) -> None:
    assert _publish(session).done
    assert _take(session).done
    before = _raw(session)
    assert _begin(session, **overrides).rejected
    assert _raw(session) == before


def test_begin_requires_a_fresh_idle_seq(session: Path) -> None:
    """At most one entry per idle sequence (§2.3.4, R3-2)."""
    for nonce in ("a", "b"):
        assert _publish(session, nonce).done
        assert _take(session, nonce).done
    assert _begin(session, "a").done
    stale = _begin(session, "b")
    assert stale.rejected
    assert _state(session, "b") == mb.STATE_TAKEN
    fresh = _begin(session, "b", idle=mb.IdleProof(EPOCH, SESSION, 6))
    assert fresh.done
    assert _raw(session)["consumed"][str(EPOCH)] == {"seq": 6, "nonce": "b"}


def test_consumption_is_per_epoch(session: Path) -> None:
    assert _publish(session, "a").done
    assert _take(session, "a").done
    assert _begin(session, "a", idle=mb.IdleProof(EPOCH, SESSION, 9)).done
    new = mb.HostBinding(EPOCH + 1, SESSION, 2000, "tok-new")
    assert _publish(session, "b", dispatch_epoch=EPOCH + 1).done
    assert _take(session, "b", binding=new, current_binding=lambda: new).done
    # A new epoch restarts idle_seq from 0; seq 1 is fresh in its namespace.
    assert _begin(
        session,
        "b",
        binding=new,
        current_binding=lambda: new,
        idle=mb.IdleProof(EPOCH + 1, SESSION, 1),
    ).done
    assert _raw(session)["consumed"] == {
        str(EPOCH): {"seq": 9, "nonce": "a"},
        str(EPOCH + 1): {"seq": 1, "nonce": "b"},
    }


def test_begin_on_offered_or_retracted_is_lost(session: Path) -> None:
    assert _publish(session).done
    result = _begin(session)
    assert result.lost
    assert result.state == mb.STATE_OFFERED


def test_replacement_poster_never_replays_taken_or_posting(session: Path) -> None:
    assert _publish(session).done
    assert _take(session).done
    assert _begin(session, poster=OTHER_POSTER).rejected
    assert _take(session, poster=OTHER_POSTER).lost
    assert _begin(session).done
    assert _take(session, poster=OTHER_POSTER).lost
    assert _begin(session, poster=OTHER_POSTER).lost
    assert mb.finish(session, CHILD, NONCE, poster=OTHER_POSTER, ok=True).rejected
    assert _state(session) == mb.STATE_POSTING


# ==========================================================================
# finish: posted / failed_before_write (+rollback) / uncertain
# ==========================================================================


def _to_posting(session: Path, nonce: str = NONCE, seq: int = 5) -> None:
    assert _publish(session, nonce).done
    assert _take(session, nonce).done
    assert _begin(session, nonce, idle=mb.IdleProof(EPOCH, SESSION, seq)).done


def test_finish_ok_is_posted(session: Path) -> None:
    _to_posting(session)
    result = mb.finish(session, CHILD, NONCE, poster=POSTER, ok=True)
    assert result.done
    assert result.state == mb.STATE_POSTED
    assert _raw(session)["consumed"][str(EPOCH)]["nonce"] == NONCE


def test_finish_after_write_started_is_uncertain_and_keeps_consumption(
    session: Path,
) -> None:
    _to_posting(session)
    result = mb.finish(
        session, CHILD, NONCE, poster=POSTER, ok=False, write_started=True
    )
    assert result.state == mb.STATE_UNCERTAIN
    assert _raw(session)["consumed"][str(EPOCH)] == {"seq": 5, "nonce": NONCE}


def test_failed_before_write_rolls_back_to_the_previous_value(session: Path) -> None:
    """R3-2: begin → pre-write failure → rollback → retry on the same seq."""
    _to_posting(session, "a", seq=4)
    assert mb.finish(session, CHILD, "a", poster=POSTER, ok=True).done
    _to_posting(session, "b", seq=5)
    result = mb.finish(
        session, CHILD, "b", poster=POSTER, ok=False, write_started=False
    )
    assert result.state == mb.STATE_FAILED_BEFORE_WRITE
    assert _raw(session)["consumed"][str(EPOCH)] == {"seq": 4, "nonce": "a"}
    # The retry (a new nonce from the lead) can use idle_seq 5 again.
    _to_posting(session, "c", seq=5)


def test_failed_before_write_with_no_prior_value_removes_the_epoch(
    session: Path,
) -> None:
    _to_posting(session)
    mb.finish(session, CHILD, NONCE, poster=POSTER, ok=False, write_started=False)
    assert _raw(session)["consumed"] == {}


def test_rollback_only_when_consumption_is_still_this_entrys(session: Path) -> None:
    """A later begin owns ``consumed[epoch]``; an older failure must not undo it."""
    for nonce in ("a", "b"):
        assert _publish(session, nonce).done
        assert _take(session, nonce).done
    assert _begin(session, "a", idle=mb.IdleProof(EPOCH, SESSION, 5)).done
    # Force "a" to lose ownership: a concurrent poster path consumed seq 6.
    assert _begin(session, "b", idle=mb.IdleProof(EPOCH, SESSION, 6)).done
    mb.finish(session, CHILD, "a", poster=POSTER, ok=False, write_started=False)
    assert _state(session, "a") == mb.STATE_FAILED_BEFORE_WRITE
    assert _raw(session)["consumed"][str(EPOCH)] == {"seq": 6, "nonce": "b"}


@pytest.mark.parametrize(
    "state_path", ["offered", "taken", "posted", "uncertain", "failed"]
)
def test_finish_outside_posting_is_lost(session: Path, state_path: str) -> None:
    assert _publish(session).done
    if state_path != "offered":
        assert _take(session).done
    if state_path in {"posted", "uncertain", "failed"}:
        assert _begin(session).done
        mb.finish(
            session,
            CHILD,
            NONCE,
            poster=POSTER,
            ok=state_path == "posted",
            write_started=state_path == "uncertain",
        )
    before = _raw(session)
    result = mb.finish(session, CHILD, NONCE, poster=POSTER, ok=True)
    assert result.lost
    assert _raw(session) == before


# ==========================================================================
# retract: exactly done / lost(<state>) / unknown
# ==========================================================================


def test_retract_offered_is_done(session: Path) -> None:
    assert _publish(session).done
    result = mb.retract(session, CHILD, NONCE)
    assert result.done
    assert result.state == mb.STATE_RETRACTED
    assert _take(session).state == mb.STATE_RETRACTED


def test_retract_of_taken_is_lost_by_default_and_done_when_allowed(
    session: Path,
) -> None:
    assert _publish(session).done
    assert _take(session).done
    lost = mb.retract(session, CHILD, NONCE)
    assert lost.lost
    assert lost.state == mb.STATE_TAKEN
    done = mb.retract(session, CHILD, NONCE, allow_taken=True)
    assert done.done
    # The poster's begin, having lost the CAS, sees retracted and stops.
    late = _begin(session)
    assert late.lost
    assert late.state == mb.STATE_RETRACTED


@pytest.mark.parametrize("final", ["posting", "posted", "uncertain", "failed"])
def test_retract_after_begin_is_lost_even_when_taken_is_allowed(
    session: Path, final: str
) -> None:
    _to_posting(session)
    if final != "posting":
        mb.finish(
            session,
            CHILD,
            NONCE,
            poster=POSTER,
            ok=final == "posted",
            write_started=final == "uncertain",
        )
    result = mb.retract(session, CHILD, NONCE, allow_taken=True)
    assert result.lost
    assert result.state != mb.STATE_RETRACTED


def test_retract_absent_and_revoked(session: Path) -> None:
    assert mb.retract(session, CHILD, NONCE).state == mb.STATE_ABSENT
    assert mb.revoke(session, CHILD, NONCE, operation_id=OP, row_is_current=_yes).done
    assert mb.retract(session, CHILD, NONCE).state == mb.STATE_REVOKED


def test_retract_persistence_failure_is_unknown(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _publish(session).done
    monkeypatch.setattr(mb, "_save", lambda path, doc: False)
    assert mb.retract(session, CHILD, NONCE).unknown
    monkeypatch.undo()
    assert _state(session) == mb.STATE_OFFERED


def test_race_take_holding_lock_then_retract(session: Path) -> None:
    assert _publish(session).done

    def first(gate):
        def current() -> mb.HostBinding:
            gate()
            return BINDING

        return _take(session, current_binding=current)

    took, retracted = _race(first, lambda: mb.retract(session, CHILD, NONCE))
    assert took.done
    assert retracted.lost
    assert retracted.state == mb.STATE_TAKEN


def test_retract_first_then_take_loses(session: Path) -> None:
    assert _publish(session).done
    assert mb.retract(session, CHILD, NONCE).done
    took = _take(session)
    assert took.lost
    assert took.state == mb.STATE_RETRACTED


# ==========================================================================
# Retention and cleanup
# ==========================================================================


def test_every_state_is_retained_until_cleanup(session: Path) -> None:
    _to_posting(session, "posted", seq=1)
    mb.finish(session, CHILD, "posted", poster=POSTER, ok=True)
    assert _publish(session, "offered").done
    assert mb.revoke(session, CHILD, "gone", operation_id=OP, row_is_current=_yes).done
    for _ in range(3):
        mb.ensure_initialised(session, CHILD)
    raw = _raw(session)
    assert set(raw["entries"]) == {"posted", "offered"}
    assert set(raw["tombstones"]) == {"gone"}


def test_cleanup_removes_only_terminal_nonces(session: Path) -> None:
    _to_posting(session, "posted", seq=1)
    mb.finish(session, CHILD, "posted", poster=POSTER, ok=True)
    assert _publish(session, "live").done
    assert mb.revoke(session, CHILD, "gone", operation_id=OP, row_is_current=_yes).done
    result = mb.cleanup(session, CHILD, {"posted", "gone", "never-existed"})
    assert result.done
    raw = _raw(session)
    assert set(raw["entries"]) == {"live"}
    assert raw["tombstones"] == {}
    # Consumption is idle-sequence bookkeeping, not an entry: it survives.
    assert raw["consumed"][str(EPOCH)]["nonce"] == "posted"


def test_cleanup_with_nothing_to_remove_does_not_rewrite(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _publish(session).done
    monkeypatch.setattr(mb, "_save", lambda path, doc: pytest.fail("rewrote"))
    assert mb.cleanup(session, CHILD, {"other"}).done


def test_read_entry_reports_state_without_writing(session: Path) -> None:
    assert _publish(session).done
    path = mb.mailbox_path(session, CHILD)
    before = path.stat().st_mtime_ns
    read = mb.read_entry(session, CHILD, NONCE)
    assert read.done
    assert read.state == mb.STATE_OFFERED
    assert path.stat().st_mtime_ns == before


def test_results_are_copies(session: Path) -> None:
    assert _publish(session).done
    read = mb.read_entry(session, CHILD, NONCE)
    assert read.entry is not None
    read.entry["state"] = "posted"
    assert _state(session) == mb.STATE_OFFERED


@pytest.mark.parametrize(
    ("nonce", "overrides"),
    [
        ("", {}),
        (NONCE, {"operation_id": ""}),
        (NONCE, {"dispatch_epoch": True}),
        (NONCE, {"text": None}),
    ],
)
def test_invalid_input_is_refused_and_never_bricks_the_mailbox(
    session: Path, nonce: str, overrides: dict
) -> None:
    """A write that would fail strict validation would read back as unknown."""
    before = _raw(session)
    with pytest.raises(ValueError, match="invalid mailbox"):
        _publish(session, nonce, **overrides)
    assert _raw(session) == before
    assert mb.read_mailbox(session, CHILD) is not None
