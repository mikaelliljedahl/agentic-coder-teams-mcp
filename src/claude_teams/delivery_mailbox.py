"""B — the delivery mailbox: the lead ⇄ Claude-child hand-off store (plan §2.3.2).

A Claude child has no native queue, so a guaranteed delivery to it is *offered*
here by the lead and *presented* by the child's own poster. The file is the
only thing the two sides share, so every rule below exists to keep one fact
recoverable: whether a nonce could have reached the child.

- **One document per child.** ``<session>/delivery-mailbox-<child>.json``,
  strictly validated, replaced atomically under ``delivery-mailbox-<child>.lock``.
- **Fail closed.** An unreadable, non-object or malformed document is
  *unknown*, never empty, and nothing is ever written over it. A missing file
  is unknown too (R3-3): :func:`ensure_initialised` creates the empty document
  under its lock **before** the first attempt to that child is marked ``sent``,
  so from then on absence can only mean the evidence was lost.
- **Every transition is one locked CAS** returning an authoritative
  :class:`MailboxResult`. A caller never decides from a preliminary read:

  .. code-block:: text

     (absent) --publish(row current, no tombstone)-->  offered
     (absent) --revoke(row current)-->                 revoked (tombstone)
     offered  --take(epoch, session, host match)-->    taken{poster}
     taken    --begin(same poster, fresh idle seq)-->  posting  (+consumed[epoch])
     posting  --finish(ok)-->                          posted
     posting  --finish(write_started=False)-->         failed_before_write (+rollback)
     posting  --finish(write_started=True)-->          uncertain
     offered  --retract-->                             retracted
     taken    --retract(allow_taken)-->                retracted

- **Replacement posters never replay.** ``take`` needs ``offered`` and
  ``begin``/``finish`` need the *same* poster, so an entry a dead poster left
  ``taken`` or ``posting`` is never presented again by its successor.
- **Retention.** Entries in every state, ``posted`` included, and tombstones
  are kept until the caller names their delivery rows terminal in
  :func:`cleanup`. Deleting an entry is never an acknowledgement.

Lock order is ``agents.lock`` ⇒ ``delivery-mailbox-<child>.lock`` ⇒
``deliveries.lock``. The callbacks passed to :func:`publish`, :func:`revoke`,
:func:`take` and :func:`begin` run *inside* the mailbox lock, so they may take
``deliveries.lock`` but must never take ``agents.lock`` (read ``agents.json``
without its lock instead: it is replaced atomically).

This module is pure storage: it never inspects processes, markers or the
delivery store itself. Everything it must not guess, the caller supplies.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from claude_teams.filelock import FileLockTimeoutError, file_lock

MAILBOX_VERSION = 1

#: Entry states.
STATE_OFFERED = "offered"
STATE_TAKEN = "taken"
STATE_POSTING = "posting"
STATE_POSTED = "posted"
STATE_FAILED_BEFORE_WRITE = "failed_before_write"
STATE_UNCERTAIN = "uncertain"
STATE_RETRACTED = "retracted"
#: Pseudo-states reported in results, never stored as an entry's ``state``.
#: ``revoked`` is a tombstone; ``absent`` is a valid mailbox with neither.
STATE_REVOKED = "revoked"
STATE_ABSENT = "absent"

ENTRY_STATES = frozenset(
    {
        STATE_OFFERED,
        STATE_TAKEN,
        STATE_POSTING,
        STATE_POSTED,
        STATE_FAILED_BEFORE_WRITE,
        STATE_UNCERTAIN,
        STATE_RETRACTED,
    }
)
#: States reached through ``take``: they must name the poster that took them.
_POSTER_STATES = frozenset(
    {
        STATE_TAKEN,
        STATE_POSTING,
        STATE_POSTED,
        STATE_FAILED_BEFORE_WRITE,
        STATE_UNCERTAIN,
    }
)

#: Result outcomes.
OUTCOME_DONE = "done"
#: The CAS found another state first; ``MailboxResult.state`` names it.
OUTCOME_LOST = "lost"
#: A caller-supplied precondition or identity check failed; nothing changed.
OUTCOME_REJECTED = "rejected"
#: The mailbox is missing, unreadable, malformed or could not be persisted,
#: or a callback raised. Nothing may be concluded, and nothing was written.
OUTCOME_UNKNOWN = "unknown"

#: Mirrors ``hooks._SAFE_AGENT_RE``: the child name is interpolated into a path.
_SAFE_CHILD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_DOC_KEYS = frozenset({"version", "entries", "tombstones", "consumed"})
_ENTRY_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "operation_id": str,
    "sender": str,
    "key": str,
    "dispatch_epoch": int,
    "backend_session_id": str,
    "text": str,
    "state": str,
    "ts": (int, float),
}
_ENTRY_OPTIONAL = frozenset({"poster", "consumed_prev"})


@dataclass(frozen=True)
class PosterIdentity:
    """The poster process incarnation that took an entry."""

    pid: int
    create_token: str

    def as_dict(self) -> dict[str, Any]:
        """Return the stored form."""
        return {"pid": self.pid, "create_token": self.create_token}


@dataclass(frozen=True)
class HostBinding:
    """What a poster is bound to: epoch, backend session and host incarnation.

    ``take`` and ``begin`` compare the poster's claimed binding with the
    entry and with the authoritative current binding (the child's agent
    record), and refuse unless all agree.
    """

    dispatch_epoch: int
    backend_session_id: str
    host_pid: int
    host_create_token: str


@dataclass(frozen=True)
class IdleProof:
    """The hook marker's idle evidence the poster read before ``begin``.

    The caller has already checked the marker says ``waiting``; the store
    checks that the proof belongs to the entry's epoch and backend session and
    that ``idle_seq`` has not been consumed in that epoch (R3-2).
    """

    dispatch_epoch: int
    backend_session_id: str
    idle_seq: int


@dataclass(frozen=True)
class MailboxResult:
    """The authoritative outcome of one mailbox operation.

    ``state`` is the entry's state after a ``done`` transition, or the state
    the CAS found for ``lost`` (``revoked`` for a tombstone, ``absent`` for no
    record at all). ``entry`` is a copy of the entry, when there is one.
    """

    outcome: str
    state: str | None = None
    reason: str = ""
    entry: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        """Whether the requested transition (or read) succeeded."""
        return self.outcome == OUTCOME_DONE

    @property
    def lost(self) -> bool:
        """Whether the CAS found a different state first."""
        return self.outcome == OUTCOME_LOST

    @property
    def rejected(self) -> bool:
        """Whether a precondition or identity check refused the transition."""
        return self.outcome == OUTCOME_REJECTED

    @property
    def unknown(self) -> bool:
        """Whether nothing can be concluded about the mailbox."""
        return self.outcome == OUTCOME_UNKNOWN


_UNKNOWN = MailboxResult(OUTCOME_UNKNOWN)


class _CallbackError(Exception):
    """A caller-supplied check raised: its answer is unknown."""


def _require_safe_child(child: str) -> str:
    if not isinstance(child, str) or not _SAFE_CHILD_RE.match(child):
        msg = f"unsafe agent name: {child!r}"
        raise ValueError(msg)
    return child


def mailbox_path(session_dir: Path, child: str) -> Path:
    """Return ``<session>/delivery-mailbox-<child>.json``."""
    return session_dir / f"delivery-mailbox-{_require_safe_child(child)}.json"


def mailbox_lock_path(session_dir: Path, child: str) -> Path:
    """Return the lock sidecar; never the data file, which is replaced."""
    return session_dir / f"delivery-mailbox-{_require_safe_child(child)}.lock"


def _empty_document() -> dict[str, Any]:
    return {"version": MAILBOX_VERSION, "entries": {}, "tombstones": {}, "consumed": {}}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_seq_nonce(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    mapping = cast("dict[str, Any]", value)
    return (
        set(mapping) == {"seq", "nonce"}
        and _is_int(mapping["seq"])
        and isinstance(mapping["nonce"], str)
        and bool(mapping["nonce"])
    )


def _valid_poster(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    mapping = cast("dict[str, Any]", value)
    return (
        set(mapping) == {"pid", "create_token"}
        and _is_int(mapping["pid"])
        and isinstance(mapping["create_token"], str)
    )


def _valid_entry(value: object) -> bool:  # noqa: PLR0911 - one return per rule.
    if not isinstance(value, dict):
        return False
    entry = cast("dict[str, Any]", value)
    if not set(_ENTRY_REQUIRED) <= set(entry):
        return False
    if set(entry) - set(_ENTRY_REQUIRED) - _ENTRY_OPTIONAL:
        return False
    for field, kind in _ENTRY_REQUIRED.items():
        field_value = entry[field]
        if isinstance(field_value, bool) or not isinstance(field_value, kind):
            return False
    if entry["state"] not in ENTRY_STATES or not entry["operation_id"]:
        return False
    if entry["state"] in _POSTER_STATES:
        if not _valid_poster(entry.get("poster")):
            return False
    elif "poster" in entry and not _valid_poster(entry["poster"]):
        return False
    prev = entry.get("consumed_prev")
    return prev is None or _valid_seq_nonce(prev)


def _valid_document(raw: object) -> bool:  # noqa: PLR0911 - one return per rule.
    """Strictly validate the whole document; any doubt means unknown."""
    if not isinstance(raw, dict):
        return False
    doc = cast("dict[str, Any]", raw)
    if set(doc) != _DOC_KEYS or doc["version"] != MAILBOX_VERSION:
        return False
    entries, tombstones, consumed = doc["entries"], doc["tombstones"], doc["consumed"]
    if not all(isinstance(part, dict) for part in (entries, tombstones, consumed)):
        return False
    for nonce, entry in entries.items():
        if not nonce or not _valid_entry(entry):
            return False
    for nonce, stone in tombstones.items():
        if not nonce or nonce in entries or not isinstance(stone, dict):
            return False
        stone_map = cast("dict[str, Any]", stone)
        if set(stone_map) != {"operation_id", "ts"}:
            return False
        if not isinstance(stone_map["operation_id"], str) or not _is_number(
            stone_map["ts"]
        ):
            return False
    for epoch, value in consumed.items():
        if not re.fullmatch(r"-?[0-9]+", epoch) or not _valid_seq_nonce(value):
            return False
    return True


def _load(path: Path) -> dict[str, Any] | None:
    """Return the validated document, or ``None`` for unknown (incl. missing)."""
    try:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
    except (OSError, ValueError):
        return None
    return cast("dict[str, Any]", raw) if _valid_document(raw) else None


def _save(path: Path, doc: dict[str, Any]) -> bool:
    """Atomically persist ``doc`` (temp file + replace); report success."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False
    return True


def _transact(
    session_dir: Path,
    child: str,
    step: Callable[[dict[str, Any]], tuple[MailboxResult, bool]],
) -> MailboxResult:
    """Run one CAS: lock, load (fail closed), decide, persist if dirty.

    ``step`` returns the result and whether it mutated the document. A lock
    failure, an unknown document, a failed write or a raising callback all
    collapse to ``unknown`` with nothing written.
    """
    path = mailbox_path(session_dir, child)
    try:
        with file_lock(mailbox_lock_path(session_dir, child)):
            doc = _load(path)
            if doc is None:
                return _UNKNOWN
            try:
                result, dirty = step(doc)
            except _CallbackError:
                return _UNKNOWN
            if dirty and not _valid_document(doc):
                # A caller bug (empty nonce, bool epoch, ...). Persisting it
                # would make every later read unknown, so refuse loudly.
                msg = f"refusing to write an invalid mailbox document to {path}"
                raise ValueError(msg)
            if dirty and not _save(path, doc):
                return _UNKNOWN
            return result
    except (OSError, FileLockTimeoutError):
        return _UNKNOWN


def _call(check: Callable[[], Any]) -> Any:
    """Run a caller check inside the lock; any exception means unknown."""
    try:
        return check()
    except Exception as exc:
        raise _CallbackError from exc


def _locate(doc: dict[str, Any], nonce: str) -> tuple[str, dict[str, Any] | None]:
    """Return ``(state, entry)`` for ``nonce``, with the pseudo-states."""
    entry = doc["entries"].get(nonce)
    if entry is not None:
        return entry["state"], entry
    if nonce in doc["tombstones"]:
        return STATE_REVOKED, None
    return STATE_ABSENT, None


def _result(
    outcome: str, state: str, entry: dict[str, Any] | None, reason: str = ""
) -> MailboxResult:
    return MailboxResult(
        outcome, state, reason, copy.deepcopy(entry) if entry is not None else None
    )


def _lost(state: str, entry: dict[str, Any] | None) -> tuple[MailboxResult, bool]:
    return _result(OUTCOME_LOST, state, entry), False


def _rejected(
    state: str, entry: dict[str, Any] | None, reason: str
) -> tuple[MailboxResult, bool]:
    return _result(OUTCOME_REJECTED, state, entry, reason), False


def _now(now: float | None) -> float:
    return time.time() if now is None else now


# ==========================================================================
# Lifecycle and reads
# ==========================================================================


def ensure_initialised(session_dir: Path, child: str) -> MailboxResult:
    """Create the empty document under its lock unless one already exists.

    Must succeed before the first attempt to ``child`` is marked ``sent``
    (R3-3). A present-but-invalid document is left alone and reported
    ``unknown``: replacing it would erase the very evidence it may hold.
    """
    path = mailbox_path(session_dir, child)
    try:
        with file_lock(mailbox_lock_path(session_dir, child)):
            if path.exists() or path.is_symlink():
                ok = _load(path) is not None
            else:
                ok = _save(path, _empty_document())
    except (OSError, FileLockTimeoutError):
        return _UNKNOWN
    return MailboxResult(OUTCOME_DONE) if ok else _UNKNOWN


def read_mailbox(session_dir: Path, child: str) -> dict[str, Any] | None:
    """Return a copy of the validated document, or ``None`` when unknown."""
    try:
        with file_lock(mailbox_lock_path(session_dir, child)):
            return _load(mailbox_path(session_dir, child))
    except (OSError, FileLockTimeoutError):
        return None


def read_entry(session_dir: Path, child: str, nonce: str) -> MailboxResult:
    """Read one nonce's state: ``done`` with a state, or ``unknown``.

    The state is an entry state, ``revoked`` or ``absent``. ``absent`` is only
    ever reported for a *valid* mailbox with no record of the nonce.
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        return _result(OUTCOME_DONE, state, entry), False

    return _transact(session_dir, child, step)


# ==========================================================================
# Lead side: publish, revoke, retract, cleanup
# ==========================================================================


def publish(  # noqa: PLR0913 - the entry's stored fields are its arguments.
    session_dir: Path,
    child: str,
    nonce: str,
    *,
    operation_id: str,
    sender: str,
    key: str,
    dispatch_epoch: int,
    backend_session_id: str,
    text: str,
    row_is_current: Callable[[], bool],
    now: float | None = None,
) -> MailboxResult:
    """Offer ``nonce`` to the child: ``absent`` ⇒ ``offered``.

    ``row_is_current`` runs under the mailbox lock and must confirm the
    delivery row is still ``sent`` with this ``operation_id`` (it may take
    ``deliveries.lock``). A tombstone or an existing entry is ``lost`` with its
    state; a false check is ``rejected``; nothing is written in either case.
    A holder whose publish lost to a tombstone must treat the row as pending.
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if state != STATE_ABSENT:
            return _lost(state, entry)
        if not _call(row_is_current):
            return _rejected(STATE_ABSENT, None, "row_not_current")
        new = {
            "operation_id": operation_id,
            "sender": sender,
            "key": key,
            "dispatch_epoch": dispatch_epoch,
            "backend_session_id": backend_session_id,
            "text": text,
            "state": STATE_OFFERED,
            "ts": _now(now),
        }
        doc["entries"][nonce] = new
        return _result(OUTCOME_DONE, STATE_OFFERED, new), True

    return _transact(session_dir, child, step)


def revoke(
    session_dir: Path,
    child: str,
    nonce: str,
    *,
    operation_id: str,
    row_is_current: Callable[[], bool],
    now: float | None = None,
) -> MailboxResult:
    """Tombstone an unpublished nonce: ``absent`` ⇒ ``revoked``.

    Recovery of a ``sent`` row with no entry. Serialised against
    :func:`publish` by the lock, so whichever runs first wins: ``done`` means
    no publish of this nonce can ever succeed and the row may go ``pending``;
    ``lost(<state>)`` means it was already published. Repeating a revoke of
    the same operation is ``done``. A false ``row_is_current`` is
    ``rejected`` (the row moved on; re-read it) and writes nothing.
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if state == STATE_REVOKED:
            same = doc["tombstones"][nonce]["operation_id"] == operation_id
            return (
                (MailboxResult(OUTCOME_DONE, STATE_REVOKED), False)
                if same
                else _lost(state, None)
            )
        if state != STATE_ABSENT:
            return _lost(state, entry)
        if not _call(row_is_current):
            return _rejected(STATE_ABSENT, None, "row_not_current")
        doc["tombstones"][nonce] = {"operation_id": operation_id, "ts": _now(now)}
        return MailboxResult(OUTCOME_DONE, STATE_REVOKED), True

    return _transact(session_dir, child, step)


def retract(
    session_dir: Path, child: str, nonce: str, *, allow_taken: bool = False
) -> MailboxResult:
    """Withdraw an unsent entry: ``offered`` ⇒ ``retracted``.

    Returns exactly one of ``done``, ``lost(<state>)`` or ``unknown``. Only
    ``done`` proves the nonce can never be presented and permits a new nonce.

    ``allow_taken`` also retracts ``taken``: ``begin`` is itself a CAS, so an
    entry still ``taken`` provably has not started a write (§2.3.2). It is off
    by default because the kill and recovery paths (§2.3.3, §2.3.5) leave
    ``taken`` unresolved.
    """
    retractable = {STATE_OFFERED, STATE_TAKEN} if allow_taken else {STATE_OFFERED}

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if entry is None or state not in retractable:
            return _lost(state, entry)
        entry["state"] = STATE_RETRACTED
        return _result(OUTCOME_DONE, STATE_RETRACTED, entry), True

    return _transact(session_dir, child, step)


def cleanup(
    session_dir: Path, child: str, terminal_nonces: Iterable[str]
) -> MailboxResult:
    """Drop the entries and tombstones of nonces whose rows are terminal.

    The caller reads the delivery store first and passes only nonces whose
    rows have settled; nothing else is ever removed, whatever its state.
    Per-epoch consumption is idle-sequence bookkeeping and is kept.
    """
    doomed = set(terminal_nonces)

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        dirty = False
        for part in ("entries", "tombstones"):
            for nonce in doomed & set(doc[part]):
                del doc[part][nonce]
                dirty = True
        return MailboxResult(OUTCOME_DONE), dirty

    return _transact(session_dir, child, step)


# ==========================================================================
# Poster side: take, begin, finish
# ==========================================================================


def _binding_mismatch(  # noqa: PLR0911 - one return per mismatch reason.
    entry: dict[str, Any],
    binding: HostBinding,
    current_binding: Callable[[], HostBinding | None],
) -> str:
    """Return why the poster may not act on ``entry``, or ``""``."""
    if entry["dispatch_epoch"] != binding.dispatch_epoch:
        return "epoch_mismatch"
    if entry["backend_session_id"] != binding.backend_session_id:
        return "backend_session_mismatch"
    current = _call(current_binding)
    if current is None:
        return "binding_unavailable"
    if current.dispatch_epoch != binding.dispatch_epoch:
        return "epoch_mismatch"
    if current.backend_session_id != binding.backend_session_id:
        return "backend_session_mismatch"
    if (current.host_pid, current.host_create_token) != (
        binding.host_pid,
        binding.host_create_token,
    ):
        return "host_mismatch"
    return ""


def take(
    session_dir: Path,
    child: str,
    nonce: str,
    *,
    poster: PosterIdentity,
    binding: HostBinding,
    current_binding: Callable[[], HostBinding | None],
    now: float | None = None,
) -> MailboxResult:
    """Claim an offered entry for ``poster``: ``offered`` ⇒ ``taken``.

    ``binding`` is what the poster's capability marker says it is bound to;
    ``current_binding`` runs under the lock and returns the child's
    authoritative binding (its agent record), or ``None`` when unavailable.
    Any mismatch with each other or with the entry is ``rejected`` and leaves
    the entry untouched. Any other state is ``lost``.
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if entry is None or state != STATE_OFFERED:
            return _lost(state, entry)
        reason = _binding_mismatch(entry, binding, current_binding)
        if reason:
            return _rejected(state, entry, reason)
        entry["state"] = STATE_TAKEN
        entry["poster"] = poster.as_dict()
        entry["ts"] = _now(now)
        return _result(OUTCOME_DONE, STATE_TAKEN, entry), True

    return _transact(session_dir, child, step)


def begin(
    session_dir: Path,
    child: str,
    nonce: str,
    *,
    poster: PosterIdentity,
    binding: HostBinding,
    current_binding: Callable[[], HostBinding | None],
    idle: IdleProof,
    now: float | None = None,
) -> MailboxResult:
    """Commit to writing: ``taken`` ⇒ ``posting``, consuming the idle seq.

    Only the poster that took the entry may begin it; anyone else is
    ``rejected``. The binding is revalidated exactly as in :func:`take`, and
    ``idle`` must belong to the entry's epoch and backend session with an
    ``idle_seq`` greater than ``consumed[epoch].seq``. In the same CAS the
    consumption becomes ``{seq: idle_seq, nonce}``; the value it replaced is
    kept on the entry so a pre-write failure can roll it back (R3-2).
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if entry is None or state != STATE_TAKEN:
            return _lost(state, entry)
        if entry.get("poster") != poster.as_dict():
            return _rejected(state, entry, "poster_mismatch")
        reason = _binding_mismatch(entry, binding, current_binding)
        if reason:
            return _rejected(state, entry, reason)
        if (idle.dispatch_epoch, idle.backend_session_id) != (
            entry["dispatch_epoch"],
            entry["backend_session_id"],
        ):
            return _rejected(state, entry, "idle_proof_mismatch")
        epoch_key = str(entry["dispatch_epoch"])
        prior = doc["consumed"].get(epoch_key)
        if prior is not None and idle.idle_seq <= prior["seq"]:
            return _rejected(state, entry, "idle_seq_consumed")
        doc["consumed"][epoch_key] = {"seq": idle.idle_seq, "nonce": nonce}
        entry["state"] = STATE_POSTING
        entry["consumed_prev"] = copy.deepcopy(prior)
        entry["ts"] = _now(now)
        return _result(OUTCOME_DONE, STATE_POSTING, entry), True

    return _transact(session_dir, child, step)


def finish(
    session_dir: Path,
    child: str,
    nonce: str,
    *,
    poster: PosterIdentity,
    ok: bool,
    write_started: bool = True,
    now: float | None = None,
) -> MailboxResult:
    """Record the write's outcome for a ``posting`` entry.

    - ``ok`` ⇒ ``posted`` (accepted by the channel, not yet a receipt);
    - failure with ``write_started=False`` ⇒ ``failed_before_write``, provably
      unsent; ``consumed[epoch]`` is rolled back to its previous value only
      while it still names this nonce, so a later ``begin`` is never undone;
    - failure with ``write_started=True`` ⇒ ``uncertain``; consumption kept.

    Only the poster that began it may finish it.
    """

    def step(doc: dict[str, Any]) -> tuple[MailboxResult, bool]:
        state, entry = _locate(doc, nonce)
        if entry is None or state != STATE_POSTING:
            return _lost(state, entry)
        if entry.get("poster") != poster.as_dict():
            return _rejected(state, entry, "poster_mismatch")
        if ok:
            new_state = STATE_POSTED
        elif write_started:
            new_state = STATE_UNCERTAIN
        else:
            new_state = STATE_FAILED_BEFORE_WRITE
            epoch_key = str(entry["dispatch_epoch"])
            owned = doc["consumed"].get(epoch_key)
            if owned is not None and owned["nonce"] == nonce:
                prev = entry.get("consumed_prev")
                if prev is None:
                    del doc["consumed"][epoch_key]
                else:
                    doc["consumed"][epoch_key] = copy.deepcopy(prev)
        entry["state"] = new_state
        entry["ts"] = _now(now)
        return _result(OUTCOME_DONE, new_state, entry), True

    return _transact(session_dir, child, step)


__all__ = [
    "ENTRY_STATES",
    "MAILBOX_VERSION",
    "OUTCOME_DONE",
    "OUTCOME_LOST",
    "OUTCOME_REJECTED",
    "OUTCOME_UNKNOWN",
    "STATE_ABSENT",
    "STATE_FAILED_BEFORE_WRITE",
    "STATE_OFFERED",
    "STATE_POSTED",
    "STATE_POSTING",
    "STATE_RETRACTED",
    "STATE_REVOKED",
    "STATE_TAKEN",
    "STATE_UNCERTAIN",
    "HostBinding",
    "IdleProof",
    "MailboxResult",
    "PosterIdentity",
    "begin",
    "cleanup",
    "ensure_initialised",
    "finish",
    "mailbox_lock_path",
    "mailbox_path",
    "publish",
    "read_entry",
    "read_mailbox",
    "retract",
    "revoke",
    "take",
]
