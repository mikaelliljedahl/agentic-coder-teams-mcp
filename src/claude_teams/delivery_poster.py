"""B — the child's delivery poster (plan §2.3.4, §2.9 R3-2).

A Claude child has no native queue. The lead offers a guaranteed delivery in
the child's delivery mailbox (:mod:`claude_teams.delivery_mailbox`); this
poster, running in the **child's own** MCP server, presents it to the child's
**own** host channel (:func:`native_wake.post_claude_notice`). It is a second
target kind of :class:`native_wake.NativeWakeNotifier`, ticked on the same
thread and holding a lifetime owner lock the same way the inbox doorbell does.

Per tick:

1. **Gate.** Only with the master flag, the downstream flag and the ``_CLAUDE``
   half on, read from this process's own environment, and only for this
   server's own identity.
2. **Own and advertise.** Hold ``native-delivery-<IDENTITY>.lock`` for life and
   refresh the capability marker ``native-delivery-<IDENTITY>.json``. The lead
   treats a Claude channel as proven (E3) only while that marker is fresh, the
   lock is held and every binding in it matches the child's record.
3. **Post at most one entry**, and only when the hook marker says ``waiting``
   in this entry's dispatch epoch and backend session with an ``idle_seq``
   above the epoch's consumption. ``take`` and ``begin`` each revalidate the
   binding against the child's record under the mailbox lock; ``begin``
   consumes the idle sequence atomically. The write's outcome is recorded by
   ``finish``: ``ok`` ⇒ posted, a failure before any byte ⇒
   ``failed_before_write`` (consumption rolled back, so the next offer re-arms
   at once), a failure after the write started ⇒ ``uncertain``.

The poster never replays: it only takes ``offered`` entries, resumes only
entries it took itself, and never touches ``posting`` or later. The residual
race is documented, not closed: a keystroke or a wake notice can start a turn
between the marker read and the post, so the rule is "never *knowingly*
mid-turn".
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from claude_teams import delivery_mailbox, filelock, native_wake

_LOG = logging.getLogger(__name__)

#: Seconds a poster that lost the owner lock waits before trying again.
_REACQUIRE_SECONDS = 10.0


def capability_file(session_dir: Path, identity: str) -> Path:
    """Return the capability marker ``native-delivery-<identity>.json``."""
    return session_dir / f"native-delivery-{identity}.json"


def owner_lock_file(session_dir: Path, identity: str) -> Path:
    """Return the poster's lifetime owner lock ``native-delivery-<identity>.lock``."""
    return session_dir / f"native-delivery-{identity}.lock"


@dataclass(frozen=True)
class PosterFacts:
    """What the poster needs from its server, injected.

    - ``host``: this server's Claude host incarnation ``(pid, creation
      token)``, or ``None`` when it cannot be proven.
    - ``current_binding``: the child's authoritative binding from its agent
      record, given this poster's host; runs under the mailbox lock, so it
      must read ``agents.json`` without ``agents.lock``.
    - ``read_marker``: the hook state marker, read without a lock (it is
      replaced atomically).
    - ``identity``: this poster process's incarnation.
    - ``epoch``: the dispatch epoch this host was started with.
    """

    host: Callable[[], tuple[int, str] | None]
    current_binding: Callable[
        [str, str, tuple[int, str]], delivery_mailbox.HostBinding | None
    ]
    read_marker: Callable[[str, str], dict | None]
    identity: delivery_mailbox.PosterIdentity
    epoch: Callable[[], int]


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _consumed_seq(doc: dict[str, Any], epoch: int) -> int:
    consumed = doc["consumed"].get(str(epoch))
    return int(consumed["seq"]) if consumed is not None else 0


def _idle_for(marker: dict | None, entry: dict[str, Any], consumed: int) -> bool:
    """Whether ``marker`` proves an unconsumed idle edge for ``entry``."""
    if not isinstance(marker, dict):
        return False
    seq = _int(marker.get("idle_seq"))
    return (
        marker.get("state") == "waiting"
        and _int(marker.get("dispatch_epoch")) == entry["dispatch_epoch"]
        and marker.get("backend_session_id") == entry["backend_session_id"]
        and seq is not None
        and seq > consumed
    )


class DeliveryPoster:
    """Present offered mailbox entries to this child's own host channel."""

    def __init__(
        self,
        facts: PosterFacts,
        *,
        get_target: Callable[[], tuple[str, str] | None],
        session_dir: Callable[[str], Path],
        channel: native_wake.ClaudeChannel,
        post: Callable[
            [native_wake.ClaudeChannel, str], native_wake.PostResult
        ] = native_wake.post_claude_notice,
        poll: float = 1.0,
    ) -> None:
        """Inject the server's facts, the channel and its transport."""
        self.facts = facts
        self.get_target = get_target
        self.session_dir = session_dir
        self.channel = channel
        self.post = post
        self.poll = poll
        self._handle: BinaryIO | None = None
        self._key: tuple[str, str] | None = None
        self._acquire_after = 0.0

    # ----------------------------------------------------------------------
    # Ownership
    # ----------------------------------------------------------------------

    def owns(self) -> bool:
        """Whether this poster currently holds its owner lock."""
        return self._handle is not None and not self._handle.closed

    def close(self) -> None:
        """Release the owner lock; the capability then stops being proven."""
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._handle.close()
        self._handle = None
        self._key = None

    def _own(self, key: tuple[str, str]) -> bool:
        if self._key != key:
            self.close()
        if self.owns():
            return True
        now = time.monotonic()
        if now < self._acquire_after:
            return False
        directory = self.session_dir(key[0])
        handle = owner_lock_file(directory, key[1]).open("a+b")
        try:
            owned = filelock.try_lock_handle(handle)
        except Exception:
            handle.close()
            raise
        if not owned:
            handle.close()
            self._acquire_after = now + _REACQUIRE_SECONDS
            return False
        self._handle = handle
        self._key = key
        return True

    # ----------------------------------------------------------------------
    # The tick
    # ----------------------------------------------------------------------

    def tick(self) -> str:
        """Run one poster step; return what happened (for logs and tests)."""
        if not native_wake.downstream_enabled("CLAUDE"):
            self.close()
            return "disabled"
        key = self.get_target()
        if key is None:
            self.close()
            return "no_identity"
        if not self._own(key):
            return "not_owner"
        session, identity = key
        directory = self.session_dir(session)
        host = self.facts.host()
        epoch = self.facts.epoch()
        marker = self.facts.read_marker(session, identity)
        self._write_capability(directory, identity, host, epoch, marker)
        if self.channel.reason != "available" or host is None:
            return "channel_unavailable"
        return self._post_one(directory, (session, identity), host, epoch, marker)

    def _write_capability(
        self,
        directory: Path,
        identity: str,
        host: tuple[int, str] | None,
        epoch: int,
        marker: dict | None,
    ) -> None:
        """Heartbeat the capability marker (atomically replaced)."""
        session_id = ""
        if isinstance(marker, dict) and _int(marker.get("dispatch_epoch")) == epoch:
            value = marker.get("backend_session_id")
            session_id = value if isinstance(value, str) else ""
        capability = {
            "pid": self.facts.identity.pid,
            "create_token": self.facts.identity.create_token,
            "host_pid": host[0] if host else 0,
            "host_create_token": host[1] if host else "",
            "backend_session_id": session_id,
            "dispatch_epoch": epoch,
            "channel": self.channel.reason,
            "heartbeat_ts": time.time(),
        }
        path = capability_file(directory, identity)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(capability), encoding="utf-8")
            tmp.replace(path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _choose(self, doc: dict[str, Any], epoch: int) -> tuple[str, dict] | None:
        """Oldest entry of this epoch: one this poster took, else an offer."""
        mine = self.facts.identity.as_dict()
        entries = [
            (entry["ts"], nonce, entry)
            for nonce, entry in doc["entries"].items()
            if entry["dispatch_epoch"] == epoch
        ]
        entries.sort(key=lambda item: (item[0], item[1]))
        taken = [
            (nonce, entry)
            for _, nonce, entry in entries
            if entry["state"] == delivery_mailbox.STATE_TAKEN
            and entry.get("poster") == mine
        ]
        if taken:
            return taken[0]
        offered = [
            (nonce, entry)
            for _, nonce, entry in entries
            if entry["state"] == delivery_mailbox.STATE_OFFERED
        ]
        return offered[0] if offered else None

    def _post_one(  # noqa: PLR0911 - one return per gate.
        self,
        directory: Path,
        key: tuple[str, str],
        host: tuple[int, str],
        epoch: int,
        marker: dict | None,
    ) -> str:
        """Take, begin, post and finish at most one entry."""
        session, identity = key
        doc = delivery_mailbox.read_mailbox(directory, identity)
        if doc is None:
            return "mailbox_unknown"
        chosen = self._choose(doc, epoch)
        if chosen is None:
            return "nothing_offered"
        nonce, entry = chosen
        consumed = _consumed_seq(doc, epoch)
        if not _idle_for(marker, entry, consumed):
            return "not_idle"
        binding = delivery_mailbox.HostBinding(
            epoch, entry["backend_session_id"], host[0], host[1]
        )

        def current() -> delivery_mailbox.HostBinding | None:
            return self.facts.current_binding(session, identity, host)

        poster = self.facts.identity
        if entry["state"] == delivery_mailbox.STATE_OFFERED:
            taken = delivery_mailbox.take(
                directory,
                identity,
                nonce,
                poster=poster,
                binding=binding,
                current_binding=current,
            )
            if not taken.done:
                return f"take_{taken.outcome}"
        # Fresh proof for begin: the child may have started a turn since.
        marker = self.facts.read_marker(session, identity)
        if not _idle_for(marker, entry, consumed):
            return "not_idle_at_begin"
        assert isinstance(marker, dict)  # noqa: S101 - guaranteed by _idle_for.
        begun = delivery_mailbox.begin(
            directory,
            identity,
            nonce,
            poster=poster,
            binding=binding,
            current_binding=current,
            idle=delivery_mailbox.IdleProof(
                entry["dispatch_epoch"],
                entry["backend_session_id"],
                int(marker["idle_seq"]),
            ),
        )
        if not begun.done:
            return f"begin_{begun.outcome}"
        try:
            result = self.post(self.channel, entry["text"])
        except Exception as err:
            # Nothing proves the write never started: uncertain, never retried.
            _LOG.warning("Delivery post raised %s", type(err).__name__)
            result = native_wake.PostResult(False, type(err).__name__, True)
        finished = delivery_mailbox.finish(
            directory,
            identity,
            nonce,
            poster=poster,
            ok=result.ok,
            write_started=result.write_started,
        )
        if not finished.done:
            _LOG.warning("Could not record the delivery post outcome for %s", nonce)
        return "posted" if result.ok else f"post_failed:{result.reason}"


__all__ = [
    "DeliveryPoster",
    "PosterFacts",
    "capability_file",
    "owner_lock_file",
]
