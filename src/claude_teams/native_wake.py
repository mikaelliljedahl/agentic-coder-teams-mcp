"""Opt-in, best-effort inbox doorbells; inbox cursors remain authoritative."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import socket
import sqlite3
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from claude_teams import filelock, messaging, procinfo, winpipe

_LOG = logging.getLogger(__name__)
_activation = threading.Event()
_registry_lock = threading.Lock()
_members: set[tuple[str, str]] = set()


def enabled(half: str = "", environ: Mapping[str, str] | None = None) -> bool:
    """Return the strict opt-in gate, optionally applying a half's kill switch."""
    values = os.environ if environ is None else environ
    return values.get("WIN_AGENT_TEAMS_NATIVE_WAKE", "").strip() == "1" and (
        not half or values.get(f"WIN_AGENT_TEAMS_NATIVE_WAKE_{half}", "").strip() != "0"
    )


def downstream_enabled(half: str, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether new downstream deliveries may use ``half``'s native carrier.

    Needs the master flag, ``WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`` and a half
    switch that is not ``0``. It gates new attempts only: it never lifts the
    barrier on native attempts that are already unresolved.
    """
    values = os.environ if environ is None else environ
    return (
        enabled(half, values)
        and values.get("WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM", "").strip() == "1"
    )


_PROPAGATED_SUB_FLAGS = (
    "WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE",
    "WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX",
)


def propagated_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the native flags a spawned child's MCP server must see (plan §2.7).

    Empty unless the master flag is effectively on, so flag-off spawns stay
    byte-identical. Otherwise the master is passed as ``1`` and the downstream
    flag and both half switches as their current values; an absent flag stays
    absent, so nothing is ever turned on implicitly.
    """
    values = os.environ if environ is None else environ
    if not enabled(environ=values):
        return {}
    env = {"WIN_AGENT_TEAMS_NATIVE_WAKE": "1"}
    env.update({key: values[key] for key in _PROPAGATED_SUB_FLAGS if key in values})
    return env


def claude_platform_supported() -> bool:
    """Linux (/proc host walk, AF_UNIX) and Windows (toolhelp walk, named pipe).

    macOS stays unsupported: nearest-host discovery cannot resolve ``claude``.
    """
    if os.name == "nt":
        return sys.platform == "win32"
    return sys.platform == "linux"


def positive_seconds(raw: str, default: float) -> float:
    """Parse finite positive seconds, falling back for invalid settings."""
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _seconds(suffix: str, default: float) -> float:
    return positive_seconds(os.environ.get(f"WIN_AGENT_TEAMS_{suffix}", ""), default)


@dataclass(frozen=True)
class ClaudeChannel:
    """Cached host channel availability, never a delivery confirmation."""

    reason: str
    path: str = ""
    token: str = field(default="", repr=False)
    owner_verified: bool = False
    # Windows: every post requires the pipe's server process to be this PID.
    host_pid: int = 0


@dataclass(frozen=True)
class PostResult:
    """One bounded transport attempt's outcome."""

    ok: bool
    reason: str = ""
    # True once the user line may have reached the host: a failure after this
    # point is uncertain and never proves non-delivery.
    write_started: bool = False


def resolve_claude_channel(  # noqa: PLR0911 - H-row decision table.
    environ: Mapping[str, str],
    resolve_host: Callable[[], procinfo.HostResolution] = procinfo.resolve_nearest_host,
) -> ClaudeChannel:
    """Resolve once, refusing inherited sockets under a non-Claude nearest host."""
    if not enabled("CLAUDE", environ):
        return ClaudeChannel("disabled")
    if not claude_platform_supported():
        return ClaudeChannel("unsupported_platform")
    path = environ.get("CLAUDE_CODE_MESSAGING_SOCKET", "")
    token = environ.get("CLAUDE_CODE_MESSAGING_TOKEN", "")
    if not path or not token:
        return ClaudeChannel("no_socket")
    try:
        host = resolve_host().host
        if host is None or not procinfo.is_claude_host(host):
            return ClaudeChannel("host_not_claude")
        if sys.platform == "win32":
            # H3-W/H4-W: a local pipe that exists. Ownership is proved per post
            # (server PID == host PID) because a pipe name carries no PID.
            if not winpipe.is_pipe_path(path) or not winpipe.pipe_exists(path):
                return ClaudeChannel("socket_missing")
            return ClaudeChannel("available", path, token, False, host.pid)
        match = re.fullmatch(r"(\d+)\.sock", Path(path).name)
        if match and int(match[1]) != host.pid:
            return ClaudeChannel("socket_not_owned")
        if not stat.S_ISSOCK(Path(path).stat().st_mode):
            return ClaudeChannel("socket_missing")
    except (OSError, ValueError):
        return ClaudeChannel("socket_missing")
    return ClaudeChannel("available", path, token, bool(match), host.pid)


def post_claude_notice(
    channel: ClaudeChannel, text: str, deadline: float = 5.0
) -> PostResult:
    """Write auth and user JSON lines with a real total operation deadline."""
    if not claude_platform_supported() or channel.reason != "available":
        return PostResult(False, channel.reason)
    wire = json.dumps({"type": "auth", "token": channel.token}) + "\n"
    wire += (
        json.dumps({"type": "user", "message": {"role": "user", "content": text}})
        + "\n"
    )
    if sys.platform == "win32":
        if not channel.host_pid:
            return PostResult(False, "socket_not_owned")
        piped = winpipe.post(
            channel.path,
            wire.encode("utf-8"),
            deadline,
            expected_pid=channel.host_pid,
        )
        return PostResult(piped.ok, piped.reason, piped.write_started)
    started = False
    try:
        end = time.monotonic() + deadline
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(deadline)
            connection.connect(channel.path)
            connection.settimeout(max(0.001, end - time.monotonic()))
            started = True
            connection.sendall(wire.encode("utf-8"))
            connection.shutdown(socket.SHUT_WR)
    except Exception as err:
        # Never include transport text: an exception may echo a credential.
        return PostResult(False, type(err).__name__, started)
    return PostResult(True, "", True)


@dataclass(frozen=True)
class NoticeConfig:
    """Coalesce and outstanding-notice windows."""

    coalesce: float = 2.0
    renotify: float = 300.0


@dataclass(frozen=True)
class Notice:
    """A proposal, committed only by the successful transport path."""

    counts: dict[str, int]
    text: str


@dataclass
class NoticeState:
    """Per-target successful progress and a pending coalesce deadline."""

    notified: dict[str, int] = field(default_factory=dict)
    first_new: float | None = None
    last_success: float | None = None
    seq: int = 0

    def observe(
        self,
        snapshot: dict[str, dict[str, int]],
        now: float,
        cfg: NoticeConfig,
    ) -> None:
        """Track coalesce timing outside the pure notice decision function."""
        if _notice_counts(self, snapshot, now, cfg) is not None:
            if self.first_new is None:
                self.first_new = now
        elif not any(row["cursor"] < row["total"] for row in snapshot.values()):
            self.first_new = None

    def succeeded(self, snapshot: dict[str, dict[str, int]], now: float) -> None:
        """Commit totals only after a successful notice."""
        self.notified = {sender: row["total"] for sender, row in snapshot.items()}
        self.first_new = None
        self.last_success = now
        self.seq += 1


def _notice_counts(
    state: NoticeState,
    snapshot: dict[str, dict[str, int]],
    now: float,
    cfg: NoticeConfig,
) -> dict[str, int] | None:
    unread = {s: r["total"] - min(r["cursor"], r["total"]) for s, r in snapshot.items()}
    unread = {s: count for s, count in unread.items() if count > 0}
    outstanding = any(
        min(r["cursor"], r["total"]) < state.notified.get(s, 0)
        for s, r in snapshot.items()
    )
    expired = (
        state.last_success is not None and now - state.last_success >= cfg.renotify
    )
    if outstanding and not expired:
        return None
    new = {
        s: r["total"] - max(state.notified.get(s, 0), min(r["cursor"], r["total"]))
        for s, r in snapshot.items()
    }
    if not unread or (not any(count > 0 for count in new.values()) and not expired):
        return None
    return unread


def plan_notice(
    state: NoticeState,
    snapshot: dict[str, dict[str, int]],
    now: float,
    cfg: NoticeConfig,
    *,
    member: bool = False,
) -> Notice | None:
    """Propose a body-free doorbell, deferring coalesce and outstanding notices."""
    unread = _notice_counts(state, snapshot, now, cfg)
    if unread is None:
        return None
    first_new = state.first_new if state.first_new is not None else now
    if now - first_new < cfg.coalesce:
        return None
    counts = ", ".join(
        f"{sender} ({count})" for sender, count in sorted(unread.items())
    )
    instruction = (
        "external_read with the member_token you saved" if member else "read_messages"
    )
    text = (
        f"[win-agent-teams wake #{state.seq + 1}] {sum(unread.values())} "
        f"unread message(s) in your team inbox from: {counts}. "
        f"Best-effort notice without content; call {instruction} to read them."
    )
    return Notice(unread, text)


@dataclass
class Backoff:
    """Bound every failed attempt with exponential retry spacing."""

    delay: float = 0.0
    until: float = 0.0

    def failed(self, now: float) -> None:
        """Schedule the next attempt, capped at five minutes."""
        self.delay = min(300.0, self.delay * 2 if self.delay else 2.0)
        self.until = now + self.delay

    def reset(self) -> None:
        """Clear failure spacing after success."""
        self.delay = self.until = 0.0


def scan_inbox(directory: Path, reader: str) -> dict[str, dict[str, int]]:
    """Read sender totals and cursors without advancing consumption."""
    senders = messaging.read_inbox_by_sender(directory / f"inbox-{reader}.jsonl")
    cursors = messaging.load_inbox_cursors(directory / f"inbox-{reader}.pos.json")
    return {
        sender: {"total": len(rows), "cursor": cursors.get(sender, 0)}
        for sender, rows in senders.items()
    }


def session_activated(session_id: str) -> None:
    """Wake the tool-independent scanner after an explicit session activation."""
    if enabled() and session_id:
        _activation.set()


def watch_member(session_id: str, name: str) -> None:
    """Register a member target; callers must already have released agents locks."""
    if enabled("CLAUDE"):
        with _registry_lock:
            _members.add((session_id, name))
        _activation.set()


@dataclass
class _Target:
    state: NoticeState = field(default_factory=NoticeState)
    backoff: Backoff = field(default_factory=Backoff)
    handle: BinaryIO | None = None
    acquire_after: float = 0.0
    signature: tuple[Any, ...] | None = None
    snapshot: dict[str, dict[str, int]] = field(default_factory=dict)
    catchup: bool = True
    # Codex lead only: the registered thread was verified in its home.
    verified: bool = False

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _signature(directory: Path, reader: str) -> tuple[Any, ...]:
    values = []
    for suffix in ("jsonl", "pos.json"):
        try:
            info = (directory / f"inbox-{reader}.{suffix}").stat()
            values.append((info.st_mtime_ns, info.st_size))
        except FileNotFoundError:
            values.append(None)
    return tuple(values)


LEAD_WAKE_STATUSES = frozenset({"provisional", "active", "cleared"})


def lead_wake_file(directory: Path, identity: str) -> Path:
    """Return the Codex lead registration ``lead-wake-<identity>.json``."""
    return directory / f"lead-wake-{identity}.json"


def _valid_lead_wake(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    generation = value.get("generation")
    host_pid = value.get("host_pid")
    return (
        isinstance(generation, int)
        and not isinstance(generation, bool)
        and generation >= 1
        and value.get("status") in LEAD_WAKE_STATUSES
        and isinstance(value.get("thread_id"), str | None)
        and isinstance(value.get("codex_home"), str | None)
        and isinstance(host_pid, int)
        and not isinstance(host_pid, bool)
        and isinstance(value.get("host_create_token"), str | None)
        and isinstance(value.get("reason"), str)
        and (value.get("status") == "cleared" or bool(value.get("thread_id")))
    )


def read_lead_wake(directory: Path, identity: str) -> dict | None:
    """Read a registration; a missing, corrupt or malformed file reads as absent."""
    try:
        value = json.loads(lead_wake_file(directory, identity).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return value if _valid_lead_wake(value) else None


def _update_lead_wake(
    directory: Path, identity: str, change: Callable[[dict | None], dict | None]
) -> dict | None:
    """Apply ``change`` under ``lead-wake-<identity>.lock``; ``None`` writes nothing."""
    path = lead_wake_file(directory, identity)
    with filelock.file_lock(path.with_suffix(".lock")):
        prior = read_lead_wake(directory, identity)
        value = change(prior)
        if value is None:
            return prior
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
            tmp.replace(path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
        return value


def register_lead_wake(
    directory: Path,
    identity: str,
    *,
    thread_id: str,
    codex_home: str,
    host: tuple[int, str | None],
    spawned: bool,
    bound: str | None,
) -> dict:
    """Register, replace or clear a Codex lead doorbell (plan §2.6, R2-6).

    Every call bumps ``generation``; a blank thread writes a ``cleared``
    tombstone. A human-started lead is ``active`` at once. A spawned lead is
    ``provisional`` until the parent-bound ``bound`` backend session exists:
    equal to the thread makes it ``active``, different makes it ``cleared``.
    The host incarnation is recorded so a new host ignores the registration.
    """

    def change(prior: dict | None) -> dict:
        generation = (prior or {}).get("generation", 0) + 1
        if not thread_id:
            status, reason = "cleared", "cleared_by_lead"
        elif not spawned:
            status, reason = "active", ""
        elif not bound:
            status, reason = "provisional", "awaiting_parent_binding"
        elif bound == thread_id:
            status, reason = "active", ""
        else:
            status, reason = "cleared", "thread_mismatch"
        return {
            "thread_id": thread_id or None,
            "codex_home": codex_home if thread_id else None,
            "generation": generation,
            "host_pid": host[0],
            "host_create_token": host[1],
            "status": status,
            "reason": reason,
        }

    stored = _update_lead_wake(directory, identity, change)
    assert stored is not None  # noqa: S101 - change() always writes.
    return stored


def corroborate_lead_wake(
    directory: Path, identity: str, generation: int, bound: str
) -> dict | None:
    """Settle a provisional registration against the parent-bound thread.

    A CAS on ``(generation, provisional)``: a newer registration or one
    already settled is left untouched and returned as it is.
    """

    def change(prior: dict | None) -> dict | None:
        if (
            prior is None
            or prior["generation"] != generation
            or prior["status"] != "provisional"
        ):
            return None
        if bound == prior["thread_id"]:
            return {**prior, "status": "active", "reason": ""}
        return {**prior, "status": "cleared", "reason": "thread_mismatch"}

    return _update_lead_wake(directory, identity, change)


def codex_lead_status(
    directory: Path,
    identity: str,
    host: Callable[[], tuple[int, str | None] | None],
) -> dict:
    """Report ``{status, generation, thread_verified}`` for ``session_info``.

    ``status`` is the stored one, or ``unregistered``, or ``stale_host`` when
    the registration belongs to another host incarnation. The host is looked
    up only when a registration exists.
    """
    registration = read_lead_wake(directory, identity)
    if registration is None:
        return {"status": "unregistered", "generation": 0, "thread_verified": False}
    status = registration["status"]
    current = host()
    if current is None or current != (
        registration["host_pid"],
        registration["host_create_token"],
    ):
        status = "stale_host"
    verified = (
        status == "active"
        and verify_codex_thread(registration["codex_home"], registration["thread_id"])[
            0
        ]
    )
    return {
        "status": status,
        "generation": registration["generation"],
        "thread_verified": verified,
    }


def codex_lead_notice(counts: Mapping[str, int], seq: int) -> str:
    """Return a body-free lead doorbell naming the full Codex tool.

    Sender names are reduced to ``[A-Za-z0-9_-]`` and the text avoids every
    ``cmd.exe`` metacharacter, so it is safe even through an npm shim.
    """
    from claude_teams.backends.codex import (  # noqa: PLC0415 - backend import cycle.
        codex_mcp_tool_name,
    )

    senders = " ".join(
        f"{re.sub(r'[^A-Za-z0-9_-]', '_', sender)}:{count}"
        for sender, count in sorted(counts.items())
    )
    return (
        f"[win-agent-teams wake #{seq}] {sum(counts.values())} unread messages "
        f"in your team inbox from {senders}. Best-effort notice without "
        f"content; call {codex_mcp_tool_name('read_messages')} to read them."
    )


def _default_lead_queue(thread_id: str, home: str, text: str) -> QueueOutcome:
    try:
        binary = _discover_codex()
    except Exception:
        return QueueOutcome(False)
    return codex_queue(binary, thread_id, home, text)


@dataclass(frozen=True)
class CodexLeadWake:
    """The Codex lead channel's injected facts (plan §2.6).

    ``host`` returns this MCP server's Codex host incarnation ``(pid,
    creation token)``, or ``None`` when the nearest host is not Codex.
    ``binding`` returns the parent-bound backend session of a spawned lead, or
    ``None``. ``queue`` runs ``codex queue``; ``verify`` checks the thread.
    """

    host: Callable[[], tuple[int, str | None] | None]
    binding: Callable[[str, str], str | None]
    queue: Callable[[str, str, str], QueueOutcome] = _default_lead_queue
    verify: Callable[[str, str], tuple[bool, str]] | None = None


class NativeWakeNotifier(threading.Thread):
    """A daemon that owns reader locks and observes only explicit session state."""

    def __init__(
        self,
        *,
        get_target: Callable[[], tuple[str, str] | None],
        session_dir: Callable[[str], Path],
        member_alive: Callable[[str, str], bool],
        channel: ClaudeChannel | None = None,
        post: Callable[[ClaudeChannel, str], PostResult] = post_claude_notice,
        clock: Callable[[], float] = time.monotonic,
        poll: float | None = None,
        codex_lead: CodexLeadWake | None = None,
        delivery: Any | None = None,
    ) -> None:
        """Inject state readers, never recovery or MCP tool calls.

        ``delivery`` is the child's ``DeliveryPoster`` (plan §2.3.4), a second
        target kind ticked here under its own gate and owner lock.
        """
        super().__init__(name="native-session-wake", daemon=True)
        self.get_target = get_target
        self.session_dir = session_dir
        self.member_alive = member_alive
        self.channel = channel or resolve_claude_channel(os.environ)
        self.post = post
        self.clock = clock
        self.poll = poll or _seconds("NATIVE_WAKE_POLL_SECONDS", 1.0)
        self.cfg = NoticeConfig(
            _seconds("NATIVE_WAKE_COALESCE_SECONDS", 2.0),
            _seconds("NATIVE_WAKE_RENOTIFY_SECONDS", 300.0),
        )
        self.targets: dict[tuple[str, str], _Target] = {}
        self.codex_lead = codex_lead
        # Keyed by (session, identity, generation, host pid, host token), so a
        # session switch, re-registration or new host never inherits state.
        self.codex_targets: dict[tuple[Any, ...], _Target] = {}
        self.delivery = delivery
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Scan on ticks and activation events; log errors only to stderr."""
        try:
            while not self._stop_event.is_set():
                try:
                    self.tick()
                except Exception:
                    _LOG.exception("Native wake scan failed")
                _activation.wait(self.poll)
        finally:
            self._release_targets()

    def tick(self) -> None:
        """Consume activation, then run each channel under its own gate."""
        activated = _activation.is_set()
        _activation.clear()
        if sys.platform == "win32":
            # Free parked pipe writes on the tick, not only on the next post.
            winpipe.reap_parked()
        self._tick_claude(activated)
        self._tick_codex_lead(activated)
        self._tick_delivery()

    def _tick_delivery(self) -> None:
        if self.delivery is None:
            return
        try:
            self.delivery.tick()
        except Exception as err:
            _LOG.warning("Delivery poster failed: %s", type(err).__name__)

    def _tick_claude(self, activated: bool) -> None:
        if self.channel.reason != "available" or not enabled("CLAUDE"):
            with _registry_lock:
                _members.clear()
            self._release_claude_targets()
            return
        lead = self.get_target()
        with _registry_lock:
            members = set(_members)
        live_members = {key for key in members if self.member_alive(*key)}
        with _registry_lock:
            _members.difference_update(members - live_members)
        wanted = live_members | ({lead} if lead else set())
        for key in self.targets.keys() - wanted:
            self.targets.pop(key).close()
        for key in wanted:
            target = self.targets.setdefault(key, _Target())
            try:
                self._check_target(key, target, lead, activated)
            except Exception as err:
                target.backoff.failed(self.clock())
                _LOG.warning("Native wake target failed: %s", type(err).__name__)

    def _check_target(
        self,
        key: tuple[str, str],
        target: _Target,
        lead: tuple[str, str] | None,
        activated: bool,
    ) -> None:
        now = self.clock()
        if now < target.backoff.until:
            return
        directory = self.session_dir(key[0])
        if target.handle is None:
            if now < target.acquire_after:
                return
            prefix = "native-wake-lead." if key == lead else "native-wake-member."
            handle = (directory / f"{prefix}{key[1]}.lock").open("a+b")
            try:
                owned = filelock.try_lock_handle(handle)
            except Exception:
                handle.close()
                raise
            if not owned:
                handle.close()
                target.acquire_after = now + 10.0
                return
            target.handle = handle
        notice = self._plan(target, directory, key[1], now, activated, key != lead)
        if notice is None:
            return
        # Session changes during a scan cannot send to a dropped lead target.
        if (
            target.handle is None
            or target.handle.closed
            or (key == lead and self.get_target() != key)
        ):
            return
        if key != lead and not self.member_alive(*key):
            target.close()
            return
        result = self.post(self.channel, notice.text)
        if result.ok:
            target.state.succeeded(target.snapshot, self.clock())
            target.backoff.reset()
            target.catchup = False
        else:
            target.backoff.failed(self.clock())

    def _plan(
        self,
        target: _Target,
        directory: Path,
        reader: str,
        now: float,
        activated: bool,
        member: bool,
    ) -> Notice | None:
        """Rescan when needed and propose a notice; catch-up skips coalescing."""
        signature = _signature(directory, reader)
        deadline = (
            target.state.first_new is not None
            and now >= target.state.first_new + self.cfg.coalesce
        ) or (
            target.state.last_success is not None
            and now >= target.state.last_success + self.cfg.renotify
        )
        if activated or signature != target.signature or target.catchup or deadline:
            target.snapshot = scan_inbox(directory, reader)
            target.signature = signature
        if target.catchup:
            target.state.notified = {s: r["cursor"] for s, r in target.snapshot.items()}
        cfg = NoticeConfig(
            0 if target.catchup else self.cfg.coalesce, self.cfg.renotify
        )
        target.state.observe(target.snapshot, now, cfg)
        notice = plan_notice(target.state, target.snapshot, now, cfg, member=member)
        if notice is None:
            target.catchup = False
        return notice

    def _codex_lead_key(self, lead: tuple[str, str] | None) -> tuple[Any, ...] | None:
        """Return the state key of a registration that may queue now, else None.

        Only an ``active`` registration made by this host incarnation
        qualifies. A ``provisional`` one is first settled against the
        parent-bound backend session, when that exists.
        """
        if lead is None or self.codex_lead is None:
            return None
        session, identity = lead
        directory = self.session_dir(session)
        registration = read_lead_wake(directory, identity)
        if registration is None or registration["status"] == "cleared":
            return None
        host = self.codex_lead.host()
        if host is None or host != (
            registration["host_pid"],
            registration["host_create_token"],
        ):
            return None
        if registration["status"] == "provisional":
            bound = self.codex_lead.binding(session, identity)
            if not bound:
                return None
            registration = corroborate_lead_wake(
                directory, identity, registration["generation"], bound
            )
            if registration is None or registration["status"] != "active":
                return None
        return (session, identity, registration["generation"], *host)

    def _tick_codex_lead(self, activated: bool) -> None:
        if self.codex_lead is None or not enabled("CODEX"):
            self._release_codex_targets()
            return
        key = self._codex_lead_key(self.get_target())
        for old in self.codex_targets.keys() - {key}:
            self.codex_targets.pop(old).close()
        if key is None:
            return
        target = self.codex_targets.setdefault(key, _Target())
        try:
            self._check_codex_lead(key, target, activated)
        except Exception as err:
            target.backoff.failed(self.clock())
            _LOG.warning("Codex lead wake failed: %s", type(err).__name__)

    def _check_codex_lead(  # noqa: PLR0911 - one return per gate.
        self, key: tuple[Any, ...], target: _Target, activated: bool
    ) -> None:
        assert self.codex_lead is not None  # noqa: S101 - gated by the caller.
        now = self.clock()
        if now < target.backoff.until:
            return
        session, identity = key[0], key[1]
        directory = self.session_dir(session)
        if target.handle is None:
            if now < target.acquire_after:
                return
            handle = (directory / f"native-wake-codex-lead.{identity}.lock").open("a+b")
            try:
                owned = filelock.try_lock_handle(handle)
            except Exception:
                handle.close()
                raise
            if not owned:
                handle.close()
                target.acquire_after = now + 10.0
                return
            target.handle = handle
        notice = self._plan(target, directory, identity, now, activated, member=False)
        if notice is None:
            return
        registration = read_lead_wake(directory, identity)
        if registration is None:
            return
        if not target.verified:
            verify = self.codex_lead.verify or verify_codex_thread
            target.verified, _ = verify(
                registration["codex_home"], registration["thread_id"]
            )
            if not target.verified:
                target.backoff.failed(self.clock())
                return
        # Revalidate target, generation and incarnation right before queueing.
        if (
            target.handle is None
            or target.handle.closed
            or self._codex_lead_key(self.get_target()) != key
        ):
            return
        current = read_lead_wake(directory, identity)
        if current is None or current["thread_id"] != registration["thread_id"]:
            return
        outcome = self.codex_lead.queue(
            current["thread_id"],
            current["codex_home"],
            codex_lead_notice(notice.counts, target.state.seq + 1),
        )
        if outcome.enqueued:
            target.state.succeeded(target.snapshot, self.clock())
            target.backoff.reset()
            target.catchup = False
        else:
            # Uncertain or failed: the notice may still arrive; retry spaced.
            target.backoff.failed(self.clock())

    def owns(self, key: tuple[str, str] | None) -> bool:
        """Report ownership without claiming notice delivery."""
        target = self.targets.get(key) if key else None
        handle = target.handle if target is not None else None
        return bool(handle and not handle.closed)

    def _release_claude_targets(self) -> None:
        for target in self.targets.values():
            target.close()
        self.targets.clear()

    def _release_codex_targets(self) -> None:
        for target in self.codex_targets.values():
            target.close()
        self.codex_targets.clear()

    def _release_targets(self) -> None:
        self._release_claude_targets()
        self._release_codex_targets()
        if self.delivery is not None:
            self.delivery.close()

    def close(self) -> None:
        """Stop the daemon and release its lifetime locks."""
        self._stop_event.set()
        _activation.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=6.0)
        else:
            self._release_targets()


def verify_codex_thread(home: str, thread_id: str) -> tuple[bool, str]:
    """Verify a live thread in its reported home with bounded read-only SQLite."""
    directory = Path(home)
    if not directory.is_absolute() or not directory.is_dir():
        return False, "home_missing"
    database = directory / "state_5.sqlite"
    if database.is_file():
        try:
            connection = sqlite3.connect(
                database.as_uri() + "?mode=ro", uri=True, timeout=0.5
            )
            try:
                row = connection.execute(
                    "SELECT rollout_path, archived FROM threads WHERE id = ?",
                    (thread_id,),
                ).fetchone()
            finally:
                connection.close()
            if row:
                path = Path(row[0])
                if row[1] != 0 or "archived_sessions" in path.parts:
                    return False, "archived"
                if path.is_relative_to(directory / "sessions"):
                    return True, ""
        except (sqlite3.DatabaseError, OSError, ValueError, TypeError):
            pass
    for pattern in (f"rollout-*-{thread_id}.jsonl", f"rollout-*-{thread_id}_*.jsonl"):
        if any(
            path.is_file() for path in (directory / "sessions").glob(f"*/*/*/{pattern}")
        ):
            return True, ""
    return False, "thread_not_found"


def _discover_codex() -> str:
    from claude_teams.backends.codex import (  # noqa: PLC0415 - backend import cycle.
        CodexBackend,
    )

    return CodexBackend().discover_binary()


_SUBMISSION_RE = re.compile(
    r"Queued message ([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


@dataclass(frozen=True)
class QueueOutcome:
    """One ``codex queue`` run. Only an exec failure proves nothing was queued.

    ``started`` means a codex process may have run. From then on, anything but
    exit 0 with a parsed submission id is uncertain: the turn may be queued
    durably (it survives the process and a reboot) even when the CLI reports a
    failure, so callers must never treat it as proof of non-delivery.
    """

    started: bool
    exit: int | None = None
    submission_id: str = ""
    timed_out: bool = False
    stderr_tail: str = field(default="", repr=False)

    @property
    def enqueued(self) -> bool:
        """Return whether the CLI confirmed a queued submission."""
        return self.exit == 0 and bool(self.submission_id)

    @property
    def provably_not_enqueued(self) -> bool:
        """Return whether no codex process ever started."""
        return not self.started


#: Largest command line handed to ``CreateProcess``, in UTF-16 code units and
#: including the terminating null. The API limit is 32 767; the margin covers
#: whatever a launcher adds (plan §2.9 R3-5).
WINDOWS_COMMAND_BUDGET = 32_000
#: Linux ``MAX_ARG_STRLEN``: one argument or environment string, null included.
POSIX_ARG_STRLEN_MAX = 128 * 1024
#: Assumed ``ARG_MAX`` when ``sysconf`` cannot say.
POSIX_ARG_MAX_FALLBACK = 128 * 1024
#: Kept free of ``ARG_MAX`` for the loader's own use (auxv, the exec path).
POSIX_ARG_MARGIN = 4 * 1024
_WINDOWS = os.name == "nt"


def _posix_arg_max() -> int:
    """Return ``sysconf(SC_ARG_MAX)``, or the fallback when it cannot say."""
    sysconf = getattr(os, "sysconf", None)  # absent on Windows
    if sysconf is None:
        return POSIX_ARG_MAX_FALLBACK
    try:
        value = int(sysconf("SC_ARG_MAX"))
    except (ValueError, OSError):
        return POSIX_ARG_MAX_FALLBACK
    return value if value > 0 else POSIX_ARG_MAX_FALLBACK


def command_fits(
    argv: list[str],
    env: Mapping[str, str] | None = None,
    *,
    windows: bool | None = None,
    arg_max: int | None = None,
) -> bool:
    """Return whether ``argv`` (and ``env``) can be launched on this platform.

    Checked before a launch whose failure would otherwise be ambiguous, so the
    caller can choose another route while nothing has run yet.

    - **Windows:** the quoted command line, as ``subprocess`` builds it,
      measured in UTF-16 code units (a non-BMP character is two) plus the
      terminating null, must be at most :data:`WINDOWS_COMMAND_BUDGET`.
    - **POSIX:** every argument and ``KEY=value`` string, encoded and
      null-terminated, must fit ``MAX_ARG_STRLEN``; all of them plus one
      pointer each (and the two terminating null pointers) must fit
      ``sysconf(SC_ARG_MAX)`` less :data:`POSIX_ARG_MARGIN`.
    """
    on_windows = _WINDOWS if windows is None else windows
    if on_windows:
        line = subprocess.list2cmdline(argv)
        units = len(line.encode("utf-16-le", "surrogatepass")) // 2
        return units + 1 <= WINDOWS_COMMAND_BUDGET
    values = os.environ if env is None else env
    strings = [*argv, *(f"{key}={value}" for key, value in values.items())]
    pointer = struct.calcsize("P")
    total = 2 * pointer
    for text in strings:
        size = len(os.fsencode(text)) + 1
        if size > POSIX_ARG_STRLEN_MAX:
            return False
        total += size + pointer
    limit = _posix_arg_max() if arg_max is None else arg_max
    return total <= limit - POSIX_ARG_MARGIN


def codex_queue_argv(binary: str, thread_id: str, message: str) -> list[str]:
    """Return the exact ``codex queue`` command :func:`codex_queue` launches."""
    return [binary, "queue", "--thread", thread_id, "--message", message]


def queue_environment(home: str) -> dict[str, str]:
    """Scrub identity/channel variables and pin ``CODEX_HOME`` for a queue run."""
    environ = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENT_")
        and key
        not in {
            "CLAUDE_CODE_MESSAGING_SOCKET",
            "CLAUDE_CODE_MESSAGING_TOKEN",
            "WIN_AGENT_TEAMS_SESSION_DIR",
        }
    }
    environ["CODEX_HOME"] = home
    return environ


def _outcome_from(returncode: int, stdout: str, stderr: str) -> QueueOutcome:
    match = _SUBMISSION_RE.search(stdout or "")
    return QueueOutcome(
        True,
        returncode,
        match[1] if match and returncode == 0 else "",
        stderr_tail=(stderr or "")[-200:],
    )


def codex_queue(  # noqa: PLR0911 - one return per staged outcome.
    binary: str,
    thread_id: str,
    home: str,
    message: str,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    runner: Callable[..., Any] | None = None,
    timeout: float | None = None,
) -> QueueOutcome:
    """Run ``codex queue`` with a real timeout; ``cwd`` is home, not CODEX_HOME.

    Process creation and communication are separate stages on purpose: only a
    failure to *construct* the process proves nothing was queued. Once a
    process object exists, every error (including an ``OSError`` from
    ``communicate``) is uncertain, because the turn may already be queued.
    ``runner`` is a legacy ``subprocess.run``-style seam; it cannot tell the
    stages apart, so none of its failures are ever reported as unstarted.
    """
    argv = codex_queue_argv(binary, thread_id, message)
    limit = timeout or _seconds("CODEX_QUEUE_TIMEOUT_SECONDS", 15.0)
    options: dict[str, Any] = {
        "env": queue_environment(home),
        "cwd": Path.home(),
        "stdin": subprocess.DEVNULL,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if runner is not None:
        try:
            completed = runner(
                argv, capture_output=True, timeout=limit, check=False, **options
            )
        except subprocess.TimeoutExpired:
            return QueueOutcome(True, timed_out=True)
        except (OSError, subprocess.SubprocessError):
            return QueueOutcome(True)
        return _outcome_from(
            completed.returncode,
            getattr(completed, "stdout", "") or "",
            getattr(completed, "stderr", "") or "",
        )
    try:
        process = popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    except (OSError, ValueError):
        # Construction failed: no codex process ever ran.
        return QueueOutcome(False)
    try:
        stdout, stderr = process.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _reap(process)
        return QueueOutcome(True, timed_out=True)
    except Exception:
        _reap(process)
        return QueueOutcome(True)
    return _outcome_from(process.returncode, stdout, stderr)


def _reap(process: Any) -> None:
    """Kill and collect a queue process without letting cleanup raise."""
    with contextlib.suppress(Exception):
        process.kill()
    with contextlib.suppress(Exception):
        process.communicate(timeout=5)


@dataclass
class _CodexState:
    generation: int
    notice: NoticeState
    backoff: Backoff = field(default_factory=Backoff)
    verified: bool = False


class CodexMemberWake:
    """Serialize per-member decide/queue/update outside the agents transaction."""

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        discover: Callable[[], str] = _discover_codex,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep process-local state only; construction has no external effects."""
        self.runner = runner
        self.discover = discover
        self.clock = clock
        self._lock = threading.Lock()
        self._member_locks: dict[tuple[str, str], Any] = {}
        self._states: dict[tuple[str, str], _CodexState] = {}

    def member_lock(self, key: tuple[str, str]) -> Any:
        """Return a member lock without acquiring it under a registry lock."""
        with self._lock:
            return self._member_locks.setdefault(key, threading.Lock())

    def wake(  # noqa: PLR0911 - explicit reported wake statuses.
        self,
        session_id: str,
        member: str,
        sender: str,
        snapshot: Callable[[], tuple[dict | None, dict[str, dict[str, int]]]],
    ) -> dict | None:
        """Revalidate generation and status immediately before the bounded queue."""
        key = (session_id, member)
        with self.member_lock(key):
            record, scan = snapshot()
            if record is None:
                return {"method": "codex_queue", "status": "coalesced"}
            registration = record.get("codex_wake") or {}
            if not registration.get("thread_id"):
                return None
            if not enabled("CODEX"):
                return {"method": "codex_queue", "status": "disabled"}
            if record.get("status") != "running":
                return {"method": "codex_queue", "status": "coalesced"}
            generation = registration["generation"]
            state = self._states.get(key)
            if state is None or state.generation != generation:
                state = _CodexState(
                    generation,
                    NoticeState(notified={s: r["cursor"] for s, r in scan.items()}),
                )
                self._states[key] = state
            now = self.clock()
            if now < state.backoff.until:
                return {"method": "codex_queue", "status": "backoff"}
            cfg = NoticeConfig(0, _seconds("NATIVE_WAKE_RENOTIFY_SECONDS", 300.0))
            state.notice.observe(scan, now, cfg)
            if plan_notice(state.notice, scan, now, cfg, member=True) is None:
                return {"method": "codex_queue", "status": "coalesced"}
            if not state.verified:
                state.verified, detail = verify_codex_thread(
                    registration["codex_home"], registration["thread_id"]
                )
                if not state.verified:
                    state.backoff.failed(self.clock())
                    return {
                        "method": "codex_queue",
                        "status": "unverified_thread",
                        "detail": detail,
                    }
            current, _ = snapshot()
            if current is None or current.get("status") != "running":
                return {"method": "codex_queue", "status": "coalesced"}
            if (current.get("codex_wake") or {}).get("generation") != generation:
                return {"method": "codex_queue", "status": "stale_registration"}
            result = self._queue(registration, sender, state.notice.seq + 1)
            if result["status"] == "queued":
                state.notice.succeeded(scan, self.clock())
                state.backoff.reset()
            else:
                state.backoff.failed(self.clock())
            return result

    def _queue(self, registration: dict, sender: str, seq: int) -> dict:
        from claude_teams.backends.codex import (  # noqa: PLC0415 - backend import cycle.
            CodexBackend,
            codex_mcp_tool_name,
        )

        result = {"method": "codex_queue", "status": "queued"}
        try:
            binary = self.discover()
        except Exception:
            return {**result, "status": "unavailable"}
        # No member-controlled free text or cmd.exe metacharacters reach the shim.
        safe_sender = re.sub(r"[^A-Za-z0-9_-]", "_", sender)
        external_key = CodexBackend._MCP_SERVER_NAME + "-external"
        notice = (
            f"win-agent-teams: wake {seq} new message from {safe_sender} "
            "in your member inbox - call "
            f"{codex_mcp_tool_name('external_read')} or "
            f"{codex_mcp_tool_name('external_read', server=external_key)} "
            "with your member_token using your configured MCP key"
        )
        outcome = codex_queue(
            binary,
            registration["thread_id"],
            registration["codex_home"],
            notice,
            runner=self.runner,
        )
        if outcome.timed_out:
            return {**result, "status": "timeout"}
        if outcome.exit is None:
            return {**result, "status": "failed"}
        if outcome.exit != 0:
            return {**result, "status": "failed", "detail": outcome.stderr_tail}
        return result
