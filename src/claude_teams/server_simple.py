"""Simplified MCP server for agent orchestration.

Watch command discovery assumes this distribution is installed in the normal
Python environment of the interpreter running the server. The generated
``python -m claude_teams.cli`` argv is not supported for embedded or frozen
hosts, and a coordinator cwd containing another ``claude_teams`` checkout can
shadow the installed module.
"""

import hashlib
import hmac
import json
import logging
import math
import os
import shlex
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastmcp import FastMCP

from claude_teams import delivery, delivery_store, hooks
from claude_teams.agent_output import (
    BINDING_LEGACY,
    CORRELATION_FIELD,
    PI_SESSION_DIR_FIELD,
    PROMPT_TRANSPORT_FIELD,
    PROMPT_TRANSPORT_SIDECAR,
    SPAWNED_BY_FIELD,
    SPAWNED_BY_SOURCE_FIELD,
    SPAWNED_BY_SOURCE_JOIN,
    SPAWNED_BY_SOURCE_SPAWN,
    BindingResult,
    _make_binder,
    classify_correlation,
    correlated_prompt,
    new_correlation_id,
    resolve_agent_binding,
)
from claude_teams.async_utils import run_blocking
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import (
    OWNERSHIP_NOT_OURS,
    OWNERSHIP_OURS,
    process_manager,
)
from claude_teams.backends.registry import registry
from claude_teams.delivery import (
    DELIVERY_DELIVERED,
    DELIVERY_FAILED,
    DELIVERY_UNCONFIRMED,
    SCAN_ABSENT,
    SCAN_FOUND,
    SCAN_INDETERMINATE,
    DeliveryOutcome,
    ReceiptScanner,
    confirm_delivery,
    delivered_prompt,
    new_delivery_nonce,
    prompt_file_name,
    remove_prompt_file,
    stale_prompt_files,
)
from claude_teams.delivery_store import (
    DELIVERIES_FILE_NAME,
    IDEMPOTENCY_CONFLICT,
    PHASE_PENDING,
    PHASE_SENT,
    PHASE_UNCONFIRMED,
    REASON_NOT_DELIVERED,
    STATUS_DELIVERED,
    STATUS_FAILED,
    STATUS_QUEUED,
    DeliveryStoreError,
    delivery_transaction,
    is_terminal,
    mark_phase,
    public_view,
    record_key,
    request_fingerprint,
    settle,
    validate_idempotency_key,
)
from claude_teams.filelock import FileLockTimeoutError, lock_handle, unlock_handle
from claude_teams.leases import (
    LEASES_FILE_NAME,
    LeaseStoreError,
    active_lease,
    finalize_lease,
    reconcile_lease,
    release_lease,
    reserve_lease,
)
from claude_teams.leases import (
    drop_agent as _drop_agent_lease,
)
from claude_teams.messaging import (
    load_inbox_cursors as _load_inbox_cursors,
)
from claude_teams.messaging import (
    purge_sender_from_inbox,
    read_inbox_by_sender,
    unread_sender_counts,
)
from claude_teams.messaging import (
    save_inbox_cursors as _save_inbox_cursors,
)

# Identity: env vars (works for Claude Code via --mcp-config)
# For Codex: the codex backend passes identity per-spawn via a `-c
# mcp_servers.<name>.env=...` override (see CodexBackend._mcp_identity_args),
# avoiding races on the shared ~/.codex/config.toml.
_AGENT_NAME: str = os.environ.get("AGENT_NAME", "").strip()
_AGENT_SESSION_ID: str = os.environ.get("AGENT_SESSION_ID", "").strip()
_AGENT_PARENT_NAME: str = os.environ.get("AGENT_PARENT_NAME", "").strip()

# The root coordinator's own inbox name. "team-lead" matches Claude Code's
# native teams convention, so a spawned Claude addressing "team-lead" lands in
# the right inbox out of the box. ("lead" and friends remain aliases below for
# backward compatibility.)
ROOT_LEAD_NAME: str = "team-lead"

# Sentinel identity used when a spawned subagent's MCP server starts with an
# empty ``AGENT_NAME`` (identity was clobbered — see ``_resolve_identity``). It
# can never be a valid inbox/lead name (contains a NUL), so any tool that keys
# off ``IDENTITY`` while unresolved fails to match a real inbox/session and, per
# the guards below, refuses outright rather than masquerading as the lead.
_UNRESOLVED_IDENTITY: str = "\x00unresolved-identity"


def _resolve_identity(environ: "os._Environ[str] | dict[str, str]") -> tuple[str, bool]:
    """Resolve ``(IDENTITY, unresolved)`` from an environment mapping.

    - A non-empty ``AGENT_NAME`` is always authoritative → ``(name, False)``.
    - Empty ``AGENT_NAME`` **with** a spawned-subagent signal
      (``WIN_AGENT_TEAMS_SESSION_DIR``, set only by ``PiBackend.build_env`` and
      inherited by children, never carried in any ``mcp.json`` ``env`` block) →
      identity is UNRESOLVED: ``(_UNRESOLVED_IDENTITY, True)``. This means the
      literal ``AGENT_*`` values were clobbered (e.g. an ``AGENT_*`` env block in
      a project ``.mcp.json`` / ``.pi/mcp.json`` shallow-overrides the inherited
      values). We refuse to default to ``team-lead`` and hijack the lead.
    - Empty ``AGENT_NAME`` with **no** signal is a legitimate human-launched root
      lead → ``(ROOT_LEAD_NAME, False)``.
    """
    name = environ.get("AGENT_NAME", "").strip()
    if name:
        return name, False
    spawned_signal = bool(environ.get("WIN_AGENT_TEAMS_SESSION_DIR", "").strip())
    if spawned_signal:
        return _UNRESOLVED_IDENTITY, True
    return ROOT_LEAD_NAME, False


IDENTITY, _IDENTITY_UNRESOLVED = _resolve_identity(os.environ)

# Names a subagent might reasonably use to mean "whoever spawned me". All of
# these resolve to the lead/parent so a message is never lost to a typo'd
# recipient. Compared case-insensitively.
_LEAD_ALIASES: frozenset[str] = frozenset(
    {
        "",
        "team-lead",
        "lead",
        "orchestrator",
        "parent",
        "boss",
        "manager",
        "up",
        "supervisor",
    }
)

_SESSION_BASE = Path.home() / ".claude" / "agent-sessions"
_TEAMS_BASE = Path.home() / ".claude" / "teams"
_SESSION_META_NAME = "session.json"
_BINDINGS_DIR_NAME = "bindings"
_AGENTS_LOCK_NAME = "agents.lock"
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.05
_LOCK_SIZE = 1
_FOLLOW_UP_IDLE_SECONDS = 60.0
_CLEANUP_STAMP_NAME = ".last-cleanup"
_RETENTION_DAYS_DEFAULT = 30.0
_NO_AUTOADOPT_ENV = "WIN_AGENT_TEAMS_NO_AUTOADOPT"
_TERMINAL_STATUSES: frozenset[str] = frozenset({"killed"})
_RETENTION_DAYS_ENV = "WIN_AGENT_TEAMS_RETENTION_DAYS"
_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60.0
_JOIN_TICKETS_NAME = "join-tickets.json"
_JOIN_TICKET_TTL_SECONDS_DEFAULT = 24 * 60 * 60.0
_JOIN_TICKET_RETENTION_SECONDS_DEFAULT = 7 * 24 * 60 * 60.0
_MEMBER_TOKEN_FIELD_COUNT = 3
_MEMBER_SECRET_LENGTH = 64
CREDENTIAL_FIELDS: frozenset[str] = frozenset({"member_token_digest"})

# A4/A4b timing. Held as module attributes rather than literals so tests can
# inject a clock and drive the poll loop deterministically: the repo already
# carries one flaky wall-clock watcher test and confirmation is far more
# timing-dense than that.
_DELIVERY_POLL_SECONDS = 0.25
#: B0 — **the** delivery bound. ONE total budget covering the wait for a busy
#: target, lease acquisition, the resume, and confirmation. Deliberately not a
#: per-step timeout: a per-step design lets a caller spend three budgets in one
#: call and makes "which of the two outcomes did I get" unanswerable.
#:
#: It is a documented server-side constant rather than an assumption about the
#: client's deadline — the server cannot know that — so it is echoed back as
#: ``call_budget_s`` on every ``follow_up_agent`` result. Keep it below the
#: smallest realistic MCP client timeout.
_DELIVERY_CALL_BUDGET_SECONDS = 45.0
#: How long after a child is proven dead an ``unconfirmed`` attempt waits for a
#: final transcript flush before it may be settled ``failed``. Buffered writes
#: routinely land after the process exits, and settling early would report a
#: delivered message as failed.
_UNCONFIRMED_FLUSH_GRACE_SECONDS = 30.0
#: How long a lease is nominally valid. Purely informational: expiry alone
#: NEVER reclaims a lease — see ``leases.reconcile_lease``.
_LEASE_TTL_SECONDS = 120.0
#: Age after which an orphaned prompt sidecar may be garbage-collected. Long
#: on purpose: deleting a file a still-running CLI may yet read is worse than
#: leaving a few kilobytes behind.
_PROMPT_GC_AGE_SECONDS = 24 * 60 * 60.0

_delivery_clock = time.monotonic
_delivery_sleep = time.sleep

#: Agent-record field holding an attempt that resumed but was never confirmed.
#: R6's "live uncertainty": it must survive the call so a retry reconciles it
#: instead of delivering the same instruction a second time.
PENDING_DELIVERY_FIELD = "pending_delivery"
_CLAUDE_PROMPT_FILE_CHARS: frozenset[str] = frozenset({"'", '"', "\n", "\r"})
logger = logging.getLogger(__name__)

if _IDENTITY_UNRESOLVED:
    logger.error(
        "win-agent-teams identity is UNRESOLVED: AGENT_NAME is empty but this "
        "process was spawned as a subagent (WIN_AGENT_TEAMS_SESSION_DIR is set). "
        "The literal AGENT_* identity was clobbered -- most likely an AGENT_* env "
        "block in a project .mcp.json / .pi/mcp.json shallow-overrides the "
        "inherited values. Team tools (send_message/read_messages/resume_session) "
        "will refuse until identity is fixed; never put AGENT_* in a project MCP "
        "config env block."
    )

# Outcome of the most recent session recovery attempt, surfaced to the lead as
# a nudge on dict-returning tools: either ``{"adopted_session": {...}}`` (a
# single-lead prior session was auto-adopted — one-shot) or
# ``{"recoverable_sessions": [...], "recovery_hint": "..."}`` (ambiguous /
# multi-lead: the lead must pick one via ``resume_session``). Empty when the
# session resolved cleanly.
_pending_recovery: dict = {}


def _idle_seconds() -> float:
    """Return the activity-fallback idle threshold, env-overridable."""
    raw = os.environ.get("WIN_AGENT_TEAMS_IDLE_SECONDS", "").strip()
    if not raw:
        return _FOLLOW_UP_IDLE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _FOLLOW_UP_IDLE_SECONDS


STALL_SECONDS: float = 300.0


def _stall_seconds() -> float:
    """Return the stall threshold, overridable via ``WIN_AGENT_TEAMS_STALL_SECONDS``."""
    raw = os.environ.get("WIN_AGENT_TEAMS_STALL_SECONDS", "").strip()
    if not raw:
        return STALL_SECONDS
    try:
        return float(raw)
    except ValueError:
        return STALL_SECONDS


def _heartbeat_fields(
    *, alive: bool, state: str, last_activity_ts: float | None, now: float | None = None
) -> tuple[float | None, bool]:
    """Return ``(heartbeat_age_s, stalled)`` derived purely from disk-backed signals.

    ``heartbeat_age_s`` is ``now - last_activity_ts`` (``None`` when no
    activity signal is available at all). ``stalled`` is ``True`` only when
    the agent is alive, its resolved ``state`` is neither ``"waiting"`` nor
    ``"dead"``, and ``heartbeat_age_s`` exceeds ``_stall_seconds()``.
    """
    if last_activity_ts is None:
        return None, False
    current = time.time() if now is None else now
    heartbeat_age_s = current - last_activity_ts
    stalled = (
        alive
        and state not in {"waiting", "dead"}
        and heartbeat_age_s > _stall_seconds()
    )
    return heartbeat_age_s, stalled


_VALID_MARKER_STATES: frozenset[str] = frozenset({"running", "waiting"})


def _resolve_agent_state(
    *,
    alive: bool,
    marker: dict | None,
    last_activity_at: float | None,
    now: float | None = None,
) -> str:
    """Resolve an agent's coarse-grained state.

    Precedence: liveness gates everything (a dead process is always
    ``"dead"``, even with a stale ``"running"`` marker); then a hook-written
    marker's state is used when it is one of the public enum values
    ``{"running", "waiting"}``; any other marker state (missing, malformed,
    or semantically invalid, e.g. ``"paused"``) is treated as absent and
    falls back to an activity-recency heuristic (``"running"`` vs
    ``"idle"``).
    """
    if not alive:
        return "dead"
    if marker is not None:
        state = marker.get("state")
        if isinstance(state, str) and state in _VALID_MARKER_STATES:
            return state
    if last_activity_at is None:
        return "idle"
    current = time.time() if now is None else now
    if current - last_activity_at < _idle_seconds():
        return "running"
    return "idle"


#: Retained under its historical name; the implementation now lives in
#: :mod:`claude_teams.filelock`, shared with the delivery status store.
AgentsFileLockTimeoutError = FileLockTimeoutError


mcp = FastMCP(
    name="win-agent-teams",
    instructions="Spawn and communicate with Claude Code and Codex agents.",
)


def _register_tool(*, external: bool = False):
    """Register one tool unless the import-time external-only gate excludes it."""

    def _decorate(fn):
        external_only = os.environ.get(
            "WIN_AGENT_TEAMS_EXTERNAL_ONLY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not external_only or external:
            return mcp.tool()(fn)
        return fn

    return _decorate


# Module-level session state for the lead role
_session_id: str = _AGENT_SESSION_ID or ""


def _session_dir(session_id: str) -> Path:
    return _SESSION_BASE / session_id


def _agents_file(session_id: str) -> Path:
    return _session_dir(session_id) / "agents.json"


def _agents_lock_file(session_id: str) -> Path:
    return _session_dir(session_id) / _AGENTS_LOCK_NAME


def _session_meta_file(session_id: str) -> Path:
    return _session_dir(session_id) / _SESSION_META_NAME


def _inbox_file(session_id: str, name: str) -> Path:
    return _session_dir(session_id) / f"inbox-{name}.jsonl"


def _inbox_cursor_file(session_id: str, name: str) -> Path:
    """Return the per-sender unread-counter sidecar beside an inbox."""
    return _session_dir(session_id) / f"inbox-{name}.pos.json"


def _state_marker_file(session_id: str, name: str) -> Path:
    """Return the hook-written state marker path for an agent (see hooks.py)."""
    return _session_dir(session_id) / f"state-{name}.json"


def _join_tickets_file(session_id: str) -> Path:
    """Return the lead-issued external-member ticket store path."""
    return _session_dir(session_id) / _JOIN_TICKETS_NAME


def _prompts_dir(session_id: str) -> Path:
    """Return the directory holding an agent's prompt sidecars."""
    return _session_dir(session_id) / "prompts"


def _prompt_file(session_id: str, name: str) -> Path:
    """Return the legacy deterministic prompt sidecar path.

    Superseded by :func:`_attempt_prompt_file` (A5). Retained only so cleanup
    still removes files written by an older version of this server.
    """
    return _prompts_dir(session_id) / f"{name}.prompt.txt"


def _attempt_prompt_file(session_id: str, name: str, nonce: str) -> Path:
    """Return this call's unique prompt sidecar path (A5).

    One path per call, keyed on the attempt's nonce, so two concurrent calls to
    the same agent can never overwrite each other's prompt — and so a file left
    behind can be attributed back to the attempt that wrote it.
    """
    return _prompts_dir(session_id) / prompt_file_name(name, nonce)


def _leases_file(session_id: str) -> Path:
    """Return the per-session operation-lease store (A4b).

    Deliberately a separate file from ``agents.json``: the registry is written
    with a plain overwrite, so a crash mid-write could otherwise destroy the
    registry *and* the lease at once.
    """
    return _session_dir(session_id) / LEASES_FILE_NAME


def _deliveries_file(session_id: str) -> Path:
    """Return the per-session delivery/audit store (B1, R4).

    A third file beside ``agents.json`` and ``operation-leases.json``, and
    deliberately **not** part of either. Unlike the inbox — which kill purges —
    nothing ever deletes rows here: the sender's whole reason for querying is
    that a settled outcome outlives the target.
    """
    return _session_dir(session_id) / DELIVERIES_FILE_NAME


def _pi_session_dir(session_id: str, name: str) -> Path:
    """Return the per-agent pi session storage dir (see PiBackend/read_pi_output)."""
    return _session_dir(session_id) / "pi-sessions" / name


def _read_state_marker(session_id: str, name: str) -> dict | None:
    """Read an agent's hook-written state marker, tolerating a missing/corrupt file."""
    path = _state_marker_file(session_id, name)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_state_marker(
    session_id: str, name: str, *, state: str, event: str
) -> None:
    """Atomically write an external member's lifecycle marker."""
    path = _state_marker_file(session_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = {"state": state, "event": event, "ts": time.time()}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(marker), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _marker_schema_valid(marker: dict | None) -> bool:
    """Return whether a marker satisfies the public state-marker schema."""
    if marker is None or marker.get("state") not in _VALID_MARKER_STATES:
        return False
    if not isinstance(marker.get("event"), str):
        return False
    ts = marker.get("ts")
    return isinstance(ts, int | float) and not isinstance(ts, bool)


def _ensure_joined_marker(session_id: str, name: str) -> bool:
    """Ensure a valid joined marker, preserving any valid newer marker."""
    marker = _read_state_marker(session_id, name)
    if _marker_schema_valid(marker):
        return True
    try:
        _write_state_marker(session_id, name, state="running", event="joined")
    except OSError:
        return False
    return True


# In-process serialization of read_messages per inbox name. FastMCP runs the
# blocking read body on a thread pool, so two concurrent tool calls for the same
# inbox in this process must be serialized while they load/advance/save the
# counter. This is in-process only; there is deliberately no cross-process file
# lock (the owning reader identity is the single writer of its counter).
_inbox_locks: dict[str, threading.Lock] = {}
_inbox_locks_guard = threading.Lock()


def _inbox_lock(name: str) -> threading.Lock:
    with _inbox_locks_guard:
        lock = _inbox_locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _inbox_locks[name] = lock
        return lock


def _bindings_dir() -> Path:
    return _SESSION_BASE / _BINDINGS_DIR_NAME


def _binding_key() -> str:
    """Return a stable key for this MCP parent/workspace identity."""
    parent_id = os.environ.get("WIN_AGENT_TEAMS_PARENT_ID", "").strip()
    if not parent_id:
        parent_id = str(os.getppid())
    cwd = str(Path.cwd().resolve())
    return f"identity={IDENTITY}\nparent={parent_id}\ncwd={cwd}"


def _binding_file() -> Path:
    digest = hashlib.sha256(_binding_key().encode("utf-8")).hexdigest()
    return _bindings_dir() / f"{digest}.json"


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _lock_file(handle) -> None:
    """Take the registry's cross-process advisory lock.

    Delegates to :mod:`claude_teams.filelock` so the delivery status store
    takes the *same* lock model rather than a second, subtly different one.
    """
    lock_handle(handle, timeout_s=_LOCK_TIMEOUT_SECONDS)


def _unlock_file(handle) -> None:
    unlock_handle(handle)


@contextmanager
def _agents_file_lock(session_id: str) -> Iterator[None]:
    path = _agents_lock_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


def _load_agents_unlocked(session_id: str) -> list[dict]:
    path = _agents_file(session_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_agents_unlocked(session_id: str, agents: list[dict]) -> None:
    _agents_file(session_id).write_text(json.dumps(agents, indent=2), encoding="utf-8")


def _load_agents(session_id: str) -> list[dict]:
    with _agents_file_lock(session_id):
        return _load_agents_unlocked(session_id)


def _save_agents(session_id: str, agents: list[dict]) -> None:
    with _agents_file_lock(session_id):
        _save_agents_unlocked(session_id, agents)


@contextmanager
def _agents_transaction(session_id: str) -> Iterator[list[dict]]:
    with _agents_file_lock(session_id):
        agents = _load_agents_unlocked(session_id)
        yield agents


def _save_agents_transaction(session_id: str, agents: list[dict]) -> None:
    _save_agents_unlocked(session_id, agents)


def _strict_positive_seconds(env_name: str, default: float) -> float:
    """Return a finite positive env override, otherwise ``default``."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _join_ticket_ttl_seconds() -> float:
    """Return the strict positive join-ticket TTL."""
    return _strict_positive_seconds(
        "WIN_AGENT_TEAMS_JOIN_TICKET_TTL_SECONDS",
        _JOIN_TICKET_TTL_SECONDS_DEFAULT,
    )


def _join_ticket_retention_seconds() -> float:
    """Return the strict positive consumed/expired ticket retention."""
    return _strict_positive_seconds(
        "WIN_AGENT_TEAMS_JOIN_TICKET_RETENTION_SECONDS",
        _JOIN_TICKET_RETENTION_SECONDS_DEFAULT,
    )


def _load_join_tickets_unlocked(session_id: str) -> list[dict]:
    """Load the ticket array while the caller holds the agents-file lock."""
    path = _join_tickets_file(session_id)
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _ticket_retained(ticket: dict, *, now: float, retention: float) -> bool:
    """Return whether an audit ticket remains inside its retention window."""
    status = ticket.get("status")
    if status == "used":
        anchor = ticket.get("used_at")
    elif status == "open":
        expires_at = ticket.get("expires_at")
        if isinstance(expires_at, int | float) and not isinstance(expires_at, bool):
            if float(expires_at) >= now:
                return True
            anchor = expires_at
        else:
            return True
    else:
        # Unknown states are retained so reconciliation can fail closed.
        return True
    if not isinstance(anchor, int | float) or isinstance(anchor, bool):
        return True
    return now - float(anchor) <= retention


def _save_join_tickets_unlocked(
    session_id: str, tickets: list[dict], *, now: float | None = None
) -> None:
    """Prune retained audit rows and atomically save the ticket array."""
    current = time.time() if now is None else now
    retention = _join_ticket_retention_seconds()
    retained = [
        ticket
        for ticket in tickets
        if _ticket_retained(ticket, now=current, retention=retention)
    ]
    tickets[:] = retained
    path = _join_tickets_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(retained, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _unique_agent_name(requested_name: str, agents: list[dict]) -> str:
    """Return an agent name that does not collide within this session."""
    requested = requested_name.strip()
    candidate = requested or f"agent-{len(agents) + 1}"
    existing = {str(agent.get("name") or "") for agent in agents}
    if candidate not in existing:
        return candidate

    counter = 2
    while True:
        suffix = f"-{counter}"
        name = f"{candidate[: 64 - len(suffix)]}{suffix}"
        if name not in existing:
            return name
        counter += 1


def _ticket_name_reserved(ticket: dict, *, now: float) -> bool:
    """Return whether an open and unexpired ticket reserves its member name."""
    expires_at = ticket.get("expires_at")
    return (
        ticket.get("status") == "open"
        and isinstance(expires_at, int | float)
        and not isinstance(expires_at, bool)
        and float(expires_at) > now
    )


def _unique_reserved_name(
    requested_name: str, agents: list[dict], tickets: list[dict], *, now: float
) -> str:
    """De-duplicate against registry records and live ticket reservations."""
    reservations = [
        {"name": ticket.get("name")}
        for ticket in tickets
        if _ticket_name_reserved(ticket, now=now)
    ]
    return _unique_agent_name(requested_name, [*agents, *reservations])


def _markdown_fence(text: str) -> str:
    """Return a backtick fence longer than every run present in ``text``."""
    longest = 0
    current = 0
    for character in text:
        if character == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(3, longest + 1)


def _build_join_prompt(
    *, session_id: str, token: str, name: str, parent: str, note: str
) -> str:
    """Build the deterministic paste-ready external-member join prompt."""
    fence = _markdown_fence(note)
    return (
        f"You are joining {parent}'s win-agent-teams session as {name!r}.\n\n"
        f"Role note:\n{fence}\n{note}\n{fence}\n\n"
        "Join protocol:\n"
        "1. Call this tool first, with these literal values:\n"
        f"   join_team(session_id={session_id!r}, token={token!r})\n"
        "2. You must save the returned member_token in this conversation. It is the "
        "credential for every external member call, including after an MCP "
        "server restart.\n"
        "3. Poll external_read(member_token=...) for work. Reply upstream with "
        "external_send(member_token=..., text=...). Do not use ambient team "
        "tools from this external-member conversation.\n"
        f"4. Optional watcher: win-agent-teams watch {_session_dir(session_id)} "
        f"--reader {name}\n"
        "5. When finished permanently, call leave_team(member_token=...).\n"
    )


def _member_secret(ticket_id: object, token: object) -> str:
    """Derive the stable external-member secret from a join ticket."""
    return hashlib.sha256(f"wam-member:{ticket_id}:{token}".encode()).hexdigest()


def _member_digest(secret: str) -> str:
    """Return the digest persisted in ``agents.json`` for a member secret."""
    return hashlib.sha256(secret.encode()).hexdigest()


def _validate_join_session_id(session_id: str) -> tuple[str, dict | None]:
    """Validate a join target exactly as the explicit resume path does."""
    sid = session_id.strip()
    if not sid:
        return sid, {"success": False, "reason": "session_id_required"}
    try:
        uuid.UUID(sid)
    except (ValueError, AttributeError):
        return sid, {
            "success": False,
            "session_id": sid,
            "reason": "invalid_session_id",
        }
    session_dir = _session_dir(sid)
    if (
        session_dir.resolve().parent != _SESSION_BASE.resolve()
        or not _agents_file(sid).exists()
    ):
        return sid, {
            "success": False,
            "session_id": sid,
            "reason": "session_not_found",
        }
    return sid, None


def _external_record_matches(
    record: dict, ticket: dict, session_id: str, expected_digest: str
) -> bool:
    """Validate immutable join-record fields before returning membership."""
    expected = {
        "name": ticket.get("name"),
        "parent": ticket.get("parent"),
        "member_token_digest": expected_digest,
        "backend": "external",
        "session_id": session_id,
        SPAWNED_BY_FIELD: ticket.get("parent"),
        SPAWNED_BY_SOURCE_FIELD: SPAWNED_BY_SOURCE_JOIN,
    }
    return all(record.get(field) == value for field, value in expected.items())


def _external_membership_result(
    session_id: str, ticket: dict, secret: str
) -> dict:
    """Build the stable successful ``join_team`` response."""
    name = str(ticket.get("name") or "")
    parent = str(ticket.get("parent") or "")
    session_dir = _session_dir(session_id)
    return {
        "success": True,
        "name": name,
        "parent": parent,
        "session_id": session_id,
        "member_token": f"wam1:{session_id}:{secret}",
        "inbox_path": str(_inbox_file(session_id, name)),
        "state_marker_path": str(_state_marker_file(session_id, name)),
        "watch_argv": _watch_argv(session_dir, reader=name),
        "instructions": (
            "Save member_token. Poll external_read for work, answer with "
            "external_send, and call leave_team only when leaving permanently."
        ),
    }


def _parse_member_token(  # noqa: PLR0911 - exact grammar rejects each deviation.
    member_token: object,
) -> tuple[str, str] | None:
    """Parse the exact external member-token grammar without raising."""
    if not isinstance(member_token, str):
        return None
    fields = member_token.split(":")
    if len(fields) != _MEMBER_TOKEN_FIELD_COUNT:
        return None
    version, session_id, secret = fields
    if version != "wam1":
        return None
    try:
        canonical = str(uuid.UUID(session_id))
    except (ValueError, AttributeError):
        return None
    if canonical != session_id:
        return None
    if (
        len(secret) != _MEMBER_SECRET_LENGTH
        or not secret.isascii()
        or any(character not in "0123456789abcdef" for character in secret)
    ):
        return None
    return session_id, secret


@contextmanager
def _member_operation(
    member_token: object, *, allow_left: bool = False
) -> Iterator[tuple[dict | None, list[dict] | None, str, dict | None]]:
    """Resolve a member and hold its registry lock through caller side effects."""
    parsed = _parse_member_token(member_token)
    if parsed is None:
        yield None, None, "", {
            "success": False,
            "reason": "invalid_member_token",
        }
        return
    session_id, secret = parsed
    if not _agents_file(session_id).exists():
        yield None, None, session_id, {
            "success": False,
            "reason": "session_not_found",
        }
        return
    computed = _member_digest(secret).encode()
    with _agents_file_lock(session_id):
        agents = _load_agents_unlocked(session_id)
        record = None
        for candidate in agents:
            if candidate.get("backend") != "external":
                continue
            stored = candidate.get("member_token_digest")
            if not isinstance(stored, str):
                continue
            try:
                stored_bytes = stored.encode()
            except UnicodeError:
                continue
            if hmac.compare_digest(computed, stored_bytes):
                record = candidate
                break
        if record is None:
            yield None, agents, session_id, {
                "success": False,
                "reason": "membership_revoked",
            }
            return
        status = record.get("status")
        if status == "left" and not allow_left:
            yield None, agents, session_id, {
                "success": False,
                "reason": "membership_revoked",
                "detail": "left",
            }
            return
        if status not in {"running", "left"}:
            yield None, agents, session_id, {
                "success": False,
                "reason": "membership_revoked",
            }
            return
        yield record, agents, session_id, None


def _retention_days() -> float:
    """Return the file-retention window in days (env-overridable, fail-safe).

    Invalid, zero, or negative ``WIN_AGENT_TEAMS_RETENTION_DAYS`` values fall
    back to the 30-day default rather than triggering mass deletion.
    """
    raw = os.environ.get(_RETENTION_DAYS_ENV, "").strip()
    if not raw:
        return _RETENTION_DAYS_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        return _RETENTION_DAYS_DEFAULT
    return value if value > 0 else _RETENTION_DAYS_DEFAULT


def _ensure_lead_token(session_id: str) -> str:
    """Return the session's stable lead recovery token, creating it if absent."""
    meta = _read_json_object(_session_meta_file(session_id))
    token = meta.get("lead_token")
    if isinstance(token, str) and token:
        return token
    return uuid.uuid4().hex


def _persist_session_binding(session_id: str) -> None:
    """Bind this parent/workspace to a session id for MCP restart recovery.

    Preserves the session's ``lead_token`` (the authoritative, re-presentable
    recovery token) across rebindings, and prunes any *other* binding files
    that reference the same session so a single lead keeps exactly one binding
    across repeated restarts (keeping gated auto-adopt automatic).
    """
    now = datetime.now(UTC).isoformat()
    key = _binding_key()
    lead_token = _ensure_lead_token(session_id)
    meta = {
        "session_id": session_id,
        "binding_key": key,
        "identity": IDENTITY,
        "cwd": str(Path.cwd().resolve()),
        "parent_id": os.environ.get("WIN_AGENT_TEAMS_PARENT_ID", "").strip()
        or str(os.getppid()),
        "lead_token": lead_token,
        "updated_at": now,
    }
    _session_meta_file(session_id).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    _bindings_dir().mkdir(parents=True, exist_ok=True)
    _binding_file().write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _prune_superseded_bindings(session_id)


def _prune_superseded_bindings(session_id: str) -> None:
    """Delete stale binding files pointing at ``session_id`` (keep the current)."""
    keep = _binding_file()
    bindings = _bindings_dir()
    if not bindings.is_dir():
        return
    for path in bindings.iterdir():
        if not path.is_file() or path.suffix != ".json" or path == keep:
            continue
        try:
            meta = _read_json_object(path)
        except OSError:
            continue
        if meta.get("session_id") == session_id:
            with suppress(OSError):
                path.unlink(missing_ok=True)


def _iter_binding_metas() -> Iterator[tuple[Path, dict, float]]:
    """Yield ``(path, meta, mtime)`` for each readable binding file."""
    bindings = _bindings_dir()
    if not bindings.is_dir():
        return
    for path in bindings.iterdir():
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            meta = _read_json_object(path)
            mtime = path.stat().st_mtime
        except OSError:
            continue
        yield path, meta, mtime


def _candidate_sessions() -> list[dict]:
    """Recoverable prior sessions for this ``identity+cwd`` within retention.

    A candidate is a binding whose ``agents.json`` still holds ≥1 (non-killed,
    hence non-terminal) agent — kill removes records, so a non-empty registry
    means resumable agents. Deduped by ``session_id`` (newest binding wins) and
    sorted newest-first. Tolerates unreadable/corrupt binding or registry files
    per candidate rather than failing recovery globally.
    """
    identity = IDENTITY
    cwd = str(Path.cwd().resolve())
    cutoff = time.time() - _retention_days() * 86400.0
    by_session: dict[str, dict] = {}
    for _path, meta, mtime in _iter_binding_metas():
        if meta.get("identity") != identity or meta.get("cwd") != cwd:
            continue
        if mtime < cutoff:
            continue
        session_id = meta.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        try:
            agents = _load_agents_unlocked(session_id)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(agents, list):
            continue
        resumable = _non_terminal_agents(agents)
        if not resumable:
            continue
        existing = by_session.get(session_id)
        if existing is None or mtime > existing["_mtime"]:
            by_session[session_id] = {
                "session_id": session_id,
                "agent_count": len(resumable),
                "last_activity": meta.get("updated_at"),
                "_mtime": mtime,
            }
    candidates = sorted(by_session.values(), key=lambda c: c["_mtime"], reverse=True)
    for c in candidates:
        c.pop("_mtime", None)
    return candidates


def _non_terminal_agents(agents: list) -> list[dict]:
    """Return only the resumable (non-terminal) agent records.

    Terminal records (legacy ``status="killed"`` left by the pre-R5 kill, which
    did not remove records) are excluded so a session holding only killed
    agents is NOT treated as recoverable and silently resumed.
    """
    return [
        a
        for a in agents
        if isinstance(a, dict) and a.get("status") not in _TERMINAL_STATUSES
    ]


def _distinct_binding_sessions() -> set[str]:
    """Distinct ``session_id``s bound to this ``identity+cwd`` within retention.

    Used as the multi-lead signal: two or more distinct sessions here means
    another lead has operated in this workspace, so single-candidate
    auto-adopt is disabled and the lead must disambiguate via resume_session.
    """
    identity = IDENTITY
    cwd = str(Path.cwd().resolve())
    cutoff = time.time() - _retention_days() * 86400.0
    sessions: set[str] = set()
    for _path, meta, mtime in _iter_binding_metas():
        if meta.get("identity") != identity or meta.get("cwd") != cwd:
            continue
        if mtime < cutoff:
            continue
        session_id = meta.get("session_id")
        if isinstance(session_id, str) and session_id:
            sessions.add(session_id)
    return sessions


def _autoadopt_enabled() -> bool:
    return os.environ.get(_NO_AUTOADOPT_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_autoadoptable_agent(session_id: str) -> bool:
    """Whether a candidate has a live non-external record for silent adoption."""
    try:
        agents = _load_agents_unlocked(session_id)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return any(
        agent.get("backend") != "external"
        for agent in _non_terminal_agents(agents)
    )


def _require_resolved_identity() -> dict | None:
    """Return a structured refusal when identity is unresolved, else ``None``.

    Applied to every identity-bearing tool so a mis-configured spawned subagent
    (empty ``AGENT_NAME`` + subagent signal) refuses loudly instead of operating
    as ``team-lead`` (spoofing ``from:``, consuming the lead's inbox, or adopting
    the lead's session). In the healthy path (Claude/Codex always carry a literal
    name; a correctly-configured pi worker gets one via the per-agent
    ``--mcp-config`` file) this never fires.
    """
    if _IDENTITY_UNRESOLVED:
        return {
            "success": False,
            "reason": "identity_unresolved",
            "hint": (
                "This agent's AGENT_NAME is empty but it was spawned as a "
                "subagent, so its MCP identity was clobbered (likely an AGENT_* "
                "env block in a project .mcp.json / .pi/mcp.json). Team tools are "
                "disabled to avoid hijacking the lead. Report this via your final "
                "output and ask the lead to fix the MCP config."
            ),
        }
    return None


def _recover_session_id() -> str:
    """Recover the persisted lead session for this MCP parent/workspace.

    Precedence: (1) ``AGENT_SESSION_ID`` env (subagent path); (2) exact
    binding-key match (fast path); (3) cwd+identity fallback. The fallback
    auto-adopts a single candidate ONLY when this workspace has a single-lead
    history (one distinct bound session); otherwise it leaves the session
    unresolved and records candidates for the recovery nudge. Sets
    ``_pending_recovery`` as a side effect.
    """
    global _pending_recovery  # noqa: PLW0603 - recovery nudge state.
    _pending_recovery = {}
    if _AGENT_SESSION_ID:
        return _AGENT_SESSION_ID
    if _IDENTITY_UNRESOLVED:
        # A mis-identified spawned child must NEVER adopt a workspace session:
        # not via the exact binding-key match, not via silent single-candidate
        # auto-adopt (the dangerous no-tool-call path), and not via the recovery
        # nudge. Return unresolved with an empty nudge before any candidate scan.
        return ""
    key = _binding_key()
    binding = _read_json_object(_binding_file())
    session_id = binding.get("session_id")
    if (
        binding.get("binding_key") == key
        and isinstance(session_id, str)
        and _agents_file(session_id).exists()
    ):
        return session_id

    candidates = _candidate_sessions()
    if not candidates:
        return ""
    single_lead_history = len(_distinct_binding_sessions()) <= 1
    if (
        _autoadopt_enabled()
        and len(candidates) == 1
        and single_lead_history
        and _has_autoadoptable_agent(str(candidates[0]["session_id"]))
    ):
        adopted = candidates[0]
        _pending_recovery = {
            "adopted_session": {
                "session_id": adopted["session_id"],
                "agent_count": adopted["agent_count"],
            }
        }
        return adopted["session_id"]
    _pending_recovery = {
        "recoverable_sessions": candidates,
        "recovery_hint": (
            "Recoverable prior session(s) exist for this workspace. Call "
            "resume_session('<session_id>') to adopt one (see session_info)."
        ),
    }
    return ""


def _recovery_note() -> dict:
    """Return the pending recovery nudge for a dict-tool result.

    ``adopted_session`` is one-shot (cleared after first surface); the
    ``recoverable_sessions`` nudge persists because the session stays
    unresolved (and is refreshed on every recovery attempt) until the lead
    calls ``resume_session``.
    """
    global _pending_recovery  # noqa: PLW0603 - recovery nudge state.
    note = dict(_pending_recovery)
    if "adopted_session" in _pending_recovery:
        _pending_recovery = {}
    return note


def _annotate(result: dict) -> dict:
    """Merge the recovery nudge into a dict tool result (no-op otherwise)."""
    note = _recovery_note()
    if isinstance(result, dict) and note:
        merged = dict(result)
        merged.update(note)
        return merged
    return result


def _active_session_id(*, create: bool = False) -> str:
    """Return the current session id, recovering persisted lead state if needed."""
    global _session_id  # noqa: PLW0603 - module-level lead session state.

    if _session_id:
        return _session_id
    recovered = _recover_session_id()
    if recovered:
        _session_id = recovered
        # A fallback auto-adoption must re-bind to the current parent key (so
        # the next call hits the fast path) and prune the stale binding.
        if _pending_recovery.get("adopted_session"):
            _persist_session_binding(recovered)
        return _session_id
    if create:
        _maybe_cleanup_old_sessions()
        _session_id = _create_session()
        return _session_id
    return ""


def _is_session_dir(path: Path) -> bool:
    """Whether ``path`` is a real session directory (UUID name + registry).

    Guards cleanup from ever treating base-level control entries (the
    ``bindings/`` dir, the ``.last-cleanup`` stamp, stray files) as sessions.
    """
    if not path.is_dir() or path.name == _BINDINGS_DIR_NAME:
        return False
    try:
        uuid.UUID(path.name)
    except ValueError:
        return False
    return (path / "agents.json").exists() or (path / _SESSION_META_NAME).exists()


def _dir_newest_mtime(path: Path) -> float:
    """Return the newest mtime across ``path`` and its descendants."""
    newest = 0.0
    with suppress(OSError):
        newest = path.stat().st_mtime
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return newest
    for entry in entries:
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, mtime)
    return newest


def _session_has_live_agent(session_id: str) -> bool:
    """Whether any agent in ``session_id`` is still a live (owned) process."""
    try:
        agents = _load_agents_unlocked(session_id)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(agents, list):
        return False
    return any(isinstance(a, dict) and _agent_alive(a) for a in agents)


def _remove_team_logs(session_id: str) -> None:
    """Remove the default team-log dir for a removed session (best-effort).

    A ``WIN_AGENT_TEAMS_LOG_DIR`` override points at a user-chosen tree and is
    intentionally left untouched (out of cleanup scope).
    """
    if os.environ.get("WIN_AGENT_TEAMS_LOG_DIR"):
        return
    with suppress(OSError):
        shutil.rmtree(_TEAMS_BASE / session_id, ignore_errors=True)


def _prune_orphan_bindings() -> None:
    """Delete binding files whose session directory no longer exists."""
    for path, meta, _mtime in _iter_binding_metas():
        session_id = meta.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or not _session_dir(session_id).is_dir()
        ):
            with suppress(OSError):
                path.unlink(missing_ok=True)


def cleanup_old_sessions(
    max_age_days: float | None = None, now: float | None = None
) -> list[str]:
    """Remove session dirs older than the retention window; return removed ids.

    Only real session directories (UUID-named, with a registry) are ever
    considered — never ``bindings/`` or the cleanup stamp. The active session
    and any session with a live agent are always kept. For each removed
    session the matching default team-log dir is deleted; orphan binding files
    are pruned afterwards. Fully best-effort per entry.
    """
    base = _SESSION_BASE
    if not base.is_dir():
        return []
    max_age = _retention_days() if max_age_days is None else max_age_days
    current = time.time() if now is None else now
    cutoff = current - max_age * 86400.0
    removed: list[str] = []
    active = _session_id
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    for path in entries:
        if not _is_session_dir(path):
            continue
        session_id = path.name
        if session_id == active:
            continue
        if _dir_newest_mtime(path) >= cutoff:
            continue
        if _session_has_live_agent(session_id):
            continue
        with suppress(OSError):
            shutil.rmtree(path, ignore_errors=True)
        _remove_team_logs(session_id)
        removed.append(session_id)
    _prune_orphan_bindings()
    return removed


def _maybe_cleanup_old_sessions() -> None:
    """Run cleanup at most once per interval; swallow every error."""
    try:
        _SESSION_BASE.mkdir(parents=True, exist_ok=True)
        stamp = _SESSION_BASE / _CLEANUP_STAMP_NAME
        now = time.time()
        if stamp.exists():
            try:
                last = float(stamp.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                last = 0.0
            if now - last < _CLEANUP_INTERVAL_SECONDS:
                return
        stamp.write_text(str(now), encoding="utf-8")
        cleanup_old_sessions()
    except Exception:
        logger.debug("session cleanup skipped", exc_info=True)


#: C3/R5 — how a ``send_message`` recipient relates to the caller. The first
#: two are deliverable (by different paths); the last three are refused.
RECIPIENT_CHILD = "child"
RECIPIENT_SPAWNER = "spawner"
RECIPIENT_SIBLING = "sibling"
RECIPIENT_UNRELATED = "unrelated"
RECIPIENT_UNKNOWN = "unknown"

#: The refused classes, with the reason each is refused rather than routed.
_UNADDRESSABLE_DETAIL = {
    RECIPIENT_SIBLING: (
        "is a sibling: you and it share a spawner, but neither spawned the "
        "other. Agent-to-agent peer messaging is an explicit non-goal — route "
        "the request through {spawner!r} instead."
    ),
    RECIPIENT_UNRELATED: (
        "exists in this session but is neither an agent you spawned nor the "
        "agent that spawned you (it may be a grandchild, or another lead's "
        "worker). Messages travel one hop along the spawn edge only: message "
        "your own child, or {spawner!r}."
    ),
    RECIPIENT_UNKNOWN: (
        "does not name any agent in this session. Check the spelling against "
        "list_agents. It was NOT re-routed to {spawner!r}: a typo silently "
        "delivered upstream is how a message ends up read by the wrong agent, "
        "or by nobody. Nothing was sent."
    ),
}


def _spawner_target() -> str:
    """Return the agent this one reports to: its spawner, or itself for root."""
    return (
        (_AGENT_PARENT_NAME or ROOT_LEAD_NAME)
        if IDENTITY != ROOT_LEAD_NAME
        else ROOT_LEAD_NAME
    )


def _classify_recipient(to: str, session_id: str) -> tuple[str, str]:
    """Classify a ``send_message`` recipient relative to the caller (C3/R5).

    Returns ``(recipient_class, resolved_name)``. The class decides the path:
    a child goes through the guaranteed (Phase B) path, the spawner goes to the
    inbox that the spawner's watcher is watching, and everything else is
    refused.

    This deliberately replaces the old "unknown name is routed to the lead with
    a warning" rule. That rule made every typo an upstream message which the
    lead would read as genuine, and it made `send_message` an accept-then-drop
    path for any recipient that could not actually be reached — exactly what R5
    forbids. A warning field is not a refusal: nothing consumed it.

    The registry is flat (one ``agents.json``, no sub-sessions), so the
    relationship is read from the record's ``spawned_by`` field, not inferred
    from a tree.
    """
    raw = to.strip()
    spawner = _spawner_target()

    # The aliases mean "whoever spawned me" by definition, so they resolve
    # before any registry lookup and can never be a typo.
    if raw.lower() in _LEAD_ALIASES:
        return RECIPIENT_SPAWNER, spawner

    agent = _find_agent(_load_agents(session_id), raw)
    if agent is None:
        # The root lead has no record of its own, so naming it explicitly is
        # still upstream rather than unknown.
        missing = RECIPIENT_SPAWNER if raw == ROOT_LEAD_NAME else RECIPIENT_UNKNOWN
        return missing, raw

    # Checked before parentage: a spawner named explicitly is the same upstream
    # path as the alias. Its own ``spawned_by`` points further up, so asking
    # only "did I spawn this?" would refuse the one path R3 depends on.
    if raw == spawner:
        return RECIPIENT_SPAWNER, raw
    parent = agent.get(SPAWNED_BY_FIELD)
    if parent == IDENTITY:
        return RECIPIENT_CHILD, raw
    if parent == spawner:
        return RECIPIENT_SIBLING, raw
    return RECIPIENT_UNRELATED, raw


def _truncate(text: str | None, max_chars: int | None) -> tuple[str, bool, int]:
    """Clip ``text`` to ``max_chars``, signalling whether it was truncated.

    Returns ``(clipped, truncated, full_len)``. ``full_len`` is the character
    count of the untruncated text. ``max_chars=None`` returns the text
    unclipped; ``max_chars<=0`` clips to an empty string (but still reports
    ``truncated=True`` when the original text was non-empty).
    """
    original = text or ""
    full_len = len(original)
    if max_chars is None:
        return original, False, full_len
    if full_len <= max_chars:
        return original, False, full_len
    return original[: max(max_chars, 0)], True, full_len


_DEFAULT_LAST_LINE_MAX_CHARS = 200


def _watch_argv(
    session_dir: str | Path,
    timeout: float | None = None,
    reader: str | None = None,
) -> list[str]:
    """Return the canonical shell-neutral argv for watching ``session_dir``."""
    argv = [sys.executable, "-m", "claude_teams.cli", "watch", str(session_dir)]
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    if reader is not None:
        argv.extend(["--reader", reader])
    return argv


def _watch_command_bash(session_dir: str | Path, timeout: float | None = None) -> str:
    """Render the watch argv for Bash."""
    return " ".join(shlex.quote(token) for token in _watch_argv(session_dir, timeout))


def _watch_command_powershell(
    session_dir: str | Path, timeout: float | None = None
) -> str:
    """Render the watch argv for PowerShell."""
    quoted = [
        "'" + token.replace("'", "''") + "'"
        for token in _watch_argv(session_dir, timeout)
    ]
    return "& " + " ".join(quoted)


# Shared disk-contract + watch-recipe note, appended verbatim to the
# docstrings of agent_status/check_agent/list_agents/agent_watch_paths (item
# 2/3 of the coordinator-event-loop plan). The consuming agent only ever
# reads tool docstrings and tool return values, never the README, so this
# text (not any repo doc) is the actual contract surface.
_DISK_CONTRACT_NOTE = """
Disk contract: each agent's state is written by an injected lifecycle hook
to `state-{name}.json` under the session dir, schema
`{"state": "running" | "waiting", "event": "<hook>", "ts": <float epoch>}`.
This file is on disk and survives MCP server restarts; this tool is cheap
and auto-restarts the server if it had died from host idle timeout. Do not
tight-poll this tool — use the watch recipe below instead.

The `win-agent-teams` console script may not be on PATH. `spawn_agent` and
`agent_watch_paths` return a ready-to-run, shell-neutral `watch_argv`, plus
`watch_command_bash` and `watch_command_powershell` renderings. Use
`watch_argv` for direct process spawning and for shells such as cmd.exe.
The watcher ignores non-actionable churn — `running` hook transitions and
`SubagentStop` (a worker's own internal Task subagent finishing) — and emits
one JSON wake record: `reason="message"` for unread inbox data,
`reason="waiting"` for a marker that settles as waiting, or `reason="output"`
for a selected output. A waiting marker must persist for a short settle window
(`WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS`, default 15s) before it wakes; one that
resumes `running` within the window is suppressed as a brief park. On a message,
call `read_messages`; on waiting, call `agent_status` or `check_agent` for the
status delta. Watch is one-shot: it exits on the first signal, so re-arm it
after every wake.

- Claude Code coordinator: run the watch as a BACKGROUND command. Its
  completion triggers a harness wake for the idle coordinator; branch on the
  emitted reason as above.
- Codex coordinator: Codex has no idle-wake, so run the watch as a BOUNDED
  FOREGROUND command within the same turn (for example, append `--timeout 60`
  to `watch_argv`, looped), then branch on its emitted reason.
  A marker read is useful for `reason="waiting"`; `reason="message"` requires
  `read_messages` because no state marker need have changed.

Timeout exit 2 means no actionable edge settled; re-check status before
starting the next watch because an agent may already be waiting due to the
small status-check/watch-baseline race, or a genuine waiting edge may have
arrived inside the final unfinished settle window.

Claude Code lead wake: a `Stop` hook now verifies watcher arming from the
harness's own `background_tasks` on every lead turn end. When a worker reply is
already unread it blocks instructing you to call `read_messages`; when no
watcher is armed it blocks instructing you to run the `watch_command_bash` /
`watch_argv` as a BACKGROUND task, so an idle lead is woken deterministically
rather than relying on the model to re-arm. The hook writes a small
`wake-progress-<reader>.json` file under the session dir to bound a no-progress
block loop and is always fail-open (never makes the lead unstoppable). Disable
it at runtime with `WIN_AGENT_TEAMS_LEAD_WAKE=0`. Server-spawned agents get this
wiring automatically; a top-level lead wires it with the `install_lead_wake`
tool.
""".strip()


def _with_disk_note(fn):
    """Append ``_DISK_CONTRACT_NOTE`` to ``fn.__doc__`` before registration.

    Must be applied BELOW ``@mcp.tool()`` (i.e. closer to the function) so it
    runs first and the note is part of the docstring FastMCP parses at
    decoration time. Appending to ``__doc__`` after ``@mcp.tool()`` has
    already run only mutates the function object, not the registered
    ``Tool.description`` that clients actually see.
    """
    fn.__doc__ = (fn.__doc__ or "") + "\n\n" + _DISK_CONTRACT_NOTE
    return fn


def _empty_agent_check(name: str, *, full: bool = False) -> dict:
    """Return a stable empty ``check_agent`` payload for an unknown agent."""
    compact = {
        "name": name,
        "state": "dead",
        "alive": False,
        "pid": None,
        "backend": None,
        "last_activity_at": None,
        "unread_count": 0,
        "last_line": "",
        "seq": 0,
        "truncated": False,
        "full_len": 0,
        "heartbeat_age_s": None,
        "stalled": False,
    }
    if not full:
        return compact
    compact.update({"last_message": None, "backend_session_id": None})
    return compact


def _safe_float(value: object) -> float:
    """Coerce persisted numeric metadata to a float."""
    try:
        # ``value or 0.0`` is evaluated unchanged at runtime; ``cast`` only
        # relaxes the static type so ``float`` accepts the (typed ``object``)
        # persisted value. This preserves conversion of any float-convertible
        # input (Decimal, Fraction, bytes, custom __float__).
        return float(cast(Any, value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _stored_backend_session_id(agent: dict) -> str | None:
    """Return the stored backend session id when present."""
    value = agent.get("backend_session_id")
    if isinstance(value, str) and value:
        return value
    return None


def _resolve_agent_binding(agent: dict, *, bounded_only: bool = False) -> BindingResult:
    """Resolve an agent record to its transcript via the A2 validation ladder.

    ``child_alive`` is passed lazily: liveness is only consulted by gate 0
    (the sidecar-pending branch of the count gate), so the common path never
    pays for a process probe.

    ``bounded_only`` is A6's "stay cheap" mode: the mtime window only, with no
    all-history fallback behind it.
    """
    if agent.get("backend") == "external":
        return BindingResult("not_applicable")
    return resolve_agent_binding(
        agent, child_alive=lambda: _agent_alive(agent), bounded_only=bounded_only
    )


def _agent_create_token(agent: dict) -> str | None:
    """Return the agent's stored PID creation token, or ``None`` if absent."""
    token = agent.get("create_token")
    return token if isinstance(token, str) and token else None


def _agent_alive(agent: dict) -> bool:
    """PID-reuse-safe liveness for an agent record.

    Passes the record's stored ``create_token`` to the process manager so a
    reused PID (after a server/host restart) is reported dead rather than
    falsely alive. Records predating tokens fall back to bare PID liveness.

    For launcher-style backends (Linux terminal) the stored ``pid`` is the
    terminal launcher; the real agent PID lives in a sidecar that survives a
    restart. We resolve it and report the agent's own liveness so a live agent
    is never seen as dead just because its launcher exited (which would let
    cleanup delete a session that still has a live agent).
    """
    if agent.get("backend") == "external":
        # External PIDs belong to a manually started, potentially shared MCP
        # server and become stale across Desktop restarts. Membership state and
        # markers are the honest liveness signal; never probe the informational
        # PID.
        return agent.get("status") == "running"
    launcher = str(agent.get("pid"))
    authoritative = process_manager.resolve_agent_pid(
        launcher, str(agent.get("session_id") or ""), str(agent.get("name") or "")
    )
    if authoritative != launcher:
        # Distinct real agent PID from the sidecar; no stored token for it, so
        # this is display/cleanup liveness only. Destructive ops still gate on
        # owns_process against the stored launcher pid+token.
        alive, _ = process_manager.health_check(authoritative)
        return alive
    alive, _ = process_manager.health_check(
        launcher, expected_token=_agent_create_token(agent)
    )
    return alive


def _sync_backend_session_id(agent: dict, binding: BindingResult) -> bool:
    """Persist a newly discovered backend session id onto an agent record.

    Only a **bound** binding may write to the record. An ``unverified`` or
    ``pending`` read has not identified a transcript we are entitled to
    attribute to this agent, and persisting an id from one would poison the
    record permanently — every later read would then trust it.
    """
    output = binding.output
    if not binding.bound or output is None or not output.backend_session_id:
        return False
    if agent.get("backend_session_id") == output.backend_session_id:
        return False
    agent["backend_session_id"] = output.backend_session_id
    return True


def _agent_check_payload(
    name: str, agent: dict, alive: bool, binding: BindingResult
) -> dict:
    """Build the rich INTERNAL check payload for an existing agent record.

    Consumed by ``follow_up_agent``/``_follow_up_failure`` (which need the
    unbounded ``last_message`` to decide busy/idle) and projected down to the
    compact public ``check_agent`` shape by ``_compact_check_view``.
    """
    output = binding.output
    public = _public_agent_record(agent)
    return {
        "name": name,
        "alive": alive,
        "pid": public["pid"],
        "backend": public.get("backend"),
        "backend_session_id": _stored_backend_session_id(agent),
        "last_activity_at": output.last_activity_at if output else None,
        "last_message": output.last_message if output else None,
        "binding": binding.outcome,
        "binding_retriable": binding.retriable,
    }


def _last_non_empty_line(text: str | None) -> str:
    """Return the last non-blank line of ``text``, or ``""`` when none exists."""
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _sender_message_count(session_id: str, reader: str, sender: str) -> int:
    """Return how many valid messages ``sender`` has sent to ``reader``'s inbox."""
    by_sender = read_inbox_by_sender(_inbox_file(session_id, reader))
    return len(by_sender.get(sender, []))


def _sender_unread_count(session_id: str, reader: str, sender: str) -> int:
    """Return ``sender``'s unread (not-yet-consumed) message count for ``reader``."""
    counts = unread_sender_counts(
        _inbox_file(session_id, reader), _inbox_cursor_file(session_id, reader)
    )
    return counts.get(sender, 0)


def _compact_check_view(
    session_id: str,
    name: str,
    internal: dict,
    *,
    max_chars: int = _DEFAULT_LAST_LINE_MAX_CHARS,
) -> dict:
    """Project the rich internal check payload to the compact public shape."""
    marker = _read_state_marker(session_id, name)
    state = _resolve_agent_state(
        alive=bool(internal.get("alive")),
        marker=marker,
        last_activity_at=internal.get("last_activity_at"),
    )
    last_line, truncated, full_len = _truncate(
        _last_non_empty_line(internal.get("last_message")), max_chars
    )
    seq = _sender_message_count(session_id, IDENTITY, name)
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    heartbeat_source = _marker_timestamp(marker)
    if heartbeat_source is None:
        heartbeat_source = internal.get("last_activity_at")
    heartbeat_age_s, stalled = _heartbeat_fields(
        alive=bool(internal.get("alive")),
        state=state,
        last_activity_ts=heartbeat_source,
    )
    return {
        "name": name,
        "state": state,
        "alive": internal.get("alive"),
        "pid": internal.get("pid"),
        "backend": internal.get("backend"),
        "last_activity_at": internal.get("last_activity_at"),
        "unread_count": unread_count,
        "last_line": last_line,
        "seq": seq,
        "truncated": truncated,
        "full_len": full_len,
        "heartbeat_age_s": heartbeat_age_s,
        "stalled": stalled,
        "binding": internal.get("binding"),
        "binding_retriable": internal.get("binding_retriable"),
    }


def _follow_up_failure(
    reason: str,
    name: str,
    status: dict | None = None,
    *,
    retriable: bool = False,
    detail: str = "",
) -> dict:
    """Build a structured ``follow_up_agent`` failure payload."""
    payload: dict[str, object] = {
        "success": False,
        "name": name,
        "reason": reason,
        "retriable": retriable,
    }
    if detail:
        payload["detail"] = detail
    if status:
        payload.update(
            {
                "alive": status.get("alive"),
                "backend_session_id": status.get("backend_session_id"),
                "last_activity_at": status.get("last_activity_at"),
                "last_message": status.get("last_message"),
                "binding": status.get("binding"),
            }
        )
    return payload


#: B3/R7 — what each unreachable target state means and what to do instead.
_NO_DELIVERY_PATH_DETAIL = {
    "record_removed": (
        "There is no record for this agent in the session: it was killed, or "
        "it never existed under this name. A killed agent is unreachable by "
        "design — its context is gone — so spawn a new agent rather than "
        "retrying. Nothing was sent."
    ),
    "no_backend_session": (
        "This agent has no resumable backend session, so there is no channel "
        "to deliver into. A dead agent WITH a valid backend session is still "
        "resumable; this one is not. Nothing was sent."
    ),
}


def _no_delivery_path(name: str, state: str, status: dict | None = None) -> dict:
    """Build the R7 refusal that names the unreachable state.

    R7 exists because the old behaviour was two tools pointing at each other:
    one returned a refusal, the other returned success without delivering. A
    recipient that genuinely cannot be reached must be said so once, plainly,
    with the state named — never a bare lookup miss.
    """
    payload = _follow_up_failure(
        "no_delivery_path",
        name,
        status,
        retriable=False,
        detail=_NO_DELIVERY_PATH_DETAIL.get(state, ""),
    )
    payload["state"] = state
    return payload


def _direction_refusal(agent: dict, name: str) -> dict | None:
    """Return a refusal payload unless this caller spawned ``agent``.

    ``None`` means the follow-up may proceed. See the call site in
    ``follow_up_agent._prepare`` for why this is an accident guard rather than
    an authorization check, and why it must run before any side effect.
    """
    spawner = agent.get(SPAWNED_BY_FIELD)
    if not isinstance(spawner, str) or not spawner:
        # Deliberately NOT a silent allow. Records written before this field
        # existed cannot be backfilled from anything trustworthy, and allowing
        # them would disable the guard during precisely the upgrade window in
        # which stale, unowned records are most likely to be around.
        return _follow_up_failure(
            "parent_unknown",
            name,
            detail=(
                f"The record for {name!r} does not say who spawned it, so the "
                "downstream-only rule cannot be evaluated. It predates spawner "
                "tracking and cannot be backfilled automatically. Either kill "
                "and respawn it, or have an operator run "
                "`win-agent-teams adopt` (CLI-only, requires the session "
                "recovery token)."
            ),
        )
    if spawner != IDENTITY:
        return _follow_up_failure(
            "not_spawner",
            name,
            detail=(
                f"follow_up_agent is downstream-only: it may only be called by "
                f"the agent that spawned the target. {name!r} was spawned by "
                f"{spawner!r}, not by {IDENTITY!r}. A follow-up is "
                "kill-and-respawn, so allowing this would restart the target's "
                "process and destroy its context. To reach an agent you did "
                "not spawn, use send_message; it appends to that agent's inbox "
                "and wakes its watcher without disturbing its process."
            ),
        )
    return None


#: Why each non-``bound`` binding outcome blocks a follow-up, and what the
#: caller should do about it. ``legacy`` names kill-and-respawn because R8
#: allows no compatibility exception: a legacy stored id may be exactly the
#: wrong pinned id this feature exists to fix, and resuming on it would let a
#: nonce be confirmed in the wrong conversation and reported as delivered.
_BINDING_REFUSAL_DETAIL = {
    "pending": (
        "The agent was launched with a prompt sidecar and has not yet recorded "
        "reading it, so its transcript cannot be identified yet. Retry shortly."
    ),
    "unverified": (
        "No transcript could be confidently attributed to this agent, so a "
        "follow-up cannot be confirmed as delivered. Do not retry; kill the "
        "agent and respawn it if you need to continue the work."
    ),
    "ambiguous": (
        "More than one transcript carries this agent's correlation marker, so "
        "resuming could continue the wrong conversation. Kill and respawn."
    ),
    "legacy": (
        "This agent predates correlation ids, so its stored session id cannot "
        "be verified and a follow-up could be delivered into the wrong "
        "conversation. Kill the agent and respawn it (kill_agent then "
        "spawn_agent); there is no way to make an existing legacy agent "
        "resumable."
    ),
    "indeterminate": (
        "A candidate transcript could not be read, so the binding is unknown "
        "rather than absent. Retry shortly."
    ),
}


@dataclass(frozen=True)
class _FollowUpPlan:
    """Everything phase 2 needs, captured while the registry lock was held."""

    agent_name: str
    backend: Any
    backend_name: str
    backend_session_id: str
    request: SpawnRequest
    old_pid: str
    old_create_token: str | None
    alive: bool
    nonce: str
    operation_id: str
    generation: int
    prompt_file: Path | None
    prompt_transport: str
    scanner: ReceiptScanner
    model: object
    permission_mode: str
    effort: str | None
    correlation_id: str | None
    agent_cwd: str
    spawned_by: str
    spawned_by_source: str
    #: Copy of the target's registry record as phase 1 saw it. Stored on the
    #: durable row so a rescan survives the record's removal (see
    #: :func:`_scan_target`).
    agent_snapshot: dict


@dataclass(frozen=True)
class _FollowUpPrep:
    """Result of phase 1: a refusal, a plan, a FIFO place, or "wait and retry".

    ``wait_reason`` is B2: a busy target is no longer refused, so phase 1 has
    to be able to say "nothing is wrong, come back in a moment" without
    returning either a plan or a refusal.
    """

    refusal: dict | None = None
    plan: _FollowUpPlan | None = None
    ticket: str | None = None
    queue_position: int = 0
    wait_reason: str = ""


#: Sentinel outcome for "the backend never returned a child at all". Distinct
#: from every scan outcome because there is nothing to have confirmed.
_DELIVERY_RESUME_FAILED = DeliveryOutcome(DELIVERY_FAILED, "resume_failed")

#: Why each terminal non-delivery blocks, in the caller's terms. R6 requires
#: that a definite non-delivery is never dressed up as progress.
_DELIVERY_FAILURE_DETAIL = {
    "resume_not_confirmed": (
        "The resumed backend process exited immediately, so the resume never "
        "attached to the conversation (typically an invalid session id). "
        "Nothing was delivered and the agent record is unchanged."
    ),
    "not_delivered": (
        "The resumed process died without the prompt ever appearing in the "
        "target's context. This is a definite non-delivery, not a timeout."
    ),
    "rotation_ambiguous": (
        "The target's transcript was replaced by more than one candidate "
        "successor, so delivery cannot be attributed to either. Kill and "
        "respawn rather than retrying."
    ),
    "resume_failed": (
        "The backend refused to resume the session. Nothing was delivered."
    ),
}


def _optional_path(value: object) -> Path | None:
    """Return ``value`` as a :class:`Path`, or ``None`` when it is not a path."""
    return Path(value) if isinstance(value, str) and value else None


def _record_generation(agent: dict) -> int:
    """Return the agent record's CAS generation (absent counts as 0)."""
    value = agent.get("generation")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _bump_generation(agent: dict) -> int:
    """Advance the record's generation and return the new value.

    The operator force path calls this *before* terminating anything, so a
    holder that is still alive can no longer finalize: its CAS is fenced out.
    """
    generation = _record_generation(agent) + 1
    agent["generation"] = generation
    return generation


def _lease_holder_probe(pid: int, token: str | None) -> str:
    """Three-valued, PID-reuse-safe ownership for a lease/claim holder.

    Returns one of ``OWNERSHIP_OURS`` / ``OWNERSHIP_NOT_OURS`` /
    ``OWNERSHIP_INDETERMINATE``. Only the middle one means "provably gone", and
    only that one may authorize reclaiming a lease or a delivery-record claim:
    a creation token we failed to read against a PID that is still alive is
    uncertainty, and resuming on uncertainty is the double delivery this whole
    feature exists to prevent.

    There is deliberately no shortcut for ``pid == os.getpid()``. A lease left
    behind by an EARLIER incarnation of the server whose PID has since been
    recycled onto this process would otherwise be treated as live forever, and
    nothing else can prove that holder is gone — which is precisely the
    PID-reuse hazard the paired token exists to close. Reservation records the
    live token for our own PID, so the honest pairing check succeeds on every
    lease this process actually holds.
    """
    return process_manager.ownership_probe(str(pid), token)


def _lease_holder_live(pid: int, token: str | None) -> bool:
    """Whether a holder is provably ours. Display and operator paths only.

    Reclaim decisions must use :func:`_lease_holder_probe`, because this
    collapses "gone" and "could not tell" into one ``False``.
    """
    return _lease_holder_probe(pid, token) == OWNERSHIP_OURS


def _delivery_successors(agent: dict, backend_name: str, session_id: str):
    """Return a callable enumerating candidate successor transcripts.

    Used only when the scanned transcript rotates, is truncated, or is
    replaced. Continuity is re-established from backend session id plus file
    identity, so this only has to supply the candidate set.
    """
    spawned_at = _safe_float(agent.get("spawned_at"))
    cwd = str(agent.get("cwd") or "")

    def _candidates() -> list[Path]:
        binder = _make_binder(backend_name, spawned_at, cwd, agent)
        if binder is None:
            return []
        try:
            return list(binder.candidates(all_history=True))
        except OSError:
            return []

    _ = session_id
    return _candidates


def _delivery_scanner(
    agent: dict,
    backend_name: str,
    backend_session_id: str,
    binding: BindingResult,
    session_id: str,
) -> ReceiptScanner:
    """Build a scanner anchored on the transcript this resume targets."""
    output = binding.output
    path = Path(output.rollout_path) if output and output.rollout_path else None
    if path is None:
        binder = _make_binder(
            backend_name,
            _safe_float(agent.get("spawned_at")),
            str(agent.get("cwd") or ""),
            agent,
        )
        if binder is not None:
            path = binder.resolve_by_session_id(backend_session_id)
    return ReceiptScanner(
        path,
        backend=backend_name,
        backend_session_id=backend_session_id,
        successors=_delivery_successors(agent, backend_name, session_id),
    )


def _pending_delivery(agent: dict) -> dict | None:
    """Return the unconfirmed attempt recorded on this agent, if any."""
    value = agent.get(PENDING_DELIVERY_FIELD)
    return value if isinstance(value, dict) else None


def _reconcile_pending_delivery(
    agent: dict,
    backend_name: str,
    backend_session_id: str,
    binding: BindingResult,
    session_id: str,
) -> bool:
    """Rescan a prior unconfirmed attempt's nonce before sending anything new.

    R6 requires this: a transcript write buffered past the previous call's
    bound can arrive afterwards, so an ``unconfirmed`` attempt may in fact have
    landed. Re-sending without checking would deliver the same instruction
    twice. Returns whether the earlier attempt is now confirmed.
    """
    pending = _pending_delivery(agent)
    if pending is None:
        return False
    nonce = pending.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        return False
    scanner = _delivery_scanner(
        agent, backend_name, backend_session_id, binding, session_id
    )
    # rewind(), not snapshot(): reconciliation must see the WHOLE transcript,
    # because the record it is looking for was written before this call began.
    scanner.rewind()
    return scanner.full_scan(nonce) == SCAN_FOUND


def _scan_for_nonce(session_id: str, agent: dict | None, nonce: str) -> str:
    """Search the whole of ``agent``'s bound transcript for ``nonce``.

    Returns one of :data:`SCAN_FOUND`, :data:`SCAN_ABSENT`,
    :data:`SCAN_INDETERMINATE` or :data:`SCAN_AMBIGUOUS`. **Not a bool.**
    Collapsing these into "delivered / not delivered" is what let an unreadable
    or ambiguous scan become a terminal ``failed``, and R6's whole point is that
    only a *definite* non-delivery may be terminal. It is the same distinction
    the binding ladder enforces: an error must not become an absence.

    ``full_scan()`` after ``rewind()``, not ``poll()``: the record being looked
    for was written before this call began, so anchoring at the current tail —
    which is what a fresh attempt does — would deliberately skip exactly what we
    came for, and ``poll``'s ``SCAN_PENDING`` cannot tell "read it all, not
    there" from "could not read".

    Takes the agent **record**, never a name, so it can run inside a registry
    transaction. Re-entering ``_load_agents`` there would try to take the
    cross-process file lock this process already holds on another handle,
    which blocks on POSIX and times out on Windows.
    """
    if not nonce or agent is None:
        # No nonce means nothing was ever sent under this row, and no record
        # means its transcript is unreachable from here. Neither is evidence
        # that a prompt did not land, so neither may read as absence.
        return SCAN_INDETERMINATE
    backend_name = str(agent.get("backend") or "")
    binding = _resolve_agent_binding(agent)
    output = binding.output
    backend_session_id = str(
        _stored_backend_session_id(agent)
        or (output.backend_session_id if output else "")
        or ""
    )
    if not backend_session_id:
        return SCAN_INDETERMINATE
    scanner = _delivery_scanner(
        agent, backend_name, backend_session_id, binding, session_id
    )
    scanner.rewind()
    return scanner.full_scan(nonce)


def _find_agent(agents: list[dict], name: str) -> dict | None:
    """Return the record named ``name``, or ``None``."""
    return next((a for a in agents if a.get("name") == name), None)


#: Copy of the target's registry record, stored on the delivery row when an
#: attempt is made. See :func:`_scan_target`.
TARGET_SNAPSHOT_FIELD = "target_snapshot"


def _scan_snapshot(agent: dict | None) -> dict | None:
    """Freeze what a later rescan needs to find this attempt's receipt."""
    return dict(agent) if agent is not None else None


def _scan_target(record: dict, agent: dict | None) -> dict | None:
    """Return the record to scan for ``record``'s nonce, agent gone or not.

    ``kill_agent`` deletes the target from ``agents.json``, and ``_scan_for_nonce``
    answers ``indeterminate`` for a missing record — so before this existed, an
    attempt whose settlement write was lost at kill time could never be
    reconciled again by anything. Kill still terminates the process even when
    that write fails (it is a lifecycle operation, and hanging on a disk error
    would be worse), but the evidence path must survive it, so the transcript
    binding is copied onto the durable row at attempt time and outlives the
    registry record.

    Liveness is deliberately NOT taken from the snapshot: it is a frozen copy,
    and a recycled PID inside it would read as a live child. Callers pass the
    real (possibly ``None``) record for that, and this only for scanning.
    """
    if agent is not None:
        return agent
    snapshot = record.get(TARGET_SNAPSHOT_FIELD)
    return cast("dict[str, Any]", snapshot) if isinstance(snapshot, dict) else None


def _reconcile_delivery_record(
    session_id: str, record: dict, agent: dict | None, *, now: float | None = None
) -> bool:
    """Actively reconcile one ``unconfirmed`` attempt. Returns whether it moved.

    This is why ``delivery_status`` is a reconciler and not a lookup: without
    it, response-loss recovery would keep answering ``unconfirmed`` forever
    after the nonce had actually landed, which is a false status in the other
    direction.

    The reaping rule is deliberately asymmetric. A **live** child with no
    receipt stays ``queued(phase=unconfirmed)`` indefinitely — that follows
    from the no-dispatcher non-goal and is honest rather than silently
    expired. Only once the child is proven dead, and one flush grace has
    passed, does a still-absent nonce become terminal.
    """
    if is_terminal(record) or record.get("phase") not in {
        PHASE_SENT,
        PHASE_UNCONFIRMED,
    }:
        return False
    nonce = str(record.get("nonce") or "")
    outcome = _scan_for_nonce(session_id, _scan_target(record, agent), nonce)
    if outcome == SCAN_FOUND:
        settle(record, STATUS_DELIVERED, reason="", now=now if now else time.time())
        remove_prompt_file(_optional_path(record.get("prompt_file")))
        return True
    if agent is not None and _agent_alive(agent):
        # Live uncertainty (R6). Not delivered, and emphatically not failed.
        mark_phase(record, PHASE_UNCONFIRMED)
        return record.get("phase") == PHASE_UNCONFIRMED
    if outcome != SCAN_ABSENT:
        # The child is gone, but the scan did not establish that the nonce is
        # not there: the transcript was unreadable, incomplete, or rotated into
        # candidates we may not choose between. R6 permits a terminal ``failed``
        # only on a COMPLETE authoritative negative, so this stays uncertain
        # rather than being reported as definite non-delivery.
        mark_phase(record, PHASE_UNCONFIRMED)
        return False
    attempted_at = _safe_float(record.get("attempted_at"))
    current = now if now is not None else time.time()
    if attempted_at and current - attempted_at < _UNCONFIRMED_FLUSH_GRACE_SECONDS:
        # The child is gone but its last transcript writes may not have hit
        # disk yet. Settling here would report a delivered prompt as failed.
        mark_phase(record, PHASE_UNCONFIRMED)
        return False
    settle(record, STATUS_FAILED, reason=REASON_NOT_DELIVERED, now=current)
    remove_prompt_file(_optional_path(record.get("prompt_file")))
    return True


def _reconcile_deliveries_for_target(
    session_id: str, agent_name: str, agent: dict | None
) -> None:
    """Settle every in-flight attempt against ``agent_name``, receipt first.

    Called from ``kill_agent`` while the registry lock is held. Order is the
    whole point: an attempt may already have an unread receipt on disk, and
    marking that message ``failed`` because the target is being killed would
    reintroduce precisely the false status this feature exists to remove.

    Nothing is deleted. Kill purges inbox lines; the audit trail is what the
    sender comes back to *after* the target is gone.

    A store write that cannot be persisted is logged and swallowed **here and
    only here**: kill is a lifecycle operation that must still terminate the
    process. The rows simply stay where they were, which is honest — nothing
    was settled — and a later ``delivery_status`` reconciles them.
    """
    try:
        _reconcile_deliveries_unchecked(session_id, agent_name, agent)
    except DeliveryStoreError:
        logger.debug("Delivery store write failed during kill", exc_info=True)


def _reconcile_deliveries_unchecked(
    session_id: str, agent_name: str, agent: dict | None
) -> None:
    """Body of :func:`_reconcile_deliveries_for_target`."""
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        for record in list(txn.data.values()):
            if record.get("to") != agent_name or is_terminal(record):
                continue
            outcome = _scan_for_nonce(
                session_id,
                _scan_target(record, agent),
                str(record.get("nonce") or ""),
            )
            if outcome == SCAN_FOUND:
                settle(record, STATUS_DELIVERED, reason="", now=time.time())
            elif outcome == SCAN_ABSENT:
                # A complete authoritative negative against a target that is
                # being killed: definite non-delivery, and terminal.
                settle(
                    record,
                    STATUS_FAILED,
                    reason=REASON_NOT_DELIVERED,
                    now=time.time(),
                )
            else:
                # Unreadable or ambiguous. Killing the target does not turn a
                # scan we could not complete into proof the prompt never
                # arrived; that is the false status this feature removes.
                mark_phase(record, PHASE_UNCONFIRMED)
            txn.touch()


def _create_session() -> str:
    sid = str(uuid.uuid4())
    base = _session_dir(sid)
    (base / "mcp").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    _save_agents_unlocked(sid, [])
    _persist_session_binding(sid)
    return sid


def _write_mcp_config(session_id: str, agent_name: str, parent_name: str) -> Path:
    """Write per-agent MCP config (used by Claude Code via --mcp-config)."""
    config = {
        "mcpServers": {
            "win-agent-teams": {
                "command": sys.executable,
                "args": ["-m", "claude_teams.server_simple"],
                "env": {
                    "AGENT_SESSION_ID": session_id,
                    "AGENT_NAME": agent_name,
                    "AGENT_PARENT_NAME": parent_name,
                },
            }
        }
    }
    path = _session_dir(session_id) / "mcp" / f"{agent_name}.mcp.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _write_pi_mcp_config(session_id: str, agent_name: str, parent_name: str) -> Path:
    """Write per-agent pi MCP config with LITERAL identity (via ``--mcp-config``).

    Mirrors :func:`_write_mcp_config` but (a) uses a distinct filename
    (``<agent>.pi.mcp.json``) so it can never collide with the Claude file, and
    (b) adds the pi-adapter-specific keys the interpolated ``~/.pi/agent/mcp.json``
    used: a server-entry ``directTools`` (per review F4 the adapter reads it at
    the server-entry level) and ``CLAUDE_TEAMS_PERMISSION_MODE=bypass``.

    The env values are written as literals (no ``${...}`` interpolation) so a
    spawned pi worker's identity survives the pi-mcp-adapter config merge even if
    a lower-precedence source carries an empty ``AGENT_*`` env block. Note this
    file only reliably owns keys that higher-precedence project sources
    (``<cwd>/.mcp.json``, ``<cwd>/.pi/mcp.json``) do not themselves declare; an
    ``AGENT_*`` env block in those sources would still shallow-override it (that
    is what the fail-loud identity guard defends against).
    """
    config = {
        "mcpServers": {
            "win-agent-teams": {
                "command": sys.executable,
                "args": ["-m", "claude_teams.server_simple"],
                "env": {
                    "AGENT_SESSION_ID": session_id,
                    "AGENT_NAME": agent_name,
                    "AGENT_PARENT_NAME": parent_name,
                    "CLAUDE_TEAMS_PERMISSION_MODE": "bypass",
                },
                "directTools": True,
            }
        }
    }
    path = _session_dir(session_id) / "mcp" / f"{agent_name}.pi.mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _needs_prompt_file(prompt: str) -> bool:
    """Return whether a Claude prompt is too CLI-sensitive for argv transport.

    Applied to the **user prompt only**, before any correlation marker is
    appended (see :func:`_materialize_prompt`).
    """
    return any(char in prompt for char in _CLAUDE_PROMPT_FILE_CHARS)


def _materialize_prompt(
    session_id: str,
    agent_name: str,
    backend_name: str,
    prompt: str,
    correlation_id: str | None,
    *,
    file_token: str,
    delivery_nonce: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the final prompt the agent will see, and choose its transport.

    The backend cannot inject the correlation marker for Claude Code: the
    sidecar is written here, before ``backend.spawn``/``backend.resume``, and
    ``ClaudeCodeBackend._prompt_arg`` then replaces argv with a fixed
    "read this file" instruction. So the server owns materialization.

    Two rules are load-bearing:

    1. Transport is decided from the **user prompt alone**, before the marker
       is appended. Testing the marked prompt instead would route every Claude
       spawn through a file read, since the multi-line marker form always
       introduces a newline.
    2. The marker form differs per transport. Argv gets a **single-line**
       marker, which introduces no CLI-sensitive character and so respects
       ``_CLAUDE_PROMPT_FILE_CHARS`` rather than bypassing it. The sidecar gets
       the newline-delimited form.

    Codex is left untouched here: its backend appends the same server-issued id
    itself (``CodexBackend._correlated_prompt``), so marking here too would
    give Codex two markers.

    A5: the sidecar path carries ``file_token`` so it is unique per call.
    ``delivery_nonce`` additionally appends the A4 delivery marker — supplied
    for a follow-up, whose delivery must be confirmed, and omitted for a spawn,
    which has nothing to confirm against.
    """
    if backend_name != "claude-code":
        if delivery_nonce:
            # Codex quotes the prompt verbatim, so the marker travels in the
            # prompt itself; the newline form matches its correlation marker.
            prompt = delivered_prompt(prompt, delivery_nonce, single_line=False)
        return prompt, {}
    use_file = _needs_prompt_file(prompt)
    if correlation_id:
        prompt = correlated_prompt(prompt, correlation_id, single_line=not use_file)
    if delivery_nonce:
        prompt = delivered_prompt(prompt, delivery_nonce, single_line=not use_file)
    if not use_file:
        return prompt, {}
    path = _attempt_prompt_file(session_id, agent_name, file_token)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return prompt, {"prompt_file_path": str(path)}


def _pi_binding_extra(
    backend_name: str, session_id: str, agent_name: str
) -> dict[str, str]:
    """Return the record field naming a pi agent's transcript directory.

    Pi is the one backend whose storage location the server chooses, so the
    reader cannot find the transcripts from ``cwd`` the way it can for Claude
    and Codex. Persisting the path here (rather than re-deriving it at read
    time) keeps the layout owned by one place and lets a record that predates
    the field be refused instead of guessed. Empty for every other backend.
    """
    if backend_name != "pi":
        return {}
    return {PI_SESSION_DIR_FIELD: str(_pi_session_dir(session_id, agent_name))}


def _correlation_extra(correlation_id: str | None) -> dict[str, str]:
    """Return the ``SpawnRequest.extra`` entry carrying the correlation id.

    ``SpawnRequest`` has no dedicated field, so the id travels in ``extra``.
    Absent means the record predates correlation; the key is then omitted
    rather than filled with an invented id.
    """
    return {CORRELATION_FIELD: correlation_id} if correlation_id else {}


def _prompt_transport(prompt_extra: dict[str, str]) -> str:
    """Return the transport that carried the final prompt, for the record.

    Gate 0 of the binding ladder needs this: a sidecar launch cannot show its
    correlation marker in the transcript until the agent has read the file, so
    zero token matches means "not yet" rather than "never". An argv launch
    carries the marker directly and gets no such grace period.
    """
    return PROMPT_TRANSPORT_SIDECAR if prompt_extra.get("prompt_file_path") else "argv"


def _ensure_pi_mcp_config() -> None:
    """Ensure ``~/.pi/agent/mcp.json`` exposes the win-agent-teams MCP server.

    The ``pi-mcp-adapter`` package reads this file to learn which stdio MCP
    servers to start. We write (idempotently, self-healing on interpreter
    moves) a ``win-agent-teams`` entry whose ``env`` uses ``${AGENT_*}``
    interpolation — resolved from each pi process's own environment
    (``PiBackend.build_env``) — so one shared static file binds every agent to
    its own identity with no per-spawn file and no race. Any other user-defined
    ``mcpServers`` in the file are preserved.
    """
    desired = {
        "command": sys.executable,
        "args": ["-m", "claude_teams.server_simple"],
        "env": {
            "AGENT_SESSION_ID": "${AGENT_SESSION_ID}",
            "AGENT_NAME": "${AGENT_NAME}",
            "AGENT_PARENT_NAME": "${AGENT_PARENT_NAME}",
            "CLAUDE_TEAMS_PERMISSION_MODE": "bypass",
        },
    }
    path = Path.home() / ".pi" / "agent" / "mcp.json"
    config: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, json.JSONDecodeError):
            config = {}
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    if servers.get("win-agent-teams") == desired:
        return
    servers["win-agent-teams"] = desired
    config["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _pi_state_extension_dir() -> Path | None:
    """Return the bundled pi state-reporting extension dir, if present.

    Overridable via ``WIN_AGENT_TEAMS_PI_EXTENSION`` (a file or directory pi's
    ``-e`` accepts); otherwise resolved relative to the repo checkout at
    ``pi-extensions/win-agent-teams-state``. Returns ``None`` when neither
    exists so the spawn still proceeds (state degrades to liveness only).
    """
    override = os.environ.get("WIN_AGENT_TEAMS_PI_EXTENSION", "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "pi-extensions" / "win-agent-teams-state"
    return candidate if candidate.exists() else None


def _pi_wake_extension_dir() -> Path | None:
    """Return the bundled pi inbox-wake extension dir, if present.

    Overridable via ``WIN_AGENT_TEAMS_PI_WAKE_EXTENSION`` (a file or directory
    pi's ``-e`` accepts); otherwise resolved relative to the repo checkout at
    ``pi-extensions/win-agent-teams-wake``. Returns ``None`` when neither
    exists so the spawn still proceeds (the lead simply is not auto-woken and
    must poll its inbox instead).
    """
    override = os.environ.get("WIN_AGENT_TEAMS_PI_WAKE_EXTENSION", "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "pi-extensions" / "win-agent-teams-wake"
    return candidate if candidate.exists() else None


def _hook_extra(session_id: str, agent_name: str, backend_name: str) -> dict[str, str]:
    """Materialise per-backend hook wiring, added to ``SpawnRequest.extra``.

    Claude Code gets a written settings-file path
    (``extra["hooks_settings_path"]``); Codex gets a JSON-encoded ``-c``
    override argv (``extra["hook_overrides"]``) evaluated only when
    ``WIN_AGENT_TEAMS_STATE_HOOKS_CODEX`` is on (see ``CodexBackend``). Pi gets
    the win-agent-teams MCP server registered for the ``pi-mcp-adapter`` and, if
    state hooks are enabled, the paths to its bundled state-reporting extension
    (``extra["pi_state_extension_path"]``) and inbox-wake extension
    (``extra["pi_wake_extension_path"]``), both loaded via ``-e``.

    Note: ``WIN_AGENT_TEAMS_STATE_HOOKS=0`` is a single kill switch for BOTH pi
    extensions — it disables state reporting AND the inbox-wake extension,
    since the early return happens before either path is added.
    """
    session_dir = _session_dir(session_id)
    if backend_name == "pi":
        _ensure_pi_mcp_config()
        # Written BEFORE the state-hooks kill switch so identity is never gated
        # behind it: the per-agent literal --mcp-config file is the primary fix
        # for the identity clobber and must exist even with state hooks off.
        extra: dict[str, str] = {
            "pi_mcp_config_path": str(
                _write_pi_mcp_config(session_id, agent_name, IDENTITY)
            )
        }
        if os.environ.get("WIN_AGENT_TEAMS_STATE_HOOKS", "1").strip() == "0":
            return extra
        ext = _pi_state_extension_dir()
        if ext:
            extra["pi_state_extension_path"] = str(ext)
        wake = _pi_wake_extension_dir()
        if wake:
            extra["pi_wake_extension_path"] = str(wake)
        return extra
    if backend_name == "claude-code":
        settings_path = hooks.write_claude_settings(session_dir, agent_name)
        return {"hooks_settings_path": str(settings_path)}
    if backend_name == "codex":
        # On Windows, Codex runs the hook via `cmd /C` and mangles a multi-token
        # double-quoted command; a bare-path launcher in `commandWindows` is the
        # cmd-safe form (see hooks.write_codex_launcher). On Linux the POSIX
        # `command` runs fine under `sh -c`, so no launcher is needed.
        launcher = (
            str(hooks.write_codex_launcher(session_dir, agent_name))
            if os.name == "nt"
            else None
        )
        overrides = hooks.codex_hook_overrides(
            session_dir, agent_name, windows_launcher=launcher
        )
        return {"hook_overrides": json.dumps(overrides)}
    return {}


@_register_tool()
async def create_join_ticket(name: str, note: str = "") -> dict:
    """Issue a replayable recovery credential for one external member.

    The returned 32-lowercase-hex ``token`` is pasted into a separate,
    interactive session's first ``join_team(session_id=..., token=...)`` call.
    During the retention window, replaying that join credential recovers the
    same membership and member token idempotently; it is not a strictly
    one-time secret. The ticket is stored in ``join-tickets.json`` under the
    active session directory, atomically and under the same cross-process lock
    as ``agents.json``. Open, unexpired tickets reserve their safe member name
    against both later tickets and ``spawn_agent``.

    ``name`` must match ``[A-Za-z0-9_-]{1,64}`` before de-duplication. The
    stored row is ``{ticket_id, name, token, parent, note, created_at,
    expires_at, status, used_at, member_name}``; ``parent`` is this caller's
    resolved identity. TTL defaults to 24 hours and retention for used/expired
    audit rows defaults to seven days. Strict finite-positive overrides are
    ``WIN_AGENT_TEAMS_JOIN_TICKET_TTL_SECONDS`` and
    ``WIN_AGENT_TEAMS_JOIN_TICKET_RETENTION_SECONDS``; invalid, zero,
    negative, NaN, or infinite values use the defaults. Store writes prune
    audit rows beyond retention, while retained expired rows no longer reserve
    names.

    ``note`` is rendered literally inside a Markdown fence chosen so its
    contents cannot close the block. The paste-ready ``join_prompt`` contains
    the complete join/read/send/restart protocol and an optional watcher
    command. The joining member must save the returned ``member_token`` and use
    only ``external_send``, ``external_read``, and ``leave_team`` with it.
    For client-surface isolation, run those calls through a separate MCP entry
    with ``WIN_AGENT_TEAMS_EXTERNAL_ONLY=1``; that entry registers only the
    four member tools plus ``list_backends``.
    """

    def _create() -> dict:
        refusal = _require_resolved_identity()
        if refusal is not None:
            return refusal
        requested = name.strip()
        if not hooks._SAFE_AGENT_RE.fullmatch(requested):
            return {"success": False, "reason": "invalid_name"}
        session_id = _active_session_id(create=True)
        now = time.time()
        with _agents_file_lock(session_id):
            agents = _load_agents_unlocked(session_id)
            tickets = _load_join_tickets_unlocked(session_id)
            reserved_name = _unique_reserved_name(
                requested, agents, tickets, now=now
            )
            ticket = {
                "ticket_id": uuid.uuid4().hex,
                "name": reserved_name,
                "token": uuid.uuid4().hex,
                "parent": IDENTITY,
                "note": note,
                "created_at": now,
                "expires_at": now + _join_ticket_ttl_seconds(),
                "status": "open",
                "used_at": None,
                "member_name": None,
            }
            tickets.append(ticket)
            _save_join_tickets_unlocked(session_id, tickets, now=now)
        return {
            "success": True,
            "ticket_id": ticket["ticket_id"],
            "name": reserved_name,
            "session_id": session_id,
            "token": ticket["token"],
            "expires_at": ticket["expires_at"],
            "join_prompt": _build_join_prompt(
                session_id=session_id,
                token=str(ticket["token"]),
                name=reserved_name,
                parent=IDENTITY,
                note=note,
            ),
        }

    return await run_blocking(_create)


@_register_tool(external=True)
async def join_team(session_id: str, token: str) -> dict:
    """Join a lead's team as a pull-only external member.

    Call this first from a manually started interactive session, using the
    literal ``session_id`` and 32-lowercase-hex join ``token`` from the lead's
    paste-ready prompt. The join credential is a replayable recovery credential
    during ticket retention: replay returns the same membership and
    ``member_token`` after an MCP restart. Save that member token; its exact
    grammar is ``wam1:<canonical-lowercase-session-uuid>:<64-lowercase-hex>``.
    ``agents.json`` stores only its SHA-256 digest, although a same-user reader
    of ``join-tickets.json`` can reconstruct it from the retained ticket.

    The registry row is ``{name, pid, backend:"external", session_id, parent,
    status:"running", spawned_at, cwd, model:null, member_token_digest,
    join_ticket_id, spawned_by, spawned_by_source:"join_ticket"}``. The secret
    is deterministically derived as
    ``sha256("wam-member:<ticket_id>:<ticket-token>")`` and the stored digest is
    ``sha256(secret)``. No public list/status/check result exposes that digest.

    The returned paths name ``inbox-<member>.jsonl`` and
    ``state-<member>.json`` in the joined session. External messaging is pull
    based and unconfirmed: use ``external_read`` to drain work and
    ``external_send`` to report to the ticket's parent. ``pid`` in the registry
    is informational and may be stale after Desktop restarts. A lead kill
    deregisters the membership and never signals that PID; leave/kill revocation
    is observed on the next token-bearing call.

    Join reconciliation is crash-safe across the registry/ticket/marker write
    windows. A missing or schema-invalid joined marker is repaired, while any
    valid newer activity marker is preserved. If the contractual marker cannot
    be written, this returns ``marker_write_failed`` with ``retriable=true``;
    replay repairs it without duplicating the registry record. This tool never
    adopts or mutates the ambient process identity or session binding.
    ``WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`` is the supported restricted server
    entry for a separate Desktop profile/client instance; a dual entry in one
    profile leaves ambient root tools selectable and is not isolation.
    """

    def _join() -> dict:  # noqa: PLR0911 - reconciliation state-machine cells.
        sid, refusal = _validate_join_session_id(session_id)
        if refusal is not None:
            return refusal
        now = time.time()
        with _agents_file_lock(sid):
            tickets = _load_join_tickets_unlocked(sid)
            ticket = next(
                (
                    row
                    for row in tickets
                    if isinstance(token, str)
                    and isinstance(row.get("token"), str)
                    and row.get("token") == token
                ),
                None,
            )
            if ticket is None:
                return {"success": False, "reason": "invalid_or_expired_token"}

            ticket_id = ticket.get("ticket_id")
            secret = _member_secret(ticket_id, ticket.get("token"))
            digest = _member_digest(secret)
            agents = _load_agents_unlocked(sid)
            matches = [
                record
                for record in agents
                if record.get("join_ticket_id") == ticket_id
            ]
            if len(matches) > 1:
                return {"success": False, "reason": "registry_corrupt"}
            record = matches[0] if matches else None
            if record is not None and not _external_record_matches(
                record, ticket, sid, digest
            ):
                return {"success": False, "reason": "registry_corrupt"}

            status = ticket.get("status")
            if status not in {"open", "used"}:
                return {"success": False, "reason": "registry_corrupt"}
            if record is not None and record.get("status") not in {
                "running",
                "left",
            }:
                return {"success": False, "reason": "registry_corrupt"}
            if record is not None and record.get("status") == "left":
                return {
                    "success": False,
                    "reason": "membership_revoked",
                    "detail": "left",
                }

            if status == "used" and record is None:
                return {"success": False, "reason": "token_already_used"}

            if status == "open" and record is None:
                expires_at = ticket.get("expires_at")
                unexpired = (
                    isinstance(expires_at, int | float)
                    and not isinstance(expires_at, bool)
                    and float(expires_at) > now
                )
                if not unexpired:
                    return {
                        "success": False,
                        "reason": "invalid_or_expired_token",
                    }
                record = {
                    "name": ticket.get("name"),
                    "pid": os.getpid(),
                    "backend": "external",
                    "session_id": sid,
                    "parent": ticket.get("parent"),
                    "status": "running",
                    "spawned_at": now,
                    "cwd": str(Path.cwd()),
                    "model": None,
                    "member_token_digest": digest,
                    "join_ticket_id": ticket_id,
                    SPAWNED_BY_FIELD: ticket.get("parent"),
                    SPAWNED_BY_SOURCE_FIELD: SPAWNED_BY_SOURCE_JOIN,
                }
                agents.append(record)
                _save_agents_unlocked(sid, agents)

            if status == "open":
                # This is either a fresh join or crash window A. Expiry is
                # intentionally irrelevant once the durable record exists.
                ticket["status"] = "used"
                ticket["used_at"] = now
                ticket["member_name"] = ticket.get("name")
                _save_join_tickets_unlocked(sid, tickets, now=now)

            name = str(ticket.get("name") or "")
            if not _ensure_joined_marker(sid, name):
                return {
                    "success": False,
                    "reason": "marker_write_failed",
                    "retriable": True,
                }
            return _external_membership_result(sid, ticket, secret)

    return await run_blocking(_join)


@_register_tool()
async def spawn_agent(
    prompt: str,
    name: str = "",
    backend: str = "",
    model: str = "",
    cwd: str = "",
    permission_mode: str = "bypass",
    reasoning_effort: str = "",
    expected_outputs: list[str] | None = None,
) -> dict:
    """Spawn a new agent process.

    model: pick by how much capability the task needs, not by a model name.
    For codex, choose one capability tier (each maps to a GPT-5.6 model at a
    fixed reasoning effort), cheapest first:
      - ``low``    -> quick, low-stakes tasks
      - ``medium`` -> token-efficient general default
      - ``high``   -> backend development, code review
      - ``xhigh``  -> genuinely hard problems
      - ``ultra``  -> the hardest problems (top tier)
    Spawning errors if the required GPT-5.6 model is not available on this
    machine (upgrade codex / check account access) — there is no silent
    downgrade. For claude-code, ``model`` is haiku/sonnet/opus.

    reasoning_effort: for claude-code, sets the effort (low/medium/high/xhigh/
    max). For codex it is silently ignored when ``model`` is a capability tier
    (the tier owns the effort); it still applies to a blank/raw-slug codex
    model. Codex accepts low/medium/high/xhigh, plus max/ultra.

    expected_outputs (optional): the exact file paths you are instructing the
    agent to create. Echoed back verbatim in the result so you can watch
    those precise paths for completion.

    The result includes the disk-backed coordination contract: an injected
    lifecycle hook writes this agent's state to ``state_marker_path``
    (absolute path to ``state-{name}.json``) on every state transition. The
    marker JSON schema is ``{"state": "running" | "waiting", "event":
    "<hook>", "ts": <float epoch seconds>}``. This file is on disk and
    survives MCP server restarts — the server auto-restarts on the next tool
    call, so you can always re-query even after it died from host
    inactivity.

    The result also includes ``session_dir``, shell-neutral ``watch_argv``, and
    the shell-specific ``watch_command_bash`` and
    ``watch_command_powershell`` renderings.

    Recommended coordination pattern: do NOT tight-poll. Run the returned
    ``watch_argv`` directly, or use the returned Bash/PowerShell rendering.
    Use a background watch for Claude Code and a bounded foreground watch for
    Codex. The watcher is one-shot, so re-arm it after every wake. Re-check
    status after timeout exit 2 before mounting the next watch.
    """

    def _do_spawn() -> dict:
        refusal = _require_resolved_identity()
        if refusal is not None:
            # An unresolved child must never create an orphan session or launch
            # a subprocess with the sentinel identity in its env.
            return refusal
        session_id = _active_session_id(create=True)
        with _agents_transaction(session_id) as agents:
            tickets = _load_join_tickets_unlocked(session_id)
            agent_name = _unique_reserved_name(
                name, agents, tickets, now=time.time()
            )

            backend_name = backend.strip() or registry.default_backend()
            b = registry.get(backend_name)

            effort = reasoning_effort.strip() or None
            # A backend may bundle a reasoning effort into a model tier (e.g.
            # Codex ``high`` -> Sol @ medium), so resolve model and effort
            # together. For codex tiers the bundled effort wins and any caller
            # ``reasoning_effort`` is ignored; other backends still honor it.
            resolved_model, effort = b.resolve_launch(model, effort)

            mcp_config_path = _write_mcp_config(session_id, agent_name, IDENTITY)

            agent_cwd = cwd.strip() or str(Path.cwd())
            # Generated before backend.spawn: the id must already be inside the
            # final initial prompt, which is materialized on the next line.
            correlation_id = new_correlation_id()
            # A5: collect anything a previous failed spawn/resume orphaned
            # before adding to the directory ourselves.
            _gc_stale_prompt_files(session_id)
            # A5: even a spawn gets a unique prompt-file path, so a spawn and a
            # concurrent follow-up for the same name cannot collide. No
            # delivery marker: a spawn either produced a PID or raised, so
            # there is nothing for A4 to confirm against.
            final_prompt, prompt_extra = _materialize_prompt(
                session_id,
                agent_name,
                backend_name,
                prompt,
                correlation_id,
                file_token=new_delivery_nonce(),
            )
            extra = {
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
                "session_dir": str(_session_dir(session_id)),
                **_correlation_extra(correlation_id),
                **prompt_extra,
                **_hook_extra(session_id, agent_name, backend_name),
            }

            request = SpawnRequest(
                agent_id=f"{agent_name}@{session_id}",
                name=agent_name,
                team_name=session_id,
                prompt=final_prompt,
                model=resolved_model,
                agent_type="worker",
                color="blue",
                cwd=agent_cwd,
                lead_session_id=IDENTITY,
                permission_mode=cast(
                    'Literal["default", "require_approval", "bypass"]',
                    permission_mode,
                ),
                reasoning_effort=effort,
                extra=extra,
            )

            result = b.spawn(request)
            pid = int(result.process_handle)
            # Capture the PID's creation token now, from the just-spawned live
            # process, so a later reused PID (after a host restart) can be told
            # apart. Captured here (not by reopening the PID cold) per the
            # plan's N1: the child is live and in-memory at this instant.
            create_token = process_manager.creation_token(str(pid))

            agents.append(
                {
                    "name": agent_name,
                    "pid": pid,
                    "backend": backend_name,
                    "session_id": session_id,
                    # The spawning lead's identity. Mirrors the AGENT_PARENT_NAME
                    # env propagated into the child's MCP config (which is also
                    # this server's IDENTITY at spawn time); recorded here so the
                    # lead-wake hook can scope "live subagents" to a caller's own
                    # children instead of every agent in the shared session.
                    "parent": IDENTITY,
                    "status": "running",
                    "spawned_at": time.time(),
                    "cwd": agent_cwd,
                    "model": resolved_model,
                    "permission_mode": permission_mode,
                    "reasoning_effort": effort,
                    "create_token": create_token,
                    CORRELATION_FIELD: correlation_id,
                    PROMPT_TRANSPORT_FIELD: _prompt_transport(prompt_extra),
                    **_pi_binding_extra(backend_name, session_id, agent_name),
                    # C1/R2 — who spawned this agent, captured here because
                    # this is the only moment it is observed rather than
                    # asserted. ``follow_up_agent`` refuses any other caller.
                    SPAWNED_BY_FIELD: IDENTITY,
                    SPAWNED_BY_SOURCE_FIELD: SPAWNED_BY_SOURCE_SPAWN,
                }
            )
            _save_agents_transaction(session_id, agents)

        session_dir = str(_session_dir(session_id))
        return {
            "name": agent_name,
            "pid": pid,
            "backend": backend_name,
            "session_id": session_id,
            "state_marker_path": str(_state_marker_file(session_id, agent_name)),
            "session_dir": session_dir,
            "watch_argv": _watch_argv(session_dir),
            "watch_command_bash": _watch_command_bash(session_dir),
            "watch_command_powershell": _watch_command_powershell(session_dir),
            "expected_outputs": list(expected_outputs) if expected_outputs else [],
        }

    return _annotate(await run_blocking(_do_spawn))


@_register_tool()
async def send_message(
    text: str, to: str = "team-lead", idempotency_key: str = ""
) -> dict:
    """Send a message one hop along the spawn edge: to your spawner, or a child.

    ``to`` defaults to ``"team-lead"``, which reaches the agent that spawned you —
    that is almost always what you want from a subagent.

    The recipient decides the path, and the two paths behave differently:

    * **Your spawner** (the default) is written to its inbox. Its watcher wakes
      it; it consumes the message with read_messages. This is the upstream path
      and it is unchanged.
    * **An external member you registered** receives one inbox line and returns
      ``delivery="inbox"``. This is pull-based and unconfirmed: it reads the
      message on its next ``external_read`` call. No idempotency key, lease,
      durable delivery row, process resume, or wake is involved.
    * **A spawned agent you spawned** goes through the guaranteed path instead
      — the same machinery as follow_up_agent, because a spawned child that is
      not polling would otherwise never see an inbox write. It therefore needs
      an idempotency_key you choose before the call, and it returns the same
      "delivered"/"failed"/"queued" statuses. Such a spawned-agent message
      does NOT enter the recipient's inbox, so read_messages cannot repeat it.

    Anyone else — a sibling, a grandchild, another lead's worker, or a name that
    matches no agent — is REFUSED, and nothing is sent. A misspelled recipient
    used to be re-routed to your lead with a warning; it no longer is, because
    that turned a typo into a real-looking upstream message.
    """
    refusal = _require_resolved_identity()
    if refusal is not None:
        return {**refusal, "to": to}
    session_id = _active_session_id()

    def _do_send() -> dict:  # noqa: PLR0911 - recipient-class contract cells.
        if not session_id:
            return {"success": False, "to": to, "reason": "session_not_found"}

        recipient_class, recipient = _classify_recipient(to, session_id)

        if recipient_class == RECIPIENT_CHILD:
            # Re-resolve and act under one registry transaction. Kill, leave,
            # and token-bearing member operations take this same lock, so a
            # validated append cannot occur after revocation removed/left the
            # record.
            with _agents_transaction(session_id) as agents:
                target = _find_agent(agents, recipient)
                if (
                    target is not None
                    and target.get("backend") == "external"
                    and target.get(SPAWNED_BY_FIELD) == IDENTITY
                ):
                    if target.get("status") == "left":
                        return {
                            "success": False,
                            "to": recipient,
                            "reason": "member_left",
                            "retriable": False,
                        }
                    if target.get("status") != "running":
                        return {
                            "success": False,
                            "to": recipient,
                            "reason": "membership_revoked",
                            "retriable": False,
                        }
                    line = json.dumps(
                        {
                            "from": IDENTITY,
                            "text": text,
                            "ts": datetime.now(UTC).isoformat(),
                        }
                    )
                    with _inbox_file(session_id, recipient).open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(line + "\n")
                    return {
                        "success": True,
                        "to": recipient,
                        "delivery": "inbox",
                        "note": (
                            "External agent; pull-based and unconfirmed. It reads "
                            "this inbox on its next external_read call."
                        ),
                    }
            # R5: the only accept-then-drop risk left is a child that is not
            # polling, so a downstream send is the guaranteed path, never an
            # inbox append. B4 stays intact — the audit row is the record, and
            # the inbox is untouched, so the text is presented exactly once.
            return _guaranteed_send(
                session_id,
                recipient,
                text,
                idempotency_key,
                True,
                tool="send_message",
            )

        if recipient_class != RECIPIENT_SPAWNER:
            detail = _UNADDRESSABLE_DETAIL[recipient_class].format(
                spawner=_spawner_target()
            )
            return {
                "success": False,
                "to": to,
                "reason": "recipient_not_addressable",
                "recipient_class": recipient_class,
                "retriable": False,
                "detail": f"{to!r} {detail}",
            }

        inbox = _inbox_file(session_id, recipient)
        line = json.dumps(
            {
                "from": IDENTITY,
                "text": text,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        with inbox.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return {"success": True, "to": recipient}

    return _annotate(await run_blocking(_do_send))


_DEFAULT_READ_LIMIT = 50


def _read_inbox(
    session_id: str,
    reader: str,
    *,
    from_agent: str = "",
    since_seq: int | None = None,
    full: bool = False,
    limit: int | None = None,
    max_chars: int | None = None,
) -> dict:
    """Read and advance one inbox; the caller supplies the serialization."""
    inbox = _inbox_file(session_id, reader)
    cursor_file = _inbox_cursor_file(session_id, reader)
    with _inbox_lock(reader):
        cursors = _load_inbox_cursors(cursor_file)
        by_sender = read_inbox_by_sender(inbox)
        original_cursors = dict(cursors)
        for sender in list(cursors):
            observed = len(by_sender.get(sender, []))
            cursors[sender] = min(cursors[sender], observed)

        relevant = [from_agent] if from_agent else list(by_sender)
        start_overrides: dict[str, int] = {}
        if since_seq is not None and from_agent:
            floor = cursors.get(from_agent, 0)
            start_overrides[from_agent] = max(since_seq, floor, 0)

        per_sender_batches: dict[str, list[tuple[int, int, dict]]] = {}
        for sender in relevant:
            entries = by_sender.get(sender, [])
            start = start_overrides.get(sender, cursors.get(sender, 0))
            per_sender_batches[sender] = [
                (position, index, msg)
                for position, (index, msg) in enumerate(entries)
                if position >= start
            ]

        effective_limit = (
            None if full else (_DEFAULT_READ_LIMIT if limit is None else limit)
        )
        selected: list[tuple[str, int, int, dict]] = [
            (sender, position, index, msg)
            for sender in relevant
            for position, index, msg in per_sender_batches[sender]
        ]
        selected.sort(key=lambda item: item[2])
        total_unread = len(selected)
        has_more = False
        if effective_limit is not None and len(selected) > effective_limit:
            selected = selected[:effective_limit]
            has_more = True

        updated = dict(cursors)
        for sender in relevant:
            entries = by_sender.get(sender, [])
            consumed_positions = [
                position
                for selected_sender, position, _, _ in selected
                if selected_sender == sender
            ]
            if consumed_positions:
                new_count = max(consumed_positions) + 1
            else:
                new_count = start_overrides.get(sender, cursors.get(sender, 0))
            floor = max(cursors.get(sender, 0), start_overrides.get(sender, 0))
            new_count = max(new_count, floor)
            if entries or sender in cursors or sender in start_overrides:
                updated[sender] = min(new_count, len(entries))

        # limit=0 is a non-consuming watermark and must leave the cursor file
        # byte-identical, including when loading revealed a clamp opportunity.
        if effective_limit == 0:
            updated = original_cursors
        else:
            _save_inbox_cursors(cursor_file, updated)

        messages: list[dict] = []
        for sender, position, _index, message in selected:
            text, truncated, full_len = _truncate(message.get("text"), max_chars)
            entry = {
                "from": sender,
                "text": text,
                "ts": message.get("ts"),
                "seq": position + 1,
            }
            if max_chars is not None:
                entry["truncated"] = truncated
                entry["full_len"] = full_len
            messages.append(entry)

        result: dict = {
            "messages": messages,
            "unread_count": total_unread,
            "has_more": has_more,
        }
        if from_agent:
            result["cursors"] = None
            result["seq"] = updated.get(from_agent, cursors.get(from_agent, 0))
        else:
            result["cursors"] = updated
            result["seq"] = None
        return result


@_register_tool()
async def read_messages(
    from_agent: str = "",
    since_seq: int | None = None,
    full: bool = False,
    limit: int | None = None,
    max_chars: int | None = None,
) -> dict:
    """Read unread messages from own inbox, delta-by-default with a watermark.

    Returns ``{messages, cursors, seq, unread_count, has_more}``. Each
    message is ``{from, text, ts, seq}`` where ``seq`` is that sender's
    1-based per-sender COUNT after this message (i.e. the sender's
    high-water mark once this message is consumed) — the same number space
    as the persisted per-sender cursor. ``cursors`` is the per-sender
    high-water map; a scalar ``seq`` (instead of ``cursors``) is returned
    only when ``from_agent`` is set.

    Default (no ``since_seq``) drains and advances the cursor as before,
    now also returning the watermark. With ``from_agent`` set, pass
    ``since_seq`` to fetch only that sender's messages with index
    ``>= since_seq`` and advance the persisted cursor to
    ``max(current, since_seq)`` (no rewind, no re-delivery/skip at the
    boundary). ``since_seq`` is invalid without ``from_agent``.

    ``limit`` bounds the batch (default 50 when ``full=False``); ``has_more``
    is set when the batch was clipped. ``full=True`` ignores ``limit``.
    ``limit=0`` is a no-body watermark poll: it returns an empty
    ``messages`` list without advancing any cursor past what was already
    read, while ``cursors``/``seq``/``has_more`` still reflect the current
    state. Negative ``limit`` values raise ``ValueError``.
    ``max_chars`` truncates each message's ``text`` (``truncated``/
    ``full_len`` added per message) when set.
    """
    if since_seq is not None and not from_agent:
        msg = "since_seq requires from_agent to be set"
        raise ValueError(msg)
    if limit is not None and limit < 0:
        msg = "limit must not be negative"
        raise ValueError(msg)

    refusal = _require_resolved_identity()
    if refusal is not None:
        # Refuse before any inbox read/cursor advance so a mis-identified child
        # never consumes (marks read) the lead's inbox.
        return refusal

    session_id = _active_session_id()

    def _do_read() -> dict:
        if not session_id:
            return {
                "messages": [],
                "cursors": {},
                "seq": None,
                "unread_count": 0,
                "has_more": False,
            }
        return _read_inbox(
            session_id,
            IDENTITY,
            from_agent=from_agent,
            since_seq=since_seq,
            full=full,
            limit=limit,
            max_chars=max_chars,
        )

    return _annotate(await run_blocking(_do_read))


@_register_tool(external=True)
async def external_send(member_token: str, text: str) -> dict:
    """Append a pull-only external member message to its parent's inbox.

    ``member_token`` must exactly match
    ``wam1:<canonical-lowercase-session-uuid>:<64-lowercase-hex>``. Every call
    re-derives authority from the external registry record; leave or lead-side
    kill is therefore enforced on the next call without relying on ambient MCP
    identity or session state. The message line is
    ``{"from": <member-name>, "text": <text>, "ts": <ISO timestamp>}`` in
    ``inbox-<parent>.jsonl``. Success means only that the line reached the
    parent's inbox: delivery is pull-based and unconfirmed.

    The session's cross-process registry lock is held through token validation,
    inbox append, and activity-marker write, linearizing the operation against
    kill and leave. Lock waiters time out after 30 seconds on Windows and block
    in POSIX ``flock``. If the append succeeds but the opportunistic
    ``running/activity`` marker write fails, the result remains successful and
    includes ``heartbeat_warning=true`` so retrying cannot duplicate a message.
    """

    def _send() -> dict:
        with _member_operation(member_token) as (
            record,
            _agents,
            session_id,
            refusal,
        ):
            if refusal is not None:
                return refusal
            if record is None:
                return {"success": False, "reason": "membership_revoked"}
            name = str(record.get("name") or "")
            parent = str(record.get("parent") or record.get(SPAWNED_BY_FIELD) or "")
            line = json.dumps(
                {
                    "from": name,
                    "text": text,
                    "ts": datetime.now(UTC).isoformat(),
                }
            )
            with _inbox_file(session_id, parent).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            result = {
                "success": True,
                "to": parent,
                "delivery": "inbox",
            }
            try:
                _write_state_marker(
                    session_id, name, state="running", event="activity"
                )
            except OSError:
                result["heartbeat_warning"] = True
            return result

    return await run_blocking(_send)


@_register_tool(external=True)
async def external_read(
    member_token: str,
    from_agent: str = "",
    since_seq: int | None = None,
    full: bool = False,
    limit: int | None = None,
    max_chars: int | None = None,
) -> dict:
    """Drain an external member's own inbox using its bearer token.

    ``member_token`` must exactly match
    ``wam1:<canonical-lowercase-session-uuid>:<64-lowercase-hex>``. Every call
    re-derives the member name and session from ``agents.json``; it never reads
    or mutates ambient identity, active-session, recovery, or binding state.
    Leave or lead-side kill is enforced on the next call.

    Cursor behavior is identical to ``read_messages``. The result contains
    ``messages, cursors, seq, unread_count, has_more``; default calls drain.
    ``from_agent`` scopes one sender; ``since_seq`` requires it. ``limit``
    defaults to 50, negative values raise ``ValueError``, ``full=true`` ignores
    it, and ``limit=0`` is a non-consuming watermark whose cursor file stays
    byte-identical. ``max_chars`` adds per-message ``truncated``/``full_len``.

    The cross-process registry lock remains held through the complete
    load/read/advance/save transaction, then the in-process inbox lock, so two
    old/new MCP processes using the same token consume each line exactly once
    under the existing at-most-once response-loss semantics. Lock waiters time
    out after 30 seconds on Windows and block in POSIX ``flock``; a large read
    can delay registry operations. A failed activity-marker write after cursor
    persistence returns success with ``heartbeat_warning=true``.
    """
    if since_seq is not None and not from_agent:
        msg = "since_seq requires from_agent to be set"
        raise ValueError(msg)
    if limit is not None and limit < 0:
        msg = "limit must not be negative"
        raise ValueError(msg)

    def _read() -> dict:
        with _member_operation(member_token) as (
            record,
            _agents,
            session_id,
            refusal,
        ):
            if refusal is not None:
                return refusal
            if record is None:
                return {"success": False, "reason": "membership_revoked"}
            name = str(record.get("name") or "")
            result = _read_inbox(
                session_id,
                name,
                from_agent=from_agent,
                since_seq=since_seq,
                full=full,
                limit=limit,
                max_chars=max_chars,
            )
            result["success"] = True
            try:
                _write_state_marker(
                    session_id, name, state="running", event="activity"
                )
            except OSError:
                result["heartbeat_warning"] = True
            return result

    return await run_blocking(_read)


@_register_tool(external=True)
async def leave_team(member_token: str) -> dict:
    """Permanently revoke this external membership without killing a process.

    ``member_token`` must exactly match
    ``wam1:<canonical-lowercase-session-uuid>:<64-lowercase-hex>``. Under the
    session's cross-process registry lock, the matching external record changes
    from ``running`` to ``left`` and its marker becomes ``waiting/left``.
    Subsequent send/read calls return ``membership_revoked`` and replaying the
    original join credential cannot resurrect the member.

    The call is idempotent: an already-left member returns success with
    ``already_left=true`` and does not rewrite the marker. The informational
    registry PID is never probed or signalled. Lock waiters time out after 30
    seconds on Windows and block in POSIX ``flock``.
    """

    def _leave() -> dict:
        with _member_operation(member_token, allow_left=True) as (
            record,
            agents,
            session_id,
            refusal,
        ):
            if refusal is not None:
                return refusal
            if record is None or agents is None:
                return {"success": False, "reason": "membership_revoked"}
            name = str(record.get("name") or "")
            if record.get("status") == "left":
                return {
                    "success": True,
                    "name": name,
                    "already_left": True,
                }
            record["status"] = "left"
            _save_agents_unlocked(session_id, agents)
            result = {"success": True, "name": name, "already_left": False}
            try:
                _write_state_marker(session_id, name, state="waiting", event="left")
            except OSError:
                result["heartbeat_warning"] = True
            return result

    return await run_blocking(_leave)


@_register_tool()
@_with_disk_note
async def check_agent(
    name: str, full: bool = False, max_chars: int = _DEFAULT_LAST_LINE_MAX_CHARS
) -> dict:
    """Check an agent's status: state, last line, and unread message count.

    Default (``full=False``) returns a compact status peek: ``{name, state,
    alive, pid, backend, last_activity_at, unread_count, last_line, seq,
    truncated, full_len, heartbeat_age_s, stalled}``. ``state`` is
    ``running``/``waiting``/``idle``/``dead``. ``last_line`` is the last
    non-empty line of the agent's most recent assistant message, clipped to
    ``max_chars`` (default 200); ``truncated`` signals clipping happened and
    ``full_len`` is the untruncated character count. ``unread_count``/``seq``
    count messages FROM this agent addressed to the caller.

    ``heartbeat_age_s`` (float, disk-derived) is seconds since the agent's
    last known activity (marker ``ts`` when available, else transcript
    activity); ``None`` when no activity signal exists yet. ``stalled``
    (bool) is ``True`` only when the agent is alive, its ``state`` is
    neither ``waiting`` nor ``dead``, and ``heartbeat_age_s`` exceeds the
    stall threshold (``STALL_SECONDS``, default 300s, env-overridable via
    ``WIN_AGENT_TEAMS_STALL_SECONDS``). Answers "alive but hung" from disk
    alone, with zero transcript bytes.

    Pass ``full=True`` to restore the full ``last_message`` (bounded to 1000
    chars) and ``backend_session_id`` for follow-up/resume workflows.
    External records return ``backend="external"`` and
    ``binding="not_applicable"``. Their PID is informational/stale and is not
    probed; membership status plus marker activity is the liveness signal.
    """
    session_id = _active_session_id()

    def _do_check() -> dict:
        if not session_id:
            return _empty_agent_check(name, full=full)
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return _empty_agent_check(name, full=full)
            alive = _agent_alive(agent)
            binding = _resolve_agent_binding(agent)
            if _sync_backend_session_id(agent, binding):
                _save_agents_transaction(session_id, agents)
            internal = _agent_check_payload(name, agent, alive, binding)
            view = _compact_check_view(session_id, name, internal, max_chars=max_chars)
            if full:
                view.update(
                    {
                        "last_message": internal.get("last_message"),
                        "backend_session_id": internal.get("backend_session_id"),
                    }
                )
            return view

    return _annotate(await run_blocking(_do_check))


def _build_resume_request(
    session_id: str,
    agent: dict,
    agent_name: str,
    agent_cwd: str,
    backend: Any,
    backend_name: str,
    prompt: str,
    nonce: str,
) -> tuple[object, str, str | None, str | None, SpawnRequest, dict[str, str]]:
    """Materialize this attempt's prompt and build the resume request.

    Extracted from ``follow_up_agent`` so the tool body stays readable; it is
    pure request construction and performs no registry or lease mutation.
    """
    mcp_config_path = _write_mcp_config(session_id, agent_name, IDENTITY)

    # Reuse the concrete model resolved at spawn. Preserve a stored blank
    # verbatim (blank means "defer to codex config"); only a genuinely absent
    # key falls back to the backend default. Do NOT coerce blank via ``or`` —
    # default_model() may be a capability tier name (e.g. "medium"), which must
    # never reach ``-c model``.
    stored_model = agent.get("model")
    model = stored_model if isinstance(stored_model, str) else backend.default_model()
    permission_mode = str(agent.get("permission_mode") or "bypass")
    effort_value = agent.get("reasoning_effort")
    effort = effort_value if isinstance(effort_value, str) else None
    # Carry the spawn-time id forward verbatim. Dropping it here would silently
    # downgrade the agent to ``legacy``, which per R8 means it can never be
    # followed up again. A legacy record stays legacy: no id is minted on
    # resume, because a fresh id would not appear anywhere in the conversation
    # that already exists.
    _, correlation_id = classify_correlation(agent)
    # A5: same sweep as the spawn path — a resume writes a sidecar too, and a
    # resume that raises is exactly how one gets orphaned.
    _gc_stale_prompt_files(session_id)
    final_prompt, prompt_extra = _materialize_prompt(
        session_id,
        agent_name,
        backend_name,
        prompt,
        correlation_id,
        file_token=nonce,
        delivery_nonce=nonce,
    )
    extra = {
        "mcp_config_path": str(mcp_config_path),
        "agent_capability": "",
        # Mirror the spawn path: the pi backend resolves its per-agent session
        # dir (``<session_dir>/pi-sessions/<name>``) and the state-marker env
        # from this. Dropping it here makes resume fall back to ``request.cwd``,
        # so ``--continue`` targets a base that has no rollout and pi exits
        # immediately (``resume_not_confirmed``).
        "session_dir": str(_session_dir(session_id)),
        **_correlation_extra(correlation_id),
        **prompt_extra,
        **_hook_extra(session_id, agent_name, backend_name),
    }
    request = SpawnRequest(
        agent_id=f"{agent_name}@{session_id}",
        name=agent_name,
        team_name=session_id,
        prompt=final_prompt,
        model=model,
        agent_type="worker",
        color="blue",
        cwd=agent_cwd,
        lead_session_id=IDENTITY,
        permission_mode=cast(
            'Literal["default", "require_approval", "bypass"]', permission_mode
        ),
        reasoning_effort=effort,
        extra=extra,
    )
    return model, permission_mode, effort, correlation_id, request, prompt_extra


def _finalize_follow_up(
    session_id: str,
    plan: _FollowUpPlan,
    outcome: DeliveryOutcome,
    new_pid: int | None,
) -> dict:
    """Phase 3 — CAS the record back under the lock, then release the lease.

    The CAS key is generation **and** ``operation_id``. Generation alone is not
    enough: an agent name reused after removal starts a fresh record that can
    legitimately be back at the same generation, and a stale finalize must not
    update it.
    """
    with _agents_transaction(session_id) as agents:
        agent = next((a for a in agents if a["name"] == plan.agent_name), None)
        # The fence is the record's CURRENT generation, not the one frozen into
        # the lease payload. ``lease force`` bumps the generation first and only
        # then terminates the child and clears the lease, with the registry lock
        # released in between; a holder finalizing inside that window still
        # finds its own lease at its own generation, so ``finalize_lease``
        # alone would say yes and the fence would lose its race.
        #
        # Checked BEFORE ``finalize_lease`` so a rejected finalize leaves the
        # lease in place for the operator to inspect and clear, rather than
        # having it dropped by the very attempt the fence just rejected.
        fenced = agent is None or _record_generation(agent) != plan.generation
        won = not fenced and finalize_lease(
            _leases_file(session_id),
            plan.agent_name,
            plan.operation_id,
            plan.generation,
        )
        if agent is None or not won:
            # Fenced out, or the record was replaced underneath us. Never
            # write: the record this attempt described no longer exists.
            return _follow_up_failure(
                "operation_superseded",
                plan.agent_name,
                retriable=True,
                detail=(
                    "This delivery attempt was fenced or its agent record was "
                    "replaced while the resume was in flight."
                ),
            )

        # Only a delivered or still-in-flight attempt writes the record. A
        # resume that never attached (A3) or a child that died without a
        # receipt leaves nothing worth tracking, and R6 forbids describing
        # either as progress — so the record stays exactly as it was.
        if new_pid is None or outcome.status not in {
            DELIVERY_DELIVERED,
            DELIVERY_UNCONFIRMED,
        }:
            if outcome.status == DELIVERY_FAILED:
                # The child is provably gone, so no CLI can still be waiting
                # to read this attempt's prompt file.
                remove_prompt_file(plan.prompt_file)
            if new_pid is None:
                return _follow_up_failure(
                    "resume_failed", plan.agent_name, retriable=True
                )
            return _follow_up_failure(
                outcome.reason or "not_delivered",
                plan.agent_name,
                retriable=False,
                detail=_DELIVERY_FAILURE_DETAIL.get(outcome.reason, ""),
            )

        agent.update(
            {
                "pid": new_pid,
                "backend": plan.backend_name,
                "session_id": session_id,
                "status": "running",
                "spawned_at": time.time(),
                "cwd": plan.agent_cwd,
                "backend_session_id": plan.backend_session_id,
                "model": plan.model,
                "permission_mode": plan.permission_mode,
                "reasoning_effort": plan.effort,
                "create_token": process_manager.creation_token(str(new_pid)),
                # Explicit even though ``update`` would preserve it: the id
                # surviving resume is the property R8 depends on.
                **_correlation_extra(plan.correlation_id),
                # Same reasoning, and it also backfills a pi record spawned
                # before the field existed, which would otherwise stay
                # unbindable for the rest of its life.
                **_pi_binding_extra(plan.backend_name, session_id, plan.agent_name),
                # Explicit for the same reason as the correlation id: a resume
                # must never launder away who spawned the agent, or the next
                # follow-up would see ``parent_unknown``. ``update`` would
                # preserve it anyway — stating it makes the invariant local.
                SPAWNED_BY_FIELD: plan.spawned_by,
                SPAWNED_BY_SOURCE_FIELD: plan.spawned_by_source,
                # The resume prompt may take a different transport than the
                # spawn prompt did, and gate 0's grace period restarts from
                # this attempt.
                PROMPT_TRANSPORT_FIELD: plan.prompt_transport,
            }
        )
        agent.pop(PENDING_DELIVERY_FIELD, None)
        if outcome.status == DELIVERY_UNCONFIRMED:
            # R6 live uncertainty: remember the attempt so the next call
            # reconciles it rather than re-sending. The prompt file stays on
            # disk — a still-running CLI may not have read it yet.
            agent[PENDING_DELIVERY_FIELD] = {
                "nonce": plan.nonce,
                "operation_id": plan.operation_id,
                "attempted_at": time.time(),
                "prompt_file": str(plan.prompt_file) if plan.prompt_file else "",
            }
        _bump_generation(agent)
        _save_agents_transaction(session_id, agents)

    if outcome.status == DELIVERY_DELIVERED:
        # The receipt record IS proof the CLI read the sidecar, so removing it
        # now cannot race a pending read.
        remove_prompt_file(plan.prompt_file)
        return {
            "success": True,
            "name": plan.agent_name,
            "status": "delivered",
            "pid": new_pid,
            "backend": plan.backend_name,
            "backend_session_id": plan.backend_session_id,
            "replaced_existing": plan.alive,
            "session_id": session_id,
        }
    return {
        "success": False,
        "name": plan.agent_name,
        "status": "queued",
        "phase": "unconfirmed",
        "reason": "delivery_unconfirmed",
        "retriable": True,
        "pid": new_pid,
        "backend_session_id": plan.backend_session_id,
        "session_id": session_id,
        "detail": (
            "The resumed child is still alive but the prompt has not been "
            "observed in its context within the scan bound. This is NOT a "
            "failure and NOT a delivery: retry this same call and it will "
            "reconcile the prior attempt before resending."
        ),
    }


def _guaranteed_delivery(  # noqa: PLR0915 - three phases of one bounded call.
    session_id: str,
    name: str,
    prompt: str,
    replace_if_idle: bool,
    record: dict,
    deadline: float,
    parentage: tuple | None = None,
) -> dict:
    """B0 — deliver ``prompt`` to ``name`` inside one bounded budget.

    Shared by ``follow_up_agent`` and ``deliver_pending``: the cooperative tail
    has to be completed by exactly the same code that would have completed it
    in the originating call, or the two could disagree about what happened.

    ``deadline`` is on the injected ``_delivery_clock`` and is the ONE budget —
    the wait for a busy target, the FIFO queue, the resume, and confirmation
    all spend from it. ``record`` is the already-durable store row; this
    function advances its phase but never creates it, because creation must
    happen before any waiting for response loss to be recoverable.
    """

    def _prepare(ticket: str | None) -> _FollowUpPrep:  # noqa: PLR0911
        """Phase 1 — validate, reserve the lease, and build the request.

        Runs entirely inside ``_agents_transaction``. Reserving the lease while
        the registry lock is held is what makes the reservation atomic against
        another server preparing the same target: without it two callers could
        both snapshot the same generation and both resume, and the losing CAS
        could not undo an already-spawned child.
        """
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                # B3/R7: name the state rather than returning a bare
                # "not found" that reads as a lookup miss. A killed agent is
                # unreachable by design, and saying so is the point.
                return _FollowUpPrep(refusal=_no_delivery_path(name, "record_removed"))

            if agent.get("backend") == "external":
                return _FollowUpPrep(
                    refusal=_follow_up_failure(
                        "external_agent_pull_only",
                        name,
                        detail=(
                            "External members cannot be resumed or receive "
                            "guaranteed delivery. Use send_message; it appends "
                            "pull-based inbox work for external_read."
                        ),
                    )
                )

            # R2/C2 — session resume is downstream-only.
            #
            # THIS IS AN ACCIDENT GUARD, NOT A SECURITY BOUNDARY. ``IDENTITY``
            # is read from an env var at import time by the caller's own
            # process, and the server evaluating this check IS that process, so
            # the identity is self-asserted and trivially forgeable. It exists
            # because a follow-up is kill-and-respawn: a worker calling this on
            # its own coordinator restarts that coordinator's process, losing
            # its context mid-task. Do not treat it as authorization.
            #
            # Placed ahead of every side effect on purpose — before the binding
            # resolve, before any record write, and in particular before the
            # A4b lease is reserved or the generation is bumped. A refusal must
            # leave the session byte-identical.
            refusal = _direction_refusal(agent, name)
            if refusal is not None:
                return _FollowUpPrep(refusal=refusal)

            # ...and the parentage the pre-flight authorized against must still
            # be the parentage now. The guard above only asks "is the CURRENT
            # spawner me"; that can still pass while the record has been
            # re-parented since the write that created the audit row was
            # authorized. Rejecting the stale authorization explicitly is what
            # lets the caller discard that row instead of keeping it as
            # evidence of a call C2 says changed nothing.
            if parentage is not None and _parentage(agent) != parentage:
                return _FollowUpPrep(
                    refusal=_follow_up_failure(
                        REASON_STALE_AUTHORIZATION,
                        name,
                        detail=(
                            f"The recorded spawner of {name!r} changed while this "
                            "call was starting, so the check that authorized it "
                            "no longer describes the record. Nothing was sent and "
                            "nothing was changed. Re-read the agent and retry."
                        ),
                    )
                )

            backend_name = str(agent.get("backend") or "")
            try:
                backend = registry.get(backend_name)
            except Exception:
                logger.debug("Failed loading backend for follow-up", exc_info=True)
                return _FollowUpPrep(
                    refusal=_follow_up_failure("backend_not_supported", name)
                )

            if not getattr(backend, "supports_resume", lambda: False)():
                return _FollowUpPrep(
                    refusal=_follow_up_failure("backend_not_supported", name)
                )

            pid = agent["pid"]
            alive = _agent_alive(agent)
            binding = _resolve_agent_binding(agent)
            changed = _sync_backend_session_id(agent, binding)
            status = _agent_check_payload(name, agent, alive, binding)

            def _refuse(
                reason: str, *, retriable: bool = False, detail: str = ""
            ) -> _FollowUpPrep:
                """Flush any newly bound session id, then refuse.

                Every refusal below shares this: a session id discovered by
                this call is still worth persisting even though the follow-up
                itself is not proceeding.
                """
                if changed:
                    _save_agents_transaction(session_id, agents)
                return _FollowUpPrep(
                    refusal=_follow_up_failure(
                        reason, name, status, retriable=retriable, detail=detail
                    )
                )

            # A2/A6/R8: a follow-up is only safe against a transcript we have
            # positively identified. Every other outcome refuses, including
            # ``legacy`` — resuming on an unverifiable stored id is precisely
            # how a nonce gets confirmed in the wrong conversation and reported
            # as delivered. ``pending`` and ``indeterminate`` are retriable;
            # the other three are terminal for this agent.
            if not binding.bound:
                return _refuse(
                    f"binding_{binding.outcome}",
                    retriable=binding.retriable,
                    detail=_BINDING_REFUSAL_DETAIL.get(binding.outcome, ""),
                )

            backend_session_id = status.get("backend_session_id")
            if not backend_session_id:
                # B3/R7 — dead or alive, without a resumable backend session
                # there is no path at all. Say which state that is.
                if changed:
                    _save_agents_transaction(session_id, agents)
                return _FollowUpPrep(
                    refusal=_no_delivery_path(name, "no_backend_session", status)
                )

            # R6: a prior attempt that expired with a live child may in fact
            # have landed, because a buffered transcript write can arrive after
            # the call returned. Reconcile BEFORE sending, or the same
            # instruction is delivered twice.
            if _reconcile_pending_delivery(
                agent, backend_name, str(backend_session_id), binding, session_id
            ):
                reconciled = _pending_delivery(agent) or {}
                remove_prompt_file(_optional_path(reconciled.get("prompt_file")))
                agent.pop(PENDING_DELIVERY_FIELD, None)
                _bump_generation(agent)
                _save_agents_transaction(session_id, agents)
                return _FollowUpPrep(
                    refusal={
                        "success": True,
                        "name": name,
                        "status": "delivered",
                        "reconciled": True,
                        "pid": agent.get("pid"),
                        "backend": backend_name,
                        "backend_session_id": str(backend_session_id),
                        "session_id": session_id,
                        "detail": (
                            "A previous attempt's prompt was confirmed in the "
                            "target's context after that call returned; it was "
                            "not sent again."
                        ),
                    }
                )

            if alive:
                last_activity_at = status.get("last_activity_at")
                # A hook-written "waiting" marker is an authoritative idle
                # signal: the agent has reached a wait/stop hook and is parked
                # awaiting input, so we can resume it immediately. It is
                # resolved FIRST, ahead of the transcript-derived checks below,
                # which are only heuristics for when no reliable marker exists.
                # Ordering matters: an agent parked at a Stop hook before
                # emitting any assistant text has no ``last_message``, and
                # checking that first reported ``agent_busy`` for precisely the
                # case we know for certain is idle.
                idle_by_marker = (
                    _resolve_agent_state(
                        alive=True,
                        marker=_read_state_marker(session_id, name),
                        last_activity_at=last_activity_at,
                    )
                    == "waiting"
                )
                if not idle_by_marker:
                    # B2/R1 — a busy target is no longer a dead end. This is
                    # the defect the whole feature exists to close: the caller
                    # is the target's own spawner (the direction guard has
                    # already run), it has something the target needs, and
                    # refusing left it with no correct tool. So we WAIT,
                    # bounded, for the target to reach a resumable point.
                    #
                    # The wait happens in the caller loop below, never here:
                    # this body holds the cross-process registry lock, and
                    # sleeping under it would block every registry reader on
                    # the machine.
                    #
                    # The three-way split below is unchanged from the refusal
                    # version; only the two "busy" arms became waits.
                    # ``agent_state_unknown`` stays a refusal: it is not a
                    # busy agent, it is one we cannot describe at all, and
                    # waiting for that resolves nothing.
                    def _wait() -> _FollowUpPrep:
                        if changed:
                            _save_agents_transaction(session_id, agents)
                        return _FollowUpPrep(wait_reason="agent_busy")

                    if status.get("last_message") is None:
                        return _wait()
                    if last_activity_at is None:
                        return _refuse("agent_state_unknown")
                    if time.time() - float(last_activity_at) < _FOLLOW_UP_IDLE_SECONDS:
                        return _wait()
                if not replace_if_idle:
                    return _refuse("agent_idle_but_alive")

            agent_name = str(agent.get("name") or name)
            agent_cwd = str(agent.get("cwd") or Path.cwd())
            generation = _record_generation(agent)
            operation_id = uuid.uuid4().hex
            nonce = new_delivery_nonce()

            # A4b — reserve the right to resume BEFORE the lock is released.
            # A second valid caller is queued behind per-target FIFO; it is
            # never refused, because refusing a valid caller would hand back
            # exactly the dead end R1 forbids.
            reservation = reserve_lease(
                _leases_file(session_id),
                agent_name,
                generation=generation,
                operation_id=operation_id,
                backend_session_id=str(backend_session_id),
                nonce=nonce,
                holder_pid=os.getpid(),
                holder_create_token=process_manager.creation_token(str(os.getpid())),
                deadline=time.time() + _LEASE_TTL_SECONDS,
                now=time.time(),
                holder_probe=_lease_holder_probe,
                ticket=ticket,
            )
            if changed:
                _save_agents_transaction(session_id, agents)
            if not reservation.granted:
                return _FollowUpPrep(
                    ticket=reservation.ticket, queue_position=reservation.position
                )

            (
                model,
                permission_mode,
                effort,
                correlation_id,
                request,
                prompt_extra,
            ) = _build_resume_request(
                session_id,
                agent,
                agent_name,
                agent_cwd,
                backend,
                backend_name,
                prompt,
                nonce,
            )

            # Anchored BEFORE the resume, on the last COMPLETE record: that is
            # what makes the later scan an observation of *this* attempt rather
            # than of history that was already there.
            scanner = _delivery_scanner(
                agent, backend_name, str(backend_session_id), binding, session_id
            )
            scanner.snapshot()

            return _FollowUpPrep(
                plan=_FollowUpPlan(
                    agent_name=agent_name,
                    backend=backend,
                    backend_name=backend_name,
                    backend_session_id=str(backend_session_id),
                    request=request,
                    old_pid=str(pid),
                    old_create_token=_agent_create_token(agent),
                    alive=alive,
                    nonce=nonce,
                    operation_id=operation_id,
                    generation=generation,
                    prompt_file=_optional_path(prompt_extra.get("prompt_file_path")),
                    prompt_transport=_prompt_transport(prompt_extra),
                    scanner=scanner,
                    model=model,
                    permission_mode=permission_mode,
                    effort=effort,
                    correlation_id=correlation_id,
                    agent_cwd=agent_cwd,
                    spawned_by=str(agent.get(SPAWNED_BY_FIELD) or ""),
                    spawned_by_source=str(
                        agent.get(SPAWNED_BY_SOURCE_FIELD) or SPAWNED_BY_SOURCE_SPAWN
                    ),
                    agent_snapshot=dict(agent),
                ),
                ticket=reservation.ticket,
            )

    def _do_follow_up() -> dict:
        if not session_id:
            return _follow_up_failure("session_not_found", name)
        # A4b — the FIFO ticket is DERIVED from the durable delivery record, so
        # it survives the end of this call. The cooperative tail promises the
        # caller that its place in the per-target queue is preserved; a fresh
        # MCP call starting from ``None`` would append a second waiter behind
        # its own orphaned head and never advance. The idempotency key is the
        # handle the caller already holds, so it is the natural durable ticket.
        ticket: str | None = _queue_ticket(record)
        waited_for = ""
        position = 0
        while True:
            prep = _prepare(ticket)
            if prep.refusal is not None:
                return prep.refusal
            if prep.plan is not None:
                break
            # Two reasons to be here, and neither is a refusal (R1): the target
            # is busy (B2), or another valid caller holds the per-target FIFO
            # (A4b). Both wait; the honest cooperative tail appears only once
            # the ONE budget is spent.
            if prep.wait_reason:
                waited_for = prep.wait_reason
            else:
                ticket = prep.ticket
                waited_for = "operation_in_progress"
                position = prep.queue_position
            if _delivery_clock() >= deadline:
                # B1: a message that was NEVER leased does not expire into
                # `failed`. There is one timeout — this budget — so the same
                # instant cannot also mean "definitely never delivered".
                return _pending_tail(session_id, name, record, waited_for, position)
            _delivery_sleep(_DELIVERY_POLL_SECONDS)

        plan = prep.plan
        # Phase 2 — the registry lock is NOT held here. Shutdown, resume and
        # confirmation all take real time, and holding a cross-process lock
        # across them would block every registry reader on the machine.
        try:
            # Crash-recovery window "after spawn, before sent": the nonce is
            # durable BEFORE the resume, so a crash here still leaves a
            # searchable receipt marker rather than an attempt nobody can
            # attribute. Inside the ``try`` so a lost write releases the lease
            # on its way out instead of stranding the target behind it.
            _mark_attempt_sent(session_id, record, plan)

            # Fail closed: only signal a PID we can prove is still ours.
            if (
                plan.alive
                and process_manager.owns_process(plan.old_pid, plan.old_create_token)
                and not process_manager.graceful_shutdown(plan.old_pid, timeout_s=5.0)
            ):
                process_manager.kill_process(plan.old_pid)

            try:
                result = plan.backend.resume(plan.request, plan.backend_session_id)
            except Exception:
                logger.debug("Failed resuming backend session", exc_info=True)
                return _record_outcome(
                    session_id,
                    record,
                    _finalize_follow_up(
                        session_id, plan, _DELIVERY_RESUME_FAILED, None
                    ),
                    plan,
                )

            new_pid = int(result.process_handle)
            outcome = confirm_delivery(
                plan.scanner,
                plan.nonce,
                child_alive=lambda: process_manager.health_check(str(new_pid))[0],
                # Whatever is LEFT of the one budget, never a fresh timer.
                # A per-step bound here would let a single call spend the
                # advertised budget twice over.
                bound_s=max(0.0, deadline - _delivery_clock()),
                poll_interval_s=_DELIVERY_POLL_SECONDS,
                clock=_delivery_clock,
                sleep=_delivery_sleep,
            )
            return _record_outcome(
                session_id,
                record,
                _finalize_follow_up(session_id, plan, outcome, new_pid),
                plan,
            )
        finally:
            # Belt and braces: a crash between here and ``_finalize`` would
            # otherwise leave the lease held by a process that is no longer
            # working on it. ``release_lease`` is a no-op once finalize won.
            #
            # The lease is deliberately NOT held across ``unconfirmed``: it
            # converts to the durable pending-delivery record, which keeps
            # serializing this target until reconciliation, so neither a
            # future delivery nor a kill is blocked by a lease nobody is
            # progressing.
            #
            # The result is checked because discarding it is the lease-side
            # twin of the delivery-claim wedge: a release whose write is lost
            # leaves the lease on disk naming THIS live server, so
            # ``_lease_holder_probe`` reports it live for as long as the
            # process runs, and every later caller queues — and every
            # ``kill_agent`` refuses — behind a holder that will never come
            # back to clear it. One retry costs nothing here and closes the
            # transient case; a persistent failure is logged loudly and needs
            # the CLI operator escape, which is what it is for.
            _release_lease_or_warn(session_id, plan)

    return _do_follow_up()


def _release_lease_or_warn(session_id: str, plan: _FollowUpPlan) -> bool:
    """Release ``plan``'s lease, retrying once, and report whether it is gone.

    ``release_lease`` returns ``False`` both when the lease was already gone
    (finalize won the CAS — the normal case) and when the write was lost. Only
    the second matters, so the store is re-read to tell them apart rather than
    warning on every healthy delivery.
    """
    path = _leases_file(session_id)
    for attempt in range(2):
        try:
            release_lease(path, plan.agent_name, plan.operation_id)
            still_held = active_lease(path, plan.agent_name)
        except LeaseStoreError:
            logger.warning(
                "Lease store unusable while releasing %s",
                plan.agent_name,
                exc_info=True,
            )
            continue
        if still_held is None or still_held.operation_id != plan.operation_id:
            return True
        if attempt == 0:
            continue
    logger.error(
        "Could not release the lease on %s held by operation %s. Until it is "
        "cleared (`win-agent-teams lease force`), deliveries to this agent "
        "queue and kill_agent refuses.",
        plan.agent_name,
        plan.operation_id,
    )
    return False


def _mark_attempt_sent(session_id: str, record: dict, plan: _FollowUpPlan) -> None:
    """Move the store row to ``queued(phase=sent)`` BEFORE the resume runs.

    Ordering is the crash-recovery contract. If this process dies between the
    spawn and the confirmation, the nonce is already on disk, so lease expiry
    can reconcile the attempt by searching for it. Writing the nonce *after*
    the resume would leave a delivered prompt no one could attribute.

    Raises :class:`DeliveryStoreError` when that write is lost, and the caller
    must abandon the attempt: a resume whose nonce never reached disk is a
    prompt nobody can attribute afterwards — exactly the case this ordering
    exists to prevent.
    """
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        stored = txn.get(
            str(record.get("sender") or ""), str(record["idempotency_key"])
        )
        target = stored if stored is not None else record
        target["nonce"] = plan.nonce
        target["operation_id"] = plan.operation_id
        target["attempts"] = int(target.get("attempts") or 0) + 1
        target["attempted_at"] = time.time()
        target["prompt_file"] = str(plan.prompt_file) if plan.prompt_file else ""
        # Pre-resume binding, so a crash between here and the resume still
        # leaves a rescannable row even if the registry record is later
        # removed. ``_record_outcome`` refreshes it with the post-resume one.
        target[TARGET_SNAPSHOT_FIELD] = _scan_snapshot(plan.agent_snapshot)
        mark_phase(target, PHASE_SENT)
        txn.put(target)
        record.update(target)


def _record_outcome(
    session_id: str, record: dict, result: dict, plan: _FollowUpPlan
) -> dict:
    """Fold one attempt's result into the durable store, then annotate it.

    The mapping is deliberately narrow, because R4 has exactly three public
    statuses and this is the only place a transport result becomes one of
    them:

    - ``delivered`` → terminal ``delivered``.
    - ``queued(phase=unconfirmed)`` → stays non-terminal. A live child with a
      buffered write is uncertainty, not failure, and settling it here is the
      contradiction R6 forbids.
    - a **retriable** failure (superseded, resume refused) → back to
      ``pending``. Nothing was confirmed and nothing is settled, so the
      cooperative tail can still complete it.
    - any other failure → terminal ``failed``. The child is provably gone with
      no receipt: definite non-delivery.
    """
    # Post-resume binding: the resume can mint a new backend session id and PID,
    # and the snapshot a later rescan uses must describe the child the nonce was
    # actually delivered to, not the one phase 1 saw. Read outside the store
    # lock; a record that has since been removed simply leaves the pre-resume
    # snapshot in place, which is still better than nothing to scan at all.
    refreshed = _scan_snapshot(_find_agent(_load_agents(session_id), plan.agent_name))
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        stored = txn.get(
            str(record.get("sender") or ""), str(record["idempotency_key"])
        )
        target = stored if stored is not None else record
        if refreshed is not None:
            target[TARGET_SNAPSHOT_FIELD] = refreshed
        if result.get("status") == "delivered":
            settle(target, STATUS_DELIVERED, reason="", now=time.time())
        elif result.get("phase") == PHASE_UNCONFIRMED:
            mark_phase(target, PHASE_UNCONFIRMED, reason="delivery_unconfirmed")
        elif result.get("retriable"):
            mark_phase(target, PHASE_PENDING, reason=str(result.get("reason") or ""))
        else:
            settle(
                target,
                STATUS_FAILED,
                reason=str(result.get("reason") or REASON_NOT_DELIVERED),
                now=time.time(),
            )
        txn.put(target)
        record.update(target)
    return _with_public_status(result, record)


def _with_delivery_identity(result: dict, record: dict) -> dict:
    """Attach the identity a sender needs to ask about this message later."""
    result["message_id"] = record.get("message_id", "")
    result["idempotency_key"] = record.get("idempotency_key", "")
    result["call_budget_s"] = _DELIVERY_CALL_BUDGET_SECONDS
    return result


def _with_public_status(result: dict, record: dict) -> dict:
    """Make the returned status and phase agree with what was durably stored.

    Not cosmetic. A caller must be able to compare this response with a later
    ``delivery_status`` answer and see the same thing; a transport-shaped
    payload that omits ``status`` would leave "which of R4's three did I get?"
    to be inferred from ``success``, which is exactly the ambiguity this
    feature removes.
    """
    result["status"] = record.get("status", STATUS_QUEUED)
    result["phase"] = record.get("phase", PHASE_PENDING)
    return _with_delivery_identity(result, record)


#: The obligation attached to the cooperative tail. R1 requires the sender be
#: told plainly that the tail is cooperative — there is no dispatcher, so
#: nothing completes it unless the sender comes back.
_TAIL_OBLIGATION = (
    "SENDER OBLIGATION: this message was NOT delivered and nothing will "
    "deliver it on your behalf — there is no background dispatcher. Call "
    "deliver_pending() (or follow_up_agent again with this same "
    "idempotency_key) to finish it. The message stays durably queryable via "
    "delivery_status(idempotency_key) until it settles."
)


def _queue_ticket(record: dict) -> str:
    """Return the durable per-target FIFO ticket for one delivery record.

    ``(sender, idempotency_key)`` is the only identity that both survives the
    end of an MCP call and is still in the caller's hands when it retries, so
    it — not a per-call uuid — is what keeps the queue place recoverable.
    Hashed rather than concatenated so an agent name containing the separator
    cannot collide with another sender's key.
    """
    sender = str(record.get("sender") or "")
    key = str(record.get("idempotency_key") or "")
    digest = hashlib.sha256(f"{sender}\x00{key}".encode())
    return digest.hexdigest()[:32]


def _pending_tail(
    session_id: str, name: str, record: dict, waited_for: str, position: int
) -> dict:
    """Return R1's cooperative tail: ``queued(phase=pending)``, never failed.

    Nothing was sent, so the record stays exactly where it was — not settled,
    not expired. An earlier draft had ``failed(reason="expired")`` here, which
    contradicted R1 outright: the budget expiring is precisely the case the
    requirement says returns ``queued`` with an obligation on the sender.
    """
    return _with_delivery_identity(
        {
            "success": False,
            "name": name,
            "status": STATUS_QUEUED,
            "phase": PHASE_PENDING,
            "reason": waited_for or "call_budget_expired",
            "retriable": True,
            "queue_position": position,
            "session_id": session_id,
            "sender_obligation": _TAIL_OBLIGATION,
            "detail": (
                f"The target did not become deliverable within the "
                f"{_DELIVERY_CALL_BUDGET_SECONDS:g}s call budget "
                f"(waited on: {waited_for or 'target'}). Nothing was sent and "
                "nothing was lost."
            ),
        },
        record,
    )


#: Field holding the process currently working one delivery record. This is the
#: delivery-record-level serialization the FIFO ticket alone cannot provide: two
#: concurrent calls under one ``(sender, key)`` derive the SAME ticket, so the
#: lease queue treats them as one caller retrying rather than as two callers.
ACTIVE_HOLDER_FIELD = "active_holder"

#: A same-key call arrived while another is still working the record.
REASON_DELIVERY_IN_PROGRESS = "delivery_in_progress"
#: A same-key call arrived while the previous attempt's fate is still unknown.
REASON_ATTEMPT_UNRESOLVED = "attempt_unresolved"
#: The durable row could not be written, so nothing may proceed on it.
REASON_STORE_UNAVAILABLE = "delivery_store_unavailable"
#: The parentage the read-only pre-flight authorized is no longer the parentage
#: under the registry lock.
REASON_STALE_AUTHORIZATION = "stale_authorization"


#: Claim ids this process is *currently* working, i.e. between
#: :func:`_claim_holder` and :func:`_release_delivery_claim`. This is what makes
#: a claim reclaimable without inventing an expiry policy: for our own PID we do
#: not have to infer anything from the OS, we simply know. Guarded by a lock
#: because ``run_blocking`` runs delivery work on a thread pool.
_ACTIVE_CLAIM_IDS: set[str] = set()
_ACTIVE_CLAIM_LOCK = threading.Lock()

#: Per-call identity of a claim, distinct from the holder PID. Two calls in one
#: server process share a PID; only this tells them apart.
CLAIM_ID_FIELD = "claim_id"


def _claim_is_held(holder: object) -> bool:  # noqa: PLR0911 - each return is a distinct liveness verdict.
    """Whether ``holder``'s claim is still being worked, so nobody else may.

    Three inputs, and keeping them apart is the point:

    - **Our own PID.** ``_ACTIVE_CLAIM_IDS`` is authoritative and exact: this
      process knows which claims it is working. A claim stamped with our PID
      whose id is NOT in that set is one we already finished — most often
      because its release write failed. Previously such a claim read as "live
      holder" forever, so every later valid caller under that key queued behind
      a claim that only this process could clear and that this process would
      never try to clear again: a permanent R1 dead end from one transient disk
      error. It is now reclaimable, and reclaiming it cannot permit concurrent
      work, because the one process that could be doing that work is us.
    - **Another process, provably gone.** Reclaimable, as before.
    - **Another process, ownership unprovable.** NOT reclaimable. An unreadable
      creation token against a live PID is uncertainty; treating it as death
      authorizes a second resume of one conversation (the same "an error is not
      an absence" shape as the store loaders).
    """
    if not isinstance(holder, dict):
        return False
    mapping = cast("dict[str, Any]", holder)
    try:
        pid = int(mapping.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    token = mapping.get("create_token")
    outcome = _lease_holder_probe(pid, str(token) if token else None)
    if outcome == OWNERSHIP_NOT_OURS:
        return False
    if pid == os.getpid():
        claim_id = mapping.get(CLAIM_ID_FIELD)
        if not isinstance(claim_id, str) or not claim_id:
            # A claim from before this field existed, or a torn write. We
            # cannot say it is ours-and-finished, so it stays held.
            return True
        with _ACTIVE_CLAIM_LOCK:
            return claim_id in _ACTIVE_CLAIM_IDS
    # Another process: live, or unprovable. Both stay held.
    return True


def _claim_holder() -> dict:
    """Stamp a claim for this call and register it as in-progress."""
    claim_id = uuid.uuid4().hex
    with _ACTIVE_CLAIM_LOCK:
        _ACTIVE_CLAIM_IDS.add(claim_id)
    return {
        "pid": os.getpid(),
        "create_token": process_manager.creation_token(str(os.getpid())),
        "claimed_at": time.time(),
        CLAIM_ID_FIELD: claim_id,
    }


def _forget_claim(record: dict) -> None:
    """Deregister the in-process claim id, whatever happens to the disk write."""
    holder = record.get(ACTIVE_HOLDER_FIELD)
    if isinstance(holder, dict):
        claim_id = cast("dict[str, Any]", holder).get(CLAIM_ID_FIELD)
        if isinstance(claim_id, str) and claim_id:
            with _ACTIVE_CLAIM_LOCK:
                _ACTIVE_CLAIM_IDS.discard(claim_id)


def _release_delivery_claim(session_id: str, record: dict) -> bool:
    """Drop this process's claim on ``record`` so the next caller may work it.

    Returns whether the release reached disk. A ``False`` is no longer a wedge:
    the in-process registration is dropped **first and unconditionally**, so
    even a stranded on-disk claim is now recognised as finished by the only
    process that could still be working it (:func:`_claim_is_held`). Callers get
    the result so they can say so rather than implying a clean handover.
    """
    _forget_claim(record)
    sender = str(record.get("sender") or "")
    key = str(record.get("idempotency_key") or "")
    if not key:
        return True
    released = True
    try:
        with delivery_transaction(_deliveries_file(session_id)) as txn:
            stored = txn.get(sender, key)
            if stored is not None:
                holder = stored.get(ACTIVE_HOLDER_FIELD)
                # Someone else's claim means it was reclaimed after we were
                # presumed dead; dropping it would undo their serialization.
                if not isinstance(holder, dict) or holder.get("pid") == os.getpid():
                    stored.pop(ACTIVE_HOLDER_FIELD, None)
                    txn.touch()
    except DeliveryStoreError:
        logger.warning(
            "Failed releasing the delivery claim for %s; the on-disk claim is "
            "stale but no longer blocks this process",
            key,
            exc_info=True,
        )
        released = False
    record.pop(ACTIVE_HOLDER_FIELD, None)
    return released


def _claim_delivery_record(session_id: str, record: dict) -> bool:
    """Take the active claim on an existing row. Returns whether we got it.

    Used by ``deliver_pending``, which reaches ``_guaranteed_delivery`` without
    going through ``_open_delivery_record``. Without this a drain and a
    ``follow_up_agent`` under the same key would derive the same FIFO ticket,
    be indistinguishable from one caller retrying, and both resume.
    """
    sender = str(record.get("sender") or "")
    key = str(record.get("idempotency_key") or "")
    if not key:
        return False
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        stored = txn.get(sender, key)
        if stored is None or is_terminal(stored):
            return False
        if _claim_is_held(stored.get(ACTIVE_HOLDER_FIELD)):
            return False
        stored[ACTIVE_HOLDER_FIELD] = _claim_holder()
        txn.touch()
        record.update(stored)
    return True


def _discard_delivery_record(session_id: str, record: dict) -> bool:
    """Remove a row this call created for a request that was then refused.

    C2 requires an authoritative refusal to change **nothing**. The row is only
    ever discarded when this call created it and nothing was sent under it, so
    no audit evidence can be destroyed: there is nothing yet to be evidence of.

    Returns whether the session really is back to byte-identical. Swallowing a
    failure here made the refusal *say* nothing changed while leaving on disk
    exactly the row the promise was about — the caller now gets the truth and
    reports it (:func:`_annotate_residual_row`), because a status that
    overstates is the defect this whole feature exists to remove.
    """
    if int(record.get("attempts") or 0) > 0 or record.get("nonce"):
        return True
    key = record_key(
        str(record.get("sender") or ""),
        str(record.get("idempotency_key") or ""),
    )
    try:
        with delivery_transaction(_deliveries_file(session_id)) as txn:
            stored = txn.data.get(key)
            if stored is not None and not int(stored.get("attempts") or 0):
                del txn.data[key]
                txn.touch()
    except DeliveryStoreError:
        logger.warning("Failed discarding a refused delivery row", exc_info=True)
        return False
    return True


def _annotate_residual_row(result: dict, idempotency_key: str) -> dict:
    """Tell the caller a refused request left its audit row behind after all."""
    result["record_discarded"] = False
    result["detail"] = (
        f"{result.get('detail', '')} NOTE: nothing was sent, but the durable "
        f"row created for this call could not be removed (the delivery store "
        f"could not be written). idempotency_key {idempotency_key!r} is "
        f"therefore consumed: reuse it only for this exact request, or pick a "
        f"new one."
    ).strip()
    return result


def _store_unavailable(name: str) -> dict:
    """Refuse because the durable row is not on disk (R4/B0)."""
    return {
        "success": False,
        "name": name,
        "status": STATUS_QUEUED,
        "phase": PHASE_PENDING,
        "reason": REASON_STORE_UNAVAILABLE,
        "retriable": True,
        "detail": (
            "The durable delivery record could not be written, so nothing was "
            "sent. That row is the only thing that would let you recover this "
            "message's outcome if you lost the response, so proceeding without "
            "it would leave you with neither a status nor a usable key. Fix the "
            "session directory (disk full, permissions, read-only mount) and "
            "retry with the same idempotency_key."
        ),
    }


def _delivery_in_progress(session_id: str, name: str, record: dict) -> dict:
    """Refuse a second concurrent call on one key, without sending anything.

    Not a dead end (R1): the record is durable, the caller holds the key, and
    the tail is completed by the call that already owns it or by a later
    ``deliver_pending``. What it prevents is two callers resuming one backend
    conversation under a single idempotency key — the recipient has no dedupe
    table, so a second resume is a second real prompt.
    """
    return _with_delivery_identity(
        {
            "success": False,
            "name": name,
            "status": STATUS_QUEUED,
            "phase": PHASE_PENDING,
            "reason": REASON_DELIVERY_IN_PROGRESS,
            "retriable": True,
            "session_id": session_id,
            "sender_obligation": _TAIL_OBLIGATION,
            "detail": (
                "Another call is already working this idempotency_key. Nothing "
                "was sent by this call. Query delivery_status(idempotency_key) "
                "for the outcome, or retry once the other call has returned."
            ),
        },
        record,
    )


def _unresolved_attempt_result(session_id: str, name: str, record: dict) -> dict:
    """Report an in-flight attempt whose fate is still unknown, without resending.

    The plan's rule is explicit: a retry must check whether the prior attempt's
    nonce already landed **before** re-sending. When that check comes back
    "still unknown" the answer is to wait, not to send again — the recipient is
    a backend conversation, not a consumer with a dedupe table, so a second
    resume delivers the same instruction twice.
    """
    return _with_public_status(
        {
            "success": False,
            "name": name,
            "reason": REASON_ATTEMPT_UNRESOLVED,
            "retriable": True,
            "session_id": session_id,
            "sender_obligation": _TAIL_OBLIGATION,
            "detail": (
                "A previous attempt under this idempotency_key was sent and its "
                "receipt has not appeared yet. It was rescanned just now and is "
                "still unresolved, so it was NOT sent again. Retry with the same "
                "key: once the receipt lands this reports delivered, and once "
                "the child is provably gone with a complete negative scan it "
                "reports failed."
            ),
        },
        record,
    )


def _reconcile_before_resend(session_id: str, name: str, record: dict) -> dict | None:
    """Re-read and reconcile the durable row before any resend is considered.

    Returns the answer when the row must NOT be sent again, or ``None`` when it
    is genuinely at ``pending`` — nothing in flight — and a fresh attempt is
    correct. ``_prepare``'s ``_reconcile_pending_delivery`` is the same rule at
    the agent-record level; this is the sender-side row, and both are needed
    because they answer for different objects.
    """
    if record.get("phase") not in {PHASE_SENT, PHASE_UNCONFIRMED}:
        return None
    agents = _load_agents(session_id)
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        stored = (
            txn.get(str(record.get("sender") or ""), str(record["idempotency_key"]))
            or record
        )
        if _reconcile_delivery_record(
            session_id, stored, _find_agent(agents, str(stored.get("to") or ""))
        ):
            txn.touch()
        record.update(stored)
    if is_terminal(record):
        return _settled_result(session_id, name, record)
    return _unresolved_attempt_result(session_id, name, record)


def _open_delivery_record(
    session_id: str, name: str, prompt: str, idempotency_key: str, options: dict
) -> tuple[dict | None, dict | None, bool]:
    """Create or reuse this sender's durable record.

    Returns ``(record, refusal, created)``.

    Runs **before any waiting**, under the store's cross-process lock, so the
    check-then-create sequence is atomic against another MCP server doing the
    same thing with the same key.

    Three outcomes, and the middle one is the whole reason the key exists:

    - no record → create one now. Creating it before the wait is what makes
      response loss survivable: a client timeout, a cancellation, or a crash
      after this point still leaves something ``delivery_status(key)`` can
      answer from.
    - a record with the SAME fingerprint → reuse it. Never a second attempt.
    - a record with ANY differing field → ``idempotency_conflict``, and
      nothing is mutated. Silently delivering the new text under the old key
      would make the audit trail lie about what was sent.

    On top of that it takes the record's **active claim**, which is the
    serialization the FIFO ticket cannot provide. The ticket is derived from
    ``(sender, key)``, so two concurrent identical calls share it and the lease
    queue cannot tell them apart from one caller retrying. Claiming the row
    under the store lock can: whoever gets the claim works the record, and
    everyone else is told so and sends nothing.
    """
    fingerprint = request_fingerprint(to=name, prompt=prompt, options=options)
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        existing = txn.get(IDENTITY, idempotency_key)
        if existing is not None:
            if existing.get("fingerprint") != fingerprint:
                return (
                    None,
                    _with_delivery_identity(
                        {
                            "success": False,
                            "name": name,
                            "reason": IDEMPOTENCY_CONFLICT,
                            "retriable": False,
                            "detail": (
                                f"idempotency_key {idempotency_key!r} was already "
                                f"used for a different message (recipient, prompt "
                                "or options differ). Nothing was sent and nothing "
                                "was changed. Use a new key, or repeat the "
                                "original request byte-for-byte."
                            ),
                        },
                        existing,
                    ),
                    False,
                )
            if is_terminal(existing):
                # Nothing left to work; the caller gets the settled answer and
                # no claim is taken, so a terminal key is never serialized.
                return dict(existing), None, False
            if _claim_is_held(existing.get(ACTIVE_HOLDER_FIELD)):
                return (
                    None,
                    _delivery_in_progress(session_id, name, existing),
                    False,
                )
            existing[ACTIVE_HOLDER_FIELD] = _claim_holder()
            txn.touch()
            return dict(existing), None, False
        record = delivery_store.new_record(
            sender=IDENTITY,
            idempotency_key=idempotency_key,
            to=name,
            fingerprint=fingerprint,
            created_at=time.time(),
        )
        record["prompt"] = prompt
        record["options"] = options
        record[ACTIVE_HOLDER_FIELD] = _claim_holder()
        txn.put(record)
        return dict(record), None, True


def _parentage(agent: dict) -> tuple[str, str]:
    """Return the parentage snapshot a C2 authorization is granted against."""
    return (
        str(agent.get(SPAWNED_BY_FIELD) or ""),
        str(agent.get(SPAWNED_BY_SOURCE_FIELD) or ""),
    )


def _preflight_refusal(session_id: str, name: str) -> tuple[dict | None, tuple | None]:
    """Refusals that must leave the session byte-identical (read-only).

    Returns ``(refusal, parentage)``. The parentage is the snapshot this
    pre-flight actually authorized against, and it is carried to the
    authoritative under-lock check so a refusal cannot be split across a change.
    Without it the two checks can legitimately disagree — the pre-flight passes,
    the durable record is created, and the locked check then refuses, leaving an
    audit row behind for a call C2 says must change nothing.

    Only the two refusals that carry the byte-identical promise live here: an
    absent record, and the R2/C2 downstream-only direction guard. Everything
    else is decided under the registry lock in ``_prepare``, where it belongs.
    """
    agent = _find_agent(_load_agents(session_id), name)
    if agent is None:
        return _no_delivery_path(name, "record_removed"), None
    refusal = _direction_refusal(agent, name)
    if refusal is not None:
        return refusal, None
    return None, _parentage(agent)


def _external_target_refusal(session_id: str, name: str) -> dict | None:
    """Return the pull-only refusal for an external target, read-only."""
    agent = _find_agent(_load_agents(session_id), name)
    if agent is None or agent.get("backend") != "external":
        return None
    return _follow_up_failure(
        "external_agent_pull_only",
        name,
        detail=(
            "External members cannot be resumed or receive guaranteed delivery. "
            "Use send_message; it appends pull-based inbox work for external_read."
        ),
    )


def _settled_result(session_id: str, name: str, record: dict) -> dict:
    """Return the already-known outcome for a terminal same-key retry."""
    delivered = record.get("status") == STATUS_DELIVERED
    return _with_delivery_identity(
        {
            "success": delivered,
            "name": name,
            "status": record.get("status"),
            "phase": record.get("phase"),
            "reason": record.get("reason", ""),
            "retriable": False,
            "session_id": session_id,
            "detail": (
                "This idempotency key has already settled; the message was "
                "not sent again."
            ),
        },
        record,
    )


def _guaranteed_send(  # noqa: PLR0911 - each return is a distinct refusal contract.
    session_id: str,
    name: str,
    prompt: str,
    idempotency_key: str,
    replace_if_idle: bool,
    *,
    tool: str,
) -> dict:
    """Open (or recover) the durable record and run the bounded delivery.

    Shared by ``follow_up_agent`` and the downstream branch of ``send_message``
    (C3/R5). Both must produce the same statuses, the same idempotency
    semantics, and the same audit rows, so they run the same code rather than
    two implementations that agree today. ``tool`` only names the caller in the
    key-validation message.
    """
    # Validation FIRST, before the record exists and before any waiting:
    # a caller that got the key wrong must learn now, not a budget later.
    invalid = validate_idempotency_key(idempotency_key)
    if invalid is not None:
        return {
            "success": False,
            "name": name,
            "reason": invalid,
            "retriable": False,
            "detail": (
                f"{tool} requires an idempotency_key you choose "
                "before the call: 1-"
                f"{delivery_store.MAX_IDEMPOTENCY_KEY_LENGTH} characters "
                "from [A-Za-z0-9._:@=+-], starting alphanumeric. It is "
                "what lets you recover the outcome if this response is "
                "lost. Nothing was sent."
            ),
        }
    if not session_id:
        return _follow_up_failure("session_not_found", name)

    # C2's invariant survives Phase B: a refused upstream follow-up must
    # leave the session byte-identical, and creating the durable record
    # first would have written a file. So the guard is re-run here, ahead
    # of the store, as a read-only pre-flight. ``_prepare`` still checks it
    # again under the registry lock — that one is authoritative; this one
    # only decides whether we may touch disk at all.
    preflight, parentage = _preflight_refusal(session_id, name)
    if preflight is not None:
        return preflight
    external_refusal = _external_target_refusal(session_id, name)
    if external_refusal is not None:
        return external_refusal

    options = {"replace_if_idle": replace_if_idle}
    try:
        record, refusal, created = _open_delivery_record(
            session_id, name, prompt, idempotency_key, options
        )
    except DeliveryStoreError:
        logger.debug("Delivery store write failed while opening", exc_info=True)
        return _store_unavailable(name)
    if refusal is not None or record is None:
        return refusal or _follow_up_failure("session_not_found", name)
    if is_terminal(record):
        return _settled_result(session_id, name, record)

    try:
        # Re-read and reconcile BEFORE anything can be granted or resent. A row
        # still at ``sent``/``unconfirmed`` has an attempt whose fate is not
        # settled, and sending again on top of it delivers one instruction
        # twice into a conversation that cannot deduplicate it.
        unresolved = _reconcile_before_resend(session_id, name, record)
        if unresolved is not None:
            return unresolved

        deadline = _delivery_clock() + _DELIVERY_CALL_BUDGET_SECONDS
        result = _guaranteed_delivery(
            session_id, name, prompt, replace_if_idle, record, deadline, parentage
        )
    except DeliveryStoreError:
        logger.debug("Delivery store write failed mid-delivery", exc_info=True)
        return _store_unavailable(name)
    except LeaseStoreError:
        # The lease store is the serialization that stops two callers resuming
        # one conversation. Unknown or unwritable, it cannot serialize anything,
        # so this fails closed exactly like the delivery store: nothing was
        # sent, the durable row still holds the key, and the tail is retriable.
        logger.warning("Lease store unusable mid-delivery", exc_info=True)
        return _store_unavailable(name)
    finally:
        _release_delivery_claim(session_id, record)
    # C2: an authoritative refusal changes nothing. The row exists only because
    # this call created it, and nothing has been sent under it.
    if (
        created
        and result.get("reason") in _C2_REFUSAL_REASONS
        and not _discard_delivery_record(session_id, record)
    ):
        return _annotate_residual_row(result, idempotency_key)
    return result


#: Refusals that must leave the session byte-identical. A row created by the
#: same call that then hit one of these is rolled back rather than kept as
#: evidence of a request that was refused.
_C2_REFUSAL_REASONS = frozenset(
    {
        "not_spawner",
        "parent_unknown",
        REASON_STALE_AUTHORIZATION,
        "external_agent_pull_only",
    }
)


@_register_tool()
async def follow_up_agent(
    name: str,
    prompt: str,
    idempotency_key: str = "",
    replace_if_idle: bool = True,
) -> dict:
    """Deliver a follow-up prompt to an agent you spawned, and confirm it landed.

    This is the mechanism for continuing a spawned agent that would otherwise
    never read an inbox message. send_message to an agent you spawned
    routes through this same path, so the two are equivalent downstream.
    A busy agent is NOT refused: the call waits, bounded, for the agent to
    reach a resumable point, then resumes and confirms.

    idempotency_key is REQUIRED and is chosen by you before the call. It is how
    you recover the outcome if you lose this response (client timeout,
    cancellation, crash): call delivery_status(idempotency_key). Reusing a key
    with a byte-identical request returns the same attempt and never sends
    twice; reusing it with any changed field returns idempotency_conflict and
    changes nothing.

    Returns exactly three statuses. "delivered" means the prompt was observed
    in the target's context — a returned PID never means that. "failed" means
    definite non-delivery (child dead, no receipt). "queued" means still in
    flight, with a phase beneath it: "pending" (nothing was sent — the call
    budget expired, and completing it is YOUR obligation via deliver_pending)
    or "unconfirmed" (it was sent and the child is alive, but the receipt has
    not appeared yet; retrying reconciles rather than resending).

    Guaranteed-path messages do NOT enter the recipient's inbox, so they cannot
    be read a second time via read_messages; they are recorded in the sender's
    delivery store instead.

    External members are pull-only and cannot be resumed. Targeting one returns
    ``reason="external_agent_pull_only"`` before any delivery row, claim, lease,
    backend lookup, or process operation. Use ``send_message`` instead; it
    appends one unconfirmed inbox message for the member's next
    ``external_read``.

    replace_if_idle defaults to True: an idle-but-alive process is gracefully
    shut down and resumed with the follow-up prompt. Set it to False to instead
    refuse such an agent with reason="agent_idle_but_alive".
    """
    session_id = _active_session_id()

    def _run() -> dict:
        return _guaranteed_send(
            session_id,
            name,
            prompt,
            idempotency_key,
            replace_if_idle,
            tool="follow_up_agent",
        )

    return _annotate(await run_blocking(_run))


@_register_tool()
async def delivery_status(idempotency_key: str = "", to: str = "") -> dict:
    """Ask what happened to a guaranteed-path message you sent (R4).

    Pass the idempotency_key you chose for that message. This works even if you
    never received the original response — that is the point of the key.

    This is an ACTIVE reconciler, not a passive lookup: an attempt sitting at
    phase="unconfirmed" is rescanned for its receipt before this answers, so a
    prompt that landed after the original call returned reports as delivered
    rather than staying unconfirmed forever.

    Passing `to` instead lists every message you sent that agent, reconciling
    each unsettled row the same way. That is a convenience view only: with
    several messages to one agent it cannot tell you which one you lost the
    response for, so it is not a substitute for the key.
    """
    session_id = _active_session_id()

    def _run() -> dict:
        try:
            return _delivery_status(session_id, idempotency_key, to)
        except DeliveryStoreError:
            logger.debug(
                "Delivery store write failed in delivery_status", exc_info=True
            )
            return _store_unavailable(to or idempotency_key)

    return _annotate(await run_blocking(_run))


def _delivery_status(session_id: str, idempotency_key: str, to: str) -> dict:
    """Body of :func:`delivery_status`, so the store failure has one handler."""
    if not session_id:
        return {"success": False, "reason": "session_not_found"}
    if not idempotency_key:
        if to:
            # Reconciled, not a stale snapshot. The tool contract promises
            # active reconciliation, and a `to` view that answered
            # ``unconfirmed`` for a row the keyed view would immediately
            # call ``delivered`` publishes two different truths about one
            # message. Every unsettled row this sender has for ``to`` is
            # rescanned before the list is returned.
            agents = _load_agents(session_id)
            with delivery_transaction(_deliveries_file(session_id)) as txn:
                rows = txn.for_sender(IDENTITY, to)
                for row in rows:
                    if _reconcile_delivery_record(
                        session_id,
                        row,
                        _find_agent(agents, str(row.get("to") or "")),
                    ):
                        txn.touch()
                deliveries = [public_view(row) for row in rows]
            return {
                "success": True,
                "to": to,
                "deliveries": deliveries,
                "note": (
                    "Convenience list. Use delivery_status(idempotency_key) "
                    "to identify one specific message."
                ),
            }
        return {
            "success": False,
            "reason": delivery_store.KEY_REQUIRED,
            "detail": "Pass either an idempotency_key or a `to` agent name.",
        }
    agents = _load_agents(session_id)
    with delivery_transaction(_deliveries_file(session_id)) as txn:
        record = txn.get(IDENTITY, idempotency_key)
        if record is None:
            # Deliberately indistinguishable from another sender's key:
            # the namespace is per-sender, so from here it does not exist.
            return {
                "success": False,
                "reason": "delivery_not_found",
                "idempotency_key": idempotency_key,
                "detail": (
                    "No message with that idempotency_key was sent by you "
                    "in this session."
                ),
            }
        if _reconcile_delivery_record(
            session_id, record, _find_agent(agents, str(record.get("to") or ""))
        ):
            txn.touch()
        return {"success": True, **public_view(record)}


@_register_tool()
async def deliver_pending(idempotency_key: str = "") -> dict:
    """Finish the guaranteed-path messages you were told to come back for.

    This is the cooperative tail R1 declares. There is no background dispatcher,
    so a message that returned status="queued" is completed only when you call
    this (or follow_up_agent again with the same key).

    For each of your unsettled messages it reconciles first — rescanning for
    the previous attempt's receipt — and only then re-delivers, so a prompt
    that already landed is never sent twice. Pass an idempotency_key to drain
    just that one.

    Draining happens HERE and in follow_up_agent, and nowhere else. agent_status,
    check_agent and list_agents stay cheap reads on purpose.

    A pending row whose target is now an external member is never claimed,
    reconciled, resumed, or leased. The audit row remains unchanged and the
    result's separate ``refusals`` list contains
    ``{idempotency_key, to, status:"refused",
    reason:"external_agent_pull_only"}``; durable delivery statuses remain
    exactly queued/delivered/failed.
    """
    session_id = _active_session_id()

    def _run() -> dict:
        if not session_id:
            return {"success": False, "reason": "session_not_found"}
        store = _deliveries_file(session_id)
        agents = _load_agents(session_id)
        # Reconcile everything first, under one lock, so a message that
        # already landed is settled before anything considers resending it.
        pending: list[dict] = []
        refusals: list[dict] = []
        with delivery_transaction(store) as txn:
            for record in txn.for_sender(IDENTITY):
                if idempotency_key and record.get("idempotency_key") != idempotency_key:
                    continue
                if is_terminal(record):
                    continue
                target = str(record.get("to") or "")
                target_record = _find_agent(agents, target)
                if (
                    target_record is not None
                    and target_record.get("backend") == "external"
                ):
                    refusals.append(
                        {
                            "idempotency_key": str(
                                record.get("idempotency_key") or ""
                            ),
                            "to": target,
                            "status": "refused",
                            "reason": "external_agent_pull_only",
                        }
                    )
                    continue
                if _reconcile_delivery_record(
                    session_id, record, target_record
                ):
                    txn.touch()
                if not is_terminal(record) and record.get("phase") == PHASE_PENDING:
                    pending.append(dict(record))

        results: list[dict] = []
        for record in pending:
            target = str(record.get("to") or "")
            # Same delivery-record serialization the originating call takes.
            # A drain racing a ``follow_up_agent`` under one key would
            # otherwise resume the conversation twice.
            if not _claim_delivery_record(session_id, record):
                results.append(_delivery_in_progress(session_id, target, record))
                continue
            try:
                deadline = _delivery_clock() + _DELIVERY_CALL_BUDGET_SECONDS
                result = _guaranteed_delivery(
                    session_id,
                    target,
                    str(record.get("prompt") or ""),
                    bool((record.get("options") or {}).get("replace_if_idle", True)),
                    record,
                    deadline,
                )
                if result.get("reason") == "external_agent_pull_only":
                    refusals.append(
                        {
                            "idempotency_key": str(
                                record.get("idempotency_key") or ""
                            ),
                            "to": target,
                            "status": "refused",
                            "reason": "external_agent_pull_only",
                        }
                    )
                else:
                    results.append(result)
            finally:
                _release_delivery_claim(session_id, record)

        return {
            "success": True,
            "attempted": len(results) + len(refusals),
            "deliveries": delivery_store.list_for_sender(store, IDENTITY),
            "refusals": refusals,
        }

    def _guarded() -> dict:
        try:
            return _run()
        except DeliveryStoreError:
            logger.debug(
                "Delivery store write failed in deliver_pending", exc_info=True
            )
            return _store_unavailable("")
        except LeaseStoreError:
            logger.warning("Lease store unusable in deliver_pending", exc_info=True)
            return _store_unavailable("")

    return _annotate(await run_blocking(_guarded))


def _cleanup_agent_artifacts(
    session_id: str, name: str, *, child_exited: bool = True
) -> None:
    """Best-effort removal of a killed agent's on-disk artifacts.

    Prevents a later agent spawned with the same name from inheriting the dead
    agent's state marker, prompt sidecar, inbox messages, or read cursors.
    Every operation is best-effort and never raises.

    A5: per-attempt prompt sidecars are removed **only** when the child is
    confirmed gone, or once they age past :data:`_PROMPT_GC_AGE_SECONDS`.
    Deleting them because a new call started would race a concurrent attempt
    whose CLI has not read its file yet, and a timeout-failure alone is
    explicitly not enough — the process may still be about to read it.
    """
    for path in (
        _state_marker_file(session_id, name),
        # The old deterministic path, written by servers predating A5.
        _prompt_file(session_id, name),
        _inbox_file(session_id, name),
        _inbox_cursor_file(session_id, name),
    ):
        with suppress(OSError):
            path.unlink(missing_ok=True)
    _gc_prompt_files(session_id, name, child_exited=child_exited)
    # Wipe the killed agent's history from the lead/parent reader inbox and drop
    # its per-sender cursor entry, so a later same-name agent starts with a clean
    # slate. Purging the delivered messages (not just the cursor) is what keeps
    # the lead's already-read backlog from resurfacing as unread: unread is
    # `total_from_sender - consumed`, so deleting the cursor alone would reset
    # `consumed` to 0 while the messages remained. Serialized against a
    # concurrent read via the reader inbox lock.
    with _inbox_lock(IDENTITY):
        purge_sender_from_inbox(_inbox_file(session_id, IDENTITY), name)
        reader_cursor = _inbox_cursor_file(session_id, IDENTITY)
        cursors = _load_inbox_cursors(reader_cursor)
        if name in cursors:
            del cursors[name]
            with suppress(OSError):
                _save_inbox_cursors(reader_cursor, cursors)


def _gc_stale_prompt_files(session_id: str) -> None:
    """Age-collect EVERY stale prompt sidecar in the session (A5).

    Reachability, not correctness, is the point. A sidecar is written before
    ``backend.spawn``/``backend.resume``; if that call raises, the file is left
    with no agent record naming it. Sweeping only on ``kill_agent`` could never
    collect it — there is no agent left to kill — and sweeping only the name
    being delivered to would miss an orphan belonging to a name that never
    comes back. So the sweep is session-wide and runs on the paths that create
    sidecars in the first place: one glob, on a directory we are about to
    write to anyway.

    Age-based only, so it can never race a concurrent attempt whose CLI has
    not read its file yet.
    """
    for path in stale_prompt_files(
        _prompts_dir(session_id),
        "*",
        older_than=_PROMPT_GC_AGE_SECONDS,
        now=time.time(),
    ):
        remove_prompt_file(path)


def _gc_prompt_files(session_id: str, name: str, *, child_exited: bool) -> None:
    """Remove this agent's per-attempt prompt sidecars, conservatively (A5)."""
    directory = _prompts_dir(session_id)
    if child_exited:
        with suppress(OSError):
            for path in directory.glob(delivery.prompt_file_glob(name)):
                remove_prompt_file(path)
        return
    for path in stale_prompt_files(
        directory, name, older_than=_PROMPT_GC_AGE_SECONDS, now=time.time()
    ):
        remove_prompt_file(path)


@_register_tool()
async def kill_agent(name: str) -> dict:
    """Force-kill an agent and remove it from the session.

    kill is terminal: the agent record is removed from ``agents.json`` (so it
    no longer appears in ``list_agents`` and ``follow_up_agent`` returns
    ``agent_not_found``), and its per-agent state marker, prompt sidecar,
    inbox, and inbox cursor are cleaned up so a later agent spawned with the
    same name starts clean. The OS process is only signalled when we can prove
    ours (matching creation token or live in-memory ownership) — a reused or
    foreign PID is left untouched. A naturally-dead agent is NOT removed until
    killed, so it remains listable and resumable.

    For ``backend="external"``, the PID belongs to a manually started,
    potentially shared MCP server and is informational/stale. It is never
    probed or signalled: kill only removes the member record, marker, inbox,
    cursor, and sender history, returning ``killed_process=false`` and
    ``reason="external_agent_deregistered"``. Its bearer token is revoked on
    the next token-bearing call.
    """
    session_id = _active_session_id()

    def _do_kill() -> dict:
        if not session_id:
            return {"success": False, "name": name, "reason": "session_not_found"}
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return {"success": False, "name": name}

            if agent.get("backend") == "external":
                remaining = [row for row in agents if row.get("name") != name]
                agents[:] = remaining
                _save_agents_transaction(session_id, agents)
                if not _drop_agent_lease(_leases_file(session_id), name):
                    logger.warning("Could not drop the lease entry for %s", name)
                _cleanup_agent_artifacts(session_id, name, child_exited=True)
                return {
                    "success": True,
                    "name": name,
                    "killed_process": False,
                    "reason": "external_agent_deregistered",
                }

            # A4b — refuse while a delivery to this agent is provably in
            # flight. Killing now could orphan an already-spawned resumed
            # child, and waiting here would deadlock: the holder needs this
            # very lock to finalize. A lease whose holder is dead, or whose
            # creation token no longer matches, is reconciled automatically
            # and the kill proceeds. Ordinary kill NEVER bypasses a live
            # lease; the operator escape lives in the CLI, behind the session
            # recovery token.
            try:
                lease = reconcile_lease(
                    _leases_file(session_id), name, holder_probe=_lease_holder_probe
                )
            except LeaseStoreError:
                # Fail closed. We cannot read (or could not clear) the store
                # that says whether a delivery to this agent is in flight, so
                # we cannot claim there is none. Killing here could orphan an
                # already-spawned resumed child. The operator escape exists
                # for exactly this, and the reason names the store so it is
                # obvious which one to repair.
                logger.warning("Lease store unusable during kill", exc_info=True)
                return {
                    "success": False,
                    "name": name,
                    "reason": "lease_store_unavailable",
                    "retriable": True,
                    "detail": (
                        "The operation-lease store could not be read or "
                        "written, so whether a delivery to this agent is in "
                        "flight is unknown. Nothing was killed. Retry, or use "
                        "`win-agent-teams lease force` once the store is "
                        "readable."
                    ),
                }
            if lease is not None:
                return {
                    "success": False,
                    "name": name,
                    "reason": "operation_in_progress",
                    "retriable": True,
                    "holder_pid": lease.holder_pid,
                    "operation_id": lease.operation_id,
                    "detail": (
                        "A follow-up delivery to this agent is in flight and "
                        "its holder is provably alive. Retry once it settles, "
                        "or use the CLI operator path "
                        "(`win-agent-teams lease force`) if the holder is hung."
                    ),
                }

            # Reconcile BEFORE concluding anything. An in-flight attempt may
            # already have an unread receipt on disk, and recording that
            # message as failed because we are killing the target would
            # reintroduce exactly the false status this feature removes.
            # Records are settled here, never deleted: unlike the inbox lines
            # purged below, the sender's audit trail must outlive the target.
            _reconcile_deliveries_for_target(session_id, name, agent)

            owned = process_manager.owns_process(
                str(agent.get("pid")), _agent_create_token(agent)
            )
            # Fail closed: never kill a PID we cannot prove is still ours.
            if owned:
                process_manager.kill_process(str(agent["pid"]))
            remaining = [a for a in agents if a.get("name") != name]
            agents[:] = remaining
            _save_agents_transaction(session_id, agents)
            if not _drop_agent_lease(_leases_file(session_id), name):
                # Not fatal — the agent is gone either way — but a surviving
                # entry would be inherited by a same-named successor, so it is
                # reported rather than silently discarded.
                logger.warning("Could not drop the lease entry for %s", name)
            # The child is gone if we just killed it or it was already dead;
            # only then may its prompt sidecars be removed outright.
            _cleanup_agent_artifacts(
                session_id, name, child_exited=owned or not _agent_alive(agent)
            )
            return {"success": True, "name": name}

    return await run_blocking(_do_kill)


@_register_tool()
async def resume_session(session_id: str) -> dict:
    """Adopt a specific prior session by id after a restart.

    Use this when ``session_info`` (or a tool's ``recoverable_sessions`` nudge)
    reports more than one recoverable prior session for this workspace — e.g.
    several leads were started in the same folder. Pass the ``session_id``
    (the recovery token echoed by ``spawn_agent``/``session_info``, and visible
    in your own earlier transcript) to re-bind this lead to that session; its
    agents then reappear in ``list_agents`` and become resumable. Returns
    ``{success, session_id, agent_count, lead_token}`` or
    ``{success: False, reason}``.
    """

    def _do_resume() -> dict:
        global _session_id  # noqa: PLW0603 - module-level lead session state.
        refusal = _require_resolved_identity()
        if refusal is not None:
            # An unresolved child must never adopt (hijack) a workspace session.
            return refusal
        sid = session_id.strip()
        if not sid:
            return {"success": False, "reason": "session_id_required"}
        # Validate the id is UUID-shaped (blocks path traversal via ".." or
        # separators) and that its directory resolves directly under the
        # session base — never adopt an agents.json from an arbitrary path.
        try:
            uuid.UUID(sid)
        except ValueError:
            return {"success": False, "session_id": sid, "reason": "invalid_session_id"}
        sdir = _session_dir(sid)
        if (
            sdir.resolve().parent != _SESSION_BASE.resolve()
            or not _agents_file(sid).exists()
        ):
            return {"success": False, "session_id": sid, "reason": "session_not_found"}
        _session_id = sid
        _persist_session_binding(sid)  # re-bind current key + prune stale bindings
        try:
            agents = _load_agents(sid)
        except (OSError, json.JSONDecodeError, ValueError):
            agents = []
        return {
            "success": True,
            "session_id": sid,
            "agent_count": len(agents),
            "lead_token": _ensure_lead_token(sid),
        }

    return await run_blocking(_do_resume)


@_register_tool()
async def session_info() -> dict:
    """Report the current session and any recoverable prior sessions.

    Call this right after a restart if ``list_agents`` is unexpectedly empty:
    it returns ``{session_id, session_dir, identity, cwd, agent_count,
    lead_token, recoverable_sessions}``. ``session_dir`` is this session's
    on-disk directory (``""`` when no session exists yet).
    ``recoverable_sessions`` lists prior sessions for
    this workspace (``{session_id, agent_count, last_activity}``) that still
    hold resumable agents; adopt one with ``resume_session('<session_id>')``.
    ``lead_token`` is this session's stable recovery token.
    """
    session_id = _active_session_id()

    def _do_info() -> dict:
        cwd = str(Path.cwd().resolve())
        recoverable = [
            c for c in _candidate_sessions() if c["session_id"] != session_id
        ]
        if not session_id:
            return {
                "session_id": "",
                "session_dir": "",
                "identity": IDENTITY,
                "cwd": cwd,
                "agent_count": 0,
                "lead_token": None,
                "recoverable_sessions": recoverable,
            }
        try:
            agents = _load_agents(session_id)
        except (OSError, json.JSONDecodeError, ValueError):
            agents = []
        return {
            "session_id": session_id,
            "session_dir": str(_session_dir(session_id)),
            "identity": IDENTITY,
            "cwd": cwd,
            "agent_count": len(agents),
            "lead_token": _ensure_lead_token(session_id),
            "recoverable_sessions": recoverable,
        }

    return await run_blocking(_do_info)


def _marker_timestamp(marker: dict | None) -> float | None:
    """Return a numeric marker ``ts`` when present, else ``None``."""
    if marker is None:
        return None
    ts = marker.get("ts")
    if isinstance(ts, int | float) and not isinstance(ts, bool):
        return float(ts)
    return None


def _public_agent_record(agent: dict) -> dict:
    """Return registry fields with every credential-bearing field omitted."""
    return {
        key: value for key, value in agent.items() if key not in CREDENTIAL_FIELDS
    }


def _list_agents_row(session_id: str, agent: dict, alive: bool) -> dict:
    """Build a compact ``list_agents`` row (no leaked internal fields)."""
    public = _public_agent_record(agent)
    name = str(public.get("name") or "")
    marker = _read_state_marker(session_id, name)
    last_activity_at = _marker_timestamp(marker)
    # The binding is resolved only on the transcript-fallback path. A compact
    # row answered entirely from the state marker must not pay for a scan.
    # External members have no transcript binding by design, so that outcome
    # is known without a scan; for other backends ``None`` means "not
    # evaluated on this call".
    binding_outcome: str | None = (
        "not_applicable" if public.get("backend") == "external" else None
    )
    if last_activity_at is None:
        binding = _resolve_agent_binding(agent)
        binding_outcome = binding.outcome
        output = binding.output
        last_activity_at = output.last_activity_at if output else None
    state = _resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_at
    )
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    return {
        "name": name,
        "state": state,
        "alive": alive,
        "pid": public.get("pid"),
        "backend": public.get("backend"),
        "last_activity_at": last_activity_at,
        "unread_count": unread_count,
        # Binding outcome is its own field, never folded into lifecycle
        # ``state``: "this process is running" and "we know which transcript
        # is its" are independent facts.
        "binding": binding_outcome,
    }


@_register_tool()
@_with_disk_note
async def list_agents(full: bool = False) -> list[dict]:
    """List all agents with compact status rows.

    Default (``full=False``) rows are ``{name, state, alive, pid, backend,
    last_activity_at, unread_count, binding}`` — no transcript bodies. Pass
    ``full=True`` to restore each agent's sanitized registry fields
    (credential fields omitted) plus
    ``last_line`` (the last non-empty line of its most recent message),
    ``truncated``, and ``full_len`` (the untruncated character count).

    Recovery: if this returns empty right after a restart and you expected
    agents, call ``session_info()`` — a prior session for this workspace may
    be recoverable and adoptable via ``resume_session('<session_id>')``.
    """
    session_id = _active_session_id()

    def _do_list() -> list[dict]:
        if not session_id:
            return []
        agents = _load_agents(session_id)
        result = []
        for agent in agents:
            alive = _agent_alive(agent)
            if not full:
                result.append(_list_agents_row(session_id, agent, alive))
                continue
            binding = _resolve_agent_binding(agent)
            output = binding.output
            last_line, truncated, full_len = _truncate(
                _last_non_empty_line(output.last_message if output else None),
                _DEFAULT_LAST_LINE_MAX_CHARS,
            )
            result.append(
                {
                    **_public_agent_record(agent),
                    "alive": alive,
                    "last_line": last_line,
                    "truncated": truncated,
                    "full_len": full_len,
                    "binding": binding.outcome,
                    # The raw record's ``backend_session_id`` is echoed here,
                    # so it must never read as verified unless it actually is.
                    "backend_session_id_verified": binding.bound,
                }
            )
        return result

    return await run_blocking(_do_list)


def _agent_status_row(session_id: str, agent: dict) -> dict:
    """Build one ``agent_status`` row (marker + cursor reads only, no scan)."""
    public = _public_agent_record(agent)
    name = str(public.get("name") or "")
    alive = _agent_alive(agent)
    marker = _read_state_marker(session_id, name)
    last_activity_ts = _marker_timestamp(marker)
    unbound = False
    binding_outcome: str | None = (
        "not_applicable" if public.get("backend") == "external" else None
    )
    if last_activity_ts is None:
        # No marker timestamp: fall back to the transcript, but stay cheap.
        # This is exactly one binding resolution AND a genuinely bounded one:
        # ``bounded_only`` drops the resolver's own all-history fallback, which
        # A6 forbids here. Without it "one call" was still unbounded work, and
        # a test that mocks the whole resolver cannot tell the difference.
        binding = _resolve_agent_binding(agent, bounded_only=True)
        binding_outcome = binding.outcome
        output = binding.output
        last_activity_ts = output.last_activity_at if output else None
        unbound = not binding.bound and binding.outcome not in {
            BINDING_LEGACY,
            "not_applicable",
        }
    state = _resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_ts
    )
    # When the fallback produced no trustworthy binding there is no activity
    # signal we are entitled to report, so say ``unknown`` rather than guess
    # ``running``/``idle`` from someone else's mtime. Liveness and an
    # authoritative marker still win: a dead process is dead, and a hook-written
    # ``waiting``/``running`` is a direct observation, not an inference.
    if unbound and alive and state not in _VALID_MARKER_STATES:
        state = "unknown"
    seq = _sender_message_count(session_id, IDENTITY, name)
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    heartbeat_age_s, stalled = _heartbeat_fields(
        alive=alive, state=state, last_activity_ts=last_activity_ts
    )
    return {
        "name": name,
        "backend": public.get("backend"),
        "state": state,
        "last_activity_ts": last_activity_ts,
        "unread_count": unread_count,
        "seq": seq,
        "heartbeat_age_s": heartbeat_age_s,
        "stalled": stalled,
        "binding": binding_outcome,
    }


@_register_tool()
@_with_disk_note
async def agent_status(names: list[str] | None = None) -> list[dict]:
    """Return cheap per-agent status rows: no bodies, no transcript scan.

    Each row is exactly ``{name, backend, state, last_activity_ts,
    unread_count, seq, heartbeat_age_s, stalled, binding}``.
    ``seq``/``unread_count`` are the caller's
    per-sender count for messages FROM that named agent. ``names=None``
    returns all agents in the session; otherwise only the named agents
    (unknown names are skipped).

    ``heartbeat_age_s`` (float, disk-derived) is seconds since the agent's
    last known activity (marker ``ts`` when available, else transcript
    activity); ``None`` when no activity signal exists yet. ``stalled``
    (bool) is ``True`` only when the agent is alive, its ``state`` is
    neither ``waiting`` nor ``dead``, and ``heartbeat_age_s`` exceeds the
    stall threshold (``STALL_SECONDS``, default 300s, env-overridable via
    ``WIN_AGENT_TEAMS_STALL_SECONDS``). Answers "alive but hung" from disk
    alone, with zero transcript bytes.

    Cost model: one state-marker read + one cursor read + one liveness check
    per agent. The marker (written by a Stop/SessionStart/etc. hook) is used
    directly when present; a transcript scan only happens as a fallback when
    no marker exists yet (e.g. hooks disabled or not yet fired).

    External rows always report ``backend="external"`` and
    ``binding="not_applicable"``. Their PID is informational and never probed;
    running/left membership plus the activity marker are the honest liveness
    signal.

    Recovery: if this returns empty right after a restart and you expected
    agents, call ``session_info()`` — a prior session for this workspace may
    be recoverable and adoptable via ``resume_session('<session_id>')``.
    """
    session_id = _active_session_id()

    def _do_status() -> list[dict]:
        if not session_id:
            return []
        agents = _load_agents(session_id)
        if names is not None:
            wanted = set(names)
            agents = [a for a in agents if a.get("name") in wanted]
        return [_agent_status_row(session_id, agent) for agent in agents]

    return await run_blocking(_do_status)


@_register_tool()
@_with_disk_note
async def agent_watch_paths(names: list[str] | None = None) -> dict:
    """Return session watch metadata and minimal agent watch-path rows.

    Rediscover exactly what to watch when you did not retain a
    ``spawn_agent`` return (e.g. after resuming a session). ``names=None``
    returns all agents in the session; otherwise only the named agents
    (unknown names are skipped). Each row contains only ``name`` and
    ``state_marker_path``. Use ``has_session`` as the authoritative
    discriminator between no session and a live session with zero agents.
    """
    session_id = _active_session_id()

    def _do_watch_paths() -> dict:
        if not session_id:
            return {
                "has_session": False,
                "session_dir": "",
                "watch_argv": [],
                "watch_command_bash": "",
                "watch_command_powershell": "",
                "agents": [],
            }
        agents = _load_agents(session_id)
        if names is not None:
            wanted = set(names)
            agents = [a for a in agents if a.get("name") in wanted]
        session_dir = str(_session_dir(session_id))
        return {
            "has_session": True,
            "session_dir": session_dir,
            "watch_argv": _watch_argv(session_dir),
            "watch_command_bash": _watch_command_bash(session_dir),
            "watch_command_powershell": _watch_command_powershell(session_dir),
            "agents": [
                {
                    "name": str(agent.get("name") or ""),
                    "state_marker_path": str(
                        _state_marker_file(session_id, str(agent.get("name") or ""))
                    ),
                }
                for agent in agents
            ],
        }

    return await run_blocking(_do_watch_paths)


@_register_tool(external=True)
async def list_backends() -> list[dict]:
    """List available spawner backends."""

    def _do_list() -> list[dict]:
        result = []
        for bname in registry.list_available():
            b = registry.get(bname)
            result.append(
                {
                    "name": bname,
                    "binary": b.binary_name,
                    "default_model": b.default_model(),
                    "supported_models": b.supported_models(),
                }
            )
        return result

    return await run_blocking(_do_list)


def _group_has_wake_token(group: object) -> bool:
    """Return whether a Stop matcher group invokes the lead-wake module."""
    if not isinstance(group, dict):
        return False
    entries = cast("dict[str, Any]", group).get("hooks")
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(h, dict)
        and hooks._WAKE_MODULE in str(cast("dict[str, Any]", h).get("command", ""))
        for h in entries
    )


def _install_wake_hook(
    config: dict[str, Any], wake_matcher: dict[str, Any], *, remove: bool
) -> dict[str, Any]:
    """Return ``config`` with the lead-wake ``Stop`` matcher upserted or removed.

    Idempotent and non-destructive: any existing lead-wake group is dropped
    first (so a re-install never duplicates), then the fresh matcher is appended
    unless ``remove``. Unrelated events and unrelated ``Stop`` groups (e.g. a
    hand-written ``emit`` group) are preserved verbatim.
    """
    result = dict(config)
    hooks_map = dict(result.get("hooks") or {})
    stop = [g for g in (hooks_map.get("Stop") or []) if not _group_has_wake_token(g)]
    if not remove:
        stop.append(wake_matcher)
    if stop:
        hooks_map["Stop"] = stop
    else:
        hooks_map.pop("Stop", None)
    if hooks_map:
        result["hooks"] = hooks_map
    else:
        result.pop("hooks", None)
    return result


def _lead_wake_settings_path(scope: str) -> Path:
    """Return the settings path for ``scope`` (``project`` cwd or ``user`` home)."""
    base = Path.home() if scope == "user" else Path.cwd()
    return base / ".claude" / "settings.json"


@_register_tool()
async def install_lead_wake(remove: bool = False, scope: str = "project") -> dict:
    """Install (or remove) the Claude Code lead inbox-wake ``Stop`` hook.

    Wires the deterministic wake hook into a settings file so an idle lead is
    reliably woken when a worker replies. On every lead turn end the hook
    verifies — from the harness's own ``background_tasks`` — that an inbox
    watcher is armed, and blocks with an operational instruction to run the
    ``watch_command_bash``/``watch_argv`` as a BACKGROUND task (or to call
    ``read_messages`` when a reply is already unread) rather than trusting the
    model to remember. Server-spawned agents get this wiring automatically; use
    this tool to wire a **top-level** lead you started yourself (e.g. an
    interactive ``claude`` in a repo).

    - ``scope="project"`` (default) writes the project ``.claude/settings.json``
      in the current working directory; ``scope="user"`` writes
      ``~/.claude/settings.json``.
    - Writes ONLY the ``Stop`` wake group (never the state-marker ``emit`` hooks,
      which are for server-spawned agents). Idempotent: re-running replaces the
      existing wake group in place and never duplicates it; unrelated hooks are
      preserved. ``remove=True`` drops only the wake group.
    - The hook writes a small ``wake-progress-<reader>.json`` file under the
      session dir to bound a no-progress block loop. Disable at runtime with the
      kill switch ``WIN_AGENT_TEAMS_LEAD_WAKE=0`` (no reinstall needed).
    """
    if scope not in {"project", "user"}:
        return {"error": f"scope must be 'project' or 'user', got {scope!r}"}

    def _do_install() -> dict:
        identity = IDENTITY
        session_id = _active_session_id(create=False)
        session_dir = _session_dir(session_id) if session_id else _SESSION_BASE
        wake_matcher = hooks._wake_hook_matcher(session_dir, identity)
        path = _lead_wake_settings_path(scope)
        updated = _install_wake_hook(
            _read_json_object(path), wake_matcher, remove=remove
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return {
            "action": "removed" if remove else "installed",
            "path": str(path),
            "reader": identity,
            "scope": scope,
        }

    return await run_blocking(_do_install)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
