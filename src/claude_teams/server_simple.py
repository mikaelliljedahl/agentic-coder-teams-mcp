"""Simplified MCP server for agent orchestration.

Watch command discovery assumes this distribution is installed in the normal
Python environment of the interpreter running the server. The generated
``python -m claude_teams.cli`` argv is not supported for embedded or frozen
hosts, and a coordinator cwd containing another ``claude_teams`` checkout can
shadow the installed module.
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from fastmcp import FastMCP

from claude_teams import hooks
from claude_teams.agent_output import (
    claude_correlation_token,
    codex_correlation_token,
    read_claude_output,
    read_codex_output,
    read_pi_output,
)
from claude_teams.async_utils import run_blocking
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import process_manager
from claude_teams.backends.registry import registry
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


class AgentsFileLockTimeoutError(TimeoutError):
    """Raised when another MCP process holds the agents registry lock too long."""


mcp = FastMCP(
    name="win-agent-teams",
    instructions="Spawn and communicate with Claude Code and Codex agents.",
)

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


def _prompt_file(session_id: str, name: str) -> Path:
    """Return the per-agent prompt sidecar path for lossless Claude launches."""
    return _session_dir(session_id) / "prompts" / f"{name}.prompt.txt"


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
    if os.name == "nt":
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_SIZE)
            except OSError as err:
                if time.monotonic() >= deadline:
                    raise AgentsFileLockTimeoutError from err
                time.sleep(_LOCK_RETRY_SECONDS)
            else:
                return
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_SIZE)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    if _autoadopt_enabled() and len(candidates) == 1 and single_lead_history:
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


def _message_recipient(to: str, session_id: str) -> tuple[str, str | None]:
    """Resolve a ``send_message`` recipient, never dropping it to a dead inbox.

    Returns ``(recipient, warning)``. Rules:

    * ``"team-lead"``/``"lead"`` (and common aliases like
      ``"orchestrator"``/``"parent"``) resolve to the agent that spawned this
      one. For the root lead they stay ``ROOT_LEAD_NAME`` (its own inbox).
    * A name that matches a known agent in this session is used verbatim
      (a lead addressing a child, or a sibling).
    * Any other / unknown name is routed to the lead anyway, with a warning,
      so a typo'd recipient can never be silently written to an inbox no one
      reads.
    """
    raw = to.strip()
    lead_target = (
        (_AGENT_PARENT_NAME or ROOT_LEAD_NAME)
        if IDENTITY != ROOT_LEAD_NAME
        else ROOT_LEAD_NAME
    )

    if raw.lower() in _LEAD_ALIASES:
        return lead_target, None

    known = {a.get("name") for a in _load_agents(session_id) if a.get("name")}
    if raw in known:
        return raw, None

    warning = (
        f"unknown recipient {to!r}; routed to {lead_target!r}. "
        'Use to="team-lead" to reach whoever spawned you.'
    )
    return lead_target, warning


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


def _watch_argv(session_dir: str | Path, timeout: float | None = None) -> list[str]:
    """Return the canonical shell-neutral argv for watching ``session_dir``."""
    argv = [sys.executable, "-m", "claude_teams.cli", "watch", str(session_dir)]
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
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


def _read_agent_output(agent: dict):
    """Read fallback output for an agent record."""
    backend = agent.get("backend")
    spawned_at = _safe_float(agent.get("spawned_at"))
    cwd = str(agent.get("cwd") or "")
    if spawned_at <= 0 or not cwd:
        return None
    backend_session_id = _stored_backend_session_id(agent)
    if backend == "codex":
        agent_id = f"{agent.get('name')}@{agent.get('session_id')}"
        return read_codex_output(
            spawned_at,
            cwd,
            backend_session_id=backend_session_id,
            correlation_token=codex_correlation_token(agent_id),
        )
    if backend == "claude-code":
        agent_id = f"{agent.get('name')}@{agent.get('session_id')}"
        return read_claude_output(
            spawned_at,
            cwd,
            backend_session_id=backend_session_id,
            correlation_token=claude_correlation_token(agent_id),
        )
    if backend == "pi":
        name = str(agent.get("name") or "")
        session_id = str(agent.get("session_id") or "")
        return read_pi_output(
            str(_pi_session_dir(session_id, name)),
            expected_session_id=backend_session_id or name,
        )
    return None


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


def _sync_backend_session_id(agent: dict, output) -> bool:
    """Persist a newly discovered backend session id onto an agent record."""
    if output is None or not output.backend_session_id:
        return False
    if agent.get("backend_session_id") == output.backend_session_id:
        return False
    agent["backend_session_id"] = output.backend_session_id
    return True


def _agent_check_payload(name: str, agent: dict, alive: bool, output) -> dict:
    """Build the rich INTERNAL check payload for an existing agent record.

    Consumed by ``follow_up_agent``/``_follow_up_failure`` (which need the
    unbounded ``last_message`` to decide busy/idle) and projected down to the
    compact public ``check_agent`` shape by ``_compact_check_view``.
    """
    backend_session_id = _stored_backend_session_id(agent)
    return {
        "name": name,
        "alive": alive,
        "pid": agent["pid"],
        "backend": agent.get("backend"),
        "backend_session_id": backend_session_id,
        "last_activity_at": output.last_activity_at if output else None,
        "last_message": output.last_message if output else None,
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
    }


def _follow_up_failure(reason: str, name: str, status: dict | None = None) -> dict:
    """Build a structured ``follow_up_agent`` failure payload."""
    payload: dict[str, object] = {
        "success": False,
        "name": name,
        "reason": reason,
    }
    if status:
        payload.update(
            {
                "alive": status.get("alive"),
                "backend_session_id": status.get("backend_session_id"),
                "last_activity_at": status.get("last_activity_at"),
                "last_message": status.get("last_message"),
            }
        )
    return payload


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


def _write_prompt_file_extra(
    session_id: str, agent_name: str, backend_name: str, prompt: str
) -> dict[str, str]:
    """Write a lossless prompt sidecar when argv transport is risky.

    Claude Code only needs it for prompts with CLI-sensitive characters. Pi
    always gets one so the backend can fall back to a ``@<file>`` include if it
    is forced through the ``pi.cmd`` shim (see ``PiBackend._prompt_args``); on
    the normal direct-``node`` launch the sidecar is written but unused.
    """
    needs_file = backend_name == "pi" or (
        backend_name == "claude-code"
        and any(char in prompt for char in _CLAUDE_PROMPT_FILE_CHARS)
    )
    if not needs_file:
        return {}
    path = _prompt_file(session_id, agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return {"prompt_file_path": str(path)}


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


@mcp.tool()
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
            agent_name = _unique_agent_name(name, agents)

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
            extra = {
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
                "session_dir": str(_session_dir(session_id)),
                **_write_prompt_file_extra(
                    session_id, agent_name, backend_name, prompt
                ),
                **_hook_extra(session_id, agent_name, backend_name),
            }

            request = SpawnRequest(
                agent_id=f"{agent_name}@{session_id}",
                name=agent_name,
                team_name=session_id,
                prompt=prompt,
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


@mcp.tool()
async def send_message(text: str, to: str = "team-lead") -> dict:
    """Write a message to an inbox for agents that actively poll read_messages.

    ``to`` defaults to ``"team-lead"``, which reaches the agent that spawned you —
    that is almost always what you want from a subagent. A lead can target a
    child by its agent name. Any unknown recipient is routed to the lead with a
    ``warning`` in the result rather than silently written to a dead inbox.

    This is not a push/resume mechanism: a spawned agent will only see this
    message if it calls read_messages after the message is sent. If the agent
    is not polling, use follow_up_agent instead.
    """
    refusal = _require_resolved_identity()
    if refusal is not None:
        return {**refusal, "to": to}
    session_id = _active_session_id()

    def _do_send() -> dict:
        if not session_id:
            return {"success": False, "to": to, "reason": "session_not_found"}
        recipient, warning = _message_recipient(to, session_id)
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
        result = {"success": True, "to": recipient}
        if warning:
            result["warning"] = warning
        return result

    return _annotate(await run_blocking(_do_send))


_DEFAULT_READ_LIMIT = 50


@mcp.tool()
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
        inbox = _inbox_file(session_id, IDENTITY)
        cursor_file = _inbox_cursor_file(session_id, IDENTITY)
        with _inbox_lock(IDENTITY):
            cursors = _load_inbox_cursors(cursor_file)
            # A missing/empty inbox is just an empty snapshot. We must NOT
            # early-return here: a stored forward cursor still needs clamping
            # and persisting, otherwise a bad value would survive and could
            # later swallow a sender's first message.
            # Group valid messages by sender in file order, tracking each
            # message's global position so the result preserves file order.
            by_sender = read_inbox_by_sender(inbox)

            # Clamp any stored count that exceeds the observed valid-message
            # count for that sender down to the observed count. This covers
            # senders absent from the current snapshot too (clamped to 0), so a
            # bad forward cursor cannot skip that sender's first future message.
            for sender in list(cursors):
                observed = len(by_sender.get(sender, []))
                cursors[sender] = min(cursors[sender], observed)

            relevant = [from_agent] if from_agent else list(by_sender)
            # start_overrides holds the effective read-start position (a
            # per-sender COUNT) for THIS call, distinct from the persisted
            # cursor value used as the no-newly-selected floor below.
            start_overrides: dict[str, int] = {}
            if since_seq is not None and from_agent:
                floor = cursors.get(from_agent, 0)
                start_overrides[from_agent] = max(since_seq, floor, 0)

            # Per-sender batch entries, each tagged with its PER-SENDER
            # position (0-based index into that sender's own entries list,
            # i.e. seq - 1) so the global-index tuples from ``by_sender``
            # never leak into cross-sender bookkeeping.
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
            selected.sort(key=lambda item: item[2])  # global file order
            # The pending backlog is measured before any limit clipping so a
            # non-consuming peek (limit=0) or a clipped batch still reports the
            # true unread count instead of just the returned batch size.
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
                    for sel_sender, position, _, _ in selected
                    if sel_sender == sender
                ]
                if consumed_positions:
                    new_count = max(consumed_positions) + 1
                else:
                    new_count = start_overrides.get(sender, cursors.get(sender, 0))
                floor = max(cursors.get(sender, 0), start_overrides.get(sender, 0))
                new_count = max(new_count, floor)
                if entries or sender in cursors or sender in start_overrides:
                    updated[sender] = min(new_count, len(entries))

            _save_inbox_cursors(cursor_file, updated)

            messages: list[dict] = []
            for sender, position, _index, msg in selected:
                text, truncated, full_len = _truncate(msg.get("text"), max_chars)
                entry = {
                    "from": sender,
                    "text": text,
                    "ts": msg.get("ts"),
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

    return _annotate(await run_blocking(_do_read))


@mcp.tool()
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
            output = _read_agent_output(agent)
            if _sync_backend_session_id(agent, output):
                _save_agents_transaction(session_id, agents)
            internal = _agent_check_payload(name, agent, alive, output)
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


@mcp.tool()
async def follow_up_agent(
    name: str,
    prompt: str,
    replace_if_idle: bool = True,
) -> dict:
    """Resume a logical agent with a follow-up prompt through the backend CLI.

    Use this instead of send_message when the target agent is not polling read_messages.
    send_message only writes to an inbox; follow_up_agent is the mechanism for
    continuing a spawned agent that would otherwise never read an inbox message.
    It only runs when the agent is dead or idle; a live busy agent is refused
    with reason="agent_busy".

    replace_if_idle defaults to True: an idle-but-alive process is gracefully
    shut down and resumed with the follow-up prompt. Set it to False to instead
    refuse such an agent with reason="agent_idle_but_alive".
    """
    session_id = _active_session_id()

    def _do_follow_up() -> dict:  # noqa: PLR0911, PLR0912 - mirrors explicit refusal reasons.
        if not session_id:
            return _follow_up_failure("session_not_found", name)
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return _follow_up_failure("agent_not_found", name)

            backend_name = str(agent.get("backend") or "")
            try:
                backend = registry.get(backend_name)
            except Exception:
                logger.debug("Failed loading backend for follow-up", exc_info=True)
                return _follow_up_failure("backend_not_supported", name)

            if not getattr(backend, "supports_resume", lambda: False)():
                return _follow_up_failure("backend_not_supported", name)

            pid = agent["pid"]
            alive = _agent_alive(agent)
            output = _read_agent_output(agent)
            changed = _sync_backend_session_id(agent, output)
            status = _agent_check_payload(name, agent, alive, output)
            backend_session_id = status.get("backend_session_id")
            if not backend_session_id:
                if changed:
                    _save_agents_transaction(session_id, agents)
                return _follow_up_failure("backend_session_missing", name, status)

            if alive:
                last_message = status.get("last_message")
                if last_message is None:
                    if changed:
                        _save_agents_transaction(session_id, agents)
                    return _follow_up_failure("agent_busy", name, status)
                last_activity_at = status.get("last_activity_at")
                if last_activity_at is None:
                    if changed:
                        _save_agents_transaction(session_id, agents)
                    return _follow_up_failure("agent_state_unknown", name, status)
                # A hook-written "waiting" marker is an authoritative idle
                # signal: the agent has reached a wait/stop hook and is parked
                # awaiting input, so we can resume it immediately. The busy_hint
                # and inactivity-timer checks below are heuristics that exist
                # only for when no reliable marker is available; a "waiting"
                # marker overrides both, avoiding a needless wait for an agent
                # we already know is idle.
                idle_by_marker = (
                    _resolve_agent_state(
                        alive=True,
                        marker=_read_state_marker(session_id, name),
                        last_activity_at=last_activity_at,
                    )
                    == "waiting"
                )
                if not idle_by_marker:
                    if output is not None and output.busy_hint:
                        if changed:
                            _save_agents_transaction(session_id, agents)
                        return _follow_up_failure("agent_busy", name, status)
                    if time.time() - float(last_activity_at) < _FOLLOW_UP_IDLE_SECONDS:
                        if changed:
                            _save_agents_transaction(session_id, agents)
                        return _follow_up_failure("agent_busy", name, status)
                if not replace_if_idle:
                    if changed:
                        _save_agents_transaction(session_id, agents)
                    return _follow_up_failure("agent_idle_but_alive", name, status)

                # Fail closed: only shut down / kill the PID when we can prove
                # it is still OUR process (in-memory ownership or a matching
                # creation token). A tokenless recovered record or a reused PID
                # is left untouched — we resume via backend_session_id instead
                # of risking a foreign kill.
                if process_manager.owns_process(
                    str(pid), _agent_create_token(agent)
                ) and not process_manager.graceful_shutdown(str(pid), timeout_s=5.0):
                    process_manager.kill_process(str(pid))

            agent_name = str(agent.get("name") or name)
            agent_cwd = str(agent.get("cwd") or Path.cwd())
            mcp_config_path = _write_mcp_config(session_id, agent_name, IDENTITY)

            # Reuse the concrete model resolved at spawn. Preserve a stored
            # blank verbatim (blank means "defer to codex config"); only a
            # genuinely absent key falls back to the backend default. Do NOT
            # coerce blank via ``or`` — default_model() may be a capability
            # tier name (e.g. "medium"), which must never reach ``-c model``.
            stored_model = agent.get("model")
            model = (
                stored_model
                if isinstance(stored_model, str)
                else backend.default_model()
            )
            permission_mode = str(agent.get("permission_mode") or "bypass")
            effort_value = agent.get("reasoning_effort")
            effort = effort_value if isinstance(effort_value, str) else None
            extra = {
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
                "session_dir": str(_session_dir(session_id)),
                **_write_prompt_file_extra(
                    session_id, agent_name, backend_name, prompt
                ),
                **_hook_extra(session_id, agent_name, backend_name),
            }
            request = SpawnRequest(
                agent_id=f"{agent_name}@{session_id}",
                name=agent_name,
                team_name=session_id,
                prompt=prompt,
                model=model,
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

            try:
                result = backend.resume(request, str(backend_session_id))
            except Exception:
                logger.debug("Failed resuming backend session", exc_info=True)
                if changed:
                    _save_agents_transaction(session_id, agents)
                return _follow_up_failure("resume_failed", name, status)

            new_pid = int(result.process_handle)
            new_create_token = process_manager.creation_token(str(new_pid))
            agent.update(
                {
                    "pid": new_pid,
                    "backend": backend_name,
                    "session_id": session_id,
                    "status": "running",
                    "spawned_at": time.time(),
                    "cwd": agent_cwd,
                    "backend_session_id": str(backend_session_id),
                    "model": model,
                    "permission_mode": permission_mode,
                    "reasoning_effort": effort,
                    "create_token": new_create_token,
                }
            )
            _save_agents_transaction(session_id, agents)
            return {
                "success": True,
                "name": agent_name,
                "pid": new_pid,
                "backend": backend_name,
                "backend_session_id": str(backend_session_id),
                "replaced_existing": alive,
                "session_id": session_id,
            }

    return _annotate(await run_blocking(_do_follow_up))


def _cleanup_agent_artifacts(session_id: str, name: str) -> None:
    """Best-effort removal of a killed agent's on-disk artifacts.

    Prevents a later agent spawned with the same name from inheriting the dead
    agent's state marker, prompt sidecar, inbox messages, or read cursors.
    Every operation is best-effort and never raises.
    """
    for path in (
        _state_marker_file(session_id, name),
        _prompt_file(session_id, name),
        _inbox_file(session_id, name),
        _inbox_cursor_file(session_id, name),
    ):
        with suppress(OSError):
            path.unlink(missing_ok=True)
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


@mcp.tool()
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
    """
    session_id = _active_session_id()

    def _do_kill() -> dict:
        if not session_id:
            return {"success": False, "name": name, "reason": "session_not_found"}
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return {"success": False, "name": name}
            # Fail closed: never kill a PID we cannot prove is still ours.
            if process_manager.owns_process(
                str(agent.get("pid")), _agent_create_token(agent)
            ):
                process_manager.kill_process(str(agent["pid"]))
            remaining = [a for a in agents if a.get("name") != name]
            agents[:] = remaining
            _save_agents_transaction(session_id, agents)
            _cleanup_agent_artifacts(session_id, name)
            return {"success": True, "name": name}

    return await run_blocking(_do_kill)


@mcp.tool()
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


@mcp.tool()
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


def _list_agents_row(session_id: str, agent: dict, alive: bool) -> dict:
    """Build a compact ``list_agents`` row (no leaked internal fields)."""
    name = str(agent.get("name") or "")
    marker = _read_state_marker(session_id, name)
    last_activity_at = _marker_timestamp(marker)
    if last_activity_at is None:
        output = _read_agent_output(agent)
        last_activity_at = output.last_activity_at if output else None
    state = _resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_at
    )
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    return {
        "name": name,
        "state": state,
        "alive": alive,
        "pid": agent.get("pid"),
        "backend": agent.get("backend"),
        "last_activity_at": last_activity_at,
        "unread_count": unread_count,
    }


@mcp.tool()
@_with_disk_note
async def list_agents(full: bool = False) -> list[dict]:
    """List all agents with compact status rows.

    Default (``full=False``) rows are ``{name, state, alive, pid, backend,
    last_activity_at, unread_count}`` — no transcript bodies. Pass
    ``full=True`` to restore each agent's raw registry record plus
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
            output = _read_agent_output(agent)
            last_line, truncated, full_len = _truncate(
                _last_non_empty_line(output.last_message if output else None),
                _DEFAULT_LAST_LINE_MAX_CHARS,
            )
            result.append(
                {
                    **agent,
                    "alive": alive,
                    "last_line": last_line,
                    "truncated": truncated,
                    "full_len": full_len,
                }
            )
        return result

    return await run_blocking(_do_list)


def _agent_status_row(session_id: str, agent: dict) -> dict:
    """Build one ``agent_status`` row (marker + cursor reads only, no scan)."""
    name = str(agent.get("name") or "")
    alive = _agent_alive(agent)
    marker = _read_state_marker(session_id, name)
    last_activity_ts = _marker_timestamp(marker)
    if last_activity_ts is None:
        output = _read_agent_output(agent)
        last_activity_ts = output.last_activity_at if output else None
    state = _resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_ts
    )
    seq = _sender_message_count(session_id, IDENTITY, name)
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    heartbeat_age_s, stalled = _heartbeat_fields(
        alive=alive, state=state, last_activity_ts=last_activity_ts
    )
    return {
        "name": name,
        "state": state,
        "last_activity_ts": last_activity_ts,
        "unread_count": unread_count,
        "seq": seq,
        "heartbeat_age_s": heartbeat_age_s,
        "stalled": stalled,
    }


@mcp.tool()
@_with_disk_note
async def agent_status(names: list[str] | None = None) -> list[dict]:
    """Return cheap per-agent status rows: no bodies, no transcript scan.

    Each row is exactly ``{name, state, last_activity_ts, unread_count, seq,
    heartbeat_age_s, stalled}``. ``seq``/``unread_count`` are the caller's
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
