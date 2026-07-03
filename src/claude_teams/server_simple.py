"""Simplified MCP server for agent orchestration."""

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from fastmcp import FastMCP

from claude_teams import hooks
from claude_teams.agent_output import (
    codex_correlation_token,
    read_claude_output,
    read_codex_output,
)
from claude_teams.async_utils import run_blocking
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import process_manager
from claude_teams.backends.registry import registry

# Identity: env vars (works for Claude Code via --mcp-config)
# For Codex: the codex backend passes identity per-spawn via a `-c
# mcp_servers.<name>.env=...` override (see CodexBackend._mcp_identity_args),
# avoiding races on the shared ~/.codex/config.toml.
_AGENT_NAME: str = os.environ.get("AGENT_NAME", "").strip()
_AGENT_SESSION_ID: str = os.environ.get("AGENT_SESSION_ID", "").strip()
_AGENT_PARENT_NAME: str = os.environ.get("AGENT_PARENT_NAME", "").strip()
IDENTITY: str = _AGENT_NAME if _AGENT_NAME else "lead"

# Names a subagent might reasonably use to mean "whoever spawned me". All of
# these resolve to the lead/parent so a message is never lost to a typo'd
# recipient. Compared case-insensitively.
_LEAD_ALIASES: frozenset[str] = frozenset(
    {"", "lead", "orchestrator", "parent", "boss", "manager", "up", "supervisor"}
)

_SESSION_BASE = Path.home() / ".claude" / "agent-sessions"
_SESSION_META_NAME = "session.json"
_BINDINGS_DIR_NAME = "bindings"
_AGENTS_LOCK_NAME = "agents.lock"
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.05
_LOCK_SIZE = 1
_FOLLOW_UP_IDLE_SECONDS = 60.0
logger = logging.getLogger(__name__)


def _idle_seconds() -> float:
    """Return the activity-fallback idle threshold, env-overridable."""
    raw = os.environ.get("WIN_AGENT_TEAMS_IDLE_SECONDS", "").strip()
    if not raw:
        return _FOLLOW_UP_IDLE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _FOLLOW_UP_IDLE_SECONDS


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
    marker's state is used verbatim; otherwise fall back to an
    activity-recency heuristic (``"running"`` vs ``"idle"``).
    """
    if not alive:
        return "dead"
    if marker is not None:
        state = marker.get("state")
        if isinstance(state, str) and state:
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


def _load_inbox_cursors(path: Path) -> dict[str, int]:
    """Load the per-sender counter sidecar, rejecting anything malformed.

    Requires a JSON object whose keys are strings and values are non-negative
    ints (``bool`` is excluded: ``isinstance(True, int)`` is ``True``). Any
    corrupt or unreadable file is treated as empty and logged; this never
    raises.
    """
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("ignoring corrupt inbox cursor file %s: %s", path, err)
        return {}
    if not isinstance(value, dict):
        logger.warning("ignoring non-dict inbox cursor file %s", path)
        return {}
    cursors: dict[str, int] = {}
    for key, count in value.items():
        if (
            isinstance(key, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        ):
            cursors[key] = count
    return cursors


def _save_inbox_cursors(path: Path, cursors: dict[str, int]) -> None:
    """Atomically persist the counter via a uniquely named temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cursors), encoding="utf-8")
    tmp.replace(path)  # atomic rename onto the sidecar


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


def _persist_session_binding(session_id: str) -> None:
    """Bind this parent/workspace to a session id for MCP restart recovery."""
    now = datetime.now(UTC).isoformat()
    key = _binding_key()
    meta = {
        "session_id": session_id,
        "binding_key": key,
        "identity": IDENTITY,
        "cwd": str(Path.cwd().resolve()),
        "parent_id": os.environ.get("WIN_AGENT_TEAMS_PARENT_ID", "").strip()
        or str(os.getppid()),
        "updated_at": now,
    }
    _session_meta_file(session_id).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    _bindings_dir().mkdir(parents=True, exist_ok=True)
    _binding_file().write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _recover_session_id() -> str:
    """Recover the persisted lead session for this MCP parent/workspace."""
    if _AGENT_SESSION_ID:
        return _AGENT_SESSION_ID
    key = _binding_key()
    binding = _read_json_object(_binding_file())
    session_id = binding.get("session_id")
    if (
        binding.get("binding_key") == key
        and isinstance(session_id, str)
        and _agents_file(session_id).exists()
    ):
        return session_id
    return ""


def _active_session_id(*, create: bool = False) -> str:
    """Return the current session id, recovering persisted lead state if needed."""
    global _session_id  # noqa: PLW0603 - module-level lead session state.

    if _session_id:
        return _session_id
    recovered = _recover_session_id()
    if recovered:
        _session_id = recovered
        return _session_id
    if create:
        _session_id = _create_session()
        return _session_id
    return ""


def _message_recipient(to: str, session_id: str) -> tuple[str, str | None]:
    """Resolve a ``send_message`` recipient, never dropping it to a dead inbox.

    Returns ``(recipient, warning)``. Rules:

    * ``"lead"`` (and common aliases like ``"orchestrator"``/``"parent"``)
      resolve to the agent that spawned this one. For the root lead they stay
      ``"lead"`` (its own inbox).
    * A name that matches a known agent in this session is used verbatim
      (a lead addressing a child, or a sibling).
    * Any other / unknown name is routed to the lead anyway, with a warning,
      so a typo'd recipient can never be silently written to an inbox no one
      reads.
    """
    raw = to.strip()
    lead_target = (_AGENT_PARENT_NAME or "lead") if IDENTITY != "lead" else "lead"

    if raw.lower() in _LEAD_ALIASES:
        return lead_target, None

    known = {a.get("name") for a in _load_agents(session_id) if a.get("name")}
    if raw in known:
        return raw, None

    warning = (
        f"unknown recipient {to!r}; routed to {lead_target!r}. "
        'Use to="lead" to reach whoever spawned you.'
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
    }
    if not full:
        return compact
    compact.update({"last_message": None, "backend_session_id": None})
    return compact


def _safe_float(value: object) -> float:
    """Coerce persisted numeric metadata to a float."""
    try:
        return float(value or 0.0)
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
        return read_claude_output(
            spawned_at, cwd, backend_session_id=backend_session_id
        )
    return None


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
    inbox = _inbox_file(session_id, reader)
    if not inbox.exists():
        return 0
    count = 0
    for raw in inbox.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            msg = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("from") == sender:
            count += 1
    return count


def _sender_unread_count(session_id: str, reader: str, sender: str) -> int:
    """Return ``sender``'s unread (not-yet-consumed) message count for ``reader``."""
    total = _sender_message_count(session_id, reader, sender)
    cursors = _load_inbox_cursors(_inbox_cursor_file(session_id, reader))
    consumed = min(cursors.get(sender, 0), total)
    return total - consumed


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
    last_line, truncated, _ = _truncate(
        _last_non_empty_line(internal.get("last_message")), max_chars
    )
    seq = _sender_message_count(session_id, IDENTITY, name)
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
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
    }


def _follow_up_failure(reason: str, name: str, status: dict | None = None) -> dict:
    """Build a structured ``follow_up_agent`` failure payload."""
    payload = {
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


def _hook_extra(session_id: str, agent_name: str, backend_name: str) -> dict[str, str]:
    """Materialise per-backend hook wiring, added to ``SpawnRequest.extra``.

    Claude Code gets a written settings-file path
    (``extra["hooks_settings_path"]``); Codex gets a JSON-encoded ``-c``
    override argv (``extra["hook_overrides"]``) evaluated only when
    ``WIN_AGENT_TEAMS_STATE_HOOKS_CODEX`` is on (see ``CodexBackend``).
    """
    session_dir = _session_dir(session_id)
    if backend_name == "claude-code":
        settings_path = hooks.write_claude_settings(session_dir, agent_name)
        return {"hooks_settings_path": str(settings_path)}
    if backend_name == "codex":
        overrides = hooks.codex_hook_overrides(session_dir, agent_name)
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
) -> dict:
    """Spawn a new agent process.

    reasoning_effort: low/medium/high/xhigh for codex,
    low/medium/high/xhigh/max for claude-code.
    """

    def _do_spawn() -> dict:
        session_id = _active_session_id(create=True)
        with _agents_transaction(session_id) as agents:
            agent_name = _unique_agent_name(name, agents)

            backend_name = backend.strip() or registry.default_backend()
            b = registry.get(backend_name)

            resolved_model = (
                b.resolve_model(model) if model.strip() else b.default_model()
            )

            mcp_config_path = _write_mcp_config(session_id, agent_name, IDENTITY)

            agent_cwd = cwd.strip() or str(Path.cwd())

            effort = reasoning_effort.strip() or None
            extra = {
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
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
                permission_mode=permission_mode,  # type: ignore[arg-type]
                reasoning_effort=effort,
                extra=extra,
            )

            result = b.spawn(request)
            pid = int(result.process_handle)

            agents.append(
                {
                    "name": agent_name,
                    "pid": pid,
                    "backend": backend_name,
                    "session_id": session_id,
                    "status": "running",
                    "spawned_at": time.time(),
                    "cwd": agent_cwd,
                    "model": resolved_model,
                    "permission_mode": permission_mode,
                    "reasoning_effort": effort,
                }
            )
            _save_agents_transaction(session_id, agents)

        return {
            "name": agent_name,
            "pid": pid,
            "backend": backend_name,
            "session_id": session_id,
        }

    return await run_blocking(_do_spawn)


@mcp.tool()
async def send_message(text: str, to: str = "lead") -> dict:
    """Write a message to an inbox for agents that actively poll read_messages.

    ``to`` defaults to ``"lead"``, which reaches the agent that spawned you —
    that is almost always what you want from a subagent. A lead can target a
    child by its agent name. Any unknown recipient is routed to the lead with a
    ``warning`` in the result rather than silently written to a dead inbox.

    This is not a push/resume mechanism: a spawned agent will only see this
    message if it calls read_messages after the message is sent. If the agent
    is not polling, use follow_up_agent instead.
    """
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

    return await run_blocking(_do_send)


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
    ``max_chars`` truncates each message's ``text`` (``truncated``/
    ``full_len`` added per message) when set.
    """
    if since_seq is not None and not from_agent:
        msg = "since_seq requires from_agent to be set"
        raise ValueError(msg)

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
            by_sender: dict[str, list[tuple[int, dict]]] = {}
            if inbox.exists():
                # Group valid messages by sender in file order, tracking each
                # message's global position so the result preserves file order.
                for index, raw in enumerate(
                    inbox.read_text(encoding="utf-8").splitlines()
                ):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        msg = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    sender = msg.get("from")
                    if not isinstance(sender, str) or not sender:
                        continue
                    by_sender.setdefault(sender, []).append((index, msg))

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

            effective_limit = None if full else (limit or _DEFAULT_READ_LIMIT)
            selected: list[tuple[str, int, int, dict]] = [
                (sender, position, index, msg)
                for sender in relevant
                for position, index, msg in per_sender_batches[sender]
            ]
            selected.sort(key=lambda item: item[2])  # global file order
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
                "unread_count": len(messages),
                "has_more": has_more,
            }
            if from_agent:
                result["cursors"] = None
                result["seq"] = updated.get(from_agent, cursors.get(from_agent, 0))
            else:
                result["cursors"] = updated
                result["seq"] = None
            return result

    return await run_blocking(_do_read)


@mcp.tool()
async def check_agent(
    name: str, full: bool = False, max_chars: int = _DEFAULT_LAST_LINE_MAX_CHARS
) -> dict:
    """Check an agent's status: state, last line, and unread message count.

    Default (``full=False``) returns a compact status peek: ``{name, state,
    alive, pid, backend, last_activity_at, unread_count, last_line, seq,
    truncated}``. ``state`` is ``running``/``waiting``/``idle``/``dead``.
    ``last_line`` is the last non-empty line of the agent's most recent
    assistant message, clipped to ``max_chars`` (default 200); ``truncated``
    signals clipping happened. ``unread_count``/``seq`` count messages FROM
    this agent addressed to the caller.

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
            pid = agent["pid"]
            alive, _ = process_manager.health_check(str(pid))
            output = _read_agent_output(agent)
            if _sync_backend_session_id(agent, output):
                _save_agents_transaction(session_id, agents)
            internal = _agent_check_payload(name, agent, alive, output)
            view = _compact_check_view(
                session_id, name, internal, max_chars=max_chars
            )
            if full:
                view.update(
                    {
                        "last_message": internal.get("last_message"),
                        "backend_session_id": internal.get("backend_session_id"),
                    }
                )
            return view

    return await run_blocking(_do_check)


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

    def _do_follow_up() -> dict:  # noqa: PLR0911 - mirrors explicit refusal reasons.
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
            alive, _ = process_manager.health_check(str(pid))
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

                if not process_manager.graceful_shutdown(str(pid), timeout_s=5.0):
                    process_manager.kill_process(str(pid))

            agent_name = str(agent.get("name") or name)
            agent_cwd = str(agent.get("cwd") or Path.cwd())
            mcp_config_path = _write_mcp_config(session_id, agent_name, IDENTITY)

            model = str(agent.get("model") or backend.default_model())
            permission_mode = str(agent.get("permission_mode") or "bypass")
            effort_value = agent.get("reasoning_effort")
            effort = effort_value if isinstance(effort_value, str) else None
            extra = {
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
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
                permission_mode=permission_mode,  # type: ignore[arg-type]
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

    return await run_blocking(_do_follow_up)


@mcp.tool()
async def kill_agent(name: str) -> dict:
    """Force-kill an agent process."""
    session_id = _active_session_id()

    def _do_kill() -> dict:
        if not session_id:
            return {"success": False, "name": name, "reason": "session_not_found"}
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return {"success": False, "name": name}
            process_manager.kill_process(str(agent["pid"]))
            for a in agents:
                if a["name"] == name:
                    a["status"] = "killed"
            _save_agents_transaction(session_id, agents)
            _state_marker_file(session_id, name).unlink(missing_ok=True)
            return {"success": True, "name": name}

    return await run_blocking(_do_kill)


def _list_agents_row(session_id: str, agent: dict, alive: bool) -> dict:
    """Build a compact ``list_agents`` row (no leaked internal fields)."""
    name = str(agent.get("name") or "")
    marker = _read_state_marker(session_id, name)
    output = _read_agent_output(agent) if marker is None else None
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
async def list_agents(full: bool = False) -> list[dict]:
    """List all agents with compact status rows.

    Default (``full=False``) rows are ``{name, state, alive, pid, backend,
    last_activity_at, unread_count}`` — no transcript bodies. Pass
    ``full=True`` to restore each agent's raw registry record plus
    ``last_line`` (the last non-empty line of its most recent message).
    """
    session_id = _active_session_id()

    def _do_list() -> list[dict]:
        if not session_id:
            return []
        agents = _load_agents(session_id)
        result = []
        for agent in agents:
            alive, _ = process_manager.health_check(str(agent["pid"]))
            if not full:
                result.append(_list_agents_row(session_id, agent, alive))
                continue
            output = _read_agent_output(agent)
            last_line, _, _ = _truncate(
                _last_non_empty_line(output.last_message if output else None),
                _DEFAULT_LAST_LINE_MAX_CHARS,
            )
            result.append({**agent, "alive": alive, "last_line": last_line})
        return result

    return await run_blocking(_do_list)


def _agent_status_row(session_id: str, agent: dict) -> dict:
    """Build one ``agent_status`` row (marker + cursor reads only, no scan)."""
    name = str(agent.get("name") or "")
    alive, _ = process_manager.health_check(str(agent.get("pid")))
    marker = _read_state_marker(session_id, name)
    last_activity_ts: float | None = None
    if marker is not None:
        ts = marker.get("ts")
        if isinstance(ts, int | float):
            last_activity_ts = float(ts)
    if last_activity_ts is None:
        output = _read_agent_output(agent)
        last_activity_ts = output.last_activity_at if output else None
    state = _resolve_agent_state(
        alive=alive, marker=marker, last_activity_at=last_activity_ts
    )
    seq = _sender_message_count(session_id, IDENTITY, name)
    unread_count = _sender_unread_count(session_id, IDENTITY, name)
    return {
        "name": name,
        "state": state,
        "last_activity_ts": last_activity_ts,
        "unread_count": unread_count,
        "seq": seq,
    }


@mcp.tool()
async def agent_status(names: list[str] | None = None) -> list[dict]:
    """Return cheap per-agent status rows: no bodies, no transcript scan.

    Each row is exactly ``{name, state, last_activity_ts, unread_count,
    seq}``. ``seq``/``unread_count`` are the caller's per-sender count for
    messages FROM that named agent. ``names=None`` returns all agents in the
    session; otherwise only the named agents (unknown names are skipped).

    Cost model: one state-marker read + one cursor read + one liveness check
    per agent. The marker (written by a Stop/SessionStart/etc. hook) is used
    directly when present; a transcript scan only happens as a fallback when
    no marker exists yet (e.g. hooks disabled or not yet fired).
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


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
