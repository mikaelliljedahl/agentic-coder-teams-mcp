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

_SESSION_BASE = Path.home() / ".claude" / "agent-sessions"
_SESSION_META_NAME = "session.json"
_BINDINGS_DIR_NAME = "bindings"
_AGENTS_LOCK_NAME = "agents.lock"
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.05
_LOCK_SIZE = 1
_FOLLOW_UP_IDLE_SECONDS = 60.0
logger = logging.getLogger(__name__)


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


def _message_recipient(to: str) -> str:
    """Resolve recipient aliases in nested agent trees."""
    recipient = to.strip()
    if recipient == "lead" and IDENTITY != "lead":
        return _AGENT_PARENT_NAME or "lead"
    return recipient


def _empty_agent_check(name: str) -> dict:
    """Return a stable empty ``check_agent`` payload."""
    return {
        "name": name,
        "alive": False,
        "pid": None,
        "backend": None,
        "backend_session_id": None,
        "last_activity_at": None,
        "last_message": None,
    }


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
    """Build the public check payload for an existing agent record."""
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
async def send_message(to: str, text: str) -> dict:
    """Write a message to an inbox for agents that actively poll read_messages.

    This is not a push/resume mechanism: a spawned agent will only see this
    message if it calls read_messages after the message is sent. If the agent
    is not polling, use follow_up_agent instead.
    """
    session_id = _active_session_id()

    def _do_send() -> dict:
        if not session_id:
            return {"success": False, "to": to, "reason": "session_not_found"}
        recipient = _message_recipient(to)
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

    return await run_blocking(_do_send)


@mcp.tool()
async def read_messages(from_agent: str = "") -> list[dict]:
    """Read unread messages from own inbox, optionally filtered by sender."""
    session_id = _active_session_id()

    def _do_read() -> list[dict]:
        if not session_id:
            return []
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
            unread: list[tuple[int, dict]] = []
            updated = dict(cursors)
            for sender in relevant:
                entries = by_sender.get(sender, [])
                start = cursors.get(sender, 0)
                unread.extend(entries[start:])
                # Avoid creating a stray zero cursor for an unknown filtered
                # sender; only persist a count for senders that have messages
                # or already had a stored cursor.
                if entries or sender in cursors:
                    updated[sender] = len(entries)

            _save_inbox_cursors(cursor_file, updated)
            unread.sort(key=lambda item: item[0])
            return [msg for _, msg in unread]

    return await run_blocking(_do_read)


@mcp.tool()
async def check_agent(name: str) -> dict:
    """Check whether an agent process is alive."""
    session_id = _active_session_id()

    def _do_check() -> dict:
        if not session_id:
            return _empty_agent_check(name)
        with _agents_transaction(session_id) as agents:
            agent = next((a for a in agents if a["name"] == name), None)
            if agent is None:
                return _empty_agent_check(name)
            pid = agent["pid"]
            alive, _ = process_manager.health_check(str(pid))
            output = _read_agent_output(agent)
            if _sync_backend_session_id(agent, output):
                _save_agents_transaction(session_id, agents)
            return _agent_check_payload(name, agent, alive, output)

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
            return {"success": True, "name": name}

    return await run_blocking(_do_kill)


@mcp.tool()
async def list_agents() -> list[dict]:
    """List all agents and their alive status."""
    session_id = _active_session_id()

    def _do_list() -> list[dict]:
        if not session_id:
            return []
        agents = _load_agents(session_id)
        result = []
        for agent in agents:
            alive, _ = process_manager.health_check(str(agent["pid"]))
            result.append({**agent, "alive": alive})
        return result

    return await run_blocking(_do_list)


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
