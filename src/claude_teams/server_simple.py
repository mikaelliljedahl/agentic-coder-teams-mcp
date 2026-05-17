"""Simplified MCP server for agent orchestration."""

import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

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
IDENTITY: str = _AGENT_NAME if _AGENT_NAME else "lead"

_SESSION_BASE = Path.home() / ".claude" / "agent-sessions"
_FOLLOW_UP_IDLE_SECONDS = 60.0
logger = logging.getLogger(__name__)

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


def _inbox_file(session_id: str, name: str) -> Path:
    return _session_dir(session_id) / f"inbox-{name}.jsonl"


def _load_agents(session_id: str) -> list[dict]:
    path = _agents_file(session_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_agents(session_id: str, agents: list[dict]) -> None:
    _agents_file(session_id).write_text(json.dumps(agents, indent=2), encoding="utf-8")


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
    _save_agents(sid, [])
    return sid


def _write_mcp_config(session_id: str, agent_name: str) -> Path:
    """Write per-agent MCP config (used by Claude Code via --mcp-config)."""
    config = {
        "mcpServers": {
            "win-agent-teams": {
                "command": sys.executable,
                "args": ["-m", "claude_teams.server_simple"],
                "env": {
                    "AGENT_SESSION_ID": session_id,
                    "AGENT_NAME": agent_name,
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
        global _session_id  # noqa: PLW0603 - module-level lead session state.

        if not _session_id:
            _session_id = _create_session()

        session_id = _session_id
        agents = _load_agents(session_id)

        agent_name = name.strip() or f"agent-{len(agents) + 1}"

        backend_name = backend.strip() or registry.default_backend()
        b = registry.get(backend_name)

        resolved_model = b.resolve_model(model) if model.strip() else b.default_model()

        mcp_config_path = _write_mcp_config(session_id, agent_name)

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
            lead_session_id="lead",
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
        _save_agents(session_id, agents)

        return {
            "name": agent_name,
            "pid": pid,
            "backend": backend_name,
            "session_id": session_id,
        }

    return await run_blocking(_do_spawn)


@mcp.tool()
async def send_message(to: str, text: str) -> dict:
    """Send a message to an agent or lead."""
    session_id = _session_id or _AGENT_SESSION_ID

    def _do_send() -> dict:
        inbox = _inbox_file(session_id, to)
        line = json.dumps(
            {
                "from": IDENTITY,
                "text": text,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        with inbox.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return {"success": True, "to": to}

    return await run_blocking(_do_send)


@mcp.tool()
async def read_messages(from_agent: str = "") -> list[dict]:
    """Read messages from own inbox, optionally filtered by sender."""
    session_id = _session_id or _AGENT_SESSION_ID

    def _do_read() -> list[dict]:
        inbox = _inbox_file(session_id, IDENTITY)
        if not inbox.exists():
            return []
        messages = []
        for raw in inbox.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if from_agent and msg.get("from") != from_agent:
                continue
            messages.append(msg)
        return messages

    return await run_blocking(_do_read)


@mcp.tool()
async def check_agent(name: str) -> dict:
    """Check whether an agent process is alive."""
    session_id = _session_id or _AGENT_SESSION_ID

    def _do_check() -> dict:
        agents = _load_agents(session_id)
        agent = next((a for a in agents if a["name"] == name), None)
        if agent is None:
            return _empty_agent_check(name)
        pid = agent["pid"]
        alive, _ = process_manager.health_check(str(pid))
        output = _read_agent_output(agent)
        if _sync_backend_session_id(agent, output):
            _save_agents(session_id, agents)
        return _agent_check_payload(name, agent, alive, output)

    return await run_blocking(_do_check)


@mcp.tool()
async def follow_up_agent(
    name: str,
    prompt: str,
    replace_if_idle: bool = False,
) -> dict:
    """Resume a logical agent with a follow-up prompt."""
    session_id = _session_id or _AGENT_SESSION_ID

    def _do_follow_up() -> dict:  # noqa: PLR0911 - mirrors explicit refusal reasons.
        agents = _load_agents(session_id)
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
                _save_agents(session_id, agents)
            return _follow_up_failure("backend_session_missing", name, status)

        if alive:
            last_message = status.get("last_message")
            if last_message is None:
                if changed:
                    _save_agents(session_id, agents)
                return _follow_up_failure("agent_busy", name, status)
            last_activity_at = status.get("last_activity_at")
            if last_activity_at is None:
                if changed:
                    _save_agents(session_id, agents)
                return _follow_up_failure("agent_state_unknown", name, status)
            if output is not None and output.busy_hint:
                if changed:
                    _save_agents(session_id, agents)
                return _follow_up_failure("agent_busy", name, status)
            if time.time() - float(last_activity_at) < _FOLLOW_UP_IDLE_SECONDS:
                if changed:
                    _save_agents(session_id, agents)
                return _follow_up_failure("agent_busy", name, status)
            if not replace_if_idle:
                if changed:
                    _save_agents(session_id, agents)
                return _follow_up_failure("agent_idle_but_alive", name, status)

            if not process_manager.graceful_shutdown(str(pid), timeout_s=5.0):
                process_manager.kill_process(str(pid))

        agent_name = str(agent.get("name") or name)
        agent_cwd = str(agent.get("cwd") or Path.cwd())
        mcp_config_path = _write_mcp_config(session_id, agent_name)

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
            lead_session_id="lead",
            permission_mode=permission_mode,  # type: ignore[arg-type]
            reasoning_effort=effort,
            extra=extra,
        )

        try:
            result = backend.resume(request, str(backend_session_id))
        except Exception:
            logger.debug("Failed resuming backend session", exc_info=True)
            if changed:
                _save_agents(session_id, agents)
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
        _save_agents(session_id, agents)
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
    session_id = _session_id or _AGENT_SESSION_ID

    def _do_kill() -> dict:
        agents = _load_agents(session_id)
        agent = next((a for a in agents if a["name"] == name), None)
        if agent is None:
            return {"success": False, "name": name}
        process_manager.kill_process(str(agent["pid"]))
        for a in agents:
            if a["name"] == name:
                a["status"] = "killed"
        _save_agents(session_id, agents)
        return {"success": True, "name": name}

    return await run_blocking(_do_kill)


@mcp.tool()
async def list_agents() -> list[dict]:
    """List all agents and their alive status."""
    session_id = _session_id or _AGENT_SESSION_ID

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
