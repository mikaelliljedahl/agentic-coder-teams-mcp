"""Simplified MCP server for agent orchestration — 7 tools, fire-and-forget."""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

from claude_teams.async_utils import run_blocking
from claude_teams.backends.contracts import SpawnRequest
from claude_teams.backends.process_manager import process_manager
from claude_teams.backends.registry import registry

# Identity: env vars (works for Claude Code via --mcp-config)
# For Codex: spawn_agent updates ~/.codex/config.toml MCP env before spawning
_AGENT_NAME: str = os.environ.get("AGENT_NAME", "").strip()
_AGENT_SESSION_ID: str = os.environ.get("AGENT_SESSION_ID", "").strip()
IDENTITY: str = _AGENT_NAME if _AGENT_NAME else "lead"

_SESSION_BASE = Path.home() / ".claude" / "agent-sessions"

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
    _agents_file(session_id).write_text(
        json.dumps(agents, indent=2), encoding="utf-8"
    )


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


def _update_codex_mcp_env(session_id: str, agent_name: str) -> None:
    """Update ~/.codex/config.toml MCP server env with agent identity.

    Codex doesn't propagate process env to MCP servers — it uses
    env from config.toml. We update it before each spawn so the
    MCP server starts with correct AGENT_NAME and AGENT_SESSION_ID.
    """
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return
    try:
        import tomllib
        content = config_path.read_text(encoding="utf-8")
        # Simple string replacement for the env block
        # Find and replace the win-agent-teams env line
        lines = content.splitlines()
        new_lines = []
        in_win_agent = False
        for line in lines:
            if line.strip().startswith("[mcp_servers.win-agent-teams]"):
                in_win_agent = True
            elif line.strip().startswith("[") and in_win_agent:
                in_win_agent = False
            if in_win_agent and line.strip().startswith("env"):
                line = f'env = {{ "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass", "AGENT_NAME" = "{agent_name}", "AGENT_SESSION_ID" = "{session_id}" }}'
            new_lines.append(line)
        config_path.write_text("\n".join(new_lines), encoding="utf-8")
    except Exception:
        pass


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
    """Spawn a new agent process. reasoning_effort: low/medium/high for codex, low/medium/high/max for claude-code."""
    global _session_id

    def _do_spawn() -> dict:
        global _session_id

        if not _session_id:
            _session_id = _create_session()

        session_id = _session_id
        agents = _load_agents(session_id)

        agent_name = name.strip() or f"agent-{len(agents) + 1}"

        backend_name = backend.strip() or registry.default_backend()
        b = registry.get(backend_name)

        resolved_model = b.resolve_model(model) if model.strip() else b.default_model()

        mcp_config_path = _write_mcp_config(session_id, agent_name)

        if backend_name == "codex":
            _update_codex_mcp_env(session_id, agent_name)

        agent_cwd = cwd.strip() or os.getcwd()

        effort = reasoning_effort.strip() or None

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
            extra={
                "mcp_config_path": str(mcp_config_path),
                "agent_capability": "",
            },
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
                "ts": datetime.now(timezone.utc).isoformat(),
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
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
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
            return {"name": name, "alive": False, "pid": None, "backend": None}
        pid = agent["pid"]
        alive, _ = process_manager.health_check(str(pid))
        return {
            "name": name,
            "alive": alive,
            "pid": pid,
            "backend": agent.get("backend"),
        }

    return await run_blocking(_do_check)


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
