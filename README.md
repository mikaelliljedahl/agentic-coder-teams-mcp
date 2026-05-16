# win-agent-teams-mcp

Minimal MCP server for spawning and communicating with Claude Code and Codex agents on Windows. Fire-and-forget agent spawning with bidirectional 1:1 messaging.

## Tools (8 total)

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start an agent process (fire-and-forget) |
| `send_message` | Send a message to an agent or lead |
| `read_messages` | Read messages from own inbox |
| `check_agent` | Check if an agent process is alive and read fallback output |
| `follow_up_agent` | Resume an existing logical agent with a follow-up prompt |
| `kill_agent` | Force-kill an agent process |
| `list_agents` | List all agents and their status |
| `list_backends` | List available backends |

## Quick Start

### Prerequisites

- Windows 10 or Windows 11
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Claude Code CLI (`claude`) and/or OpenAI Codex CLI (`codex`) on `PATH`

### Setup — Claude Code as Lead

Add to your project's `.mcp.json` so Claude Code can spawn agents:

```json
{
  "mcpServers": {
    "win-agent-teams": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["-m", "claude_teams.server_simple"]
    }
  }
}
```

Spawned Claude Code agents get the MCP server automatically via `--mcp-config`.
Spawned Codex agents need the Codex setup below only when they must call MCP
tools themselves, for example to `send_message` back to lead. Passive Codex
workers can still be observed through `check_agent` output fallback.

### Setup — Codex as Lead (or as Spawned Agent)

Add to `~/.codex/config.toml` so Codex can use the MCP tools (both as lead and as spawned agent):

```toml
[mcp_servers.win-agent-teams]
command = "C:\\path\\to\\.venv\\Scripts\\python.exe"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass" }
enabled = true
```

This is required in two scenarios:
1. **Codex as lead** — Codex calls `spawn_agent` to start Claude Code or other Codex agents.
2. **Codex as spawned agent using MCP tools** — when Claude Code spawns a Codex agent and expects that agent to call tools such as `send_message`.

The server auto-injects `AGENT_NAME` and `AGENT_SESSION_ID` into the Codex config env before each spawn so the MCP server knows agent identity.

## How It Works

### Spawning

`spawn_agent` starts a CLI process in its own console window and returns immediately with `{name, pid, backend, session_id}`. The agent runs independently — output is visible in the console window.

### Messaging

Bidirectional 1:1 messaging between lead and agents via JSONL files:

```
~/.claude/agent-sessions/{session-id}/
    agents.json              # agent registry
    inbox-lead.jsonl         # messages TO lead
    inbox-{agent}.jsonl      # messages TO agent
```

Each line: `{"from": "agent-1", "text": "done", "ts": "2026-05-11T..."}`

### Output Fallback

`check_agent(name)` returns `{name, alive, pid, backend, backend_session_id, last_activity_at, last_message}`. For Codex and Claude Code workers, these fields are read from the CLIs' existing JSONL session logs. This is a fallback for workers that finish without calling `send_message`; it does not replace explicit agent-to-lead messaging.

### Follow-up / Resume

`follow_up_agent(name, prompt, replace_if_idle=false)` continues the same logical agent by starting a new backend process with the CLI's native resume mechanism. Codex uses `codex resume` with the same permission/cwd/reasoning settings as spawn; Claude Code uses `claude --resume`. If the old process is still alive, the tool only replaces it when it looks idle and `replace_if_idle=true`.

The tool relies on `backend_session_id`, which `check_agent` exposes from the backend's JSONL session logs. Once known, that session id is used as the correlation key so resume follow-ups keep reading the correct rollout even when multiple agents share a working directory.

Recommended follow-up pattern:

```
1. Lead calls spawn_agent(..., name="worker")
2. Lead polls check_agent("worker") until last_message and backend_session_id are present
3. Lead calls follow_up_agent("worker", prompt="next task", replace_if_idle=true)
4. Lead polls check_agent("worker") for the follow-up last_message
```

### Identity

The server detects its role from environment variables:
- **Lead mode**: No `AGENT_NAME` set → identity = `"lead"`
- **Agent mode**: `AGENT_NAME` + `AGENT_SESSION_ID` set → identity = agent name

### Example Flow

```
1. Lead calls spawn_agent(prompt="Review auth.py and send_message results to lead", backend="codex", name="reviewer")
2. Codex opens in a new console window, starts working
3. Codex calls send_message(to="lead", text="Found 3 issues in auth.py")
4. Lead calls read_messages() → sees the message
5. Lead calls kill_agent(name="reviewer") when done
```

### Messaging Model

Agents are still best treated as **single-prompt workers**. The prompt at spawn is their task. They execute it, optionally send a status message back to lead, and then go idle.

- **Agent → Lead**: Works well. Include "send results to lead via send_message" in the prompt. The agent sends a message when done.
- **Lead → Agent via inbox**: The message is written to the agent's inbox, but the agent only sees it if it actively polls `read_messages`. Most agents don't poll after completing their initial task.
- **Lead → Agent via resume**: Use `follow_up_agent`. It replaces/resumes the logical agent through the backend CLI's native resume command instead of relying on inbox polling.
- **Multi-turn conversation**: Keep it deliberate. Use explicit `follow_up_agent` calls and verify each turn through `check_agent`.

**Recommended pattern**: Spawn an agent per task. Put everything it needs in the prompt. Have it report back via `send_message` or rely on `check_agent` fallback output. For a second turn, use `follow_up_agent` and verify that `backend_session_id` remains stable.

### Smoke-Tested Resume Chain

The native resume path has been smoke-tested with this chain:

```
lead -> Claude Code orchestrator -> Codex CLI target
```

The Claude orchestrator spawned a passive Codex target, observed its base answer with `check_agent`, called `follow_up_agent(..., replace_if_idle=true)`, and observed the follow-up answer. The Codex `backend_session_id` stayed unchanged across the base and follow-up turns, confirming that the follow-up resumed the same backend session.

## Spawn Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Task prompt for the agent |
| `name` | auto (`agent-1`) | Agent name |
| `backend` | `claude-code` | `claude-code` or `codex` |
| `model` | backend default | Model to use |
| `reasoning_effort` | none | `low`/`medium`/`high`/`xhigh` (codex), `low`/`medium`/`high`/`xhigh`/`max` (claude-code) |
| `permission_mode` | `bypass` | `bypass`, `default`, or `require_approval` |
| `cwd` | server cwd | Working directory for the agent |

## CLI

```powershell
win-agent-teams serve      # Start the MCP server
win-agent-teams backends   # List available backends
```

## Development

```powershell
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync --group dev
```

## License

[MIT](./LICENSE)
