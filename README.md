# win-agent-teams-mcp

Minimal MCP server for spawning and communicating with Claude Code and Codex agents on Windows. Fire-and-forget agent spawning with bidirectional 1:1 messaging.

## Tools (7 total)

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start an agent process (fire-and-forget) |
| `send_message` | Send a message to an agent or lead |
| `read_messages` | Read messages from own inbox |
| `check_agent` | Check if an agent process is alive |
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
Spawned Codex agents need the Codex setup below to message back.

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
1. **Codex as lead** — Codex calls `spawn_agent` to start Claude Code or other Codex agents
2. **Codex as spawned agent** — when Claude Code spawns a Codex agent, the agent needs this config to call `send_message` back to lead

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

### Messaging Model — Fire-and-Forget

Agents are **single-prompt workers**, not chatbots. The prompt at spawn is their task. They execute it, optionally send a status message back to lead, and then go idle.

- **Agent → Lead**: Works well. Include "send results to lead via send_message" in the prompt. The agent sends a message when done.
- **Lead → Agent (follow-up)**: The message is written to the agent's inbox, but the agent only sees it if it actively polls `read_messages`. Most agents don't poll after completing their initial task.
- **Multi-turn conversation**: Not practical. Agents don't run inbox-polling loops.

**Recommended pattern**: Spawn an agent per task. Put everything it needs in the prompt. Have it report back via `send_message` when done. For file output, tell the agent which file to write to in the prompt — that's more reliable than messaging for large results.

## Spawn Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Task prompt for the agent |
| `name` | auto (`agent-1`) | Agent name |
| `backend` | `claude-code` | `claude-code` or `codex` |
| `model` | backend default | Model to use |
| `reasoning_effort` | none | `low`/`medium`/`high` (codex), `low`/`medium`/`high`/`max` (claude-code) |
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
