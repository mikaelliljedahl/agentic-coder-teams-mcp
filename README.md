# agentic-coder-teams-mcp

Minimal MCP server for spawning and communicating with Claude Code and Codex agents on Windows or Linux. Fire-and-forget agent spawning with bidirectional 1:1 messaging.

## Tools (8 total)

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start an agent process (fire-and-forget) |
| `send_message` | Send a message to an agent or lead |
| `read_messages` | Read unread messages from own inbox |
| `check_agent` | Check if an agent process is alive and read fallback output |
| `follow_up_agent` | Resume an existing logical agent with a follow-up prompt |
| `kill_agent` | Force-kill an agent process |
| `list_agents` | List all agents and their status |
| `list_backends` | List available backends |

## Quick Start

### Prerequisites

- Windows 10/11 or Linux
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Claude Code CLI (`claude`) and/or OpenAI Codex CLI (`codex`) on `PATH`
- `tmux` on Linux

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

On Linux, use your virtualenv's Python path instead, for example:

```json
{
  "mcpServers": {
    "win-agent-teams": {
      "command": "/path/to/.venv/bin/python",
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

Linux example:

```toml
[mcp_servers.win-agent-teams]
command = "/path/to/.venv/bin/python"
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

`spawn_agent` starts a CLI process and returns immediately with `{name, pid, backend, session_id}`. The agent runs independently.

Display mode is selected automatically:
- Windows uses native processes. Interactive agents can open their own console window, and captured output can be tailed in Windows Terminal when `wt.exe` is available.
- Linux/POSIX defaults to spawning each agent in its own terminal emulator window. Set `WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux` to use `tmux` instead. In tmux mode, if the MCP server is already inside tmux, agents are spawned as split panes by default (set `USE_TMUX_WINDOWS=1` to use tmux windows); if the server is not inside tmux, agents are spawned into a detached session named `win-agent-teams-<session>`.

If your MCP client does not pass its `TMUX` environment variable through to the
MCP server, set `WIN_AGENT_TEAMS_TMUX_TARGET` to an existing session or pane,
for example `codex-lead`. Agents will then spawn into that target instead of a
detached session.

On Linux, the terminal-window launcher is the default
(`WIN_AGENT_TEAMS_LINUX_LAUNCHER=terminal`), which spawns each agent in a
visible terminal emulator window. This is the recommended mode when running from
**Claude Desktop**, which is not attached to a tmux session — tmux mode would
spawn agents into a detached session you would never see. The launcher probes
common terminals such as `qterminal` (the LXQt/Lubuntu default),
`gnome-terminal`, `x-terminal-emulator`, `xfce4-terminal`, and `xterm`; set
`WIN_AGENT_TEAMS_LINUX_TERMINAL` to force a specific terminal command. Set
`WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux` only if you are running the server inside a
tmux session and prefer panes/windows.

> Note: `qterminal` may hand launches to an already-running instance via D-Bus.
> The auto-discovery path skips it when one is already running and falls through
> to the next available terminal. Set `WIN_AGENT_TEAMS_LINUX_TERMINAL` if you
> want to force a specific emulator.

Spawned agents are intentionally detached from the MCP server lifetime by
default. This keeps a long-running Claude Code or Codex worker alive if Codex
restarts, idles out, or closes its MCP server process. To restore strict child
cleanup when the MCP process exits, set `WIN_AGENT_TEAMS_KILL_ON_EXIT=1` before
starting the server. This cleanup option is Windows-specific.

Interactive agents launch in their own console window by default so their live
UI remains visible while they work. On Windows, set
`WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` to capture stdout/stderr to the
per-agent log file instead.

Lead MCP sessions are also persisted per parent process, workspace, and
identity. If Codex or Claude restarts only the MCP server process, tools such as
`list_agents`, `check_agent`, and `follow_up_agent` recover the prior
`agents.json` and can resume stored backend sessions. If the parent process
itself may change but should keep the same orchestration session, set a stable
`WIN_AGENT_TEAMS_PARENT_ID` value before starting the MCP server.

Nested orchestration shares the same session intentionally. For example, a
Claude Desktop lead can spawn a Claude Code agent, and that agent can spawn a
Codex worker under the same `agents.json`. Registry updates are guarded by a
cross-process lock, and duplicate requested names are made unique within the
session, for example `worker` then `worker-2`.

### Messaging

Bidirectional 1:1 messaging between lead and agents via JSONL files:

```
~/.claude/agent-sessions/{session-id}/
    agents.json              # agent registry
    inbox-lead.jsonl         # messages TO lead
    inbox-{agent}.jsonl      # messages TO agent
    inbox-{agent}.pos.json   # per-sender unread cursor for that inbox
```

Each line: `{"from": "agent-1", "text": "done", "ts": "2026-05-11T..."}`

`read_messages` returns only **unread** messages. A per-sender counter
sidecar (`inbox-{name}.pos.json`) tracks how much of each sender's stream
the reader has already consumed, so reads stay O(n) instead of re-returning
the whole inbox every call. `read_messages(from_agent="x")` advances only
`x`'s cursor. Delivery is best-effort: a crashed reader or a lost/corrupt
cursor file may cause a message to be re-delivered or, in the rare case of
two lead processes sharing one inbox, consumed by the other process.

### Output Fallback

`check_agent(name)` returns `{name, alive, pid, backend, backend_session_id, last_activity_at, last_message}`. For Codex and Claude Code workers, these fields are read from the CLIs' existing JSONL session logs. This is a fallback for workers that finish without calling `send_message`; it does not replace explicit agent-to-lead messaging.

`last_message` is the **tail** of the worker's most recent assistant message,
truncated to a 1000-character budget so repeated polling stays cheap. When
truncated it is prefixed with a marker, e.g. `[truncated: showing last 950 of
8200 chars]`. It is a status peek, not the full output — read the agent's own
session log if you need the complete text.

### Follow-up / Resume

`follow_up_agent(name, prompt, replace_if_idle=true)` continues the same logical agent by starting a new backend process with the CLI's native resume mechanism. Codex uses `codex resume` with the same permission/cwd/reasoning settings as spawn; Claude Code uses `claude --resume`. If the old process is still alive but idle, the tool replaces it by default; pass `replace_if_idle=false` to instead refuse with `agent_idle_but_alive`. A live, busy process is always refused with `agent_busy`.

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
2. Codex opens in a console window and starts working
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

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync --group dev
```

## License

[MIT](./LICENSE)
