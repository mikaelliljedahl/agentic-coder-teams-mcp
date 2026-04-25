# win-agent-teams-mcp

Windows-native MCP server for orchestrating teams of agentic coding agents.

This fork is focused on running directly on Windows with Claude Code and
OpenAI Codex CLI as the primary workflow. It exposes a FastMCP server plus a
small operator CLI named `win-agent-teams`.

## Quick Start

### Prerequisites

- Windows 10 or Windows 11
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Claude Code CLI available as `claude` on `PATH`
- OpenAI Codex CLI available as `codex` on `PATH`

### Run with uvx

```powershell
uvx --from git+https://github.com/mikaelliljedahl/agentic-coder-teams-mcp win-agent-teams serve
```

### Verify available backends

```powershell
uvx --from git+https://github.com/mikaelliljedahl/agentic-coder-teams-mcp win-agent-teams backends
```

The server auto-discovers supported agent CLIs from your Windows `PATH`.

## Claude Code MCP Setup

Add the server to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "win-agent-teams": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mikaelliljedahl/agentic-coder-teams-mcp",
        "win-agent-teams",
        "serve"
      ]
    }
  }
}
```

Start Claude Code in the project directory after saving the MCP config. Claude
Code can then use the MCP tools to create a team, spawn teammates, assign tasks,
send messages, and check teammate health.

## Codex Flow On Windows

Use Claude Code as the team lead and spawn Codex teammates when you want an
OpenAI Codex CLI worker to implement, review, or investigate a task.

Typical flow:

1. Start Claude Code in a Windows workspace with the MCP server configured.
2. Use `team_create` to create a team.
3. Use `list_backends` to confirm `claude-code` and `codex` are available.
4. Use `spawn_teammate` with `backend_name` set to `codex` for Codex workers.
5. Use `task_create`, `send_message`, `read_inbox`, and `check_teammate` to
   coordinate the work.

Example `spawn_teammate` arguments:

```json
{
  "team_name": "windows-port",
  "name": "codex-worker",
  "prompt": "Implement the focused change and report back with tests run.",
  "options": {
    "backend_name": "codex",
    "model": "balanced",
    "cwd": "C:\\code\\github\\my-project"
  }
}
```

## Windows Smoke Test

Use this smoke test to verify the Windows-native Codex lead -> Claude Code
teammate path. It exercises MCP spawning, inbox messaging, PID handles, log
capture, health checks, and cleanup without tmux, WSL, Cygwin, or psmux.

Refresh the GitHub install first so Codex uses the latest fork revision:

```powershell
uvx --refresh --from git+https://github.com/mikaelliljedahl/agentic-coder-teams-mcp win-agent-teams --help
uvx --refresh --from git+https://github.com/mikaelliljedahl/agentic-coder-teams-mcp win-agent-teams backends --json
```

Start a fresh Codex CLI session with the `win-agent-teams` MCP server
configured, then run this prompt:

```text
Run a Windows-native smoke test for win-agent-teams MCP using the team name `codex-lead-smoke-visual`.

Use the `win-agent-teams` MCP tools only, not direct filesystem reads unless an MCP tool fails.

Steps:

1. Create team `codex-lead-smoke-visual`.
2. Spawn teammate:
   - name: `claude-worker`
   - backend: `claude-code`
   - model: `sonnet`
   - prompt: `Read your inbox, reply to the team lead with a one-sentence confirmation using send_message(sender="claude-worker", recipient="team-lead"), then wait for further instructions.`
3. Read config and verify:
   - `claude-worker` exists
   - `backendType` is `claude-code`
   - `processHandle` is a decimal PID string
   - `pid` is populated
4. Send a follow-up message from `team-lead` to `claude-worker`:
   `Please confirm you received the follow-up message.`
5. Poll or read `team-lead` inbox using MCP tools until a message from `claude-worker` appears.
6. Report the exact received message text.
7. Call `health_check` for `claude-worker`.
8. Call `get_agent_logs` for `claude-worker` with `tail=80`.
9. Stop `claude-worker` using `force_kill_teammate`.
10. Delete the team.
```

Expected result:

- A real Claude Code terminal window opens for `claude-worker`.
- The backend list includes `claude-code` and `codex`.
- `processHandle` and `pid` are decimal PID values.
- Claude replies to `team-lead` through MCP `send_message`.
- Codex can read the reply through MCP inbox tools.
- `health_check` returns process-based detail.
- `get_agent_logs` returns a path under
  `%USERPROFILE%\.claude\teams\codex-lead-smoke-visual\logs\claude-worker.log`.
- `force_kill_teammate` and `team_delete` complete successfully.
- No tmux, pane, WSL, Cygwin, or psmux errors appear.

If Codex cannot see Claude's inbox replies but direct file inspection shows
messages in `team-lead.json`, refresh the `uvx` install and restart Codex. Older
builds read inbox files using the Windows process codepage; current builds use
UTF-8 explicitly for inbox state.

## CLI Reference

```powershell
win-agent-teams serve              # Start the MCP server
win-agent-teams backends           # List available backends
win-agent-teams templates          # List registered agent templates
win-agent-teams presets            # List registered team presets
win-agent-teams config TEAM        # Show team config
win-agent-teams status TEAM        # Show member status and task summary
win-agent-teams inbox TEAM AGENT   # Read an agent inbox
win-agent-teams health TEAM AGENT  # Health-check a teammate process
win-agent-teams kill TEAM AGENT    # Force-kill a teammate
```

All commands support `--json` / `-j` for machine-readable output.

## Development

```powershell
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync --group dev
```

Run the same checks as CI:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

CI runs on `windows-latest` only.

## Project Notes

- Package name: `win-agent-teams-mcp`
- Console script: `win-agent-teams`
- FastMCP app name: `win-agent-teams`
- Supported platform for this fork: Windows
- Primary documented lead/worker flow: Claude Code plus Codex CLI

## License

[MIT](./LICENSE)
