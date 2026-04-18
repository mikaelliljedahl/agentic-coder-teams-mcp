<div align="center">

# agentic-coder-teams-mcp

Multi-backend MCP server for orchestrating teams of agentic coding agents.

**644 tests | 93% coverage | 17 backends | Python 3.12+**

</div>

https://github.com/user-attachments/assets/531ada0a-6c36-45cd-8144-a092bb9f9a19

---

## Table of Contents

- [What is this?](#what-is-this)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Supported Backends](#supported-backends)
- [MCP Tools](#mcp-tools)
- [MCP Prompts](#mcp-prompts)
- [Skills](#skills)
- [CLI Reference](#cli-reference)
- [How It Works](#how-it-works)
- [Development](#development)
  - [Observability](#observability)
- [Contributing](#contributing)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## What is this?

Claude Code has a built-in [agent teams](https://code.claude.com/docs/en/agent-teams) feature that lets multiple Claude Code instances coordinate as a team with shared task lists, inter-agent messaging, and tmux-based spawning. But the protocol is internal and tightly coupled to Claude Code's own tooling.

This MCP server reimplements that protocol as a standalone [Model Context Protocol](https://modelcontextprotocol.io/) server with one major addition: **pluggable backend support for 17 agentic coding CLIs**. Any MCP client can use it to spawn and coordinate heterogeneous teams of coding agents across different tools and providers.

### What you can do with it

- Spawn a team of agents using different backends (Claude Code, Codex, Gemini, Aider, etc.)
- Coordinate work through shared task lists with dependency tracking
- Send messages between agents (direct, broadcast, shutdown requests)
- Monitor agent health and force-kill unresponsive agents
- Drive common lead operations with templated MCP **prompts** (`status_check`, `health_sweep`, `task_handoff`, `wrap_up`, `unblock_teammate`)
- Surface agent **skills** to any connected client via a bundled team-orchestration skill and per-backend skill roots
- Export traces over OTLP when the optional `[otel]` extra is installed
- Use any MCP-compatible client as the orchestrator

---

## Quick Start

### 1. Install

```bash
# Using uvx (recommended, no install needed)
uvx --from git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp claude-teams serve

# Or clone and install locally
git clone https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync
```

### 2. Prerequisites

- **Python 3.12+**
- **[tmux](https://github.com/tmux/tmux)** - agents spawn in tmux panes
- At least one supported agentic CLI on your `PATH` (e.g., `claude`, `codex`, `gemini`)
- **Platform**: Linux or macOS. tmux has no native Windows port; on Windows, run the server inside WSL2, Cygwin, or MSYS2. MCP *clients* on Windows work fine against a server running elsewhere.

### 3. Verify backends

```bash
claude-teams backends
```

This shows which backends were auto-discovered on your system.

### 4. Connect your MCP client

Add the server to your client's MCP config (see [Installation](#installation) below for client-specific examples).

---

## Installation

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "claude-teams": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp", "claude-teams", "serve"]
    }
  }
}
```

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "claude-teams": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp", "claude-teams", "serve"],
      "enabled": true
    }
  }
}
```

### Any MCP client

The server speaks standard MCP over stdio. Point your client at:

```
uvx --from git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp claude-teams serve
```

---

## Supported Backends

The server auto-discovers which backends are available based on binaries found on your `PATH`:

| Backend | CLI binary | Description |
|---------|-----------|-------------|
| `claude-code` | `claude` | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (default) |
| `codex` | `codex` | [OpenAI Codex CLI](https://github.com/openai/codex) |
| `gemini` | `gemini` | [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
| `opencode` | `opencode` | [OpenCode](https://opencode.ai) (multi-provider) |
| `aider` | `aider` | [Aider](https://aider.chat) |
| `copilot` | `copilot` | [GitHub Copilot CLI](https://github.com/github/copilot-cli) |
| `auggie` | `auggie` | [Augment Code](https://www.augmentcode.com/) |
| `goose` | `goose` | [Goose](https://github.com/block/goose) |
| `qwen` | `qwen` | [Qwen Chat CLI](https://github.com/QwenLM) |
| `vibe` | `vibe` | [Vibe](https://github.com/thevibe-ai/vibe) |
| `kimi` | `kimi` | [Kimi CLI](https://kimi.ai) |
| `amp` | `amp` | [Amp](https://amp.dev) |
| `rovodev` | `rovodev` | [Rovo Dev](https://www.atlassian.com/software/rovo) |
| `llxprt` | `llxprt` | [LLXpert](https://llxpert.ai) |
| `coder` | `coder` | [Coder](https://coder.com) |
| `claudish` | `claudish` | [Claudish](https://github.com/claudish-dev/claudish) (multi-provider) |
| `happy` | `happy` | [Happy](https://happy.dev) |

### Model tiers

All backends support generic model tiers that map to backend-specific models:

| Tier | Meaning | Example mapping |
|------|---------|----------------|
| `fast` | Fastest/cheapest | `haiku`, `gpt-4o-mini` |
| `balanced` | Default, good tradeoff | `sonnet`, `gpt-4o` |
| `powerful` | Most capable | `opus`, `o3` |

You can also pass backend-specific model names directly.

### Third-party backends

Register your own backend via Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) using the `claude_teams.backends` group. See [Adding a Backend](#adding-a-backend) for details.

---

## MCP Tools

Tools are organized into three tiers using **progressive disclosure**. At startup, only bootstrap tools are visible. Higher tiers unlock as you create teams and spawn teammates, which keeps the cold-start tool surface small.

### Tier 0: Bootstrap (always visible)

| Tool | Description |
|------|-------------|
| `team_create` | Create a new agent team and return the lead capability for later re-attachment. One team per server session. |
| `team_attach` | Attach a new MCP session to an existing team using a lead or agent capability. |
| `team_delete` | Delete a team and all its data. Fails if teammates are still active. |
| `list_backends` | List all available backends and their supported models. |
| `read_config` | Read team configuration and member list (lead capability required). |

### Tier 1: Team (visible after `team_create`)

| Tool | Description |
|------|-------------|
| `spawn_teammate` | Spawn a coding agent via any backend. Specify backend, model, prompt, optional absolute `cwd`, and optional `permission_mode`. |
| `send_message` | Send direct messages, broadcasts, or shutdown/plan-approval responses. |
| `read_inbox` | Read messages from an agent's inbox with pagination metadata, optional unread-only filtering, and `oldest`/`newest` ordering. |
| `task_create` | Create a new task with auto-incrementing ID. |
| `task_update` | Update task status, owner, dependencies, or metadata. |
| `task_list` | List team tasks in canonical task-ID order with pagination metadata. |
| `task_get` | Get full details of a specific task. |

### Tier 2: Teammate (visible after first `spawn_teammate`)

| Tool | Description |
|------|-------------|
| `force_kill_teammate` | Forcibly kill a teammate's process and remove from team. |
| `check_teammate` | Inspect one teammate's liveness, lead-facing unread messages, unread count, and optional captured output. |
| `poll_inbox` | Long-poll an inbox for new messages (blocks up to 30 seconds). |
| `process_shutdown_approved` | Cleanly remove a teammate after graceful shutdown approval. |
| `health_check` | Check if a teammate's process is still running. |

### Capability model

- `team_create` returns a `lead_capability`. Keep it if you want to attach another MCP session to the same team later.
- `spawn_teammate` issues a per-agent capability and includes `team_attach(...)` instructions in the worker's bootstrap prompt.
- A session attached with the lead capability can perform all lead actions.
- A session attached with an agent capability can perform agent-scoped actions as that agent, but cannot use lead-only tools such as `broadcast`, `force_kill_teammate`, `check_teammate`, or `health_check`.

### Permission modes

- `spawn_teammate` accepts `permission_mode` with three values:
  - `default`: preserve the backend's normal automation behavior
  - `require_approval`: strip backend auto-approval flags when supported
  - `bypass`: require an explicit approval-bypass mode; unsupported backends reject the spawn
- You can also set `CLAUDE_TEAMS_PERMISSION_MODE` to change the server-wide default when `permission_mode` is omitted.
- Backends vary in how they implement it. `claude-code` uses `--permission-mode bypassPermissions`; one-shot CLIs such as `codex`, `coder`, `gemini`, `aider`, `amp`, `copilot`, `happy`, `llxprt`, `qwen`, `rovodev`, and `claudish` use their backend-native bypass flags.

### Pagination

- `task_list` and `read_inbox` now return paginated envelopes with:
  - `items`
  - `total_count`
  - `limit`
  - `offset`
  - `has_more`
  - `next_offset`
- `task_list` keeps its canonical task-ID ordering.
- `read_inbox` keeps `order="oldest"` by default and also accepts `order="newest"` so recent mail can be fetched from page 1.
- This avoids unbounded MCP payloads without truncating message or task content.

Example `read_inbox` response:

```json
{
  "items": [
    {
      "from": "team-lead",
      "text": "Please review task 7",
      "timestamp": "2026-04-07T18:22:11.531Z",
      "read": false,
      "summary": "review request"
    }
  ],
  "totalCount": 1,
  "limit": 100,
  "offset": 0,
  "hasMore": false
}
```

### Optional Inbox Encryption

- Set `CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY` to enable inbox-at-rest encryption.
- The server derives a per-team inbox key from that master key and encrypts inbox entries on disk.
- Existing plaintext inbox entries remain readable; they are opportunistically rewritten encrypted on the next modifying inbox operation.
- If encrypted inbox entries exist and the master key is missing or wrong, inbox reads fail closed.

---

## MCP Prompts

Templated prompts let clients drive common team-lead operations without
re-writing orchestration instructions each time. All prompts are tagged
`team`, so they become visible alongside team-tier tools after
`team_create` or `team_attach` via the same progressive-disclosure
mechanism.

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `status_check` | Check a teammate's health, current task, and blockers. | `team`, `teammate` |
| `health_sweep` | Run health check across all teammates and flag issues. | `team` |
| `task_handoff` | Hand off completed work from one teammate to another. | `team`, `from_teammate`, `to_teammate`, optional `context` |
| `wrap_up` | Collect all task results and prepare a team summary. | `team` |
| `unblock_teammate` | Send a targeted message to unblock a stuck teammate. | `team`, `teammate`, optional `hint` |

Each prompt returns an assistant-prefilled conversation that skips
preamble and drives the model straight into the supporting tool calls.

---

## Skills

The server acts as a skill provider for connected clients, surfacing
agentic-skills content alongside its tools.

### Bundled skill

The server ships with `team-orchestration/SKILL.md`, a complete guide to
leading a team through this MCP server: lifecycle, delegation patterns,
coordination, monitoring, and teardown. It is served to any client that
supports agent skills.

### Custom backend skill roots

Beyond FastMCP's built-in providers for Claude, Codex, Copilot, Cursor,
Gemini, Goose, OpenCode, and VSCode, this server registers nine
additional `SkillsDirectoryProvider`s covering the remaining backends in
the registry:

| Backend | Skill roots (in lookup order) |
|---------|-------------------------------|
| `amp` | `~/.config/amp/skills/`, `~/.config/agents/skills/`, `~/.claude/skills/` |
| `auggie` | `~/.augment/skills/`, `~/.claude/skills/`, `~/.agents/skills/` |
| `coder` | `~/.code/skills/`, `~/.codex/skills/` |
| `goose` | `~/.agents/skills/`, `~/.goose/skills/`, `~/.claude/skills/` |
| `kimi` | `~/.kimi/skills/`, `~/.claude/skills/`, `~/.codex/skills/`, `~/.config/agents/skills/`, `~/.agents/skills/` |
| `llxprt` | `~/.llxprt/skills/` |
| `qwen` | `~/.qwen/skills/` |
| `rovodev` | `~/.rovodev/skills/` |
| `vibe` | `~/.vibe/skills/` |

Earlier roots win on name collision. The `goose` override exists because
FastMCP's built-in `GooseSkillsProvider` points at a path Block Goose
does not actually use.

---

## CLI Reference

The `claude-teams` CLI provides terminal commands for inspecting and managing teams. The CLI and MCP server can run concurrently -- they share the same file-based state with `fcntl.flock()` guards.

```
claude-teams serve              # Start the MCP server
claude-teams backends           # List available backends
claude-teams config TEAM        # Show team config
claude-teams status TEAM        # Show member status and task summary
claude-teams inbox TEAM AGENT   # Read an agent's inbox messages
claude-teams health TEAM AGENT  # Health-check a teammate's process
claude-teams kill TEAM AGENT    # Force-kill a teammate
```

All commands support `--json` / `-j` for machine-readable output.

### Examples

```bash
# List what backends are available on your system
claude-teams backends

# Show team config with member list
claude-teams config my-team

# Check if all agents are still running
claude-teams health my-team worker-1

# Read the team lead inbox in newest-first order
claude-teams inbox my-team team-lead --order newest

# Force-kill an unresponsive agent
claude-teams kill my-team stuck-worker
```

---

## How It Works

### Spawning

Teammates launch as separate processes in tmux panes via `tmux split-window`. Each agent gets:

- A unique agent ID (`name@team`)
- An assigned color from a rotating palette
- Backend-specific CLI flags and environment variables
- Its initial prompt delivered to its inbox

### Messaging

JSON-based inboxes stored under `~/.claude/teams/<team>/inboxes/`. File locking via `fcntl.flock()` prevents corruption from concurrent reads and writes. Message types:

- **Direct messages** -- one-to-one communication between agents
- **Broadcasts** -- send to all teammates at once
- **Shutdown requests/responses** -- graceful shutdown protocol
- **Plan approval** -- request and grant plan approval

### Task tracking

JSON task files stored under `~/.claude/tasks/<team>/`. Tasks support:

- Status progression: `pending` -> `in_progress` -> `completed`
- Ownership assignment to specific agents
- Dependency graphs (`blocks` / `blockedBy`) with cycle detection
- Arbitrary metadata
- Auto-incrementing IDs

### Storage layout

```
~/.claude/
├── teams/<team-name>/
│   ├── config.json          # Team config + member list
│   ├── inboxes/
│   │   ├── team-lead.json   # Lead agent inbox
│   │   ├── worker-1.json    # Teammate inboxes
│   │   └── .lock
│   └── runs/                # One-shot backend result files (auto-cleaned)
└── tasks/<team-name>/
    ├── 1.json               # Task files (auto-incrementing IDs)
    ├── 2.json
    └── .lock
```

### Concurrency safety

| Operation | Safety mechanism |
|-----------|-----------------|
| Config writes | Atomic via `tempfile.mkstemp()` + `os.replace()` to prevent partial reads |
| Inbox operations | Guarded by `fcntl.flock()` file locks |
| Task operations | Guarded by `fcntl.flock()` file locks with validation-then-write phasing |

### Progressive tool disclosure

The server uses FastMCP's tag-based visibility system to progressively reveal tools and prompts as state evolves:

1. **Cold start** -- only 5 bootstrap tools visible; team-tagged prompts hidden
2. **After `team_create`** -- 7 team-tier tools and 5 team-tagged prompts become visible
3. **After first `spawn_teammate`** -- 5 teammate-tier tools become visible
4. **After `team_delete`** -- resets back to bootstrap-only

This is transparent to MCP clients -- the server sends `ToolListChangedNotification` automatically.

---

## Development

### Setup

```bash
git clone https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync
```

### Running tests

```bash
uv run pytest                           # Run all tests
uv run pytest --cov=claude_teams        # With coverage
uv run pytest tests/test_server_bootstrap.py -v  # Single module
uv run pytest -k "test_spawn"           # Filter by name
```

### Linting and type checking

```bash
uv run ruff format                      # Format
uv run ruff check                       # Lint
uv run ty check                         # Type check (Astral's ty)
```

### Observability

OpenTelemetry tracing is optional. Install the extra and set the usual
OTel environment variables — FastMCP emits server spans for every tool,
prompt, and resource invocation once a tracer provider is registered.

```bash
uv sync --extra otel
# Or with pip
pip install 'claude-teams[otel]'
```

Configuration (standard OTel environment variables):

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_SERVICE_NAME` | Service name stamped on every span | `claude-teams` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/HTTP collector endpoint | unset (no export) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers for the collector | unset |
| `OTEL_SDK_DISABLED` | Set to `true`/`1` to disable without uninstalling | unset |

If the `otel` extra is not installed, telemetry setup is a silent no-op
and the server runs normally.

### Project structure

```
src/claude_teams/
├── server.py            # Thin MCP assembly surface and entrypoint
├── server_runtime.py    # Shared MCP app, auth, pagination, annotations, lifecycle
├── server_schema.py     # Reusable validated parameter types for tools/prompts
├── server_bootstrap.py  # Bootstrap-tier tool registration
├── server_team_spawn.py # Team-tier spawn and message tools
├── server_team_tasks.py # Team-tier task and inbox tools
├── server_team_relay.py # One-shot backend relay helpers
├── server_teammate.py   # Teammate-tier tool registration
├── server_prompts.py    # Team-tier MCP prompt templates
├── skill_providers.py   # Custom SkillsDirectoryProvider table for non-default backends
├── telemetry.py         # Optional OpenTelemetry tracer-provider setup
├── errors.py            # Central exception taxonomy
├── cli.py               # Typer CLI commands
├── async_utils.py       # Thread offload helper for blocking local operations
├── models.py            # Pydantic models (TeamConfig, Task, InboxMessage, etc.)
├── teams.py             # Team CRUD (config read/write, member management)
├── tasks.py             # Task CRUD (create, update, list, dependencies)
├── messaging.py         # Inbox operations (read, write, structured messages)
├── capabilities.py      # Lead/agent capability storage and resolution
├── inbox_crypto.py      # Optional inbox-at-rest encryption helpers
├── filelock.py          # Shared fcntl-based file locking
├── skills/
│   └── team-orchestration/SKILL.md  # Bundled team-orchestration skill
└── backends/
    ├── base.py          # Compatibility re-exports for backend contracts/base
    ├── contracts.py     # Backend protocol and spawn result/request types
    ├── tmux_base.py     # Shared tmux-backed BaseBackend implementation
    ├── registry.py      # Auto-discovery and registration
    ├── claude_code.py   # Claude Code backend
    ├── codex.py         # OpenAI Codex backend
    ├── gemini.py        # Gemini CLI backend
    └── ...              # 14 more backend implementations
```

---

## Contributing

### Adding a backend

1. Create `src/claude_teams/backends/your_backend.py` inheriting from `BaseBackend`
2. Implement these methods:
   - `build_command(request) -> list[str]` -- CLI args to spawn the agent
   - `build_env(request) -> dict[str, str]` -- extra environment variables
   - `supported_models() -> list[str]` -- available model names
   - `default_model() -> str` -- default model when none specified
   - `resolve_model(model) -> str` -- map generic tiers to backend-specific names
   - optionally `default_permission_args() -> list[str]` when the backend normally auto-approves
   - optionally `bypass_permission_args() -> list[str]` when explicit bypass differs from the default
3. Set class attributes: `name`, `binary_name`
4. Add the entry to `_BUILTIN_BACKENDS` in `registry.py`
5. Add tests in `tests/test_backends/test_your_backend.py`

Example backend (minimal):

```python
from claude_teams.backends.base import BaseBackend, SpawnRequest

class MyBackend(BaseBackend):
    _name = "my-tool"
    _binary_name = "mytool"

    def supported_models(self) -> list[str]:
        return ["default", "large"]

    def default_model(self) -> str:
        return "default"

    def resolve_model(self, model: str) -> str:
        return {"fast": "default", "balanced": "default", "powerful": "large"}.get(model, model)

    def build_command(self, request: SpawnRequest) -> list[str]:
        return [self.binary_name, "--model", request.model, "--prompt", request.prompt]

    # build_env is inherited from BaseBackend and returns {} by default.
    # Override only when your backend needs to export custom env vars.
```

Or register externally via the `claude_teams.backends` entry point group in your package's `pyproject.toml`:

```toml
[project.entry-points."claude_teams.backends"]
my-tool = "my_package.backend:MyBackend"
```

### Code style

- Format with `ruff format`, lint with `ruff check`
- Type check with `ty check`
- All tool functions include docstrings (behavioral description only -- FastMCP generates parameter docs from type annotations)
- Tests use `pytest` with `pytest-asyncio` in auto mode

### Pull requests

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass: `uv run pytest`
4. Ensure coverage stays above 90%: `uv run pytest --cov=claude_teams`
5. Format and lint: `uv run ruff format && uv run ruff check`

---

## Architecture Deep Dive

<details>
<summary>Click to expand -- detailed technical architecture for engineers</summary>

### Backend protocol

Backends implement a `Backend` protocol providing:

- **Lifecycle**: `spawn`, `health_check`, `kill`, `graceful_shutdown`
- **Interactivity**: `capture`, `send`, `wait_idle`, `execute_in_pane`
- **Model resolution**: Map generic tiers (`fast`, `balanced`, `powerful`) to backend-specific model IDs

A `BaseBackend` class provides shared tmux lifecycle management via [`claude-code-tools`](https://pypi.org/project/claude-code-tools/). Concrete backends only need to implement `build_command`, `build_env`, and model resolution.

### FastMCP integration

The server is built on [FastMCP 3.x](https://github.com/jlowin/fastmcp) using:

- **Lifespan management** -- `_LifespanState` tracks registry, session ID, active team, and teammate state
- **Tag-based tool visibility** -- tools tagged as `bootstrap`, `team`, or `teammate` with per-session enable/disable
- **Automatic `ToolListChangedNotification`** -- clients are notified when the visible tool set changes

### Message protocol

Messages follow a structured format. The `send_message` tool supports five message types:

| Type | Purpose | Required fields |
|------|---------|----------------|
| `message` | Direct message between team members | `recipient`, `content`, `summary` |
| `broadcast` | Send from `team-lead` to all teammates | `content`, `summary` |
| `shutdown_request` | Ask agent to stop | `recipient` |
| `shutdown_response` | Reply to shutdown | `sender`, `request_id`, `approve` |
| `plan_approval_response` | Approve/reject plan | `recipient`, `request_id`, `approve` |

For `message`, `sender` defaults to `team-lead`; teammates should pass their own
agent name explicitly when sending direct messages. `broadcast` remains
lead-only. Sessions attached with a worker capability may send direct
messages only as that worker unless a lead capability is used.

### Task state machine

```
pending ──> in_progress ──> completed
                │
                └──> deleted (removes file)
```

Transitions are validated -- you cannot go backwards (e.g., `in_progress` -> `pending` is rejected). Dependency cycles are detected and rejected on creation.

### Session constraints

- One team per server session (enforced in `team_create`)
- Team names must match `[a-zA-Z0-9_-]+` and be at most 64 characters
- Agent names follow the same rules and `team-lead` is reserved
- The `has_teammates` flag prevents redundant `enable_components` calls on subsequent spawns

</details>

---

## Acknowledgments

This project stands on the shoulders of giants. The original implementation and protocol reverse-engineering was done by [Victor](https://github.com/cs50victor) in [claude-code-teams-mcp](https://github.com/cs50victor/claude-code-teams-mcp), based on his [deep dive into Claude Code's internals](https://gist.github.com/cs50victor/0a7081e6824c135b4bdc28b566e1c719). His work cracking open the agent teams protocol and building the first standalone MCP server for it made everything here possible. Thank you, Victor.

## License

[MIT](./LICENSE)
