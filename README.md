# agentic-coder-teams-mcp

Minimal MCP server for spawning and communicating with Claude Code, Codex, and Pi agents on Windows or Linux. Fire-and-forget agent spawning with bidirectional 1:1 messaging.

## Tools (20 total)

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start an agent process (fire-and-forget); returns its state-marker path |
| `create_join_ticket` | Reserve a name and issue a paste-ready external-member join prompt |
| `join_team` | Register a manually started interactive session with a join ticket |
| `external_send` | Send from an external member to its parent using a member token |
| `external_read` | Drain an external member's inbox using a member token |
| `leave_team` | Permanently leave an external membership |
| `send_message` | Send a message to an agent or lead |
| `read_messages` | Read unread messages from own inbox (delta + watermark) |
| `check_agent` | Check an agent's status: state, last line, unread count, stall signal |
| `follow_up_agent` | Resume an existing logical agent with a follow-up prompt |
| `delivery_status` | Reconcile/query one guaranteed-delivery attempt |
| `deliver_pending` | Complete queued guaranteed-delivery attempts |
| `kill_agent` | Force-kill an agent process |
| `resume_session` | Explicitly adopt a recoverable prior session |
| `session_info` | Report the active and recoverable sessions |
| `list_agents` | List all agents with compact status rows |
| `agent_status` | Cheap per-agent state/watermark/stall rows (no bodies) |
| `agent_watch_paths` | Session watch envelope plus minimal `{name, state_marker_path}` agent rows |
| `list_backends` | List available backends |
| `install_lead_wake` | Install/remove the Claude Code lead inbox-wake `Stop` hook for a top-level lead |
| `install_member_wake` | Install/remove the external-member inbox-wake `Stop` hook for a joined member session |

## Quick Start

> Full install, upgrade, and troubleshooting guide (including the "MCP server
> doesn't show up in Claude Code" checklist): **[INSTALL.md](INSTALL.md)**.

### Prerequisites

- Windows 10/11 or Linux
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Claude Code CLI (`claude`), OpenAI Codex CLI (`codex`), and/or Pi CLI (`pi`) on `PATH`
- Linux: a terminal emulator or `tmux` for visible agent windows

### Step 0 — Install the server (required for every client)

The package is not on PyPI; install from a clone. All the config snippets
below point at the virtualenv this creates — **run this first**:

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync
```

The launch command everywhere is the venv's Python running the server module:
`/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python -m claude_teams.server_simple`
(Windows: `C:\abs\path\to\agentic-coder-teams-mcp\.venv\Scripts\python.exe`).

### Setup — Claude Code as Lead

Recommended: register once at **user scope** with the `claude mcp add` CLI —
available in every project, no approval prompt, no hand-edited JSON:

```bash
# Linux
claude mcp add --scope user win-agent-teams -- \
  /abs/path/to/agentic-coder-teams-mcp/.venv/bin/python -m claude_teams.server_simple
```

```powershell
# Windows (PowerShell)
claude mcp add --scope user win-agent-teams -- C:\abs\path\to\agentic-coder-teams-mcp\.venv\Scripts\python.exe -m claude_teams.server_simple
```

Verify with `claude mcp list` (should report **connected**) or `/mcp` inside a
session.

Alternatively, register per-project via a `.mcp.json` in the root of **the
project where you run `claude`** (not inside this repo's clone). Note that
project-scope servers require a one-time interactive approval in Claude Code:

```json
{
  "mcpServers": {
    "win-agent-teams": {
      "command": "/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python",
      "args": ["-m", "claude_teams.server_simple"]
    }
  }
}
```

On Windows use the venv's `Scripts\python.exe` path with **doubled
backslashes** (`"C:\\abs\\path\\to\\...\\.venv\\Scripts\\python.exe"`) —
single backslashes are invalid JSON and break the whole file.

Spawned Claude Code agents get the MCP server automatically via `--mcp-config`.
Spawned Codex agents need the Codex setup below only when they must call MCP
tools themselves, for example to `send_message` back to lead. Passive Codex
workers can still be observed through `check_agent` output fallback.

### Setup — Codex as Lead (or as Spawned Agent)

Add to `~/.codex/config.toml` (Windows: `C:\Users\<you>\.codex\config.toml`) so
Codex can use the MCP tools (both as lead and as spawned agent):

```toml
[mcp_servers.win-agent-teams]
command = "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass" }
enabled = true
```

Linux example:

```toml
[mcp_servers.win-agent-teams]
command = "/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass" }
enabled = true
```

This is required in two scenarios:
1. **Codex as lead** — Codex calls `spawn_agent` to start Claude Code or other Codex agents.
2. **Codex as spawned agent using MCP tools** — when Claude Code spawns a Codex agent and expects that agent to call tools such as `send_message`.

The server auto-injects `AGENT_NAME` and `AGENT_SESSION_ID` into the Codex config env before each spawn so the MCP server knows agent identity.

### Setup — Pi (as Lead or Spawned Agent)

[Pi](https://github.com/earendil-works/pi) (`pi`) has, by design, no built-in MCP
support, so it reaches the win-agent-teams tools through the official
[`pi-mcp-adapter`](https://pi.dev/packages/pi-mcp-adapter) package. One-time setup:

```bash
# 1. Install pi and log into your model provider (e.g. ChatGPT Plus/Pro):
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
pi            # then run /login inside pi and pick your provider

# 2. Install the MCP adapter so pi can call the win-agent-teams tools:
pi install npm:pi-mcp-adapter
```

You do **not** hand-write any MCP config. Identity is delivered differently for
the two pi scenarios:

- **Human-launched pi *lead*** — the server writes `~/.pi/agent/mcp.json`
  (idempotently, on each pi spawn) with a `win-agent-teams` entry whose identity
  env uses `${AGENT_NAME}`/`${AGENT_SESSION_ID}`/`${AGENT_PARENT_NAME}`
  interpolation, resolved from the pi process's own environment. With no
  `AGENT_*` set, identity resolves to `team-lead` (the project `.mcp.json` may
  also apply — see the warning below).
- **Spawned pi *worker*** — the server writes a **per-agent MCP config** with
  **literal** `AGENT_*` values at `<session_dir>/mcp/<agent>.pi.mcp.json` and the
  pi backend passes it to pi via `--mcp-config`. This mirrors the Claude path and
  does not rely on `${AGENT_*}` interpolation, so a spawned worker's identity is
  never left to the environment (which lower-precedence MCP config sources can
  clobber).

The adapter starts the server **lazily** (only when pi actually calls a tool), so
a normal `pi` run you do yourself is unaffected apart from one ~200-token proxy
tool.

> **WARNING — never put `AGENT_*` in a project MCP config.** Do not add an
> `AGENT_NAME` / `AGENT_SESSION_ID` / `AGENT_PARENT_NAME` `env` block to a project
> `.mcp.json` or `.pi/mcp.json` `win-agent-teams` entry. The pi-mcp-adapter merges
> config sources **later-wins** and replaces the whole `env` map per server entry,
> so an empty (or literal) `AGENT_*` value there overwrites the correct per-agent
> identity delivered via `--mcp-config`. This now causes the win-agent-teams MCP
> server to **refuse** identity-bearing tools (`send_message` / `read_messages` /
> `resume_session` return `{"success": false, "reason": "identity_unresolved"}`)
> rather than silently masquerade as `team-lead` and hijack the lead's session.

This covers both scenarios:
1. **Pi as lead** — pi calls `spawn_agent` to start Claude Code, Codex, or other pi agents.
2. **Pi as spawned agent using MCP tools** — a lead spawns a pi agent that calls `send_message` (etc.) back.

A spawned pi agent reports lifecycle state through a small bundled extension
(`pi-extensions/win-agent-teams-state`, loaded via `-e`) that writes the same
`state-<agent>.json` marker the coordinator watches. Pi launches via `node` +
its bundled `dist/cli.js` directly (bypassing the `pi.cmd` shim) so multi-line
prompts survive verbatim; session binding uses `--session-id <agent>` for
deterministic resume.

## How It Works

### External members

A lead can attach a manually started interactive session—such as a visual QA
session with browser access—without spawning its process:

1. The lead calls `create_join_ticket(name, note)` and gives the returned
   `join_prompt` to the human.
2. The new session calls `join_team(session_id, token)` and saves the returned
   `member_token`.
3. The lead uses ordinary `send_message(to=<member>)`; the external member
   polls with `external_read(member_token)` and reports with
   `external_send(member_token, text)`.
4. The member calls `leave_team(member_token)` when finished. The lead may
   instead call `kill_agent`, which only deregisters an external member and
   never probes or signals its informational PID.

External delivery is inbox-only, pull-based, and unconfirmed. External members
cannot be resumed with `follow_up_agent`. Join/member tokens are bearer
credentials in the same-user disk threat model; `agents.json` stores only the
member-token digest, and public status/list tools redact it.

For the external session, run a separate Desktop profile or separate client
instance whose MCP configuration contains only a
`win-agent-teams-external` entry with
`WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`. That server exposes only `join_team`,
`external_send`, `external_read`, `leave_team`, and `list_backends`. A dual
registration in the same profile is a degraded mode: the normal ambient root
tools remain selectable, so it does not provide client-surface isolation. If
your client cannot scope MCP configuration to a separate profile or instance,
ambient-tool isolation is unavailable on that client.

### Registry backend labels

| Registry `backend` | Process origin | Lead → agent delivery |
|---|---|---|
| `claude-code` | Spawned CLI | Guaranteed resume |
| `codex` | Spawned CLI | Guaranteed resume |
| `pi` | Spawned CLI | Guaranteed resume |
| `external` | Manually started interactive session; not a spawn backend | Pull-only inbox |

### Spawning

`spawn_agent` starts a CLI process and returns immediately with `{name, pid, backend, session_id}`. The agent runs independently.

Display mode is selected automatically:
- Windows uses native processes. When Windows Terminal (`wt.exe`) is available, interactive agents open as **tabs grouped in one window per team** (`wt -w wt-team-<team>`), each tab titled `<agent>@<team>`; the tab closes when the agent exits or is killed. If `wt.exe` is not on PATH, each interactive agent falls back to its own console window. Set `WIN_AGENT_TEAMS_NO_WT_TABS=1` to force the classic one-window-per-agent console even when `wt.exe` is present.
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

Interactive agents launch in a visible console by default so their live UI
remains visible while they work — a Windows Terminal tab (grouped per team)
when `wt.exe` is available, otherwise a standalone console window. On Windows,
set `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` to capture stdout/stderr to the
per-agent log file instead, or `WIN_AGENT_TEAMS_NO_WT_TABS=1` to keep the
visible console but force a separate window per agent instead of tabs.

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

`read_messages` returns only **unread** messages, delta-by-default, plus a
watermark. A per-sender counter sidecar (`inbox-{name}.pos.json`) tracks how
much of each sender's stream the reader has already consumed, so reads stay
O(n) instead of re-returning the whole inbox every call.

Return shape: `{messages, cursors, seq, unread_count, has_more}`. Each
message is `{from, text, ts, seq}`, where `seq` is that sender's per-sender
COUNT after the message (the same number space as the persisted cursor).
`cursors` (a `{sender: count}` map) is returned when `from_agent` is unset;
a scalar `seq` is returned instead when `from_agent` is set. `read_messages(
from_agent="x")` advances only `x`'s cursor. Pass `since_seq=<count>` (only
valid together with `from_agent`) to fetch a specific tail and advance the
cursor to `max(current, since_seq)` — no rewind, no boundary re-delivery.
`limit` bounds a batch (default 50; `has_more` signals more remain);
`full=True` ignores the limit. `max_chars` truncates each message's `text`
(adds `truncated`/`full_len` per message). Delivery is best-effort: a
crashed reader or a lost/corrupt cursor file may cause a message to be
re-delivered or, in the rare case of two lead processes sharing one inbox,
consumed by the other process.

### Output Fallback

`check_agent(name, full=False, max_chars=200)` returns a compact status peek
by default: `{name, state, alive, pid, backend, last_activity_at,
unread_count, last_line, seq, truncated}`. `state` is one of
`running`/`waiting`/`idle`/`dead` (see "State" below). `last_line` is the
last non-empty line of the worker's most recent assistant message, clipped
to `max_chars` (default 200) with `truncated` signalling clipping.
`unread_count`/`seq` count messages FROM this agent addressed to the caller.

Pass `full=True` to restore the previous behavior: the full `last_message`
(still bounded to 1000 chars, tail-truncated with a
`[truncated: showing last N of M chars]` marker) plus `backend_session_id`,
both read from the CLIs' existing JSONL session logs. This is a fallback for
workers that finish without calling `send_message`; it does not replace
explicit agent-to-lead messaging, and it is a status peek, not the full
output — read the agent's own session log if you need the complete text.

`list_agents(full=False)` returns compact rows by default: `{name, state,
alive, pid, backend, last_activity_at, unread_count}` — no transcript
bodies. Pass `full=True` to restore each agent's raw registry record plus a
`last_line` peek.

`agent_status(names=None)` returns the cheapest possible per-agent rows —
`{name, state, last_activity_ts, unread_count, seq, heartbeat_age_s,
stalled}`, no bodies at all. It costs one state-marker read + one cursor read
+ one liveness check per agent (no transcript scan) once a marker exists,
which is what makes it suitable for a coordinator sweep across many agents.
`stalled` is `true` when an agent is alive and not `waiting`/`dead` but has
produced nothing for longer than `WIN_AGENT_TEAMS_STALL_SECONDS` (default
300) — i.e. alive-but-hung, detectable with zero transcript bytes.

### State

Agent `state` is hook-driven when possible: a small hook command
(`python -m claude_teams.hooks emit`) is injected into spawned Claude Code
agents (`--settings <path>`, on by default; disable with
`WIN_AGENT_TEAMS_STATE_HOOKS=0`) and writes a per-agent marker file
(`state-{name}.json`) mapping `SessionStart`/`UserPromptSubmit`/
`PreToolUse`/`PostToolUse` → `running` and `Stop`/`SubagentStop` →
`waiting`. Codex agents get the same marker via an inline `-c` hook override
plus `--dangerously-bypass-hook-trust` (Codex silently skips injected hooks
without that flag); this is **on by default** — set
`WIN_AGENT_TEAMS_STATE_HOOKS_CODEX=0` to disable it (or
`WIN_AGENT_TEAMS_STATE_HOOKS=0` to disable hooks for both backends). Pi has no
hook CLI, so a spawned pi agent writes the same marker from a bundled extension
(`pi-extensions/win-agent-teams-state`, loaded via `-e`): pi's `session_start`/
`turn_start`/`tool_call` → `running` and `agent_settled` → `waiting`. It is
also gated by `WIN_AGENT_TEAMS_STATE_HOOKS`; override the extension path with
`WIN_AGENT_TEAMS_PI_EXTENSION`.
When no marker exists yet (hooks disabled, or not fired), `state` falls back
to an activity-recency heuristic: `running` if the agent produced output
within the last `WIN_AGENT_TEAMS_IDLE_SECONDS` (default 60s), else `idle`. A
dead process always reports `state="dead"`, regardless of a stale marker.
`kill_agent` deletes the marker so a reused agent name never inherits a dead
predecessor's state.

State-marker environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `WIN_AGENT_TEAMS_STATE_HOOKS` | `1` (on) | Master switch for hook-driven state injection (both backends). `0` disables it; `state` then uses the activity-recency fallback only. |
| `WIN_AGENT_TEAMS_STATE_HOOKS_CODEX` | `1` (on) | Codex-specific switch. `0` leaves Claude hooks on but skips Codex `-c` injection and its `--dangerously-bypass-hook-trust` flag. |
| `WIN_AGENT_TEAMS_IDLE_SECONDS` | `60` | Age (seconds) beyond which an alive-but-quiet agent with no marker is reported `idle` instead of `running`. |
| `WIN_AGENT_TEAMS_STALL_SECONDS` | `300` | Age (seconds) beyond which an alive, non-`waiting` agent is flagged `stalled` (alive-but-hung). |
| `WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS` | `1.5` | Seconds a `waiting` marker must persist before `watch` wakes on it; a marker that resumes `running` within the window is suppressed as a brief park. `0` disables settling. Malformed values fall back to the default. |

### Coordinating without polling (the marker file bus)

A coordinator cannot be *pushed* to directly by MCP, and this MCP server itself
may be shut down after a few minutes of host inactivity. What survives is the
**file bus** — workers are detached processes, state markers and inboxes are on
disk — so coordination is built on a semantic background watcher, not on the
server staying up. (The tool descriptions carry the same recipe, since a
spawned agent only reads those, not this README.)

The loop:

1. `spawn_agent(..., expected_outputs=["report.md", ...])` returns
   `state_marker_path`, `session_dir`, shell-neutral `watch_argv`,
   `watch_command_bash`, `watch_command_powershell`, and echoes
   `expected_outputs`. Declare the exact files the agent is told to create so
   you can watch precise paths. There is intentionally no unqualified
   `watch_command` field.
2. Execute `watch_argv` directly, or use the rendering for your current shell.
   The `win-agent-teams` console script may not be on PATH; `watch_argv` uses
   the server's interpreter with `-m claude_teams.cli` and is the canonical
   value (also use it for cmd.exe). The watcher ignores non-actionable churn
   and exits only for actionable work: unread lead inbox data, a selected output
   change, or a marker that settles as `waiting`. Two writes are treated as
   churn and never wake on their own: `running` lifecycle transitions, and
   `SubagentStop` (a worker's own internal Task subagent finishing while the
   worker keeps going). A `waiting` marker must also *persist* as waiting for a
   short settle window (`WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS`, default `1.5`)
   before it wakes — one that flips back to `running` inside the window is
   suppressed as a brief park. A genuine `waiting` that arrives in the final
   settle window can fall past the deadline and surface as `exit 2`; the
   mandated status re-check after a timeout recovers it. Each watch is
   one-shot and exits on the first signal, so re-arm it after every wake.
3. Branch on its single JSON wake record: `reason="message"` → call
   `read_messages`; `reason="waiting"` → call `agent_status(names)` (or read the
   marker directly); `reason="output"` → inspect the output. Use
   `agent_watch_paths(names)` to rediscover an envelope containing the same
   session-wide watch metadata and minimal `{name, state_marker_path}` agent
   rows. Its `has_session` boolean distinguishes no active session from a live
   session with zero agents.

Wake wiring differs by orchestrator:

- **Claude Code coordinator** — run the watch as a **background** command; its
  completion triggers the harness background-task notification, waking you.
  Then call `agent_status`. You can stay idle in between.
- **Codex coordinator** — Codex has no idle-wake, so run a **bounded
  foreground** watch inside the turn and branch on its JSON reason; loop as
  needed. On timeout exit 2, re-check status before starting the next watch to
  close the small status-check/watch-baseline race.

A ready-made bounded watcher ships as a CLI (needs no MCP server). The command
below documents the CLI shape; coordinators should use the returned
`watch_argv` or shell-specific rendering because the console script may not be
on PATH:

```bash
win-agent-teams watch <session-dir> [--timeout SECONDS] [--pattern 'state-*.json']
# exit 0 + one JSON reason (message/waiting/output); exit 2 on timeout
# add --no-inbox when a custom pattern must remain artifact-only
```

Inbox wake is now enabled even when `--pattern` is supplied. Existing scripts
that used a custom pattern exclusively to await an artifact should add
`--no-inbox` to preserve that behavior.

### Claude Code lead wake

The watch recipe above depends on the lead actually *arming* a background
watcher before it goes idle. A `Stop` hook makes that deterministic instead of
trusting the model to remember: on every lead turn end it inspects the harness's
own `background_tasks` array (delivered in the `Stop` payload) and decides, in a
single read-only inbox scan:

- **unread reply already waiting** → block, naming the sender(s), instructing the
  lead to call `read_messages` and keep draining while `has_more` is true;
- **no unread and no watcher armed** → block with the operational instruction to
  start the rendered `watch_command_bash` as a background task (`run_in_background`);
- **watcher armed** (a running `background_tasks` entry whose command is this
  session's `claude_teams.cli watch <session-dir>`) → allow; the tracked watcher
  carries the wait and the harness wakes the lead when it exits;
- **nothing to wait for** (no live subagents, or no session) → allow immediately.

The hook is **zero-token** (it returns instantly in every branch; the long wait
lives in the tracked watcher, not the hook) and always **fail-open** — it never
emits `{"continue":false}` and never exits non-zero, so it can never make a lead
unstoppable. A per-lead progress guard (`wake-progress-<reader>.json` under the
session dir, keyed on inbox **cursor** advance) caps a no-progress block loop at
`WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` (default `3`, well under the harness's
8-block ceiling). Identity is per-lead: a spawned nested lead resolves its own
`inbox-<AGENT_NAME>.jsonl`, never a hardcoded `team-lead`.

Server-spawned Claude Code agents get this wiring automatically (it is added as a
second `Stop` matcher group alongside the state-marker `emit` hook). For a
**top-level** lead you start yourself, wire it in one step with the
`install_lead_wake` MCP tool:

- `install_lead_wake()` writes the wake `Stop` hook into the project
  `.claude/settings.json` in the lead's cwd (`scope="user"` targets
  `~/.claude/settings.json`). It writes only the wake group, is idempotent, and
  preserves unrelated hooks.
- `install_lead_wake(remove=true)` removes only the wake group.

See [INSTALL.md](INSTALL.md) section 6 for the full automatic-vs-manual matrix,
verification steps, and wake-related troubleshooting.

Environment tunables:

| Env var | Default | Meaning |
|---------|---------|---------|
| `WIN_AGENT_TEAMS_LEAD_WAKE` | `1` | Kill switch. `0` disables the hook at runtime (even for already-wired sessions). |
| `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` | `3` | Consecutive no-progress blocks before the guard fails open. |
| `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_WAIT` | `0` | Optional in-hook grace-wait seconds (reserved; default `0` = no in-hook wait). |

The downstream direction (lead → external member) has a member-shaped twin:
after `join_team`, the member session calls
`install_member_wake(joined_session_id, member_name)` (default `scope="user"`,
i.e. `~/.claude/settings.json`) to bake a `claude_teams.member_wake` `Stop`
hook that watches `inbox-<member>` in the **joined** lead's session dir. It
blocks with an `external_read(member_token=...)` instruction while unread work
waits, otherwise verifies a `watch <joined dir> --reader <member>` background
watcher is armed, and fails open when the membership is no longer `running` or
the joined session has been inactive longer than
`WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS` (default `21600` = 6h). Kill switch:
`WIN_AGENT_TEAMS_MEMBER_WAKE` (falls back to `WIN_AGENT_TEAMS_LEAD_WAKE` when
unset); the no-progress guard shares `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS`
and writes `wake-progress-member-<member>.json`. No `member_token` is ever
baked into settings. The lead-wake and member-wake `Stop` groups coexist and
are installed/removed independently.

#### Manual smoke test (interactive; not run in CI)

This is the end-to-end delivery check. It has **not** been run in CI or this
repo's automated suite (the automated proof is `tests/test_lead_wake.py`, which
drives the decision function with faked payloads). Run it with your own Claude
Code and model configuration; record harness version, model, sender, and wake
content per run — do not treat this document as evidence the run occurred.

1. In a repo cwd, run `install_lead_wake` and confirm `.claude/settings.json` has
   the `Stop` wake group and `win-agent-teams session-dir` reports the lead
   identity.
2. Start an interactive `claude` lead; `spawn_agent` a worker; go idle.
3. Observe the hook blocks once with the arm instruction and the lead starts
   `watch …` as a background task.
4. Confirm arming detection's happy path: after the lead starts the watcher via
   the real `watch_command_bash` rendering, the next `Stop` is **allowed** (the
   `background_tasks` entry is recognised) rather than re-blocking the arm
   instruction — this closes the persistent-false-negative mode that unit token
   tests cannot reach.
5. Have the worker `send_message`; confirm the harness re-invokes the idle lead
   when the background watcher exits, and the lead calls `read_messages` and
   drains while `has_more` is true.
6. Confirm one wake per generation (no storm), re-arm after each wake, the cursor
   never double-advances, and the lead stays interruptible/typeable. Repeat on
   Opus `--effort low` for ≥ 10 idle turns for a 100% wake rate, and run on both
   Linux and Windows.

### Follow-up / Resume

`follow_up_agent(name, prompt, replace_if_idle=true)` continues the same logical agent by starting a new backend process with the CLI's native resume mechanism. Codex uses `codex resume` with the same permission/cwd/reasoning settings as spawn; Claude Code uses `claude --resume`. If the old process is still alive but idle, the tool replaces it by default; pass `replace_if_idle=false` to instead refuse with `agent_idle_but_alive`. A live, busy process is always refused with `agent_busy`.

The tool relies on `backend_session_id`, which `check_agent(name, full=True)` exposes from the backend's JSONL session logs. Once known, that session id is used as the correlation key so resume follow-ups keep reading the correct rollout even when multiple agents share a working directory.

Recommended follow-up pattern:

```
1. Lead calls spawn_agent(..., name="worker")
2. Lead polls check_agent("worker", full=True) until last_message and backend_session_id are present
3. Lead calls follow_up_agent("worker", prompt="next task", replace_if_idle=true)
4. Lead polls check_agent("worker", full=True) for the follow-up last_message
```

### Identity

The server detects its role from environment variables:
- **Lead mode**: No `AGENT_NAME` set → identity = `"team-lead"`
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

**Recommended pattern**: Spawn an agent per task. Put everything it needs in the prompt. Have it report back via `send_message` or rely on `check_agent(name, full=True)` fallback output. For a second turn, use `follow_up_agent` and verify that `backend_session_id` remains stable.

### Smoke-Tested Resume Chain

The native resume path has been smoke-tested with this chain:

```
lead -> Claude Code orchestrator -> Codex CLI target
```

The Claude orchestrator spawned a passive Codex target, observed its base answer with `check_agent(name, full=True)`, called `follow_up_agent(..., replace_if_idle=true)`, and observed the follow-up answer. The Codex `backend_session_id` stayed unchanged across the base and follow-up turns, confirming that the follow-up resumed the same backend session.

## Spawn Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Task prompt for the agent |
| `name` | auto (`agent-1`) | Agent name |
| `backend` | `claude-code` | Spawn backends: `claude-code`, `codex`, or `pi`. `external` is a joined registry label, not a valid spawn backend. |
| `model` | backend default | Model to use. For `codex`/`pi`, a capability tier (`low`/`medium`/`high`/`xhigh`/`ultra`) that bundles a model + effort. Pi tiers soft-fall-back to pi's default model when the tier's model is absent (e.g. after switching provider) rather than erroring. |
| `reasoning_effort` | none | `low`/`medium`/`high`/`xhigh` (codex), `low`/`medium`/`high`/`xhigh`/`max` (claude-code). Ignored for `codex`/`pi` tiers (the tier owns the effort). |
| `permission_mode` | `bypass` | `bypass`, `default`, or `require_approval` |
| `cwd` | server cwd | Working directory for the agent |

## CLI

```powershell
win-agent-teams serve      # Start the MCP server
win-agent-teams backends   # List available backends
win-agent-teams watch DIR  # Block until an actionable edge (settled waiting / message / output) under DIR, or --timeout
```

## Roadmap / future work

Shipped since the initial monitoring model: the disk-derived **stall/heartbeat** signal
(`stalled` / `heartbeat_age_s`) and the **marker file-bus** coordination loop
(`agent_watch_paths`, `spawn_agent` marker paths, the `watch` CLI) — see *Coordinating without
polling* above.

Still open, but constrained by the host harnesses rather than by this server:

- **True push over poll.** A model-facing event that wakes an *idle* coordinator on an external
  MCP event. Currently impossible: no harness surfaces MCP notifications (or a `FileChanged`
  hook) as a mid-idle wake — only a background command's completion wakes the coordinator, which
  is exactly what the marker + `watch` loop rides. A real push needs a harness change.
- **`wait_for(agent, condition, timeout)`.** Deliberately *not* built as a blocking MCP tool:
  the server can be shut down after a few minutes of host inactivity, so a long blocking call is
  unreliable. The bounded `watch` CLI (server-independent) is the robust substitute.

## Development

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync --group dev
```

## License

[MIT](./LICENSE)
