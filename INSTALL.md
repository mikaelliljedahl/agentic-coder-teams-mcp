# Installing and upgrading win-agent-teams

This guide covers a **fresh install** of the `win-agent-teams` MCP server for
Claude Code and Codex on Linux and Windows, plus **upgrading** an existing
setup, wiring the **lead-wake `Stop` hook**, and **troubleshooting** the common
failure modes (especially "the server never shows up in Claude Code").

Setup is always **two parts**:

1. **Register the MCP server** with your client (Claude Code and/or Codex) so
   the tools (`spawn_agent`, `send_message`, `read_messages`, …) exist.
2. **Optionally install the lead-wake `Stop` hook** for a top-level Claude Code
   lead — one `install_lead_wake()` tool call. For a spawned nested lead, pass
   `enable_spawned_lead_wake=true` to `spawn_agent` instead.

Optionally, **native session wake** (section 6a) lets messages wake idle
sessions directly, including Codex Desktop members and Codex leads. With
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` as well, a follow-up reaches a live child
in its running session instead of restarting it. It is off unless the lead's
MCP entry (and each external member's) sets `WIN_AGENT_TEAMS_NATIVE_WAKE=1`;
spawned children inherit the lead's flags.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.12+** | The repo pins `3.12` in `.python-version`; `uv` will fetch it if missing. |
| [uv](https://docs.astral.sh/uv/) | Creates the virtualenv and installs dependencies. |
| `git` | To clone (and later pull upgrades). |
| Backend CLIs | Whichever agents you want to spawn: Claude Code (`claude`), OpenAI Codex (`codex`), and/or Pi (`pi`) on `PATH`. |
| Linux only | A terminal emulator (`qterminal`, `gnome-terminal`, `xterm`, …) or `tmux` for visible agent windows. |
| OS | Windows 10/11 or Linux. |

The package is **not on PyPI** — you install from a git clone.

## 2. Install the server (required for every client)

This step is the same no matter which client (Claude Code, Codex, Pi) you use.
Skipping it is the #1 install failure: the config snippets below point at a
virtualenv Python that only exists **after** `uv sync`.

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync
```

`uv sync` creates a `.venv` inside the clone with the `claude_teams` package
installed. The launch command used in every registration below is that venv's
Python running the server module:

| OS | Command | Args |
|---|---|---|
| Linux | `/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python` | `-m claude_teams.server_simple` |
| Windows | `C:\abs\path\to\agentic-coder-teams-mcp\.venv\Scripts\python.exe` | `-m claude_teams.server_simple` |

Always use the **absolute** venv path — never bare `python`, which would miss
the installed package.

**Sanity check before registering anything** (catches a broken install in
seconds — a working server starts silently and waits on stdin; press `Ctrl+C`
to stop it):

```bash
# Linux
/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python -m claude_teams.server_simple
# Windows (PowerShell)
C:\abs\path\to\agentic-coder-teams-mcp\.venv\Scripts\python.exe -m claude_teams.server_simple
```

If that prints `No module named claude_teams` or a Python error instead, fix
this first — no client-side config can work until this command does.

## 3. Register with Claude Code

Claude Code stores MCP servers at three scopes, and picking the wrong one is
the main reason installs "silently" fail. Recommended: **user scope** via the
`claude mcp add` CLI — one global registration, available in every project, no
approval prompt, no hand-edited JSON.

### 3.1 Recommended: user scope via `claude mcp add`

Linux:

```bash
claude mcp add --scope user win-agent-teams -- \
  /abs/path/to/agentic-coder-teams-mcp/.venv/bin/python -m claude_teams.server_simple
```

Windows (PowerShell):

```powershell
claude mcp add --scope user win-agent-teams -- C:\abs\path\to\agentic-coder-teams-mcp\.venv\Scripts\python.exe -m claude_teams.server_simple
```

Everything after `--` is the literal command + args. This writes the server
into your user-level Claude config (`~/.claude.json`), so it is available in
every project without further steps.

### 3.2 Alternative: project scope via `.mcp.json`

Only use this when you want the registration checked into (or scoped to) one
specific project. Two rules people trip over:

- The `.mcp.json` goes in the root of **the project where you run `claude`**
  — *not* inside the win-agent-teams clone.
- On first use Claude Code shows an **approval prompt** for project-scope
  servers. If it was ever declined, the server stays hidden until you run
  `claude mcp reset-project-choices` in that project and approve again.

`.mcp.json` in your project root — Linux:

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

Windows (note the **doubled backslashes** — single backslashes are invalid
JSON escapes and make the whole file unparseable):

```json
{
  "mcpServers": {
    "win-agent-teams": {
      "command": "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "claude_teams.server_simple"]
    }
  }
}
```

### 3.3 Verify it worked

1. From a terminal: `claude mcp list` — `win-agent-teams` should be listed and
   report **connected** (it launches each server to health-check it).
2. Inside an interactive `claude` session: run `/mcp` — the server should show
   as connected with its 15+ tools.
3. Ask the model to call `list_backends` — it should return the installed
   backend CLIs (claude-code / codex / pi).

If any of these fail, jump to [Troubleshooting](#8-troubleshooting).

## 4. Register with Codex

Codex has exactly **one** config file, which is why this path rarely fails:
add a `mcp_servers` block to `~/.codex/config.toml` (same path on Windows:
`C:\Users\<you>\.codex\config.toml`).

Linux:

```toml
[mcp_servers.win-agent-teams]
command = "/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass" }
enabled = true
```

Windows (TOML also needs doubled backslashes in double-quoted strings):

```toml
[mcp_servers.win-agent-teams]
command = "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass" }
enabled = true
```

This single entry covers both roles: **Codex as lead** (spawning agents) and
**Codex as a spawned agent** that calls tools like `send_message` back to its
lead (the server injects `AGENT_NAME`/`AGENT_SESSION_ID` into the spawned
Codex's environment so identity resolves correctly).

Verify: start a new `codex` session and ask it to call `list_backends`, or
check `/mcp` (recent Codex builds list configured MCP servers on startup).

## 4a. Separate external-member profile (recommended for Desktop QA)

External members carry a returned `member_token` on every call; they do not
rebind the MCP server's process-global identity. For client-surface isolation,
use a **separate Claude Desktop profile, OS profile, or separate client
instance** whose MCP configuration contains only this restricted entry (plus
the browser tools the QA session needs):

Linux:

```json
{
  "mcpServers": {
    "win-agent-teams-external": {
      "command": "/abs/path/to/agentic-coder-teams-mcp/.venv/bin/python",
      "args": ["-m", "claude_teams.server_simple"],
      "env": {"WIN_AGENT_TEAMS_EXTERNAL_ONLY": "1"}
    }
  }
}
```

Windows:

```json
{
  "mcpServers": {
    "win-agent-teams-external": {
      "command": "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "claude_teams.server_simple"],
      "env": {"WIN_AGENT_TEAMS_EXTERNAL_ONLY": "1"}
    }
  }
}
```

The restricted server exposes exactly `join_team`, `external_send`,
`external_read`, `leave_team`, and `list_backends`. The ordinary lead creates a
ticket with `create_join_ticket` and hands its returned `join_prompt` to the
external session.

Do not describe two entries in one profile as isolation. If the normal
`win-agent-teams` entry is also available to that conversation, ambient root
tools remain selectable; this is a degraded dual-entry setup. If your Desktop
version cannot scope MCP configuration to a separate profile or client
instance, ambient-tool isolation is unavailable there. The token-carried
membership and revocation semantics still work, but the client surface is not
contained.

## 5. Pi

Pi setup is different (no hand-written MCP config; identity is delivered via
generated configs). Follow **"Setup — Pi"** in the [README](README.md) — it is
already the authoritative guide, including the warning about never putting
`AGENT_*` variables in a project MCP config.

## 6. The lead-wake `Stop` hook (Claude Code leads only)

The coordination loop relies on an idle lead being *woken* when a worker
replies. A `Stop` hook makes that deterministic. Who needs to install what:

| Situation | Action needed |
|---|---|
| **Server-spawned** Claude Code agent expected to spawn and wait for children | Pass `enable_spawned_lead_wake=true` to `spawn_agent`. The wake hook is then a second `Stop` matcher group in the agent's per-agent `hooks-<name>.settings.json` (passed via `--settings`). Ordinary spawned agents default to state-marker hooks only. |
| **Top-level lead you start yourself** (an interactive `claude` in your repo) | Run the `install_lead_wake` MCP tool **once** (see below). |
| Codex or Pi lead | Not applicable — the hook is Claude Code-specific. Codex leads use the bounded foreground `watch` loop instead (see README "Coordinating without polling"). |

### 6.1 Install

In the lead's Claude Code session (with the MCP server registered), ask it to
call:

```
install_lead_wake()
```

- Default scope (`scope="project"`) writes the `Stop` wake group into
  **`.claude/settings.local.json` in the lead's working directory** — i.e. the
  project where you started `claude`. This is the right choice almost always.
  It is deliberately the personal, git-ignored file and not the checked-in
  `.claude/settings.json`: the hook bakes this machine's absolute paths and the
  lead's PID, so a committed copy breaks every other clone and worktree (a
  Windows install shows up as a `Stop hook error` on Linux). A wake group an
  older version left in `.claude/settings.json` is removed on the next
  install — commit that removal.
- `install_lead_wake(scope="user")` writes it into **`~/.claude/settings.json`**
  instead (applies to every project; usually more than you want).

The tool is **idempotent** (re-running replaces the wake group in place, never
duplicates it) and **preserves unrelated hooks** in the file. It writes *only*
the wake group — never the state-marker `emit` hooks, which belong to
server-spawned agents.

### 6.2 Verify

- The tool's return value names the exact file it wrote
  (`{"action": "installed", "path": ..., "reader": ..., "scope": ...}`).
- Open that settings file and confirm a `hooks.Stop` entry whose command runs
  `claude_teams.hooks` with your session directory.
- `.../.venv/bin/python -m claude_teams.cli session-dir` (or
  `win-agent-teams session-dir` when the console script is on PATH) reports the
  lead identity and session dir the hook will watch.
- Restart the `claude` session (hooks are read at startup), spawn a worker that
  replies via `send_message`, go idle, and confirm the lead wakes and drains
  `read_messages`.

### 6.3 Remove / disable

- `install_lead_wake(remove=true)` removes only the wake group (same scope
  selection), leaving all other hooks intact.
- Kill switch without touching config: set `WIN_AGENT_TEAMS_LEAD_WAKE=0` in the
  lead's environment — the hook then no-ops at runtime, even for
  already-wired sessions.

The hook is fail-open by design: it never blocks indefinitely and can never
make a lead unstoppable (a no-progress guard caps repeated blocks, default 3).

## 6a. Native session wake (optional, opt-in)

Native session wake is an **opt-in doorbell**. It lets a message wake an idle
session directly, without a watcher or a human nudge:

| Direction | Mechanism | Platforms |
|---|---|---|
| Lead → Codex member (Codex TUI or **Codex Desktop**) | `codex queue` on the member's registered thread | Linux, Windows |
| Member/child → Claude Code lead | Body-free notice on Claude Code's own session channel: a Unix socket on Linux, a named pipe on Windows | Linux, **native Windows** (macOS reports `unsupported_platform`) |
| Child → Codex lead | `codex queue` on the lead's thread, registered with `set_lead_wake` (6a.3) | Linux, Windows |
| Lead → live Codex or Claude child | The follow-up itself goes into the child's running session instead of a kill and resume (6a.4) | Linux, Windows |

It is **off by default**. Without the flag the server behaves exactly as before:
the same tool list, the same join prompt and the same results. The feature is
additive. Keep the `Stop` hook (section 6) and the `watch` recipe, because the
doorbell is best-effort and never a delivery receipt. For semantics and
tunables, see the README "Native session wake (opt-in)" section.

### 6a.1 Enable

Set `WIN_AGENT_TEAMS_NATIVE_WAKE=1` in the `env` of **every** win-agent-teams
MCP entry that takes part: the lead's **and** the member's. Setting it on one
side is not enough. `external_set_wake` only exists on a member's server that
started with the flag, and only a flag-on lead sends the doorbell.

Claude Code (user scope). Remove and re-add an existing registration:

```bash
claude mcp remove --scope user win-agent-teams
claude mcp add --scope user win-agent-teams -e WIN_AGENT_TEAMS_NATIVE_WAKE=1 -- \
  /abs/path/to/agentic-coder-teams-mcp/.venv/bin/python -m claude_teams.server_simple
```

Claude Desktop, or any JSON MCP config (Windows paths need doubled
backslashes):

```json
"win-agent-teams": {
  "command": "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe",
  "args": ["-m", "claude_teams.server_simple"],
  "env": {"WIN_AGENT_TEAMS_NATIVE_WAKE": "1"}
}
```

Codex (`~/.codex/config.toml`; add the key to the existing `env` table):

```toml
[mcp_servers.win-agent-teams]
command = "C:\\abs\\path\\to\\agentic-coder-teams-mcp\\.venv\\Scripts\\python.exe"
args = ["-m", "claude_teams.server_simple"]
env = { "CLAUDE_TEAMS_PERMISSION_MODE" = "bypass", "WIN_AGENT_TEAMS_NATIVE_WAKE" = "1" }
enabled = true
```

An external-only member entry (section 4a) needs the flag too, next to
`WIN_AGENT_TEAMS_EXTERNAL_ONLY`. Restart the clients afterwards (Codex Desktop:
quit it fully and start it again), because the flag is read at server
startup.

Spawned agents inherit the flags from their lead. A flag-on lead passes
`WIN_AGENT_TEAMS_NATIVE_WAKE=1`, and `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM`,
`WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE` and `WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX`
with the values the lead has, to every Claude and Codex child it spawns or
resumes. A flag the lead does not have stays unset in the child, and a
flag-off lead passes nothing. For spawned children you only configure the
lead's entry; an external member still needs the flag in its own entry.

### 6a.2 Codex members register their thread

No extra install step is needed. With the flag on, the `join_prompt` from
`create_join_ticket` tells a Codex member to run one shell command, which
prints `CODEX_THREAD_ID` and `CODEX_HOME`, and to pass both to
`external_set_wake(member_token, codex_thread_id, codex_home)`. The member then
ends its turn. The next lead `send_message` queues a wake on that thread.

### 6a.3 A Codex lead registers its own thread

A Claude Code lead is woken through its own session channel. A Codex lead is
woken by `codex queue` on its own thread, which it registers once with
`set_lead_wake(codex_thread_id, codex_home)`. The tool exists only with the
flag on. With the flag on, `session_info` and `resume_session` (and, for a
Codex child spawned with `enable_spawned_lead_wake=true`, its spawn and resume
prompts) tell the lead to run one shell command and pass the two values it
prints:

```bash
echo "$CODEX_THREAD_ID ${CODEX_HOME:-$HOME/.codex}"
```

```powershell
"$env:CODEX_THREAD_ID $(if ($env:CODEX_HOME) {$env:CODEX_HOME} else {Join-Path $HOME '.codex'})"
```

A lead you started yourself is active at once. A lead spawned by another agent
stays `provisional`, and is never queued, until its own thread is confirmed by
its parent's binding. Register again after every restart or resume of the
Codex session; `session_info.native_wake.codex_lead.status` reads
`stale_host` until you do.

### 6a.4 Native downstream delivery (`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`)

Without it, `follow_up_agent` (and `send_message` to a child you spawned)
kills a live child and resumes it with the prompt. With
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` next to `WIN_AGENT_TEAMS_NATIVE_WAKE=1`
in the lead's entry, a live, interactive child gets the message in its running
session instead, keeping its PID:

- **Codex child**: `codex queue` on its thread, when it is idle.
- **Claude Code child**: the lead offers the message in a mailbox file, and
  the child's own MCP server posts it to its own session channel at its next
  idle point. The child inherits the flags from the lead (6a.1).

A dead or headless child, a Pi child, a message over 16 KiB, or a child whose
channel cannot be proven still resumes as before.
`WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE=0` or `WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX=0`
turns off one half, here and for wake.

A native message whose receipt has not appeared yet is reported as
`queued`/`unconfirmed` with reason `native_unresolved`. It can still run, even
after a kill or a reboot, so later messages to that child wait
(`prior_native_attempt_unresolved`) until it is seen or you release it. To give
up on one, use the `lead_token` that `session_info` returns:

```bash
win-agent-teams deliveries release-native <session_id> <idempotency_key> --token <lead_token>
```

The row becomes `failed(operator_released)`. The message may still run.

### 6a.5 Verify

1. `session_info` on the lead returns a `native_wake` object. Under Claude Code
   on Linux or Windows, `claude_channel` should be `available`. On macOS it is
   `unsupported_platform`, which is expected. A Codex lead sees
   `native_wake.codex_lead`, with status `active` after registration.
2. The member's tool list contains `external_set_wake`. If it is missing,
   the member's entry lacks the flag or the client was not restarted.
3. `list_agents(full=true)` on the lead shows a `codex_wake` block on the
   member after registration.
4. `send_message` to the member returns
   `wake: {method: "codex_queue", status: "queued"}`, and the idle Codex thread
   starts a new turn by itself.
5. With `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`, `list_agents(full=true)` shows
   `interactive: true` and a `dispatch_epoch` on a child spawned afterwards. A
   `follow_up_agent` to it while idle returns `method: "codex_queue"` or
   `"claude_mailbox"`, and its `pid` does not change.

### 6a.6 Disable

Remove the variable, or set it to anything other than `1`, and restart the
clients. `WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE=0` or
`WIN_AGENT_TEAMS_NATIVE_WAKE_CODEX=0` turns off one half only. Removing
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM` alone keeps wake and returns follow-ups to
resume. Native messages that are already unresolved keep blocking their
target until they settle or are released, even with every flag off.

## 7. Upgrading an existing install

```bash
cd /abs/path/to/agentic-coder-teams-mcp
git pull
uv sync
```

Then:

1. **Restart your client sessions** (Claude Code and/or Codex) so they relaunch
   the MCP server process with the new code. In Claude Code, `/mcp` →
   reconnect also works.
2. **Re-registration is normally NOT needed** — the launch command
   (`<venv-python> -m claude_teams.server_simple`) is stable across versions.
   Re-register only if you moved the clone/venv or a release note says the
   entry point changed.
3. **`install_lead_wake()` is safe to re-run** after an upgrade (idempotent);
   do so if release notes mention changes to the wake hook so the settings
   file picks up the new hook command.
4. If you maintain orchestration prompts/skills that drive this server, check
   [AGENT_UPGRADE_NOTES.md](AGENT_UPGRADE_NOTES.md) for instruction-level
   changes (e.g. the watch recipe replacing tight polling).

## 8. Troubleshooting

### "win-agent-teams doesn't show up in Claude Code"

Work down this list — it covers every failure mode we've seen in the field:

1. **The venv doesn't exist.** Run the sanity-check command from step 2. If it
   errors, you skipped `uv sync` (or the clone moved). Nothing downstream can
   work until this runs cleanly.
2. **Registered in the wrong scope / wrong directory.** `claude mcp list`
   shows what Claude Code actually sees *from your current directory*. A
   project `.mcp.json` only applies when you launch `claude` from that
   project's root; a `local`-scope add only applies to the project where you
   ran it. When in doubt, re-add with `--scope user` (section 3.1).
3. **Project-scope approval was declined.** Project `.mcp.json` servers need a
   one-time interactive approval. Run `claude mcp reset-project-choices` in the
   project, restart `claude`, and approve the prompt.
4. **Invalid JSON on Windows.** Single backslashes in `.mcp.json` paths break
   the file silently. Use `\\` (or forward slashes, which Windows Python
   accepts). Validate with `python -m json.tool .mcp.json`.
5. **Server listed but "failed to connect".** Run `claude --debug` and check
   the MCP log output, or run the launch command by hand and read the stderr.
   Typical causes: wrong Python (bare `python` instead of the venv path),
   Python < 3.12, or a half-synced venv (`uv sync` again).
6. **Old session.** MCP config is read at session start — restart `claude`
   after any registration change.

### "Codex installed it fine but Claude Code can't" 

That asymmetry is expected and is exactly what this guide fixes: Codex reads
one global `~/.codex/config.toml`, while Claude Code resolves three scopes plus
a per-project approval gate. Use the user-scope `claude mcp add` command
(section 3.1) and the asymmetry disappears.

### "The lead never wakes up"

1. Confirm the hook is present in the settings file `install_lead_wake`
   reported, and that you restarted the `claude` session afterwards.
2. Confirm `WIN_AGENT_TEAMS_LEAD_WAKE` is not set to `0` in the lead's
   environment. The spawn path deliberately sets it to `0` for ordinary
   spawned Claude agents; a nested lead must be spawned with
   `enable_spawned_lead_wake=true`. The internal
   `WIN_AGENT_TEAMS_LEAD_WAKE_BASELINE` preserves an operator-level kill switch
   through deeper nesting.
3. Scope mismatch: with `scope="project"` the hook lands in the settings of
   the directory the MCP **server** runs in — normally the project where you
   started `claude`. If you registered the server with an explicit `cwd`
   pointing elsewhere, the file lands there instead; use `scope="user"` or fix
   the registration.
4. Check the tool output's `reader` field matches the lead's identity
   (`team-lead` for a human-started lead).

### "No Python at '…\AppData\Roaming\uv\python\…'" (Windows, Codex Desktop shows no tools)

Claude Desktop on Windows is an MSIX-packaged app. When it runs `uv sync` or
`uv python install` (for example, when you ask Claude Desktop to install this
server), the writes to `AppData\Roaming` are redirected into the package's
private `AppData\Local\Packages\Claude_…` store. Claude sees the interpreter
there, but other programs do not. Codex Desktop then fails to start the venv:
its MCP handshake closes immediately, and the conversation has no
win-agent-teams tools. Running the section 2 sanity check from a **regular**
PowerShell window, outside Claude, shows the same `No Python at` error.

Fix: install the interpreter outside `AppData`, then rebuild the venv.

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$HOME\.local\share\uv\python"
uv python install 3.12
cd C:\abs\path\to\agentic-coder-teams-mcp
Remove-Item -Recurse -Force .venv
uv sync
```

The simplest fix is to run `uv sync` from a regular terminal, not from inside
Claude Desktop. To avoid the problem for good, set `UV_PYTHON_INSTALL_DIR` as a
user environment variable.

### Spawned agents open no visible window (Linux)

The Linux launcher probes common terminal emulators; force one with
`WIN_AGENT_TEAMS_LINUX_TERMINAL`, or use tmux mode with
`WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux`. See the README "Spawning" section.
