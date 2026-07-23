# Installing and upgrading win-agent-teams

This guide covers a **fresh install** of the `win-agent-teams` MCP server for
Claude Code and Codex on Linux and Windows, plus **upgrading** an existing
setup, wiring the **lead-wake `Stop` hook**, and **troubleshooting** the common
failure modes (especially "the server never shows up in Claude Code").

Setup is always **two parts**:

1. **Register the MCP server** with your client (Claude Code and/or Codex) so
   the tools (`spawn_agent`, `send_message`, `read_messages`, …) exist.
2. **Optionally install the lead-wake `Stop` hook** for a top-level Claude Code
   lead — one `install_lead_wake()` tool call. Spawned agents get it
   automatically; you only ever run this for a lead you started yourself.

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

If any of these fail, jump to [Troubleshooting](#7-troubleshooting).

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
| **Server-spawned** Claude Code agent (any nesting level) | **Nothing.** The spawn path auto-wires the wake hook as a second `Stop` matcher group in the agent's per-agent `hooks-<name>.settings.json` (passed via `--settings`). |
| **Top-level lead you start yourself** (an interactive `claude` in your repo) | Run the `install_lead_wake` MCP tool **once** (see below). |
| Codex or Pi lead | Not applicable — the hook is Claude Code-specific. Codex leads use the bounded foreground `watch` loop instead (see README "Coordinating without polling"). |

### 6.1 Install

In the lead's Claude Code session (with the MCP server registered), ask it to
call:

```
install_lead_wake()
```

- Default scope (`scope="project"`) writes the `Stop` wake group into
  **`.claude/settings.json` in the lead's working directory** — i.e. the
  project where you started `claude`. This is the right choice almost always.
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
   environment.
3. Scope mismatch: with `scope="project"` the hook lands in the settings of
   the directory the MCP **server** runs in — normally the project where you
   started `claude`. If you registered the server with an explicit `cwd`
   pointing elsewhere, the file lands there instead; use `scope="user"` or fix
   the registration.
4. Check the tool output's `reader` field matches the lead's identity
   (`team-lead` for a human-started lead).

### Spawned agents open no visible window (Linux)

The Linux launcher probes common terminal emulators; force one with
`WIN_AGENT_TEAMS_LINUX_TERMINAL`, or use tmux mode with
`WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux`. See the README "Spawning" section.
