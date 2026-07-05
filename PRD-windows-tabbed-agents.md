# PRD: Group agent consoles into tabs per Team Lead (Windows)

**Status:** Draft
**Date:** 2026-07-05
**Repo:** `win-agent-teams-mcp` (`agentic-coder-teams-mcp/`)
**Target platform:** Windows 11 with Windows Terminal (`wt.exe`)

---

## 1. Background & problem

When the MCP server spawns interactive agents (Claude Code / Codex) on Windows, each agent opens its **own separate console window**. With several agents running under one team lead, these windows stack on top of each other and clutter the desktop. The lead wants the agents **grouped as tabs inside a single Windows Terminal window** — one window per team lead — while keeping the interactive console (the live agent UI must stay visible and usable).

### 1.1 Root cause (verified in code)

The stacked windows come from the **interactive spawn path**, not from the log tail:

- `WindowsProcessManager.spawn_process` adds `creationflags |= CREATE_NEW_CONSOLE` when `interactive_console` is true — `process_manager.py:363–376`.
- `_should_use_interactive_console` returns `True` on Windows by default unless `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` — `process_manager.py:625–636`.
- Both backends are interactive: `ClaudeCodeBackend.is_interactive → True` (`claude_code.py:32`) and `CodexBackend.is_interactive → True` (`codex.py:40`).

So every interactive agent gets `CREATE_NEW_CONSOLE` → its own classic console window → stacked windows.

### 1.2 The Windows mechanism already exists in the codebase

`_open_windows_terminal_tail` (`process_manager.py:646–676`) already invokes:

```
wt.exe -w 0 nt --title "<agent>@<team>" -- powershell -NoExit -Command Get-Content ... -Wait
```

`wt -w <id> nt` is exactly the mechanism that groups tabs into one named window. But this helper runs only on the **non-interactive** branch (`process_manager.py:424–425`) to tail a log — it does not run in the default interactive mode that produces the windows. The building block is present; it just needs to drive the *agent process itself* instead of `CREATE_NEW_CONSOLE`.

---

## 2. Goals

1. On Windows, launch interactive agents as **tabs inside a shared Windows Terminal window**, grouped per team lead.
2. Each tab is titled with the agent name (`<agent>@<team>`).
3. Preserve the **interactive console** experience — the agent's live TUI runs in the tab, fully attached to a real console.
4. **No regression in lifecycle management**: `health_check`, `kill_agent`, `owns_process`/`creation_token`, restart recovery, and `list_agents`/`check_agent` must remain correct.
5. Opt-in and backward compatible — the current one-window-per-agent behavior stays the default.

## 3. Non-goals

- No changes to the Linux / tmux launchers (they already have window/pane/tab modes).
- No custom GUI or window manager outside Windows Terminal.
- No change to the lead↔agent messaging/inbox protocol.
- Not solving codex stdout log capture (see §6, out of scope but noted).

---

## 4. Proposed solution

Add a **tabs mode** to `WindowsProcessManager` that, instead of `CREATE_NEW_CONSOLE`, starts the agent through Windows Terminal:

```
wt.exe -w <window-id> nt --title "<agent>@<team>" -- <launcher-shell> <agent-cmd...>
```

- `-w <window-id>` — all agents sharing the id land in the **same** window.
- `nt` — new tab.
- `--title "<agent>@<team>"` — tab title.
- `-- <launcher-shell> <agent-cmd...>` — a thin shell wrapper that first records the real agent PID (see §5), then `exec`s the agent so its interactive UI owns the tab's console.

### 4.1 Activation flag

Introduce `WIN_AGENT_TEAMS_WINDOWS_TABS` (default `0`). When `1` **and** the run is interactive **and** `wt.exe` is on PATH, use tabs mode. This mirrors the Linux launcher-selection convention (`WIN_AGENT_TEAMS_LINUX_LAUNCHER`).

### 4.2 Interaction with existing flags

Tabs mode is a *variant* of interactive mode, so precedence is:

1. `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` → non-interactive: capture stdout/stderr to the log file, no window/tab. (Unchanged; highest priority — user explicitly opted out of a visible console.)
2. Else if `WIN_AGENT_TEAMS_WINDOWS_TABS=1` and `wt.exe` present → **tabs mode** (this PRD).
3. Else → current `CREATE_NEW_CONSOLE` per-agent window (default).

### 4.3 Window grouping id

Use `request.team_name` — already in scope at spawn (used for the tab title and log directory today, `process_manager.py:659`). One window per team ≈ one window per lead in practice.

**Constraint:** Windows Terminal treats a purely numeric `-w` value as a *window ID* (e.g. `-w 0` = most-recent window). `team_name` is validated as `^[A-Za-z0-9_-]+$` and may be all digits, so it must be prefixed to force name semantics — e.g. `-w "wt-team-<team_name>"`. Whatever prefix is chosen must be applied consistently so every agent in the team resolves to the same window.

*(Alternative grouping key — one window per orchestrating lead via `WIN_AGENT_TEAMS_PARENT_ID` — is rejected for v1: `PARENT_ID` is optional (falls back to `os.getppid()`), is only used for session binding in `server_simple.py`, and is not threaded into `process_manager`. See §9.)*

---

## 5. Critical design point: launcher PID vs. agent PID

The entire lifecycle assumes `handle = str(process.pid)` is the **agent's** PID (`process_manager.py:412`). `wt.exe` breaks that assumption: it is a launcher. When a Windows Terminal window already exists, `wt.exe` forwards the new-tab request to the existing terminal process and the invoked `wt.exe` process **exits immediately**. If we naively kept `handle = wt.pid`:

- `health_check` (`process_manager.py:465–484`) reads `process.poll()`, sees `wt.exe` already gone, and reports the agent as `exited (0)` while it is actually running.
- `kill_process`, `graceful_shutdown`, `owns_process`, `creation_token`, and restart recovery all point at the dead/wrong PID.

**This must be solved before tabs mode is viable.** The fix already exists in this codebase for the analogous Linux case: `LinuxTerminalProcessManager` uses a **sidecar `.pid` file** — the launched shell writes its own `$$` to a `.pid` file before exec-ing the agent, and the manager reads it via `resolve_agent_pid` / `_agent_pid_health` (`process_manager.py:1148–1164`, `1243–1266`, `_with_agent_pid_file` at `1243`).

**Requirement:** the Windows tabs path must adopt the same pattern:

1. Wrap the agent command in a shell (`powershell`/`cmd`) that first writes its own PID to `<log_path>.pid`, then `exec`s / starts the agent so the agent's console UI is what the user sees in the tab.
2. Track both the launcher handle and the sidecar-derived agent PID in a dedicated `ProcessInfo` variant (mirror `LinuxTerminalProcessInfo`).
3. Make `health_check`, `kill_process`, `graceful_shutdown`, `owns_process`, `resolve_agent_pid`, and `creation_token` resolve to the **agent** PID from the sidecar, not the `wt.exe` PID.
4. Persist enough (the sidecar path is deterministic from the log path) that restart recovery can re-derive the agent PID after an MCP-server restart — as the Linux path already does.

> Note the PID-reuse-safety machinery (`creation_token`, `owns_process`, `_pid_health_with_token`) must gate on the **agent** PID's creation token, not the launcher's, to keep the fail-closed guarantees intact.

---

## 6. Other details to handle

- **Log capture.** Interactive claude-code already gets `--debug-file` injected so logs still reach the file (`process_manager.py:368–369`, `_with_debug_file`). Codex does **not** get `--debug-file`, so codex output inside a tab is not captured to the log file — identical to today's `CREATE_NEW_CONSOLE` codex, so **no regression**, but `capture()`/tail for codex-in-tab stays empty. Out of scope here; flag as a known gap.
- **stdin / interactivity.** Interactive agents inherit the console (`stdin=None/stdout=None`). Inside a WT tab the agent gets a real console → the interactive UI works. `send()` already no-ops in interactive mode (no stdin pipe), and both backends use native messaging rather than stdin — so no behavioral change.
- **Fallback.** If `wt.exe` is not on PATH, fall back to the current `CREATE_NEW_CONSOLE` path (never crash). `shutil.which("wt.exe")` is already used in the codebase.
- **Killing does not close the tab.** `taskkill /T /F` on the agent PID (from the sidecar) kills the agent, but the WT tab does not auto-close unless its shell exits. Cosmetic; acceptable for v1. Optionally launch the wrapper shell so it exits when the agent exits.
- **`CREATE_BREAKAWAY_FROM_JOB`.** The existing breakaway logic (`_popen`, `process_manager.py:428–463`) exists so agents survive the server's Job Object. `wt.exe` runs in its own process tree anyway; verify the spawned launcher still breaks away as intended (or is intentionally detached) so the tabs outlive the MCP server per the project's detach-by-default policy.

---

## 7. Acceptance criteria

1. With `WIN_AGENT_TEAMS_WINDOWS_TABS=1`, two agents in the same team open as **two tabs in one** Windows Terminal window, each with the correct tab title, each showing a live interactive agent UI.
2. Agents in different teams open in **separate** windows.
3. `list_agents` / `check_agent` report `running` while the agent runs and `exited` only when the **agent** (not `wt.exe`) actually exits.
4. `kill_agent` terminates the correct (agent) process.
5. Restart recovery re-derives the agent PID and continues to report correct liveness after an MCP-server restart.
6. With the flag off (default), behavior is unchanged.
7. With `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0`, output is captured to the log file and no window/tab opens (unchanged).
8. Without `wt.exe` on PATH, it falls back to the current per-agent console with no crash.
9. Test suite green on **both** Windows and Linux (Linux paths untouched).

---

## 8. Implementation sketch (where the code goes)

- All changes are in `src/claude_teams/backends/process_manager.py`, inside/around `WindowsProcessManager` — this is the module that owns Windows spawning. `process_base.py` and the backends are unchanged (they already pass `is_interactive` through `spawn_process`).
- Add a `WindowsTerminalProcessInfo` dataclass (sidecar `.pid` path + launcher `Popen`), analogous to `LinuxTerminalProcessInfo`.
- Add a `_build_windows_terminal_tab_command(...)` that produces the `wt.exe -w <prefixed-team> nt --title ... -- <shell-that-writes-pid-then-execs-agent>` argv.
- Branch in `spawn_process`: interactive + `WIN_AGENT_TEAMS_WINDOWS_TABS=1` + `wt.exe` present → tabs path; otherwise the existing branches.
- Route `health_check` / `kill_process` / `graceful_shutdown` / `owns_process` / `resolve_agent_pid` / `creation_token` to the sidecar agent PID for tab-launched agents.
- Unit tests mirroring `tests/test_backends/test_process_manager_windows.py`; keep Linux tests unaffected.

---

## 9. Open question

- **Grouping key:** one window per **team** (`team_name`, in scope, simplest — recommended for v1) vs. one per **orchestrating lead** (`WIN_AGENT_TEAMS_PARENT_ID`, requires threading it through `SpawnRequest`). Recommendation: ship `team_name` (prefixed) in v1; revisit a lead-level key only if a single lead routinely runs multiple teams that should share one window.

---

## 10. Addendum — as shipped

The implemented behavior refines §4 in three ways (all in `WindowsProcessManager`, verified by unit tests + a real `wt.exe` smoke and a headless end-to-end run through a second Claude Code CLI):

1. **Tabs are the default, not opt-in.** Interactive agents on Windows go to a per-team Windows Terminal tab whenever `wt.exe` is on PATH — no activation flag. The classic `CREATE_NEW_CONSOLE` window remains the fallback when `wt.exe` is absent, and there is a single escape hatch `WIN_AGENT_TEAMS_NO_WT_TABS=1` to force the classic path. (`WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` still disables the visible console entirely.)

2. **Grouping key = `team_name`, prefixed.** Window id is `wt-team-<team_name>` (the prefix avoids `wt` misreading an all-numeric team as a numeric window id). The `PARENT_ID`/lead-level option in §9 was not needed for v1.

3. **Tab title is pinned; tabs auto-close on exit *and* kill.** The `nt` command passes `--suppressApplicationTitle` so the agent CLI can't overwrite the `<agent>@<team>` tab title. The wrapper shell always ends with `exit 0`, and `kill_agent` terminates the **agent subtree** (children of the wrapper shell) rather than force-killing the shell — so the shell returns and exits 0, letting Windows Terminal's graceful `closeOnExit` remove the tab, matching the old console's auto-close instead of leaving a dead "[process exited with code 1]" tab.

The launcher-PID risk in §5 was resolved exactly as designed: the wrapper writes its own PID to a sidecar, that PID (not `wt.exe`'s) is the handle, and the server captures its creation token — so health/kill/ownership/restart-recovery all work unchanged. The codex log-capture gap (§6) remains out of scope.
