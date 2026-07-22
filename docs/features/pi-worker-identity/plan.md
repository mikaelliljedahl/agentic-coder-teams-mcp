# Plan: Fix Pi worker identity clobbering (session hijack)

> `docs/` is gitignored. Commit this file with `git add -f docs/features/pi-worker-identity/plan.md`.

## Problem statement

A spawned **pi** worker's `win-agent-teams` MCP server process starts with an
**empty** `AGENT_NAME` / `AGENT_SESSION_ID`, so its module-level `IDENTITY`
falls back to `team-lead`. Consequences observed:

1. A pi worker's `send_message` was recorded with `from: team-lead` (it wrote
   `IDENTITY`, which was wrongly `team-lead`).
2. With an empty `AGENT_SESSION_ID`, `_recover_session_id` produced a
   `recovery_hint` / `recoverable_sessions` nudge. The pi worker followed it and
   called `resume_session(<lead session id>)`, **hijacking the root lead's
   session** — breaking the lead's session ownership and its wake watcher.

Claude Code workers are unaffected because they receive a per-agent
`--mcp-config` file with **literal** `AGENT_*` values.

## Verified root cause (more precise than the original hypothesis)

The original hypothesis was "`${AGENT_*}` interpolation yields `""` because the
adapter's env lacks `AGENT_*`". That is **not** the mechanism. The pi-mcp-adapter
runs in-process with pi, and the spawned MCP server subprocess *does* inherit
pi's `process.env` (which `PiBackend.build_env` populates with the correct
`AGENT_NAME`). The real cause is the adapter's **config-source merge order**
combined with the repo `.mcp.json`'s hardcoded empty `env` block.

Evidence (pi-mcp-adapter, resolved from the installed package at
`~/.pi` / `node_modules/@earendil-works/pi-coding-agent`, decompiled bundles in
`/tmp/jiti/pi-mcp-adapter-*.mjs`):

- **Config sources, merged low→high precedence** (`config.mjs`
  `getConfigSources` + `loadMcpConfig`): iterated in order and merged with
  `mergeConfigs`, where **later sources win**:
  1. `~/.config/mcp/mcp.json`   (shared-global)
  2. `~/.pi/agent/mcp.json`     (pi-global; overridable via `--mcp-config`)
  3. `<cwd>/.mcp.json`          (shared-project)
  4. `<cwd>/.pi/mcp.json`       (pi-project — **highest precedence**)
  So the README's "precedence 1→4" list is *read order*; effective override
  precedence is the **reverse**.
- **Server-object merge is shallow at the server-entry level**
  (`config.mjs` `mergeServerMaps`: `merged[name] = {...base[name], ...next[name]}`).
  Because `env` is a single key of the server object, a later source that has an
  `env` key **replaces the entire env map** of an earlier source.
- **The env the server subprocess receives** (`server-manager.mjs`
  `resolveEnv`, line ~476): `{...process.env, ...interpolateEnvRecord(definition.env)}`
  — pi's full `process.env` (which includes the correct `AGENT_NAME` and
  `WIN_AGENT_TEAMS_SESSION_DIR`), then **overridden** by the merged config
  `env`.
- **`interpolateEnvVars`** (`utils.mjs` line ~62): `${VAR}` → `process.env[VAR] ?? ""`;
  a literal `""` stays `""`.

The repo `.mcp.json` (`/home/mikael/code/agentic-coder-teams-mcp/.mcp.json`,
tracked, not gitignored) declares:

```json
"win-agent-teams": {
  ...,
  "env": { "AGENT_NAME": "", "AGENT_SESSION_ID": "", "AGENT_PARENT_NAME": "" },
  ...
}
```

For a spawned pi worker whose cwd is the repo root, `.mcp.json` (source #3) is
merged **after** `~/.pi/agent/mcp.json` (source #2). Its explicit empty `env`
**replaces** the whole env map, so the server subprocess ends up with
`AGENT_NAME=""` — even though pi's `process.env` carried the correct value.
`IDENTITY` then falls back to `team-lead`, and the empty `AGENT_SESSION_ID`
triggers the recovery nudge that leads to the hijack.

Note: `WIN_AGENT_TEAMS_SESSION_DIR` is **not** present in any `mcp.json` `env`
block, so it is never clobbered — it survives from `process.env` into the server
subprocess. This makes it a **reliable signal that the process is a spawned
subagent** (used by the defense-in-depth guards below).

## Current behavior — verified citations

All line numbers are `src/claude_teams/server_simple.py` unless noted.

- `_AGENT_NAME` / `_AGENT_SESSION_ID` / `_AGENT_PARENT_NAME` read from env
  (lines 61-63); `IDENTITY = _AGENT_NAME if _AGENT_NAME else ROOT_LEAD_NAME`
  (line 70) with `ROOT_LEAD_NAME = "team-lead"` (line 69). This is the silent
  fallback that masquerades a child as the lead.
- `_write_mcp_config(session_id, agent_name, parent_name)` writes a per-agent
  file `<session_dir>/mcp/<agent>.mcp.json` with **literal** `AGENT_SESSION_ID`
  / `AGENT_NAME` / `AGENT_PARENT_NAME` (lines 1169-1186). Used by Claude via
  `--mcp-config`. It is already called for **every** backend at spawn (lines
  1405 and 1836) and its path is placed in `extra["mcp_config_path"]` (line
  1409/…). **Pi's `build_command` currently ignores `mcp_config_path`.**
- `_ensure_pi_mcp_config()` writes `~/.pi/agent/mcp.json` with
  `${AGENT_*}` interpolation (lines 1211-1249); called from `_hook_extra`'s pi
  branch (line 1305), *before* the `WIN_AGENT_TEAMS_STATE_HOOKS=0` kill switch
  (line 1306).
- `_recover_session_id` (lines 572-616): returns early on non-empty
  `_AGENT_SESSION_ID` (lines 584-585); otherwise, on ambiguity, sets
  `_pending_recovery = {"recoverable_sessions": ..., "recovery_hint": ...}`
  (lines 609-615). `_recovery_note` / `_annotate` (lines 619-641) surface it on
  dict tool results.
- `send_message` writes `"from": IDENTITY` into the recipient inbox (line 1500).
- `resume_session` (lines 1986-2031): rebinds the module `_session_id` to any
  UUID-shaped session under `_SESSION_BASE` that has an `agents.json`
  (lines 2012-2019) — no check that the caller is entitled to that session. This
  is what allowed the child to adopt the lead's session.
- `backends/pi.py`: `build_command` (lines 281-295) and `build_resume_command`
  (lines 297-322) assemble the pi argv (launcher, headless flags, permission
  args, `--session-dir`, `--session-id`, model args, `-e` extensions, prompt).
  `build_env` (lines 390-406) sets `AGENT_NAME` / `AGENT_SESSION_ID` /
  `AGENT_PARENT_NAME` and `WIN_AGENT_TEAMS_SESSION_DIR` in pi's process env.

## Key design question — RESOLVED

**How does each spawned pi worker get literal `AGENT_*` values, without breaking
the root lead, the adapter, or the Claude path?**

Answer — two independent facts drive the mechanism:

1. **pi-mcp-adapter registers a `--mcp-config <path>` flag** (`index.mjs`
   line ~85: `pi.registerFlag("mcp-config", {type:"string"})`), and at runtime
   `initializeMcp` reads it: `loadMcpConfig(pi.getFlag("mcp-config"), ctx.cwd)`
   (`init.mjs` line ~37). When set, it replaces the **pi-global** source's read
   path (source #2) with our file. `pi.getFlag` returns a value **only because
   the adapter itself registered the flag** — so `--mcp-config` is safe to pass
   (pi will not reject it as unknown, and the adapter consumes it).

2. **`--mcp-config` alone is insufficient**: sources #3 (`.mcp.json`) and #4
   (`.pi/mcp.json`) are still merged **after** it and, if they carry an `env`
   block, replace ours. The repo `.mcp.json` carries an empty `env` block — that
   is the actual clobber. So the fix **must also remove that env block**.

**Chosen mechanism (belt-and-suspenders, matches the "literal, not
interpolation" directive):**

- Point each spawned pi at a **per-agent literal** MCP config via
  `--mcp-config <session_dir>/mcp/<agent>.pi.mcp.json` (mirrors Claude's
  `--mcp-config`). This is per-agent by construction (keyed by session-dir +
  agent name), so **no cwd collision** and **no repo-tree pollution** — unlike
  writing `<cwd>/.pi/mcp.json`, which would collide when multiple agents share
  the default cwd (repo root; `agent_cwd = cwd.strip() or Path.cwd()`, line
  1407).
- **Remove the `env` block from the repo `.mcp.json` `win-agent-teams` entry**
  so nothing at higher precedence (source #3) replaces our literal env. This is
  safe for the root lead: with `AGENT_*` unset, the server's `IDENTITY` defaults
  to `team-lead` (line 70) exactly as intended, and the root lead's session
  binding/wake are unaffected.

Why not just fix `_ensure_pi_mcp_config`'s interpolation? Because interpolation
was never the failing link (see root cause); the empty `env` in `.mcp.json`
would still win. Removing that env block is required either way; giving pi a
literal per-agent file is the requested robust primary fix.

### Rejected alternatives

- **Write `<cwd>/.pi/mcp.json` (highest precedence, source #4).** Collides
  across agents that share the default cwd (repo root) and pollutes the working
  tree. Rejected.
- **Only remove the `.mcp.json` env block, rely on `~/.pi/agent/mcp.json`
  `${AGENT_*}` interpolation from inherited `process.env`.** Correct in
  practice, but keeps identity on `${}` interpolation, which the task explicitly
  wants to stop depending on. Kept only as a fallback discussion.

## Proposed design

### Primary fix

1. **Per-agent literal pi MCP config.** Add
   `_write_pi_mcp_config(session_id, agent_name, parent_name) -> Path` mirroring
   `_write_mcp_config` (lines 1169-1186), writing
   `<session_dir>/mcp/<agent>.pi.mcp.json` with literal
   `AGENT_SESSION_ID`/`AGENT_NAME`/`AGENT_PARENT_NAME`, plus the pi-specific keys
   already used in `_ensure_pi_mcp_config`: `CLAUDE_TEAMS_PERMISSION_MODE:
   "bypass"` and top-level `directTools: true`. Omit `cwd` (the adapter defaults
   the server's cwd to pi's cwd = the agent cwd).
   - Wire it in `_hook_extra`'s pi branch, returning
     `{"pi_mcp_config_path": str(path)}`, and write it **before** the
     `WIN_AGENT_TEAMS_STATE_HOOKS=0` early return (line 1306) so identity is
     never gated behind the state-hooks kill switch.
   - (Reusing the existing `extra["mcp_config_path"]` is possible but that file
     lacks the pi-specific `directTools`/permission keys; a dedicated writer is
     cleaner and keeps the Claude file untouched.)
2. **Pass `--mcp-config` to pi.** In `PiBackend.build_command` and
   `build_resume_command`, when `extra["pi_mcp_config_path"]` is present, append
   `["--mcp-config", <path>]`. Add it near the `--session-dir`/`--session-id`
   args.
3. **Remove the clobbering env block from the repo `.mcp.json`.** Delete the
   `"env": { "AGENT_NAME": "", ... }` key from the `win-agent-teams` entry in
   `/home/mikael/code/agentic-coder-teams-mcp/.mcp.json`. Keep `command`,
   `args`, `cwd`, `lifecycle`, `directTools`.

### Defense in depth

4. **Do not silently masquerade as `team-lead` when the process is clearly a
   spawned subagent.** Replace the bare `IDENTITY` fallback (line 70) with logic
   that treats an **empty `AGENT_NAME` + spawned-subagent signal** as an
   unresolved identity rather than the lead. Signal:
   `os.environ.get("WIN_AGENT_TEAMS_SESSION_DIR")` is set (verified reliable and
   never clobbered). Behavior:
   - Root lead (no `WIN_AGENT_TEAMS_SESSION_DIR`, empty `AGENT_NAME`) →
     `IDENTITY = "team-lead"` **unchanged**.
   - Spawned subagent with empty `AGENT_NAME` → mark identity unresolved (e.g.
     module flag `_IDENTITY_UNRESOLVED = True`, and set `IDENTITY` to a sentinel
     that is never a valid inbox/lead name). Guard the identity-bearing tools:
     `send_message` refuses (`{"success": False, "reason":
     "identity_unresolved"}`) instead of writing `from: team-lead` (line 1500);
     `resume_session` refuses (so a mis-identified child can never adopt a
     session). Emit a loud `logger.error`. Normally identity resolves correctly
     after the primary fix, so this guard is a safety net that never fires in the
     healthy path.
   - Claude workers always have a literal `AGENT_NAME` → guard never triggers.
5. **Suppress recovery nudges for a child process.** In `_recover_session_id`,
   when the spawned-subagent signal is present but `_AGENT_SESSION_ID` is empty,
   do **not** populate `_pending_recovery` with `recoverable_sessions` /
   `recovery_hint` (return `""` with `_pending_recovery = {}`). A child must
   never be invited to adopt a workspace session. (With the primary fix,
   `_AGENT_SESSION_ID` is always set for a child, so `_recover_session_id`
   returns early at line 584 and this branch is not reached — this is a
   belt-and-suspenders guard against the exact hijack path.)

## Exact files to change

- `/home/mikael/code/agentic-coder-teams-mcp/.mcp.json` — remove the `env` block
  from the `win-agent-teams` server entry (fix #3).
- `src/claude_teams/server_simple.py`
  - add `_write_pi_mcp_config` near `_write_mcp_config` (~line 1186) (#1);
  - wire it into `_hook_extra` pi branch before the state-hooks kill switch
    (~lines 1304-1315) (#1);
  - harden `IDENTITY` derivation (~lines 61-70), add `_IDENTITY_UNRESOLVED`
    signal helper (#4);
  - guard `send_message` (~line 1494-1500) and `resume_session`
    (~lines 2000-2018) on unresolved identity (#4);
  - suppress recovery nudge for children in `_recover_session_id`
    (~lines 596-616) (#5).
- `src/claude_teams/backends/pi.py` — append `--mcp-config` in `build_command`
  (~line 287) and `build_resume_command` (~line 314) (#2).
- Tests under `tests/` (see test plan).

## Cross-platform notes (Windows)

- The per-agent config path is created with `pathlib` and JSON-serialized (no
  shell), so no quoting concerns for the file contents.
- `--mcp-config <path>` is passed as a **separate argv token** (not embedded in
  a string), and pi is launched via direct `node <cli.js>` (see
  `PiBackend._launcher`, lines 226-241), bypassing the `pi.cmd`/`cmd.exe` shim,
  so a Windows path with spaces or backslashes survives verbatim through
  `CreateProcess` / `CommandLineToArgvW`. If the rare shim-fallback path is hit
  (`_launches_via_shim`, line 249), the argv still passes as a discrete token;
  the path contains no newline or `< > | & ^`, so the shim does not mangle it.
- Session-dir paths already round-trip on Windows elsewhere in the codebase
  (`_pi_session_dir`, lines 324-334); reuse the same `Path` construction.

## Risks

- **Root lead regression (highest risk).** Removing `.mcp.json`'s `env` and the
  `IDENTITY` guard must keep the human-launched root lead as `team-lead`.
  Mitigation: the guard keys off `WIN_AGENT_TEAMS_SESSION_DIR`, which the root
  lead never has; with `AGENT_*` unset the server defaults to `team-lead`.
  Covered by a dedicated test (root-lead env → `team-lead`, guard inactive).
- **Claude path regression.** No change to `_write_mcp_config` or the Claude
  branch; Claude always sets a literal `AGENT_NAME`, so the guard cannot fire.
  Covered by a regression test.
- **Other users' `.mcp.json`.** External consumers copy the README `.mcp.json`
  snippet (README lines ~32). Removing the `env` block from the repo file should
  be reflected in the README/ADDING-A-BACKEND docs so downstream configs do not
  reintroduce the clobber. (Doc update is in scope for the PR; note in
  implementation.md.)
- **`~/.config/mcp/mcp.json` (source #1).** If a user has a global entry it is
  merged first (lowest precedence) and cannot clobber our per-agent `env`; no
  action needed, but note it in implementation.md.
- **Adapter version drift.** The `--mcp-config` flag and merge order are
  observed in the currently installed pi-mcp-adapter. Pin the observed version
  in implementation.md and add a smoke test so a future adapter change is caught.

## Test plan (red first, then green)

Focused unit tests to write **failing first**:

1. **Pi per-agent MCP config carries literal `AGENT_*`.**
   `test_backends/test_pi.py` (or a server test): call the new
   `_write_pi_mcp_config(session_id, "worker-1", "team-lead")` and assert the
   written JSON's `mcpServers["win-agent-teams"].env` has literal
   `AGENT_NAME == "worker-1"`, `AGENT_SESSION_ID == session_id`,
   `AGENT_PARENT_NAME == "team-lead"` and **no** `${` substring anywhere in the
   file. Assert `directTools` / `CLAUDE_TEAMS_PERMISSION_MODE` present.
2. **Pi launch passes `--mcp-config` pointing at that file.**
   `test_backends/test_pi.py`: build a `SpawnRequest` with
   `extra["pi_mcp_config_path"]` set and assert both `build_command` and
   `build_resume_command` include `"--mcp-config"` immediately followed by that
   path. Also assert it is a discrete argv token (no shell string).
3. **`_hook_extra` writes the pi config even with state hooks disabled.**
   Set `WIN_AGENT_TEAMS_STATE_HOOKS=0`, call `_hook_extra(sid, "worker-1",
   "pi")`, assert `pi_mcp_config_path` is present and the file exists.
4. **IDENTITY does not silently become `team-lead` for a spawned-subagent env.**
   Simulate empty `AGENT_NAME` + `WIN_AGENT_TEAMS_SESSION_DIR` set (reload the
   module under patched `os.environ`, or factor the derivation into a testable
   `_resolve_identity(environ)` helper). Assert identity is flagged unresolved
   (not `"team-lead"`), and that `send_message` returns
   `{"success": False, "reason": "identity_unresolved"}` and `resume_session`
   refuses.
5. **Root lead still resolves to `team-lead`.** Empty `AGENT_NAME` and **no**
   `WIN_AGENT_TEAMS_SESSION_DIR` → identity `"team-lead"`, guard inactive,
   `send_message`/`resume_session` behave as today.
6. **`recovery_hint` suppressed for children.** With the spawned-subagent signal
   present and empty `AGENT_SESSION_ID`, assert `_recover_session_id()` returns
   `""` and `_pending_recovery` has **no** `recovery_hint` /
   `recoverable_sessions`. With no signal (root lead, ambiguous history), assert
   the nudge is still produced (unchanged behavior).

Gates: focused tests, then the full `pytest` suite and `ruff check` across the
whole repo, on Linux (per CLAUDE.md / MEMORY). A pi live smoke (spawn a pi
worker, have it `send_message`, assert the recorded `from` is its own name and
no `resume_session` on the lead's session) is the acceptance check for the PR.

## Open decisions for the user before implementation

- **Guard failure mode (#4):** refuse-at-tool-call (recommended — server still
  starts, tools return structured errors) vs. hard-fail the server process at
  startup (louder but crashes the MCP server). Recommend refuse-at-tool-call.
- **README/ADDING-A-BACKEND update:** confirm the doc snippet should drop the
  `env` block too (recommended, to prevent downstream reintroduction).
