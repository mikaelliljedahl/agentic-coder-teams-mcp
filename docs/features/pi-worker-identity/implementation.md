# Implementation: Fix Pi worker identity clobbering (session hijack)

> `docs/` is gitignored. Commit this file with
> `git add -f docs/features/pi-worker-identity/implementation.md`.

Branch: `feature/pi-lead-autoload`. This documents the already-implemented,
green (uncommitted) fix for the pi worker identity clobber / session-hijack bug.

## Root cause

A spawned **pi** worker's `win-agent-teams` MCP server process started with an
**empty** `AGENT_NAME` / `AGENT_SESSION_ID`, so its module-level `IDENTITY` fell
back to `team-lead`. The mechanism was not `${AGENT_*}` interpolation failing —
the pi-mcp-adapter runs in-process with pi and the spawned MCP server subprocess
*does* inherit pi's `process.env` (which `PiBackend.build_env` populates with the
correct `AGENT_NAME`). The real cause is the adapter's **config-source merge
order** combined with the repo-root `.mcp.json`'s hardcoded empty `env` block:

1. The adapter merges MCP config sources **later-wins**:
   `~/.config/mcp/mcp.json` (source #1) → `~/.pi/agent/mcp.json` (source #2,
   overridable via `--mcp-config`) → `<cwd>/.mcp.json` (source #3) →
   `<cwd>/.pi/mcp.json` (source #4, highest precedence).
2. The server-entry merge is **shallow** (`merged[name] = {...base, ...next}`),
   so a later source that carries an `env` key **replaces the entire env map**.
3. For a spawned pi worker whose cwd was the repo root, `<cwd>/.mcp.json`
   (source #3) merged **after** the inherited/interpolated identity and its
   `"env": { "AGENT_NAME": "", "AGENT_SESSION_ID": "", "AGENT_PARENT_NAME": "" }`
   block wiped the correct values to `""`.
4. Empty `AGENT_NAME` → `IDENTITY` fell back to `team-lead`; empty
   `AGENT_SESSION_ID` → `_recover_session_id` emitted a `recovery_hint` /
   `recoverable_sessions` nudge. The pi worker followed the nudge and called
   `resume_session(<lead session id>)`, **hijacking the root lead's session**
   (breaking its ownership and wake watcher). A single-lead history could also
   trigger `_recover_session_id`'s **silent single-candidate auto-adopt** with no
   tool call at all — a worse, quieter variant of the same hijack.

`WIN_AGENT_TEAMS_SESSION_DIR` is set only by `PiBackend.build_env`, never appears
in any `mcp.json` `env` block, and survives from `process.env` into the server
subprocess — making it a **reliable signal that the process is a spawned
subagent**, used by the guards below.

## Final design (as implemented)

Two-part fix (primary + defense-in-depth), plus docs.

### Primary — per-agent literal `--mcp-config` for pi workers

- `server_simple._write_pi_mcp_config(session_id, agent_name, parent_name)`
  writes `<session_dir>/mcp/<agent>.pi.mcp.json` with **literal** `AGENT_*`
  values (no `${...}` interpolation), plus `CLAUDE_TEAMS_PERMISSION_MODE=bypass`
  and a **server-entry-level** `directTools: true` (per review F4). The distinct
  `.pi.mcp.json` filename can never collide with the Claude `<agent>.mcp.json`.
- `server_simple._hook_extra` (pi branch) calls it and returns
  `{"pi_mcp_config_path": ...}`. It is written **before** the
  `WIN_AGENT_TEAMS_STATE_HOOKS=0` kill switch so identity is never gated behind
  the state-hooks toggle.
- `backends/pi.PiBackend._mcp_config_args` emits `["--mcp-config", <path>]` (a
  discrete argv token) when `extra["pi_mcp_config_path"]` is present, wired into
  both `build_command` and `build_resume_command`. Passing `--mcp-config`
  redirects the pi-global config source (#2) to our literal file; the adapter
  registers the flag, so pi does not reject it.

### Defense in depth — fail loud instead of masquerading

- `server_simple._resolve_identity(environ) -> (IDENTITY, unresolved)`: a
  non-empty `AGENT_NAME` is authoritative; empty `AGENT_NAME` **with** the
  spawned-subagent signal (`WIN_AGENT_TEAMS_SESSION_DIR`) yields the sentinel
  `_UNRESOLVED_IDENTITY` (a NUL-bearing string that can never match a real inbox
  or lead name) and `unresolved=True`; empty `AGENT_NAME` with **no** signal is a
  legitimate human-launched root lead → `team-lead`. Module globals
  `IDENTITY, _IDENTITY_UNRESOLVED` are bound from this at import.
- Import-time `logger.error` fires when identity is unresolved, telling the
  operator the likely cause (an `AGENT_*` env block in a project MCP config) and
  that team tools will refuse.
- `_require_resolved_identity()` returns a structured refusal
  `{"success": False, "reason": "identity_unresolved", "hint": ...}` and is
  applied at the top of `send_message`, `read_messages`, and `resume_session`.
  `read_messages` refuses **before** any inbox read/cursor advance so a
  mis-identified child never consumes the lead's inbox (F3).
- `_recover_session_id` returns `""` **immediately** after the
  `_AGENT_SESSION_ID` early-return when identity is unresolved — before the
  binding-key match, the single-candidate auto-adopt, and the recovery nudge —
  closing the silent auto-adopt path (F2). Guard failure mode is refuse-at-tool
  -call (open decision (a)); the server still starts and returns actionable
  structured errors rather than crashing.

### Docs

- `README.md`: pi identity section rewritten to describe the two mechanisms
  (human lead → `~/.pi/agent/mcp.json` interpolation; spawned worker → per-agent
  literal `--mcp-config`) with an explicit WARNING never to put `AGENT_*` in a
  project MCP config; fixed the `"lead"` → `"team-lead"` drift in the Identity
  section (F7).
- `ADDING-A-BACKEND.md`: pi bullet updated with the per-agent `--mcp-config`
  note and the same warning (F7).

## Red/green test evidence

New/updated tests (all green):

- `tests/test_pi_worker_identity.py`
  - `test_pi_config_has_literal_identity` — literal `AGENT_*`, no `${`,
    server-entry `directTools`, `CLAUDE_TEAMS_PERMISSION_MODE=bypass`,
    `<agent>.pi.mcp.json` name. **Red** before `_write_pi_mcp_config` existed.
  - `test_pi_config_distinct_from_claude_config` — pi file distinct from and
    non-polluting to the Claude file.
  - `test_build_command_includes_mcp_config` /
    `test_build_resume_command_includes_mcp_config` — `--mcp-config` emitted as a
    discrete token pointing at the per-agent file. **Red** before
    `_mcp_config_args` wiring.
  - `test_build_command_omits_mcp_config_when_absent` — degradation/compat.
  - `test_hook_extra_writes_pi_config_with_state_hooks_off` — config written even
    with `WIN_AGENT_TEAMS_STATE_HOOKS=0`; extensions still suppressed. **Red**
    before the pre-kill-switch write.
  - `test_resolve_identity_*` (spawned-subagent unresolved / root-lead team-lead
    / named worker) — **Red** against the old bare `IDENTITY` fallback.
  - `test_send_message_refuses_when_identity_unresolved`,
    `test_read_messages_refuses_and_does_not_touch_lead_inbox` (asserts the lead
    cursor file is never written),
    `test_resume_session_refuses_when_identity_unresolved` — **Red** before the
    guards; old code would have spoofed `from:`, consumed the lead inbox, and
    adopted the lead's session.
  - `test_recover_session_id_no_silent_autoadopt_for_unresolved_child` — the
    dangerous no-tool-call path; **Red** against old auto-adopt.
  - `test_root_lead_send_message_works` /
    `test_recovery_hint_retained_for_root_lead` — regression: root lead still
    resolves to `team-lead`, tools work, nudge preserved.
- `tests/test_server_simple_guards.py` — updated
  `test_hook_extra_pi_emits_both_extension_keys` and
  `test_hook_extra_pi_state_hooks_off_disables_both` to account for the always
  -written `pi_mcp_config_path` key.

## Disposition of plan-review findings F1–F8 + open decisions

- **F1** (`.mcp.json` is gitignored, not tracked). Accepted. Fix #3 is a **local
  environment remediation only** — the repo-root `.mcp.json` env block was
  removed on this machine, but it is invisible to the PR diff and unenforceable
  by tests. The durable protection is therefore the guard (#4/#5) + the per-agent
  `--mcp-config` file (#1/#2) + the docs warning (F7), not the local edit. Any
  user hand-adding an `AGENT_*` env block re-creates the clobber and is caught
  only by the fail-loud guard, which is treated as first-class behavior (tested).
- **F2** (guard #5 missed the silent hijack paths). Accepted. `_recover_session_id`
  returns `""` immediately on unresolved identity, before the binding-key match
  and the single-candidate auto-adopt; a red-first test covers the silent
  auto-adopt path.
- **F3** (guard covered too few tools; `read_messages` reads the lead inbox).
  Accepted. Centralized `_require_resolved_identity()` applied to `send_message`,
  `read_messages` (before any read/cursor), `resume_session`, and `spawn_agent`
  (all return the structured `identity_unresolved` refusal). `follow_up_agent`
  is inert by construction (`session_not_found` before any side effect). A test
  asserts `read_messages` never touches `inbox-team-lead` / its cursor.
- **F4** (`directTools` placement). Accepted. `directTools` is written at the
  **server-entry** level, asserted by `test_pi_config_has_literal_identity`.
- **F5** (remaining `.mcp.json` keys still shallow-override). Accepted as
  documented invariant: the per-agent file only reliably owns keys that
  higher-precedence project sources do not declare. Noted in the
  `_write_pi_mcp_config` docstring.
- **F6** (session id derivable from the signal). Not adopted; refuse-at-tool
  -call was chosen (open decision (a)). Deriving the session id from the
  directory basename is a possible future enhancement but was intentionally left
  out to keep the guard simple and loud.
- **F7** (docs: README has no env block to drop; reframed). Accepted. README pi
  section + ADDING-A-BACKEND updated with the warning and the per-agent mechanism
  description; README `"lead"` → `"team-lead"` fixed.
- **F8** (adapter drift). Accepted. Environment pinned below; the live smoke test
  asserts the observable contract (spawned pi worker's `send_message` records its
  own name), not adapter internals.
- **Open decision (a) — guard failure mode**: refuse-at-tool-call (recommended
  by both plan and review). Server still starts; tools return structured errors;
  import-time `logger.error` tells the operator.
- **Open decision (b) — README env block**: reframed per F7 — there was no env
  block in the shipped README snippet to drop; instead the warning + mechanism
  update + `"lead"`→`"team-lead"` fix were made.

## Deviations from plan

- Fix #3 reclassified from a shippable change to a **local remediation** (F1);
  the durable channel to the file that caused the bug is the docs warning.
- The guard was centralized into `_require_resolved_identity()` and extended to
  `read_messages` (not only `send_message`/`resume_session` as the plan's first
  draft implied) per F3.
- `_recover_session_id` early-return placed before the binding-key match and
  auto-adopt scan (F2), not only suppressing the nudge.

## Pinned environment

- pi `0.80.10`
- pi-mcp-adapter `2.11.0`

The `--mcp-config` flag, config source order, and shallow server-entry merge are
observed in this stack; a future adapter change is caught by the live smoke test.

## Validation commands (whole-repo gates, Linux)

```bash
ruff check .
python -m pytest tests/test_pi_worker_identity.py tests/test_server_simple_guards.py -q
python -m pytest -q          # full suite
```

Live pi smoke (deferred acceptance check for the PR): spawn a pi worker, have it
`send_message`, assert the recorded `from` is the worker's own name and that no
`resume_session` fires on the lead's session.
