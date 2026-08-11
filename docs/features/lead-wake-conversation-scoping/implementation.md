# Lead-wake conversation scoping implementation

## Outcome

The shared `install_lead_wake` hook is now bound to the Claude conversation
that installed it. Installation resolves the nearest host over the full
Claude/Codex/Pi host set using both executable names and launch argv, requires
that nearest host to be Claude, captures its
PID plus creation token, and requires a concrete active agent session directory
before the settings path is computed or touched. Foreign, restarted, legacy,
or inconsistently configured hook processes fail silent at `D0b`, before any
session discovery or guard write.

The server-spawned per-agent hook remains explicitly `private`. Member wake
behavior is unchanged: its guard filename is still
`wake-progress-member-<name>.json`, its schema remains
`lead-wake-progress/1`, and no `owner_generation` field is written to member
guard files.

## Red and green evidence

The inherited full-suite state reported at handoff was `3 failed, 138 passed,
14 errors`. The focused reproduction command below failed with the same 3
member-wake coexistence failures and 14 install-side errors. The install errors
first exposed the intentionally red test seam (`server_simple.procinfo` was not
yet present); the coexistence failures reached the old
`hooks._wake_hook_matcher(session_dir, identity)` call and raised because the
new keyword-only `owner_mode` was absent.

```powershell
$env:PYTHONPATH="C:\code\github\win-agent-teams-mcp\wt-lead-wake-scoping\src"
& "C:\code\github\win-agent-teams-mcp\agentic-coder-teams-mcp\.venv\Scripts\python.exe" -m pytest tests/test_install_lead_wake.py tests/test_member_wake.py::TestCoexistence -q
```

Reproduced result: exit 1, `3 failed, 14 errors in 3.03s`.

After the server implementation and coexistence fixture update, the focused
feature set completed with `155 passed in 7.46s`. Additional plan-matrix tests
were then added for Windows command parsing, refusal timing and diagnostics,
the Codex-before-Claude F6 chain, access-denied/disappeared host tokens,
ownership restart/handoff, distinct shared settings scopes plus an explicit
private scope, old and different owner guard generations, subprocess-level
fail-open behavior, and the frozen member guard schema.

The accepted post-implementation review findings were first reproduced with
new focused tests. The initial remediation run was red for Node/Pi ancestry,
Linux Node-shim classification, Toolhelp retry, the unified creation-token seam,
and settings-write result handling. After the minimal production changes, the
focused `procinfo`/lead-wake/install/member-wake set passed `126 passed`.

## Final design

- `procinfo` performs one bounded ancestor walk (64 rows). On Linux it reads
  `/proc/<pid>/cmdline` first and uses `/proc/<pid>/comm` plus
  `/proc/<pid>/status` as fallback. On Windows it pairs the Toolhelp process
  name/parent snapshot with command lines from one CIM query and retries one
  transient `ERROR_BAD_LENGTH` snapshot failure. Direct Claude/Codex/Pi image
  names still match exactly; Node launchers are classified only when argv names
  a recognized package script (`@anthropic-ai/claude-code`, `@openai/codex`, or
  `@earendil-works/pi-coding-agent`). Thus the real
  `node .../pi-coding-agent/dist/cli.js` layer stops the walk before an outer
  Claude host.
- `install_lead_wake(remove=False)` completes host resolution, Claude-host
  validation, creation-token capture, and concrete active-session-directory
  validation before computing the settings path. Refusals return one stable
  reason (`host_not_found`, `host_token_unavailable`, `host_walk_failed`, or
  `no_active_session`) plus a sanitized ancestry chain. They neither alter an
  existing settings file nor create its directory. `_SESSION_BASE` is no
  longer a fallback session directory.
- Successful install/removal results now include `"success": true`; settings
  write failures return `{"success":false,"reason":"settings_write_failed"}`
  while preserving the prior file and cleaning the temp. The tool docstring
  documents the active-session remedy, refusal shapes/reasons, both kill
  switches, and the auto-adoption nuance of the baked session-dir fallback.
- Successful installation bakes `--owner-mode bound`, `--owner-host-pid`, and
  `--owner-host-token`. Reinstallation replaces the prior lead-wake group, so
  the last successful installer owns that settings scope while unrelated hook
  groups and settings remain intact.
- Settings writes use a unique sibling temp file followed by `Path.replace`.
  A failed replace leaves the prior bytes valid and unchanged, cleans the temp
  file, and returns `settings_write_failed`.
- `remove=True` performs neither owner nor session resolution and removes only
  the lead-wake group.
- `lead_wake` evaluates `D0b` immediately after the master kill switch. A
  matching bound owner continues through D1-D6; a foreign or unknown owner
  allows without session or inbox work. The owner-only generation resets a
  stale lead guard counter after handoff, while the shared guard helper's
  default preserves the member schema.
- The entire hook entrypoint contains argument parsing, evaluation, logging,
  output, and explicit stdout flush inside a `BaseException`-safe boundary.
- `evaluate` now defaults `owner_mode` to `None` (legacy/unbound → D0b allow).
  Legacy private-wiring tests opt into `owner_mode="private"`; hook-side owner
  tests cover a multi-level non-Claude chain and prove D0b precedes session
  resolution. The vacuous unrelated-path guard assertion was removed.
- Both install and hook paths patch/call the module-level
  `process_manager.creation_token` seam.
- The install result explicitly returns
  `{"binding":{"scope":"conversation","survives_restart":false}}` and a
  note instructing the lead to rerun `install_lead_wake` after restart. The
  protocol reference records the same limitation and handoff semantics.

## Deviations and unperformed supplemental checks

Production behavior follows approved plan v3 after the post-review argv-aware
host correction. One test-design deviation remains explicit: plan test 11 asked
for every pre-existing decision test to inject a matching bound owner, while the
legacy decision tests instead pass `owner_mode="private"` through their module
fixture. Dedicated bound-owner tests traverse the real comparison path. This
keeps private per-agent decision coverage semantically accurate but is not the
literal test arrangement requested by plan test 11.

The member-wake coexistence tests now provision the active session and Claude
owner required by the new lead-install contract; member-wake production code
was not changed.

The plan's manual two-conversation smoke was not run during this implementation
continuation. Windows paired ancestry was already measured in `spike.md`; this
run relied on that evidence and the synthetic/live-Windows process-walk tests.

## Validation

Commands were run from
`C:\code\github\win-agent-teams-mcp\wt-lead-wake-scoping` with the prescribed
interpreter and `PYTHONPATH`.

```powershell
& "C:\code\github\win-agent-teams-mcp\agentic-coder-teams-mcp\.venv\Scripts\python.exe" -m ruff check
```

Result: exit 0, `All checks passed!`.

```powershell
$env:PYTHONPATH="C:\code\github\win-agent-teams-mcp\wt-lead-wake-scoping\src"
& "C:\code\github\win-agent-teams-mcp\agentic-coder-teams-mcp\.venv\Scripts\python.exe" -m pytest tests/ -q
```

Final post-review result: exit 0,
`1254 passed, 2 skipped in 93.65s (0:01:33)`.

The two skips are the repository's platform-conditional tests; this Windows
run is not Linux verification.

## Open release gate: Linux paired ancestry

The Linux half of the R2-3 release gate remains open. Before release, repeat the
paired live capture on the Lubuntu VM and record all three required shapes:

1. server-side and hook-side chains for one top-level Claude conversation must
   select the same nearest host PID;
2. a second simultaneous conversation must select a different nearest host
   PID on both sides; and
3. a server-spawned Claude agent's server and hook chains must select the same
   agent host PID.

The synthetic ancestry suite and Windows Toolhelp smoke do not discharge this
Linux gate.
