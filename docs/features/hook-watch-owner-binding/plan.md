# Plan: stop wake hooks from emitting dead-PID owner-bound watch commands

## Problem / current behavior

The lead-wake and member-wake Stop hooks block an idle coordinator with an
operational reason containing a ready-to-run watch command:

- `lead_wake._arm_reason` (`src/claude_teams/lead_wake.py:221-230`) calls
  `server_simple._watch_command_bash(session_dir)`.
- `member_wake._member_arm_reason` (`src/claude_teams/member_wake.py:154-165`)
  calls `server_simple._watch_command_bash(joined_session_dir, reader=member)`.

`_watch_command_bash` → `_watch_argv`
(`src/claude_teams/server_simple.py:1508-1523`) bakes in
`--owner-pid os.getppid()` + `--owner-token` **when the parent's creation
token is available** (the exit-4 field evidence shows the pair was emitted).
That is correct **in the MCP server process**, whose parent is the long-lived
coordinator. But the hooks run
as short-lived `python -m claude_teams.lead_wake` / `member_wake` processes
whose parent is the transient hook shell wrapper. By the time the model runs
the suggested command, that PID is gone (or reused), so the watcher immediately
and correctly exits 4 ("owner gone"). Every re-arm attempt gets a fresh equally
dead command; the no-progress guard eventually fails open and the lead sleeps
unwakeable. Observed in the field (2026-08-09): every hook-suggested watcher
died instantly with exit 4; a manually started **unbound** watcher
(`--reader team-lead --timeout 1800`) lived.

The MCP-tool paths (`spawn_agent`, `agent_watch_paths`, `session_info` returns)
are NOT affected — verified working owner-bound watchers in live sessions.

## Design

Hook-emitted commands must omit the owner binding: a hook cannot know the
coordinator's PID, and since #46 the 1800 s default `--timeout` alone bounds
an unbound watcher's lifetime (the Stop hook's re-arm loop re-arms after
timeouts but does not bound an already-running watcher).

1. Add a keyword-only `bind_owner: bool = True` parameter to `_watch_argv` and
   thread it through `_watch_command_bash` (PowerShell rendering untouched —
   no hook uses it). When `bind_owner=False`, skip the
   `--owner-pid`/`--owner-token` pair entirely.
2. `lead_wake._arm_reason` passes `bind_owner=False`.
3. `member_wake._member_arm_reason` passes `bind_owner=False`.
4. `_DISK_CONTRACT_NOTE`: leave the owner-binding sentence untouched (it is
   correctly scoped to `spawn_agent`/`agent_watch_paths` returns). Add the
   clarification to the later **Stop-hook paragraph** instead: hook-suggested
   commands are deliberately unbound and bounded by the default `--timeout`.

Default stays `True` so every existing MCP-tool call site keeps its binding
with zero diff.

## Files affected

- `src/claude_teams/server_simple.py` — `_watch_argv`, `_watch_command_bash`,
  one clause in `_DISK_CONTRACT_NOTE`.
- `src/claude_teams/lead_wake.py` — `_arm_reason`.
- `src/claude_teams/member_wake.py` — `_member_arm_reason`.
- `tests/test_lead_wake.py` (or the file holding `_arm_reason` coverage) — new
  assertion: the D5 reason's command contains `claude_teams.cli`/`watch` and
  the session dir but NOT `--owner-pid`.
- `tests/test_member_wake.py` — same for the M5 reason (plus `--reader`).
- `tests/test_watch_command_discovery.py` — new direct unit test:
  `_watch_argv(dir, bind_owner=False)` contains no owner flags even when a
  creation token is available; default call unchanged.

## Risks

- An unbound hook-armed watcher outlives a dead coordinator for up to 30 min
  (the default timeout). Accepted: bounded, harmless (it only reads), and the
  alternative is a watcher that never lives at all.
- `_is_armed` matches commands by `claude_teams.cli` + `watch` + session-dir
  tokens only (`lead_wake._command_matches_session`), so binding removal does
  not affect armed detection.

## Test cases (red first)

1. `_watch_argv(dir, bind_owner=False)` → no `--owner-pid` / `--owner-token`
   tokens, even with a live creation token.
2. `lead_wake._arm_reason(session_dir)` reason string: contains the watch
   command for the session dir, does NOT contain `--owner-pid`.
3. `member_wake._member_arm_reason(dir, "alice")` reason string: contains
   `--reader alice`, does NOT contain `--owner-pid`.
4. Existing default-path tests (owner flags present in canonical argv when a
   token is available) stay green.
5. Runtime regression (review finding 1): `_watch_argv(dir, bind_owner=False)`
   executed as a subprocess with a short explicit `--timeout` — arranged so the
   default path WOULD have bound (live parent, token available) — exits 2
   (quiet timeout), not 4. Mirrors the existing subprocess seams in
   `tests/test_watch_command_discovery.py:160-201`.
6. Armed-detection loop closure (review finding 2): a running
   `background_tasks` entry whose command came from
   `_watch_command_bash(..., bind_owner=False)` is accepted as armed — one
   direct `_command_matches_session` case plus one D4 decision-path assertion
   (member-wake shares `_is_armed`, so M4 needs no duplicate matcher suite).
