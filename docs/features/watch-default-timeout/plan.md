# Plan: raise `watch` default `--timeout` from 60 s to 1800 s

## Problem / current behavior

`claude_teams.cli watch` is a one-shot blocking command. Its `--timeout`
defaults to **60 s** ([src/claude_teams/cli.py:519](../../../src/claude_teams/cli.py)),
and every canonical command the server hands out (`spawn_agent`,
`agent_watch_paths`, `session_info` return values; the lead-wake / member-wake
arm nags) omits `--timeout` (`_watch_argv` in
[src/claude_teams/server_simple.py:1508](../../../src/claude_teams/server_simple.py)),
so it inherits that default.

Real worker turnaround is rarely under 5 minutes, so in practice the watcher
times out (exit 2) long before any actionable edge, the coordinator gets a
spurious background-task wake every minute, and the Stop hook immediately nags
it to re-arm. Observed in the field: leads conclude "the watcher died right
away" and re-launch manually with `--timeout 86400`.

Since PR #45 canonical watchers are owner-bound **when a PID creation token is
available** (`--owner-pid`/`--owner-token`, exit 4 when the coordinator
disappears) — `_watch_argv` omits the owner flags in the supported fallback
where `process_manager.creation_token()` returns `None` (pinned by
`test_watch_argv_omits_owner_binding_when_token_is_unavailable`). For the
bound (normal) case the short timeout no longer serves its original
orphan-protection purpose.

## Decision (user-approved)

Raise the CLI default `--timeout` to **1800 s** (30 min). 86400 was judged too
long; 1800 keeps a bounded worst-case for *unbound* manual invocations (which
have no owner binding as a lifetime guard) while eliminating the once-a-minute
spurious wake cycle for canonical, owner-bound watchers.

Canonical commands keep **omitting** `--timeout` and inherit the new default —
no change to `_watch_argv` output (and the existing
`test_watch_command_discovery` assertion that canonical argv contains no
`--timeout` stays valid).

## Changes

1. **`src/claude_teams/cli.py`** — `watch` option default `60.0` → `1800.0`;
   keep the help text accurate.
2. **`docs/reference/agent-messaging-protocol.md`** — update the two stale
   statements of the default (the synopsis `[--timeout 60]` around line 1405
   and "Default timeout: **60 s**" around line 1456), **and repair the stale
   source citations next to them**: the synopsis cites `cli.py:182-199` and the
   default bullet cites `cli.py:186`, while the watch command now lives around
   `cli.py:517-552` (default at `:519-520`). Cite without hard line pins or
   with the corrected lines.
3. **Tests** (red first): add a metadata-only default assertion in
   `tests/test_cli_watch.py` — do **not** invoke `watch` without `--timeout`
   (that would block until the new 1800 s deadline). Instead obtain the Click
   command via `typer.main.get_command(app)`, select `commands["watch"]`, find
   the option whose `opts` contains `"--timeout"`, and assert
   `option.default == 1800.0`. All existing tests that *execute* the blocking
   watcher pass an explicit timeout and are unaffected; the
   `test_watch_command_discovery` tests deliberately exercise timeout-less
   `_watch_argv` rendering (canonical omission) and stay valid.

Non-changes, deliberately:

- `_DISK_CONTRACT_NOTE`'s Codex guidance "append `--timeout 60` … looped" is an
  *explicit* foreground-loop pattern for a harness with no idle-wake; it stays.
- `AGENT_UPGRADE_NOTES.md:44` and `:118` show the same explicit bounded Codex
  foreground-loop recipe (`--timeout 60`), not statements of the CLI default;
  they stay.
- `README.md:446` and `:615` use symbolic `[--timeout SECONDS]` wording; no
  update needed.
- Canonical commands keep `timeout=None` (omission) as the single source of
  truth rather than pinning an explicit `--timeout` in `_watch_argv`.
- No change to exit-code semantics (0 wake / 2 timeout / 4 owner gone) or to
  the settle-window logic.

## Files affected

- `src/claude_teams/cli.py`
- `tests/test_cli_watch.py`
- `docs/reference/agent-messaging-protocol.md`

## Risks

- Unbound watchers now linger up to 30 min instead of 1 min if forgotten. This
  covers both hand-typed invocations and the canonical fallback where no PID
  creation token is available (owner flags omitted). Accepted: still bounded,
  and the token-less fallback is rare (creation-token probe failing on the
  local platform).
- Coordinators relying on the 60 s timeout as a de-facto polling cadence
  (Codex foreground loop) are unaffected because that pattern passes
  `--timeout 60` explicitly.

## Test cases

1. Metadata-only: `typer.main.get_command(app)` → `commands["watch"]` → the
   option with `"--timeout"` in `opts` has `default == 1800.0`. Never invoke
   `watch` without an explicit timeout in tests.
2. Existing suite (`tests/test_cli_watch.py`, `tests/test_watcher_contract.py`,
   `tests/test_watch_command_discovery.py`) stays green.
