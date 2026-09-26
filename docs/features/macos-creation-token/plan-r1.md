# macOS process liveness and verified `kill_agent`

## Status note

The implementation predates this plan: it was written as local macOS
adaptations on a Mac host and is being brought into the repository workflow
retroactively. This plan documents the design so it can be reviewed on its own
terms before the implementation review. Anything the review rejects is changed
before the PR is opened.

## Scope

1. **Creation tokens and liveness on macOS.** `process_manager.creation_token`
   returns a token on Windows (WMI) and Linux (`/proc/<pid>/stat`) but `None`
   for every PID on macOS, which has no `/proc`.
2. **`kill_agent` reports whether the process actually died.** Today it removes
   the registry row and returns `{"success": true}` whether or not the OS
   process was signalled or survived.

Out of scope: `procinfo.resolve_nearest_host()` also returns an empty chain on
macOS (`tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain`
fails on `origin/main` and after this change). It affects conversation-scoped
lead-wake (#48) and watcher-to-coordinator binding (#45) and is a separate fix.

## Current behavior (origin/main e5c8e98, on macOS)

- `creation_token()` → `None` for every PID, so every agent record stores
  `create_token: None`.
- `owns_process()` therefore answers `False` for any agent whose spawning
  manager no longer holds it in memory (any MCP server restart).
- `kill_agent` fails closed on that `False`: it removes the row, signals
  nothing, returns `success: true`. The agent keeps running as an orphan
  (~400 MB for a claude agent, ~110 MB for codex) and the caller is told the
  reap succeeded.
- Launcher-style backends record the terminal PID; the agent is a grandchild
  recorded in a sidecar. After a restart only the launcher would be signalled.
- Full suite on macOS: 13 failed, 2 errors (pid_reuse creation tokens,
  delivery_integrity, follow_up_delivery leases, procinfo).

## Design

### `process_manager.py`
- `_read_darwin_proc_info(pid)`: `sysctl(CTL_KERN, KERN_PROC, KERN_PROC_PID,
  pid)` via ctypes → `kinfo_proc`; reads `p_starttime` (timeval at offset 0)
  and `p_stat` (offset 36). Returns `None` when the PID does not exist.
  `sysctl` is used instead of libproc `proc_pidinfo`, which returns nothing for
  PID 1 and for zombies (indistinguishable from death).
- `_read_darwin_creation_token(pid)`: `"<sec>.<usec:06d>"` of start time;
  immutable per process, so a reused PID gets a different token.
- `creation_token()` dispatches to it when `sys.platform == "darwin"`.
- `_posix_pid_is_zombie`, `_posix_pid_alive`: zombie-excluding liveness
  (`SZOMB` on macOS, `Z` state in `/proc/<pid>/stat` on Linux). On macOS
  `sysctl` also answers for other users' processes where `os.kill(pid, 0)`
  would raise `EPERM`.
- Public `pid_alive(handle)` and `wait_pid_exit(handle, timeout_s)` (50 ms
  poll) so the server can verify a kill without reaching into a manager's
  private `_pid_alive`.

### `server_simple.py`
- `_terminate_agent_process(agent)` replaces the inline `owns_process` +
  `kill_process`:
  - uses `ownership_probe`; if not provably ours: `process_already_dead` when
    gone, otherwise `ownership_not_ours_process_alive` /
    `ownership_indeterminate_process_alive` with `orphan_pid` and a `detail`
    naming the manual command. Still fail-closed: never signals an unproven PID.
  - if ours: resolves the sidecar agent PID (`resolve_agent_pid`), calls
    `kill_process`, then `_kill_and_confirm_gone` for launcher and agent PID
    (re-signal if still alive, wait up to `_KILL_VERIFY_TIMEOUT_SECONDS = 10`).
  - returns `process_terminated` or `process_survived_kill` (+ `orphan_pid`,
    `detail`).
- `kill_agent` merges the outcome into its result; `killed_process` replaces
  the old `owned` flag when deciding `child_exited` for artifact cleanup.
- The `kill_agent` tool docstring tells callers to check `killed_process`,
  since the consuming agent only reads tool descriptions.

### Docs
`docs/reference/agent-messaging-protocol.md` §kill documents the new return
fields and reasons.

## Files affected
- `src/claude_teams/backends/process_manager.py`
- `src/claude_teams/server_simple.py`
- `tests/test_process_liveness.py` (new, real processes)
- `tests/test_kill_agent.py`
- `docs/reference/agent-messaging-protocol.md`

## Risks
1. **Hard-coded `kinfo_proc` offsets** (0, 36; min size 64). Correct for
   64-bit macOS (arm64 and x86_64 share the layout). A wrong offset would give a
   plausible-but-wrong token; guarded by real-process tests on macOS.
2. **Lock hold time.** `_terminate_agent_process` runs inside
   `_agents_transaction`. Verification can wait up to 10 s per PID (launcher +
   agent → worst case ~20 s) while other tools touching that session's
   `agents.json` block. Only in the survivor case; a normal kill returns as
   soon as the PID is gone.
3. **Return-shape change.** `kill_agent` gains fields; `success` semantics are
   unchanged. Additive for existing callers.
4. **Linux behavior change.** `pid_alive` now excludes zombies on Linux too.
5. CI runs on Linux only, so the macOS-specific paths are exercised only by
   local macOS runs.
6. A comment references "CLAUDE.md §7.11.1", which is not a section of this
   repository's CLAUDE.md (it came from a consuming project). To be removed.

## Test cases
- `test_process_liveness.py`: live child has a token on this platform; token
  stable across calls; killed child reads dead (`pid_alive`, `wait_pid_exit`);
  zombie reads dead (POSIX); ownership of a real untracked child after a
  simulated restart.
- `test_kill_agent.py`: `killed_process` true when verified gone; survivor
  reported not absorbed; unowned live PID reported as orphan; indeterminate
  ownership reason; sidecar agent PID killed too; already-dead PID not an orphan.
- Existing `test_pid_reuse.py`, `test_delivery_integrity.py`,
  `test_follow_up_delivery.py` must pass on macOS (they fail on origin/main).

## Validation
`uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
`uv run pytest` — on macOS locally; Linux via PR CI.
