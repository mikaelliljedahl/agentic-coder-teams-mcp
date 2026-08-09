# Implementation: stop wake hooks from emitting dead-PID owner-bound watch commands

Branch: `fix/hook-watch-owner-binding` (worktree `C:\code\github\win-agent-teams-mcp\wt-hook-watch-owner`).

All commands below were run from the worktree root with the shared venv and
`PYTHONPATH=C:\code\github\win-agent-teams-mcp\wt-hook-watch-owner\src`, verified once with:

```
$ python -c "import claude_teams; print(claude_teams.__file__)"
C:\code\github\win-agent-teams-mcp\wt-hook-watch-owner\src\claude_teams\__init__.py
```

## Red evidence

Tests were written first. Focused run before any production change:

```
$ python -m pytest tests/test_watch_command_discovery.py tests/test_lead_wake.py tests/test_member_wake.py -q
...
E       TypeError: _watch_command_bash() got an unexpected keyword argument 'bind_owner'

tests\test_lead_wake.py:445: TypeError
___ TestDecisionCore.test_not_armed_blocks_with_reader_scoped_watch_command ___
...
        # Hook-emitted commands are deliberately unbound: the hook's parent is a
        # transient wrapper, so a baked-in owner PID would die instantly (exit 4).
>       assert "--owner-pid" not in result.reason
E       AssertionError: assert '--owner-pid' not in 'An inbox wa...e reminders.'
E
E         '--owner-pid' is contained here:
E           ned-sess' --owner-pid 30444 --owner-token 134307534416295032 --reader ext  Once it is running in the background, you may end your turn. If you are finished as a member of this team, call leave_team(member_token=...) instead to stop these reminders.
E         ?           +++++++++++

tests\test_member_wake.py:315: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_watch_command_discovery.py::test_watch_argv_omits_owner_binding_when_bind_owner_is_false
FAILED tests/test_watch_command_discovery.py::test_watch_command_bash_omits_owner_binding_when_bind_owner_is_false
FAILED tests/test_watch_command_discovery.py::test_watch_subprocess_exits_when_bound_owner_dies
FAILED tests/test_watch_command_discovery.py::test_unbound_watch_argv_times_out_instead_of_exiting_owner_gone
FAILED tests/test_watch_command_discovery.py::test_watch_command_powershell_executes_and_times_out_quietly
FAILED tests/test_watch_command_discovery.py::test_watch_argv_runs_from_unrelated_cwd_without_pythonpath
FAILED tests/test_lead_wake.py::TestDecisionCore::test_wake_allows_when_armed_bg_task_is_unbound
FAILED tests/test_lead_wake.py::TestDecisionCore::test_wake_blocks_arm_instruction_when_not_armed_no_unread
FAILED tests/test_lead_wake.py::TestIdentityAndArming::test_arming_match_is_separator_insensitive_and_session_scoped
FAILED tests/test_member_wake.py::TestDecisionCore::test_not_armed_blocks_with_reader_scoped_watch_command
10 failed, 67 passed, 1 skipped in 51.36s
```

Three of the ten (`test_watch_subprocess_exits_when_bound_owner_dies`,
`test_watch_command_powershell_executes_and_times_out_quietly`,
`test_watch_argv_runs_from_unrelated_cwd_without_pythonpath`) are pre-existing
subprocess tests that are load-sensitive under this file's 10 s `_run` timeout;
they pass when re-run in isolation and were unaffected by this change:

```
$ python -m pytest tests/test_watch_command_discovery.py::test_watch_subprocess_exits_when_bound_owner_dies tests/test_watch_command_discovery.py::test_watch_argv_runs_from_unrelated_cwd_without_pythonpath tests/test_watch_command_discovery.py::test_watch_command_powershell_executes_and_times_out_quietly tests/test_watch_command_discovery.py::test_unbound_watch_argv_times_out_instead_of_exiting_owner_gone -q
...F                                                                     [100%]
>       argv = server_simple._watch_argv(tmp_path, timeout=1, bind_owner=False)
E       TypeError: _watch_argv() got an unexpected keyword argument 'bind_owner'
1 failed, 3 passed in 21.87s
```

The seven genuine reds are exactly the new/extended assertions: the missing
`bind_owner` keyword on `_watch_argv`/`_watch_command_bash`, and the D5/M5 hook
reasons still containing `--owner-pid`/`--owner-token`.

## Production change

- `src/claude_teams/server_simple.py`
  - `_watch_argv` gains keyword-only `bind_owner: bool = True`. When false, both
    the `os.getppid()`/`process_manager.creation_token` lookup **and** the
    `--owner-pid`/`--owner-token` pair are skipped. Docstring updated to state
    why (the caller's parent is not the eventual runner) and what bounds an
    unbound watcher (the CLI default `--timeout`).
  - `_watch_command_bash` gains keyword-only `bind_owner: bool = True` and
    passes it through. `_watch_command_powershell` untouched (no hook uses it).
  - `_DISK_CONTRACT_NOTE`: the `spawn_agent`/`agent_watch_paths` owner-binding
    sentence is unchanged; one clause added to the later "Claude Code lead wake"
    Stop-hook paragraph saying hook-suggested commands are deliberately NOT
    owner-bound and are bounded by the default `--timeout` instead.
- `src/claude_teams/lead_wake.py` — `_arm_reason` calls
  `_watch_command_bash(session_dir, bind_owner=False)`.
- `src/claude_teams/member_wake.py` — `_member_arm_reason` calls
  `_watch_command_bash(joined_session_dir, reader=member, bind_owner=False)`.

## Tests added

- `tests/test_watch_command_discovery.py`
  - `test_watch_argv_omits_owner_binding_when_bind_owner_is_false` — plan case 1
    (monkeypatched live `getppid` + available creation token).
  - `test_watch_command_bash_omits_owner_binding_when_bind_owner_is_false` —
    the Bash rendering threads the flag and keeps `--reader`.
  - `test_unbound_watch_argv_times_out_instead_of_exiting_owner_gone` — plan
    case 5 runtime regression: precondition asserts the default path WOULD bind
    (skips if creation tokens are unavailable), then runs the unbound argv with
    `--timeout 1` and asserts exit 2 with empty stdout/stderr, not exit 4.
  - Plan case 4 (default path still owner-bound) already existed as
    `test_watch_argv_keeps_session_dir_with_spaces_as_one_token` and
    `test_watch_argv_binds_concurrent_leads_independently`; not duplicated.
- `tests/test_lead_wake.py`
  - `test_wake_blocks_arm_instruction_when_not_armed_no_unread` extended: D5
    reason contains the session dir and no `--owner-pid`/`--owner-token`.
  - `test_wake_allows_when_armed_bg_task_is_unbound` — plan case 6 decision path:
    a running `background_tasks` entry built from
    `_watch_command_bash(..., bind_owner=False)` is accepted as D4/allow.
  - `test_arming_match_is_separator_insensitive_and_session_scoped` extended
    with a direct `_command_matches_session` case for the unbound rendering.
- `tests/test_member_wake.py`
  - `test_not_armed_blocks_with_reader_scoped_watch_command` extended: the M5
    reason keeps `--reader <member>` and carries no owner flags. Per plan case 6,
    member-wake shares `_is_armed`, so no duplicate M4 matcher suite was added.

## Green evidence

Focused:

```
$ python -m pytest tests/test_watch_command_discovery.py tests/test_lead_wake.py tests/test_member_wake.py -q
77 passed, 1 skipped in 22.56s
```

Full suite:

```
$ python -m pytest -q
1192 passed, 2 skipped in 74.70s (0:01:14)
```

Lint:

```
$ python -m ruff check .
All checks passed!
```

Type check (CI runs `uv run ty check`, `.github/workflows/ci.yml:31`):

```
$ uv run ty check
error[unresolved-attribute]: Object of type `BaseContext` has no attribute `Process`
   --> tests\test_join_team.py:730:9
Found 1 diagnostic
```

This single diagnostic is **pre-existing** and unrelated: `tests/test_join_team.py`
is not touched by this change (see `git diff --stat` below). Reported, not hidden.
Note `uv run` provisions its own `.venv/` in this worktree as a side effect; it is
gitignored and does not appear in the working tree status.

## Files changed

```
$ git diff --stat
 src/claude_teams/lead_wake.py         |  2 +-
 src/claude_teams/member_wake.py       |  4 ++-
 src/claude_teams/server_simple.py     | 29 ++++++++++++++-----
 tests/test_lead_wake.py               | 36 +++++++++++++++++++++++
 tests/test_member_wake.py             |  4 +++
 tests/test_watch_command_discovery.py | 54 +++++++++++++++++++++++++++++++++++
 6 files changed, 120 insertions(+), 9 deletions(-)
```

## Deviations from the plan

- Added one test beyond the plan's list:
  `test_watch_command_bash_omits_owner_binding_when_bind_owner_is_false`. The
  plan only named `_watch_argv` directly, but `_watch_command_bash` is the actual
  hook-facing entry point being threaded, so its rendering is pinned directly.
- No plan test case was skipped. Case 4 was satisfied by existing coverage and
  deliberately not duplicated.
- Nothing committed, per instructions.
