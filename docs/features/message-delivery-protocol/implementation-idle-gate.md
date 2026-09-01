# Delivery idle-gate implementation note

## Change

P1a now requires an authoritative hook-written `waiting` marker before a live
agent can be shut down and resumed. Transcript inactivity remains a busy/wait
signal and can never authorize replacement. P1b adds the 60-second retry hint
and changes the cooperative-tail and tool-description pacing to send once,
poll `delivery_status`, then drain only when appropriate.

## Files changed

The final `git diff --stat` for tracked changes was:

```text
 src/claude_teams/server_simple.py |  40 ++++++++++---
 tests/test_bounded_delivery.py    | 118 +++++++++++++++++++++++++++++++++++++-
 tests/test_follow_up_delivery.py  |  14 ++++-
 3 files changed, 161 insertions(+), 11 deletions(-)
```

This requested note is a new untracked artifact and is therefore not included
by `git diff --stat`. The pre-existing untracked proposal file was read only
and was not modified.

## Mandated tests: red before the production change

Command used the worktree source first on `PYTHONPATH` and selected the four
mandated tests in `tests/test_bounded_delivery.py`.

Captured output before the production change:

```text
FF..                                                                     [100%]
FAILED ...::test_busy_target_that_never_becomes_resumable_returns_the_pending_tail
E       KeyError: 'retry_after_s'
FAILED ...::test_stale_transcript_activity_without_marker_never_kills_live_child
E       AssertionError: assert 'unconfirmed' == 'pending'
2 failed, 2 passed in 3.34s
```

The two waiting-marker tests were already green; the stale unmarked-agent test
demonstrated the old shutdown/resume path by returning `unconfirmed`.

## Mandated tests: green after the change

```text
....                                                                     [100%]
4 passed in 2.06s
```

The existing follow-up tests that implicitly treated a resumed live child as
idle were updated to write the authoritative `waiting` marker before their
next operation. No old heuristic-kill assertion remains in the two edited
delivery test files.

## Quality gates

- `uv run ruff format --check .`: PASS — 79 files already formatted.
- `uv run ruff check .`: PASS — all checks passed.
- `uv run ty check`: FAIL — one unchanged Windows/platform diagnostic at
  `tests/test_join_team.py:730`: `BaseContext` has no attribute `Process`.
- `PYTHONPATH=<worktree>/src uv run pytest`: FAIL — `1392 passed, 2 skipped,
  5 failed` in 291.55s.

The full-suite failures were:

- `tests/test_agent_output.py::test_follow_up_agent_refuses_idle_live_agent_without_replace`
  — still expects transcript age to yield `agent_idle_but_alive`.
- `tests/test_agent_output.py::test_follow_up_agent_replaces_idle_live_agent_when_allowed`
  — still expects transcript age to authorize replacement.
- `tests/test_restart_safety.py::test_tokenless_recovered_record_never_graceful_shutdowns_pid`
  — still expects the no-marker recovered live case to proceed.
- `tests/test_restart_safety.py::test_reused_pid_does_not_get_graceful_shutdown`
  — still expects the no-marker live case to proceed.
- `tests/test_follow_up_delivery.py::test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`
  — unrelated `kill_agent` token-mismatch failure; that production path was not
  changed.

No mandated test was skipped.

## Fixup round 2

### Per-test outcomes

- `tests/test_agent_output.py::test_follow_up_agent_refuses_idle_live_agent_without_replace` — added an explicit `waiting` marker; PASS.
- `tests/test_agent_output.py::test_follow_up_agent_replaces_idle_live_agent_when_allowed` — added an explicit `waiting` marker; PASS.
- `tests/test_restart_safety.py::test_tokenless_recovered_record_never_graceful_shutdowns_pid` — added an explicit `waiting` marker so the original resume/token-safety intent remains exercised; PASS, with no graceful or hard-kill calls.
- `tests/test_restart_safety.py::test_reused_pid_does_not_get_graceful_shutdown` — added an explicit `waiting` marker so the original PID-reuse safety intent remains exercised; PASS, with no graceful or hard-kill calls.
- `tests/test_follow_up_delivery.py::test_kill_agent_proceeds_when_the_holder_token_no_longer_matches` — run after `git stash` and again after `git stash pop`; FAIL in both states with `assert result["success"] is True` receiving `False`. Confirmed pre-existing and unrelated; left unchanged.

The four corrected tests ran together as:

```text
....                                                                     [100%]
4 passed in 3.25s
```

### Round-2 quality gates

- `uv run ruff format --check .`: PASS — 79 files already formatted.
- `uv run ruff check .`: PASS — all checks passed.
- `uv run ty check`: FAIL — one unchanged Windows/platform diagnostic at
  `tests/test_join_team.py:730`: `BaseContext` has no attribute `Process`.
- `PYTHONPATH=C:/code/github/win-agent-teams-mcp/wt-delivery-idle-gate/src uv run pytest`: FAIL — `1396 passed, 2 skipped, 1 failed` in 112.50s. The sole failure is the pre-existing token-mismatch test listed above.

No mandated test was skipped in fixup round 2, and no commit was created.
