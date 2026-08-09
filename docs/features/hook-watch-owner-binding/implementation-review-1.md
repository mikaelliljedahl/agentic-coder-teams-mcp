CHANGES-REQUIRED

1. MAJOR — `tests/test_watch_command_discovery.py:238-255`: the new runtime regression inherits `_run`'s 10-second outer subprocess ceiling (`tests/test_watch_command_discovery.py:20-34`) and failed with `TimeoutExpired` in the exact focused gate; give this test a materially larger harness timeout while retaining the watcher's explicit `--timeout 1`, then rerun the exact focused gate from a cold process.

## Evidence for finding 1

With `PYTHONPATH=C:\code\github\win-agent-teams-mcp\wt-hook-watch-owner\src` and `.venv\Scripts\python.exe`, the requested focused command exited 1:

```text
2 failed, 75 passed, 1 skipped in 129.74s (0:02:09)
```

The failures were:

- `test_watch_subprocess_exits_when_bound_owner_dies`: the pre-existing owner-death subprocess exceeded its 5-second `communicate` ceiling.
- `test_unbound_watch_argv_times_out_instead_of_exiting_owner_gone`: the newly added finding-1 regression exceeded `_run`'s 10-second ceiling.

Both passed when rerun individually. The new regression took 8.58 seconds in isolation, leaving only about 1.4 seconds of margin below its 10-second outer limit; the existing owner-death test passed alone in 5.81 seconds. The full suite subsequently passed, but that warm/full-run success does not erase the independently reproduced failure of the reported focused gate. The implementation note itself records that this subprocess file is load-sensitive (`docs/features/hook-watch-owner-binding/implementation.md:49-60`); the new load-bearing regression must not inherit the same fragility.

The test's semantic assertions are otherwise correct: it first verifies that the default helper path would produce `--owner-pid`, generates an unbound argv with a one-second watcher timeout, asserts both owner flags are absent, and expects quiet exit 2 rather than exit 4. The fix is test-harness headroom, not a production-code change. For example, allow `_run` to accept a per-call timeout and use a larger value for this subprocess startup while keeping `timeout=1` in `_watch_argv`.

## Production implementation assessment

The implementation matches the approved design:

- `_watch_argv` adds keyword-only `bind_owner: bool = True` and guards both `os.getppid()`/`creation_token` lookup and owner-argument emission (`src/claude_teams/server_simple.py:1508-1533`). Default behavior and the token-unavailable fallback remain intact.
- `_watch_command_bash` adds and forwards the keyword-only option without changing reader/timeout ordering (`src/claude_teams/server_simple.py:1536-1547`). PowerShell remains on the default bound path.
- Lead D5 and member M5 pass `bind_owner=False`; member M5 retains `reader=member` (`src/claude_teams/lead_wake.py:221-230`; `src/claude_teams/member_wake.py:154-167`).
- Caller tracing found no missed production hook-context caller. The only production calls opting out are `_arm_reason` and `_member_arm_reason`; MCP/result discovery callers retain the default owner binding.

No production correctness defect was found.

## Prior-review closure

- Finding 1: implemented but not yet sound as a gate because the new runtime regression is load-sensitive; this is the sole change-required item.
- Finding 2: closed. `tests/test_lead_wake.py:276-295` places an unbound Bash command in a running `background_tasks` entry and proves D4/allow. `tests/test_lead_wake.py:429-484` also directly proves `_command_matches_session` accepts the unbound rendering. Member-wake shares `_is_armed`, so no duplicate M4 matcher suite is required.
- Finding 3: closed. The plan now qualifies token availability and attributes lifetime bounding to the CLI timeout (`plan.md:13-16`, `:33-36`). The existing MCP-return owner-binding sentence is untouched; the new unbound-command clause is placed in the later Stop-hook paragraph (`src/claude_teams/server_simple.py:1612-1615`).

The D5/M5 tests assert the session/reader contract and absence of both `--owner-pid` and `--owner-token`. The added direct Bash test also proves `bind_owner=False` is threaded through while preserving `--reader alice`.

## Scope and gate verification

The source/test diff is limited to the six declared files: three production modules and three corresponding test modules. The untracked feature directory contains only `plan.md`, `plan-review-1.md`, and `implementation.md` before this review report. No scope creep was found, and `git diff --check` passed.

Independent verification used this worktree's local `.venv` with explicit `PYTHONPATH`; import provenance was:

```text
C:\code\github\win-agent-teams-mcp\wt-hook-watch-owner\src\claude_teams\__init__.py
```

Gate results:

- Focused tests: **FAILED**, exit 1 — 75 passed, 1 skipped, 2 failed in 129.74s.
- New runtime regression alone: passed — 1 passed in 8.58s.
- Existing owner-death regression alone: passed — 1 passed in 5.81s.
- Full suite: passed — 1192 passed, 2 skipped in 116.28s, exit 0.
- Ruff: `All checks passed!`, exit 0.
- `uv run ty check`: exit 1 with exactly one diagnostic, `tests/test_join_team.py:730:9`, unresolved `BaseContext.Process`.

The `ty` diagnostic is confirmed unrelated to this implementation: `git diff --quiet -- tests/test_join_team.py` returned 0, scoped status was clean, and the file is absent from `git diff --name-only`. The implementation did not introduce or modify that diagnostic.
