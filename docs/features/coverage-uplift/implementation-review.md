# Implementation review: coverage uplift — first increment

## Verdict: REQUEST-CHANGES

The increment is test-only, the focused suite is green, and most review dispositions are
implemented correctly. However, two entry-guard tests are not hermetic under the failure mode they
are intended to detect: if either guard regresses, the tests can inspect the developer's real
`~/.codex` or `~/.claude`. The review brief makes that a blocker. Several negative-only tests also
do not distinguish the intended production branch from a broken implementation, contrary to the
approved observable-assertion disposition.

## Findings

### BLOCKER — entry-guard tests can read real user state when the guard regresses

`test_read_codex_output_rejects_bad_inputs` and
`test_read_claude_output_rejects_bad_inputs` do not patch `Path.home`
(`tests/test_agent_output.py:1335` and `tests/test_agent_output.py:1348`). They are safe only while
the exact guards under test remain intact (`src/claude_teams/agent_output.py:54` and
`src/claude_teams/agent_output.py:87`). If the Codex guard is removed or weakened, execution reaches
the real `Path.home() / ".codex" / "sessions"` at
`src/claude_teams/agent_output.py:225`; if the Claude guard regresses, execution reaches the real
`Path.home() / ".claude" / "projects"` at `src/claude_teams/agent_output.py:95`. A regression test
must remain isolated while exercising the regression. Patch `Path.home` to `tmp_path` in both tests
(or apply an autouse home-isolation fixture to the appended section).

### MAJOR — several tests cover lines but do not prove the claimed branch behavior

The approved disposition requires skip-logic tests to retain a valid older record and assert its
selection (`docs/features/coverage-uplift/plan.md:193`). The following assertions are too weak:

- `test_read_codex_output_skips_non_session_meta_and_bad_payload`
  (`tests/test_agent_output.py:1406`) supplies only rejected candidates and asserts `None`. Removing
  only the non-`session_meta` check at `src/claude_teams/agent_output.py:135` still returns `None`,
  because that record has no usable cwd. The test detects the bad-payload guard, but not the
  non-session-meta skip it claims. Add a valid older rollout and assert that it is selected.
- `test_last_codex_message_none_without_assistant` (`tests/test_agent_output.py:1522`) does not prove
  the role-mismatch skip at `src/claude_teams/agent_output.py:261`: removing the role check still
  returns `None` because the record has no matching content. A valid older assistant record would
  make the selection observable.
- `test_content_text_non_list_returns_none` (`tests/test_agent_output.py:1603`) still returns `None`
  if the non-list guard at `src/claude_teams/agent_output.py:349` is deleted: iterating the supplied
  string produces no dict parts, then the later empty-parts guard returns `None`.
- `test_goose_recipes_skips_blank_and_missing_dirs`
  (`tests/test_agent_discovery.py:151`) does not distinguish either continue at
  `src/claude_teams/backends/_agent_discovery.py:118` or
  `src/claude_teams/backends/_agent_discovery.py:121` from falling through: globbing the current
  directory or a nonexistent directory can still yield the asserted empty list. Use portable fault
  injection or a controlled current directory containing a supported recipe to make the blank
  segment behavior observable. The nonexistent-directory branch likewise needs an assertion that
  proves it was not scanned.
- `test_discover_claude_agents_no_dirs_returns_empty`
  (`tests/test_agent_discovery.py:41`) would also pass if the `is_dir` skip at
  `src/claude_teams/backends/_agent_discovery.py:46` were removed, because globbing the nonexistent
  directories naturally yields no files.

These are coverage-gaming risks under the review's explicit standard: they execute the branch but
would pass for relevant broken variants of production code.

### MAJOR — the promised `server_simple` isolation fixture is incomplete

The accepted disposition specifies an autouse fixture that patches `_SESSION_BASE`, `_TEAMS_BASE`,
`_session_id`, `_AGENT_SESSION_ID`, `_AGENT_PARENT_NAME`, `_pending_recovery`, `_inbox_locks`, cwd,
and environment overrides (`docs/features/coverage-uplift/plan.md:183`). The actual fixture is not
autouse (`tests/test_server_simple_guards.py:24`) and does not patch `_AGENT_PARENT_NAME` or
`IDENTITY` (`tests/test_server_simple_guards.py:38`). Those values are bound from the developer's
environment at import time (`src/claude_teams/server_simple.py:49` and
`src/claude_teams/server_simple.py:58`), and `_message_recipient` consumes them
(`src/claude_teams/server_simple.py:792`). In particular, the root-lead expectation at
`tests/test_server_simple_guards.py:264` can vary when tests are launched with `AGENT_NAME` or
`AGENT_PARENT_NAME` set. Normalize these import-time globals in the fixture as approved, and make
the fixture autouse (or document and prove why every non-user of it is state-free).

### MINOR — one contract-isolation test is not labelled as such

The Codex normalization test explicitly says it is contract isolation
(`tests/test_agent_output.py:1341`), but the adjacent Claude resolution test does not
(`tests/test_agent_output.py:1353`). Its name reads like a natural filesystem edge even though it
monkeypatches `_resolve_path_text` to an impossible empty result. This only partially honors the
labelling disposition at `docs/features/coverage-uplift/plan.md:174`.

### MINOR — validation evidence has an incorrect focused-test count

`implementation.md` reports 13 passing discovery + entrypoint tests
(`docs/features/coverage-uplift/implementation.md:34`), but there are 12 discovery tests and 2
entrypoint tests, and the focused run executes 14. The overall increase of 88 is internally correct:
33 appended `agent_output` tests + 41 `server_simple` tests + 12 discovery tests + 2 entrypoint
tests. Correct the local red/green evidence count to 14.

### MINOR — two deferred `server_simple` lines are easy, not runtime/subprocess-hard

The 30-line deferred set in `implementation.md:25` matches the source and rounds to the claimed
97% (891/921 = 96.74%). Windows locking and the `follow_up_agent` changed/save combinations are
reasonably hard or platform-specific. In contrast, `server_simple.main()` at
`src/claude_teams/server_simple.py:2105` can be tested by patching `mcp.run` and asserting one call;
the `__main__` call at `src/claude_teams/server_simple.py:2110` can use the same runpy technique
already used by `tests/test_server_entrypoint.py:20`. Deferral was explicitly approved in the plan,
so this is not a scope violation, but `implementation.md:67` should not characterize those two
lines as requiring a server runtime or subprocess.

### NIT — entrypoint test emits a runpy warning

The focused run emits `RuntimeWarning: 'claude_teams.server' found in sys.modules ... prior to
execution` because `tests/test_server_entrypoint.py:10` imports the module before
`runpy.run_module` at line 26. The assertion passed, but removing the cached module before runpy or
otherwise isolating the executions would avoid the warning's stated unpredictable-behavior risk.

## Confirmed dispositions and behavior

- No production changes: `git status --porcelain` reports only
  `tests/test_agent_output.py` plus the three untracked test files; no `src/` path is modified.
  `git diff --stat` reports only `tests/test_agent_output.py` (354 insertions). Git omits untracked
  files from `diff --stat`, and `docs/` is ignored by `.gitignore`, so status/stat must be interpreted
  with those limitations.
- Corrupt-registry tolerance is asserted at the MCP-tool level only for `resume_session` and
  `session_info` (`tests/test_server_simple_guards.py:416` and
  `tests/test_server_simple_guards.py:425`), matching the catches at
  `src/claude_teams/server_simple.py:1837` and `src/claude_teams/server_simple.py:1878`. The
  `list_agents`, `agent_status`, and `kill_agent` tests use only no-session/not-found registries.
- `list_backends` uses a deterministic fake registry and asserts the full returned row
  (`tests/test_server_simple_guards.py:444`); it does not invent a session guard.
- Fault injection is portable (`Path.open`, `read_text`, `stat`, `rglob`, `iterdir`); no chmod is
  used. The reverse-line OSError test consumes the generator and asserts `[]`
  (`tests/test_agent_output.py:1614`).
- Discovery has an autouse `Path.home` isolation fixture
  (`tests/test_agent_discovery.py:26`), so those tests do not inspect real `~/.claude` or
  `~/.codex` state.
- Entrypoint identity and invocation assertions are genuine
  (`tests/test_server_entrypoint.py:13` and `tests/test_server_entrypoint.py:20`).

## Verification performed

- `git status --porcelain`
- `git diff --stat` / `git diff --name-only`
- Focused tests: **135 passed**, with the one runpy warning described above.
- Ruff on the four reviewed test files: **clean**.
- Scoped coverage: `server.py` **100%**, `_agent_discovery.py` **100%**, and `agent_output.py`
  **100%**. A focused run cannot reproduce the whole-suite `server_simple.py` 97% or total 83.94%
  because existing tests provide most of that module's accumulated coverage; the full suite was
  intentionally not rerun. Static inspection supports the stated 30-line remaining set and the
  rounded module percentage, but the total 83.94% remains based on the implementer's recorded full
  run.
