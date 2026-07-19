# Verdict: APPROVE-WITH-CHANGES

The first-increment scope is sound. Deferring `process_manager.py` is preferable to mixing a large subprocess/process-lifecycle test effort into this guard-focused PR, and openly reporting that the repository-wide 90% floor remains unmet is honest. The PR should remain test-only and should not be enlarged merely to make the aggregate number green.

The plan does, however, need the concrete corrections below before implementation. In particular, it currently overstates the attainable `server_simple.py` coverage, groups `list_backends` under inapplicable session guards, and does not map corrupt-registry expectations to the tools that actually tolerate corruption.

> **MAJOR — Validation must distinguish passing tests from the knowingly failing coverage gate**
>
> `docs/features/coverage-uplift/plan.md:144-153` says to run `uv run pytest --cov` while also calling the full suite “green except” for `fail_under`. That command exits nonzero while total coverage is below 90%, so it is not a green validation result and may make the PR unmergeable if it is a required CI check.
>
> Keep the incremental scope, but add a passing full-suite command such as `uv run pytest --cov --cov-fail-under=0` (or the project's equivalent), then run/report the configured command separately and explicitly identify its sole expected failure. Before implementation, state how this first PR can be merged while the baseline gate remains red; if CI requires the configured gate, the sequencing/CI policy must be resolved, not hidden by wording.

> **MAJOR — The `server_simple.py` target and remaining-gap inventory are inaccurate**
>
> The current missing-line report includes the intentionally deferred Windows paths, `follow_up_agent` combinations, and `main` paths, totaling roughly 31 lines by themselves (`src/claude_teams/server_simple.py:17-18`, `293-303`, `310-311`, `1608-1708`, `2107`, `2111`). The plan also omits the error branches in `_dir_newest_mtime` at `src/claude_teams/server_simple.py:668-676`. Leaving about 35 of 921 statements uncovered yields approximately 96%, not “~98%” as claimed at `docs/features/coverage-uplift/plan.md:21`.
>
> Either add genuine `_dir_newest_mtime` tests (mock `Path.rglob` and a descendant `stat` to raise `OSError`, asserting the retained/newest mtime) and recalculate the target, or explicitly defer those lines too. In either case, update the module target and success criterion to an evidence-based value. This does not justify pulling `process_manager.py` into scope.

> **MAJOR — `list_backends` has no session guard, and corrupt registries are not universally tolerated**
>
> The grouped tool bullet at `docs/features/coverage-uplift/plan.md:98-101` is too ambiguous to implement correctly. `list_backends` does not call `_active_session_id` at all; it always enumerates `registry.list_available()` and calls `registry.get()` plus the backend metadata methods (`src/claude_teams/server_simple.py:2084-2102`). A “no active session” test is therefore inapplicable and would either assert nothing relevant or require a production change. Cover it with a deterministic fake registry and assert the complete returned backend row and call/order behavior.
>
> Likewise, corrupt `agents.json` is deliberately caught by `resume_session` and `session_info` (`src/claude_teams/server_simple.py:1837-1845`, `1878-1888`), but it is not caught by `list_agents`, `agent_status`, `kill_agent`, or `_message_recipient`, all of which eventually call an unguarded `_load_agents_unlocked` (`src/claude_teams/server_simple.py:328-353`). Do not write tests expecting those tools to tolerate corruption unless production behavior is changed, which this PR forbids. Replace the grouped bullet with a per-tool matrix of setup, expected payload/reason, and the exact branch covered.

> **MINOR — “Path that resolves empty” is not a real filesystem case**
>
> `docs/features/coverage-uplift/plan.md:43,65-67` calls out an “unresolvable path” / “path that resolves empty.” For a nonempty string, `_resolve_path_text` either returns the resolved path or, after `OSError`/`RuntimeError`, returns the expanded input (`src/claude_teams/agent_output.py:417-423`). It does not naturally return an empty string. The `read_*` empty-normalization branches at `src/claude_teams/agent_output.py:57-59` and `90-92` are reachable only by mocking the helper (the literal empty-cwd case returns earlier).
>
> Remove the claim that this is a real path edge. If those defensive branches are retained as unit targets, label them explicitly as contract-isolation tests with `_normalize_path` / `_resolve_path_text` mocked to return `""`; otherwise omit them. No production change is needed.

> **MINOR — The plan includes already-covered cases and misses other named-module branches**
>
> The current coverage data shows that several proposed additions already execute today: no-candidate returns (`agent_output.py:64-65`, `105-106`), Claude string content (`agent_output.py:347-348`; existing `test_read_claude_output_accepts_string_content`), and the too-small truncation marker branch (`agent_output.py:402-404`; existing small-budget tests). These are meaningful behaviors, but duplicating them does not uplift coverage and unnecessarily enlarges the PR.
>
> Reconcile the case list against a fresh missing-lines report before writing tests. For the advertised `agent_output.py` target, explicitly decide whether to cover or defer malformed Codex metadata/payload (`src/claude_teams/agent_output.py:134-139`), per-file `stat()` failure (`204-207`), and an invalid/out-of-range spawn timestamp (`216-219`). The projected reduction and per-module targets should be recalculated from the final list.

> **MINOR — The shared `server_simple` fixture needs a stronger isolation contract**
>
> The risk section at `docs/features/coverage-uplift/plan.md:122-131` correctly identifies `_SESSION_BASE`, `_session_id`, `_pending_recovery`, and `IDENTITY`, but the proposed cleanup/recovery/tool tests also depend on `_TEAMS_BASE`, `_AGENT_SESSION_ID`, `_AGENT_PARENT_NAME`, `_inbox_locks`, the process cwd, and recovery/retention/log-directory environment variables (`src/claude_teams/server_simple.py:49-101`, `267-273`, `546-552`, `692-701`). Patching `Path.home` after `server_simple` has been imported does not relocate `_SESSION_BASE` or `_TEAMS_BASE`, because both were bound at import time.
>
> Require an autouse or explicitly shared fixture for the new file that patches both base directories, sets `_session_id = ""`, `_AGENT_SESSION_ID = ""`, `_pending_recovery = {}`, resets `_inbox_locks`, establishes an isolated cwd, and deletes relevant environment overrides unless a test sets one. Tests that monkeypatch `IDENTITY` or `_AGENT_PARENT_NAME` should do so locally. This prevents accidental recovery from or cleanup of real user state.

> **MINOR — Entrypoint patching must target the imported source symbol**
>
> The entrypoint test at `docs/features/coverage-uplift/plan.md:105-108` is feasible, but patching `claude_teams.server.main` after importing it will not affect the fresh namespace created by `runpy.run_module`. Patch `claude_teams.server_simple.main` before the `runpy` call, then assert it was called exactly once; the fresh execution of `server.py:3` imports that patched object. The re-export test should assert identity (`server.main is server_simple.main` and `server.mcp is server_simple.mcp`), not merely attribute presence.

> **MINOR — Error-path tests need observable assertions and portable fault injection**
>
> The proposed private-helper tests are genuine when they prove fallback semantics, but `OSError` cases should not rely on chmod (unreliable under elevated users) or merely “not raising.” Monkeypatch the relevant `Path.open`, `read_text`, `stat`, `rglob`, or `iterdir` call to raise and assert the documented return plus any preserved state/no-write property. Because `_iter_lines_reverse` is a generator (`src/claude_teams/agent_output.py:363-383`), its `OSError` branch executes only when consumed; assert `list(_iter_lines_reverse(path)) == []`. For skip logic in `_last_codex_message` and `_last_claude_message`, place malformed/newer records after a valid older record and assert that the valid older text is returned.

> **NIT — Avoid duplicate tests where an existing behavior test can be extended**
>
> Existing recovery and follow-up suites already establish much of the fixture vocabulary and neighboring behavior (`tests/test_session_recovery.py`, `tests/test_restart_safety.py`, and the latter half of `tests/test_agent_output.py`). The missing `resume_session("")`, no-session, registry-raises, and corrupt-active-registry branches are valid additions, but names and assertions should identify the distinct branch rather than restating existing “unknown UUID,” “backend reports no resume support,” or “no candidates” tests.

## Required changes before implementation

1. Keep `process_manager.py` deferred, but document a passing validation command and how the knowingly red 90% gate is handled for this PR.
2. Replace the grouped MCP-tool bullet with a per-tool expectation matrix; give `list_backends` a fake-registry behavior test and limit corrupt-registry tolerance assertions to the functions that catch those errors.
3. Correct the `server_simple.py` target and add or explicitly defer `_dir_newest_mtime` error branches.
4. Relabel/remove the impossible natural “resolves empty” case and deduplicate cases already covered by existing tests.
5. Define the full isolation fixture, including `_TEAMS_BASE`, `_AGENT_SESSION_ID`, cwd, inbox locks, and relevant environment variables.
6. Specify that `server_simple.main` is patched for the `runpy` entrypoint test and that every guard/error test asserts an observable return, selection, state change, or absence of a write.

With those changes, the planned tests exercise real defensive behavior and remain an appropriately reviewable first coverage increment. No proposed correction requires changing production behavior.
