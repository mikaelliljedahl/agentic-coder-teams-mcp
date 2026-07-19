# Increment 1 implementation review

## Summary

The implementation delivers the approved Increment 1 type cleanup: `ty` is now
at **12 diagnostics**, all in the deferred `process_manager.py` work, and Ruff
is clean. The changed-test selection passes (**195 passed**). The full suite
was run; its pytest process completed and the failure cache is empty (the
implementation reports 497 passed / 2 skipped).

I found no new type error, production regression, or scope expansion. One
extremely narrow semantic edge exists in `_content_text` for hostile/custom
`dict` subclasses whose `get` and `__getitem__` disagree; the real callers
operate on JSON-decoded built-in dictionaries, so this is a note rather than a
blocker.

## Score (0-100)

**96/100**

## Prior Findings

| Iteration-2 plan blocker | Status in code | Evidence |
|---|---|---|
| `_safe_float` must not reject `Decimal`/`Fraction`/bytes/custom float-convertible values | **Confirmed resolved** | The implementation uses `float(cast(Any, value or 0.0))` at [server_simple.py:907](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:907)-[916](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:916). `cast` returns its argument unchanged at runtime, so the expression evaluates and converts exactly the same object as the old `float(value or 0.0)`. The regression tests include non-zero `Decimal`, `Fraction(7, 2)`, and `True` with correct expected floats at [test_safe_float.py:18](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_safe_float.py:18)-[32](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_safe_float.py:32). |
| `_content_text` needs an actual typed mapping/local solution for `Never` keys and `object` join parts | **Confirmed resolved** | The explicit loop guards a real `dict`, casts it to `dict[str, object]`, binds `text`, requires `str`, and appends only to `parts: list[str]` at [agent_output.py:344](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:344)-[363](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:363). `ty` reports none of the four former errors. |
| Fake `last_request` requires typed `spawn`, typed field, request assertion, and optional-`extra` assertion | **Confirmed resolved** | Both fakes now declare `last_request: SpawnRequest | None` and `spawn(request: SpawnRequest)` ([test_hooks_integration.py:16](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:16)-[33](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:33), [test_spawn_agent_watch_contract.py:12](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:12)-[29](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:29)). Each relevant reader binds and asserts both `request` and `request.extra` before use ([hooks:63](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:63)-[66](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:66), [watch:116](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:116)-[120](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:120)). |

## Correctness/Behavior Verification

### Production edits

- **`_content_text`:** For ordinary dictionaries, the new loop has the same
  acceptance predicate and ordering as the old comprehension: list input only;
  each dictionary must have matching `type` and a string `text`; the accepted
  strings are joined in original list order ([agent_output.py:349](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:349)-[363](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:363)). The production callers obtain their content from decoded transcript payload/message dictionaries
  ([agent_output.py:260](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:260)-[280](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:280)), where `.get("text")` and `["text"]` agree. See the minor custom-subclass caveat in New Regressions.

- **`_annotate`:** Its narrow signature is valid. There are exactly five calls:
  [server_simple.py:1310](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1310),
  [1347](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1347),
  [1512](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1512),
  [1567](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1567), and
  [1749](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1749). Each passes a nested `_do_*` declared `-> dict` through generic `run_blocking`; no non-dict caller exists. The merge/no-merge logic remains unchanged at [server_simple.py:618](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:618)-[625](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:625).

- **`_safe_float`:** Behavior is equivalent for the requested cases. Non-zero
  `Decimal`, `Fraction`, and bytes remain truthy and reach `float`; `None`,
  `""`, and zero become `0.0`; `True` remains an `int`-like truthy value and
  converts to `1.0`. Invalid truthy strings still raise `ValueError` and return
  `0.0` under the unchanged exception handler ([server_simple.py:909](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:909)-[916](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:916)).

- **`permission_mode`:** Both `SpawnRequest` sites now apply a type-only
  Literal cast ([server_simple.py:1267](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1267)-[1272](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1272),
  [1705](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1705)-[1710](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1710)). `cast` returns the original string, exactly like the removed comment had no runtime effect. No validation behavior was added.

- **`payload`:** `payload: dict[str, object]` precisely matches its subsequent
  insertion of `status.get(...)` object values at
  [server_simple.py:1083](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1083)-[1099](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1099). A local variable annotation has no data-path effect.

- **Backend protocol:** The diff adds only `resume` to `Backend` at
  [contracts.py:310](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/contracts.py:310)-[316](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/contracts.py:316); it does not add `supports_resume` or `build_resume_command`. The built-in registry contains only Claude Code and Codex
  ([registry.py:15](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/registry.py:15)-[18](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/registry.py:18)); both subclass `BaseBackend`, which implements `resume`
  ([process_base.py:89](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_base.py:89)-[93](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_base.py:93)). Their `supports_resume` and concrete resume-command implementations also remain present.

## Scope

The modified production files exactly match the plan: `agent_output.py`,
`server_simple.py`, and `backends/contracts.py`. The changed test files match
the approved list, including the plan-authorized new
`tests/test_safe_float.py`. `git diff --name-only` contains no unexpected code
or test file. No `process_manager.py`, CLI/watch code, dependency, configuration,
or unrelated behavior was touched.

## Test Fidelity

- The fake-backend assertions are strengthening assertions, not masking:
  they run after the real `spawn_agent` call and fail if the fake did not
  receive a request or if the expected `extra` payload was absent. The original
  checks for the same hook paths, prompt, and prompt-file data remain intact
  ([test_hooks_integration.py:59](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:59)-[70](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:70),
  [test_spawn_agent_watch_contract.py:106](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:106)-[120](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:120)).
- The five `dict[str, object]` record annotations are type-only; their
  `update(overrides)` and persisted data are unchanged (representatively
  [test_agent_status.py:23](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_agent_status.py:23)-[35](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_agent_status.py:35)). `_read(**kwargs: Any)` remains the same forwarding call
  ([test_read_messages.py:13](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_read_messages.py:13)-[14](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_read_messages.py:14)).
- `Tool | None` assertions correctly make a missing registered tool fail the
  test instead of dereferencing it accidentally ([test_session_recovery.py:249](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_session_recovery.py:249)-[252](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_session_recovery.py:249),
  [test_tool_descriptions.py:65](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_tool_descriptions.py:65)-[67](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_tool_descriptions.py:67)).
- The two `# ty: ignore[unresolved-attribute]` directives are justified:
  these tests deliberately synthesize Windows `OSError.winerror` values to
  exercise the breakaway fallback, while the checker uses Linux stubs
  ([test_process_manager_windows.py:120](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_backends/test_process_manager_windows.py:120)-[146](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_backends/test_process_manager_windows.py:120)).
- Replacing `(time.time() - ..., ) * 2` with a once-computed `old` and
  `(old, old)` preserves the exact two equal timestamps while satisfying the
  `os.utime` tuple shape ([test_cleanup.py:113](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_cleanup.py:113)-[114](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_cleanup.py:114)).
- `test_safe_float` has 13 test cases: Decimal `2.5`, Fraction `7/2 -> 3.5`,
  true `-> 1.0`, ordinary numeric/string cases, six falsy cases, and an invalid
  string fallback ([test_safe_float.py:18](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_safe_float.py:18)-[40](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_safe_float.py:40)). All expected values are correct and specifically prevent the rejected isinstance-guard regression.

## ty/ruff/pytest confirmation

- `uv run ty check` independently reports **12 diagnostics**. All are in
  `src/claude_teams/backends/process_manager.py`, at the planned ctypes,
  `_popen`/log-handle, and three `_tracked_alive` sites; there are no
  Increment-1 or new-file diagnostics.
- `uv run ruff check src/ tests/` independently reports **All checks passed**.
- The full `uv run pytest -q` process completed during review with an empty
  `.pytest_cache/v/cache/lastfailed`; the directly rerun changed-test selection
  passed **195 tests in 16.79s**. This is consistent with the implementation's
  full-suite report. The stated `test_cli_watch` issue is plausibly a
  pre-existing timing flake: the untouched test creates a thread that sleeps
  0.07 seconds and then waits under a poll-driven two-second watch
  ([test_cli_watch.py:558](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_cli_watch.py:558)-[590](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_cli_watch.py:590)). Deferring it is reasonable because this change touches no CLI/watch path. Its claimed clean-main history was not independently reproducible without stashing the requested implementation, so treat that attribution as supporting context rather than proof.

## New Regressions

No actionable regression found.

Minor semantic edge: the old comprehension checked `item.get("text")` but
then appended `item["text"]`; the new code appends the result of `.get()`
([agent_output.py:355](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:355)-[360](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:360)). A pathological `dict` subclass could make those methods disagree or give them side effects. That differs from the old helper for that unsupported shape. JSON decoding supplies normal dictionaries to the actual callers, so it has no production impact; if exact generic-dict-subclass parity is desired, append a type-cast of `mapping["text"]` after the existing string check instead.

## Final Verdict

**APPROVED WITH NOTES — 96/100.** Increment 1 is correct, scoped, type-clean
to the intended 12 deferred diagnostics, lint-clean, and covered by the changed
test suite. The custom-dict caveat and unverified clean-main flake attribution
are non-blocking notes.
