# Type-cleanup plan re-review — iteration 2

## Summary

**Approved.** The revised plan resolves all three iteration-1 blockers. The
live baseline remains 44 `ty` diagnostics; the revised 32/12 split still
accounts for every one. The proposed Increment 1 edits are now specific enough
to implement and should preserve behavior. I found two small documentation
follow-ups, neither of which blocks implementation.

## Prior Findings Resolution

| Prior finding | Status | Evidence and verification |
|---|---|---|
| `_safe_float` guard would reject valid float-convertible objects such as non-zero `Decimal` | **Resolved** | The updated plan says: “**Type-only fix, runtime expression unchanged.** Bind a local `coerced: Any = value or 0.0` then `return float(coerced)`” ([plan.md:32](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:32)). This evaluates the same `value or 0.0` expression once, then passes the same object to `float`; assigning to `Any` has no runtime effect. It therefore preserves the existing implementation at [server_simple.py:906](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:906)-[911](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:911), including `Decimal`, `Fraction`, bytes, custom `__float__`, bool, `0`, empty string, and `None`. `float(coerced)` accepts `Any`, so this removes the error at line 909. The mandated `Decimal("2.5")`, `""`, `None`, and `0` regression cases are correctly listed ([plan.md:108](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:108)-[109](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:109)). |
| `_content_text` narrowing was vague and did not prove removal of the `Never` key/join errors | **Resolved** | The plan now explicitly says: “**Rewrite the comprehension as an explicit loop**” and uses `mapping = cast("dict[str, object]", item)`, a typed `text` local, and `parts: list[str]` ([plan.md:30](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:30)). This addresses the exact baseline failure: even after `isinstance(item, dict)`, the current code has unknown dict keys/values at [agent_output.py:351](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:351)-[360](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:360). After the cast, `.get("type")` and `.get("text")` are legal; `isinstance(text, str)` narrows before appending, so `"".join(parts)` receives `list[str]`. For the JSON-derived ordinary dictionaries used by the callers ([agent_output.py:260](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:260)-[280](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/agent_output.py:280)), the loop preserves the old filter order and concatenation. Ensure `cast` is imported from `typing`. |
| Fake `last_request` proposal left an `object` assignment and optional `extra` errors | **Resolved** | The updated plan mandates all three needed changes: import `SpawnRequest`, change the fake parameter and field types, and “bind `request = backend.last_request`, `assert request is not None`, and ... `assert request.extra is not None`” ([plan.md:37](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:37)). That directly fixes the current `request: object` assignment in [test_hooks_integration.py:15](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:15)-[32](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_hooks_integration.py:32) and [test_spawn_agent_watch_contract.py:11](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:11)-[28](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/tests/test_spawn_agent_watch_contract.py:28). Both assertions are required because `SpawnRequest.extra` is `dict[str, str] | None` ([contracts.py:202](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/contracts.py:202)-[214](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/contracts.py:214)). This clears the four listed read sites without a residual object-parameter error. `contracts` imports neither test module nor `server_simple`, so there is no cycle. |

## Additional Verification

### `Backend.resume`

**Resolved / approved.** The plan now specifies adding “**only** `resume(...)`
... Do **not** add `supports_resume`/`build_resume_command`” and records the
third-party structural-protocol compatibility impact ([plan.md:34](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:34)). That is exactly the minimal remedy for the call at
[server_simple.py:1703](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1703); `supports_resume` remains safely dynamic through `getattr` at [server_simple.py:1597](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/server_simple.py:1597).

The only built-in registry entries are Claude Code and Codex
([registry.py:15](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/registry.py:15)-[18](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/registry.py:18)); both inherit `BaseBackend`, whose `resume` method is present at
[process_base.py:89](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_base.py:89)-[93](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_base.py:93). No official concrete gap is masked.

### Increment 2 disposition

**Reasonable.** The correction explicitly says the diagnostics “are **not all
Windows-only**” and identifies the Windows, tmux, and Linux-terminal locations
([plan.md:45](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:45)-[49](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:49)). It also correctly recognizes that a result-only `Popen` cast cannot cure the
`**kwargs: object` overload diagnostics ([plan.md:54](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:54)); the current `_popen` signature and uses confirm this
([process_manager.py:573](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:573)-[608](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:608)).

Deferring the three apparently easy `_tracked_alive` entries with the file is
also justified. Each subclass currently keeps `info: object`
([process_manager.py:634](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:634),
[1366](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:1366),
[1748](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:1748)) because the shared abstract method accepts `object`
([process_manager.py:234](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:234)-[241](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/src/claude_teams/backends/process_manager.py:241)). Narrowing a method parameter in an override can violate parameter contravariance/LSP. A focused decision—typed generic base design or local casts—is preferable to a rushed three-line change.

### Coverage / live spot-check

`uv run ty check` still reports the unmodified 44-diagnostic baseline. The
inventory arithmetic remains correct: Increment 1 is 32 and Increment 2 is 12,
for 44. The plan's expected post-Increment-1 result of 12 diagnostics, all in
`process_manager.py`, is therefore consistent with the live inventory.

## New Gaps / Risks

- The Scope still says Increment 2 is deferred so “the **Windows-runtime**
  typing gets its own focused review” ([plan.md:13](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:13)-[16](/home/mikael/code/agentic-coder-teams-mcp/.claude/worktrees/unruffled-meitner-991ba7/docs/features/ty-diagnostics-zero/plan.md:16)). This conflicts with the correctly revised statement below that the subsystem also contains tmux and Linux-terminal diagnostics. Change it to “process-manager typing” for consistency.
- The new `_safe_float` regression test has no named destination in “Files
  affected.” Add the chosen test module to that list and import `Decimal` there.
  This is a planning/documentation omission, not a correctness blocker.
- Implementation must add the ordinary `typing` imports (`Any`, `cast`, and
  the existing plan's `Literal`/`cast` imports) and let Ruff sort them. These
  are routine integration details; no pytest or Ruff regression is inherent in
  the planned changes.

## Score

**94/100.** The substantive correctness blockers are resolved. The remaining
issues are small documentation/scope wording omissions.

## Ready for Implementation?

**Yes.** Implement Increment 1 as specified, including the required focused
`_safe_float` regression test, then run `uv run ty check`, `uv run pytest -q`,
and `uv run ruff check src/ tests/`. Clean up the two documentation details
above while doing so.
