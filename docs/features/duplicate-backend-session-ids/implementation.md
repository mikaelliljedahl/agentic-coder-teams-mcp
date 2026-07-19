# Implementation: deterministic backend_session_id binding for concurrent Claude Code agents

## Summary

Mirrored the proven Codex correlation-token mechanism for the Claude Code
backend so that concurrent agents spawned in the same `cwd` bind to distinct
`backend_session_id` values instead of racing onto the newest-mtime transcript.

Four production changes were made, exactly as described in `plan.md`.

## Final design

1. **`src/claude_teams/agent_output.py`**
   - Renamed the module constant `_CODEX_CORRELATION_PREFIX` to the shared
     `_CORRELATION_PREFIX` (behavior-preserving; same value `"wat-corr:"`).
   - `codex_correlation_token` now references `_CORRELATION_PREFIX`.
   - Added `claude_correlation_token(agent_id: str) -> str` returning
     `f"{_CORRELATION_PREFIX}{agent_id}"` with a docstring.
   - `read_claude_output` gained a keyword-only `correlation_token: str | None
     = None`. After `candidates` are built and before `max(...)`, when
     `backend_session_id is None and correlation_token`, candidates are filtered
     to those whose transcript contains the token via the existing
     `_rollout_contains_token`; if any match, the filtered set is used, else the
     unfiltered set is kept. This parallels `_matching_codex_rollouts`.

2. **`src/claude_teams/backends/claude_code.py`**
   - Imported `claude_correlation_token`.
   - `build_command` now appends a correlation suffix to the value returned by
     `_prompt_arg(request)` (so the marker lands in the first recorded user
     message even when a prompt file is used).
   - Added a private static helper `_correlation_suffix(request)` producing
     `"\n\n[win-agent-teams correlation id: {token} — internal marker, ignore
     this line]"` with `token = claude_correlation_token(request.agent_id)`.
   - `build_resume_command` is unchanged (resume already knows the session id).

3. **`src/claude_teams/server_simple.py`**
   - Imported `claude_correlation_token`.
   - `_read_agent_output`'s `claude-code` branch now passes
     `correlation_token=claude_correlation_token(f"{agent.get('name')}@{agent.get('session_id')}")`,
     matching the spawn-time `SpawnRequest.agent_id` format.

## TDD evidence

### Red

Tests were written first in `tests/test_agent_output.py`. Initial focused run
failed at collection:

```
ImportError: cannot import name 'claude_correlation_token' from
'claude_teams.agent_output'
```

### Green

After implementing the four changes:

```
uv run pytest tests/test_agent_output.py -q
86 passed in 12.87s
```

Five new tests cover all plan cases:
- `test_read_claude_output_disambiguates_concurrent_agents_by_token`
- `test_read_claude_output_falls_back_when_token_absent`
- `test_read_claude_output_ignores_token_when_session_id_known`
- `test_claude_build_command_embeds_correlation_token` (also asserts the resume
  command does NOT embed it)
- `test_claude_build_command_embeds_token_with_prompt_file`

Five pre-existing `tests/test_backends/test_claude_code.py` build-command tests
that asserted the exact prompt argv were updated to accommodate the trailing
marker (assert `startswith(prompt)` + marker present), matching how the Codex
suite already handles its equivalent suffix.

## Validation commands

- `uv run ruff format --check .` -> `50 files already formatted`
- `uv run ruff check .` -> `All checks passed!`
- `uv run ty check` -> `All checks passed!`
- `uv run pytest` -> `3 failed, 612 passed, 3 skipped`

### Pre-existing breakage (NOT caused by this change)

The 3 failures are all in `tests/test_watch_command_discovery.py`
(`test_watch_argv_executes_and_times_out_quietly`,
`test_watch_command_bash_executes_and_times_out_quietly`,
`test_watch_argv_runs_from_unrelated_cwd_without_pythonpath`). They fail with
`subprocess.TimeoutExpired` when spawning `python -m claude_teams.cli watch`.
Verified pre-existing by stashing all working-tree changes and rerunning: the
same 3 tests fail on the clean tree (`3 failed, 8 passed, 1 skipped`). They are
environmental `watch` CLI subprocess timeouts unrelated to this feature.

## Review

Independent post-implementation review was performed by a **Codex** agent
(`gpt-5.6`, high reasoning) via the win-agent-teams MCP — the opposite tool from
the Claude implementer, satisfying separation of duties. Result: **APPROVED,
99/100, zero blockers, zero warnings** (see `implementation-review.md`). Per the
user's direction and the small size of the change, the separate pre-code plan
review (workflow step 2) was folded into this post-implementation Codex review,
which validated the implementation against `plan.md` item-by-item.

## Deviations from plan

None. All four changes were implemented as specified. The only work beyond the
plan's explicit list was updating the five pre-existing `test_claude_code.py`
build-command assertions, which the plan's marker-injection change necessarily
required (the plan's own test case 4/5 assert the new behavior).
