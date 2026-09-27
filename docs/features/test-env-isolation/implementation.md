# Implementation: test-env-isolation

## Final design

`tests/conftest.py` defines one predicate, `is_ambient_agent_env(key)`. It
matches:

- the `WIN_AGENT_TEAMS_*` prefix;
- the `CLAUDE_TEAMS_*` prefix;
- `AGENT_NAME`, `AGENT_SESSION_ID`, `AGENT_PARENT_NAME`, `CODEX_HOME`,
  `CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`.

Two layers use the predicate:

1. **Load-time scrub.** When `conftest.py` loads, it deletes matching keys
   from `os.environ` before any `claude_teams` import. That covers
   import-time reads and module- or session-scoped fixtures.
2. **Autouse fixture.** `_clear_inherited_agent_env` now sweeps with the same
   predicate via `monkeypatch.delenv`, replacing its hand-picked list. It
   also resets `server_simple._AGENT_NAME`.

New guard test: `tests/test_env_isolation.py::test_ambient_agent_env_is_cleared`.
It asserts that no ambient variable is visible inside a test. It uses its own
list, not the conftest predicate, so a regression in the predicate is caught.

## Deviations from the original plan

- The plan first proposed a function-scoped sweep only. Plan review finding 1
  and the per-variable probe showed that import-time reads need clearing
  before import, so the load-time scrub was added. The plan was revised
  before implementation.
- `_AGENT_NAME` reset added (plan review finding 3).

## Red / green evidence (Linux, base `1388c37`)

| Run | Before | After |
|---|---|---|
| `uv run pytest` (clean env) | 2821 passed, 14 skipped | 2840 passed, 14 skipped |
| `NATIVE_WAKE=1 NATIVE_DOWNSTREAM=1` | **5 failed** (the 5 named tests) | green |
| `WIN_AGENT_TEAMS_SESSION_DIR=/tmp/x` only | **6 failed** (`test_join_team`, `test_watch_command_discovery`) | green (in broad run) |
| `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` only | **18 failed** (tool-description, direction-guard, native-wake tests) | green (in broad run) |
| `CODEX_HOME=/tmp/ch` only | 0 failed on the existing suite (guard test red) | green |
| Guard test with ambient vars | **red** | green |
| Broad ambient run (native flags + sub-flags, `SESSION_DIR`, `EXTERNAL_ONLY`, `STATE_HOOKS=0`, `PARENT_ID`, `CODEX_DIRECT_LAUNCH`, `LINUX_LAUNCHER=tmux`, `AGENT_*`, `CODEX_HOME`, `CLAUDE_CODE_MESSAGING_*`, `CLAUDE_TEAMS_PERMISSION_MODE`) | — | **2840 passed, 14 skipped** |

## Tests that need a flag set it explicitly

A grep of `tests/` for the native flags shows every consumer does one of
three things:

- sets the flag via `monkeypatch.setenv`;
- passes an explicit mapping, e.g. `native_wake.enabled(..., {...})`;
- builds a subprocess env with the flag assigned, e.g.
  `test_tool_descriptions.py:298`, `test_codex_lead_wake.py:742`,
  `test_native_downstream_tool_text.py:24`.

A full run with nothing set is green, which confirms it.

## Validation commands

```bash
uv run ruff format --check .   # 114 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 2840 passed, 14 skipped (unset)
env WIN_AGENT_TEAMS_NATIVE_WAKE=1 WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1 ... uv run pytest
                               # 2840 passed, 14 skipped (broad ambient set)
```

Not run: the Windows suite and the Lubuntu VM run (this was done in a Linux
cloud container). CI covers Linux.

Post-review addition (implementation-review finding 5): parametrised tests
of `is_ambient_agent_env` itself, so the predicate is exercised in CI, where
the ambient guard passes trivially. Final count: 2840 passed, 14 skipped,
both clean and with the broad ambient set exported.
