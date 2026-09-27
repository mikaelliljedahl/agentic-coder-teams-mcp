# Isolate the test suite from ambient win-agent-teams environment

## Scope

Test-only change. Make `uv run pytest` produce the same result regardless of
win-agent-teams / agent-identity variables exported in the developer's shell.
No production behaviour changes.

## Observed failure

Claude Desktop's MCP config sets `WIN_AGENT_TEAMS_NATIVE_WAKE=1` and
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`, and both leak into every agent shell it
launches. With those exported, five tests fail locally (CI never sets them, so
CI is green):

- `tests/test_agent_output.py::test_spawn_agent_persists_output_lookup_metadata`
- `tests/test_backends/test_base_runtime.py::TestBaseBackendSpawn::test_calls_process_manager_with_command_and_env`
- `tests/test_backends/test_base_runtime.py::TestBaseBackendSpawn::test_env_values_are_passed_unquoted_to_process_manager`
- `tests/test_backends/test_codex.py::TestCodexMcpIdentity::test_build_command_injects_identity_env_override`
- `tests/test_join_team.py::test_external_only_mode`

Reproduced on Linux at `1388c37`:
`WIN_AGENT_TEAMS_NATIVE_WAKE=1 WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1 uv run pytest`
→ `5 failed, 2816 passed, 14 skipped`.

## Current behaviour (why)

`native_wake.native_wake_env()` / backend spawn code forward the native flags
into the child env and spawn record, so tests asserting the *exact* env dict or
record contents see extra keys. `tests/conftest.py` already has an autouse
fixture, `_clear_inherited_agent_env`, that `delenv`s the identity variables
(`AGENT_*`) and a hand-picked subset of `WIN_AGENT_TEAMS_*` wake flags, but the
native flags were added later (#75) and never joined that list. A hand-picked
list will keep drifting every time a new flag is added.

## Proposed design

*Revised after plan review (see `plan-review.md`, findings 1–3).*

A shared predicate `is_ambient_agent_env(key)` in `tests/conftest.py` matches:

- every `WIN_AGENT_TEAMS_*` name (native flags and their per-backend
  sub-flags, `SESSION_DIR`, `EXTERNAL_ONLY`, launcher/terminal selection,
  timing knobs, and any future flag);
- every `CLAUDE_TEAMS_*` name (`PERMISSION_MODE`, `AGENT_CAPABILITY`);
- `AGENT_NAME`, `AGENT_SESSION_ID`, `AGENT_PARENT_NAME`, `CODEX_HOME`,
  `CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`.

The list is inspired by the scrub production does in
`native_wake.queue_environment`.

Two layers use the predicate:

1. **Load-time scrub.** When `conftest.py` loads, before any `claude_teams`
   import, it deletes matching keys from `os.environ`. This is required
   because several modules read these variables at import time:
   `server_simple` reads identity and `EXTERNAL_ONLY` tool registration,
   and `process_manager` picks the Linux launcher. Module- and session-scoped
   fixtures also run before function-scoped autouse fixtures, for example the
   module-scoped `tools` fixture in `test_native_downstream_tool_text.py`.
   Probe evidence: `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` alone fails 18 tests,
   and `WIN_AGENT_TEAMS_SESSION_DIR` alone fails 6. Neither is fixable by
   a function-scoped fixture alone.
2. **Per-test autouse fixture** (`_clear_inherited_agent_env`, extended). It
   `monkeypatch.delenv`s every matching key still present. This catches
   direct `os.environ` writes by earlier tests. It replaces the old
   hand-picked list, and it also resets `server_simple._AGENT_NAME`
   alongside the globals it already reset.

Tests that need a flag set it explicitly *after* both layers
(`monkeypatch.setenv`, or an explicit `env` dict for subprocesses / pure
functions taking a mapping).

Deliberately **not** cleared: generic OS, desktop and multiplexer host state
(`HOME`, `PATH`, `COMSPEC`, `LC_*`, `TMUX`, `DISPLAY`, `WAYLAND_DISPLAY`,
`XDG_RUNTIME_DIR`, `USE_WINDOWS_TERMINAL`, `HERDR_CONFIG_PATH`,
`GOOSE_RECIPE_PATH`). These describe the host, not agent identity. Tests that
depend on them set them explicitly, and a full run with `TMUX` exported is
green.

## Files affected

- `tests/conftest.py` — the fixture.
- `tests/test_env_isolation.py` — new regression test.
- `docs/features/test-env-isolation/*` — this record.

## Risks

- A test that silently depended on an ambient variable would start failing.
  Mitigation: the full suite is green in CI where none of these are set, so
  clearing them locally only converges on CI's environment.
- Mutating `os.environ` at conftest load is process-wide for the pytest run.
  That is intended, and it only affects the test process.
- Prefix sweep also removes `WIN_AGENT_TEAMS_LOG_DIR`; tests then use the
  default/tmp-patched location, which is what CI does.

## Test cases

- **Red:** new `tests/test_env_isolation.py` asserts that, inside a test, no
  `WIN_AGENT_TEAMS_*`, `AGENT_NAME/SESSION_ID/PARENT_NAME`, `CODEX_HOME`,
  `CLAUDE_CODE_MESSAGING_*`, `CLAUDE_TEAMS_*` variable is present. Run it with
  those exported → fails (along with the five named tests).
- **Green:** after the fixture change, the new test and the five named tests
  pass with the variables exported.
- Full gates (`ruff format --check`, `ruff check`, `ty check`, `pytest`) with
  flags unset and with a broad set of ambient variables exported.
- Verify every test that needs a native flag sets it explicitly (grep +
  full-suite green with flags unset is the empirical check).
