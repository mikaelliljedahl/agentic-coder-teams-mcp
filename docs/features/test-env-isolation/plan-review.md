# Plan review: test-env-isolation

**Reviewer:** an independent Claude general-purpose subagent with a fresh
context, run in the cloud session.

**Deviation from the CLAUDE.md workflow:** the required *opposite-family*
reviewer (GPT/Codex) was **not available**. The container has no Codex CLI
and no OpenAI credentials. This review is a best-available substitute, and a
GPT/Codex review is still owed before merge. The PR says so too.

The reviewer ran the full suite on a scratch copy with the originally proposed
function-scoped sweep and the native flags, `WIN_AGENT_TEAMS_LINUX_LAUNCHER=tmux`,
`AGENT_NAME`, `CODEX_HOME`, `TMUX`, `STATE_HOOKS=0` and
`CLAUDE_TEAMS_PERMISSION_MODE` exported. Result: 1 failed, 2821 passed.

## Findings and dispositions

1. **Blocking: import-time launcher selection.** `process_manager` picks the
   global manager from `WIN_AGENT_TEAMS_LINUX_LAUNCHER` at import time
   (`process_manager.py:3767-3770`).
   `test_defaults_to_terminal_on_posix_and_windows_manager_on_windows` re-reads
   the variable at test time, which gives a mismatch once the fixture has
   cleared it.
   **Accepted, fixed differently:** the design now also scrubs at conftest
   load, before any `claude_teams` import, so import-time state and test-time
   reads agree. This also fixes the wider class the probe found:
   `EXTERNAL_ONLY=1` → 18 failures from import-time tool registration. The
   test itself is unchanged.
2. **Non-blocking: module-scoped fixture copies `os.environ` before autouse
   runs** (`test_native_downstream_tool_text.py:21-23`). **Accepted:** the
   load-time scrub fixes it. The predicate is a single shared helper
   (`is_ambient_agent_env`).
3. **Non-blocking: `server_simple._AGENT_NAME` not reset.** **Accepted:** now
   reset in the autouse fixture.
4. **Non-blocking: other host vars read by production** (`TMUX`, `DISPLAY`,
   `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, `USE_WINDOWS_TERMINAL`,
   `HERDR_CONFIG_PATH`, `GOOSE_RECIPE_PATH`). **Accepted as recommended:**
   they are left alone and listed under "deliberately not cleared".
5. **Root cause verified.** Accepted.
6. **Prefix sweep safe (verified).** Accepted.
7. **Nit: guard uses a `CLAUDE_TEAMS_` prefix.** **Accepted:** the plan now
   says "prefix".
8. **Nit: wording of the "mirrors `queue_environment`" claim.** **Accepted:**
   reworded to "inspired by".
