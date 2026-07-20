# Pi lead auto-load — implementation

Implements `docs/features/pi-lead-autoload/design.md` (auto-load the wake
extension for every Pi lead at any nesting level, plus the §6 correction that
generalizes the merged wake feature beyond a root `team-lead`).

Branch: `feature/pi-lead-autoload` (worktree `../wt-pi-lead-autoload`, from
`main` @ `4c90a39`, which equals `origin/main`).

## Locked design decision

Every spawned Pi agent gets the wake watcher, but it watches ONLY its OWN
identity. The reader is `AGENT_NAME or team-lead` — the CLI default WITHOUT
`--reader`. `--reader team-lead` is never hardcoded.

## Final design

1. **Auto-load (package.json)** — added `"pi": { "extensions": ["./index.ts"] }`
   so `pi install <dir>` / the `packages` array treats the wake directory as a
   loadable extension (root-lead path). MCP server stays project-scoped in the
   repo `.mcp.json` (not globalized).
2. **Activation guard (index.ts)** — `activate()` returns immediately unless
   `WIN_AGENT_TEAMS_SESSION_DIR` is set (spawned agent, backend-injected) OR
   `WIN_AGENT_TEAMS_LEAD === "1"` (root-lead opt-in). Plain `pi` → true no-op
   (no handlers, no loop).
3. **Backend load (backends/pi.py + server_simple.py)** — `_extension_args` now
   emits a second `-e <wake dir>` alongside the state extension, threaded via
   `request.extra["pi_wake_extension_path"]`. The server resolves it in
   `_pi_wake_extension_dir()` (overridable via
   `WIN_AGENT_TEAMS_PI_WAKE_EXTENSION`) and adds it in `_hook_extra` for the pi
   backend. Nested subagents-as-lead are thus covered with no per-agent setup.
4. **§6 generalization (cli.ts + state-machine.ts)**
   - Dropped the forced `--reader team-lead`. `runWatch` / `runInboxStatus` take
     an optional reader and omit `--reader` when it is `undefined`, letting the
     CLI apply its ambient `AGENT_NAME`-or-`team-lead` default.
   - `WakeMachine` no longer defaults its reader to `team-lead`
     (`readerOverride: string | undefined`). The active identity for inbox-path
     matching is `readerOverride ?? discovery.identity` (`activeReader()`).
   - `discovering()` / `refresh()` gate on "a live session exists"
     (`r.kind === "ok"`) instead of `identity === "team-lead"`; the watched
     inbox binds to the reported `discovery.identity`.
   - Renamed `leadInboxPath` → `ownInboxPath` and `isLeadInboxPath` →
     `isOwnInboxPath`; identity is now a required argument (no `team-lead`
     default).
5. **Docs** — extension README rewritten for the any-nesting-level behavior,
   the activation guard, and both auto-load paths; `index.ts` header docstring
   updated.

## Red → green evidence (TDD)

TypeScript (`vitest`, `pi-extensions/win-agent-teams-wake/`):

- New/changed tests written first against the new API:
  - `test/cli.test.ts`: `ownInboxPath` / `isOwnInboxPath` with an
    `identity` argument, incl. a nested-lead `inbox-worker-1.jsonl` case.
  - `test/state-machine.test.ts`: `machine()` helper drops the `team-lead`
    reader; replaced the old "fails closed when identity is not team-lead" test
    with:
    - "§6: a nested lead (identity=<AGENT_NAME>) reaches WATCHING and wakes on
      its own inbox",
    - "§6: a nested lead ignores a message on the team-lead inbox",
    - "§6: shells out WITHOUT --reader",
    - "§6: an explicit reader override is passed through as --reader".
  - `test/index.test.ts`: activation-guard suite — no-op when neither env is
    set; activates for a spawned agent (session dir); activates for the flagged
    root lead. Existing T9 wiring test now sets `WIN_AGENT_TEAMS_SESSION_DIR`.
  - `test/harness.ts`: `okSessionDir` gained an optional `identity` arg.
- Red: before the src changes these referenced non-existent exports
  (`ownInboxPath`/`isOwnInboxPath`) and asserted behavior the old machine did
  not have.
- Green: after cli.ts / state-machine.ts / index.ts changes — 57/57 tests pass;
  `tsc --noEmit`, `eslint .`, `prettier --check` all clean.

Python (`pytest`, `tests/test_backends/test_pi.py`):

- New tests first: `test_wake_extension_loaded_alongside_state` (both `-e`
  values present) and `test_wake_extension_absent_when_not_provided`.
- Red: `assert r"C:\ext\wat-wake" in e_values` failed
  (`['C:\\ext\\wat-state']`).
- Green: after the `_extension_args` change both pass.

## Validation commands (verbatim results)

- TS `tsc --noEmit`: clean (exit 0).
- TS `vitest run`: 57 passed (5 files).
- TS `eslint .`: clean (exit 0).
- TS `prettier --check`: all files use Prettier code style.
- `uv run ruff check .`: All checks passed.
- `uv run ruff format --check .`: 52 files already formatted.
- `uv run ty check`: All checks passed.
- `uv run pytest -q`: 662 passed, 3 skipped, **3 failed** — all three failures
  are in `tests/test_watch_command_discovery.py`
  (`test_watch_argv_executes_and_times_out_quietly`,
  `test_watch_command_bash_executes_and_times_out_quietly`,
  `test_watch_argv_runs_from_unrelated_cwd_without_pythonpath`). They spawn
  `python -m claude_teams.cli watch <dir> --timeout 1` as a real subprocess and
  hit a 10 s `TimeoutExpired`. **Pre-existing and unrelated**: they reproduce on
  the base tree and this change does not touch the `watch` CLI command. Surfaced
  here, not fixed (out of scope; a real behavior/env issue in the watch CLI, not
  cosmetic).

## Deviations from design.md

- Design §5 task 5 (a `pi-lead` launcher subcommand) and task 7 (optionally
  flipping `.mcp.json` `lifecycle` to `lazy`) were left as documented options
  only; the `alias pi-lead='WIN_AGENT_TEAMS_LEAD=1 pi'` is documented in the
  extension README rather than shipped as code. No `.mcp.json` change made
  (design recommends keeping it as-is).
- `DEFAULT_READER = "team-lead"` is retained in `cli.ts` solely as the
  `parseInboxStatus` JSON-field fallback; it no longer drives any `--reader`
  flag or gate.
- Independent cross-family (Codex) reviews deferred — Codex unavailable.
