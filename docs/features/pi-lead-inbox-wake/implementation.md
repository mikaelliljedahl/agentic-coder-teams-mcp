# Implementation — Pi-lead inbox wake

Branch: `feature/pi-lead-inbox-wake` (worktree
`agentic-coder-teams-mcp.pi-lead-inbox-wake`), on `main` @ `3085295`.
Implements the APPROVED-WITH-NOTES plan ([plan.md](plan.md) v4, review score 96 —
[plan-review.md](plan-review.md)). Implementer: Claude subagents (TDD); reviewer of
this implementation: Codex (see [implementation-review.md](implementation-review.md)).

## Final design (as built)

Two components, matching plan §3:

1. **Python read-only CLI surface** (`src/claude_teams/cli.py`,
   `src/claude_teams/server_simple.py`) — additive, backward-compatible:
   - `session_info()` now returns `session_dir` (active → `str(_session_dir(id))`,
     no-session → `""`).
   - `win-agent-teams session-dir` — discovery-only (`create=False`); exit 0 one
     tab line `id\tdir\tidentity`; exit 3 empty when none; exit 1 stderr on error.
   - `win-agent-teams inbox-status <dir> [--reader]` — non-consuming generation
     probe; emits `inbox-status/1` JSON `{senders:{from:{total,cursor,unread}}}`;
     exit 4 for bad/outside-base dir.
   - `win-agent-teams watch … --reader NAME` — additive reader override; omitted →
     exact prior env behavior. Both `--reader` paths validate via
     `_require_safe_reader`, reusing `claude_teams.hooks._SAFE_AGENT_RE`.

2. **Pi TypeScript extension** (`pi-extensions/win-agent-teams-wake/`) — pinned to
   `@earendil-works/pi-coding-agent@0.80.10`, written against the API in plan §10
   (verified against the installed `types.d.ts`/`exec.d.ts`). Modules: `cli.ts`
   (shell-out wrappers + strict/tolerant parsers), `generation.ts` (ACK contract),
   `state-machine.ts` (`WakeMachine` single-flight loop with DISCOVERING/WATCHING/
   WAKE_PENDING/ACK_WAIT/ACK_STALLED), `lifecycle.ts` (owned `AbortController` on
   `session_start`, aborted on `session_shutdown`), `index.ts` (`ExtensionFactory`).
   Injection: `sendMessage({customType:"win-agent-teams/wake", …, display:true},
   {triggerTurn:true, deliverAs:"steer"})`, exactly once per generation.

## Red → green evidence

**Part 1 (Python):** new tests failed RED first (13 failures — missing subcommands
`SystemExit(2)`, missing `session_dir` key), then GREEN: 56 new tests pass
(`tests/test_cli_watch.py`, `tests/test_session_recovery.py`) covering T1, T2, T2b,
T3, T3b.

**Part 2 (TS):** suites failed RED first (4 files, missing `src/` modules), then
GREEN: 43 tests across 4 suites covering T4–T11 (incl. the blocker regressions:
T5b one-injection/no-rearm, T5d generation-ACK race, T7/T7b/T7c ACK_STALLED, T10/T11
lifecycle).

## Validation commands (re-run to reproduce)

Python (repo root, needs `uv`):
```
uv run pytest -q                     # 497 passed, 2 skipped
uv run ruff check .                  # All checks passed
uv run pytest tests/test_cli_watch.py tests/test_session_recovery.py -q   # 56 passed
```
TypeScript (`pi-extensions/win-agent-teams-wake/`, needs node/npm):
```
npm install
npx tsc --noEmit                     # clean
npx vitest run                       # 50 passed (5 files)
npx eslint .                         # clean
# Prettier: the whole-tree `npx prettier --check .` FAILS on `.eslintrc.cjs`
# and README.md (not covered by the package format config). The gate that the
# package actually enforces is the scoped script, which is green:
npx prettier --check "src/**/*.ts" "test/**/*.ts"   # All matched files use Prettier code style
```

## Quality gates (whole repo — honest)

- **Python full pytest: GREEN** — 500 passed, 2 skipped (independently re-run).
- **`ruff check .`: GREEN.**
- **`uv run ty check` (CI gate, `.github/workflows/ci.yml`): RED — 44 diagnostics,
  ALL pre-existing.** Verified by stashing this feature's Python changes: the count
  is 44 with and without them, so the feature introduces **zero** new type errors.
  Diagnostics touch pre-existing code (e.g. `server_simple.py` send/recipient
  paths, `tests/test_tool_descriptions.py`); none fall on the added `session_dir`/
  `session-dir`/`inbox-status`/`--reader` lines, and `cli.py` has no ty diagnostics.
  Same disposition as the `ruff format` drift: pre-existing, not absorbed here.
- **TS `tsc --noEmit` / vitest / eslint: GREEN** (independently re-run: tsc exit 0,
  50/50 across 5 suites, eslint clean).
- **Prettier: scoped GREEN, whole-tree RED (accurately).** The package's enforced
  scope `npx prettier --check "src/**/*.ts" "test/**/*.ts"` passes. A repo-wide
  `npx prettier --check .` reports `.eslintrc.cjs` and `README.md`, which the
  package's `format` script does not target; this is not the gate the package
  runs. (Corrects the earlier overstated "prettier --check . GREEN" claim.)
- **`ruff format --check .`: RED — pre-existing, not this feature.** 7 files would
  reformat (`src/claude_teams/backends/{codex,contracts,process_manager,registry}.py`,
  `tests/test_agent_output.py`, `tests/test_backends/{test_base_runtime,test_codex}.py`).
  None are touched by this feature; confirmed red on `main` too. Deliberately not
  absorbed into the feature branch (isolation; a parallel agent is cleaning these).
  **Recommendation:** leave to the parallel cleanup, or a separate cosmetic commit.

## Deviations from plan

1. **TS `AbortError` class instead of `DOMException`** — the `ES2022` lib excludes
   the DOM lib; a custom `AbortError` (`name==="AbortError"`) is behavior-equivalent
   and `isAbortError` also matches Node's real abort `DOMException` by name.
2. **ACK_STALLED checks session-change first** — plan §3.1 lists it third; the build
   checks discovery at the top of every poll (plan-review-4 §6 note 2's *preferred*
   ordering). Late-drain then new-generation follow in plan order. No behavior gap.
3. **Unsafe `--reader` value → exit 1 + stderr** (plan enumerated exit 4 only for
   bad `session_dir`); treated as a validation/internal error.
4. **No separate Pi manifest file** — pi 0.80.10 loads a default-exported factory
   module; the root `index.ts` (`main` in `package.json`, importing the `src/`
   modules) is that entry, matching the `pi -e <dir>` convention.

## Post-review fixes (implementation-review-1.md, blockers 1–4 + non-blocking)

Addressed the Codex post-implementation review (`implementation-review-1.md`),
TDD red→green:

1. **Blocker 1 — cross-platform inbox-path guard.** `cli.ts` gains
   `isLeadInboxPath()` (separator-normalizing) and `state-machine.ts:147` now uses
   it instead of a hardcoded-`/` strict equality, so a Windows `session_dir` +
   backslash wake `path` is no longer dropped. Tests: `cli.test.ts`
   "isLeadInboxPath (cross-platform guard)" + `state-machine.test.ts`
   "T4/blocker-1: a Windows session_dir + backslash wake path injects exactly once"
   (RED against the old strict guard, GREEN after).
2. **Blocker 2 — T5d proves the invariant.** Rewrote T5d to capture at
   `total=2,cursor=0` (unread 2) and resolve at `total=4,cursor=2` (unread STILL 2,
   `min(cursor,total)=2 ≥ target 2`), asserting no watch re-arm before the cursor
   threshold and re-arm only at/after it. A naive "unread decreased / any progress"
   ACK impl now FAILS this test (verified by mutation).
3. **Blocker 3 — exit-0 stderr rule.** `parseWatch()` now treats exit 0 with
   non-empty stderr as `malformed` (backoff, no wake). Test: `cli.test.ts`
   "treats exit 0 with non-empty stderr as malformed" (RED then GREEN).
4. **Blocker 4 — T9 delivery evidence (committed).** Added `test/index.test.ts`
   driving the default-export factory with a fake `ExtensionAPI` (asserts
   `session_start`/`session_shutdown` registration, exactly one owned loop/live
   child on start, abort on shutdown), and `README.md` documenting the exact Pi
   load/run procedure and the manual smoke-test steps. The live Pi smoke test is
   NOT claimed as run.

Non-blocking: strengthened T11 to drive the real `WakeMachine` and count live
production `pi.exec` children (≤ 1 across start/reload/resume); added unsafe-`--reader`
exit-1 tests for both `watch` and `inbox-status` (Python); strengthened T2b to follow
the non-consuming probe with the real consuming `read_messages` and assert identical
messages; corrected the prettier gate claim above.

## Live Pi smoke test — PASSED (2026-07-19)

- The **real Pi smoke test** (plan §6 T9 live run) was validated interactively by
  the user on 2026-07-19. A Codex worker was spawned by a Pi lead via
  win-agent-teams; when the Codex worker called `send_message` to team-lead, the
  Pi lead (running `pi -e pi-extensions/win-agent-teams-wake` with the
  pi-mcp-adapter + `.mcp.json`) was woken **immediately** and read the message with
  no manual input. This confirms FR1–FR3 across a heterogeneous team (Codex worker
  → Pi lead). The committed automated loadability proof is `test/index.test.ts`;
  the manual procedure is in `pi-extensions/win-agent-teams-wake/README.md`.
