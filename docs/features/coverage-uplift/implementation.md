# Implementation: Test coverage uplift — first increment

Test-only increment. **No `src/` changes.** Adds genuine tests for previously
untested guard / error / edge branches in four modules.

## Files added / changed

- `tests/test_server_entrypoint.py` (new) — `server.py` re-export identity + `__main__`.
- `tests/test_agent_discovery.py` (new) — `_agent_discovery.py` error/skip branches + collisions.
- `tests/test_agent_output.py` (extended) — appended a "Guard / error-branch coverage" section
  (private-helper guards, IO error handlers, malformed-record skips).
- `tests/test_server_simple_guards.py` (new) — pure guards, binding/recovery filtering, and the
  per-tool "no session / not found / corrupt-registry-tolerated / fake-registry" matrix.

## Per-module coverage (measured)

| Module | Before | After |
|---|---|---|
| `server.py` | 0% | **100%** |
| `backends/_agent_discovery.py` | 60% | **100%** |
| `agent_output.py` | 82% | **100%** |
| `server_simple.py` | 89% | **~98%** (deferred set is now Windows + `follow_up` combos only) |

**Total suite coverage: 79.47% → 84.00%** (+4.53 pts). Test count: 484 → 575 passed (+91), 2
skipped. `server_simple.py` = 97% (28 missed). After the post-implementation review (below), `main()`/`__main__`
(2107, 2111) are now covered, so the `server_simple.py` remaining set is just the Windows `msvcrt`
branches (`18, 293-303, 310-311`) and the `follow_up_agent` `if changed`/escalate combos
(`1608, 1615, 1619-1621, 1639-1641, 1648, 1659, 1704-1708`).

## Red → green evidence

Each new test file was run in isolation and asserted to fail against the *untested* branch before
the assertions were finalised, then run green:

- `_agent_discovery` / `server` entrypoint: `14 passed` (12 discovery + 2 entrypoint; later grown to
  4 entrypoint tests in the review pass); then coverage confirmed
  `_agent_discovery.py 100%`, `server.py 100%`.
- `agent_output` guards: `80 passed`; coverage `agent_output.py 100%`.
- `server_simple` guards: initial run surfaced one genuinely-wrong test
  (`test_iter_binding_metas_skips_unstattable`) — the injected `stat` fault fired inside
  `Path.is_file()` (line 456) and a generic `OSError` propagated out of `is_file` rather than being
  caught at 461-462. Corrected to inject a `read_text` fault (the branch's actual catchable IO),
  renamed `_skips_unreadable`. Final: `41 passed`.

This is the intended TDD signal: the fault-injection point must match the branch under test, or the
test proves nothing. The failure caught a mis-aimed test before it could mask real behaviour.

## Notable correctness decisions (from plan-review dispositions)

- **Hermetic isolation.** `server_simple._SESSION_BASE` / `_TEAMS_BASE` are import-time-bound, so an
  autouse-style `isolated` fixture patches the module attributes directly (not `Path.home`) plus
  `_session_id`, `_AGENT_SESSION_ID`, `_pending_recovery`, `_inbox_locks`, cwd, and the relevant env
  overrides. The discovery tests likewise patch `Path.home` to a clean tmp dir so they never read the
  developer's real `~/.codex` / `~/.claude`.
- **No corruption-tolerance tests where production does not tolerate it.** Verified `_load_agents`
  → `_load_agents_unlocked` has no `try/except`; corrupt-`agents.json` tolerance is asserted ONLY
  for `resume_session` and `session_info` (which catch it), never for `list_agents` / `agent_status`
  / `kill_agent`.
- **`list_backends`** has no session guard — tested with a deterministic fake registry asserting the
  full returned row, not an inapplicable "no session" path.
- **Contract-isolation branches** (`agent_output` 57-59 / 90-92) are labelled as such — reached by
  monkeypatching `_normalize_path` / `_resolve_path_text` to `""`, not presented as filesystem edges.
- **Portable fault injection** — monkeypatch `Path.open`/`read_text`/`stat`/`rglob`/`iterdir`; no
  `chmod`. `_iter_lines_reverse`'s OSError branch asserted via `list(...) == []` (generator consumed).

## Known remaining gaps in `server_simple.py` (intentionally deferred)

- Windows-only `msvcrt` locking (lines 17-18, 293-303, 310-311) — not reachable on Linux CI.
- The fiddliest `follow_up_agent` `if changed: save` / escalate-to-kill combinations (subset of
  1597-1728) — require a contrived `_sync_backend_session_id`-True + failure state; deferred rather
  than asserting on mock internals. The cleanly-reachable refusals (`session_not_found`,
  `backend_not_supported`) are covered.

(`main()`'s `mcp.run()` at 2107 and the `__main__` guard at 2111 were initially deferred but are now
covered — see the post-implementation review dispositions.)

## Post-implementation review dispositions (Codex, independent — verdict: REQUEST-CHANGES)

All findings accepted and addressed; verified against source.

- **BLOCKER — entry-guard tests could read real user state on regression.**
  `test_read_codex_output_rejects_bad_inputs` / `test_read_claude_output_rejects_bad_inputs` now
  patch `Path.home` to `tmp_path`, so a regressed guard can never reach the real `~/.codex` /
  `~/.claude` trees.
- **MAJOR — weak tests that covered a line without proving its branch.** Strengthened to
  mutation-resistant selection assertions: the Codex malformed-meta test now includes a valid OLDER
  matching rollout and asserts *it* is selected (so removing the `session_meta`/dict guards changes
  the result or raises); added `test_last_codex_message_skips_non_assistant_returns_older` (proves the
  role skip via selection); `test_content_text_non_list_returns_none` now uses non-iterable inputs
  (`None`, `int`) so the guard's absence would raise rather than silently return `None`.
  *Residual (documented):* `discover_claude_agents` no-dirs (line 46) and `discover_goose_recipes`
  blank/missing-dir (118/121) are behaviourally-equivalent guards on those inputs — `glob` over a
  nonexistent path yields nothing either way — so no output-level assertion can distinguish the
  mutant; line coverage is the honest ceiling for them and they are kept as coverage tests.
- **MAJOR — isolation fixture incomplete.** The `isolated` fixture is now `autouse=True` and also
  patches the import-time-bound `_AGENT_PARENT_NAME` and `IDENTITY` (to `ROOT_LEAD_NAME`), so the
  root-lead expectation in `_message_recipient` cannot vary with a developer's `AGENT_NAME` /
  `AGENT_PARENT_NAME` environment.
- **MINOR — contract-isolation label.** The Claude resolve-empty test is now explicitly labelled as
  contract isolation, matching its Codex sibling.
- **MINOR — evidence count.** Corrected 13 → 14 above.
- **MINOR — `main`/`__main__` mischaracterised as runtime/subprocess.** Rather than reword, they are
  now covered: `test_server_simple_main_runs_mcp` (patches `FastMCP.run` at the class, asserts
  `main()` forwards once → line 2107) and `test_server_simple_main_block` (runpy `__main__` → 2111).
- **NIT — runpy warning.** The benign in-`sys.modules` `RuntimeWarning` is suppressed via a
  `_run_as_main` helper (deleting the cached module would risk a divergent second copy).

## Whole-repo status (honest)

The 90% `fail_under` gate remains **red** — it was already red on `main` (79.47%, pre-existing
debt). This increment lowers the debt materially (→ 84.00%) but does not clear it; the dominant
remaining gap is `backends/process_manager.py` (65%, 388 missed lines), scoped to a later increment.
Runners-up for the next increment: `cli.py` (73%), `hooks.py` (86%), `codex.py` (88%),
`process_base.py` (85%).

## Validation commands

- Passing suite: `uv run pytest --cov --cov-fail-under=0` (all tests green).
- Configured gate: `uv run pytest --cov` — expected to exit non-zero solely on the `fail_under`
  coverage gate (pre-existing repo debt), not on any test failure.
- Lint: `uv run ruff check src/ tests/` → clean.
