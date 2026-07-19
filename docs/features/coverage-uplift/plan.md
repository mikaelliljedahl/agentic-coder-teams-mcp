# Plan: Test coverage uplift — first increment

## Scope

The repo enforces a 90% coverage floor (`pyproject.toml [tool.coverage.report] fail_under = 90`),
but `uv run pytest --cov` currently reports **79.47%** total — pre-existing debt on `main`, not
introduced by any recent PR. This is the first of what will be several increments toward the floor.

**This increment is test-only.** No production code changes. We add genuine tests for
currently-untested branches (guards, error handlers, edge cases). We do **not** alter production
behavior to game coverage, and we do **not** add trivial "call it and ignore the result" tests
that assert nothing.

### In scope (targeted modules)

| Module | Stmts | Missed | Cover | Plan |
|---|---|---|---|---|
| `src/claude_teams/server.py` | 4 | 4 | 0% | → 100% (import + `__main__` via `runpy` with patched `main`) |
| `src/claude_teams/backends/_agent_discovery.py` | 50 | 20 | 60% | → ~100% (pure functions, error branches) |
| `src/claude_teams/agent_output.py` | 275 | 50 | 82% | → ~97%+ (file-read guards/error handlers) |
| `src/claude_teams/server_simple.py` | 921 | 98 | 89% | → ~96% (pure guards, tool error paths); ~31 lines deferred (Windows/`main`/`follow_up` combos) |

Projected reduction: ~150 missed lines → total coverage rises from **79.47%** to roughly **84–85%**.

### Explicitly OUT of scope (deferred to later increments)

- `src/claude_teams/backends/process_manager.py` — **388 missed lines (65%)**. This is the single
  largest gap and the true reason the floor is unmet. It requires substantial process/subprocess
  mocking and would blow up PR size. It gets its own increment(s).
- Other partial modules (`cli.py` 73%, `codex.py` 88%, `process_base.py` 85%, `hooks.py` 86%,
  `messaging.py` 92%) — candidates for a second increment.

**The floor (90%) will NOT be met by this increment.** That is expected and will be reported
honestly. Reaching 90% requires the `process_manager.py` work above.

## Current behavior

The targeted modules already work; they are simply under-tested on their defensive branches:

- `_agent_discovery.py`: `discover_codex_style_agents` error branches (TOML decode failure,
  non-dict `agents`, non-dict entry, missing/empty `config_file`) and the entire
  `discover_goose_recipes` function (env-var-driven recipe scan) are untested.
- `agent_output.py`: early-return guards (`spawned_at <= 0`, empty cwd, unresolvable path) and
  error handlers on malformed/empty JSONL rollout files are untested.
- `server_simple.py`: dozens of small guards — corrupt-JSON tolerances, "no active session" tool
  returns, "agent not found" returns, unknown-backend fall-throughs, bad-numeric coercions.
- `server.py`: never imported by any test.

## Proposed design (tests only)

All new tests use `tmp_path` + `monkeypatch` (of `Path.home`, module globals like `_SESSION_BASE`,
and env vars), following the established style in `tests/test_agent_output.py` (helpers like
`_write_jsonl`, `_codex_path`). No network, no real subprocesses, no real MCP server runtime.

### 1. `tests/test_agent_discovery.py` (new)
- `discover_codex_style_agents`: TOML decode error → skipped; `agents` not a dict → skipped;
  entry not a dict → skipped; `config_file` missing/empty/non-str → skipped; project-local
  overrides user-global on name collision (write both, assert project wins).
- `discover_goose_recipes`: empty/unset `GOOSE_RECIPE_PATH` → `[]`; empty dir segment skipped;
  non-existent dir skipped; `*.yaml`/`*.yml`/`*.json` discovered; first-seen wins on collision.
- (Optionally exercise `discover_claude_agents` project-shadows-home if not already covered.)

### 2. `tests/test_agent_output.py` (extend)
Add focused cases for the uncovered guard/error lines:
- `read_codex_output` / `read_claude_output`: `spawned_at <= 0` → `None`; empty `cwd` → `None`;
  path that resolves empty → `None`; no candidates → `None`; message `None` and session id `None`
  → `None`.
- `_rollout_contains_token`: token beyond `max_lines` → `False`; `OSError` on open → `False`.
- `_first_json_object`: blank lines skipped; `JSONDecodeError` line skipped; `OSError` → `None`.
- `_last_codex_message` / `_last_claude_message`: non-`response_item` / non-`assistant` skipped;
  payload not dict skipped; role mismatch skipped.
- `_claude_session_id` / `_claude_started_at`: blank/invalid-JSON/non-dict lines skipped;
  `OSError` → `None`.
- `_parse_timestamp`: non-str / empty / bad-format → `None`.
- `_content_text`: `allow_string` path; non-list → `None`; no matching parts → `None`.
- `_iter_lines_reverse`: multi-chunk reverse read; `OSError` → empty.
- `_truncate_tail`: budget `<= 0` → `""`; budget too small for marker → raw tail.
- `_resolve_path_text` / `_normalize_path`: empty → `""`.

### 3. `tests/test_server_simple_guards.py` (new)
Group the pure-guard / tool-error-path cases. Representative set (from line map):
- `_idle_seconds`: non-float env → default.
- `_read_state_marker`, `_read_json_object`: corrupt JSON → `None` / `{}`.
- `_unique_agent_name`: multi-collision → `-3`.
- `_prune_superseded_bindings`, `_iter_binding_metas`, `_candidate_sessions`,
  `_distinct_binding_sessions`: no-dir, non-json, mismatched identity/cwd, corrupt registry,
  non-list agents, stale-mtime branches (craft on-disk binding + session fixtures matching module
  `IDENTITY`/cwd; use `os.utime` for mtime).
- `_recovery_note`: one-shot `adopted_session` nudge clears the global.
- `_is_session_dir`: non-UUID → `False`.
- `_session_has_live_agent`: corrupt/non-list registry → `False`.
- `_remove_team_logs`: custom `WIN_AGENT_TEAMS_LOG_DIR` set → no removal.
- `cleanup_old_sessions` / `_maybe_cleanup_old_sessions`: missing base → `[]`; corrupt stamp → 0.0.
- `_message_recipient`: name matches known agent → verbatim.
- `_safe_float`: bad value → 0.0. `_marker_timestamp`: non-numeric / bool → `None`.
- `_read_agent_output` / `_hook_extra`: unknown backend → `None` / `{}`.
- `_last_non_empty_line`: all-blank → `""`.
- MCP tools `send_message`, `read_messages`, `kill_agent`, `resume_session`, `session_info`,
  `list_agents`, `agent_status`, `list_backends`: "no active session" / "agent not found" /
  "session_id_required" / corrupt-registry-tolerated paths, driven by forcing
  `_active_session_id` to `""` (monkeypatch) or crafting session dirs.
- `follow_up_agent`: at minimum `session_not_found` (empty session) and `backend_not_supported`
  (registry rejects backend name). Deeper interior branches — see "Known remaining gaps".

### 4. `tests/test_server_entrypoint.py` (new)
- Import `claude_teams.server`, assert `main`/`mcp` re-exported.
- Run the `__main__` block via `runpy.run_module("claude_teams.server", run_name="__main__")`
  with `main` patched to a no-op, to cover the guarded `main()` call without starting the server.

## Files affected

- `docs/features/coverage-uplift/plan.md` (this file)
- `docs/features/coverage-uplift/plan-review.md` (Codex review)
- `docs/features/coverage-uplift/implementation.md` (red/green evidence)
- `docs/features/coverage-uplift/implementation-review.md` (Claude Opus review)
- `tests/test_agent_discovery.py` (new)
- `tests/test_agent_output.py` (extend)
- `tests/test_server_simple_guards.py` (new)
- `tests/test_server_entrypoint.py` (new)
- **No `src/` changes.**

## Risks

- **Fixture coupling to module globals.** `server_simple` reads module-level `IDENTITY`, `_SESSION_BASE`,
  `_session_id`, `_pending_recovery`. Tests must monkeypatch these and restore them. Mitigation:
  monkeypatch via the `monkeypatch` fixture (auto-restored) and prefer `_SESSION_BASE` override so
  tests never touch the real `~/.claude` tree.
- **Global state bleed between tests.** `_session_id` / `_pending_recovery` are module globals.
  Mitigation: set/reset within each test via `monkeypatch.setattr`.
- **Async tools.** MCP tool functions are async. Mitigation: use `pytest.mark.asyncio` /
  `asyncio.run`, consistent with existing async tests in the suite.
- **Over-mocking `follow_up_agent`.** The `if changed: save` interior branches require a
  contrived `_sync_backend_session_id`-True + failure combination. Mitigation: cover the
  cleanly-reachable refusal reasons; document the rest as remaining gaps rather than writing
  brittle tests that assert on mock internals.

## Known remaining gaps (documented, not covered by this increment)

- `server_simple.py`: Windows-only `msvcrt` locking branches (lines 17-18, 293-303, 310-311),
  `main()` blocking `mcp.run()` and `__main__` (2107, 2111), and the fiddliest `follow_up_agent`
  `if changed` combinations (subset of 1608/1615/1619-1621/1639-1641/1648/1659/1704-1708).
- `process_manager.py` in full (deferred increment).

## Test / validation commands

- Focused (red→green per file): `uv run pytest tests/test_agent_discovery.py tests/test_server_simple_guards.py tests/test_server_entrypoint.py tests/test_agent_output.py -q`
- Full suite + coverage: `uv run pytest --cov` (report the new total honestly; floor still unmet).
- Lint: `uv run ruff check src/ tests/`

## Plan-review dispositions (Codex, opposite family — verdict: APPROVE-WITH-CHANGES)

All findings accepted; load-bearing claims independently verified against source before acceptance.

1. **MAJOR — validation command / red gate.** `uv run pytest --cov` exits non-zero while below
   `fail_under`. **Accepted.** Passing validation command is
   `uv run pytest --cov --cov-fail-under=0` (all tests green). The configured
   `uv run pytest --cov` is run separately and its *only* expected failure is the coverage gate,
   reported explicitly. **Merge policy:** this repo's gate is already red on `main` (79.47%), so
   the gate is pre-existing debt, not introduced here; the PR lowers the debt and states the floor
   remains unmet. Surfaced to the user in the summary/PR.
2. **MAJOR — server_simple target overstated + `_dir_newest_mtime` omitted.** Verified: 668-676
   are reachable OSError branches. **Accepted.** Added `_dir_newest_mtime` tests (monkeypatch
   `Path.rglob` → OSError; and a descendant `stat` → OSError). Module target corrected to **~96%**
   (deferring Windows 17-18/293-303/310-311, `follow_up_agent` interior combos 1608-1708 subset,
   `main` 2107/2111 ≈ 31 lines). Success criteria updated below.
3. **MAJOR — `list_backends` has no session guard; corrupt registry not universally tolerated.**
   Verified: `list_backends` (2088-2102) never calls `_active_session_id`; `_load_agents` →
   `_load_agents_unlocked` (328-332) has **no** try/except, so `list_agents`/`agent_status`/
   `kill_agent`/`_message_recipient` do **not** tolerate corrupt `agents.json`; only
   `resume_session` (1837-1840) and `session_info` (1878-1888) catch it. **Accepted.** The grouped
   tool bullet is replaced by the per-tool matrix below. `list_backends` gets a fake-registry
   behavior test (assert full row). Corrupt-registry tolerance is asserted ONLY for
   `resume_session` + `session_info`.
4. **MINOR — "path resolves empty" is not a real fs case.** Verified against
   `_resolve_path_text` (417-423). **Accepted.** Dropped that claim. `agent_output.py` 57-59/90-92
   are covered as explicit contract-isolation tests (monkeypatch the normalize helper to `""`),
   labelled as such — not as a filesystem edge.
5. **MINOR — some cases already covered; reconcile against fresh missing-lines.** **Accepted.** The
   case list is reconciled against the current `--cov` missing report (below); already-covered
   behaviors (no-candidate returns 64-65/105-106, string content 347-348, small-budget marker
   402-404) are dropped. Decision on 134-139 (malformed Codex meta/payload), 204-207 (per-file
   `stat` OSError), 216-219 (bad spawn timestamp): **cover all three** (cheap, genuine).
6. **MINOR — isolation fixture must be stronger.** Verified import-time binding of `_SESSION_BASE`
   and `_TEAMS_BASE`. **Accepted.** New `tests/test_server_simple_guards.py` uses an autouse fixture
   that monkeypatches `_SESSION_BASE`, `_TEAMS_BASE`, `_session_id=""`, `_AGENT_SESSION_ID=""`,
   `_AGENT_PARENT_NAME`, `_pending_recovery={}`, resets `_inbox_locks`, sets an isolated cwd, and
   clears recovery/retention/log env overrides. `IDENTITY`/`_AGENT_PARENT_NAME` patched locally
   where a test needs them.
7. **MINOR — entrypoint patch target.** **Accepted.** Patch `claude_teams.server_simple.main`
   before `runpy.run_module("claude_teams.server", run_name="__main__")`; assert called once.
   Re-export test asserts identity (`server.main is server_simple.main`,
   `server.mcp is server_simple.mcp`).
8. **MINOR — observable assertions + portable fault injection.** **Accepted.** No chmod; fault
   injection via monkeypatching `Path.open`/`read_text`/`stat`/`rglob`/`iterdir`. `_iter_lines_reverse`
   OSError asserted via `list(...) == []` (generator consumed). Skip-logic tests place a valid
   *older* record before malformed/newer ones and assert the older text is returned.
9. **NIT — dedupe vs existing suites.** **Accepted.** New tests name the distinct branch and avoid
   restating existing recovery/resume/no-candidate cases.

### Per-tool expectation matrix (replaces the grouped MCP-tool bullet in §3)

| Tool | Setup | Expected | Branch |
|---|---|---|---|
| `send_message` | force `_active_session_id()` → `""` | `success=False, reason=session_not_found` | 1322 |
| `read_messages` | no session | empty snapshot | 1390 |
| `kill_agent` | no session / agent-not-found | `session_not_found` / `success=False, name` | 1784 / 1788 |
| `resume_session` | `""` id / valid dir + **corrupt** agents.json | `session_id_required` / `success=True, agent_count=0` | 1821 / 1839-1840 |
| `session_info` | resolved session + **corrupt** agents.json | `agent_count=0` | 1878-1882 |
| `list_agents` | no session | `[]` | 1946 |
| `agent_status` | no session | `[]` | 2034 |
| `list_backends` | fake registry (2 backends) | full rows `{name,binary,default_model,supported_models}` | 2088-2102 |
| `follow_up_agent` | no session / registry rejects backend | `session_not_found` / `backend_not_supported` | 1584 / 1593-1595 |

## Success criteria

1. All new tests pass; full suite stays green (except the expected `fail_under` gate, which we
   report as still red because the floor is genuinely not yet met).
2. Total coverage rises materially (target ~84%), with the four targeted modules at/near the
   percentages in the scope table (server_simple ~96%, agent_output ~97%+, _agent_discovery ~100%,
   server 100%). Actual measured numbers reported in `implementation.md`.
3. `ruff` clean on `src/` and `tests/`.
4. No production (`src/`) behavior changes.
5. The summary/PR states plainly that the 90% floor remains unmet and names `process_manager.py`
   as the remaining blocker.
