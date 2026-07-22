# Implementation — Deterministic inbox-wake for Claude Code lead agents

Feature slug: `claude-lead-inbox-wake`.
Branch: `feature/claude-lead-inbox-wake`, worktree `/home/mikael/code/wt-claude-lead-wake`.
Built with red-green TDD against the APPROVED `plan.md` (§2 design, §3 files,
§4 test plan), honouring `plan-review.md` F1 (guard sources `{total, cursor,
unread}` from `read_inbox_by_sender` + `load_inbox_cursors`) and F2 (persistent
arming false-negative is bounded by the no-progress guard + smoke assertion).

---

## Task 0 (plan §5) — `background_tasks` re-confirmed on this harness

**Verdict: PASS.** A throwaway `Stop` hook (scratchpad, delivered via
`claude -p --settings <file>`, real `~/.claude/settings.json` and every project
`.claude/settings.json` left untouched) logged the raw stdin while a harmless
background task (`python3 -c "import time; time.sleep(20)"`) was running.

- Harness: `claude --version` → **2.1.215 (Claude Code)** (same as the spike).
- Observed `background_tasks` entry shape (matches spike-results.md:137–141
  exactly):

  ```json
  {"id":"b15bzmocl","type":"shell","status":"running",
   "description":"Sleep for 20 seconds in background",
   "command":"python3 -c \"import time; time.sleep(20)\""}
  ```

- The full Stop payload carried `session_id, transcript_path, cwd, prompt_id,
  permission_mode, hook_event_name, stop_hook_active, last_assistant_message,
  background_tasks, session_crons` — `background_tasks` populated, `command`
  string preserved verbatim. No `effort` field (as the spike flagged).

The field is present and reliable, so the `background_tasks` arming-match logic
was built as designed (plan §2.4). No switch to the pid/marker fallback.

---

## Final design as built

- **`src/claude_teams/lead_wake.py`** (new). Pure decision core `evaluate(payload,
  *, reader_arg, session_dir_arg)` returning a `WakeDecision(action, code,
  reason, log)`, plus a `__main__` (`_parse_args` → `_read_payload` from stdin →
  `evaluate` → structured stderr log + optional `{"decision":"block",...}` on
  stdout). Decision table D0–D6 exactly as plan §2.2. Helpers:
  `_resolve_identity` (`AGENT_NAME` authoritative, `--reader` default,
  `team-lead` fallback), `_resolve_session_dir` (reuses
  `server_simple._active_session_id`/`_session_dir`; baked `--session-dir` is a
  fallback only), `_live_subagent_names` (non-terminal `_load_agents`),
  `_scan_senders` (one read-only `read_inbox_by_sender` + `load_inbox_cursors`
  scan, mirrors `inbox-status`), `_is_armed`/`_command_matches_session`
  (separator-insensitive, session-scoped `background_tasks` match), and the
  progress guard (`_read_guard`/`_write_guard`/`_cursor_advanced`/`_apply_guard`
  writing `wake-progress-<reader>.json`, cursor-keyed).
- **`src/claude_teams/hooks.py`**. Added `_WAKE_MODULE`,
  `_WAKE_HOOK_TIMEOUT_SECONDS`, `_wake_command`, `_wake_hook_matcher`; extended
  `write_claude_settings` so ONLY the `Stop` event gets a SECOND matcher group
  (the wake hook) alongside the existing `emit` group. All other events
  unchanged.
- **`src/claude_teams/server_simple.py`**. New MCP tool `install_lead_wake(remove,
  scope)` (idempotent write/remove of the top-level `Stop` wake hook into project
  `.claude/settings.json`, `scope="user"` → `~/.claude/settings.json`), pure
  helper `_install_wake_hook` + `_group_has_wake_token` +
  `_lead_wake_settings_path`; extended `_DISK_CONTRACT_NOTE` (plan §2.10).
- **`README.md`**. New "Claude Code lead wake" section + interactive manual smoke
  test mirroring the Pi wake README (incl. F2's happy-path arming-recognition
  assertion). Tool count 10 → 11 + table row.

Decision precedence as built: D0 kill switch → D1 no session → D2 no live
subagents → (unread? → D3 block-read via guard) → (armed? → D4 allow) → D5
block-arm via guard. The guard (consulted only on would-be D3/D5 blocks) resets
`noprogress_blocks` on cursor advance and, only when `stop_hook_active` is true,
increments and fails open at the cap (D6).

---

## Red → green evidence (per cluster, in plan §4.1 order)

Focused tests were run with `uv run python -m pytest`. `python` is not on PATH
on this VM; `uv run` provides the project interpreter.

### Cluster: hooks wiring (tests #1, #2)
- **RED:** `tests/test_hooks.py::...::test_write_claude_settings_stop_has_two_matcher_groups`
  and `...::test_wake_command_argv_shape` →
  `AttributeError: module 'claude_teams.hooks' has no attribute '_wake_command'`
  and `StopIteration` (no `lead_wake` group). (`3 failed, 1 passed`.)
- **GREEN:** after adding `_WAKE_MODULE`/`_wake_command`/`_wake_hook_matcher` and
  the `Stop`-only second group → `48 passed, 2 skipped` (Windows-only cmd tests).

### Cluster A: D0/D1/D2 (tests #13, #3, #4)
- **RED:** `ImportError: cannot import name 'lead_wake' from 'claude_teams'`
  (`1 error during collection`).
- **GREEN:** module skeleton implementing D0/D1/D2 (block branch stubbed) →
  `3 passed`.

### Cluster B: D3/D4/D5 (tests #5, #6, #7)
- **RED:** `AssertionError: assert 'allow' == 'block'` (stub returned allow where
  a block was expected). (`3 failed`.)
- **GREEN:** `_is_armed`/`_command_matches_session`, `_read_reason`/`_arm_reason`,
  and the unread/armed/arm branch → `6 passed`.

### Cluster C: progress guard (tests #8, #9, + a stop-hook-active gate test)
- **RED** (captured by temporarily bypassing `_apply_guard` with `return block`):
  `test_wake_progress_guard_fail_open_after_cap` →
  `AssertionError: assert 'block' == 'allow'`;
  `test_wake_progress_guard_resets_after_productive_wake` →
  `assert 2 == 0` (counter never reset, no guard file written). (`2 failed,
  1 passed` — the gate test passes vacuously without the guard, by design.)
- **GREEN:** `_apply_guard` + `_read_guard`/`_write_guard`/`_cursor_advanced` →
  full `test_lead_wake.py` `9 passed`.

  Note on discipline: the cluster-C tests and the guard implementation were first
  written in the same cycle; the RED above was captured retroactively by
  bypassing the guard and re-running, then reverted. Honest disclosure per
  CLAUDE.md — the red is real and reproducible, but was not observed strictly
  before the code on the first pass for this cluster.

### Cluster D: identity + arming match (tests #10, #11)
- Added `test_wake_nested_lead_uses_agent_name_inbox` and
  `test_arming_match_is_separator_insensitive_and_session_scoped`; both **GREEN**
  against the already-built helpers (`2 passed`). #10 asserts `AGENT_NAME=mid`
  scans `inbox-mid.jsonl` (bob) and never `inbox-team-lead.jsonl` (zoe) — the
  regression vs the Pi team-lead-only bug. #11 asserts a `\`-separated command
  for THIS session matches while a DIFFERENT session's watcher does not.

### `__main__` entrypoint (3 tests)
- `TestMainEntrypoint` uses `io.StringIO` stdin faking (mirrors `test_hooks.py`):
  block prints the decision JSON to stdout and a body-free log line to stderr;
  allow prints nothing to stdout; corrupt stdin fails open. All **GREEN**.
- Subprocess smoke: `echo '{...Stop...}' | python -m claude_teams.lead_wake
  --session-dir <none> --reader team-lead` → stderr
  `win-agent-teams/lead-wake {"code":"D1",...,"decision":"allow","why":"no-session"}`,
  empty stdout, exit 0.

### `install_lead_wake` (test #12 + tool tests)
- **RED:** `AttributeError` — `_install_wake_hook`/`install_lead_wake` absent
  (`6 failed`).
- **GREEN:** pure `_install_wake_hook` upsert (idempotent, preserves unrelated
  hooks, `remove` drops only the wake group) + the async tool → `6 passed`.

### Docstring contract (`_DISK_CONTRACT_NOTE`, install docstring)
- **RED:** `assert 'background_tasks' in note` failed.
- **GREEN:** extended `_DISK_CONTRACT_NOTE` (background_tasks / arm-or-read /
  `wake-progress-<reader>.json` / kill switch) → `test_tool_descriptions.py`
  `12 passed`.

---

## Deviations from the plan (minimal, justified)

1. **Top-level `from claude_teams import server_simple` in `lead_wake.py`**
   rather than a function-local import. The plan (§2.1) justified a *separate
   module* to avoid an import cycle, and verified `server_simple` does **not**
   import `lead_wake`; a top-level import is therefore cycle-free and matches
   `cli.py`'s established `from claude_teams import server_simple as _ss`
   pattern. This also satisfies ruff `PLC0415` (no inline imports) for production
   code. No behavioural change.
2. **Extra tests beyond the plan's 13.** Added `test_wake_command_argv_shape`,
   `test_wake_guard_does_not_fail_open_without_stop_hook_active` (enforces the D6
   `stop_hook_active` gate), 3 `__main__` entrypoint tests, and 2 pure
   upsert-helper tests. All strengthen coverage of load-bearing seams; none
   weaken or replace a planned test. Test #12 is realised as the upsert-helper
   pair plus the tool idempotency/removal tests.
3. **`install_lead_wake` `scope` parameter** is `"project"` (default) / `"user"`
   strings rather than a bare `scope="user"` flag — same behaviour, clearer API,
   validated with an error return for other values.
4. **`--session-dir` baked but discovery-primary at runtime.** Faithful to §2.3
   and the §8 note: the runtime resolves the session via `server_simple`
   discovery and uses the baked `--session-dir` only as a fallback when discovery
   yields nothing but the baked dir still exists.

No design decisions were relitigated; the `background_tasks` variant, the
two-`Stop`-group wiring, the cursor-keyed guard, and the `install_lead_wake`
tool are all as approved.

---

## Validation commands and outcomes (whole repo, this Linux VM = the CI gate)

- `uv run ruff check` → **All checks passed!**
- `uv run ruff format --check .` → **55 files already formatted** (the 6 touched
  files were auto-formatted with `ruff format`; no unrelated files changed).
- `uv run python -m pytest -q` → **688 passed, 3 skipped** (skips are the
  Windows-only `cmd.exe` launcher tests + one platform-guarded test).
- `uv run python -m pytest --cov=claude_teams --cov-report=term-missing` →
  **Total coverage 83.74%** (`fail_under = 80` reached); `lead_wake.py` **88%**
  (uncovered lines are the discovery seams tests monkeypatch and env
  malformed-value branches).

No red anywhere in the tree; no pre-existing breakage encountered or absorbed.

---

## Open items surfaced for review / manual verification

- **M3 interactive wake + F2 arming happy-path** cannot be closed headless
  (spike-results.md:168–177): that the harness re-invokes an idle lead when the
  background watcher exits, and that the real `_watch_command_bash` rendering is
  recognised by `_command_matches_session` end-to-end, are in the README manual
  smoke test (§ "Manual smoke test", steps 3–5). This is the same interactive
  assumption the already-shipped watch recipe makes.
- **GC1 (Claude Desktop honours the settings `Stop` hook)** remains the 1-minute
  manual check (spike M1); every headless signal points to it holding on harness
  2.1.215.

---

## Post-merge fix (Fynd 1): leaf agents wrongly self-counting as live subagents

Surfaced by the interactive test after this branch was rebased onto
`origin/main` (which had absorbed PR #34's Pi-wake auto-loading, wiring the wake
hook onto *every* spawned Claude Code agent, not just top-level leads).

### The bug

`lead_wake._live_subagent_names(session_dir)` returned **all** non-terminal
agents in the shared session's `agents.json`. It did not exclude the agent's
own record, nor scope to the agent's own children. Observed on live disk: a
leaf worker `worker1` — the only record in `agents.json`, `parent=None` — saw
itself as a "live subagent", so it skipped D2 (no-live-subagents → allow),
reached D5, and armed a watcher for **its own** inbox. A leaf agent with no
children must hit D2 and allow immediately; only an agent that actually leads
live children should arm/block.

### Root cause

Two gaps: (1) the agent record written at spawn
(`server_simple._do_spawn`) carried no parent linkage, so there was no way to
tell a caller's children from unrelated agents; #34 propagated
`AGENT_PARENT_NAME` into the child's *env/MCP config* but never onto the
*record*. (2) `_live_subagent_names` had no self-exclusion and no child
scoping, and the only existing test faked an empty list — so the self-count
path was never exercised.

### The fix

1. **Record the parent at spawn.** The spawn record now stores
   `"parent": IDENTITY` — the spawning lead's identity, the same value #34
   propagates as `AGENT_PARENT_NAME` into the child's MCP config
   (`_write_mcp_config`/`_write_pi_mcp_config` `parent_name=IDENTITY`). Naming
   follows #34's parent convention rather than inventing a parallel one.
2. **Scope + self-exclude in `_live_subagent_names(session_dir, identity)`.** It
   now returns only records whose `parent == identity`, always excluding the
   record whose `name == identity` (self). `identity` is the value `evaluate`
   already resolved via `_resolve_identity` (AGENT_NAME else `--reader`
   fallback). A leaf → empty → D2 allow; a sibling (same parent, parent != me)
   is not a child → still D2; a real live child → non-empty → proceeds.
3. **Legacy fallback.** When **no** record in the registry carries a `parent`
   key (pre-fix sessions), scoping by parentage is impossible, so it falls back
   to "every non-terminal agent except self" — still self-excluding, so a
   legacy leaf can never regress into the self-count bug, while a legacy lead
   still sees its (unscoped) live agents. In a mixed session the scoped
   predicate applies and parentless records simply do not match.

### Red → green evidence

New `TestLiveSubagentScoping` in `tests/test_lead_wake.py` drives the real
`_live_subagent_names` through `evaluate()` with `server_simple._load_agents`
monkeypatched to faked `agents.json` content:

- `test_leaf_agent_only_self_record_allows` — only-self record (the exact
  observed bug) → D2 allow.
- `test_sibling_is_not_counted_as_child` — self + sibling (same parent) → D2
  allow.
- `test_real_live_child_proceeds_past_d2` — self + a live child (parent==me) →
  NOT D2 (lands on D5 arm instruction).
- `test_terminal_child_is_not_live` — a killed child → treated as not-live →
  D2 allow.
- `test_legacy_records_without_parent_still_exclude_self` — pre-fix record with
  no parent field → fallback still excludes self → D2 allow.

Red (fix stashed, source reverted): the four scoping tests failed on the
self-count / signature, and the six existing tests that monkeypatch
`_live_subagent_names` failed on the new two-arg signature — `15 failed, 4
passed`. Green (fix restored): `19 passed`. Full suite: `713 passed, 3 skipped`
→ **718 passed, 3 skipped** after the fix.
