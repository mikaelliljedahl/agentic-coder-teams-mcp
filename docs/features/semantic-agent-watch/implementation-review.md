# Independent post-implementation review — semantic agent watcher

**Reviewer:** Claude Code (Opus) — `semantic-watch-implementation-review`
**Reviewed:** complete uncommitted diff in `wt-semantic-agent-watch` against
`plan.md` and `plan-review.md`.
**Files inspected:** `src/claude_teams/messaging.py` (new),
`src/claude_teams/cli.py`, `src/claude_teams/server_simple.py`,
`src/claude_teams/backends/claude_code.py`,
`tests/test_cli_watch.py`, `tests/test_backends/test_claude_code.py`,
`tests/test_tool_descriptions.py`, `tests/test_read_messages.py`, `README.md`,
`CLAUDE.md`, `implementation.md`.
**Date:** 2026-07-19

---

## Verdict

**Approved — no blocking findings.**

The implementation faithfully realizes the approved plan and closes every accepted
plan-review finding (B1, N1–N5, T1–T6). The core watcher logic is correct:
edge-triggered state-marker semantics with unconditional baseline advancement,
independent non-consuming inbox detection with a pre-loop check that closes the
send-before-watch race, deterministic `message > waiting > output` precedence, and
a stable single-line JSON wake contract. The nested-identity fix mirrors the Codex
backend exactly and is verified against the `SpawnRequest` contract. I verified the
suites locally:

```
uv run pytest tests/test_cli_watch.py tests/test_backends/test_claude_code.py \
  tests/test_tool_descriptions.py tests/test_read_messages.py -q   # 92 passed
uv run pytest -q                                                    # 473 passed
uv run ruff check <modified files>                                 # All checks passed
```

The remaining items are non-blocking test-coverage and hygiene notes.

---

## Verification of accepted plan-review findings

| Finding | Status | Evidence |
|---|---|---|
| **B1** identity in `build_env` | ✅ Implemented | `claude_code.py:280-282` sets `AGENT_NAME=request.name`, `AGENT_SESSION_ID=request.team_name`, `AGENT_PARENT_NAME=request.lead_session_id` — byte-identical to `CodexBackend.build_env` (`codex.py:566-570`). `merged_env.update(env)` in `process_manager` guarantees it overrides inherited values, fixing both the absent (root) and leaked-from-parent (nested) cases. Test `test_supplies_child_identity_for_nested_orchestration` + updated key-count assertion (`len(env) == 5`). |
| **N1** custom-pattern × inbox | ✅ Implemented | `--inbox/--no-inbox` flag (default on). `test_watch_no_inbox_preserves_artifact_only_behavior`. |
| **N2** contract note branches on reason | ✅ Implemented | `_DISK_CONTRACT_NOTE`, `spawn_agent`, `agent_watch_paths` all instruct `message→read_messages`, `waiting→status`, `output→inspect`, plus exit-2 re-check. `test_tool_descriptions.py` asserts `reason="message"`, `read_messages`, `reason="waiting"`, `exit 2`, `re-check`. |
| **N3** exit-2 recovery + edge-trigger test | ✅ Implemented | Documented in note/README/docstrings; `test_watch_preexisting_waiting_marker_is_not_a_new_edge`. |
| **N4** shared unread/validity helper | ✅ Implemented | `messaging.py` provides `read_inbox_by_sender` / `load_inbox_cursors` / `unread_sender_counts`; `server_simple.read_messages`, `_sender_message_count`, `_sender_unread_count`, and the CLI watcher all consume them. The prior duplicate copies were deleted from `server_simple.py`. |
| **N5** partial-append coverage | ✅ Implemented | `test_watch_detects_completed_append_after_partial_line`. |
| **T1** `build_env` assertion | ✅ | see B1 |
| **T2** precedence | ✅ | `test_watch_message_reason_wins_over_waiting` |
| **T3** startup `waiting` not an edge | ✅ | `test_watch_preexisting_waiting_marker_is_not_a_new_edge` |
| **T4** wake JSON shape | ✅ | exact-dict assertions for all three reasons |
| **T5** semantic filter under custom state pattern | ✅ | `test_watch_custom_state_pattern_remains_semantic` |
| **T6** single JSON object | ✅ | `len(lines) == 1` + `json.loads` assertions |

Existing `test_cli_watch.py` cases are unchanged in behavior (only cosmetic
line-joining); their `"<filename>" in stdout` substring assertions still hold
because the filename (no path separators) survives inside the JSON-escaped path —
no Windows backslash-escaping hazard for the substring checks.

---

## Correctness assessment (verified OK)

- **No missed message (send-before-watch).** `inbox_before` is snapshotted
  (`cli.py:169`), then an *absolute* `unread_sender_counts` pre-loop check
  (`171-181`) catches anything already present, and any append arriving before the
  first loop iteration changes `(mtime_ns,size)` so `inbox_after != inbox_before`
  re-checks. No append can slip through the pre-loop/loop seam.
- **Precedence is structural, not timing-dependent.** Within one iteration the
  message branch (`190-200`) is evaluated before the waiting/output branches
  (`218-224`), so `message` wins whenever both are ready in the same interval,
  regardless of thread scheduling. `waiting` is checked before `output`. Matches
  plan §4.
- **Cursor-only change does not wake.** The watcher tracks the inbox file's
  `(mtime_ns,size)`, never `.pos.json`; a `read_messages` drain rewrites only the
  cursor, leaving inbox identity unchanged → no spurious wake. Confirmed by
  `test_watch_does_not_wake_for_consumed_message` (consumed count yields empty
  `unread`).
- **Baseline advances on every edge** (`before = after`, `cli.py:216`, and
  `inbox_before = inbox_after`, `201`) including ignored `running`/corrupt markers
  and consumed inbox changes, so a non-ready write cannot re-fire forever.
- **Stale `waiting` at startup is not an edge** — it is captured in the initial
  `before` snapshot and never appears in `changed`.
- **JSON serialization is safe.** `_snapshot_mtimes` keys and `_changed_paths`
  return `str` paths (`str(entry)`, `cli.py:86`), so the `path` embedded in the
  wake record is always JSON-serializable — no `Path`-not-serializable crash.
- **Nested-identity side effects are benign.** The per-agent `--mcp-config` env
  block still sets the MCP subprocess identity (unchanged in `server_simple.py`);
  the CLI process now *also* carries the same `AGENT_NAME=request.name`, so the
  subprocess's inherited value and its config-block value agree. Grandchild spawns
  get their own name via `build_env` override. No identity regression.
- **`read_messages` clamp invariant preserved.** `read_inbox_by_sender` returns
  `{}` for a missing inbox but does **not** short-circuit the function, so the
  "must not early-return — a stored forward cursor still needs clamping"
  invariant (`server_simple.py:1393-1407`) is intact.
- **Windows.** `(mtime_ns, size)` snapshotting is retained; cross-process
  partial-append tolerance is exercised on the actual watched inbox file.

---

## Non-blocking findings

### NB1. TDD case #3 (inbox wake *while the state marker is actively `running`*) not tested as a combined scenario
The literal plan/red case #3 — "an inbox append wakes while the state marker
remains `running`" — is not tested as one scenario. Inbox wake is tested standalone
(`test_watch_wakes_for_preexisting_unread_message`, partial-append), and
message-over-waiting precedence is tested, but no test drives a concurrent
`running` marker churn together with an inbox append and asserts the message still
wins. The code path is clearly correct (running markers are dropped by
`_waiting_agent`, and the message branch precedes the state branch), so this is a
coverage gap, not a defect.
**Fix:** add a test where `state-worker.json` is (re)written `running` in the same
interval as an inbox append and assert `reason == "message"` and exit 0.

### NB2. No positive test that inbox wake stays active alongside a custom `--pattern`
Only `--no-inbox` is pinned. The actual backward-incompatible behavior change from
N1 — that a *custom-pattern* caller now *also* wakes on inbox traffic by default —
is neither asserted nor guarded. A future refactor could silently disable inbox
watching under a custom pattern and no test would catch it.
**Fix:** add a test invoking `watch <dir> --pattern report.md` (no `--no-inbox`)
with an unread inbox message and assert it wakes with `reason == "message"`.

### NB3. Corrupt `.pos.json` tolerance is only tested transitively through `read_messages`
`load_inbox_cursors`'s malformed-cursor handling is exercised by
`tests/test_read_messages.py::test_corrupt_cursor_file_treated_as_empty` (via the
shared helper), which satisfies plan red-case #7 at the helper level. There is no
CLI-watcher-level test writing a garbage `.pos.json` and asserting no false wake,
and there is no dedicated `test_messaging.py` for the new module.
**Fix (optional):** a one-line CLI test (unread inbox + corrupt `.pos.json` →
still wakes) or a small `test_messaging.py` unit test would lock the module's
contract directly. Low risk given the transitive coverage.

### NB4. Default-on inbox under a custom pattern is a silent behavior change for existing scripts — surface it more loudly
This is the intended, plan-accepted N1 resolution, but existing callers running
`win-agent-teams watch <dir> --pattern output-*.md` purely to await an artifact
will now *also* wake on unrelated inbox traffic unless they add `--no-inbox`. The
README mentions `--no-inbox` but frames it as a feature, not as a migration note
for existing artifact-only callers.
**Fix:** one sentence in the README/PR description calling out the behavior change
and the `--no-inbox` opt-out for pre-existing custom-pattern callers.

### NB5. Timing-based tests may be mildly flaky under heavy CI load
Several new tests (`test_watch_ignores_running_transitions_until_waiting`,
`test_watch_detects_completed_append_after_partial_line`,
`test_watch_message_reason_wins_over_waiting`) coordinate a writer thread with
`time.sleep` steps against a reduced `_WATCH_POLL_SECONDS`. They passed reliably
locally and carry generous timeout margins (2 s vs. sub-100 ms steps), and the
precedence test is robust by construction. Noted as a watch-item, not a defect; no
change required now.

### NB6. Minor hygiene
- `claude_code.py` gained a blank line after the module docstring and `cli.py`'s
  unrelated `backends` command was reflowed — both are ruff-format cosmetics that
  rode along with the feature. Harmless, but `CLAUDE.md` asks to keep unrelated
  changes out of the branch.
- In the loop, `path.name.startswith("state-") and path.suffix == ".json"`
  (`cli.py:207`) duplicates the identical guard inside `_waiting_agent`
  (`115`). Trivial; the outer check could be dropped and non-state paths routed
  to `outputs` by `_waiting_agent` returning `None` — but that would change
  output classification, so leaving it is the safer choice. No action needed.

---

## Documentation & scope

- `implementation.md` accurately records red (11 failures), green (110/473
  passed), lint, and the deviation rationale (identity supplied only to the MCP
  subprocess before the fix). The duplicate-test-class defect and its correction
  are honestly disclosed.
- README and MCP tool descriptions consistently teach the reason-branching
  contract and exit-2 re-check.
- `CLAUDE.md` (new) captures the standing repository workflow as planned.
- Scope is disciplined; changes map to the plan's "files expected to change".

## Recommendation

Approve for PR. NB1–NB3 (small test additions) and NB4 (a one-line migration note)
are worth folding in before merge but do not block; NB5–NB6 are informational.

---

## Addendum — follow-up verification of NB1–NB4 fixes (2026-07-19)

**Scope:** re-inspection of the changes made after the review above. No production
code was re-reviewed for new behavior because none changed: `cli.py` (+139),
`server_simple.py` (+140/-95), `claude_code.py` (+6), and `messaging.py` are
byte-identical to the reviewed state. All fixes are test-only plus documentation.

**Result: all four fixes are correct. Approval stands — still no blocking findings.**

### NB1 — message wake during active `running` churn ✅
`test_watch_message_wakes_during_running_marker_churn` (`test_cli_watch.py:313`)
writes a `running` marker and an inbox append in the same instant and asserts
`reason == "message"` with `from == ["worker"]`. This is exactly plan red-case #3
("an inbox append wakes while the state marker remains `running`") and closes the
gap as specified. Correct.

### NB2 — inbox stays enabled under a custom pattern ✅
`test_watch_custom_pattern_keeps_inbox_enabled_by_default` (`:338`) invokes
`--pattern report.md` with no `--no-inbox`, with an unread inbox present, and
asserts a `message` wake. This pins the N1 behavior-change *direction* that was
previously unguarded, so a future refactor disabling inbox under a custom pattern
would now fail. Correct, and it pairs properly with the existing
`test_watch_no_inbox_preserves_artifact_only_behavior` to cover both directions.

### NB3 — corrupt cursor handling at the watcher level ✅
`test_watch_treats_corrupt_cursor_as_unread` (`:352`) writes a malformed
`inbox-team-lead.pos.json` (`{broken`) alongside an unread message and asserts the
watcher still wakes with `reason == "message"`. The asserted semantics are right:
`load_inbox_cursors` returns `{}` on `JSONDecodeError`, so `consumed = 0` and the
message reads as unread. This is the safe failure direction — a corrupt cursor
causes a redundant wake rather than silently swallowing a message — and the test
name states that intent accurately. Correct.

### NB4 — README migration note ✅
README now carries an explicit paragraph after the CLI block: *"Inbox wake is now
enabled even when `--pattern` is supplied. Existing scripts that used a custom
pattern exclusively to await an artifact should add `--no-inbox` to preserve that
behavior."* This frames the change as a migration for existing callers rather than
only advertising the flag, which is what NB4 asked for. Correct.

### Verification run

```
uv run pytest tests/test_cli_watch.py tests/test_backends/test_claude_code.py \
  tests/test_tool_descriptions.py tests/test_read_messages.py \
  tests/test_agent_status.py -q      # 113 passed
uv run pytest -q                     # 476 passed
uv run ruff check tests/test_cli_watch.py   # All checks passed
```

`test_cli_watch.py` went from 16 to 19 tests; the full suite from 473 to 476, which
matches `implementation.md`.

### One trivial doc nit (non-blocking, not a code issue)

`implementation.md:46` still records **`110 passed`** for the focused run. That
command's actual current result is **`113 passed`** — the focused count was not
updated when the three new tests landed, although the full-suite figure on line 53
*was* correctly updated to 476 and the NB1–NB4 summary paragraph (line 82) is
accurate. Recommend changing line 46 from `110 passed` to `113 passed` so the
recorded evidence matches a re-run. Documentation only; no bearing on the verdict.

**Final verdict: approved for PR.**
