# Implementation review — codex-trust-prompt

## Part A

Reviewer: Claude Opus (independent post-implementation review), 2026-09-26.
Scope: the uncommitted diff on `fix/codex-trust-prompt` against HEAD `c2868be`, plus the untracked
`tests/test_startup_diagnosis.py`. I reviewed it against `plan.md` rev 3 §3, all three rounds of
`plan-review.md` (round-3 dispositions 3 and 6 in particular) and `implementation.md`.

### Verification performed

- Focused run: `tests/test_startup_diagnosis.py`, `test_correlation_transport.py`, `test_follow_up_delivery.py`,
  `test_agent_status.py`, `test_agent_output.py`, `test_join_team.py` and `test_native_wake_flag_off.py`.
  **361 passed.**
- Full suite: **1882 passed, 6 skipped**. `ruff format --check`, `ruff check` and `ty check` all pass on Linux.
- A semantic JSON diff of `tests/fixtures/native_wake/flag_off.json` against HEAD. Only
  `tools.check_agent`, `tools.list_agents` and `tools.agent_status` differ.
- I decoded the new fixture as cp1252, which is what `Path.read_text()` does on Windows under Python 3.12.
  The em dashes come out garbled, so the golden no longer matches there (finding 1).
- `grep trust_cwd|projects=` over `src/` finds nothing. **Part B is not implemented here.** That is correct.

### Checklist results

| Area | Result |
|---|---|
| Launch-start capture (spawn) | `server_simple.py:3516-3518`. Mode and hook fields are computed first, `launch_started_at = time.time()` is taken on the line before `b.spawn`, and both are persisted in the appended record (`:3540-3541`). `spawned_at` is unchanged. OK. |
| Launch-start capture (follow-up) | `:4848-4853`. The capture happens after the old-PID graceful shutdown or kill (`:4839-4846`) and immediately before `plan.backend.resume`. It is passed to `_finalize_follow_up` (`:4398`), which refreshes the record on delivered/unconfirmed. On a failed resume the record is not written, so the metadata is discarded, as before. OK. |
| Marker comparison | `_startup_diagnosis` (`:6360-6414`) treats a marker as absent when `ts < launch_started_at` or `ts` is non-numeric/bool (via `_marker_timestamp`). The comparison is diagnostic-only; `_resolve_agent_state`/`_heartbeat_fields` inputs are untouched. OK. |
| Missing capability (R3 disp. 3) | `_state_hook_args` uses `getattr(...)`/`callable` and yields `[]`. Spawn with a minimal `_FakeBackend` records `hooks_wired=False` (`test_correlation_transport.py`). `ty` is green. The claude kill switch `WIN_AGENT_TEAMS_STATE_HOOKS=0` returns `[]`, so `hooks_wired=False`. OK (see finding 3 for the exception path). |
| Surface consistency | All five surfaces call the same helper with the record and the raw marker. `_empty_agent_check` returns `None`/`None` in both modes. No new binding resolution was added; the spy count is 3, all pre-existing. `state`, `stalled` and `heartbeat_age_s` are unchanged. OK. |
| Legacy / external | A missing `launch_started_at` or a non-bool `hooks_wired` gives `None`. `backend == "external"` gives `None`. Join records carry no launch fields. OK. |
| Docstrings (R3 disp. 6) | The exact tri-state predicate, the env override, `heuristic`, `folder-trust` and the clock caveat appear in all three tools. The headless hint says "CLI startup problem or hook failure". OK, with nits 6 and 7. |
| Part B | Absent. OK. |

### Findings

1. **Major — the `flag_off.json` golden was re-serialised with literal UTF-8, which breaks the golden test on Windows.**
   `tests/fixtures/native_wake/flag_off.json` (14 changed lines).
   - **What changed:** the file is 28 lines of diff, but only three values changed semantically: the three tool
     descriptions that gained the new fields. The rest of the diff converts every `—`/`→` escape into
     a literal `—`/`→`, apparently because the file was regenerated with `ensure_ascii=False`. At HEAD the file
     was pure ASCII; it now has 14 lines with non-ASCII bytes.
   - **Why it matters:** `tests/test_native_wake_flag_off.py:20` reads the file with `read_text()` and no
     encoding. On Windows under Python 3.12 (the `.python-version`) that decodes as cp1252: `—` becomes `â€”`,
     and `test_tools_list_identical`'s `== expected` fails. Linux CI stays green, so the regression hides on
     exactly the platform the repo targets.
   - **Why the change is needed at all:** the fixture is a golden of every tool description. Changing three
     docstrings legitimately requires updating those three values; nothing else should change.
   - **Fix:** regenerate with the default `json.dumps(..., ensure_ascii=True, indent=2)` (matching HEAD's
     format) so the diff is limited to the three description strings. Optionally also make the loader
     `read_text(encoding="utf-8")` so the test is encoding-independent.

2. **Minor — the follow-up ordering test cannot detect a capture after `resume`.**
   `tests/test_follow_up_delivery.py:401-431`. The `env` fixture freezes `server_simple.time.time` at a constant
   `1_000.0` (`:198`). As a result, `launch_started_at <= marker_ts` holds whether the capture happens before or
   after `resume()`: both are 1000.0. Plan §3.A5(3) asks the test to prove the capture precedes the child
   marker.
   **Fix:** inside this test, patch `server_simple.time.time` with a strictly increasing counter (for example
   `itertools.count(1000.0, 0.001)`) and assert `launch_started_at < marker_ts`. Optionally also record the
   `graceful_shutdown`/`kill_process` call time and assert the capture comes after it, which covers "after the
   old PID shuts down".

3. **Minor — in follow-up, an exception from an optional adapter capability now fires after the old PID is stopped.**
   `server_simple.py:4848-4850` (`_launch_mode_fields`), via `_state_hook_args` at `:297-300`.
   - **Why it matters:** the call sits outside the inner `try`, so a third-party adapter whose
     `state_hook_args` raises would kill the old agent, skip `resume` and `_finalize_follow_up`, and propagate.
     The outer `finally` still releases the lease. Built-in adapters are pure and cannot hit this. Still, the
     disposition-3 intent was that an unknown adapter must never break a launch because of this diagnostic.
   - **Fix:** make `_state_hook_args` fail-safe with `try: ... except Exception: logger.debug(...); return []`,
     which records `hooks_wired=False`. Alternatively compute `launch_fields` before the old-PID shutdown. Add
     one test with an adapter whose `state_hook_args` raises.

4. **Minor — the end-to-end tests do not cover a follow-up refreshing an existing `launch_started_at`.**
   The follow-up test starts from a record without launch fields. **Fix:** seed the record with
   `launch_started_at=1.0, hooks_wired=True, launch_interactive=False` and assert that all three values are
   replaced after a delivered follow-up. This is the path where a stale value would silently mask or mis-date
   a new launch.

5. **Nit — Pi's `state_hook_args` duplicates rather than reuses the builder's logic.**
   `backends/pi.py:520-523`. The plan says the method should return "the exact argv helper each builder uses".
   Pi re-implements the `pi_state_extension_path` branch of `_extension_args`. It is equivalent today, but the
   two can drift. **Fix:** extract `_state_extension_args(request)`, call it from both `_extension_args` and
   `state_hook_args`, and keep the wake extension separate.

6. **Nit — the `list_agents` docstring mentions a field that `list_agents` does not return.**
   `server_simple.py:6483` says "Neither field changes `state` or `stalled`", but `list_agents` has no
   `stalled` field. **Fix:** drop `or stalled` there.

7. **Nit — the reference doc is imprecise in two places.**
   - `docs/reference/agent-messaging-protocol.md:96` says "A successful follow-up refreshes all three fields".
     The finalizer also refreshes them on `delivery_unconfirmed` with a live new PID. **Fix:** say "a follow-up
     that records a new PID (delivered or unconfirmed)".
   - `:677` exceeds the file's wrap width. **Fix:** re-wrap it.

8. **Nit — `implementation.md` omits the fixture re-serialisation and the outstanding Linux smoke.**
   **Fix:** after fixing finding 1, note that `flag_off.json` changes only the three tool descriptions.
   Plan §3.A5 requires the Linux smoke (interactive codex in a fresh `/tmp` dir, which should show `True` plus
   the Codex hint) before the PR, and it is still marked TODO. Record its result before opening PR 1.

### Verdict

**APPROVE WITH CHANGES.**
- **Before PR 1:** fix finding 1 (a Windows-only red golden) and run the Linux smoke (finding 8).
- **Should fix in this PR:** findings 2-4. Each is small and test-local or a one-line guard.
- **Optional:** findings 5-7.

The core design matches rev 3 Part A and round-3 dispositions 3 and 6. Capture points precede process start on
both paths. The diagnostic uses only the record and the raw marker, all five surfaces agree, public
`state`/`stalled`/`heartbeat_age_s` are untouched, legacy and external records yield `None`, and Part B is
absent.

## Dispositions (lead, 2026-09-26) — Part A

| # | Disposition |
|---|---|
| 1 | Accepted (major). Regenerate `flag_off.json` with `ensure_ascii=True`, so only the three description values differ from main. Read goldens with `encoding="utf-8"` in `tests/test_native_wake_flag_off.py`, and check any other golden readers. |
| 2 | Accepted. The follow-up ordering test uses a monotonically increasing fake clock and asserts capture happens after the old-PID stop and before `resume`. |
| 3 | Accepted. The capability helper catches exceptions from third-party `state_hook_args` and returns `[]`, which gives `hooks_wired=False`. Test it, including that follow-up still resumes. |
| 4 | Accepted. Add a test that a follow-up replaces existing launch fields on the record. |
| 5 | Accepted. Pi shares its hook-arg logic with the builder. `list_agents` docstring drops the `stalled` mention. Fix the reference-doc wording. Smoke recorded in `implementation.md` (lead run: PASS). |
