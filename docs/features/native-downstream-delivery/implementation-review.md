VERDICT: CHANGES_REQUESTED

Independent Codex post-implementation review of `feat/native-downstream-delivery`
at `ffa3a44`, against plan v3.1 (including the §2.9 overrides), plan-review
rounds 1–4 and their dispositions, spikes, implementation notes, and the Linux
smoke runbook. Scope: `git diff 471a175..HEAD -- src tests` and the commit log;
upstream #69 and #71 are excluded from feature findings. Source and tests were
left unchanged. Reproductions used temporary directories and injected transports.

1. **MAJOR — Flag-off resume of a previously native child disables its new hooks and preserves stale eligibility.**
   `src/claude_teams/server_simple.py:3062`,
   `src/claude_teams/server_simple.py:3078`,
   `src/claude_teams/server_simple.py:5074`,
   `src/claude_teams/backends/process_base.py:104`.

   Start a child with native wake enabled and an epoch-5 waiting marker, then
   restart the parent server with the master flag off and resume that child.
   `_dispatch_extra` returns no epoch; the process environment exports none;
   finalization merges an empty `_native_record_fields` result into the old
   record, preserving epoch 5 and the old `interactive`/`codex_home` facts.
   The new root-spawned child's hooks use epoch 0. `hooks.py:145` therefore
   drops **every** new hook, including `UserPromptSubmit` and `SessionStart`,
   leaving the predecessor's waiting marker visible. A subsequent follow-up
   treats the actively running child as idle and authorizes another shutdown
   and resume. Re-enabling native flags also exposes stale Codex eligibility;
   Claude capability/marker epochs cannot agree until another flag-on resume.
   This is the unimplemented recovery case explicitly identified in round-4
   implementation note R4-2; pristine flag-off compatibility does not require
   retaining invalid recovery metadata on an already-native record.

   **Reproduction:** using the real `follow_up_agent` path, real hook writes,
   and the existing `test_native_selection.env` fixture with its fake backend:
   seed epoch 5 and emit `Stop`; remove the epoch/master environment variables;
   deliver a flag-off follow-up; emit the replacement host's
   `UserPromptSubmit`; deliver a second follow-up. Observed:
   `first_status=delivered`, `first_resume_epoch=None`, `record_epoch=5`,
   `marker_state=waiting`, `second_status=delivered`, `resume_count=2`.

   **Suggested fix:** for records with native recovery metadata, mint a fresh
   epoch before resume even when the master flag is off, export that epoch
   independently of enabling native delivery, and persist the matching epoch
   and current launch facts during finalization. Preserve the existing empty
   metadata/env behavior for pristine flag-off records. Add an end-to-end
   on → off resume → on regression that verifies request/env/record epoch
   equality, accepts the replacement host's running hook, refuses to replace
   it while busy, and updates TTY/home facts. Existing record-field tests
   cover fresh flag-off spawns, not this transition.

2. **MAJOR — Codex lead wake can queue a replacement provisional registration using an unverified home.**
   `src/claude_teams/native_wake.py:834`.

   `_check_codex_lead` verifies registration A, checks its generation and
   incarnation through `_codex_lead_key`, then reads the registration again.
   This last read checks only `thread_id`, but uses the newly read
   `codex_home` for the queue call. A concurrent `set_lead_wake` with the same
   thread and a different home can replace A after the key check. The queue
   then uses registration B despite B having a different generation and
   potentially being `provisional` or belonging to a different incarnation.
   The verified state still belongs to A's home. This violates §2.6's rule
   that provisional registrations never queue and that the queued target,
   generation and incarnation are revalidated.

   **Reproduction:** use the existing `test_codex_lead_wake` notifier/fakes;
   inject replacement on the fourth registration read of its first tick
   (the read at line 834), retaining the thread but setting home to
   `/other-home`, `spawned=True`, and `bound=None`. Observed:
   `final_status=provisional`, verification only of `/codex-home`, and one
   queue call using `/other-home`. No actual Codex process was launched.
   The existing revalidation tests mutate during verification, before the
   key check, so they miss this interleaving.

   **Suggested fix:** validate the final registration snapshot's active status,
   generation, host incarnation, thread and home against the state key and
   verified snapshot, and queue that same validated snapshot. Serialize the
   dispatch decision with registration replacement/clear under the
   registration lock so a replacement cannot bypass the gate. Add a
   deterministic barrier test replacing the registration between the key
   check and final read, including same-thread/different-home and provisional
   replacements; neither should queue under the old verified state.

Counts: **0 BLOCKER, 2 MAJOR, 0 MINOR, 0 NIT**.

Validation: focused delivery/selection/mailbox/poster/lead-wake/runner/hooks/
pipe/propagation suites: **710 passed, 1 skipped**. Full `uv run pytest -q`:
**2587 passed, 10 skipped**. `uv run ruff format --check .` and
`uv run ruff check .` passed. `uv run ty check` failed with the two existing
Windows `unresolved-attribute` diagnostics at
`scripts/herdr_nested_check.py:152` (`pane_id`) and
`tests/test_join_team.py:750` (`BaseContext.Process`); both offending
expressions are present in upstream baseline `202cfc4`.

No separate duplicate-presentation defect was found in the reviewed N5,
native absence settlement, frozen-carrier scanning, carrier-ref CAS, mailbox
recovery/retention, epoch-fenced taken retraction, poster consumption rollback,
or pipe cancel/drain/parking paths. This is not live transport approval:
S-2/S-3 and the required real-host merge smokes remain unverified in the
provided records. Run and record the plan's N1/N3/N5/N6/N7/N8 gates before merge.

## Disposition (lead, round 1)

1. **Accepted, fixed.** A record that already carries a `dispatch_epoch` gets a
   fresh epoch on every resume, whatever the flags: `_dispatch_extra` mints it,
   `process_base` exports it whenever the request carries one, and
   `_native_record_fields` persists it with current `interactive`/`codex_home`.
   Pristine flag-off records are unchanged (no epoch, no env, no
   `dispatch-epochs.json`). Regression: `test_flag_off_resume_of_native_record_refreshes_recovery_metadata`
   (on -> off -> on, real hooks) and `test_flag_off_resume_of_pristine_record_adds_no_native_metadata`,
   plus three unit tests in `test_native_record_fields.py`.
2. **Accepted, fixed.** `_check_codex_lead` now requires the post-plan read to be
   `active` at the key's generation and host incarnation, binds verification to
   the `(thread_id, codex_home)` pair, and makes the final check plus the
   `codex queue` call under the registration lock, queueing exactly the verified
   pair. The lock is held across the bounded queue call (default 15 s, under the
   30 s lock timeout). Regression: `test_replacement_after_the_key_check_never_queues_it`
   (same-thread/other-home and provisional replacements at reads 2-4).

Gates after the fixes: ruff format/check clean; ty only the 2 pre-existing
Windows diagnostics; pytest 2598 passed, 10 skipped.
