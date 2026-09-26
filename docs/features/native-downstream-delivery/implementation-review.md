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

## Round 2

VERDICT: CHANGES_REQUESTED

Reviewed `git diff ffa3a44..57e4341` and the lead's dispositions at HEAD
`57e4341`. Both original reproductions are fixed for the scenarios reviewed:

- **Round-1 finding 1:** the direct flag-off resume now uses epoch 6 in the
  request, exported environment and record. Re-running the hook/follow-up
  reproduction with the actual exported epoch gives `marker_state=running`,
  `second_status=queued`, and `resume_count=1`. Current launch facts are
  refreshed, and pristine flag-off requests without an ambient epoch remain
  unchanged. A remaining nested-spawn case is reported below.
- **Round-1 finding 2:** verification is bound to the thread/home pair, and
  the final active/generation/incarnation/pair check and queue share the
  registration lock. The old fourth-read window no longer exists. Replacing
  before the final check produces zero queues. With an actual concurrent
  registration writer during the injected queue, the writer waits, the queue
  uses the original verified active registration, and the writer completes
  after return. It also completes after an injected queue exception: the
  context manager releases the lock and the notifier backs off. No lock
  inversion was found in this path; registration updates/corroboration use
  the same lock without acquiring the notifier owner lock. The default queue
  timeout is 15 seconds, with up to 5 seconds of bounded reap on failure;
  holding the registration lock across that call delays updates deliberately.

1. **MAJOR — A new flag-off descendant inherits its parent's epoch without recording it, so a later resume rejects its hooks.**
   `src/claude_teams/backends/process_base.py:115`,
   `src/claude_teams/server_simple.py:3069`,
   `src/claude_teams/backends/process_manager.py:705`.

   Resume a previously native nested lead with the master flag off. The fix
   correctly exports its new epoch, for example 6. Its own MCP server then
   spawns a new child while the flag is still off. This child's record has no
   `dispatch_epoch`, so `_dispatch_extra` mints none; `process_base` supplies
   no epoch override either. The process manager merges `os.environ` into
   the child's environment, so the child inherits the **parent's** epoch 6.
   Its hooks write epoch-6 markers while its record and per-name watermark
   remain pristine. After the master flag is enabled, the first resume of
   that child mints epoch 1. Every replacement-host hook is rejected behind
   the epoch-6 marker. The old waiting marker again permits shutdown/resume
   of a busy child. Thus the newly exported recovery epoch must also be
   isolated from descendants that do not receive their own epoch.

   **Reproduction:** pass a request with empty `extra` through the real
   `ClaudeCodeBackend._spawn_with_command` and
   `WindowsProcessManager.spawn_process`, with master off and ambient epoch
   6, faking only `_popen` and ancillary window actions. The environment at
   `_popen` contains epoch 6 although the request has none. Then, using the
   real hook/store/follow-up path in `test_native_selection.env`, write that
   pristine child's `Stop` at inherited epoch 6, enable flags, resume it, and
   emit `UserPromptSubmit` at the newly minted epoch. Observed:
   `first_resume_epoch=1`, `marker_state=waiting`,
   `second_status=delivered`, `resume_count=2`. All artifacts were temporary;
   no real agent process was launched.

   **Suggested fix:** prevent a child from inheriting the parent's dispatch
   epoch when its request has no minted epoch. An explicit blank override
   when an ambient epoch exists makes the child hooks use epoch 0 despite
   the process manager's environment merge; alternatively allocate and
   persist a distinct child epoch in this recovery context. Keep genuinely
   pristine flag-off environments unchanged. Add a nested flag-off spawn →
   flag-on resume regression that observes the **merged launch environment**,
   then real hooks, and verifies a busy replacement cannot be resumed again.
   The new pristine-record test only inspects the backend's override dict,
   and its fake `spawn_process` omits the inherited environment merge.

Round-2 counts: **0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT**.

Validation at `57e4341`: all **11 new regression cases passed**; broader
selection/record/lead-wake/propagation/runner/native-dispatch/poster/hooks/tool
text suites returned **606 passed, 1 skipped**. Both original reproductions,
concurrent registration success/exception checks, and the new nested-spawn
reproduction ran separately with temporary artifacts and fake transports.
Repository-wide ruff format/check passed. `ty check` still reports only the
two previously documented Windows diagnostics. The full pytest suite was
not rerun in this round; the lead's full-suite result above is unchanged and
is not presented as an independent run. Required live merge gates remain
unverified. Only this review file was edited; no source/tests or commits.

## Disposition (lead, round 2)

1. **Accepted, fixed.** `process_base._spawn_with_command` never lets a child
   inherit the parent's `WIN_AGENT_TEAMS_DISPATCH_EPOCH`: a minted epoch is
   exported as before; otherwise, when the server itself has an ambient epoch,
   the child gets an explicit blank override (Popen env, Windows Terminal
   wrapper, tmux/terminal/herdr shell `export ...=''`), which hooks read as
   epoch 0. With no ambient epoch nothing is added, so pristine launches and
   the flag-off goldens are unchanged. MCP configs and the Codex `-c` override
   never carry the epoch. Regression:
   `test_flag_off_nested_spawn_does_not_inherit_the_parents_epoch` (real
   `_spawn_with_command` + `WindowsProcessManager.spawn_process` merged env,
   real hooks, busy replacement not resumed), 7 cases in
   `test_native_record_fields.py`, `test_blank_epoch_is_no_epoch`.
   **Accepted residual:** a tmux pane inherits the tmux *server's* environment;
   if that server was started by an agent with an epoch while this MCP server
   has none, the pane can still see it. Closing it would add a key to every
   pristine launch, which the flag-off baseline forbids.

Gates: ruff format/check clean; ty only the 2 pre-existing Windows
diagnostics; pytest 2607 passed, 10 skipped.


## Round 3

VERDICT: **CHANGES_REQUESTED**

Reviewed `git diff 57e4341..c1e78d1` and the lead's round-2 disposition.
The original round-2 finding is fixed when the epoch is in the MCP process's
ambient environment: an unversioned child gets a blank override, a minted
child epoch takes precedence, and pristine launches without an ambient epoch
add no key. Blank is correctly interpreted as epoch 0. Re-running the nested
spawn through the real Claude backend and Windows process manager, with only
process creation/window effects faked, gives `initial_child_epoch=""`,
`first_resume_epoch=1`, `marker_state=running`, `resume_count=1`; the busy
follow-up queues. No additional regression in the changed branch was found.
The accepted tmux residual remains a correctness failure, as detailed below.

1. **MAJOR - The accepted tmux-server residual still lets a busy replacement be resumed again.**
   `src/claude_teams/backends/process_base.py:121`,
   `src/claude_teams/backends/process_manager.py:1561`,
   `src/claude_teams/server_simple.py:3050`,
   `src/claude_teams/hooks.py:145`.

   A tmux server started from an epoch-bearing agent can retain epoch 6.
   A later MCP process without that variable launches a pristine child into
   that server with the master flag off. The new conditional supplies no
   blank export; the pane inherits the server's 6 while the child's record
   and epoch watermark have no epoch. Its `Stop` writes a waiting marker at
   6. Enable native wake and resume the child: `_next_dispatch_epoch` consults
   only the record and watermark, allocating 1. The replacement's
   `UserPromptSubmit` at 1 is dropped behind marker 6. A further follow-up
   therefore treats the busy replacement as waiting and shuts it down/resumes
   it again. Accepting this residual leaves the round-2 failure reachable on
   a supported launcher.

   **Reproduction:** run the real `ClaudeCodeBackend._spawn_with_command`
   and `TmuxProcessManager.spawn_process` with no MCP ambient epoch, faking
   only the tmux subprocess/availability and selecting an existing session.
   Execute the resulting shell command with Git Bash under an inherited
   epoch-6 environment to model the pane's tmux-server environment: it prints
   6; neither the tmux client environment nor the shell's exports supplies
   an epoch override. Then use real hooks, store and follow-up code in the
   temporary `test_native_selection.env` fixture. Observed
   `first_resume_epoch=1`, `marker_state=waiting`, `marker_epoch=6`,
   `second_status=delivered`, `resume_count=2`. This tests the generated shell
   command and recovery path; it is not a live Linux/tmux smoke test.

   **Suggested fix:** preserve pristine flag-off launch bytes, but make a
   subsequent minted recovery epoch outrank the existing valid state-marker
   epoch as well as the record/watermark, persisting that higher watermark.
   The marker already represents native recovery state. Alternatively,
   detect and clear a retained launcher epoch before launch. Closing this
   failure does not require adding an environment key to every pristine
   launch. Keep the old-hook fence; add a server-only inherited epoch followed by
   flag-on resume regression which checks that the running hook is accepted
   and the busy replacement cannot be resumed again.

Round-3 counts: **0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT**.

Validation at `c1e78d1`: all **9 new regression cases passed**. Broader
selection/record/lead-wake/flag-propagation/queue-runner/native-dispatch/
poster/hooks/tool-text/flag-off-golden suites returned **653 passed,
1 skipped**. Repository-wide ruff format/check passed. `ty check` remains
red with the same two previously documented Windows `unresolved-attribute`
diagnostics (`scripts/herdr_nested_check.py:152`, `tests/test_join_team.py:750`).
The full pytest suite was not rerun independently this round; the lead's
2607-pass result above is attributed to the lead. Live Linux merge gates
remain unverified (WSL is not installed here). Only this review file was
edited; no source/tests or commits.

## Disposition (lead, round 3)

1. **Accepted, fixed; the round-2 residual is withdrawn.** `_next_dispatch_epoch`
   now mints `max(high-water, record dispatch_epoch, marker dispatch_epoch) + 1`,
   reading `state-<name>.json` defensively (missing, unreadable or non-positive
   int ignored). `_dispatch_extra` also mints with the flag off when the marker
   carries a nonzero epoch: such a marker is native recovery metadata, and a
   replacement at epoch 0 would otherwise have every hook dropped behind it.
   Launches with no marker epoch (including ordinary flag-off children, whose
   hooks write 0) are unchanged. Kill/force fences use the same function, so
   they also land above the marker. Regression:
   `test_flag_on_resume_mints_above_an_inherited_marker_epoch` (the tmux case),
   `test_flag_off_resume_of_pristine_record_mints_above_inherited_marker`, and
   mint unit tests.

Gates: ruff format/check clean; ty only the 2 pre-existing Windows
diagnostics; pytest 2623 passed, 10 skipped. (One run by the implementer saw
a single timing failure in the pre-existing stress test
`test_codex_member_wake.py::test_registration_send_stress_no_deadlock`; it
passed in isolation and on rerun.)


## Round 4

VERDICT: **APPROVED**

Reviewed `git diff c1e78d1..9cf207b` and the lead's round-3 disposition.
The round-3 MAJOR is resolved; withdrawing the tmux residual is appropriate.
No new findings.

- **Tmux recovery, master on and off:** repeated the round-3 reproduction
  through the real Claude backend and tmux process manager, executing the
  generated command in Git Bash with a simulated retained server epoch 6.
  The initial unversioned pane still inherits 6, but recovery now exports
  and persists epoch 7 under either flag setting. Real hooks/store/follow-up
  code gives `marker_state=running`, `second_status=queued`, `resume_count=1`.
  The inherited epoch no longer poisons replacement hooks. This is an
  independent generated-command/recovery reproduction, not live tmux smoke.
- **Ordinary flag-off compatibility:** independently resumed an ordinary
  child after its real hook wrote an epoch-0 waiting marker. Request and
  environment have no epoch, record has no native launch fields, and no
  watermark file is created. Existing flag-off goldens also pass. A positive
  marker now triggers recovery only where the old-hook fence requires it;
  missing, malformed, noninteger, boolean, negative and zero marker epochs
  do not create a positive recovery epoch by themselves.
- **Kill/force fences:** independently seeded offered, taken and posting
  mailbox entries at record epoch 3, with marker epoch 9. Both actual kill
  and CLI force paths persisted record/watermark epoch 10 before the first
  retract. Offered/taken became retracted; posting stayed posting and
  unresolved. The existing fence-failure tests also pass. Record removal
  still leaves the durable watermark for name reuse.
- **Concurrency:** the marker is read through its existing atomic-file
  reader without taking `state-<agent>.lock`; the watermark is still minted
  under its lock. The fix introduces no lock-order inversion and preserves
  the old-hook fence and native N5 barriers.

Round-4 counts: **0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT**.

Validation at `9cf207b`: all **16 new regression cases passed**. The broader
selection/record/lead-wake/flag-propagation/queue-runner/native-dispatch/
poster/hooks/tool-text/flag-off-golden suites returned **669 passed,
1 skipped**; operation-lease and kill suites returned **47 passed**.
Repository-wide ruff format/check passed. `ty check` remains red with the
same two pre-existing Windows `unresolved-attribute` diagnostics
(`scripts/herdr_nested_check.py:152`, `tests/test_join_team.py:750`). The full
pytest suite was not independently rerun this round; the lead's 2623-pass
result above remains attributed to the lead. This approval covers the code
review; required live Linux merge gates remain unverified here. Only this
review file was edited; no source/tests or commits.
