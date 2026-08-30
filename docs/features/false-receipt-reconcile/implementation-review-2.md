# Implementation review, round 2: false receipt reconciliation

Verdict: **APPROVED**

The two round-1 blockers are fixed in the implementation and covered by tests
that now reach the intended paths. I accept the three explicitly dispositioned
scope choices: `_RECONCILE_SAME_KEY` remains defensive, separate send-through
tests for `deliver_pending`/`send_message` are deferred because they share the
tested implementation, and the failed-row test seeds the barrier state rather
than invoking out-of-scope `kill_agent`.

## Round-1 blocker verification

### Failed-row marker removal: RESOLVED

The `_reconcile_before_resend` cleanup is now guarded by
`record.get("status") == STATUS_DELIVERED`; a terminal `failed` settlement
returns without calling `_clear_reconciled_marker`. `src/claude_teams/server_simple.py:5079-5097`

That is the correct location: `_reconcile_delivery_record` has already
finished and `record.update(stored)` has made the settled status authoritative,
so a failed result cannot accidentally clear the marker. The terminal-failed
barrier in `_settle_reconciled_attempt` still retains the marker as well.
`src/claude_teams/server_simple.py:5069-5078`, `src/claude_teams/server_simple.py:2271-2276`, `src/claude_teams/server_simple.py:2323-2328`

`test_a_late_receipt_after_a_failed_same_key_retry_does_not_resend` drives the
required sequence: first call leaves an unconfirmed row/marker, liveness is
then switched to dead, the clock is moved beyond the flush grace, the same-key
retry settles `failed`, the late nonce receipt is appended, and a new-key call
must return `prior_attempt_settled_failed` without a second backend resume.
`tests/test_follow_up_delivery.py:671-714`

Without the status guard, the same-key retry would clear the marker at
`_reconcile_before_resend`; the new-key call would miss the terminal-failed
barrier and enter `_prepare`'s normal send path. The test's `len(resume_calls) ==
1` assertion would then fail. `src/claude_teams/server_simple.py:4260-4340`, `src/claude_teams/server_simple.py:5079-5097`

### Settlement-write false green: RESOLVED

The revised `test_a_lost_settlement_write_sends_nothing_and_keeps_the_marker`
installs the failing `save_records` only after the old attempt exists. Its
one-shot function lets the new caller row be written on write 1 and fails from
write 2 onward; the `writes["n"] > 1` assertion proves the settlement
transaction was attempted. It also proves `k117` exists and that neither row
became terminal. `tests/test_follow_up_delivery.py:790-824`

This now reaches `_settle_reconciled_attempt`: `_open_delivery_record` writes
the new row first, then the reconciliation transaction mutates the old and
caller rows and fails on commit. `DeliveryStoreError` reaches the existing
fail-closed handler, so no backend resume starts and the marker remains.
`src/claude_teams/server_simple.py:2244-2297`, `src/claude_teams/server_simple.py:5291-5315`, `src/claude_teams/server_simple.py:5317-5332`

With settlement incorrectly allowed to proceed, the second save would not
return `delivery_store_unavailable`, and the test would fail its reason and/or
row-state assertions. This is no longer the earlier pre-creation false green.

## New tests

### Duplicate-nonce barrier: RESOLVED

`test_two_rows_carrying_one_nonce_are_a_barrier_too` copies the old durable row
under a second key, then calls the real follow-up path. The transcript scan is
positive, `_resolve_pending_row` sees two sender/target/nonce matches, and the
test asserts no second resume, marker retention, and no settled rows.
`tests/test_follow_up_delivery.py:639-668`, `src/claude_teams/server_simple.py:2224-2242`

The direct `_rows()` status check is appropriate here because
`delivery_status` is itself an active reconciler; using it could settle the
rows and hide the barrier state. `tests/test_follow_up_delivery.py:653-668`

### Option-only different request: RESOLVED

The original option-only test still proves the non-alias refusal when
`replace_if_idle=False` rejects an idle target. The paired
`test_an_option_only_difference_is_actually_sent` sets the child dead, so the
same option difference falls through to a real second resume, writes a receipt,
and verifies that the second row carries that resume's nonce. `tests/test_follow_up_delivery.py:516-565`

The implementation's fingerprint includes options, and the different-
fingerprint arm returns `None` from `_answer_reconciled_attempt`, allowing the
ordinary reservation/mark/resume path to run. `src/claude_teams/server_simple.py:2280-2287`, `src/claude_teams/server_simple.py:2330-2335`, `src/claude_teams/server_simple.py:4336-4340`

### Dispositioned coverage choices: accepted

- `_RECONCILE_SAME_KEY` is defensive for a partially written/inconsistent
  state. A healthy same-key `sent`/`unconfirmed` retry settles earlier in
  `_reconcile_before_resend`, and the new test covers that public path. Keeping
  the defensive verdict is reasonable. `src/claude_teams/server_simple.py:5066-5088`, `tests/test_follow_up_delivery.py:716-730`
- The `deliver_pending` and child `send_message` tests exercise the real public
  entry points and their alias behavior. Their send-through arms delegate to
  the same `_guaranteed_send`/`_guaranteed_delivery` implementation already
  exercised by the follow-up different-request tests; I accept the narrow
  decision not to duplicate those send tests. `tests/test_follow_up_delivery.py:733-786`, `src/claude_teams/server_simple.py:3362-3420`, `src/claude_teams/server_simple.py:5512-5620`
- The failed-row test directly seeds a terminal row because the implementation
  under test is the barrier decision, while ordinary `kill_agent` removes the
  agent after its row reconciliation. This is a valid scoped fixture, not a
  false claim that the kill entry point was exercised. `tests/test_follow_up_delivery.py:603-636`, `src/claude_teams/server_simple.py:5731-5847`

## A. Correctness of the implemented path

No remaining false-delivery or response/row contradiction was found in the
changed path.

- `_resolve_pending_row` filters the current sender and target and accepts only
  one matching nonce. `src/claude_teams/server_simple.py:2224-2242`
- `_settle_reconciled_attempt` settles the old row and, for an alias, the
  caller's row in one delivery transaction. It updates `record` after every
  persisted caller-row mutation. `src/claude_teams/server_simple.py:2263-2297`
- `_answer_reconciled_attempt` builds a delivered answer only for the
  `same_key`/`alias` verdicts. The alias provenance is copied from the updated
  caller record, and `_with_public_status` then reflects the persisted status,
  phase, nonce and identity. `src/claude_teams/server_simple.py:2323-2353`
- The `send` verdict returns `None`, leaving the current row for the normal
  `_mark_attempt_sent` and `_record_outcome` path; it does not report success
  for an unsent prompt. `src/claude_teams/server_simple.py:2330-2335`, `src/claude_teams/server_simple.py:4462-4508`
- Barrier rows are marked `queued/pending` with the barrier reason, while the
  agent marker is not popped. The response is built from the same updated
  caller record. `src/claude_teams/server_simple.py:2263-2269`, `src/claude_teams/server_simple.py:2323-2328`, `src/claude_teams/server_simple.py:5006-5024`
- `public_view` preserves the ordinary exact key set and conditionally exposes
  only alias provenance. The existing exact-key test still passes, and the new
  test covers both shapes. `src/claude_teams/delivery_store.py:232-260`, `tests/test_delivery_store.py:229-260`

The registry mutation in `_answer_reconciled_attempt` is correctly placed.
That helper is invoked inside `_prepare`'s existing registry transaction, so it
mutates the already-held `agent`, bumps its generation on successful marker
cleanup, and saves before returning. A send-through answer then reads that
bumped generation before `reserve_lease`; a barrier answer does not change the
agent and only saves when the pre-existing `changed` flag requires it.
`src/claude_teams/server_simple.py:4140-4141`, `src/claude_teams/server_simple.py:2323-2335`, `src/claude_teams/server_simple.py:4336-4365`

`_clear_reconciled_marker` is safe with respect to lock re-entry. Its only call
site is after `_reconcile_before_resend` has exited its delivery transaction,
and the helper opens the registry transaction itself. `_answer_reconciled_attempt`
does not call it while `_prepare` holds the registry lock. `src/claude_teams/server_simple.py:2356-2378`, `src/claude_teams/server_simple.py:5069-5097`

## B. Test quality and validation

The focused changed-file run completed with **76 passed, 1 failed**. The one
failure was the documented Windows baseline
`test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`, at
`tests/test_follow_up_delivery.py:1018`; it is consistent with the
implementation report's pre-existing ownership-probe explanation.
`docs/features/false-receipt-reconcile/implementation.md:94-113`

The updated regression assertions are meaningful: the old response-only
assertions still pass with the original bug, but the added provenance and both
keyed status lookups fail until the caller row is actually settled.
`tests/test_follow_up_delivery.py:398-429`

The new tests use the real transcript scanner, delivery store and public tool
entry points, with the fake backend only at the resume boundary. The focused
test execution confirms they pass in the current worktree; the full-suite
counts supplied for this Windows machine retain the documented baseline and
timing failure rather than being treated as implementation failures.
`tests/test_follow_up_delivery.py:1-8`, `tests/test_follow_up_delivery.py:125-187`

## C. Remaining notes

**NIT — `_RECONCILE_SAME_KEY` is defensive rather than normally reachable.**
The implementation and tests document that fact, and there is no correctness
issue from retaining the branch. `src/claude_teams/server_simple.py:2215-2221`, `tests/test_follow_up_delivery.py:716-730`

No dead added helper, inconsistent public docstring, or unintended in-scope
behavior change was found. The `reconciled_from_key` store field and the
spawned-child `send_message` documentation match the implemented contract.
`src/claude_teams/delivery_store.py:67-70`, `src/claude_teams/delivery_store.py:232-244`, `src/claude_teams/server_simple.py:3362-3370`, `src/claude_teams/server_simple.py:5375-5394`

## Final verdict

**APPROVED.** The round-1 blockers are closed, the new barrier and option-only
tests are substantive, the focused changed tests pass apart from the documented
baseline failure, and the remaining dispositioned items are acceptable within
the chosen narrow scope.
