# Implementation review: false receipt reconciliation

Verdict: **REJECTED**

This review evaluates the implementation diff on branch
`claude/false-receipt-delivery-status-ad3d77`, not the plan in isolation. The
narrow deferrals recorded in revision 4 are accepted as scope decisions and are
not re-raised here: markerless retriable-row nonce overwrite, non-`FOUND` scan
outcomes, and damaged-store repair. The first finding below is a new regression
in the changed same-key cleanup path, not a re-opening of those deferrals.

## A. Correctness

### BLOCKER — same-key cleanup clears a failed row's marker and reopens a duplicate window

The new `_clear_reconciled_marker` is called for every terminal result from
`_reconcile_before_resend`, including `failed`, not only for a positively found
receipt. `_reconcile_delivery_record` settles an absent nonce as
`STATUS_FAILED` after the child is dead and the grace period has passed; the
caller then clears the agent marker unconditionally. `src/claude_teams/server_simple.py:2461-2509`, `src/claude_teams/server_simple.py:5079-5088`

That changes behavior in a concrete sequence:

1. An attempt is `unconfirmed` with nonce N and the agent marker still names N.
2. The child dies and a same-key retry gets a complete negative scan, so the
   row becomes terminal `failed`.
3. The new cleanup removes the marker.
4. The delayed transcript write for N arrives after the negative scan. A later
   new-key call has no marker to reconcile, so it follows the normal send path
   and can deliver the same prompt again. `src/claude_teams/server_simple.py:2488-2509`, `src/claude_teams/server_simple.py:5066-5088`, `src/claude_teams/server_simple.py:4260-4340`

Before this diff, `_reconcile_before_resend` did not clear the marker. With the
marker retained, the later positive scan would reach the new terminal-failed
barrier instead of silently sending the same request. The intended case-C
implementation also retains the marker, but the same-key path bypasses that
case entirely. `src/claude_teams/server_simple.py:2205-2213`, `src/claude_teams/server_simple.py:2271-2328`, `src/claude_teams/server_simple.py:5079-5088`

The cleanup must distinguish a terminal `delivered` result from terminal
`failed`, or otherwise preserve an explicit conflict barrier for a failed row
whose nonce may still appear later. Add a regression covering negative scan ->
same-key failure -> late receipt -> new-key retry, and assert no second resume.

### Correctness confirmed — identified reconciliation and response bookkeeping

The new resolver filters the current sender and target and requires exactly one
row whose nonce matches. It returns no row for zero or multiple matches.
`src/claude_teams/server_simple.py:2224-2242`

The settlement helper performs the old-row settlement and alias-row settlement
in one delivery transaction. It updates the caller's `record` after each
barrier/alias/same-key mutation, and the send-through verdict leaves the caller
row pending for the existing `_mark_attempt_sent` path. `src/claude_teams/server_simple.py:2244-2297`

The answer helper calls that transaction before clearing the marker. For a
barrier it never pops the marker; for a successful reconciliation it pops the
marker, bumps the agent generation, saves the registry, and only then returns a
public answer. For the different-request verdict it returns `None`, so
`_prepare` continues to the normal reservation path. `src/claude_teams/server_simple.py:2300-2353`, `src/claude_teams/server_simple.py:4264-4283`, `src/claude_teams/server_simple.py:4336-4374`

The returned alias answer is built through `_with_public_status` after
`record.update(caller)`, so its `status`, `phase`, identity fields, and
`reconciled_from_key` describe the row that was persisted. `src/claude_teams/server_simple.py:2288-2297`, `src/claude_teams/server_simple.py:2337-2353`, `src/claude_teams/server_simple.py:4560-4579`

### Correctness confirmed — barrier retention and lock use

The barrier branch marks the caller row pending, persists its reason in the
delivery transaction, and does not mutate the agent marker. If the existing
`changed` flag is true, it saves the unchanged marker; if false, the enclosing
registry transaction exits without a save. Either way the marker remains on
disk. `src/claude_teams/server_simple.py:2263-2269`, `src/claude_teams/server_simple.py:2323-2328`, `src/claude_teams/server_simple.py:602-610`

`_answer_reconciled_attempt` does not call `_clear_reconciled_marker`. It is
called from `_prepare` while `_agents_transaction` is held, but it directly
mutates that already-held `agent`, bumps its generation, and saves it. The
later `generation = _record_generation(agent)` and `reserve_lease` therefore
read the bumped generation before a send-through attempt is reserved.
`src/claude_teams/server_simple.py:4140-4141`, `src/claude_teams/server_simple.py:4272-4283`, `src/claude_teams/server_simple.py:4336-4359`

`_clear_reconciled_marker` has one call site, in `_reconcile_before_resend`,
after its `delivery_transaction` has closed. It opens its own
`_agents_transaction`, so it is not re-entered while the registry lock is
already held. `src/claude_teams/server_simple.py:2356-2378`, `src/claude_teams/server_simple.py:5057-5088`

The lock order is consequently safe for the new `_prepare` path: registry
then delivery. The existing kill path has the same nested order, while
`_record_outcome`, `_reconcile_before_resend`, `delivery_status`, and
`deliver_pending` load agents before opening the delivery transaction and do
not hold the registry lock across it. `src/claude_teams/server_simple.py:602-610`, `src/claude_teams/server_simple.py:2536-2565`, `src/claude_teams/server_simple.py:4642-4670`, `src/claude_teams/server_simple.py:5066-5078`, `src/claude_teams/server_simple.py:5457-5484`, `src/claude_teams/server_simple.py:5531-5550`

### MINOR — defensive same-key verdict is not exercised by the public healthy path

The normal same-key `sent`/`unconfirmed` retry is intercepted by
`_reconcile_before_resend` before `_prepare`, and the new test explicitly says
it exercises that path. `_RECONCILE_SAME_KEY` remains useful for an inconsistent
state, but no public test directly reaches `_answer_reconciled_attempt` with
that verdict. This is acceptable defensive code, but its intended corrupted/
partial-state use should be documented or covered by a focused helper test.
`src/claude_teams/server_simple.py:5066-5088`, `src/claude_teams/server_simple.py:2244-2282`, `tests/test_follow_up_delivery.py:609-623`

## B. Test quality

### MAJOR — the “lost settlement write” test fails before settlement

`test_a_lost_settlement_write_sends_nothing_and_keeps_the_marker` patches
`delivery_store.save_records` to fail before invoking the second
`follow_up_agent`. The second call first enters `_open_delivery_record` and
tries to persist the new key; that is where the patched write fails. It never
reaches `_reconcile_pending_delivery`, `_answer_reconciled_attempt`, or the
settlement transaction under test. `tests/test_follow_up_delivery.py:682-703`, `src/claude_teams/server_simple.py:5092-5170`, `src/claude_teams/server_simple.py:4264-4283`

This test would still pass with the original reconciliation bug, because the
new caller row creation would fail before the old hard-coded shortcut ran. The
test must first create/persist the caller row, then enable a one-shot failure
for the subsequent settlement write, and assert that the actual settlement
helper was reached (or use a fault hook at that exact transaction boundary).

### MINOR — several assertions in the updated regression still pass with the old bug

The pre-existing portion of
`test_a_later_flush_reconciles_and_does_not_resend` — `status="delivered"`,
`reconciled=True`, one backend resume, and marker absence — would all pass with
the old shortcut. The new provenance and both keyed `delivery_status` checks
are the assertions that catch the original contradiction. `tests/test_follow_up_delivery.py:417-429`

The new alias test adds the stronger checks, including an empty nonce on the
caller row, so the overall alias coverage is meaningful. `tests/test_follow_up_delivery.py:462-487`

### MINOR — option-only test proves refusal after fall-through, not a real send

`test_an_option_only_difference_is_a_different_request` flips
`replace_if_idle=False`, then expects `agent_idle_but_alive`. It proves the
implementation did not alias the row, but it does not prove that the
different-fingerprint arm can send a new prompt. The test reaches the normal
path and is refused by the option itself. `tests/test_follow_up_delivery.py:516-535`, `src/claude_teams/server_simple.py:4285-4334`

Keep this refusal test, but add a second option-only case with an actually
resumable/busy-state fixture so the new attempt, nonce, and persisted row are
verified.

### MINOR — shared entry points are real, but not all declared arms are tested

The `deliver_pending` test calls the real public `deliver_pending` tool and
exercises the alias arm. The `send_message` test calls the real public
`send_message` tool and exercises the spawned-child alias route. `src/claude_teams/server_simple.py:3345-3420`, `src/claude_teams/server_simple.py:5512-5620`, `tests/test_follow_up_delivery.py:626-679`

Neither test exercises its entry point's different-fingerprint send-through
arm. The shared implementation makes the alias tests valuable, but a routing
or argument mistake could still leave those public variants unproven. Add one
different-prompt `deliver_pending` case and one different-prompt child
`send_message` case, or explicitly narrow the acceptance claim to the shared
alias path.

### MINOR — duplicate-match barrier is implemented but not tested

The implementation treats zero and multiple nonce matches identically in
`_resolve_pending_row`, but the changed tests remove the old row and cover only
the zero-match case. There is no test for two rows carrying the same nonce,
marker retention, caller-row reason, and no resume. `src/claude_teams/server_simple.py:2224-2242`, `tests/test_follow_up_delivery.py:538-570`

Add the duplicate-row case because it is the only protection against a rare
nonce collision or store corruption, and it is a separate branch in the
resolver's contract.

### MINOR — the failed-row test seeds the state directly

`test_a_prior_attempt_settled_failed_is_not_resurrected` directly calls
`delivery_store.settle` rather than driving `kill_agent`'s reconciliation. The
direct fixture is sufficient to test the barrier decision, and `kill_agent` is
outside the chosen implementation scope, but the test docstring's claim about
kill-time cleanup is not itself exercised. `tests/test_follow_up_delivery.py:573-606`, `src/claude_teams/server_simple.py:2536-2565`, `src/claude_teams/server_simple.py:5750-5847`

### Validation evidence

I ran:

```
rtk uv run pytest tests/test_follow_up_delivery.py tests/test_delivery_store.py -q
```

Observed result: **73 passed, 1 failed**. The failure was the documented
baseline `test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`, at
`tests/test_follow_up_delivery.py:898`; it matches the implementation report's
Windows ownership-probe explanation. `docs/features/false-receipt-reconcile/implementation.md:94-113`

The focused run did not expose the false-green settlement test because that
test's assertions pass; the source inspection above shows why it does not reach
the intended failure point. No full-suite result was independently established
in this review.

## C. Dead code, reachability, and unintended behavior

No newly added helper is dead: `_resolve_pending_row` is called by
`_settle_reconciled_attempt`, that helper is called by `_answer_reconciled_attempt`,
and the answer helper is called from `_prepare`. `_reconcile_barrier` is used by
the barrier verdict, and `RECONCILED_FROM_FIELD` is used by both settlement and
`public_view`. `src/claude_teams/server_simple.py:2224-2353`, `src/claude_teams/server_simple.py:4264-4283`, `src/claude_teams/delivery_store.py:67-70`, `src/claude_teams/delivery_store.py:257-260`

The delivery-store public contract change is internally consistent: ordinary
rows retain the exact existing key set, while aliased rows expose the optional
provenance. The exact ordinary-row assertion remains valid and the new unit
test covers the conditional field. `src/claude_teams/delivery_store.py:232-260`, `tests/test_delivery_store.py:229-260`

The only material unintended behavior found in the diff is the failed-row
marker removal described above. It conflicts with the implementation's own
case-C barrier, which says a settled failed row keeps the marker and sends
nothing. `src/claude_teams/server_simple.py:2205-2213`, `src/claude_teams/server_simple.py:2271-2276`, `src/claude_teams/server_simple.py:5079-5088`

## Required changes before approval

1. Do not clear `pending_delivery` when `_reconcile_before_resend` settles the
   old row as `failed`, or add an equivalent durable conflict barrier that
   prevents a late receipt/new-key retry from sending the same request.
2. Repair the settlement-write test so the failure is injected after the
   caller row exists and inside `_settle_reconciled_attempt`'s transaction.
3. Add the failed-row late-receipt regression and the duplicate-nonce barrier
   test.
4. Add real send-through coverage for option-only differences and at least one
   different-request case through each shared public entry point.

The explicitly deferred markerless nonce overwrite, non-`FOUND` scan behavior,
and damaged-store repair are not included in this verdict; they remain the
chosen narrow-scope follow-ups.

## Final verdict

**REJECTED.** The main reconciliation path is well-bookkept and lock-safe, but
the changed same-key cleanup can clear a failed attempt's only duplicate guard,
and the settlement-failure test does not test settlement at all. Both must be
fixed before the implementation can be approved.
