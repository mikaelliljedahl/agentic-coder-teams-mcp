# Independent plan review: false receipt reconciliation, revision 3

Verdict: **REJECTED**

Revision 3 is a substantial improvement: it removes the legacy identity split, retains an unresolved marker instead of clearing it blindly, specifies a store transaction before registry cleanup, and adds the missing public-path tests. It is still not safe to implement as written. The nonce is not immutable on every same-key retry path, case D can permanently block ordinary delivery when the store cannot be repaired through the named tools, and case C can double-deliver after a late receipt contradicts a terminal `failed` row.

## Disposition of review 2 findings

| Review 2 finding | Status | Revision 3 disposition |
| --- | --- | --- |
| Row 4's queued/pending shape was legal only if it was a non-C2 refusal | **RESOLVED** | Revision 3 replaces row 4 with case D, retains the marker, marks the caller row's reason, and specifies the full retriable response. `docs/features/false-receipt-reconcile/plan.md:132-159` |
| BLOCKER: clearing an unidentified marker removed the duplicate guard | **PARTIALLY RESOLVED** | The new store lookup avoids the old clear-and-send path, and case D retains the marker. However, zero/multiple-match case D is not actually recoverable through `delivery_status(to)` or `deliver_pending`; that new availability blocker is below. `docs/features/false-receipt-reconcile/plan.md:75-93`, `docs/features/false-receipt-reconcile/plan.md:132-159` |
| Lock-order review | **RESOLVED** | Section 4 now explicitly specifies registry -> delivery and closing the delivery transaction before the later registry transaction. `docs/features/false-receipt-reconcile/plan.md:161-179` |
| `public_view` conditional provenance contract | **RESOLVED** | Revision 3 names the docstring change and test 15. `docs/features/false-receipt-reconcile/plan.md:214-224`, `docs/features/false-receipt-reconcile/plan.md:300-313` |
| MAJOR: test 11 and registry-save `OSError` behavior | **PARTIALLY RESOLVED** | The plan now splits alias and send-through tests and requires an `OSError` boundary, but the exact state/response after that boundary and the claimed cleanup behavior still fail for terminal-row early returns. `docs/features/false-receipt-reconcile/plan.md:181-201`, `docs/features/false-receipt-reconcile/plan.md:300-313` |
| MAJOR: marker identity validation | **PARTIALLY RESOLVED** | Sender/target/nonce matching is now specified, but type validity, durable-row field consistency, missing-row recovery, and terminal-state outcomes remain incomplete. `docs/features/false-receipt-reconcile/plan.md:89-99`, `docs/features/false-receipt-reconcile/plan.md:335-349` |
| MAJOR: terminal-`failed` asymmetry | **PARTIALLY RESOLVED** | Case C states the no-overwrite rule, but its same-fingerprint immediate retry still sends after the marker is cleared and can duplicate a prompt whose nonce was found. `docs/features/false-receipt-reconcile/plan.md:107-112`, `docs/features/false-receipt-reconcile/plan.md:281-284` |
| Branch-specific `send_message` wording | **RESOLVED** | Revision 3 scopes the documentation to the spawned-child branch and keeps inbox/external branches out of scope. `docs/features/false-receipt-reconcile/plan.md:9-22`, `docs/features/false-receipt-reconcile/plan.md:214-221` |
| Concrete `deliver_pending`/concurrency state machines | **RESOLVED in the test list** | Tests 9 and 11 now name the rows, keys and expected resumes; the D barrier still needs a recovery assertion beyond a restored row. `docs/features/false-receipt-reconcile/plan.md:287-298` |
| `public_view` shape tests | **RESOLVED** | Test 15 covers ordinary and alias shapes. `docs/features/false-receipt-reconcile/plan.md:311-313` |
| Same-key cleanup failure rule | **PARTIALLY RESOLVED** | Revision 3 gives a registry-save failure rule, but the “benign residue” claim is too broad and the same-key/terminal early-return interaction remains. `docs/features/false-receipt-reconcile/plan.md:196-211` |
| “Only `deliver_pending` could pick up the current row” precision issue | **RESOLVED** | Revision 3's current-behaviour text names both drain and same-key retry. `docs/features/false-receipt-reconcile/plan.md:50-63` |
| Withdrawn `_FollowUpPlan` identity plumbing | **RESOLVED** | The store lookup makes the marker identity fields unnecessary. `docs/features/false-receipt-reconcile/plan.md:75-93`, `docs/features/false-receipt-reconcile/plan.md:226-235` |

## 1. Is the nonce a sound lookup key?

**BLOCKER — it is a sound probabilistic handle only while the row's nonce is immutable. The current retry state machine does not guarantee that.** Every normal follow-up/delivery attempt allocates a fresh nonce in `_prepare`, and the nonce generator uses 16 cryptographically random bytes. That gives practical uniqueness, but not a store-enforced uniqueness constraint. `src/claude_teams/server_simple.py:4160-4164`, `src/claude_teams/delivery.py:53-54`, `src/claude_teams/delivery.py:101-108`

The important ordering claim is correct: `_mark_attempt_sent` writes the nonce, increments attempts, and marks the row `sent` inside `delivery_transaction` before the backend resume is invoked. The shared delivery path reaches that call before `plan.backend.resume`. `src/claude_teams/server_simple.py:4282-4303`, `src/claude_teams/server_simple.py:4393-4422`

The plan's stronger implication — that a durable row keeps a unique nonce for each attempt — is false on an existing retriable-failure path:

1. `_mark_attempt_sent` writes nonce N1 and attempts 1.
2. `_finalize_follow_up` can return a retriable `operation_superseded` result, or `resume_failed` when no PID is returned. `src/claude_teams/server_simple.py:3813-3847`
3. `_record_outcome` maps a retriable result back to `phase="pending"` without clearing the existing nonce or attempts. `mark_phase` changes only status, phase and reason. `src/claude_teams/server_simple.py:4457-4472`, `src/claude_teams/delivery_store.py:218-225`
4. A same-key retry sees `phase="pending"`, so `_reconcile_before_resend` returns `None` instead of reconciling N1. `_prepare` then creates N2, and `_mark_attempt_sent` unconditionally overwrites the row's nonce with N2. `src/claude_teams/server_simple.py:4847-4862`, `src/claude_teams/server_simple.py:4406-4414`

The normal `sent`/`unconfirmed` path is protected because `_reconcile_before_resend` refuses to send again until that row settles. That protection does not cover a row returned to `pending` after a retriable failure. `src/claude_teams/server_simple.py:4876-4885`, `src/claude_teams/server_simple.py:4919-4931`

This matters to the new design. If a marker still names N1 but its row was overwritten with N2, the nonce lookup reports zero matches and falls into case D. The old receipt can no longer be attributed through that row, and the retained marker can wedge the agent. Even without a marker, a prompt that landed during an `operation_superseded`/`resume_failed` window loses its only durable nonce and can be sent again under the same key. Revision 3 does not mention this transition or test it. `docs/features/false-receipt-reconcile/plan.md:75-99`, `docs/features/false-receipt-reconcile/plan.md:132-159`

The plan must either preserve every prior nonce in an attempt history, or forbid replacing a non-empty nonce until that attempt is conclusively reconciled/terminal. It needs a regression test for a retriable failure followed by a same-key retry. Case B is safe only because a terminal row is returned before another attempt; case D is not safe if its marker's original row has already lost the marker nonce. `src/claude_teams/server_simple.py:5072-5075`, `src/claude_teams/delivery_store.py:194-215`

## 2. Case D: zero or multiple matching rows

**BLOCKER — retaining the marker is fail-closed, but it is a real delivery dead end until manual store repair or agent destruction.** The queued/pending caller row is legal and should remain: it is an unsent request, not evidence of delivery. Case D correctly avoids both `delivered` and `failed`. `src/claude_teams/delivery_store.py:162-191`, `src/claude_teams/delivery_store.py:218-225`, `docs/features/false-receipt-reconcile/plan.md:132-155`

The named recovery paths do not themselves clear a D marker:

- `delivery_status(to=name)` enumerates existing rows and reconciles them, but it never reads or mutates `pending_delivery`. With zero matching rows it has nothing to reconcile, so the marker remains. `src/claude_teams/server_simple.py:5221-5247`
- `deliver_pending` enumerates rows, reconciles them, and only sends copied rows that remain `phase="pending"`. It then calls the same `_prepare`; a still-found D marker returns the same refusal again. It does not clear the marker. `src/claude_teams/server_simple.py:5309-5353`
- `kill_agent` can remove the marker only as a consequence of deleting the agent record, after lease checks; it does not recover a missing old row or deliver the caller's pending request. A live lease blocks even that escape. `src/claude_teams/server_simple.py:5514-5569`, `src/claude_teams/server_simple.py:5577-5589`

A restored single row can make the next call resolve, but only if it is restored with the exact current sender, target, nonce, and a transcript state that reaches the positive pending-marker scan. A duplicate pair remains a duplicate pair: every status/drain pass may settle rows, but the lookup still sees more than one match and case D retains the marker. The plan therefore must describe manual repair as a required recovery operation, not call this “not a dead end,” and must test that `deliver_pending` cannot bypass the barrier. `docs/features/false-receipt-reconcile/plan.md:134-159`, `docs/features/false-receipt-reconcile/plan.md:275-280`

The response detail names `delivery_status(to=...)` and `kill_agent` but omits `deliver_pending`, even though it is in scope. More importantly, the user-facing contract should say that zero/multiple matches block all automatic sends until the durable store is repaired or the agent is killed. `docs/features/false-receipt-reconcile/plan.md:147-159`, `docs/features/false-receipt-reconcile/plan.md:203-213`

## 3. Case C: terminal failed row whose nonce is now found

**BLOCKER — the same-fingerprint C arm can double-deliver.** The plan clears the marker, leaves the new caller row pending, returns `prior_attempt_settled_failed`, and says the immediate retry sends normally. `docs/features/false-receipt-reconcile/plan.md:107-112`, `docs/features/false-receipt-reconcile/plan.md:281-284`

That is not safe when the scan has just found the old nonce. The current caller row is still `pending`, so `_reconcile_before_resend` returns `None`; with the marker cleared, `_prepare` takes the normal send path and allocates a new nonce. `src/claude_teams/server_simple.py:4847-4862`, `src/claude_teams/server_simple.py:5074-5088`, `src/claude_teams/server_simple.py:4160-4164`

`kill_agent` can create the old `failed` row by treating a complete `SCAN_ABSENT` result as definite non-delivery before removing the agent. If a buffered transcript write appears later, the nonce is found but `settle()` deliberately refuses to overwrite the terminal failure. `src/claude_teams/server_simple.py:2362-2381`, `src/claude_teams/delivery_store.py:199-215`

Normally kill also removes the agent, so an active follow-up cannot reach case C afterward. The case is nevertheless reachable after a crash or independent registry-save failure between delivery-row settlement and agent removal, and the test must model that state rather than merely call kill and then follow up on a deleted agent. `src/claude_teams/server_simple.py:5495-5506`, `src/claude_teams/server_simple.py:5571-5579`, `docs/features/false-receipt-reconcile/plan.md:281-284`

For the same fingerprint, retain a barrier/refusal and require explicit operator resolution, or introduce a separately documented conflict status. Do not clear the marker and automatically resend the same request. The different-fingerprint C arm is safe against duplicating the old prompt because the old nonce was found; it can fall through only for the genuinely different request. `docs/features/false-receipt-reconcile/plan.md:107-120`

## 4. The step-3 `OSError` boundary

**The catch is implementable where the plan puts it, but its state and response contract remain incomplete.** `_agents_transaction` loads under the lock, yields, and then only releases the file lock; it does not perform an implicit save in `__exit__`. `_save_agents_transaction` is the explicit plain write that can raise. `src/claude_teams/server_simple.py:567-608`

Thus a `try/except OSError` around step 3 inside `_prepare` can catch the failure while the registry lock is held. Returning from that handler lets the context manager release the lock; it does not persist the in-memory marker removal. The delivery transaction has already committed, so the on-disk state is exactly: terminal delivery row(s), old marker still present, and no automatic rollback. `src/claude_teams/server_simple.py:3959-3960`, `src/claude_teams/delivery_store.py:368-377`, `docs/features/false-receipt-reconcile/plan.md:163-195`

For the alias arm, returning `delivered` after this failure is defensible because the caller's row is already terminal and the marker is only stale residue. For send-through arms, returning the existing `_store_unavailable` shape is not self-explanatory: that helper says the durable delivery row could not be written, although step 2 did write the old row and the current row may already exist. The plan must define a registry-store-unavailable response or explicitly document why reusing that delivery-store reason is truthful. `src/claude_teams/server_simple.py:4759-4775`, `src/claude_teams/server_simple.py:5090-5101`, `docs/features/false-receipt-reconcile/plan.md:187-195`

The same boundary must be specified for the post-`_reconcile_before_resend` cleanup transaction, not only `_prepare`; that path is outside the registry transaction and has its own store-then-registry sequence. `docs/features/false-receipt-reconcile/plan.md:203-211`, `src/claude_teams/server_simple.py:4849-4862`

## 5. The “benign stale marker” argument

**MAJOR — case B itself is correct, but the claim that only the same-key terminal early return can leave a marker is false.** When case B is actually reached, `R` is terminal `delivered`, so `settle()` does nothing; the subsequent fingerprint arm can clear the marker without re-settling `R`. `src/claude_teams/delivery_store.py:199-215`, `docs/features/false-receipt-reconcile/plan.md:107-120`

However, many paths return before `_prepare` can reach case B: preflight refusal, external-target refusal, idempotency conflict, a terminal current row, an unavailable backend/binding, or a queued call that never obtains a plan. `_guaranteed_send` also returns immediately for any terminal current row, regardless of whether it is the same or a different key. `src/claude_teams/server_simple.py:5051-5075`, `src/claude_teams/server_simple.py:4894-4931`

`delivery_status` and `deliver_pending` do not clear an agent marker by themselves, and `_reconcile_pending_delivery` only enters the new resolution branch after the old nonce scan returns `SCAN_FOUND`. Therefore “any subsequent call under any other key clears it” is true only for a valid nonterminal call that passes all guards, finds the old nonce, and reaches case B. `src/claude_teams/server_simple.py:2188-2200`, `src/claude_teams/server_simple.py:5221-5247`, `src/claude_teams/server_simple.py:5310-5353`, `docs/features/false-receipt-reconcile/plan.md:196-201`

The plan should narrow the claim and define whether stale markers are merely tolerated or must be cleaned by a dedicated terminal-row cleanup path. Test 13 must include the alias retry under the same key, an already-terminal different key, and a path that returns before `_prepare`; otherwise it cannot establish the stated cleanup invariant.

## 6. `mark_phase(C, PHASE_PENDING, reason=...)`

**No correctness finding: this is legal.** `mark_phase` explicitly sets a nonterminal row to `status="queued"`, the requested phase, and the supplied reason, and `public_view` exposes `reason`. `PHASE_PENDING` is the store's defined nonterminal phase. `src/claude_teams/delivery_store.py:50-62`, `src/claude_teams/delivery_store.py:218-245`

No exact public-view key-set assertion in the two requested test files is affected; the exact dictionaries in `tests/test_delivery_integrity.py:495-509` test raw store writes, and the delivery-status test checks selected values. `tests/test_delivery_integrity.py:658-692`

The implementation must mutate the row loaded inside the active delivery transaction and persist it (`txn.put` or `txn.touch` as appropriate), then build the response through `_with_delivery_identity`. Otherwise the plan's “same reason” statement can still diverge between the returned refusal and keyed `delivery_status`. `src/claude_teams/server_simple.py:4475-4494`, `src/claude_teams/server_simple.py:4524-4553`, `docs/features/false-receipt-reconcile/plan.md:143-152`

## 7. Remaining underspecification

**BLOCKER — the plan does not define what happens when the marker nonce is not found in the transcript.** `_reconcile_pending_delivery` returns only a Boolean and returns `False` for every scan outcome other than `SCAN_FOUND`; current `_prepare` then falls through to the normal send path. The four store cases in revision 3 are therefore not reached for `SCAN_ABSENT`, `SCAN_INDETERMINATE`, or `SCAN_AMBIGUOUS`. `src/claude_teams/server_simple.py:2188-2200`, `src/claude_teams/server_simple.py:4083-4109`, `docs/features/false-receipt-reconcile/plan.md:101-130`

If the old prompt's transcript write is merely delayed, sending a new same-fingerprint request before a positive scan can duplicate it. The plan must specify whether all non-`FOUND` outcomes retain the marker and refuse, or whether a particular outcome is allowed to fall through, and must test that choice. The current `delivery.py` rules explicitly distinguish absent, indeterminate and ambiguous scans rather than collapsing them. `src/claude_teams/delivery.py:25-34`, `src/claude_teams/server_simple.py:2203-2228`

**MAJOR — the resolver's row-consistency predicate needs to be complete.** Sender, target and nonce agreement is necessary, but the plan should require a typed nonempty nonce, a row fingerprint, a valid row status/phase, and a matching `idempotency_key`/target before treating `R` as authoritative. It must state what happens if the row is missing fields, has a different target, has a different nonce representation, or has no fingerprint. `docs/features/false-receipt-reconcile/plan.md:84-99`, `src/claude_teams/delivery_store.py:123-159`

**MAJOR — case C's test setup and public contract are still ambiguous.** Because ordinary `kill_agent` removes the agent after reconciling rows, a test that calls kill normally cannot then exercise `_prepare` on that agent. The plan must say whether it injects a failed registry save, leaves a crash fixture, or directly seeds the post-kill state, and must specify the caller-visible result for the same-fingerprint conflict. `src/claude_teams/server_simple.py:5495-5506`, `src/claude_teams/server_simple.py:5571-5579`, `docs/features/false-receipt-reconcile/plan.md:281-284`

**MAJOR — nonce collision handling is only described as “more than one row.”** `new_delivery_nonce()` is random rather than collision-checked, so the plan should say whether a rare collision is treated exactly like corruption and whether all matching rows remain untouched. Test 6 names the outcome but not whether any matching rows are settled before case D is returned. `src/claude_teams/delivery.py:101-108`, `docs/features/false-receipt-reconcile/plan.md:275-280`

**MINOR — test 13's “next call under a different key” is not enough to prove stale-marker cleanup.** It bypasses the terminal-row early return by construction and does not cover the same-key terminal path that the plan calls the one exception. The test should assert both the intended residue and the exact path that removes it. `docs/features/false-receipt-reconcile/plan.md:196-201`, `docs/features/false-receipt-reconcile/plan.md:304-308`, `src/claude_teams/server_simple.py:5072-5075`

**NIT — “unique” should be qualified as practical cryptographic uniqueness.** There is no deterministic uniqueness check in the generator or store. Case D's duplicate-row branch is the fail-closed backstop. `src/claude_teams/delivery.py:53-54`, `src/claude_teams/delivery.py:101-108`, `docs/features/false-receipt-reconcile/plan.md:75-93`

## Required changes before approval

1. Make durable nonce history immutable across retriable same-key failures, or explicitly reconcile/preserve the prior nonce before allowing a replacement attempt.
2. Treat case D as a manual-recovery barrier unless the plan adds a real store-repair/marker-clear path; do not claim `delivery_status` or `deliver_pending` can resolve a missing/duplicate row.
3. Do not clear case C for a same-fingerprint retry that can immediately resend after a positive nonce scan; retain a barrier or define an explicit conflict resolution.
4. Define the registry-save failure response/state for both `_prepare` and `_reconcile_before_resend`, including terminal-row early returns.
5. Specify behavior for every non-`FOUND` transcript scan and complete the resolver's typed row-consistency predicate.

## Final verdict

**REJECTED.** Revision 3 closes most revision-2 findings, but the central nonce-authority assumption is false on retriable pending rows, case D can wedge the agent without a real repair path, and case C's immediate retry can duplicate a prompt whose nonce was found. These are implementation-blocking correctness issues.
