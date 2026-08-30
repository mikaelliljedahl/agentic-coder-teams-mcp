# Independent plan review: false receipt reconciliation, revision 2

Verdict: **REJECTED**

Revision 2 fixes most of review 1's explicitly identified design errors. It is still not ready to implement because the unidentified-marker recovery path can resend an already observed prompt, and the crash/save-failure claim in test 11 is not true for all four-way outcomes. A few identity and response-shape details also remain underspecified.

## Disposition of review 1 findings

| Review 1 finding | Status | Revision 2 disposition |
| --- | --- | --- |
| Core root-cause analysis | **RESOLVED** | The rewritten current-behaviour section still correctly identifies the agent-scoped marker, the nonce-only scan, the orphaned caller row, and the absence of a C2 reason. `docs/features/false-receipt-reconcile/plan.md:24-67` |
| Precision correction: the old row/current row were described too absolutely | **PARTIALLY RESOLVED** | The “only `deliver_pending`” statement about the current pending row remains inaccurate: a later same-key `follow_up_agent`/child `send_message` can also reach the normal path once the old marker has been cleared. `docs/features/false-receipt-reconcile/plan.md:50-63`, `src/claude_teams/server_simple.py:5074-5088`, `src/claude_teams/server_simple.py:5276-5353` |
| Stale `_finish_follow_up` / `fingerprint` names | **RESOLVED** | Revision 2 names `_finalize_follow_up`, `_FollowUpPlan`, and `request_fingerprint`. `docs/features/false-receipt-reconcile/plan.md:71-92` |
| Same-key retry is intercepted by `_reconcile_before_resend` | **RESOLVED for the normal path** | Section 4 now explicitly requires marker cleanup there and test 5 exercises it. A separate save-failure limitation remains below. `docs/features/false-receipt-reconcile/plan.md:154-161`, `docs/features/false-receipt-reconcile/plan.md:209-211` |
| New-key alias needed durable/public provenance | **RESOLVED** | Revision 2 makes `reconciled_from_key` durable/public and keeps the new row's nonce empty. `docs/features/false-receipt-reconcile/plan.md:107-116`, `docs/features/false-receipt-reconcile/plan.md:171-179` |
| Identity could not reach the write site | **RESOLVED** | It now explicitly adds sender, key, and fingerprint to `_FollowUpPlan`. `docs/features/false-receipt-reconcile/plan.md:71-86` |
| Different fingerprint includes option-only changes | **RESOLVED** | The option boundary is deliberate and has a dedicated test. `docs/features/false-receipt-reconcile/plan.md:88-92`, `docs/features/false-receipt-reconcile/plan.md:181-185`, `docs/features/false-receipt-reconcile/plan.md:205-208` |
| Legacy shortcut itself was a false receipt | **RESOLVED as to the response status** | Revision 2 replaces it with a queued retriable refusal rather than `delivered`. The marker-clearing consequence is a new blocker below. `docs/features/false-receipt-reconcile/plan.md:94-126`, `docs/features/false-receipt-reconcile/plan.md:188-189` |
| Cross-store commit order was unspecified | **RESOLVED as stated** | Section 3 now requires one delivery transaction to commit before clearing/saving the registry marker. The registry-save failure after that commit is a separate unresolved issue below. `docs/features/false-receipt-reconcile/plan.md:128-152` |
| `DeliveryStoreError` was swallowed | **RESOLVED** | Revision 2 propagates it to the existing fail-closed handler and explicitly retains the marker. `docs/features/false-receipt-reconcile/plan.md:144-147`, `src/claude_teams/server_simple.py:5077-5101` |
| `deliver_pending` shared the shortcut | **RESOLVED in scope** | It is now in scope and test 7 is named. The test still needs the concrete setup/assertions described in the test findings below. `docs/features/false-receipt-reconcile/plan.md:15-17`, `docs/features/false-receipt-reconcile/plan.md:162-163`, `docs/features/false-receipt-reconcile/plan.md:216-217` |
| `_reconcile_before_resend` interaction was missing | **RESOLVED in scope** | Section 4 states that a matching marker is cleared after the store commit and test 5 is assigned to it. `docs/features/false-receipt-reconcile/plan.md:156-161`, `docs/features/false-receipt-reconcile/plan.md:209-211` |
| `send_message` child route and documentation were missing | **RESOLVED in scope** | Both entry points are named and test 8 is added. The wording still needs to be scoped to the spawned-child branch because the other `send_message` branches have no durable delivery row. `docs/features/false-receipt-reconcile/plan.md:15-17`, `docs/features/false-receipt-reconcile/plan.md:164-169`, `src/claude_teams/server_simple.py:3175-3189` |
| `kill_agent` reconciliation must remain separate | **RESOLVED** | It is explicitly out of scope and the plan preserves the durable-row/target-snapshot path. `docs/features/false-receipt-reconcile/plan.md:19-22`, `docs/features/false-receipt-reconcile/plan.md:181-192` |
| Missing marker-identity test | **RESOLVED in test list** | Test 1 covers the three fields and test 6 covers missing fields, though type/row-consistency validation remains underspecified. `docs/features/false-receipt-reconcile/plan.md:194-215` |
| Missing `deliver_pending`, `send_message`, concurrency, and both-key view tests | **RESOLVED in test list** | Tests 2, 7, 8, and 9 now name those surfaces. `docs/features/false-receipt-reconcile/plan.md:201-221` |
| Missing settlement fault injection/crash test | **PARTIALLY RESOLVED** | Tests 10 and 11 are added, but test 11's expected state is not valid for both alias and different-fingerprint arms, and an `agents.json` save failure is not handled by the current exception boundary. `docs/features/false-receipt-reconcile/plan.md:223-229`, `src/claude_teams/server_simple.py:600-608`, `src/claude_teams/server_simple.py:5090-5101` |
| Existing regression test needed updating | **RESOLVED in plan** | The exact new-key regression, same-key test, and marker-content test are named for changes. `docs/features/false-receipt-reconcile/plan.md:231-241` |

## 1. Row 4: is the queued/pending result legal?

**No finding against retaining the freshly created row. This is an acceptable shape, but only as an explicitly non-C2 safety refusal.** The caller's row represents a real request that has not been sent: `new_record()` deliberately initializes `status="queued"`, `phase="pending"`, and `attempts=0`. `src/claude_teams/delivery_store.py:162-191`

The existing C2 rollback is narrower. `_guaranteed_send` calls `_discard_delivery_record` only when the result reason is in `_C2_REFUSAL_REASONS`, whose members are authorization/target refusals; the discard helper removes only a zero-attempt row. `src/claude_teams/server_simple.py:4715-4743`, `src/claude_teams/server_simple.py:5102-5123`

Therefore `pending_attempt_unidentified` should remain outside `_C2_REFUSAL_REASONS`, and the current row should remain durably `queued/pending`, not be settled `delivered` or `failed`, and not be discarded. That preserves the caller's retry handle and makes the response's status/phase honest. Revision 2's row-4 action is correct on this point. `docs/features/false-receipt-reconcile/plan.md:96-105`

The plan must still specify the complete result contract: `success=false`, `retriable=true`, `message_id`, `idempotency_key`, `call_budget_s`, and a sender obligation should be attached in the same style as `_pending_tail`; otherwise the hard-coded refusal could omit the identity that normal queued responses provide. `src/claude_teams/server_simple.py:4524-4553`, `src/claude_teams/server_simple.py:4475-4494`

There is also a smaller status-detail gap. The new response reason is `pending_attempt_unidentified`, but the untouched caller row's `reason` remains the empty string from `new_record`, and `public_view` exposes that row reason. If response/status agreement includes reason, the plan must persist the same reason on the current pending row; if it intentionally guarantees only status/phase, say so. `src/claude_teams/delivery_store.py:176-190`, `src/claude_teams/delivery_store.py:234-245`, `docs/features/false-receipt-reconcile/plan.md:9-13`

## 2. Clearing an unidentified marker and old-row recoverability

**BLOCKER — clearing the marker before resolving the old row removes the only automatic duplicate guard.** Revision 2 says to settle nothing, clear the unidentified marker, and let an immediate retry under the current key enter the normal path. `docs/features/false-receipt-reconcile/plan.md:100-105`, `docs/features/false-receipt-reconcile/plan.md:119-126`

That old row is normally still recoverable from the durable store. Attempts are marked durable before resume, and delivery records are not deleted. `src/claude_teams/server_simple.py:4393-4422`, `src/claude_teams/delivery_store.py:1-15`

After the marker is gone:

- keyed `delivery_status(old_key)` is not available if the caller truly does not know the old key;
- `delivery_status(to=name)` still enumerates and reconciles all rows for that target, and the public view includes each row's key; `src/claude_teams/server_simple.py:5221-5247`, `src/claude_teams/delivery_store.py:227-245`
- `deliver_pending()` still enumerates the sender's rows, reconciles the old nonce first, and only then considers pending rows; `src/claude_teams/server_simple.py:5304-5333`
- `kill_agent` still scans and settles all durable rows for the target before deleting the agent; `src/claude_teams/server_simple.py:2362-2388`, `src/claude_teams/server_simple.py:5563-5589`

So the durable old row is not lost in the normal case, but the marker's association and guard are lost. The immediate retry under the newly created key is `phase="pending"`; `_reconcile_before_resend` returns `None` for that phase, and `_prepare` sees no marker and proceeds to reserve/build/send. `src/claude_teams/server_simple.py:4847-4862`, `src/claude_teams/server_simple.py:5074-5088`

If the current request is the same prompt as the already-found legacy attempt, that retry sends the prompt a second time. `deliver_pending()` has the same consequence after it first reconciles the old row, because the current row is then the pending item it drains. `src/claude_teams/server_simple.py:5309-5353`

The plan must choose one safe migration rule: either locate the old row by the unique `(to, nonce)` evidence inside the existing delivery transaction and then apply the identified four-way branch, or retain an ambiguity marker/tombstone and refuse further automatic sends until the old row is reconciled. “Refuse once, clear, then send normally” is honest about the current response but not safe against duplicate delivery.

## 3. Lock order

**No new lock-order deadlock is present if the plan's stated order is implemented literally.** The registry transaction is `_agents_transaction`, which holds the agents lock through its body; the delivery transaction holds the separate delivery lock through its body and commits on exit. `src/claude_teams/server_simple.py:567-608`, `src/claude_teams/delivery_store.py:358-377`

The relevant existing paths are:

| Path | Lock relationship |
| --- | --- |
| Proposed `_prepare` reconciliation | Registry first at `_prepare`; proposed delivery transaction second. `src/claude_teams/server_simple.py:3950-3960`, `docs/features/false-receipt-reconcile/plan.md:128-142` |
| `kill_agent` | Registry first at `kill_agent`; `_reconcile_deliveries_unchecked` then opens delivery transaction. `src/claude_teams/server_simple.py:5495-5569`, `src/claude_teams/server_simple.py:2358-2387` |
| `_record_outcome` | Loads agents before opening delivery transaction; the registry lock is released by `_load_agents` first. `src/claude_teams/server_simple.py:4444-4450` |
| `_reconcile_before_resend` | Loads agents before opening delivery transaction. Revision 2's later registry cleanup must happen after that delivery transaction has closed. `src/claude_teams/server_simple.py:4847-4856`, `docs/features/false-receipt-reconcile/plan.md:156-161` |
| `delivery_status` | Loads agents before the keyed/list delivery transaction. `src/claude_teams/server_simple.py:5228-5255` |
| `deliver_pending` | Loads agents before its delivery transaction. `src/claude_teams/server_simple.py:5303-5309` |
| `send_message` child route | Its external-target registry transaction ends before the child branch calls `_guaranteed_send`. `src/claude_teams/server_simple.py:3212-3257` |
| Delivery-only paths | `_mark_attempt_sent`, `_open_delivery_record`, claim release/claim, and discard take only the delivery lock. `src/claude_teams/server_simple.py:4406-4422`, `src/claude_teams/server_simple.py:4670-4703`, `src/claude_teams/server_simple.py:4734-4743`, `src/claude_teams/server_simple.py:4894-4943` |

I found no existing nested delivery-first-then-registry path in the requested implementation. A future implementation must still close the delivery transaction before `_reconcile_before_resend` opens its normal `_agents_transaction`; holding both in the reverse order would create the cycle the current code avoids. `docs/features/false-receipt-reconcile/plan.md:156-161`

## 4. `public_view` and its tests

**MINOR — the addition is legal as an intentional contract extension, but the contract text must be updated.** `public_view` currently promises an exact documented projection and returns a fixed set of keys. `reconciled_from_key` is not currently documented there. `src/claude_teams/delivery_store.py:227-245`

Revision 2's conditional addition does not break any exact-key assertion found in the two requested test files: those tests inspect selected delivery fields, while the exact dictionaries at `tests/test_delivery_integrity.py:495-509` test raw `save_records` contents, not `public_view`. The existing delivery-status test likewise checks selected status fields only. `tests/test_delivery_integrity.py:658-692`

Nevertheless, a strict consumer can observe a new key only on alias rows. The plan must update the `public_view` docstring/query contract and test both cases: ordinary rows omit the optional provenance, alias rows include it, and `delivery_status(idempotency_key)` and `delivery_status(to=...)` expose the same value. `docs/features/false-receipt-reconcile/plan.md:107-116`, `src/claude_teams/server_simple.py:5221-5247`

## 5. Test 11: store commit succeeds, registry save fails

**MAJOR — the failure is reachable, but the claimed resulting state is not universal.** `_prepare` holds the registry lock, the proposed delivery transaction can commit at its context exit, and the subsequent `_save_agents_transaction` is a separate plain file write. There is no rollback coupling between those stores. `src/claude_teams/server_simple.py:3959-3960`, `src/claude_teams/server_simple.py:600-608`, `src/claude_teams/delivery_store.py:368-377`

The current exception boundary also matters: `_guaranteed_send` catches `DeliveryStoreError` and `LeaseStoreError`, but not an `OSError` from `_save_agents_transaction`. A real agents-file write failure can therefore escape as an exception after the delivery rows have committed. `src/claude_teams/server_simple.py:586-608`, `src/claude_teams/server_simple.py:5090-5101`

The post-failure state depends on the branch:

- For row 2, both durable rows may be terminal, but the old marker remains on disk because the registry save failed. A retry with the same current key returns at the earlier `is_terminal(record)` check and never reaches `_prepare`, so it does not re-clear that marker. `src/claude_teams/server_simple.py:5072-5075`, `docs/features/false-receipt-reconcile/plan.md:149-152`, `docs/features/false-receipt-reconcile/plan.md:228-229`
- For row 3, the old row is terminal but the caller's row remains `queued/pending`; a retry reaches `_prepare`, can clear the stale marker, and then sends the new request. Thus the plan's generic assertion that “rows are terminal” is false for this arm. `docs/features/false-receipt-reconcile/plan.md:102-105`, `docs/features/false-receipt-reconcile/plan.md:228-229`

The “next call re-scans, finds the rows terminal, and clears the marker” crash argument is consequently valid only for a nonterminal current row. Test 11 must split the alias and different-fingerprint cases and define the first-call behavior for an unhandled registry-save exception, or the implementation must add a deliberate registry-save failure contract and a cleanup path that also handles terminal-row early returns.

## 6. Remaining underspecification and new implementation choices

**MAJOR — “present and non-empty” is not enough identity validation.** Revision 2 does not say that the marker sender must equal the current durable row sender/`IDENTITY`, that the key must be a valid string key, or that the row loaded by `record_key(sender, key)` must have the expected target, nonce, and fingerprint. Without those checks, a stale or malformed marker can settle a row for another target or alias the current request to a row whose nonce merely happens to be found. `docs/features/false-receipt-reconcile/plan.md:94-105`, `src/claude_teams/delivery_store.py:123-159`, `src/claude_teams/server_simple.py:4894-4943`

The plan should define the identified predicate as typed, current-sender-owned metadata plus a matching durable old row: sender, key, target, nonce, and fingerprint must agree before any settlement or aliasing. Missing row, duplicate nonce match, target mismatch, fingerprint mismatch, and terminal `failed` row also need explicit outcomes; `settle()` deliberately refuses to overwrite terminal rows. `src/claude_teams/delivery_store.py:194-215`, `docs/features/false-receipt-reconcile/plan.md:190-192`

**MAJOR — row-4 migration and row-2 terminal asymmetry are not fully specified.** If the old row is already terminal `failed`, `settle()` leaves it failed even after the transcript scan finds the nonce. The plan acknowledges this but does not say whether row 2 may still mark a new alias `delivered` while `delivery_status(old_key)` remains `failed`, or whether aliasing must refuse in that state. `src/claude_teams/delivery_store.py:199-215`, `docs/features/false-receipt-reconcile/plan.md:190-192`

**MINOR — `send_message` wording must be branch-specific.** The default/upstream `send_message` path writes an inbox line and the external-member path is pull-based with no idempotency key or delivery row; only the spawned-child branch uses `_guaranteed_send`. A generic statement that all `send_message` reconciliation is fingerprint-based would misdocument the other two branches. `src/claude_teams/server_simple.py:3175-3189`, `src/claude_teams/server_simple.py:3205-3264`, `docs/features/false-receipt-reconcile/plan.md:164-169`

**MINOR — test 7 and test 9 need concrete state machines, not only labels.** `deliver_pending` first reconciles every row under one delivery lock and only later drains copied pending rows; the test must show which row is old, which row is current, and whether the result is an aggregate refusal or delivery. The concurrency test must state the expected outcome when two different keys see the same found marker: one old instruction, plus at most one new attempt per deliberately distinct current request. `src/claude_teams/server_simple.py:5309-5366`, `docs/features/false-receipt-reconcile/plan.md:216-221`

**MINOR — the same-key cleanup requirement needs a failure rule.** Section 4 says `_reconcile_before_resend` commits the store and then opens a normal registry transaction, but it does not say what the public result is if that second write fails, nor how a later terminal-row early return can clean the marker. This is the same cross-store state problem exposed concretely by test 11. `docs/features/false-receipt-reconcile/plan.md:156-161`, `src/claude_teams/server_simple.py:4849-4862`, `src/claude_teams/server_simple.py:5072-5075`

## Required changes before approval

1. Do not clear an unidentified marker and then permit an unqualified retry to send; recover the old row by nonce or retain an ambiguity barrier.
2. Keep row 4's fresh caller row queued/pending and outside C2, but define its complete response and reason-persistence contract.
3. Split test 11 by branch, define the registry-save failure response, and handle or explicitly test the current unhandled `OSError` path.
4. Specify typed sender/key/row validation, missing-row and terminal-failed behavior, and the branch-specific `send_message` documentation.

## Final verdict

**REJECTED.** Revision 2 is materially improved and resolves the original review's main blockers, but the unidentified-marker retry can still double-deliver an already observed prompt, and the crash-window/test-11 contract is not correct for all branches. Those issues must be closed before implementation.
