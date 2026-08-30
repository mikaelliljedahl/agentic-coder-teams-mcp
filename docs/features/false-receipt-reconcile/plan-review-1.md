# Independent plan review: false receipt reconciliation

Verdict: **REJECTED**

The plan identifies the field defect correctly, but it is not safe to implement as written. The legacy arm still permits a false receipt, the proposed store-error handling violates the delivery store's fail-closed contract, and the cross-file commit order is unspecified. Those are blockers, not hardening notes.

## A. Root-cause analysis

**Correct, with two precision corrections.**

- The pending object is agent-scoped: `_finalize_follow_up` writes it into the agent record while the registry transaction is active, and the current object contains only `nonce`, `operation_id`, `attempted_at`, and `prompt_file`; it has no sender, idempotency key, or request fingerprint. `src/claude_teams/server_simple.py:3781-3795`, `src/claude_teams/server_simple.py:3881-3893`
- `_pending_delivery` returns that object, and `_reconcile_pending_delivery` uses only its nonce. It rewinds and performs a full scan, but it never compares the pending attempt with the current prompt or key. `src/claude_teams/server_simple.py:2168-2200`
- The current caller's row is created before the delivery wait. `new_record` initializes it as `queued/pending` with `attempts: 0`, including the caller's fingerprint. `src/claude_teams/server_simple.py:5022-5075`, `src/claude_teams/delivery_store.py:162-191`
- When the old nonce is found, `_prepare` clears the agent field and returns a hard-coded successful `delivered/reconciled` result directly. `_guaranteed_delivery` returns that result without `_record_outcome`, and `_guaranteed_send` only rolls back a newly created row for reasons in `_C2_REFUSAL_REASONS`; this result has no such reason. Thus the new row remains pending. `src/claude_teams/server_simple.py:4079-4107`, `src/claude_teams/server_simple.py:4259-4263`, `src/claude_teams/server_simple.py:4425-4472`, `src/claude_teams/server_simple.py:5102-5123`
- `delivery_status` cannot reconcile a row that is still `pending`, because `_reconcile_delivery_record` handles only `sent` and `unconfirmed` phases. `src/claude_teams/server_simple.py:2299-2307`, `src/claude_teams/server_simple.py:5253-5272`

The plan's core claim is therefore verified. Two statements should be narrowed: the old row can also be settled by `kill_agent` or a drain, not only by a later keyed `delivery_status`, and the new row is not literally permanent because `deliver_pending` can eventually process a pending row. `src/claude_teams/server_simple.py:5276-5373`, `src/claude_teams/server_simple.py:5563-5589`

**NIT — stale symbol and line references.** The current equivalent of the plan's `_finish_follow_up` is `_finalize_follow_up`, and the current store helper is named `request_fingerprint`, not `fingerprint`. The implementation plan should name the current symbols. `docs/features/false-receipt-reconcile/plan.md:63-80`, `src/claude_teams/server_simple.py:3781-3786`, `src/claude_teams/delivery_store.py:147-159`

## B. Review of the four-way branch

### Same fingerprint, same key

**MAJOR — the plan does not account for the earlier resend gate.** A normal same-key retry with a `sent`/`unconfirmed` row is handled by `_reconcile_before_resend` before `_prepare`; it rescans and returns the settled result without entering the pending-agent shortcut. That path also does not clear `pending_delivery`. Therefore the proposed same-key arm is not the unchanged path the plan assumes, and case 4 will not prove the new `_prepare` branch. `src/claude_teams/server_simple.py:4838-4862`, `src/claude_teams/server_simple.py:5074-5084`, `docs/features/false-receipt-reconcile/plan.md:82-91`

The fix must define one cleanup rule for both paths: when the row whose nonce was found is the pending agent attempt, settle the row and clear the matching agent field, without reloading `agents.json` while the registry lock is held. `src/claude_teams/server_simple.py:2219-2223`

### Same fingerprint, different key

**Conditionally safe and honest.** `request_fingerprint` covers the target, prompt, and options, and `SCAN_FOUND` after `rewind()`/`full_scan()` is positive evidence that the old prompt was observed in the target context. If the pending sender, old key, old fingerprint, and current request fingerprint all match the intended alias relation, `delivered` is still truthful for the prompt even though the new row has no nonce of its own. No fourth public status is needed: the store explicitly has three statuses, with transport state represented by `phase`. `src/claude_teams/delivery_store.py:50-62`, `src/claude_teams/delivery_store.py:147-159`, `src/claude_teams/server_simple.py:2194-2200`, `src/claude_teams/server_simple.py:5148-5154`

The plan should keep `status="delivered"`, leave the new row's nonce empty, and persist/surface `reconciled_from_key` as provenance. It must not copy the old nonce into the new row. As written, this is underspecified: `settle()` only sets status, phase, reason, and settled time, while `public_view()` drops all extra fields. The plan must say whether `reconciled_from_key` is a durable/public field or response-only, and test the promised surface. `src/claude_teams/delivery_store.py:199-215`, `src/claude_teams/delivery_store.py:227-245`, `docs/features/false-receipt-reconcile/plan.md:88-97`

**MAJOR — the identity plumbing is missing.** The current `_finalize_follow_up` has no `record` parameter, and `_FollowUpPlan` has no sender, idempotency key, or fingerprint fields. The code snippet in the plan cannot be applied at the write site without explicitly carrying those values from `_guaranteed_delivery` into `_finalize_follow_up` (or adding them to the plan). `src/claude_teams/server_simple.py:1992-2020`, `src/claude_teams/server_simple.py:3781-3786`, `src/claude_teams/server_simple.py:4303-4332`, `docs/features/false-receipt-reconcile/plan.md:65-75`

### Different fingerprint

**Safe against duplicating the old attempt, but not universally safe as phrased.** Once the old nonce is positively found, the old prompt has been observed, so deliberately sending the current request does not resend that old attempt. `src/claude_teams/server_simple.py:2197-2200`, `docs/features/false-receipt-reconcile/plan.md:121-124`

However, a different fingerprint does not necessarily mean different prompt text: the fingerprint also includes `options`, and the current option set includes `replace_if_idle`. The same prompt with only that option changed will be sent again. That may be an intentional “new request” policy, but the plan must state it and test it; the old-nonce proof does not by itself prove that the new prompt is semantically different. `src/claude_teams/server_simple.py:147-159`, `src/claude_teams/server_simple.py:5064-5067`

This arm is safe only if the old row is durably settled before the agent field is cleared and before the new attempt is marked/sent. The plan currently does not specify that ordering. `docs/features/false-receipt-reconcile/plan.md:88-106`

### No fingerprint / legacy

**BLOCKER — the proposed conservative default is still a false receipt.** A legacy object has no evidence connecting the old nonce to the current caller's prompt, key, or sender. Keeping the shortcut still drops the current prompt and returns `delivered`; it also leaves the newly created current row at `queued/pending`. That directly violates the plan's own “response and `delivery_status` must not contradict” scope. `docs/features/false-receipt-reconcile/plan.md:5-12`, `docs/features/false-receipt-reconcile/plan.md:90-97`, `docs/features/false-receipt-reconcile/plan.md:146-149`, `src/claude_teams/server_simple.py:4079-4107`, `src/claude_teams/server_simple.py:5148-5154`

The default must be an explicit refusal/uncertain result for the current request, with no claim that its prompt was delivered. A legacy object also cannot safely identify an old delivery row to settle. Missing or malformed sender/key fields must take the same ambiguous path; checking only whether `fingerprint` exists is insufficient. `src/claude_teams/server_simple.py:3886-3891`, `src/claude_teams/delivery_store.py:142-159`

## C. Concurrency, locks, and crash windows

**No inherent deadlock if the implementation follows the existing lock order.** `_prepare` runs under `_agents_transaction`; `_scan_for_nonce` explicitly accepts the already-loaded agent record and warns that re-entering `_load_agents` while holding the registry lock deadlocks or times out. The existing kill path already uses registry lock → delivery transaction, while resend/status paths load agents before entering the delivery transaction and do not reacquire the registry lock inside it. `src/claude_teams/server_simple.py:3950-3960`, `src/claude_teams/server_simple.py:2219-2223`, `src/claude_teams/server_simple.py:4838-4856`, `src/claude_teams/server_simple.py:5228-5238`, `src/claude_teams/server_simple.py:5563-5569`, `src/claude_teams/delivery_store.py:358-377`

The new helper may therefore open a delivery transaction while the existing registry transaction is held, but it must use the `pending` object and the current `record`; it must not call `_load_agents`, `_reconcile_before_resend`, or any helper that re-enters the registry lock. `src/claude_teams/server_simple.py:3959-4090`, `src/claude_teams/server_simple.py:2219-2223`

**BLOCKER — the cross-store commit order is missing.** The delivery transaction persists only when its context exits, and a failed dirty write raises `DeliveryStoreError`. The safe sequence inside the existing registry transaction is:

1. Perform the positive scan using the already-held agent record.
2. In one delivery transaction, settle the old row and, for the same-fingerprint/different-key arm, the caller's row; add `reconciled_from_key` before the transaction commits.
3. Only after that transaction exits successfully, clear `pending_delivery` and save `agents.json` while still inside the registry transaction.
4. For the different-fingerprint arm, only then fall through to reserve/build the new attempt; the new row still must be marked durable before resume.

If a process dies after step 2 and before step 3, the agent field may be stale, but the terminal rows make a repeat idempotent and the next call can clear the stale field. If step 3 happens first, or if the store write is ignored, a retry can see no pending marker while the caller's row is still pending and send the same request again. The plan must state this ordering and the atomicity of the two row settlements. `src/claude_teams/delivery_store.py:358-377`, `src/claude_teams/server_simple.py:4406-4422`, `src/claude_teams/server_simple.py:4450-4472`

**BLOCKER — swallowing `DeliveryStoreError` is incompatible with this path.** The store documents persistence failure as a condition on which nothing may rely, and the transaction raises on a lost write. `_guaranteed_send` already catches that exception and returns the fail-closed `delivery_store_unavailable` result before any resume can rely on an undurable row. The proposed helper would bypass that handling; in the alias arm it can return `delivered` while the caller's row remains `queued/pending`, reproducing the reported defect. `src/claude_teams/delivery_store.py:86-95`, `src/claude_teams/delivery_store.py:368-377`, `src/claude_teams/server_simple.py:5077-5099`, `docs/features/false-receipt-reconcile/plan.md:99-106`

The store-failure test must expect a non-success/fail-closed result, no new resume, both rows unchanged, and the pending agent marker retained. “Does not raise” is true at the public tool boundary only because `_guaranteed_send` catches the propagated exception; it is not a reason to swallow it in the helper. `src/claude_teams/server_simple.py:5090-5099`, `src/claude_teams/server_simple.py:4760-4775`

## D. Missing paths and contract changes

**MAJOR — `deliver_pending` cannot be out of scope.** It calls `_guaranteed_delivery`, whose `_prepare` is the exact shortcut under review. A pending delivery drained by this tool can therefore encounter the new pending-agent identity, and the four-way behavior must be defined for it. `src/claude_teams/server_simple.py:5276-5297`, `src/claude_teams/server_simple.py:5309-5353`, `src/claude_teams/server_simple.py:4259-4282`, `docs/features/false-receipt-reconcile/plan.md:11-12`

**MAJOR — `_reconcile_before_resend` needs an explicit interaction rule.** It runs before `_prepare` for every nonterminal row, and it can settle a same-key old row before the agent-level shortcut is reached. It currently updates only the delivery row and the caller's local record; it does not clear `pending_delivery`. The plan must either make this path clear the matching marker or explicitly make the later `_prepare` cleanup safe and test both paths. `src/claude_teams/server_simple.py:4838-4862`, `src/claude_teams/server_simple.py:5074-5084`, `src/claude_teams/server_simple.py:3881-3893`

**MAJOR — the `send_message` child route is part of the same public behavior.** For a spawned child it calls `_guaranteed_send` with the same delivery machinery, so the fix automatically changes that route. Its tool docstring must describe the same-key/new-key reconciliation and legacy refusal semantics, not only `follow_up_agent`'s docstring. `src/claude_teams/server_simple.py:3167-3190`, `src/claude_teams/server_simple.py:3205-3264`, `src/claude_teams/server_simple.py:5022-5028`

The `follow_up_agent` docstring also needs the new rule: a new key is treated as an alias only for a fingerprint-identical request whose old nonce was found; a different fingerprint is sent as a new request; a legacy marker is not enough to claim delivery. The current text documents only same-key retries and the old three statuses. `src/claude_teams/server_simple.py:5126-5158`, `docs/features/false-receipt-reconcile/plan.md:115-116`

**MINOR — kill-time reconciliation is already a separate path and must stay so.** `kill_agent` holds the registry lock, calls `_reconcile_deliveries_for_target`, and that helper scans/settles every delivery row for the target before deleting the agent. It does not need the agent's pending blob, because the durable rows carry their own nonce and snapshot. The new helper must not be inserted in a way that makes kill re-enter the registry lock or depends on the soon-to-be-deleted agent record. `src/claude_teams/server_simple.py:2334-2388`, `src/claude_teams/server_simple.py:5495-5589`

## E. Test adequacy

Cases 1–4 are at the right broad level in `test_follow_up_delivery.py`: that fixture uses a real transcript file, the real scanner and lease/store files, a fake resume backend only at the backend boundary, and an injected clock. Those are appropriate for proving actual `follow_up_agent` behavior rather than only helper behavior. `tests/test_follow_up_delivery.py:1-8`, `tests/test_follow_up_delivery.py:49-80`, `tests/test_follow_up_delivery.py:125-187`

The store-failure case belongs in `test_delivery_integrity.py`, but it must seed the old and caller rows first and inject the failure after the initial row creation. Failing `save_records` before the call would fail `_open_delivery_record` and never exercise pending-row settlement. The existing test suite already uses OS-boundary failure injection for this style of test. `tests/test_delivery_integrity.py:462-483`, `tests/test_delivery_integrity.py:779-796`, `tests/test_delivery_integrity.py:955-982`

**The six cases are not sufficient as written.** Add or amend the following:

- **MAJOR:** Assert that an unconfirmed attempt writes sender, idempotency key, and fingerprint into `pending_delivery`, not only its nonce. The existing test checks only the nonce. `tests/test_follow_up_delivery.py:379-396`
- **MAJOR:** Exercise the public `deliver_pending` path with a pending caller row and an already-found agent-level nonce. It shares `_prepare` despite the plan declaring it out of scope. `src/claude_teams/server_simple.py:5276-5353`
- **MAJOR:** Exercise `send_message(to=child)` for both alias and different-fingerprint branches, or explicitly prove the shared public contract through that entry point. `src/claude_teams/server_simple.py:3184-3189`, `src/claude_teams/server_simple.py:3253-3264`
- **MAJOR:** Add a legacy/malformed-identity test whose expected result is an honest refusal or queued non-delivery result, not `delivered/reconciled`. A blob with a fingerprint but no sender/key must also be covered. `docs/features/false-receipt-reconcile/plan.md:90-97`, `src/claude_teams/server_simple.py:3886-3891`
- **MAJOR:** Add a fault-injection test for the settlement commit and assert no resume, no marker clearing, and a fail-closed response. Also cover the crash-safe ordering assumption, at least by making the store commit succeed while the subsequent agent save fails and verifying that a retry does not resend the old request. `src/claude_teams/delivery_store.py:358-377`, `src/claude_teams/server_simple.py:5090-5099`
- **MINOR:** Add a concurrent/retry test that proves two callers cannot both consume the same found pending marker and that only the deliberately distinct request is sent. The existing concurrency tests cover same-key claims/leases, not this found-marker branch. `tests/test_follow_up_delivery.py:494-551`, `tests/test_delivery_integrity.py:229-305`
- **MINOR:** Assert both keyed views after an alias: the old key is delivered, the new key is delivered with the promised provenance, and the new row retains no false nonce. `src/claude_teams/server_simple.py:5186-5272`, `src/claude_teams/delivery_store.py:227-245`

**Existing test that must change.** `test_a_later_flush_reconciles_and_does_not_resend` is the exact current regression: it creates `k29`, retries with new key `k30`, asserts `delivered/reconciled`, and asserts only one backend resume, but never queries `k30` or `k29` in the delivery store. It must assert the new-key row is terminal and the old row is settled, plus the provenance contract. `tests/test_follow_up_delivery.py:398-423`

The same-key test also exercises `_reconcile_before_resend`, not the new `_prepare` branch; update it if the fix promises marker cleanup there. `tests/test_delivery_integrity.py:400-455`

## Final verdict

**REJECTED.** Approve only after the plan (1) replaces the legacy shortcut with an honest refusal/uncertain result, (2) propagates delivery-store failures to the existing fail-closed boundary, (3) specifies and tests delivery-store commit before registry-marker clearing, preferably settling both rows in one delivery transaction, (4) carries identity into the actual `_finalize_follow_up` write site, and (5) covers `deliver_pending`, `send_message`, `_reconcile_before_resend`, public provenance, and the updated regression test.
