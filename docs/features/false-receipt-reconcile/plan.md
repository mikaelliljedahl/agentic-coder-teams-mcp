# Plan: false `delivered` receipt from the pending-delivery reconcile shortcut

Revision 4 — **scoped**. Revisions 1-3 were rejected by independent review
(`plan-review-1.md`, `plan-review-2.md`, `plan-review-3.md`) with findings that
progressively widened from "the receipt lies" to "harden the whole nonce
lifecycle". A second independent assessment (Fable) judged the widening to be
scope escalation beyond what the defect warrants. Revision 4 fixes the receipt
and dispositions the rest as named follow-up work.

## Scope

Fix the false receipt: `follow_up_agent` must never answer `delivered` for a
prompt it did not send, and `delivery_status(key)` must never contradict the
response that produced that key.

In scope, because they share `_guaranteed_delivery`/`_prepare` or the same
marker: `follow_up_agent`, `send_message` **to a spawned child**,
`deliver_pending`, `_reconcile_before_resend`.

**Explicitly out of scope**, documented here and filed as follow-ups:

1. **Nonce overwrite on markerless retriable retries.** A row returned to
   `phase=pending` by `_record_outcome` (`server_simple.py:4465`) keeps its
   nonce, and the next same-key attempt overwrites it
   (`_mark_attempt_sent`, `server_simple.py:4406-4414`). This cannot corrupt
   the lookup this plan adds: the marker is written only together with
   `phase=unconfirmed` (`server_simple.py:3881-3893`), and an `unconfirmed` row
   is resend-blocked by `_reconcile_before_resend` (`server_simple.py:4847-4862`)
   and is never demoted back to `pending` by `_reconcile_delivery_record`
   (`server_simple.py:2299-2331`). It is a real, separate duplicate window.
2. **Non-`FOUND` scan outcomes.** `_reconcile_pending_delivery` returns a bool;
   `SCAN_ABSENT`/`SCAN_INDETERMINATE`/`SCAN_AMBIGUOUS` fall through to a normal
   send today (`server_simple.py:2188-2200`, `4083-4109`). Unchanged here.
3. **Repair machinery for a damaged delivery store.** `kill_agent` remains the
   escape.

Also out of scope: the `unconfirmed` phase itself, the binding ladder,
`kill_agent`'s `_reconcile_deliveries_for_target`, and the inbox/external
branches of `send_message`, which have no durable delivery row
(`server_simple.py:3175-3189`).

## Current behaviour

1. A follow-up that resumes the child but does not observe its nonce within the
   scan bound returns `queued/unconfirmed` and records the attempt on the
   **agent record** (`_finalize_follow_up`, `server_simple.py:3881-3893`) as
   `{nonce, operation_id, attempted_at, prompt_file}` — no sender, no key, no
   fingerprint.
2. The next delivery to that agent — any key, any prompt, from
   `follow_up_agent`, a child `send_message`, or `deliver_pending` — opens a
   durable row for its own key (`server_simple.py:5022-5075`), then reaches
   `_reconcile_pending_delivery` in `_prepare` (`server_simple.py:4079-4107`).
   On `SCAN_FOUND` it returns a hard-coded
   `{"status": "delivered", "reconciled": True}`.
3. Three consequences: the new prompt is silently dropped; the caller's row is
   orphaned at `queued/pending, attempts: 0` (no reason in
   `_C2_REFUSAL_REASONS`, `server_simple.py:5102-5123`; and
   `_reconcile_delivery_record` only moves `sent`/`unconfirmed` rows,
   `2299-2307`); and the pending attempt's own row stays in flight. The orphan
   is later drainable by `deliver_pending`, which can deliver the same content
   a second time under the abandoned key.

Reproduced live while preparing this plan: a follow-up under a fresh key
answered `delivered / reconciled: true` while `delivery_status(that key)`
answered `queued / pending / attempts: 0 / nonce: ""`, and the prompt was never
sent.

## Design

### 1. Resolve the pending attempt from the store, by nonce

The marker is **not** extended. Its `nonce` is already a durable handle, and
every attempt writes its nonce onto its row before the resume
(`_mark_attempt_sent`, `server_simple.py:4393-4422`). Inside one
`delivery_transaction`, the pending attempt's row `R` is the row of **this
sender**, for **this target**, whose `nonce` equals the marker's:
`txn.for_sender(sender, to=name)` filtered on a non-empty equal nonce
(`delivery_store.py:337-352`). A marker written by an older server resolves
identically — there is no legacy case.

Comparison then uses `R`'s own `fingerprint`
(`request_fingerprint(to, prompt, options)`, `delivery_store.py:147-159`), the
value already behind `idempotency_conflict`. It covers `options`, so a request
differing only in `replace_if_idle` is a different request.

### 2. The branch

`C` is the caller's row. Reached only on `SCAN_FOUND`, with the registry
transaction held by `_prepare` and the store work in one delivery transaction.

| Case | Action |
| --- | --- |
| exactly one `R`, `R` is `C` (same key) | settle `R` `delivered`; clear the marker; return `delivered/reconciled` |
| exactly one `R`, different key, `R.fingerprint == C.fingerprint` | settle `R` `delivered`; settle `C` `delivered` with `reconciled_from_key=R.idempotency_key` and an **empty** nonce; clear the marker; return `delivered/reconciled` + `reconciled_from_key` |
| exactly one `R`, different key, different fingerprint | settle `R` `delivered`; clear the marker; **fall through to the normal send path** — the new prompt is really sent |
| `R` already terminal `failed`, or zero/several matches | **barrier**: settle nothing, send nothing, **keep the marker**, set `C`'s reason, return a retriable refusal |

`delivered` on an aliased `C` is honest: `SCAN_FOUND` after
`rewind()`/`full_scan()` is positive evidence that a request with *this
fingerprint* was observed in the target's context. The old nonce is never
copied onto `C`; `reconciled_from_key` carries the provenance instead. The
different-fingerprint arm cannot resend the old instruction — it is reached
only after that instruction's nonce was positively found.

The barrier row covers two states the store cannot corroborate: a row already
settled `failed` (kill-time `SCAN_ABSENT` cleanup, `server_simple.py:2362-2381`,
which `settle()` will not overwrite, `delivery_store.py:199-215`), and zero or
several nonce matches (a damaged store, or a ~2^-128 collision). Both keep the
marker, because it is the only remaining duplicate guard. Reasons:
`prior_attempt_settled_failed` and `pending_attempt_unresolvable`. Both are
retriable and re-evaluated on every call; `kill_agent` is the documented
escape. Neither is reachable from healthy operation.

`C` stays `queued/pending, attempts: 0` under the barrier — an unsent request,
which is the truth — with `mark_phase(C, PHASE_PENDING, reason=<the same
reason>)` so `delivery_status(C.key)` reports what the response reported. The
response is built through `_with_public_status`/`_with_delivery_identity`
(`server_simple.py:4475-4494`) like every other queued answer.

### 3. Order and failures

Inside the registry transaction `_prepare` already holds:

1. Scan with the **already-held** agent record — never re-enter `_load_agents`
   from here (`server_simple.py:2219-2223`).
2. One `delivery_transaction`: resolve `R`, settle `R`/`C`, or set the barrier
   reason. Close it.
3. Only then clear `PENDING_DELIVERY_FIELD` and `_save_agents_transaction`.
4. Send-through arm: only then reserve the lease and mark the new attempt.

Lock order registry → delivery matches `kill_agent`
(`server_simple.py:5495-5569`, `2358-2387`); no path takes the reverse.
`DeliveryStoreError` from step 2 is propagated to `_guaranteed_send`'s
fail-closed handler (`server_simple.py:5077-5101`): nothing sent, marker
retained, rows unchanged.

A marker left behind (crash between steps 2 and 3) is benign: the rows are
terminal, so the next call that reaches the shortcut re-settles nothing and
clears it. A same-key retry returns earlier at `is_terminal(record)`
(`server_simple.py:5072-5075`) and leaves the residue in place; that is
accepted, not an invariant.

### 4. Adjacent paths

- **`_reconcile_before_resend`** (`server_simple.py:4838-4862`) settles a
  same-key row before `_prepare` is reached and leaves the marker behind. It
  clears the marker when the row it settled carries the marker's nonce, in a
  separate `_agents_transaction` opened after its delivery transaction closes.
  A failure of that registry write leaves the same benign residue; the delivery
  result stands.
- **`deliver_pending`** (`server_simple.py:5276-5353`) reaches the same
  `_prepare`; all arms apply.
- **Docstrings.** `follow_up_agent` (`server_simple.py:5126-5158`) and the
  spawned-child branch of `send_message` (`server_simple.py:3184-3189`) state:
  reconciliation applies to a fingerprint-identical request; a new key for an
  identical request returns `delivered` with `reconciled_from_key`; a different
  request is sent; an unresolvable prior attempt is refused, not claimed. The
  inbox and external branches are unchanged.
- **`public_view`** (`delivery_store.py:227-245`) surfaces
  `reconciled_from_key` when present, documented as optional provenance.

## Files affected

- `src/claude_teams/server_simple.py` — the `_prepare` shortcut (~4079-4107), a
  new `_resolve_pending_row`/`_settle_reconciled_attempt` pair,
  `_reconcile_before_resend` (~4838), the two tool docstrings.
- `src/claude_teams/delivery_store.py` — `public_view` and its contract text.
- `tests/test_follow_up_delivery.py`, `tests/test_delivery_integrity.py`.

`_FollowUpPlan`, `_finalize_follow_up`, `_mark_attempt_sent`,
`_record_outcome`, the row schema and `kill_agent` are **untouched**.

## Risks

- **A new send that did not happen before** (different-fingerprint arm),
  guarded by the positive scan of the old nonce. The fingerprint includes
  `options`, so the boundary is broader than "different prompt text" — stated
  and tested.
- **`delivered` on a row with no nonce of its own** (alias arm) — provenance is
  durable and public rather than implied.
- **The barrier arms refuse where the old code answered.** Only reachable from
  a crash-orphaned `failed` row or a damaged store; retriable; `kill_agent`
  escapes.
- **Out-of-scope item 1** leaves a duplicate window on markerless retriable
  retries, unchanged by this fix.

## Test cases (red first)

`tests/test_follow_up_delivery.py`:

1. **Alias.** Unconfirmed under `k1`; receipt flushed; follow-up under `k2`
   with an identical request → `delivered` + `reconciled_from_key="k1"`;
   `delivery_status("k2")` is `delivered` with that provenance and an empty
   nonce; `delivery_status("k1")` is `delivered`; one resume; marker cleared.
2. **Different prompt.** Same setup, different prompt under `k2` → two resumes,
   `k2` carries its own nonce, response is not `reconciled`, `k1` `delivered`.
3. **Option-only difference.** Same prompt, `replace_if_idle=False` → behaves
   as 2.
4. **Same key.** Retry under `k1` → one row `delivered`, nothing resent, marker
   cleared (via `_reconcile_before_resend`).
5. **Barrier: unresolvable.** Marker nonce matches no row → reason
   `pending_attempt_unresolvable`, nothing sent, nothing `delivered`, **marker
   retained**, `delivery_status(k2)` shows `queued/pending` with the same
   reason.
6. **Barrier: prior failed.** `R` seeded terminal `failed` with its nonce in
   the transcript → reason `prior_attempt_settled_failed`, nothing sent, marker
   retained, `R` still `failed`.
7. **`deliver_pending`** with `k1` unconfirmed (receipt flushed) and `k2`
   pending and identical → `k1` `delivered`, `k2` aliased, no resume.
8. **`send_message(to=<spawned child>)`** through the alias arm.

`tests/test_delivery_integrity.py`:

9. Delivery-store write fails during the settlement: fail-closed, no resume,
   rows unchanged, marker retained.
10. `public_view` shape: ordinary rows omit `reconciled_from_key`, alias rows
    include it, and both `delivery_status(key)` and `delivery_status(to=...)`
    expose it.

**Existing tests that change**

- `test_a_later_flush_reconciles_and_does_not_resend`
  (`tests/test_follow_up_delivery.py:398-423`) encodes the bug: it retries under
  a new key, asserts `delivered/reconciled`, and never queries either key. It
  gains assertions that both rows are terminal and the provenance is present.
- The marker-content assertion (`tests/test_follow_up_delivery.py:379-396`) is
  unchanged — no new marker fields.

## Validation

```
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

## Disposition of review 3

| Finding | Disposition |
| --- | --- |
| BLOCKER: nonce not immutable across retriable same-key failures | **Deferred, with a reason.** Verified real, but unreachable while a marker exists: the marker is written only with `phase=unconfirmed`, which is resend-blocked and never demoted to `pending`. Filed as follow-up 1 |
| BLOCKER: case D is a manual-recovery barrier, not self-healing | **Accepted as described.** §2 states it plainly as a barrier with `kill_agent` as the escape, reachable only from a damaged store |
| BLOCKER: case C same-fingerprint retry can double-deliver | **Accepted.** The marker is now **retained** for a terminal-`failed` row; nothing is cleared and nothing is sent |
| MAJOR: registry-save `OSError` contract | **Deferred.** Revision 4 does not add an `OSError` boundary; a failed registry write leaves the benign stale-marker residue described in §3, which the next call clears |
| BLOCKER: behaviour for non-`FOUND` scan outcomes undefined | **Deferred, unchanged from today.** Filed as follow-up 2 |
| MAJOR: resolver predicate completeness | **Accepted in part.** The predicate is sender + target + non-empty equal nonce against a durable row; anything else is the barrier arm |
| MAJOR: case C test setup ambiguity | **Accepted.** Test 6 seeds the post-kill state directly rather than calling `kill_agent` |
| MAJOR: nonce collision handling | **Accepted.** Treated exactly like corruption — several matches take the barrier arm, settling nothing |
| MINOR: stale-marker claim too broad | **Accepted.** §3 states it as an accepted residue, not an invariant |
| MINOR: `public_view` contract text and both shapes | **Accepted.** §4 and test 10 |
| MINOR/NIT: "unique" nonce wording; `deliver_pending` missing from the detail text | **Accepted** in the wording of §1 and the refusal detail |
