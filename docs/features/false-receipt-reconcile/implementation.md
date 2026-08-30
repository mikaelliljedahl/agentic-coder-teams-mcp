# Implementation: request-scoped reconciliation of a found receipt

Implements `plan.md` revision 4 (option A). Three plan reviews
(`plan-review-1..3.md`) and one independent risk assessment preceded it; the
scope decision and the deferred findings are recorded in the plan's disposition
table.

## What changed

`src/claude_teams/server_simple.py`

- **`_resolve_pending_row(txn, sender, to, nonce)`** — the agent marker holds a
  nonce and nothing else, so the delivery store decides which request that
  nonce belonged to: the one row of this sender, for this target, carrying that
  nonce. Zero or several matches means the store cannot answer.
- **`_settle_reconciled_attempt(session_id, name, nonce, record)`** — one
  delivery transaction that settles what the found receipt proves and returns
  the verdict: `same_key`, `alias`, `send`, or a barrier reason. The two rows
  can never settle apart, because they settle together.
- **`_answer_reconciled_attempt(...)`** — turns a verdict into the caller's
  answer inside `_prepare`'s registry transaction: store commit first, then the
  marker clear and `agents.json` save; `None` means "different request, send
  it".
- **`_clear_reconciled_marker(session_id, agent_name, nonce)`** — used by
  `_reconcile_before_resend`, which settles the same-key row before `_prepare`
  is ever reached and previously left the marker behind.
- **`_reconcile_barrier(...)`** plus `REASON_PENDING_UNRESOLVABLE` /
  `REASON_PRIOR_SETTLED_FAILED` — retriable refusals that send nothing, claim
  nothing, and **keep** the marker, because it is the only remaining guard
  against re-sending a prompt that provably landed.
- The `_prepare` shortcut now calls `_answer_reconciled_attempt` instead of
  returning a hard-coded `delivered`.
- Docstrings: `follow_up_agent` and the spawned-child branch of `send_message`
  state the request-scoped rule. That docstring is the only thing a consuming
  agent reads.

`src/claude_teams/delivery_store.py`

- `RECONCILED_FROM_FIELD` (`reconciled_from_key`) and its exposure in
  `public_view` when present — the one optional member of the query contract,
  so an aliased row's `delivered`-with-empty-nonce is explainable.

Untouched, as planned: `_FollowUpPlan`, `_finalize_follow_up`,
`_mark_attempt_sent`, `_record_outcome`, the row schema, `kill_agent`.

## Behaviour, before and after

| Situation | Before | After |
| --- | --- | --- |
| New key, identical request | `delivered/reconciled`; caller's row left `queued/pending, attempts: 0` | `delivered` + `reconciled_from_key`; both rows terminal |
| New key, different request | `delivered/reconciled` — **the new prompt was silently dropped** | old row settled `delivered`; the new prompt is really sent |
| Same key retry | `delivered`; marker left behind | `delivered`; marker cleared |
| Marker's row missing/duplicated | `delivered/reconciled` on no evidence | `pending_attempt_unresolvable`, retriable, nothing sent, marker kept |
| Marker's row already `failed` | `delivered/reconciled`, contradicting that row | `prior_attempt_settled_failed`, retriable, nothing sent, marker kept |

## Red → green

Tests were written first and run red before any production change:

```
6 failed, 28 deselected      # KeyError: 'reconciled_from_key'; assert True is False; ...
```

Failures, in order: `test_a_later_flush_reconciles_and_does_not_resend` (no
provenance), `test_a_new_key_for_the_same_request_settles_the_callers_own_row`,
`test_a_new_key_for_a_different_request_is_really_sent`,
`test_an_option_only_difference_is_a_different_request`,
`test_an_unresolvable_prior_attempt_is_refused_not_claimed`,
`test_a_prior_attempt_settled_failed_is_not_resurrected`.

Green after the change, plus the later additions:

- `tests/test_follow_up_delivery.py` — alias, different prompt, option-only
  difference, same-key marker cleanup, both barriers, `deliver_pending` alias,
  `send_message` to a child, and a lost settlement write (fail-closed: nothing
  sent, rows unchanged, marker retained).
- `tests/test_delivery_store.py` — provenance appears only on an aliased row's
  public view.

Deviation from the plan: the store-failure test lives in
`tests/test_follow_up_delivery.py` rather than `tests/test_delivery_integrity.py`,
because the end-to-end fixture there (real transcript, real scanner, real store)
is what makes "nothing was sent and the marker survived" assertable.

## Review round 1 (`implementation-review-1.md`) — REJECTED, then fixed

| Finding | Fix |
| --- | --- |
| **BLOCKER**: `_reconcile_before_resend` cleared the marker on ANY terminal row, including `failed`. A late transcript write after a negative scan would then meet no marker, and the next call would resend a prompt the target already had. | The cleanup now runs only when the row settled `delivered` (`server_simple.py:5079-5097`). Regression: `test_a_late_receipt_after_a_failed_same_key_retry_does_not_resend`. |
| **MAJOR**: the lost-settlement-write test injected the failure before the caller's row was created, so it never reached the settlement transaction and would have passed with the old bug. | The failure is now one-shot: the first store write (row creation) succeeds, the settlement write fails, and the test asserts `writes["n"] > 1`, that `k117` exists, and that neither row settled. |
| MINOR: duplicate-nonce barrier untested. | `test_two_rows_carrying_one_nonce_are_a_barrier_too`. |
| MINOR: the option-only test proved a refusal, not a real send. | Kept, and paired with `test_an_option_only_difference_is_actually_sent`, which follows the fall-through to a genuine second resume with its own nonce. |
| MINOR: `_RECONCILE_SAME_KEY` is not reachable from a healthy public path (the same-key retry settles earlier, in `_reconcile_before_resend`). | Accepted as defensive; it is what answers a partially-written state, and the code says so. |
| MINOR: `deliver_pending`/`send_message` cover the alias arm but not their own send-through arm. | Accepted: the arms share one implementation, and the different-request path is proven through `follow_up_agent` twice. |
| MINOR: the failed-row test seeds the state instead of driving `kill_agent`. | Accepted, as planned — `kill_agent` is out of scope and the barrier decision is what is under test. |

## Review round 2 (`implementation-review-2.md`) — APPROVED

Both round-1 blockers verified fixed at the right place and genuinely covered:
the `failed` settlement retains the marker, and the settlement-write test now
reaches `_settle_reconciled_attempt`'s transaction. The three dispositions above
were accepted. One NIT remains, recorded rather than changed:
`_RECONCILE_SAME_KEY` is defensive, not reachable from a healthy public path.

## Validation

```
uv run ruff format --check .     # clean
uv run ruff check .              # All checks passed!
uv run ty check                  # 1 pre-existing diagnostic, see below
uv run pytest                    # 1337 passed, 2 failed - see below
```

### Pre-existing breakage, not introduced here

Both reproduce on `HEAD` with this branch's changes reverted, and both look
Windows-only:

1. `tests/test_follow_up_delivery.py::test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`
   fails. The test patches `process_manager.owns_process`, but the kill path
   probes ownership through `process_manager.ownership_probe`
   (`server_simple.py:2089-2107`). On this machine the probe answers
   *indeterminate* for the fabricated PID 4242 rather than *not ours*, so
   `kill_agent` refuses with "holder is provably alive". Verified against a
   clean `HEAD` checkout of `src/`.
2. `tests/test_watch_command_discovery.py::test_watch_argv_executes_and_times_out_quietly`
   failed in one full-suite run and passes in isolation. It launches a real
   watcher subprocess and asserts a timeout, so it is load-sensitive; the run
   was concurrent with a Codex agent and a second pytest process.
3. `ty check` reports `tests/test_join_team.py:730` —
   `BaseContext has no attribute Process`. Untouched by this branch; the repo's
   `CLAUDE.md` notes Windows-only `ty` diagnostics that do not appear in CI.

Neither is fixed here: (1) is a real question about `ownership_probe`'s
platform behaviour or the test's patch target, not a cosmetic lint, and fixing
it inside this PR would mix an unrelated behaviour change into a delivery-
protocol fix.

## Deferred, by decision

Named in the plan and left for separate work:

1. A row returned to `phase=pending` by a retriable failure keeps its nonce, and
   the next same-key attempt overwrites it — a duplicate window on the
   markerless path. Unreachable while a marker exists, because the marker is
   written only with `phase=unconfirmed`, which is resend-blocked.
2. `SCAN_ABSENT` / `SCAN_INDETERMINATE` / `SCAN_AMBIGUOUS` still fall through to
   a normal send, exactly as before this change.
3. No repair machinery for a damaged delivery store; `kill_agent` remains the
   escape from either barrier.
