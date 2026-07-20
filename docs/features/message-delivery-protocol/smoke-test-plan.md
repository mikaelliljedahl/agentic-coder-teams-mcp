# Smoke test plan — lead-orchestrated

How a **lead agent** verifies the message-delivery protocol end to end against
real spawned processes, using only MCP tools plus the `win-agent-teams` CLI.

## Why this exists

The unit suite is green at 924 tests, and that is not sufficient evidence.
Three independent reviews each found live critical defects *behind* a fully
green suite — six in Phase A, four in the final pass. The pattern held every
time: the defects lived in **concurrency, crash windows and error states**,
which unit tests on an injected clock do not reach. The final round only went
red once someone wrote a genuinely threaded test and induced a real `OSError`.

So this plan is deliberately *not* a reimplementation of the unit suite in
process form. It covers what unit tests structurally cannot:

- real backend CLIs that take real wall-clock time to become idle;
- a real busy agent, mid-turn, that must be waited for rather than refused;
- real process death, real PID reuse windows, real transcript flushes arriving
  after a call has already returned;
- several MCP server processes writing the same session directory at once.

**A green smoke run is not proof of correctness.** It is proof that the
protocol's happy paths and its named failure modes behave as documented on this
machine, with these backends. Absence of a failure here is weaker evidence than
a unit test, because timing is uncontrolled.

## Preconditions

- A worktree with the feature branch installed (`uv sync`).
- At least one backend available (`list_backends`); prefer running the matrix
  twice, once per backend, since Claude and Codex differ in prompt transport,
  receipt-record shape, and session-id discovery.
- A **fresh session directory**. A recovered session carries other agents'
  records and will corrupt the parentage assertions.
- Nothing else spawning into the same session — `list_agents` must be empty at
  the start. All agents in a session share one flat `agents.json`.

Server-side constants the plan depends on (read them, do not assume):

| Constant | Value today | Where |
|---|---|---|
| Delivery call budget | `45.0` s | `_DELIVERY_CALL_BUDGET_SECONDS` |
| Follow-up idle threshold | `60.0` s | `_FOLLOW_UP_IDLE_SECONDS` |
| Watcher settle window | `15` s | `WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS` |

The call budget is **one total budget** covering wait, lease acquisition,
resume and confirmation — not a per-step timeout. Several cases below hinge on
work outlasting it, so a task that takes ~2–3× the budget is the useful shape.

## Vocabulary under test

`send_message(text, to, idempotency_key)` classifies the recipient and the
class decides the path:

| Class | Path |
|---|---|
| `child` | guaranteed (Phase B) delivery — resume + confirm |
| `spawner` | inbox + watcher (upstream, unchanged) |
| `sibling` / `unrelated` / `unknown` | refused, never rerouted |

Delivery statuses are exactly three — `delivered`, `failed`, `queued` — with
`pending` / `sent` / `unconfirmed` as **phases beneath `queued`**, never
statuses in their own right. Assert on the pair, and never treat a phase as a
status.

---

## Phase 1 — happy path, downstream

**S1. Guaranteed delivery to an idle child.**
Spawn `worker-a` with a short task. Wait for `check_agent` to report
`state="waiting"`. Then `send_message(to="worker-a", idempotency_key="s1")`.

Expect `status="delivered"`. Then assert the two things that make `delivered`
meaningful rather than decorative:

1. The message is **not** in `worker-a`'s actionable inbox (B4 — a
   guaranteed-path message never enters it, or a polling worker acts on the
   same text twice).
2. `delivery_status(idempotency_key="s1")` returns `delivered` with a nonce.

**S2. `follow_up_agent` and `send_message` are the same path downstream.**
Repeat S1 with `follow_up_agent(name="worker-a", ...)` and a fresh key. The
result shape must match S1's. This is the C3 consolidation: if these diverge,
the lead is back to guessing which tool to use — the original defect.

**S3. Upstream still works.**
Have `worker-a` call `send_message(to="team-lead")`. Assert it lands in the
lead's inbox and `read_messages` returns it. Run `win-agent-teams watch
<session_dir>` **before** the send and confirm it wakes with
`reason="message"`. R3 makes this path load-bearing; it must not have been
disturbed by the downstream rework.

---

## Phase 2 — the original defect

**S4. A busy agent is waited for, not refused.**
This is the bug that started the feature.

Give `worker-b` a task lasting well beyond the 45 s budget. While it is
**provably mid-turn** (`check_agent` → `state="running"`, `stalled=false`),
call `send_message(to="worker-b", idempotency_key="s4")`.

- It must **not** return `agent_busy`. That reason no longer exists.
- It either blocks and returns `delivered` once the agent reaches a resumable
  point, or returns `queued(phase="pending")` with a stated sender obligation.
- Then drive the tail: `deliver_pending(idempotency_key="s4")` must complete
  it. Assert the worker's transcript contains the prompt **exactly once**.

**S5. The prompt actually arrives.**
Not the same claim as S4. The original field report was `success` plus a new
PID with the prompt never arriving — a false receipt. Verify from the
**worker's** side: its transcript contains the text, and its subsequent
behaviour references it. A status field is not evidence that a prompt landed;
that conflation is precisely what R6 exists to prevent.

---

## Phase 3 — direction guard (R2)

**S6. A worker cannot restart its lead.**
Have `worker-a` call `follow_up_agent(name="team-lead", ...)`. Expect refusal.
Then assert the refusal **changed nothing**: the lead's PID is unchanged, no
MCP config was regenerated, no prompt sidecar was written, no lease was taken,
and no generation was bumped. Capture `agents.json` and the session dir before
and after and diff them — a refusal must leave the session byte-identical.

**S7. A worker may follow up its own child.**
Have `worker-a` spawn `worker-a1` and follow it up. Expect success. The guard
must be directional, not a blanket ban on workers.

**S8. Siblings and typos are refused, never rerouted.**
From `worker-a`, `send_message(to="worker-b")` (sibling) and
`send_message(to="worker-typo")` (unknown). Both refuse. **Critically: assert
neither text reached the lead's inbox.** The old rule silently rerouted an
unknown name upstream, so the lead read a typo as a genuine message — the
specific R5 failure.

`send_message(to="team-lead")` must keep working from a worker throughout;
a classifier that calls the spawner a sibling breaks upstream messaging
entirely, and that trap has been hit once already.

---

## Phase 4 — honesty under failure

These matter most and are the hardest to trigger. Prefer running them last,
and treat "could not reproduce" as *not tested*, not as a pass.

**S9. Response loss is recoverable.**
Send with key `s9`, then discard the response without reading it (kill the
lead's client, or simply drop it). Recover with
`delivery_status(idempotency_key="s9")`. This is the entire reason the key is
caller-supplied: a server-generated id would only have arrived in the response
you just lost.

**S10. Idempotency.**
Same key + byte-identical payload → one attempt, not two; assert the worker's
transcript shows the prompt **once**. Same key + any differing field →
`idempotency_conflict` with no mutation. Missing key downstream → validation
error **before** any waiting (it must return fast, not after 45 s).

**S11. Bound expiry with a live child is not terminal.**
Arrange a resume whose confirmation cannot complete inside the budget. Expect
`queued(phase="unconfirmed")` — **never** `failed`. Then let the transcript
flush and call `delivery_status` again: it must reconcile to `delivered`. A
`failed` here would be a false terminal status, contradicted moments later.

**S12. Definite non-delivery is terminal.**
Kill the child before the receipt exists. Only then may the status settle
`failed(reason="not_delivered")`.

**S13. Kill-time reconciliation.**
Send, and kill the target while an attempt is in flight but its receipt has
already been written. The record must settle `delivered`, not `failed` — kill
rescans before concluding. Also assert the **delivery record survives the
kill**, even though kill purges the sender's inbox messages. The two stores
differ deliberately: the inbox is live state, the delivery store is an audit
trail whose value is surviving the target.

**S14. Concurrency — two callers, one key.**
The defect the final review found: two concurrent calls sharing sender+key
resumed the conversation **twice**. Issue two `send_message` calls with the
same key from two threads/processes. Assert exactly **one** resume and the
prompt **once** in the transcript. This is the case a sequential test cannot
reach, and the guard that carries it is the delivery-record claim.

---

## Phase 5 — operator paths

**S15. Kill refuses under a live lease** with `operation_in_progress`, and
proceeds once the holder is dead or its creation token no longer matches.

**S16. `win-agent-teams lease inspect|clear|force`** — force must refuse a
holder that is not overdue (exit 3), and must bump the fence *before*
terminating, so a late finalize by the original holder is rejected.

**S17. `win-agent-teams adopt`** requires the recovery token and the expected
generation; refuses a stale generation; and refuses an **already-parented**
record. Assert no `adopt_agent` tool is reachable over MCP — if one ever
appears, C2's guard is void, since a worker could adopt its lead and then pass
the check.

---

## Reporting

For each case record: expected, observed, and the **evidence** (result payload,
transcript excerpt, `agents.json` diff). A case whose failure mode could not be
triggered is reported as **not tested** — never as passed. Given that a green
suite has hidden critical defects three times on this branch, a smoke report
that says "all green" without naming what it could not provoke is the least
useful possible outcome.

## Known non-goals

- **Spoofing.** `IDENTITY` is self-asserted from an env var by the caller's own
  process. R2 is an accident guard, not a security boundary, and no smoke test
  should imply otherwise.
- **`kill_agent` caller identity** — deliberately unguarded, out of scope.
- **Sibling messaging** — refused by design, not a gap.
- Two known issues are tracked separately and should not be re-reported:
  `test_watch_settle_wakes_persistent_waiting` is flaky on a wall-clock race,
  and a stale inbox cursor can silently consume messages if a crash lands
  between the inbox purge and the cursor delete.
