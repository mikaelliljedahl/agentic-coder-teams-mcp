# Fix proposal: token-burning retry loop after PR #53

Source: user problem report from a live orchestration session (post-#53).
Status: proposal, not yet planned/implemented.

## What the report describes vs. what the code does

The report's core claim — "retry on `agent_busy` is a respawn that destroys a
busy agent's context" — is *partially* out of date after #53, but the loop it
describes is real. Current behavior (`_guaranteed_delivery`,
`server_simple.py:4109`):

1. A busy target is **waited on**, not killed: `_prepare` returns
   `wait_reason="agent_busy"` and the caller loop polls until the 45 s call
   budget (`_DELIVERY_CALL_BUDGET_SECONDS`, line 222) expires, then returns the
   cooperative tail with `SENDER OBLIGATION` text (line 4676).
2. The kill+resume only fires once the target is judged **idle**. Resume goes
   through the backend's session-resume (`--session-id`/`-r`), so
   conversational context is preserved on disk — but every resume re-feeds the
   whole transcript to the API from a fresh process. That is the token burn:
   each retry cycle that reaches the resume step costs one full-context replay
   in the *target*, invisible to the orchestrator.
3. **The idleness judgment is unsafe.** Marker says "waiting" → fine. But when
   there is no marker, the fallback (lines 4327–4332) treats
   `last_activity_at` older than `_FOLLOW_UP_IDLE_SECONDS = 60 s` as idle. An
   agent inside one long tool call or a long thinking stretch writes nothing to
   its transcript for well over 60 s, so a retry landing in that window
   **kills it mid-task**. It resumes, replays context, re-reads files to
   recover in-flight work, goes busy again — the next retry repeats. This is
   the loop.

So the fix is not "stop respawning on busy" (already true); it is (a) never
kill on a heuristic, and (b) stop the API from telling senders to hammer
retries.

## P1 — Close the loop

### P1a. Kill only on authoritative idle evidence

In `_prepare` (`server_simple.py:4285–4334`):

- Keep: marker `waiting` → resume immediately.
- Change: when there is **no** state marker for the target, never take the
  `replace_if_idle` kill path off the `last_activity_at` heuristic. Treat the
  agent as busy and wait (i.e. the heuristic may only classify *busy*, never
  *idle-enough-to-kill*). The `agent_idle_but_alive` refusal path stays for
  `replace_if_idle=False`.
- The transcript-inactivity fallback remains only for backends that genuinely
  cannot write markers — today all three built-ins write them, so in practice
  this is: marker present and `waiting` → deliver; anything else → wait, then
  cooperative tail.

Effect: a busy agent can never be killed by a retry. Worst case a delivery
stays `queued(pending)` until the target parks at its Stop/idle hook — which
is exactly "deliver at the next natural stop".

### P1b. Rewrite the sender-obligation contract to "send once, then poll status"

The current tail text (line 4676) reads as "you must immediately retry".
Replace with an explicit protocol and a pacing hint:

- Add `retry_after_s` (suggest: 60) to `_pending_tail` results.
- Reword `_TAIL_OBLIGATION`: the durable row is safe; the *cheap* next step is
  `delivery_status(idempotency_key)` (read-only, actively reconciles);
  call `deliver_pending()` only when `delivery_status`/`check_agent` shows the
  target `waiting`, or after `retry_after_s`. Never in a tight loop.
- Mirror the same guidance in the `follow_up_agent` / `deliver_pending`
  docstrings (the consuming agent reads only tool descriptions).

### P1c. (Follow-up, larger) true no-respawn delivery for parked agents

When the target's marker is `waiting` and its backend has a wake mechanism
(member-wake / Stop-hook), prefer appending to the inbox + letting the wake
hook feed the message on the *existing* process instead of kill+resume. This
removes even the idle-resume context replay. Scope it as its own feature
(`docs/features/…`); P1a+P1b alone close the destructive loop.

## P2 — Scope the big read payloads

### P2a. `deliver_pending` result

`server_simple.py:5616` returns `delivery_store.list_for_sender(store,
IDENTITY)` — the sender's entire history, unbounded (the observed 143 KB).
Change the result to:

- `results` (the rows actually attempted this call, public view),
- `refusals` (unchanged),
- `still_pending_count` + at most N (e.g. 20) summarized pending rows.

When `idempotency_key` was passed, return only that row.

### P2b. `list_agents`

- Default `full=False` already returns compact rows; additionally default to
  **current-session, non-retired** agents and add `include_dead=False` /
  `all_sessions=False` opt-ins.
- Fix the `status:"running"` + `alive:false` contradiction: derive the
  reported state through `_resolve_agent_state(alive=…)` so a dead process can
  never report `running`. Stale markers must lose to liveness.
- Cap `full=True` per-row transcript fragments (`max_chars`) and total payload.

## P3 — Let a never-sent row be superseded under the same key

`_open_delivery_record` (line 5119) refuses any fingerprint change. Relax it
only for rows that are provably unsent: `status=queued`, `phase=pending`,
`attempts=0`, no nonce, claim not held. For those, atomically (under the store
lock) replace prompt/options/fingerprint, append the old fingerprint to a
`superseded` audit list on the row, and proceed. Any row with a sent/leased
attempt keeps today's hard `idempotency_conflict` — that is the audit-trail
case the refusal exists for. This removes the "stale text or new key +
double-delivery risk" dilemma.

## P4 — Watcher default pattern

`cli.py:35` `_WATCH_DEFAULT_PATTERN = "state-*.json"` wakes every reader on
every agent's transition (N² wakeups). Default should be identity-scoped:
`state-<identity>.json` where identity = `AGENT_NAME` env else `team-lead`
(consistent with the "lead is a role" rule in CLAUDE.md). Keep `--pattern`
for explicit broad watching.

## P5 — Diagnostic traps

- `check_agent` on a pi target returns empty `last_line`: fix the pi branch of
  the output reader (`agent_output.py`) for the `--mode json` event format so
  a healthy pi agent shows its last event instead of looking hung.
- `check_agent` on an unknown name: return
  `reason="unknown_agent"` + "never spawned in this session (check
  session/namespace)" instead of a payload that reads as a dead agent
  (`_empty_agent_check`, line 1641). A false death report triggers respawns of
  healthy agents.

## Dispositioned, not proposed

- **Downstream-only `follow_up_agent`** stays. The direction guard exists
  because resume restarts the target's process; grandparent→grandchild resume
  would let two coordinators race one conversation. Cross-level *information*
  flow already exists via `send_message` upstream + relay. If relay overhead
  becomes the bottleneck, address it in P1c's inbox-based delivery (inbox
  append is direction-agnostic), not by widening resume authority.

## Suggested implementation order

| # | Change | Size | Risk |
|---|--------|------|------|
| 1 | P1a marker-only kill gate | small | low — makes delivery strictly more conservative |
| 2 | P1b tail text + `retry_after_s` + docstrings | small | none |
| 3 | P2a `deliver_pending` payload scoping | small | low (response-shape change; callers per report already can't consume today's shape) |
| 4 | P2b `list_agents` state fix + scoping | medium | low |
| 5 | P4 watcher default | small | low |
| 6 | P5 diagnostics | small–medium | low |
| 7 | P3 supersede-unsent | medium | medium (touches idempotency invariants; needs its own tests) |
| 8 | P1c inbox-based no-respawn delivery | large | own feature + plan review |

Each of 1–7 fits one feature branch (`docs/features/` plan + opposite-family
review per repo workflow); P1a+P1b together are the minimal change that stops
the token-burning loop.
