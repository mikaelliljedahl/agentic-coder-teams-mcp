# Implementation plan: directional, non-dropping message delivery

Requirements: `requirements.md` (same directory). Current behaviour reference:
`docs/reference/agent-messaging-protocol.md`. Reviews: `plan-review-1.md`,
`plan-review-2.md`.

Worktree: `C:\code\github\win-agent-teams-mcp\wt-message-delivery-protocol`,
branch `feature/message-delivery-protocol`, from `main` at `3085295`. All
implementation happens there; the primary worktree is not touched.

Revision 5. Rewritten wholesale at revision 3 (review 2 found stale sections);
revisions 4 and 5 applied review findings plus two contradiction sweeps.

## Scope and order

Whole chain in one branch. Commit boundaries, in order:

0. **Phase 0** — remove dead `busy_hint`. No dependency on anything below.
1. **Phase A (R8, R6)** — resume correctly identifies its target and proves
   delivery.
2. **C1 + C2 (R2)** — persist `spawned_by`, enforce the direction guard.
   Deliberately ahead of Phase B: queue-on-busy must not exist before the
   downstream-only guard, or a worker can queue an upstream resume to its lead.
3. **Phase B (R1, R7, R4)** — bounded in-call delivery on the trustworthy
   primitive, behind the guard.
4. **C3 + C4 (R5, R3)** — send-path consolidation, watcher contract.

## Current behaviour — verified

Every citation was read in this worktree.

**Claude session-id selection has no correlation.** `read_claude_output`
(`agent_output.py:79-118`) globs the project dir; with no stored id it keeps
anything passing `_started_after(..., spawned_at)` (`:103-104`) and takes
`max(candidates, key=mtime)` (`:108`). Nothing ties a transcript to *this* agent.
`_started_after(None, ...)` returns `True` (`:307-331`), so a transcript with no
parseable timestamp is accepted on mtime alone.

**Codex solves only half the problem.** `_correlated_prompt`
(`backends/codex.py:543-556`) appends a name+session-derived token on initial
spawn, and token-bearing candidates are preferred *only before an id is known*
(`agent_output.py:156-169`). A stored Codex id is also an absolute filter
(`:140-147`). So Codex addresses initial same-cwd ambiguity, **not** stale-id
revalidation.

**A stored Claude id is an exclusive filter** (`agent_output.py:97-102`), and
`_read_agent_output` passes a correlation token for Codex but nothing for Claude
(`server_simple.py:922-941`). Combined with `_sync_backend_session_id`
(`:980-987`), which can only persist an id the reader returns, a wrong stored id
can never be corrected: **once wrong, permanently wrong.**

**Resume trusts the process handle.** `_do_follow_up` calls `backend.resume`
(`:1703`), takes `new_pid` from the handle (`:1710`), rewrites the record and
returns success (`:1712-1736`). `BaseBackend.resume` merely builds a command and
spawns (`backends/process_base.py:89-104`). No liveness, exit-code, or transcript
check.

**The original process can survive resume.** Shutdown happens only if ownership
is proven (`:1651-1659`); `owns_process` returns `False` for a tokenless
recovered record (`backends/process_manager.py:253-267`).

**Prompt files collide**, scoped to CLI-sensitive Claude prompts: deterministic
path (`:228-230`), unconditional overwrite (`:1127-1138`).

**Busy detection rides on the same read** (`:1600-1645`), and has a second
defect: `last_message is None` returns `agent_busy` at `:1611-1616`, *before* the
authoritative `waiting` marker is consulted at `:1629-1636`.

**`busy_hint` is dead** (`agent_output.py:34-42`; read only at
`server_simple.py:1638`; never assigned `True` in src or tests), and follow-up
uses the bare `_FOLLOW_UP_IDLE_SECONDS` (`:85,1642`) while `_idle_seconds()` owns
the env override (`:104-112`) — so `WIN_AGENT_TEAMS_IDLE_SECONDS` does not affect
the follow-up gate.

**No persisted spawner.** The record (`:1275-1289`) has no parent field;
`_message_recipient` (`:777-809`) resolves lead aliases from the caller's own
`_AGENT_PARENT_NAME`. Resume regenerates the MCP config and `lead_session_id`
(`:1661-1699`), rewiring the resumed *process's* parent — but there is no
registry parent to overwrite.

**Nothing can execute work from a marker.** Hooks only write marker state
(`hooks.py:61-89`); the watcher reads files, prints one JSON record and exits
(`cli.py:183-220`). Markers are also lossy: hooks can be disabled
(`backends/claude_code.py:243-256`), malformed input writes nothing, and
`SubagentStop` writes `waiting` while the watcher suppresses it
(`hooks.py:19-22`, `cli.py:200-207`).

## Phase 0 — dead-code removal

Remove `busy_hint`: the field, the branch at `server_simple.py:1638-1641`, the
comment at `:1624-1628` describing it as a live heuristic, the ten
`busy_hint=False` test call sites, and row 7 of the refusal table in the protocol
reference. Provably behaviour-preserving (the branch is unreachable repo-wide),
per the boy-scout rule in `CLAUDE.md`.

Deliberately **not** wired up instead: it looks like an intended but incomplete
busy signal, and a real transcript-derived signal is a design decision with its
own tests, not a resurrection of an unused field. After removal the only
non-marker heuristic is the idle timer, which is already the truth.

Landed as its own commit with no dependency on the design below. Review 2
suggested deferring it to avoid line churn; keeping it first is deliberate —
it is three deletions, and doing it after Phase A would mean re-touching the
same function twice.

## Phase A — trustworthy resume

### A1. Server owns final prompt materialization

The backend cannot inject the marker: the server writes the sidecar from the
unmodified prompt *before* `backend.spawn`/`backend.resume`
(`server_simple.py:1127-1138`, `:1240-1267`), and `_prompt_arg` then replaces
argv with a fixed "read this file" instruction
(`backends/claude_code.py:258-271`).

So the server builds the final correlated string and chooses transport. Two
rules, both load-bearing:

1. **Transport is decided from the user prompt alone** (`server_simple.py:92`),
   before any marker is appended.
2. **The marker form differs per transport.** Argv branch: a single-line marker,
   no newline. Sidecar branch: newline-delimited, as Codex does today.

Rule 2 is the fix for review 2's objection that rule 1 alone merely hides a
newline from the sensitivity test rather than making argv safe. A single-line
argv marker introduces no sensitive character at all, so the existing safety rule
is respected rather than bypassed. `_CLAUDE_PROMPT_FILE_CHARS` is left unchanged.

Reader side: `read_claude_output` takes the correlation id and filters on it.
The backend must not mutate a server-owned file.

### A1b. Per-spawn correlation id — full data flow

`codex_correlation_token` derives from agent name + session id
(`agent_output.py:22-31`), but a killed name can be reused once its record is
removed (`server_simple.py:1771-1778`), so one token could identify two
conversations. Replace with a per-spawn random id. The complete flow, which
review 2 correctly said was only a concept:

1. **Generation** — before `backend.spawn`, i.e. before `server_simple.py:1267`,
   since the id must be inside the final initial prompt.
2. **Registry key** — `correlation_id` on the agent record (`:1275-1289`),
   written after a successful spawn. Validity: non-empty string; anything else is
   resolved by A2's metadata gate (absent → `legacy`, present-but-invalid →
   `unverified`).
3. **Preservation** — resume's record update (`:1712-1726`) and every CAS update
   must carry it forward. Losing it silently downgrades the agent to legacy.
4. **Server → Claude** — carried in the final prompt per A1, both transports.
5. **Server → Codex** — `_correlated_prompt` (`backends/codex.py:543-556`) stops
   deriving its own token and consumes the persisted id instead. This is what
   prevents review 2's double-marker risk; Codex must not carry both.
   `SpawnRequest` has no dedicated field, so it travels in `extra`
   (`backends/contracts.py:198-214`).
6. **Read path** — `_read_agent_output` (`:922-941`) loads the persisted id and
   passes it to both readers. Classification follows A2's metadata gate:
   **absent** (old-scheme record) → `legacy`; **present but malformed or
   wrong-type** → `unverified`. Never silently re-derived. The two must not be
   collapsed — absent means "predates correlation", malformed means "corrupt",
   and only the first is a compatibility case.

### A2. Explicit validation ladder

Replace the exclusive filter (`agent_output.py:97-102`) with **sequential gates**.
Review 3 correctly found the previous ladder logically inconsistent — 4a/4b/4c
were nested under a "zero matches plus valid id" premise that none of them
actually satisfies. Each gate below is evaluated in order and returns:

0. **Sidecar-pending gate.** For a Claude sidecar spawn, the binding cannot exist
   until the agent reads the file and its tool result is recorded — argv carries
   only the read instruction (`backends/claude_code.py:258-271`). While the
   record has a sidecar attempt with no receipt yet **and** the child is alive
   **and** the spawn is within the pending window, zero matches is `pending`, a
   call-local result. Pending ends — and scanning re-enters at gate 1 — when the
   receipt appears, the child dies, or the window expires. Outside those
   conditions, zero matches falls through to the count gate normally. `pending`
   is never cached and never persisted.
1. **Metadata gate.** Correlation field absent (record predates correlation) →
   `legacy`. Present but empty, malformed, or wrong type → `unverified`.
2. **Scan gate.** Enumerate candidates and scan for the token. Any incomplete
   scan → `indeterminate`; do **not** compute a match count. The current scanner
   collapses `OSError` and not-found into `False` (`:173-191`) and must be split.
   `indeterminate` is retriable; `unverified` is terminal for the call.
3. **Count gate.** Zero → `unverified`, never max-mtime fallback. Two or more →
   `ambiguous`, no guess.
4. **Session-id gate.** Exactly one match, but no parseable `sessionId` (the
   parser returns `None`, `:286-304`) → `unverified`; there is no id to re-pin
   to. Otherwise bind, re-pinning if it differs from the stored id.

**Two-tier candidate enumeration**, replacing "drop the mtime cutoff entirely".
Review 3 flagged that an all-history scan on every read is a performance risk:
`_read_agent_output` is reached by `check_agent`, both `list_agents` forms and the
`agent_status` fallback, so scanning hundreds of transcripts per agent becomes
quadratic in a long-lived project.

- **Tier 1 — validate the stored binding directly.** Open the stored session's
  transcript by id, with no mtime cutoff. This is the case the cutoff would
  wrongly exclude, and it is a single file open, not a scan.
- **Tier 2 — correction scan**, only when tier 1 is missing or mismatched. Keeps
  the existing time window (`:194-210`) as a first pass, with an all-history
  fallback if the window yields nothing. The token, not mtime, decides identity;
  the window is only an ordering heuristic.
- **Cache** validated bindings. Key: `{backend, normalized cwd, correlation_id,
  backend_session_id, canonical path, file identity, parsed session id,
  grammar_version}`. Invalidate on correlation/session/cwd change, path
  disappearance, file replacement or truncation, parsed-session mismatch, or a
  `grammar_version` bump — so an entry written by older code is never trusted.
  Appends do not invalidate. Only successful bindings are cached: `pending`,
  `unverified`, `ambiguous` and `indeterminate` never are. An in-memory cache
  simply disappears across restart; a persisted one must be revalidated before
  use. `(mtime_ns, size)` (`cli.py:141-147`) is a change detector, not a
  collision-proof identity — use an OS file id where available, otherwise stat
  identity plus a stable header hash.

Note that "open the stored session by id" is **not an existing primitive**:
Claude globs candidates and parses `sessionId` from contents
(`agent_output.py:79-109,286-304`), and Codex scans candidate dirs even with a
stored id (`:121-155`). Tier 1 therefore needs either a persisted validated path
or a small id→path resolver. This is new work, not a reuse.

**Legacy records refuse.** Superseding the previous "proceed, current behaviour":
a legacy stored id may be exactly the wrong pinned id this feature exists to fix,
and A4 would then confirm the nonce in the wrong conversation and report
`delivered` — the original bug with a false receipt. Requirements now state this
explicitly (R8, no compatibility exception). Legacy records refuse follow-up with
an error naming kill-and-respawn as the recovery. Read-only consumers keep
working.

Sidecar timing is handled by gate 0 above, not as a separate rule — an earlier
draft stated it as prose after the count gate, which assigned two different
outcomes to the same observable input.

### A3. Child liveness — early-failure signal only

Re-check `new_pid` after a bounded settle; fail fast on an immediately-dead child
(`resume_not_confirmed`). This catches `claude --resume <bad-id>` exiting within
a second. It is **not** delivery evidence.

### A4. Nonce confirmation (R6)

Liveness, transcript growth and marker transitions are all wrong oracles: markers
are keyed on agent name (`server_simple.py:223-230`) and hooks write only
state/event/ts (`hooks.py:81-89`), so a surviving old process and a resumed one
write the same marker.

Embed a cryptographically random nonce per delivery attempt in the final prompt;
confirm that exact nonce in the transcript whose `backend_session_id` is being
resumed. Confirmation requires **both** child survival (A3) and the nonce record.

Review 2's scanner failure modes, each with its rule:

- **Semantic, not substring.** Match a parsed user-prompt or tool-result record,
  not raw bytes — a nonce echoed in a CLI diagnostic or serialized argv must not
  confirm.
- **Record boundary, not raw EOF.** Capture the offset of the last *complete*
  JSONL record before resume. Starting mid-record can yield an unparsable
  fragment, and the readers skip malformed lines permanently
  (`agent_output.py:269-283`), so a partial record would never be reconsidered
  once its completion arrives.
- **Retain partial bytes between polls** rather than advancing past them.
- **Detect identity/size regression** (rotation, truncation, replacement). On
  rotation, establish continuity by **backend session id plus file identity**,
  using the correlation token as corroboration when present but never as a
  precondition — a successor may legitimately not replay the initial marker, so
  requiring it would falsely fail. More than one candidate successor →
  `ambiguous`, not a guess. This is the single rotation rule; an earlier draft
  also said "exactly one token-validated successor", which contradicted it.
- **Bounded scan expiry is not terminal.** With a live child it becomes
  `queued(phase=unconfirmed)` per B1; only a dead child with no receipt is
  `failed(reason="not_delivered")`. A clean negative must stay distinguishable
  from an exhausted scan so a retry reconciles before re-sending.

**A4 is only as strong as A2.** Finding the nonce in a conversation identified by
a wrong stored id confirms delivery to the wrong target. This is why Phase A
lands as one unit and why `unverified`/`ambiguous` must block delivery rather
than degrade to best effort.

**Named receipt records** — review 3 correctly said "decided in this phase" was
still a TODO, not a specification:

- **Claude**: the `type: "user"` record whose message content carries the nonce.
  For a sidecar spawn this is the tool-result record for the file read, since
  argv carries only the read instruction (`backends/claude_code.py:258-271`).
  Existing parsers only handle assistant messages (`agent_output.py:252-283`), so
  a user/tool-result parser is new work in this phase.
- **Codex**: the rollout record for user input, the same record class
  `_rollout_contains_token` already scans (`agent_output.py:173-191`), tightened
  from substring to a parsed field.

Marker grammar is strict — a fixed delimiter plus the high-entropy id — and
matching is on the parsed payload, not a substring. A test must include user text
that looks like a marker to prove only the full random id matches.

**Late writes and rotation**, refining the rules above:

- **Rotation without replay** is covered by the single continuity rule above.
- **A late flush after the call returns** cannot be solved by retaining bytes
  during the call. Terminal `failed` is therefore only permitted when the child
  is dead before receipt; a bound expiry with a live child is *uncertain*, and
  retry or status reconciliation must rescan the prior attempt's nonce before
  re-sending. See B1.
- **Sidecar cleanup** (A5) must not delete a file a still-running CLI may yet
  read. Cleanup requires confirmed child exit or the age threshold — never
  timeout-failure alone.

### A4b. Per-target operation lease

Releasing the registry lock during confirmation is necessary — `_agents_transaction`
holds a cross-process lock for its whole body (`server_simple.py:349-357`) and
`_do_follow_up` currently keeps it across shutdown and spawn (`:1582-1703`), so a
confirmation poll inside would block every registry reader.

But CAS after the fact is **not sufficient**, as review 2 showed: two callers can
snapshot the same generation, both resume the same conversation, and both deliver
distinct nonces. The losing CAS does not undo an irreversible side effect.

So: **atomically reserve a per-agent operation lease before releasing the lock**,
holding `{generation, operation_id, backend_session_id, holder_pid,
holder_create_token, deadline}`. It does not resume.

**A second valid caller queues behind per-target FIFO — it is not refused.**
"Queues or refuses" was not an implementable outcome rule, and refusing a valid
caller would hand back exactly the dead end R1 forbids. Refusal is reserved for
an invalid caller or request (direction guard, `idempotency_conflict`) or a
genuine no-path condition (R7, or an `unverified`/`ambiguous` binding).

Finalization CASes on generation *and* `operation_id`.

**Crash-atomic storage.** Review 3 found the lease could not live in the registry
as-is: `_save_agents_unlocked` overwrites `agents.json` with `write_text`
(`:328-336`), so a crash mid-write can destroy the registry *and* the lease. The
file lock serializes writers but does not prevent a torn write. Leases go in
their own file written with the temp-file + atomic-replace pattern that cursor
persistence already uses (`messaging.py:29-34`). The status store (B1) does the
same. This also keeps lease churn out of the registry's write path.

**Expiry is not fencing, and neither is a PID.** A holder that is alive but slow
after spawning would otherwise let a second caller observe expiry, fail to find a
not-yet-flushed nonce, and retry into a delivery still in progress. So recovery
checks holder liveness and applies a grace period before reclaiming; wall-clock
expiry alone never justifies a resend.

`holder_pid` alone is insufficient because a dead holder's PID can be reused —
which is exactly why this repo already pairs PIDs with creation tokens
(`backends/process_manager.py:170-184,253-267`). The lease stores
`holder_create_token` alongside `holder_pid` and validates the pair with the same
fail-closed model.

**Kill policy — refuse, with an operator escape.** The previous "cancel or wait"
was an unresolved choice and both branches are unsafe: waiting while holding the
registry lock deadlocks a holder that needs it to finalize, and cancel-and-delete
can orphan an already-spawned resume. So `kill_agent` (`:1782-1797`) refuses with
`operation_in_progress` while a *provably live* lease exists; a lease whose
holder is dead or whose token no longer matches is reconciled automatically, then
the kill proceeds.

Unconditional refusal would make a hung-but-live holder's agent permanently
unkillable, so a CLI-only operator path (session recovery token required) can:
inspect the attempt nonce and resumed-child liveness; clear a dead or
token-mismatched lease; and, for a live but overdue holder, force — which first
bumps a fencing generation so the holder can no longer finalize, then terminates
the owned resumed child if ownership is provable, reconciles the nonce, and only
then permits the kill. Ordinary `kill_agent` never bypasses the lease.

Also: name reuse after removal must not let a stale finalize update the
replacement record — hence `operation_id` in the CAS key, not generation alone.
`check_agent` re-pinning the backend id (`:1540-1548`) while a resume holds an
old snapshot is reconciled by the lease, not last-writer-wins.

### A5. Unique prompt files

Per-call filename carrying the delivery nonce. Cleanup is tied to a confirmed or
failed attempt, plus conservative age-based GC — never "delete this agent's stale
files" during a new call, which would race a concurrent attempt.
`_cleanup_agent_artifacts` (`:1741-1755`) removes only the deterministic path
today and needs the new scheme.

### A6. Consumer decisions for the new outcomes

Review 2 was right that naming outcomes is not specifying behaviour. Decisions:

There are **five** non-success binding outcomes, not four: gate 0's `pending`
joins `unverified`, `ambiguous`, `legacy` and `indeterminate`. `pending` is
reachable by these consumers because a read can land inside a sidecar spawn's
confirmation window, so it needs its own column rather than being treated as an
internal detail.

| Consumer | `pending` | `unverified` | `ambiguous` | `legacy` | `indeterminate` |
|---|---|---|---|---|---|
| `check_agent` (`:1540-1548`) | report `pending`, **do not persist** an id | report state, **do not persist** an id | same | current behaviour, flag in payload | report, no persist |
| `follow_up_agent` (`:1600-1605`) | refuse, **retriable** — the binding may yet appear | refuse — cannot confirm delivery | refuse | **refuse** — see below | refuse, retriable |
| `list_agents` compact/full (`:1904-1913`, `:1949-1966`) | show `pending`, no id | show state, no id | same | show state, id marked unverified | show state |
| `agent_status` no-marker fallback (`:1973-1984`) | `state="unknown"`, **stay cheap** | `state="unknown"`, **stay cheap** — no extra scan | same | current behaviour | `state="unknown"` |

`pending` and `indeterminate` are both retriable and must be distinguishable
from the three terminal outcomes: a caller that retries on `unverified` would
spin forever, and one that gives up on `pending` would fail a spawn that was
about to bind normally.

**`legacy` refuses follow-up.** An earlier draft of this table said "proceed,
current behaviour"; that contradicted A2 and R8 and is superseded. A legacy
stored id may be the wrong pinned id this feature exists to fix, and proceeding
would let A4 confirm the nonce in the wrong conversation and report `delivered`.
The refusal names kill-and-respawn as the recovery. Read-only consumers keep
working, but full `list_agents` must not present a legacy stored id as if
verified — the binding outcome is a separate field, not overloaded onto lifecycle
`state`.

The non-persist rule matters: `check_agent` currently persists any newly surfaced
id, which would let an unverified read poison the record. `agent_status` must
keep its advertised cheap marker/cursor/liveness contract (`:2003-2024`) — "stay
cheap" means no second scan and no all-history scan.

### A7. Fix the pre-marker busy check

Move the `last_message is None` → `agent_busy` branch (`:1611-1616`) below the
`waiting` marker check (`:1629-1636`), so the authoritative signal is not
shadowed by a quiet transcript.

## C1 + C2 — direction guard (before Phase B)

**C1. Persist the spawner.** `spawned_by` on the agent record (`:1275-1289`),
written at spawn from `IDENTITY`, preserved through resume and CAS updates like
`correlation_id`.

Existing records cannot be backfilled and are **not** silently allowed — that
would make the guard ineffective exactly during upgrade. They refuse with
`parent_unknown`.

**Recovery is an operator action, not an agent-callable tool.** The previous
`adopt_agent` MCP tool reintroduced the hole C2 closes: "callable only by a
caller claiming parentage" is tautological, because the operation itself writes
the caller as `spawned_by` and `IDENTITY` is self-asserted (`:49-58`). A confused
worker could adopt its lead and then pass C2 — turning a one-call mistake into a
two-call one, both discoverable. Adoption therefore lives in the `win-agent-teams`
CLI, requires the session recovery token and the expected current record
generation, and records that the parentage was operator-asserted rather than
spawn-derived. A worker with filesystem access can still bypass it, which is an
accepted non-goal — but it must not be a normal agent-reachable escape hatch.

**C2. Enforce downstream-only.** `follow_up_agent` refuses a caller that is not
`spawned_by`, naming the rule and pointing at `send_message`. Refusal must change
nothing — no PID, no MCP config regeneration, no prompt sidecar.

A comment at the check site and a line in the protocol reference must state
plainly that this prevents accidents and is **not** a security boundary:
`IDENTITY` is read from an env var at import (`:49-58`) and the enforcing server
is the caller's own process.

## Phase B — bounded in-call delivery

### B0. Delivery model

Decided after review 2 showed the previous "sender drives it" design did not
satisfy R1: a message could pass its resumable point and deadline while queued
forever, because no process exists to observe the deadline.

**The originating call performs the delivery.** When the target is busy,
`follow_up_agent` does not refuse and does not return immediately — it waits,
bounded, for the target to reach a resumable point, then resumes and confirms
per A4, returning `delivered` or `failed`. Within the bound this is genuine
guaranteed delivery with no queue and no dependency on anyone returning.

If the bound expires, the call returns `queued` plus an explicit sender
obligation to call `deliver_pending`, and the message stays durably queryable.
The tail is cooperative and is stated as such in the return payload and the
protocol reference.

**Response loss must not defeat the honesty property.** Review 3 found three
holes in "the sender always knows which of the two it got":

- **The bound is one total budget** covering wait, lease acquisition, resume and
  confirmation — not a per-step timeout. The server cannot know each client's
  deadline, so the budget is a documented server-side constant that callers can
  read, not an assumption about the client.
- **The message is created durably before waiting**, keyed by a
  **caller-supplied idempotency key**. If the response is lost — client timeout,
  cancellation, or a server crash after creation — the sender recovers the
  outcome with `delivery_status(key)`. Without this, a sender that loses the
  response has no id to ask about, since the id would only have arrived in the
  lost response.
- **Late confirmation is reconciled, not contradicted.** See B1: bound expiry
  with a live child is uncertain, not terminal.

Rejected: a persistent dispatcher. It is the only way to guarantee the tail for a
passive sender, but it adds a daemon with its own lifecycle, locking and restart
recovery to a project built on per-agent MCP servers and a one-shot watcher.
Recorded as a non-goal in `requirements.md` rather than left implicit.

**Opportunistic drain is bounded to an explicit allow-list**: `deliver_pending`
and `follow_up_agent` only. Not `agent_status`, `check_agent` or `list_agents` —
review 2 correctly noted that draining on those would turn an advertised cheap
read (`:2003-2024`) into a slow mutator.

Waiting must not hold the registry lock (A4b) and must not hold the operation
lease longer than its deadline.

### B1. Delivery state machine

Legal states and transitions, with the owner of each:

Public statuses are exactly R4's three. `sent` and `unconfirmed` are **phases
beneath `queued`**, not additional statuses — review 3 flagged that a four-state
machine contradicted R4, and that exposing `sent` invites reading it as
"arrived".

```
queued(phase=pending) ──(lease acquired, resume spawned)──> queued(phase=sent)
queued(phase=sent)    ──(A4 nonce confirmed)─────────────> delivered [terminal]
queued(phase=sent)    ──(child dead before receipt)──────> failed(reason="not_delivered") [terminal]
queued(phase=sent)    ──(bound expired, child alive)─────> queued(phase=unconfirmed)
queued(phase=unconfirmed) ──(nonce found on rescan)──────> delivered [terminal]
queued(phase=unconfirmed) ──(child dead, no nonce)───────> failed(reason="not_delivered") [terminal]
queued(phase=pending) ──(call budget expires, never leased)─> queued(phase=pending) [the cooperative tail]
```

**A never-leased message does not expire into `failed`.** An earlier draft had
`failed(reason="expired")` here, which directly contradicted R1's cooperative
tail: the call budget expiring is exactly the case R1 says returns `queued` with
a sender obligation. There is one timeout — the total call budget — not a second
post-call expiry, so the same instant cannot mean both. A pending message stays
queued and queryable until it is delivered, reconciled, or the session is cleaned
up. If a distinct later expiry is ever wanted it must be introduced in
requirements as its own contract, with a reason it does not re-break the tail.

The `unconfirmed` phase is the fix for review 3's finding that a terminal
`failed` could be contradicted by a transcript flush arriving after the call
returned. **A timeout with a live child never terminates as failed.** It stays
non-terminal until the child is dead and the transcript quiescent, and any retry
must rescan for the prior nonce first. Only definite non-delivery — a dead child
with no receipt record — is terminal.

- **Cross-process atomicity.** Several per-agent MCP servers share the session
  dir. Inbox cursor locking is in-process only; the registry uses a file lock
  (`:339-357`). The status store uses the same file-lock transaction model.
- **Crash recovery**, per window: crash after spawn before `sent` → lease expiry
  reconciles by nonce search; after nonce before `delivered` → nonce search finds
  it and completes; mid-PID-update → CAS on `operation_id` makes the partial
  update detectable.
- **Per-target serialization and FIFO** via the A4b lease. One in-flight delivery
  per target; queued messages delivered in order.
- **Retry** checks whether the prior attempt's nonce already landed before
  re-sending. This, not receiver-side dedupe, prevents a duplicate prompt — the
  recipient is a backend conversation, not a consumer with a dedupe table.
- **Idempotency keys are required, not optional.** Every downstream
  guaranteed-path call supplies one; generating it server-side would recreate the
  lost-response problem. Uniqueness namespace is
  `(session_id, sender_identity, idempotency_key)`, so two senders may reuse the
  same textual key. Same key with a byte-identical recipient/prompt/options →
  return or reconcile the existing attempt, never create a second. Same key with
  any differing field → `idempotency_conflict`, no mutation. Missing, empty,
  malformed or overlong → validation error **before** waiting. Terminal
  tombstones are retained for at least the session retention period; GC'ing them
  earlier would let key reuse duplicate a delivery.
- **Query contract.** `delivery_status(idempotency_key)` returning
  `{message_id, idempotency_key, to, status, phase, reason, attempts, nonce,
  created_at, settled_at}`, in the same sender namespace. A `to` lookup is a
  convenience list only — it cannot serve response-loss recovery when several
  messages target one agent. Survives restart. Kill-time cleanup **reconciles
  before concluding**: an in-flight `sent`/`unconfirmed` attempt may already
  have an unread receipt, so cleanup rescans for the nonce first and records
  `delivered` if it landed; only a genuinely receipt-less attempt becomes
  `failed`. Records are never deleted. Marking a delivered message failed at
  kill time would reintroduce exactly the false-status problem this feature
  exists to remove.
- **`delivery_status` is an active reconciler, not a passive lookup.** When the
  phase is `unconfirmed` it performs a bounded reconciliation scan before
  answering. Without this, response-loss recovery could return `unconfirmed`
  forever even after the nonce landed. The same reconciliation runs in
  `deliver_pending` and in any same-key retry, always before a resend.
- **Reaping `unconfirmed`.** A live child with no receipt legitimately stays
  `queued(phase=unconfirmed)` indefinitely — that follows directly from the
  no-dispatcher non-goal, and it is honest rather than silently expired. Once the
  child is proven dead, one reconciliation runs after a transcript-flush grace;
  a still-absent nonce then becomes `failed(reason="not_delivered")`. The lease
  is not held across `unconfirmed` — it converts to a durable
  unconfirmed-attempt record that keeps serializing that target until
  reconciliation, so neither future delivery nor kill is blocked by a lease that
  nobody is progressing.

### B2. No dead end

`agent_busy` ceases to be a returned refusal for a spawner addressing its own
child; it becomes the wait described in B0.

### B3. `no_delivery_path` (R7)

Enumerated, not deferred:

| Target state | Behaviour |
|---|---|
| alive, busy | wait per B0 |
| alive, waiting/idle | resume and confirm |
| dead, record present, valid backend session | resume and confirm — intentionally supported (`:1767-1778`) |
| dead, no resumable backend session | `no_delivery_path`, state named |
| killed / record removed | `no_delivery_path`, state named |
| `unverified` / `ambiguous` binding (A2) | refuse — cannot prove the target |

### B4. Inbox history must not double-present

`send_message` appends actionable text to the recipient's inbox (`:1324-1333`).
If the guaranteed path also resumes the target with that text while the inbox
retains it, a polling worker can read and act on it a second time.

**Guaranteed-path messages do not enter the actionable inbox at all.** They are
recorded in the delivery/audit store beside it.

The previous design — write to the inbox, then pre-advance the recipient's
cursor — was unsafe, and review 3's counterexample is decisive: the cursor is a
per-sender consumed **count**, not a message-id set (`messaging.py:9-26,64-75`),
and `read_messages` advances to the maximum selected position (`:1417-1475`). So
with sender S's cursor at 0, an unread actionable S-message #1, and an audit
S-message #2 appended, advancing S's cursor to 2 **silently destroys message
#1**. "Never advance past unrelated senders" did not address earlier messages
from the *same* sender.

It also broke the single-writer invariant: cursor locking is deliberately
in-process because only the inbox owner writes its own cursor (`:245-260`), while
this had the sender's server writing the recipient's cursor concurrently with the
recipient's `read_messages`.

A separate audit store avoids both, needs no cursor changes, and is smaller.

## C3 + C4

**C3. Consolidate the send path (R5).** Downstream `send_message` routes through
the B0 path or refuses with a pointer. Requires explicit recipient classification,
which does not exist today — `_message_recipient` reroutes an unknown name to the
lead with a warning (`:777-809`). Behaviour per class: own child → guaranteed
path; own spawner → inbox + watcher (upstream, unchanged); sibling → refuse, do
not reroute (non-goal); root lead → as spawner; unknown/typo → refuse. A typo
must never become a silent upstream message. Result shapes are defined alongside
B1's schema so the two are consistent.

**C4. Watcher contract (R3).** Document and test the existing guarantee rather
than change it: wake-without-consume, cursor clamping, exit-2-does-not-strand. No
change to wake priority or the settle window; the `before = after` output-edge
coupling (`cli.py:280`) stays untouched.

## Files affected

| File | Phase | Change |
|---|---|---|
| `src/claude_teams/agent_output.py` | 0, A | remove `busy_hint`; correlation-id param and validation ladder for `read_claude_output`; split `OSError` from not-found in the token scanner; record-boundary scanning |
| `src/claude_teams/backends/codex.py` | A | `_correlated_prompt` consumes the persisted per-spawn id instead of deriving its own — prevents double markers |
| `src/claude_teams/backends/claude_code.py` | A | none for injection (the server owns the final prompt); only whatever `_prompt_arg` needs to carry the argv marker unchanged |
| `src/claude_teams/backends/contracts.py` | A | correlation id travels in `SpawnRequest.extra` |
| `src/claude_teams/server_simple.py` | 0,A,B,C | `busy_hint` removal; final-prompt materialization; correlation id generation/persistence/propagation; consumer decisions; operation lease; resume confirmation; unique prompt files; `spawned_by` + direction guard; delivery state machine and `delivery_status`; recipient classification |
| `src/claude_teams/messaging.py` | B | delivery status store and audit-record write, both atomic temp+replace. **No cursor changes** — B4 explicitly does not touch cursors |
| `src/claude_teams/cli.py` | A,C | operator recovery only: `adopt_agent` (token + generation gated) and stuck-lease reconciliation, neither MCP-reachable. The `watch` command itself is unchanged — C4 is contract tests |
| `docs/reference/agent-messaging-protocol.md` | 0,A,B,C | refusal table row 7 removal; new outcomes, states, direction rule, and the accident-guard caveat |
| `tests/` | all | below |

## Risks

1. **A2 outcome semantics leak into public APIs.** `unverified` and `ambiguous`
   are new states four consumers must present. A6 fixes the behaviour; the risk
   is that callers (including the orchestration skill) treat them as errors.
   Mitigation: distinguish retriable `indeterminate` from terminal `unverified`.
2. **A4 backend evidence may not exist.** If a backend's transcript cannot
   expose the nonce as a semantic record, R6 needs an acknowledgement path for
   that backend. Surfaced as a design change, not absorbed.
3. **A4b lease is new concurrency machinery** in a codebase whose only existing
   cross-process primitive is the registry file lock. Getting it wrong risks
   deadlock or a stuck agent. Mitigation: creation-token-fenced holder identity,
   deadline plus grace on every lease, reconciliation by nonce search, and the
   operator escape path — refuse-on-kill without it could make an agent
   permanently unkillable.
4. **B0 waiting occupies the caller.** A bounded wait inside an MCP call
   interacts with client-side timeouts. The budget must be below the smallest
   realistic client timeout, and the `queued` tail must be genuinely usable when
   it fires — not a theoretical branch.
5. **Audit store visibility.** R5 is now scoped in requirements to
   inbox-delivered and upstream messages, so the boundary is settled; the risk
   is that implementation quietly re-breaches it. Guaranteed-path messages no
   longer appear in the recipient's inbox, so the recipient cannot re-read them
   via `read_messages`.
   That is intended (it is what removes the double-presentation), but R5's
   "history and re-reading" phrasing must be scoped to inbox-delivered and
   upstream messages so the two do not read as contradictory.
6. **Phase A blast radius.** The read path serves `check_agent`, `agent_status`,
   `list_agents` and follow-up. A6 gives each an explicit decision; the full
   suite is a backstop, not the evidence.
7. **Legacy agents stop being followable.** Agents spawned under the old Codex
   token scheme, and any record without a correlation id, fall to `legacy` in A2
   and now **refuse** follow-up — they must be killed and respawned. Read-only
   tools keep working. This is a deliberate, requirement-level decision (R8, no
   compatibility exception), not an oversight: the alternative lets a nonce be
   confirmed in the wrong conversation and reported as delivered.

## Test cases (red first)

Deterministic filesystem unit tests use the existing JSONL fixtures
(`tests/test_agent_output.py:465-579`). Timing cases take an injected clock and
poll interval — no real sleeps.

**Phase 0**
- Full suite green after removal; no test references `busy_hint`.

**A — binding**
- Two Claude transcripts in one project dir, the foreign one newer → the agent's
  own is selected (fails today; the core bug).
- Wrong stored id whose transcript is still **live and active**, token matching
  another → re-pins.
- Stored id valid → unchanged, no flapping across repeated reads.
- Two token matches, one of them the stored transcript → `ambiguous` (guards the
  ordering fix; the old ladder returned "keep").
- Zero matches, valid correlation id, **outside gate 0's pending conditions**
  (non-sidecar spawn, or child dead, or window expired) → `unverified`, not
  max-mtime.
- Gate 0 entry: sidecar spawn, child alive, inside the window, no receipt yet →
  `pending`, never cached, never persisted.
- Gate 0 exit, each path re-entering the count gate: receipt appears → binds;
  child dies → falls through; window expires → falls through.
- Malformed/empty/wrong-type correlation field → `unverified`, not `legacy`.
- Single match with no parseable `sessionId` → `unverified`.
- Token scan hits `OSError` → `indeterminate`, distinct from `unverified`.
- Legacy record → `legacy`, never re-pinned on mtime, and follow-up **refuses**
  with kill-and-respawn named; read-only tools still work.
- Validated-binding cache: hit is reused; entry is invalidated on each of
  correlation/session/cwd change, path disappearance, file replacement,
  truncation, parsed-session mismatch, and grammar-version bump. An append does
  not invalidate. `pending`/`unverified`/`ambiguous`/`indeterminate` are never
  cached. A cache entry written by an older grammar version is not trusted.
- Transcript with no parseable timestamp → not accepted on mtime alone.
- Eligibility ignores the mtime cutoff: a stored transcript older than the cutoff
  is still revalidated.
- Agent name reused after kill → the per-spawn id keeps the two apart.

**A — transport and correlation flow**
- Plain prompt → argv, with a single-line marker, and the sensitivity test still
  sees only the user prompt.
- Sensitive prompt → sidecar, newline-delimited marker; asserts final file
  contents **and** the argv instruction together.
- A representative Claude transcript fixture proves the marker is visible in
  context, not merely present on disk — for both transports.
- Codex receives exactly **one** marker after the change to `_correlated_prompt`.
- Correlation id survives: spawn → restart → read; and spawn → resume → read.
- **Absent** persisted id after restart → `legacy`, never re-derived.
- **Malformed or wrong-type** persisted id after restart → `unverified`, not
  `legacy`. The two must not be conflated: absent means "predates correlation",
  malformed means "corrupt", and only the first is a compatibility case.

**A — confirmation**
- Child exits immediately → `resume_not_confirmed` via a real poll/exit-code
  transition, not a mocked "confirmed"; record unchanged.
- Must **not** confirm: old surviving process grows the transcript; unrelated
  assistant message; old process writes the shared marker after resume; nonce
  appears only in a CLI diagnostic or serialized argv.
- Pre-resume EOF mid-record → the completed record is still matched when it
  arrives (guards the skip-malformed-permanently trap).
- Rotation/truncation/replacement → detected; continuity established by backend
  session id **plus file identity**, with the token only corroborating when
  present. Includes a successor that does **not** replay the initial marker —
  it must still be followed, not rejected. Two candidate successors →
  `ambiguous`.
- Nonce in the correct transcript → `delivered`.
- Bound expiry with a **live** child → `queued(phase=unconfirmed)`, never
  terminal; a later flush containing the nonce reconciles to `delivered`.
- Dead child with no receipt → `failed(reason="not_delivered")`.
- Confirmation does not hold the registry lock: a concurrent registry read
  succeeds during a confirmation poll.

**A — lease**
- Two concurrent **valid** callers → exactly one resume and one delivered nonce;
  the other **queues** behind per-target FIFO. It is not refused — refusing a
  valid caller would be the dead end R1 forbids. (The pre-lease design allowed
  both to resume.)
- `kill_agent` during a provably live lease → refuses with
  `operation_in_progress`; no orphaned confirmed process.
- `kill_agent` where the lease holder is dead or its creation token no longer
  matches → lease reconciled automatically, kill proceeds.
- CLI force path on a live-but-overdue holder → fencing generation bumped first,
  so a late finalize by the original holder is rejected.
- Name reused after removal → a stale finalize does not update the replacement.
- Expired lease → recovery finds the nonce before retrying; no duplicate prompt.

**A — consumers (A6)**
- Each of `check_agent`, `list_agents` compact/full, `agent_status` no-marker
  fallback, follow-up × each of the **five** outcomes (`pending`, `unverified`,
  `ambiguous`, `legacy`, `indeterminate`). `check_agent` must not persist an
  unverified or pending id; `agent_status` must not add a scan.
- `pending` and `indeterminate` are reported as retriable; the three terminal
  outcomes are not. A consumer must not retry `unverified` nor give up on
  `pending`.

**A7**
- `last_message is None` with a `waiting` marker → not reported busy.

**C1 + C2**
- Worker follow-ups its lead → refused; PID, MCP config and prompt sidecar all
  unchanged.
- Worker follow-ups an agent it spawned → succeeds. Nested parentage and a
  sibling both exercised with distinct caller identities against the flat registry.
- Record without `spawned_by` → `parent_unknown`, not a silent allow.
- `adopt_agent` is **not reachable over MCP** — assert no such tool is
  registered. Invoked via the CLI with the session recovery token and expected
  record generation, it writes the field and records the parentage as
  operator-asserted; with a stale generation it refuses.
- Spoofing is out of scope and documented as such in the test file.

**Phase B**
- Busy target that reaches waiting inside the bound → `delivered` from the
  originating call, no queue, no second call.
- Busy target that never reaches waiting → `queued` with the sender obligation
  stated; status queryable; not reported delivered.
- Call budget expires with no lease ever acquired → stays
  `queued(phase=pending)` — the cooperative tail — **not**
  `failed(reason="expired")`. Only R4's three public statuses are ever exposed;
  `sent`/`unconfirmed`/`pending` are phases beneath `queued`.
- Drain allow-list: `deliver_pending` and `follow_up_agent` drain;
  `agent_status`/`check_agent`/`list_agents` do not and stay cheap.
- Retry after an unconfirmed attempt detects the prior nonce and does not deliver
  twice.
- Crash windows: after spawn before `sent`; after nonce before `delivered`;
  mid-PID-update.
- Cross-process atomicity: two MCP servers mutating the status store.
- `delivery_status` schema, sender isolation, restart persistence, kill-time
  transition to `failed` without deletion.
- Every B3 target-state row → its documented behaviour.
- B4: a guaranteed-path message is resumed into context **and** does not appear
  as new in the recipient's `read_messages`.

**C3 + C4**
- Recipient classification: own child, own spawner, sibling (refused), root lead,
  unknown/typo (refused, not silently upstream).
- AC10 integration: append upstream message → watcher wakes → `read_messages`
  consumes once → next watch does not wake for it → a timeout or restart between
  those steps does not lose it.

**Live-backend acceptance** (not mocks) for AC1–AC3: send during a real tool
call, assert no `read_messages` in the worker transcript, verify the nonce and
the resulting action; one real invalid-session-id resume proving the CLI's fast
exit becomes a failure.

Regression: agents spawned under the old Codex token scheme remain readable by
`check_agent`/`list_agents`/`agent_status`, and **refuse** follow-up with
kill-and-respawn named — per R8's no-compatibility-exception rule. An earlier
draft of this line demanded they "keep working", which contradicted R8.

## Validation

All four gates CI runs (`.github/workflows`), across the whole repository, not
only changed files:

1. `uv run ruff format --check .`
2. `uv run ruff check .`
3. `uv run ty check`
4. `uv run pytest` (coverage `fail_under = 80`)

An earlier draft listed only ruff and pytest, which would have let a red
`ruff format` or `ty` gate through unnoticed.

Any pre-existing red is reported explicitly with file and rule code rather than
scoped out or described as green. Known pre-existing red on `main` at the time of
writing, to be re-checked rather than assumed:

- `ruff format --check` — 3 unformatted test files
  (`test_agent_discovery.py`, `test_agent_output.py`,
  `test_server_simple_guards.py`). Cosmetic and behaviour-preserving, so fixed
  here in its own commit per the boy-scout rule.
- `ty check` — 2 diagnostics for `signal.SIGKILL`, which does not exist on
  Windows. Platform-specific rather than trivially cosmetic, so **reported, not
  silently fixed**; needs a decision on whether the gate is meant to pass on
  Windows at all.

## Rebase note

This branch was re-based onto `main` after `main` advanced 8 commits mid-plan.
Two of those changed facts this plan depends on:

- `29b166f` raised the watcher settle window from 1.5 s to **15.0 s**. The
  protocol reference has been corrected; C4's contract tests must assert against
  the constant, not a literal.
- `c6a022e` added substantial coverage to `agent_output` and `server_simple` —
  the files Phase A rewrites. Phase A's tests must be reconciled with those,
  not written as if they are absent.
