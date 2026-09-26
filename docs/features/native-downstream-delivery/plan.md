# Native downstream delivery — plan

Status: **v2**. It revises v1 (07eaf67) after `plan-review.md` round 1 and the
Windows spikes in `spikes.md`. Every finding is dispositioned in
`plan-review.md` § Disposition.
Branch: `feat/native-downstream-delivery`, from `origin/main` 471a175 (after #70).
Predecessor: `docs/features/native-session-wake/` (PR #70): §9.1, F3, F5.

## 0. Summary

The goal is to make the native session channels the normal way messages reach
live agents.

- **A. Codex child, downstream.** `follow_up_agent`, and `send_message` to an
  own child, put the message into the **live** Codex thread with
  `codex queue`. They no longer kill the child and respawn it through
  `backend.resume`.
- **B. Claude child, downstream.** Same idea for a live Claude Code child. The
  lead first persists the attempt, then publishes a **delivery mailbox**
  entry. The **child's own** MCP server posts that entry to its own host
  channel, which is the own-child trust path.
- **C. F5: Windows Claude channel.** Overlapped, cancellable named-pipe I/O,
  owner-verified on every post. This is **implemented** (71b5e7a, 11addf5).
- **D. F3: Codex lead wake.** A Codex lead is woken by `codex queue` on its
  own thread when a child replies.

Resume (kill and respawn) stays as the fallback. It is still the path for a
dead child, including after a reboot, and for a headless child, a child whose
native channel cannot be proven, and Pi.

**Flags and baselines** (review #14):

| Setting | Behaviour |
|---|---|
| master `WIN_AGENT_TEAMS_NATIVE_WAKE` off | Byte-identical to `main`: tools, results, env, and no `method` field, even on rows persisted earlier |
| master on, `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM` off | #70 behaviour plus C and D. Downstream delivery is always `resume` |
| master on, downstream on | A and B where eligible |
| `…_CLAUDE=0` / `…_CODEX=0` | Disables that half everywhere: wake, B or A, and D |

Exactly-once is defined narrowly. A nonce receipt proves the message was
*presented* to the child. It does not prove that every requested action ran
exactly once. The invariant this feature keeps is that **one message is never
presented twice through two carriers**.

## 1. Current behaviour (facts, cited to 471a175)

- `follow_up_agent` (`server_simple.py:5655`) and `send_message` to an own
  child (`:3635-3642`) both call `_guaranteed_send` (`:5542`). That calls
  `_guaranteed_delivery`, which runs `_prepare` (`:4413-4711`, under
  `_agents_transaction`) and then `_do_follow_up` (`:4713-4822`).
- **`_prepare`:**
  - Enforces the gates: record, direction (2a/2b), backend `supports_resume`,
    binding (3a), and `backend_session_id`.
  - Reconciles prior attempts.
  - For a live target, returns `wait_reason` **before** reserving a lease
    unless the marker says `waiting`. The same applies to
    `replace_if_idle=False` (`:4567-4624`).
  - Mints `generation`, `operation_id` and `nonce` (`:4628-4630`), reserves
    the lease (`:4636-4649`), builds the request (`:4162-4241`) and snapshots
    the scanner.
- **`_do_follow_up`:**
  1. Persists the row as `sent` (`:4859-4888`) **before** the transport.
  2. Shuts the child down, gracefully first and forcibly only if needed
     (`:4761-4766`).
  3. Calls `backend.resume`, which spawns a new process (`:4769`).
  4. Calls `confirm_delivery` (`delivery.py:572-616`), which scans for the
     nonce in a **user** record (extractors `:162-195`) and probes the new PID.
  5. `_finalize_follow_up` (`:4244-4388`) fences on generation and lease,
     writes the new PID and `pending_delivery`, and deletes the prompt file on
     receipt (`:4361-4362`).
- **Settlement.** Reconciliation (`:2587-2635`) and kill (`:2638-2691`,
  `:6116-6123`) settle `absent` as `failed` once the child is dead and the
  grace has passed. That is sound only because resume is **not** a durable
  carrier (review #1).
- **Guaranteed messages never enter `inbox-<child>.jsonl`** (protocol §2).
- **Delivery rows** are free-form dicts. `public_view` whitelists the fields;
  the fingerprint covers `to`, `prompt` and `options`.
- **Children are interactive TUIs by default** (`process_manager.py:939-962`),
  but `agents.json` does not record that. Headless Codex uses `codex exec`.
- **Spawned Codex children** do not get a per-agent `CODEX_HOME`
  (`codex.py:612-637`).
- **Native wake (#70):**
  - The notifier starts only under the Claude half and platform gate
    (`server_simple.py:7149-7155`).
  - `CodexMemberWake` queues a fixed notice for external members, with
    `CODEX_HOME=<home>` and `cwd=Path.home()` (`native_wake.py:589-638`).
  - A Claude child spawned by a flag-on lead did **not** get the flag in its
    own MCP server, even though the process environment is inherited. The
    explicit MCP config carries only identity variables (`:2704-2723`).
- **Spikes.** S-1 shows that `codex queue` into a live TUI child is detected
  by the **unchanged** scanner and that the queue returns a queued submission
  id. S-4 shows that the pipe server PID equals the nearest `claude.exe`.

## 2. Design

### 2.1 Method selection in two stages (A, B; review #8)

**Stage 1: provisional eligibility.** This runs in `_prepare`, **before** the
idle/replace gate. It computes `native_candidate ∈ {codex_queue,
claude_mailbox, None}` from the following conditions. Evaluating eligibility
this early also avoids a lease or kill.

| # | Condition | Codex (A) | Claude (B) |
|---|---|---|---|
| N0 | downstream on, master on, half not `0` | `_CODEX` | `_CLAUDE` |
| N1 | alive, binding bound, `backend_session_id` set | ✓ | ✓ |
| N2 | record `interactive: true`, written at spawn and resume from `provides_tty(...)`. A missing field means ineligible. | ✓ | ✓ |
| N3 | channel proof | `verify_codex_thread(record.codex_home, backend_session_id)`. `codex_home` is recorded at spawn. | a fresh capability marker **bound to this incarnation** (§2.3), with its owner lock held |
| N4 | transport safety | native binary, never the `.cmd` shim | the child's channel is `available` |
| N5 | no unresolved native attempt to this target from **another** row (§2.2.4) | ✓ | ✓ |

With a candidate, the busy and `replace_if_idle=False` branches **do not
return `wait_reason`**. The call goes on to reserve the lease.

**Stage 2: commit under the lease.** Once the lease is granted, the
conditions are re-evaluated. If they still hold, the method is committed on
the `_FollowUpPlan`.

If they no longer hold, the method becomes `resume`. The call then **re-enters
the existing idle/replace checks** within the same call budget: it releases
the lease and keeps its FIFO ticket. It never calls `backend.resume` from a
native error branch.

The method is written to the row in `_mark_attempt_sent` as
`method ∈ {codex_queue, claude_mailbox, resume}`, together with
`carrier_ref` (§2.2, §2.3). It is exposed in `public_view` and in results
**only** when the master flag is on. It is never part of `options`.

### 2.2 A: Codex child via `codex queue`

**2.2.1 Carrier and text.** The message is the follow-up itself, with the
nonce marker appended (`delivered_prompt(single_line=False)`). S-1 shows this
arrives intact through the native binary, including multiple lines and
metacharacters.

The size budget is computed on the **encoded full command** (review #13):

- POSIX: the UTF-8 bytes of argv, capped at `ARG_MAX/4`.
- Windows: the UTF-16 units of the quoted command line, capped at 30 000.

Above the budget, the message is a **sidecar wrapper line**, as for Claude,
with the marker in the wrapper.

**2.2.2 Order.** The attempt is persisted before anything is queued:

1. `_mark_attempt_sent` durably writes `method`, `nonce`, `operation_id`,
   `generation`, the lease identity and `target_snapshot`.
2. The queue call runs.
3. `carrier_ref = <queued submission id>`, parsed from `Queued message <id>`,
   is written with a second store write.

If step 3's write fails, the row stays at `sent` without a `carrier_ref`.
That is still conservative, because it stays unresolved (§2.2.4).

**2.2.3 Outcome of the queue call** (review #1, #13, §8 Q1):

| Outcome | Classification | Row |
|---|---|---|
| discovery fails, or `unverified_thread`, before any subprocess starts | provably not enqueued | back to `pending`; stage 2 falls back (§2.1) |
| `OSError`/`FileNotFoundError` raised by `Popen` itself, so no process started | provably not enqueued | back to `pending` |
| exit 0 with a parsed submission id | enqueued | `sent`, then confirm |
| exit 0 without a parsable id, **any** non-zero exit, a timeout, or an error after spawn | **uncertain** | `unconfirmed` with `native_unresolved`; never fall back |

**2.2.4 Settlement of a durable carrier** (review #1). A queued Codex turn can
outlive the process, a kill, a record removal and a reboot. For
`method == codex_queue` rows:

- `confirm_delivery` probes the **existing** PID for liveness, but child death
  does **not** make an absent nonce terminal.
- `_reconcile_delivery_record`, `kill_agent` settlement and `delivery_status`
  keep the row `queued(unconfirmed, reason="native_unresolved")` until one of
  two things happens:
  - (a) the nonce is found, and the row becomes `delivered`; or
  - (b) **authoritative removal**: `thread/queue/delete` with
    `{threadId, queuedSubmissionId}` returns `deleted:true`, and a rescan
    after the delete is `absent`. The row becomes `failed(not_delivered)`.
    This needs spike **S-5**. Until S-5 passes, (b) is not implemented and
    rows stay unresolved. The limitation is documented in the tool text.
- **Barrier (N5).** While a target has an unresolved native row, no **other**
  guaranteed delivery to that target is sent. The call returns
  `queued(pending, reason="prior_native_attempt_unresolved")` with
  `sender_obligation`, which reuses the existing agent-level
  `pending_delivery` barrier.
  - The same key reconciles as today.
  - This blocks a "same instruction, new key" duplicate. It also blocks the
    reload-dispatch race: a resume for a different message cannot load the
    thread, and so run a stale queued turn, while the sender believes it
    failed.
  - An operator escape exists: CLI `deliveries release-native <key>` marks
    the row `failed(operator_released)`. It is documented as **"may still
    execute"**.
- `kill_agent` still succeeds. It keeps unresolved native rows
  **unresolved** (not failed), keeps their `target_snapshot`, and reports
  them in its result as `native_unresolved: [keys]`.
  - A same-name successor inherits the N5 barrier: the barrier is keyed on
    `(session, name)` and is **not** cleared by removal.
  - Removal is refused only for the lease reasons that already exist.

**2.2.5 Finalisation.** A native branch of `_finalize_follow_up` leaves `pid`,
`create_token`, `spawned_at` and `prompt_transport` untouched. On
`unconfirmed` it writes `pending_delivery {…, method, carrier_ref}`.

**2.2.6 Runner.** The subprocess part of `CodexMemberWake._queue` is
extracted into `codex_queue(binary, thread_id, home, message, timeout) ->
QueueOutcome{started, exit, submission_id, timed_out}`. It keeps `CODEX_HOME`
set to the home, `cwd=Path.home()`, the same environment scrub and a real
timeout. `CodexMemberWake` keeps its own notice text (review #9).

### 2.3 B: Claude child via delivery mailbox and own channel

**2.3.1 Mailbox.** The mailbox is `<session>/delivery-mailbox-<child>.json`:
one JSON document, validated strictly, written atomically under
`delivery-mailbox-<child>.lock` (§8 Q2). Reads and writes **fail closed**, in
the style of `delivery_store`: an unreadable or non-object file means
"unknown", never "empty".

Each entry, keyed by nonce, carries:

- `operation_id`, `generation`, `sender`
- `target {name, backend_session_id, host_pid, host_create_token}` (host
  incarnation, review #6)
- `text`, or `prompt_file`
- `state`, `ts`

State transitions, each **one locked compare-and-swap (CAS) that returns an
authoritative result** (review #5):

```text
offered --take(host incarnation matches)--> taken
taken   --begin-->                          posting
posting --PostResult ok-->                  posted
posting --write_started=False failure-->    failed_before_write   (provably unsent)
posting --write_started=True failure-->     uncertain             (never retried)
offered --retract(lead)-->                  retracted             (provably unsent)
```

`retract` returns exactly one of `retracted`, `lost_to_take` (the current
state), or `unknown` (lock, IO or corruption). Only a durably written
`retracted` or `failed_before_write` permits `pending` and a new nonce.

**2.3.2 Lead order** (review #3):

1. `_mark_attempt_sent`, durably: method, nonce, operation, generation,
   lease, snapshot.
2. Publish `offered` with an atomic replace, then read it back.
3. Confirm on the bound transcript, probing the existing PID.

**Crash recovery:** a `sent` row with no entry for its nonce is **provably
unsent**, because the poster only posts published entries. So an `absent`
entry means the row goes back to `pending`. A mailbox that cannot be read
means "unknown", and the row stays `sent`. On budget expiry, the lead calls
`retract`, and the table above applies.

**2.3.3 Child poster.** The child's MCP server runs a `DeliveryPoster`, which
is a second target kind in `NativeWakeNotifier` that shares its owner-lock
pattern.

- **Capability marker.** `native-delivery-<IDENTITY>.json` records `{pid,
  create_token, host_pid, host_create_token, backend_session_id, channel,
  heartbeat_ts}`. It is refreshed every poll and removed on shutdown.
  Eligibility (N3) requires all of the following:
  - The heartbeat is at most `3 × poll` old.
  - The owner lock is held.
  - `host_pid` and `host_create_token` equal the **child record's**
    `pid`/`create_token`, or its resolved host PID on the tmux/terminal
    launchers.
  - `backend_session_id` equals the record's.
- **Take, and again immediately before the post.** The poster checks:
  - Its own host PID and create token equal `entry.target`.
  - The record's current generation is at least `entry.generation`.
  - The session is active.

  If any check fails, it leaves the entry alone (review #6).
- **Idle rule** (review #7, §8 Q3):
  - The poster requires an **affirmative** `waiting` marker for this
    incarnation. A missing or stale marker means no post.
  - It posts at most **one** entry per `running→waiting` edge. After a post,
    it re-arms only when it sees a `running` marker newer than the post and
    then a later `waiting` marker.
  - If no `running` marker appears within `RENOTIFY` (300 s), the entry stays
    `uncertain` (accepted but no turn started). It is never reposted, and the
    next entry waits.
  - **Residual race, documented:** a manual keystroke or an ordinary wake
    notice can start a turn between the marker read and the post. The host
    offers no atomic idle-only enqueue, so the plan does not promise that a
    post can never land mid-turn. It promises only that one is never
    *knowingly* posted mid-turn.
- **Text.** Up to 32 KiB inline. Longer text goes through the sidecar wrapper
  line with the marker (§2.4).
- **Receipt.** S-3 (Linux) must confirm that the persisted record is
  `type:"user"` with string or text content.

**2.3.4 Kill and force** (review #6):

- **Lock order:** `agents.lock`, then `delivery-mailbox-<child>.lock`, then
  `deliveries.lock`.
- `kill_agent` first retracts every `offered` entry for the target (CAS). It
  then runs the existing rescan. `taken`, `posting` and `uncertain` entries
  make their rows `native_unresolved`, as in §2.2.4.
- CLI `force_clear_lease` also retracts `offered` entries for that target.
- Cleanup removes the mailbox only when it holds no `taken`, `posting` or
  `uncertain` entries.

### 2.4 Prompt sidecars for native wrappers (review #4)

A native wrapper's marker proves that the wrapper was presented, not that the
file was read. So native sidecars are **pinned**:

- They are not deleted on receipt, by finalisation or by reconciliation.
- They are excluded from age GC.
- They are released only by `kill_agent` cleanup of the target or by session
  cleanup.

For wrappers, `delivered` means "presented, body available at `prompt_file`",
and the tool text says so. Inline delivery (under the budget) is preferred,
which avoids most sidecars.

### 2.5 C: F5, Windows Claude channel (implemented)

- `claude_platform_supported()` is true on Linux and Windows. macOS stays
  `unsupported_platform`.
- **H3-W.** The path must be `\\.\pipe\…` and exist. On **every** post,
  `GetNamedPipeServerProcessId` is called on the **same handle** that will be
  written, before any byte is written. It must equal the nearest `claude`
  host PID. A query failure or a mismatch means nothing is written
  (review #12). S-4 confirmed that the pipe server is the CLI `claude.exe`
  under Claude Desktop.
- **Transport:**
  - Overlapped `WriteFile` with an event.
  - The wait is bounded by the deadline. On expiry, `CancelIoEx` is followed
    by a **bounded** drain (`CANCEL_GRACE_MS`).
  - If cancellation lost to a completed write, the result is read from the
    completion.
  - An undrained operation is **parked**: its handle, event, OVERLAPPED
    structure and buffer are kept alive and reaped later (review #11).
  - `WaitNamedPipeW` is never called with 0 remaining.
- **Outcome.** `PipeResult` and `PostResult` carry `write_started`. Every
  failure after `WriteFile` is uncertain.
- **Tests** (done): a real pipe server (happy path, stalled reader, busy pipe,
  PID mismatch, missing pipe) plus a scripted kernel32 (immediate completion,
  cancel lost to completion, cancelled write, undrained cancel parked and
  reaped, synchronous refusal, owner-query failure, short write).

### 2.6 D: F3, Codex lead wake (review #9, §8 Q4)

- **Notifier start.** `main()` starts the notifier when **either** the Claude
  channel is eligible (`_CLAUDE` on and platform supported) **or** a Codex
  lead registration is possible (`_CODEX` on). Each channel is gated
  independently inside `tick()`. An unavailable Claude channel no longer
  drains everything; it only disables the Claude targets.
- **Registration.** `set_lead_wake(codex_thread_id, codex_home="")` is
  registered with the master flag and writes
  `<session>/lead-wake-<IDENTITY>.json` under a lock. It carries forward the
  predecessor's rules:
  - **R2-B:** clearing keeps a tombstone and bumps a monotonic `generation`.
  - **R3-A:** notice, backoff and verification caches are keyed by
    `(identity, generation)`.
  - **R3-C:** the target and generation are revalidated immediately before
    each queue call. Catch-up happens only on transitions.
- **Proof of the thread (Q4).** The tool is called by the Codex lead with the
  output of the same one-line shell command that the join prompt uses, which
  prints its host-provided `CODEX_THREAD_ID` and `CODEX_HOME`. For a
  **spawned** Codex lead (`enable_spawned_lead_wake=true`, an existing
  parameter), the spawn prompt instructs exactly this call. The server then
  also requires the supplied thread to equal the parent-bound
  `backend_session_id` once binding exists. Before binding, it accepts and
  re-checks at every tick; a later mismatch clears the registration with a
  reason.
- **Incarnation.** A registration is bound to the registering MCP server's
  host PID and create token. A different host incarnation, for example after
  a resume, ignores it until the new host registers again.
- **Delivery.** The lead notice goes through `codex_queue(...)` with the lead
  text ("call read_messages"). It is body-free and shares the coalesce,
  renotify, backoff and owner-lock machinery. The inbox stays authoritative,
  and `watch` stays valid.
- `session_info.native_wake` gains `codex_lead: {registered, generation,
  thread_verified}`.

### 2.7 Flag propagation (review #10)

After S-2, propagate the effective master flag, the downstream flag and
**both half switches** as values. They are never turned on implicitly.

- **Claude children:** through the generated `--mcp-config` env
  (`server_simple.py:2704-2723`) and the process env.
- **Codex children and nested leads:** through the `-c
  mcp_servers.win-agent-teams.env.*` overrides the Codex backend already
  builds.

The socket and token scrub (`process_base.py:104-108`) is unchanged.

### 2.8 Tool text

The consuming agent reads only docstrings. With the master flag on, these get
decorated:

- `follow_up_agent`, `send_message`, `delivery_status`: `method`,
  `native_unresolved`, the barrier, the wrapper meaning of `delivered`, and
  "do not resend an unresolved native message".
- `kill_agent`: `native_unresolved`.
- `spawn_agent`: the `interactive` and `codex_home` record fields.
- `session_info`: `codex_lead`.
- `set_lead_wake`: new.

Flag-off descriptions stay identical to the golden files.

## 3. Files affected

- `src/claude_teams/winpipe.py`: done (C).
- `src/claude_teams/native_wake.py`: done for C. New: `codex_queue` runner,
  `DeliveryPoster`, the Codex lead channel, and independent channel gating.
- `src/claude_teams/delivery_mailbox.py`: new. Store, CAS and validation.
- `src/claude_teams/server_simple.py`:
  - two-stage selection in `_prepare`
  - native dispatch and finalisation
  - carrier-aware reconcile, kill and status
  - the N5 barrier
  - record fields at spawn and resume
  - flag propagation, `set_lead_wake`, `session_info` and docstrings
- `src/claude_teams/delivery.py`: confirmation without new-PID assumptions
  for native methods.
- `src/claude_teams/delivery_store.py`: `method`, `carrier_ref`,
  `public_view` gating.
- `src/claude_teams/leases.py`, `cli.py`: retract offers in
  `force_clear_lease`, and `deliveries release-native`.
- `src/claude_teams/backends/claude_code.py`, `codex.py`: MCP env propagation.
- Docs: the protocol reference, `README.md`, `INSTALL.md` §6a, and the skills.

## 4. Tests (red first)

1. **Baselines:**
   - master off: golden-identical, with no `method` even for persisted native
     rows.
   - master on and downstream off: always `resume`.
   - each `_CLAUDE=0` / `_CODEX=0` combination.
2. **Two-stage selection:**
   - each of N0–N5 false means `resume`.
   - busy, native succeeds without waiting.
   - busy, native fails before enqueue, target still busy: waits, no kill.
   - eligibility lost while waiting for the lease.
   - `replace_if_idle=False` with native success and with native failure.
   - original budget exhausted.
3. **A:**
   - same PID, no `backend.resume`, receipt, `carrier_ref` stored.
   - each outcome in the §2.2.3 table.
   - enqueue, then non-zero exit, stays unresolved.
   - encoded-budget cases: quote-heavy, non-BMP, long path, multibyte.
   - shim means resume.
4. **Durable carrier:**
   - enqueue, timeout, child death, negative scan: stays unresolved, never
     `failed`.
   - new-key retry: barrier.
   - kill, then same-name successor: barrier inherited.
   - thread reload dispatches the old item: `delivered`.
   - operator release.
5. **Mailbox CAS** (deterministic barriers):
   - retract wins, take wins.
   - retract persistence failure means unknown.
   - corrupt or non-object file fails closed.
   - `posting → uncertain` is never retaken after a poster crash.
6. **Lead order:** a crash or failed write at every boundary, including
   "publish succeeded, row write failed" and "sent without entry means
   pending".
7. **Poster:**
   - affirmative idle only; a missing or stale marker means no post.
   - one post per edge, with re-arm.
   - `waiting → running` between take and post.
   - accepted but no turn started means `uncertain`.
   - incarnation mismatch at take and at post.
   - lead death while the child poster survives.
   - force-clear racing take or post.
   - MCP restart while waiting.
   - two consecutive offers.
8. **Transport uncertainty:**
   - POSIX: full send then shutdown failure; connect failure.
   - Windows: the `winpipe` suite (done).
9. **Sidecars:** wrapper receipt, then immediate reconcile, then delayed file
   read. The file survives reconcile, GC and process death, and is released
   on kill cleanup.
10. **D:**
    - `_CLAUDE=0/_CODEX=1`, Windows Codex lead.
    - clear then set, backoff then replacement.
    - undrained A→B.
    - delayed binding, then a mismatch clears.
    - restart backlog.
    - session switch during a queue call.
    - the external-member notice text is unchanged.
11. **Flag propagation:** spawn and resume env for Claude and Codex, every
    combination.
12. **Guaranteed messages never appear in `inbox-<child>.jsonl`.**

## 5. Live verification

**Spikes:**

| Spike | Question | Status |
|---|---|---|
| S-1 | Queued-turn shape and receipt | **done** (`spikes.md`) |
| S-2 | Claude child MCP env | blocked on Windows (CLI not logged in); run on Linux |
| S-3 | Socket-posted record shape | Linux |
| S-4 | Pipe owner | **done** |
| S-5 | `codex app-server` `thread/queue/delete` against a TUI's queue | new: can the app-server reach that queue, and does `deleted:true` guarantee no later dispatch? Decides §2.2.4(b) |
| S-6 | Does a Codex-hosted MCP server see `CODEX_THREAD_ID`? | decides whether D can self-register without a tool call |

**Smokes (Linux and Windows unless noted):**

| Smoke | What it shows |
|---|---|
| N1 | Codex child, idle, delivered in place with the same PID |
| N2 | Codex child, busy: dispatched after the turn (closes V2) |
| N3 | Claude child, idle, in place, bypass mode, no approval prompt (Linux; Windows after S-2) |
| N4 | Claude child, busy: held until `waiting` |
| N5 | Dead child: resume fallback. Dead child with an unresolved queued item: barrier, then a resume dispatches the old item |
| N6 | Windows Claude lead woken by a child's `send_message`, CLI and Desktop |
| N7 | Codex lead woken by a child's reply, TUI and Desktop |
| N8 | Baselines from §0 |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Duplicate presentation through the durable queue plus a resume | Carrier-aware settlement: unresolved, never failed. N5 barrier. Authoritative removal only via S-5 |
| Duplicate post after an uncertain write | `write_started`; `uncertain` is never retried |
| Crash between persist and publish | Persist first; "sent without entry" means provably unsent |
| Stale poster posting into a new incarnation | Incarnation-bound entries and capability, checked at take and at post |
| Mid-turn post | Affirmative idle, edge rule, documented residual race |
| Pipe write hang | Overlapped I/O, bounded drain, parking |
| Sidecar deleted too early | Pinned until kill or session cleanup |
| An unresolved row blocks a target for a long time | Visible reason, `kill_agent` report, operator release (documented "may still execute") |

## 7. Non-goals

- Pi native delivery.
- Making native delivery the default. That is a separate PR once N1–N8 are
  green on both platforms.
- Removing `watch` or the `Stop` hook.
- Bodies in *wake* notices. D stays body-free; A and B carry bodies because
  they are deliveries with nonce receipts.
- A mid-turn delivery option (§8 Q3).
