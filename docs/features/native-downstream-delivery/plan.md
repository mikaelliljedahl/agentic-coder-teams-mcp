# Native downstream delivery — plan

Status: **v3**. It revises v2 (976ea09) after round 2 of `plan-review.md`. The
user decided on **one PR** for A–D. Every finding is dispositioned in
`plan-review.md` (§ Disposition for round 1, § Round 2 disposition for round 2).
Branch: `feat/native-downstream-delivery`, from `origin/main` 471a175 (after #70).
Predecessor: `docs/features/native-session-wake/` (PR #70): §9.1, F3, F5.

## 0. Summary

The goal is to make the native session channels the normal way messages reach
live agents.

- **A. Downstream to a Codex child.** `follow_up_agent`, and `send_message` to
  an own child, put the message into the **live** Codex thread with
  `codex queue`. They no longer kill the child and respawn it through
  `backend.resume`.
- **B. Downstream to a Claude child.** The same for a live Claude Code child.
  The lead persists the attempt, then publishes an entry in a **delivery
  mailbox**. The **child's own** MCP server posts it to its own host channel
  (own-child trust).
- **C. F5: a Windows Claude channel.** Overlapped, cancellable named-pipe I/O,
  owner-verified on every post, with bounded parking. **Implemented**
  (71b5e7a, 11addf5, 16fdd4f).
- **D. F3: Codex lead wake.** A Codex lead is woken by `codex queue` on its
  own thread when a child replies.

Resume (kill-and-respawn) stays as the fallback. It is still the path for:

- a dead child, including after a reboot;
- a headless child;
- a child whose native channel cannot be proven;
- a message larger than the native inline limit (§2.4);
- Pi.

**Flags and baselines** (reviews #14, R2-1):

| Setting | Behaviour |
|---|---|
| master `WIN_AGENT_TEAMS_NATIVE_WAKE` off | Byte-identical to `main` for sessions **without native recovery state**: tools, results, env, and no `method` field. The N5 barrier (§2.2.4) stays active for unresolved native rows persisted while the flags were on. |
| master on, `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM` off | #70 behaviour plus C and D. New downstream attempts are always `resume`. |
| master on, downstream on | A and B where eligible |
| `…_CLAUDE=0` / `…_CODEX=0` | Disables that half's *new* use everywhere: wake, B or A, and D. It never lifts N5. |

Exactly-once is defined narrowly. A nonce receipt proves that a message was
*presented* to the child. It does not prove that every action requested in it
ran exactly once. The invariant this feature keeps is **that one message is
never presented twice through two carriers**.

## 1. Current behaviour (facts, cited to 471a175)

**Entry points.** `follow_up_agent` (`server_simple.py:5655`) and
`send_message` to an own child (`:3635-3642`) both call `_guaranteed_send`
(`:5542`), which calls `_guaranteed_delivery`. That runs `_prepare`
(`:4413-4711`, inside `_agents_transaction`) and then `_do_follow_up`
(`:4713-4822`).

**`_prepare`:**

- enforces the gates;
- reconciles prior attempts;
- for a live target, returns `wait_reason` **before** reserving a lease unless
  the state marker says `waiting`. The same happens with
  `replace_if_idle=False` (`:4567-4624`);
- mints `generation`, `operation_id` and `nonce`, reserves the lease, builds
  the request, and snapshots the scanner.

**`_do_follow_up`:**

- persists the row as `sent` (`:4859-4888`) before the transport;
- shuts the child down, gracefully first and forcibly only if needed
  (`:4761-4766`);
- calls `backend.resume` (`:4769`);
- calls `confirm_delivery` (`delivery.py:572-616`);
- `_finalize_follow_up` (`:4244-4388`) fences, writes the new PID and deletes
  the prompt file on receipt.

**Settlement** (R2-13). Two baseline paths turn `absent` into `failed`:

- `_reconcile_delivery_record` (`:2587-2635`): the child must be dead **and**
  past the flush grace;
- `kill_agent`'s `_reconcile_deliveries_for_target` (`:2638-2691`, called at
  `:6116-6123`): it settles `absent` **before** killing, with no grace.

Both are sound only because resume is not a durable carrier.

**Other constraints:**

- `_scan_target` prefers a live registry record with the same name over
  `target_snapshot` (`:2566-2585`), which matters for R2-9.
- `leases.release_lease` removes the holder's waiter (`leases.py:423-435`).
- Guaranteed-path messages never enter `inbox-<child>.jsonl`.
- `agents.json` does not record whether a child is interactive.
- Codex children get no per-agent `CODEX_HOME`.
- The hook marker is `{state, event, ts, gen}`. It is written without
  read-modify-write, and `gen` is random per write (`hooks.py:84-119`).

**Spikes.**

- S-1: a `codex queue` turn into a live, idle TUI child is detected by the
  unchanged scanner, and the queue returns a submission id.
- S-4: the pipe server PID is the nearest `claude.exe`.

## 2. Design

### 2.1 The N5 barrier first, then method selection in two stages

**N5 — the unresolved-native barrier** (R2-1). It is **not** an eligibility
condition.

- **When.** It is evaluated at the start of `_prepare`, before any carrier
  choice, and again under the granted lease. It runs **regardless of flags**.
- **What it looks for.** Every delivery row in the session store with
  `to == target`, from any sender, where:
  - `method ∈ {codex_queue, claude_mailbox}`;
  - `status == queued`;
  - `phase ∈ {sent, unconfirmed}`.

  A row whose queue call, ref write or finalisation is still in flight is
  already `sent`, so it counts.
- **Result.** For a different row it returns
  `queued(pending, reason="prior_native_attempt_unresolved", blocking_key=…)`
  with `sender_obligation`, and **never** falls through to resume. The same
  key reconciles, as it does today.
- **Storage.** N5 reads the delivery store, not the single `pending_delivery`
  field, so it survives record removal and name reuse.

**Stage 1 — provisional eligibility** (in `_prepare`, before the idle/replace
gate). The candidate is `native_candidate ∈ {codex_queue, claude_mailbox,
None}`:

| # | Condition | Codex (A) | Claude (B) |
|---|---|---|---|
| E0 | downstream on, master on, half switch not `0` | `_CODEX` | `_CLAUDE` |
| E1 | alive; binding bound; `backend_session_id` set | ✓ | ✓ |
| E2 | the record has `interactive: true`, written at spawn and resume from `provides_tty(...)`; missing ⇒ false | ✓ | ✓ |
| E3 | channel proof | `verify_codex_thread(record.codex_home, backend_session_id)` | a fresh capability marker bound to this dispatch epoch and backend session (§2.3.3), owner lock held |
| E4 | transport safety | native binary, not the `.cmd` shim | child channel `available` |
| E5 | encoded message ≤ `NATIVE_INLINE_MAX` (§2.4) | ✓ | ✓ |
| E6 | **Codex only:** the target is idle by marker, until N2 passes live (R2-10) | ✓ | — (the poster waits for idle itself) |

- **Candidate found.** A Claude target that is busy or has
  `replace_if_idle=False` goes on to reserve the lease, and the poster enforces
  idleness. For Codex, E6 keeps today's wait loop for busy targets.
- **No candidate.** Today's path runs unchanged.

**Stage 2 — commit under the lease.** First N5 is re-checked, then E0–E6 are
re-evaluated.

- If they no longer hold, the call **gives up the lease together with its FIFO
  position**. This is deliberate and documented, and the order is tested
  (R2-11).
- It then re-enters the loop at the idle/replace gate within the same call
  budget, and never calls `backend.resume` from a native branch.

**Recorded fields.** `_mark_attempt_sent` durably writes:

- `method`;
- the frozen **carrier** (R2-9): `{backend, backend_session_id, codex_home,
  rollout_path | transcript_path, dispatch_epoch, host_pid,
  host_create_token}`;
- `nonce`, `operation_id`, `generation`, the lease identity.

`method` is exposed in `public_view` and in results **only** when the master
flag is on. It is never part of `options`.

### 2.2 A: a Codex child via `codex queue`

**2.2.1 Text.** The message is the follow-up itself with the multi-line nonce
marker appended; S-1 shows it arrives intact. There is **no sidecar** (§2.4).

**2.2.2 Order.**

1. `_mark_attempt_sent`: method, carrier and identity, durable.
2. `codex_queue(...)`: the runner exists (ebe5b4a).
3. On `enqueued`, write `carrier_ref=<submission id>` with a CAS on
   `(sender, key, nonce, operation_id)`.

The CAS never reverts a terminal row and never attaches the ref to another
attempt. If the ref is missing, unparsable or its write fails, the row stays
unresolved and S-5 cancellation is not available for it.

**2.2.3 Queue outcome** (runner `QueueOutcome`):

| Outcome | Row |
|---|---|
| discovery failure, or `unverified_thread`, before the subprocess starts; exec failure (`provably_not_enqueued`) | back to `pending`, then stage-2 fallback (§2.1) |
| `enqueued` | `sent`, then confirm |
| anything else: exit 0 without an id, non-zero exit, timeout, error after spawn | `unconfirmed`, reason `native_unresolved` |

**2.2.4 Settlement of a durable carrier.** For `codex_queue` rows:

- **Scanning.** Rows are always scanned against the **frozen carrier's**
  rollout, never against a same-name successor's transcript (R2-9). The
  current record is used only to judge liveness of a *matching* dispatch
  epoch.
- **No terminal failure from absence.** Child death, kill (both baseline
  paths), record removal and reboot never make an absent nonce terminal.
- **Leaving the unresolved state.** A row stays unresolved until one of:
  - (a) the nonce is found ⇒ `delivered`;
  - (b) authoritative removal via S-5 (`thread/queue/delete` returns
    `deleted:true`, a later rescan of the frozen carrier is `absent`, and a
    `carrier_ref` exists) ⇒ `failed(not_delivered)`. This is gated on S-5;
  - (c) the operator runs `deliveries release-native <key>`, a CLI command
    documented as "may still execute" ⇒ `failed(operator_released)`.
- **`kill_agent`** reports `native_unresolved: [keys]`, leaves those rows
  unresolved, and N5 keeps blocking the name.
- **The N5 smoke** (R2-1). The *independent* host reloading the thread (the
  Codex TUI or Desktop reopening it), or the operator releasing the row. An
  ordinary resume by this server is blocked by N5.

**2.2.5 Finalisation.** The native branch leaves `pid`, `create_token`,
`spawned_at` and `prompt_transport` untouched. On `unconfirmed` it writes
`pending_delivery {…, method, carrier_ref}`.

### 2.3 B: a Claude child via the delivery mailbox and its own channel

**2.3.1 Epochs and identities** (R2-4):

- **Host incarnation.** `(host_pid, host_create_token)` of the child's
  `claude` process.
- **Dispatch epoch.** A new `dispatch_epoch` field on the agent record.
  - It is incremented only by kill, force-clear, resume/respawn, record
    removal and re-spawn under the same name.
  - It is **not** incremented by native finalisation, which bumps
    `generation` as today.
  - Normal native deliveries therefore keep an offered entry eligible, and
    every revoking mutation invalidates it.
- **Operation identity.** `(nonce, operation_id)`.

**2.3.2 Mailbox.**

- **File.** `<session>/delivery-mailbox-<child>.json`: one JSON document with
  strict validation, written atomically under `delivery-mailbox-<child>.lock`.
- **Fail closed.** An unreadable or non-object file means *unknown*, never
  empty. Mailbox absence means *unknown* for any row that references it.
- **Entry fields.** `{operation_id, sender, key, dispatch_epoch,
  backend_session_id, text, state, poster?, ts}`.
- **Transitions.** Each is one locked CAS on `(nonce, state, dispatch_epoch)`
  and returns an authoritative result:

```text
(absent) --publish(row sent & op matches & no tombstone)--> offered
(absent) --revoke(recovery)-->                              revoked   (tombstone; blocks a late publish)
offered  --take(epoch & backend session & host match)-->    taken{poster pid,create token}
taken    --begin(same poster)-->                            posting
posting  --ok-->                                            posted
posting  --write_started=False-->                           failed_before_write   (provably unsent)
posting  --write_started=True failure-->                    uncertain
offered  --retract(lead/kill/force)-->                      retracted             (provably unsent)
```

- **Replacement posters.** A `taken` or `posting` entry whose poster is
  provably gone is **never** replayed by a replacement poster. `taken` becomes
  `uncertain` only if `begin` could have run. Because `begin` is itself a CAS,
  `taken` without `posting` is provably unsent and can be retracted.
- **`retract` / `revoke` result.** Exactly one of `done`, `lost(<state>)` or
  `unknown`.
- **Retention** (R2-2). Entries in **every** state, including `posted`, are
  kept until their delivery row is terminal. Cleanup removes only entries
  whose rows are terminal, and it runs in the order delivery-store read ⇒
  mailbox write. Deleting an entry is never an acknowledgement.

**2.3.3 Lead order and recovery** (R2-2):

1. `_mark_attempt_sent` (durable).
2. `publish` CAS. Under the mailbox lock it verifies that the row is still
   `sent` with this `operation_id`, and that no tombstone exists.
3. Read back the entry.
4. Confirm against the frozen carrier.

**Recovery of a `sent` row with no entry.** This is allowed only when the
row's `active_holder` is `not_ours`, using the existing three-valued ownership
probe, so no live publisher exists. It first writes a `revoked` tombstone
(CAS), then sets the row to `pending`. A late publish by the old holder hits
the tombstone and aborts. Any other case (holder live or indeterminate,
mailbox unknown) keeps the row `sent`.

**Recovery with an entry.** Receipts are inspected first; a found nonce means
`delivered`.

- `offered` ⇒ retract, and `done` ⇒ `pending`.
- `failed_before_write` or `retracted` ⇒ `pending`.
- `taken`, `posting`, `posted` or `uncertain` ⇒ `native_unresolved`, as in
  §2.2.4 (N5 blocks).

**Budget expiry in the live call.** The lead retracts. `done` ⇒ `pending`
(new nonce allowed later); anything else ⇒ `unconfirmed`.

**2.3.4 Child poster (`DeliveryPoster`).** This is a second target kind in
`NativeWakeNotifier`, using the same owner-lock pattern.

- **Capability marker.** `native-delivery-<IDENTITY>.json` holds `{pid,
  create_token, host_pid, host_create_token, backend_session_id,
  dispatch_epoch, channel, heartbeat_ts}`. Eligibility E3 requires all of:
  - a heartbeat no older than `3 × poll`;
  - the owner lock held;
  - `host_*` equal to the child record's `pid` and `create_token` (or its
    resolved host on the tmux/terminal launchers);
  - `backend_session_id` and `dispatch_epoch` equal to the record's.
- **Idle proof** (R2-5). The hook marker gains:
  - `backend_session_id`, taken from the hook payload's `session_id`;
  - `idle_seq`, a monotonic integer incremented on every `waiting` event;
  - `turn_seq`, incremented on `UserPromptSubmit`.

  They are written read-modify-write under `state-<agent>.lock`. The poster
  persists `consumed_idle_seq` in the mailbox document. It posts **only** when
  all of these hold:
  - the marker says `waiting`;
  - `marker.backend_session_id == entry.backend_session_id`;
  - `marker.idle_seq > consumed_idle_seq`.

  At `begin` it sets `consumed_idle_seq = marker.idle_seq`, in the same CAS.
- **What this gives:**
  - A short turn between polls still raises `idle_seq`, so no edge is missed.
  - A restart while already idle posts once for the current `idle_seq`,
    because it is greater than the persisted consumed value.
  - At most one entry is posted per idle sequence. The next one needs a new
    `waiting` event, meaning a turn happened.
  - A `failed_before_write` does **not** consume the sequence, so the poster
    re-arms immediately.
  - A receipt always wins. `posted` without a receipt stays unresolved, is
    never relabelled, and is never reposted. The wording is "no evidence of a
    turn", not "no turn started".
- **Residual race, documented.** A manual keystroke or an ordinary wake notice
  can start a turn between the marker read and the post. There is no atomic
  idle-only enqueue, so the rule is "never *knowingly* mid-turn".
- **Validation.** At take **and** at begin: epoch, backend session and host
  identity. Failure leaves the entry untouched.

**2.3.5 Kill and force.**

- **Lock order.** `agents.lock` ⇒ `delivery-mailbox-<child>.lock` ⇒
  `deliveries.lock`.
- **`kill_agent`:**
  1. Bumps `dispatch_epoch`, so posters stop at take or begin.
  2. Retracts `offered` entries. An `unknown` result keeps the row
     unresolved.
  3. Runs the rescan against the frozen carriers.
  4. Leaves `taken` and later entries unresolved (N5).
- **CLI `force_clear_lease`** bumps `dispatch_epoch` and retracts `offered`
  entries as well.

### 2.4 No native sidecars (R2-3, R2-12)

Native carriers only carry **inline** text. `NATIVE_INLINE_MAX` is **16 KiB of
UTF-8** for the encoded delivered text, marker included. This is far below
Linux `MAX_ARG_STRLEN` (128 KiB per string), below the Windows 32 767-unit
command line after quoting, and conservative for the Claude socket line (S-3
confirms the socket). A larger message fails E5 and uses **resume**, whose
existing sidecar lifecycle is unchanged. No native attempt ever depends on a
file body, so kill and session cleanup cannot strand one.

### 2.5 C: F5, the Windows Claude channel (implemented)

- **Platforms.** Linux and Windows are supported; macOS is
  `unsupported_platform`.
- **H3-W owner check.** It runs on **every** post, on the writing handle,
  before any byte is written, and fails closed. S-4 confirmed it under Claude
  Desktop.
- **Writes.** Overlapped `WriteFile`, bounded by the deadline. On expiry, the
  write is cancelled with `CancelIoEx`, followed by a bounded drain. When the
  completion races the cancel, the completion decides the result.
- **Parking** (R2-7). An undrained write is parked with its storage. While a
  path has a parked write, it takes no new write (`channel_busy`). The process
  parks at most `MAX_PARKED` writes (`parked_cap`). The notifier reaps parked
  writes on every tick.
- **Exception safety** (R2-8). After `WriteFile`, every exit, including
  exceptions, either observes completion or parks. Deadlines that are not
  finite, or are ≤ 0, are refused before open.
- **Outcome.** `write_started` is part of every outcome.

### 2.6 D: F3, Codex lead wake (R2-6)

**Notifier start.** The notifier starts when either the Claude channel is
eligible or `_CODEX` is on. Each channel is gated independently in `tick()`.

**Registration.**

- **Tool.** `set_lead_wake(codex_thread_id, codex_home="")` is registered with
  the master flag. It writes `<session>/lead-wake-<IDENTITY>.json` under a
  lock.
- **Record contents.** `{thread_id, codex_home, generation, host_pid,
  host_create_token, status: provisional|active|cleared, reason}`.
- **Clear.** A clear keeps a tombstone and bumps `generation` (R2-B).

**State keys.** Owner lock, notice, cache, backoff and catch-up state are all
keyed by `(session, identity, generation, host incarnation)`. This prevents
S1→S2 suppression.

**Which registrations may queue:**

- **A human-started lead** (no parent binding) is `active` on a verified
  thread. This is the same trust level as `external_set_wake`, supplied from
  the host's `CODEX_THREAD_ID` through the shell command.
- **A spawned lead** (`enable_spawned_lead_wake=true`) is **`provisional`** and
  **never queued** until the parent-bound `backend_session_id` exists and
  equals the supplied thread. Then it becomes `active`. A mismatch sets it to
  `cleared` with a reason.
- **Proof from S-6.** If S-6 shows that the MCP server sees `CODEX_THREAD_ID`,
  the server can self-register with that value as independent current-host
  proof. It still needs corroboration when bound.

**Registration instruction.** The instruction (the one-line shell command plus
`set_lead_wake`) appears in the spawn prompt, in the **resume** prompt for
nested leads, and in the recovery text of `session_info` and `resume_session`.
A new host incarnation ignores the old registration until it re-registers.

**Delivery.**

- The body-free lead notice ("call read_messages") goes through
  `codex_queue(...)`.
- Target, generation and incarnation are revalidated immediately before each
  queue.
- The external-member notice is unchanged.
- `session_info.native_wake.codex_lead` reports `{status, generation,
  thread_verified}`.

### 2.7 Flag propagation (review #10)

Children receive the effective master flag, the downstream flag and **both
half-switch values**, passed as values and never turned on implicitly:

- **Claude children** receive them through the generated `--mcp-config` env
  and the process env.
- **Codex children and nested leads** receive them through the
  `-c mcp_servers.win-agent-teams.env.*` overrides.

The socket and token scrub is unchanged. S-2 (Linux) verifies propagation.

### 2.8 Tool text

The consuming agent reads only docstrings. With the master flag on, these
tools are decorated:

- **`follow_up_agent`, `send_message`, `delivery_status`:** `method`,
  `native_unresolved`, the N5 barrier and `blocking_key`, "do not resend an
  unresolved native message; it may still run", and the inline limit (larger
  messages resume).
- **`kill_agent`:** `native_unresolved`.
- **`spawn_agent`:** the `interactive`, `codex_home` and `dispatch_epoch`
  record fields.
- **`session_info`:** `codex_lead`.
- **`set_lead_wake`:** new tool.

The flag-off descriptions are golden.

### 2.9 Round-3 amendments (v3.1)

The rules below override the corresponding text in §2.3–§2.4.

**R3-2: markers are bound to an epoch, and `idle_seq` counts transitions.**

- **Epoch in the environment and the marker.** Spawn and resume pass
  `WIN_AGENT_TEAMS_DISPATCH_EPOCH=<record.dispatch_epoch>` in the child's
  environment. Hooks inherit it from the `claude` or `codex` process, and the
  marker records it as `dispatch_epoch`.
- **Old hooks lose.** When a hook's epoch is lower than the prior marker's, it
  drops the write, so a late hook from an old host cannot overwrite a new
  incarnation's marker.
- **A new epoch starts a new namespace.** When the epoch differs from the prior
  marker's, `idle_seq` and `turn_seq` restart from 0.
- **`idle_seq` counts transitions only.** It increases only when a `waiting`
  event follows a non-`waiting` state, or when the epoch is new. A duplicate
  `Stop` within one idle period does not increase it.
- **When the poster may post.** It requires
  `marker.dispatch_epoch == entry.dispatch_epoch`, together with
  `backend_session_id` and `idle_seq` as before.
- **Consumption is per epoch.** It is stored as
  `consumed[epoch] = {seq, nonce}`.
- **Rollback.** `failed_before_write` rolls the consumption back to the previous
  value, with a CAS, only when `consumed[epoch].nonce` is still this entry's
  nonce. An uncertain write keeps it.
- **Locking.** The poster reads the marker without a lock (the marker is
  replaced atomically). `state-<agent>.lock` is taken only by hooks, so it
  never nests inside `agents.lock`.
- **New tests:**
  - a resume within the same backend session, with an old waiting marker;
  - a late hook write from the old host;
  - a duplicate `Stop`;
  - begin → pre-write failure → rollback → retry.

**R3-3: recovery is serialised by the mailbox lock, not by holder liveness.**

- **The mailbox exists before the first attempt.** It is created (an empty,
  valid document) under its lock **before** the first attempt to that child is
  marked `sent`. From then on:
  - a missing mailbox *file* means **unknown**;
  - a valid mailbox with no entry for the nonce means **no entry**.
- **Publishing and revoking are both CAS operations under the mailbox lock.**
  - `publish` requires that there is no entry and no tombstone for the nonce,
    that the row is `sent`, and that `operation_id` matches.
  - `revoke` requires that the row is `sent` and that there is no entry. It
    writes the tombstone.
  - Whichever runs first wins. After a successful `revoke`, the row goes to
    `pending`. A live holder that later tries to publish sees the tombstone
    and returns `queued(pending)`.
- **Safe regardless of the holder.** Correctness does not depend on whether the
  holder is alive, has released its claim, or has no token. The claim is used
  only to avoid disrupting a call in progress: recovery is skipped while an
  `active_holder` claim is held by a *live* call in this process
  (`_ACTIVE_CLAIM_IDS`).
- **New tests:**
  - a completed call in a live server;
  - a removed holder;
  - a failed holder release;
  - a holder without a token;
  - the first publication failing;
  - revoke racing publish in both orders.

**R3-5: the inline limit plus a real command budget.**

- **16 KiB stays** the limit on the delivered text.
- **Codex also checks the full command.** Before launching, it checks the real
  command, not just the text:
  - **Windows:** `len(subprocess.list2cmdline(argv))` must be at most 32 000.
  - **POSIX:** each argument must be under 128 KiB, and argv plus the
    environment must fit within `sysconf(SC_ARG_MAX)` minus a 4 KiB margin.
    If `sysconf` is unavailable, the check assumes 128 KiB.
- **Pre-launch failure.** When the budget check fails, the attempt is marked
  `native_ineligible_this_call`. It then goes through the idle gate as a
  forced resume, so it does not loop back into native.
- **Size limits on resume.**
  - Codex resume passes the prompt through argv, so the same budget applies. A
    message that fits neither path fails before launch with
    `failed(message_too_large)`, which is the same as on `main`.
  - Claude resume keeps its sidecar transport.

## 3. Files affected

- `src/claude_teams/winpipe.py`, `native_wake.py`: done for C and the runner.
  New in `native_wake.py`: `DeliveryPoster`, the Codex lead channel,
  independent gating.
- `src/claude_teams/delivery_mailbox.py` (new): the store, CAS, tombstones and
  validation.
- `src/claude_teams/hooks.py`: marker `backend_session_id`, `idle_seq` and
  `turn_seq` under a lock.
- `src/claude_teams/server_simple.py`:
  - N5;
  - two-stage selection;
  - native dispatch and finalisation;
  - frozen-carrier reconcile, kill and status;
  - `dispatch_epoch`;
  - record fields;
  - flag propagation;
  - `set_lead_wake`, `session_info` and the docstrings.
- `src/claude_teams/delivery.py`: confirmation without new-PID assumptions for
  native methods.
- `src/claude_teams/delivery_store.py`: `method`, `carrier`, the `carrier_ref`
  CAS, `public_view` gating and the N5 query.
- `src/claude_teams/leases.py` and `cli.py`: force-clear bumps the epoch and
  retracts; `deliveries release-native`.
- `src/claude_teams/backends/claude_code.py` and `codex.py`: MCP env
  propagation.
- Docs: the protocol reference, `README.md`, `INSTALL.md` §6a and the skills.

## 4. Tests (red first)

1. **Baselines.**
   - Master off with no native state: golden.
   - Master off **with** a persisted unresolved native row: N5 still blocks.
   - Master on, downstream off.
   - Each half-switch combination.
2. **N5.**
   - It blocks, across senders and phases, including an in-flight `sent` row.
   - Native → disable flags → retry: still blocked.
   - N5 appears between stage 1 and stage 2.
   - Holder crash after enqueue, before ref or finalisation.
   - Same-name replacement.
   - Operator release.
3. **Selection.**
   - Each false E0–E6 selects `resume` (unless N5 blocks).
   - Busy Claude reserves the lease; busy Codex waits (E6).
   - Eligibility lost under the lease: the FIFO position is given up and the
     order is tested.
   - `replace_if_idle=False`.
   - The original budget is exhausted.
4. **A.**
   - Same PID; receipt; the `carrier_ref` CAS (receipt arrives before the ref,
     the ref write fails, an operator action races the ref write).
   - Each outcome in §2.2.3.
   - Encoded-limit cases: quotes, non-BMP, multibyte; >16 KiB ⇒ resume.
   - The shim ⇒ resume.
5. **Frozen carrier.** Kill → same-name successor → a receipt in the old
   thread settles the old row, and the successor transcript is never used.
6. **Mailbox CAS**, with deterministic barriers.
   - Every transition.
   - Retract vs. take, both ways.
   - Publish vs. revoke tombstone.
   - A persistence failure ⇒ `unknown`.
   - A corrupt or absent file ⇒ `unknown`.
   - Retention of `posted` until terminal.
   - Cleanup order.
7. **Recovery.**
   - A live publisher is paused between `sent` and publish while
     `delivery_status` runs: no pending, no double publish.
   - Holder `not_ours`: tombstone, then pending; the late publish aborts.
   - socket write → posted → kill cleanup → delayed receipt: `delivered`, with
     no retry in between.
8. **Poster.**
   - A full turn between ticks (`idle_seq` +2): exactly one post.
   - A restart on the same waiting marker: one post; after that, none.
   - A stale predecessor marker with a different `backend_session_id`: no
     post.
   - `failed_before_write` re-arms.
   - A receipt without an observed running marker settles.
   - `waiting→running` between take and begin.
   - An epoch bump racing take or begin.
   - A replacement poster never replays `posting`.
   - Two consecutive offers.
   - Lead death while the poster survives.
9. **Hook marker.** Monotonic sequences under concurrent emits; payload
   `session_id` persisted.
10. **Transport uncertainty.** POSIX tests (done); `winpipe` suites (done).
11. **D.**
    - S1→S2 with equal generations and a backlog.
    - A provisional wrong thread gets zero queues.
    - Binding arrives after the first ticks.
    - MCP restart under the same host.
    - A resumed nested lead under a new host re-registers.
    - `_CLAUDE=0/_CODEX=1`.
    - A Windows Codex lead.
    - The external-member notice is unchanged.
12. **Flag propagation.** Every combination, on spawn and resume, for both
    backends.
13. **Guaranteed messages never appear in `inbox-<child>.jsonl`.**

## 5. Live verification

**Spikes:**

| Spike | Question | Status |
|---|---|---|
| S-1 | queued-turn shape and receipt | **done** |
| S-4 | pipe owner | **done** |
| S-2 | Claude child MCP env | Linux; or Windows after `claude /login` |
| S-3 | persisted shape of a socket-posted record; size limit | Linux |
| S-5 | `thread/queue/delete` via `codex app-server` against a TUI's queue | Optional; without it, (b) in §2.2.4 is not implemented |
| S-6 | does a Codex-hosted MCP server see `CODEX_THREAD_ID`? | Decides D self-registration |

**Smokes**, on Linux and Windows unless noted. **Merge gates:** N1, N3, N5,
N6, N7, N8.

| Smoke | Check |
|---|---|
| N1 | Codex child, idle, same PID |
| N2 | Codex child, busy. If it passes, E6 is lifted in this PR. If it fails, E6 stays and the result is recorded. |
| N3 | Claude child, idle, in place, bypass mode, no approval prompt (Linux; Windows after S-2) |
| N4 | Claude child, busy ⇒ held until the next `idle_seq` |
| N5 | Dead child ⇒ resume. An unresolved queue item ⇒ the barrier holds; the independent host reload or an operator release clears it |
| N6 | Windows Claude lead woken by a child's `send_message`, CLI and Desktop |
| N7 | Codex lead woken by a child's reply, TUI and Desktop |
| N8 | The baselines in §0 |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Duplicate presentation: durable queue + resume | N5 barrier independent of flags; frozen carrier; no terminal failure from absence; S-5 or operator release |
| Duplicate post after an uncertain write | `write_started`; `uncertain` and `posted` are never retried; tombstones |
| Crash between persist and publish | Publish CAS checks the row; recovery requires the holder `not_ours` and a tombstone |
| A stale poster reaches a new incarnation | `dispatch_epoch` + backend session + host identity at take and begin |
| Missed or mid-turn idle edges | `idle_seq`/`consumed_idle_seq`; the residual race is documented |
| Pipe write hangs or piles up | Overlapped I/O, bounded drain, per-path exclusion, cap, reaper |
| Sidecar body lost | No native sidecars; larger messages resume |
| Long N5 blocks | Visible `blocking_key`, a `kill_agent` report, operator release |

## 7. Non-goals

- Pi native delivery.
- Native delivery as the default.
- Removing `watch` or the `Stop` hook.
- Bodies in wake notices.
- A mid-turn delivery option.
- Native sidecars.
