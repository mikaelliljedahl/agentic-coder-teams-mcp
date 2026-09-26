# Native downstream delivery — plan

Status: draft for independent review (opposite model family).
Branch: `feat/native-downstream-delivery` (from `origin/main` at 471a175, after #70).
Predecessor: `docs/features/native-session-wake/` (PR #70), §9.1 and follow-ups F3/F5.

## 0. Summary

Make the native session channels the normal way messages reach live agents:

- **A. Downstream, Codex child** — `follow_up_agent` and `send_message` to an own
  child deliver *into the live Codex thread* with `codex queue`, instead of
  kill-and-respawn via `backend.resume`.
- **B. Downstream, Claude child** — the same, for a live Claude Code child: the
  lead writes an exactly-once **delivery mailbox** entry, and the **child's own**
  MCP server posts it to its own Claude inbox socket (own-child trust).
- **C. F5 — Windows Claude channel** — overlapped, cancellable named-pipe I/O
  so Claude hosts on Windows get the same channel Linux has (lead wake *and* B).
- **D. F3 — Codex lead wake** — a Codex lead (human-started or spawned) is woken
  by `codex queue` on its own thread when a child replies with `send_message`.

Resume (kill-and-respawn) stays as the fallback, and is still the path for a
dead child (e.g. after a reboot), a headless child, a child whose native
channel cannot be proven, and Pi.

Everything new is **opt-in**. A and B sit behind a new sibling flag
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`, which requires the master flag
`WIN_AGENT_TEAMS_NATIVE_WAKE=1`. C and D are wake channels and follow the
master flag and its existing half switches (`_CLAUDE`, `_CODEX`). With the
flags off, behaviour is byte-identical to `main`.

## 1. Current behaviour (facts, cited to 471a175)

- `follow_up_agent` (`server_simple.py:5655`) and `send_message` to an own
  child (`:3635-3642`) both call `_guaranteed_send` (`:5542`) →
  `_guaranteed_delivery` → `_prepare` (`:4413-4711`, under
  `_agents_transaction`) + `_do_follow_up` (`:4713-4822`).
- `_prepare` enforces the gates (record, direction 2a/2b, backend
  `supports_resume`, binding 3a, `backend_session_id`), reconciles prior
  attempts, and for a **live** target waits until the state marker says
  `waiting` (`:4567-4624`); a busy target yields `wait_reason="agent_busy"`
  until the 45 s call budget (`_DELIVERY_CALL_BUDGET_SECONDS`, `:223`) runs out.
  It mints `generation`, `operation_id`, `nonce` (`:4628-4630`), reserves the
  operation lease (`:4636-4649`), builds the resume request with the nonce
  marker (`_build_resume_request`, `:4162-4241`) and snapshots the receipt
  scanner.
- `_do_follow_up` marks the row `sent` (`_mark_attempt_sent`, `:4859-4888`),
  **kills the live child** (`:4761-4766`), calls `backend.resume` (`:4769`),
  which spawns a **new process**, then `confirm_delivery` waits for the nonce in
  a **user** transcript record of the bound session (`delivery.py:162-195`,
  strict regex `:58-61`), probing the new PID for liveness. `_finalize_follow_up`
  (`:4244-4388`) writes the new PID / create token / `pending_delivery`.
- Guaranteed-path messages **never** enter `inbox-<child>.jsonl`
  (protocol doc §2 `send_message`), because of double execution and the
  per-sender count-cursor hazard.
- Delivery rows (`delivery_store.py`) are free-form dicts; `public_view`
  whitelists fields; the fingerprint covers `to`, `prompt`, `options`
  (`options` = `replace_if_idle` only).
- Children are interactive TUIs on Linux (tmux/terminal/herdr) and on Windows
  unless `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE` is false (`process_manager.py:
  939-962`); headless Codex uses `codex exec` and exits after the turn. **Nothing
  in `agents.json` records whether a child is interactive.**
- Spawned Codex children get no per-agent `CODEX_HOME` (`codex.py:612-637`);
  rollout discovery assumes `~/.codex` (`agent_output.py:495`).
- Native wake (#70): `native_wake.py` — Claude channel is Linux-only
  (`claude_platform_supported`, H1b); the notifier watches the lead's own inbox
  and external members; `CodexMemberWake` runs
  `codex queue --thread <id> --message <fixed notice>` for external Codex
  members only. The socket wire is `{"type":"auth",...}` +
  `{"type":"user","message":{"role":"user","content":text}}`. A post only counts
  as own-child when it comes from the host's direct child (the MCP server);
  a lead→child post is a peer message and can be held for approval.
- Live observation (#70 Linux smoke): a Claude child spawned by a flag-on lead
  did **not** end up with the flag in its own MCP server.

## 2. Design

### 2.1 Per-attempt method selection (A + B)

Selection happens at the end of `_prepare`, **after the lease is granted**,
while the registry lock is held and `alive`, marker state, binding and
`backend_session_id` are known. The chosen method is a new `_FollowUpPlan`
field and is written to the delivery row in `_mark_attempt_sent` as
`method ∈ {"codex_queue", "claude_mailbox", "resume"}` (added to `public_view`
and to the result as `method`). It is **not** part of `options`, so it never
affects the idempotency fingerprint.

`native` is chosen only when **all** hold; otherwise `resume` exactly as today:

| # | Condition | Codex (A) | Claude (B) |
|---|---|---|---|
| N0 | `NATIVE_DOWNSTREAM=1`, master on, half switch not `0` | `_CODEX` | `_CLAUDE` |
| N1 | target alive (`_agent_alive`) and binding bound, `backend_session_id` set | ✓ | ✓ |
| N2 | child spawned **interactive** — new record field `interactive: true` written at spawn/resume from `provides_tty(...)`; absent (old records) ⇒ resume | ✓ | ✓ |
| N3 | channel proof | `verify_codex_thread(codex_home, backend_session_id)` passes; `codex_home` recorded at spawn (server's effective `CODEX_HOME` or `~/.codex`) | fresh **poster capability** marker from the child's own server + its owner lock is held (§2.3) |
| N4 | transport safety | native `codex.exe`/POSIX binary (never the `.cmd` shim, `_launches_via_cmd_shim`) | channel `available` in the child (reported by the capability marker) |

Busy targets: with a native method the lead **does not wait for idle**. Codex
queues the turn and dispatches it after the current turn (V2, must be verified,
§5 N3); Claude's poster holds the mailbox entry until its own marker says
`waiting` (§2.3). If V2 fails live, Codex native is restricted to
`idle_by_marker` and busy Codex targets keep today's wait-then-decide loop.

`replace_if_idle` is irrelevant to a native attempt (nothing is replaced); it
keeps its meaning for the resume fallback.

### 2.2 A — Codex child via `codex queue`

- Carrier: Codex's own persistent queue. The **message is the follow-up
  itself** (user decision), with the nonce marker appended exactly as the resume
  path does (`delivered_prompt`, multi-line form).
- Size/charset: argv goes to the native binary verbatim (no shell). If the
  delivered text exceeds a conservative argv budget (`_NATIVE_QUEUE_MAX_CHARS`,
  e.g. 24 000 chars on Windows, 100 000 on POSIX), use a **prompt file**:
  the queued message is the single line
  `Read your complete follow-up from UTF-8 file <path>, then follow it exactly. [win-agent-teams delivery id: wat-deliver:<nonce> — internal marker, ignore this line]`
  — the marker lives in the queued user message itself, so the receipt does not
  depend on the child reading the file. The file uses the existing prompt-file
  lifecycle (§4c of the protocol doc).
- Runner: extract the subprocess part of `CodexMemberWake._queue` into a
  shared `codex_queue(binary, thread_id, home, message, timeout)` helper (same
  env scrub, `cwd=home`, `stdin=DEVNULL`, real timeout).
- Outcome mapping — the key exactly-once rule is **"only a provable
  non-enqueue may fall back to resume in the same call"**:

  | `codex queue` outcome | Row | Next |
  |---|---|---|
  | exit 0 | `sent` | `confirm_delivery` on the bound rollout; `child_alive` probes the **existing** PID |
  | discovery failure, `unverified_thread`, non-zero exit **with** a recognised pre-enqueue error | back to `pending` | same call falls back to `resume` (fresh attempt, fresh nonce) |
  | timeout, non-zero exit with unrecognised stderr, `OSError` after spawn | `unconfirmed` | never resend; reconcile by nonce later (existing rescan-before-resend) |

  A queued-but-undispatched row that later dispatches (e.g. after the child is
  resumed for a *different* message) carries the original nonce, so it settles
  that row as `delivered` late — never a duplicate of a *different* row.
- Finalisation: a native `_finalize_follow_up` branch that keeps `pid`,
  `create_token`, `spawned_at`, `prompt_transport` untouched; on `unconfirmed`
  it writes `pending_delivery` with `method` so kill/reconcile know no new
  process exists. `kill_agent` reconciliation is unchanged (rescan first).

### 2.3 B — Claude child via delivery mailbox + own socket

Constraint (§9.1): only the child's own server may post to its host socket, and
guaranteed messages may not use the actionable inbox. New carrier:

- **Mailbox** `<session>/delivery-mailbox-<child>.json` (single JSON document,
  atomic write, guarded by `delivery-mailbox-<child>.lock`), a map
  `nonce → {operation_id, sender, text | prompt_file, state, ts}`.
  `state` transitions are **CAS under the lock**:
  `offered → taken(poster pid, create token) → posted | post_failed`, and
  `offered → retracted` (lead only). No counters, no cursors, so none of the
  inbox hazards apply.
- **Lead side** (`_do_follow_up`, method `claude_mailbox`): write `offered`,
  mark the row `sent`, then `confirm_delivery` on the bound transcript
  (existing PID for liveness). On budget expiry: if still `offered`, **retract**
  it (CAS) and return `queued(pending)` — nothing was presented, so a later
  retry may resend (new nonce). If `taken`/`posted`, return
  `queued(unconfirmed)` and leave reconciliation to the nonce scan.
- **Child side**: the child's MCP server runs a `DeliveryPoster` alongside
  (or inside) `NativeWakeNotifier`:
  - Owner lock `native-delivery-<IDENTITY>.lock` (same pattern as the notifier).
  - Capability marker `native-delivery-<IDENTITY>.json`
    `{pid, create_token, channel, heartbeat_ts}` refreshed every poll while the
    channel is `available`; deleted on shutdown. N3 requires a heartbeat within
    `3 × poll` **and** the owner lock currently held (`try_lock_handle` probe
    fails).
  - Loop: when its own state marker is `waiting` (or no marker yet after
    SessionStart), take the oldest `offered` entry (CAS → `taken`), post the
    delivered text (nonce marker included; long text via the prompt-file line,
    as Claude's resume sidecar already does) as the `user` line, then CAS →
    `posted`. A failed post → `post_failed`; the lead's reconcile treats
    `post_failed` + absent nonce as a provable non-delivery (back to `pending`).
  - It never posts while the marker says `running`; one entry per idle edge
    (the next waits for the next `waiting`), so a follow-up never lands between
    tool calls.
- **Flag propagation to the child** (fixes the #70 observation): the lead's
  `build_env` passes `WIN_AGENT_TEAMS_NATIVE_WAKE=1` and
  `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` to Claude children when on. Whether
  Claude Code forwards those to the child's MCP server must be verified
  (spike S-2); if not, the capability marker simply never appears and N3 falls
  back to resume — safe by construction.
- Receipt: the socket content becomes a `type:"user"` transcript record
  (spike S-3 must confirm the persisted shape; the scanner already accepts
  string and text-list content and `tool_result` for the file variant).

### 2.4 C — F5: Windows Claude channel

- `claude_platform_supported()` becomes `linux or win32`; macOS stays
  `unsupported_platform` (H2 cannot resolve `claude` there).
- Windows H-rows: H1 unchanged; H2 via the existing toolhelp host walk; **H3-W**:
  the socket value must start with `\\.\pipe\`; after `CreateFileW`,
  `GetNamedPipeServerProcessId(handle)` must equal the nearest `claude` host
  PID (real owner verification, better than the POSIX filename check) —
  mismatch ⇒ close without writing, `socket_not_owned`. H4: open failure ⇒
  `socket_missing`.
- Transport (`_post_pipe`, ctypes via `WinDLL("kernel32", use_last_error=True)`
  as `process_manager.py`/`procinfo.py` already do; no new dependency):
  `CreateFileW(GENERIC_WRITE, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED)`;
  `ERROR_PIPE_BUSY` ⇒ `WaitNamedPipeW` bounded by the remaining deadline;
  one `WriteFile` with an `OVERLAPPED` + manual-reset event;
  `WaitForSingleObject(remaining_ms)`; on timeout `CancelIoEx` then
  `GetOverlappedResult(wait=True)` so the buffer is never freed under a pending
  write; `FlushFileBuffers` is **not** used (it blocks). Always `CloseHandle`.
  Same two JSON lines, auth line mandatory.
- The existing `sys.platform` narrowing for `ty` stays; ctypes access is behind
  `getattr(ctypes, "WinDLL")` as elsewhere.

### 2.5 D — F3: Codex lead wake

- Registration: new tool `set_lead_wake(codex_thread_id, codex_home="")`
  (registered only with the master flag), storing
  `<session>/lead-wake-<IDENTITY>.json {thread_id, codex_home, generation}`
  after `verify_codex_thread`. Blank thread clears. Its docstring gives the
  same one-line shell command the join prompt uses
  (`echo "$CODEX_THREAD_ID ${CODEX_HOME:-$HOME/.codex}"` / PowerShell form).
- **Automatic for spawned Codex leads**: a spawned Codex agent's own server
  knows `AGENT_NAME`; when its record in `agents.json` has a bound
  `backend_session_id` and recorded `codex_home`, it self-registers, so nested
  Codex leads need no tool call.
- Delivery: `NativeWakeNotifier` gains a second channel kind. For the lead
  target, when the host is Codex (nearest host `codex`) and a registration
  exists, it runs the shared `codex_queue` helper with the **body-free** notice
  (`plan_notice` text, same coalesce/renotify/backoff/owner lock). This is a
  doorbell; the inbox remains authoritative and `watch` stays valid.
- `session_info.native_wake` gains `codex_lead: {registered, thread_verified}`.

### 2.6 Tool text

The consuming agent only reads docstrings. With the flags on, decorate
`follow_up_agent`, `send_message`, `delivery_status`, `spawn_agent` (new record
fields), `session_info`, and the new `set_lead_wake`; flag-off descriptions
stay golden-identical (`test_registered_tools_golden`).

## 3. Files affected

- `src/claude_teams/native_wake.py` — Windows pipe transport + H3-W,
  `codex_queue` helper, `DeliveryPoster`, notifier Codex-lead channel.
- `src/claude_teams/delivery_mailbox.py` (new) — mailbox store + CAS.
- `src/claude_teams/server_simple.py` — method selection in `_prepare`, native
  dispatch in `_do_follow_up`, native `_finalize_follow_up` branch,
  reconcile/kill awareness of `method`, record fields `interactive` +
  `codex_home` at spawn/resume, `set_lead_wake`, `session_info`, docstrings.
- `src/claude_teams/delivery_store.py` — `method` in `public_view`.
- `src/claude_teams/backends/process_base.py`, `claude_code.py` — flag
  propagation to Claude children.
- `docs/reference/agent-messaging-protocol.md`, `README.md`, `INSTALL.md` §6a,
  skills (`agent-orchestration`, `external-member-*`).
- Tests: see §4.

## 4. Tests (red first)

Unit / integration (fakes for subprocess, sockets, pipes, clock):

1. Flag-off golden: tools, join prompt, spawn/resume env, delivery results
   byte-identical; `method` absent from results.
2. Method matrix: each of N0–N4 false ⇒ `resume`; all true ⇒ native, per backend.
3. Old record without `interactive` ⇒ resume.
4. Codex native idle: no kill, no `backend.resume`, same PID, nonce in a queued
   user record ⇒ `delivered`; row `method=codex_queue`.
5. Codex outcome table: pre-enqueue failure falls back to resume in the same
   call with a fresh nonce; timeout ⇒ `unconfirmed`, never resent; later nonce
   ⇒ `delivered`.
6. Codex long prompt ⇒ prompt-file line carries the marker; shim binary ⇒
   resume.
7. Busy Codex target ⇒ queued immediately (flagged by V2 switch) / waits when
   the V2 switch is off.
8. Mailbox CAS: offered→taken→posted; retract only from offered; concurrent
   lead retract vs poster take (stress); crash between take and post ⇒
   reconcile by nonce.
9. Poster: posts only on `waiting`, one entry per idle edge, never on
   `running`; owner lock exclusivity; capability heartbeat freshness gate.
10. Claude native: lead budget expiry with `offered` ⇒ retract + `pending`;
    with `taken` ⇒ `unconfirmed`; `post_failed` + absent ⇒ `pending`.
11. Guaranteed messages still never appear in `inbox-<child>.jsonl`.
12. `kill_agent` with an outstanding native attempt: rescan first; retract
    `offered` mailbox entries; no stranded rows.
13. Windows pipe (real named pipe server in-test via ctypes, Windows-only):
    happy path; `ERROR_PIPE_BUSY` bounded wait; **stalled server** (never reads,
    tiny buffer) ⇒ `CancelIoEx` within deadline, no thread left behind;
    server-PID mismatch ⇒ no write.
14. H-table on Windows: pipe prefix, H3-W, macOS still unsupported.
15. Codex lead wake: registration validation/clear; notifier posts body-free
    notice via `codex_queue` on child reply; coalesce/backoff; spawned Codex
    lead self-registers from its bound record.
16. `ty` narrowing: no new platform-only diagnostics on Windows or Linux.

## 5. Live verification (before merge)

Spikes (first, they can change the design):

- **S-1** `codex queue` on a TUI child: what the rollout records for a queued
  turn (must be `response_item`, `role:user`, `input_text` containing the
  marker); is there a `codex queue` list/remove?
- **S-2** Does a Claude child's MCP server see env vars set on the `claude`
  process? (decides flag propagation).
- **S-3** Persisted transcript shape of a socket-posted user line; size limit
  of `content`.
- **S-4** `GetNamedPipeServerProcessId` on Claude Code's pipe equals the
  `claude.exe` host PID (CLI and Claude Desktop).

Smokes (Linux and Windows unless noted):

- **N1** Codex child idle follow-up delivered in place (same PID).
- **N2** Codex child busy follow-up dispatched after the turn (closes V2).
- **N3** Claude child (Linux, then Windows after C) idle follow-up in place,
  bypass mode, no approval prompt.
- **N4** Claude child busy: held until `waiting`, then delivered.
- **N5** Dead child (kill the process / reboot) ⇒ resume fallback, delivered.
- **N6** Windows Claude lead woken by a child's `send_message` (closes F5);
  stalled-pipe test harness.
- **N7** Codex lead (TUI and Desktop) woken by a child's reply (closes F3).
- **N8** All flags off: identical to `main`.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Double execution (queued native row + resume) | Only provable pre-enqueue failures fall back in-call; everything else is `unconfirmed` and reconciled by nonce; mailbox CAS retract-or-taken |
| Queued Codex turn never dispatches (thread unloaded, V1) | Row stays `unconfirmed` with live child (honest); a later resume loads the thread and dispatches it with its original nonce |
| Wrong-session post | H2/H3(-W) + capability marker + owner lock; scrub on spawn/resume unchanged |
| Pipe write hang | Overlapped I/O + `CancelIoEx`, tested with a stalled server |
| Claude holds/drops socket posts (`crossSessionInbound`, rate limits) | Post is never a receipt; budget expiry with `offered` retracts; `taken` stays `unconfirmed`; nonce decides |
| Poster missing in child (flag not propagated) | Capability marker absent ⇒ resume (safe default) |
| Behaviour change surprises | Sibling opt-in flag; flag-off golden tests |

## 7. Non-goals

- Pi native delivery.
- Making native delivery the default (separate PR once smokes are green on
  both platforms).
- Removing the `watch` recipe or the `Stop` hook.
- Message bodies in *wake* notices (D stays body-free; A/B carry bodies because
  they are deliveries with nonce receipts, not doorbells).

## 8. Open questions for review

1. Is "provable pre-enqueue failure" for `codex queue` decidable from exit
   code/stderr, or should every non-zero exit be `unconfirmed`?
2. Mailbox as one JSON document vs one file per entry (contention vs atomic
   CAS simplicity)?
3. Should the Claude poster also deliver mid-turn (between tool calls) when the
   sender opts in, or is idle-edge-only the right invariant?
4. F3 auto-registration for spawned Codex leads: trust the parent-bound
   `backend_session_id`, or require the child to confirm via `CODEX_THREAD_ID`?
