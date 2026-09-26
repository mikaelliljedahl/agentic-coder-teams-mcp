# Native session wake (Claude inbox socket + `codex queue`) — implementation plan

Feature slug: `native-session-wake` · Branch `feat/native-session-wake` ·
Worktree `/home/mikael/code/github/wt-native-session-wake`, created from
`origin/main` at `e5c8e98`.

Evidence:

- [`smoke-run-2026-09-25.md`](smoke-run-2026-09-25.md) — the pre-implementation
  smoke run, on Linux.
- The Claude Code doc
  <https://code.claude.com/docs/en/cross-session-messaging.md>, section "The
  session's inbox socket".
- The codex source at tag `rust-v0.156.1`, checked out in the session
  scratchpad (`codex-rs/…`).

Revision 2 applies [`plan-review.md`](plan-review.md) round 1; its
"Dispositions (round 1)" section maps each finding to the sections below. Every
citation was opened and checked at `path:line`. In `server_simple.py`, the
**symbol name** is authoritative because line numbers drift
(`docs/reference/agent-messaging-protocol.md:12-18`).

---

## 0. Scope

**Goal.** A Claude Code lead (Claude Desktop, `claude` 2.1.280) and a Codex
Desktop session that joined as an external member are both woken
**natively** when a message lands:

- an idle recipient starts a new turn;
- a busy recipient sees the message between tool calls.

**Opt-in, off by default.** Everything here is gated by
**`WIN_AGENT_TEAMS_NATIVE_WAKE=1`**; unset or any other value means off. With
the flag off:

- no thread starts, no lock file is created, and no subprocess runs;
- the spawn environment is not touched;
- the tool list, join prompt, and every tool result are **byte-identical to
  `main`**;
- `watch`, `lead_wake`, and `member_wake` are unchanged.

Flag-on differences are limited to one new tool, which is registered only then
(§2.3), and docstring sentences that name the flag.

**Additive, not a replacement.** A native notice is a **best-effort doorbell**.
Claude can silently hold or drop it under `crossSessionInbound` hold/refuse, or
under rate limits, and a Codex thread must be loaded for the queued row to fire.
So the notice never counts as delivery. The existing `watch` guidance stays in
every docstring, and agents are **not** told to stop arming a watcher.

**In scope**

- **A.** Each MCP server wakes **its own** Claude host through the session
  inbox socket (§2.1).
- **B.** Downstream wake for a Codex member via `codex queue` (§2.2–2.3).
- **C.** Stale-environment safety (§2.4).
- **D.** Contract text and skills (§2.6).
- **E.** Windows support, reduced to what is safe:
  - the Codex `codex queue` path, a subprocess with a real timeout;
  - the flag-off identity guarantee.

  The **Claude socket channel is POSIX-only** in this PR. On native Windows it
  reports `unsupported_platform` and does no pipe I/O at all. Cancellable
  overlapped named-pipe I/O is follow-up F5.

**Out of scope (follow-ups, §9):**

- Stop-hook rows that trust the native path (the former D4n/M4n);
- a native downstream transport for spawned children (§9.1);
- Codex-lead self-wake;
- a blocking `external_read(wait_seconds=…)`;
- the user-level recipe `~/.claude/commands/codex-visual-qa.md`.

---

## 1. Current behaviour

1. **Upstream (member → lead) has no push.** `external_send`
   (`server_simple.py:3720-3775`) appends to `inbox-<parent>.jsonl`. The lead
   learns of it only from a one-shot `watch` it armed itself
   (`agent-messaging-protocol.md:1456-1488`). The visual-QA recipe re-arms it
   with a 28-iteration background loop
   (`~/.claude/commands/codex-visual-qa.md:85-87`).
2. **Downstream (lead → external member) is pull-only.**
   - The external branch of `send_message` (`server_simple.py:3458-3502`) does
     one append under `_agents_transaction` (`:3460`).
   - Its docstring says "No idempotency key, lease, durable delivery row,
     process resume, or wake is involved" (`:3424-3428`).
   - The join prompt says "Poll external_read" (`:770`).
   - Codex has no idle wake.
3. **Children inherit the socket.**
   - Nothing in `src/` references `CLAUDE_CODE_MESSAGING_*`.
   - Every process manager starts from `os.environ.copy()`
     (`process_manager.py:636`, `:1096`, `:1499`, `:1907`).
   - A Codex or Pi child spawned by a Claude lead therefore inherits the lead's
     socket and token.
4. **Identity and session are process-global.**
   - `IDENTITY` is resolved once at import (`server_simple.py:145-168`).
   - The session is the module global `_session_id` (`:376`). It is set by
     `_active_session_id` (`:1260-1278`) and by `resume_session` (`:5968`).
   - Recovery has side effects: it sets `_pending_recovery` and can persist a
     binding (`:1177-1232`).
5. **No background task, and a fixed tool set.**
   - `main()` only calls `mcp.run()` (`:6884-6886`).
   - Tools are registered at import through `_register_tool`, whose only gate is
     `WIN_AGENT_TEAMS_EXTERNAL_ONLY` (`:361-373`).

**Verified host facts (2026-09-25):**

- **Socket path.** All 14 live Claude-hosted `claude_teams` servers carry
  `CLAUDE_CODE_MESSAGING_SOCKET=/run/user/1000/cc-socks/<N>.sock`, with `<N>`
  equal to the parent `claude` PID. This holds for both `cli` and
  `claude-desktop` entrypoints; tokens were not printed.
- **Codex filters the environment.** The three servers under Codex Desktop's
  `app-server` carry no `CLAUDE_CODE_*` variables, and that app-server has no
  `CODEX_HOME` override.
- **Documented vs observed export.** The doc guarantees a fresh socket export
  only to hooks and Bash ("never one inherited from a parent session"). Export
  to MCP servers is **observed only** (risk R1).
- **Where Codex threads are recorded.** A Desktop thread appears in
  `~/.codex/state_5.sqlite` table `threads` (`id`, `rollout_path`). Its rollout
  lives under `sessions/`, or under `archived_sessions/` once archived; the
  smoke thread `01a0d9a9-…` is now archived.

---

## 2. Design

### 2.0 Gate

`native_wake.enabled()` returns `os.environ.get("WIN_AGENT_TEAMS_NATIVE_WAKE",
"").strip() == "1"`. Sub-switches `…_CLAUDE=0` and `…_CODEX=0` disable one half
when the master flag is on; they never enable anything.

The gate is checked at each site:

- notifier start in `main()`;
- the Codex branch of `send_message`;
- registration of `external_set_wake`;
- the join-prompt text;
- the spawn scrub;
- `session_info.native_wake`.

The **acting** server's flag decides, so the lead's flag governs `codex queue`.
Children inherit the flag through `os.environ.copy()`, so nested Claude leads
get native wake too; H2 (§2.1) still stops non-Claude children from posting.

### 2.1 Claude host notifier — new module `src/claude_teams/native_wake.py`

It is import-light: `messaging`, `procinfo`, `filelock`,
`process_manager.creation_token`, and the stdlib. It never imports
`server_simple`.

**(a) Channel resolution — `resolve_claude_channel(environ, resolve_host=procinfo.resolve_nearest_host)`**

| Row | Condition | Result |
|---|---|---|
| H0 | gate off, or `…_CLAUDE=0` | `disabled` |
| H1b | native Windows (`os.name == "nt"`), evaluated **before** any env or host check | `unsupported_platform`; no env read, no host resolution, no pipe I/O (review R2-C/R3-B; follow-up F5) |
| H1 | socket or token env absent or empty | `no_socket` |
| H2 | nearest host is not Claude (`procinfo.is_claude_host`, `procinfo.py:78-80`) | `host_not_claude` |
| H3 | the basename matches `^\d+\.sock$`, and the digits ≠ the host PID | `socket_not_owned` |
| H4 | the path is not a socket | `socket_missing` |
| H5 | otherwise | available; `owner_verified` is true iff H3's pattern matched |

`procinfo._walk` (`procinfo.py:83-101`) stops at the nearest process in
`{"claude","codex","pi"}` (`procinfo.py:16`). A server under a Codex or Pi child
of a Claude lead therefore resolves to `codex` or `pi`, and H2 refuses it.

Resolution runs **once** per notifier start and is cached. H1b short-circuits
before host resolution, so Windows never pays for the PowerShell CIM query
(`procinfo.py:189-240`).

**(b) Poster — `post_claude_notice(channel, text, deadline=5.0) -> PostResult`**

The poster connects only once the text is ready, because the socket closes
connections that send no complete line within 30 s. It writes two lines, the
wire format proven in `smoke-run-2026-09-25.md:21-24`:

```text
{"type":"auth","token":"<CLAUDE_CODE_MESSAGING_TOKEN>"}
{"type":"user","message":{"role":"user","content":"<notice>"}}
```

The auth line is always sent. It is optional on POSIX, and sending it keeps
the wire format ready for the Windows follow-up. The message keeps
`role:"user"` (§10 answer 4).

- **Transport (POSIX only):** `AF_UNIX` with `settimeout(deadline)`. That
  timeout is a real operation deadline for both connect and `sendall`. Then
  `shutdown(SHUT_WR)` and close. No response is read. There is no worker thread.
- **Errors.** All errors become `PostResult(ok=False, reason)`. Nothing raises
  into the notifier.

**(c) Notice policy — `plan_notice(state, snapshot, now, cfg)`, pure**

The same rule drives Codex (§2.2), except for coalescing. `snapshot` is `{sender: {total, cursor}}`
from one read-only scan (`read_inbox_by_sender` + `load_inbox_cursors`, as in
`lead_wake._scan_senders`, `lead_wake.py:229-247`). State is
`{notified: {sender: total}, first_new, last_success, seq}`.

- `new[s] = total − max(notified[s], min(cursor, total))`; keep only positive
  values.
- **Coalesce (Claude notifier only).** Post only when
  `now − first_new ≥ COALESCE` (2 s). The notifier's own tick re-evaluates, so
  the deadline always gets another decision. The Codex send path passes
  `coalesce=0`: it has no timer, so it must decide immediately (§2.2).
- **Immediate backlog check.** The baseline catch-up after a session activates
  (f) also passes `coalesce=0`.
- **Outstanding.** While any `cursor < notified[s]` and
  `now − last_success < RENOTIFY` (300 s), defer.
- **Only a successful post advances state:** `notified = totals`, `seq += 1`,
  `last_success = now`.
- **Baseline.** A new target starts with `notified = cursors`, so a backlog at
  start produces one catch-up notice.

The notice is one line, carries no body and no token, and includes `seq` and
counts so identical-repeat dropping cannot swallow it:

```text
[win-agent-teams wake #<seq>] <N> unread message(s) in your team inbox from: alice (2). Best-effort notice without content; call read_messages to read them.
```

A member target says "call external_read with the member_token you saved".

**(d) Backoff — one policy for every post failure, Claude and Codex**

- Per channel: exponential, starting at 2 s, doubling, capped at 300 s, and
  reset on success.
- The Claude channel key is the target. For Codex the key is
  `(session, member, registration generation)`, so a new registration starts
  with no backoff (R3-A).
- While backing off, the notifier does not post, and a Codex send returns
  `wake.status:"backoff"` without a subprocess.
- `seq` never advances on failure.

**(e) Single owner per reader — OS lock, no heartbeat**

- **Lock file.** `native-wake-<reader>.lock` in the session dir; for a member,
  `native-wake-member-<member>.lock` in the joined dir.
- **New primitive.** `filelock.try_lock_handle(handle) -> bool` makes one
  non-blocking attempt. It is needed because the existing `lock_handle` blocks
  on POSIX (`filelock.py:43-58`).
  - **POSIX:** `fcntl.flock(LOCK_EX | LOCK_NB)`.
  - **Windows:** `handle.seek(0)` first, as `lock_handle` does at
    `filelock.py:49`, then a single `msvcrt.locking(LK_NBLCK)`. It exists for
    shared use; the Claude notifier does not run on Windows.
  - It returns `False` **only for contention**: `EWOULDBLOCK`/`EAGAIN`, or
    `EACCES`/`EDEADLK` from `msvcrt`. Every other `OSError` propagates, and the
    notifier treats it as a post failure with backoff.
- **Holding it.** The notifier opens the file `a+b`, never truncating, as
  `file_lock` does (`filelock.py:70-84`). It keeps the handle open, holding the
  lock for the **target's lifetime**, and releases it on target drop or session
  switch.
- **Release on death is automatic**, because the kernel drops the lock with
  the process. There is no heartbeat, no staleness timeout, and no displacement
  of a live owner.
- **Children do not inherit it.** Python opens files non-inheritable, and
  `Popen` closes fds.
- **Non-owners** do not notify for that reader and retry acquisition every
  10 s.
- **Before each post** the notifier rechecks that it still holds the target's
  handle and that the target's session is unchanged.
- **Lock ordering.** The agents lock is never held while posting; the notifier
  never takes it except for a read-only membership check.
- **Why this suffices.** Two leads bound to one session as `team-lead` is
  already unsupported (`agent-messaging-protocol.md:1787-1795`). The lock
  simply makes them produce one notice stream.

**(f) Notifier thread — `NativeWakeNotifier`**

The thread is a daemon started in `main()` before `mcp.run()`, only when the
gate is on. It is not started through the FastMCP lifespan, so tool tests never
start it. It never writes to stdout, which carries the MCP protocol; it logs
through `logging` to stderr.

Each tick (default 1 s):

1. **Targets.**
   - **The lead target** exists only when:
     - the module global `_session_id` is non-empty;
     - identity is resolved;
     - the server is not in external-only mode.

     The thread reads the global only. It never calls `_active_session_id` or
     recovery (§10 answer 1).
   - **Immediate backlog check on activation** (incident, 2026-09-26). Every
     code path that assigns the global calls
     `native_wake.session_activated(session_id)` right after the assignment.
     Those paths are:
     - `_active_session_id`, on recovery, auto-adopt, or create
       (`server_simple.py:1260-1278`);
     - `resume_session` (`:5968`).

     The call is a no-op when the gate is off. It sets a `threading.Event` that
     wakes the notifier immediately instead of on the next tick. The event is
     set from the tool thread; all I/O stays in the notifier.

   - **Activation semantics** (review R3-C):
     - **Consume, then snapshot.** The notifier first clears the event, *then*
       reads the latest `_session_id`. An activation arriving during a scan
       therefore re-sets the event and is processed on the next iteration; it
       is never lost.
     - **Catch-up fires only on a transition.** The lead target is keyed by
       `(session_id, IDENTITY)`, and the baseline catch-up notice
       (`coalesce=0`) is posted only when that key **differs** from the
       current target. That covers first activation after a restart, S1→S2,
       and a return to a previously dropped target.
     - **Repeated same-session activation** (for example `resume_session` on
       the session that is already active, `server_simple.py:5946-5968`) keeps
       the existing target unchanged. Its notice state, outstanding window,
       backoff, and owner lock all survive, so it cannot trigger a duplicate
       notice.
     - **Rapid S1→S2.** Only the latest snapshot is processed. S1's target and
       lock are released, S2 gets exactly one catch-up, and S1 gets no late
       notice.
   - **Restart procedure (documented).** The notifier cannot choose a session
     after a restart, because that would mean side-effectful recovery. The lead
     must therefore call `session_info` or `resume_session` first. With the flag
     on, the `resume_session` and `session_info` docstrings say that a backlog
     notice follows immediately when unread messages are waiting. The same
     sentence goes in the external-member skills' restart notes.
   - **Member targets** come from `watch_member(session_id, name)`, which
     `external_read`/`external_send`/`external_set_wake` call after token
     resolution (`_member_operation`, `server_simple.py:880-960`). A member
     target is dropped when its record is no longer `running`.
2. **Change detection.** Stat `(mtime_ns, size)` of the inbox and `.pos.json`.
   Rescan only on change, on a pending deadline, or on an activation event.
3. **Post.** If this server is the owner (e), run `plan_notice`, then post.

**Lock order for the notifier.**

- The target-registry lock is an in-process `threading.Lock` guarding the set
  of targets. It is held only to copy or mutate the set, never while acquiring
  anything else.
- `watch_member` is called **after** `_member_operation`'s `with` block has
  exited, so the agents lock is already released. It takes only the registry
  lock.

### 2.2 Codex member wake (lead side)

**The lead pushes, not the member.**

- The external branch of `send_message` is the **only** writer to a member
  inbox. It requires `target.spawned_by == IDENTITY` (`server_simple.py:3461-3466`).
- The lead gets a status back.
- Codex filters its MCP servers' environment (§1).

**Notice state is keyed by registration** (review R3-A). The Codex
`notified`/`last_success`, the thread-verification cache, and the backoff are
all keyed by `(session, member, codex_wake.generation)`, not by member alone.

- A new generation starts with `notified` initialised from the member's
  **current cursors**, with no `last_success` and no backoff.
- So after a successful wake to thread A, a change to B while A's notice is
  still undrained makes the next send **queue to B**. It is not coalesced
  against A's state.
- Entries for older generations are dropped when a newer one is first used.

**Per-member decision under an in-process lock.**

- The decision takes a `threading.Lock` per `(session, member)` and holds it
  across decide → queue → update.
- An in-process lock suffices because one lead identity per session is the
  supported configuration, and only that identity's server writes the member's
  inbox.
- The worst case is two processes sharing one identity; that yields one extra
  doorbell, which is harmless because the inbox stays the source of truth.

**Global lock order.** It is stated once and applies to every code path:

    per-member lock  →  agents lock (short)  →  release  →  queue subprocess

- The append in `send_message` happens under `_agents_transaction`
  (`:3460-3500`), unchanged, **without** the per-member lock. The per-member
  lock is taken only after that transaction has exited.
- The per-member lock is **never** awaited while holding the agents lock.
- The agents lock is **never** held across the subprocess.
- The target-registry lock (§2.1f) is a leaf lock: nothing else is acquired
  while it is held.
- **Registration generation.** `external_set_wake` stores `codex_wake` with an
  integer `generation`, incremented under the agents lock on every set, change,
  or clear. `leave_team` and `kill_agent` change `status` or remove the record,
  which the revalidation below also catches.

**The decision** runs after the append's transaction has exited:

1. Take the per-member lock.
2. Take the agents lock **briefly**. Snapshot `status`, `codex_wake`
   (thread, home, generation), and the member inbox scan; then release.
3. If the member is not `running`, or has no `codex_wake`, there is no queue
   call: the status is `coalesced` for a member that left, or the `wake` field
   is omitted when the registration was cleared.
4. Decide, using the §2.1(c) rule with `coalesce=0`:
  - `last_success` is the time of the last **successful** `codex queue`
    insertion.
  - A failed, timed-out, or unverified attempt updates nothing, so the next
    send retries, subject to backoff (§2.1d).
   - **The first send with no outstanding wake queues immediately.**
   - Later sends, while the member still has unread **and** `last_success` is
     within the re-notice floor, get `wake.status:"coalesced"`.
   - Two concurrent sends produce one queue call: the second waits on the
     per-member lock, then sees the first as outstanding.
5. **Revalidate immediately before the subprocess.** Take the agents lock again
   briefly and compare `status == "running"` and `codex_wake.generation` with
   the snapshot, then release.
   - On a mismatch there is no queue call: `wake.status:"stale_registration"`,
     or `coalesced` if the member left.
   - The residual window between this check and the subprocess is one queue
     row to a just-replaced thread. That is harmless, because the inbox stays
     the source of truth.
6. Queue with the agents lock free, then update `notified`/`last_success` on
   success.

**The queue call.**

- **Binary.** `CodexBackend().discover_binary()` (`backends/codex.py:283-302`).
  When it is not found, the status is `unavailable`.
- **Environment.** The call runs with `CODEX_HOME=<member-reported home>`.
  Codex resolves its home from `CODEX_HOME` before `thread/queue/add`
  (`codex-rs/tui/src/session_queue_commands.rs:37-40`, `find_codex_home` in
  `codex-rs/utils/home-dir/src/lib.rs:13-18`).
- **Working directory.** `cwd=Path.home()`. The queue path starts or attaches
  an app-server and loads config non-interactively
  (`codex-rs/tui/src/session_archive_commands.rs:285-304`). It runs no TUI and
  has no trust prompt, so cwd only selects project config layers; home avoids
  pulling in an unrelated project's config.
- **Subprocess.** `stdin=DEVNULL`, UTF-8 `capture_output`, and a timeout of
  `WIN_AGENT_TEAMS_CODEX_QUEUE_TIMEOUT_SECONDS` (default 15).
- **Outcome.** `queued`, `failed` (non-zero exit, stderr tail ≤ 200
  characters), or `timeout`. The daemon refusal
  (`session_queue_commands.rs:32-50`) surfaces as `failed`.

**Notice text** contains no `cmd.exe` metacharacters (`( ) < > | & ^ % ! " '`),
no newline, and no member-controlled free text. Only `[A-Za-z0-9_-]` names are
interpolated. This keeps it safe on the `codex.cmd` shim fallback
(`_launches_via_cmd_shim`, `codex.py:553-561`):

```text
win-agent-teams: new message from team-lead in your member inbox - call external_read with your member_token
```

**Result.** `wake: {method:"codex_queue", status:
queued|coalesced|backoff|failed|timeout|unavailable|unverified_thread|stale_registration|disabled,
detail?}`. It is added only when the flag is on **and** the member registered a
thread. The wake never fails the send, and `success`/`delivery:"inbox"` are
unchanged.

**Verifying the thread before the first queue for each registration.** This is
cached on success and re-run when the registration changes.

1. The reported home must be an absolute path that is an existing directory on
   the lead's machine.
2. **Only if the db file exists**, open `<home>/state_5.sqlite` with the URI
   `file:<path>?mode=ro`.
   - The filename is the constant `STATE_DB_FILENAME`,
     `codex-rs/state/src/sqlite.rs:33`.
   - The mode is `ro`, not `immutable`, so uncheckpointed WAL rows are
     visible. Codex runs WAL with a 5 s busy timeout
     (`codex-rs/state/src/sqlite.rs:297-330`).
   - Set `timeout=0.5` as the busy timeout.
   - Query `SELECT rollout_path, archived FROM threads WHERE id = ?`. The
     columns are defined in `codex-rs/state/migrations/0001_threads.sql:1-14`.
   - **Close the connection before the queue call.**
   - Outcomes:
     - `archived = 0` with a rollout under `sessions/` passes;
     - `archived != 0`, or a rollout under `archived_sessions/`, gives
       `unverified_thread` (detail `archived`);
     - a locked, busy, or schema error (`sqlite3.OperationalError`,
       `DatabaseError`) falls back to step 3.
3. If the db is missing, errors, or has no row, glob `<home>/sessions/*/*/*/`
   for `rollout-*-<id>.jsonl` or `rollout-*-<id>_*.jsonl`. The name format is
   `codex-rs/rollout/src/rollout_file_name.rs:62-73`, and `sessions` is
   `SESSIONS_SUBDIR`, `codex-rs/rollout/src/lib.rs:84`. The glob is kept for
   schema drift.
4. If nothing is found: `unverified_thread`, and no queue call.

### 2.3 Registration tool `external_set_wake(member_token, codex_thread_id, codex_home)`

This replaces the `join_team` parameter from revision 1.

- **Registration.** It is registered through `_register_tool(external=True)`
  and **only when the gate is on at import**. The flag-off tool list is
  therefore byte-identical, and the tool is also available under
  `WIN_AGENT_TEAMS_EXTERNAL_ONLY`.
- **Consequence.** The **member's** MCP entry needs the flag too. The join
  prompt says so, and says to keep polling if the tool is absent.
- **Authentication.** The token is resolved via `_member_operation`
  (`server_simple.py:880-960`). The call requires `status:"running"` and can be
  repeated at any time; there is no ticket-retention limit.
- **Validation.**
  - `codex_thread_id` must be a canonical UUID.
  - `codex_home` must be a non-empty absolute path; blank means `~/.codex`.
  - A blank thread id clears the registration.
  - A validation failure gives `{success:false, reason:"invalid_codex_thread_id"|"invalid_codex_home"}`
    and writes nothing.
- **Storage.** It stores `codex_wake: {thread_id, codex_home, registered_at,
  generation}` on the record under the agents lock, inside `_member_operation`.
  `generation` is incremented on every change, including a clear, and is
  **monotonic for the life of the record**:
  - A clear keeps the field as `codex_wake: {thread_id: null, codex_home: null,
    generation: n+1, registered_at}` rather than deleting it, so a later set
    becomes `n+2` and never reuses an old value.
  - "Registered" means `thread_id` is non-null.

  The field is mutable, which `_external_record_matches` ignores
  (`:815-828`). It is not a credential.
- **Return and side effect.** It returns `{success, name, codex_wake}`, and
  calls `watch_member` after the `with` block exits (§2.1f lock order).
- **Join prompt, flag on only.** One added step:

  > If you are a Codex session, run one shell command and pass both values to
  > `external_set_wake(member_token=…, codex_thread_id=…, codex_home=…)`:
  > Unix `echo "$CODEX_THREAD_ID ${CODEX_HOME:-$HOME/.codex}"`; PowerShell
  > `"$env:CODEX_THREAD_ID $(if ($env:CODEX_HOME) {$env:CODEX_HOME} else {Join-Path $HOME '.codex'})"`.
  > The lead can then wake you with a queued message. This is best effort, so
  > still check `external_read` before ending a long wait.

  `CODEX_THREAD_ID` reaches shell commands per
  `codex-rs/protocol/src/shell_environment.rs:7,152`, and the smoke run
  confirmed it in Desktop (`smoke-run-2026-09-25.md:50-51`).

### 2.4 Stale-environment safety

- **Primary guard: H2/H3** (§2.1a).
- **Defence in depth, gate on only.** `BaseBackend._spawn_with_command`
  (`backends/process_base.py:95-105`) is used by **both** `spawn` and `resume`
  (`:82-93`). It sets `CLAUDE_CODE_MESSAGING_SOCKET=""` and
  `CLAUDE_CODE_MESSAGING_TOKEN=""` in `env_vars`.
  - Empty rather than deleted, because managers merge `os.environ` first and
    `_build_posix_shell_command` exports the dict (`process_manager.py:146-150`).
  - H1 treats empty as absent.
  - This matters most on Windows, where own-child verification is token-only
    (doc).
  - A spawned `claude` exports its own socket to hooks and Bash (doc). Export
    to its MCP child is verified in smoke S2.

### 2.5 Env vars

| Env var | Default | Meaning |
|---|---|---|
| `WIN_AGENT_TEAMS_NATIVE_WAKE` | unset (off) | Opt-in master flag. Exactly `1` enables |
| `WIN_AGENT_TEAMS_NATIVE_WAKE_CLAUDE` / `…_CODEX` | `1` | With the flag on, `0` disables that half |
| `WIN_AGENT_TEAMS_NATIVE_WAKE_POLL_SECONDS` | `1.0` | Notifier tick |
| `WIN_AGENT_TEAMS_NATIVE_WAKE_COALESCE_SECONDS` | `2.0` | Burst window |
| `WIN_AGENT_TEAMS_NATIVE_WAKE_RENOTIFY_SECONDS` | `300` | Re-notice floor while outstanding |
| `WIN_AGENT_TEAMS_CODEX_QUEUE_TIMEOUT_SECONDS` | `15` | Bound on `codex queue` |

Seconds use strict finite-positive parsing, the same as
`_strict_positive_seconds` (`server_simple.py:616-625`), copied into
`native_wake` to avoid an import cycle.

### 2.6 Contract text

Every changed docstring says the native path is active **only** with
`WIN_AGENT_TEAMS_NATIVE_WAKE=1`, is a **best-effort doorbell**, and never
replaces the watcher. The existing watcher guidance is kept verbatim.

- **`send_message`:** the `wake` field and its statuses; the wake never fails
  the send. The "No … wake is involved" sentence becomes "no wake unless …".
- **`read_messages` / `external_read`:** a `[win-agent-teams wake #n]` or
  `win-agent-teams:` notice has no content, so read the inbox when you get one.
  Duplicates are harmless.
- **`external_set_wake`** (new): the full contract of §2.3.
- **`create_join_ticket`:** the join prompt carries the §2.3 step when the flag
  is on.
- **Platform wording (R3-B).** Every changed description that mentions the
  Claude notice (`read_messages`, `external_read`, `session_info`,
  `resume_session`, `_DISK_CONTRACT_NOTE`) states: "Claude session wake is
  POSIX-only (Linux/macOS); on native Windows it is unavailable and the
  watcher is the wake path." Codex `codex queue` wake works on all platforms.
- **`session_info`** (flag on only): `native_wake: {claude_channel:
  "available"|<H-row reason>, owner_verified, notifier_owner: bool}`. This
  reports **channel availability, never delivery**.
- **`resume_session` / `session_info`** (flag on only): one sentence each.
  After a restart, call `session_info` or `resume_session` first. When the
  Claude channel is available and unread messages are waiting, a backlog
  notice follows immediately (§2.1f).
- **`_DISK_CONTRACT_NOTE`** (`:1573-1642`): one flag-on paragraph covering the
  lock files, the kill switches, and "keep arming `watch`".
- **Skills.** `.claude/skills/external-member-join` gains a Codex subsection
  (`external_set_wake`, both shells). `external-member-invite` gains the `wake`
  field. Both keep their watcher steps.
- **Docs.** `README.md` (external members, env table) and
  `agent-messaging-protocol.md` (external members, on-disk table, §5 note).

---

## 3. Files affected

- **New** `src/claude_teams/native_wake.py`.
- `src/claude_teams/filelock.py`: `try_lock_handle`.
- `src/claude_teams/server_simple.py`:
  - `main()` starts the notifier;
  - the Codex wake in `send_message`;
  - the `external_set_wake` tool, registered only when the gate is on;
  - `watch_member` calls in `external_read`/`external_send`;
  - `_build_join_prompt`;
  - `session_info`;
  - docstrings and `_DISK_CONTRACT_NOTE`.
- `src/claude_teams/backends/process_base.py`: the scrub.
- **New tests** `tests/test_native_wake.py` and
  `tests/test_codex_member_wake.py`.
- **Extended tests:** `test_join_team.py`, `test_tool_descriptions.py`,
  `test_backends/*`, and the filelock tests.
- **Docs:** `README.md`, `docs/reference/agent-messaging-protocol.md`, the two
  skills, and `docs/features/native-session-wake/{implementation,implementation-review}.md`.

`lead_wake.py` and `member_wake.py` are **unchanged** (§9 F2).

---

## 4. Test cases (red → green)

### 4.0 Flag off is byte-identical — written first, green throughout

They run with the flag unset, and with it set to `0`, `true`, and empty.

- F0: `main()` never constructs the notifier.
- F1: the golden-string join prompt, captured from `main`.
- F2: the send result to an external member, even one with a `codex_wake`
  record; the fake runner asserts no call.
- F3: the spawn **and resume** environment is unchanged for claude-code, codex,
  and pi.
- F4: `session_info` has no `native_wake` key.
- F5: no `native-wake-*` files after a join/send/read cycle.
- F6: the registered tool list equals the golden list, with no
  `external_set_wake`, including under `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`.
- F7: the existing suites (`test_join_team`, `test_send_path`, `test_lead_wake`,
  `test_member_wake`, `test_cli_watch*`) pass unmodified.

### 4.1 `tests/test_native_wake.py`

1. The gate truth table and sub-switches (H0).
2. H1: absent or empty env → `no_socket`.
3. H2: an injected resolver returning `codex`, then `pi` → `host_not_claude`.
   This is the stale-environment regression.
4. H3: `999.sock` with host 123 → `socket_not_owned`; `123.sock` →
   `owner_verified`; a non-numeric name → available but unverified.
5. POSIX: a real `AF_UNIX` server in `tmp_path` receives exactly the auth line
   and the user line, then EOF.
6. **Windows is unsupported (H1b).** With `os.name` patched to `"nt"`, the
   result is `unsupported_platform`: no `open`, no socket call, and no host
   resolution (spies assert zero calls). This holds **both with and without**
   the socket and token env set; without them the result is not
   `no_socket` (R3-B). With the gate on, `main()` does not
   start the notifier thread on Windows.
7. Every post error returns `ok=False` and never raises.
8. `plan_notice`: growth after coalesce; defer while outstanding; re-notify
   after the floor; re-notify after a drain; baseline catch-up; a failed post
   does not advance state.
9. The notice text has no body or token, and consecutive notices differ.
10. **Backoff.** A persistent `OSError` produces attempts at t≈0, 2, 6, 14…,
    not one per tick. A success resets the backoff.
11. **Owner lock.**
    - Two notifiers on one reader in one process (separate open file
      descriptions) → exactly one owner.
    - A subprocess holds the lock and is killed → the next attempt from this
      process acquires it.
    - The owner rechecks the lock before each post; a switched session releases
      it.
12. The lead target follows a `_session_id` switch. There is no lead target
    when identity is unresolved or in external-only mode, and recovery
    functions are never called (asserted with a patched spy).
13. A member target is registered and then dropped when the member leaves.
14. The notifier never writes to stdout (`capsys`).
15. `filelock.try_lock_handle`: a free lock → `True`; a held lock → `False`
    immediately, without blocking. A non-contention `OSError` propagates. On
    Windows, `seek(0)` is asserted on a patched `msvcrt`.
15a. **Immediate backlog on activation (incident test).** A notifier runs with
    no active session, and the lead inbox already holds an unread message.
    - Calling `resume_session(sid)` sets the event; the notice is posted
      **within one second without waiting for a tick or for coalesce**, and it
      names the sender.
    - The same holds for the first `_active_session_id` auto-adopt.
    - The flag-on `resume_session` and `session_info` docstrings contain the
      backlog-notice sentence.
15b. **Repeated same-session resume.** An outstanding notice plus backoff are
    set, then `resume_session(sid)` is called twice on the already-active
    session: there is no extra notice, and the outstanding state, backoff, and
    lock are unchanged.
15c. **Rapid S1→S2.** Activating S1 and then S2 before the notifier runs
    produces exactly one catch-up, for S2. There is no S1 notice, and S1's lock
    is released.
15d. **Activation during a scan.** A patched scan blocks; S2 is activated while
    it runs. After the scan returns, the next iteration processes S2 — the
    event is not lost — with one catch-up.

### 4.2 `tests/test_codex_member_wake.py`

All of these run with the flag on.

16. `external_set_wake` stores the registration, can be re-called to update or
    clear it, rejects a malformed id or home with no write, is refused for a
    left member, and exists under `EXTERNAL_ONLY`.
17. A send to a registered member → the runner gets argv `[bin, "queue",
    "--thread", tid, "--message", notice]`, `env CODEX_HOME=<member-reported
    home>`, and `cwd=Path.home()` (the lead's home, not the Codex home); the
    result has `wake.status:"queued"` **in the same call**.
17a. **An isolated single send.**
    - One send and no further activity; the runner is called exactly once
      during that `send_message` call.
    - Real time then advances past 2 s and the re-notice floor is shortened.
      There is no second call, because the member has not drained; there is no
      stranded "pending" state, because the send path never waits for a timer.
      (Review R2-A.)
18. **Notice charset.** No `( ) < > | & ^ % ! " '` or newline, and no body or
    token.
19. **Shim path.** With `discover_binary` forced to a `codex.cmd` path, the argv
    and notice are unchanged and pass through with no metacharacters. This is
    asserted on the built argv; the Windows run is gate W2.
20. A failed first send, then a second send → a second queue attempt is made
    (after backoff), not `coalesced`.
21. Two concurrent sends (threads) → exactly one runner call; the other gets
    `coalesced`.
22. `external_read` drains, then a new send queues again.
23. **Thread verification.**
    - A `state_5.sqlite` row under `sessions/` → queued.
    - An archived row → `unverified_thread`.
    - No db but a rollout glob hit → queued.
    - A **home mismatch** (thread absent from the reported home) →
      `unverified_thread` with no runner call.
    - A non-existent home → `unverified_thread`.
24. Timeout, failure, and missing-binary cases → `success:true` with
    `wake.status` set.
25. The runner is invoked with the agents lock free: a probe acquires
    `_agents_file_lock` within 0.5 s.
26. `…_CODEX=0` → `disabled`, with no runner call.
26a. **A send racing re-registration.** The runner is blocked on an event
    while a first send holds the per-member lock. `external_set_wake` changes
    the thread, and then a second send proceeds. After the first returns, the
    second send's revalidation sees the new `generation`: it queues to the
    **new** thread, or reports `stale_registration`. No queue ever goes to the
    old thread after the change is committed, except in the documented
    single-row residual window.
26b. **A send racing `leave_team`.** The member leaves between snapshot and
    revalidation → no runner call and `wake.status:"coalesced"`. A subsequent
    send gets `member_left`, as today.
26c. **Lock order.** `external_set_wake` and `send_message` running
    concurrently in threads for 200 iterations produce no deadlock (bounded
    join). A spy asserts that the per-member lock is never requested while
    `_agents_file_lock` is held.
26e. **Generation keying** (R3-A):
    - A successful wake to thread A, then `external_set_wake` changes to B with
      A's notice undrained → the next send queues to **B** (runner argv has B).
    - Set A (gen 1) → clear (gen 2) → set A again (gen 3). The generation never
      repeats, the state is fresh, and the send queues.
    - Thread A fails into backoff, then the registration changes to B → the
      next send attempts B immediately (no inherited backoff) and re-runs
      verification for B.
26d. **SQLite verification** (review R2-D):
    - A row that exists only in an **uncheckpointed WAL**, written by a
      connection that stays open, is found.
    - An **exclusively locked** db and a **schema without `archived`** both fall
      back to the glob.
    - `archived=1` with a `sessions/` path → `unverified_thread`.
    - The connection is closed before the runner is called (spy).
    - The **mtime, size, and file set** of `state_5.sqlite*` and the `home` dir
      are unchanged afterwards, and **no db file is created** when absent.

### 4.3 Contract

27. `test_tool_descriptions`:
    - `send_message` mentions `wake` and `WIN_AGENT_TEAMS_NATIVE_WAKE`;
    - `read_messages`/`external_read` mention the notice and "best-effort";
    - **the watcher guidance is retained** — `_DISK_CONTRACT_NOTE` still
      contains "run the watch as a BACKGROUND command" and the flag-off join
      prompt still contains "Optional watcher";
    - no docstring says to stop arming `watch`;
    - every changed description that mentions the Claude notice contains
      "POSIX-only" (R3-B).
28. The flag-on join prompt contains `external_set_wake`, both shell commands,
    and still contains `--reader <name>`.

**Order.** F0–F7, then 1–15 (red: the module is absent), then 16–26, then
27–28. After that the whole-repo gates on Linux:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Finally the Windows checks (§5).

---

## 5. Smoke gates

- **Codex Desktop queue wake: PASSED** (`smoke-run-2026-09-25.md:44-63`).
  - The queue drained in about 4–5 s and a new turn started, whether the thread
    was visible or open but not visible.
  - The round trip took about 11–14 s.
  - A live direct-child prototype woke the idle Claude lead every time.
- **Post-implementation (Linux), recorded in `implementation.md`.**
  - **S1.** Claude Desktop lead plus Codex Desktop member with the real server
    and flag on: a round trip with no human nudge.
  - **S2.** A spawned Claude child's win-agent-teams MCP process has its **own**
    non-empty socket (stem = the child's PID) after the scrub.
  - **S3.** Codex and Pi children log `host_not_claude`.
  - **S4.** A `resume_session` switch. Also a **restart with a backlog**:
    restart the Claude lead, leave an unread upstream message, call
    `resume_session`, and the backlog notice arrives immediately (the incident
    of 2026-09-26).
  - **S5.** A lead in bypass mode gets notices without approval.
  - **S6.** With `crossSessionInbound=refuse`, the notice is dropped silently
    and the watcher still wakes the lead. This demonstrates the additive
    design.
  - **S7.** With the flag off, there are no notices, no lock files, and no
    queue rows.
- **Before merge (Windows; the user has a machine).**
  - **W1.** With the flag **off**, the tool list, join prompt, and results
    match `main`, and the full suite is green on Windows.
  - **W2.** With the flag on, a lead sends to a Codex Desktop member on Windows.
    `codex queue` runs through the resolved native binary, or the `.cmd`
    fallback, with the notice intact, and the member is woken. Its timeout is a
    real `subprocess` timeout.
  - **W3.** With the flag on, `session_info.native_wake.claude_channel ==
    "unsupported_platform"` and no pipe is opened. A spawned child's env has the
    scrubbed empty variables.
- **Open verification, not blockers.**
  - **V1.** Desktop closed or the thread unloaded: expected to stay queued
    until the thread is next loaded.
  - **V2.** Queued while a turn is running: expected to dispatch after the
    turn.
  - Record both outcomes in the `external_set_wake` docstring.

---

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | MCP-server export of the socket env is undocumented | Opt-in; H1 degrades to today's behaviour; additive design; S2 |
| R2 | Wrong-session notify through inherited env | H2 (primary), H3, and the scrub on spawn and resume; tests 3–4, F3; S2/S3; W2 |
| R3 | Duplicate notifiers or storms | OS owner lock (§2.1e), coalesce/outstanding, backoff, and `seq` in the text |
| R4 | Notice held or dropped (`crossSessionInbound`, receiver limits, unloaded Codex thread) | Additive: the watcher guidance is kept, `session_info` reports availability only, and S6 |
| R5 | `codex queue` is experimental; CLI 0.156.1 vs Desktop 0.155.0-alpha | Bounded, reported, never fatal; `…_CODEX=0`; the gate passed |
| R6 | Wrong `CODEX_HOME` | The member reports its home; the thread is verified in that home before queueing; test 23 |
| R7 | Windows pipe write hangs | Not reachable: the Claude channel is `unsupported_platform` on Windows (H1b); overlapped I/O is follow-up F5 |
| R10 | Codex queue races re-registration or leave | One global lock order; generation revalidated before the subprocess; tests 26a–c |
| R11 | Restarted lead with a backlog is never woken | Immediate backlog check on session activation; the restart procedure is in the docstrings; test 15a; S4 |
| R8 | The notifier harms stdio or the server | Daemon thread, never stdout, everything caught; not started in tests |
| R9 | Opt-in leaks | A single gate; F0–F7 written first |

---

## 7. Alternatives rejected

- **A blocking `external_read(wait_seconds)`.** It costs tokens per poll and
  holds the turn open. It remains an optional follow-up.
- **A Codex Stop hook that blocks.** It holds the turn and needs hook trust.
- **Desktop `ipc.sock`.** Undocumented and churns between versions.
- **A detached poster.** It loses own-child status
  (`smoke-run-2026-09-25.md:28`).
- **A member-side Codex self-notify.** Its environment is filtered, it gives
  the lead no status, and its registration would be lost on restart.
- **A heartbeat/stale-takeover claim.** It is not atomic (review #1); replaced
  by the OS lock.

## 8. Non-goals

- Message bodies in notices.
- Credentials in notices, the queue db, or new files.
- Waking on worker `waiting` parks.
- Treating a doorbell as delivery.

## 9. Follow-ups

- **F1** — `~/.claude/commands/codex-visual-qa.md` (outside the repo):
  - add the flag to the lead's and the member's MCP entries;
  - add the `external_set_wake` step to the prompt template;
  - replace "Poll every 2–3 minutes" with "end your turn; you will usually be
    woken; check `external_read` if nothing arrives";
  - **keep** the watcher, but it may use a long `--timeout` instead of the
    28 × 20 s loop.
- **F2** — Stop-hook rows trusting the native path (the former D4n/M4n). A hook
  can use `try_lock_handle` on the owner lock as a liveness probe. This needs
  the delivery semantics to be tightened first (review #3).
- **F3** — Codex-lead self-wake: a `codex_queue` host channel fed by a
  tool-supplied thread id and home.
- **F4** — Native wake on worker `waiting` parks.
- **F5** — A Windows Claude channel: overlapped, cancellable named-pipe I/O
  (`CreateFile` with `FILE_FLAG_OVERLAPPED`, `WriteFile`, and `CancelIoEx` on
  the deadline), with the auth line. It must be verified live, including a
  stalled pipe, before H1b is lifted.

### 9.1 Follow-up (separate PR): native transport for `send_message` / `follow_up_agent` downstream

It is opt-in through a sibling flag (for example
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`, which requires the master flag), with
byte-identical fallback.

**Today.** A downstream send to a spawned child uses `_guaranteed_send`
(`server_simple.py:3505`). That is kill-and-respawn via `backend.resume`
(`agent-messaging-protocol.md:896-910`), confirmed by a nonce receipt.

**Direction.** A **live interactive** child is woken in place:

- **A Claude child** is woken by **its own** server's notifier (§2.1 running in
  the child), never by a lead→child socket post. A lead→child post is a
  peer message, and a bypass-mode receiver's inbound default holds peer
  messages for approval.
- **A Codex child** is woken with `codex queue --thread <backend_session_id>`,
  which was verified on a TUI (`smoke-run-2026-09-25.md:35-42`).
- **Headless or dead children** keep the resume fallback.
- **Pi** is unchanged.

**Constraints (redesign, do not bypass):**

1. The socket and the queue are only doorbells; the store is the truth.
2. Guaranteed-path messages **never enter the actionable inbox**
   (`agent-messaging-protocol.md:251-263`), because of the double-execution
   risk and the count-cursor hazard. This needs a new exactly-once carrier,
   such as a per-child mailbox with its own cursor and nonce.
3. `delivered` still requires a consumption receipt carrying the nonce; a
   successful post is never a receipt. Idempotency, `delivery_status`,
   `deliver_pending`, and the rescan-before-resend rule are preserved.
4. Choosing native or resume is a per-attempt decision on the delivery row,
   under the lease.

**Upstream is already covered.** A child's upstream send lands in
`inbox-<parent>`, and the parent's own notifier from this PR wakes it when the
flag is on.

## 10. Answers to the review's §11 questions (adopted)

1. **Notifier and session recovery.** The notifier never recovers a session.
   A restarted lead needs one team tool call or `resume_session` before
   catch-up wake; this is documented.
2. **Registering a thread id.** Thread registration uses the token-authenticated
   `external_set_wake` (§2.3), not a `join_team` replay.
3. **Stop hooks.** D4n/M4n are split out as follow-up F2; D5/M5 already
   protect the idle path.
4. **Wire role.** Keep the smoke-tested `role:"user"`. There is no documented
   notice role; revisit if Claude publishes one.
5. **The Windows pipe address and I/O** are deferred to F5. This PR does no
   pipe I/O (H1b).
6. **`codex queue` and folder trust.** The queue path does not prompt for
   trust: there is no TUI, and config loads non-interactively
   (`session_archive_commands.rs:285-304`). It runs with `cwd=Path.home()`;
   the member's cwd is not needed.
