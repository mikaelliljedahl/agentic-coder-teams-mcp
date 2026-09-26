VERDICT: CHANGES_REQUESTED

1. [BLOCKER] `docs/features/native-session-wake/plan.md:324-345` — Atomic replacement does not atomically acquire the owner claim. Two servers can both read an absent/stale claim, replace it, and post; a live owner whose heartbeat is delayed past 120 seconds can also be displaced and continue posting. Add a per-reader cross-process claim lock or an atomic compare-and-claim protocol, recheck ownership immediately before posting, and test simultaneous starts and stale takeover. Do not hold the agents lock while posting.

2. [MAJOR] `docs/features/native-session-wake/plan.md:401-424` — `last_wake` is process-local and the plan does not say whether failed or timed-out queue attempts update it. If they do, the next send while the first line remains unread is reported `coalesced` without another queue attempt. Define the timestamp as the last *successful* queue insertion, serialize the decision for concurrent sends to the same member, and test failed-first-send/second-send plus two simultaneous sends. The append under `_agents_transaction` is correctly before the subprocess; keep that lock boundary (`src/claude_teams/server_simple.py:3460-3500`).

3. [MAJOR] `docs/features/native-session-wake/plan.md:237-260,476-479` — `session_info.native_wake.claude="active"` proves a socket/channel exists, not that Claude accepts notices. The [Claude socket contract](https://code.claude.com/docs/en/cross-session-messaging.md#the-sessions-inbox-socket) says `crossSessionInbound=refuse` still binds a socket but drops messages. The proposed tool text then tells the agent to stop arming `watch`, leaving it deaf. Report channel readiness separately from confirmed delivery, or retain the watcher fallback when inbound policy cannot be verified; add a refusal-mode test/contract case.

4. [MAJOR] `docs/features/native-session-wake/plan.md:225-239,401-420` — The Windows named-pipe path uses blocking `open()`/write without an effective per-attempt timeout; five retries only bound `ERROR_PIPE_BUSY`, and `timeout=5.0` does not constrain a stuck write. Use a Windows pipe API with bounded connect and write deadlines, or explicitly mark Windows unsupported until live W1/W2 verification. Also make the notice safe for the `.cmd` fallback: it contains parentheses at line 419, which `src/claude_teams/backends/codex.py:283-296` identifies as `cmd.exe` metacharacters. Test the actual shim fallback, not only mocked `open()`.

5. [MAJOR] `docs/features/native-session-wake/plan.md:350-364,410-420` — Same user and machine do not establish the same `CODEX_HOME`. Codex resolves that home before `thread/queue/add` (`codex-rs/tui/src/session_queue_commands.rs:37-40,54-60` in the supplied rust-v0.156.1 tree); a successful queue into another home can be reported `queued` while the member never wakes. Specify how the lead learns or configures the member's home, verify it before queueing, and test a mismatch. `CODEX_THREAD_ID` in shell commands is confirmed by `codex-rs/protocol/src/shell_environment.rs:7,152`.

6. [MINOR] `docs/features/native-session-wake/plan.md:366-400,504-510` — Replay of `join_team` expires with ticket retention even though a valid `member_token` remains usable (`src/claude_teams/server_simple.py:3063-3076,908-960`). Put thread-id registration/update on a token-authenticated member call so a Desktop thread can change later. The proposed `echo $CODEX_THREAD_ID` join prompt is Unix-only; include PowerShell syntax for Windows.

7. [MINOR] `docs/features/native-session-wake/plan.md:431-450,603-605` — Scrubbing the spawn environment through `BaseBackend._spawn_with_command` covers both spawn and resume (`src/claude_teams/backends/process_base.py:82-105`), but test 29 names spawn alone. Test resume and the actual MCP child environment under a spawned Claude host: the cited Claude document guarantees fresh socket export to hooks/Bash, while export to MCP servers is only observed. The empty inherited values must be shown to be replaced by the child's own socket.

8. [MINOR] `docs/features/native-session-wake/plan.md:257-271,293-322` — The notice policy does not define a minimum retry interval after a failed post, and the stated 60-second backoff covers only selected failure reasons. A persistent `OSError` or Windows write failure could retry each one-second poll, while each attempt increments `seq`. Apply one bounded backoff to every post failure and test it. The Claude document's repeat/rate limits are receiver protections, not a sender retry policy.

## §11 answers

1. Start the thread without recovering the session. The current `_active_session_id` recovery writes bindings and changes `_pending_recovery` (`src/claude_teams/server_simple.py:1177-1280`); a background scan must not silently choose a session. Document that a restarted lead needs one team tool call or explicit `resume_session` before catch-up wake.
2. Use a token-authenticated update on `external_read` or a small dedicated member tool. Join replay is useful for crash reconciliation but is not a durable re-registration path after seven-day ticket retention.
3. Split D4n/M4n into a follow-up unless the liveness claim and `active` semantics are tightened as above. Existing D5/M5 protect the idle path (`src/claude_teams/lead_wake.py:474-487`, `src/claude_teams/member_wake.py:219-233`); changing them is not needed to prove native delivery.
4. Keep the smoke-tested `role:"user"` wire message. The cited socket documentation specifies the auth line and own-child handling, but does not document a system/notice wire role; do not invent one. Revisit if Claude publishes one.

## Dispositions (round 1)

Applied in plan.md revision 2. Coordinator dispositions: all accepted.

| # | Severity | Disposition | What changed (plan.md sections) |
|---|---|---|---|
| 1 | BLOCKER | ACCEPTED | The heartbeat/stale-takeover claim is replaced by an OS-level exclusive lock per reader (`native-wake-<reader>.lock`), held for the target's lifetime through a new non-blocking `filelock.try_lock_handle` (flock `LOCK_EX\|LOCK_NB` / `msvcrt LK_NBLCK`). The lock is released automatically on process death, so there is no staleness timeout and no displacement of a live owner. Non-owners do not notify, and ownership is rechecked before each post. The agents lock is never held while posting. §2.1(e); tests 11 and 15; §7 |
| 2 | MAJOR | ACCEPTED | `last_success` counts only successful queue insertions. The decision is serialized by a per-member in-process `threading.Lock` across decide→queue→update; this is justified because only the lead identity's server writes member inboxes, and a cross-process race costs one harmless extra doorbell. The queue runs after the agents transaction exits. §2.2; tests 20, 21, 25 |
| 3 | MAJOR | ACCEPTED | `session_info` reports `claude_channel: "available"`, never delivery. Native wake is additive: docstrings call it a best-effort doorbell and keep the watcher guidance. D4n/M4n moved to follow-up F2. A contract test asserts the watcher guidance is retained, and smoke S6 covers `crossSessionInbound=refuse`. §0, §2.6, §5 S6, §9 F2, §10.3; test 27 |
| 4 | MAJOR | ACCEPTED | Windows pipe post runs in a daemon worker thread joined with a hard deadline. On timeout it is abandoned and backs off, with at most one outstanding poster thread per target (a bounded leak). W1/W2/W3 are live gates before merge; if W1 fails, the channel is reported `unsupported_platform`. The Codex notice contains no cmd.exe metacharacters or quotes. Tests cover the charset and the `.cmd` shim argv path. §2.1(b), §2.2, §5; tests 6, 18, 19 |
| 5 | MAJOR | ACCEPTED | The member reports its Codex home with the thread id (Unix and PowerShell commands). The lead runs `codex queue` with `CODEX_HOME=<home>` (cited `find_codex_home`) and first verifies the thread in that home: a `state_5.sqlite` `threads` row (`STATE_DB_FILENAME`, `state/src/sqlite.rs:33`), then a rollout filename glob (`rollout_file_name.rs:62-73`). A mismatch, an archived thread, or an unverifiable thread gives `unverified_thread` and no queue. §2.2, §2.3; test 23 |
| 6 | MINOR | ACCEPTED | The `join_team` parameter is replaced by the token-authenticated `external_set_wake(member_token, codex_thread_id, codex_home)`. It is an external tool that can be re-called at any time and works under `EXTERNAL_ONLY`. It is registered only when the flag is on, so the flag-off tool list stays byte-identical; the member's MCP entry therefore needs the flag too. The join prompt (flag on only) gives Unix and PowerShell commands. §2.3; tests 16, 28, F6 |
| 7 | MINOR | ACCEPTED | The scrub test covers spawn and resume (F3). A spawned Claude host's MCP child having its own fresh socket is manual smoke S2, because it is not unit-testable without a real `claude`. §2.4, §5 |
| 8 | MINOR | ACCEPTED | One bounded exponential backoff applies to every post failure, Claude and Codex (2 s doubling, 300 s cap, reset on success). `seq` never advances on failure. A test checks that a persistent `OSError` does not retry every poll. §2.1(d); test 10 |

§11 answers 1–4: adopted as §10.1–10.4. The author's own open questions are
answered in §10.5, which covers the Windows pipe address through live gate W1
before merge. §10.6 covers `codex queue` folder trust: the source shows no TUI
and non-interactive config loading (`session_archive_commands.rs:285-304`), so
it runs with `cwd=Path.home()`.

## Round 2

VERDICT: CHANGES_REQUESTED

### Round 1 finding status

1. **RESOLVED.** The lifetime OS lock in `plan.md:233-258` removes the read/replace claim race. `filelock.py:43-58,61-84` supplies the existing platform primitives; the new try-lock must seek to byte 0 on Windows and return `False` only for contention, not every `OSError`. Tests 11 and 15 cover competing owners and release on death.
2. **NOT RESOLVED.** Successful-only `last_success` and the per-member lock fix failed-attempt coalescing (`plan.md:295-316`), but the shared two-second coalesce has no timer on the send path; finding R2-A below.
3. **RESOLVED.** `session_info` now reports channel availability, and watcher guidance remains (`plan.md:47-51,436-457`). D4n/M4n moved to a follow-up; refusal smoke S6 is specified.
4. **NOT RESOLVED.** The safe Codex notice and shim test address the argv issue, but the Windows thread join bounds the caller only; finding R2-C below.
5. **RESOLVED.** `external_set_wake` carries the member's Codex home and the lead sets `CODEX_HOME` before queueing (`plan.md:318-325,369-404`). Local thread verification is an additional guard, with a new issue below.
6. **RESOLVED.** Token-authenticated `external_set_wake` is independent of ticket retention and the join prompt covers Unix and PowerShell (`plan.md:369-404`). Import-time flag-only registration preserves the flag-off tool list; both MCP entries must enable the flag, as the prompt now states.
7. **RESOLVED.** Spawn and resume are both covered by F3, with the real spawned-Claude MCP environment checked by smoke S2 (`plan.md:406-419,497-498,614-615`). This remains a required implementation gate.
8. **RESOLVED.** All post failures now share exponential backoff and failures do not advance `seq` (`plan.md:225-231,528-529`).

### New findings

R2-A. [BLOCKER] `docs/features/native-session-wake/plan.md:199-214,307-316` — The same `plan_notice` requires a two-second coalesce for Codex, but Codex decision-making runs only during `send_message`. A lone send sets `first_new`, returns before the deadline, and no timer invokes the decision again; the member is never queued. Make the first Codex send queue immediately, or schedule a bounded deferred queue independent of another send. Test one isolated send with real time advancing past the window; test 17's immediate `queued` expectation (`plan.md:552-554`) currently contradicts the algorithm.

R2-B. [MAJOR] `docs/features/native-session-wake/plan.md:295-316,379-392` — A send copies the registration under the agents lock, releases it, then waits on the per-member lock and can queue to an old `thread_id` after `external_set_wake` has changed or cleared that registration. The plan also leaves `watch_member`'s registry-lock position relative to `_member_operation`'s agents lock unspecified (`src/claude_teams/server_simple.py:908-910`). Define one order for member lock, agents lock, and target-registry lock; revalidate a registration generation before queueing, and test a send racing re-registration and leave. Never wait on the per-member lock while holding the agents lock or hold a registry lock while acquiring it.

R2-C. [MAJOR] `docs/features/native-session-wake/plan.md:187-196,516-522` — `Thread.join(timeout)` returns while a blocked Windows named-pipe `open` or write continues. One stuck worker can permanently suppress future notices for that target, and a late write can occur after a session switch or leave. Use cancellable overlapped pipe I/O with a real operation deadline, or explicitly ship the Windows channel disabled until that is implemented and verified. W1's successful-post test does not exercise a stalled pipe.

R2-D. [MINOR] `docs/features/native-session-wake/plan.md:352-367,565-571` — The direct `state_5.sqlite` query should specify a short SQLite busy timeout, closing the read-only connection before the queue call, and fallback on lock/schema errors. Codex uses WAL with a five-second busy timeout (`codex-rs/state/src/sqlite.rs:297-330`), and `threads` also has an `archived` column (`codex-rs/state/migrations/0001_threads.sql:1-14`). Check that column as well as the rollout path; test an uncheckpointed WAL row, a locked/schema-changed DB, and that verification creates or changes no Codex DB files. Keep the rollout glob fallback for schema drift.

R2-E. [NIT] `docs/features/native-session-wake/plan.md:326-330,552-554` — The design specifies `cwd=Path.home()` but test 17 expects `cwd=home`, where `home` is the member's `CODEX_HOME`. Align the test with the intended working directory.

## Dispositions (round 2)

Applied in plan.md revision 3. Coordinator dispositions: all accepted. The
round-1 try-lock note (Windows `seek(0)`; return `False` only for contention)
is also applied in §2.1(e) and test 15.

| # | Severity | Disposition | What changed (plan.md sections) |
|---|---|---|---|
| R2-A | BLOCKER | ACCEPTED | The Codex send path uses `plan_notice` with `coalesce=0`: the first send with no outstanding wake queues immediately, and later sends are `coalesced` only while the member still has unread and `last_success` is within the re-notice floor. The 2 s coalesce stays only in the Claude notifier thread, whose tick re-evaluates. §2.1(c), §2.2 step 4; test 17 fixed; new test 17a (isolated single send, real time past the window) |
| R2-B | MAJOR | ACCEPTED | One global lock order: per-member lock → agents lock (short snapshot) → release → queue. The per-member lock is never awaited while holding the agents lock, and the target-registry lock is a leaf. `external_set_wake` bumps `codex_wake.generation` under the agents lock. The send revalidates status and generation (a second short agents lock) immediately before the subprocess, giving `stale_registration` or `coalesced` on mismatch. `watch_member` is called after `_member_operation`'s `with` block exits. §2.1(f) lock order, §2.2 "Global lock order" and steps 1–6, §2.3 Storage/Return; tests 26a (re-registration race), 26b (leave race), 26c (lock order, no deadlock) |
| R2-C | MAJOR | ACCEPTED (simplest option) | The Claude socket channel is POSIX-only: on native Windows, H1b reports `unsupported_platform` with no pipe I/O and no host resolution, and the notifier thread is not started. The worker-thread design is removed. Overlapped, cancellable pipe I/O is follow-up F5. Windows keeps the Codex `codex queue` path (a real subprocess timeout) and the flag-off identity guarantee; W1–W3 now verify exactly those. §0 E, §2.1(a)(b), §5, R7, §9 F5, §10.5; test 6 rewritten |
| R2-D | MINOR | ACCEPTED | Verification opens the db only if it exists, via URI `mode=ro` (not `immutable`, so WAL rows are visible), with a 0.5 s busy timeout. It selects `rollout_path, archived` (cited `0001_threads.sql:1-14`, WAL and busy timeout `sqlite.rs:297-330`), closes the connection before the queue call, and falls back to the rollout glob on lock or schema errors. §2.2 verification step 2; test 26d (uncheckpointed WAL row, locked db, schema drift, `archived=1`, connection closed first, no Codex db files created or changed) |
| R2-E | NIT | ACCEPTED | Test 17 now asserts `cwd=Path.home()` (the lead's home, not the member's `CODEX_HOME`). §4.2 test 17 |
| Incident (2026-09-26) | new requirement | ACCEPTED | When a session becomes active, `native_wake.session_activated()` sets an event: from `_active_session_id` (recover, auto-adopt, create) and from `resume_session`. The notifier then runs an immediate backlog check and posts a catch-up notice with `coalesce=0` if anything is unread. There is still no side-effectful recovery from the thread (§11 Q1 kept). The restart procedure (call `session_info`/`resume_session` first) is documented, and the flag-on `resume_session`/`session_info` docstrings state that the backlog notice follows. §2.1(c)(f), §2.6, R11, S4; test 15a |

## Round 3

VERDICT: CHANGES_REQUESTED

### Round 2 finding status

- **R2-A — RESOLVED.** Codex uses `coalesce=0` and queues within the original send call (`plan.md:204-211,364-379`). Tests 17/17a explicitly exercise an isolated send.
- **R2-B — NOT RESOLVED.** The lock order and leaf target-registry lock now prevent the stated inversion (`plan.md:318-325,346-356`), and generation revalidation narrows the stale snapshot window. However, notice state is not tied to that generation; R3-A below can still strand the replacement thread. The remaining post-check race is explicitly accepted at `plan.md:385-387`, rather than eliminated.
- **R2-C — RESOLVED.** Windows Claude pipe I/O and the worker thread are removed; the channel is unsupported and the notifier does not start (`plan.md:60-66,191-195,606-609`). Codex queue remains available. One platform-gate ordering inconsistency is noted below.
- **R2-D — RESOLVED.** The database check now specifies read-only URI access, a 0.5-second busy timeout, the `archived` column, explicit connection closure, and schema/lock fallback (`plan.md:430-451`). Test 26d covers WAL visibility and database side effects. These remain implementation checks, not already-run evidence.
- **R2-E — RESOLVED.** Test 17 now asserts the lead's `Path.home()` working directory (`plan.md:648-651`).

### New findings

R3-A. [MAJOR] `docs/features/native-session-wake/plan.md:336-339,364-389,473-478` — `notified`/`last_success` and backoff are per member, while the queue destination changes by registration generation. After a successful wake to thread A, registration changes to B while the inbox remains unread; the next send sees A's outstanding notification and returns `coalesced` without waking B. Revalidation is skipped because the decision already deferred. Key notice state, verification cache, and backoff by registration generation; initialize a new generation from current cursors. Specify how a clear retains the monotonically increasing generation so clear→set cannot reuse an old value. Test successful wake to A → change to B without draining → one send queues to B, plus clear→set and failure-backoff→change cases. Test 26a currently permits `stale_registration` but does not cover this sequential case.

R3-B. [MINOR] `docs/features/native-session-wake/plan.md:158-174,536-542,606-609,761-763` — H1b appears after H1 and H2 in the decision table, yet the text and tests require Windows to return `unsupported_platform` without host resolution. Following the table would return `no_socket` on a Windows host lacking socket env, or run CIM before reaching H1b. Put H1b immediately after H0, and test Windows both with and without socket env. Explicitly name POSIX-only Claude wake in the changed tool descriptions and assert that wording in the contract tests.

R3-C. [MINOR] `docs/features/native-session-wake/plan.md:286-308,632-639` — Catch-up activation is correctly signalled from tool threads without background recovery, but every `resume_session` assignment signals even when the same session is already active (`src/claude_teams/server_simple.py:5946-5968`). Define catch-up as a transition to a new target/activation generation, preserve outstanding-notice and failure-backoff state for repeated activation of the same target, and consume the event before processing the latest session snapshot. Add tests for repeated same-session resume, rapid S1→S2 activation, and an activation arriving during a scan. These pin the immediate-backlog requirement without enabling duplicate notice storms or losing the latest activation.

## Dispositions (round 3)

Applied in plan.md revision 4. Coordinator dispositions: all accepted as the
reviewer recommends.

| # | Severity | Disposition | What changed (plan.md sections) |
|---|---|---|---|
| R3-A | MAJOR | ACCEPTED | The Codex notice state (`notified`/`last_success`), thread-verification cache, and backoff are keyed by `(session, member, codex_wake.generation)`. A new generation is initialised from the member's current cursors, with no inherited outstanding state or backoff. `generation` is monotonic for the life of the record: a clear keeps the field with `thread_id: null` and `generation: n+1`, so clear→set gives `n+2` and never reuses an old value. §2.1(d), §2.2 "Notice state is keyed by registration", §2.3 Storage; test 26e (wake A → change to B undrained → queues to B; set→clear→set; backoff→change) |
| R3-B | MINOR | ACCEPTED | H1b now comes directly after H0, before any env read or host resolution. Test 6 covers Windows with and without the socket env. Changed tool descriptions name Claude session wake as POSIX-only, and contract test 27 asserts that wording. §2.1(a), §2.6; tests 6, 27 |
| R3-C | MINOR | ACCEPTED | Catch-up fires only on a transition to a new `(session_id, IDENTITY)` target. A repeated same-session activation keeps the notice state, outstanding window, backoff, and owner lock. The notifier consumes the event before reading the latest `_session_id`, so an activation during a scan is processed on the next iteration. Rapid S1→S2 processes only the latest session. §2.1(f) "Activation semantics"; tests 15b (repeated same-session resume), 15c (rapid S1→S2), 15d (activation during a scan) |

## Round 4

VERDICT: APPROVED

### Round 3 finding status

- **R3-A — RESOLVED.** Notice state, verification cache, and backoff now include the registration generation; a new generation starts from current cursors (`plan.md:351-360`). Clear retains a tombstone and increments the generation (`plan.md:499-506`). Test 26e covers undrained A→B, clear→set, and backoff→change (`plan.md:745-752`).
- **R3-B — RESOLVED.** H1b immediately follows H0 and precedes socket-env/host checks (`plan.md:158-166`). The contract explicitly states POSIX-only Claude wake (`plan.md:569-576`), and tests cover Windows with and without socket env and no notifier start.
- **R3-C — RESOLVED.** Catch-up is transition-only, repeated activation preserves state and the owner lock, and the event is cleared before reading the latest session (`plan.md:301-318`). Tests 15b–15d cover repeated resume, rapid transitions, and activation during a scan (`plan.md:680-689`). The existing pre-post session check remains applicable.

### Minor item for implementation

R4-A. [MINOR] `docs/features/native-session-wake/plan.md:393-395,503-506` — Clearing now leaves a truthy `codex_wake` dictionary with `thread_id: null`, whereas decision step 3 still says to skip when there is "no codex_wake". Use the explicit registered predicate (`thread_id` non-null) for every send eligibility check, and assert that a send after clear omits `wake` and invokes neither verification nor the runner. This is a wording/test alignment item; the intended predicate is already specified at line 506.

No new blocker or major finding in the reviewed revisions. Approval is for the plan; the stated red-first tests and smoke gates remain required during implementation.
