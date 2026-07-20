# Plan — Pi-lead real-time wake via win-agent-teams extension

Feature branch: `feature/pi-lead-inbox-wake`. Worktree:
`agentic-coder-teams-mcp.pi-lead-inbox-wake`. Built on `main` @ `3085295`
(includes the `watch` CLI from #25).

See [requirement.md](requirement.md) for the agreed requirement and the
investigation behind ruling out ACP / Event Bus / coms-net.

**Revision:** v4, incorporating disposition of `plan-review-1.md` (56),
`plan-review-2.md` (76), and `plan-review-3.md` (85). Finding-by-finding
disposition in §9; the exact, upstream-verified Pi API this plan is written against
is pinned in §10.

## 1. Scope

**In scope**
- A Pi extension (`win-agent-teams` Pi plugin, TypeScript) that runs inside a Pi
  lead session, wakes the lead when a worker sends a message, and drives it to
  read and act — via a cursor-aware single-flight state machine.
- Minimal, additive **read-only** CLI surface this repo must expose so the
  extension can **shell out** (it cannot call MCP tools directly — see §3.0):
  `session-dir` (discovery) and `inbox-status` (non-consuming cursor-generation
  probe), plus an additive `--reader` flag on `watch`. Plus an additive
  `session_dir` field on the `session_info()` MCP tool for the lead LLM's benefit.

**Out of scope** (see requirement §5)
- ACP, new spawn backend, changed server/CLI wake semantics, Pi-as-worker,
  symmetric peer-to-peer.

## 2. Current behavior (verified)

Citations against `main` @ `3085295` (independently re-verified in
`plan-review-1.md` §8):

- Session layout: `_SESSION_BASE = Path.home()/".claude"/"agent-sessions"`
  ([server_simple.py:77](../../../src/claude_teams/server_simple.py));
  `_session_dir(session_id) = _SESSION_BASE/session_id`
  ([:198](../../../src/claude_teams/server_simple.py)); lead inbox
  `inbox-team-lead.jsonl` via `_inbox_file(session_id, "team-lead")` where
  `team-lead` comes from the caller/identity, not the helper
  ([:214](../../../src/claude_teams/server_simple.py)).
- Lead identity: `IDENTITY` is captured **at import** from `AGENT_NAME`, else
  `team-lead` ([:45-58](../../../src/claude_teams/server_simple.py)). A fallback,
  not proof the env is unset.
- `watch` CLI ([cli.py:183](../../../src/claude_teams/cli.py)): default `--inbox`
  wakes on unread for the reader `os.environ.get("AGENT_NAME","").strip() or
  "team-lead"`, evaluated dynamically ([cli.py:222-224](../../../src/claude_teams/cli.py)).
  On wake `_emit_wake` writes **one** compact JSONL line and exits 0
  ([cli.py:177-179](../../../src/claude_teams/cli.py)); timeout = exit 2, silent
  ([cli.py:307-308](../../../src/claude_teams/cli.py)). The inbox check is
  **non-consuming** — it calls `unread_sender_counts()` and never saves a cursor
  ([cli.py:228,255](../../../src/claude_teams/cli.py),
  [messaging.py:53-64](../../../src/claude_teams/messaging.py)).
- The cursor is advanced **only** by `read_messages`
  ([server_simple.py:1345,1458-1475](../../../src/claude_teams/server_simple.py)),
  which drains at most 50 by default and reports `has_more`
  ([:1440-1456](../../../src/claude_teams/server_simple.py)). Its lock is
  **process-local** and assumes a single cursor writer
  ([:245-249](../../../src/claude_teams/server_simple.py)).
- `session_info` returns `session_id`/`identity`/`cwd`/counts but **not**
  `session_dir` ([server_simple.py:1852,1869-1889](../../../src/claude_teams/server_simple.py)).
- Session discovery is **not** simply "empty until first spawn":
  `_active_session_id(create=False)` first attempts recovery / auto-adoption
  ([:627-645,555-599](../../../src/claude_teams/server_simple.py)); a new session
  is created lazily only when nothing was recovered, and only `spawn_agent` passes
  `create=True` ([:1225-1227,1097-1104](../../../src/claude_teams/server_simple.py)).

Gap today: nothing wakes a **lead** Pi session — the lead must run `watch` itself.

## 3. Proposed design

### 3.0 Feasibility constraints (decide the whole shape)

Verified against upstream Pi (`main`, 2026-07-19; exact signatures quoted in §10):

- **The extension cannot invoke MCP tools directly.** `pi.getAllTools()` returns
  tool *metadata*, not callable implementations. `pi.exec(command, args, options)`
  runs subprocesses. **→ the extension shells out to the `win-agent-teams` CLI**
  (resolves OQ2; drives §3.2). An additive `session_dir` on `session_info()` is
  still added for the lead LLM, but it is not the extension's data path.
- **`ExecOptions` = `{ signal?, timeout?, cwd? }` — no `env`.** The child inherits
  Pi's environment; the extension **cannot** unset `AGENT_NAME` via `pi.exec`. **→
  the CLI must take an explicit `--reader` override** (additive; defaults to the
  current env-based behavior) so the extension passes `--reader team-lead` and never
  depends on the ambient env (§3.2, §3.3). Fixes review-2 blocker 4.1.
- **`isIdle()` is only on an event/command `ExtensionContext`, not on
  `ExtensionAPI`.** A free-running background loop cannot query idle state. **→ the
  extension always injects with `deliverAs: "steer"`** (immediate turn when idle,
  queued after the current turn when busy — one mode covers both), so no background
  `isIdle()` call is needed (§3.1). Fixes review-2 blocker 4.3, supersedes OQ4.
- **`sendMessage(...) : void`** — returns no turn handle. **→ the extension does not
  correlate turns at all.** Re-arm is driven purely by observed **cursor
  generation** progress (§3.1), which also removes the mid-turn correlation race
  (review-1 finding #2, review-2 blocker 4.4).

Consequence for cursor ownership: the extension must **never** advance the cursor
(a second writer would violate the process-local single-writer assumption,
[server_simple.py:245-249](../../../src/claude_teams/server_simple.py)). **The lead
is the single cursor owner.** The extension only (a) blocks on `watch`, (b) injects
a wake, and (c) reads a **non-consuming generation snapshot** to know when re-arm is
safe.

### 3.1 Pi extension `win-agent-teams` (TypeScript)

**Lifecycle (owned controller — fixes review-3 blocker 4.5).** `session_shutdown`
carries **no** `AbortSignal`, and `ExtensionContext.signal` is the per-turn signal
(often `undefined` while idle), not a lifetime signal. So: register `session_start`
and `session_shutdown` at activation; on `session_start` create an
**extension-owned `AbortController`**, start exactly one loop, and pass
`controller.signal` to every `pi.exec` and every backoff wait; on `session_shutdown`
call `controller.abort()` and await loop completion. Reset the controller/loop refs
idempotently across `reason` = reload/new/resume/fork. Exactly one live
`watch`/`inbox-status` child at any time (single-flight).

**One injection per generation (resolves review-3 §4.6).** A generation is injected
**exactly once**; there is no re-inject-within-a-generation edge. ACK_WAIT waits for
acknowledgement or, on a no-progress budget (time/probe count), transitions to
ACK_STALLED. Only a genuinely *newer* generation is ever injected again.

**Generation model (the ACK contract).** A wake defines a *generation*: at wake
time the extension reads the non-consuming **inbox-status** snapshot (§3.2),
`{sender:{total,cursor,unread}}`, and records `target_total[sender]` for every
sender that currently has `unread > 0`. The generation is **acknowledged** when, for
**all** captured senders, `min(current.cursor, current.total) >= target_total[sender]`
— i.e. the lead's cursor has reached at least the message count present at wake.
This is race-safe under concurrent arrivals: later messages only raise `total`, they
never hide consumption of the captured generation (fixes review-2 blocker 4.2). Bare
unread counts are **not** used for ACK.

State machine:

```
DISCOVERING ─session─▶ WATCHING ─reason=message─▶ WAKE_PENDING ─inject once─▶ ACK_WAIT
     ▲                    ▲                                                      │
     │ session_id changed │ generation acknowledged                             │
     │                    └──────────────────────────────────────────────────┐ │
     └───────────(rebind)                                     retry budget spent │
                          ACK_STALLED ◀───────────────────────────────────────┘ │
                              │  strictly-newer generation observed → one inject ┘
                              └─ session_id changed → DISCOVERING
```

- **DISCOVERING** — `pi.exec("win-agent-teams", ["session-dir"])` (§3.2) → `{session_id,
  session_dir, identity}`. **Fail closed unless `identity == "team-lead"`.** While no
  session exists (exit 3), poll with capped backoff (R3). Treat `session_id` as a
  **changing binding** (§3.3).
- **WATCHING** — `pi.exec("win-agent-teams", ["watch", session_dir, "--reader",
  "team-lead", "--timeout", T])`. Parse strictly-but-tolerantly (R6): exit 0 + one
  JSON line `reason=message` → capture generation → WAKE_PENDING; exit 2 → re-arm
  WATCHING; `reason=output`/`waiting` → ignored in v1 (OQ6); other exit / malformed
  / stderr noise → capped-backoff retry. Reject a `path` ≠ the discovered lead inbox.
- **WAKE_PENDING → inject once** — inject exactly one custom message and go to
  ACK_WAIT:
  ```
  pi.sendMessage(
    { customType: "win-agent-teams/wake",
      content: "📨 New inbox message(s) from <senders>. Call read_messages and keep "
             + "draining while has_more, then act on each.",
      display: true },                            // display is a boolean in CustomMessage (review-3 4.3)
    { triggerTurn: true, deliverAs: "steer" })    // steer: immediate if idle, else after current turn
  ```
- **ACK_WAIT** — re-check discovery (§3.3), then poll the non-consuming
  **inbox-status** snapshot on a capped-backoff timer:
  - generation acknowledged (all captured senders) → WATCHING.
  - progress but not complete (a `>50` backlog drain, finding #8) → keep waiting;
    progress refreshes the budget so genuine draining is never cut off.
  - **no progress within a capped retry budget** → ACK_STALLED (do **not** re-arm an
    inbox watcher — that would re-fire the same level-triggered unread and reset the
    budget; fixes review-2 blocker 4.1).
- **ACK_STALLED** — the captured generation is unread and the lead is not draining.
  Do **not** run an inbox-enabled `watch`. Poll inbox-status on backoff and evaluate,
  **in this order** (fixes review-3 blocker 4.1):
  1. **Late drain →** if every captured target is now acknowledged
     (`min(cursor,total) >= target_total` for all captured senders, treating a
     sender that vanished as satisfied), transition to **WATCHING**. This covers the
     lead reading the stale generation later without any new injection.
  2. **New generation →** otherwise compare the **union** of captured and current
     sender keys, treating a missing captured target as **0**. A newer generation
     exists only when at least one sender has `current.total > captured_target`
     **and currently has `unread > 0`** (the unread guard avoids re-injecting for
     messages that arrived and were independently consumed between probes). Capture a
     fresh `target_total` for **every currently-unread sender**, give the new
     generation its own budget, and inject once → ACK_WAIT.
  3. **Session change →** a changed `session_id` → DISCOVERING.

  Paid turns are thus bounded **per message generation**; a first message from a new
  sender while another generation is stalled is recognized via the union/default-0
  rule (not missed forever).

The extension never reads message bodies and never advances the cursor. The lead,
via its own MCP `read_messages`, is the sole reader/cursor owner (reconciled Design
A; see §9 and requirement §3).

### 3.2 Repo change (Python) — read-only CLI shell-out surface

All additive. The two new subcommands are read-only and non-consuming; the existing
`watch` gains one additive optional flag; no wake/read/send semantics change for
existing callers.

1. **`win-agent-teams session-dir`** (new subcommand,
   [cli.py](../../../src/claude_teams/cli.py)) — **non-creating / discovery-only**
   (`_active_session_id(create=False)`; it never creates a session directory,
   though fallback auto-adoption may persist a session *binding* —
   [server_simple.py:627-644](../../../src/claude_teams/server_simple.py) — so it is
   "discovery-only", not literally side-effect-free). Contract:
   - success: exit 0, stdout = exactly one line `<session_id>\t<session_dir>\t<identity>`,
     no Rich/decoration, nothing on stderr;
   - no session: exit 3, empty stdout;
   - internal error: exit 1, message on stderr only.
2. **`win-agent-teams inbox-status <session_dir> [--reader NAME]`** (new subcommand)
   — the ACK generation probe. Uses the existing non-consuming messaging helpers
   (`read_inbox_by_sender` + `load_inbox_cursors`, clamped like
   [messaging.py:53-64](../../../src/claude_teams/messaging.py)); **never** writes a
   cursor. Contract:
   - `--reader` defaults to `team-lead`;
   - success: exit 0, stdout = **exactly one JSON object**, schema-versioned:
     `{"schema":"inbox-status/1","reader":"team-lead","senders":{"<from>":{"total":N,"cursor":M,"unread":K}}}`
     (`unread = total - min(cursor,total)`); empty inbox → `"senders":{}`;
     nothing on stderr;
   - bad/nonexistent/outside-base `session_dir`: exit 4, stderr message, no stdout;
   - internal error: exit 1, stderr only.
3. **`win-agent-teams watch … --reader NAME`** (additive flag on the existing
   `watch`) — overrides the inbox reader instead of relying on
   `os.environ["AGENT_NAME"]` ([cli.py:222-224](../../../src/claude_teams/cli.py)).
   Omitted → current env-based behavior (backward compatible). The extension always
   passes `--reader team-lead`, so it does not depend on `ExecOptions` supporting an
   `env` override (which it does not, §3.0/§10). Validate `--reader` (and the
   `inbox-status` reader) against the existing safe-agent-name constraints
   (reject path separators / control chars) before interpolating it into inbox/cursor
   filenames (review-3 §4.4 hardening).
4. **`session_info()` gains `session_dir`** — add `session_dir:
   str(_session_dir(session_id))` to the active return
   ([server_simple.py:1882-1889](../../../src/claude_teams/server_simple.py)) **and**
   `session_dir: ""` to the no-session return
   ([:1869-1877](../../../src/claude_teams/server_simple.py)); update the docstring
   ([:1853-1860](../../../src/claude_teams/server_simple.py)). Purely additive; for
   the lead LLM, not the extension's path.

### 3.3 Identity / session binding

- **Reader identity via flag, not env.** The extension passes `--reader team-lead`
  to `watch`/`inbox-status` (§3.2.3), so it never manipulates the child env
  (`ExecOptions` has no `env`, §3.0). It additionally **validates** `identity ==
  "team-lead"` from `session-dir` and **fails closed** otherwise (finding #7).
- **`session_dir` only from the CLI**, never hardcoded. **Refresh timing (fixes
  review-2 blocker 4.5):** re-run `session-dir` (a) after every `watch` exit and
  (b) immediately before every injection. If `session_id` changed (recovery /
  resume / new session), abort the current generation, tear down the watcher, and
  re-enter DISCOVERING — never inject against the old session (finding #6).

## 4. Files affected

| File | Change |
|------|--------|
| `src/claude_teams/cli.py` | New read-only subcommands `session-dir` and `inbox-status`; additive `--reader` flag on `watch`. |
| `src/claude_teams/server_simple.py` | Additive `session_dir` on both `session_info()` return branches + docstring. |
| `tests/test_session_recovery.py` | `session_info` `session_dir` (both branches); `session-dir` CLI incl. recovery/no-session/no-create. |
| `tests/test_cli_watch.py` | `watch --reader` override; new `inbox-status` CLI (schema, non-consuming, exit codes). |
| `pi-extensions/win-agent-teams-wake/` (new, TS) | The extension: manifest, single-flight state machine, `pi.exec` calls, injection, tests. `package.json` pins `@earendil-works/pi-coding-agent` to an exact version recorded at implementation start; written against the API in §10. Placement decided (OQ5). |
| `docs/features/pi-lead-inbox-wake/*` | plan/review/implementation artifacts. |

## 5. Risks

- **R1 Unbounded paid-turn loop.** Re-arm that lets a level-triggered inbox `watch`
  re-see the same unread would reset the budget forever. **Mitigation:** ACK is a
  cursor *generation* threshold (§3.1); on budget exhaustion the machine enters
  **ACK_STALLED**, which never runs an inbox watcher for an unread generation and
  only re-injects for a **strictly newer** generation. Paid turns bounded per
  generation. Tested (T5, T5b, T5c). Fixes review-2 blocker 4.1.
- **R2 ACK race under concurrent arrivals.** A consume + a new arrival can leave a
  bare unread count unchanged. **Mitigation:** ACK uses per-sender `{total,cursor}`
  and requires `min(cursor,total) >= target_total` for all captured senders; later
  arrivals only raise `total` (§3.1). No turn correlation is used (`sendMessage`
  returns void). Tested (T5d). Fixes review-2 blocker 4.2.
- **R3 Session not yet created / changes after startup.** Discovery tolerates "no
  session yet" and rebinds when `session_id` changes. **Mitigation:** DISCOVERING
  backoff + refresh after every watch exit and before every injection + watcher
  teardown (§3.1, §3.3). Tested (T6, T6b).
- **R4 Pi API mismatch.** Written against the exact upstream signatures pinned in
  §10 (`ExecOptions` has no `env`; `sendMessage` returns void; `isIdle` is
  event-context only). `package.json` pins an exact Pi version; a thin adapter
  isolates the surface. (Review-2 blocker 4.)
- **R5 Two cursor writers.** The extension must never call a consuming read. Only
  the lead advances the cursor. `watch`/`inbox-status` are non-consuming by
  construction.
- **R6 Subprocess robustness / lifecycle.** Missing binary, malformed/extra stdout,
  non-0/non-2 exit, timeout, cancellation, shutdown → exactly one live child, capped
  backoff, teardown via the **extension-owned `AbortController`** aborted from
  `session_shutdown` (§3.1, §10). Tested (T10, T11). Fixes review-2 blocker 5 /
  finding #10.

## 6. Test cases (red-green)

**Python (this repo):**
- **T1** `session_info()` returns `session_dir == str(_session_dir(session_id))`
  (active) and `""` (no session); existing fields unchanged.
- **T2** `session-dir` CLI: prints `id\tdir\tidentity` (exit 0) for an
  active/recovered session; empty + exit 3 when none; internal error → exit 1 +
  stderr only; **does not create** a session (assert no new dir under
  `_SESSION_BASE`); respects recovery/auto-adoption.
- **T2b** `inbox-status` CLI: emits the `inbox-status/1` JSON schema with correct
  `{total,cursor,unread}` per sender for `--reader team-lead`; **non-consuming**
  (a following `read_messages` still returns the same messages); `"senders":{}` when
  empty; bad/outside-base `session_dir` → exit 4 + stderr.
- **T3** Regression: existing `session_info`/watch/read tests green; `watch` with no
  `--reader` behaves exactly as before.
- **T3b** `watch --reader team-lead` overrides env: with `AGENT_NAME=other` in the
  environment, the flag still watches the `team-lead` inbox.

**Pi extension (TS, own suite, mocked `pi.exec`/`sendMessage`):**
- **T4** Parse: `reason=message` (exit 0) → capture generation → inject; exit 2 →
  loop; `output`/`waiting` → no inject; malformed line / stderr / other exit →
  backoff, no tight loop; wrong-inbox `path` rejected.
- **T5** Single generation → exactly one injection; re-arm (→WATCHING) only after the
  generation is acknowledged (`min(cursor,total) >= target_total`).
- **T5b** Lead never reads (no progress) → **exactly one** injection, then
  ACK_STALLED; **no inbox watcher is started while the generation is unread**, so no
  budget reset (the review-2 blocker-4.1 regression test).
- **T5c** `>50` backlog: cursor advances across probes but stays `< target_total` →
  extension keeps waiting; progress refreshes budget; no premature re-inject.
- **T5d** Concurrent arrival race: consume 1 + arrive 1 leaves bare unread
  unchanged, but generation ACK (cursor/total) still resolves correctly (blocker
  4.2).
- **T6** "No session yet" (exit 3) → waits, starts watching once `session-dir`
  returns one.
- **T6b** `session_id` changes between refreshes → current generation aborted, old
  child torn down, rebinds to the new session; never injects against the old one.
- **T7** ACK_STALLED **late drain**: lead reads the stalled generation with no new
  injection → next probe acknowledges captured targets → returns to WATCHING
  (review-3 4.1).
- **T7b** ACK_STALLED **new sender while stalled**: sender B's first-ever message
  arrives while stalled on A → union/default-0 rule recognizes B as a new generation
  → exactly one injection; not missed forever.
- **T7c** ACK_STALLED **arrive-then-consume between probes**: a message arrives and
  is independently consumed before the next probe → the `unread>0` guard suppresses
  a redundant injection.
- **T8** Two senders in one generation: ACK requires **both** to reach their
  captured `target_total`.
- **T10** `session_shutdown` fires → the **extension-owned** `AbortController` is
  aborted; the child receives the signal and is terminated; no further injection.
- **T11** At most one live `watch`/`inbox-status` child at any time across the whole
  lifecycle, including across `session_start`/reload/resume (single-flight, owned
  controller reset idempotently).

**Manual / e2e:**
- **T9** Bare `pi` + loaded extension as lead; spawn a worker; worker
  `send_message` → the lead is woken and drains its inbox without anyone manually
  starting a watcher. (The end-to-end fixture that proves FR1–FR3.)

## 7. Open questions — dispositions

| OQ | Status | Decision |
|----|--------|----------|
| OQ1 re-arm strategy | **Resolved** | Cursor-*generation* ACK single-flight machine with ACK_STALLED (§3.1); no turn correlation. |
| OQ2 extension MCP vs shell | **Resolved** | **Shell out.** Extension API exposes tool metadata only + `pi.exec` (§3.0). |
| OQ3 Design A vs B | **Resolved** | Reconciled Design A: extension wakes + observes non-consuming generation snapshot; lead is sole cursor owner/reader. B (raw JSONL) rejected. Requirement §3 updated to match. |
| OQ4 steer/followUp + idle | **Resolved** | Always `deliverAs:"steer"` with `triggerTurn:true` (immediate if idle, else queued); no background `isIdle()` needed since it is event-context-only (§3.0/§3.1/§10). |
| OQ5 location/loading | **Resolved** | `pi-extensions/win-agent-teams-wake/` in this repo, pinned Pi version, with the T9 e2e fixture. |
| OQ6 output/waiting | **Deferred** | v1 acts only on `reason=message`; other one-shot exits ignored, documented. |
| OQ7 timeout `T` | **Deferred** | Bounded, configurable default; paired with R6 backoff. |

## 8. Next steps (per CLAUDE.md workflow)

1. **Re-review (iteration 4)** by the same Codex agent via `follow_up_agent` —
   state Resolved/Not-Resolved per prior blocker with evidence, re-check AC matrix.
2. Red-green-refactor TDD implementation.
3. Post-implementation review (Claude Opus), `implementation.md` +
   `implementation-review.md`.
4. PR against `mikaelliljedahl/agentic-coder-teams-mcp`.

## 9. Disposition of `plan-review-1.md` findings

| # | Finding | Disposition |
|---|---------|-------------|
| 1 | `turn_end` re-arm ≠ cursor progress; unbounded paid loop | **Accepted.** `inbox-status` generation ACK + ACK_STALLED, one injection per generation (§3.1, R1, T5/T5b). |
| 2 | Mid-turn correlation race | **Accepted.** v3+ **drops turn correlation entirely** (`sendMessage` returns void); re-arm is cursor-generation-driven (§3.0/§3.1, R2). |
| 3 | Design A contradicts requirement §3 | **Accepted.** Reconciled: lead is sole cursor owner; requirement §3 updated. |
| 4 | OQ2 is a feasibility gate; extension can't invoke MCP | **Accepted.** Shell-out design (§3.0, §3.2). |
| 5 | Wrong injection API (`steer`/`followUp` are SDK) | **Accepted.** `pi.sendMessage(customMessage, {deliverAs:"steer"})` (§3.1; v3 supersedes the isIdle branch — see review-2 4.3). |
| 6 | Discovery can bind to recovered/changing session | **Accepted.** Treat `session_id` as changing binding; rebind (§3.1, §3.3, T6b). |
| 7 | Identity should be validated, not assumed | **Accepted.** Validate `identity==team-lead`, fail closed; v3 uses `--reader` flag instead of env removal (§3.3; review-2 4.1). |
| 8 | `>50` backlog → one turn per page | **Accepted.** Generation ACK tolerates partial cursor progress; notice tells lead to drain `has_more` (§3.1, T5c). |
| 9 | Wake-to-read window safe only if read happens; single cursor writer | **Accepted.** Single cursor owner enforced (R5); no second consuming reader. |
| 10 | Subprocess failure/backoff/shutdown absent | **Accepted.** R6 + one-live-watcher lifecycle. |
| 11 | Parse strictly but tolerantly; reject arbitrary `path` | **Accepted.** T4 parse cases + inbox-path check. |
| 12 | Test location imprecise | **Accepted.** `tests/test_session_recovery.py` (§4). |
| Warnings (§6) | `send_message` phrasing, default watch pattern, additive-key caveat, no question classification, pin Pi version | **Accepted/noted** — phrasing corrected in §2; question classification out of v1; Pi version pinned (R4). |

### Disposition of `plan-review-2.md` blockers (score 76 → v3)

| # | review-2 blocker | Disposition |
|---|------------------|-------------|
| 4.1 | Retry-exhaustion `→ WATCHING` re-fires the same level-triggered unread; loop still unbounded | **Accepted.** New **ACK_STALLED** state never runs an inbox watcher for an unread generation; only a strictly-newer generation re-injects; budget keyed per generation (§3.1, R1, T5b/T7). |
| 4.2 | Unread counts alone are not race-safe (consume+arrival cancels out) | **Accepted.** ACK is a per-sender `{total,cursor}` generation threshold `min(cursor,total) >= target_total`; `inbox-status/1` schema exposes it (§3.1, §3.2, R2, T5d). |
| 4.3 | CLI contracts under-specified | **Accepted.** `session-dir` and `inbox-status` fully specified (schema, exit 0/1/3/4, empty, validation, non-creating wording) (§3.2, T2/T2b). |
| 4.4 | Pi API: no `env` in `ExecOptions`; `sendMessage:void`; `isIdle` context-only; message object undefined; pin version | **Accepted.** `--reader` flag replaces env removal; turn correlation dropped (ACK is cursor-driven); always `deliverAs:"steer"`; custom-message object defined; API pinned in §10 (§3.0, §3.1, R4). |
| 4.5 | Session refresh timing unspecified | **Accepted.** Refresh after every watch exit and before every injection; rebind on id change (§3.3, T6b). |
| 5 | Shutdown/reload/one-child lifecycle tests missing | **Accepted.** T10 (shutdown abort) + T11 (single-flight) (§6, R6). |
| non-blocking | additive-key exact-equality caveat; name exact Pi version | **Noted** — T1/T3 cover field regression; exact Pi version recorded in the extension `package.json` at implementation start, API facts pinned in §10. |

### Disposition of `plan-review-3.md` blockers (score 85 → v4)

| Item | review-3 point | Disposition |
|------|----------------|-------------|
| ACK_STALLED transitions | 4.1 — no late-ack→WATCHING edge; new-sender undefined → possible permanent missed wake | **Accepted.** Explicit ordered evaluation: late-drain→WATCHING, then union/default-0 new-generation rule with an `unread>0` guard (§3.1, T7/T7b/T7c). |
| Lifecycle signal | 4.5 — `session_shutdown` has no `AbortSignal`; `ctx.signal` is per-turn | **Accepted.** Extension-owned `AbortController` created at `session_start`, aborted at `session_shutdown`, idempotent across reload/new/resume/fork (§3.1, §10, T10/T11). |
| `display` type | 4.3 — `display` is `boolean` in `CustomMessage` | **Accepted.** Sample fixed to `display:true`; §10 quotes `messages.ts`. |
| Injection budget ambiguity | 4.6 — one injection vs reinjection unclear | **Accepted.** Exactly one injection per generation; no in-generation reinject edge (§3.1, T5b). |
| `--reader` hardening | 4.4 — validate reader before filename interpolation | **Accepted.** Safe-agent-name validation added (§3.2). |
| Editorial | stale §9 `turnIndex` row; scope said `unread` | **Accepted.** §9 row 2 updated; §1 scope now says `inbox-status`. |

## 10. Verified Pi extension API (pinned facts)

Quoted from upstream `earendil-works/pi` (`main`, fetched 2026-07-19). The extension
is written against these; `package.json` records the exact resolved version at
implementation start and CI verifies the signatures still hold.

- `packages/coding-agent/src/core/exec.ts`:
  ```ts
  export interface ExecOptions { signal?: AbortSignal; timeout?: number; cwd?: string; }
  // no `env` field — child inherits Pi's environment → use CLI --reader, not env
  ```
- `packages/coding-agent/src/core/extensions/types.ts`:
  ```ts
  exec(command: string, args: string[], options?: ExecOptions): Promise<ExecResult>;

  sendMessage<T = unknown>(
    message: Pick<CustomMessage<T>, "customType" | "content" | "display" | "details">,
    options?: { triggerTurn?: boolean; deliverAs?: "steer" | "followUp" | "nextTurn" },
  ): void;                                   // returns void → no turn handle

  isIdle(): boolean;                         // on ExtensionContext only, not ExtensionAPI

  on(event: "session_start",    handler: ExtensionHandler<SessionStartEvent>): void;
  on(event: "session_shutdown", handler: ExtensionHandler<SessionShutdownEvent>): void;
  on(event: "turn_end",         handler: ExtensionHandler<TurnEndEvent>): void;
  // SessionShutdownEvent = { reason: "quit"|"reload"|"new"|"resume"|"fork";
  //                          targetSessionFile?: string }   — NO AbortSignal
  // ExtensionContext.signal is the per-turn signal (undefined while idle),
  //   NOT a lifetime/shutdown signal.
  ```
- `packages/coding-agent/src/core/messages.ts`:
  ```ts
  interface CustomMessage<T> { customType: string; content: string;
                               display: boolean; details?: T; /* … */ }
  // display is a BOOLEAN (show/hide), not banner text
  ```

Implications already baked into §3: shell out via `exec`; drive the reader with
`--reader team-lead` (no `env`); inject one `{customType:"win-agent-teams/wake",
content, display:true}` custom message with `deliverAs:"steer"`; do not rely on a
returned turn handle or on `isIdle()` from the background loop; own an
`AbortController` created at `session_start` and abort it from `session_shutdown`
(there is **no** shutdown `AbortSignal` to reuse).
