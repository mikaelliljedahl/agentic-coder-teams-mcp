# PRD — Deterministic inbox-wake for Claude Code lead agents

Status: draft, revised per `prd-review.md` (findings F1–F5 applied).
Feature slug: `claude-lead-inbox-wake`.
Branch: `feature/claude-lead-inbox-wake` (based on `origin/main`).

## 1. Background & problem statement

win-agent-teams is an MCP server that lets a "lead" agent spawn and message
Claude Code / Codex / Pi subagents. Messaging is **pull-based**: a subagent's
reply lands in `inbox-<reader>.jsonl` under
`~/.claude/agent-sessions/<session-id>/`, and the lead only sees it when it
calls the `read_messages` MCP tool. Nothing pushes; nothing wakes the lead on
its own. So a lead that has spawned workers and is otherwise idle needs an
external, reliable trigger to call `read_messages` the moment a reply arrives.

**Pi leads already have a deterministic trigger.** The bundled Pi extension
`pi-extensions/win-agent-teams-wake/` runs a single-flight state machine
(`DISCOVERING → WATCHING → ACK_WAIT → ACK_STALLED`, see `src/state-machine.ts`)
that shells out to the read-only `win-agent-teams` CLI (`session-dir`,
`inbox-status`, `watch`) and, on a new inbox message, injects exactly one
steering turn (`customType: "win-agent-teams/wake"`) telling the lead to call
`read_messages`. It never consumes the inbox (the lead stays the only cursor
writer) and uses a cursor/total-based "generation ACK" so concurrent arrivals
do not re-fire a wake (`src/generation.ts`). Merged in PR #31; generalized to
nested leads on `feature/pi-lead-autoload`.

**Claude Code leads have no deterministic equivalent.** The current recipe
(documented in the `_DISK_CONTRACT_NOTE` docstring appended to
`agent_watch_paths` / `check_agent` / `agent_status` / `list_agents` in
`server_simple.py`, and surfaced via the `watch_argv` /
`watch_command_bash` / `watch_command_powershell` fields of `spawn_agent` and
`agent_watch_paths`) is:

> Run the `watch` CLI (`python -m claude_teams.cli watch <session_dir>`) as a
> **background** Bash task. When it exits (a message/waiting/output edge), the
> Claude Code harness re-invokes the model, which branches on the emitted
> `reason` and calls `read_messages`.

This works, but only if the model **remembers to start the background watcher**
every idle turn. The user reports this is probabilistic in practice: "Fable
always remembers, Opus on low reasoning effort sometimes forgets." A lead that
forgets goes to sleep with unread worker replies and never wakes until the
human nudges it. Probabilistic orchestration wiring is unacceptable — the
trigger must fire every turn regardless of model discipline or reasoning
effort.

The nested-orchestration model (`CLAUDE.md` "Nested orchestration: lead is a
role, not an identity", commit `b5627ed`) makes this a per-level requirement:
**any** spawned Claude Code agent can itself be a lead for the children it
spawns, watching `inbox-<its AGENT_NAME>.jsonl`; the root/human-launched lead
watches `inbox-team-lead.jsonl`. The Pi wake shipped as team-lead-only and had
to be corrected (the "over-constrained bug"). This feature must not repeat that
mistake: the Claude wake must work for a lead at **every** nesting level.

## 2. Verified Claude Code Stop-hook semantics

Verified against the official docs on 2026-07-20:

- Hooks reference: https://code.claude.com/docs/en/hooks
- Hooks guide: https://code.claude.com/docs/en/hooks-guide

Hooks are "user-defined shell commands that execute at specific points in
Claude Code's lifecycle … deterministic control … ensuring certain actions
always happen rather than relying on the LLM to choose to run them" (hooks
guide). This determinism is exactly the property the file-watcher recipe lacks.

**Registration (settings.json).** Three levels of nesting: event → matcher
group → hook handler. `Stop` does **not** support matchers ("Stop … don't
support matchers and always fire on every occurrence. If you add a `matcher`
field to these events, it is silently ignored"). Shape:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "<cmd>", "timeout": 600 } ] }
    ]
  }
}
```

`SubagentStop` **does** support a matcher (by agent type). This is the
Claude-Code-internal Task subagent finishing — **not** a win-agent-teams
spawned worker — and the existing `watch` CLI already ignores `SubagentStop`
markers as non-actionable churn. This feature targets the lead's own `Stop`.

**Input (stdin JSON).** `Stop`/`SubagentStop` receive `session_id`,
`transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, plus
(reference) `last_assistant_message` and an `effort` object. **Note:** the
`session_id` here is Claude Code's own transcript session id — it is **not**
the win-agent-teams `AGENT_SESSION_ID`; identity resolution (§6.2) must not
rely on this field.

**`stop_hook_active` + the infinite-loop guard.** The hooks *guide* documents a
guard the hooks *reference* page omits (candor: our first reference-page fetch
did not surface it; the guide is authoritative here):

> "Claude Code overrides a Stop hook after it blocks eight times in a row
> without progress. Your hook script needs to check whether it already
> triggered a continuation. Parse the `stop_hook_active` field from the JSON
> input and exit early if it's `true`."

So (a) there is a hard cap of **8 consecutive blocks without progress**, after
which Claude Code ignores the block and stops anyway (turn ends with a
warning); and (b) `stop_hook_active: true` tells the hook it is already running
inside a stop-triggered continuation — which includes a *legitimate*
continuation such as a message-wake that led to a successful `read_messages`.
Both facts are load-bearing for the loop-guard requirement (§6.7), but
`stop_hook_active` alone cannot distinguish a productive wake cycle from a
stuck loop; only observed progress (cursor advance) can — see FR14.

**Output / decision.** On **exit 0**, stdout is parsed as JSON. For `Stop`,
`{"decision": "block", "reason": "<text>"}` "Prevents Claude from stopping,
continues the conversation"; the `reason` "is fed back to Claude so it keeps
working". Omitting `decision` (or emitting
`hookSpecificOutput.additionalContext`) allows the stop. A top-level
`{"continue": false, "stopReason": "…"}` halts Claude entirely and "takes
precedence over any event-specific decision fields" — an available hard escape
hatch. **Exit 2** also blocks and feeds stderr back; other non-zero exits are
non-blocking errors (shown as a hook-error notice, stop proceeds).

**Timeout & long-running/blocking hooks.** Command hooks take a `timeout` field
in **seconds**; the default for `Stop` is **600 s (10 min)**. A hook may sleep
and poll for up to its timeout — long-running/blocking hooks are supported.
There is also an `async: true` field ("runs in the background without
blocking"), but an async hook cannot return a blocking `decision` — so it
cannot itself wake the lead. Additionally, a process **forked by a hook** is
not a *tracked* background task (those are started by the model via its Bash
tool), so a hook-spawned watcher almost certainly cannot trigger a harness
re-invocation by itself. These constraints shape the candidate designs in §3.

## 3. Candidate designs

Two first-class candidate designs (plus a hybrid) can satisfy "deterministic
trigger". Choosing between them is the most consequential decision of this
feature and is resolved by an early spike (§9 GC2), not by default.

### Design A — block-in-hook (synchronous long-poll)

On `Stop` with live subagents, the hook itself waits inside the hook process
(blocking `watch --timeout`, or an `inbox-status` poll loop). On a new message
it returns `{"decision":"block","reason":"New message from <sender> — call
read_messages …"}`; on max-wait timeout it allows (or, under a strict budget,
re-blocks once with a "still waiting" nudge).

- Pros: zero extra model turns per wake; wake latency ≈ poll interval; no new
  disk state.
- Cons: **the UI is frozen while the hook waits** — the human cannot type a new
  prompt until the hook returns; the wait ceiling is the hook `timeout`
  (default 600 s), so a lead is deaf to replies arriving later than ~10 min
  after its last turn; repeated timeout-re-blocks press against the 8-block cap
  (~8 × max-wait, under ~80 min, after which it stops anyway).

### Design B — verified arming (hook verifies, tracked watcher wakes)

The wake path stays the **existing tracked background watcher**; the hook's job
is to make its arming deterministic instead of model-discretionary:

1. On `Stop` with live subagents, the hook checks whether a background watcher
   is **actually armed** for this lead — via an arming marker (pid/marker file
   under the session dir, written by the `watch` CLI or a thin wrapper, and
   staleness-checked against pid liveness).
2. **Armed → allow the stop.** The turn ends, the UI stays typeable, and the
   already-running tracked watcher wakes the harness on the next message — the
   existing recipe's wake path, unchanged.
3. **Not armed → block**, with a `reason` containing the exact command to run
   as a background Bash task now (the `watch_command_bash` / `watch_argv`
   rendering for this session dir). Following a direct injected instruction is
   near-certain even on Opus-low — the observed failure mode was *remembering
   unprompted*, not *refusing when told*.
4. On the next `Stop`, the hook re-verifies. Still not armed → block again
   (subject to the FR14 progress guard). The lead can never go to sleep
   unarmed; **determinism lives in the verification step**, not in the model.

- Pros: no UI freeze; no 600 s ceiling (the watcher can run for hours —
  overnight waits work); no 8-cap pressure in normal operation (a successful
  arming is progress); reuses the shipped, documented wake path.
- Cons: one short extra model turn per arming (and per re-arm after each wake,
  since `watch` is one-shot); adds a small arming-marker file to the disk
  contract; the arming command is still executed by the model (mitigated by
  the verified retry loop).

### Hybrid — grace wait, then arm-and-release

On `Stop` with live subagents: wait in-hook for a short grace period (seconds,
not minutes) to catch replies already in flight (zero extra turns for the
common fast-reply case); if nothing arrives, fall back to Design B's
verify/arm/allow so the turn ends typeable and long waits are carried by the
tracked watcher.

### Disk-contract consideration (Design B / hybrid)

Design B introduces one new session-dir artifact: an **arming marker**
(e.g. `wake-armed-<reader>.json` — exact name/schema is a plan choice)
recording at least the watcher pid and start time, written on watcher start and
removed/invalidated on exit; the hook treats a marker with a dead pid (or a
malformed marker) as not-armed. Like all session-dir contract surface, it must
be documented in the relevant MCP tool docstrings (FR26), not only prose docs.

## 4. Goals & non-goals

### Goals

- G1. A Claude Code lead — at any nesting level, on any model/effort — reliably
  processes a subagent's inbox reply without the model having to remember to
  arm a watcher. The trigger is deterministic (hook-driven), not
  model-discretionary.
- G2. Waiting burns **zero model tokens** (the wait happens inside the hook
  process or a background watcher process, not in a model turn).
- G3. Behavioral parity with the Pi wake where it makes sense: identity = the
  agent's **own** identity; one wake per generation of unread messages; never
  advance the inbox cursor (the lead remains the only cursor writer).
- G4. Cross-platform: Windows **and** Linux. The hook entrypoint must not be
  bash-only.
- G5. Deterministic installation for both lead kinds: (a) auto-wired at spawn
  for server-spawned Claude Code agents, and (b) a documented one-step install
  for the human-launched top-level lead.
- G6. The lead is never made unstoppable: a human interrupt and a bounded
  max-wait escape hatch always let the turn end.

### Non-goals

- N1. Push-based messaging or any change to the pull-based on-disk contract
  (`inbox-<reader>.jsonl`, `inbox-<reader>.pos.json`, `state-<agent>.json`) —
  beyond the additive arming marker introduced by Design B/hybrid (§3).
- N2. Waking a chat-only Claude Desktop session that does **not** run the hook
  system. (The Claude Code harness that Claude Desktop embeds is in scope
  **iff** it honors `Stop` hooks and settings; confirming that is gating check
  GC1, §9. A pure chat app with no hooks cannot be served by this mechanism
  and is explicitly out of scope.)
- N3. Replacing or removing the existing background-watcher recipe. Under
  Design A it remains the documented fallback (and the mechanism for Codex
  leads, which have no idle-wake). Under Design B/hybrid it is not a fallback
  at all — it **is** the wake path, with the hook guaranteeing it is armed.
  Either way this feature adds determinism; it does not delete the recipe.
- N4. Changing how Codex or Pi leads wake.
- N5. Waking on non-message edges (`waiting` / `output`) in v1 — parity with the
  Pi wake, which wakes only on `reason="message"` and ignores `output`/`waiting`
  in v1.

## 5. User stories

- U1 (top-level interactive lead). As a human running `claude` in a repo, I
  spawn workers, then keep chatting/idling. When a worker replies, my lead
  session picks it up and calls `read_messages` on its own, even on Opus-low,
  without my having told it to arm a watcher.
- U2 (spawned nested lead). As a mid-level Claude Code agent spawned by the
  server (I have `AGENT_NAME` set), I spawn my own children. When a child posts
  to `inbox-<my name>.jsonl`, I am woken and drain it — with no extra setup
  beyond what the server injected at my spawn.
- U3 (mixed Pi + Claude team). As a team with a Pi root lead and Claude
  mid-level leads (or vice-versa), every lead wakes deterministically by the
  mechanism native to its backend; the two mechanisms share the same CLI/disk
  contract and never fight over the cursor.
- U4 (leaving a lead overnight). As a human, I leave an idle lead with live
  workers and it wakes on a reply that arrives hours later. **This story is
  satisfiable only under Design B or the hybrid** (a long-running tracked
  watcher): under pure Design A the lead is deaf ≤ 10 min after its last turn
  (allow-on-timeout) or hits the 8-block cap within ~80 min (re-block). If the
  gating spike (§9 GC2) selects pure Design A, U4 is explicitly descoped to:
  "fails safe overnight — stoppable, no runaway loop, no token burn — and wakes
  on the next human nudge".

## 6. Functional requirements (numbered, testable)

### 6.1 Deterministic trigger

- FR1. The wake MUST be driven by a Claude Code `Stop` hook, so it runs every
  time the lead's turn ends regardless of model or reasoning effort. Under
  Design A the hook itself detects and blocks on new messages; under Design B
  the hook deterministically **verifies watcher arming** and blocks until the
  watcher is verifiably armed. In neither design may correct wiring depend on
  the model choosing, unprompted, to invoke a tool or start a background
  command.
- FR2. On turn end with **no live subagents** for this lead's session, the hook
  MUST allow the stop (exit 0, no `decision`) promptly (target < 1 s) so an
  ordinary non-orchestrating session is unaffected.
- FR3. On turn end **with** live subagents and at least one unread inbox
  message already present, the hook MUST immediately block
  (`{"decision":"block","reason":"…"}`) naming the sender(s) and instructing
  `read_messages`, without waiting.
- FR4. On turn end with live subagents and no unread message, the design MUST
  guarantee a wake on the next message: Design A by waiting in-hook (§6.4) and
  blocking on arrival; Design B by verifying an armed watcher (blocking with
  the exact arm command until verified); hybrid by grace-wait then
  verify-and-release.

### 6.2 Identity resolution (lead at every level)

- FR5. The hook MUST resolve the lead's **own** identity and session the same
  way the CLI already does: identity = `AGENT_NAME` when set, else `team-lead`;
  session via `AGENT_SESSION_ID` (spawned path) or the workspace binding
  (`_binding_file` / `_recover_session_id`, cwd+identity fallback) for the
  top-level lead. It MUST reuse the read-only `session-dir` CLI (which prints
  `id<TAB>dir<TAB>identity`) rather than reimplementing discovery.
- FR6. The hook MUST watch `inbox-<own-identity>.jsonl` and MUST NOT hardcode
  `team-lead`. A nested Claude lead (`AGENT_NAME` set) MUST be woken by messages
  to its own inbox. (This is the explicit anti-repeat of the Pi team-lead-only
  bug.)
- FR7. If the hook cannot resolve a live win-agent-teams session for this
  process (no session, or a non-lead/unknown identity), it MUST fail **open** —
  allow the stop — never block. A hook must never trap a session that is not a
  win-agent-teams lead.

### 6.3 No cursor mutation

- FR8. The hook MUST NOT write the inbox cursor (`inbox-<reader>.pos.json`) or
  mutate any session file other than its own wake bookkeeping (the Design-B
  arming marker and the FR14 progress-guard state file). Read-only access to
  inbox metadata is permitted and SHOULD go through the read-only CLI surface
  (`session-dir`, `inbox-status`, `watch`) — note `inbox-status` already
  exposes per-sender `{total, cursor, unread}`, which satisfies FR11 without
  opening message bodies. The lead calling `read_messages` remains the sole
  cursor writer.

### 6.4 No-token-burn waiting

- FR9. All waiting MUST occur outside model turns — inside the hook process
  (Design A) and/or inside the background watcher process (Design B) —
  consuming **zero** model tokens while waiting.
- FR10. Any **in-hook** wait MUST be bounded by a configured maximum (§6.6)
  and MUST honor the interrupt (§6.7). A Design-B tracked watcher may run
  unbounded (or with its own generous `--timeout`, re-armed after each wake),
  since it does not hold a turn open.

### 6.5 One-wake-per-generation semantics

- FR11. A block SHOULD name the unread sender(s) at wake time (available from
  `inbox-status`'s senders map without reading message bodies — see FR8) and
  instruct the lead to call `read_messages` and keep draining while
  `has_more`. (Mirror the Pi wake message content and the generation-ACK
  intent so a single burst of replies produces a single actionable wake rather
  than a storm.)
- FR12. The mechanism MUST NOT enter a tight block/allow loop while messages
  stay unread. Because each block is followed by the lead doing work
  (`read_messages`, or arming the watcher under Design B), consecutive
  **no-progress** blocks MUST be avoided so the 8-block cap (§2) is not
  reached in normal operation.

### 6.6 Timeout / re-block behavior

- FR13. The design MUST define behavior when an in-hook max-wait elapses with
  live subagents but no new message. For Design A this is a plan decision
  between: (a) **allow stop** (lead sleeps until the human nudges — safest wrt
  the 8-block cap and UX), or (b) **re-block once** with a "still waiting"
  nudge under a strict consecutive-timeout budget. For Design B/hybrid the
  question largely dissolves (the hook releases after verifying arming; the
  watcher carries the long wait). The chosen default MUST be documented and
  MUST NOT risk an unstoppable lead (§6.7). See §10 OQ2.
- FR14. The loop-guard MUST be **progress-based**, not a blanket
  `stop_hook_active` skip. The hook maintains a small state file under the
  session dir (name/schema a plan choice) recording the per-sender
  cursor/total snapshot at its previous block. It fails toward **allow** only
  when a would-be block shows **no cursor advance since the previous block**
  (i.e. the lead was woken/instructed and made no progress). A continuation
  after a *productive* wake (cursor advanced) is normal operation and MUST NOT
  shorten or skip the next cycle. `stop_hook_active` is retained as a
  belt-and-braces input combined with the progress check (e.g. the guard is
  consulted only when `stop_hook_active` is true), never as a standalone
  trigger — a blanket "skip when true" would fire on the very next Stop after
  each successful wake and degrade the feature to first-wake-only.

### 6.7 Loop-guard safety (never make a lead unstoppable)

- FR15. A human interrupt (Esc/Ctrl-C) MUST abort any in-hook wait promptly and
  let the turn end. (Verified empirically in §9 GC2(b).)
- FR16. A single hook invocation MUST NOT wait longer than its configured
  max-wait, which MUST be strictly less than the Claude Code `Stop` hook
  `timeout` (default 600 s) so the hook returns cleanly rather than being
  hard-killed by the harness.
- FR17. There MUST be a documented kill switch that fully disables the wake
  (env flag, §6.10) and, per FR7/FR14, guaranteed fail-open paths so no
  combination of state leaves the lead unable to stop.

### 6.8 Cross-platform (Windows + Linux)

- FR18. The hook command MUST run on Windows and Linux. It MUST NOT be a
  bash-only script. Reuse the repo's Python entrypoint pattern (the state
  emitter is invoked as `"<python>" -m claude_teams.hooks emit …`; a wake
  entrypoint should follow the same `python -m …` convention), so the same code
  path runs on both OSes.
- FR19. Windows launch/quoting gotchas MUST be respected: settings-file JSON
  command strings are double-quoted per token (see
  `hooks._shell_quote_command`); paths use forward slashes
  (`Path.as_posix()`) as the state hooks already do. Message/inbox path
  comparison MUST be separator-insensitive (the Pi wake normalizes `\`↔`/`;
  the Python `watch` already emits `str(Path)` whose separator is host-native).
  Design-B pid-liveness checks MUST work on both OSes.

### 6.9 Installation / wiring

- FR20. **Spawned Claude Code agents:** the wake hook MUST be auto-wired at
  spawn, alongside the existing state-marker hooks, with **no** per-agent user
  action. Today `hooks.write_claude_settings` wires every lifecycle event
  (including `Stop`) to the state-marker `emit` command and the server passes it
  via `--settings` (`claude_code._hooks_settings_args`,
  guarded by `WIN_AGENT_TEAMS_STATE_HOOKS`). The wake hook MUST coexist with the
  existing `Stop` state-marker hook (a `Stop` event may hold multiple hooks in
  its array; ordering/'`decision` from whichever hook' semantics must be
  respected — §10 OQ5). Mechanism (extend `write_claude_settings`, a second
  hook entry, or a combined command) is a plan choice; the **requirement** is
  zero-touch auto-wiring at spawn/resume.
- FR21. **Top-level human-launched lead:** because the server does not launch
  this process, wake installation MUST be delivered by a documented one-step
  mechanism — either a copy-paste settings snippet or an MCP tool that
  emits/installs the correct `Stop` hook into the user's Claude Code settings.
  The requirement is that it be **one deterministic step**, not per-turn model
  discipline. Mechanism choice is deferred to the plan (§10 OQ1).
- FR22. Installation MUST be idempotent and reversible (re-running does not
  duplicate hooks; a documented removal/disable exists).

### 6.10 Config surface

- FR23. Tunables MUST use the existing `WIN_AGENT_TEAMS_*` env-var convention
  (cf. `WIN_AGENT_TEAMS_STATE_HOOKS`, `WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS`,
  `WIN_AGENT_TEAMS_IDLE_SECONDS`). At minimum: an on/off switch and the
  in-hook max-wait seconds. Names to be finalized in the plan (candidates:
  `WIN_AGENT_TEAMS_LEAD_WAKE=0/1`, `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_WAIT`).
- FR24. Defaults MUST be safe out of the box: wake enabled for spawned agents,
  in-hook max-wait comfortably under the 600 s hook timeout.

### 6.11 Observability

- FR25. The hook MUST emit enough diagnostics (to stderr and/or a log under the
  session dir) to answer "did the hook run, did it resolve a session, did it
  wait/verify, why did it allow vs block?" without a debugger — the manual
  smoke test (§7) depends on it. Logging MUST NOT leak message bodies (parity
  with the read-only, non-consuming contract).

### 6.12 Documentation / tool-docstring contract

- FR26. Per the repo rule that the consuming agent reads only MCP tool
  docstrings (not prose docs), any disk/behavior contract an orchestrating lead
  must know MUST be reflected in the relevant tool docstrings (e.g. the
  `_DISK_CONTRACT_NOTE`, `spawn_agent`, `agent_watch_paths`) — including the
  Design-B arming marker if adopted — and, if an install tool is added (FR21),
  in that tool's docstring. Prose docs (README/ADDING-A-BACKEND) are updated in
  addition, not instead.

## 7. Success criteria / acceptance

- AC1. **Determinism across effort.** A Claude Code lead on Opus-`low` (worst
  reported case), with zero watcher discipline, processes a subagent reply
  within N seconds of it landing, for M consecutive idle turns, with a 100% wake
  rate. Suggested targets for the plan: N ≤ watch poll interval + a few seconds
  (Design B: plus one bounded arming turn when unarmed), M ≥ 10. Verified on
  **both** Linux and Windows.
- AC2. **Zero-token wait.** While waiting, no model turn/token is consumed
  (observable: no assistant output between turn end and the wake; Design B's
  arming turn is bounded and occurs only when unarmed).
- AC3. **Nested lead.** A spawned Claude lead (with `AGENT_NAME`) is woken by a
  message from its own child to `inbox-<AGENT_NAME>.jsonl` — not only the root
  `team-lead` (direct regression test for the Pi-style over-constrained bug).
- AC4. **No-op safety.** A plain `claude` session with no win-agent-teams
  session, and a lead with no live subagents, stop normally with no perceptible
  delay.
- AC5. **Stoppability.** The lead can always be interrupted; no state sequence
  yields an unstoppable lead; the 8-block override is never reached in normal
  drain operation (per the FR14 progress guard).
- AC6. **No cursor interference.** Running the wake alongside a Pi lead in a
  mixed team never advances or corrupts a cursor; each lead drains its own
  inbox.
- AC7. **Automated proof of wiring.** Unit tests prove the emitted settings
  wire the `Stop` wake hook (mirroring `hooks.write_claude_settings` tests) and
  that identity/decision/progress-guard logic blocks vs allows correctly
  against faked CLI output and state files. A documented manual end-to-end
  smoke test exists (mirroring the Pi wake README's step-by-step), explicitly
  marked as not auto-run in CI.
- AC8. **Overnight wake (Design B/hybrid only).** A lead left idle with live
  workers processes a reply arriving > 1 hour later without human input. If
  pure Design A is selected, this criterion is replaced by: the lead fails
  safe (stoppable, silent, zero tokens) and wakes on the next human nudge (per
  the U4 descope).

## 8. Risks & mitigations

- R1. **UI-blocking long-poll (Design A only).** A synchronous `Stop` hook that
  sleeps holds the turn open, so the human cannot type a new prompt while it
  waits (unlike the Pi background loop + steer injection, and unlike the
  tracked-watcher wake path). *Mitigation:* Design B/hybrid eliminates this
  entirely (the hook returns quickly once armed); if Design A is chosen
  anyway, keep max-wait modest, ensure the interrupt aborts the wait (FR15),
  and default to allow-on-timeout so the session becomes typeable again. The
  A/B/hybrid choice is gating check GC2 (§9).
- R2. **8-block override.** Repeated no-progress blocks (e.g. re-block on every
  timeout, or a model that never obeys the arming instruction) can hit the
  8-in-a-row cap and defeat determinism. *Mitigation:* block only on events
  that lead to progress (a message-wake ending in `read_messages`; an arming
  instruction ending in a verified armed watcher) and apply the FR14
  progress-based guard.
- R3. **Hook hard-timeout kill.** An in-hook wait ≥ the 600 s hook timeout is
  killed mid-flight. *Mitigation:* max-wait strictly < hook timeout (FR16).
  Design B avoids long in-hook waits altogether.
- R4. **Orphaned/crashed subagents.** A dead worker that never replies could
  keep a lead waiting forever if "live subagents exist" is judged only from a
  stale marker. *Mitigation:* any in-hook wait is always bounded (FR10/FR16),
  and liveness should reuse the existing stall/idle detection
  (`WIN_AGENT_TEAMS_STALL_SECONDS`, `agent_status`) rather than a naive
  "any marker present" check.
- R5. **Cursor race.** The wake must reflect the same "unread" view the lead
  acts on. *Mitigation:* reuse `inbox-status`/`watch` `unread` semantics
  (`unread = total − min(cursor, total)`) exactly, and never write the cursor
  (FR8) — the same discipline the Pi generation-ACK relies on.
- R6. **Session-id confusion.** The `session_id` in the Stop-hook stdin is
  Claude Code's transcript id, not the win-agent-teams session. *Mitigation:*
  resolve identity/session only via env + the `session-dir` CLI (FR5), never
  from the hook payload's `session_id`.
- R7. **Double-wiring / event collision.** Adding a `Stop` wake hook next to the
  existing `Stop` state-marker hook could reorder or shadow decisions.
  *Mitigation:* define coexistence explicitly (§10 OQ5) and cover it with the
  settings-generation unit test (AC7).
- R8. **Stale arming marker (Design B).** A watcher that crashed (or a marker
  left by a previous boot) could make the hook believe the lead is armed when
  it is not. *Mitigation:* the marker records the watcher pid; the hook treats
  a dead-pid or malformed marker as not-armed and re-instructs.
- R9. **Docs drift on hook semantics.** The reference vs guide pages disagreed
  on `stop_hook_active` visibility. *Mitigation:* the plan must re-verify
  against the live docs at implementation time and pin the exact fields used.

## 9. Pre-plan gating checks (ordered — resolve before plan.md is written)

These two checks gate the plan; every other open question (§10) can be settled
inside the plan itself.

- **GC1 — Claude Desktop harness honors `Stop` hooks + settings.** The user's
  primary lead runs in Claude Desktop's embedded Claude Code harness. Verify
  empirically that this harness executes `Stop` command hooks registered via
  user/project settings (expectation: yes, but it must be observed, not
  assumed). If it does not, the feature misses its main consumer and the
  approach must be reconsidered before any further investment.
- **GC2 — Design spike: choose Design A / Design B / hybrid (§3).** Build the
  smallest possible probe of each mechanism and decide. The spike must
  establish at minimum: (a) whether a Stop-hook block whose `reason` contains
  the exact arm command reliably causes an Opus-low lead to start the tracked
  watcher (Design B's load-bearing step); (b) whether an in-hook sleep is
  interruptible by the human, and what the UX of a blocked turn actually looks
  like (Design A's load-bearing constraint, FR15); (c) arming-marker
  read/write and pid-liveness mechanics on both OSes. Note: the earlier
  hypothesis that a hook-forked process might itself trigger a harness
  re-invocation is almost certainly false (hook children are not tracked
  background tasks) and is **not** the question to spike — the question is
  A vs B vs hybrid.

## 10. Open questions (deferred to the plan)

- OQ1. Top-level-lead install mechanism (FR21): copy-paste settings snippet vs a
  new MCP tool that writes the hook into the user's settings. If a tool: which
  settings file (project `.claude/settings.json` vs user settings), and how to
  make it idempotent/reversible.
- OQ2. Design-A timeout disposition (FR13): allow-stop (sleep until nudged) vs
  bounded re-block. Which is the default, and is it configurable? (Moot if GC2
  selects Design B.)
- OQ3. Exact grace-wait duration for the hybrid, if chosen.
- OQ4. Should v1 wake only on `reason="message"` (Pi parity, N5), or also
  surface `waiting`/`output` edges to the lead?
- OQ5. `Stop` multi-hook semantics: with both the state-marker `emit` hook and
  the wake hook registered on `Stop`, what is the ordering, and how is a
  `decision:block` from one hook combined with a plain exit-0 from the other?
  Should they be merged into one command?
- OQ6. Reader/identity for a spawned nested lead whose `AGENT_SESSION_ID`
  points at a shared session dir — confirm `inbox-<AGENT_NAME>.jsonl` is the
  correct inbox in the multi-level case and that `session-dir`'s printed
  identity matches.
- OQ7. Final `WIN_AGENT_TEAMS_*` env-var names and defaults (FR23/FR24), plus
  the arming-marker and progress-guard file names/schemas (§3, FR8, FR14).

## 11. References

- Pi wake extension: `pi-extensions/win-agent-teams-wake/` (state machine,
  generation ACK, CLI wrappers) and its `README.md`.
- Pi auto-load / nested-lead correction design (reference only):
  `docs/features/pi-lead-autoload/design.md` on branch
  `feature/pi-lead-autoload`.
- State-hook wiring: `src/claude_teams/hooks.py`
  (`write_claude_settings`, `emit`, `_emit_command`).
- Claude Code launch/env injection: `src/claude_teams/backends/claude_code.py`
  (`_hooks_settings_args`, `build_env`).
- CLI surface: `src/claude_teams/cli.py` (`watch`, `session-dir`,
  `inbox-status` — note `inbox-status` exposes per-sender
  `{total, cursor, unread}`).
- Disk contract, binding, watch recipe: `src/claude_teams/server_simple.py`
  (`_binding_file`, `_recover_session_id`, `_watch_argv`, `_DISK_CONTRACT_NOTE`,
  `agent_watch_paths`).
- Nested-orchestration model: `CLAUDE.md` "Nested orchestration: lead is a role,
  not an identity" (commit `b5627ed`).
- PRD review applied here: `prd-review.md` (same directory).
- Claude Code hooks: https://code.claude.com/docs/en/hooks and
  https://code.claude.com/docs/en/hooks-guide (verified 2026-07-20).
