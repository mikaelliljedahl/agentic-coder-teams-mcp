# Agent messaging and lifecycle protocol

Reference for the `win-agent-teams` MCP server as it behaves today. Audience: an
agent that must spawn, message, and retire other agents **without reading the
source**.

This document describes current behavior only. It proposes nothing. Known
defects are stated as facts in [Sharp edges](#sharp-edges).

## How to read the citations

**Symbol names are authoritative; line numbers are not.** Claims cite
`path:line`, but `server_simple.py` in particular churns fast — a verification
pass found 112 of its 113 line citations had drifted by 9–50 lines within a
handful of merged PRs, while every named constant, function, field and value
was still correct. Search for the named symbol rather than jumping to the
number, and treat a line that does not contain what the claim describes as
drift, not as a contradiction.

If a **claim** turns out to be false, that is a defect in this document: fix it
here in the same change that alters the behaviour. A reference that is trusted
and wrong is worse than no reference.

---

## 1. Actors and identity

There are three actors.

| Actor | What it is |
|-------|------------|
| **Lead / spawner** | The process whose MCP client calls `spawn_agent`. May itself be a spawned worker. |
| **Worker** | An OS process (`claude` or `codex`) started by the lead, with its own MCP server instance. |
| **Watcher** | A short-lived CLI process, `win-agent-teams watch <session_dir>`, that blocks until something actionable happens and then exits. |

### Identity is process-global and read once at import

The MCP server module reads three environment variables at import time and
never re-reads them (`src/claude_teams/server_simple.py:49-51`):

- `AGENT_NAME` → `_AGENT_NAME`
- `AGENT_SESSION_ID` → `_AGENT_SESSION_ID`
- `AGENT_PARENT_NAME` → `_AGENT_PARENT_NAME`

`IDENTITY` is `AGENT_NAME` when set, otherwise the constant `"team-lead"`
(`src/claude_teams/server_simple.py:57-58`). `IDENTITY` is *the* inbox name this
process reads (`src/claude_teams/server_simple.py:1397`) and *the* `from` field
stamped on every message it sends (`src/claude_teams/server_simple.py:1326`).

A root lead — one launched by a human, with no `AGENT_NAME` in its environment —
is therefore always `team-lead`.

### How a worker receives its identity

The two backends inject identity differently, because Codex does not propagate
process environment to its MCP servers.

- **claude-code**: the server writes a per-agent MCP config file containing
  `AGENT_SESSION_ID` / `AGENT_NAME` / `AGENT_PARENT_NAME` in its `env` block
  (`src/claude_teams/server_simple.py:1107-1124`), passed as
  `--mcp-config <path>` (`src/claude_teams/backends/claude_code.py:173-175`).
  The same three variables are also set in the child process environment
  (`src/claude_teams/backends/claude_code.py:283-289`).
- **codex**: identity is passed as a per-process config override,
  `-c mcp_servers.win-agent-teams.env={ ... }`
  (`src/claude_teams/backends/codex.py:465-487`). The comment there is explicit
  that writing to the shared `~/.codex/config.toml` would be racy. The same
  variables also go into the child's own environment
  (`src/claude_teams/backends/codex.py:558-570`).

`AGENT_PARENT_NAME` is set from `request.lead_session_id`, which `spawn_agent`
fills with the spawner's `IDENTITY` (`src/claude_teams/server_simple.py:1261`).

### Parentage is a flat field, not a tree

A spawned agent's record is
`{name, pid, backend, session_id, status, spawned_at, cwd, model,
permission_mode, reasoning_effort, create_token, correlation_id,
prompt_transport, spawned_by, spawned_by_source}` (`spawn_agent._do_spawn`),
plus two fields written only by `follow_up_agent`: `generation` (int, the CAS
counter — absent counts as 0) and `pending_delivery` (present only while an
attempt is unconfirmed, see [section 4a](#4a-delivery-confirmation)).

`spawned_by` holds the spawning server's `IDENTITY` at the moment of the spawn,
and `spawned_by_source` records how that parentage was established: `spawn`
(observed at spawn, the normal case) or `operator_asserted` (written by the CLI
recovery path described in
[gates 2a/2b](#gates-2a2b--the-direction-guard-not-a-security-boundary)).
Both are preserved verbatim through every resume. `spawned_by` is what the
`follow_up_agent` direction guard reads *and* what `send_message`'s C3
classification reads; a record without it is refused — `parent_unknown` from
`follow_up_agent`, `recipient_class="unrelated"` from `send_message` — rather
than allowed.

This is a single field per record, not a tree: it names one parent and supports
no ancestry query. Since the registry is flat, that single field is exactly what
separates a child from a sibling, and it is why `send_message` can refuse a
sibling without needing an ancestry walk. It is also **not** the mechanism that
resolves the `"team-lead"` alias — that still comes from the
`AGENT_PARENT_NAME` env var inside the child process (`_spawner_target`).

`correlation_id` is **required and load-bearing**: a non-empty string, written
at spawn and preserved through resume. Constructing or migrating a record
without it is not a harmless omission — an absent field classifies the agent as
`legacy`, which under R8 makes it permanently ineligible for follow-up and
recoverable only by kill-and-respawn. `prompt_transport` records which transport
that spawn used (`argv` or `sidecar`) and is what lets the binding ladder's
gate 0 tell a not-yet-readable sidecar spawn from a genuinely unbindable one.

**All agents in a session share one flat registry.** `AGENT_SESSION_ID` is
propagated verbatim to every descendant, and `_active_session_id` returns it
before doing any recovery (`src/claude_teams/server_simple.py:567-568`). A
worker that spawns its own workers writes into the *same* `agents.json` as the
root lead. There are no sub-sessions.

### Session directory resolution

`_SESSION_BASE` is `~/.claude/agent-sessions`
(`src/claude_teams/server_simple.py:77`); the session dir is
`_SESSION_BASE / <uuid>` (`src/claude_teams/server_simple.py:198-199`).

For a root lead with no `AGENT_SESSION_ID`, the session is recovered via a
binding file keyed by `sha256(identity + parent_pid + cwd)`
(`src/claude_teams/server_simple.py:267-278`), with a cwd+identity fallback that
auto-adopts a single candidate only when this workspace has exactly one bound
session (`src/claude_teams/server_simple.py:555-599`). Otherwise the session is
left unresolved and a `recoverable_sessions` nudge is merged into subsequent
dict-returning tool results (`src/claude_teams/server_simple.py:602-624`).

---

## 2. Tools

`_annotate` merges the recovery nudge into the return value of `spawn_agent`,
`send_message`, `read_messages`, `check_agent`, and `follow_up_agent` only
(`src/claude_teams/server_simple.py:1302`, `1339`, `1504`, `1559`, `1738`).
`kill_agent`, `resume_session`, `session_info`, `list_agents`, `agent_status`,
`agent_watch_paths`, and `list_backends` are not annotated.

Every tool body runs on a worker thread via `run_blocking`
(`src/claude_teams/async_utils.py:7-21`).

`_with_disk_note` appends `_DISK_CONTRACT_NOTE` — the state-marker schema and
the `win-agent-teams watch` recipe — to the *registered* descriptions of
`check_agent`, `list_agents`, `agent_status` and `agent_watch_paths`. That
appended text, not the raw docstring in the source, is what a calling agent
actually reads, so a change there changes the contract those agents see. It must
sit below the `@mcp.tool()` decorator to take effect.

### `spawn_agent(prompt, name, backend, model, cwd, permission_mode, reasoning_effort, expected_outputs)`

**Mechanically**, under the `agents.json` file lock
(`src/claude_teams/server_simple.py:1227`):

1. Creates the session if none exists (`create=True`,
   `src/claude_teams/server_simple.py:1226`).
2. De-duplicates the requested name by appending `-2`, `-3`, …
   (`src/claude_teams/server_simple.py:360-374`). **The name you asked for is
   not necessarily the name you got — read `result["name"]`.**
3. Resolves backend (default `claude-code` when available, else the first
   available, `src/claude_teams/backends/registry.py:92-108`) and resolves
   `(model, effort)` together (`src/claude_teams/server_simple.py:1238`).
4. Generates a fresh per-spawn correlation id (`new_correlation_id`) **before**
   the backend is called, because the id has to be inside the final prompt.
5. Writes the per-agent MCP config, materializes the final prompt and its
   transport (`_materialize_prompt`, see §4), and puts the correlation id, the
   optional prompt sidecar path, and the hook wiring into `SpawnRequest.extra`.
6. Calls `backend.spawn(request)`, which builds argv and starts **a new OS
   process** (`src/claude_teams/backends/process_base.py:82-105`).
7. Captures a PID creation token from the just-live child and appends the
   registry record, including `correlation_id` and `spawned_by` (the spawning
   server's `IDENTITY`, with `spawned_by_source: "spawn"`). This is the only
   point at which parentage is *observed* rather than asserted.

**Returns** `{name, pid, backend, session_id, state_marker_path, session_dir,
watch_argv, watch_command_bash, watch_command_powershell, expected_outputs}`
(`src/claude_teams/server_simple.py:1338-1349`). The three `watch_*` fields are
ready-to-run renderings of the watcher invocation for this session, so a
coordinator does not have to assemble the command itself.

`expected_outputs` is echoed back verbatim and is **not** validated, watched, or
acted upon anywhere.

**What the return does not prove:** only that `CreateProcess` (or equivalent)
succeeded and returned a PID. It does not prove the CLI parsed its argv, that
the model was accepted, that hooks fired, or that the prompt was received. The
first real evidence of life is the appearance of `state-{name}.json`.

**Failure modes** are exceptions, not result fields: `BackendNotRegisteredError`
(`src/claude_teams/backends/registry.py:78-79`),
`BackendBinaryNotFoundError` (`src/claude_teams/backends/claude_code.py:121`),
`BackendModelUnavailableError` for a Codex tier whose GPT-5.6 model this install
does not expose (`src/claude_teams/backends/codex.py:237-246`) — there is no
silent downgrade.

### `send_message(text, to="team-lead", idempotency_key="")`

**The recipient decides the path.** `_classify_recipient` classifies `to`
relative to the caller, and only two of the five classes are deliverable:

| Class | When | Behaviour |
|---|---|---|
| `spawner` | a lead alias, the name of `AGENT_PARENT_NAME`, or `team-lead` | append to `inbox-{spawner}.jsonl` — the upstream path, unchanged |
| `child` | the record's `spawned_by` is the caller | the **guaranteed path**: `_guaranteed_send`, the same code `follow_up_agent` runs |
| `sibling` | the record's `spawned_by` is the caller's own spawner | **refused** |
| `unrelated` | in `agents.json`, but neither of the above (a grandchild, another lead's worker, a pre-C1 record with no `spawned_by`) | **refused** |
| `unknown` | matches no agent in this session | **refused** |

Lead aliases are `""`, `team-lead`, `lead`, `orchestrator`, `parent`, `boss`,
`manager`, `up`, `supervisor` (case-insensitive, `_LEAD_ALIASES`). They resolve
before any registry lookup, so they can never be a typo. A spawner named
explicitly is checked **before** parentage: the spawner's own `spawned_by`
points further up, so asking only "did I spawn this?" would refuse the upstream
path R3 depends on.

**Returns:**

- Upstream: `{"success": true, "to": <resolved recipient>}`.
- Downstream: whatever `_guaranteed_send` returns — the same
  `delivered`/`failed`/`queued(phase=…)` schema as `follow_up_agent`, including
  `idempotency_conflict` and `idempotency_key_required`. A downstream send
  without a key is refused before anything is sent.
- Refused: `{"success": false, "to": ..., "reason": "recipient_not_addressable",
  "recipient_class": ..., "retriable": false, "detail": ...}`.
- No session: `{"success": false, "to": ..., "reason": "session_not_found"}`.

**A typo is refused, never re-routed (R5).** Until C3, an unknown recipient was
written to the lead's inbox with a `warning` field in the result. Nothing ever
consumed that field, so in practice a misspelled name became a real-looking
upstream message that the lead read as genuine — and every unreachable recipient
had an accept-then-drop path. Both are gone: the refusal names the class and
says plainly that nothing was sent.

**For the upstream class, `success: true` still proves only that a line was
appended to a file.** There is no push and no wake from `send_message` itself;
the guarantee comes from the recipient's watcher (§5), which is why R3 makes the
watcher a protocol component rather than a convenience.

**Guaranteed-path messages never enter the actionable inbox.** A message
delivered by `follow_up_agent` — or by `send_message` to an agent you spawned,
which is the same code — is resumed into the target's context and recorded
in the *sender's* delivery store; it is never written to `inbox-{name}.jsonl`.
If it were, a polling worker could read and act on the same instruction a second
time. The rejected alternative — write to the inbox, then pre-advance the
recipient's cursor — is unsafe: the cursor is a per-sender consumed **count**,
not a message-id set, and `read_messages` advances to the maximum selected
position, so advancing past an audit record silently destroys every earlier
unread message from that same sender. It also breaks the single-writer
invariant, and since kill purges a sender's inbox lines, an inbox-resident audit
record would be destroyed by killing the **sender** — losing exactly the trail
R4 requires to survive.

### `read_messages(from_agent, since_seq, full, limit, max_chars)`

Reads **the caller's own** inbox — `inbox-{IDENTITY}.jsonl` — never anyone
else's (`src/claude_teams/server_simple.py:1397`). Serialized per inbox name by
an in-process lock only; there is deliberately no cross-process file lock
(`src/claude_teams/server_simple.py:245-260`, `1399`).

Mechanics per call:

1. Load cursors; group inbox lines by `from`, keeping global file position
   (`src/claude_teams/messaging.py:37-61`). Malformed lines, non-dict entries,
   and entries without a non-empty string `from` are skipped silently.
2. **Clamp** every stored cursor down to the observed message count for that
   sender, including senders absent from the current snapshot (clamped to 0)
   (`src/claude_teams/server_simple.py:1413-1415`).
3. Select unread entries, sort by global file order
   (`src/claude_teams/server_simple.py:1443-1448`), clip to `limit` (default 50
   unless `full=True`, `src/claude_teams/server_simple.py:1440-1442`).
4. Advance and persist cursors (`src/claude_teams/server_simple.py:1458-1475`).

**Returns** `{messages, cursors, seq, unread_count, has_more}`. Each message is
`{from, text, ts, seq}` where `seq` is the sender's 1-based per-sender count —
plus `truncated` and `full_len` on each message when `max_chars` is set.
`unread_count` is measured *before* limit clipping, so it is the true backlog,
not the returned batch size (`src/claude_teams/server_simple.py:1449-1452`).
`cursors` is the map when `from_agent` is unset; `seq` is a scalar and `cursors`
is `None` when `from_agent` is set (`src/claude_teams/server_simple.py:1496-1501`).

Preconditions and refusals: `since_seq` without `from_agent` raises `ValueError`
(`src/claude_teams/server_simple.py:1379-1381`); negative `limit` raises
`ValueError` (`src/claude_teams/server_simple.py:1382-1384`). `limit=0` is a
non-consuming watermark peek. With no session, an empty result is returned
rather than an error (`src/claude_teams/server_simple.py:1389-1396`).

**This call is destructive by default** — it advances the persisted cursor, so
messages are drained. Use `limit=0` to peek.

### External members: `create_join_ticket`, `join_team`, `external_*`, `leave_team`

An external member is a manually started interactive session registered as a
child without a spawned backend process. The lead calls
`create_join_ticket(name, note)`, which reserves a safe, de-duplicated name
against both registry records and other open/unexpired tickets. Tickets live in
the atomic `join-tickets.json` array under the session directory. Their TTL is
24 hours by default; used and expired audit rows are retained for seven days.
Both durations have strict finite-positive environment overrides:
`WIN_AGENT_TEAMS_JOIN_TICKET_TTL_SECONDS` and
`WIN_AGENT_TEAMS_JOIN_TICKET_RETENTION_SECONDS`.

The paste-ready prompt tells the interactive session to call
`join_team(session_id, token)`. The join credential is replayable during
retention: reconciliation replays return the same membership after any of the
registry/ticket/marker crash windows. `join_team` returns a bearer token with
the exact grammar
`wam1:<canonical-lowercase-hyphenated-session-uuid>:<64-lowercase-hex>`.
`agents.json` stores only `sha256(secret)` as `member_token_digest`; public
agent results omit credential fields. A same-user reader of
`join-tickets.json` still has the derivation inputs, consistent with the
existing file-level threat model.

The external record has `backend: "external"`,
`spawned_by_source: "join_ticket"`, and `status: "running"`. Its PID is only
informational and can be stale after a Desktop restart. Binding is
`not_applicable`. External-only sessions remain discoverable and explicitly
resumable, but are not silently auto-adopted unless a live non-external record
also exists.

Every external call re-resolves authority from the token and registry while
holding the cross-process agents lock through all side effects. The lock order
is agents lock → per-inbox process lock → file I/O. Windows waiters time out
after 30 seconds; POSIX `flock` waiters block. Consequently:

- `external_send(member_token, text)` appends one line to the ticket parent's
  inbox and returns `delivery: "inbox"`.
- `external_read(member_token, ...)` uses the exact `read_messages` cursor,
  filtering, limiting, and truncation contract, but its complete cursor
  transaction is cross-process serialized.
- Successful send/read operations write `running/activity`; marker failure is
  reported as `heartbeat_warning: true` without turning the successful inbox
  operation into a retryable failure.
- `leave_team(member_token)` changes the record to `left` and writes
  `waiting/left`. Repeating leave is successful and does not rewrite the
  marker.
- Lead `send_message` to a running external child performs one locked inbox
  append. It never creates a delivery row or lease and needs no idempotency
  key. `follow_up_agent` and stale `deliver_pending` rows refuse with
  `external_agent_pull_only`.
- Lead `kill_agent` removes the record and artifacts without probing or
  signalling the informational PID. The next token call reports revocation.

Lead → external is pull-only, but the member session can make the pull
hands-free with `install_member_wake(joined_session_id, member_name)`: it bakes
a `Stop` hook (`python -m claude_teams.member_wake --joined-session-dir <dir>
--member <name>`, default `scope="user"` → `~/.claude/settings.json`) that on
every member turn end blocks while unread messages wait in the joined
`inbox-<member>` (instructing `external_read(member_token=...)`) and otherwise
verifies a reader-scoped `watch <joined dir> --reader <member>` background
watcher is armed. No credential is baked; the hook fails open when the
membership is not `running`, when the joined session shows no activity within
`WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS` (default 6h), or via the
`WIN_AGENT_TEAMS_MEMBER_WAKE` kill switch (falls back to
`WIN_AGENT_TEAMS_LEAD_WAKE` when unset). It coexists with the lead-wake `Stop`
group; the harness must support Claude Code `Stop` hooks for it to do anything.

`WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` registers only `join_team`,
`external_send`, `external_read`, `leave_team`, and `list_backends`. Use that
entry alone in a separate Desktop profile/client instance for client-surface
isolation. Running it beside the ordinary server in one profile leaves the
ambient root tools selectable and is only a degraded compatibility setup.

### `check_agent(name, full=False, max_chars=200)`

Reads the registry record, does a liveness check, and resolves the agent's
transcript through the binding ladder (`_resolve_agent_binding`, see
[section 3a](#3a-transcript-binding)). Persists a newly discovered
`backend_session_id` back into `agents.json` **only when the binding outcome is
`bound`** (`_sync_backend_session_id`); a `pending`, `unverified`, `ambiguous`,
`legacy` or `indeterminate` read never writes an id to the record, because
persisting one would make every later read trust it.

**Returns** `{name, state, alive, pid, backend, last_activity_at, unread_count,
last_line, seq, truncated, full_len, heartbeat_age_s, stalled, binding,
binding_retriable}`. `full=True` adds `last_message` and `backend_session_id`.

`binding` is the binding outcome for this call and `binding_retriable` says
whether retrying could change it. `last_message` / `last_activity_at` are
populated only for `bound` and `legacy`; the other four outcomes carry no
transcript-derived data at all.

`state` resolution precedence (`src/claude_teams/server_simple.py:154-182`):

1. Not alive → `"dead"` (liveness gates everything; a stale `running` marker
   never wins).
2. Marker state in `{"running", "waiting"}` → that value.
3. Otherwise activity recency: `< _idle_seconds()` → `"running"`, else
   `"idle"`. `_idle_seconds()` defaults to 60 s, overridable via
   `WIN_AGENT_TEAMS_IDLE_SECONDS` (`src/claude_teams/server_simple.py:104-112`).

`stalled` is `True` only when alive, state is neither `waiting` nor `dead`, and
heartbeat age exceeds `WIN_AGENT_TEAMS_STALL_SECONDS` (default 300 s)
(`src/claude_teams/server_simple.py:118-148`).

**`unread_count` and `seq` count messages *from* this agent *to the caller***,
read out of the caller's own inbox — not the agent's inbox
(`src/claude_teams/server_simple.py:1051-1052`, `1020-1031`).

An unknown agent name returns a stable all-zero payload with `state: "dead"`,
`alive: false` — **not** an error (`src/claude_teams/server_simple.py:883-903`,
`1543`). You cannot distinguish "never existed" from "killed" from this tool.

### `follow_up_agent(name, prompt, idempotency_key, replace_if_idle=True)`

See [section 4](#4-follow_up_agent-in-detail).

`idempotency_key` is **required** and is chosen by the caller *before* the call.
That ordering is the point: a server-generated id would only ever reach the
sender inside the response, so a sender that loses the response would have
nothing to ask about. Validation runs before anything is created and before any
waiting — missing/empty (`idempotency_key_required`), malformed
(`idempotency_key_malformed`), or over
`delivery_store.MAX_IDEMPOTENCY_KEY_LENGTH` (`idempotency_key_too_long`).

The uniqueness namespace is `(session, sender IDENTITY, key)`, so two senders
may reuse one textual key. Reusing a key with a byte-identical
recipient/prompt/options returns or reconciles the existing attempt and never
creates a second; reusing it with **any** differing field returns
`idempotency_conflict` and mutates nothing.

### `delivery_status(idempotency_key="", to="")`

The sender's query contract (R4). Returns
`{message_id, idempotency_key, to, status, phase, reason, attempts, nonce,
created_at, settled_at}` for one of **your own** messages. Another sender's key
is reported as `delivery_not_found` — the namespace is per-sender, so from here
it genuinely does not exist. Survives a server restart, because the store is a
file.

**It is an active reconciler, not a passive lookup.** An attempt sitting at
`phase="unconfirmed"` is rescanned for its receipt before this answers.
Without that, response-loss recovery would keep reporting `unconfirmed` forever
after the nonce had actually landed — a false status in the other direction.

Passing `to` instead returns a convenience list, reconciling **every** unsettled
row it is about to return, so the two views can never publish different
statuses for one message. It still cannot serve response-loss recovery: with
several messages to one agent, nothing identifies which row is the one whose
response was lost.

### `deliver_pending(idempotency_key="")`

Completes the cooperative tail. Reconciles every unsettled message of yours
first — rescanning for the previous attempt's nonce — and only then re-delivers,
so a prompt that already landed is never sent a second time.

**The drain allow-list is exactly `deliver_pending` and `follow_up_agent`.**
`agent_status`, `check_agent` and `list_agents` deliberately do **not** drain:
they are advertised as cheap reads, and draining there would turn each into a
slow mutator.

### `kill_agent(name)`

Terminal and destructive (`kill_agent._do_kill`):

0. **Refuses outright while a delivery to this agent is in flight** and its
   lease holder is provably alive, returning `reason="operation_in_progress"`
   with `retriable: true`, `holder_pid` and `operation_id`. Nothing is killed
   and the record survives. Killing under a live lease could orphan an
   already-spawned resumed child, and *waiting* here would deadlock — the
   holder needs this very lock to finalize. A lease whose holder is dead, or
   whose creation token no longer matches (PID reuse), is reconciled
   automatically and the kill proceeds. **Ordinary `kill_agent` never bypasses
   a live lease**; the escape hatch is CLI-only (see
   [section 4b](#4b-the-operation-lease)).
1. Signals the OS process **only** when `owns_process` proves it is still ours —
   live in-memory ownership or a matching creation token
   (`src/claude_teams/server_simple.py:1790-1793`,
   `src/claude_teams/backends/process_manager.py:253-267`). A reused or foreign
   PID is left running.
2. Removes the record from `agents.json`.
3. Deletes `state-{name}.json`, `inbox-{name}.jsonl`, and
   `inbox-{name}.pos.json` (`_cleanup_agent_artifacts`) — **any unread messages
   addressed to that agent are destroyed** — plus the legacy deterministic
   `prompts/{name}.prompt.txt`.
3a. Deletes `prompts/{name}.<nonce>.prompt.txt` **only when the child is
   confirmed gone** (we just killed it, or it was already dead). Otherwise only
   files past the 24 h age threshold are collected. A still-running CLI may not
   have read its prompt file yet, and a timeout is explicitly not licence to
   delete it (see [section 4c](#4c-prompt-file-lifecycle)).
4. Deletes the killed agent's sender entry from the caller's own cursor file
   (`src/claude_teams/server_simple.py:1759-1764`).

**Returns** `{"success": true, "name": name}`; `{"success": false, "name":
name, "reason": "operation_in_progress", ...}` under a live lease; `{"success":
false, "name": name}` with **no `reason` field** when the agent is not in the
registry
(`src/claude_teams/server_simple.py:1788`); `{"success": false, "name": name,
"reason": "session_not_found"}` with no session.

`success: true` does not prove the process died — step 1 is conditional.

A naturally-dead agent is **not** auto-removed; it stays listable and resumable
until killed (`src/claude_teams/server_simple.py:1777-1778`).

### `resume_session(session_id)`

Re-binds this lead to a prior session id after a restart
(`src/claude_teams/server_simple.py:1817-1846`). Validates the id is UUID-shaped
(blocking path traversal) and that its directory resolves directly under
`_SESSION_BASE` (`src/claude_teams/server_simple.py:1825-1834`).

**Returns** `{success, session_id, agent_count, lead_token}`, or `{success:
false, reason}` where `reason` is `session_id_required`, `invalid_session_id`,
or `session_not_found`.

### `session_info()`

Returns `{session_id, identity, cwd, agent_count, lead_token,
recoverable_sessions}` (`src/claude_teams/server_simple.py:1882-1889`).
`recoverable_sessions` excludes the current session
(`src/claude_teams/server_simple.py:1866-1868`) and lists only bindings whose
`agents.json` still holds a non-`killed` agent within the retention window
(`src/claude_teams/server_simple.py:466-521`).

### `list_agents(full=False)`

Compact rows `{name, state, alive, pid, backend, last_activity_at,
unread_count, binding}`. `full=True` returns the raw registry record plus
`last_line`, `truncated`, `full_len`, `binding`, and
`backend_session_id_verified`.

`binding` is the binding outcome (see [section 3a](#3a-transcript-binding)),
kept as its own field and never folded into lifecycle `state` — "this process is
running" and "we know which transcript is its" are independent facts. In a
compact row it is `null` when the row was answered entirely from the state
marker and no transcript was consulted.

`full=True` echoes the raw record, which includes whatever
`backend_session_id` is stored. `backend_session_id_verified` is `true` only
when this call actually bound that transcript, so a `legacy` record's stored id
is never presented as if it were verified. Empty list when no session
resolved (`src/claude_teams/server_simple.py:1945-1946`) — indistinguishable
from a session with no agents; call `session_info()` to tell them apart.

### `agent_status(names=None)`

The cheap path. Rows are exactly `{name, state, last_activity_ts, unread_count,
seq, heartbeat_age_s, stalled}`. Cost is one marker read + one cursor read + one
liveness check per agent; a rollout-log scan happens **only** as a fallback when
no marker exists. Unknown names are silently skipped.

When that fallback runs and the binding is neither `bound` nor `legacy`, `state`
is `"unknown"` and `last_activity_ts` is `null`: there is no activity signal
this tool is entitled to report, and guessing `running`/`idle` from an unrelated
transcript's mtime is exactly the bug the ladder exists to prevent. Liveness and
an authoritative marker still win — a dead process is `"dead"`, and a
hook-written `waiting`/`running` is a direct observation. The fallback stays
cheap: exactly one binding resolution, and a genuinely bounded one. It asks the
resolver for `bounded_only` mode, which drops the resolver's own all-history
fallback — without it a single call could still escalate to a full history walk
internally, so "one resolution" was not the same thing as "cheap".

### `agent_watch_paths(names=None)`

Returns `{has_session, session_dir, watch_argv, watch_command_bash,
watch_command_powershell, agents: [{name, state_marker_path}]}`
(`src/claude_teams/server_simple.py:2121-2135`) — no longer a bare list. Pure
path computation: it does not check that any file exists. Use it to rediscover
what to watch when a `spawn_agent` return was not retained; the `watch_*` fields
give the same ready-to-run command `spawn_agent` returns.

### `list_backends()`

`[{name, binary, default_model, supported_models}]` for backends whose binary is
on `PATH` (`src/claude_teams/server_simple.py:2088-2100`,
`src/claude_teams/backends/registry.py:82-90`).

---

## 3. On-disk contract

Everything lives under `~/.claude/agent-sessions/<session-uuid>/`
(`src/claude_teams/server_simple.py:77`, `198-199`).

| Path | Written by | Read by |
|------|-----------|---------|
| `agents.json` | server, under an exclusive file lock (`src/claude_teams/server_simple.py:316-346`) | server |
| `agents.lock` | lock file only (`src/claude_teams/server_simple.py:81`, `206-207`) | server |
| `session.json` | server, on binding (`src/claude_teams/server_simple.py:423-426`) | server recovery |
| `inbox-{name}.jsonl` | an agent's upstream `send_message` (append); since C3 only the owner's own children can write it (`src/claude_teams/server_simple.py:1332-1333`) | the owner's `read_messages`; watcher (`src/claude_teams/cli.py:223`) |
| `inbox-{name}.pos.json` | the owner's `read_messages` (`src/claude_teams/server_simple.py:1475`) | owner; watcher (read-only) |
| `state-{name}.json` | the **worker's own** lifecycle hook process (`src/claude_teams/hooks.py:88-89`) | server status tools; watcher |
| `prompts/{name}.<nonce>.prompt.txt` | server, before a claude-code spawn/resume (`_materialize_prompt`) | the worker itself, as a file read |
| `operation-leases.json` | server, temp-file + atomic replace (`src/claude_teams/leases.py:save_leases`) | server |
| `deliveries.json` | server, under `deliveries.lock`, temp-file + atomic replace (`src/claude_teams/delivery_store.py:save_records`) | server (`delivery_status`, `deliver_pending`) |
| `deliveries.lock` | lock file only (`src/claude_teams/filelock.py:file_lock`) | server |
| `mcp/{name}.mcp.json` | server (`src/claude_teams/server_simple.py:1122-1123`) | the claude CLI |
| `hooks-{name}.settings.json` | server (`src/claude_teams/hooks.py:147-148`) | the claude CLI |
| `codex-hook-{name}.cmd` | server, Windows only (`src/claude_teams/hooks.py:182-184`) | Codex's hook runner |

Two directories are created eagerly at session creation, `mcp/` and `logs/`
(`src/claude_teams/server_simple.py:1100-1101`).

### Inbox format

One JSON object per line, appended:
`{"from": <sender IDENTITY>, "text": <str>, "ts": <ISO-8601 UTC>}`
(`src/claude_teams/server_simple.py:1325-1331`). Append-only; nothing ever
rewrites or compacts it. The file is deleted only by `kill_agent`.

**`deliveries.json` is the opposite, on purpose.** Kill purges inbox lines and
cursor entries so a same-named successor does not inherit live actionable state.
Delivery records are **never deleted** — not by kill, not by agent-record
cleanup. The inbox is live state; the delivery store is the sender's audit
trail, and its whole value is that a settled outcome stays queryable after the
target is gone. Do not extend the inbox's delete-on-kill to it.

It takes the same cross-process file-lock transaction model as the registry
(`src/claude_teams/filelock.py`), not the inbox's `_inbox_lock` — that one is a
`threading.Lock` and is in-process only by design, because an inbox's owner is
the single writer of its own cursor. Several per-agent MCP servers share one
session dir, so the delivery store needs the real thing.

**Persistence fails closed.** `delivery_store.save_records` returns whether the
atomic replace actually reached disk, and `delivery_transaction` raises
`DeliveryStoreError` when a dirty transaction was lost. **No resume may begin
unless the pre-wait row is durably visible**: that row is the sender's only
handle on an in-flight message, so a lost write plus a lost response would leave
neither a reliable status nor a recoverable key — the exact hole the
caller-supplied idempotency key exists to close (R4/B0). The delivery tools
answer `queued(phase="pending", reason="delivery_store_unavailable",
retriable=true)` and send nothing.

**Reads fail closed too, and absence is not the same as an error.**
`delivery_store.load_records` and `leases.load_leases` return `{}` only when the
file does not exist yet — the legitimately-empty case. A store that exists but
cannot be read or parsed raises (`DeliveryStoreUnreadableError`,
`LeaseStoreUnreadableError`), because treating unknown state as empty would let
the next dirty transaction atomically **replace** the audit trail, make a key
whose row could not be read look unused (authorizing a duplicate delivery), and
grant a fresh lease over a live holder. The same rule applies one level in: a
malformed lease payload is not "no lease held". Callers inherit the fail-closed
behaviour they already had for a lost write.

Three writes are reached *after* something has already been mutated or decided,
and each now reports rather than swallows:

- **Kill-time reconciliation** still logs and continues — `kill_agent` is a
  lifecycle operation that must terminate the process, and leaving the rows
  unsettled is honest, nothing *was* settled. What changed is that the rescan
  is still possible afterwards: see "Kill-time cleanup" below.
- **`_release_delivery_claim`** returns whether the release reached disk, and
  drops its in-process claim registration first and unconditionally, so a lost
  write can no longer wedge the key (see `active_holder`, below).
- **`_discard_delivery_record`** (the C2 refusal rollback) returns whether the
  session really is byte-identical again. When it is not, the refusal says so
  (`record_discarded: false`, plus a note that the idempotency key is consumed)
  instead of promising that nothing changed while its row survives on disk.

`leases.save_leases` follows the same rule: a lease that did not reach disk is
not a lease, so `reserve_lease` reports `queued` rather than `granted`,
`reconcile_lease` raises `LeaseNotPersistedError` rather than reporting the
lease reclaimed, and `drop_agent` returns whether the entry actually went.

### Cursor semantics

`inbox-{name}.pos.json` is a **per-sender consumed-count**, not a byte or line
offset: `{"<sender>": <int count of that sender's messages already read>}`.

- Loading discards non-string keys, non-int values, booleans, and negatives
  (`src/claude_teams/messaging.py:19-26`).
- Saving is atomic via a uniquely named temp file and `replace`
  (`src/claude_teams/messaging.py:29-34`).
- **Clamping** happens on every read: a stored count larger than the observed
  message count for that sender is reduced to the observed count, including
  senders with zero observed messages
  (`src/claude_teams/server_simple.py:1413-1415`; also in
  `unread_sender_counts`, `src/claude_teams/messaging.py:70-71`). This is what
  prevents a corrupt forward cursor from permanently swallowing a sender's
  future messages.
- Unread for a sender is `len(messages) - min(cursor, len(messages))`, and only
  positive counts are returned (`src/claude_teams/messaging.py:64-75`).

### State marker schema

`{"state": "running" | "waiting", "event": "<hook event name>", "ts": <float
epoch seconds>}` (`src/claude_teams/hooks.py:88`). Written atomically via a
temp file + `replace` (`src/claude_teams/hooks.py:53-58`).

Readers tolerate a missing or corrupt file by returning `None`
(`src/claude_teams/server_simple.py:233-242`, `src/claude_teams/cli.py:162-165`).
A marker state outside `{"running", "waiting"}` is treated as absent and falls
through to the activity heuristic (`src/claude_teams/server_simple.py:151`,
`174-177`).

### Backend rollout logs (outside the session dir)

The server does not capture agent transcripts. It reads the CLIs' own logs.

- **claude-code**: `~/.claude/projects/<cwd with `\`/`:` replaced by `-`>/*.jsonl`.
- **codex**: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`, scoped by the
  rollout's `session_meta` `cwd`.

Which of those files belongs to a given agent is decided by the binding ladder
(next section), not by recency. `read_claude_output` / `read_codex_output`
remain as the **legacy** readers — they match on start time and mtime and are
used only for records that predate correlation ids.

#### Correlation ids

The correlation id is **generated once per spawn** (`new_correlation_id`) and
persisted as `correlation_id` on the agent record. It is deliberately *not*
derived from agent name + session id: a killed agent's name can be reused once
its record is removed, so a derived token could identify two different
conversations.

The id is **never re-derived** when a record does not carry one.
`classify_correlation` distinguishes:

| Record state | Classification | Meaning |
|---|---|---|
| `correlation_id` key absent | `legacy` | Record predates correlation. Compatibility case. |
| Present but empty/blank/wrong type | `unverified` | Corrupt. |
| Present, non-blank string | `valid` | Bindable. |

The two non-`valid` cases are deliberately **not** collapsed: absent means
"predates correlation", malformed means "corrupt", and only the first is a
compatibility case.

The record also carries `prompt_transport` (`"argv"` or `"sidecar"`), written at
spawn and rewritten at every resume. Gate 0 of the ladder needs it.

The id survives resume: `follow_up_agent` reads it off the record, passes it
back in `SpawnRequest.extra`, and writes it back on the updated record. Dropping
it there would silently reclassify a live agent as `legacy`.

`last_message` is the last assistant message, capped at 1000 chars with a
truncation marker (`src/claude_teams/agent_output.py:19`, `386-407`).
`last_activity_at` is the rollout file's mtime, not a semantic timestamp
(`src/claude_teams/agent_output.py:67`, `71-72`).

Separately, per-agent stdout/stderr logs go to
`~/.claude/teams/<session-id>/logs/<agent>.log`, overridable with
`WIN_AGENT_TEAMS_LOG_DIR` (`src/claude_teams/backends/process_manager.py:698-707`).

---

## 4. `follow_up_agent` in detail

### The delivery model: bounded in-call delivery with a cooperative tail

**The originating call performs the delivery.** When the target is busy the call
does not refuse and does not return immediately — it waits, bounded, for the
target to reach a resumable point, then resumes and confirms. Within the bound
this is genuine guaranteed delivery: no queue, no dependency on anyone coming
back.

**The bound is ONE total budget** (`_DELIVERY_CALL_BUDGET_SECONDS`), covering
the wait, lease acquisition, the resume, and confirmation — deliberately not a
per-step timeout, which would let one call spend the advertised budget several
times over. The server cannot know each client's deadline, so this is a
documented server-side constant rather than an assumption about the caller; it
is echoed back as `call_budget_s` on every result.

If the budget expires without a lease ever being acquired, the call returns
`queued(phase="pending")` together with an explicit `sender_obligation`, and the
message stays durably queryable. **The tail is cooperative**: there is no
dispatcher (see the non-goal in `requirements.md`), so nothing completes it
unless the sender calls `deliver_pending` or repeats `follow_up_agent` with the
same key.

### The delivery state machine

Public statuses are exactly R4's three. `sent` and `unconfirmed` are **phases
beneath `queued`**, never additional statuses — a four-state machine would
contradict R4, and exposing `sent` invites reading it as "arrived".

```text
queued(pending)     --(lease acquired, resume spawned)--> queued(sent)
queued(sent)        --(nonce confirmed)----------------->  delivered [terminal]
queued(sent)        --(child dead before receipt)------->  failed(not_delivered) [terminal]
queued(sent)        --(budget expired, child alive)----->  queued(unconfirmed)
queued(unconfirmed) --(nonce found on rescan)----------->  delivered [terminal]
queued(unconfirmed) --(child dead, grace passed, no nonce)-> failed(not_delivered) [terminal]
queued(pending)     --(budget expired, never leased)---->  queued(pending)  [cooperative tail]
```

Three rules that are easy to get wrong:

- **A never-leased message does not expire into `failed`.** There is one
  timeout — the call budget — so the same instant cannot mean both "come back
  for it" and "it will never happen". A pending message stays queued and
  queryable until delivered, reconciled, or the session is cleaned up.
- **A timeout with a live child never terminates as failed.** It stays
  non-terminal until the child is dead *and* one transcript-flush grace
  (`_UNCONFIRMED_FLUSH_GRACE_SECONDS`) has passed. A live child with no receipt
  legitimately stays `queued(unconfirmed)` indefinitely — honest, rather than
  silently expired. Only definite non-delivery is terminal.
- **A retry rescans for the prior attempt's nonce before re-sending.** This, not
  receiver-side dedupe, is what prevents a duplicate prompt: the recipient is a
  backend conversation, not a consumer with a dedupe table. A row still at
  `sent`/`unconfirmed` when a same-key call arrives is reconciled and then
  **reported, not resent**: if the receipt landed it answers `delivered`, if the
  child is provably gone with a complete negative scan it answers `failed`, and
  otherwise it answers `queued(unconfirmed)` with `reason="attempt_unresolved"`.
  Only a row genuinely back at `pending` is sent again.
- **One key is worked by one call at a time.** The FIFO ticket is derived from
  `(sender, key)`, so two concurrent identical calls share it and the lease
  queue cannot tell them apart from one caller retrying. The delivery record
  therefore carries an `active_holder` claim, taken under the store lock. A
  second concurrent call gets `queued(phase="pending",
  reason="delivery_in_progress")` and sends nothing.

  **When a claim may be reclaimed** — a crashed or wedged call must not hold a
  key forever (itself an R1 dead end), but reclaiming on a guess authorizes a
  second resume of one conversation. There are three cases, not two:

  - *Our own process.* `_ACTIVE_CLAIM_IDS` is exact: this process knows which
    per-call `claim_id`s it is working. A claim stamped with our PID whose id is
    not in that set is one we already finished — most often because its release
    write was lost — and is reclaimable. Reclaiming it cannot permit concurrent
    work, because the only process that could be doing that work is us.
  - *Another process, provably gone.* Reclaimable.
  - *Another process, ownership unprovable.* **Not** reclaimable.

  That third case is why liveness is three-valued.
  `process_manager.ownership_probe` returns `ours` / `not_ours` /
  `indeterminate`, and only `not_ours` authorizes a reclaim. `owns_process`
  remains a bool and remains the gate for *destructive* operations, where
  "unproven" and "not ours" must behave identically — but a reclaim gate refuses
  in the opposite direction, so the two cannot share one bool. An unreadable
  creation token against a live PID is uncertainty, not proof of death; the
  operation lease applies the same rule via `leases._holder_reclaimable`.

The lease is **not** held across `unconfirmed`. It converts to the durable
pending-delivery record on the agent, and the sender-side row stays at
`unconfirmed` — so a later delivery under that key reconciles rather than
resending, while neither a future delivery to that target nor a kill is blocked
by a lease nobody is progressing.

### Kill-time cleanup reconciles before concluding

`kill_agent` settles every in-flight attempt against the target, but **rescans
for the nonce first** and records `delivered` if it landed. An in-flight
attempt may already have an unread receipt on disk; marking that message failed
because the target is being killed would reintroduce exactly the false-status
problem this protocol exists to remove.

The scan reports one of four outcomes and they are **not** collapsed to a bool
(`_scan_for_nonce`, `ReceiptScanner.full_scan`):

| Outcome | Meaning | Result |
|---|---|---|
| `found` | the nonce is in a named receipt record | `delivered` |
| `absent` | every record parsed, none carried it | `failed(not_delivered)` |
| `indeterminate` | missing/unreadable transcript, a record that would not parse, or a partial trailing write | stays `queued(unconfirmed)` |
| `ambiguous` | the transcript rotated into more than one candidate successor | stays `queued(unconfirmed)` |

Only `absent` — a **complete authoritative negative** — may become terminal. An
error is not an absence, exactly as in the binding ladder. Nothing is deleted.

**The rescan survives the kill.** `kill_agent` removes the agent from
`agents.json`, and `_scan_for_nonce` answers `indeterminate` for a missing
record — so if the kill-time settlement write is lost (the one place a store
write is logged and swallowed), the row would previously have been stranded at
`unconfirmed` with nothing able to move it, because the only evidence path had
just been deleted. The transcript binding is therefore copied onto the durable
row at attempt time (`target_snapshot`, written by `_mark_attempt_sent` and
refreshed post-resume by `_record_outcome`), and `_scan_target` falls back to it
when the registry record is gone. A later `delivery_status` can then still find
the receipt and settle honestly.

The snapshot supplies **scanning only, never liveness**: it is a frozen copy,
and a recycled PID inside it would read as a live child. Callers pass the real
(possibly absent) record for the liveness question.

### It starts a NEW OS PROCESS

This is the single most important fact about this tool. `follow_up_agent` does
**not** write to the running agent's stdin, does not attach to its terminal, and
does not send it a message. It calls `backend.resume(...)`
(`src/claude_teams/server_simple.py:1703`), which builds a resume command and
hands it to `process_manager.spawn_process`
(`src/claude_teams/backends/process_base.py:89-93`) — a brand-new process, with
a new PID recorded over the old one
(`src/claude_teams/server_simple.py:1710-1726`).

Continuity comes purely from the backend CLI's own `--resume` /
`resume <id>` mechanism replaying the prior conversation
(`src/claude_teams/backends/claude_code.py:193-196`,
`src/claude_teams/backends/codex.py:387-404`).

### The liveness gate

Evaluated in this order, in `_do_follow_up`:

| Order | Condition | Refusal reason |
|-------|-----------|----------------|
| 1 | No session resolved | `session_not_found` |
| 2 | `name` not in `agents.json` | `no_delivery_path` (`state="record_removed"`) |
| 2a | Record has no `spawned_by` | `parent_unknown` |
| 2b | Caller's `IDENTITY` is not the record's `spawned_by` | `not_spawner` |
| 3 | Backend not loadable, or `supports_resume()` false | `backend_not_supported` |
| 3a | Binding outcome is not `bound` | `binding_<outcome>` |
| 4 | No `backend_session_id` known | `no_delivery_path` (`state="no_backend_session"`) |
| — | *`idle_by_marker` resolved here; a `waiting` marker skips 5–7* | |
| 5 | Alive, **not idle by marker**, and `last_message is None` | *not a refusal* — **bounded wait** (B2) |
| 6 | Alive, **not idle by marker**, and `last_activity_at is None` | `agent_state_unknown` |
| 7 | Alive, **not idle by marker**, and last activity < 60 s ago | *not a refusal* — **bounded wait** (B2) |
| 8 | Alive, judged idle, and `replace_if_idle=False` | `agent_idle_but_alive` |
| 8a | Another caller holds the operation lease | *queued, never refused* — see [4b](#4b-the-operation-lease) |
| 9 | `backend.resume(...)` raised | `resume_failed` |
| 10 | Resumed child exited inside the settle window | `resume_not_confirmed` |
| 11 | Resumed child died with no receipt | `not_delivered` |
| 12 | Scan bound expired, child still alive | *not a refusal* — `queued(phase="unconfirmed")` |
| 13 | The whole call budget expired without ever leasing | *not a refusal* — `queued(phase="pending")`, the cooperative tail |

Gates 2 and 2a/2b are ALSO evaluated once up front, read-only, before the
delivery record is created — a refusal from either must leave the session
byte-identical, and creating the record first would have written a file. They
are then re-evaluated under the registry lock, and that evaluation is the
authoritative one. Gates 2a/2b are evaluated first of all, before the binding
resolve and before any write; gate 3a is evaluated before the lease is reserved; gates 10-12 happen
**after** the registry lock has been released (see
[4a](#4a-delivery-confirmation)).

#### Gates 2a/2b — the direction guard (NOT a security boundary)

`follow_up_agent` is downstream-only: only the agent recorded in the target's
`spawned_by` may call it. A sibling, an unrelated agent, and — the case this
exists for — a worker targeting its own coordinator are all refused with
`reason="not_spawner"`, and the `detail` names the rule and points at
`send_message` as the upstream path. The reason it matters is that a follow-up
is kill-and-respawn: letting a worker follow up its lead would restart the
coordinator's process mid-task.

**This is an accident guard, not an authorization check, and must never be
described as one.** `IDENTITY` is read from an env var at import time by the
caller's own process (see
[Identity is process-global and read once at import](#identity-is-process-global-and-read-once-at-import)),
and the MCP server evaluating the guard *is* that process. A caller can assert
any identity it likes, and anything with filesystem access can edit
`agents.json` directly. Resisting a deliberate bypass would need a shared
broker, minted credentials and worker-unwritable state — an explicit non-goal.
The guard stops mistakes; it stops nothing else.

A record with no `spawned_by` refuses with `reason="parent_unknown"` rather
than being silently allowed: records written before the field existed cannot be
backfilled from anything trustworthy, and allowing them would disable the guard
during exactly the upgrade window in which unowned records exist. Recovery is
`kill_agent` + `spawn_agent`, or the operator CLI:

```
win-agent-teams adopt <session-id> <agent> <parent> \
    --token <session recovery token> --expect-generation <n>
```

Adoption is deliberately **not** reachable over MCP. An agent-callable version
would reintroduce the hole the guard closes: "callable only by a caller claiming
parentage" is tautological, because the operation itself writes the caller as
the spawner and identity is self-asserted — a confused worker could adopt its
own coordinator and then legitimately follow it up. The CLI form is gated on the
session recovery token *and* the record generation the operator observed (so an
adoption cannot be replayed against a record that moved underneath it), and it
records `spawned_by_source: "operator_asserted"` so a later reader can tell an
asserted parentage from an observed one.

**Adoption fills in a MISSING `spawned_by`; it does not re-parent.** A record
that already names a spawner is refused (exit 4). Without that the token plus a
generation would move *any* record to *any* parent, handing one agent
kill-and-respawn rights over another agent's child — broader than this contract,
and operator-gated is not the same as in-contract. Recovery for a wrong
parentage is `kill_agent` + `spawn_agent`.

A refusal at 2a/2b **changes nothing**: no PID, no regenerated MCP config, no
prompt sidecar, no lease acquired, no generation bump — and no delivery record.
That is why the check sits ahead of every side effect rather than alongside the
other gates.

The two evaluations are joined by a **parentage snapshot**, not just re-run.
The read-only pre-flight returns the `(spawned_by, spawned_by_source)` it
authorized against, and the under-lock check refuses with
`reason="stale_authorization"` when the record no longer matches it. Without
that the two can legitimately disagree — the pre-flight passes, the durable row
is created, and the locked check then refuses — leaving an audit row behind for
a call that was refused. A row created by a call that then hits `parent_unknown`,
`not_spawner` or `stale_authorization` is discarded, and only ever when this
call created it and nothing has been sent under it, so no evidence can be lost.

Checks 5–7 all sit **behind** the marker resolution. An earlier revision of this
document listed 5 and 6 as unconditional on `alive`, and carried an extra row for
an `output.busy_hint` check; that field and its branch no longer exist, and the
marker now precedes the transcript-derived checks rather than following two of
them.

Both `claude-code` and `codex` return `supports_resume() == True`
(`src/claude_teams/backends/claude_code.py:113-115`,
`src/claude_teams/backends/codex.py:252-254`), so reason 3 in practice means the
backend binary vanished from `PATH`.

**The authoritative signal is the `waiting` marker.** `idle_by_marker` is
computed by re-running `_resolve_agent_state` and testing for `"waiting"`. When
it is true, checks 5, 6 and 7 — the entire transcript-derived heuristic block —
are skipped. The comment above it explains why the ordering matters: an agent
parked at a Stop hook before emitting any assistant text has no `last_message`,
so checking that first would have made us WAIT out the whole call budget for
precisely the case we know for certain is idle. (Before B2 the same ordering
bug produced an `agent_busy` refusal; the ordering fix predates the change from
refusal to wait and still matters for the same reason.)

The inactivity heuristic (check 8) compares against the module constant
`_FOLLOW_UP_IDLE_SECONDS = 60.0` **directly**
(`src/claude_teams/server_simple.py:85`, `1642`) — not via `_idle_seconds()`, so
`WIN_AGENT_TEAMS_IDLE_SECONDS` does **not** affect this gate even though it does
affect the `state` field reported by `check_agent` / `agent_status`.

#### Gate 3a — the binding gate

A follow-up starts a new process against a stored `backend_session_id`. That is
only safe when we have positively identified which transcript belongs to this
agent, so **every non-`bound` outcome refuses**, with reason `binding_pending`,
`binding_unverified`, `binding_ambiguous`, `binding_legacy` or
`binding_indeterminate`.

| Outcome | `retriable` | Meaning for the caller |
|---|---|---|
| `pending` | `true` | Sidecar spawn whose read receipt has not landed. Retry shortly. |
| `indeterminate` | `true` | A candidate could not be read. Retry shortly. |
| `unverified` | `false` | No transcript can be attributed to this agent. Kill and respawn. |
| `ambiguous` | `false` | Two transcripts carry the marker. Kill and respawn. |
| `legacy` | `false` | Record predates correlation ids. Kill and respawn. |

The `retriable` flag is the whole point of the split: a caller that retries on
`unverified` spins forever, and one that gives up on `pending` fails a spawn
that was about to bind normally.

**`legacy` refuses, with no compatibility exception (R8).** An agent spawned
before correlation ids existed cannot be followed up. Its stored session id may
be exactly the wrong pinned id this mechanism exists to fix, and resuming on it
would let a delivery nonce be confirmed in someone else's conversation and
reported as `delivered` — the original bug with a false receipt attached. The
only recovery is `kill_agent` followed by a fresh `spawn_agent`; the refusal's
`detail` field says so. Read-only tools (`check_agent`, `list_agents`,
`agent_status`) keep working against legacy records unchanged.

Failure payloads carry `{success: false, name, reason, retriable}`, a `detail`
string for the binding and direction refusals, plus `alive`, `backend_session_id`,
`last_activity_at`, `binding`, and the unbounded `last_message` when a status
was computed.

### Kill-and-respawn under `replace_if_idle`

When the agent is alive and judged idle and `replace_if_idle=True` (the
default), the existing process is shut down before the resume
(`src/claude_teams/server_simple.py:1656-1659`):

```
if process_manager.owns_process(str(pid), _agent_create_token(agent)) \
   and not process_manager.graceful_shutdown(str(pid), timeout_s=5.0):
    process_manager.kill_process(str(pid))
```

The **creation-token ownership check fails closed**
(`src/claude_teams/backends/process_manager.py:253-267`): it returns `True` only
when this manager still holds a live in-memory child for that handle, or the
live PID's creation token equals the stored one. A tokenless recovered record,
an unreadable token, or a mismatch all yield `False`, and the process is left
untouched — the resume then proceeds anyway against `backend_session_id`,
potentially leaving the old process running.

Tokens are captured at spawn time from the just-live child
(`src/claude_teams/server_simple.py:1269-1273`) and re-captured after every
resume (`src/claude_teams/server_simple.py:1711`).

### 4a. Delivery confirmation

A returned PID is **not** evidence that a follow-up arrived, and neither is
transcript growth or a state-marker transition. Markers are keyed on agent
*name* and hooks write only `state`/`event`/`ts`, so a surviving old process and
a freshly resumed one write byte-identical markers.

So each attempt embeds a cryptographically random **delivery nonce** in the
final prompt (`wat-deliver:<32 hex>`, `src/claude_teams/delivery.py`), and
`delivered` is set only when that exact nonce is found in a **named receipt
record** of the transcript whose `backend_session_id` is being resumed.
Confirmation requires **both** child survival and the receipt record.

| Backend | Receipt record |
|---|---|
| `claude-code` | the `type: "user"` record — either literal user text (argv transport) or the `tool_result` carrying the prompt-file contents (sidecar transport) |
| `codex` | the rollout record for user input (`response_item` with `payload.role == "user"`) |

Four scanner rules are load-bearing:

- **Semantic, not substring.** The nonce is read from a *parsed* receipt
  record. A nonce echoed in a CLI diagnostic, in an assistant reply, or in
  serialized argv does not confirm anything.
- **Marker grammar is strict.** A fixed delimiter plus the full 32-hex id,
  with hex lookarounds on both sides — the prefix alone, a truncated id, or an
  id embedded in a longer hex run never match.
- **Record boundary, not raw EOF.** The pre-resume anchor is the offset of the
  last *complete* JSONL record. Partial bytes are retained between polls rather
  than skipped, because the readers drop malformed lines **permanently** and a
  fragment consumed at EOF would never be reconsidered once its remainder
  arrived.
- **Replacement means rotation, not absence.** Continuity is re-established by
  backend session id **plus** file identity. More than one candidate successor
  is `ambiguous`, unconditionally — the correlation token is **never** consulted
  to choose between candidates. It is written at spawn and a successor may
  legitimately not replay it, so its presence in one file is no evidence that
  the other is not the live conversation; selecting on it would be a guess.
  The scanner takes no correlation token at all.
- **Replacement detection is not size alone.** A transcript truncated and
  rewritten in place keeps its inode, and can be back to its old size before
  the next poll. Alongside `(st_dev, st_ino)` and a size regression, the leading
  bytes captured at snapshot time are re-checked: an append leaves them
  byte-identical, a rewrite does not.

#### The three non-delivery outcomes (R6)

| Situation | Result | Terminal? |
|---|---|---|
| Child exited inside the settle window | `reason="resume_not_confirmed"` | yes |
| Child died later with no receipt | `reason="not_delivered"` | yes |
| Bound expired, child still alive | `status="queued", phase="unconfirmed"` | **no** |

The third is R6's *live uncertainty*: a transcript write buffered past the bound
can still arrive, so it is neither delivered nor terminally failed. The attempt
is recorded on the agent as `pending_delivery` `{nonce, operation_id,
attempted_at, prompt_file}`, and the **next** `follow_up_agent` call rescans for
that nonce across the whole transcript *before* sending anything. When it is
found the call returns `{success: true, status: "delivered", reconciled: true}`
and **no second prompt is sent**.

Only a `delivered` or `unconfirmed` outcome writes the agent record. A resume
that never attached, or a child that died without a receipt, leaves the record
exactly as it was — there is nothing worth tracking and R6 forbids describing
either as progress.

### 4b. The operation lease

Confirmation polls for tens of seconds and therefore cannot run inside
`_agents_transaction`, which holds a **cross-process** file lock for its whole
body. The lock is released around resume-and-confirm.

Compare-and-swap after the fact is not sufficient: two callers could snapshot
the same generation, both resume, and both deliver distinct nonces, and the
losing CAS cannot undo an irreversible side effect. So a caller **atomically
reserves a per-agent lease while the registry lock is still held**
(`src/claude_teams/leases.py`). The lease holds `{generation, operation_id,
backend_session_id, nonce, holder_pid, holder_create_token, deadline}` and does
not itself resume. Finalization CASes on `operation_id` **and the agent
record's CURRENT generation**, read under the registry lock — not the
generation frozen into the lease payload, which is written at reservation time
and therefore always agrees with itself. `operation_id` is in the key because
generation alone is not enough: a name reused after removal starts a fresh
record that can legitimately be back at the same generation. The record
generation is in the key because that is what the operator force path bumps,
and the fence has to survive the window between the bump and the lease being
cleared. A fenced finalize writes nothing and leaves the lease alone.

- **A second valid caller queues** behind per-target FIFO with a ticket; it is
  **not** refused. Refusing a valid caller would hand back exactly the dead end
  R1 forbids. Once the wait budget is spent it gets the honest cooperative tail
  `{status: "queued", phase: "pending", reason: "operation_in_progress",
  retriable: true, queue_position}` — and nothing was sent.
- **The ticket is durable, and derived — not returned.** It is a hash of
  `(sender, idempotency_key)`, so a genuinely fresh `follow_up_agent` call with
  the same key reclaims the same queue place. Nothing has to be carried across
  calls beyond the key the caller already chose. A per-call ticket would leave
  every retry appending a new waiter behind its own orphaned head, and the
  queue would grow by one per attempt while the caller never advanced.
- **Promotion re-points the ticket at the granted `operation_id`.** A retrying
  caller mints a fresh `operation_id` on every poll, so the waiter record is
  updated to the id the lease is actually granted (and later finalized or
  released) under. Otherwise the promoted waiter is never dropped from the
  queue and later valid callers wait behind an orphan indefinitely.
- **Persistence failure fails closed.** A reservation whose atomic replace
  failed returns `queued`, never `granted`: the on-disk store still shows the
  target free, so granting would let another process resume the same
  conversation. A finalize or release that did not reach disk did not win.
- **Storage is crash-atomic and outside the registry.** `agents.json` is
  overwritten with a plain write, so a crash mid-write could destroy the
  registry *and* the lease. Leases live in `operation-leases.json`, written with
  the temp-file + atomic-replace pattern `save_inbox_cursors` uses.
- **Expiry is not fencing, and neither is a bare PID.** A holder that is alive
  but slow after spawning would otherwise let a second caller observe expiry,
  fail to find a not-yet-flushed nonce, and retry into a delivery still in
  flight. Reclaiming therefore checks holder liveness; **wall-clock expiry alone
  never justifies a resend.** Because a dead holder's PID can be reused,
  `holder_pid` is paired with `holder_create_token` and validated fail-closed,
  exactly as `owns_process` does — **including when the PID equals the server's
  own**. There is no self-PID shortcut: a lease left by an earlier incarnation
  whose PID was recycled onto this process would otherwise read as live
  forever, and nothing else can prove that holder gone.

#### The operator escape (CLI only)

Unconditional refusal would make a hung-but-live holder's agent permanently
unkillable. `win-agent-teams lease {inspect,clear,force} <session> <agent>
--token <lead_token>` is gated on the session recovery token and is not
reachable over MCP:

- `inspect` — reports the attempt nonce, holder liveness and the lease's
  generation, so an operator can check by hand whether the prompt landed.
- `clear` — removes a lease whose holder is dead or token-mismatched. It
  **refuses a provably live holder** (exit 3).
- `force` — for a live-but-overdue holder, and "overdue" is **enforced**: a
  holder that is provably live and still inside `lease.deadline` is refused
  (exit 3), because forcing it would kill a delivery that is doing exactly what
  the lease is for. Order is load-bearing: it **bumps the fencing generation
  first**, so the original holder can no longer win its finalize CAS; only then
  does it terminate the resumed child (and only when ownership is provable);
  only then does it release the lease.

`force` runs **all three steps inside one registry transaction**, and
revalidates the operation id against the store as its first act inside that
lock. Validating only at the final clear was not enough: the lease is read
before the lock is taken, and if the inspected operation finalized while a
queued caller was legitimately granted the target in between, the command fenced
and killed *that* caller's live delivery and reported the mismatch afterwards,
when nothing could be undone. `reserve_lease` grants only while holding this
same lock, so holding it across fence, kill and clear is what makes the
validation meaningful. Registry readers block for the duration; `kill_process`
is a terminate rather than the graceful-shutdown path, so that is bounded.

Both `clear` and `force` release the lease under the registry lock, as a
compare-and-swap on the operation id they read. A clear that does not match, or
that fails to reach disk, exits 4 and reports which operation actually holds the
lease — it never reports success for a lease it did not release. A lease store
that cannot be read at all exits 5.

`kill_agent` inherits the same fail-closed rule: if the lease store is unreadable
or a reclaim could not be persisted, it refuses with
`reason="lease_store_unavailable"` (retriable) and kills nothing, because it
cannot claim there is no delivery in flight.

### 4c. Prompt-file lifecycle

Prompt sidecars are `prompts/{name}.<nonce>.prompt.txt` — one per call, so two
concurrent calls to the same agent can never overwrite each other's prompt.

| Attempt outcome | Sidecar |
|---|---|
| `delivered` | removed immediately — the receipt *is* proof the CLI read it |
| `resume_not_confirmed` / `not_delivered` | removed — the child is provably gone |
| `unconfirmed` | **kept** — a live CLI may not have read it yet |
| `kill_agent`, child confirmed gone | all of the agent's sidecars removed |
| anything else | age-based GC only, 24 h |

There is deliberately **no** "delete this agent's stale files at the start of a
new call": that would race a concurrent attempt whose CLI has not read its file
yet. Timeout-failure alone is never sufficient grounds for deletion.

### `backend_session_id` scraping

The server never receives a session id from the backend. It **scrapes** it out
of the CLI's rollout log:

- claude-code: the first line's `sessionId` field
  (`src/claude_teams/agent_output.py:286-304`, `109`).
- codex: the `session_meta` record's `payload.id`
  (`src/claude_teams/agent_output.py:134-140`).

`_sync_backend_session_id` persists it onto the registry record the first time
it is observed **and only when the binding outcome is `bound`**, and both
`check_agent` and `follow_up_agent` write the registry back when it changes —
`follow_up_agent` at every one of its refusal exits, so a newly discovered id is
not lost when the call goes on to refuse. An earlier revision of this document
said any newly surfaced id is persisted; that is superseded, and the change is
load-bearing: an unverified read that wrote an id would poison the record for
every later call.

Consequence: **`follow_up_agent` returns `no_delivery_path` with
`state="no_backend_session"` until the target CLI has flushed enough of its
rollout file to be discovered.** A freshly spawned agent is not immediately
followable-up.

### 3a. Transcript binding

`_resolve_agent_binding` maps an agent record to exactly one transcript, or to a
named reason why it cannot. It returns a `BindingResult` with an `outcome` and,
for `bound` and `legacy` only, an `output`. The other four outcomes carry no
transcript-derived data, because there is no transcript we are entitled to
attribute to the agent.

#### The gates, in order

| Gate | Condition | Outcome |
|---|---|---|
| 1. Metadata | `correlation_id` absent | `legacy` |
| 1. Metadata | present but empty/blank/wrong type | `unverified` |
| 2. Scan | any candidate could not be read | `indeterminate` |
| 3. Count | zero matches, inside gate 0's window | `pending` |
| 3. Count | zero matches, otherwise | `unverified` |
| 3. Count | two or more matches | `ambiguous` |
| 4. Session id | one match, no parseable `sessionId` | `unverified` |
| 4. Session id | one match with an id | `bound` (re-pinning if it differs) |

**Gate 0 (sidecar-pending).** For a Claude sidecar spawn, argv carries only a
"read this file" instruction, so the correlation marker cannot appear in the
transcript until the agent has read the file and its tool result is recorded.
While `prompt_transport == "sidecar"`, the child is alive, and the spawn is
inside the pending window (`DEFAULT_SIDECAR_PENDING_WINDOW_S`, 120 s), zero
matches is `pending` rather than `unverified`. The window ends when the receipt
appears (the marker shows up, so the count gate binds normally), the child dies,
or the deadline passes — after which zero matches falls through to the count
gate like any other. `pending` is **call-local**: never cached, never persisted.

Gate 0 is numbered first in the design because it changes the meaning of an
observation, but it is evaluated where zero matches is decided: it is a branch
of the count gate, and it cannot precede the metadata gate because a `legacy`
record has no token to scan for at all.

**Gate 2 never guesses.** The scanner returns three states — present, definitely
absent, and *could not read*. An unreadable candidate makes the whole scan
`indeterminate`; a match count computed from a partially failed scan is not a
count we are entitled to use. `indeterminate` is retriable; `unverified` is
terminal.

**Gate 3 has no max-mtime fallback.** The token decides identity, never
recency. Two Claude sessions in one project directory, with the foreign one more
recently written, must still resolve to the agent's own transcript — that is the
core bug this ladder exists to fix.

#### Candidate enumeration

Two tiers, so a correct binding does not cost an all-history scan on every read
(`_resolve_agent_binding` is reached by `check_agent`, both `list_agents` forms,
and the `agent_status` fallback):

- **Tier 1** resolves the stored `backend_session_id` to a concrete path with
  **no mtime cutoff**, so a long-running session older than the window is still
  revalidated. This is not an existing primitive: Claude names transcripts after
  the session id but that is a convention, so the resolver tries the direct name
  first and falls back to parsing `sessionId`; Codex matches the `session_meta`
  `payload.id`.
- **Tier 2** is the correction scan: the existing mtime window first, then all
  history if the window yields nothing.

The tier-1 path **joins** the candidate set rather than short-circuiting it. A
tier-1 hit alone cannot answer the count gate, because a second transcript may
also carry the marker — and that is `ambiguous`, not a licence to keep the
stored binding.

#### The validated-binding cache

Successful bindings are cached in memory, keyed by `{backend, normalized cwd,
correlation_id, stored backend session id}`. Only `bound` is ever cached;
`pending`, `unverified`, `ambiguous` and `indeterminate` never are.

Each entry stores the canonical path, an OS file identity (`st_dev`, `st_ino`),
the size, a fixed-length header hash, the parsed session id, and
`BINDING_GRAMMAR_VERSION`. An entry is discarded when the path disappears, the
file identity changes, the file shrinks (truncation), the header hash changes
(replacement), the parsed session id no longer matches, or the grammar version
differs — so an entry written by older code is never trusted. **Appends do not
invalidate**: the header length is frozen at the size seen when the entry was
written, so growth leaves the hash untouched.

`(mtime_ns, size)` alone — the watcher's change detector — is deliberately not
used here. It detects change; it is not collision-proof identity.

### Prompt materialization and transport (claude-code)

The **server** builds the final prompt string and chooses its transport
(`_materialize_prompt`), because the backend cannot do it: the sidecar is
written before `backend.spawn`/`backend.resume`, and
`ClaudeCodeBackend._prompt_arg` then replaces argv with a fixed instruction.

Two rules, both load-bearing:

1. **Transport is decided from the user prompt alone.** `_needs_prompt_file`
   tests the *unmarked* user prompt against `_CLAUDE_PROMPT_FILE_CHARS`
   (`'`, `"`, `\n`, `\r`) **before** the correlation marker is appended. Testing
   the marked prompt would route every Claude spawn through a file read, since
   the multi-line marker form always adds a newline.
2. **The marker form differs per transport.** The argv branch appends a
   **single-line** marker joined by a space, which introduces no CLI-sensitive
   character — so the safety rule is respected, not bypassed. The sidecar branch
   appends the newline-delimited form. `_CLAUDE_PROMPT_FILE_CHARS` is unchanged.

When the sidecar is used, the server writes the **marked** prompt to a
per-attempt file, `prompts/{name}.<nonce>.prompt.txt` (`_attempt_prompt_file`),
passes `prompt_file_path` in `extra`, and the
backend puts only this fixed instruction on the command line
(`src/claude_teams/backends/claude_code.py:258-271`):

> `Read your complete task prompt from UTF-8 file path <path> then follow the file contents exactly.`

**The worker must actually read that file.** The real prompt never reaches the
model directly — and neither does the marker, until the read lands in context.

This applies identically to `spawn_agent` and `follow_up_agent`, and since A5
each attempt gets its **own** path: the per-attempt nonce is in the filename, so
a follow-up cannot overwrite the spawn prompt file and two concurrent calls to
one agent cannot collide on a single path. Cleanup is by age or on confirmed
child exit, never "delete the others because a new call started".

Codex has no prompt-file path: `_materialize_prompt` returns the prompt
unchanged for any non-`claude-code` backend. Codex relies on passing the prompt
as a verbatim argv token via the native `.exe`, falling back to a JSON-encoded
single-line form only when it is forced through the `codex.cmd` npm shim
(`src/claude_teams/backends/codex.py`, `_prompt_arg`). Its marker is appended by
`CodexBackend._correlated_prompt`, which **consumes** the server-issued id from
`extra["correlation_id"]` rather than deriving one — so a Codex prompt carries
exactly one marker, never two. With no id in `extra` the prompt goes out
unmarked; no id is ever invented.

---

## 5. The watcher

`win-agent-teams watch <session_dir> [--timeout 1800] [--pattern state-*.json] [--inbox/--no-inbox]`
(`src/claude_teams/cli.py`, the `watch()` command).

### The R3 delivery contract

Upstream messaging has no push: `send_message` appends a line and returns. The
recipient learns about it because its watcher wakes it. That makes the watcher a
**protocol component with obligations**, not a convenience — and the three
obligations below are load-bearing for every upstream message in the system.

They describe behaviour that already holds. C4 states them and pins them with
characterisation tests (`tests/test_watcher_contract.py`); it changed no watcher
code. Wake priority and the settle window are explicitly out of scope.

1. **Wake without consume.** A `reason="message"` wake reports which senders
   have unread messages and advances **no cursor**. The message is still unread
   afterwards, so the recipient — and only the recipient, via `read_messages` —
   consumes it. A second watcher started before the read wakes for the same
   message. Consequence: an unread message that is never drained re-wakes every
   subsequent watch immediately.
2. **Cursor clamping.** `unread_sender_counts` clamps a per-sender cursor to the
   message count (`consumed = min(cursor, total)`,
   `src/claude_teams/messaging.py`). A cursor ahead of the count — which
   `kill_agent` can produce, since it purges a sender's inbox lines — yields
   zero unread rather than a negative count or a phantom wake, and it is clamped
   **per sender**, so one stale cursor cannot mute another sender. Messages
   appended past the stale mark are reported normally.
3. **Exit 2 does not strand.** A timeout is not a consumption. Whatever caused
   the watcher to exit 2 — deadline, `--no-inbox`, an edge still inside its
   settle window — the unread message and its cursor are untouched and the next
   watch wakes for it. This is why "re-check status after exit 2" is a
   correctness instruction and not just advice.

**What the contract does not promise:** a *bounded* wake latency in the face of
the settle window. An unread inbox message wakes within one poll (0.5 s), but a
`waiting` marker waits out `_WATCH_SETTLE_SECONDS` (15 s default) first, and a
watcher that is not running at all wakes nobody. The guarantee is "no message is
lost between wakes", not "a wake is running".

### Reader identity

The watcher determines whose inbox to watch from its **own process
environment**: `AGENT_NAME`, falling back to `team-lead`
(`src/claude_teams/cli.py:222`). It watches `inbox-{reader}.jsonl` and
`inbox-{reader}.pos.json` (`src/claude_teams/cli.py:223-224`). Both backends do
set `AGENT_NAME` in the child environment, so a worker running the watcher gets
its own inbox.

### Poll loop

- Poll interval: **0.5 s** (`src/claude_teams/cli.py:19`, `309`).
- Default timeout: **1800 s** (30 min) (`src/claude_teams/cli.py`, the `watch()`
  `--timeout` option).
- File identity is `(mtime_ns, size)`, not bare mtime, so a same-tick atomic
  replace that changes size is still detected
  (`src/claude_teams/cli.py:109-126`).

### Wake priority: message > output > waiting

1. **Message** is checked at the top of each iteration, and once before the loop
   starts against the current unread state (`src/claude_teams/cli.py:227-237`,
   `254-264`).
2. **Output** is emitted next (`src/claude_teams/cli.py:292-294`).
3. **Waiting** is emitted last, and only after settling
   (`src/claude_teams/cli.py:300-306`).

### The actionable-edge rule

A changed `state-*.json` wakes the watcher only when its `state` is `"waiting"`
**and** its `event` is not in `_NON_ACTIONABLE_WAITING_EVENTS`
(`src/claude_teams/cli.py:150-174`). That set contains exactly `SubagentStop`
(`src/claude_teams/cli.py:26`) — an agent's *own* built-in Task subagent
finishing, while the agent itself is still mid-task.

`running` markers are never actionable. A marker with no `event` field, or a
non-string `event`, is treated as actionable for backward compatibility
(`src/claude_teams/cli.py:168-173`).

### Settle window

An actionable waiting marker is registered with its first-seen monotonic time
(`src/claude_teams/cli.py:284-285`) and only wakes once it has **persisted** as
waiting for `_WATCH_SETTLE_SECONDS` (`src/claude_teams/cli.py:304`). Each marker
path settles independently, so overlapping waits across several agents do not
drop each other (`src/claude_teams/cli.py:239-244`). A candidate that flips back
to `running` inside the window is dropped
(`src/claude_teams/cli.py:301-303`).

Default is **15.0 s** (`_WATCH_SETTLE_DEFAULT_SECONDS`), overridable via
`WIN_AGENT_TEAMS_WATCH_SETTLE_SECONDS`
(`src/claude_teams/cli.py:28`, `31-56`). A malformed, `NaN`, infinite, or
negative override falls back to the default; `0` is a valid value meaning
"settle disabled".

### Wake-without-consume for the inbox

The watcher calls `unread_sender_counts`, which is explicitly documented as *not*
advancing cursors (`src/claude_teams/messaging.py:64-65`) and never calls
`save_inbox_cursors`. **A `reason="message"` wake leaves the message unread; you
must still call `read_messages`.** Consequently, an unread message that is never
drained will re-wake every subsequent watch immediately.

### Exit codes and output

- **Exit 0** — one JSON object printed on stdout
  (`src/claude_teams/cli.py:177-179`), one of:
  - `{"reason":"message","from":[<senders>],"path":"<inbox path>"}`
  - `{"reason":"output","path":"<changed path>"}`
  - `{"reason":"waiting","agent":"<name>","path":"<marker path>"}`
- **Exit 2** — deadline reached, **nothing printed**
  (`src/claude_teams/cli.py:307-308`). Exit 2 does not mean nothing happened:
  the docstring notes a waiting transition may precede the initial snapshot, or
  a genuine edge may still be inside its settle window at the deadline
  (`src/claude_teams/cli.py:214-216`). Re-check status after exit 2.

### `reason="output"` requires a non-default `--pattern`

Classification splits changed paths: anything named `state-*.json` goes to the
waiting bucket, everything else to outputs (`src/claude_teams/cli.py:269-276`).
The default pattern is `state-*.json` (`src/claude_teams/cli.py:20`), so with
default arguments the outputs bucket is **always empty** and `reason="output"`
can never fire. It only becomes reachable with a widened `--pattern`.

---

## 6. Lifecycle hooks

The marker is written by a hook command the server injects into the worker's
CLI configuration. The hook runs `python -m claude_teams.hooks emit
--session-dir <dir> --agent <name>` and reads the event JSON from stdin
(`src/claude_teams/hooks.py:92-112`, `297-304`).

### Event → state mapping

| Hook event | Marker state |
|-----------|--------------|
| `SessionStart` | `running` |
| `UserPromptSubmit` | `running` |
| `PreToolUse` | `running` |
| `PostToolUse` | `running` |
| `Stop` | `waiting` |
| `SubagentStop` | `waiting` |

(`src/claude_teams/hooks.py:19-22`, `44-50`.)

Any unrecognized event name writes **nothing** and leaves the prior marker
intact, as does empty, non-JSON, or non-dict stdin
(`src/claude_teams/hooks.py:61-89`).

Note the asymmetry that the watcher exists to handle: `SubagentStop` writes a
`waiting` marker on disk, but the watcher filters it out as non-actionable
(section 5). Status tools do **not** filter it — `check_agent` and
`agent_status` will report `state: "waiting"` for a `SubagentStop` marker,
because `_resolve_agent_state` looks only at `state`, never at `event`
(`src/claude_teams/server_simple.py:154-182`).

### Wiring per backend

- **claude-code**: a settings file `hooks-{name}.settings.json` is written with
  every event mapped to the emit command
  (`src/claude_teams/hooks.py:136-149`), passed as `--settings <path>`
  (`src/claude_teams/backends/claude_code.py:242-256`). Suppressed by
  `WIN_AGENT_TEAMS_STATE_HOOKS=0`.
- **codex**: a list of `-c hooks.<Event>=[...]` overrides
  (`src/claude_teams/hooks.py:188-227`), plus
  `--dangerously-bypass-hook-trust` — without which Codex silently skips
  injected hooks (`src/claude_teams/backends/codex.py:407-437`). Suppressed by
  `WIN_AGENT_TEAMS_STATE_HOOKS=0` or `WIN_AGENT_TEAMS_STATE_HOOKS_CODEX=0`
  (`src/claude_teams/backends/codex.py:439-449`).
- **codex on Windows** additionally needs a `.cmd` launcher referenced as a
  single bare path in `commandWindows`, because Codex's `cmd /C` argv escaping
  corrupts a multi-token quoted command
  (`src/claude_teams/hooks.py:152-185`, `236-265`).

Hook wiring is regenerated on **every** spawn and every follow-up
(`src/claude_teams/server_simple.py:1249`, `1685`).

---

## Sharp edges

Facts about current behavior. No remedies are proposed here.

### A busy worker is now reachable — within a bound (was: reachable by neither tool)

`send_message` upstream still writes to a file and nothing more; downstream it
now routes through the guaranteed path (C3). `follow_up_agent` no
longer refuses a busy agent: both former `agent_busy` sites (the
no-`last_message` check and the inactivity timer, both behind the marker
resolution) are now bounded waits. A spawner addressing its own child gets
`delivered`/`failed` inside the budget, or `queued(phase="pending")` with a
stated obligation if the target never parks in time.

What remains true: there is still no push, stdin write or signal. Delivery
happens by *resume*, so it needs the target to reach a resumable point — and if
it never does within the budget, completing the message is the sender's job.

### Codex workers do not poll `read_messages`

Nothing in the codex spawn path arranges for polling. The prompt is the caller's
text plus the server-issued correlation marker
(`CodexBackend._correlated_prompt`); the environment is
`AGENT_NAME` / `AGENT_SESSION_ID` / `AGENT_PARENT_NAME` only
(`src/claude_teams/backends/codex.py:566-570`). There is no Codex equivalent of
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, which the claude-code backend sets to
put its workers in Claude Code's native team-messaging mode
(`src/claude_teams/backends/claude_code.py:284-285`). Messages sent to a Codex
worker's inbox therefore accumulate unread unless the caller's own prompt text
instructs the agent to poll. Since C3 this is much less likely to be reached by
accident: a spawner addressing its own Codex child gets the guaranteed path, not
an inbox append. It still applies to anything that writes upstream into a Codex
lead's inbox.

### `kill_agent` still has no caller-identity check

`follow_up_agent` now compares the caller's `IDENTITY` against the target's
`spawned_by` (`_direction_refusal`, gates 2a/2b above), so the flat registry no
longer means any agent can kill-and-respawn any other one through that tool.

`kill_agent` has no equivalent check: it looks the target up by name in the
session-wide `agents.json` and never consults the caller's identity. Because all
descendants share one flat registry (section 1), **any agent can kill any other
agent in the session, including its own lead**, provided that lead is itself a
spawned agent record. This is the same hazard class as the one gates 2a/2b
close, but it is lifecycle control rather than message delivery and is tracked
as a separate follow-up.

Note that closing it would have the same character as the direction guard —
an accident guard, not authorization — for the same reason: identity is
self-asserted (see [gates 2a/2b](#gates-2a2b--the-direction-guard-not-a-security-boundary)).

### `before = after` is load-bearing and ordering-sensitive

At `src/claude_teams/cli.py:280` the snapshot is advanced unconditionally —
including for ignored `running` and corrupt markers — so a non-ready write
cannot be rediscovered forever. Because that assignment has already consumed the
current output edge, the outputs emit **must** come before the settled-wait
emit; a wait matures over several polls and can settle in the same tick an
output arrives. Waking on the wait first would drop that output permanently.
The code carries this reasoning inline
(`src/claude_teams/cli.py:277-291`). Reordering these three statements silently
loses wake events without failing loudly.

### The only non-marker busy heuristic is a 60-second timer

When no `waiting` marker is available, busy/idle rests entirely on
`_FOLLOW_UP_IDLE_SECONDS` — was there transcript activity in the last 60 s.
There is no signal derived from what the agent is actually doing.

(A vestigial `AgentOutput.busy_hint` field once suggested otherwise. It was
never set `True` by any code path, so its branch was unreachable; both were
removed. Any document or memory describing it is stale.)

### `WIN_AGENT_TEAMS_IDLE_SECONDS` does not affect the follow-up gate

It is read by `_idle_seconds()` (`src/claude_teams/server_simple.py:104-112`),
which is used only by `_resolve_agent_state`
(`src/claude_teams/server_simple.py:180`). The `follow_up_agent` inactivity
check uses the bare constant (`src/claude_teams/server_simple.py:1642`). Tuning
the env var changes reported `state` but not who is judged busy enough to be
waited for rather than resumed immediately.

### `kill_agent` destroys the target's unread inbox (but not delivery records)

`inbox-{name}.jsonl` and its cursor are deleted outright
(`src/claude_teams/server_simple.py:1751-1752`). Any message sent to that agent
and not yet read is gone. Delivery records in `deliveries.json` are deliberately
**not** touched: they are the sender's audit trail and must outlive the target.

### A follow-up no longer overwrites the spawn prompt file

Historical, and recorded because the old behaviour was published here. Prompt
sidecars used to be keyed by agent name alone, so a claude-code follow-up whose
prompt contained a quote or newline rewrote the same file the original spawn
used, and a worker re-reading that path saw the follow-up prompt instead of the
original. Since A5 the path is `prompts/{name}.<nonce>.prompt.txt`
(`_attempt_prompt_file`), unique per attempt, so neither a follow-up nor a
concurrent second call can overwrite another attempt's prompt.

### A spawned agent may not have the name you requested

`_unique_agent_name` appends `-2`, `-3`, … on collision and truncates the base to
fit 64 characters (`src/claude_teams/server_simple.py:360-374`). Kill does free
the name for reuse, since it removes the record
(`src/claude_teams/server_simple.py:1794-1795`).

### `send_message` to a child no longer behaves like `send_message`

Resolved as of C3 — recorded because the *shape* of the result now depends on
the recipient. An upstream send returns `{"success", "to"}`; a downstream send
returns the full delivery schema (`status`, `phase`, `idempotency_key`, …) and
requires a key. A caller that branches on `result["to"]` alone, or that assumes
`send_message` never blocks, will be surprised: the downstream branch spends the
same bounded call budget as `follow_up_agent`.

The older hazard this replaces — an unknown recipient silently rerouted to the
lead with only a `warning` field, which callers checking `success` never
noticed — no longer exists. Unknown, sibling, and unrelated recipients are
refused with `reason="recipient_not_addressable"` and nothing is written.

### `check_agent` cannot distinguish "unknown" from "dead"

An unrecognized name returns the same `state: "dead"`, `alive: false` payload as
a genuinely dead agent (`src/claude_teams/server_simple.py:883-903`, `1543`).

### Ambient inbox cursor writes are not cross-process locked

The lock guarding load/advance/save is a `threading.Lock` scoped to one Python
process, deliberately so (`src/claude_teams/server_simple.py:245-249`). The
invariant relied upon is that only the inbox owner's identity writes its own
cursor. Two ambient MCP server processes sharing an `IDENTITY` and session
would race. `external_read` is the deliberate exception: it holds the
session's cross-process agents lock across the full cursor transaction because
the portable member token can be used by an old and new server process at once.
