# PRD: External agent join — let a manually started session register as a child of the lead

> **Design pivot (2026-07-23, after plan review 1):** the "rebind process
> identity" mechanism proposed under *Proposed design* below was rejected in
> review ([plan-review-1.md](plan-review-1.md), 42/100) — it is unsafe under a
> shared Claude Desktop MCP process, non-atomic, and restart-unsafe. The
> committed design is **token-carried identity**: `join_team` returns a
> `member_token` and new `external_send`/`external_read`/`leave_team` tools
> carry it per call; no process-global identity is ever mutated. See
> [plan.md](plan.md) §0. The goals, non-goals, user story, and acceptance
> intent below are unchanged.

## Problem

Spawned agents run headless in the Claude Code CLI and therefore have no access
to the Claude Desktop preview browser. When an orchestrator session needs a
**visual QA agent** (one that can drive the in-app browser, take screenshots,
and file visual bug reports), the human today has to:

1. Ask the orchestrator to produce a prompt for a new, manually started
   Claude Desktop session.
2. Start that session by hand.
3. Have the two sides communicate through an ad-hoc agreed file location
   (bug-report files the orchestrator polls), with no identity, no inbox, no
   delivery semantics, and no visibility in `list_agents` / `agent_status`.

The MCP server already has everything needed for structured two-way
communication (per-agent inboxes, cursors, state markers, a flat registry) —
but there is no way for a process that was **not** spawned by `spawn_agent` to
obtain a child identity and a registry record.

### Why the current system cannot do this

- Identity is a **read-once module global** resolved from `AGENT_NAME` /
  `WIN_AGENT_TEAMS_SESSION_DIR` at import (`_resolve_identity`,
  `src/claude_teams/server_simple.py:137`). A fresh Desktop session has neither
  env var, so its win-agent-teams server resolves to `team-lead` — it becomes a
  *second root lead in its own session*, not a child.
- The only writers of `agents.json` records are `spawn_agent`
  (`spawned_by_source: "spawn"`) and the CLI-only `adopt` command, which merely
  backfills a missing `spawned_by` on an *existing* record. There is no
  MCP-callable registration path.
- Session discovery (`_recover_session_id`) is keyed on identity + parent PID +
  cwd. A separate Desktop process in a different cwd will never auto-find the
  lead's session; the session id must be handed over explicitly.
- Downstream guaranteed delivery (`_guaranteed_send`, PR #36) works by
  **resuming the child's CLI process** and confirming a nonce receipt in its
  transcript. A Desktop conversation cannot be resumed by the server, so that
  machinery cannot apply to an externally attached agent.

## Goals

1. A human can start a fresh Claude Desktop (or any interactive) session and,
   on the orchestrator's instruction, register it via a single MCP tool call as
   a **child of the lead** in the lead's existing team session.
2. After joining, messaging is symmetric with spawned agents wherever possible:
   - child → lead: `send_message(to="team-lead", ...)` appends to the lead's
     inbox exactly like a spawned child (works unchanged).
   - lead → child: `send_message(to="<child>", ...)` succeeds and delivers via
     the child's inbox (pull), instead of being refused or attempting a
     CLI resume.
3. The external child appears in `list_agents` / `agent_status` / `check_agent`
   with an honest backend label and liveness signal, and the lead's existing
   watch tooling can observe it.
4. Joining is **authorized by the lead** — an arbitrary process that merely
   knows a session id cannot attach itself and read/write team traffic.
5. `kill_agent` and session cleanup handle external children safely
   (deregister, never kill the PID — it is the user's own MCP server process).

## Non-goals

- No push/wake of the Desktop session from the server side (no resume, no
  process signaling). Delivery to an external child is pull-based; an optional
  Stop-hook wake (reusing `install_lead_wake`, which is already
  identity-generic) is a follow-up, not part of this feature.
- No guaranteed-delivery receipts (PR #36 semantics) for external children.
  The lead gets `delivery: "inbox"` — honest, weaker semantics.
- No re-parenting, no joining as a child of an arbitrary mid-level lead in v1
  (the ticket names the parent; v1 only exercises `team-lead`, but the design
  must not hard-code it).
- No multi-tenant hardening beyond the ticket token — the threat model is
  "processes on the same machine under the same user", consistent with the
  rest of the disk contract.

## User story (primary)

> As an orchestrator (root lead) running a team of spawned workers, I want to
> issue a "join ticket" for a visual QA agent, hand the human a paste-ready
> prompt, and have the manually started Desktop session register itself as my
> child — so that I can send it QA tasks and receive its bug reports through
> the normal inbox protocol instead of an ad-hoc file convention.

## Proposed design

### New tool 1: `create_join_ticket` (called by the lead)

```
create_join_ticket(name: str, note: str = "") -> {
  success, name,            # de-duplicated via _unique_agent_name
  session_id, token,        # one-time secret, 32-hex nonce
  expires_at,               # default TTL 24h
  join_prompt               # paste-ready text for the human (see below)
}
```

- Requires resolved identity; the caller becomes the `parent` recorded on the
  ticket.
- Writes the ticket into the session dir, e.g.
  `join-tickets.json` (array, guarded by its own lock, same
  temp-file-plus-atomic-replace pattern as `deliveries.json`):

  ```json
  {"name": "visual-qa", "token": "<32hex>", "parent": "team-lead",
   "created_at": 169..., "expires_at": 169..., "note": "...",
   "status": "open"}
  ```

- Reserves the name (de-dup against both `agents.json` and open tickets) so
  the ticket's name is the name the joiner will get.
- `join_prompt` is a complete instruction block the human pastes into the new
  session: what the agent's role is (from `note`), plus the literal
  `join_team(session_id=..., token=...)` call to make first, plus the
  read/answer protocol (poll `read_messages` after each task, report to
  `team-lead` via `send_message`). This replaces today's hand-written prompt.

### New tool 2: `join_team` (called by the external session)

```
join_team(session_id: str, token: str) -> {
  success, name, parent, session_id,
  inbox_path, state_marker_path,
  instructions   # short protocol reminder for the joining model
}
```

Behavior, in order:

1. **Eligibility guard.** Only allowed when the calling process currently has
   the *default root* identity (`IDENTITY == team-lead`,
   `_IDENTITY_UNRESOLVED == False`) **and** has not spawned agents or bound a
   session with activity in it. A spawned worker (real `AGENT_NAME`) or an
   unresolved-identity process (#24 sentinel) is refused with a clear reason.
   This prevents identity hijack and double-join.
2. Validate `session_id` shape and location exactly like `resume_session`
   (UUID directly under `~/.claude/agent-sessions/`).
3. Under the `agents.json` lock: look up an open, unexpired ticket matching
   `token`; consume it (`status: "used"`, record joiner pid + ts). Invalid,
   expired, or already-used tokens are refused — tickets are strictly
   one-time.
4. Append the agent record:

   ```json
   {"name": "<ticket.name>", "pid": <server pid>, "backend": "external",
    "session_id": "<session_id>", "parent": "<ticket.parent>",
    "status": "running", "spawned_at": <ts>, "cwd": "<server cwd>",
    "model": null, "create_token": "<new nonce>",
    "spawned_by": "<ticket.parent>", "spawned_by_source": "join_ticket"}
   ```

5. Write `state-<name>.json` = `{state: "running", event: "joined", ts}` so
   the lead's watch tooling sees it immediately (external agents run no
   lifecycle hooks; see "Liveness" below).
6. **Rebind process identity**: set the module globals (`IDENTITY`,
   `_AGENT_PARENT_NAME`, `_session_id`) to the joined values and persist a
   session binding (`_persist_session_binding`) under the new identity, so an
   MCP server restart in the same Desktop conversation recovers the same
   identity instead of reverting to `team-lead`. This is the first legitimate
   post-import identity mutation in the codebase; it must be a single
   dedicated function (`_assume_identity(...)`) with the eligibility guard
   inside it, not scattered global writes.

### New tool 3: `leave_team` (called by the external session)

Marks the record `status: "completed"`, updates the state marker
(`state: "waiting", event: "left"`), and reverts nothing else. Idempotent.
The lead can also retire the child with `kill_agent` (see below).

### Changes to existing tools

- **`send_message` / `_classify_recipient`** (`server_simple.py:1030`): a
  target record whose `backend == "external"` and `spawned_by == IDENTITY` is
  classified as an **inbox child**: append `{"from","text","ts"}` to
  `inbox-<child>.jsonl` (same code path as the spawner direction) and return
  `{success: true, delivery: "inbox", note: "external agent; pull-based —
  it reads on its next read_messages call"}`. No `idempotency_key` required,
  no leases, no `deliveries.json` entry.
- **`follow_up_agent`**: refuse for `backend == "external"` with reason
  `external_agent_pull_only` and a hint to use `send_message`.
- **`kill_agent`**: for `backend == "external"`, never signal the PID (it is
  the user's Desktop MCP server, possibly shared). Deregister only: mark the
  record `killed`, purge inbox/cursor per existing semantics, update the state
  marker. Return `{success: true, killed_process: false, reason:
  "external_agent_deregistered"}`.
- **`check_agent` / `agent_status` / `list_agents`**: PID liveness of the MCP
  server process is a usable but weak signal (the server may outlive the
  conversation). Surface `backend: "external"` and the last state-marker /
  inbox-read timestamp so the lead can judge staleness.

### Liveness / heartbeat

External agents run no hooks, so the state marker is bumped opportunistically:
every `read_messages` and `send_message` call made *by* an external identity
also rewrites its own `state-<name>.json` (`state: "running", event:
"activity"`). Cheap (one small atomic write) and gives the lead a
last-seen signal through the existing watch CLI.

### Message flow after join

```
lead ── send_message(to="visual-qa") ──▶ inbox-visual-qa.jsonl   (append)
visual-qa ── read_messages() ──────────▶ drains own inbox        (pull)
visual-qa ── send_message(to="team-lead") ─▶ inbox-team-lead.jsonl
lead ── read_messages() / lead-wake hook ──▶ drains own inbox
```

The joining prompt instructs the external agent to call `read_messages`
whenever it finishes a task or is asked to check in. Optionally the human can
run the existing `watch` CLI pointed at the child's inbox, and the external
session can install the existing Stop-hook wake (`install_lead_wake` is
identity-generic) — both are usability follow-ups, not requirements.

## Files affected (expected)

- `src/claude_teams/server_simple.py` — new tools `create_join_ticket`,
  `join_team`, `leave_team`; `_assume_identity`; `_classify_recipient` branch;
  guards in `follow_up_agent` / `kill_agent`; heartbeat in
  `read_messages` / `send_message`.
- `src/claude_teams/` (possibly a new `join_tickets.py` mirroring
  `delivery_store.py` for the ticket store + lock).
- `tests/test_join_team.py` (new), plus additions to existing
  `send_message` / `kill_agent` / classification tests.
- `docs/reference/agent-messaging-protocol.md` — new section on external
  agents; README backend table gains an `external` row (not a backend adapter;
  no spawn path).
- Tool docstrings — the consuming agent reads only these; the entire disk
  contract for joining/messaging external agents must be stated there.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Shared MCP server process in Claude Desktop.** If the Desktop app shares one win-agent-teams process across conversations, `join_team` rebinds identity for *all* of them. | Eligibility guard refuses join once the process has session activity; document the limitation; verify Desktop's actual process model during implementation (one server per conversation vs per app) and record the finding in `implementation.md`. |
| **Identity hijack / spoofing** — anything that mutates `IDENTITY` post-import is adjacent to bug #24. | One-time lead-issued token; join refused for unresolved-identity processes and for processes that already hold a real `AGENT_NAME`; all mutation confined to `_assume_identity` with the guard inside. |
| **`kill_agent` killing the user's Desktop MCP server.** | Hard rule: `backend == "external"` ⇒ never signal the PID. Covered by a dedicated test. |
| **Stale external children** (human closes the Desktop window; nothing updates the registry). | Heartbeat timestamps + lead-side staleness display; `kill_agent` deregisters cleanly; ticket TTL prevents stale open tickets. |
| **Message loss on the pull path** — the lead believes "sent" means "seen". | `send_message` return explicitly says `delivery: "inbox"` (pull, unconfirmed); tool docstring spells out the semantics so orchestrator prompts set expectations. |
| **Name collisions** between tickets and future spawns. | Ticket names reserved through the same `_unique_agent_name` de-dup, checked against open tickets as well. |

## Open questions

1. Should `join_team` also be allowed for **mid-level leads** as ticket
   issuers in v1 (any resolved identity can call `create_join_ticket`), or
   root-lead-only? Proposal: allow any resolved identity — "lead is a role,
   not an identity" — the ticket's `parent` is simply the issuer.
2. Should the joining session be allowed to **spawn its own children** after
   joining (it is now a mid-level lead by role)? Proposal: yes, no extra code
   needed — worth a smoke test.
3. Ticket TTL default (proposal: 24 h) and whether `create_join_ticket`
   should support revocation (proposal: defer; consuming or expiring covers
   v1).

## Acceptance criteria

1. Lead calls `create_join_ticket(name="visual-qa", note=...)` → gets a
   one-time token and a paste-ready `join_prompt`.
2. A fresh session (default root identity) calls
   `join_team(session_id, token)` → record appears in the lead's
   `list_agents` with `backend: "external"`, `spawned_by: <lead>`; state
   marker exists; second use of the same token is refused.
3. Lead `send_message(to="visual-qa", ...)` succeeds with
   `delivery: "inbox"`; the external session's `read_messages` returns it.
4. External session `send_message(to="team-lead", ...)` lands in the lead's
   inbox unchanged from spawned-child behavior.
5. A spawned worker (real `AGENT_NAME`) and an unresolved-identity process
   (#24 sentinel) are both refused by `join_team`.
6. `kill_agent("visual-qa")` deregisters without signaling the PID; the
   external session's subsequent calls report it is no longer registered.
7. `follow_up_agent("visual-qa")` is refused with
   `external_agent_pull_only`.
8. Full-repo quality gates (ruff, ty, pytest) green on Linux.

## Test cases (minimum)

- Ticket lifecycle: create, consume, reuse-refused, expired-refused, name
  de-dup against registry and open tickets.
- Join happy path: record shape, state marker, identity rebind, binding
  persistence (simulated server restart recovers the joined identity).
- Eligibility refusals: already-named identity, unresolved sentinel,
  post-activity root.
- Classification: lead→external = inbox append; sibling→external refused;
  external→parent = spawner path.
- `kill_agent` on external: no signal sent (assert via monkeypatched
  `os.kill`), record + inbox cleanup.
- Heartbeat: `read_messages` by an external identity bumps its state marker.
- Smoke (manual, documented): real Desktop session joins a live orchestrator
  session, exchanges one round-trip message pair.
