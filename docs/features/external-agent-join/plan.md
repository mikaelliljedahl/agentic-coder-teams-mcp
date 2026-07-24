# Implementation plan v4: External agent join (token-carried identity)

Lets a manually started, interactive session (Claude Desktop) register itself,
on the lead's instruction, as a **child** of the lead in the lead's existing
team session — registry record, inbox, two-way messaging — without being
launched by `spawn_agent`.

- **Spec inputs (priority order):** [PRD.md](PRD.md),
  [plan-review-1.md](plan-review-1.md), [plan-review-2.md](plan-review-2.md),
  `CLAUDE.md`, `docs/reference/agent-messaging-protocol.md`.
- **Stack / tests:** Python 3, FastMCP, pytest. New `tests/test_join_team.py`
  plus edits to existing suites (§4).
- **Revision history:** v1 (process-global identity rebind) rejected 42/100;
  v2 (token-carried identity) 58/100; v3 74/100 — mechanics largely accepted,
  three blockers remained (reconciliation state machine, delivery-row
  cleanup, containment/recovery semantics). v4 fixes those; §8 dispositions
  review 2, §9 dispositions review 3 ([plan-review-3.md](plan-review-3.md)).

## 0. Design (unchanged from v2 in direction)

No process-global identity is ever mutated. `join_team` mints a
self-locating bearer **member token**; dedicated tools `external_send`,
`external_read`, `leave_team` carry it per call. Shared-Desktop-process and
MCP-restart safe by construction: the token lives in the conversation
transcript, and every call re-derives authority from the registry.

**Token grammar (exact contract, review-2 NB1):**
`wam1:<session-uuid>:<secret>` — exactly three colon-delimited fields;
version literal `wam1`; canonical lowercase hyphenated UUID text (round-trips
`str(uuid.UUID(x)) == x`); secret exactly 64 lowercase hex chars. Any
deviation ⇒ `invalid_member_token` (no exception escapes; non-string/corrupt
stored values are treated as no-match, never raised).

**Credential at rest (review-2 blocker 1):** `agents.json` stores only
`member_token_digest = sha256(secret).hexdigest()` — never the secret.
Verification: digest the presented secret, compare via
`hmac.compare_digest(computed.encode(), stored.encode())` (fixed-type bytes).
The secret itself is **derived deterministically** from the join ticket:
`secret = sha256(f"wam-member:{ticket.ticket_id}:{ticket.token}").hexdigest()`
— so an idempotent `join_team` replay re-derives and returns the same
`member_token`. **Honest scope of this scheme (review-3 NB1):** it removes
the member secret from `agents.json` and closes every *tool-level* disclosure
path; it does NOT protect against a reader of `join-tickets.json`, who holds
the derivation inputs (`ticket_id` + `token`) and can reconstruct the secret
— exactly as a reader of `session.json` holds today's `lead_token`.
File-level access is the accepted same-user threat model. (A server-keyed
HMAC would harden the ticket-store case but adds key
persistence/recovery obligations; deliberately not done.)

## 1. Current state (all pointers opened and verified in this working tree)

1. Identity globals read once at import
   ([server_simple.py:119](../../../src/claude_teams/server_simple.py#L119)–121,
   `_resolve_identity` [:137](../../../src/claude_teams/server_simple.py#L137)–160,
   `ROOT_LEAD_NAME` [:127](../../../src/claude_teams/server_simple.py#L127)).
   **Never written by this feature.**
2. `_session_id` [:345](../../../src/claude_teams/server_simple.py#L345);
   `_active_session_id` [:841](../../../src/claude_teams/server_simple.py#L841)–859;
   `_pending_recovery` [:250](../../../src/claude_teams/server_simple.py#L250),
   read/cleared by `_annotate` [:831](../../../src/claude_teams/server_simple.py#L831)–838.
   `external_*` tools call **none** of these (enforced by fail-fast
   monkeypatch tests, §5 test 12).
3. Recovery: `_candidate_sessions`
   [:649](../../../src/claude_teams/server_simple.py#L649)–690 (identity+cwd,
   parent-PID-blind); `_non_terminal_agents`
   [:693](../../../src/claude_teams/server_simple.py#L693)–704 — its **only**
   consumer of `_TERMINAL_STATUSES`
   [:191](../../../src/claude_teams/server_simple.py#L191) (verified by grep;
   v2's "kill-related listing" claim was wrong and is withdrawn).
4. Registry lock `_agents_file_lock`
   [:499](../../../src/claude_teams/server_simple.py#L499)–508 (cross-process);
   plain non-atomic registry save
   [:518](../../../src/claude_teams/server_simple.py#L518)–519. Inbox cursor
   lock `_inbox_lock` [:449](../../../src/claude_teams/server_simple.py#L449)–455
   is **explicitly in-process only** (comment
   [:440](../../../src/claude_teams/server_simple.py#L440)–444) and relies on a
   single-process single-reader premise that a portable bearer token breaks —
   §3.5 adds cross-process serialization for external readers.
5. Spawn: name de-dup from `agents` only, under the lock (`_do_spawn`
   [:2287](../../../src/claude_teams/server_simple.py#L2287)–2296); record
   shape [:2363](../../../src/claude_teams/server_simple.py#L2363)–2391.
6. Routing: `_classify_recipient`
   [:1030](../../../src/claude_teams/server_simple.py#L1030)–1073;
   `send_message` [:2412](../../../src/claude_teams/server_simple.py#L2412)–2487
   (CHILD ⇒ `_guaranteed_send` [:2453](../../../src/claude_teams/server_simple.py#L2453);
   SPAWNER ⇒ append [:2475](../../../src/claude_teams/server_simple.py#L2475)–2485).
7. Guaranteed path — **there is no single pre-mutation boundary** (review-2
   blocker 5): `_guaranteed_send`
   [:3997](../../../src/claude_teams/server_simple.py#L3997) validates the key
   [:4016](../../../src/claude_teams/server_simple.py#L4016), preflights
   [:4041](../../../src/claude_teams/server_simple.py#L4041), then **creates**
   the durable row via `_open_delivery_record`
   [:4047](../../../src/claude_teams/server_simple.py#L4047)–4049 before
   `_guaranteed_delivery`; `deliver_pending`
   [:4246](../../../src/claude_teams/server_simple.py#L4246) **claims** an
   existing row (`_claim_delivery_record`, call at
   [:4290](../../../src/claude_teams/server_simple.py#L4290)) before its
   `_guaranteed_delivery` call [:4296](../../../src/claude_teams/server_simple.py#L4296);
   backend is first resolved only inside `_guaranteed_delivery._prepare`
   ([:2940](../../../src/claude_teams/server_simple.py#L2940)–2969,
   [:3018](../../../src/claude_teams/server_simple.py#L3018)–3030). ⇒ guards
   must sit at **two early call sites + one authoritative recheck** (§3.6).
8. `kill_agent` [:4411](../../../src/claude_teams/server_simple.py#L4411)–4512
   holds `_agents_transaction` through record removal + cleanup; signals the
   PID when `owns_process` proves ownership
   [:4491](../../../src/claude_teams/server_simple.py#L4491)–4496.
9. Status: `check_agent` payload has `backend`
   [:1313](../../../src/claude_teams/server_simple.py#L1313); compact
   `list_agents` row has it [:4648](../../../src/claude_teams/server_simple.py#L4648);
   `agent_status` row does **not** ([:4741–4749], docstring promise). **Full
   `list_agents` returns the raw record — docstring promise at
   [:4665](../../../src/claude_teams/server_simple.py#L4665)–4667 and
   `{**agent, ...}` splat at
   [:4691](../../../src/claude_teams/server_simple.py#L4691)–4694** — the
   token-leak surface §3.7 closes. Shared binding wrapper
   [:1232](../../../src/claude_teams/server_simple.py#L1232)–1244 is called by
   compact list [:4634](../../../src/claude_teams/server_simple.py#L4634)–4638,
   full list [:4685](../../../src/claude_teams/server_simple.py#L4685), and the
   `agent_status` fallback [:4722](../../../src/claude_teams/server_simple.py#L4722).
10. Marker: `_state_marker_file`
    [:373](../../../src/claude_teams/server_simple.py#L373); atomic write
    pattern [hooks.py:63](../../../src/claude_teams/hooks.py#L63)–68; safe-name
    regex `_SAFE_AGENT_RE` [hooks.py:39](../../../src/claude_teams/hooks.py#L39).
11. CLI `watch --reader` exists and validates
    ([cli.py:504](../../../src/claude_teams/cli.py#L504)–510,
    `_require_safe_reader` [cli.py:473](../../../src/claude_teams/cli.py#L473)–484);
    server-side `_watch_argv`
    [:1096](../../../src/claude_teams/server_simple.py#L1096)–1101 has **no**
    reader parameter — §3.3 extends it.
12. Session-id validation to copy: `resume_session`
    [:4541](../../../src/claude_teams/server_simple.py#L4541)–4550.
13. TTL-parser precedent `_idle_seconds`
    [:253](../../../src/claude_teams/server_simple.py#L253)–261 accepts
    zero/negative/NaN/inf — the ticket TTL parser must **not** copy it
    (review-2 NB3): require finite and `> 0`, else the 24 h default.

## 2. Disk & API contract (committed)

### 2.1 Ticket store — `join-tickets.json`

JSON array in the session dir; mutated **only under `_agents_file_lock`**;
saved atomically (temp + `os.replace`). Row:

```json
{"ticket_id": "<uuid4 hex>", "name": "visual-qa", "token": "<uuid4 hex>",
 "parent": "team-lead", "note": "...", "created_at": 0.0, "expires_at": 0.0,
 "status": "open", "used_at": null, "member_name": null}
```

- TTL default 24 h; env `WIN_AGENT_TEAMS_JOIN_TICKET_TTL_SECONDS`, parsed
  strict-finite-positive (§1.13).
- **Retention:** `used`/expired rows kept ≥ 7 days
  (`WIN_AGENT_TEAMS_JOIN_TICKET_RETENTION_SECONDS`, same parser) so replay
  keeps answering `token_already_used` / idempotent-rejoin; pruned on any
  ticket-store write.
- **Name reservation = `open` AND unexpired** tickets only (review-2 NB6):
  expired rows retained for audit do not reserve names. Requested names must
  match hooks' `_SAFE_AGENT_RE` before de-dup (`invalid_name` otherwise).
- Docstring wording: the join token is a **replayable recovery credential
  during retention**, not strictly one-time — replay returns the same
  membership idempotently (review-2 NB2).

### 2.2 External agent record

```json
{"name": "<ticket.name>", "pid": <os.getpid()>, "backend": "external",
 "session_id": "<sid>", "parent": "<ticket.parent>", "status": "running",
 "spawned_at": <ts>, "cwd": "<cwd>", "model": null,
 "member_token_digest": "<sha256 hex>", "join_ticket_id": "<ticket_id>",
 "spawned_by": "<ticket.parent>", "spawned_by_source": "join_ticket"}
```

- `status`: `running` → `left` (`leave_team`); kill removes the record.
- **Recovery candidacy — discoverable, never auto-adopted (review-3 blocker
  3b):** "session discoverable/addressable" and "silently auto-adoptable"
  are split. `_non_terminal_agents` is **unchanged** (external records still
  count), so an external-only team stays visible in
  `session_info.recoverable_sessions` and remains explicitly adoptable via
  `resume_session` — a legitimate lead whose exact binding missed can still
  recover the session to message, inspect, or kill its external member. What
  changes is the **silent** path only: `_recover_session_id`'s
  single-candidate auto-adopt ([:797](../../../src/claude_teams/server_simple.py#L797)–805)
  additionally requires the candidate to hold ≥ 1 non-terminal
  **non-external** record; a session whose only live records are external is
  never adopted without an explicit `resume_session` call. This closes the
  review-2 hole (a lingering external row silently re-adopting the lead's
  session into a fresh ambient process) without losing legitimate recovery.
- `pid` is informational only (stale after Desktop restarts; docstring says
  so).

### 2.3 Tool surface

As v2 (§2.3 there), with these deltas:

- `join_team` returns `watch_argv` built by a new
  `_watch_argv(session_dir, reader=None)` parameter that appends
  `--reader <name>` (test asserts the token **following** `--reader` equals
  the member name).
- `external_read` mirrors `read_messages` params (`from_agent`, `since_seq`,
  `full`, `limit`, `max_chars`) with identical semantics/validation, reader =
  member name.
- Marker/heartbeat failure semantics (review-2 NB5): the inbox append /
  cursor advance is the user-visible operation; if it succeeded but the
  activity-marker write fails, the tool returns **success** with
  `heartbeat_warning: true` — never a failure that invites a duplicate send.
  In `join_team` only, the `joined` marker is part of the contract: a marker
  write failure there returns `{success: false, reason:
  "marker_write_failed", retriable: true}`; the replay path repairs it
  (§3.4).

### 2.4 Membership resolution & locking (review-2 blockers 3–4)

One helper owns the whole lifecycle:

```python
@contextmanager
def _member_operation(member_token, *, allow_left=False):
    # parse & validate token grammar (§0)  -> refusal outside the lock
    # validate session id like resume_session -> session_not_found
    with _agents_file_lock(sid):
        # locate record: backend=="external" AND
        #   hmac.compare_digest(sha256(secret), record["member_token_digest"])
        # no record            -> membership_revoked
        # status=="left"       -> membership_revoked(detail="left")
        #                         unless allow_left
        yield record, agents, sid     # ALL side effects happen HERE
```

- **The registry file lock is held through every side effect** — inbox
  append, the complete cursor load/read/advance/save, marker write, status
  flip. Since `_agents_file_lock` is cross-process (§1.4), this
  simultaneously provides the **cross-process cursor serialization** that
  review-2 blocker 4 demanded: two *token callers* presenting the same token
  (old and new MCP server) serialize on the session's lock file; each message
  is consumed by exactly one of them. (Scope honestly stated per review 3:
  response loss after cursor persistence keeps the existing at-most-once API
  behavior, and an ambient reader with a forged `AGENT_NAME` bypasses this
  lock — same-user threat model.) External traffic is low-volume;
  serializing it against registry ops is an accepted cost. **Platform note
  (review-3 NB2):** lock waiters time out after 30 s on Windows
  ([:184](../../../src/claude_teams/server_simple.py#L184)) but block
  indefinitely in `flock` on POSIX (`filelock.py:43–58`); a large
  `external_read` can therefore stall concurrent spawn/kill/status calls.
  Documented in the tool docstrings, and §5 test 28 bounds the contention
  with a large-inbox fixture.
- **Lock order (committed):** agents file lock → `_inbox_lock(name)`
  (in-process) → file I/O. `kill_agent` already mutates under
  `_agents_transaction` (§1.8), so kill/leave/send/read are mutually
  linearized; a validated send can never append after a kill removed the
  record.
- `leave_team` uses `allow_left=True`: already-`left` ⇒ `{success: true,
  already_left: true}` with **no** marker rewrite (fixes the v2
  `_resolve_member`-vs-idempotency contradiction).

## 3. Work breakdown (TDD; red tests in §5)

### 3.1 Constants & helpers
`SPAWNED_BY_SOURCE_JOIN = "join_ticket"` (agent_output.py);
`CREDENTIAL_FIELDS = frozenset({"member_token_digest"})`;
`_join_tickets_file` / atomic load/save; strict TTL+retention parsers;
`_member_secret(ticket_id, token)` derivation; `_member_operation` (§2.4);
`_write_state_marker` (atomic, hooks pattern); `_build_join_prompt` (pure;
deterministic layout: fenced sections whose fence for the `note` block is
**chosen to not occur in the note text** (extend backtick run, the standard
Markdown technique), so even a note containing ``` cannot terminate its block
— review-3 NB6 replaces v3's unqualified "cannot alter" claim; test 1 includes
a delimiter-bearing note).

### 3.2 `create_join_ticket` + spawn reservation (both orders)
Tool as v2; de-dup input = `agents` ∪ **open-and-unexpired** ticket names.
`_do_spawn` de-dup input likewise (same lock already held). Tests cover
ticket-then-spawn, spawn-then-ticket, and concurrent (§5 t3).

### 3.3 `join_team` — reconciliation-first algorithm (review-2 blocker 2)

Outside the lock: validate `session_id` (§1.12). Under
`_agents_file_lock(sid)`, with tickets and agents both loaded:

1. Locate ticket by token; no ticket ⇒ `invalid_or_expired_token`. **Do NOT
   apply expiry yet** (review-3 blocker 1: expiry must never mask crash
   recovery).
2. **Locate record(s) by `join_ticket_id`, regardless of ticket status.**
   Multiple matches ⇒ `{success: false, reason: "registry_corrupt"}`, no
   mutation. One match: validate immutable fields — `name`, `parent`,
   `member_token_digest` (against the ticket derivation), **and**
   `backend == "external"`, `session_id == sid`,
   `spawned_by == ticket.parent`, `spawned_by_source == "join_ticket"`
   (review-4 NB2: a corrupt same-`join_ticket_id` non-external row must never
   receive a membership return); any mismatch ⇒ `registry_corrupt`, no
   mutation.
3. Dispatch on **(ticket.status, record presence, record.status)**:
   - `open` + no record + ticket **unexpired** → insert record (§2.2), save
     agents; mark ticket `used` (+`member_name`, `used_at`), save tickets;
     ensure marker; return fresh membership.
   - `open` + no record + ticket **expired** → `invalid_or_expired_token`.
   - `open` + record(`running`) — crash window A, **regardless of expiry**
     (the join already happened; expiry only gates *new* joins) → mark ticket
     `used`, save; ensure/repair marker; return idempotent success.
   - `used` + record(`running`) — normal replay or crash window B →
     ensure/repair marker; return idempotent success with the re-derived
     `member_token`.
   - any-status + record(`left`) → `{success: false, reason:
     "membership_revoked", detail: "left"}` — replay after `leave_team` NEVER
     rewrites the marker or resurrects membership (review-3 blocker 1).
   - `used` + no record (kill removed it, or record lost) →
     `token_already_used`.
   - **Corruption precedence (review-4 NB1):** unknown ticket status ⇒
     `registry_corrupt` and it takes precedence over every record-state rule
     (including any-status-left) — a malformed ticket store has exactly one
     stable answer. Likewise `record.status not in {"running", "left"}` ⇒
     `registry_corrupt` (fail-closed default), no mutation.
4. **Marker ensure/repair is conservative and schema-aware (review-4 NB3):**
   it writes `running/joined` only when the marker file is missing,
   unparseable, **or parseable but schema-invalid** (`state` not in
   `{"running","waiting"}`, missing `event`, or non-numeric `ts` — state
   readers already treat such markers as absent, so leaving one would not
   satisfy the joined-marker contract); a schema-valid existing marker (e.g.
   a newer `running/activity`) is left byte-untouched.
5. Marker ensure failure ⇒ `marker_write_failed, retriable: true` (§2.3);
   every replay re-runs step 3's dispatch and step 4's ensure/repair.

Return: `{success, name, parent, session_id, member_token, inbox_path,
state_marker_path, watch_argv (with --reader), instructions}`. No
`_annotate`, no `_active_session_id`, no `_require_resolved_identity`.

### 3.4 `external_send` / `external_read` / `leave_team`
All three run entirely inside `_member_operation` (§2.4). `external_read`
extracts the current `read_messages` body into `_read_inbox(session_id,
reader, ...)` (both call it; ambient behavior unchanged — the in-process
single-reader premise still holds for ambient identities, documented at the
`_inbox_lock` comment). Heartbeat bump per §2.3 semantics.

### 3.5 Lead→external delivery (`send_message`)
CHILD case becomes a single `_agents_transaction`: re-resolve the record by
name **inside the lock** and recheck `spawned_by == IDENTITY`, backend, and
status before acting (review-2 §7-row-6 requirement): external+`running` ⇒
append + `{"delivery": "inbox"}`; external+`left` ⇒ `member_left`;
non-external ⇒ existing `_guaranteed_send` path unchanged.

### 3.6 Guaranteed-path guards — two call sites + authoritative recheck
1. `_guaranteed_send`: after key validation and `_preflight_refusal`
   (read-only, [:4041]), a read-only external-target check **before
   `_open_delivery_record`** ⇒ `external_agent_pull_only`; `deliveries.json`
   byte-identical, no lease, no claim.
2. `deliver_pending`: resolve the pending row's target record **before
   reconciliation and before `_claim_delivery_record`** ([:4290]) ⇒ same
   refusal, store byte-identical.
3. Authoritative recheck in `_guaranteed_delivery._prepare` under the agents
   lock, **before backend registry lookup or lease reservation**
   ([:3018–3030], [:3169 area]) ⇒ closes the target-replacement race (name
   killed and re-joined as external between preflight and prepare), without
   `registry.get("external")` ever being called.
4. **Durable-row settlement for the authoritative race (review-3 blocker 2):**
   `external_agent_pull_only` is treated as a **C2 no-side-effect refusal** —
   it is added to `_C2_REFUSAL_REASONS`
   ([:4083](../../../src/claude_teams/server_simple.py#L4083)–4100 rollback),
   so when the *current* `_guaranteed_send` call created the row, the row is
   discarded on refusal, the claim is released, and the idempotency key
   becomes reusable — **qualified honestly (review-4 NB4): "reusable once the
   discard persisted"; `_release_delivery_claim`/`_discard_delivery_record`
   persistence failures keep their existing report-and-log behavior
   (`record_discarded: false`), never a false clean-rollback promise.** For a
   **pre-existing** row (a `deliver_pending` drain, or a retry of a row
   created before the target became external), the claim is released and the
   row is left **unmutated** (audit preserved) with the stable refusal
   returned; it is never attempted again and never consumes a lease.
   `deliver_pending`'s public contract gains a **separate `refusals` list**
   (review-4 NB5 — the durable/public delivery statuses stay exactly
   `queued/delivered/failed`; refusal is an attempt outcome, not a stored
   status): entries `{idempotency_key, to, reason:
   "external_agent_pull_only"}` alongside the existing attempted count and
   the unchanged `deliveries` audit rows.

### 3.7 Credential sanitizer + status honesty
- `_public_agent_record(agent)`: strips every `CREDENTIAL_FIELDS` key. Applied
  in **full `list_agents`** (the `{**agent}` splat, [:4691]), compact list
  rows, `check_agent` payload assembly, and `agent_status` rows. Leak tests
  assert none of: `member_token_digest`, ticket `token`, derived secret
  appear in any tool result (§5 t23). **The full-list docstring is updated**
  from "raw registry record" ([:4665–4667]) to "sanitized registry fields
  (credential fields omitted)" so the published schema matches the redaction
  (review-3 NB7).
- `agent_status` row gains `backend` (docstring schema + existing schema
  tests updated — module identified at implementation time by grepping for
  the schema assertion; review-2 NB5/NB-workflow accepted).
- External binding centralized: the shared wrapper ([:1232–1244]) returns
  `not_applicable` for `backend == "external"`, so compact list / full list /
  `agent_status` fallback all inherit it (review-2 NB4-centralization).

### 3.8 Ambient-tool containment (review-2 blocker 6)
Honest scoping first: token-carried identity eliminates **global-state
contamination**; it does not remove the **ambient root toolset** from the
member's MCP process, and a wrong ambient call (`send_message`,
`list_agents`, `kill_agent`…) still acts as a root lead and can adopt a
same-cwd session via `_active_session_id` (§1.2–1.3). The plan therefore
ships a **technical control**, not prompt discipline alone:

- **`WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` server mode:** the server registers
  **only** `join_team`, `external_send`, `external_read`, `leave_team` (and
  `list_backends` for diagnostics). Implementation note (review-3 NB3):
  registration happens at decoration time for *every* tool
  ([:2233 et al.]), so the mode is implemented as one gate function wrapping
  ALL `@mcp.tool()` registrations (e.g. a `_register_tool(fn, external=bool)`
  helper or a post-definition registration loop) — not a condition sprinkled
  near the new tools. `_with_disk_note`'s decoration-time ordering
  requirement ([:1176–1184], `tests/test_tool_descriptions.py:130–142`) is
  preserved; tests assert both registered names AND client-visible
  descriptions per mode.
- **Supported isolated deployment (review-3 blocker 3a):** the *supported*
  QA setup is a **separate Desktop profile / separate client instance whose
  MCP config contains ONLY the `win-agent-teams-external` entry** (plus
  whatever browser tooling QA needs). In that deployment the ambient root
  toolset is absent from the client surface — actual isolation, not prompt
  discipline. INSTALL.md documents this as the recommended configuration.
- **Dual-entry fallback is named for what it is:** if the user runs the
  external entry alongside the main server in one profile, ambient tools
  remain selectable and only the `join_prompt` discourages them. The plan
  does not call this containment; it is a documented degraded mode. Whether
  Desktop profiles can scope MCP configs per profile is verified during the
  smoke test and recorded in `implementation.md`; **if no isolated
  deployment exists on the user's client, shipping proceeds but README/
  INSTALL must state plainly that ambient-tool isolation is unavailable
  there** (release-note gate, not a code gate — the feature's correctness
  never depends on it).
- The "recoverable session" half of review-2 blocker 6 is fixed by §2.2's
  auto-adopt/discoverability split.

### 3.9 Docs & workflow artifacts
Tool docstrings carry the full contract (token grammar, replayable-recovery
wording, `delivery:"inbox"` pull semantics, revocation-on-next-call,
kill-never-signals, `pid` staleness, EXTERNAL_ONLY mode).
`docs/reference/agent-messaging-protocol.md` "External members" section;
README backend-table row + EXTERNAL_ONLY note; INSTALL.md config block;
`implementation.md` + implementation-review artifact per repo workflow.

## 4. File change list

| File | Change |
|---|---|
| `src/claude_teams/server_simple.py` | **Edit** — everything in §3 except below. |
| `src/claude_teams/agent_output.py` | **Edit** — `SPAWNED_BY_SOURCE_JOIN`. |
| `tests/test_join_team.py` | **New** — matrix §5. |
| existing suites: `agent_status`/`list_agents` schema tests, send/kill/classify tests, recovery tests | **Edit** — external rows, sanitizer, `backend` field, auto-adopt non-external-record condition (modules named at implementation time via grep; noted per review-2 NB5). |
| `docs/reference/agent-messaging-protocol.md`, `README.md`, `INSTALL.md` | **Edit** — §3.9. |
| `docs/features/external-agent-join/implementation.md` | **New** at implementation time. |
| `registry.py`, `backends/` | **No change.** |

## 5. Test matrix (red-first; exact assertions)

Tickets & join:

| # | Test | Asserts |
|---|---|---|
| 1 | `test_create_ticket_token_prompt_exact` | 32-hex token; reserved name; `join_prompt` contains the **literal** session id, the literal token, the exact strings `join_team(`, `external_send`, `external_read`, "save", "member_token", restart instruction, and `--reader <name>`; `note` appears only inside its fenced block. |
| 2 | `test_ticket_name_safe_and_dedup` | `../evil` ⇒ `invalid_name`; collision vs agents ⇒ `-2`; vs open ticket ⇒ `-2`; vs **expired** ticket ⇒ NOT bumped (expired rows don't reserve). |
| 3 | `test_spawn_ticket_reservation_both_orders` | ticket→spawn ⇒ spawn gets `-2`; spawn→ticket ⇒ ticket gets `-2`; concurrent (threads + barrier) ⇒ names disjoint. |
| 4 | `test_join_happy_path` | record exact fields incl. `member_token_digest` (sha256 of derived secret), `join_ticket_id`; exactly one row; marker `running/joined`; return `member_token` matches grammar and re-derivation; `watch_argv[watch_argv.index("--reader")+1] == name`; lead `list_agents` shows `backend=="external"`. |
| 5 | `test_join_replay_idempotent_and_used_no_record` | replay ⇒ same member_token, still one record; `used`+record-deleted ⇒ `token_already_used`; expired ⇒ `invalid_or_expired_token`. |
| 6 | `test_used_ticket_retention` | within retention, old used token ⇒ `token_already_used` even after later ticket writes; post-retention row pruned. |
| 7 | `test_concurrent_join_same_token` | two threads, barrier ⇒ exactly 1 record, both get identical member_token. |
| 8 | `test_crash_window_A_open_ticket_existing_record` | state: record saved, ticket `open` ⇒ replay marks ticket used, repairs marker, **no duplicate record**; **same outcome when the ticket has meanwhile EXPIRED** (crash A survives TTL). |
| 9 | `test_crash_window_B_marker_missing_and_marker_preservation` | state: both stores saved, no marker ⇒ replay writes `running/joined`; state: valid newer `running/activity` marker present ⇒ replay leaves it byte-identical (conservative repair). |
| 9b | `test_join_replay_after_leave_and_after_kill` | after `leave_team`: replaying the join token ⇒ `membership_revoked, detail:"left"`, marker still `waiting/left` (mtime unchanged); after `kill_agent`: replay ⇒ `token_already_used`. |
| 10 | `test_join_marker_write_failure_retriable` | injected marker failure ⇒ `marker_write_failed, retriable: true`; replay succeeds and repairs. |
| 11 | `test_registry_corrupt_states` | parameterized: duplicate `join_ticket_id` rows; immutable-field mismatch (incl. non-external backend); unknown ticket status (wins over left-rule); `record.status` outside {running,left} ⇒ all return `registry_corrupt`, stores byte-identical. |
| 12 | `test_token_grammar_matrix` | wrong version, 2/4 fields, uppercase UUID, non-canonical UUID, 63/65-char secret, non-hex secret, non-ASCII ⇒ `invalid_member_token`; wrong-but-well-formed secret ⇒ `membership_revoked`; deleted session dir ⇒ `session_not_found`; corrupt stored digest (int/None) ⇒ `membership_revoked`, no exception. |

Messaging, revocation, cursors:

| # | Test | Asserts |
|---|---|---|
| 13 | `test_lead_send_to_external_inbox` | `{success, delivery:"inbox"}`; 1 inbox line; deliveries.json byte-identical; no lease. |
| 14 | `test_external_read_full_cursor_semantics` | drain-once; `from_agent`+`since_seq` windowing; `since_seq` **without** `from_agent` ⇒ error; negative `limit` ⇒ `ValueError`; `full=True` ignores `limit`; multi-sender ordering; exact `unread_count`/`has_more`; `limit=0` watermark no-consume **and cursor file byte-stable**; `max_chars` truncation flags — identical results to `read_messages` on the same fixtures. |
| 15 | `test_external_send_and_heartbeat` | line in parent inbox `from==name`; marker `activity`; injected marker failure ⇒ success + `heartbeat_warning: true`, message still delivered exactly once. |
| 16 | `test_two_process_external_read_exactly_once` | `multiprocessing`: both present the same token concurrently ⇒ each message returned by exactly one reader (cross-process lock). |
| 17 | `test_no_ambient_reads` | `_active_session_id`, `_recover_session_id`, `_candidate_sessions`, `_persist_session_binding`, `_annotate` monkeypatched to raise ⇒ join/send/read/leave all succeed; `IDENTITY`/`_session_id` untouched; no binding file. |
| 18 | `test_revocation_races` | barrier races in **all four** combinations: {kill, leave} × {`external_send`, `external_read`} ⇒ outcome is either clean pre-revocation success or `membership_revoked`; never an append after record removal (inbox absent ⇒ no file recreated); racing leave never double-writes the `left` marker. |
| 19 | `test_leave_idempotent_and_lead_send_refused` | leave ⇒ `left` + `waiting/left` marker; second leave ⇒ `already_left`, marker mtime unchanged; `send_message(to=member)` ⇒ `member_left`. |
| 20 | `test_guaranteed_guards_both_paths` | `follow_up_agent(member, key)` ⇒ `external_agent_pull_only`, deliveries.json byte-compare, no lease/claim/backend-registry lookup (registry.get monkeypatched to raise); stale pending row whose target name is now an external member ⇒ `deliver_pending` refuses pre-claim AND its result contains the per-row `{idempotency_key, to, status:"refused", reason:"external_agent_pull_only"}` entry. |
| 20b | `test_prepare_race_row_settlement` | swap target to external between `_open_delivery_record` and `_prepare` ⇒ refusal without `registry.get("external")`; the row this call created is **discarded** (deliveries.json byte-identical to pre-call), no `active_holder` survives, and the same idempotency key is immediately reusable; for a **pre-existing** row the audit row survives unmutated with its claim released; injected `_discard_delivery_record` persistence failure ⇒ `record_discarded: false` reported, no false clean-rollback. |

Safety, status, restart:

| # | Test | Asserts |
|---|---|---|
| 21 | `test_kill_external_never_probes_pid` | `owns_process` **and** `kill_process` raise if called; `killed_process: False`, reason `external_agent_deregistered`; record/inbox/cursor/marker cleaned. |
| 22 | `test_external_only_session_discoverable_not_autoadopted` | session whose only live record is a `running` external member: `_candidate_sessions()` **includes** it and `session_info.recoverable_sessions` lists it; `_recover_session_id` single-candidate auto-adopt **skips** it (returns unresolved + nudge); explicit `resume_session(sid)` succeeds and the lead can then `send_message` to / `kill_agent` the member. With a `running` spawned worker added, auto-adopt behaves as today. |
| 23 | `test_no_credential_in_any_tool_result` | after join: full `list_agents`, compact `list_agents`, `check_agent(full=True)`, `agent_status` results serialized to JSON contain neither the digest, the ticket token, nor the derived secret. |
| 24 | `test_agent_status_backend_and_binding_na` | `agent_status` exact row schema + `backend`; external binding reported `not_applicable` in check/full-list/status-fallback (all three call sites). |
| 25 | `test_restart_with_real_lead_binding` | fixture: real `team-lead` binding + same-cwd candidate session exists; fresh subprocess with clean env runs `external_read(token)` ⇒ succeeds; recovery helpers fail-fast-patched out in-process variant; separately assert an *ambient* call in that subprocess CAN adopt (documents the §6 residual honestly). |
| 26 | `test_external_only_mode` | `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` ⇒ registered tool names == {join_team, external_send, external_read, leave_team, list_backends} **and their client-visible descriptions are intact** (disk-note ordering preserved); without env ⇒ full set with unchanged descriptions. |
| 27 | `test_ttl_and_retention_parsers` | `0`, `-1`, `nan`, `inf`, `"x"` ⇒ defaults; valid override honored. |
| 28 | `test_large_inbox_read_contention_bounded` | 10k-line inbox fixture, **barrier-based (review-4 NB6)**: prove the concurrent `agent_status` blocks while `external_read` holds the lock, then completes promptly after release; Windows-marked variant asserts completion below the 30 s lock deadline. |
| 29 | `test_join_prompt_delimiter_injection` | `note` containing ``` and a fake `REPORTING PROTOCOL` heading stays fully inside its (extended) fence; protocol lines outside appear exactly once. |

## 6. Risks & open questions (residual)

- **Ambient toolset in the member's conversation.** Mitigated by the
  EXTERNAL_ONLY server mode (§3.8) + join-prompt discipline; not eliminated
  while Desktop lacks per-conversation MCP scoping. An ambient misuse acts as
  a root lead and may adopt a same-cwd session — pre-existing recovery
  behavior, now *documented* (v2's "unchanged by this feature" was
  imprecise; the structural half — external records keeping sessions
  adoptable — is fixed by §2.2).
- **Serialization cost:** external send/read hold the session's registry file
  lock; low-volume by design; documented.
- **Bearer token in transcript** — same-user threat model, accepted (no
  member secret in `agents.json` or any tool result; the ticket store retains
  the derivation inputs — §0).
- **`agents.json` non-atomic save** — pre-existing ([:518]); unchanged scope.
- **Stale `running` external records** — never trigger silent fallback
  auto-adopt (§2.2), though the session stays discoverable and explicitly
  resumable by design; visible via `heartbeat_age_s`; lead retires with
  `kill_agent`. Optional auto-`left` threshold deferred.
- **Desktop process model** — informational only (affects `pid` field
  usefulness); observe and record in `implementation.md`.
- **Mid-level parent gone before consumption** — accepted, docstring'd
  (review-1 NB3 disposition stands).

## 7. Review-1 disposition (unchanged from v2)

Rows 1–9 + NB1–NB5 as in v2 §7; superseded only where §8 tightens them.

## 8. Review-2 blocker disposition (historical — where §9 tightens a row, §9 wins)

| Review-2 finding | Disposition in v3 |
|---|---|
| B1 credential exposure via full list | **Fixed** — digest-only at rest, deterministic re-derivation for replay, `_public_agent_record` sanitizer on all four surfaces, leak test 23 (§0, §3.7). |
| B2 reconciliation not idempotent; marker window | **Fixed** — record-by-`join_ticket_id` lookup first, all four (status, record) states dispatched, marker ensure/repair on every replay, `registry_corrupt` for duplicates, marker-failure retriability defined (§3.3, tests 8–11). |
| B3 lock not held through side effects; leave contradiction | **Fixed** — `_member_operation` holds the registry lock through every side effect; committed lock order; `allow_left` mode; race tests 18–19 (§2.4). |
| B4 cross-process cursor race | **Fixed** — external reads run under the cross-process registry file lock end-to-end; two-process exactly-once test 16 (§2.4). |
| B5 no single delivery boundary | **Fixed** — guards at both early call sites (pre-`_open_delivery_record`, pre-`_claim_delivery_record`) + authoritative `_prepare` recheck before backend/lease; byte-compare tests 20 (§3.6, §1.7). |
| B6 prompt-only containment; inaccurate "unchanged" claim | **Mitigated + made honest** — EXTERNAL_ONLY server mode as a shippable technical control; §6 wording corrected; external records structurally excluded from recovery candidacy; residual documented as such, wrong-tool behavior asserted in test 25 (§3.8, §2.2). Full elimination is a Desktop-client capability gap, stated openly. |
| B7 matrix gaps | **Fixed** — tests 1–27 cover every enumerated gap: redaction, fail-fast ambient patches, real-binding restart, two-process read, cursor matrix, both crash windows + marker failure, both reservation orders + concurrent, revocation races both directions, token grammar matrix, join-prompt exactness, TTL parsers, EXTERNAL_ONLY registration. |
| NB1 token grammar | **Adopted** (§0, test 12). |
| NB2 "one-time" wording | **Adopted** — replayable-recovery-credential docstring (§2.1). |
| NB3 TTL parser | **Adopted** — strict finite-positive (§1.13, test 27). |
| NB4 centralize binding | **Adopted** — shared wrapper returns `not_applicable` (§3.7, test 24). |
| NB5 heartbeat failure semantics | **Adopted** — success + `heartbeat_warning`; join marker contractual + retriable (§2.3, tests 10/15). |
| NB6 reservation expiry + both orders | **Adopted** (§2.1, §3.2, tests 2–3). |
| NB7 join-prompt exactness/injection | **Adopted** — deterministic fenced layout, literal-value tests (§3.1, test 1). |

## 9. Review-3 blocker disposition

| Review-3 finding | Disposition in v4 |
|---|---|
| B1 state machine: expired crash-A unrecoverable; replay-after-leave resurrects marker | **Fixed** — expiry check moved after record reconciliation (expiry gates only *new* joins); dispatch extended to (ticket status, record presence, record.status); `left` ⇒ `membership_revoked/detail:"left"`, marker never rewritten; conservative marker repair (only missing/unparseable markers are written); immutable-field mismatch and unknown ticket status ⇒ `registry_corrupt` (§3.3, tests 8/9/9b). |
| B2 durable-row settlement for the authoritative race | **Fixed** — `external_agent_pull_only` added to `_C2_REFUSAL_REASONS` (row created by this call ⇒ discarded, claim released, key reusable); pre-existing rows: claim released, audit row unmutated, stable refusal; `deliver_pending` result gains per-row refused entries (§3.6.4, tests 20/20b). |
| B3a EXTERNAL_ONLY is not isolation in a dual-entry setup | **Fixed by honest re-scoping** — supported deployment = separate Desktop profile whose config contains only the external entry (actual client-surface isolation); dual-entry named a documented degraded mode, never called containment; profile capability verified in smoke test; release-note gate if the client cannot isolate (§3.8). |
| B3b external-only sessions invisible to legitimate recovery | **Fixed** — discoverability/auto-adopt split: `_non_terminal_agents` unchanged (sessions stay in `recoverable_sessions` and `resume_session` works); only the silent single-candidate auto-adopt additionally requires a non-external live record (§2.2, test 22). |
| NB1 at-rest wording | **Adopted** — claim narrowed to "no member secret in agents.json + no tool-level disclosure"; ticket-store reconstruction stated openly (§0). |
| NB2 lock-stall documentation | **Adopted** — platform-specific behavior documented (30 s Windows timeout vs blocking POSIX flock); contention bounded by test 28 (§2.4). |
| NB3 EXTERNAL_ONLY registration mechanics | **Adopted** — one gate wrapping ALL registrations; decoration-time description ordering preserved; names + descriptions asserted per mode (§3.8, test 26). |
| NB4 cursor contract completeness | **Adopted** — test 14 extended (invalid since_seq, negative limit, full=True, multi-sender ordering, unread_count/has_more, cursor-file byte stability). |
| NB5 leave-vs-read race | **Adopted** — test 18 covers all four {kill,leave}×{send,read} combinations. |
| NB6 fence injection claim | **Adopted** — extended-fence technique; claim narrowed; adversarial test 29 (§3.1). |
| NB7 full-list docstring | **Adopted** — docstring updated to "sanitized registry fields" alongside the sanitizer (§3.7). |
