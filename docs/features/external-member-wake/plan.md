# External-member wake (downstream, lead → external) — implementation plan

Branch: `feature/external-agent-join` (same branch as the shipped join feature;
no merge until this is done). Companion to `install_lead_wake` (upstream).
Design basis: [`design.md`](design.md).

## 0. Scope decision

Ship the **downstream hands-free pickup** as *Option A ergonomics on Option B's
module* (design.md §5 recommendation): a new `member_wake` Stop-hook module + an
`install_member_wake` MCP tool that bakes **explicit** `--joined-session-dir` and
`--member` into the settings file. The hook watches `inbox-<member>` in the
**joined lead's** session dir and refuses to let the member's turn end while
unread work waits, otherwise verifies a background watcher is armed.

**Out of scope (documented follow-up):** the fully self-locating, multi-membership
variant (design.md Option B) — `join_team`/`leave_team` dropping a token-free
membership-pointer file that a single user-scope hook enumerates. The module and
gate written here are the substrate that follow-up extends; no rework needed.

## 1. Current behavior (the gap)

`lead_wake.evaluate` derives three things from **process state**, all wrong for a
Desktop member (which keeps `IDENTITY = team-lead` by design):

1. `_resolve_identity` → `team-lead` (member inbox is `inbox-<member>`);
2. `_resolve_session_dir` → the Desktop session's **own** dir (member inbox lives
   in the **joined** dir);
3. D2 gate `_live_subagent_names(dir, "team-lead")` → a member leads no children
   → D2 fast-allow every turn, so it never scans any inbox.
4. D3 reason names `read_messages`; a member must call `external_read`.

## 2. Design

### 2.1 New module `src/claude_teams/member_wake.py`

CLI (Stop hook): `python -m claude_teams.member_wake --joined-session-dir <dir>
--member <name>`. Reuses `lead_wake` helpers **by import** (they already take
explicit `(session_dir, identity)` and are token-free): `_scan_senders`,
`_is_armed`, `_command_matches_session`, `_read_guard`/`_write_guard`/
`_cursor_advanced`/`_apply_guard`, `WakeDecision`, `_read_payload`, `_log_line`,
`_kill_switch_on` semantics. No duplication of the scan/guard/never-unstoppable
machinery.

`evaluate_member(payload, *, member, joined_session_dir)` decision path:

- **M0 kill switch** — `WIN_AGENT_TEAMS_MEMBER_WAKE` governs member-wake; it is
  ON unless explicitly set to `0`. When that var is **unset**, fall back to
  `WIN_AGENT_TEAMS_LEAD_WAKE` (so a single `…LEAD_WAKE=0` disables both, but an
  explicit `…MEMBER_WAKE=1` re-enables member-wake independently). Off → allow.
  This is a **new** helper `_member_kill_switch_on()` — `lead_wake._kill_switch_on`
  reads only the lead var (review-1 Major 3), so it cannot be reused as-is.
- **M1 session** — resolve `joined_session_dir` from the **arg only** (never from
  `_active_session_id`, which would find the Desktop session's own dir). Not a
  directory → allow.
- **M2 membership gate (replaces D2)** — read the joined `agents.json`; find the
  record with `name == member` AND `backend == "external"` AND
  `spawned_by_source == "join_ticket"`. **Fail-open allow** when any of: no such
  record (not a member here — the self-disable case); `status == "left"`;
  `status in _TERMINAL_STATUSES` (`{"killed"}`); the registry can't be read; or —
  see M2b — the joined session looks **abandoned**. Live only when
  `status == "running"`. (Note: the statuses are `running` / `left` / `killed`;
  "revoked" is not a real status — review-1 minor.)
- **M2b abandoned-team liveness TTL (review-1 Major 1)** — a member record left
  `status:"running"` in an `agents.json` that outlives the lead would make M5
  block at *every* member turn end forever (M2 has no D2-style liveness analogue;
  this is exactly the live symptom seen with `visual-qa` this session). So M2
  also **fails open** when the joined session shows no lead-side activity within a
  TTL: the newest mtime across the joined `state-*.json` markers and
  `inbox-*.jsonl` is older than `WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS`
  (generous default, e.g. 6h). Any recent lead activity keeps the gate armed.
- **M3 unread** — scan `inbox-<member>` in the joined dir; if unread, **block**
  with `_member_read_reason` (names `external_read(member_token=...)`, never
  `read_messages`). Apply the progress guard.
- **M4 armed** — a running watcher whose command matches the **joined** dir →
  allow.
- **M5 not armed** — **block** with `_member_arm_reason`, which renders the
  joined-dir watch command including `--reader <member>` **and** the escape
  hatch: "if you are finished as a member, call `leave_team(member_token=…)` to
  stop these reminders." Apply the progress guard.

Never-unstoppable contract identical to `lead_wake`: exit 0 always, print block
only as `{"decision":"block","reason":...}`, no-progress guard caps consecutive
unproductive blocks then fails open. The guard reuses `lead_wake._apply_guard`,
whose cap comes from `_max_noprogress()` reading
`WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` — member-wake **deliberately shares
that cap env** (documented here + asserted by a test, per review-1 Major 3);
no member-specific cap var is added. Guard file
`wake-progress-member-<member>.json` in the joined dir — the `member-` prefix
avoids colliding with the lead's `wake-progress-<reader>.json` even when a member
is named `team-lead` (review-1 minor). Observability: member-wake needs its own
log line (`lead_wake._log_line` hardcodes the `lead-wake` prefix — review-1
Major 3), rendered as `win-agent-teams/member-wake …`.

### 2.2 Reason builders

- `_member_read_reason(senders)` → "Unread messages are waiting in your member
  inbox from: … Call external_read(member_token=…) to process them, and keep
  calling it while has_more is true before ending your turn."
- `_member_arm_reason(joined_session_dir, member)` → instruct running the
  reader-scoped watch as a background task. Must render `--reader <member>`; use a
  reader-aware bash rendering of `_watch_argv(joined_dir, reader=member)`.
  (`_watch_argv` already accepts `reader`; `_watch_command_bash` currently does
  not forward it — add an optional `reader` param to `_watch_command_bash`, or
  build the string locally in `member_wake`. Prefer extending
  `_watch_command_bash(session_dir, timeout=None, *, reader=None)` so both call
  sites share one renderer.)

### 2.3 New MCP tool `install_member_wake(joined_session_id, member_name, remove=False, scope="user")`

- Validates `joined_session_id` (uuid + `agents.json` exists, same as
  `_validate_join_session_id`) and non-empty `member_name`.
- Builds the baked Stop-hook command via new `hooks._member_wake_command` /
  `hooks._member_wake_hook_matcher` mirroring `_wake_command` but
  `module = claude_teams.member_wake` with `--joined-session-dir` / `--member`.
- Idempotent install/replace/remove of **only** the member-wake Stop group,
  keyed on the `member_wake` module string (distinct from the lead-wake group) so
  the two coexist and removing one preserves the other. New helper
  `_install_member_wake_hook` mirroring `_install_wake_hook`.
- **`scope="user"` (default — review-1 Major 2).** For a Desktop-launched MCP
  server, `scope="project"` resolves to `Path.cwd()/.claude/settings.json` **in
  the MCP server process**, whose cwd is arbitrary and which the member's own
  Claude harness never reads — so a project install would report `installed`
  while nothing is actually armed (or pollute an unrelated repo). User scope
  (`~/.claude/settings.json`) is read by the member's interactive session.
  `scope="project"` remains selectable for the repo-launched case.
- Returns `{action, path, member, joined_session_dir, scope}`.
- **Tool docstring** is the contract the orchestrating agent reads. It must
  state: called from the **joined member session** after `join_team`; the member
  harness **must support Claude Code `Stop` hooks** (review-1 Major 2) — this is
  useless in a client without them; it bakes **no credential** (member name +
  joined dir only); lead→member is pull-only; and that a finished member should
  call `leave_team` (or `install_member_wake(remove=True)`) to stop reminders.

### 2.4 Coexistence

`_install_wake_hook` detects its group by the `_WAKE_MODULE` substring;
`member_wake` is a different module string, so lead-wake removal never touches the
member group and vice versa. A session that is both a lead and a member keeps two
independent Stop groups.

## 3. Files affected

- **NEW** `src/claude_teams/member_wake.py`
- `src/claude_teams/hooks.py` — `_MEMBER_WAKE_MODULE`, `_member_wake_command`,
  `_member_wake_hook_matcher`.
- `src/claude_teams/server_simple.py` — `install_member_wake` tool,
  `_install_member_wake_hook`, reader-aware `_watch_command_bash`.
- **NEW** `tests/test_member_wake.py`
- Docs: flip `design.md` status, add a README / `agent-messaging-protocol.md`
  note, and record red/green + validation in the feature `implementation.md`.

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Desktop conversation trapped in a block loop | M2 fail-open membership gate + no-progress guard cap + never-nonzero-exit |
| Token at rest | hook never receives/handles `member_token`; test asserts settings + argv carry only member name + joined dir |
| Stale baked hook after team ends | M2 gate fail-opens → no-op allow |
| Multi-membership unsupported (Option A) | documented Option-B follow-up |
| No-progress guard not failing open (observed live with lead-wake this session) | keep `lead_wake`'s contract verbatim + a unit test asserting allow at the cap; the live symptom is likely harness `stop_hook_active` propagation and is **not** in scope to fix here — note it |

## 5. Test matrix (`tests/test_member_wake.py`, mirrors lead-wake)

1. unread in joined `inbox-<member>` → block; reason contains `external_read`,
   not `read_messages`.
2. no unread + armed watcher matching joined dir → allow.
3. no unread + not armed → block; arm reason renders `watch <joined dir>
   --reader <member>`.
4. membership `left` / terminal → fail-open allow.
5. no membership record → fail-open allow (self-disable).
6. joined dir missing / not a dir → fail-open allow.
7. kill switch off → allow.
8. guard: cursor advance resets counter; `stop_hook_active` increments; at cap →
   allow.
9. credential: `install_member_wake` settings + hook argv contain member +
   joined dir only, no token/secret.
10. coexistence: install member then lead (and reverse) → both groups present;
    removing one preserves the other.
11. install idempotency: re-run replaces in place, no duplicate.
12. install validation: bad `joined_session_id` → error; empty `member_name` →
    error.
13. **abandoned team (review-1 Major 1):** record `status:"running"` but joined
    markers/inboxes older than the TTL → fail-open allow; fresh activity within
    TTL → still blocks when unread/not-armed.
14. **member kill switch (review-1 Major 3):** `MEMBER_WAKE=0` → allow;
    `MEMBER_WAKE` unset + `LEAD_WAKE=0` → allow; `MEMBER_WAKE=1` +
    `LEAD_WAKE=0` → still evaluates (member overrides).
15. **shared cap env (review-1 Major 3):** guard cap honours
    `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` (asserts the deliberate reuse).
16. **member-wake log line:** stderr line prefixed `win-agent-teams/member-wake`,
    not `lead-wake`.
17. **guard-file isolation:** a member named `team-lead` writes
    `wake-progress-member-team-lead.json`, never the lead's guard file.
18. **status vocabulary:** `status:"killed"` and `status:"left"` → fail-open
    allow; only `"running"` (within TTL) blocks.
19. **armed near-miss:** a running watcher command for a *different* session dir
    does NOT count as armed → still blocks.
20. **garbage / empty Stop payload:** never raises, never exits non-zero.

## 6. TDD sequence

1. **Red:** author `test_member_wake.py` evaluate tests → fail (module absent).
2. **Green:** minimal `member_wake.evaluate_member`.
3. **Red/green:** install-tool + hooks-wiring tests.
4. **Refactor:** dedupe shared helpers via import from `lead_wake`; extend
   `_watch_command_bash` with `reader`.
5. Gates whole-repo: `uv run ruff format --check`, `uv run ruff check`,
   `uv run ty check`, `uv run pytest -q`.

## 7. Non-goals

- Pushing/resuming into a Desktop conversation (impossible — pull-only stays).
- Persisting the member token anywhere on disk.
- Full self-locating multi-membership (Option B) — follow-up.

## 8. Plan-review-1 disposition (Fable, independent; see `plan-review-1.md`)

Verdict APPROVED-WITH-CHANGES, 72/100, no blockers. Note the workflow deviation:
CLAUDE.md wants an opposite-family (GPT/Codex) plan review, but those tokens are
exhausted, so cross-model independence is provided by a **Fable** reviewer against
an **Opus**-authored plan; the authoritative gate remains the Opus
post-implementation review of the real diff.

| # | Finding | Severity | Disposition |
|---|---|---|---|
| M1 | Abandoned team (record stuck `running`) → infinite per-turn blocks; M2 lacks a D2-style liveness analogue | Major | **Accepted** → new §2.1 **M2b** liveness TTL fail-open + `leave_team` escape hatch in the M5 reason + test 13 |
| M2 | Default `scope="project"` resolves to the MCP server cwd → no-ops for Desktop; harness Stop-hook requirement unstated | Major | **Accepted** → default `scope="user"`; docstring states harness requirement (§2.3) |
| M3 | `_log_line`/`_kill_switch_on`/`_max_noprogress` are lead-hardcoded, so three "reuse by import" items are actually new code | Major | **Accepted** → member `_log_line` prefix, `_member_kill_switch_on`, documented+tested shared cap env (§2.1, tests 14–16) |
| — | Guard-file collision if member named `team-lead`; "revoked" not a real status; kill-switch truth table; 6 missing test cells | Minor | **Accepted** → `wake-progress-member-` prefix, status vocabulary fixed to `running`/`left`/`killed`, M0 truth table spelled out, tests 13–20 added |

All other review-1 claims verified against code and confirmed correct (record
fields, `leave_team`→`left`, `_watch_argv`/`_watch_command_bash` signatures,
`_WAKE_MODULE`-substring coexistence, `external_read` cursor coupling).
