# Plan review 1 — external-member-wake (independent adversarial review)

Reviewer: Claude (Opus family), 2026-07-24. Target:
[`plan.md`](plan.md) against `design.md`, `lead_wake.py`, `hooks.py`, and
`server_simple.py` as of branch `feature/external-agent-join` (HEAD `71a11ff`).

## Claim verification (plan statements checked against real code)

| Plan claim | Verdict |
|---|---|
| `_scan_senders`, `_is_armed`, `_command_matches_session`, `_read_guard`/`_write_guard`/`_cursor_advanced`/`_apply_guard`, `WakeDecision`, `_read_payload` take explicit `(session_dir, identity)`/payload args and are token-free | **True** (`lead_wake.py:157-333, 406-418`) — reusable by import as claimed. |
| `_log_line` reusable by import | **False** — it hardcodes the `"win-agent-teams/lead-wake "` prefix (`lead_wake.py:421-425`). See finding 3. |
| `_kill_switch_on` "semantics" reusable | **Misleading as listed** — it hardcodes `WIN_AGENT_TEAMS_LEAD_WAKE` (`lead_wake.py:36, 59-61`); cannot implement M0's member-env-with-fallback by import. See finding 3. |
| External record carries `name` / `backend=="external"` / `spawned_by_source=="join_ticket"` / status in `{running,left}` | **True** — record written at `server_simple.py:2844-2858` (`SPAWNED_BY_SOURCE_JOIN = "join_ticket"`, `agent_output.py:57`); `leave_team` mutates `status` → `"left"` (`server_simple.py:3529`). Note the only terminal status in `_TERMINAL_STATUSES` is `"killed"` (`server_simple.py:194`); `"revoked"` is not a stored status — prose only. |
| `_watch_argv` accepts `reader`; `_watch_command_bash` does not forward it | **True** — `server_simple.py:1508-1519` vs `1522-1524`. |
| `_install_wake_hook` keys the group on the `_WAKE_MODULE` substring, so a `member_wake` group coexists | **True** — `_group_has_wake_token` substring-matches `hooks._WAKE_MODULE == "claude_teams.lead_wake"` (`server_simple.py:5866-5877`, `hooks.py:30`). `"claude_teams.member_wake"` does not contain that substring and vice versa, so mutual removal is safe. |
| Never-unstoppable contract (exit 0 always, print-only block, guard cap fail-open) | **True** in `lead_wake.main`/`_apply_guard` (`lead_wake.py:295-333, 428-448`); reusable if `main` is mirrored faithfully. |
| `external_read` drains the same cursor `_scan_senders` reads | **Verified consistent** — both use `inbox-<name>.pos.json` (`server_simple.py:394, 3206, 3266`; `lead_wake.py:165-166`), so M3's unread signal really clears when the model obeys the block reason. Not stated in the plan; worth a test cell (finding 6d). |

## Findings

### 1. MAJOR — "Stale baked hook after team ends is harmless because M2 fail-opens" is false for the dominant staleness case

M2 fail-opens on: record gone, status `left`/terminal, registry unreadable, dir
missing. But when the joined lead simply **stops being used** (Desktop closed,
session abandoned), nothing ever mutates the record: it stays `status:
"running"` in an `agents.json` that remains on disk indefinitely. Every future
turn-end in the member session then lands in M5 (no watcher armed) and blocks.
The guard only caps *consecutive* unproductive blocks and **resets to 0 at the
D6 fail-open** (`lead_wake.py:325`), so the member gets ~cap (default 3) blocks
at **every** turn end, forever, for a team that is dead. Lead-wake does not have
this failure because D2 self-disables when children go terminal; M2 has no
analogous liveness signal — external "leaving" requires an explicit
`leave_team` or lead-side kill that will never happen for an abandoned session.

**Fix:** add a staleness discriminator to M2 (e.g. fail-open when the joined
session shows no activity for N hours — lead inbox/marker mtime — or when the
lead's own state marker is stale), or at minimum: (a) delete the "harmless"
claim from §2.3 and the risk table, (b) make the M5 arm-reason include the
self-service escape hatch ("if this team is finished, run
`install_member_wake(..., remove=True)` or set the kill switch"), and (c) add a
test cell for the abandoned-team scenario documenting the accepted behavior.

### 2. MAJOR — default `scope="project"` resolves against the MCP **server process** cwd, which is not the member conversation's workspace

`_lead_wake_settings_path("project")` is `Path.cwd()/.claude/settings.json`
(`server_simple.py:5906-5909`) evaluated in the MCP server process. For the
feature's primary client — a manually-started Claude Desktop session — the
server cwd is wherever Desktop launched the server (app dir, `/`, home), not a
project the member harness reads settings from. Two failure shapes: (a)
**silent no-op** — the hook lands in a `.claude/settings.json` no harness ever
loads, the tool returns `installed`, and the user believes hands-free pickup is
armed; (b) **pollution** — it writes into an unrelated repo that happens to be
the server cwd. "Mirrors `install_lead_wake`" is not a justification:
`install_lead_wake` targets an interactive `claude` **CLI** lead whose cwd is
the repo; the member case is exactly the one where that assumption breaks (the
design's own Option-A con already says so).

Additionally, the plan never states the supported member harness. Stop hooks
are a Claude Code harness feature; if the member session is the Claude Desktop
chat app without hook support, the entire feature is inert. The plan (and the
tool docstring, which is the contract surface per repo rules) must say which
harnesses the install is valid for.

**Fix:** default `install_member_wake` to `scope="user"` (or require an
explicit scope with no default), return the resolved path prominently, and
document in the docstring that project scope only works when the member
conversation's workspace cwd equals the MCP server cwd. State the harness
requirement explicitly.

### 3. MAJOR — the §2.1 reuse list is partially wrong: `_log_line`, `_kill_switch_on`, and the guard cap env are lead-hardcoded

- `_log_line` renders the fixed prefix `"win-agent-teams/lead-wake"`
  (`lead_wake.py:423`). Importing it mislabels every member decision in stderr
  observability — the one signal used to debug wake behavior live.
- `_kill_switch_on` reads only `WIN_AGENT_TEAMS_LEAD_WAKE`
  (`lead_wake.py:36, 59-61`); M0's "member env, fallback to lead env" cannot be
  an import — it is new code and should be listed as such.
- `_apply_guard` internally calls `_max_noprogress`, which reads
  `WIN_AGENT_TEAMS_LEAD_WAKE_MAX_NOPROGRESS` (`lead_wake.py:37, 64-73`). Reusing
  `_apply_guard` verbatim means the member guard cap is silently governed by the
  **lead** env var. Possibly acceptable — but the plan neither says so nor tests
  it.

**Fix:** amend §2.1 to move these three out of the "reuse by import" list;
either parameterize (prefix arg for `_log_line`, env-name args for the
switch/cap) or write small member-local variants, and state which env governs
the member cap. Add a test pinning the chosen cap-env behavior.

### 4. MINOR — "guard file never collides" is overstated: the joined dir holds guards for every wake-writing identity, keyed only by name

All of a team's agents share one session dir, so the joined dir already
contains `wake-progress-<identity>.json` for the joined lead and any nested
leads. `wake-progress-<member>.json` avoids collision only if the member name
can never equal a wake-writing identity there. Ticket names are deduped against
registry records and live tickets (`_unique_reserved_name`,
`server_simple.py:724-734`), which covers spawned agents — but the **root lead
identity `team-lead` has no registry record**, so `create_join_ticket(name=
"team-lead")` reserves it and the member's guard file becomes the same
`wake-progress-team-lead.json` the joined lead's own lead-wake writes,
cross-contaminating cursor snapshots and noprogress counters (both directions
mostly fail toward extra fail-opens, but the "never collides" invariant in
§2.1 and risk-table reasoning is false).

**Fix (cheap):** name the member guard `wake-progress-member-<member>.json`
(distinct prefix, still identity-keyed), or reserve `team-lead` in
`create_join_ticket`. Adjust the plan text either way.

### 5. MINOR — M0 kill-switch truth table is ambiguous and can strand a user

"Member-specific env (default on), falling back to the shared one so one
switch can disable both" admits two implementations: (a) member env, when
*set*, wins (LEAD=0 + MEMBER=1 → member on); (b) either being 0 disables
(member can never stay on while lead is off). The plan's "one switch disables
both" phrasing implies (b), which removes the ability to disable lead-wake
alone. Also note the env must be visible to the **member harness's** hook
process — env set on the MCP server does not reach the Stop hook; the plan
should say where to set it.

**Fix:** write the four-cell truth table into the plan (recommend (a):
`MEMBER` set → authoritative; unset → inherit `LEAD`), and add both
disagreement cells to the test matrix (§5 currently has only "kill switch off").

### 6. MINOR — §5 test-matrix gaps (adversarial cells missing)

a. **Registry unreadable** — M2 prose covers "registry can't be read" but no
   matrix cell exercises corrupt/unparseable `agents.json` → fail-open (cell 6
   only covers the missing dir).
b. **Name-shadowing record** — a record with `name == member` but
   `backend != "external"` or `spawned_by_source != "join_ticket"` must
   fail-open (the M2 predicate is conjunctive; test that it is).
c. **Armed-mismatch** — a *running* watcher whose command references the
   member's **own** session dir (or another session) must NOT satisfy M4 for
   the joined dir. `_command_matches_session` also matches on the bare
   basename/session-id substring (`lead_wake.py:189-193`), so include a
   near-miss command in the cell.
d. **Cursor consistency** — after a simulated `external_read` drain
   (cursor file advanced), the hook allows; pins the
   `inbox-<member>.pos.json` coupling verified above so a future cursor-file
   rename cannot silently break M3.
e. **Payload garbage** — empty/non-dict stdin payload → still exits 0
   (never-unstoppable holds for member `main` too, not only `evaluate_member`).
f. **Abandoned team** (from finding 1) — record `running`, dir present, no
   watcher: document the block-per-turn behavior or the staleness fail-open.

### 7. NIT — `"revoked"` in M2 prose is not a real stored status

Statuses observable on an external record are `running`, `left`, and (per
`_TERMINAL_STATUSES`) `killed`; lead-side kill deregistration otherwise removes
the record ("no record" cell). Keep the M2 predicate written in terms of the
real values so the implementer doesn't invent a `revoked` branch.

### 8. NIT — plan should pin where `_member_read_reason`'s token placeholder comes from

The reason text renders `external_read(member_token=…)` with a literal
ellipsis placeholder — correct (the hook never holds the token). Say so
explicitly in §2.2 so a well-meaning implementer never "improves" it by
threading the real token through, and extend matrix cell 9 to assert the
reason string contains no `wam1:` prefix.

## Points verified sound (no change requested)

- M1 "arg only, never `_active_session_id`" correctly forks
  `_resolve_session_dir`'s discovery-first behavior, which would indeed find
  the wrong (own) dir.
- Coexistence design (§2.4) is real: substring keying confirmed at
  `server_simple.py:5866-5877`; distinct module strings cannot cross-match.
- Extending `_watch_command_bash` with keyword-only `reader` is the right
  call-site-sharing fix; existing callers (`:3053`, `:5829`) are unaffected.
- Install validation reusing `_validate_join_session_id` (uuid + base-dir
  containment + `agents.json` exists, `server_simple.py:782-805`) is accurate.
- No-token-at-rest holds by construction: installer inputs are
  `joined_session_id` + `member_name`; the baked argv carries dir + name only.

## Verdict

**APPROVED-WITH-CHANGES** — the module split, reuse boundary, M-gate shape,
and never-unstoppable carry-over are sound, but findings 1-3 must be resolved
in the plan text (and their test cells added) before implementation: the
staleness claim is false as written, the default install scope likely no-ops
for the primary Desktop-member use case, and three "reused" helpers are not
actually reusable as listed.

Score: **72/100**.
