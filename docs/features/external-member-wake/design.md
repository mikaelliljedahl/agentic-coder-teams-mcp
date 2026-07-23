# External-member wake (downstream, lead → external) — follow-up design sketch

Status: **design sketch** (not planned/implemented). Follow-up to the shipped
`external-agent-join` feature (PR #38). Companion to the upstream wake that
already works via `install_lead_wake`.

## 1. Problem

A manually-started interactive session (Claude Desktop) can now `join_team` as
an external member and exchange messages with the lead. Two directions:

| Direction | Transport | Hands-free today? |
|---|---|---|
| **external → lead** (member replies) | `external_send` → `inbox-<lead>` | **Yes.** `install_lead_wake` on the lead blocks its turn-end until the reply is drained. Done in this session. |
| **lead → external** (lead assigns/answers) | `send_message` → `delivery:"inbox"` → `inbox-<member>` | **No.** The member must *manually* poll `external_read`. This sketch closes that gap. |

The asymmetry is fundamental, not a bug: PR #36 guaranteed delivery wakes a
spawned child by **resuming its CLI process**. A Desktop conversation cannot be
resumed or injected into from outside, so lead → external is **pull-only by
design** (an explicit PRD non-goal). "Hands-free" here therefore means the same
thing lead-wake means: a deterministic **Stop hook** in the member's own session
that refuses to let its turn end while unread work sits in the member inbox, and
otherwise verifies a background watcher is armed to carry the idle wait. No push;
the member's own harness turn-end is still the trigger.

## 2. Why the existing hook doesn't cover it

`lead_wake.evaluate` derives three things from the **process identity**, and the
external session keeps `IDENTITY = team-lead` **by design** (the token-carried
model never rebinds process identity — that safety property is the whole reason
the feature shipped). So if `install_lead_wake` were run in the Desktop session:

1. `_resolve_identity(reader)` → `team-lead` — but the member's inbox is
   `inbox-<member>.jsonl`, never `inbox-team-lead`.
2. `_resolve_session_dir(...)` → the Desktop session's **own** session dir — but
   the member inbox lives in the **joined lead's** session dir.
3. D2 gate `_live_subagent_names(dir, "team-lead")` → the Desktop session leads
   **no children**, so it hits D2 fast-allow every turn and never even looks at
   an inbox.
4. The D3 reason tells the model to call `read_messages`; a member must call
   `external_read(member_token=...)` (ambient team tools are off-limits from an
   external-member conversation).

All four are wrong for a member. The reader and the session dir are the wrong
values, the *liveness gate* asks the wrong question, and the *instruction* names
the wrong tool.

## 3. Building blocks already in place (no change needed)

- **Watch CLI is already `(session_dir, reader)`-parameterized.** `join_team`
  returns `watch_argv = _watch_argv(session_dir, reader=name)` and the join
  prompt step 4 emits `watch <joined-session> --reader <member>`. The background
  watcher side is solved.
- **The scan/armed/guard machinery is already session-dir-parameterized and
  token-free:** `_scan_senders`, `_command_matches_session` / `_is_armed`,
  `_read_guard`/`_write_guard`/`_cursor_advanced`, and the never-unstoppable
  contract (exit 0, print-only-block, no-progress cap) are all reusable verbatim
  against the joined dir + `inbox-<member>`.
- **`join_team` already returns everything the installer needs:** `session_id`
  (the joined lead's session), `name` (the member), and `watch_argv`. No new
  data has to be computed at join time.

The gap is *only* in the Stop-hook decision path — the reader/session-dir/gate
selection — not in the watcher, the delivery path, or the messaging store.

## 4. Hard constraint: no token at rest

`external-agent-join`'s core invariant is **digest-only at rest** — the
`member_token` secret never touches disk. The member-wake hook **must not** bake
the token into `settings.json`. It doesn't need to: detecting unread is a
read-only scan of `inbox-<member>.jsonl` + its cursor (no auth), and the hook
only *instructs* the model to call `external_read` — the model already holds the
token in its conversation transcript. So the hook parameters are the member
**name** and the **joined session dir** only; both are non-secret.

## 5. Design options

### Option A — baked per-member hook via a new `install_member_wake` tool
The Desktop session, right after `join_team`, calls a new MCP tool
`install_member_wake(joined_session_id, member_name)` that bakes
`python -m claude_teams.member_wake --joined-session-dir <lead dir> --member <name>`
into a settings file.

- Pro: smallest change; mirrors `install_lead_wake` one-to-one.
- Con: a baked hook is member-and-session-specific. Project scope writes into the
  Desktop workspace's cwd (often not a repo we want to touch); user scope
  (`~/.claude/settings.json`) fires on every Desktop conversation but can only
  carry **one** baked member. A Desktop user who joins two teams, or rejoins
  after the joined session ends, has a stale baked hook. Re-install per join is
  required and easy to forget.

### Option B — self-locating member-wake (recommended end state)
A single hook, installed **once at user scope**, that on each turn-end discovers
*all* active memberships this Desktop session holds and watches each
`inbox-<member>` whose membership is still live.

Requires one new **token-free** artifact: `join_team` drops a membership pointer
in the **member session's own** dir (e.g. `memberships.jsonl`:
`{joined_session_id, member_name, ts}`), and `leave_team` marks it left. Nothing
ties a Desktop conversation to its joined memberships on disk today (membership
records live only in the *joined* lead's `agents.json`, keyed by member name),
so this pointer is the missing link — and it carries no secret.

`member_wake.evaluate` then:
1. resolves the member session's own dir (same discovery as lead-wake — this is
   where the pointer lives);
2. reads active memberships from the pointer;
3. for each, rechecks liveness against the **joined** `agents.json` (membership
   still `running`, not `left`/revoked) — the authoritative gate replacing D2;
4. scans the joined `inbox-<member>`; blocks (external_read reason) if any
   unread, else verifies an armed watcher for the joined dir, else blocks
   (arm reason). Progress guard + fail-open identical to lead-wake.

- Pro: install-once, multi-membership, survives rejoins, never writes into an
  unknown cwd, self-disables (no active membership ⇒ D2-equivalent allow) on
  every non-member Desktop conversation, and keeps the token off disk.
- Con: needs the pointer-file write in `join_team`/`leave_team` and a small
  discovery/liveness path. Slightly more than Option A.

### Recommendation
Ship **Option B**. The pointer file is cheap and token-free, and it is the only
option that behaves correctly for a Desktop client that joins more than one team
or rejoins over time — which is exactly the interactive-session use case this
whole feature exists for. Option A's baked-single-member hook would be a
foot-gun in precisely that workflow. If a minimal first cut is wanted, land
Option B's `member_wake` module first driven by an explicit
`install_member_wake(joined_session_id, member_name)` (Option A ergonomics) and
add the pointer-file auto-discovery in a second step — same module, same gate.

## 6. Fail-open / never-unstoppable (unchanged contract)

Carry `lead_wake`'s guarantees verbatim: the hook never emits `continue:false`,
never exits non-zero, prints a block only as `{"decision":"block","reason":...}`,
and the no-progress guard caps consecutive unproductive blocks before failing
open. Additional member-specific fail-open cases → **allow**:

- membership pointer absent / no active membership (the "self-disable on a
  non-member Desktop conversation" case — the D2 analogue);
- the joined session dir no longer exists (lead ended) or its `agents.json`
  marks the membership `left`/revoked — never trap a Desktop conversation in a
  block loop for a team that's gone.

## 7. Test sketch (mirrors `tests/` lead-wake coverage)

- member with unread in joined `inbox-<member>` → **block**, reason names
  `external_read` (not `read_messages`);
- member, no unread, armed watcher matches the **joined** dir → **allow**;
- member, no unread, not armed → **block**, arm reason renders the joined-dir
  watch command;
- membership revoked / `left` in joined `agents.json` → **fail-open allow**;
- no membership pointer (ordinary Desktop conversation) → **allow**;
- joined session dir deleted → **fail-open allow**;
- progress-guard cap reached → **fail-open allow**; kill switch off → **allow**;
- **credential check:** installed settings + any wake artifact contain no
  `member_token` / secret (only member name + joined dir).

## 8. Non-goals

- Pushing into or resuming a Desktop conversation from outside (impossible;
  that's why this is pull-only + turn-end-gated, not delivery-resume like PR#36).
- Persisting the member token anywhere on disk.
- Changing the upstream lead-wake behavior or the shipped `external-agent-join`
  wire format.
