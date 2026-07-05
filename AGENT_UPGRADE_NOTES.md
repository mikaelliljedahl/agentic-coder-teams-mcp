# Orchestration instruction changes since the pre-PR#6 version

For anyone (human or agent) who **writes the instructions that drive
`win-agent-teams` agents** — orchestration skills, slash-commands, or
coordinator prompts (e.g. a `/codex-agent-orchestration`-style skill). If your
instructions were written against the version **before PR #6**, update them as
below. This is only about *what to tell a coordinator to do* — internal fixes
that are transparent to callers are intentionally left out.

If you change one thing: **replace "poll `check_agent` in a loop" with the
watch recipe**, and **add a restart-recovery step** so an empty `list_agents`
after a restart isn't mistaken for "no agents".

---

## Instruction-rewrite cheatsheet

| If your instructions currently say… | Change it to… |
|---|---|
| "Poll `check_agent` every few seconds until done" | "Watch the state-marker path, then read once on change" (§1) |
| "Read `check_agent`/`list_agents` for the full transcript" | "These are compact by default; pass `full=True` only when you need bodies" (§3) |
| "Re-read all messages each time" | "`read_messages` is delta; advance via the returned `seq`/`cursors`" (§3) |
| "If `list_agents` is empty, the team is gone" | "After a restart, call `session_info()` → `resume_session('<id>')`" (§2) |
| "`kill_agent` to pause/park an agent you'll resume" | "`kill_agent` is terminal removal; use `follow_up_agent` to resume a dead/idle agent" (§4) |
| "Kill the MCP server to stop the agents" | "Agents survive server death; stop them with `kill_agent`" (§5) |

---

## 1. Tell coordinators to *watch*, not tight-poll  *(PR #6)*

Each spawned agent writes a `state-<agent>.json` marker under the session dir on
every lifecycle transition (`{ "state": "running"|"waiting", "event", "ts" }`).
`spawn_agent` returns `state_marker_path` and `session_dir`;
`agent_watch_paths()` rediscovers them.

Put this in the "wait for the agent" step of your instructions:

- **Claude Code coordinator** → run the watch as a **background** command; its
  completion wakes the coordinator, which then calls `agent_status` /
  `check_agent` for the delta.
- **Codex coordinator** (no idle-wake) → run a **bounded foreground** watch,
  looped, within the turn:
  ```
  win-agent-teams watch <session-dir> --timeout 60 --pattern "state-*.json"
  ```
  On exit 0, read the marker JSON from disk and/or call `agent_status`.

Explicitly forbid a tight `check_agent` spin loop — it wastes tokens; the marker
watch is cheap and server-independent.

## 2. Add a restart-recovery step  *(PR #10)* — the biggest instruction gap

After the coordinator's own process restarts (crash, `/compact`, host reboot),
`list_agents` can return **empty** while agents still exist. Your instructions
must not treat that as "no agents". Add:

1. Call **`session_info()`** → `{session_id, identity, cwd, agent_count,
   lead_token, recoverable_sessions}`.
2. If `recoverable_sessions` is non-empty, adopt one with
   **`resume_session('<session_id>')`**; its agents reappear and become
   resumable.
3. Dict-returning tool results also carry a `recoverable_sessions` /
   `recovery_hint` nudge (or a one-shot `adopted_session`) — instruct the
   coordinator to act on it.

Notes to bake in: a single-lead workspace auto-adopts the newest session; if
**several coordinators** ran in the same folder, auto-adopt is off and the
coordinator must pick via `resume_session`. Tell long-running orchestrations to
**record `lead_token`** (their stable recovery handle). Session dirs are
auto-pruned after 30 days (`WIN_AGENT_TEAMS_RETENTION_DAYS`); active/live
sessions are never removed.

## 3. Expect delta + compact responses  *(PR #6)*

- `read_messages` drains and advances a per-sender cursor and returns
  `{messages, cursors, seq, unread_count, has_more}`. Instruct: advance using
  the returned `seq`/`cursors`, don't assume you re-receive the whole history.
  `limit` defaults to 50; `limit=0` is a no-body watermark poll; `full=True`
  returns everything.
- `check_agent` / `list_agents` / `agent_status` are **compact by default** (no
  transcript bodies) and expose a coarse `state`
  (`running`/`waiting`/`idle`/`dead`) plus `heartbeat_age_s` and `stalled`.
  Instruct: use `stalled: true` to detect "alive but hung"; pass `full=True`
  only when you actually need `last_message` / `backend_session_id` (e.g. before
  a follow-up).

## 4. Fix the `kill` vs `follow_up` guidance  *(PR #10)* — semantics changed

- `kill_agent` is now **terminal removal**: it deletes the record (gone from
  `list_agents`; `follow_up_agent` → `agent_not_found`) and cleans the agent's
  marker/inbox/cursor. It is **not** a pause/park.
- A **naturally dead** agent is **kept** and stays **resumable via
  `follow_up_agent`** until you explicitly kill it.

So instruct: use `follow_up_agent(name, prompt)` to continue a dead/idle agent
(re-review, "fix and continue"); use `kill_agent(name)` only to retire it for
good or to free a name for a clean respawn. (Kills are fail-closed on PID reuse,
so a stale/reused PID is never signalled — no caller action needed.)

## 5. Don't tell coordinators to kill the server to stop agents  *(PR #7)*

On Windows, spawned agents break away from the MCP server's job object and keep
running when the (idle) server is killed. The server auto-restarts on the next
tool call and state markers persist on disk. Instruction impact: to stop an
agent, use `kill_agent` — never "kill the server". (Host-level opt-out:
`WIN_AGENT_TEAMS_NO_BREAKAWAY=1`.)

---

## Drop-in replacement for the "wait / status / continue" block

If your skill/command has a polling block, replace it with:

```
spawn_agent(...)                      # keep state_marker_path + session_dir
# ── wait: watch, don't poll ──────────────────────────
#   Claude coordinator:  background   win-agent-teams watch <session_dir>
#   Codex coordinator:   foreground   win-agent-teams watch <session_dir> --timeout 60   (looped)
# ── on change ────────────────────────────────────────
agent_status([...])                   # coarse state + heartbeat + stalled (cheap; no bodies)
check_agent(name, full=True)          # only when you need last_message / backend_session_id
read_messages(from_agent=name)        # delta; advance via returned seq/cursors
# ── continue / retire ────────────────────────────────
follow_up_agent(name, prompt)         # resume a dead/idle agent (NOT kill_agent)
kill_agent(name)                      # terminal: removes the record for good
# ── after a coordinator restart, if list_agents is empty ─
session_info() → resume_session('<session_id>')
```
