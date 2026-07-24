---
name: external-member-invite
description: Attach a manually-started interactive session (e.g. a browser-capable Claude Desktop QA) to your team as a pull-only external member, and exchange messages with it. Use from the LEAD/orchestrator side when a teammate needs interactive tools a headless spawned agent lacks (a real browser, a logged-in app, a human at the keyboard); for ordinary headless workers use agent-orchestration instead. Covers create_join_ticket, handing off the join prompt, the read/send loop, pull-only delivery semantics, and teardown.
---

# Invite an external member (lead / orchestrator side)

You are the **lead**. You want a **manually-started interactive session** — most
often a Claude Desktop session that has a real browser and a human present — to
join your team so you can hand it work and read its replies. This is the mirror
of the [`external-member-join`](../external-member-join/SKILL.md) skill the other
session runs.

## When to use this vs spawning a normal worker

- **Spawn a normal headless worker** (see `agent-orchestration`) for anything a
  fresh CLI process can do on its own. That is the default.
- **Invite an external member** only when the teammate needs something a spawned
  headless agent *cannot* have: an interactive browser, an already-logged-in
  application, a specific machine/profile, or a human in the loop. Example: a
  **visual QA** that must drive the Claude Desktop browser to check a UI.

An external member is **pull-only and unconfirmed** downstream: your messages
land in its inbox and it reads them on its own schedule. There is no process
resume or guaranteed delivery toward it (a Desktop conversation cannot be
resumed from outside), so never rely on a member having "received" a message —
rely on its reply.

## Steps

### 1. Mint a join ticket

```
create_join_ticket(name="<member-name>", note="<one-line role brief>")
```

`name` must match `[A-Za-z0-9_-]{1,64}`. The result contains `session_id`,
`token`, and a paste-ready `join_prompt`. The ticket is a **replayable recovery
credential** during its retention window (default 24h) — replaying it after an
MCP restart returns the same membership, it is not strictly one-time.

### 2. Hand the join prompt to the other session

Give the returned `join_prompt` (or your own equivalent) to the interactive
session. It contains the literal `join_team(session_id=..., token=...)` call and
the member protocol. If you want a specific connection mode, tell that session
which mode to use from [`external-member-join`](../external-member-join/SKILL.md)
(watcher-only vs member-wake) — see that skill's mode table.

### 3. Make sure YOU get woken when the member replies (upstream)

The member's reply lands in **your** inbox (`inbox-<your-identity>.jsonl`). To be
woken instead of polling:

- If you are a **server-spawned** lead, upstream wake is already wired.
- If you are a **top-level** lead you started yourself, run `install_lead_wake`
  once, then keep an inbox watcher armed as a **background** task (the lead-wake
  Stop hook will print the exact `watch` command if none is armed). Re-arm it
  after each wake; use a long `--timeout` so an idle wait rarely re-fires.

### 4. Exchange messages

- **Read replies:** `read_messages` (delta-by-default; keep calling while
  `has_more` is true).
- **Send work:** `send_message(to="<member-name>", text="...")`. To an external
  member this returns `delivery:"inbox"` — pull-only, unconfirmed. Do **not**
  pass an `idempotency_key` (that is for guaranteed delivery to spawned agents).
- Confirm round-trips by content: include a distinctive marker in a probe and
  check the member echoes it back.

### 5. Teardown

- The member should call `leave_team(member_token=...)` when it is permanently
  done. Once it is `left`, it no longer counts as a live child, so your
  lead-wake stops nagging you to watch for it.
- If the member is gone and did not leave, you may `kill_agent("<member-name>")`
  from the lead. For an external record this only **deregisters** the membership;
  it never probes or signals the informational PID.

## Gotchas

- Each interactive session gets its **own** win-agent-teams server process, but
  sessions started in the **same folder** share one on-disk session/team (the
  workspace binding is cwd-based). So a member you invite from the *same folder*
  is joining the *same session you already lead* — fine, but see the member
  skill's Mode A for why such a member should stay **watcher-only** (installing
  member-wake there would nag your own orchestrator session).
- "Done" ≠ "correct": verify a member's deliverable independently, exactly as
  with a spawned worker.
