---
name: external-member-join
description: Join another agent's team from a manually-started interactive session as a pull-only external member (join_team + external_read / external_send + leave_team), so a lead can hand you work and you can reply upstream. Use when you were given a join ticket or join prompt. Presents two connection modes to choose from — watcher-only (member co-located with the lead on the same machine/folder) and member-wake (member in a separate profile/machine) — with the exact steps and trade-offs for each.
---

# Join a team as an external member (member side)

You were started manually (e.g. a Claude Desktop session with a browser) and
given a **join ticket** by a lead. This skill connects you as a **pull-only
external member**: the lead can put work in your inbox, and you reply with
`external_send`. It is the mirror of the lead's
[`external-member-invite`](../external-member-invite/SKILL.md) skill.

Rules that hold in every mode:

- Use **only** `join_team`, `external_read`, `external_send`, `leave_team`, and
  (Mode B) `install_member_wake`. Do **not** use ambient team tools like
  `send_message`/`read_messages` from this external-member conversation.
- **Save the `member_token`** returned by `join_team` in this conversation. It is
  your credential for every call below and survives an MCP server restart.
- Downstream (lead → you) is **pull-only**: nothing is pushed into your
  conversation. You are woken only because a background **watcher** you armed
  fires when a message lands — then you drain it. That watcher is the actual
  wake mechanism in both modes.

## Choose your mode

| | **Mode A — watcher-only** | **Mode B — member-wake** |
|---|---|---|
| **Use when** | You are on the **same machine + same folder/home** as the lead (two Claude sessions side by side — e.g. a visual QA next to the build orchestrator) | You are in a **separate profile or machine** from the lead (ideally launched with `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`) |
| **How you stay woken** | You arm and re-arm the reader-scoped watcher yourself | `install_member_wake` adds a Stop hook that *enforces* the watcher is armed and blocks your turn-end while unread work waits |
| **Why not the other** | Installing member-wake here writes a user-scope hook that also fires in the **lead's** co-located session and nags it (harmless but noisy) — so skip it | Without the hook you must remember to re-arm the watcher every time; the enforcement is worth it when you are a dedicated remote member |

If unsure and you are on one shared machine/folder: pick **Mode A**.

## Mode A — watcher-only (co-located)

1. **Join** with the literal values from your ticket:
   ```
   join_team(session_id='<session-id>', token='<token>')
   ```
   Save the returned `member_token`. The result also gives `watch_argv` (a
   shell-neutral watch command already scoped to your member name).

2. **Arm the reader-scoped watcher** as a **background** task
   (`run_in_background = true`), with a long timeout so it carries a real idle
   wait rather than expiring every minute:
   ```
   <watch_argv ...> --timeout 3600
   ```
   (`watch_argv` is `... -m claude_teams.cli watch <joined-session-dir> --reader
   <your-name>`; append `--timeout 3600`.) When it fires (`reason:"message"`) it
   re-invokes you.

3. **Drain and reply** whenever the watcher wakes you:
   ```
   external_read(member_token=...)          # read the lead's work
   external_send(member_token=..., text=...) # reply upstream
   ```
   Keep reading while there is more. Announce readiness once after joining so the
   lead knows you are live.

4. **Re-arm** the watcher after each wake (it is one-shot). This is the one thing
   Mode A does not enforce for you — do not forget it, or you will go idle
   without a wake.

5. **Leave** when permanently done:
   ```
   leave_team(member_token=...)
   ```

## Mode B — member-wake (separate profile/machine)

Do steps 1–3 of Mode A, and additionally, right after joining:

- **Install the enforcing Stop hook:**
  ```
  install_member_wake(joined_session_id='<session-id>', member_name='<your-name>')
  ```
  Expect `action:"installed"`, `scope:"user"`. Now, on every turn end, the hook
  checks your member inbox in the lead's session and refuses to let you go idle
  without an armed watcher — printing the exact watch command if none is running.
  It bakes **no credential** (only your name + the joined session dir).

- Still arm the watcher (step 2); the hook enforces it but does not start it.
- The hook **fails open** (allows the stop) once your membership is
  `left`/killed or the joined session goes stale, so a stale install is
  harmless.
- Runtime kill switch: `WIN_AGENT_TEAMS_MEMBER_WAKE=0`. Remove the hook with
  `install_member_wake(..., remove=True)`.

Shared-home caveat: if you are actually on the same home as the lead, this hook
also fires in the lead's session — that is why Mode A exists. Prefer a separate
profile via `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1` for a real remote member.

## After an MCP restart

Your `member_token` still works — just resume calling `external_read`/
`external_send` with it. If you lost it, replay the original `join_team(session_id,
token)` during the ticket's retention window to recover the same membership and
token.
