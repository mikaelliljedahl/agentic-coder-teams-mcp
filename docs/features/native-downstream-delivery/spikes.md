# Native downstream delivery — spikes

Windows 11 Pro 26200 test VM, 2026-09-26. codex-cli 0.157.1 (Codex Desktop
26.924.22138 install, native `codex.exe`), Claude Code CLI 2.1.281 hosted by
Claude Desktop 2.9939.2.

## S-1 — `codex queue` into a live Codex TUI child

**Setup.** The target was a real interactive Codex child spawned by
win-agent-teams: `plan-reviewer`, bound `backend_session_id`
`01a0ddc6-…`. It was idle after its review turn. The spike script
`s1_queue.py` (in the session scratchpad) ran
`codex.exe queue --thread <id> --message <text>` with `CODEX_HOME=~/.codex`,
`cwd=~`, and `AGENT_*` / Claude messaging variables removed. `<text>` was
`delivery.delivered_prompt(..., single_line=False)`: three lines containing
`"`, `(`, `)`, `&`, `<`, `>`, followed by the multi-line
`[win-agent-teams delivery id: wat-deliver:<nonce> …]` marker.

**Evidence.**

- `codex queue` exited 0 and printed
  `Queued message 01a0ddd0-2200-… for thread 01a0ddc6-…`. That is a queued
  submission id.
- The nonce first appeared in the rollout 8.3 s after the queue call.
- Two records carried the nonce:
  - `type:"response_item"`, `payload.type:"message"`, `role:"user"`, content
    `[input_text]`.
  - `type:"event_msg"`, `payload.type:"item_completed"`. The scanner ignores
    this one.
- `delivery.receipt_nonces(record, "codex")` returned exactly the nonce. The
  existing scanner detects a queued turn **unchanged**.
- The child answered `ACK-S1`. The multi-line text and the metacharacters
  arrived intact through the native `codex.exe`.

**CLI surface.** `codex queue --help` has only `--thread` and `--message`,
with no list or remove. `codex app-server generate-json-schema --experimental`
exposes `thread/queue/{add,list,delete,reorder,start,update}` and a
`thread/queue/changed` notification. The relevant methods are:

- `ThreadQueueDeleteParams {threadId, queuedSubmissionId}` → `{deleted: bool}`
- `ThreadQueueAddParams {threadId, clientUserMessageId, input: [UserInput]}`

**Elevation.** A new Codex TUI **cannot** be started from an elevated shell:
`Error: start the Windows daemon from a non-elevated terminal; shared clients
must not inherit administrator privileges`. `codex queue` itself works from
the same elevated shell once the daemon is running. win-agent-teams spawns
from the (non-elevated) MCP server, so it is unaffected.

**Not yet measured:** the busy-turn case (V2) and a 20 000-character argv.

**Impact on the plan.**

- A is feasible on Windows. The receipt path needs no scanner change.
- The queued submission id should be stored on the delivery row. With it,
  `thread/queue/delete` could become the **authoritative removal proof** that
  review finding 1 asks for. That is a new spike, S-5, because the method is
  experimental.

## S-2 — env propagation to a Claude child's MCP server

**Blocked.** The `claude` CLI on this VM is not logged in (`Not logged in ·
Please run /login`); Claude Desktop uses its own authentication. Claude
children cannot be spawned here until the CLI is logged in. Run S-2 on Linux,
or after a `claude /login` on this VM.

## S-3 — persisted shape of a socket-posted user line

**Not run on Windows.** There is no safe target: posting into the pipe of the
session that runs the spike would inject into the live orchestrating
conversation. Run it on Linux against a throwaway child, or on Windows once
Claude children can be spawned (S-2).

## S-4 — Windows pipe owner

**Setup.** `s4_pipe_owner.py` opened `CLAUDE_CODE_MESSAGING_SOCKET`
(`\\.\pipe\LOCAL\cc-msg-…`) with
`CreateFileW(GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED)`,
queried the server PID, and closed the handle **without writing**.

**Evidence.**

- `GetNamedPipeServerProcessId` on the **client-opened** handle returned
  11928.
- That equals `CLAUDE_PID`, and it is the nearest `claude.exe` ancestor in
  python ← powershell ← cmd ← claude.exe 11928 ← Claude.exe 2596 (Desktop) ←
  explorer.
- 11928 is the Claude Code CLI (`…\Roaming\Claude\claude-code\2.1.281\claude.exe`)
  that Claude Desktop starts.
- The Desktop app's own image is also named `Claude.exe`. Only the **nearest**
  match is the pipe owner.
- `GetNamedPipeInfo`: byte mode, 64 KiB in and out buffers, 255 instances.

**Impact on the plan.**

- H3-W (server PID equals the nearest `claude` host PID) holds for the
  CLI-under-Desktop arrangement. That answers review finding 12's request for
  a live proof.
- The host walk must keep stopping at the nearest match.
- A standalone CLI (not under Desktop) still needs the same check (smoke N6).

## Windows live smokes (2026-09-26)

Windows 11 Pro 26200 test VM. The MCP server ran from branch
`feat/native-downstream-delivery` at 147efa6, with
`WIN_AGENT_TEAMS_NATIVE_WAKE=1` and `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` in
both the Claude and the Codex MCP config. codex-cli 0.157.1; the lead was
Claude Code CLI 2.1.281 hosted by Claude Desktop. Child: an interactive Codex
TUI `n1probe` (tier `cheapest`), spawned by the lead.

| Smoke | Result | Evidence |
|---|---|---|
| N1: idle Codex child | **pass** | Two follow-ups (`n1-first`, `n1-second`): `status: delivered`, `method: codex_queue`, `replaced_existing: false`, pid 3820 both times; the child answered `ACK-N1`. Settled 2.5 to 7.6 s after creation. The marker carried `idle_seq`, `turn_seq`, `backend_session_id` and `dispatch_epoch: 1`. |
| §2.7 propagation to a Codex child | **pass** | The child's command line had `WIN_AGENT_TEAMS_NATIVE_WAKE = "1"` and `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM = "1"` inside `-c mcp_servers.win-agent-teams.env={...}`. |
| N5 (dead child resumes) | **pass** | After `Stop-Process` on pid 3820, follow-up `n5-dead-resume` returned `method: resume`, new pid 15216, same backend session. Epoch 2 in the marker, the record and `dispatch-epochs.json`. The next follow-up (`n5-native-after-resume`) went native again: `codex_queue`, pid 15216. |
| N6: Claude lead woken on Windows (CLI under Desktop) | **pass** | The child ran `send_message` to `team-lead` after a 20 s sleep while the lead's turn had ended and no watcher was armed. The lead session received `[win-agent-teams wake #1] 1 unread message(s) ... from: n1probe (1)` and `read_messages` returned `N6-WAKE from n1probe`. `session_info.native_wake.claude_channel` was `available`. |

**Not run on this VM.**

- The N5 barrier with a live unresolved queue item was not run; unit tests
  cover it.
- N6 with a standalone CLI (not under Desktop) was not run.
- N7 needs a Codex lead (TUI or Desktop).
- N8 needs server restarts with other flag values; golden tests cover it.
- N3 and N4 need a logged-in `claude` CLI.

| Smoke | Result | Evidence |
|---|---|---|
| N7: Codex lead woken by `codex queue` (spawned nested lead) | **pass** | `n7lead` (Codex, `enable_spawned_lead_wake=true`) ran the shell command and `set_lead_wake`: `status: provisional`, `reason: awaiting_parent_binding`. After the parent's `check_agent(full=True)` bound `backend_session_id` equal to the thread, the registration became `active`. Phase 2 reached `n7lead` via `codex_queue`. It spawned `n7child`, ended its turn without polling or a watcher, and was woken by `[win-agent-teams wake #1] 1 unread messages in your team inbox from n7child:1 ...`. `n7child` sent at 20:38:38.99 UTC; `n7lead` had read it and replied upstream by 20:38:58.22 (about 19 s, model turn included). The same reply woke this Claude lead again through the pipe (`wake #2`), a second N6 pass. |

**Observation (NIT, follow-up).** With the flags on, the Claude delivery poster
also runs in Codex agents' MCP servers. It writes
`native-delivery-<name>.json` with `channel: no_socket` and `host_pid: 0` every
tick. That marker can never pass E3, so it is harmless, but the poster could
skip agents whose channel is unavailable.

**Still not run:** N7 with a human-started lead (Codex TUI or Desktop; the
user is running it) and the other items listed above.

### Codex Desktop, human-started (2026-09-26, 20:40 to 20:43 UTC)

Codex Desktop was opened in `C:\code`, the same workspace as this Claude
lead, so its win-agent-teams server resolved to the **same session**
(`bcba8d19…`) with the **same identity**, `team-lead`.

| Smoke | Result | Evidence |
|---|---|---|
| N7, human-started Codex lead | **wake: pass; body: taken by the other lead** | `set_lead_wake` returned `active` at once (no provisional step), generation 1, `thread_verified=true`, host pid 4796. It spawned `n7child-2`, whose reply at 20:41:18 woke **both** leads: Codex Desktop received `[win-agent-teams wake #1] 1 unread messages in your team inbox from n7child-2:1` with no polling or watcher; this Claude lead received `wake #3` through the pipe. The Claude lead read the message first, which advanced the shared `team-lead` cursor, so Codex Desktop's `read_messages` came back empty. |
| External Codex member (`codexdesk`), member → lead | **pass** | The same Desktop thread joined as `codexdesk` via a join ticket and `external_set_wake`. Its `external_send` woke the Claude lead through the pipe (`wake #4`, `wake #5`). |
| External Codex member, lead → member | **round trip: pass; wake source ambiguous** | `send_message(to="codexdesk")` returned `wake: {method: codex_queue, status: queued}`, and `EXT-ACK codexdesk` came back at 20:42:42. The wake text Codex quoted was the **lead** notice (`… from codexdesk:1 … call read_messages`), not the member notice: the same thread was registered both as lead `team-lead` and as member `codexdesk`, so the member's own upstream message also woke it in its lead role. |

**Finding (pre-existing, made visible by D): two human-started leads in one
workspace share `team-lead`.** Session and identity are derived from the
workspace, so a Claude lead and a Codex lead started in the same directory
share the inbox, the read cursor and the wake. Each is woken; whichever
reads first takes the message. This is not introduced by this feature, but
with lead wake on both hosts it now shows. **Follow-up:** give each
human-started lead its own identity or session, or refuse a second live lead
registration for the same identity. **For a clean N7:** open the Codex lead
in its own directory.

### Codex Desktop, isolated workspace (2026-09-26, 20:46 to 20:50 UTC)

Codex Desktop was reopened in its own folder, `C:\code\Ccoden7test`, so its
server had its own session (`a51f94f9…`). Reported by the Desktop thread
itself after joining this lead as external member `codexn7`.

| Smoke | Result | Evidence |
|---|---|---|
| N7, human-started Codex lead (clean) | **pass** | `codex_lead: {status: active, generation: 1, thread_verified: true}` straight after `set_lead_wake`. Spawned `n7child` at 20:46:38. The child sent at 20:47:34.50. The lead, with its turn ended and no polling, watcher or sleep, got `[win-agent-teams wake #1] 1 unread messages in your team inbox from n7child:1. … call mcp__win_agent_teams__read_messages …` at 20:47:50 (about 16 s after the send). `read_messages` returned `N7-WAKE from n7child`. |
| External Codex member, member → lead | **pass** | `codexn7` joined via a join ticket and `external_set_wake` (thread `01a0df77-…`, generation 1). Its `external_send` report woke this Claude lead through the pipe (`wake #6`). |
| External Codex member, lead → member (clean) | **pass** | `send_message(to="codexn7")` returned `wake: {method: codex_queue, status: queued}`. The member was woken by the **member** notice: `win-agent-teams: wake 1 new message from team-lead in your member inbox - call mcp__win_agent_teams__external_read …` (seen 20:50:05). It replied `EXT-ACK codexn7` at 20:50:14, which woke this lead again (`wake #7`). |

With separate workspaces, every direction works on Windows without a watcher:
Claude lead ↔ Codex child, Codex lead ← Codex child, and Claude lead ↔
external Codex member. The earlier ambiguity came only from the
shared-identity collision described above.

### Claude children on Windows: N3 and N4 (2026-09-26, after `claude /login`)

**First attempt: B never engaged (bug, fixed in 6f7ef3b).** The claude-code
child `n3probe` ran in a Windows Terminal tab. Its record `pid` was the in-tab
PowerShell launcher (1236), while the child's poster reported the `claude.exe`
host that owns the pipe (17208, a direct child of 1236). `_record_hosts` had no
Windows launcher resolution, so E3 never matched. Two follow-ups went to
`method: resume` with a new pid, although the capability marker said
`channel: available`. That fallback was safe but meant B was never used on
Windows. Unit tests and the review missed it because they used fake PIDs.
**Fix:** for claude-code records on Windows, `_record_hosts` also accepts the
single direct `claude.exe` child of a token-verified wrapper, created no
earlier than it. The lead's E3 and the poster's `take`/`begin` binding both
read this set.

**After the fix** (server restarted; fresh child `n3b`, haiku, WT tab):

| Smoke | Result | Evidence |
|---|---|---|
| N3: idle Claude child | **pass** | `n3b-first`: `status: delivered`, `method: claude_mailbox`, pid 2656, `replaced_existing: false`. The child answered `ACK-N3`. Afterwards the mailbox had no entries (retention removed the settled entry) and `consumed["1"] = {seq: 1}`. |
| N4: busy Claude child | **pass** | `n4-busy-task` (native) started a turn at 21:13:57 UTC. The child's harness blocked a foreground `Start-Sleep`, so it backgrounded it and ended the turn at 21:14:10. `n4-while-busy` was sent at about 21:14:01 while the marker said `running`. The poster held it and posted at 21:14:11, right after the idle transition. The transcript has exactly one user record with the prompt, and the child answered `ACK-N4` at 21:14:12. `consumed["1"]` moved to `seq: 3`. The busy window was about 10 s rather than the planned 25 s, but the hold-until-idle behaviour is shown. |

Socket-posted messages appear in the child as a user turn prefixed "Another
Claude session sent a message:", with the delivery marker intact. The receipt
scanner found them, and both deliveries settled `delivered`.
