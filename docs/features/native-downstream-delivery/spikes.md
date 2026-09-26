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
