# Linux smoke prompt (paste into a Claude Code agent on the Linux machine)

---

You are verifying the **last open release gate** on PR #48 of
`mikaelliljedahl/agentic-coder-teams-mcp` (branch
`feat/lead-wake-conversation-scoping`), on Linux. Everything else is green;
this gate is the reason the PR is still a draft.

## Setup

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
git checkout feat/lead-wake-conversation-scoping
uv sync --locked --group dev
```

Read these first — do not re-derive the design from the code:

- `docs/features/lead-wake-conversation-scoping/plan.md` (approved plan, v3)
- `docs/features/lead-wake-conversation-scoping/spike.md` (the Windows
  measurements, especially **F6**)
- `src/claude_teams/procinfo.py` (`_read_linux_process`, `resolve_nearest_host`,
  `host_kind`, `is_claude_host`)

## Background: what the feature does and why this gate exists

The lead-wake `Stop` hook was nagging Claude Code conversations that had never
used the win-agent-teams MCP, because `install_lead_wake` writes a **shared**
settings file and ownership was resolved per-workspace, not per-conversation.
The fix binds the owner at install time: it resolves the **nearest host process
ancestor** of the installing conversation, captures its pid + creation token,
bakes both into the hook command, and the hook compares them against its own
resolved host before doing anything.

The whole feature therefore rests on one empirical claim:

> The MCP server and the Stop hook of the *same* conversation independently
> resolve the *same* nearest host process — and two different conversations
> resolve *different* ones.

That was measured on Windows. On Linux it has **never been run against real
processes** — only synthetic pid→ppid maps. That is what you are testing.

## Why this needs a human-driven check rather than the test suite

The failure mode is **silence, not a crash**. If host resolution misbehaves on
Linux, `install_lead_wake` refuses with `host_not_found` and the hook then
allows every conversation — the suite stays green and everything looks calm
while the protection is simply gone. So: never conclude "it works" from an
absence of errors. Confirm positively that the pids match, or report failure.

Two specific hazards, both already fixed in code but unverified on Linux:

1. `/proc/<pid>/comm` reports the **thread** name, truncated to 15 bytes, and
   for an npm/node install of Claude Code (`claude` is a `#!/usr/bin/env node`
   shim) it reports `node`, not `claude`. The reader therefore prefers
   `/proc/<pid>/cmdline` argv0 and falls back to `comm`. Check that this is
   right for *your* installation method.
2. Host selection is **two-step**: find the nearest ancestor in the full host
   set (claude, codex, pi — including node-launched shims), *then* require it to
   be a Claude host. A Claude-only search walks past an intervening `codex`/
   `node` layer and lands on the lead's own conversation (spike F6).

## Task 1 — paired ancestry capture (the gate)

Write a small throwaway script that, for a given pid, prints the ancestor chain
and the resolved nearest host using the repo's own code:

```python
from claude_teams import procinfo
# procinfo.resolve_nearest_host() resolves from the CURRENT process;
# read procinfo.py for the snapshot/walk entry points to resolve from a
# given pid instead.
```

Then capture, with `win-agent-teams` configured as an MCP server in Claude Code:

1. **Conversation A** — start a Claude Code conversation, call any
   win-agent-teams tool so its MCP server is running, and record:
   - the **server-side** chain (the `claude_teams.server_simple` process), and
   - the **hook-side** chain (a process launched the way a `Stop` hook is —
     via a shell from the same conversation).

   → Both must resolve the **same** nearest host pid.

2. **Conversation B** — a second, simultaneous Claude Code conversation **in the
   same folder**. Capture the same two chains.

   → Must resolve a **different** nearest host pid from A's.

3. **A server-spawned Claude agent** (`spawn_agent(backend="claude-code")`).
   Capture its server-side and hook-side chains.

   → Both must resolve **that agent's own** host, not the lead's.

Record every chain verbatim (pid + image name per level, plus argv0 where it
matters). Note the depth: on Windows there were up to four intermediate levels,
including a venv→uv python re-exec and a console shim.

## Task 2 — end-to-end behavior

With conversations A and B still running in the same folder:

1. In **A**: `install_lead_wake()`. It must **succeed**, not refuse. If it
   refuses, report the `reason` and the `chain` it returned — that is the gate
   failing, and it is the single most important thing to report.
2. In **A**: `spawn_agent(...)` a trivial worker. The result now carries
   `wake_binding`; it should be `{"state": "bound"}`.
3. In **B** (which has used no team tools): end a turn. B must **not** be
   blocked or nagged.
4. In **A**: end a turn without arming a watcher. A **should** be nudged
   (D3/D5).
5. Restart **A**, then `spawn_agent` again: `wake_binding` should now be
   `"stale"` with a hint to re-run `install_lead_wake`.

The hook writes one structured stderr line per evaluation, starting
`win-agent-teams/lead-wake`. Capture those lines — `why="owner-unknown"` is the
signature of host resolution failing, and `why="not-owner"` is the intended
bystander path. Distinguishing those two is the point of the exercise.

## Task 3 — the suite on Linux

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Report the real results. Never report a gate green when it is red; state
whether any failure pre-dates this branch.

## Deliverable

Write `docs/features/lead-wake-conversation-scoping/linux-smoke.md` on the
branch containing: every captured chain verbatim, an explicit
pass/fail per numbered item above, the relevant `win-agent-teams/lead-wake`
stderr lines, the four gate results, your Claude Code installation method
(npm/native/other) and `claude --version`, and a clear verdict on whether the
paired-ancestry gate is discharged on Linux.

If something fails, do **not** fix it — report the measurement. The design
decision belongs upstream. Commit only that one file; do not push unless asked.
