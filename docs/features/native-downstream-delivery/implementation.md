# Native downstream delivery: implementation status

Status as of 2026-09-26. This file is the living handoff for the feature.

**Branch and location.** Branch `feat/native-downstream-delivery`, worktree
`C:\code\wt-native-downstream`, base `origin/main` 471a175 (after #70). Pushed.

**Plan.** [plan.md](plan.md) v3.1. §2.9 overrides §2.3 and §2.4 where they
differ.

**Review.** [plan-review.md](plan-review.md). Round 4 was **APPROVED_WITH_NITS**
from a Codex (tier high) reviewer, with 0 BLOCKER, 0 MAJOR, 2 MINOR and 1 NIT.
All three were addressed during implementation (steps 1, 2 and the epoch
work).

**Spikes.** [spikes.md](spikes.md). S-1 and S-4 are done. S-2 and S-3 need
Linux, or `claude /login` on the Windows VM. S-5 is optional. S-6 decides
whether D can self-register.

**Workflow rule from the user.** From here on, subagents do **all** the
coding: Claude subagents or Codex via win-agent-teams. The lead orchestrates,
reviews and runs the gates.

## Done

All implementation steps are done, each by a subagent with red/green TDD,
and every commit passed the four gates on Windows.

| Step | Commit | What |
|---|---|---|
| Plan and reviews | 07eaf67 → 59ca4c9 | plan v1→v3.1; four plan-review rounds; spikes.md |
| C: Windows pipe | 71b5e7a, 11addf5, 16fdd4f, 412ab23 | `winpipe.py`, Windows H3-W resolve and post, `write_started` |
| Runner, hooks, record fields | ebe5b4a, 36f317e, 91c7b79, 1459cc8 | `codex_queue`, epoch-bound `idle_seq`, `interactive`/`codex_home`/`dispatch_epoch` |
| Linux runbook | 0501439 | [smoke-linux.md](smoke-linux.md) |
| 1. N5 and two-stage selection | 48653d7 | `DeliveryTransaction.unresolved_native`, `_native_candidate`, stage 2 under the lease, `_NATIVE_DISPATCH` seam |
| 2. A: Codex child | b5c06c9 | `_dispatch_codex_queue`, `carrier_ref` CAS, outcome table, receipt-only settlement, kill `native_unresolved`, CLI `deliveries release-native`, command budget (UTF-16 on Windows) |
| 3. B: mailbox store | 3674463 | `delivery_mailbox.py` |
| 4–5. B: lead side and poster | 1b58767 | `_dispatch_claude_mailbox`, recovery table, kill/force epoch bump and retract, `delivery_poster.py` |
| 6. D: Codex lead wake | 4290a48 | `set_lead_wake`, provisional/active/cleared, incarnation-keyed state |
| 7–8. Flags, tool text, docs | ffa3a44 | `propagated_env`, flag-on notes, protocol reference §4d, README, INSTALL §6a, skills |
| 9. Implementation review | 57e4341, c1e78d1, 9cf207b | [implementation-review.md](implementation-review.md): four rounds, 4 MAJOR fixed, round 4 **APPROVED** |

**Decisions made during implementation** (all recorded in the commit messages
or review dispositions):

- A `taken` mailbox entry is retracted only by `kill_agent` and
  `force_clear_lease` (CLI `lease force` and `lease clear`), after the epoch
  bump.
- `message_too_large` for Codex resume is enforced only with the master and
  downstream flags on; flag-off behaviour matches `main`.
- Epochs are minted above `max(high-water, record, state marker)`, and a
  record or marker that already carries an epoch gets a fresh one on every
  resume, whatever the flags. A child never inherits its parent's epoch.
- Codex lead wake queues under the registration lock, which is held across
  the bounded `codex queue` call (15 s default).
- D self-registration (S-6) and S-5 authoritative removal are not
  implemented.
- E6 (a Codex target must be idle) stays until N2 has run live.

**Gates at 9cf207b (Windows):**

- `pytest`: 2623 passed, 10 skipped.
- `ruff format` and `ruff check`: clean.
- `ty`: only the 2 diagnostics that already exist on `main`
  (`herdr_nested_check.py:152`, `test_join_team.py:750`).

## Remaining work

1. **Linux smokes, part 1** (L-1 to L-4 in [smoke-linux.md](smoke-linux.md)):
   N2 (decides E6), S-6, S-2, S-3. Run by the user.
2. **Live merge gates.** Windows: N1, N5, N6, N7, N8. Linux: N3, N4 and
   part 2 of the runbook. These need the MCP server running from this branch
   with `WIN_AGENT_TEAMS_NATIVE_WAKE=1` and
   `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1`.
3. **Open the PR** with `--repo mikaelliljedahl/agentic-coder-teams-mcp`.

Possible follow-ups, not in this PR: a spawned Codex lead binding its own
record so it leaves `provisional` without the parent syncing; D
self-registration if S-6 passes; S-5 removal.

## Environment notes (Windows test VM)

- **Python.** Claude Desktop is an MSIX app, and uv writes into its private
  `AppData`. The venvs point at `~\.local\share\uv\python`; set
  `UV_PYTHON_INSTALL_DIR` to that before running `uv`.
- **Codex CLI.** It refuses to start its daemon from an elevated shell.
  Spawned agents started through the MCP server work. `codex queue` works from
  anywhere once the daemon is running.
- **Claude CLI.** It is not logged in, so no Claude children can be spawned on
  this VM.
- **Plan reviewer.** The Codex agent `plan-reviewer` (session `bcba8d19-…`,
  thread `01a0ddc6-…`) is still alive and can be reused for the
  implementation review.
