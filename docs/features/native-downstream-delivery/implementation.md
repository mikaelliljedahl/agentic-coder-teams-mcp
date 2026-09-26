# Native downstream delivery: implementation status

Status as of 2026-09-26. This file is the living handoff for the feature.

**Branch and location.** Branch `feat/native-downstream-delivery`, worktree
`C:\code\wt-native-downstream`, base `origin/main` 471a175 (after #70). Not
pushed yet.

**Plan.** [plan.md](plan.md) v3.1. §2.9 overrides §2.3 and §2.4 where they
differ.

**Review.** [plan-review.md](plan-review.md). Round 4 was **APPROVED_WITH_NITS**
from a Codex (tier high) reviewer, with 0 BLOCKER, 0 MAJOR, 2 MINOR and 1 NIT.
The three items still open:

- Measure the command budget in UTF-16 units.
- Word the flag-off error so it does not leak the feature.
- Allocate the epoch and pass the same one before the child starts.

**Spikes.** [spikes.md](spikes.md). S-1 and S-4 are done. S-2 and S-3 need
Linux, or `claude /login` on the Windows VM. S-5 is optional. S-6 decides
whether D can self-register.

**Workflow rule from the user.** From here on, subagents do **all** the
coding: Claude subagents or Codex via win-agent-teams. The lead orchestrates,
reviews and runs the gates.

## Done

| Part | Commits | What |
|---|---|---|
| Plan and reviews | 07eaf67 → 59ca4c9 | plan v1→v3.1; four review rounds with dispositions; spikes.md |
| C: F5 Windows pipe | 71b5e7a, 11addf5, 16fdd4f, 412ab23 | `winpipe.py` (details below); `native_wake` Windows H3-W resolve and post; `write_started` on POSIX and Windows |
| Runner | ebe5b4a, 412ab23 | `native_wake.codex_queue` (details below); `CodexMemberWake` uses it |
| Hook marker | 36f317e, 1459cc8 | `hooks.py` (details below) |
| Record fields | 91c7b79, 1459cc8 | `interactive`, `codex_home` and `dispatch_epoch` on spawn and resume, flag-gated (details below) |

**`winpipe.py`:**

- overlapped writes with a bounded cancel-drain;
- the owner PID is checked on the handle used for writing;
- undrained writes are parked, with per-path reservation that ignores case;
- `MAX_PARKED` cap, and a reaper that runs on every tick;
- writes are exception-safe;
- invalid deadlines are refused.

**`codex_queue`:**

- stages `Popen` then `communicate`: only a failure to construct the process
  proves nothing was queued;
- returns `QueueOutcome` with the submission id.

**`hooks.py`:**

- the marker carries `idle_seq` (counts transitions only), `turn_seq`,
  `backend_session_id` and `dispatch_epoch` (taken from
  `WIN_AGENT_TEAMS_DISPATCH_EPOCH`);
- writes are serialised under `state-<agent>.lock`;
- a write from an older epoch is dropped;
- a new epoch or a new session resets the counters.

**Record fields:**

- the epoch is minted from a per-session `dispatch-epochs.json` high-water
  mark, so it stays monotonic when a name is reused;
- `process_base` exports it to the child's environment.

**Gates at 1459cc8 (Windows):**

- `pytest`: 1899 passed, 7 skipped.
- `ruff format` and `ruff check`: clean.
- `ty`: only the 2 diagnostics that already exist on `main` (`herdr_nested_check.py:152`, `test_join_team.py:731`).

## Remaining work

Each step goes to a subagent, with red-green TDD and the relevant plan
sections in the prompt.

1. **N5 barrier and two-stage selection** (plan §2.1).
   - N5 query over `deliveries.json`, keyed by `(sender, key)`, independent of
     the flags.
   - E0–E6 eligibility.
   - Commit the choice under the lease, with fallback through the idle gate.
   - Delivery-row fields: `method`, frozen `carrier`, and `public_view` gated
     by the flag.
   - Tests 1–3.
2. **A: Codex native dispatch** (§2.2, §2.9 R3-5).
   - `carrier_ref` compare-and-swap (CAS).
   - The outcome table.
   - Durable settlement: never terminal on absence; the kill report;
     `deliveries release-native` in the CLI.
   - Native finalisation branch.
   - Command-budget check.
   - Tests 4–5.
3. **B: `delivery_mailbox.py` store** (§2.3.2, §2.9 R3-3).
   - Initialise before the first `sent`.
   - The publish, revoke, take, begin and retract state machine, as
     authoritative CAS operations.
   - Tombstones, retention, and fail-closed handling.
   - Test 6.
4. **B: lead order, recovery and kill/force** (§2.3.3, §2.3.5).
   - Bump the epoch.
   - Retract offered entries.
   - Tests 7 and 12.
5. **B: child `DeliveryPoster`** (§2.3.4, §2.9 R3-2).
   - Capability marker.
   - `consumed[epoch]` with CAS rollback.
   - Validation at take and begin.
   - Tests 8–9.
6. **D: Codex lead wake** (§2.6).
   - `set_lead_wake`.
   - Independent channel gating.
   - State keyed per session and incarnation.
   - Provisional registrations for spawned leads.
   - Recovery text.
   - Test 11.
7. **Flag propagation** (§2.7), test 12. **Tool docstrings** (§2.8), plus
   flag-off golden tests.
8. **Docs**: the protocol reference, `README.md`, `INSTALL.md` §6a, and the
   skills.
9. **Post-implementation review** by Codex (tier high). Save it as
   `implementation-review.md`.
10. **Live gates.**
    - On Windows: N1, N5, N6, N7, N8.
    - On Linux: S-2, S-3, N3, N4, and N2 (busy) to decide E6.
11. **Open the PR** with `--repo mikaelliljedahl/agentic-coder-teams-mcp`.

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
