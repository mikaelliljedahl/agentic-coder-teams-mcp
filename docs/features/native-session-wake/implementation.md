# Native session wake — implementation

Implemented in `/home/mikael/code/github/wt-native-session-wake`, on
`feat/native-session-wake`. No commit or push. The worktree was clean before
implementation. The pristine server source matched `e5c8e98` byte-for-byte
(blob `4f00813f5aed0dbab7cc9fbdbcfca62a078f176d`); that source supplied the
flag-off golden tool descriptions and join prompt.

## Final design

- `native_wake.py` provides strict master/sub-switch gates, finite-positive
  settings, Claude host/socket resolution, bounded POSIX JSON-line posting,
  pure notice decisions, successful-only progress, and exponential backoff.
  `NoticeState.observe` tracks coalesce timing outside the pure `plan_notice`.
  Retry deadlines and success timestamps start at transport completion.
- `NativeWakeNotifier` is started only by `main`, as a daemon. It observes
  explicit session state without recovery, scans changed inbox/cursor files or
  pending deadlines, and holds non-inheritable lifetime reader locks. Owner
  contention retries every ten seconds; no heartbeat or stale takeover exists.
  Native Windows and macOS short-circuit before socket/environment/host discovery,
  never start this notifier, and reports `unsupported_platform`.
- Explicit recovery/adoption/create/resume signals activation. Catch-up bypasses
  coalescing only on a target transition. Same-session activation preserves
  notice, backoff, and lock state. Event consumption precedes the latest
  session snapshot; a switch during a scan cannot post to the old lead target.
- External token operations register member targets after releasing the agents
  lock. Membership checks drop targets that stop running. The target-registry
  lock is a leaf lock.
- `external_set_wake` is registered only at flag-on import, also in
  external-only mode. It authenticates running membership, validates canonical
  thread UUID/absolute home, and writes monotonic registration generations.
  Clear preserves a null-thread tombstone.
- The external `send_message` append retains its existing transaction. Wake
  happens afterwards under a per-member lock, with short agents-lock snapshots
  and generation/status revalidation. Notice state, verification cache, and
  backoff are fresh for each registration generation. Older state is replaced.
  Queue never holds the agents lock. Unexpected wake-side errors cannot undo
  a successful inbox append.
- Verification uses existing `state_5.sqlite` via read-only URI and 0.5-second
  timeout, observes WAL rows, refuses archives, closes before queue, and falls
  back to rollout filenames for missing/schema/lock cases. The runner uses the
  discovered binary, reported `CODEX_HOME`, lead home as cwd, DEVNULL stdin,
  UTF-8 capture, and a real subprocess timeout. Its notice is safe for `.cmd`
  and contains no body or credential. No deferred Codex queue timer exists.
- **R4-A addressed:** every eligibility check uses a non-null `thread_id`;
  truthy clear tombstones omit `wake` and call neither verification nor runner.
- Spawn and resume scrub both inherited Claude messaging variables to empty
  strings only with the master flag on. Tests cover all three backends.
- Flag-on tool descriptions carry the best-effort contract, watcher fallback,
  Linux-only Claude restriction, restart procedure, and applicable statuses.
  Flag-off descriptions, tool lists (including external-only), prompts,
  results, environment behavior, and files match the golden baseline. README,
  protocol reference, and both external-member skills describe the same
  additive behavior. `lead_wake.py`, `member_wake.py`, and `watch` are unchanged.

## Red / green evidence

The plan's order was followed: flag-off identity first, Claude group, Codex
member group, then contracts. All queue runners are fakes. The only socket
posted in tests is a test-created AF_UNIX listener under `tmp_path`; no real
Claude channel is contacted. Test interpreters/subprocesses exercise only tool
registration or an advisory lock holder, never a real Codex thread.

| Group | Red evidence before production change | Green evidence |
|---|---|---|
| §4.0 F0–F6, with F7 legacy suites | Baseline characterizations intentionally green; 44 parameterized cases passed before any production edit. Initial test-only setup mistakes (nonexistent registry helper and incorrect external-only expected list) were corrected against the pristine source first. | Every subsequent focused run included the identity tests. F7's existing cases passed in the initial legacy run and the final whole-repo suite. |
| §4.1 1–15, 15a–d; spawn/resume scrub | `ImportError: cannot import name 'native_wake' from 'claude_teams'`; `1 error during collection`. | `87 passed in 6.06s` (43 new cases plus 44 flag-off cases). |
| §4.2 16–26, 17a, 26a–e, R4-A | `AttributeError: module 'claude_teams.native_wake' has no attribute 'CodexMemberWake'`; missing verifier/tool registration; `10 failed, 17 errors`. | `114 passed in 9.00s` (Claude, Codex, and flag-off groups). |
| §4.3 27–28 and availability result | Description lacked `WIN_AGENT_TEAMS_NATIVE_WAKE=1`; prompt lacked `external_set_wake`; `KeyError: 'native_wake'`; `3 failed`. | `47 passed in 5.59s` (contracts plus flag-off cases). |
| Final boundary regressions | Unchanged inbox rescanned each tick; snapshot `OSError` escaped after append; removed member omitted expected `coalesced`; `3 failed`. | `124 passed in 9.10s`, including the four existing entrypoint tests. |
| Pure policy and completion-based backoff | Policy mutated `first_new`; slow failed post retried immediately; slow queue returned `timeout` instead of `backoff`; `3 failed`. | `123 passed in 9.08s` across all new cases before moving contract/scrub tests into the planned existing suites/backend directory. |

Short captured excerpts and focused green outputs are in [evidence/](evidence/).
Before review fixes, added coverage consisted of **123 collected test cases**:

- 44 flag-off identity cases;
- 46 Claude/policy/ownership/activation/scrub cases, including six in
  `tests/test_backends/test_native_wake_env.py`;
- 30 Codex registration/queue/verification/race cases;
- three contract cases added to `test_tool_descriptions.py` and
  `test_join_team.py`.

Refactoring retained the behavioral contracts: shared policy/backoff state,
small injected callbacks, conditional description decoration before MCP
registration, and the existing registry transaction boundaries. The first
whole-suite run found a test isolation mistake introduced here: patching the
MCP instance's `run` left an instance-bound method that defeated an existing
class-level entrypoint patch. Changing the new tests to patch the class fixed
it; this was **not pre-existing breakage**. No existing tests were weakened.

## Deviations

None. Pure policy timing bookkeeping lives in `NoticeState.observe`; the
approved `plan_notice` remains free of mutation and I/O. All review
Dispositions, including transition-only activation, generation keying,
Windows exclusion, read-only WAL verification, and R4-A, are applied.
No follow-up or non-goal was implemented.

## Validation

Final whole-repository Linux gates (not scoped to changed files):

| Command | Result |
|---|---|
| `uv run ruff format --check .` | PASS — 91 files already formatted |
| `uv run ruff check .` | PASS — All checks passed |
| `uv run ty check` | PASS — All checks passed |
| `uv run pytest` | PASS — 1832 passed, 4 skipped |

`git diff --check` passes. No pre-existing gate failure remains or was hidden.
The four skipped tests are the suite's existing platform/dependency skips;
none was added for this feature to bypass a failure. Final outputs are captured
in `evidence/*-final.txt`.

## Manual smokes remaining for the lead

The 2026-09-25 prototype evidence is retained as historical evidence, not claimed
as validation of this implementation. **All implementation smokes remain:**

| Smoke | Remaining verification |
|---|---|
| S1 | Real Claude Desktop lead ↔ Codex Desktop member round trip, flag on, no human nudge |
| S2 | Spawned Claude host exports its own non-empty socket to its MCP child after scrub |
| S3 | Codex/Pi children refuse inherited Claude channels as `host_not_claude` |
| S4 | Real session switch and restart/backlog catch-up after `resume_session` |
| S5 | Bypass-mode lead accepts its own native notice without approval |
| S6 | `crossSessionInbound=refuse` silently drops notice while watcher still wakes |
| S7 | Flag off produces no notices, native lock files, or Codex queue rows |
| W1 | Windows flag-off byte identity and complete Windows suite |
| W2 | Windows Codex Desktop wake through native binary or `.cmd`, intact notice and bounded timeout |
| W3 | Windows `unsupported_platform`, no pipe I/O, and spawned environment scrub |
| V1 | Desktop closed/thread unloaded: observe queued-row persistence until loaded |
| V2 | Queue during a busy turn: observe dispatch timing after that turn |

V1/V2 remain explicitly unverified in `external_set_wake`'s tool description;
update it after recording live outcomes. Independent post-implementation review
and these real-host smokes belong to the lead's review/merge workflow. No live
queue/socket probes, user configuration edits, commit, or push were performed.

## Review fixes

Addressed all five MINOR and four NIT findings in
[implementation-review.md](implementation-review.md), as requested by the lead.
These corrections supersede the plan's macOS availability claim and original
lock names; no deferred transport or other follow-up was implemented.

| Finding | Fix and verification |
|---|---|
| 1 — macOS availability | Claude wake is explicitly Linux-only. Native Windows and macOS report `unsupported_platform` before host lookup and never start the notifier. Tool descriptions (including `session_info`), join prompt, README, reference, both skills, and current implementation wording agree. Tests cover macOS rejection and H2 with an absent `/proc`. |
| 2 — inherited queue environment | The fake runner sees neither Claude messaging variable, any `AGENT_*` variable (including capability), nor `WIN_AGENT_TEAMS_SESSION_DIR`. Unrelated settings and the reported `CODEX_HOME` remain, and the lead's environment is unchanged. |
| 3 — registry I/O on unavailable channels | `tick` consumes activation, drains member registrations, releases existing targets, and returns before any member registry read for unavailable/disabled Claude channels. Four channel-reason cases assert zero registry reads and no files/posts. |
| 4 — member restart contract | `external_read` and `external_set_wake` explain that a Claude-hosted member must call `external_read` once after an MCP restart to re-arm notices; no notice from its notifier arrives until the next `external_read`, `external_send`, or `external_set_wake`. The reference and join skill agree. Persisted Codex registration still supports the lead-side queue path. |
| 5 — member posting coverage | A live member test asserts immediate baseline catch-up, the saved-token `external_read` instruction, sender name, member lock ownership, and a second notice after drain/new growth and the coalesce deadline. |
| 6 — send description | Flag-on registration uses an explicit external-member paragraph, with grammatical `or process resume is involved` wording. It no longer replaces a source sentence. Structural markers fail loudly if the documented paths change. The contract test rejects `or wake is involved`; flag-off golden descriptions are unchanged. |
| 7 — deterministic restart test | A tracking activation event signals when the daemon has completed its empty first tick and entered its 30-second wait. Only then does the test resume or auto-adopt. Both paths post within one second, exactly once, with `alice (1)` in the notice. |
| 8 — lock collision | Lead and member locks are `native-wake-lead.<reader>.lock` and `native-wake-member.<reader>.lock`. The separator cannot occur in a valid name. A lead named `member-alice` and member `alice` both own distinct locks and post successfully. Contract/reference lock names are updated. |
| 9 — member-supplied home | The registration description and reference identify `codex_home` as member-supplied, checked as an existing directory before queueing, and used only as `CODEX_HOME` for the subprocess, never as cwd. They explain the same-user config-selection trust model. The redundant post-null-check condition was removed. |

Red-first evidence was captured before production changes:

- Channel/registry/member/collision run: **7 failed, 1 passed**, including
  host lookup on macOS, registry reads with an unavailable channel, missing
  member lock name, and one notice instead of two on colliding reader names.
  The passing case characterizes existing fail-safe H2 behavior without `/proc`.
- Queue environment/contract run: **2 failed** — inherited variables reached
  the fake runner and descriptions still claimed `Linux/macOS`.
- Green focused run: **124 passed in 8.97s**, including all flag-off identity
  cases. The activation test was then made deterministic without changing
  production behavior.

Nine new collected cases bring feature coverage to **132 added cases**.
Diagnostic excerpts are in `evidence/review-red.txt` and
`evidence/review-contract-env-red.txt`; the focused green run is in
`evidence/review-green.txt`. Whole-repository review-fix gate outputs are in
`evidence/review-*-final.txt`.

All S1–S7, W1–W3, and V1/V2 manual smokes listed above remain for the lead.
No real queue or Claude socket was contacted, no external configuration was
changed, and no commit or push was made.

Final review-fix validation, whole repository on Linux:

| Command | Result |
|---|---|
| `uv run ruff format --check .` | PASS — 91 files already formatted |
| `uv run ruff check .` | PASS — All checks passed |
| `uv run ty check` | PASS — All checks passed |
| `uv run pytest` | PASS — 1832 passed, 4 skipped in 42.24s |

`git diff --check` passes. No pre-existing failure was hidden or remains.

## Live smoke results (2026-09-26, Linux, lead run by the Claude Opus reviewer)

Setup: a Claude Code 2.1.282 lead and a Codex 0.156.1 TUI member, each in its
own herdr tab, both running this worktree's server with
`WIN_AGENT_TEAMS_NATIVE_WAKE=1` (lead via `--mcp-config --strict-mcp-config`,
member via `-c mcp_servers.win-agent-teams.*` overrides). Neither side armed a
watcher, polled or slept.

| Smoke | Result | Evidence |
|---|---|---|
| S1 round trip, flag on | PASS | Three full PING→PONG cycles. Every lead `send_message` returned `wake: {method: codex_queue, status: queued}` and the idle Codex member started a new turn; every member `external_send` woke the idle lead with `[win-agent-teams wake #n] 1 unread message(s) ... from: smoke-codex`. `session_info.native_wake = {claude_channel: available, owner_verified: true, notifier_owner: true}`. |
| S2 spawned Claude child has its own socket | PASS | Child server env `CLAUDE_CODE_MESSAGING_SOCKET=/run/user/1000/cc-socks/3949699.sock`, stem = its own `claude` host pid (not the lead's 3903430). |
| S4 restart with backlog | PASS | Lead `claude` killed; member sent `BACKLOG-1`; new lead called only `resume_session` and received `[win-agent-teams wake #1]` immediately, then read `BACKLOG-1`. |
| S5 bypass-mode lead | PASS | The lead ran with `--permission-mode bypassPermissions`; every notice was delivered without an approval prompt (own-child). |
| S3, S6, S7 | Not run live | Covered by unit/flag-off tests; S7 is the default for every other session on this machine, which runs `main`. |
| Codex **Desktop** member | Covered by the pre-implementation smoke run | Same `codex queue` path; see smoke-run-2026-09-25.md §3. |
| W1–W3 (Windows) | **Pending** | Must run on the Windows machine before merge. |
| V1 / V2 | Open | Not blockers (plan §5). |

Observation: a Claude child spawned by a flag-on lead runs the same server
command but does not inherit `WIN_AGENT_TEAMS_NATIVE_WAKE`; its own wake stays
off unless its MCP entry sets the flag. This matches the per-installation
opt-in.
