# Plan: a parked agent must be able to wake a coordinator

## Problem (observed 2026-09-19, session `6f42b20e…`)

Two spawned `claude-code` planners finished and parked (`Stop` → marker
`{"state": "waiting", "event": "Stop"}`). The coordinator's later `watch` never
woke for one of them, for two independent reasons:

1. **Late `SubagentStop` downgrades a parked marker.** 16 minutes after
   `plan-p883` parked, Claude Code ran an "away summary" in the idle agent and
   fired `SubagentStop`. `hooks.emit` mapped it to `waiting` and overwrote the
   marker's `event` from `Stop` to `SubagentStop`. `cli.watch` deliberately
   treats `SubagentStop` as non-actionable churn (an internal Task subagent
   finishing while the worker keeps going), so from then on the agent looked
   permanently unactionable although it was parked and done.
2. **A watch armed after the transition has nothing to see.** `watch` wakes on
   *edges* (mtime/size changes after its start snapshot). It does check the
   inbox for already-unread messages at start, but never checks for markers that
   are *already* `waiting`. The coordinator armed its watch at 11:42 for agents
   that parked at 11:24 and 11:30, and it sat there until timeout.

## Current behavior

- `hooks.emit` (`src/claude_teams/hooks.py:78-106`) writes
  `{state, event, ts}` for every recognized event unconditionally.
- `cli.watch` (`src/claude_teams/cli.py:575-680`): initial inbox check →
  poll loop on edges; `pending_waits` settle for `_WATCH_SETTLE_SECONDS`
  (default 15 s) before a `waiting` wake; `_waiting_agent` returns `None` for
  `event == "SubagentStop"`.

## Proposed design (v2 — after plan-review round 1, VERDICT REJECTED)

### A. `SubagentStop` writes no marker

`hooks.emit` treats `SubagentStop` like an unrecognized event: nothing is
written, the prior marker stays. `_WAITING_EVENTS` becomes `{"Stop"}`. This
cannot overwrite `running` or `Stop`, needs no read-modify-write, and removes
the `agent_status` lie ("waiting" for an agent mid-Task). The watch keeps its
`_NON_ACTIONABLE_WAITING_EVENTS` filter for markers written by older installs.

### B. Generation-acknowledged parked markers in `watch`

- **Generation.** `emit` adds `"gen": uuid4().hex` to every marker it writes
  (additive; readers ignore it). For a legacy marker without a string `gen`,
  the generation is the SHA-256 of the file bytes. Equality of generations, not
  ordering of timestamps, decides "seen before" — immune to clock rollback,
  equal timestamps, missing/non-numeric `ts`.
- **Candidate record.** `_waiting_agent(path)` is replaced by
  `_parked_candidate(path) -> Parked | None` (`agent`, `gen`, `path`) from ONE
  parse. `_NON_ACTIONABLE_WAITING_EVENTS` and the no/odd-`event` tolerance are
  unchanged.
- **Ack store.** `<session_dir>/.watch/ack-<reader>.json` →
  `{"acked": {"state-<agent>.json": "<gen>"}}`. It lives in a subdirectory,
  and `_snapshot_mtimes` only lists files directly in the session dir, so no
  `--pattern` (including `*`) can classify it as output. Read tolerant of a
  missing/corrupt file (→ empty). Read-modify-write under
  `claude_teams.filelock.file_lock` on `ack-<reader>.json.lock` (cross-platform,
  already used by the registry); the lock is held only for the write, never
  through the settle window.
- **Start-up scan** (default on; `--no-parked` restores edge-only behavior):
  after the unread-inbox check, every `state-*.json` in the session dir that
  matches `pattern`, yields a candidate, and whose `gen` differs from the acked
  one is registered in `pending_waits` with the start time. It then follows the
  existing settle window and the message > output > waiting priority.
- **Settle and ack.** When a pending wait matures, the marker is parsed ONCE
  (one `read_bytes`, hashed and decoded from the same buffer); that candidate's
  `gen` is what gets acked, and the wake names that candidate. Edge-triggered
  waits ack the same way, so a re-armed watch never re-fires on a generation
  it already delivered. **Delivery guarantee: at-least-once.** The wake is
  printed FIRST, then the ack is written (locked load/merge, unique temp file,
  atomic replace, bounded lock timeout); an ack failure of any kind is logged
  to stderr and never changes the exit code (the next watch may deliver the
  same generation again). Two live watchers for the same reader may therefore
  both deliver one park; the lock only guarantees no ack entry is lost.
  Single-live-watcher-per-reader remains the documented operating rule.
- **Reader identity.** The resolved reader (explicit `--reader`, else
  `AGENT_NAME`, else `team-lead`) is validated with `_require_safe_reader`
  regardless of its source before any inbox or ack path is built (exit 1 on an
  unsafe value). Nested leads therefore get their own ack file.
- **Unchanged:** success JSON `{"reason","agent","path"}`, exit 0; timeout exit
  2 with empty stdout; owner-gone exit 4; `--no-inbox`; message/output shapes.
- **Cut:** any `kill_agent` ack cleanup. Stale entries are harmless — a later
  same-name agent parks with a fresh `gen`. Sessions are removed whole.

### C. Contract surfaces

- Tool docstrings in `src/claude_teams/server_simple.py`: `spawn_agent`
  (the `watch_command_*` note) and `agent_watch_paths` state: parked markers
  wake a fresh watch by default, once per reader per park (ack), after the
  settle delay; priority message > output > waiting; `--no-parked`; nested
  reader identity; at-least-once; unchanged JSON/exit codes.
- `docs/reference/agent-messaging-protocol.md` (§ actionable edge, § settle,
  marker table, session-dir file table: add `.watch/ack-<reader>.json`),
  `.claude/skills/agent-orchestration/SKILL.md` (§ actionable edge), README.
- `tests/test_tool_descriptions.py` pins the docstring facts.

## Files affected

- `src/claude_teams/hooks.py`, `src/claude_teams/cli.py`,
  `src/claude_teams/server_simple.py` (docstrings only)
- `tests/test_hooks.py` (SubagentStop expectation flips),
  `tests/test_hooks_parked_marker.py` (new), `tests/test_cli_watch.py`
  (`test_watch_preexisting_waiting_marker_is_not_a_new_edge` becomes the
  `--no-parked` case), `tests/test_cli_watch_parked.py` (new),
  `tests/test_tool_descriptions.py`, `tests/test_watcher_contract.py` if it
  enumerates session files
- docs listed in C; `docs/features/watch-parked-agents/*`

## Risks

- **Two same-reader watchers** → possible duplicate delivery (at-least-once,
  documented). No lost ack entries (locked r/m/w).
- **Legacy markers** (no `gen`) are fingerprinted by bytes; a rewrite with
  identical bytes is indistinguishable from no rewrite — acceptable, `ts`
  changes on every real write.
- **Old installs still emitting `SubagentStop` markers** → filtered as today.
- **Windows**: the ack lock reuses the registry's lock model; the subdirectory
  path is built with `Path`. Not run on Windows here; flagged in the PR.

## Test cases (red first)

hooks:
1. `SubagentStop` with a parked `Stop` marker → bytes untouched.
2. `SubagentStop` with a `running` marker → bytes untouched (no downgrade,
   no false `waiting`).
3. `SubagentStop` with no marker → no file created.
4. `Stop` writes `gen` (32 hex chars); two `Stop`s → different `gen`.
5. `UserPromptSubmit` over `Stop` → `running` (only SubagentStop is inert).

watch (settle 0 unless stated; every assertion checks exact JSON / exit / stdout):
6. Parked `Stop` marker before start → exit 0, `reason: waiting`, ack file
   holds that marker's `gen`.
7. Same again → exit 2, empty stdout (acked).
8. Marker rewritten by a new `Stop` (new `gen`) → wakes again; ack updated.
9. Legacy marker without `gen` → wakes; ack stores the byte fingerprint;
   rerun → no wake; rewrite with a different `ts` → wakes.
10. Legacy `SubagentStop` marker at start → no wake.
11. `--no-parked` + parked marker → exit 2; then a genuine new edge → wakes.
12. Parked marker + pre-existing unread message → `message` wins; ack not
    written.
13. Parked marker + output edge before settle (settle 0.3) → `output` wins.
14. Timeout before settle (settle 1, timeout 0.2) → exit 2, no ack.
15. Corrupt ack file → treated as empty; wakes; file rewritten valid.
16. Ack write failure (unwritable `.watch`) → still exit 0 with the wake;
    stderr mentions the ack.
17. Edge-triggered wait is acked → immediate re-run does not re-fire.
18. Two readers (`--reader a`, `--reader b`) each wake once for one park.
19. Ambient `AGENT_NAME=child` → ack at `.watch/ack-child.json`; unsafe
    `AGENT_NAME` → exit 1.
20. `--pattern "*"`: an ack write by another reader is not an `output` wake.
21. Two parked markers → first run wakes for one and acks only it; second run
    wakes for the other.
22. Tool descriptions for `spawn_agent` and `agent_watch_paths` contain the
    contract words.

## Plan-review round-1 dispositions (`plan-review.md`, REJECTED)

1. BLOCKER, SubagentStop race — ACCEPTED: SubagentStop now writes nothing (A).
2. BLOCKER, ts ordering — ACCEPTED: generation ids + byte fingerprint for legacy (B).
3. MAJOR, ack generation not atomic with wake — ACCEPTED: one parse yields the
   candidate that is both acked and reported; at-least-once, ack before print.
4. MAJOR, concurrent same-reader watchers — ACCEPTED: locked r/m/w (no lost
   entries); duplicate delivery documented as at-least-once; single-watcher
   remains the operating rule. Tests 18, 21.
5. MAJOR, ack file as spurious output — ACCEPTED: ack lives in `.watch/`
   subdirectory, never snapshotted. Test 20.
6. MAJOR, reader validation — ACCEPTED: validated regardless of source. Test 19.
7. MAJOR, compat + failure contract — ACCEPTED: spelled out under B (Unchanged,
   Delivery guarantee). Tests 11, 14, 16.
8. MINOR, kill cleanup — ACCEPTED: cut.
9. MAJOR, interleavings — ACCEPTED: tests 12-21 added.
10. MAJOR, tool docstrings — ACCEPTED: section C; test 22.

## Plan-review round-2 dispositions (`plan-review-2.md`, APPROVED WITH CHANGES)

1. MAJOR, ack-before-print is at-most-once — ACCEPTED: print first, then ack
   (`_emit_parked_wake`). Plan text corrected above.
2. MAJOR, same-reader merge untested — ACCEPTED: `TestAckHelper::
   test_concurrent_same_reader_acks_are_merged` (8 threads on a barrier, one
   file, all entries present, no temp files).
3. MINOR, atomic replace + single buffer — ACCEPTED: stated in the plan and
   implemented; `test_failed_replace_leaves_old_ack_and_no_temp`,
   `test_temp_name_is_unique_per_process_and_call`.
4. MINOR, "once" overpromises — ACCEPTED: tool text says an acknowledged
   generation is suppressed and delivery is at-least-once; pinned in
   `test_watch_contract_documents_parked_marker_delivery`.
5. MINOR, lock timeout — ACCEPTED: `_WATCH_ACK_LOCK_TIMEOUT_SECONDS = 5`,
   `FileLockTimeoutError` folded into the non-fatal ack-failure path. POSIX
   `flock` remains blocking (repo lock model); the watch deadline does not
   bound post-wake ack work — documented here, not claimed otherwise.
6. NIT — noted.
