# Implementation: a parked agent must be able to wake a coordinator

## Red

- `tests/test_hooks_parked_marker.py` (7 cases) and
  `tests/test_cli_watch_parked.py` (21 cases) were written against the v2 plan
  before any production change: `5 failed` (hooks) and `16 failed` (watch) —
  the passing ones asserted behavior that already held (`--no-parked`-like
  edge wakes, legacy `SubagentStop` filtering, message priority, timeout
  shape, unsafe-reader handling for the explicit flag).
- Three tests added after plan-review round 2 (`TestAckHelper`) were red for
  the concurrency/atomicity contract before `_acknowledge` existed.

## Green

Focused: `tests/test_hooks_parked_marker.py tests/test_cli_watch_parked.py
tests/test_cli_watch.py tests/test_hooks.py` → `122 passed, 2 skipped`.

Existing expectations that changed on purpose:
- `tests/test_hooks.py::test_emit_maps_subagentstop_to_waiting` →
  `test_emit_subagentstop_writes_nothing`; four hook-wiring enumerations no
  longer list `SubagentStop` (it is not wired any more).
- `tests/test_cli_watch.py::test_watch_preexisting_waiting_marker_is_not_a_new_edge`
  → `test_watch_no_parked_treats_preexisting_waiting_marker_as_no_edge`.

## Final design

- `hooks.py`: `_WAITING_EVENTS = {"Stop"}`; `SubagentStop` is unrecognized →
  nothing written, and it is no longer wired into the generated hook
  settings. Every marker gains `"gen": uuid4().hex`.
- `cli.py`: `_Parked` record from one `read_bytes` (`gen` field or SHA-256 of
  the bytes); `.watch/ack-<reader>.json` via `_read_acked`/`_acknowledge`
  (locked with `filelock.file_lock`, 5 s timeout, unique temp + atomic
  replace, temp always unlinked); `--parked/--no-parked` (default on) seeds
  `pending_waits` at start with unacknowledged parked markers; every settled
  wait goes through `_emit_parked_wake`: **print, then ack**, ack failure →
  stderr warning, exit stays 0. The resolved reader is validated whatever its
  source (`--reader`, `AGENT_NAME`, default).
- `server_simple.py`: docstring text only (`_DISK_CONTRACT_NOTE`, `spawn_agent`).

## Deviations from the plan

- Plan v2 first said ack-before-print; changed to print-then-ack per
  plan-review round 2 (at-least-once). Plan text updated to match.
- `SubagentStop` hook wiring removed entirely (the plan only said the emitter
  ignores it); wiring an inert hook would cost a process spawn per event.

## Contract surfaces updated

`_DISK_CONTRACT_NOTE` + `spawn_agent` docstring (pinned by
`test_watch_contract_documents_parked_marker_delivery`),
`docs/reference/agent-messaging-protocol.md` (actionable-edge rule, new
"Parked markers and the per-reader acknowledgement" section, marker table,
session-file table, exit codes), `.claude/skills/agent-orchestration/SKILL.md`,
`README.md`.

## Validation

```bash
uv run ruff format --check .   # 84 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1553 passed, 4 skipped (Linux)
```

Not run on Windows. The lock path with a bounded timeout (`msvcrt`) is
exercised only by the POSIX `flock` branch here; the ack-permission test is
skipped on Windows and as root.

## Manual smoke (to do before merge)

1. Spawn a `claude-code` agent, let it finish its turn (marker `waiting/Stop`
   with a `gen`), and only then run `watch_command_bash` → expect a
   `reason="waiting"` wake after the settle window and
   `.watch/ack-team-lead.json` naming that `gen`.
2. Run the same watch again → expect exit 2 after the timeout.
3. `follow_up_agent` the agent, let it park again → the next watch wakes.

## Post-implementation review dispositions (`implementation-review.md`, REJECTED)

1. MAJOR, acked generation re-delivered on a later edge — FIXED. `acked` is
   loaded once per invocation and consulted through `_unacked()` at start-up
   seeding, at edge registration and at settle. `--no-parked` deliberately
   bypasses it (raw edge-only compatibility mode) — documented in the flag
   help and docstring. Tests: `TestAckedGenerationSuppression` (touch with the
   same `gen` → timeout; byte-identical legacy rewrite → timeout; new `gen`
   edge → wake; `--no-parked` → wake).
2. MAJOR, full gate red (`tests/test_follow_up_delivery.py::
   test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`,
   `agent_busy` instead of `resume_not_confirmed`) — NOT REPRODUCIBLE here:
   8/8 green on this branch and 3/3 on `main`, plus green in every full run
   (`1553 passed`). The test spawns a real short-lived process and reads its
   PID back; on a host where many agents are being spawned (the reviewer's
   Codex session) that PID can be reused by a live process before the check,
   which reads as `agent_busy`. It is unrelated to this diff (the
   `server_simple.py` change is docstring-only) and pre-exists this branch.
   Surfaced as a separate follow-up rather than swallowed.
3. MINOR, contract text — FIXED: `gen` in the disk-contract schema; flag help
   and `_DISK_CONTRACT_NOTE` now say at-least-once, not exactly-once; the
   exit-2 explanation lists the three real causes; test-module docstring says
   print-then-ack; the description test pins the qualified sentence.
4. MINOR, vacuous `.watch/` test + skipped ordering coverage — FIXED: the
   broad-pattern test now runs with nothing actionable and asserts a timeout
   after the foreign ack write; `TestEmitParkedWakeOrdering` pins print→ack
   order, exit 0 + stderr warning on `OSError`/lock timeout, and
   `FileLockTimeoutError → OSError` inside `_acknowledge` — all
   platform-independent via monkeypatch.
5. MINOR, Windows — the simulated lock-timeout test from (4) is added; a real
   Windows run is still outstanding and flagged in the PR.

Final gates after the fixes: format/lint/ty clean, `1553 passed, 4 skipped`.
