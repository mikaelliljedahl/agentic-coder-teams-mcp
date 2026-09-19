# Harden `test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`

## Scope

Make the A3 test in `tests/test_follow_up_delivery.py` deterministic on hosts
that are concurrently spawning many processes. Test-only change; no production
behaviour changes.

## Observed failure

On a busy host the test failed once with `result["reason"] == "agent_busy"`
instead of `resume_not_confirmed`. Passes 8/8 locally, 3/3 on main.

## Current behaviour (why it is host-dependent)

Two unpinned real-OS probes leak into the test:

1. **Pre-existing child liveness.** The `env` record stores `pid: 123` with no
   `create_token`. Unlike every other test in this file, this test installs
   neither `_dead_agent` nor `_child_alive`, so `follow_up_agent` runs the real
   `process_manager.health_check("123")` → `_pid_health_with_token` with no
   token → bare `os.kill(123, 0)`. On this dev host PID 123 is a kernel thread
   and `os.kill` raises `EPERM`, which `_pid_alive` maps to *dead*; on a host
   where PID 123 is a live process the caller may signal (containers, low-PID
   reuse), it reads *alive*, no `waiting` marker exists, so prep returns
   `_FollowUpPrep(wait_reason="agent_busy")` — exactly the observed failure.
2. **Post-resume child liveness (in this test double only).** `exited_pid` is
   a genuinely exited PID, and
   `confirm_delivery(child_alive=lambda: process_manager.health_check(str(new_pid))[0])`
   passes **no** `expected_token`, so a PID reused by a live process between
   `proc.wait()` and the probe reads alive → `scan_expired`/`delivery_unconfirmed`
   instead of `resume_not_confirmed`. This is **not** a demonstrated production
   defect: the built-in backends resume through `process_manager.spawn_process`,
   which tracks the child in `_processes`, so confirmation hits the tracked
   `Popen.poll()`/pane branch before the bare-PID fallback. It is reachable
   here because `_FakeResumeBackend` returns an untracked handle. Test-only
   fix; no production change (see plan-review finding 5).

## Proposed design

Keep a real process termination, pin both probes to *that* process (revised
per plan-review findings 2 and 3):

- Rename the fixture to `exited_child` and have it capture the child's
  `creation_token` **while its liveness is still guaranteed** (the child
  holds a pipe open until the fixture closes it; after exit the token may
  become unreadable, and on Windows can linger only while a process handle is
  retained), returning `SimpleNamespace(pid, token)`. Assert the
  token is non-`None` rather than silently degrading to bare-PID liveness.
- **Repoint the agent record** at that process — `pid` *and* `create_token` —
  before the call. `_agent_alive` then probes with a token, so the ambient
  occupant of an unrelated numeric PID is never consulted; the fixture's own
  real termination decides. No special-casing of the unrelated `"123"`
  sentinel, and no stub on that path at all.
- For the confirmation probe, which calls `health_check` without a token,
  install a narrow wrapper that captures the **original bound**
  `health_check` first (otherwise delegation recurses), supplies the captured
  token when none was passed, and asserts the probed handle is the expected
  one rather than inventing answers for unexpected handles.

With the token pinned, a recycled PID reports dead (`pid reused (token
mismatch)`). The A3 property retained is "a real child was created, really
terminated, and confirmation observed it dead with no receipt" — it exercises
the token-aware OS-PID fallback, not the manager's tracked-`Popen` branch
(the fixture process is not registered in `_processes`). Token parsing and
reuse semantics themselves are covered by `tests/test_pid_reuse.py`.

## Files affected

- `tests/test_follow_up_delivery.py` (fixture, two helpers, one hardened
  test, one new regression test)

## Risks

- Over-stubbing would turn the test into a mock-driven one, losing the A3
  signal. Mitigated by delegating to the real `health_check` for the child.
- All `"123"`-sentinel special-casing is deliberately absent: the record is
  repointed instead, so no probe consults an ambient host PID.

## Test cases

- **Red evidence (scratch, not committed):** with the ambient record PID
  forced alive, the unmodified test yields `agent_busy` — the exact observed
  failure; with the post-resume probe forced alive, it yields
  `delivery_unconfirmed`.
- The hardened test still passes and still asserts the resume was attempted,
  the record's `pid`/`create_token` are unchanged, and no pending-delivery
  field was left behind.
- **Green evidence:** with `creation_token` returning a stranger's token and
  `_pid_alive` forced `True` (a fully recycled PID), the hardened test still
  yields `resume_not_confirmed`.
- **Committed regression test:** `test_a_recycled_pid_cannot_masquerade_as_the_exited_child`
  makes the green evidence durable instead of narrative.
- Repeat runs stay green; the full suite and all four gates are green.
