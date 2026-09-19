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
2. **Post-resume child liveness.** `exited_pid` is a genuinely exited PID, and
   `confirm_delivery(child_alive=lambda: process_manager.health_check(str(new_pid))[0])`
   passes **no** `expected_token`, so a PID reused by a live process between
   `proc.wait()` and the probe reads alive → `scan_expired`/`unconfirmed`
   instead of `resume_not_confirmed`. Not the reason seen, but the same class
   of flake and reachable on the same busy host.

## Proposed design

Keep the real process exit (the docstring's intent), pin the two probes:

- Change the `exited_pid` fixture to capture the child's `creation_token`
  **while it is still alive**, and return `(pid, token)` — or keep returning
  the pid and expose the token via a small `SimpleNamespace`. The token is read
  from `/proc/<pid>` start time on Linux, so it must be read before `wait()`.
- In the test, install a `health_check` wrapper on
  `server_simple.process_manager` that:
  - returns `(False, "dead")` for the stale record handle `"123"` (matching
    `_dead_agent`, the convention the rest of the file already uses), and
  - delegates to the **real** `process_manager.health_check(handle,
    expected_token=<captured token>)` for the exited child handle.

  With the token pinned, a reused PID reports dead (`pid reused (token
  mismatch)`), and the real `poll`/exit transition still drives the result.

Fallback if the token cannot be read on some platform (`creation_token`
returns `None` before `wait()`): the delegate then behaves as today's bare PID
liveness; to stay deterministic we assert the token was captured in the
fixture (`pytest.skip` is not needed — token read is supported on both CI
platforms).

## Files affected

- `tests/test_follow_up_delivery.py` (fixture + one test)

## Risks

- Over-stubbing would turn the test into a mock-driven one, losing the A3
  signal. Mitigated by delegating to the real `health_check` for the child.
- The `"123"` sentinel is duplicated; kept consistent with `_dead_agent`.

## Test cases

- The hardened test still passes and still fails if `confirm_delivery`'s
  settle-window branch is broken.
- Repeat runs (`-n` repeats / `--count`-style loop by rerunning pytest) stay
  green.
- Simulated reuse: with the token pinned, a live PID substituted for the
  exited one still yields `resume_not_confirmed`.
