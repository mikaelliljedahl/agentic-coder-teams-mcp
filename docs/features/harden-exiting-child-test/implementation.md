# Implementation: harden the A3 exiting-child test

## What changed

`tests/test_follow_up_delivery.py` only. No production change (plan-review
finding 5).

- `exited_pid` → `exited_child`: spawns a child that blocks on stdin, reads
  its `creation_token` **while it is live** (asserting it is readable),
  closes stdin and reaps it. Returns `SimpleNamespace(pid, token)`.
- `_repoint_record_at(child)`: rewrites the stored agent record's `pid` and
  `create_token` to the exited child's, so `_agent_alive` probes
  token-aware and never consults whatever process owns the old `123`
  sentinel on the host.
- `_pin_liveness_to(monkeypatch, child)`: captures the **original bound**
  `health_check` before patching (no recursion), supplies the captured token
  when the confirmation loop probes without one, and asserts the probed
  handle is the expected one.

## Red → green evidence

Scratch tests (not committed), against the *unmodified* test body:

| Forced host condition | Observed reason |
| --- | --- |
| record's numeric PID alive on this host | `agent_busy` ← the reported flake |
| post-resume PID recycled by a live process | `delivery_unconfirmed` |

After the change, with `creation_token` returning a stranger's token *and*
`_pid_alive` forced `True` (a fully recycled PID), the test still yields
`resume_not_confirmed`.

Root cause of the host dependence: Linux `_pid_alive` maps every `OSError`
from `os.kill(pid, 0)` — `EPERM` included — to *dead*, so PID 123 reads dead
on a dev box where it is a kernel thread and alive on a host where the caller
may signal it.

## Deviations from the plan

Adopted plan-review findings 2–5: record repointing instead of special-casing
`"123"`, original-`health_check` capture, unexpected-handle assertion, an
asserted-non-`None` token, and no production change. The plan was revised in
place before coding.

## Validation

```
uv run ruff format --check .   # 85 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1635 passed, 4 skipped
```
