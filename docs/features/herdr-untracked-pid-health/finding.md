# Finding: `HerdrProcessManager` reports a foreign PID alive where the other managers report it dead

Surfaced while validating the `foot-terminal` change; **not fixed there** —
it is a pre-existing behaviour difference introduced by PR #58 (the Herdr
launcher), it touches liveness semantics shared with the kill paths, and it is
unrelated to terminal discovery. Sibling of
`docs/features/ownership-indeterminate-kill/finding.md`.

## What differs

For a handle this manager does **not** track, with no `expected_token`, all
three POSIX managers fall through to the same
`_PidOwnershipMixin._pid_health_with_token`, which ends in `self._pid_alive(handle)`.
The overrides disagree:

```
LinuxTerminalProcessManager  health_check("123") -> (False, 'process not found')
TmuxProcessManager           health_check("123") -> (False, 'process not found')
HerdrProcessManager          health_check("123") -> (True,  'process exists by pid')
```

(PID 123 on this machine is `kcompactd0`, a live root-owned kernel thread.)

The cause is the `_pid_alive` override each class uses:

- `LinuxTerminalProcessManager._pid_alive` / `TmuxProcessManager._pid_alive`
  call `os.kill(pid, 0)` and treat **any** `OSError` as "not alive". For a
  process owned by another user that raises `EPERM`, so an existing foreign
  process reads as dead.
- `HerdrProcessManager._pid_alive` delegates to the module-level `_pid_is_live`,
  which is `/proc`-based and zombie-aware, so it reports the same process alive.

Neither is dangerous for destructive operations — those gate on `owns_process`
and the creation token, and a token turns both into `pid reused (token
mismatch)`. The divergence only affects the token-less display/liveness path
kept for records that predate tokens.

## Why it matters

`follow_up_agent` on a record whose PID has since been reused by a *foreign*
process returns `agent_busy` under the Herdr launcher, where the other
launchers correctly conclude the old child is gone and proceed. The agent is
then unrecoverable through follow-up until the record is killed.

Narrow, but it is a real behavioural difference between launchers that no test
pins.

## Evidence

`tests/test_follow_up_delivery.py::test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`
fails **only** when `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` is set in the
environment:

```
AssertionError: assert 'agent_busy' == 'resume_not_confirmed'
```

The test hardcodes `pid: 123` in its agent record. CI never sets the launcher
env, so CI is green and this is invisible there.

## Options

1. **Settle on one liveness definition.** Make the EPERM case mean "exists" in
   all three managers (`_pid_is_live` everywhere). Most correct, but it changes
   what "alive" means on the shared path used by kill/health decisions, so it
   needs its own plan, review and test pass.
2. **Make the token non-optional** for records written by current code, so the
   token-less fallback stops mattering. Larger change; overlaps the
   `ownership-indeterminate-kill` finding.
3. **Accept and document**, adding a test that pins each manager's answer so the
   difference is at least deliberate.

Independently of which is chosen, the test above should not hardcode a PID that
can be live on the host — that is a test-portability bug of the same family as
the Windows `shlex.quote` fix in PR #58.

## Recommendation

Own plan, own review, own PR — together with
`ownership-indeterminate-kill`, since both are about what this repo is willing
to conclude from an unproven PID.
