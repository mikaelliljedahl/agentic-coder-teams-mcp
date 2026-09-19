# Implementation: non-interactive children no longer inherit the server's stdin

## Red

`tests/test_subprocess_stdin.py` (new ratchet, AST-based) and
`tests/test_wake_local_settings.py::TestGitIgnoreLayouts::test_git_calls_detach_stdin`
(behavioral) were written before any production change:

```
FAILED tests/test_subprocess_stdin.py::test_every_subprocess_run_names_stdin[codex.py]
FAILED ...[pi.py]  ...[process_base.py]  ...[process_manager.py]
FAILED ...[procinfo.py]  ...[server_simple.py]
FAILED tests/test_wake_local_settings.py::TestGitIgnoreLayouts::test_git_calls_detach_stdin
7 failed, 51 passed
```

## Green

`stdin=subprocess.DEVNULL` added to all 20 literal `subprocess.run` sites and
to the one `Popen` that lacked it (the Windows Terminal log tail) — 21 added
lines in `src/`, each a single keyword. Per file: `server_simple.py` 1,
`procinfo.py` 1, `codex.py` 1, `pi.py` 1, `process_base.py` 2,
`process_manager.py` 15 (14 `run` + the log-tail `Popen`).

Focused: `tests/test_subprocess_stdin.py tests/test_wake_local_settings.py
tests/test_subprocess_decoding.py` → `106 passed`.

Two existing tests asserted an exact kwargs set and were updated to include the
new keyword (the risk the plan predicted):
`test_base_runtime.py::TestBaseBackendWaitIdle::test_polls_after_version_command_when_timeout_is_set`
and `::TestTmuxProcessManager::test_kill_process_uses_tmux_target_when_known`.

## Final design

- Every **non-interactive** child started by the package now gets an explicitly
  detached stdin. No such call site wants the parent's stdin: they are
  fixed-argv utilities (git plumbing, tmux, pgrep, taskkill, PowerShell
  `-Command`, `--version` / model-discovery probes, herdr CLI) or captured
  non-interactive shell runs.
- Two `Popen` paths keep `stdin=None` **deliberately** and are unchanged: the
  Windows `CREATE_NEW_CONSOLE` agent launch and the Linux terminal-emulator
  launcher, where the interactive CLI needs a real console. Forcing `DEVNULL`
  there would break that contract.
- The ratchet (`test_subprocess_stdin.py`) admits only the two shapes that
  prove detachment statically: literal `stdin=subprocess.DEVNULL`, or `input=`
  with a value that is not a literal `None`. `stdin=None`, `input=None`,
  `stdin=0`/`False`/`sys.stdin`, a conditional `stdin=`, and every `**kwargs`
  splat fail — deliberately, including shapes that may be correct at runtime;
  widening means adding a named shape, not loosening the rule. Same stated
  limits as `test_subprocess_decoding.py`: literal `subprocess.run` spelling
  only, `Popen` not covered (its five sites are pinned by their own tests).

## Plan-review dispositions (`plan-review.md`, APPROVED WITH CHANGES)

1. **MAJOR, event-loop claim contradicted by the source — ACCEPTED.**
   `install_lead_wake` already runs `_do_install` through `run_blocking`
   (`asyncio.to_thread`), so the stall is a 10 s latency on one call and a
   wrong `git_ignore`, not a frozen loop. Plan corrected; the `to_thread`
   follow-up dropped and replaced with a narrower one (unbounded
   `subprocess.run` calls still pin a worker thread).
2. **MAJOR, the log-tail `Popen` had no mutation-sensitive test — FIXED.**
   `TestWindowsTerminalTail::test_opens_windows_terminal_by_default_when_available`
   now asserts `stdin == subprocess.DEVNULL`.
3. **MAJOR, the ratchet checked spelling, not detachment — FIXED** (further
   tightened after the implementation review, below).
4. **MINOR, inventory overstated — FIXED.** 20 literal `subprocess.run` sites
   (14 in `process_manager.py`), not 22/16.
5. **MINOR, blanket "no child wants stdin" claim — FIXED.** The two
   console-launch `Popen` paths are excluded explicitly in the plan, the risk
   section and the ratchet's docstring.

## Implementation-review dispositions (`implementation-review.md`, CHANGES REQUIRED)

1. **MAJOR, ratchet still permitted inherited stdin — FIXED.** "Not the literal
   `None`" accepted `input=None`, `stdin=0`, `stdin=False`, `stdin=sys.stdin`
   and `stdin=PIPE if flag else None`. `_detaches_stdin` now requires the exact
   `stdin=subprocess.DEVNULL` attribute shape or a non-`None` `input=`. Six new
   mutation self-tests cover exactly those shapes; the deliberate false
   positives (`**kwargs`, including a literal dict) are stated in the docstring
   with the reason.
2. **MAJOR, `implementation.md` inaccurate — FIXED.** Counts (20 `run` + 1
   `Popen`; 14 `run` in `process_manager.py`), the final-design statement
   (non-interactive children, with the console exceptions named), the splat
   policy and every test total below are re-derived from the current tree.
3. **MINOR, rejected event-loop claim still in two test narratives — FIXED**
   (`test_subprocess_stdin.py` docstring, `test_git_calls_detach_stdin`
   docstring).
4. **MINOR, plan's test-case summary described the superseded ratchet — FIXED**
   (`plan.md` scope item 3 and test case 2 now match the implemented rule; the
   log-tail assertion is listed as its own case).

The reviewer noted the manual smoke was supplied evidence, not independently
reproduced. The driver is `smoke3.py` (session scratchpad, not committed); it
needs a real stdio server and a scratch git repo, which is why the check is
manual — see the follow-up on a Windows CI job.

## Deviations from the plan

None beyond the dispositions above.

## Validation

```bash
uv run ruff format --check .   # 86 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # 2 diagnostics, both present on main
uv run pytest                  # 1677 passed, 3 skipped (Windows)
```

`ty` reports the same two pre-existing diagnostics on `main`
(`ProcessInfo.pane_id`, `BaseContext.Process`), in files this branch does not
touch; the branch adds none. Linux via CI.

## Manual smoke (the original repro)

Real stdio MCP server, `spawn_agent` → `install_lead_wake()` in a scratch repo
whose path contains a space:

| | before | after |
|---|---|---|
| duration | 10.5 s | 0.6 s |
| `git_ignore` | `failed` | `already_ignored` |
