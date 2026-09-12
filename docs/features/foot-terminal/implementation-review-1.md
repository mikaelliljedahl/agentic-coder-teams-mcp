# Implementation review 1 — codex (gpt-5.6-sol)

Independent post-implementation reviewer, opposite model family to the
implementer (Claude). Reviewed the uncommitted diff against `plan.md`
(revision 2) and `plan-review-1.md`.

> Review result: the one-line production change is correct and complete for the
> stated goal. It inserts `foot` at the approved boundary, affects only
> `LinuxTerminalProcessManager._discover_terminal`, intentionally displaces only
> `xterm`, and no other tuple consumer exists. Local `foot 1.28.0`/`footclient`
> help confirms the documented `-T`, ignored `-e`, server requirement, and
> footclient default-wait behavior. The normal internal `spawn_agent` path does
> not write a state marker; the lifecycle hook does. No additional functional
> work is required.

Blocker findings: none. Major findings: none. Nit findings: none.

1. [minor] `README.md:264-267` — Disposition 2 was not fully carried into the
   README. "a Wayland-only session that still has an X11-only terminal
   installed, that terminal is picked" is broader than the actual ordered scan:
   only an *earlier* candidate can mask `foot`; `xterm`, for example, is
   X11-only but now loses to `foot`. "the spawn fails" is also too categorical
   because `spawn_agent` may initially return a launcher PID before the unusable
   terminal exits. Recommendation: say "if an earlier X11-only candidate is
   installed, discovery may select it and the terminal launch can then fail
   without DISPLAY," matching plan.md's known limitation.

2. [minor] `implementation.md:69-72`; `test_base_runtime.py:1549-1560,1587-1619`
   — Disposition 3 is claimed "accepted in full," but the tests/documentation
   overstate the evidence. The preference guard samples only `xfce4-terminal`,
   so it does not itself establish "nothing above xterm changes." The spawn test
   claims the "full Popen argv" but asserts only a six-item prefix plus
   substrings in the last item; extra/misplaced arguments could pass. Because
   `Popen` is mocked, it also proves only that PID-file/exec text is
   constructed, not that a sidecar is written or `exec` occurs. This is not a
   CLI-assumption correctness hole — the exact argv characterization plus
   recorded live probe cover that separately — but the stated test guarantees
   are inaccurate.

3. [minor] `implementation.md:74-83` — The listed exact command `uv run pytest`
   is not green in this configured worktree: reproduced `2 failed, 1497 passed,
   4 skipped` because the environment carries
   `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`, triggering two unrelated
   environment-sensitive failures. Running with that variable unset gives the
   claimed result. Recommendation: record the command actually needed for the
   claimed whole-repo result.

## Disposition (implementer)

1. **Accepted.** README reworded to "if an **earlier** X11-only candidate is
   installed, discovery may select it and the terminal launch can then fail
   without `DISPLAY`."
2. **Accepted in full, and the overclaims were real.**
   `test_prefers_established_terminal_over_foot` is now parameterized over every
   candidate ahead of `foot`, so it actually establishes what it says.
   `test_spawn_through_foot_*` now asserts the **complete** argv rather than a
   prefix, is renamed to `..._builds_full_argv_and_shell_command`, and its
   docstring says plainly that a mocked `Popen` establishes construction only —
   execution is established by the recorded live probe.
3. **Accepted, and it found a genuine defect of mine.** Investigated rather than
   just re-worded:
   - `TestPlatformProcessManagerSelection` did not know about the `herdr`
     launcher — a real omission from PR #58. Fixed here, own commit.
   - The second failure is **not** cosmetic. `HerdrProcessManager` reports a
     foreign PID alive where `LinuxTerminalProcessManager` and
     `TmuxProcessManager` report it dead, because their `_pid_alive` overrides
     disagree about `EPERM`. Written up in
     `docs/features/herdr-untracked-pid-health/finding.md`, surfaced to the
     user, and deliberately left for its own PR — it changes liveness semantics
     shared with the kill paths.
   - `implementation.md` now states the gate results honestly, including what
     still fails under `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` and why CI is
     unaffected.
