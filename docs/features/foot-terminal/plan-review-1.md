# Plan review 1 — codex (gpt-5.6-sol, medium)

Reviewer: independent codex agent `foot-plan-reviewer`, opposite model family to
the implementer (Claude). Reviewed plan revision 1 against the worktree source.

Blocker findings: none.

1. [major] `plan.md:30-33`; `process_manager.py:2058-2067` — Putting `foot`
   immediately before `xterm` is safe mechanically (the tuple is a first-match
   scan; only qterminal has special handling), but it does not keep existing
   resolution unchanged: a host with both xterm and foot changes from xterm to
   foot. Recommendation: either append foot after xterm to preserve all
   precedence, or explicitly call the xterm displacement intentional and test
   the exact foot+xterm case.

2. [major] `plan.md:5-7,30-33`; `process_manager.py:2058-2072` — The broad
   "Wayland-only desktops" claim is not guaranteed by late insertion. Discovery
   is PATH-only and can still select an earlier X11-only/unusable candidate
   (notably lxterminal, or an x-terminal-emulator alternative) when DISPLAY is
   absent, then never reach foot. Recommendation: narrow the promise to stock
   Omarchy's stated package set, or make discovery session-aware
   (WAYLAND_DISPLAY/DISPLAY) and add a Wayland-only test with an earlier
   unusable X11 candidate present.

3. [minor] `plan.md:67-78`; `tests/test_backends/test_base_runtime.py:1339-1526`
   — The proposed preference test uses xfce4-terminal, so it stays green even if
   foot is inserted in the wrong place relative to xterm; the command test only
   characterizes list construction and does not establish execution, PID-file
   creation, foreground lifetime, or health behavior. Recommendation: make the
   precedence boundary test use foot+xterm, add a foot-specific spawn test
   asserting the full wrapped shell command/PID sidecar passed to Popen, and
   describe the argv-only test as characterization rather than proof foot
   executes it.

4. [minor] `plan.md:47-48`; `process_manager.py:2079-2126` — The reason given
   for excluding footclient is factually wrong for foot 1.28: footclient waits
   by default until its terminal/child exits (only `--no-wait` detaches),
   accepts the same `-T -e command...` form, and the existing wrapper exports
   agent env and `exec`s while preserving the sidecar PID. Auto-discovery may
   still be undesirable because footclient requires a running server.
   Recommendation: exclude it on that basis and document it as a viable opt-in
   via `WIN_AGENT_TEAMS_LINUX_TERMINAL`.

5. [nit] `process_manager.py:78-83,2128-2135` — No X11 hard requirement exists
   here: missing DISPLAY merely prevents the early return and causes a
   parent-environment recovery attempt; WAYLAND_DISPLAY is preserved/copied
   independently. Functionally fine on Wayland, though the parent read happens
   on every spawn because `all(...)` can never be true without DISPLAY.
   Recommendation: no feature-blocking change; optionally add a focused test
   showing WAYLAND_DISPLAY/XDG_RUNTIME_DIR/DBus survive with DISPLAY absent.

Confirmed, no finding: `plan.md:34-44` is correct for standalone foot.
`bash -lc` runs the PID-file prefix, `_build_shell_command` ends in `exec`, so
the recorded shell PID becomes the agent PID; `_agent_pid_health` takes
precedence over launcher state, while standalone foot itself waits and returns
the client exit status. README coverage is already in the plan, and the existing
env-override precedence test remains applicable.

## Disposition (implementer)

1. **Accepted, option B.** Keep `foot` immediately before `xterm` and state the
   displacement as intentional: the tuple is ordered by desirability and `xterm`
   is the deliberate last resort, so on a host with both, `foot` is the better
   window. Plan revised to say so instead of claiming no change, and the
   precedence test now uses the exact `foot` + `xterm` pair.
2. **Accepted, narrow the promise.** Session-aware discovery is a real feature
   with its own risks and is out of scope here; the plan and README now claim
   only what is true — `foot` is found when no earlier candidate is installed.
   The X11-only-candidate-wins-without-DISPLAY case is recorded as a known,
   unchanged limitation.
3. **Accepted in full.** Precedence test switched to foot+xterm, a foot spawn
   test asserting the Popen argv and the PID sidecar added, and the argv test
   relabelled as characterization.
4. **Accepted.** The stated reason was wrong. Corrected to the real one —
   `footclient` needs a running `foot --server`, which auto-discovery cannot
   assume — and documented as a supported opt-in.
5. **Accepted as test only.** The `all(...)` shortcut is left alone (cleanup
   outside this change), with a test pinning that the Wayland keys survive when
   DISPLAY is absent.
