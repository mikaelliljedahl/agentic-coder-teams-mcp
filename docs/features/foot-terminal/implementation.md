# Implementation — `foot` in the Linux terminal launcher

Plan: `plan.md` (revision 2). Review: `plan-review-1.md` (codex), all five
findings dispositioned there.

## Final design

One line of production code: `"foot"` added to the candidate tuple in
`LinuxTerminalProcessManager._discover_terminal`, immediately before `"xterm"`.
No `_terminal_command` branch — the generic
`[terminal, "-T", title, "-e", "bash", "-lc", shell_command]` fallback is
correct for `foot`, which documents `-e` as ignored for xterm compatibility.

Deviation from plan revision 1: the claim that this leaves every existing
desktop's resolution unchanged was wrong and was removed. A host with both
`xterm` and `foot` now gets `foot`. That displacement is intentional — the tuple
is ordered by desirability and `xterm` is the deliberate last resort — and is
pinned by `test_prefers_foot_over_xterm`.

## Red/green evidence

Red, before the production change (`pytest -k "foot or wayland"`):

```
FAILED ...::test_discovers_foot_when_no_other_terminal_is_installed
FAILED ...::test_prefers_foot_over_xterm
2 failed, 4 passed
```

Exactly the two cases the plan predicted would be red; the other four are
characterization/guard tests that were green before and after.

Green, after adding `"foot"` to the tuple: `6 passed`.

## Live validation

Unit tests can only assert the argv we believe is right — the failure mode four
Herdr CLI assumptions hit in PR #58 — so the argv and the launch were checked
against the real binary, `foot 1.28.0`.

1. Direct probe: `foot -T probe -e bash -lc '<script>'` ran the script, wrote
   the expected PID, and the `foot` process stayed in the foreground for the
   command's whole lifetime (it does not fork to a server and exit), so the
   terminal-pid health path behaves as it does for `xterm`.
2. End-to-end through `server_simple.spawn_agent`, in-process, with
   `WIN_AGENT_TEAMS_LINUX_LAUNCHER=terminal` and `WIN_AGENT_TEAMS_LINUX_TERMINAL`
   **unset** so discovery had to find `foot` itself:

```
PASS  manager is LinuxTerminalProcessManager  LinuxTerminalProcessManager
PASS  auto-discovery picks foot with no override  /usr/bin/foot
PASS  fresh state marker written by the agent's own hook
      {'state': 'waiting', 'event': 'Stop', 'ts': 1789157570.1912303}
```

`spawn_agent` writes no marker itself — the marker is written by
`claude_teams.hooks emit` running inside the spawned agent's own process — so a
marker newer than the spawn is proof the agent CLI really started inside the
`foot` window. `kill_agent` then cleaned up.

## Tests added

`tests/test_backends/test_base_runtime.py::TestLinuxTerminalProcessManager`:

| Test | What it establishes |
| --- | --- |
| `test_discovers_foot_when_no_other_terminal_is_installed` | The Omarchy case: only `foot` on PATH resolves. |
| `test_prefers_foot_over_xterm` | The boundary this change moves, asserted deliberately. |
| `test_prefers_established_terminal_over_foot` | Parameterized over every candidate ahead of `foot`, so only `xterm` is displaced. |
| `test_foot_command_uses_generic_title_and_shell_form` | Characterization of the argv the live probe validated. |
| `test_spawn_through_foot_builds_full_argv_and_shell_command` | The complete `Popen` argv, and that the wrapped shell command contains the `.pid` sidecar write and the `exec` of the agent. `Popen` is mocked, so this establishes construction only — that it runs is established by the live probe. |
| `test_desktop_env_preserves_wayland_keys_without_display` | `_desktop_env` has no hidden X11 requirement. |

## Validation commands

```
uv run ruff format --check .   # all files formatted
uv run ruff check .            # All checks passed
uv run ty check                # All checks passed
uv run pytest                  # 1504 passed, 4 skipped
```

Whole-repo, all green — in a shell with `WIN_AGENT_TEAMS_LINUX_LAUNCHER` unset,
which is how CI runs.

**With `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr` set in the environment, one test
still fails**, and it is not this change's:
`test_follow_up_delivery.py::test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`
returns `agent_busy` instead of `resume_not_confirmed`, because
`HerdrProcessManager` reports a foreign PID alive where the other managers
report it dead. That is a pre-existing difference from PR #58, written up in
`docs/features/herdr-untracked-pid-health/finding.md` and deliberately left for
its own PR. CI does not set the launcher env, so CI is unaffected.

A second failure under that env — `TestPlatformProcessManagerSelection` not
knowing about the `herdr` launcher — *was* trivially mine from PR #58 and is
fixed here in its own commit.
