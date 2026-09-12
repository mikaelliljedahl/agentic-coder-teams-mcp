# Discover `foot` in the Linux terminal launcher

*Revision 2 — incorporates `plan-review-1.md` (codex).*

## Scope

Add the `foot` terminal emulator to `LinuxTerminalProcessManager._discover_terminal`'s
auto-discovery list so the default Linux launcher works on a machine where none
of the currently probed terminals is installed but `foot` is — stock
Omarchy/Hyprland being the case that prompted this.

Out of scope: other modern emulators (`alacritty`, `kitty`, `ghostty`,
`wezterm`), session-aware discovery (see "Known limitation"), the `herdr` and
`tmux` launchers, and any change to `_terminal_command`'s dispatch structure.

## Current behavior

`_discover_terminal` (`src/claude_teams/backends/process_manager.py:2047`)
honours `WIN_AGENT_TEAMS_LINUX_TERMINAL` first, then probes, in order:

```
qterminal, gnome-terminal, x-terminal-emulator, xfce4-terminal,
konsole, mate-terminal, lxterminal, xterm
```

The tuple is a plain first-match scan on `shutil.which`; only `qterminal` has
extra handling (skipped when an instance is already running, because it hands
launches to it over D-Bus). None of these exist on a stock Omarchy install, so
`spawn_agent` fails with `Could not find a supported terminal emulator on PATH.`
even though `foot` is present and works. The env override is the only way
through today — that is what this machine is currently configured with.

## Proposed design

1. Insert `"foot"` into the candidate tuple **immediately before `"xterm"`**.

   This is not a no-op: a host with both `xterm` and `foot` installed switches
   from `xterm` to `foot`. That displacement is **intentional**. The tuple is
   ordered by desirability and `xterm` sits last as the deliberate "anything is
   better than nothing" fallback; where `foot` also exists it is the better
   window, and nothing else in the list is affected. A test pins this exact
   boundary.

2. **No `_terminal_command` branch.** The generic fallback
   `[terminal, "-T", title, "-e", "bash", "-lc", shell_command]` is already
   correct for `foot`:
   - `foot --help` (1.28.0) documents `-T,--title=TITLE` and `-e` as
     *"ignored (for compatibility with xterm -e)"*, so `bash -lc <cmd>` is taken
     as the command to run.
   - Verified live on this machine: `foot -T probe -e bash -lc '<script>'` ran
     the script and the `foot` process stayed in the foreground for the
     command's whole lifetime — it does **not** fork to a server and exit.
   - The reviewer independently confirmed the downstream chain: `bash -lc` runs
     the PID-file prefix, `_build_shell_command` ends in `exec` so the recorded
     shell PID becomes the agent PID, and `_agent_pid_health` takes precedence
     over launcher state — so the sidecar/health path behaves exactly as it does
     for `xterm`.

3. Update the README's list of probed terminals to mention `foot`, and document
   `footclient` as a supported opt-in.

### `footclient`

Not auto-discovered, because it requires a running `foot --server` and
discovery cannot assume one exists. It is otherwise fine: `footclient` waits for
its child by default (only `--no-wait` detaches) and accepts the same
`-T … -e command …` form, so `WIN_AGENT_TEAMS_LINUX_TERMINAL=footclient` works
as a deliberate opt-in on a machine that runs the server.

### Known limitation (unchanged by this change)

Discovery is PATH-only and not session-aware. On a Wayland-only session that
nonetheless has an X11-only candidate installed — `lxterminal`, or an
`x-terminal-emulator` alternative pointing at one — that earlier candidate is
still selected and `foot` is never reached; the spawn then fails for want of
`DISPLAY`, exactly as it does today. Making discovery consult
`WAYLAND_DISPLAY`/`DISPLAY` is a real feature with its own risks and is left out
of this change. `WIN_AGENT_TEAMS_LINUX_TERMINAL` remains the escape hatch.

## Files affected

- `src/claude_teams/backends/process_manager.py` — one tuple entry.
- `tests/test_backends/test_base_runtime.py` — new discovery + spawn tests.
- `README.md` — probed-terminal list and the `footclient` note.
- `docs/features/foot-terminal/` — plan, review, implementation notes.

## Risks

- **Displacing `xterm`.** Accepted and tested, see design point 1.
- **Wrong argv silently opening an idle window.** A unit test can only assert
  the argv we believe is right — the exact failure mode that four Herdr CLI
  assumptions hit in PR #58. Mitigated by the live probe above and by an
  end-to-end spawn before the PR, not by the unit tests.

## Test cases

Red first where marked; the rest are characterization/guard tests that pin
decisions this change makes.

1. `test_discovers_foot_when_no_other_terminal_is_installed` — `which` answers
   only for `foot`; discovery returns `/usr/bin/foot`. **Red today** (raises
   `FileNotFoundError`).
2. `test_prefers_foot_over_xterm` — `which` answers for both `foot` and
   `xterm`; discovery returns `foot`. **Red today** (returns `xterm`). This is
   the boundary the placement decision actually moves.
3. `test_prefers_established_terminal_over_foot` — `which` answers for both
   `xfce4-terminal` and `foot`; discovery returns `xfce4-terminal`. Green today;
   guards against inserting `foot` too early.
4. `test_foot_command_uses_generic_title_and_shell_form` — characterization:
   `_terminal_command` for `/usr/bin/foot` yields
   `[..., "-T", title, "-e", "bash", "-lc", cmd]`. This pins the argv the live
   probe validated; it does not itself establish that `foot` executes it.
5. `test_spawn_through_foot_wraps_shell_command_and_pid_sidecar` — spawns with
   `_discover_terminal` stubbed to `foot` and `Popen` mocked, asserting the full
   argv **and** that the wrapped shell command still writes the `.pid` sidecar
   and `exec`s the agent. Covers the plumbing the argv-only test does not.
6. `test_desktop_env_preserves_wayland_keys_without_display` — `_desktop_env`
   keeps `WAYLAND_DISPLAY`/`XDG_RUNTIME_DIR`/DBus when `DISPLAY` is absent.
   Green today; pins that the Wayland path has no hidden X11 requirement.

## Validation

Four gates whole-repo: `ruff format --check .`, `ruff check .`, `ty check`,
`pytest`. Plus the live `foot` probe already run, and one real `spawn_agent`
with `WIN_AGENT_TEAMS_LINUX_TERMINAL` unset to prove discovery picks `foot` end
to end.
