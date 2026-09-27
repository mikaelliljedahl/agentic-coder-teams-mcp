# Windows Terminal command-line hardening — implementation

Plan: `plan.md` rev 2. Codex approved it in round 2 of `plan-review.md`. This
feature ships in the same PR as `powershell-quote-hardening` (PR #74).

## Final design (`backends/process_manager.py`)

- **Module helpers.** These sit next to `_powershell_quote`, under a comment
  block that lists the four wt stages with source references:
  - `_wt_escape_delimiters(token)`: rewrites `;` as `\;`. This is exact under
    `Commandline::AddArg`.
  - `_wt_child_arg(arg)`: takes `list2cmdline([arg])`, strips its outer
    quotes when `arg` has a space (wt adds them back), then applies
    `_wt_escape_delimiters`.
  - `_wt_argv(wt, options, child)`: the single builder for every wt launch.
    It escapes the options, adds `--`, and encodes the child argv. The `wt`
    path itself is left untouched.
  - `_wt_may_expand(values)`: true when the values contain 2 or more `%` in
    total, since `ExpandEnvironmentStringsW` needs a `%name%` pair.
  - `_powershell_encoded(script)`: Base64 of the UTF-16LE script.
- **Tail.** `_open_windows_terminal_tail` now runs
  `powershell -NoExit -EncodedCommand <payload>`. The payload contains the
  same script as before, still with `_powershell_quote(log_path)`, but the log
  path never appears on wt's command line.
- **Tab spawn.** `_spawn_in_terminal_tab` changes in three ways:
  - **Codex direct launch.** It is used only when
    `_codex_direct_launch_blocker(cmd, cwd)` returns `None`. That function
    returns a reason when `cmd[0]` is a `.bat`/`.cmd` file, or when the argv
    or cwd contains a `%` pair. When there is a reason, it logs
    `[wt direct launch skipped] <reason>` and uses the wrapper tab instead.
  - **Wrapper branch.** When the `.launch.ps1` path contains a `%` pair, it
    raises the new `WindowsTerminalTabUnsafeCommandLineError`. The raise
    happens **before** the sidecar is cleared, the wrapper is written or
    `Popen` is called.
  - **Builder.** Both branches now build their argv with `_wt_argv`. The
    static `_escape_wt_passthrough` is removed and replaced by
    `_wt_escape_delimiters`.
- **`spawn_process`.** It now catches `WindowsTerminalTabUnsafeCommandLineError`
  in the same fallback as `WindowsTerminalTabImmediateExitError`: it reopens
  the log and launches in a classic console, with no wt involved. There is no
  double-run risk because nothing had been started. The new exception is a
  plain `RuntimeError`, not a `WindowsTerminalTabSpawnError`.

## Deviations from the plan

- `_wt_may_expand` takes a list of values rather than one joined string. The
  count is the same, and a list makes the cross-argument case explicit.
- The direct-launch check counts `%` over the raw `cmd` rather than the
  encoded command line. Encoding never adds or removes `%`, so the result is
  the same.
- `docs/reference/agent-messaging-protocol.md` does not describe the wt argv,
  so it was not changed. The historical root file
  `PLAN-codex-wt-semicolon-tabs.md` still names `_escape_wt_passthrough`; it
  is a record of an earlier plan and was left as is.

## Red / green evidence

- **Red.** `uv run pytest tests/test_backends/test_wt_command_line.py
  tests/test_backends/test_powershell_quote.py -q` failed before the
  production change:
  - the new module failed to import (`ImportError: cannot import name
    '_wt_argv'`);
  - `test_terminal_tail_quotes_log_path` failed because it now decodes an
    `-EncodedCommand` payload, and production still sent `-Command`.
- **Green.** The same command gives **162 passed, 1 skipped**.
- **Mutation checks.** Each helper was reverted to its old behaviour inside
  the test process, and the new module fails in every case (exit 1):
  - `_wt_child_arg` escaping only `;`: the quote cases fail;
  - `_wt_escape_delimiters` as the identity: the `;` cases fail;
  - `_wt_may_expand` always false: the `%` routing cases fail.
- **Negative-control tests.** These are in the suite and prove the models can
  see each bug:
  - an unescaped `-d C:\p;calc.exe` gives 2 sub-commands, the second starting
    with `calc.exe`;
  - `;`-only passthrough corrupts a prompt containing `"`;
  - an expanded variable that contains `"` re-splits the child argv;
  - an expanded variable that contains `'` breaks the old `-Command` tail
    literal.

## Validation (Linux, all four CI gates, whole repo)

```bash
uv run ruff format --check .   # 97 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 2047 passed, 0 failed, 7 skipped (no pwsh)
# with pwsh 7.6.6 on PATH:     # 2048 passed, 0 failed, 6 skipped
```

## Manual Windows verification

`docs/features/wt-semicolon-hardening/verify_windows_wt.py` has to be run by
hand on Windows with Windows Terminal installed:

```powershell
uv run python docs/features/wt-semicolon-hardening/verify_windows_wt.py
```

Every result is labelled with the boundary it exercised. For the design, see
the script's docstring and the plan.

| Label | What it runs |
|---|---|
| `wt->native(py)` | hostile `-d` directories and argv combinations through real wt to a Python recorder |
| `wt->native(rust)` | the same, to a Rust recorder, when `rustc` is on PATH |
| `wt->ps-file` | the real tab spawn under a hostile log dir and cwd, with a PowerShell stub as the agent |
| `console-fallback` | a `%` pair in the log dir, which must lead to a classic-console launch |
| `wt->tail` | the real tail script, with `Get-Content` stubbed to record the path it receives |
| `inject` | a positive control where the raw token must run `mk.cmd`, and a protected run where it must not |
| `expand` | a proof that the expansion stage exists, plus the guard; needs an isolated Terminal process |

More about how it runs:

- Cases run in both fresh-window and existing-window (anchor tab) modes.
- It records the resolved `wt.exe` path, the `WindowsTerminal.exe` file
  version, the PowerShell version and the Python version.
- `expand` and the `-w 0` tail case need an isolated Terminal process, so
  they only run when no Terminal is running at start. Otherwise they are
  reported as NOT EXECUTED.

**Coverage limits:**

- PowerShell `-File` runs under `powershell.exe` 5.1 only. Production
  hard-codes `powershell`, so the plan's "and `pwsh` if installed" does not
  apply.
- Only the native round trip runs in both fresh-window and existing-window
  modes. The injection, wrapper, fallback and tail cases run in the
  existing (anchor) window only.
- The Rust recorder checks argv, not cwd.
- A user profile with `elevate: true` sends the request through a second,
  unmodelled wt parse (see "Remaining separate boundaries").
- `wt->ps-file` covers the wrapper path and PowerShell literal binding. It
  does not cover Windows PowerShell's later native-argv marshalling, which is
  a pre-existing, separate boundary.
- A run without `rustc` only has Python native-consumer evidence.

**Status:** run on Windows on 2026-09-27, after merging `origin/main`
(18be2a1): **18 passed, 0 failed, 1 not executed.** Exit code 0.

- Environment: Windows 11 Pro 10.0.26200; `wt.exe` at
  `%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe`; `WindowsTerminal.exe`
  1.24.2607.10001 (package 1.24.11911.0); PowerShell 5.1.26100.7920;
  Python 3.12.14.
- Isolated Terminal process: True. So `expand` and the `-w 0` tail case
  ran.
- PASS: `wt->native(py)`, all 5 argv sets in both fresh-window and
  existing-window modes.
- PASS: `inject`. The positive control ran `mk.cmd`, and an encoded `;`
  stayed data.
- PASS: `wt->ps-file`. Real tab spawn with a hostile log dir, using the
  args `a;b`, `say "hi"`, `x'’y` and `tab\tz`.
- PASS: `console-fallback`, a `%` pair in the log dir.
- PASS: `wt->tail`, in a named window and with `-w 0`. The path was exact,
  and `%USERNAME%` was not expanded.
- PASS: `expand`. wt expanded `%WTV_PROBE%` and re-split argv into
  `['x a', 'b y']`, and the direct-launch guard refused the `%` pair.
- NOT EXECUTED in that first run: `wt->native(rust)`, because `rustc` was
  not on PATH.

Rerun on the same day with rustc 1.98.1 (toolchain
`stable-x86_64-pc-windows-gnu`, installed via rustup) on PATH: **28 passed,
0 failed, 0 not executed.** Exit code 0.

- The isolated Terminal process and the Rust recorder were both present.
- `wt->native(rust)` passed for all 5 argv sets in both fresh-window and
  existing-window modes. The Rust recorder's argv matched the Python
  recorder's exactly.
- All the other cases passed again.

## Remaining separate boundaries (not changed)

These are listed in the plan and need their own consumer analysis:

- CMD/batch:
  - `hooks.write_codex_launcher`;
  - `hooks._shell_quote_command`;
  - the codex/pi npm-shim fallbacks (the codex shim can no longer reach wt
    through direct launch);
  - `native_wake.CodexMemberWake._queue`.
- `ProcessBackend.execute_in_pane` (`cmd /c`, which is by design).
- The TOML `commandWindows` hook.
- wt profile elevation (`elevate: true`):
  - `NewTerminalArgs::ToCommandline` → `elevate-shim.exe` re-serialises the
    request for a second wt parse, which the encoding does not model;
  - it needs UAC consent and an unusual setting;
  - it was found in the implementation review (finding 1).

## Implementation-review dispositions

The Opus review is `implementation-review.md`, verdict APPROVED.

1. **minor**, the elevation re-serialisation path — **documented** as a
   separate, unmodelled boundary in the plan and here. No code change.
2. **minor**, the verifier's route oracles — **fixed**:
   - `wt->ps-file` requires `[wt command]` and no `[fallback]`;
   - `console-fallback` requires `[fallback]` and the `'%' pair` reason, and
     no `[wt command]`.
3. **minor/nit**, the expansion probe only proved the rewrite — **fixed**:
   the argument is now `x %WTV_PROBE% y`, and the probe asserts the re-split
   `["x a", "b y"]`.
4. **nit**, verifier claims that were not exercised — **documented** under
   coverage limits.
5. **nit**, the .NET wording — **fixed** in the plan's "Supported domain".
6. **nit**, the cwd `%` check wording — **fixed**. The docstring now calls it
   defensive, it has its own skip reason, and the call is simplified to
   `_wt_may_expand(cmd)`.
7. **nit**, the unclosed log handle in a test — **fixed** with a `with` block.
