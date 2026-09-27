# Windows Terminal command-line hardening — plan

Revision 2 — incorporates `plan-review.md` (Codex, CHANGES REQUESTED).
Dispositions are at the end.

Ships on the same branch/PR as `powershell-quote-hardening` (PR #74, per the
user). That feature fixed the PowerShell-literal layer; this one fixes the
layer in front of it: how `wt.exe` itself parses the command line we hand it.

## How wt parses its command line (verified from source)

Source: `microsoft/terminal` @ `8c0a234f`,
`src/cascadia/TerminalApp/AppCommandlineArgs.cpp` and `Commandline.cpp`.

1. **Split.** `BuildCommands` runs `_addCommandsForArg` on every argv element
   *after* Windows quote processing (`CommandLineToArgvW`). Each element is
   searched with `_commandDelimiterRegex = ^;|[^\\];`: any `;` not preceded by
   `\` ends the current sub-command and starts a new one
   (`[wt.exe, <rest of element>, <following elements>…]`). **Quoting gives no
   protection** — the quotes are gone before the split.
2. **Unescape.** `Commandline::AddArg` replaces every `\;` with `;` (left to
   right, resuming after the replacement) in **every** arg, including option
   values such as `-d`.
3. **Rebuild the child command line.** `_getNewTerminalArgs` joins the args
   after `--` with single spaces, wrapping an arg in `"…"` **only if it
   contains a space**. Nothing is escaped: an embedded `"`, a tab, an empty
   arg, or trailing backslashes inside the added quotes are emitted raw, and
   the child re-parses the string with the normal Windows rules.
4. **Expand environment variables.** `ConptyConnection::_LaunchAttachedClient`
   (`src/cascadia/TerminalConnection/ConptyConnection.cpp:55`) runs
   `wil::ExpandEnvironmentStringsW` over the **whole rebuilt child command
   line** before `CreateProcessW` (line 166, `lpApplicationName = nullptr`, so
   no `cmd.exe` is involved). There is no escape for `%`. The environment is
   the one of whichever Terminal process handles the request — with the
   single-instance `WindowEmperor` a hand-off to an already-running Terminal
   uses *its* environment, not ours — so the result cannot be predicted from
   our side. Substitution needs a `%name%` pair: a command line with fewer
   than two `%` is never changed.

Forwarding to an existing window (`wt/shim.cpp` → `WindowEmperor.cpp` →
`TerminalApp/Remoting.cpp` at this revision) passes the raw
`GetCommandLineW()` string and re-parses it with `CommandLineToArgvW`; it adds
no second split or unescape pass (confirmed by the reviewer). Older
monarch/peasant builds are not analysed; the manual verifier records the
build it exercised.

**Unmodelled path — profile elevation** (found in implementation review):
when the resolved profile (the "Defaults" profile for a bare command line)
has `elevate: true` and Terminal is not elevated, `TerminalPage` re-serialises
the request with `NewTerminalArgs::ToCommandline` (`"<Commandline>"`, no `;` or
quote escaping) and hands it to `elevate-shim.exe` for a second wt parse. Our
encoding does not cover that second pass. It needs a UAC consent and an
unusual setting; it is recorded as a separate boundary, not fixed here.

Consequences:

- A `;` anywhere in a wt-visible value opens a new sub-command. A fragment
  with no spaces (e.g. `-d C:\p;calc.exe`) becomes `[wt, calc.exe, <our
  remaining argv>…]` — the default `new-tab` — so wt launches it. That is
  command injection, not just a junk tab.
- The vulnerable pattern is an **unescaped** `;` (not preceded by `\`).
- Escaping every `;` as `\;` is **exact**: after escaping, every `;` has one
  inserted `\` immediately before it, so the split regex never matches, and
  `AddArg`'s left-to-right `\;` → `;` removes exactly those inserted
  backslashes (an original `\;` becomes `\\;` → `\;`). Proof in the tests.
- Step 3 means a child arg containing `"` (for example a codex prompt, which
  the lead often writes with quotes) is re-split into several argv elements
  for the child. Today that corrupts the codex prompt on the direct-launch
  path, and it lets prompt text inject extra codex flags (for example `-c`
  config overrides).
- Step 4 means any `%name%` in a child value is silently rewritten, and an
  expansion whose value contains `"` or `'` can re-split argv or break a
  PowerShell literal that `_powershell_quote` had made safe. Encoding cannot
  fix this: the only safe options are to keep the value off the command line
  or to refuse it.

## Current behaviour — wt-visible values

In `backends/process_manager.py`:

| Site | Value | Source | Today |
|------|-------|--------|-------|
| `_spawn_in_terminal_tab`, codex direct launch (`WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1`) | `-d request.cwd` | caller | **unescaped** |
| same | codex argv after `--` | caller prompt, cwd, model… | `;`→`\;` only; `"`/tab/empty not handled |
| `_spawn_in_terminal_tab`, wrapper branch (default, all backends) | `.launch.ps1` path after `-File` | home dir / `WIN_AGENT_TEAMS_LOG_DIR` | **unescaped** |
| `_open_windows_terminal_tail` | whole `-Command` script, including the log path | same | **unescaped** |
| both | `-w wt-team-<team>`, `--title <agent>@<team>` | validated names (`[A-Za-z0-9_-]`) | safe; no `;` possible |

## Decision per value

- **All wt option values** (`-d`, `--title`, `-w`): escape `;` → `\;`. Only
  `-d` can carry a `;` today; the other two go through the same helper so a
  future unvalidated value is safe by construction. (This is delimiter
  escaping, not general CLI11 option validation.) Rejecting `;` was
  rejected: the escape is exact and `;` is a legal directory-name character.
- **All child argv after `--`**: encode each element so wt's step-3 rebuild
  yields the canonical `subprocess.list2cmdline` command line, then escape
  `;` (see design).
- **Tail (`_open_windows_terminal_tail`) — avoid.** Pass the script with
  `powershell -NoExit -EncodedCommand <Base64 UTF-16LE>`. Base64 has no
  `;`, `%`, quote or space, so the log path never reaches wt's parser or the
  expansion stage. The tail is a best-effort convenience and never refuses.
- **Wrapper branch (`-File <.launch.ps1>`) — degrade to the classic
  console.** `-File` stays (it gives `exit 0` / `$PSCommandPath` semantics).
  - A `%` pair can only come from the wrapper path, i.e. the home dir or
    `WIN_AGENT_TEAMS_LOG_DIR`. If the encoded child command line contains two
    or more `%`, `_spawn_in_terminal_tab` raises the new
    `WindowsTerminalTabUnsafeCommandLineError`. It does this **before**
    writing the wrapper or calling `Popen`, so nothing has run.
  - `spawn_process` catches that error the same way it already handles
    `WindowsTerminalTabImmediateExitError`: it logs the reason and launches in
    a new classic console. That path is `Popen` → `list2cmdline` →
    `CreateProcessW` directly, with no wt and no expansion stage.
  - There is no double-run risk, because nothing was started. Refusing to
    spawn at all was rejected: it would break every agent spawn for a user
    whose profile path contains `%…%`.
  - Relocating the wrapper was also rejected: a staging dir is still under
    the user profile.
- **Codex direct launch (opt-in) — only when it is exact; otherwise use the
  default wrapper tab.** The prompt, `-C <cwd>` and `-d <cwd>` all reach
  step 4. Direct launch is used only when:
  - `cmd[0]` is not a `.cmd`/`.bat` (case-insensitive). The npm-shim
    fallback of `CodexBackend.discover_binary` would otherwise hand a batch
    file to `CreateProcessW` and a CMD parser, which this encoder does not
    make safe.
  - Neither the encoded child command line nor `-d` contains two or more
    `%`.

  Otherwise it logs the reason and takes the default wrapper branch, which
  bakes the argv into the `.ps1` and keeps it off the command line. That
  branch is already the default for codex and is known to work. No new
  failure mode is introduced. Routing the prompt through a file was
  considered, but the TUI entrypoint has no such input.
- argv[0] of our own `Popen` (the `wt.exe` path) is **not** encoded: it is
  the executable Python launches.

### Supported domain

The exactness claims cover ordinary child arguments: any `str` without NUL
(NUL cannot cross a Windows command line), valid Unicode, and a total command
line within the 32,767-character `CreateProcess` limit. Child argv[0] must be
a normal executable path (no `"`, no trailing backslash). The encoding
produces `list2cmdline`'s canonical form, which the tested child runtimes
(Python/UCRT, Rust `std`, and PowerShell for the `-File` path) parse back
exactly for ordinary arguments with a normal argv[0]; no claim is made that
they share one parser implementation. PowerShell's `-File`
path goes through native argv decoding only; the tail script is no longer on
the command line.

## Proposed design

In `process_manager.py`, replace the static `_escape_wt_passthrough` with
module functions and one builder used by all three sites:

```python
def _wt_escape_delimiters(token: str) -> str:
    """Escape ``;`` so wt does not split on it (wt turns ``\\;`` back into ``;``)."""
    return token.replace(";", r"\;")

def _wt_child_arg(arg: str) -> str:
    """Encode one child argv element for wt's naive re-join (see plan)."""
    encoded = subprocess.list2cmdline([arg])
    if " " in arg:
        encoded = encoded[1:-1]  # wt re-adds the quotes
    return _wt_escape_delimiters(encoded)

def _wt_argv(wt: str, options: list[str], child: list[str]) -> list[str]:
    """Full wt argv: escaped options, ``--``, encoded child argv."""
    return [
        wt,
        *(_wt_escape_delimiters(o) for o in options),
        "--",
        *(_wt_child_arg(a) for a in child),
    ]
```

- Codex direct launch: `_wt_argv(wt, [*head, "-d", cwd], cmd)`.
- Wrapper branch: `_wt_argv(wt, head, ["powershell", …, "-File",
  str(wrapper_path)])`.
- Tail: `_wt_argv(wt, ["-w", "0", "nt", "--title", title], ["powershell",
  "-NoExit", "-EncodedCommand", _powershell_encoded(script)])`, where
  `_powershell_encoded` is Base64 of the UTF-16LE script. The script still
  uses `_powershell_quote(log_path)` inside.
- `_wt_may_expand(text) -> bool`: `text.count("%") >= 2`.
  - Evaluated on the joined encoded child command line, as wt step 3 will
    produce it, and for direct launch also on `-d`.
  - Direct launch: if it is `True`, or `cmd[0]` is a batch file, fall back to
    the wrapper branch and write a `[wt direct launch skipped] <reason>` log
    line.
  - Wrapper branch: if it is `True`, raise
    `WindowsTerminalTabUnsafeCommandLineError`. The new exception is a plain
    `RuntimeError`, not a `WindowsTerminalTabSpawnError`, because it means
    "nothing launched", not "launched, PID unknown".
  - The check runs before `_write_tab_wrapper`, the sidecar, and `Popen`.
- `spawn_process`: add `WindowsTerminalTabUnsafeCommandLineError` to the
  existing fallback `except`, with the same log reopening and a distinct log
  line.
- Docstrings carry the wt rules with the source reference; the old
  `_escape_wt_passthrough` docstring's claim is kept but corrected (it
  covered only `;`).
- The logged `[wt command]` line is unchanged (it logs the argv as sent).

Codex PID discovery (`_find_codex_pid_by_token`) matches the correlation
token as a substring of codex.exe's command line. The token
(`wat-corr:<id>`) has no quote, backslash, or `;`, so the encoding never
alters it.

## Files affected

- `src/claude_teams/backends/process_manager.py`
- `tests/test_backends/test_wt_command_line.py` (new)
- `docs/reference/agent-messaging-protocol.md` if it describes the Windows
  launch argv (checked during implementation)
- `tests/test_backends/test_base_runtime.py` (update the direct-launch and
  tail assertions that pin the old strings; `test_escape_wt_passthrough_*`
  moves to the new helper)
- `tests/test_backends/test_powershell_quote.py`
  (`test_terminal_tail_quotes_log_path` reads the script token back through
  the wt model instead of indexing the raw argv)
- `docs/features/wt-semicolon-hardening/*`, including
  `verify_windows_wt.py`

## Test cases (red first)

Test-only models, each citing the source above, kept as separate stages:
`_cmdline_to_argv(cmdline)` (the outer `CommandLineToArgvW` rules, including
its argv[0] rule — applied to the **whole** serialized command, argv[0]
included), `_wt_split(argv)` (regex split + `AddArg` unescape),
`_wt_rebuild(child)` (step 3) and `_crt_argv(cmdline)` (modern CRT / Rust
rules for the child, including argv[0]). Pipeline:
`list2cmdline(wt_argv)` → `_cmdline_to_argv` → drop argv[0] → `_wt_split` →
per sub-command, the tail after `--` → `_wt_rebuild` → `_crt_argv`. Step 4 is
modelled by `_expand(cmdline, env)` for the controlled-variable tests.

1. `_wt_escape_delimiters` + `_wt_split` round-trip: `;` at start/end,
   `;;`, `\;`, `\\;`, `\\\;`, trailing `\`, `C:\dir\;x` → exactly one
   sub-command, value unchanged.
2. `_wt_child_arg` pipeline round-trip after a real executable token, for a
   corpus of single values **and** combinations: spaces, tabs, `"`, `\"`,
   backslash runs before quotes and at the end (`a b\`, `a b\\`,
   `a\"b c`), the empty string, `;` mixed with all of those, `'`, smart
   quotes, newlines, a realistic prompt → the child argv comes back exactly.
3. Full `_wt_argv` pipeline with hostile `-d` values (`C:\p;calc.exe`,
   `C:\a b;x`, `C:\p\;q`) → one sub-command, the `-d` value exact, the
   child argv exact.
4. Negative controls that prove the models see each bug: the old unescaped
   `-d C:\p;calc.exe` → 2 sub-commands, the second starting `calc.exe`; the
   old `;`-only passthrough with a `"` in the prompt → a different child argv;
   `_expand` with a controlled var whose value holds `"` → re-split argv, and
   one whose value holds `'` → the old `-Command` tail script no longer scans
   as a single PowerShell literal.
5. `%` policy.
   - `_wt_may_expand`: false for 0 or 1 `%`, true for 2 or more, including a
     pair split across two args (e.g. `50%` and `80%`).
   - The tail's encoded argv contains no `%`, `;`, `"` or space, even for a
     log path containing `%X%`, `;`, `'` and `’`. Decoding the Base64 gives
     the exact script, and its literal still scans back to the path (reusing
     the PowerShell scanner).
6. Call sites, with `Popen` mocked:
   - Codex direct launch with a cwd and prompt containing `;` and `"`: decoded
     through the full model, there is one sub-command, the `-d` value is exact
     and the argv is exact.
   - Codex direct launch with `%A%` in the prompt, and separately with `%A%`
     in the cwd: the wt argv is the **wrapper** form (`powershell … -File`),
     the prompt is inside the `.ps1`, and the skip reason is logged.
   - Codex direct launch with `cmd[0] = …\codex.cmd`: the wrapper form is
     used and the reason is logged.
   - Wrapper branch with `WIN_AGENT_TEAMS_LOG_DIR` in a dir containing `;`
     and a space: the wrapper path is exact after decoding.
   - Wrapper branch with a log dir containing `%A%%B%`:
     `_spawn_in_terminal_tab` raises
     `WindowsTerminalTabUnsafeCommandLineError`, with no wrapper, no sidecar
     and no `Popen` call.
   - Through `spawn_process`, the same case falls back to the classic console
     (`CREATE_NEW_CONSOLE` `Popen` of the agent `cmd`, with no wt) and logs
     the reason.
   - Tail with a log path containing `;`, `%X%`, `'` and `’`: there is one
     sub-command, and `-EncodedCommand` decodes to the exact script.
7. Existing tests keep passing with pinned strings updated (e.g.
   `r"Implement the parser\; run the tests\; report back"` is unchanged).

## Manual Windows verification

`docs/features/wt-semicolon-hardening/verify_windows_wt.py`, run by hand on
Windows with Windows Terminal installed. Everything lives in a private temp
dir; every run uses a unique nonce and its own output file, and a missing
completion artifact is a **failure**, never "protected".

- **Records the environment:** the resolved `wt.exe` path and the version of
  the `WindowsTerminal.exe` it launches (file version, not just the Store
  package), `$PSVersionTable` for each PowerShell, the Python version.
- **Window modes.** Every case runs twice:
  - fresh: a new named window;
  - existing: a named window kept alive by an anchor tab running a
    long-sleeping recorder. The anchor must report ready before any case
    runs, and it is closed at the end.

  The tail's `-w 0` policy is exercised separately, the same way.
- **Native argv round trip.** `_wt_argv(wt, ["-w", name, "nt", "-d", <hostile
  real dir>], [sys.executable, rec.py, out.json, *hostile args])`. The
  recorder writes `sys.argv[2:]` (**all** hostile args), `os.getcwd()` and
  the nonce. The whole list is compared, including count and order.
  - Hostile dirs: `a;b`, `a b;c`, a dir named `;y` (path `…\x\;y`), and a
    dir with a trailing-backslash-sensitive name.
  - Hostile args: combinations of space, tab, `"`, `;` and backslash runs;
    the empty string; a realistic quoted prompt.
  - A Rust recorder is built and run too when `rustc` is on PATH, since
    codex is Rust; the run records it as skipped otherwise.
- **PowerShell `-File` round trip.** The real `_write_tab_wrapper` +
  `_spawn_in_terminal_tab` wt argv is built with `WIN_AGENT_TEAMS_LOG_DIR`
  set to a dir named `l;o g's`. The wrapper runs a PowerShell stub that
  writes its args and nonce. This runs under `powershell.exe` 5.1, and under
  `pwsh` if installed.
- **Tail round trip.** Capture the argv of the real
  `_open_windows_terminal_tail` (with `Popen` monkeypatched) for a log path
  in a dir named `t;a%USERNAME%il'’s`, and decode its `-EncodedCommand`
  script. Prepend a `Get-Content` function that writes its `-LiteralPath`
  argument and the nonce to an output file. Re-encode the script and launch
  it through wt with the same builder, dropping `-NoExit` so the tab closes.
  The recorded path must equal the log path exactly, with `%USERNAME%`
  **not** expanded.
- **Injection probes.**
  1. A child arg `x;cmd.exe`, followed by `/c` and `<mk.cmd>`.
     - The **raw** (old) token must produce the sentinel. The script waits
       for it and fails if it doesn't appear, so a blind oracle is caught.
     - The protected token must deliver the recorder output **and** produce
       no sentinel within the wait.
  2. An expansion probe:
     - With `WTV_PROBE` set to `a" "b` in the environment Terminal is started
       from (fresh-window mode only), the raw rebuilt command line must
       re-split the recorder's argv. This proves step 4 exists.
     - The production helper must report `_wt_may_expand` for the same value,
       so the direct launch takes the wrapper path. The script asserts the
       argv it would send is the wrapper form.

  Each probe uses its own nonce and cleans up before the protected run.

Prints PASS/FAIL per case; exits 0 only if everything passes. Results are
recorded in `implementation.md` as observed.

## Risks

- **The wt parser could change upstream.** The behaviour has been stable and
  documented since the `;` feature shipped, and the tests pin our model of
  it. The manual check is the real-wt evidence.
- **Old wt versions** that predate `\;` escaping would pass a literal `\;`.
  The existing codex passthrough already relies on `\;`, so this adds no new
  dependency.
- **Encoding change for the default wrapper path.** The wrapper path had no
  space issues before (it has no `"`). The new encoding is identical for
  paths without `;`, `"`, or a tab.

## Separate boundaries — listed, not changed

- **CMD/batch.**
  - `hooks.write_codex_launcher` (`hooks.py:284`): runtime paths inside
    `"…"` in a `.cmd`, where `%VAR%` expands even inside quotes.
  - `hooks._shell_quote_command` (`hooks.py:250`).
  - codex/pi npm batch-shim fallbacks: `codex.py` resolves the native binary
    and falls back to the `.cmd` shim; `cmd.exe` truncates at a newline and
    interprets `& | < > ^ %`.
  - `native_wake.CodexMemberWake._queue` (`native_wake.py:589`).
- **Intentional command execution.** `ProcessBackend.execute_in_pane`
  (`process_base.py:177`, `cmd /c <command>`) is by design.
- **TOML.** `hooks._codex_hook_overrides_windows` (`hooks.py:368`) sets
  `commandWindows`.
- **wt profile elevation** (`elevate: true` in the applied profile): a second
  wt parse of a re-serialised command line (see "How wt parses").

Each of these needs its own consumer analysis; neither this helper nor
`_powershell_quote` makes them safe. One intersection with wt exists: the
codex `.cmd` shim fallback can land in `cmd[0]` of the direct launch. This
plan closes that route by refusing a batch `cmd[0]` there (see decisions),
rather than claiming the encoder covers CMD.

## Plan-review dispositions

1. blocker, `ExpandEnvironmentStringsW` stage omitted — **accepted**.
   - Verified in source at `ConptyConnection.cpp:55`.
   - Step 4 was added to the model.
   - Tail: now avoided via `-EncodedCommand`.
   - A `%` pair (2+ `%`) is the trigger: the rule is independent of the
     environment and valid for both fresh and existing windows.
   - Direct launch: now degrades to the wrapper tab on a `%` pair.
   - Wrapper: now degrades to the classic console on a `%` pair.
   - Neither refuses the spawn.
   - Added controlled-variable model tests (plain substitution, `"`, `'`) and
     a real-Terminal expansion probe.
2. major, the real-Windows oracle missed the production consumers and window
   reuse — **accepted**.
   - Anchor tab, readiness check, and fresh vs existing window modes.
   - A separate `-w 0` tail run.
   - Nonces and completion artifacts.
   - Real wrapper `-File` under 5.1/pwsh, plus a real tail script round trip.
   - An optional Rust recorder.
   - Combination corpus.
   - Records the wt path and file version.
3. minor, `sys.argv[3:]` dropped an argument — **accepted**: now `[2:]`,
   comparing the full list.
4. major, the `.cmd` positive control was not launchable by `CreateProcessW`
   — **accepted**.
   - Now `cmd.exe` + `/c` + batch path as separate argv elements.
   - Mandatory positive control, separate nonces, cleanup between runs.
   - A missing artifact counts as failure.
5. minor, parser contracts and domain — **accepted**.
   - Separate outer `CommandLineToArgvW` and child CRT/Rust models, with
     argv[0] included in the serialized command.
   - A "Supported domain" section: no NUL, the argv[0] precondition, the
     length limit.
6. minor, the batch fallback can reach wt — **accepted**: direct launch is
   skipped (wrapper tab used) for a `.cmd`/`.bat` `cmd[0]`, and the scope
   statement is corrected.
