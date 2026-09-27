# Windows Terminal command-line hardening — implementation review

Reviewer: Claude Code (Opus), independent post-implementation review.
Scope: the uncommitted diff on `fix/powershell-quote-hardening` (HEAD
`30a3ebc`), meaning `process_manager.py`, the new `test_wt_command_line.py`,
the edits to `test_base_runtime.py` and `test_powershell_quote.py`, and
`implementation.md` / `verify_windows_wt.py`. Reviewed against `plan.md`
rev 2 and `plan-review.md` round 2. No source, test or plan files were
changed.

## What was checked

- **wt stage models compared with the C++ source at `8c0a234f`.**
  - `_wt_split` matches `_addCommandsForArg`:
    - the same regex;
    - `matchedFirstChar = match[0].length() == 1`;
    - an empty prefix is dropped;
    - a placeholder `wt.exe` starts each new sub-command;
    - the loop runs `do … while (!remaining.empty())`.
  - `_add_arg` matches `Commandline::AddArg`: it resumes at
    `pos + Delimiter.length()`.
  - `_wt_rebuild` matches `_getNewTerminalArgs`: it quotes only when an arg
    contains `" "`, with no other escaping.
  - `_expand` stands in for `ExpandEnvironmentStringsW` at
    `ConptyConnection.cpp:55`. It is only used for the controlled-variable
    negative controls.
- **What sits between the rebuild and `CreateProcessW`.**
  - `TerminalSettings::CreateWithNewTerminalArgs` copies `Commandline` and
    `StartingDirectory` verbatim.
  - `GetProfileForArgs` → `_getProfileForCommandLine` only uses
    `NormalizeCommandLine` to *match* a profile. It does not rewrite.
  - `_evaluatePathForCwd` only path-joins. So `-d` is **not**
    env-expanded, and the cwd `%` check in `_codex_direct_launch_blocker` is
    conservative, not required. It is harmless, because `-C <cwd>` is in the
    argv anyway.
- **Exactness of the encoding.** Beyond the tests, I ran an extra property
  fuzz: 50,000 random argv/`-d` combinations over
  `{a, space, tab, ", \, ;, %, \n, '}`, through the test file's own full
  pipeline (`list2cmdline` → `CommandLineToArgvW` → split/unescape →
  rebuild → CRT). The result was 0 mismatches for the child argv and for
  `-d`.
- **`%` routing.**
  - Direct launch → wrapper: `_codex_direct_launch_blocker` runs first. The
    wrapper `%` guard then applies whether or not direct launch was skipped,
    as round 2 required.
  - The guard runs before the sidecar unlink, before `_write_tab_wrapper`
    and before `Popen`. Only a log line is written before it.
  - `spawn_process` catches the new exception in the existing fallback. It
    closes and reopens the log and then takes the classic-console `Popen`.
  - Nothing was launched, so there is no double-run.
  - `WindowsTerminalTabSpawnError` still propagates. The new exception is
    correctly *not* a subclass of it.
- **Round-2 notes.**
  - (1) Isolated process for the expansion probe: **folded**. The anchor is
    launched first with `WTV_PROBE` only when no `WindowsTerminal.exe` is
    running. `expand` and `-w 0` otherwise report NOT EXECUTED.
  - (2) Boundary labelling: **folded** in the verifier's docstring, the
    per-result labels, and the coverage-limits section of `implementation.md`.
  - (3) Cross-argument `%` test: **folded**. The test is
    `test_percent_pair_in_direct_and_wrapper_falls_back_to_classic_console`,
    with `"50%"` and `"80% …"` as two args plus a `%` log dir.
    - Checked by hand: a per-argument predicate would let direct launch
      proceed. That gives a second `Popen`, which `call_count == 1` would
      catch.
  - (4) Base64 / .NET wording: **partly folded**. See finding 5.
- **Gates (Linux, whole repo):**
  - `ruff format --check`: 97 files already formatted;
  - `ruff check`: all checks passed;
  - `ty check`: all checks passed;
  - `pytest`: 2047 passed, 7 skipped.

## Findings

1. **Minor — an unanalysed wt re-serialisation path: profile elevation.**
   - When the resolved profile has `elevate: true` and Terminal is not
     elevated, `TerminalPage` (≈ line 4963) rebuilds the request as
     `new-tab … -- "<Commandline>"` via `NewTerminalArgs::ToCommandline`
     (`ActionArgs.cpp:213`) and hands it to `elevate-shim.exe`.
   - That path wraps the already-rebuilt child command line in raw `"…"`,
     with no `;` escape and no quote escape. A second wt parse then re-splits
     it.
   - With a command line and no `-p`, the "Defaults" profile applies. So a
     user whose `profiles.defaults` sets `elevate: true` would put our encoded
     argv through a pass the encoding does not model.
     - The impact is limited: UAC consent is required, and the setting is
       unusual.
     - It is still a real boundary that sits outside the documented model.
   - **Fix:** add this to the plan's "How wt parses" forwarding note and to
     "Separate boundaries" as unmodelled. Optionally have the verifier record
     `profiles.defaults.elevate`. No code change is needed.

2. **Minor — the verifier's `wt->ps-file` and `console-fallback` oracles do
   not check which route actually ran.**
   - `wrapper_case` passes when the stub's output appears. It never asserts
     that the log has `[wt command]` and has no `[fallback]`.
   - `console_fallback_case` asserts `[fallback]` in the log. It does not
     assert that `[wt command]` is **absent**, or that the recorded reason is
     the `'%' pair` message.
   - Today neither gap produces a false pass:
     - the log dir of `wrapper_case` has no `%`;
     - a regression in the `%` guard would, with settle `0`, leave no
       `[fallback]`.
   - But the labels claim a specific boundary.
   - **Fix:**
     - in `wrapper_case`, require `"[wt command]" in log and "[fallback]" not
       in log`;
     - in `console_fallback_case`, require `"[wt command]" not in log` and
       `"'%' pair" in log`.

3. **Minor / nit — the verifier's `expand` probe proves that expansion
   happens, not that argv is re-split as the plan says.**
   - With `WTV_PROBE='a" "b'` and the arg `p%WTV_PROBE%q`, the child sees
     `pa" "bq`. The CRT parses that as **one** arg, `pa bq`.
   - So the probe proves the rewrite, which is honestly what the docstring
     and `implementation.md` say. The plan text ("must re-split the recorder's
     argv") is not demonstrated.
   - **Fix:** either put the variable in its own space-delimited position
     (for example the arg `x %WTV_PROBE% y`, which becomes
     `"x a" "b y"` → `["x a", "b y"]`, and assert the count changed), or
     amend the plan wording to "rewritten".

4. **Nit — some plan claims in the verifier are not exercised, and the plan
   was not updated to say so.**
   - PowerShell `-File` is exercised only under `powershell.exe` 5.1. That
     is correct, because production hard-codes `powershell`. But the plan
     says "and under `pwsh` if installed".
   - The injection probe and the wrapper, fallback and tail cases run only in
     existing-window mode. Only `native_case` runs fresh vs existing.
   - The Rust recorder does not check the cwd.
   - **Fix:** record these as coverage limits in `implementation.md`, or
     loop the injection probe over both modes.

5. **Nit — the round-2 note 4 wording about .NET was not folded.**
   - `plan.md` "Supported domain" still says ".NET's
     `CommandLineToArgvW`-based parsing".
   - The payload-only Base64 assertion *was* folded:
     `re.fullmatch` runs on `child[3]` only.
   - **Fix:** reword to "canonical `list2cmdline` form, compatible with the
     tested runtimes (Python/UCRT, Rust `std`)", or note the disposition in
     `implementation.md`.

6. **Nit — the `%` check on `cwd` in `_codex_direct_launch_blocker` is
   conservative, but its wording implies it is required.**
   - The skip reason says the cwd "would be expanded by wt". `-d` is not
     env-expanded (see "What was checked").
   - Refusing is still the right call: codex's `-C <cwd>` is on the command
     line and *is* expanded.
   - **Fix:** in the docstring, say the cwd check is defensive, or drop it
     in favour of the argv check.
   - `_wt_may_expand([*cmd])` can simply be `_wt_may_expand(cmd)`.

7. **Nit — test hygiene.** `test_wrapper_path_with_percent_pair_refuses_before_writing`
   opens `log_path.open("a")` inline and never closes it.
   - This can emit a `ResourceWarning`, and on Windows it can hold the file
     open until the test ends.
   - **Fix:** use a `with` block, or close it in `finally`.

## Assessment

The production change is correct and tight:
- the three wt launch sites go through one builder;
- the encoding is exact within its stated domain (confirmed by fuzzing
  through the source-faithful models);
- the `%` policy counts across arguments, as required;
- the unsafe-command-line error is raised before any side effect;
- the fallback reuses the existing no-double-run path.

The tests pin each routing decision, and they include negative controls that
prove the models can see each bug. The composition test would catch the
per-argument regression that round 2 warned about. Every finding is either
documentation or verifier-oracle tightening. None of them changes what
production does. Real-Windows evidence is still pending, and
`implementation.md` says so accurately.

VERDICT: APPROVED
