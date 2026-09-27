# PowerShell single-quote hardening — implementation review

Reviewer: Claude Opus (independent post-implementation review).
Reviewed: `plan.md` rev 2, `plan-review.md`, `implementation.md`,
`verify_windows.py`, the uncommitted diff to `backends/process_manager.py` and
`server_simple.py`, and `tests/test_backends/test_powershell_quote.py`.

## How this was checked

- The four gates on Linux, without PowerShell on PATH: `ruff format --check`,
  `ruff check` and `ty check` pass. `pytest` gives 1994 passed and 83 skipped.
- **Real PowerShell was installed for this review.** I installed
  `dotnet tool install PowerShell --tool-path <scratchpad>`, which gave pwsh
  7.6.6 on Linux, and reran the relevant tests and the verifier with it on
  PATH:
  - `test_powershell_quote.py` with pwsh on PATH: **the optional real-parser
    test fails all 77 cases** (finding 1).
  - I ran the same AST probe through `-File` and through `-EncodedCommand`.
    The real parser gives one string constant with a round-tripping value for
    **all 77** non-NUL corpus values. When the probe is given ASCII-only
    quoting instead (`x’; calc`), it reports a parse error, so the probe can
    tell bad quoting from good.
  - `verify_windows.check_shell(pwsh)` on Linux pwsh 7.6.6: **0 failures**.
    The control run with the old quoting created the sentinel, which shows the
    oracle is not blind. All ~60 literals and the wrapper argv round-trip
    matched.
  - This is pwsh 7 on Linux only. Windows PowerShell 5.1 is still unverified,
    as `implementation.md` already says.

## Core change: correct

- `_powershell_quote` doubles each of the five quote characters with itself
  and wraps the result in ASCII `'…'`. This matches `ScanStringLiteral`, which
  appends the second character of a pair. The real parser confirms it.
- Callers are complete for PowerShell-literal renderers:
  - the wrapper covers env values, cwd, sidecar and every argv element;
  - `_watch_command_powershell` now uses the shared helper, which removes the
    duplicate quoting code;
  - `_open_windows_terminal_tail` quotes the log path, which it did not quote
    at all before.
  - `grep` finds no other inline `replace("'", "''")` in `src/`.
- The env-key guard uses `fullmatch`, validates before anything is written,
  and is tested for no-write on rejection. The env passed to
  `_write_tab_wrapper` is the backend override dict from `build_env`, not the
  merged `os.environ`. So Windows keys such as `ProgramFiles(x86)` never reach
  it, and all current keys (`AGENT_*`, `WIN_AGENT_TEAMS_*`, `PATH`,
  `CODEX_MANAGED_BY_NPM`, the lead-wake keys) pass.
- The scanner model matches the cited tokenizer: any quote opens the literal,
  a pair keeps the second character, and an unpaired quote ends it.
  `test_old_ascii_only_quoting_was_breakable` guards the model itself. The
  exact-output tests would catch a regression on their own, without the model.

## Findings

1. **blocker — The optional real-PowerShell test never runs its probe and fails
   wherever PowerShell exists, including both CI jobs.**
   - `test_real_powershell_parser_reads_single_literal` feeds `_AST_PROBE` to
     `pwsh -Command -` over stdin. In that mode PowerShell reads input like an
     interactive prompt: a statement spanning several lines (`ParseInput(` …,
     `FindAll({ …`) waits for a blank line that never comes. At EOF the
     buffered statements are silently dropped.
   - Observed on pwsh 7.6.6: return code 0, stdout contains only terminal
     escape sequences (`'\x1b[?1h\x1b[?1l…'`), and nothing is evaluated.
     `base64.b64decode(<escape junk>)` then raises `UnicodeDecodeError`.
     Result: 77 of 77 cases fail.
   - Both CI jobs will hit this. `.github/workflows` runs `uv run pytest` on
     `ubuntu-latest`, which ships pwsh, and on `windows-latest`, which has
     both pwsh and powershell. The test is therefore not "optional" in CI; it
     turns the PR red on both runners. It would also fail on every Windows
     developer machine.
   - The empty-string case is separately broken: the Base64 of `""` is an
     empty line, so after `.strip()` there is no last line and the test gets
     an `IndexError` or a wrong value.
   - **Fix:**
     - Pass the probe with `-EncodedCommand`, which takes Base64 of the
       UTF-16LE script. This avoids stdin mode, temp files and execution
       policy. Use `stdin=subprocess.DEVNULL`.
     - Prefix the output with a marker, e.g.
       `'OK:' + [Convert]::ToBase64String(...)`, and assert that the last
       line starts with `OK:` before decoding.
   - I checked this exact fix against pwsh 7.6.6: **77/77 pass**, including
     `""`. The shape I checked:

     ```python
     probe = _AST_PROBE.replace(
         "[Convert]::ToBase64String", "'OK:' + [Convert]::ToBase64String"
     ).format(src=...)
     encoded = base64.b64encode(probe.encode("utf-16-le")).decode("ascii")
     subprocess.run([_PWSH, "-NoProfile", "-NonInteractive",
                     "-EncodedCommand", encoded], stdin=subprocess.DEVNULL, ...)
     ```

     Alternatively, write the probe to `tmp_path` as `utf-8-sig` and run it
     with `-ExecutionPolicy Bypass -File`, which I also checked. Without
     `Bypass`, Windows PowerShell 5.1's default `Restricted` policy blocks
     `-File` on client machines.
   - Also update `implementation.md`. It currently says "It runs automatically
     wherever one is on PATH", which implies the test was a working
     independent check. Record the real-parser result once the test is fixed.

2. **minor — `implementation.md` says "No deviations from the plan", but two
   planned test updates were not made.**
   - The plan lists `tests/test_backends/test_base_runtime.py` ("tail test
     update") and `tests/test_watch_command_discovery.py` under "Files
     affected". Neither changed.
   - `test_base_runtime.py:1051` still asserts the old unescaped form
     `f"Get-Content -LiteralPath '{log_path}' -Wait -Tail 80"`. It only passes
     because that `log_path` contains no quote.
   - **Fix:** either
     - change that assertion to
       `f"Get-Content -LiteralPath {_powershell_quote(str(log_path))} -Wait -Tail 80"`,
       or
     - record in `implementation.md` that the new
       `test_terminal_tail_quotes_log_path` and
       `test_watch_command_powershell_quotes_all_quote_chars` replace those
       planned edits, and drop the "No deviations" claim.

3. **minor — The wt follow-up is not linked.**
   - Plan-review round 2 asked for the Windows Terminal `;` follow-up to be
     linked from the implementation record or the PR. `implementation.md`
     describes it but gives no issue reference.
   - **Fix:** file the issue, covering:
     - wt `;` splitting of the codex direct-launch `-d cwd`;
     - the wrapper `-File` path;
     - the tail `-Command` script;
     - the CMD/batch boundaries.

     Then link it from `implementation.md` and the PR body.

4. **nit — Record the Linux pwsh evidence as partial verification.**
   - `verify_windows.check_shell` is sound. I found no false-pass or
     spurious-fail path:
     - the control proves the oracle can see an injection;
     - an injected parse would fail the string-count check before
       dot-sourcing;
     - wrapper success is judged from captured stub output, not the exit code;
     - no shells found means exit 1.
   - It passes unchanged on pwsh 7.6.6 on Linux when `check_shell` is called
     directly.
   - **Fix:** add one line to `implementation.md` recording this as pwsh-7
     evidence. Keep "not yet run on Windows (5.1)" as the open item.

5. **nit — Duplicate corpus entries.**
   - `_hostile_corpus()` produces duplicates: `q * 2` comes from both the runs
     and the 25 ordered pairs.
   - pytest disambiguates the IDs, so this is harmless, but it inflates the
     count, including the 77 real-shell subprocess launches.
   - **Fix:** `list(dict.fromkeys(corpus))`.

6. **nit — Cross-module import of a private helper.**
   - `server_simple` imports `_powershell_quote` from `process_manager`.
   - This is acceptable, since it removes the duplicate, and it matches
     existing private imports. Consider dropping the underscore, or add a
     one-line comment that it is shared on purpose.

## Summary

The production change is correct, minimal, and applied to every PowerShell
literal renderer in scope. The real parser confirms it on pwsh 7.6.6. The
documentation is honest that the guarantee covers the parser only. The one
blocker is test infrastructure: the "optional" real-PowerShell test is broken
and will run, and fail, on both CI runners because both have pwsh. Fix
finding 1, re-run the gates with pwsh on PATH, and address findings 2–3.

VERDICT: CHANGES REQUESTED

## Round 2

I re-reviewed the current diff: the unchanged production code, the
`test_base_runtime.py` tail assertion, the rewritten real-parser test, the
deduplicated corpus, the import comment, and `implementation.md`.

### Checks

- **Gates.**
  - Without pwsh: `ruff format --check`, `ruff check` and `ty check` are
    green.
  - With pwsh 7.6.6 on PATH: `test_powershell_quote.py` plus
    `test_base_runtime.py` give **201 passed**. The real-parser test now runs
    and passes.
- **Negative control.** In a throwaway test that I deleted afterwards, I
  monkeypatched `_powershell_quote` back to the old ASCII-only rule. The
  real-parser test then raises `AssertionError`. This confirms the
  implementer's claim that the test can see the bug.
- **Command-line size.** The single `-EncodedCommand` for the 72-value
  corpus is **7,764 characters**, well under Windows' 32,767-character
  `CreateProcess` limit. There is headroom to roughly quadruple the corpus.
  `-EncodedCommand` is also not subject to execution policy, so Windows
  PowerShell 5.1 on a `Restricted` client machine will not block it.
- **Probe design.**
  - The `OK:`/`FAIL:` markers are prefix-filtered, so terminal escape noise is
    ignored.
  - The count check plus `zip(strict=True)` catches both missing and extra
    results.
  - The empty-value case now works.

### Round 1 dispositions

1. blocker, real-parser test: **resolved**, as verified above.
2. minor, stale tail assertion and "no deviations" claim: **resolved**. The
   assertion now uses `_powershell_quote`, and `implementation.md` lists the
   deviations.
3. minor, wt follow-up not linked: **resolved enough**. It is queued as a
   separate session, recorded in `implementation.md`, and will be in the PR
   body.
4. nit, record partial evidence: **resolved**. The pwsh 7.6.6 Linux result
   is recorded, and Windows 5.1 is kept as the open item.
5. nit, duplicate corpus entries: **resolved** with `dict.fromkeys`.
6. nit, private import: **resolved** with a comment.

### Remaining findings

1. **nit — Make the wt follow-up durable.**
   - A local task chip is not visible to other contributors.
   - **Fix:** when opening the PR, also file a GitHub issue (`--repo
     mikaelliljedahl/agentic-coder-teams-mcp`) and link it from the PR body.
     This does not block the PR.

No correctness or test-quality issues remain. The Windows 5.1 run of
`verify_windows.py` is still an open manual item. `implementation.md`
describes it accurately as unverified and does not claim otherwise.

VERDICT: APPROVED
