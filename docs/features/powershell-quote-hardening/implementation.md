# PowerShell single-quote hardening — implementation

Plan: `plan.md` (revision 2, Codex-approved in `plan-review.md` round 2).

## Final design

- `backends/process_manager.py`
  - `_POWERSHELL_SINGLE_QUOTES` = ASCII `'` + U+2018, U+2019, U+201A, U+201B.
  - `_powershell_quote` doubles each of the five **with itself** and wraps the
    result in ASCII `'…'`. Docstring states the tokenizer rule and that the
    guarantee is parser-level only (not wt `;`, not CMD, not native argv).
  - `_write_tab_wrapper` validates every env key with
    `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*")` before writing anything and
    raises `ValueError` otherwise.
  - `_open_windows_terminal_tail` now quotes `log_path` with
    `_powershell_quote` (previously interpolated with no escaping).
- `server_simple._watch_command_powershell` uses the shared
  `_powershell_quote` instead of its ASCII-only inline copy.

Deviations from the plan:

- The optional real-PowerShell test (plan test 7) runs **one** PowerShell
  launch that parses the whole corpus, not one launch per value, and passes
  the probe via `-EncodedCommand`. The first version piped it over
  `-Command -`, which silently evaluates nothing for multi-line statements
  (implementation-review finding 1). CI runners ship pwsh, so this test does
  run in CI.
- `test_watch_command_discovery.py` was not edited: the new
  `test_watch_command_powershell_quotes_all_quote_chars` covers the plan's
  watch-command case, and the existing ASCII `O'Brien` tests still pass
  unchanged. `test_base_runtime.py`'s tail assertion now expects
  `_powershell_quote(log_path)` rather than a hand-built `'…'`.

## Red / green evidence

- Red: `uv run pytest tests/test_backends/test_powershell_quote.py -q` →
  **78 failed, 31 passed, 77 skipped** before the production change
  (failures: the four extra quote chars in exact-output and round-trip tests,
  wrapper fields, env-key guard, watch command, tail). The 31 passing were
  ASCII-only / control cases, including `test_old_ascii_only_quoting_was_breakable`
  which proves the scanner model detects the old bug. (The 77 skips were the
  first, per-value version of the real-parser test.)
- Green: same file plus `test_base_runtime.py` and
  `test_watch_command_discovery.py`, with pwsh 7.6.6 on PATH →
  **218 passed, 1 skipped** (the skip is unrelated).
- Real-parser negative control: with `_powershell_quote` monkeypatched back to
  the old ASCII-only rule, `test_real_powershell_parser_reads_single_literal`
  **fails** (`'x’; calc; ’': FAIL:shape`), so the test can see the bug.

## Validation (Linux, all four CI gates, whole repo)

```bash
uv run ruff format --check .   # 95 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1989 passed, 0 failed, 7 skipped (no pwsh)
# with pwsh 7.6.6 on PATH:     # 1990 passed, 0 failed, 6 skipped
```

## Manual Windows verification

Not runnable on Linux CI. Committed verifier:
`docs/features/powershell-quote-hardening/verify_windows.py`.

```powershell
uv run python docs/features/powershell-quote-hardening/verify_windows.py
```

Per installed shell (`powershell.exe` 5.1 and `pwsh` 7), in a private temp dir:

1. Control: a script built with the **old** ASCII-only quoting and a `’`
   break-out payload must create the `$env:PSQ_SENTINEL` file (proves the
   oracle can see an injection).
2. For ~60 hostile values: `Parser::ParseInput` reports no errors and exactly
   one string constant; the dot-sourced `$v` matches the Python-computed
   Base64; the sentinel file is never created.
3. A real `_write_tab_wrapper` output (hostile cwd directory, env value,
   sidecar path, argv) runs a PowerShell-script stub that writes each received
   argument as Base64; the captured values must match exactly, the sidecar must
   exist at the hostile path, and `PWNED` must not appear. (PowerShell script
   argument binding — not native-exe argv marshalling.)

Prints `PSVersion` and PASS/FAIL per shell; exits 0 only if all pass.

**Partial evidence:** `check_shell` run with **pwsh 7.6.6 on Linux** (dotnet
global tool): 0 failures. The control created the sentinel; all literals and
the wrapper argv round-tripped. Both the reviewer and I ran this independently.

**Still open: not yet run on Windows** (Windows PowerShell 5.1 and pwsh 7 on
Windows). Record the observed output here before relying on the 5.1 claim.

## Follow-ups (out of scope, see plan "Separate boundaries")

Queued as a separate follow-up session ("Harden wt command-line against ;
splitting"); it is not in this PR.

- Windows Terminal `;` sub-command splitting of wt-visible values (`-d cwd` in
  the codex direct-launch branch, the wrapper `-File` path, the tail
  `-Command` script). Command-injection-class; needs its own plan and a
  real-wt test.
- CMD/batch boundaries (`hooks.write_codex_launcher`, codex/pi batch shims,
  `native_wake` codex queue).
