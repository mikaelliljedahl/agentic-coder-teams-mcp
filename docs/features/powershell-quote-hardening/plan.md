# PowerShell single-quote hardening — plan

Revision 2 — incorporates `plan-review.md` (Codex, CHANGES REQUESTED).
Dispositions are at the end.

## Scope

Make every place that embeds a runtime value in a PowerShell single-quoted
string literal safe against PowerShell's *full* set of single-quote characters,
not just ASCII `'`. Independent of the codex `trust_cwd` feature (PR #73),
which rejects these characters for `trust_cwd` paths only.

Explicitly **not** in scope: Windows Terminal (`wt`) command-line parsing,
CMD/batch quoting, TOML hook commands. Those are separate parser boundaries
(see "Separate boundaries — follow-up"). This patch does not claim to secure
the full Windows launch pipeline.

## Current behaviour

PowerShell's tokenizer treats **five** characters as a single quote, both as
literal delimiters and as the escape inside one
([`CharTraits.cs` `IsSingleQuote`, v7.4.6](https://github.com/PowerShell/PowerShell/blob/v7.4.6/src/System.Management.Automation/engine/parser/CharTraits.cs#L253-L260)):

| Char | Code point | Name |
|------|------------|------|
| `'`  | U+0027 | APOSTROPHE |
| `‘`  | U+2018 | LEFT SINGLE QUOTATION MARK |
| `’`  | U+2019 | RIGHT SINGLE QUOTATION MARK |
| `‚`  | U+201A | SINGLE LOW-9 QUOTATION MARK |
| `‛`  | U+201B | SINGLE HIGH-REVERSED-9 QUOTATION MARK |

[`ScanStringLiteral`](https://github.com/PowerShell/PowerShell/blob/v7.4.6/src/System.Management.Automation/engine/parser/tokenizer.cs#L2135-L2172):
inside a single-quoted literal, on any quote char the scanner peeks the next
char; if that is *also* a quote char (any of the five) it consumes both and
appends the **second**; otherwise the literal ends. Nothing else is special in
a verbatim literal: no `$` expansion, no backtick escapes; newlines, NUL,
interior U+FEFF, double-quote look-alikes (U+201C–U+201E) and here-string-like
text (`'@`, `"@`) are all plain content.

The repo only doubles ASCII `'`, so a value such as `x’; calc; ’` closes the
literal at `’` and the remainder is parsed as PowerShell.

### Safety argument for the fix

Encode each non-quote char as itself and each quote char `q` as `qq`, then wrap
in ASCII `'…'`. Decoding: a non-quote char appends itself; `qq` is a pair and
appends `q`. Because every quote in the payload is emitted as a pair, the
scanner never sees an unpaired quote inside the payload, regardless of
mixed/adjacent runs (pairs are consumed left-to-right and never straddle a
boundary, since each encoded unit is self-contained). The first unpaired quote
is the closing ASCII `'`. Therefore the output is exactly one literal whose
value equals the input. This is parser-level safety for a correctly decoded
script (the wrapper is already written as `utf-8-sig` bytes, which Windows
PowerShell 5.1 needs for non-ASCII). It says nothing about native-process argv
fidelity (e.g. NUL cannot travel in a Windows command line at all).

### Affected sites (PowerShell-literal renderers — fixed here)

1. `backends/process_manager.py` `_powershell_quote`, used by
   `WindowsProcessManager._write_tab_wrapper` (the per-spawn `.launch.ps1` run
   in a Windows Terminal tab, all three backends on the default tab path).
   Quoted values: env values (`AGENT_NAME`, `AGENT_SESSION_ID`,
   `WIN_AGENT_TEAMS_SESSION_DIR`, codex `PATH`, …), `Set-Location -LiteralPath
   <cwd>`, `Out-File -FilePath <sidecar_path>`, and every agent argv element
   (`-C <cwd>`, prompt, model, executable path, …). The cwd and prompt are
   caller-controlled.
2. `server_simple.py` `_watch_command_powershell` — duplicated inline
   ASCII-only quoting. Renders `watch_command_powershell` returned by
   `spawn_agent`/`agent_watch_paths`, which the coordinator runs verbatim.
   Tokens: python executable path, session dir (under the home dir).
3. `backends/process_manager.py` `_open_windows_terminal_tail` —
   `f"Get-Content -LiteralPath '{log_path}' …"` with **no** escaping. Team and
   agent names are validated by `log_path()` (`_validate_safe_name`), so the
   realistic sources of quote chars are the user's home directory and the
   `WIN_AGENT_TEAMS_LOG_DIR` override.
4. `$env:{key}` in `_write_tab_wrapper` — key interpolated bare. All keys are
   internal constants today; add a fail-closed guard.

Checked, not affected: `procinfo._windows_command_lines`,
`_find_codex_pid_by_token` (fixed script text), `_child_pids` (interpolates a
parsed `int`).

### Separate boundaries — follow-up, not changed here

- **Windows Terminal `;` splitting.** wt splits its command line on `;` even
  inside quoted tokens, and a split fragment is a new wt sub-command that can
  launch an arbitrary executable in a new tab. Unescaped wt-visible values:
  `-d request.cwd` in the codex direct-launch branch (outside
  `_escape_wt_passthrough`), the `.launch.ps1` wrapper path passed to `-File`,
  and the whole tail `-Command` script (log path from home dir /
  `WIN_AGENT_TEAMS_LOG_DIR`). This is command-injection-class and needs its own
  analysis plus a real-wt test; I will file it as a follow-up task and say so
  in the PR. This patch's tail change fixes only the PowerShell layer.
- **CMD / batch:** `hooks.write_codex_launcher` (`.cmd` with runtime paths in
  double quotes — `%` expansion), `hooks._shell_quote_command`, codex/pi
  batch-shim fallbacks, `native_wake.CodexMemberWake._queue`.
- **Intentional command execution:** `ProcessBackend.execute_in_pane`
  (`cmd /c <caller command>`) — by design, not a data literal.
- **TOML:** `_codex_hook_overrides_windows` `commandWindows` path.

`_powershell_quote` must not be applied to any of these.

## Proposed design

- Keep one helper in `process_manager.py`:
  ```python
  _POWERSHELL_SINGLE_QUOTES = frozenset("'‘’‚‛")

  def _powershell_quote(value: str) -> str:
      text = str(value)
      return "'" + "".join(
          ch * 2 if ch in _POWERSHELL_SINGLE_QUOTES else ch for ch in text
      ) + "'"
  ```
  Doubling each quote **with itself** preserves the value exactly. Docstring
  cites the tokenizer rule and the five characters.
- `server_simple._watch_command_powershell` uses `_powershell_quote`
  (removes the duplicate).
- `_open_windows_terminal_tail` uses `_powershell_quote(str(log_path))`.
- `_write_tab_wrapper` validates every env key with
  `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)` **before** writing anything;
  raises `ValueError` naming the key otherwise.
- Keep the existing raw-bytes `utf-8-sig` write unchanged.

## Files affected

- `src/claude_teams/backends/process_manager.py`
- `src/claude_teams/server_simple.py`
- `tests/test_backends/test_powershell_quote.py` (new)
- `tests/test_backends/test_base_runtime.py` (tail test update)
- `tests/test_watch_command_discovery.py`
- `docs/features/powershell-quote-hardening/*`

## Test cases (red first)

A test-only scanner model `_ps_scan_single_quoted(text, start) -> (value, end)`
implements `ScanStringLiteral` as cited. It is regression evidence, not a
proof; exact-output assertions sit alongside it.

1. Exact output: each of the five chars (parametrised) → doubled with itself;
   empty string → `''`; `x’; calc; ’` and `a'‘’‚‛b` → exact expected literals.
2. Round-trip via the scanner for a hostile corpus: each quote alone, runs of
   1–4, at start/end, all 25 adjacent ordered pairs, empty string, NUL
   (parser-level only), U+FEFF, CR/LF/CRLF, `'@`/`"@` lines, `$(calc)`,
   backticks, U+201C–U+201E. Assert the value round-trips and the literal
   ends exactly at the end of the quoted string.
3. `_write_tab_wrapper` with a hostile value (every quote char + `; calc; `)
   in cwd, an env value, an argv element (executable and prompt) and the
   sidecar path: scan each full generated fragment (not per line) and assert
   each literal decodes to its input and the statement contains nothing after
   the expected literals.
4. Env-key guard: rejects `""`, `"A;B"`, `"A}"`, `"A:B"`, `"A B"`, `"A\n"`,
   `"A\r"`, `"A’"`, `"1A"`; accepts `AGENT_NAME`, `PATH`,
   `WIN_AGENT_TEAMS_SESSION_DIR`, `_X1`; no file written on rejection.
5. `_watch_command_powershell` with a session dir containing all five quote
   chars: each token scans back to the `_watch_argv` token.
6. `_open_windows_terminal_tail` (wt + Popen mocked) with a log path
   containing `'` and `’`: the `-Command` string is
   `Get-Content -LiteralPath <_powershell_quote(log_path)> -Wait -Tail 80`
   and the literal scans back to the path. (Proves the Python argv only; wt
   behaviour is the follow-up.)
7. Optional real-parser test, skipped when neither `pwsh` nor `powershell` is
   on PATH: use `[System.Management.Automation.Language.Parser]::ParseInput`
   on `$v = <literal>` and assert no parse errors, a single
   `StringConstantExpressionAst` whose value (emitted as Base64 UTF-8) equals
   the input, for the corpus minus NUL.

Existing byte-level BOM/newline wrapper tests stay as-is.

## Risks

- Behaviour changes only for values containing the four extra quote chars
  (now preserved exactly instead of breaking out).
- Env-key guard could reject a future legitimate key with unusual chars —
  acceptable, fails loudly at spawn.
- Scanner model could share a misunderstanding with the implementation;
  mitigated by the cited source, test 7 where available, and the manual Windows
  check.

## Manual Windows verification (record results in implementation.md)

Standalone script `docs/features/powershell-quote-hardening/verify_windows.py`
(committed, run by hand on Windows from the repo root with the venv active):

- Builds the corpus with `chr()` / `\u` escapes only (no editor/shell
  ambiguity), including every quote char, all adjacent mixed pairs, and
  `x’; Set-Content -LiteralPath $env:TEMP\pwned.txt 1; ’`.
- For each value writes a `utf-8-sig` `.ps1`:
  `$v = <_powershell_quote(value)>` then compares
  `[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($v))` against the
  independently computed Base64 of the value, `exit 1` on mismatch. Also runs
  the Parser AST check (no errors, one string constant).
- Deletes `%TEMP%\pwned.txt` beforehand; runs each script with
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -File` and
  `pwsh -NoProfile -File` (if installed); asserts exit 0 and that the sentinel
  file was **not** created; prints a summary with `$PSVersionTable.PSVersion`.
- Runs the same check against a real `_write_tab_wrapper` output with a stub
  "agent" (`cmd` replaced by a PowerShell script that echoes its args as
  Base64) to confirm argv arrives intact.

Windows 5.1/7 results are recorded as observed, not assumed.

## Plan-review dispositions

1. major, manual verification broken — **accepted**: replaced with the
   standalone `verify_windows.py` described above (Base64 oracle, separate
   sentinel file, AST check, both shells, versions recorded).
2. major, wt risk understated / provenance wrong — **accepted**: corrected
   provenance (names validated; home dir / `WIN_AGENT_TEAMS_LOG_DIR` / cwd are
   the sources), restated wt `;` as command-injection-class, inventoried wt
   arguments, and scoped it as a tracked follow-up; PR will say so.
3. minor, broader cmd/wt audit — **accepted**: listed as separate boundaries;
   helper not applied to them.
4. minor, cite source / proof boundary — **accepted**: citations, safety
   argument, parser-vs-transport distinction, extended corpus.
5. minor, fullmatch / full-fragment tests / sidecar — **accepted**: all
   folded into tests 3–7.
