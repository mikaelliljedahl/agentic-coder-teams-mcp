# Independent security plan review

Reviewed the proposed plan against the current worktree, including a whole-`src/` search for PowerShell, pwsh, cmd, wt, command switches, shell execution, and command renderers. No source code was modified. This is source-based review; neither Windows PowerShell 5.1 nor pwsh is available in this environment, so Windows execution is not claimed.

1. **major — The manual Windows verification does not test the intended script.**

   In the documented PowerShell command, the outer double-quoted `python -c` argument expands both occurrences of `$v` before Python receives them. With `$v` unset, the generated assignment loses its left-hand side; with it set, the result depends on ambient state. Also, Python interprets the adjacent literals in `v='…‛''z'` as concatenation, so the intended ASCII apostrophe is absent. Printing a payload that itself contains `PWNED` is also a weak oracle.

   **Recommendation:** put the generator in a standalone `.py` file and run `python <file>`; construct quote characters with `chr()` or Unicode escapes to avoid editor/outer-shell ambiguity. Generate BOM-bearing script bytes, and compare the evaluated literal against an independently encoded expected value (for example Base64 of UTF-8), failing with a nonzero exit status on mismatch. Use a separate sentinel variable/file to detect execution. Run the resulting script explicitly with both `powershell.exe -NoProfile -File ...` and `pwsh -NoProfile -File ...`, and record versions, exit codes, and assertions. Include every quote character and adjacent mixed pairs, not just the single example. A PowerShell parser AST check should assert one string constant with the expected value and no parse errors. This fixes the crucial independent check of the Python scanner model.

2. **major — The wt residual-risk description is incorrect and the reachability audit is incomplete.**

   The plan's claim that semicolon splitting would “at worst split the tail tab” understates the consequence. Windows Terminal supports semicolon-separated commands, and a new-tab command can launch an executable; attacker-controlled fragments can therefore potentially request another process, subject to the complete argument layout. This is a separate parser boundary from PowerShell quoting. [Microsoft's Terminal command-line documentation](https://learn.microsoft.com/en-us/windows/terminal/command-line-arguments) describes both behaviors.

   In this repo, `WindowsProcessManager.log_path` validates team/agent names before constructing paths (`process_manager.py:933`), and `_tab_window_id` validates the team name. Consequently the plan should not identify unrestricted team/agent names as the ordinary source of smart quotes or semicolons in these paths. The home directory and `WIN_AGENT_TEAMS_LOG_DIR` override remain relevant. The direct-launch branch also passes `request.cwd` to `wt -d` outside `_escape_wt_passthrough`; the wrapper branch passes the derived `.launch.ps1` path unescaped. Tail mode passes its entire script through wt.

   **Recommendation:** correct the risk and input provenance, inventory all wt arguments, and explicitly track this as a separate command-injection/argument-integrity follow-up if it remains outside this narrowly scoped patch. Do not claim that the tail change secures the complete launch pipeline. Verify semicolon handling through actual wt before declaring that boundary safe; a mocked Popen assertion only proves the Python argument list. A larger wt redesign need not be bundled into this quote-helper change.

3. **minor — The broader cmd/wt caller audit omits several distinct execution boundaries.**

   I found no additional PowerShell single-quoted runtime-value renderer beyond `_powershell_quote`/`_write_tab_wrapper`, `_watch_command_powershell`, and `_open_windows_terminal_tail`. The excluded CIM scripts are correctly characterized: `procinfo._windows_command_lines` and `_find_codex_pid_by_token` use fixed script text; `_child_pids` interpolates a parsed integer. However, the requested broader audit also includes:

   - `hooks.write_codex_launcher` (`hooks.py:284`): builds a `.cmd` file containing runtime interpreter and session paths inside double quotes. CMD percent expansion is a separate concern; PowerShell quote doubling does not address it.
   - `hooks._shell_quote_command` (`hooks.py:250`) and hook producers: build command strings from runtime paths, later interpreted by the hook consumer's shell. `_codex_hook_overrides_windows` transports a bare `.cmd` path through TOML to `commandWindows`; it is not PowerShell text.
   - `ProcessBackend.execute_in_pane` (`backends/process_base.py:169`): passes the caller's explicit command to `cmd /c` on Windows. This is an intentional command-execution API, not an unquoted data literal.
   - Codex and pi batch-shim fallbacks (`backends/codex.py`, `backends/pi.py`), plus `native_wake.CodexMemberWake`'s `_queue` path (`native_wake.py:589`): native argv can ultimately cross a CMD boundary. `_queue` sanitizes the sender in its generated notice. These paths need their own argument/consumer analysis, not this helper.
   - `_spawn_in_terminal_tab`, `_escape_wt_passthrough`, and the tail launcher: wt interpretation precedes any PowerShell interpretation, as detailed above.

   **Recommendation:** add these to the plan's audit as separately scoped boundaries, with the distinction between intentional commands and embedded data. Do not expand this patch into generic shell quoting or apply `_powershell_quote` to CMD/TOML strings.

4. **minor — The core tokenizer claim is correct, but state the proof's boundary and exceptional transport cases.**

   The five-character set is confirmed by [PowerShell v7.4.6 `CharTraits.cs`, `IsSingleQuote`](https://github.com/PowerShell/PowerShell/blob/v7.4.6/src/System.Management.Automation/engine/parser/CharTraits.cs#L253-L260). [The matching `ScanStringLiteral` implementation](https://github.com/PowerShell/PowerShell/blob/v7.4.6/src/System.Management.Automation/engine/parser/tokenizer.cs#L2135-L2172) consumes any two quote characters as a pair and retains the second. Embedded NUL is distinguished from end-of-input and retained. Here-string footer syntax, backticks, dollar signs, double quotes, line breaks, and an interior U+FEFF do not switch this scanner to another mode.

   The proof is simple: each nonquote maps to itself; each quote maps to two identical quotes, which decode to one original quote. Concatenation preserves those decoding boundaries, even for mixed runs. The final ASCII delimiter is the first unpaired quote after the encoded payload. This establishes one literal and its value in a correctly decoded script, not arbitrary native-process argv fidelity. NUL cannot be transported as ordinary Windows command-line data; native argument marshalling and batch shims have additional rules.

   `_write_tab_wrapper` already writes raw `utf-8-sig` bytes and deliberately preserves payload newlines (`process_manager.py:1271–1278`). Retain this: Windows PowerShell interprets BOM-less non-ASCII script files using its legacy encoding assumptions. [Microsoft encoding documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding?view=powershell-5.1).

   **Recommendation:** cite the pinned source, add this short proof, and distinguish parser safety from transport validity. Add empty strings, embedded NUL (parser-only), U+FEFF, CR/LF/CRLF, and apparent here-string delimiters to the corpus. Preserve existing byte-level BOM/newline tests. The inspected public source establishes the PowerShell 7 behavior; record actual 5.1 results instead of presenting those as already verified.

5. **minor — Make the integration tests and env-key guard precise.**

   The shared helper and its current module location are reasonable; the env-key guard is small, useful defense in depth. However, Python `$` permits a match before a final newline when used with `re.match`, so a literal translation of the proposed anchored pattern can accept `"A\n"`. Also, “every line parses back” is unsuitable for literals containing newlines, and the proposed wrapper test omits the quoted sidecar path.

   **Recommendation:** use `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)` and test trailing newline, CR, whitespace, colon, Unicode quotes, and valid representative internal keys. Validate before writing the wrapper. Test all five quote characters through each renderer, including executable paths, sidecar paths, and watch executable/session paths. Parse complete generated fragments rather than splitting payloads into lines. Retain exact expected-output tests alongside the scanner model; include a hostile mixed run and empty value. Describe the corpus test as regression evidence, not a proof by itself. An optional real-PowerShell parser integration test, skipped when unavailable, would preserve the independent verification beyond the one-time Windows check.

The implementation design is appropriately narrow and the quote transform is sound. Approval is withheld for the broken independent verification and misleading residual-risk analysis; the additional audit and test clarifications above should be incorporated before implementation sign-off.

VERDICT: CHANGES REQUESTED

## Round 2

Reviewed revision 2 and its dispositions against the previously inspected callers, and rechecked the wrapper and watch renderer in the worktree. The five original findings are addressed at plan level: the standalone verifier avoids outer-shell interpolation, the wt risk and provenance are corrected, separate parser boundaries are inventoried, the proof and encoding boundary are explicit, and the guard/test design is substantially strengthened. No source code was edited and no Windows execution was performed.

Remaining findings are non-blocking implementation clarifications:

1. **minor — Isolate the manual verification sentinel.** The proposed verifier deletes a fixed `%TEMP%\pwned.txt`, which could belong to another test or user, and concurrent runs could interfere with its absence check. **Recommendation:** allocate a private temporary directory for each run, place scripts and the sentinel there, and clean up only that directory. Make the injected sentinel command robust to spaces in the path so that absence is a meaningful check. Keep the independent Base64 and parser assertions as the primary correctness oracles.

2. **minor — Make wrapper verification assertions independent of its exit status and label the tested boundary accurately.** `_write_tab_wrapper` deliberately ends with `exit 0`; therefore a zero wrapper exit status alone cannot establish that the stub ran or received correct arguments. A PowerShell script stub exercises PowerShell argument binding, not native-executable argument marshalling. **Recommendation:** require captured stub output, assert its argument count and each Base64-encoded value against independently computed expectations, and fail the Python verifier on missing or mismatched output. Describe this as PowerShell script argument verification; use a native executable stub only if claiming native argv fidelity, which remains outside this patch's stated guarantee.

The revised scope and implementation approach are approved. Actual Windows 5.1/7 results remain implementation verification work, and the promised wt follow-up should be linked in the implementation record or PR. Approval does not certify those separate parser boundaries.

VERDICT: APPROVED
