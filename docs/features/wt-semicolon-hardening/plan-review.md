# Independent security review

Reviewed the plan, the three launch paths in `process_manager.py`, the related runtime and PowerShell tests, the sibling PowerShell plan, and Microsoft Terminal source at `8c0a234f`. No source or plan files were changed. Real Windows execution was not available in this Linux worktree.

1. **Blocker — the pipeline omits Terminal's environment expansion after quoting.**

   [`ConptyConnection::_LaunchAttachedClient`](https://github.com/microsoft/terminal/blob/8c0a234f/src/cascadia/TerminalConnection/ConptyConnection.cpp#L55) calls `ExpandEnvironmentStringsW` on the complete rebuilt child command line, before passing it to `CreateProcessW` (around lines 163–176). This is a Terminal boundary, including for native executables; it is not the separately deferred CMD `%VAR%` issue. For example, a literal `%USERNAME%` prompt is changed even when both proposed helpers work perfectly. An expansion containing `"` can change child argument boundaries after encoding. In the tail script, an expansion containing an apostrophe can break a previously safe PowerShell literal: with a controlled variable whose value is `x'; <sentinel command>; '`, a literal directory component `%WT_REVIEW_VALUE%` is expanded after `_powershell_quote` ran. Expansion does not rerun the wt semicolon splitter, but can inject into the subsequent child parser.

   **Recommendation:** add this stage to the threat model and make an explicit escape/reject/avoid decision for percent-bearing child values. Do not assume doubling `%` escapes `ExpandEnvironmentStringsW`. Practical choices include avoiding command-line script text with `-EncodedCommand`, routing direct prompts through a script/data file, and using a safe staging path or rejecting `%` in remaining exposed paths. Specify how existing-window environment differences affect the policy. Add controlled-variable tests for ordinary substitution, embedded double quotes, and PowerShell single quotes; establish the variable before starting the private Terminal instance. The current model and corpus would all pass while missing this vulnerability.

2. **Major — the proposed real-Windows oracle does not establish the production consumer and window-reuse behavior.**

   A Python argv recorder validates one native runtime, but the actual paths launch Rust Codex and Windows PowerShell 5.1. A visual tail check cannot establish exact script/path preservation or absence of an additional side effect. Repeated use of one window name does not guarantee reuse: the recorder exits immediately and Terminal may close its last tab/window before the next case.

   **Recommendation:** keep an anchor process/tab alive, wait for readiness, then run and label both fresh-window and existing-named-window cases; separately exercise the `-w 0` tail policy in an isolated session. Require completion artifacts and unique nonces, not merely launcher exit or lack of a sentinel. Exercise the real wrapper `-File` path with a hostile directory and assert its output; execute a PowerShell script/path round trip under 5.1, and under pwsh if claiming support for it. A small Rust argv recorder would independently cover the direct native consumer. Include combinations of spaces, tabs, quotes, semicolons and trailing backslash runs, not just separate simple examples. Record actual tested executable paths/versions: the stable Store package version alone need not identify the `wt` resolved from PATH (portable/Preview installations exist).

3. **Minor — the recorder slices off the first hostile argument.**

   For `[sys.executable, stub.py, out.json, *hostile_args]`, Python receives `sys.argv == [stub.py, out.json, *hostile_args]`. The proposed `sys.argv[3:]` drops the first hostile value; the correct slice is `sys.argv[2:]`.

   **Recommendation:** fix the planned index and compare the entire received list, including count and ordering. Do not compensate by slicing the expected input; that would create an untested first argument.

4. **Major — the injection positive control is not a dependable executable-launch oracle.**

   The proposed split fragment is an absolute `mk.cmd` path. Terminal's launch path above invokes `CreateProcessW` directly with no `cmd /c` wrapper. A batch file is not a native executable, so making the path space-free does not establish that the positive control can run. If the control fails and is properly mandatory, the verifier becomes inconclusive, rather than demonstrating the security property.

   **Recommendation:** use a native sentinel executable, or deliberately construct the injected command as `cmd.exe` followed by separate `/c` and batch-path argv elements. Prove the raw variant produces the sentinel and the protected variant both completes its expected recorder output and does not produce it. Give each run a separate nonce/output location, wait for positive-control completion, and clean up before testing the protected run. Failure to launch or a missing completion artifact must fail the verification, not count as protection.

5. **Minor — distinguish parser contracts and bound the exactness claim.**

   `_wt_child_arg` is sound for ordinary non-NUL child arguments under the described rebuild rule, before the omitted expansion stage: delimiter decoding restores the encoded token; if the original contains an ASCII space, removing and restoring the outer quotes recreates `list2cmdline([arg])` exactly; otherwise wt leaves that encoding untouched. This covers tabs, empty arguments, embedded quotes, backslash runs, trailing backslashes, newlines and semicolon combinations. Encoding does not introduce or remove ASCII spaces, so the predicate remains consistent.

   However, the plan's single “MSVC/CommandLineToArgvW” model conflates parsers that are not identical. Rust explicitly uses modern C/C++ argument rules rather than `CommandLineToArgvW`; see its [parser and argv[0] special case](https://github.com/rust-lang/rust/blob/1.90.0/library/std/src/sys/args/windows.rs#L31-L95). This does not invalidate the helper's canonical encoding for ordinary arguments, but it matters for the claimed proof and negative controls. Child argv[0] is special, too: it cannot be treated as arbitrary prompt data containing escaped quotes or a quoted trailing backslash. Normal executable paths satisfy a narrower contract. NUL cannot traverse a Windows command line; invalid Unicode and total command-line length are also outside an unrestricted “all inputs” claim.

   **Recommendation:** state the supported domain and executable-path preconditions, test data after a real/dummy executable token, and model the outer `CommandLineToArgvW` stage separately from child CRT/Rust parsing. Include argv[0] in the outer serialized test command rather than calling an argv[0]-aware parser on `argv[1:]`. Windows native API/runtime probes should validate the test models. For PowerShell, distinguish native argv decoding from its later `-Command` script parsing; the one complete script argument must survive both.

6. **Minor — correct the scope statement about batch fallbacks.**

   `CodexBackend.discover_binary()` can return the `.cmd` shim, `build_command()` places it in `cmd[0]`, and `_spawn_in_terminal_tab` selects direct launch solely from backend type and the direct-launch environment flag. Consequently that fallback can be supplied to wt. The plan's assertion that none of the listed separate boundaries passes through wt is too broad. A WT encoder does not make a CMD consumer safe, and a raw batch executable may fail to launch at this boundary altogether.

   **Recommendation:** explicitly restrict the direct-path contract to a native executable, or document and test the fallback route and its separate guarantees. Keeping general batch hardening outside this feature is reasonable, but do not assert that the routes cannot intersect.

## Confirmed analysis and forwarding behavior

The split/unescape reading is correct at the pinned revision. `BuildCommands` processes each already-parsed argv element; `_addCommandsForArg` applies `^;|[^\\];` independently, and every resulting argument goes through `Commandline::AddArg`, including `-d` and other option values. Inserting one backslash immediately before every semicolon prevents every delimiter match. For an original run of *n* backslashes before a semicolon, decoding removes the inserted last backslash from the resulting *n+1* run. The replacement loop resumes after the resulting semicolon, so it cannot recursively strip the original run. Adjacent semicolons work for the same reason. The plan should say “unescaped semicolon,” rather than literally “a semicolon anywhere,” when describing the vulnerable behavior.

CLI11 receives a reversed vector of strings, with the wt executable removed; it does not lex a newly joined string at this point. The explicit `--` and `positionals_at_end(true)` protect child option-like arguments, while semicolon splitting happens earlier. The three identified production construction paths cover the wt launch sites found in `src`; escaping all option tokens is appropriate for this delimiter boundary. Future unvalidated option values still need to satisfy CLI11 option semantics—delimiter escaping is not general validation.

At **8c0a234f**, the forwarding implementation is already the single-instance `WindowEmperor` implementation, not the older monarch/peasant architecture suggested by some stale comments:

- [`wt/shim.cpp`](https://github.com/microsoft/terminal/blob/8c0a234f/src/cascadia/wt/shim.cpp) preserves `pCmdLine` and only prefixes its executable name when starting `WindowsTerminal.exe`.
- [`WindowEmperor.cpp`](https://github.com/microsoft/terminal/blob/8c0a234f/src/cascadia/WindowsTerminal/WindowEmperor.cpp#L120-L146) serializes the raw `GetCommandLineW()` string in the handoff payload. The receiver uses `commandlineToArgArray`/`CommandLineToArgvW` on that raw string (around line 1272), just as fresh startup does around line 598.
- [`TerminalApp/Remoting.cpp`](https://github.com/microsoft/terminal/blob/8c0a234f/src/cascadia/TerminalApp/Remoting.cpp#L43-L52) stores the argument array and parses it. Window selection uses the parsed target and dispatches the resulting object; it does not join decoded arguments into another wt command line.

Thus forwarding/reuse at this revision does **not** add an unsafe argv reserialization or a second delimiter-unescape pass. This is a source conclusion for this revision, not proof about every older shipped monarch/peasant build; the Windows verifier should record which build it actually exercises.

The semicolon escape and the proposed inverse of wt's rebuild rule are good foundations. Approval is blocked by the unmodeled environment expansion and the verification gaps above.

VERDICT: CHANGES REQUESTED

## Round 2

Reviewed revision 2 against the existing launch/fallback code and the previously inspected Terminal sources. The six first-round findings are addressed at the design level. No source or plan changes were made; this is plan approval, not evidence that the implementation or Windows verification has passed.

The revised expansion policy is sound for the stated boundary. `ExpandEnvironmentStringsW` needs two percent delimiters; fewer than two percent signs in its entire input precludes substitution, independently of the receiving Terminal's environment. Counting across the complete rebuilt child command line, rather than separately per argument, correctly catches cross-argument pairs. Counting `-d` separately is a conservative additional path check. Encoding and delimiter unescaping do not change the percent count. Base64 UTF-16LE removes the tail script from this boundary while preserving the sibling PowerShell-literal protection.

The direct-to-wrapper and wrapper-to-classic-console decisions also fit the existing control flow. The new exception must remain a distinct pre-launch failure, with the check moved ahead of the current sidecar unlink and wrapper write as specified. The existing fallback handler already closes/reopens the log and then reaches the classic-console branch; extending that handler as planned does not create a double-run risk. The percent guard must apply to the wrapper command even after direct launch has been skipped. Batch launch remains a separate consumer boundary, appropriately excluded from the native direct route.

Remaining non-blocking findings:

1. **Minor — the controlled-expansion probe needs a fresh Terminal process, not merely a fresh window.** The revised verifier says `WTV_PROBE` is set in the environment Terminal is started from, in “fresh-window mode only.” At the pinned revision a new named window can still be created by the existing single-instance Terminal process, whose environment does not acquire the probe variable from this invocation. Starting anchor windows before setting the variable has the same problem. The mandatory negative control should fail in that situation, so this is a reproducibility issue rather than a false security pass.

   **Recommendation:** explicitly establish an isolated Terminal process with `WTV_PROBE` set before its first launch, then create both fresh and reused windows under that process. If isolation cannot be established, fail/report the probe as not executed, rather than treating an unchanged raw value as evidence against the expansion stage. The `-w 0` case should use that same isolated process so it cannot target an unrelated user window.

2. **Minor — distinguish the wrapper-file boundary from the wrapper's subsequent native invocation.** The planned PowerShell stub validates the wrapper path and PowerShell parameter/literal handling. It does not prove that Windows PowerShell 5.1 preserves every quote/empty-argument combination when the wrapper later invokes a native executable. That later native-marshalling boundary already exists in the default wrapper and was explicitly outside the sibling feature's guarantee; the new fallback inherits it. The Rust recorder's optional status also means some runs will provide Python-only native-consumer evidence.

   **Recommendation:** label verifier results and exactness claims by the boundary actually exercised. Do not describe the fallback as guaranteeing arbitrary native argv fidelity solely because a PowerShell stub passed. Keep the existing separate-boundary limitation visible, and record a skipped Rust probe as a coverage limitation. This does not require expanding this wt feature into general PowerShell native-argument hardening.

3. **Minor — add one integrated cross-argument policy regression.** Revision 2 correctly requires the production check on the whole child command line and tests the percent predicate with a split pair, but its mocked direct-launch examples use `%A%` inside one value. A future implementation could accidentally apply the correct predicate separately to each argument and still pass those call-site examples.

   **Recommendation:** add a mocked direct-launch case with one `%` in each of two different child arguments and assert wrapper selection. Also combine that direct-launch fallback with a percent-pair wrapper directory and assert exactly one classic-console launch, no wt launch, and no wrapper/sidecar mutation. These are meaningful tests of the two new routing decisions and their composition.

4. **Nit — make the Base64 assertion and .NET wording precise.** “The tail's encoded argv contains no `%`, `;`, `"` or space” is true of the Base64 payload, not necessarily of the complete argv: the resolved wt executable path can contain spaces. Also, grouping .NET as generally “CommandLineToArgvW-based” is stronger than needed; runtime implementations and this Windows API are not a single parser contract.

   **Recommendation:** apply the alphabet assertion specifically to the `-EncodedCommand` payload, and state canonical-encoding compatibility for the supported/tested runtimes without claiming that they all call `CommandLineToArgvW`. Retain the separate outer-API and child-runtime test models already specified.

No blocker or major finding remains in revision 2. The revised design closes the identified wt delimiter, rebuild, and environment-expansion issues within its declared scope; the notes above can be handled during implementation and verifier construction.

VERDICT: APPROVED
