# Code review 1

## Verdict

CHANGES REQUESTED

The UTF-8 pinning in `_windows_command_lines` is sound, but the blanket
`errors="replace"` policy is unsafe for two tmux machine-protocol responses.
One of those responses feeds the ownership proof and can now turn corrupt
output into a false "pane running" answer. The package sweep found no current
captured-and-decoded call omitted by the commit, but the new AST guard has
several false negatives and the procinfo test does not execute a decoder.

## Findings

### BLOCKER

1. **BLOCKER - replacement decoding can turn corrupt tmux status into a false ownership proof**

   `src/claude_teams/backends/process_manager.py:1670`

   `_pane_alive` now decodes `tmux display-message ... "#{pane_dead}"` with
   `errors="replace"`, but lines 1677-1681 treat every return-code-zero response
   other than the exact string `"1"` as `True, "tmux pane running"`. An
   undecodable byte therefore becomes U+FFFD, misses the `== "1"` check, and is
   accepted as alive. This is worse than the former loud failure. It is also an
   ownership bug, not just a health-display bug: `_tracked_alive` at lines
   1467-1470 uses this result as proof that the in-memory tmux process is ours.

   **Concrete fix:** treat this output as a strict protocol. Decode strictly
   (ASCII or UTF-8), accept only exactly `"0"` or `"1"`, and make any other
   value a failure/indeterminate result that cannot prove ownership. Add a test
   in which the command returns invalid bytes or unexpected text with exit code
   zero and assert that `_tracked_alive`/ownership does not return ours.

### MAJOR

2. **MAJOR - replacement decoding can register a tmux spawn with a corrupted target ID**

   `src/claude_teams/backends/process_manager.py:1427`

   The spawn response is a three-field machine protocol, but
   `_parse_tmux_spawn_output` at lines 1655-1666 validates only the field count
   and the PID. With `errors="replace"`, an invalid byte in the window or pane
   field can become a replacement character and still be stored as
   `target_id`/`pane_id`; spawn then reports success even though later health,
   send, and kill commands cannot address that pane. A replacement inside the
   PID does fail `int()`, so this path cannot synthesize a different PID, but it
   can silently create an unmanageable spawned agent.

   **Concrete fix:** decode this protocol strictly and validate all three
   fields (`@<digits>`, `%<digits>`, and a positive decimal PID) before
   registering the process. Convert a decode/validation failure to the existing
   explicit `RuntimeError` spawn failure rather than replacing bytes.

3. **MAJOR - the procinfo regression test never runs subprocess decoding**

   `tests/test_procinfo.py:218`

   The test replaces `subprocess.run` and returns an already-decoded `str` at
   lines 231-239. It can verify call construction, but it cannot produce a
   reader-thread decode failure, prove that UTF-8 bytes are decoded as UTF-8,
   or detect BOM/leading-output behavior. Its replacement character is already
   present in the Python string. The test would *not* pass if either current
   half were simply dropped: removing `encoding="utf-8"` fails line 243, and
   removing the OutputEncoding prefix fails line 246. It is therefore not a
   pure tautology, but its name and docstring claim a behavioral regression
   test that it does not provide. The `"UTF8" in argv` assertion is also much
   weaker than checking the exact command prefix.

   **Concrete fix:** add a Linux-runnable integration test using an executable
   `tmp_path` shim in place of `powershell.exe`, while leaving
   `subprocess.run` real. The shim should require the exact OutputEncoding
   prefix and write raw JSON bytes containing both valid UTF-8 non-ASCII text
   and an invalid byte inside a quoted command-line value. Force the default
   subprocess text encoding to ASCII in the test (for example via the runtime's
   locale/text-encoding seam) so dropping explicit `encoding="utf-8"` changes
   the result; dropping `errors="replace"` must raise; dropping the child prefix
   must make the shim fail. Assert the parsed argv. Keep a small separate unit
   assertion for the exact PowerShell command if desired.

4. **MAJOR - the package-wide AST guard silently misses captured decoders**

   `tests/test_subprocess_decoding.py:22`

   The advertised whole-package guard recognizes only the literal spelling
   `subprocess.run`, only `stdout=subprocess.PIPE` (not a stderr-only pipe), and
   only literal `True` for `capture_output`, `text`, and
   `universal_newlines` (lines 22-57). Consequently each of these escapes it:

   - `subprocess.run(..., stderr=subprocess.PIPE, text=True)`
   - an imported or aliased `run`/`PIPE`
   - `capture_output=SOME_TRUE_VALUE` or `text=SOME_TRUE_VALUE`
   - a captured/decoded `subprocess.Popen`

   These false negatives were reproduced directly against the helper. No
   current package call is missed by this commit's sweep, but the guard does
   not enforce the invariant its module docstring claims.

   `Popen(..., stdin=PIPE, text=True)` in `process_manager.py` is deliberately
   outside this output-decoding change because it controls encoding on writes,
   not a captured output stream. That exclusion is reasonable and is stated in
   `implementation.md:75`, but the guard's own opening claim, "Every captured
   subprocess stream", is not honest about only inspecting `run` calls.

   **Concrete fix:** at minimum recognize stderr pipes and document the exact
   `subprocess.run`-only scope. Prefer resolving the import aliases used in the
   package and conservatively treating non-literal capture/text keyword values
   as candidates. If Popen output capture remains deliberately excluded, say
   so in this test module and test name. Also change the failure message at
   lines 79-82 to require an explicit call-site-appropriate policy, rather than
   prescribing `replace` for strict machine protocols.

### MINOR

5. **MINOR - the tests describe the wrong Windows failure mechanics**

   `tests/test_subprocess_decoding.py:3`

   Both this module (lines 3-8) and the procinfo test docstring
   (`tests/test_procinfo.py:221`) say the `UnicodeDecodeError` surfaces from
   `subprocess.run`. In a direct Windows CPython 3.12 reproduction, the reader
   thread printed `UnicodeDecodeError`, `subprocess.run` returned with
   `stdout=None`, and the next `.strip()` in `_windows_command_lines` would
   raise `AttributeError`. The end-user outcome is still a crashed helper and
   the production fix still prevents it, but the stated causal contract is not
   what the target runtime did.

   **Concrete fix:** describe both relevant modes accurately: strict decoding
   can raise in the caller on runtimes/code paths that decode there, while the
   Windows reader-thread path can lose the stream and cause the subsequent
   consumer to fail. The real-child regression proposed above should lock down
   the behavior the code actually needs, rather than an assumed exception
   propagation detail.

### NIT

No NIT findings.

## Required judgments

### 1. PowerShell UTF-8 pinning

No finding. The command at `src/claude_teams/procinfo.py:208` sets
`[Console]::OutputEncoding` before the pipeline writes JSON, and the parent
uses matching `encoding="utf-8"` at line 226. A direct redirected-stdout probe
on this Windows host produced BOM-free JSON beginning with byte `0x7b` (`{`),
with no leading blank line, under both Windows PowerShell 5.1
(`powershell.exe`, version 10.0.26100.8875) and pwsh 7.6.5. The assignment itself
does not emit a pipeline object. Even a leading CR/LF would be JSON whitespace;
the material concern was a BOM, and neither runtime emitted one.

The helper explicitly searches for `powershell.exe`, not `pwsh.exe`, at lines
198-206. Therefore running the server from pwsh does not change which engine
this helper launches; on normal Windows it still launches Windows PowerShell
5.1. If `powershell.exe` is not on PATH it falls back to the pre-existing
System32 Windows PowerShell path. A missing fallback executable still yields
`{}` through the existing `OSError` guard. I found no new case introduced by
the prefix where a previously successful CIM query becomes empty.

### 2. All 14 other `errors="replace"` sites

| Site | Decoded output use | Assessment |
|---|---|---|
| `process_base.py:146` | `wait_idle`; stdout/stderr ignored | Replacement is harmless; decoding could be omitted entirely. |
| `process_base.py:169` | Returned as user-visible command output | Replacement is appropriate loss-tolerant display behavior. |
| `process_manager.py:113` | `taskkill`; output ignored | Harmless; no PID is parsed from output. |
| `process_manager.py:1248` | PowerShell JSON, then ASCII correlation-token and PID match | Replacement can cause a fail-closed miss/JSON rejection; it cannot manufacture the token or a different numeric PID. |
| `process_manager.py:1361` | Whitespace-separated child PIDs | U+FFFD cannot become a digit; a corrupted token is skipped, not converted to a wrong PID. |
| `process_manager.py:1427` | tmux window ID, pane ID, PID protocol | **MAJOR finding 2:** malformed IDs can be accepted. |
| `process_manager.py:1490` | tmux Ctrl-C response ignored | Harmless. |
| `process_manager.py:1516` | User-visible captured pane text | Replacement is appropriate loss-tolerant display behavior. |
| `process_manager.py:1533` | tmux literal-send response ignored | Harmless. |
| `process_manager.py:1541` | tmux Enter-send response ignored | Harmless. |
| `process_manager.py:1670` | Exact `pane_dead` status, then ownership | **BLOCKER finding 1:** malformed output becomes alive/ours. |
| `process_manager.py:1686` | tmux kill response ignored | Harmless. |
| `process_manager.py:1732` | Session existence uses return code only | Harmless. |
| `process_manager.py:2107` | `pgrep` existence uses return code only | Harmless. |

There are no `wmic` or `tasklist` calls in the reviewed package version; the
Windows process queries here use PowerShell CIM, and the only `taskkill` output
is ignored.

### 3. Sweep completeness and Popen scope

A narrow package search found all `subprocess.run` and `subprocess.Popen`
sites. The two pre-existing discovery calls in `codex.py` and `pi.py` already
pin UTF-8 with replacement. No current `run` call captures and decodes a stream
without an `errors` policy. No current Popen captures stdout/stderr into a
decoded pipe: its outputs are inherited, discarded, or redirected directly to
files. The `stdin=PIPE, text=True` Popen path at
`process_manager.py:631-645` is intentionally a write-encoding issue and was
not missed by this output-decoding sweep. The AST guard's future false
negatives and scope wording are finding 4.

### 4. Test quality

The focused tests pass (`26 passed` for `tests/test_subprocess_decoding.py` plus
the new procinfo test), but the procinfo test is only a kwargs/command-shape
test. Finding 3 gives a Linux-runnable real-subprocess replacement.

### 5. Updated exact-kwargs test doubles

No finding. The updates in `tests/test_backends/test_base_runtime.py:193` and
`:1225` correctly keep existing exact invocation-contract assertions aligned
with production. Those mocks return strings or ignore output, so they neither
prove nor conceal decoder behavior; the problem is relying on the separate
fake-run/AST tests as behavioral coverage, addressed in findings 3 and 4.

## Verification notes

- Reviewed commit: `de57e2f850f3a7604ee4cfba560743342416bcc6`
- Diff basis: `git diff main...HEAD`
- Direct PowerShell byte probes: Windows PowerShell 5.1 and pwsh 7.6.5,
  redirected stdout, exact command prefix, no BOM/leading blank output
- Focused tests: `26 passed in 16.96s`
- The worktree's pre-existing uncommitted `implementation.md` rewrite and
  untracked `smoke-test.md` were treated as context only, not as part of
  `main...HEAD`.

## Overall verdict

CHANGES REQUESTED
