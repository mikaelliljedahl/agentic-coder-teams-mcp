# Code review 2

## Verdict

CHANGES REQUESTED

The ownership blocker in `_pane_alive` is fixed, and the response now runs a
real child process for the procinfo decoder. The branch's focused and full test
results also reproduce as documented. The response is not ready, however:
the supposedly strict tmux spawn parser accepts Unicode decimal ids that tmux
can never emit or address, one round-1 regression test still cannot distinguish
the explicit encoding from a UTF-8 locale default, and two smoke-test phases
can pass or print a nominal result without exercising the behavior they claim.

## Review basis

- Reviewed commit: `ef196313d29840baafec9debfacdc8a8e872cc09`
- Merge base / `main`: `816d476f6791d3b9d17f06a1ea0834b241026c72`
- Diffs read: `git diff main...HEAD` and `git diff de57e2f..HEAD`
- Scope: only the files named in the review request, plus primary tmux source
  and manual pages needed to verify the external protocol grammar
- No codebase-memory graph lookup or indexing was performed.

## A. Round-1 finding dispositions

### 1. BLOCKER `_pane_alive`: RESOLVED

Evidence: `src/claude_teams/backends/process_manager.py:1456-1473` and
`:1687-1709`.

The production fix now maps `"1"` to dead, `"0"` to alive, and every other
decoded value to `False, "unreadable tmux pane status"`. That closes the
round-1 ownership bug: `_tracked_alive` at lines 1470-1473 returns that false
verdict and therefore cannot use unreadable text as proof that the tracked pane
is ours.

This does not turn a merely unreadable pane response directly into a false
health verdict while the PID is still live. `health_check` at lines 1456-1468
first calls `_pane_alive`, then falls back to `_pid_alive(handle)` and reports
`True, "process exists by pid"`. Ownership remains fail-closed while general
liveness retains the existing PID fallback. If the PID is also absent, the
health result is false, which is appropriate because neither pane nor process
liveness can then be established.

I found no real tmux response in the checked release history that would make a
valid target return exit code zero with localized text, an empty value, or
another status. tmux 1.8 formats `pane_dead` with `%d` from a boolean
([tmux 1.8 `format.c`](https://github.com/tmux/tmux/blob/1.8/format.c#L370-L403));
2.0 does the same
([tmux 2.0 `format.c`](https://github.com/tmux/tmux/blob/2.0/format.c#L599-L648));
3.0a emits `0` or `1`
([tmux 3.0a `format.c`](https://github.com/tmux/tmux/blob/3.0a/format.c#L2113-L2146));
and current tmux explicitly returns only `"0"` or `"1"`
([current `format.c`](https://github.com/tmux/tmux/blob/master/format.c#L2092-L2103)).
`display-message -p` supplies the ordinary line terminator that `.strip()`
removes; the format value itself is neither localized nor decorated. I found
no supported-version counterexample to the strict verdict.

Concrete fix: none required for production. The caller-level test gap is
reported separately under B.

### 2. MAJOR `_parse_tmux_spawn_output`: PARTIAL

Evidence: `src/claude_teams/backends/process_manager.py:24-27` and
`:1658-1685`; `tests/test_backends/test_base_runtime.py:1106-1139`.

The intended grammar is correct but the Python regex is not strict enough.
Across tmux 1.8, 2.0, 3.0a, and current source, window ids are emitted with
`@%u` and pane ids with `%%%u`, so the legitimate language is ASCII
`@[0-9]+` and `%[0-9]+`
([tmux 1.8 source](https://github.com/tmux/tmux/blob/1.8/format.c#L327-L398),
[current window id](https://github.com/tmux/tmux/blob/master/format.c#L3004-L3010),
[current pane id](https://github.com/tmux/tmux/blob/master/format.c#L2260-L2266)).
No legitimate id in those versions is rejected by the new patterns.

However, Python `\d` is Unicode-aware. The branch accepts
`@\u0661<TAB>%\u0662<TAB>+3` and returns the invalid target ids plus PID `3`.
`int()` likewise accepts Unicode decimal digits and a leading plus sign. That
is wider than tmux's machine protocol and can still register a spawn that later
cannot be addressed. The new tests cover U+FFFD and `-1`, but not this valid
Unicode / invalid-protocol class.

The `pid <= 0` check itself is sound. A successfully created pane's child PID
is positive; zero and negative values are not legitimate `pane_pid` results.
Current tmux only formats a pane PID while the pane fd is live
([current `pane_pid`](https://github.com/tmux/tmux/blob/master/format.c#L2387-L2393)).
Raising the existing `RuntimeError` is the right response to any impossible
value because the manager must not register it.

Concrete fix: compile the id patterns with ASCII digits (`@[0-9]+` and
`%[0-9]+`, or `re.ASCII`) and require `pid_text` to full-match `[1-9][0-9]*`
before conversion. Add cases for Unicode decimal digits, `+3`, and embedded
whitespace.

### 3. MAJOR procinfo decoder test: PARTIAL

Evidence: `tests/test_procinfo.py:220-247`, `:250-286`, and `:289-324`;
`docs/features/subprocess-utf8-decoding/implementation.md:70-74`.

`_run_helper_against_shim` now genuinely runs the decoder. It captures the real
`subprocess.run` before monkeypatching, and the replacement at lines 278-279
invokes that real function against a real Python child that writes raw bytes.
The invalid-byte case therefore fails if `errors="replace"` is removed.

Replacing `_split_windows_command_line` does not weaken the decoder claim: the
bytes still cross a real pipe, are decoded by `subprocess`, and are parsed by
`json.loads`. It does narrow the test to decoding and JSON handoff; it does not
prove Windows `CommandLineToArgvW` behavior. That limitation is reasonable for
a Linux-runnable decoder test and the test name
`test_windows_command_lines_decodes_real_utf8_bytes` is honest about what it
does.

The explicit encoding is still not behaviorally proved. Under UTF-8 mode, I
executed an in-memory mutation replacing `encoding="utf-8"` with `text=True`;
the real-child test still returned the intact non-ASCII argv successfully. The
separate call-shape assertion at lines 220-247 catches removal of the keyword,
and the implementation document admits the limitation, but that is not the
round-1 requested proof that the encoding changes behavior independently of
the host locale.

Concrete fix: run the helper in a nested interpreter whose default text
encoding is deliberately non-UTF-8/ASCII (for example with UTF-8 mode and
locale coercion disabled), while the shim emits UTF-8. The real-child test must
then fail when only `encoding="utf-8"` is removed. Keep the call-shape test as a
separate exact-contract check.

### 4. MAJOR AST guard: PARTIAL

Evidence: `tests/test_subprocess_decoding.py:1-24`, `:51-85`, and `:115-159`.

`_is_pipe` correctly recognizes `subprocess.PIPE` for the declared literal
scope, stderr-only capture is now included, and non-literal capture/text flags
are conservatively treated as candidates. The module docstring is now candid:
it says the guard covers only calls spelled `subprocess.run`, explicitly
excludes Popen and aliases, and calls itself a ratchet rather than a proof.

The new `_not_false` introduces real false positives. It returns true for every
AST expression except an absent keyword or literal `False`. Reproduced examples:

- `subprocess.run(a, capture_output=None, text=True)` is classified as capture.
- `subprocess.run(a, capture_output=0, text=True)` is classified as capture.
- `subprocess.run(a, capture_output=True, text=None)` is classified as text
  decoding.

All three calls are fine for this invariant: the first two do not capture, and
the third captures bytes rather than decoded text. The "does not cry wolf"
table at lines 144-159 does not cover the new literal-falsey cases.

Concrete fix: distinguish non-literal unknowns from statically false literals.
Treat `None`, `False`, numeric zero, and empty literal strings/containers as
false where the corresponding subprocess option uses truthiness, then add each
case to `test_guard_does_not_cry_wolf`.

### 5. MINOR failure-mechanics docstrings: PARTIAL

Evidence: `src/claude_teams/procinfo.py:208-237`;
`tests/test_subprocess_decoding.py:3-9`; `tests/test_procinfo.py:311-319`;
`docs/features/subprocess-utf8-decoding/implementation.md:8-21`.

The mis-decode description is technically sound. A raw U+001A inside a JSON
string is an unescaped control character, so `json.loads` rejects the payload;
the helper catches `JSONDecodeError` and returns `{}`, losing the whole table.

The Windows reader-thread description is also a real pre-fix CPython failure
mode: a decode exception can prevent the reader buffer from receiving a value,
leaving `stdout is None`, after which the old unconditional `.strip()` raised
`AttributeError`. But the current wording is incomplete and one current test
docstring is stale:

- On POSIX/select-based subprocess paths, strict decoding can raise
  `UnicodeDecodeError` in the caller instead of producing `stdout=None`.
- After the round-2 `(completed.stdout or "")` guard at `procinfo.py:232`,
  merely removing `errors="replace"` on Windows no longer makes the helper's
  `.strip()` raise; it returns `{}`. `test_procinfo.py:316-318` still says the
  current no-`errors` mutation raises `AttributeError`.

Concrete fix: qualify the `stdout=None`/`AttributeError` sequence as the old
Windows reader-thread behavior, state the caller-side strict-decode exception
mode for other runtimes, and update the invalid-byte test docstring to account
for the new null guard.

## B. New findings in the round-2 response

### BLOCKER

No BLOCKER findings.

### MAJOR

1. **MAJOR - the strict tmux id parser accepts Unicode ids outside tmux's grammar**

   `src/claude_teams/backends/process_manager.py:24-27, 1671-1684`

   `\d` and `int()` accept Unicode decimal forms and signs. A reproduced
   `@\u0661<TAB>%\u0662<TAB>+3` response is accepted even though tmux emits
   ASCII `%u` ids and a plain decimal PID. This preserves the original failure
   class for a different malformed input: a successful spawn can be registered
   with an unaddressable target.

   **Concrete fix:** use ASCII digit regexes for all three fields and add
   Unicode-digit/sign/whitespace rejection tests.

2. **MAJOR - smoke Phase 2 can pass against the broken main implementation**

   `docs/features/subprocess-utf8-decoding/smoke-test.md:93-117`

   The phase checks only whether argv tuples are empty. On this machine,
   `main` returned a populated process table whose argv was mojibake; Phase 2
   printed `argv empty: []`, satisfying its PASS criterion even though Phase 1
   proved the marker was corrupted. It also launches Python, not a node shim,
   and never calls `host_kind`, so it does not test the Pi-vs-Claude
   classification described above the command.

   **Concrete fix:** launch a process whose executable/name and script argv
   actually exercise the node-shim classification seam, then assert the exact
   marker and expected `host_kind`. At minimum, make any corrupted marker an
   explicit FAIL even when argv is non-empty.

3. **MAJOR - smoke Phase 4c never inspects tmux or a decoded production path**

   `docs/features/subprocess-utf8-decoding/smoke-test.md:184-202`

   `TmuxProcessManager` has no `list_sessions` method, so the command always
   prints `inspect manually`. It neither reads the non-ASCII window name nor
   invokes a production captured-output path. The stated PASS criterion
   (window name round-trips or is replaced) is therefore unobservable, and the
   stated `UnicodeDecodeError` FAIL cannot be produced by this command.

   **Concrete fix:** use a real production method that captures displayed tmux
   text, or run an explicit `tmux list-windows -F ...` through the same decode
   policy and assert its output. Remove the `hasattr(...)/inspect manually`
   fallback.

### MINOR

1. **MINOR - the ownership regression test never reaches ownership**

   `tests/test_backends/test_base_runtime.py:1141-1166`

   `test_unreadable_pane_status_does_not_prove_ownership` calls only
   `_pane_alive`. It proves helper behavior, but a mutation making
   `_tracked_alive` return `True` would leave this test green. The test name and
   docstring overstate the covered path.

   **Concrete fix:** construct a `TmuxProcessInfo` and assert
   `_tracked_alive(info)`/the ownership probe is false. Add a separate
   `health_check` assertion documenting the intentional live-PID fallback.

2. **MINOR - Phase 3 is source-shape theatre, not the advertised byte check**

   `docs/features/subprocess-utf8-decoding/smoke-test.md:119-137`

   The regex only proves that `OutputEncoding` occurs in the inspected source
   slice. It does not run PowerShell, inspect stdout bytes, prove the assignment
   precedes JSON emission, or detect a BOM/blank line. Phase 1's successful
   `json.loads` is the actual end-to-end check; it rejects a BOM but permits
   harmless JSON whitespace.

   **Concrete fix:** either delete/rename Phase 3 as a static sanity check, or
   execute the resolved PowerShell command and inspect raw stdout bytes for the
   expected first byte and absence of a BOM.

3. **MINOR - the exact Windows row-count evidence is not reproducible or stable**

   `docs/features/subprocess-utf8-decoding/implementation.md:94-100` and
   `docs/features/subprocess-utf8-decoding/smoke-test.md:77-91`

   A current reproduction returned 285 rows on this branch with the marker
   intact, and 287 rows against `main` with a populated but mojibake argv. It
   did not reproduce `291` versus `0`. Process-table counts are snapshots, and
   the pre-fix failure may be corruption without total-table loss depending on
   the active code page and bytes in the table.

   **Concrete fix:** label the table as a dated single-machine observation and
   make the invariant the marker's exact round-trip. Define FAIL as empty,
   missing, or corrupted argv, not only the one observed `0`/`None` shape.

### NIT

No NIT findings.

### Round-2 production guard judgment

`(completed.stdout or "").strip()` at `src/claude_teams/procinfo.py:232` is not
ordinary-path logic once `errors="replace"` is present: normal captured text is
a string, including `""`. It still earns its place. The helper's contract is
quiet degradation to an empty table, and a reader-thread failure unrelated to
Unicode decoding, a test double, or a future subprocess implementation can
still provide `None`. The guard is cheap and prevents the exact secondary
`AttributeError` that made the original incident opaque. No production finding
is raised for it.

The spawn validation test is behavioral because it drives `spawn_process` and
asserts the registry remains empty. The status test is only helper-level and
does not prove the ownership call path, as reported above. The AST helper tests
exercise the implementation but omit the new false-positive boundary.

## C. Smoke-test plan judgment

- Phase 1 is falsifiable and still detects the regression because PASS requires
  the exact marker. Its enumerated pre-fix FAIL output is too narrow: current
  `main` returned rows and argv but corrupted the marker.
- Phase 2 is not a valid downstream regression check; it passed against that
  corrupted `main` result and does not exercise a node host classifier.
- Phase 3 is a static source grep dressed as a raw-output check. The end-to-end
  parse in Phase 1 is stronger.
- Phase 4a genuinely needs no tmux binary: `_tmux_binary()` can return the
  fallback string and the mocked `subprocess.run` intercepts execution. The
  command produced `(False, "unreadable tmux pane status: '?'")` here.
- Phase 4b also needs no tmux: `_parse_tmux_spawn_output` is pure parsing.
- Phase 4c is not executable evidence for its claim because it always takes the
  `inspect manually` fallback.
- Phase 5 is sound. Against `main` it failed exactly the package rows for
  `procinfo.py`, `process_base.py`, and `process_manager.py`.
- The Windows commands are Bash/MSYS syntax (`cat`, `/tmp`, backslash line
  continuations). The plan should say to run them in Git Bash; they are not
  literal PowerShell commands despite being labeled only as Windows phases.

## D. Author-claim verification

### Red against main

The table at `implementation.md:78-88` reproduced. With
`PYTHONPATH=C:/code/github/win-agent-teams-mcp/agentic-coder-teams-mcp/src`, the
selected run produced 12 failures and 29 passes:

- UTF-8 call-shape test: `KeyError: 'encoding'`.
- Two real-child procinfo cases: `KeyError: 7`.
- Three spawn protocol cases: no `RuntimeError`.
- Three unreadable pane-status cases: reported alive.
- AST guard: failed on exactly `procinfo.py`, `process_base.py`, and
  `process_manager.py`.

One qualification: the two real-child tests fail against `main` because their
shim also requires the new child-side OutputEncoding prefix. That RED result
does not prove the parent-side `encoding="utf-8"` independently; the UTF-8-mode
mutation described in A3 stayed green without it.

### Current branch and gates

- Focused response set: 42 passed.
- `ruff format --check .`: 79 files already formatted, exit 0.
- `ruff check .`: all checks passed, exit 0.
- `ty check`: only the documented `tests/test_join_team.py:730`
  `unresolved-attribute`, exit 1.
- Full `pytest -q`: 1380 passed, 2 skipped, and only the documented
  `test_kill_agent_proceeds_when_the_holder_token_no_longer_matches` failed.

The gate claims are therefore current and accurate. The exact row-count table
is not reproducible as a stable result, as reported in B.

## Overall verdict

CHANGES REQUESTED
