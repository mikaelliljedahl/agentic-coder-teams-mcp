# Code review 3

## Verdict

CHANGES REQUESTED

The production tmux grammar and ownership fixes are now sound, and the branch's
runtime/gate evidence reproduces. I would still not merge this as the intended
last round: the new locale test does not isolate `encoding="utf-8"` (it removes
`errors=` at the same time and can skip), and the AST guard still has the
statically-falsey-container false positives that round 2 explicitly required it
to close. Two smoke/documentation claims also remain only partially repaired.

## Review basis and limitations

- Reviewed tip: `057b6ec` on `fix/subprocess-utf8-decoding`.
- Baseline: `main` at `816d476f6791d3b9d17f06a1ea0834b241026c72`.
- Read `git diff main...HEAD`, `git diff ef19631..HEAD`, both prior reviews,
  `implementation.md`, `smoke-test.md`, and the five named source/test files.
- No codebase-memory graph lookup or indexing was performed.
- The worktree was clean before this report was created.
- Native Linux execution was unavailable: WSL could not start because its VM
  feature is unavailable, and Docker's Linux engine was not running. The forced
  locale test was run on Windows and its Linux behavior was inspected, but I do
  not claim a native-Linux execution result.

## Open findings

### BLOCKER

No BLOCKER findings.

### MAJOR

1. **MAJOR - the locale regression does not isolate the explicit encoding and
   may turn into a permitted skip**

   `tests/test_procinfo.py:341-400`; `implementation.md:103-111`

   The test claims to prove the result changes when only
   `encoding="utf-8"` is removed, but the driver removes both `encoding` and
   `errors` at lines 365-368. The stripped run can therefore differ because
   strict decoding raises, rather than because the host-default decoder was
   selected. This does not meet round 1/2's requested mutation: remove only
   `encoding`, retain `errors="replace"`, and observe UTF-8 become locale-decoded
   replacement/mojibake.

   The test also calls `pytest.skip` at lines 396-398 if the nested interpreter
   remains UTF-8. On this Windows host the forced environment reported `cp1252`
   and the test passed; against `main` it failed with the documented mojibake.
   On a Linux runner where the environment cannot force a non-UTF-8 default,
   however, the only behavioral encoding proof becomes a CI-accepted skip.

   **Concrete fix:** in `strip` mode remove only `encoding`; leave
   `errors="replace"` in place. Make inability to establish the test's
   non-UTF-8 prerequisite a failure, or replace the platform prerequisite with
   a deterministic locale/default-encoding seam so the test cannot skip.

2. **MAJOR - `_not_false` still misclassifies statically falsey literal
   containers**

   `tests/test_subprocess_decoding.py:51-64, 148-174`

   `bool(value.value)` is correct for parsed `ast.Constant` values: `None`,
   `False`, numeric zero, and `""` are false; non-empty strings are true. It
   does not cover all literals. `()`, `[]`, and `{}` parse as `ast.Tuple`,
   `ast.List`, and `ast.Dict`, so `_not_false` returns `True` for each. Negative
   numbers parse as `ast.UnaryOp`; `-1` being treated as true is correct, but
   `-0` is another false positive. The current "does not cry wolf" table covers
   only `None` and numeric zero, not the empty containers required by round 2.

   I found no new false negative from the `ast.Constant` change: every constant
   it now dismisses is falsey to `subprocess` too. The remaining defect is false
   positives, not missed defended calls.

   **Concrete fix:** statically evaluate safe literal AST nodes (for example via
   guarded `ast.literal_eval`) and use their truthiness; leave names/calls/other
   unknown expressions conservative. Add empty tuple/list/dict (and `-0`)
   no-wolf cases.

### MINOR

1. **MINOR - the production comment still describes an impossible current
   `.strip()` failure**

   `src/claude_teams/procinfo.py:226-240`

   The invalid-byte test now correctly distinguishes old Windows reader-thread
   behavior from POSIX caller-side `UnicodeDecodeError`, but the production
   comment still says `stdout=None` makes "the `.strip()` below" raise. The
   code below is now `(completed.stdout or "").strip()`, so that statement is
   only true of the pre-fix implementation.

   **Concrete fix:** explicitly label the `AttributeError` sequence as the old
   behavior and state that the current null guard degrades it to `{}`.

2. **MINOR - smoke phase 2 still claims classification coverage it does not
   execute**

   `docs/features/subprocess-utf8-decoding/smoke-test.md:104-132`

   The important false-pass defect is fixed: the phase now requires the exact
   non-ASCII path, and it printed `True` on this branch and `False` against
   `main`. It still starts `python.exe`, not `node`, and never calls
   `host_kind`/`resolve_nearest_host`, despite the heading and explanation
   presenting it as a Pi-vs-Claude classification check.

   **Concrete fix:** either rename the phase and remove the classifier claim so
   it is honestly an argv round-trip probe, or launch/address the node seam and
   assert the resulting host classification.

3. **MINOR - smoke phase 3 still permits the leading blank output its heading
   says it rejects**

   `docs/features/subprocess-utf8-decoding/smoke-test.md:134-156`

   The revised phase now runs the real PowerShell command and inspects raw
   bytes, which closes most of round 2's finding. But `out.lstrip()[:1]` accepts
   any leading spaces or blank lines. That contradicts "no BOM, no blank line"
   and the stated requirement to inspect the expected first byte.

   **Concrete fix:** test `out[:1] in (b"[", b"{")` directly (alongside the BOM
   check), or rename the phase and explain that leading JSON whitespace is
   deliberately accepted.

### NIT

No NIT findings.

## A. Round-2 section A closeout

### A1. BLOCKER `_pane_alive`: RESOLVED

`src/claude_teams/backends/process_manager.py:1688-1710` accepts only stripped
`"0"`/`"1"`, fails every other successful response closed, and
`tests/test_backends/test_base_runtime.py:1150-1189` now exercises the actual
`_tracked_alive` ownership path. The PID fallback in `health_check` remains a
separate liveness result and does not become ownership proof.

### A2. MAJOR `_parse_tmux_spawn_output`: RESOLVED

`src/claude_teams/backends/process_manager.py:24-30, 1664-1686` now uses
ASCII-only `@[0-9]+`, `%[0-9]+`, and `[1-9][0-9]*` full matches. The seven
rejection cases at `tests/test_backends/test_base_runtime.py:1106-1148` cover
replacement characters, zero/negative PID, Unicode digits, and a sign.

The field grammar is now exact after framing whitespace is normalized. Direct
checks showed LF and CRLF both parse, as does harmless leading/trailing outer
whitespace because `output.strip()` runs before the tab split. Whitespace inside
a field and a leading-zero PID are rejected. That outer tolerance cannot store
an unaddressable id because the returned fields are already stripped. tmux's
`%u` ids and unpadded positive pane PID have no legitimate form rejected here.

### A3. MAJOR procinfo decoder test: PARTIAL

`tests/test_procinfo.py:268-327` still provides a real decoder test for UTF-8
and invalid bytes. The new test at `:341-400` passes on this branch and fails on
`main`, but lines 365-368 mutate both encoding and error handling, and lines
396-398 allow the required proof to skip. See MAJOR finding 1.

### A4. MAJOR AST guard: PARTIAL

The `None`/zero false positives are closed, and unknown non-literals remain
conservative, but empty literal containers are still false positives because
they are not `ast.Constant`. See MAJOR finding 2.

### A5. MINOR failure-mechanics wording: PARTIAL

`tests/test_procinfo.py:312-327` now accurately describes the Windows and POSIX
failure shapes. `src/claude_teams/procinfo.py:226-240` retains the stale
present-tense `.strip()` claim. See MINOR finding 1.

## B. Round-2 section B closeout and fresh 057b6ec review

### Round-2 BLOCKER

No BLOCKER finding existed.

### Round-2 MAJOR findings

1. **Strict tmux ids accept Unicode forms: RESOLVED.** Evidence:
   `process_manager.py:24-30, 1664-1686` and
   `test_base_runtime.py:1106-1148`.
2. **Smoke phase 2 can pass broken `main`: PARTIAL.** Exact-marker falsifiability
   is resolved and reproduced branch/main as `True`/`False`; the advertised
   downstream classification is still not called. See MINOR finding 2.
3. **Smoke phase 4c never inspects tmux/a production path: RESOLVED by code.**
   `smoke-test.md:209-235` now creates a real pane and calls production
   `_pane_alive` for live and bogus ids. I could not execute it without a native
   Linux/tmux runtime, so this status is source-verified rather than a live
   result. The non-ASCII window name is not itself in `_pane_alive`'s captured
   `#{pane_dead}` output, but the phase no longer claims to inspect the name.

### Round-2 MINOR findings

1. **Ownership test stops at `_pane_alive`: RESOLVED.**
   `test_base_runtime.py:1158-1189` constructs `TmuxProcessInfo` and asserts
   `_tracked_alive(info) is False`.
2. **Phase 3 is source-shape theatre: PARTIAL.** It is now a real raw-byte
   probe, but `lstrip()` does not prove the advertised absence of a leading
   blank line. See MINOR finding 3.
3. **Row-count evidence presented as invariant: RESOLVED.**
   `implementation.md:135-147` labels the counts dated observations and
   `smoke-test.md:78-94` makes exact marker preservation the invariant.

### Round-2 NIT

No NIT finding existed.

### Fresh findings in 057b6ec

No additional finding distinct from the PARTIAL round-2 items above. In
particular, the nested-driver construction is otherwise robust:

- `str(shim)!r` produces a valid path literal, including Windows backslashes.
- The environment override reaches both nested executions.
- The inner production helper already supplies a 10-second child timeout, and
  both real `subprocess.run` calls wait for their children; I found no ordinary
  leak or indefinite shim wait. The outer driver has no timeout, but there is no
  new unbounded operation outside ordinary interpreter startup/import.
- `check=True` is correct: an unexpected driver/import/crash must fail the test,
  not be parsed as evidence. The wrong-reason risk is the two-keyword mutation
  and skip path already reported in A3, not `check=True`.

## C. Whole-change judgment

### Re-audit of all 14 sweep sites

| Site | Output contract | Judgment |
|---|---|---|
| `process_base.py:146-153` | `--version`; output ignored | Replacement harmless; byte capture/no capture would also suffice. |
| `process_base.py:169-180` | user-visible command output | Replacement is appropriate loss-tolerant display behavior. |
| `process_manager.py:122-128` | `taskkill`; output ignored | Replacement harmless. |
| `process_manager.py:1257-1288` | PowerShell JSON plus exact ASCII token/PID selection | Machine-shaped, but fail-closed: corrupt JSON is rejected and corrupt command text cannot manufacture the exact token. |
| `process_manager.py:1370-1389` | whitespace-separated child PIDs | Machine-shaped, but each invalid token is skipped; U+FFFD cannot become a different PID. |
| `process_manager.py:1436-1445` | tmux spawn ids/PID | Correctly paired with strict full-field validation. |
| `process_manager.py:1499-1505` | tmux send response ignored | Replacement harmless. |
| `process_manager.py:1525-1534` | user-visible pane capture | Replacement appropriate. |
| `process_manager.py:1542-1548` | tmux literal-send response ignored | Replacement harmless. |
| `process_manager.py:1550-1556` | tmux Enter-send response ignored | Replacement harmless. |
| `process_manager.py:1690-1710` | exact `pane_dead` protocol/ownership | Correctly paired with strict `0`/`1` validation. |
| `process_manager.py:1715-1721` | tmux kill response ignored | Replacement harmless. |
| `process_manager.py:1761-1768` | session existence by return code | Output is irrelevant; replacement harmless. |
| `process_manager.py:2136-2143` | process existence by `pgrep` return code | Output is irrelevant; replacement harmless. |

I found no third unsafe protocol consumer. The JSON and child-PID sites are
machine-shaped, but their consumers reject/skip damaged values rather than
turning them into successful ownership/addressing answers. The two tmux sites
remain the only ones where decoded stdout directly establishes an address or
ownership fact, and both now validate.

### Linux/CI behavior and coverage

The sweep intentionally changes invalid-byte behavior on Linux at the base,
tmux, and `pgrep` call sites from possible `UnicodeDecodeError` to replacement.
The AST test proves every literal `subprocess.run` call names a policy, and the
procinfo shim exercises a real decoder; most individual sweep sites are not
given their own real-invalid-byte subprocess test. I found no wrong Linux
semantic consequence in the site audit, so this is a coverage limitation, not
an additional finding.

The locale test's skip branch is different: it can remove the only behavioral
proof of the explicit encoding on a Linux runner. That is already MAJOR finding
1. The claim in `implementation.md:156-157` that the two baseline failures "do
not appear in CI" was not independently checked within the named-file scope.

### Scope

The diff is large mainly because it commits two prior reviews plus the
implementation/smoke narrative. The production change remains localized to
one decoding invariant, the Windows procinfo encoding contract, and two strict
tmux consumers. It is still reviewable as one change; scope size is not a
finding.

## D. Author-claim verification

### Red against `main`

Running the complete Evidence-table selection with `main`'s `src` on
`PYTHONPATH` produced **17 failed, 32 passed**:

- four procinfo failures: `KeyError: 'encoding'`, two `KeyError: 7` results, and
  the documented `['pi', 'â‰¥'] != ['pi', '≥']` locale mismatch;
- all **7** spawn cases failed because no `RuntimeError` was raised;
- all **3** unreadable pane cases were reported alive;
- the package guard failed on exactly `process_base.py`, `process_manager.py`,
  and `procinfo.py`.

The same selected set on this branch produced **49 passed**.

The locale test alone was also run with
`PYTHONUTF8=0`, `PYTHONCOERCECLOCALE=0`, `LC_ALL=C`, and `LANG=C`. Its nested
Windows encoding was `cp1252`, so it did not skip and passed. The same test
against `main` failed at the documented mojibake comparison. Native-Linux
execution could not be performed for the environment reason stated above.

### Live Windows smoke evidence

- Phase 1, branch: 255 rows, exact marker argv, `MARKER kept : True`.
- Phase 1, `main`: 254 rows, mojibake argv, `MARKER kept : False`.
- Phase 2, branch/main: `script argv intact` was `True` / `False`.
- Phase 3: `BOM : False`, `starts JSON: True`, first bytes `b'[{"Proce'`.

This confirms the document's corrected invariant and its warning that a broken
`main` can return a populated but mangled table rather than zero rows.

### Gates

- `ruff format --check .`: **79 files already formatted**, exit 0.
- `ruff check .`: **All checks passed**, exit 0.
- `ty check`: the single documented
  `tests/test_join_team.py:730 unresolved-attribute`, exit 1. Checking tracked
  `src` and `tests` against `main` produced the same sole diagnostic.
- `pytest -q`: **1388 passed, 2 skipped, 1 failed**. The only failure was the
  documented
  `test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`.
- That test was run separately with `main`'s sources and failed at the same
  `assert result["success"] is True`, confirming it as baseline.

## Overall verdict

CHANGES REQUESTED
