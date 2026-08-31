# Captured subprocess output must decode without raising

## Defect

`subprocess.run(..., capture_output=True, text=True)` decodes the child's
streams with the *locale* encoding — cp1252 on a Swedish Windows.

**The failure is silent, which is worse than the crash it looks like.** Two
modes were reproduced on this machine, and neither one raises where a caller
would notice:

1. **Mis-decode.** PowerShell writes a redirected stream in the console code
   page. A planted argv of `≥→é中` came back as `=\x1a‚?` — and `\x1a` (SUB) is
   a control character, which `json.loads` refuses inside a string. One process
   with a non-ASCII command line therefore discards the **entire 170 KB process
   table**: `_windows_command_lines` returns `{}` and
   `resolve_nearest_host` reports every ancestry entry with `argv=()`.
2. **Undecodable byte.** subprocess' reader thread dies, `completed.stdout`
   comes back `None`, and the helper's own `.strip()` raises `AttributeError`
   past an `except (OSError, subprocess.SubprocessError)` guard written for a
   different failure.

Everything built on argv — the node-shim rule that tells a `node.exe` running
Pi apart from one running Claude Code — degrades to a guess, and nothing says
why. Observed during the PR #53 smoke test as a `Thread-N (_readerthread)`
traceback in the middle of a `deliver_pending()` call. That call completed; it
was `procinfo` that went blind.

## Change

1. `procinfo._windows_command_lines` pins **both** ends to UTF-8: the child is
   told `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`, and the
   parent decodes `encoding="utf-8", errors="replace"`. Its `.strip()` also no
   longer assumes `stdout` is a string.
2. Every other `subprocess.run` in the package that captures *and* decodes a
   stream gains `errors="replace"`. Their locale decoding is unchanged — this
   only removes the ability to raise. (`codex.py` and `pi.py` already had an
   explicit UTF-8 policy and were left alone.)
3. **The two tmux machine protocols are validated, not just decoded** — see
   "Review round 1" below. `errors="replace"` alone would have converted a loud
   failure into a wrong answer at those two sites.

## Review round 1

`docs/features/subprocess-utf8-decoding/code-review-1.md` (Codex, adversarial):
CHANGES REQUESTED. It confirmed the PowerShell UTF-8 pinning against real
byte-level probes on both Windows PowerShell 5.1 and pwsh 7.6.5 (no BOM, no
leading blank line), and audited all 14 `errors="replace"` sites individually.
Two of them were wrong, and both are fixed here:

- **BLOCKER — `TmuxProcessManager._pane_alive`.** It read *anything that is not
  `"1"`* as "pane running", and `_tracked_alive` uses that as **proof of
  ownership**. A replacement character would therefore have claimed ownership
  of a pane we cannot address — strictly worse than the crash it replaced.
  `#{pane_dead}` is now treated as the machine protocol it is: exactly `"0"` or
  `"1"`, anything else fails closed as `unreadable tmux pane status`.
- **MAJOR — `_parse_tmux_spawn_output`.** It counted three fields and validated
  only the PID, so a corrupted window or pane id could be registered. Spawn
  would report success and leave an agent that can never be health-checked,
  signalled or killed. All three fields are now validated (`@\d+`, `%\d+`,
  positive PID).
- **MAJOR — the procinfo test never ran a decoder.** Replaced by a
  real-subprocess test (below).
- **MAJOR — the AST guard had false negatives** (stderr-only pipes, non-literal
  `True`) and a docstring claiming more scope than it had. Both fixed, and the
  guard now has tests of its own proving it catches those shapes.
- **MINOR — the docstrings described the wrong failure mechanics.** Corrected
  above and in the tests; both modes are now stated.

## Review round 2

`code-review-2.md` (Codex, adversarial, on the round-1 response): CHANGES
REQUESTED again. It confirmed the `_pane_alive` blocker RESOLVED — including
checking tmux's own `format.c` across 1.8, 2.0, 3.0a and master to establish
that `#{pane_dead}` really is only ever `0` or `1`, so failing closed breaks no
legitimate caller — and reproduced the whole red-against-`main` table. Four
things were still wrong:

- **MAJOR — the "strict" tmux id parser was not strict.** Python's `\d` is
  Unicode-aware and `int()` accepts a sign, so `@١  %٢  +3` was *accepted* and
  registered a target that can never be addressed. The same failure class the
  round-1 fix was supposed to close, through a different input. Now ASCII-only:
  `@[0-9]+`, `%[0-9]+`, and a PID that must full-match `[1-9][0-9]*`.
- **MAJOR — smoke phase 2 could pass against broken `main`.** It only checked
  that argv was non-empty, and on this machine `main` returned a populated
  table full of mojibake. Rewritten to assert the exact non-ASCII script path
  round-trips, and verified to print `True` on this branch and `False` against
  `main`.
- **MAJOR — smoke phase 4c tested nothing.** `TmuxProcessManager` has no
  `list_sessions`, so my `hasattr` fallback always printed "inspect manually".
  Replaced with a real pane driven through the production `_pane_alive`.
- **MINOR — the ownership test never reached ownership** (it called
  `_pane_alive`, not `_tracked_alive`), **phase 3 was source-grep theatre**
  (now runs PowerShell and inspects the raw first bytes for a BOM), and the
  **row-count evidence was presented as an invariant** when it is a dated
  single-machine snapshot.

It also found a genuine false positive I introduced: `_not_false` flagged
`capture_output=None` and `capture_output=0`, which are "off". The guard now
distinguishes statically-falsey literals from non-literal unknowns, and the
"does not cry wolf" table covers those cases.

**The encoding is now proved behaviourally.** Round 1 asked for a test that
fails when only `encoding="utf-8"` is dropped, independent of the host locale;
round 1's response admitted it did not have one, and round 2 demonstrated the
gap by mutating the keyword away and watching the test stay green.
`test_explicit_encoding_is_what_decodes_utf8_not_the_host_locale` re-runs the
scenario in a nested interpreter forced to an ASCII default
(`PYTHONUTF8=0`, `PYTHONCOERCECLOCALE=0`, `LC_ALL=C`) and asserts the two
answers differ. Against `main` it fails with
`['pi', 'â‰¥'] != ['pi', '≥']` — the mojibake, named.

## Review round 3

`code-review-3.md`: CHANGES REQUESTED. It closed out the tmux work as RESOLVED
— checking that `output.strip()` before the tab split makes both LF and CRLF
parse, and that no legitimate tmux id is now rejected — and re-audited all 14
`errors="replace"` sites independently rather than trusting round 1's table,
finding no third machine protocol. Four things were still open:

- **MAJOR — the locale test did not isolate the encoding.** Its mutation
  removed `encoding` *and* `errors` together, so the stripped run could differ
  merely by raising rather than by picking the locale decoder. It now removes
  `encoding` only; `errors="replace"` stays, so the sole variable is which
  decoder subprocess chooses. Verified: still passes here, still fails against
  `main` with `['pi', 'â‰¥'] != ['pi', '≥']`.
- **MAJOR — that test could silently skip.** A `pytest.skip` when the non-UTF-8
  default could not be forced would have deleted the only behavioural proof of
  the encoding half on exactly the runner (Linux) where it matters. It is now a
  **failure** with a message naming what to fix. `LC_CTYPE=C` was added to the
  forcing.
- **MAJOR — `_not_false` still had false positives.** `bool(value.value)` only
  covers `ast.Constant`; `()`, `[]`, `{}` and `-0` parse as other node types and
  were flagged. Now `ast.literal_eval` with a conservative fallback, and the
  "does not cry wolf" table covers all of them.
- **MINOR ×3** — the production comment still described a `.strip()` failure the
  null guard has since made impossible; smoke phase 2's heading claimed a
  classification check it does not perform (it is an argv round-trip probe, and
  now says so, pointing at the two tests that do cover the classifier); and
  phase 3's `out.lstrip()[:1]` accepted the leading blank line its own heading
  promised to reject. All three corrected.

Round 3 could not execute on native Linux (no WSL, no Docker), so its
tmux-on-Linux verification is source-level. **The smoke test's phase 4 has not
been run on Linux by any round.**

## Evidence

Red first. Every test below was run against `main`'s sources via `PYTHONPATH`
and fails there:

| test | red against `main` |
|---|---|
| `test_windows_command_lines_asks_the_child_for_utf8` | `KeyError: 'encoding'` |
| `test_windows_command_lines_decodes_real_utf8_bytes` | `KeyError: 7` |
| `test_windows_command_lines_survives_an_undecodable_byte` | `KeyError: 7` |
| `test_explicit_encoding_is_what_decodes_utf8_not_the_host_locale` | `['pi', 'â‰¥'] != ['pi', '≥']` |
| `test_spawn_refuses_a_tmux_target_it_could_never_address` (×7) | no `RuntimeError` |
| `test_unreadable_pane_status_does_not_prove_ownership` (×3) | reported alive |
| `test_subprocess_decoding.py` | fails on `procinfo.py`, `process_base.py`, `process_manager.py` |

The `_run_helper_against_shim` tests drive `_windows_command_lines` with a
**real child process** and a real `subprocess.run`; only the executable and the
Windows-only argv splitter are substituted, so they run on Linux too. The
locale test goes further and runs the whole thing in a nested interpreter whose
default text encoding is forced to ASCII, which is what makes it independent of
the runner's own locale.

Behavioural checks on Windows, `smoke-test.md` phases 1–3. These are **dated
single-machine observations (2026-08-30)**, not stable invariants — the process
table is a snapshot and the pre-fix damage takes more than one shape:

| check | `main` | this branch |
|---|---|---|
| planted argv `≥ → é 中` round-trips (phase 1) | no — 0 rows, `argv: None` | yes |
| planted non-ASCII node script path survives (phase 2) | `False` | `True` |
| PowerShell output has no BOM and starts with `[` (phase 3) | n/a | `True` |

A second run on the same machine produced 287 rows against `main` with argv
present but *mangled*, rather than 0 rows. The invariant that holds either way
is the marker round-trip, which is what the smoke test now judges on.

Two test doubles in `tests/test_backends/test_base_runtime.py` assert the exact
kwargs of a `subprocess.run` call and were updated to include
`errors="replace"`.

## Gates

`ruff format --check`, `ruff check`, `ty check` and `pytest` were run over the
whole repository on Windows. All green except two **pre-existing, Windows-only**
results that also fail on `main` and do not appear in CI:

- `tests/test_follow_up_delivery.py::test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`
- `ty` diagnostic `unresolved-attribute` on `tests/test_join_team.py:730`

## Known risk before merge

`test_explicit_encoding_is_what_decodes_utf8_not_the_host_locale` now **fails**
rather than skips if it cannot force a non-UTF-8 default in its nested
interpreter. That is deliberate — a skip would silently remove the only
behavioural proof of the encoding — but it has only been exercised on Windows,
where the default is cp1252 anyway. On Linux it relies on `LC_ALL=C` plus
`PYTHONUTF8=0` and `PYTHONCOERCECLOCALE=0` yielding an ASCII default. **Run the
suite on the Lubuntu VM before merging.** If that assumption does not hold, the
test fails loudly with a message naming the fix, which is the intended
behaviour — but it would be a red CI run.

## Not fixed here

`Popen(..., text=True)` with `stdin=PIPE` in `process_manager.py` encodes what
we *write* to a child with the locale encoding too. That is the likely source
of the em-dash corruption noted in the Pi messaging work. It changes spawn
behaviour, so it needs its own branch and its own smoke test.
