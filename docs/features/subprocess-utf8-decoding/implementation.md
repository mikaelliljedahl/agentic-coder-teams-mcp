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

Codex's remaining suggestion — forcing the process' default text encoding to
ASCII so that dropping `encoding="utf-8"` fails on a UTF-8 Linux runner — was
not implemented. On Linux the real-subprocess tests prove `errors=`; `encoding=`
is pinned by the call-shape test. This is stated in the test docstrings rather
than papered over.

## Evidence

Red first. Every test below was run against `main`'s sources via `PYTHONPATH`
and fails there:

| test | red against `main` |
|---|---|
| `test_windows_command_lines_asks_the_child_for_utf8` | `KeyError: 'encoding'` |
| `test_windows_command_lines_decodes_real_utf8_bytes` | `KeyError: 7` |
| `test_windows_command_lines_survives_an_undecodable_byte` | `KeyError: 7` |
| `test_spawn_refuses_a_tmux_target_it_could_never_address` (×3) | no `RuntimeError` |
| `test_unreadable_pane_status_does_not_prove_ownership` (×3) | reported alive |
| `test_subprocess_decoding.py` | fails on `procinfo.py`, `process_base.py`, `process_manager.py` |

The two `_run_helper_against_shim` tests drive `_windows_command_lines` with a
**real child process** and a real `subprocess.run`; only the executable and the
Windows-only argv splitter are substituted, so they run on Linux too.

Behavioural check on Windows — a child was started whose argv carries `≥ → é 中`,
then the live process table was read back (`smoke-test.md`, phase 1):

| sources | rows returned | child's argv | marker intact |
|---|---|---|---|
| `main` | 0 | `None` | no |
| this branch | 291 | full argv | yes |

Two test doubles in `tests/test_backends/test_base_runtime.py` assert the exact
kwargs of a `subprocess.run` call and were updated to include
`errors="replace"`.

## Gates

`ruff format --check`, `ruff check`, `ty check` and `pytest` were run over the
whole repository on Windows. All green except two **pre-existing, Windows-only**
results that also fail on `main` and do not appear in CI:

- `tests/test_follow_up_delivery.py::test_kill_agent_proceeds_when_the_holder_token_no_longer_matches`
- `ty` diagnostic `unresolved-attribute` on `tests/test_join_team.py:730`

## Not fixed here

`Popen(..., text=True)` with `stdin=PIPE` in `process_manager.py` encodes what
we *write* to a child with the locale encoding too. That is the likely source
of the em-dash corruption noted in the Pi messaging work. It changes spawn
behaviour, so it needs its own branch and its own smoke test.
