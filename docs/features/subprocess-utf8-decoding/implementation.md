# Captured subprocess output must decode without raising

## Defect

`subprocess.run(..., capture_output=True, text=True)` decodes the child's
streams with the *locale* encoding — cp1252 on a Swedish Windows. An
undecodable byte raises `UnicodeDecodeError` from subprocess' reader thread,
and that exception surfaces from the `subprocess.run` call itself, so the
`except (OSError, subprocess.SubprocessError)` guards at these call sites do
not catch it.

Observed during the PR #53 smoke test: one non-ASCII byte in a running
process' command line took down a whole `deliver_pending()` call from inside
`procinfo._windows_command_lines()`.

The same helper also *silently mangled* command lines that did decode:
Windows PowerShell writes a redirected stream in the console code page, and
`procinfo` compares argv to decide process ownership, so a mangled argv is a
wrong answer rather than a loud one.

## Change

1. `procinfo._windows_command_lines` pins **both** ends to UTF-8: the child is
   told `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`, and the
   parent decodes `encoding="utf-8", errors="replace"`.
2. Every other `subprocess.run` in the package that captures *and* decodes a
   stream gains `errors="replace"`. Their locale decoding is unchanged — this
   only removes the ability to raise. (`codex.py` and `pi.py` already had an
   explicit UTF-8 policy and were left alone.)

## Evidence

Red first:

- `tests/test_procinfo.py::test_windows_command_lines_decodes_utf8_and_never_raises`
  failed with `KeyError: 'encoding'` before the fix.
- `tests/test_subprocess_decoding.py` — an AST guard over the whole package —
  failed on `procinfo.py`, `process_base.py` and `process_manager.py` when run
  against `main`'s sources, and passes against this branch.

Behavioural check on Windows, same live process table:

| sources | rows | argv parts containing non-ASCII |
|---|---|---|
| `main` | 284 | 0 (characters lost) |
| this branch | 278 | 2 (`≥` preserved) |

Two test doubles in `tests/test_backends/test_base_runtime.py` asserted the
exact kwargs of a `subprocess.run` call and were updated to include
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
