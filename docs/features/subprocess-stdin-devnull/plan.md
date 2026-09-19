# Plan: child processes must not inherit the MCP server's stdin

## Problem

Found in the Windows smoke of #61/#63 on `85e9aef` (2026-09-19).
`install_lead_wake()` (project scope) took ~10.5 s under a real stdio MCP
server and returned `git_ignore: "failed"` although the file was ignored. The
desktop host surfaced the call as "interrupted before a result was received".

Root cause: `server_simple._git()` calls `subprocess.run` without `stdin`, so
`git.exe` inherits the server's stdin — the JSON-RPC pipe the host holds open.
On Windows the first call (`git rev-parse --show-toplevel`) blocks until the
10 s `timeout` → `TimeoutExpired` → `"failed"`.

The event loop is *not* starved: `install_lead_wake` already dispatches
`_do_install` through `run_blocking` (`asyncio.to_thread`), so the cost is a
10 s latency on that one call plus a wrong `git_ignore` value, not a stalled
server. A `--timeout`-less git call would hang the worker thread forever, so
the 10 s bound is the only reason this is merely slow.

Evidence (`subprocess.run` instrumented inside a stdio server):

```
10.0s rc=TimeoutExpired stdin=None ['git', 'rev-parse', '--show-toplevel']
# same setup, stdin=DEVNULL forced for git:
0.0s rc=0 stdin=-3 ['git', 'rev-parse', '--show-toplevel']   → install 0.6 s, already_ignored
```

It does not reproduce in a plain script, an in-memory FastMCP client, a child
with an idle `stdin=PIPE`, or on Linux — so unit tests and CI stayed green.
PowerShell/taskkill children in the same server happened not to hang, but they
inherit the same pipe, and no *non-interactive* child in the package ever wants
the protocol stream: one that read it would also steal JSON-RPC bytes. The
exception is the two console-launch `Popen` paths, which inherit deliberately
so an interactive agent CLI gets a real console; they are out of scope below.

## Current behavior

AST scan of `src/claude_teams`: 20 literal `subprocess.run` sites, none passes
`stdin`/`input` — `procinfo.py` 1, `server_simple.py` 1 (`_git`), `codex.py` 1,
`pi.py` 1, `process_base.py` 2, `process_manager.py` 14 (taskkill, PowerShell
CIM queries, tmux, pgrep, herdr). No `asyncio.create_subprocess_*`,
`check_call`, `check_output`, or aliased call sites exist.

Of the 5 literal `Popen` sites (all in `process_manager.py`), 4 set `stdin`
deliberately and stay unchanged — including two that inherit **on purpose**:
the Windows `CREATE_NEW_CONSOLE` agent launch and the Linux terminal-emulator
launcher pass `stdin=None` so the interactive CLI gets a real console. Only the
Windows Terminal log tail (`process_manager.py:966`) omits it.

## Scope

In:
1. `stdin=subprocess.DEVNULL` on every `subprocess.run` in the package that
   does not pass `input=`. None of these children reads stdin by design (fixed
   argv utilities; `process_base.execute_in_pane`-style `cmd /c`/`sh -lc` runs
   captured, non-interactive, with a timeout).
2. `stdin=subprocess.DEVNULL` on the log-tail `Popen` (`process_manager.py:966`),
   pinned by extending its existing argv test — the ratchet does not cover
   `Popen`, so without that assertion the change could be reverted unnoticed.
3. A ratchet test, modelled on `tests/test_subprocess_decoding.py`. Every call
   spelled `subprocess.run` must use one of the two shapes that prove the
   child's stdin statically: literal `stdin=subprocess.DEVNULL`, or `input=`
   with a value that is not a literal `None`. Everything else fails —
   `stdin=None`, `input=None`, `stdin=0`/`False`/`sys.stdin`, a conditional,
   and any `**kwargs` splat — even where it might be correct at runtime.
   Stated limits: literal spelling only, `Popen` not covered.
4. A behavioral regression test for `_git`: monkeypatch `subprocess.run`,
   assert `_ensure_locally_ignored` passes `stdin=subprocess.DEVNULL` on every
   git call.

Out (follow-ups, noted in `implementation.md`):
- A real stdio-server integration test on Windows CI (no Windows CI job exists).
- Auditing `subprocess.run` sites that have no `timeout=`: with stdin detached
  they can no longer block on the protocol pipe, but an unbounded call still
  pins a `run_blocking` worker thread.

## Files affected

- `src/claude_teams/server_simple.py` (`_git`)
- `src/claude_teams/procinfo.py`
- `src/claude_teams/backends/{codex,pi,process_base,process_manager}.py`
- `tests/test_subprocess_stdin.py` (new)
- `tests/test_wake_local_settings.py` (git stdin assertion)
- `tests/test_backends/test_base_runtime.py` (log-tail `Popen` assertion; two
  exact-kwargs assertions gain the new keyword)
- `docs/features/subprocess-stdin-devnull/*`

## Risks

- A child that *did* rely on inherited stdin would now see EOF. Audit of the
  changed sites says none: tmux `send-keys`/`display-message`/`has-session`,
  `pgrep`, `taskkill`, PowerShell `-Command`, `codex debug models`,
  `pi --list-models`, `--version`, herdr CLI client calls, git plumbing.
  `powershell -Command <script>` with a redirected empty stdin is fine
  (`-Command -` is not used). The two console-launch `Popen` paths that *do*
  rely on it are deliberately left alone.
- Existing tests that assert exact `subprocess.run` kwargs may need the new
  keyword added.

## Test cases (red first)

1. `test_subprocess_stdin.py::test_every_subprocess_run_names_stdin[<file>]` —
   red for the 6 files listed above, green after.
2. Guard self-tests. Flagged: a bare call; a captured call; `stdin=None`;
   `input=None`; `stdin=0` / `False` / `sys.stdin`; a conditional `stdin=`; and
   any `**kwargs` splat, literal dict included. Not flagged: exactly
   `stdin=subprocess.DEVNULL`, `input=<not a literal None>`, and calls that are
   not `subprocess.run`.
3. `test_wake_local_settings.py::test_git_calls_detach_stdin` — red before the
   `_git` change.
4. `test_base_runtime.py::TestWindowsTerminalTail::test_opens_windows_terminal_by_default_when_available`
   gains a `stdin` assertion — red before the log-tail `Popen` change.
5. Full gate: `ruff format --check`, `ruff check`, `ty check`, `pytest`
   (Windows here; Linux via CI).
6. Manual: re-run the stdio smoke (`spawn_agent` → `install_lead_wake`) and
   expect < 2 s and `already_ignored`/`excluded`, not `failed`.
