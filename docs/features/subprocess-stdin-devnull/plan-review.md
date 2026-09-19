# Independent plan review: subprocess stdin detachment

VERDICT: APPROVED WITH CHANGES

The proposed production change is directionally correct: every affected `subprocess.run` call is non-interactive, and detaching it from the stdio MCP server's JSON-RPC input is safer than inheriting that stream. The plan is not ready unchanged because its event-loop diagnosis and call-site counts do not match the source, and the planned tests do not fully protect the stated invariant or the sole `Popen` change.

1. **MAJOR — The claimed event-loop freeze and the `asyncio.to_thread` follow-up are contradicted by the current source.**

   The plan says the synchronous Git timeout freezes the whole event loop and lists moving this work to `asyncio.to_thread` as future work (`plan.md:13-14,54-56`). In fact, `install_lead_wake()` returns `await run_blocking(_do_install)` (`src/claude_teams/server_simple.py:6775`), and `run_blocking()` already delegates to `await asyncio.to_thread(...)` (`src/claude_teams/async_utils.py:7-21`). `_ensure_locally_ignored()` and `_git()` run inside `_do_install`, so a blocked Git child delays this tool call and occupies a worker thread, but it does not block the MCP event loop. Keep the observed 10-second latency/root child-stdin diagnosis, but correct the event-loop claim and remove or narrow the already-completed follow-up.

2. **MAJOR — The Windows Terminal tail `Popen` change has no mutation-sensitive test.**

   Scope item 2 changes the only `Popen` that previously omitted `stdin` (`src/claude_teams/backends/process_manager.py:940-971`), while the ratchet explicitly excludes `Popen` (`plan.md:45-49`). The existing `TestWindowsTerminalTail.test_opens_windows_terminal_by_default_when_available` checks only the argv (`tests/test_backends/test_base_runtime.py:987-1015`); deleting `stdin=subprocess.DEVNULL` from the production call leaves that test and the new ratchet green. Extend this existing test to assert `popen_mock.call_args.kwargs["stdin"] == subprocess.DEVNULL`.

3. **MAJOR — The ratchet checks keyword spelling, not the stated non-inheritance policy.**

   `_names_stdin()` treats any `stdin=` value and every `**kwargs` splat as compliant (`tests/test_subprocess_stdin.py:37-50`). Consequently, `subprocess.run(..., stdin=None)`—which inherits stdin—and `subprocess.run(..., **{})` both pass. The self-tests expressly bless `**kw` and have no `stdin=None` mutation (`tests/test_subprocess_stdin.py:68-97`). That does not enforce scope item 1's promise that all no-input `run` calls use `DEVNULL`. Tighten the AST rule/self-tests to reject a literal `stdin=None` and, for this package's direct calls, require either `input=` or the explicit `stdin=subprocess.DEVNULL` shape. If dynamic forwarding must remain allowed, document it as an unchecked escape hatch and cover each such production wrapper behaviorally.

4. **MINOR — The call-site inventory is overstated.**

   A complete AST scan of `src/claude_teams` finds 20, not 22, literal `subprocess.run` calls: `procinfo.py` 1, `server_simple.py` 1, `codex.py` 1, `pi.py` 1, `process_base.py` 2, and `process_manager.py` 14 (not 16). It finds five literal `subprocess.Popen` calls, all in `process_manager.py`. Searches for `asyncio.create_subprocess_exec`, `asyncio.create_subprocess_shell`, loop `subprocess_exec`/`subprocess_shell`, imported subprocess call aliases, `check_call`, and `check_output` found no additional production subprocess creation. Correct `plan.md:32-36`; the proposed six-file production scope is otherwise complete.

5. **MINOR — The blanket child-stdin claim should distinguish intentional console-launch paths.**

   The plan says no package child wants inherited stdin (`plan.md:27-28`), but two unchanged launcher paths deliberately pass `stdin=None`: Windows interactive agents launched with `CREATE_NEW_CONSOLE` (`src/claude_teams/backends/process_manager.py:650-665`) and the Linux terminal-emulator launcher (`src/claude_teams/backends/process_manager.py:1878-1887`). Existing tests pin those values (`tests/test_backends/test_base_runtime.py:299-342,375-412,1341-1383`). Forcing `DEVNULL` into the Windows path would change the console/TTY contract and can break interactive CLIs; the current plan correctly does not change either path. Reword the title/risk statement to cover non-interactive `run` calls plus the log-tail launcher and explicitly list these console-launch exceptions.

## Verified call-site behavior

- `tmux` calls (`new-window`/`split-window`, `send-keys`, `capture-pane`, `display-message`, `kill-*`, `has-session`) communicate entirely through argv/stdout/stderr and do not consume stdin.
- Herdr client invocations use argv and captured JSON/text; the long-running Herdr server `Popen` already uses `DEVNULL` for all three standard streams.
- PowerShell CIM queries use `-Command <script>`, never `-Command -`; `taskkill`, `pgrep`, Git plumbing, Codex model discovery, Pi model discovery, and backend `--version` probes do not need stdin.
- `BaseBackend.execute_in_pane()` invokes `cmd /c` or `sh -lc` as a captured, timed, non-interactive API and offers no input channel. Giving such a command EOF is consistent with that contract and prevents arbitrary commands from consuming JSON-RPC bytes.
- The five literal production `Popen` expressions are the two `_popen()` retry branches, Windows Terminal log tail, Linux terminal launcher, and Herdr server. Internal `_popen()` callers always pass `stdin` explicitly (`None` for a real Windows console, otherwise `PIPE` or `DEVNULL`).

## Verification performed

- Fresh codebase-memory index plus graph searches/traces; coverage reported no parse/skipped source files. Coverage freshness remained advisory, so exhaustive claims above were cross-checked directly with AST and literal source scans. Only generated `__pycache__` directories were excluded in the relevant source/test scopes.
- `uv run pytest -q`: **1669 passed, 3 skipped**.
- Focused affected suites: **337 passed**.
- `git diff --check`: clean.
- `ruff format --check`, `ruff check`, `ty check`, Linux CI, and the manual real-stdio Windows smoke were not executed for this review; the plan must not present them as independently verified.
