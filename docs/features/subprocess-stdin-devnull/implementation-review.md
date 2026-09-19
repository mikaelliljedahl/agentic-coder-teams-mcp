# Post-implementation review: subprocess stdin detachment

VERDICT: CHANGES REQUIRED

The production diff itself is correct and mechanically scoped: all 20 literal `subprocess.run` calls now pass `stdin=subprocess.DEVNULL`, the Windows Terminal log-tail `Popen` is detached, and the two intentional console-launch `stdin=None` paths remain unchanged. Every production hunk is a single added keyword; no argv, positional argument, existing keyword, call order, or surrounding control flow changed. The implementation is not ready because the ratchet still permits inherited stdin and the implementation record materially contradicts the source, tests, and corrected plan.

1. **MAJOR — The ratchet still has false negatives and does not enforce detachment.**

   `_detaches_stdin()` accepts any `input=` and any `stdin=` whose AST value is not the literal constant `None` (`tests/test_subprocess_stdin.py:52-64`). That incorrectly passes at least:

   - `subprocess.run(a, input=None)`: `subprocess.run` does not create a pipe when `input is None`, so stdin is inherited.
   - `subprocess.run(a, stdin=0)` and `stdin=False`: these designate file descriptor 0 rather than detaching it.
   - `subprocess.run(a, stdin=sys.stdin)`: this explicitly passes the parent's stdin.
   - `subprocess.run(a, stdin=subprocess.PIPE if flag else None)`: the expression can evaluate to inheritance.

   Direct execution of the helper returned `True` for all of those shapes. Conversely, a statically safe splat such as `**{"stdin": subprocess.DEVNULL}` is rejected, so the rule also has a deliberate false positive. This does not satisfy the prior finding's requested policy-level check, and disposition 3 in `implementation.md:55-58` is therefore not genuinely fixed. Since every current production call uses the same literal form, make the ratchet require the exact `stdin=subprocess.DEVNULL` AST shape, with an explicit narrowly tested alternative for genuine `input=` calls; add mutations for `input=None`, fd `0`/`False`, `sys.stdin`, and a conditional that can yield `None`. If arbitrary handles or splats are intentionally supported, the test must define and verify a sound static allowlist instead of treating every non-literal-`None` expression as proof.

2. **MAJOR — `implementation.md` is materially inaccurate and internally contradictory.**

   The requested implementation record is not source-accurate:

   - Lines 19-22 say 21 `subprocess.run` changes and 15 in `process_manager.py`; the verified counts are 20 `run` calls total, 14 in `process_manager.py`, plus one separate `Popen` change.
   - Lines 34-37 and the title claim every package child has detached stdin/no child wants the parent, contradicting the unchanged Windows and Linux console-launch `stdin=None` exceptions documented later at lines 61-64.
   - Line 41 says a `**kwargs` splat is trusted, while the implemented ratchet and disposition 3 explicitly reject it.
   - Lines 24-25 report 98 focused passes; the stated three-file command currently produces 100 passes.
   - Line 79 reports 1,669 full-suite passes; the current full suite produces 1,671 passes and 3 skips.

   Correct the implementation counts, final-design statement, splat policy, and validation totals. The two type diagnostics are in unchanged files and the current `ty check` output matches the report; formatting and Ruff claims also reproduced.

3. **MINOR — The rejected event-loop-freeze diagnosis remains in both newly added test narratives.**

   The corrected plan and `implementation.md` properly say the Git timeout delays one worker-thread call rather than freezing the event loop, but `tests/test_subprocess_stdin.py:5-7` and `tests/test_wake_local_settings.py:484-488` still state that `install_lead_wake` froze the whole event loop. Update those comments/docstrings so accepted disposition 1 is reflected everywhere. This is documentation-only; test behavior is unaffected.

4. **MINOR — The corrected plan's test-case summary still describes the superseded ratchet.**

   `plan.md:61-65` correctly says `**kwargs` is not evidence and must fail, but `plan.md:101-103` still says the guard does not flag `**kw` and generically says it does not flag `stdin=`. Update the test-case summary to match the intended tightened rule and its self-tests. This also prevents `implementation.md` from pointing to two conflicting definitions of the accepted plan.

## Accepted parts and verification

- Plan-review finding 2 is genuinely fixed: `TestWindowsTerminalTail.test_opens_windows_terminal_by_default_when_available` now asserts `stdin == subprocess.DEVNULL`, and removing the production keyword would fail it.
- Plan-review finding 4 is corrected in `plan.md`: the exhaustive AST inventory is 20 literal `subprocess.run` calls across the six planned files, with 14 in `process_manager.py`; no additional asyncio subprocess, `check_call`, `check_output`, or aliased production call was found.
- Plan-review finding 5 is correctly reflected in the plan and ratchet docstring: the Windows `CREATE_NEW_CONSOLE` and Linux terminal-emulator launchers retain `stdin=None` intentionally. Only the contradictory `implementation.md` summary remains to fix.
- `git diff --unified=0` confirms production changes are additions only. Adding a final keyword does not reorder or alter evaluation of existing arguments; the intended stdin behavior is the only semantic change.
- `uv run ruff format --check .`: 86 files formatted.
- `uv run ruff check .`: passed.
- `uv run ty check`: the same two diagnostics named in `implementation.md`; neither file is changed by this worktree.
- Focused three-file test command: 100 passed.
- Full `uv run pytest -q`: 1,671 passed, 3 skipped.
- `git diff --check`: clean.

The reported real-stdio Windows smoke was not independently rerun in this review. `implementation.md` identifies only `smoke3.py` in an unspecified session scratchpad, so the before/after timing is supplied evidence rather than independently reproducible evidence from this worktree.
