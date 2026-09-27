# Implementation review: test-env-isolation

**Reviewer:** an independent Claude general-purpose subagent with a fresh
context. The CLAUDE.md-specified post-implementation reviewer (Claude Code
Opus) *is* the same model family as this implementer. The opposite-family
(GPT/Codex) **plan** review is still owed; see `plan-review.md`.

**Verdict:** approve, no blocking findings. The reviewer independently re-ran
all four gates with a clean env and with a broad ambient set: 2822 passed,
14 skipped, and ruff format, ruff check and ty check all green.

## Findings and dispositions

1. **Load-time scrub runs before any `claude_teams` import.** Verified:
   `pytest_plugins` is processed after the conftest module body, pyproject has
   no `addopts`/`-p`, and the nested `tests/test_backends/conftest.py` imports
   only `pytest`. No action.
2. **Strict `monkeypatch.delenv(key)` cannot raise**, because keys come from
   `os.environ` immediately before. No action.
3. **No test loses coverage.** No test was gated on an inherited value. No
   action.
4. **Windows case-insensitive `os.environ`** is safe: keys are uppercased,
   so lowercase exports are also caught. No action.
5. **Guard test passes trivially in CI.** **Accepted:** added parametrised
   tests of `is_ambient_agent_env` itself. Positive cases are the known-leaky
   names; negative cases are host state (`HOME`, `PATH`, `TMUX`, `DISPLAY`,
   `XDG_RUNTIME_DIR`, `COMSPEC`). They exercise the predicate in CI.
6. **Nit: stale guard docstring.** **Accepted:** reworded to describe the
   load-time plus per-test layers.
7. **Process: reviews are not from the opposite family.** **Acknowledged:**
   disclosed in the PR body.
