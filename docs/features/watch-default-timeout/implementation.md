# Implementation: raise `watch` default `--timeout` from 60 s to 1800 s

## Red evidence

Added the metadata-only test first (`tests/test_cli_watch.py::test_watch_default_timeout_is_1800_seconds`)
and ran it before touching `cli.py`:

```
    def test_watch_default_timeout_is_1800_seconds() -> None:
        """Metadata-only: never invoke `watch` without an explicit timeout here."""
        command = typer.main.get_command(app)
        watch_cmd = command.commands["watch"]
        timeout_opt = next(
            p for p in watch_cmd.params if "--timeout" in getattr(p, "opts", [])
        )

>       assert timeout_opt.default == 1800.0
E       assert 60.0 == 1800.0
E        +  where 60.0 = <TyperOption timeout>.default

tests\test_cli_watch.py:29: AssertionError
====================== 1 failed, 42 deselected in 2.43s =======================
```

## Green evidence

Focused:

```
tests\test_cli_watch.py .                                                [100%]
====================== 1 passed, 42 deselected in 1.54s =======================
```

Full suite:

```
1188 passed, 2 skipped in 58.58s
```

Lint:

```
ruff check .  ->  All checks passed!
```

No pre-existing failures anywhere in the tree.

### Test-environment note (deviation-adjacent, worth recording)

The `.venv` lives at the **primary** repo root
(`agentic-coder-teams-mcp/.venv`) and its editable install resolves
`claude_teams` to the **primary** worktree's `src/`, not this feature
worktree's. A first full-suite run therefore exercised `main`'s `cli.py` and
reported the new test still failing (`1 failed, 1187 passed`) even though the
edit was in place. All authoritative runs above were made with

```
PYTHONPATH=<worktree>/src
```

verified via `python -c "import claude_teams; print(claude_teams.__file__)"`
pointing into the worktree before running pytest.

## Files changed

```
docs/reference/agent-messaging-protocol.md |  7 ++++---
 src/claude_teams/cli.py                    |  2 +-
 tests/test_cli_watch.py                    | 12 ++++++++++++
 3 files changed, 17 insertions(+), 4 deletions(-)
```

- `src/claude_teams/cli.py` — `watch()` `--timeout` default `60.0` → `1800.0`.
  Help text ("Seconds to wait before giving up.") left unchanged; it states no
  number and stays accurate.
- `tests/test_cli_watch.py` — added `import typer.main` and the metadata-only
  default assertion. No test invokes `watch` without an explicit `--timeout`.
- `docs/reference/agent-messaging-protocol.md` — synopsis `[--timeout 60]` →
  `[--timeout 1800]`; "Default timeout: **60 s**" → "**1800 s** (30 min)". Both
  stale line-pinned citations (`cli.py:182-199` and `cli.py:186`) replaced with
  unpinned citations naming the `watch()` command / its `--timeout` option, per
  the plan's preference for avoiding hard line pins.

## Deviations from the plan

None functionally. The plan allowed either corrected line numbers or unpinned
citations for the doc references; unpinned form was chosen so the doc does not
re-rot on the next edit to `cli.py`.

Deliberate non-changes held as planned: `_DISK_CONTRACT_NOTE`,
`AGENT_UPGRADE_NOTES.md`, `README.md`, `_watch_argv` (canonical commands still
omit `--timeout` and inherit the new default), and the exit-code / settle-window
semantics.

2026-08-09 12:09
