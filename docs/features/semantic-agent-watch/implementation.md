# Semantic agent watcher implementation

## Summary

Implemented the approved semantic watcher design:

- state marker rewrites wake only when the changed marker says `waiting`,
- unread messages for the current orchestrator wake without being consumed,
- custom output paths retain edge-trigger behavior,
- `--no-inbox` preserves artifact-only waiting,
- each successful wake emits one JSONL record with reason `message`, `waiting`, or `output`,
- Claude Code child processes now receive explicit agent/session/parent environment identity so nested Claude orchestrators watch their own inbox,
- MCP descriptions tell coordinators how to handle each reason and to re-check status after timeout exit 2.

Inbox validity and cursor interpretation now live in `src/claude_teams/messaging.py` and are shared by the MCP server and watcher.

## Red evidence

The focused red run was:

```bash
uv run pytest tests/test_cli_watch.py \
  tests/test_backends/test_claude_code.py \
  tests/test_tool_descriptions.py -q
```

Before production changes it produced 11 failures. The failures demonstrated:

- `running` transitions incorrectly completed the old watcher,
- inbox writes and pre-existing unread messages did not wake,
- nested identity inbox selection was absent,
- output was not the new JSON reason contract,
- `--no-inbox` did not exist,
- MCP descriptions did not document message/waiting reasons or timeout recovery.

A duplicate test-class name initially prevented the new backend identity test from being collected. This test defect was corrected before green by merging it into the existing `TestClaudeCodeBuildEnv` class; the existing environment-size assertion was updated from two to five keys.

## Green and refactor evidence

Focused behavior and regression run:

```text
uv run pytest tests/test_cli_watch.py tests/test_backends/test_claude_code.py \
  tests/test_tool_descriptions.py tests/test_read_messages.py \
  tests/test_agent_status.py -q
113 passed
```

Full suite:

```text
uv run pytest -q
476 passed
```

Lint for every modified Python file:

```text
uv run ruff check <modified Python files>
All checks passed
```

`git diff --check` also passed.

The repository-wide `ty check src` still reports 25 existing diagnostics in `agent_output.py`, `process_manager.py`, and pre-existing `server_simple.py` typing. No diagnostic points to the new `messaging.py` or watcher code. The repository-wide coverage invocation reports 79.63% against a configured 90% floor; the new `messaging.py` itself reports 92%, and all 476 tests pass. These baseline-wide quality issues are not expanded into this scoped feature.

## Deviations and review-driven additions

The initial plan assumed a nested Claude process inherited its own `AGENT_NAME`. Opus review showed identity was supplied only to the MCP subprocess. The implementation therefore also updates `ClaudeCodeBackend.build_env`, matching Codex identity propagation.

Opus also requested:

- explicit custom-pattern compatibility (`--no-inbox`),
- shared inbox parsing instead of duplicated validity rules,
- timeout race documentation,
- message precedence over simultaneous waiting/output edges,
- partial-append recovery coverage,
- strict one-record JSON wake schemas.

All were accepted and implemented.

The independent implementation review approved the change with no blocking findings. Before PR, its non-blocking recommendations NB1–NB4 were also accepted: tests now cover message wake during active `running` churn, inbox wake with a custom pattern, and corrupt-cursor handling; README now calls out the custom-pattern migration and `--no-inbox` opt-out. The final suite increased from 473 to 476 passing tests.

## Remaining intentional behavior

A `waiting` marker already present when the watcher starts is not considered a new edge. This avoids stale-marker wake after resume. Coordinators must check status before watching and re-check after timeout exit 2, closing the small check-to-baseline race without introducing a turn-generation protocol in this feature.
