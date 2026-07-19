# Semantic agent watcher plan

## Goal

Make the existing disk-backed watcher a reliable wake-up bridge for a Claude Code orchestrator. A background watcher should complete only when coordination work is actionable:

1. a spawned agent reaches hook state `waiting`,
2. the current lead has unread inbox messages, or
3. a caller-selected output file changes.

The MCP server remains the orchestration API. `win-agent-teams watch` remains an auxiliary process because completion of a Claude Code background bash task is the harness mechanism that wakes an idle orchestrator.

## Current behavior

- Spawned Claude Code and Codex agents write `state-{name}.json` on lifecycle hooks.
- `PreToolUse` and `PostToolUse` both write `state="running"`.
- `win-agent-teams watch` exits on every matching file creation or rewrite, regardless of marker content.
- A Claude Code lead can therefore wake while its child is still working, including around a long-running bash tool.
- `send_message` appends to `inbox-{recipient}.jsonl`, and `agent_status` derives `unread_count`, but the default watcher observes only `state-*.json`. A message does not wake the lead until an unrelated marker transition occurs.

## Plan-review disposition

Claude Code Opus returned **changes requested** in `plan-review.md`. All findings are accepted as follows before implementation:

- **B1 accepted:** mirror Codex identity environment in `ClaudeCodeBackend.build_env` so nested Claude orchestrators and their background shells receive their own `AGENT_NAME`, `AGENT_SESSION_ID`, and `AGENT_PARENT_NAME`. Add a backend test; the per-agent MCP config remains authoritative and unchanged.
- **N1 accepted with compatibility switch:** inbox watching remains enabled by default for coordination, including custom patterns, but add `--no-inbox` so artifact-only callers can retain the old behavior. Test both modes.
- **N2 accepted:** all MCP contract descriptions must explain wake reasons and direct `message` wakes to `read_messages` rather than marker reads.
- **N3 accepted:** document that timeout exit `2` requires a status re-check, and test that a pre-existing `waiting` marker is not treated as a new edge.
- **N4 accepted:** extract shared pure inbox parsing/counting helpers rather than duplicate message validity rules in the watcher.
- **N5 accepted:** test a partial trailing line followed by a completed append.
- **T1–T6 accepted:** add the requested backend-env, precedence, startup-edge, JSON-schema, custom-state-pattern, and single-JSON-line tests.

With these changes the blocking identity issue is resolved and the plan is approved for TDD implementation.

## Scope

### In scope

- Make state-marker watching semantic: ignore changed markers unless their parsed state is `waiting`.
- Independently detect unread messages for the watcher process's lead identity.
- Supply Claude Code processes with the same explicit agent/session/parent identity environment already supplied to Codex processes, enabling nested Claude watcher routing.
- Preserve custom output-pattern behavior: non-state files matching an explicit pattern still wake on any creation/change.
- Emit a concise machine-readable wake reason.
- Update MCP tool descriptions and user documentation with the new contract.
- Add focused tests and run the full suite.

### Out of scope

- HTTP/SSE or Pi-to-Pi transport.
- Direct user-message injection into Claude Code, Codex, or Pi.
- Changing lifecycle state when a message is sent.
- Consuming inbox messages inside the watcher.
- General reorganization of the existing `docs/` tree beyond this feature directory.

## Design

### 1. Separate edge detection from readiness

Retain the watcher's portable polling and `(mtime_ns, size)` snapshots. After a matching path changes:

- For `state-*.json`, parse the marker and accept it only when `state == "waiting"`.
- Ignore missing, corrupt, non-object, or `running` markers.
- Advance the baseline after ignored changes so the same edge does not trigger repeatedly.
- For other files selected through `--pattern`, retain the existing any-change behavior.

The watcher remains edge-triggered for state markers. It does not immediately accept a pre-existing `waiting` marker because that may be stale from a previous resumed turn. The coordinator should perform its normal status check before mounting the next watcher. A later improvement may add an explicit generation/cursor to remove this ambiguity.

### 2. Supply and derive the watcher reader identity

`ClaudeCodeBackend.build_env` will mirror Codex by explicitly setting:

```python
{
    "AGENT_NAME": request.name,
    "AGENT_SESSION_ID": request.team_name,
    "AGENT_PARENT_NAME": request.lead_session_id,
}
```

This corrects both an absent identity on root-spawned Claude workers and an inherited parent identity on nested spawns. It does not replace or weaken the identity in the generated per-agent MCP config; both the Claude process and its MCP subprocess receive the same intended identity.

The watcher then uses the same identity convention as the MCP server:

```python
reader = os.environ.get("AGENT_NAME", "").strip() or "team-lead"
```

This supports both a root lead and a spawned agent acting as a nested orchestrator because its background shell inherits `AGENT_NAME`.

The relevant files are:

```text
inbox-{reader}.jsonl
inbox-{reader}.pos.json
```

### 3. Detect unread messages without consuming them

Extract pure `Path`-based messaging helpers into a small shared module used by both the MCP server and CLI watcher. The helpers will:

- tolerate missing/corrupt inbox and cursor files,
- parse only JSON objects with a non-empty string `from`, preserving file order and sender sequence,
- load non-negative per-sender cursor counts,
- clamp consumed counts to available totals consistently,
- report senders where `message_count > consumed_count`.

`read_messages` will retain delivery/cursor mutation ownership while consuming the shared parser/counting result. The watcher uses only the read-only count result.

The watcher must never update `.pos.json`; only the owning MCP `read_messages` call consumes messages.

Check unread state:

- once before entering the polling loop, preventing a send-before-watch race,
- after the inbox file changes.

Track the inbox with `(mtime_ns, size)` like other watched files. Cursor changes alone do not create work and should not wake the coordinator.

### 4. Wake result

On success, print one JSON object and exit `0`:

```json
{"reason":"waiting","agent":"worker","path":".../state-worker.json"}
{"reason":"message","from":["worker"],"path":".../inbox-team-lead.jsonl"}
{"reason":"output","path":".../report.md"}
```

Continue to exit `2` silently on timeout. Preserve useful path information for compatibility with callers that inspect output text.

If several predicates become true in one polling interval, prefer `message`, then `waiting`, then `output`, while including all relevant senders/agents where practical. Inbox messages are preferred because they carry explicit communication that should be drained promptly.

### 5. Command compatibility

Keep the existing command shape and add one compatibility escape hatch:

```bash
win-agent-teams watch <session-dir> [--timeout SECONDS] [--pattern GLOB] [--no-inbox]
```

Inbox watching is enabled by default and uses the inherited lead identity, including when a custom output pattern is supplied. `--no-inbox` restores artifact-only behavior for callers that must wake exclusively on the selected pattern. This behavior is explicit and tested.

Default `state-*.json` becomes semantic. A custom non-state pattern keeps its current generic behavior. A custom pattern that selects state files still applies semantic state filtering.

## Files expected to change

- `src/claude_teams/cli.py`
  - marker parsing and semantic filtering,
  - inbox snapshot/check,
  - JSON wake output and `--no-inbox`.
- `src/claude_teams/messaging.py` (new)
  - shared pure inbox parsing/counting helpers.
- `src/claude_teams/backends/claude_code.py`
  - explicit agent/session/parent environment for nested watcher identity.
- `tests/test_backends/test_claude_code.py`
  - production identity propagation assertion.
- `tests/test_cli_watch.py`
  - focused red/green behavioral tests.
- `src/claude_teams/server_simple.py`
  - update `_DISK_CONTRACT_NOTE`, `spawn_agent`, and `agent_watch_paths` descriptions.
- `tests/test_tool_descriptions.py`
- `tests/test_spawn_agent_watch_contract.py` and/or `tests/test_agent_watch_paths.py`
  - contract assertions where appropriate.
- `README.md`
  - describe `waiting OR unread` semantics.
- `docs/features/semantic-agent-watch/*`
  - plan, reviews, and implementation record.
- `CLAUDE.md`
  - standing repository workflow requested for future work.

## TDD sequence

### Red

Add tests proving:

1. Creation/rewrite of `state-worker.json` with `state="running"` does not complete the watcher.
2. Multiple running transitions followed by `waiting` complete only on `waiting`.
3. An inbox append wakes while the state marker remains `running`.
4. Unread inbox content present before watcher startup wakes immediately.
5. Messages already covered by `.pos.json` do not wake.
6. `AGENT_NAME=parent-agent` selects `inbox-parent-agent.jsonl`.
7. Malformed inbox rows and malformed cursor content do not crash or create false unread work.
8. Existing custom output-pattern behavior remains intact.
9. Timeout remains exit code `2` with no output.
10. `ClaudeCodeBackend.build_env` supplies the child identity rather than an inherited parent identity.
11. A pre-existing `waiting` marker is not a new edge and times out; contract text requires a status re-check after exit `2`.
12. Simultaneous message and waiting edges emit `reason="message"`.
13. Every wake reason emits exactly one JSON object with the documented schema.
14. A custom state pattern still ignores `running`.
15. `--no-inbox` preserves artifact-only custom-pattern behavior.
16. A partial trailing inbox line is ignored and a later completed append is detected.

Run the focused tests and retain the failing output in `implementation.md`.

### Green

Implement only enough parsing and predicate evaluation to satisfy the focused tests. Run:

```bash
uv run pytest tests/test_cli_watch.py
```

### Refactor and regression

Remove duplication where it improves clarity without coupling the CLI watcher to MCP process state. Then run formatting/lint checks configured by the project and:

```bash
uv run pytest
```

## Risks and mitigations

### Identity mismatch

Codex already supplies explicit process identity, but Claude Code currently supplies identity only to its MCP subprocess. Add the same `AGENT_NAME`/`AGENT_SESSION_ID`/`AGENT_PARENT_NAME` values to `ClaudeCodeBackend.build_env`, where they override any inherited parent values. Cover the production launch environment and watcher lookup separately. Root leads still fall back to `team-lead` when the watcher itself is not a spawned agent.

### Inbox parsing cost

Do not rescan JSONL every 0.5 seconds. Scan once at startup and thereafter only when the inbox `(mtime_ns, size)` changes. This remains O(n) per message edge, matching current status counting behavior; incremental indexing can be a later optimization.

### Partial append

A cross-process Windows read racing an append may observe a malformed final line. Ignore malformed lines. Because the final write changes size/mtime again, the next poll re-evaluates it. `send_message` currently writes one JSON line per append. Test the two-stage partial-then-complete sequence against the watched inbox.

### Stale waiting marker

Do not treat a marker present at startup as a new state edge. This preserves current edge semantics and avoids immediately completing after `follow_up_agent` while its prior `waiting` marker remains. Document both the required pre-watch status check and a status re-check after timeout exit `2`, which closes the check-to-baseline TOCTOU window.

### Message duplication

The watcher does not advance cursors. Claude wakes and calls `read_messages`, which remains the sole consumer. If the coordinator fails to drain unread messages and starts another watcher, immediate re-wake is intentional.

## Acceptance criteria

- Tool activity that writes `running` no longer wakes a Claude Code orchestrator.
- A transition to `waiting` still wakes it.
- `send_message(to="team-lead")` wakes it even while the sender remains running.
- Messages are consumed only through MCP `read_messages`.
- Nested Claude and Codex lead identity is respected by the production process environment and watcher.
- Custom-pattern callers can disable inbox wake with `--no-inbox`.
- Timeout guidance closes the pre-check/watch-baseline race by requiring a status re-check.
- Existing custom output watching and timeout behavior continue to work.
- Focused and full test suites pass.
- Claude Code Opus approves the plan before implementation and reviews the final diff afterward.
