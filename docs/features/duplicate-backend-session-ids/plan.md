# Plan: deterministic backend_session_id binding for concurrent Claude Code agents

## Problem / motivation

When several Claude Code agents are spawned in the **same `cwd`** at nearly the
same time, some of them are bound to the **same** `backend_session_id`. Observed
in the field: three of five freshly-woken agents shared one id (`0b548227…`),
and their `last_line` values were identical because the fallback reader was
returning the *same session transcript* for all of them.

Consequences:

- `backend_session_id` is not unique per agent, so `resume`/`follow_up` can
  target the wrong native session.
- `check_agent`/`follow_up_agent` report the same `last_message` for distinct
  agents, so their outputs look (falsely) in agreement. Reports that appear to
  corroborate each other may just be one session read N times.

## Current behavior (verified against source)

`backend_session_id` is **not** assigned at spawn. It is discovered after the
fact by scanning Claude's per-project transcript directory.

`read_claude_output` (`src/claude_teams/agent_output.py:79-118`), when no id is
yet known, collects every `*.jsonl` in
`~/.claude/projects/<encoded_cwd>/` whose mtime is fresh and whose first
timestamp is after spawn, then:

```python
mtime, path = max(candidates, key=lambda item: item[0])   # newest-modified wins
backend_session_id = _claude_session_id(path)
```

The selection key is only **cwd + start-time + mtime**. Nothing ties a given
transcript file to a given logical agent. All agents in one `cwd` share one
project directory, so whichever transcript was most recently written at the
instant of discovery wins — and multiple agents racing discovery latch onto the
same file. Hence duplicate ids and identical `last_line`.

### The Codex backend already solved this

Codex faces the identical ambiguity and fixed it with a per-agent **correlation
token** injected into the initial prompt:

- `codex_correlation_token(agent_id)` — `src/claude_teams/agent_output.py:22-31`
- `_correlated_prompt` appends the marker to the spawn prompt —
  `src/claude_teams/backends/codex.py:543-556`
- `_matching_codex_rollouts` prefers rollouts that contain the token when the id
  is still unknown, falling back to the unfiltered set —
  `src/claude_teams/agent_output.py:162-170`
- `_rollout_contains_token` — bounded forward scan of the first
  `_CORRELATION_SCAN_MAX_LINES` (500) lines — `agent_output.py:173-191`
- Regression test: `test_read_codex_output_disambiguates_concurrent_agents_by_token`
  (`tests/test_agent_output.py:408`).

The Claude Code path has **no** equivalent: `read_claude_output` neither accepts
nor uses a correlation token, and `ClaudeCodeBackend.build_command`
(`src/claude_teams/backends/claude_code.py:141-185`) injects no marker.

## Proposed design

Mirror the proven Codex mechanism for Claude Code. Four changes:

1. **`agent_output.py` — token helper.**
   Generalize the marker prefix constant `_CODEX_CORRELATION_PREFIX` to a shared
   `_CORRELATION_PREFIX` (behavior-preserving rename; same string value
   `"wat-corr:"`), keep `codex_correlation_token` using it, and add
   `claude_correlation_token(agent_id)` returning `f"{_CORRELATION_PREFIX}{agent_id}"`.
   The token string is identical to what the read side scans for.

2. **`agent_output.py` — `read_claude_output` accepts `correlation_token`.**
   New keyword-only param `correlation_token: str | None = None`. After building
   `candidates`, when `backend_session_id is None and correlation_token`, filter
   to candidates whose transcript contains the token (reuse the existing
   `_rollout_contains_token`); if any match, use that filtered set, otherwise
   keep the unfiltered set. Then `max(...)` by mtime as today. This exactly
   parallels `_matching_codex_rollouts:162-170` and preserves the transient
   fallback (before Claude has flushed the prompt to the transcript).

3. **`claude_code.py` — inject the marker on spawn only.**
   Append a correlation suffix to the argv prompt in `build_command` (NOT
   `build_resume_command` — resume already knows the id). The suffix must be
   appended to the value returned by `_prompt_arg(request)` so it lands in the
   first recorded user message **even when a prompt file is used** (in that case
   the argv is only the "Read your complete task prompt from …" instruction).
   Suffix format matches Codex:
   `"\n\n[win-agent-teams correlation id: {token} — internal marker, ignore this line]"`.

4. **`server_simple.py` — pass the token when reading.**
   In `_read_agent_output` (`src/claude_teams/server_simple.py:966-986`), the
   `claude-code` branch passes
   `correlation_token=claude_correlation_token(agent_id)`, where
   `agent_id = f"{agent.get('name')}@{agent.get('session_id')}"` — identical to
   the spawn-time `SpawnRequest.agent_id` (`server_simple.py:1296`), so the
   token matches.

## Files affected

- `src/claude_teams/agent_output.py` — constant rename, new
  `claude_correlation_token`, new param + filter in `read_claude_output`.
- `src/claude_teams/backends/claude_code.py` — marker injection in
  `build_command`; small private helper for the suffix.
- `src/claude_teams/server_simple.py` — pass `correlation_token` in
  `_read_agent_output`; import `claude_correlation_token`.
- `tests/test_agent_output.py` — new tests (see below).

## Test cases (red first)

1. **Disambiguation:** two transcripts in one encoded-cwd project dir, both
   started after spawn; transcript B has the newer mtime but transcript A
   contains agent A's token. `read_claude_output(..., correlation_token=<A>)`
   returns A's `backend_session_id`, not B's. (Mirrors the Codex test.)
2. **Fallback when no token present yet:** neither transcript carries the token
   → falls back to newest-mtime (current behavior preserved).
3. **Exact-id path unaffected:** when `backend_session_id` is supplied, the
   token is ignored and matching stays by session id.
4. **`build_command` embeds the token:** the last argv element contains
   `claude_correlation_token(request.agent_id)`; `build_resume_command` does NOT.
5. **Marker present even with `prompt_file_path`:** token still appears in the
   argv prompt when the extra carries a prompt file path.

## Risks / mitigations

- **Marker leaks into the agent's context.** Same as Codex today; the line is
  self-labeled "ignore this line" and is a single trailing line. Acceptable.
- **Token absent during the first split-second** (prompt not yet flushed to the
  transcript): handled by the unfiltered fallback — no regression vs. today.
- **Constant rename churn:** confined to one module; `grep` for
  `_CODEX_CORRELATION_PREFIX` before/after to ensure no dangling refs.
- **Windows path encoding** for the project dir is unchanged
  (`_encode_claude_cwd`).

## Out of scope

- Assigning the session id deterministically at spawn (would require Claude CLI
  to accept a caller-supplied session id; not available). The token approach is
  the same pragmatic fallback Codex uses.
- Any change to Codex behavior beyond the behavior-preserving constant rename.
