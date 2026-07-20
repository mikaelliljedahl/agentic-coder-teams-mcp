# Code Review: Duplicate Claude Backend Session IDs

## Summary

The uncommitted change correctly mirrors Codex's correlation-token selection for
Claude Code. It deterministically prefers the transcript containing the
spawned agent's marker before a native session ID is known, retains the prior
newest-mtime fallback, and leaves the known-session-ID path and resume prompts
unaffected. No source defects or scope violations were found.

Targeted validation passed: `git diff --check` and `uv run pytest
tests/test_agent_output.py tests/test_backends/test_claude_code.py -q` (`129
passed in 18.45s`). The known environmental watch-test failures were not run or
attributed to this change.

## Score (0-100)

99/100

## AC/Plan Compliance

| Plan item | Result | Evidence |
| --- | --- | --- |
| (a) Share the correlation prefix and add a Claude helper | Pass | `src/claude_teams/agent_output.py:17` renames the constant to `_CORRELATION_PREFIX`; both helpers use it at `:31` and `:44`. Repository search found no `_CODEX_CORRELATION_PREFIX` references. |
| (b) Add token-aware Claude transcript selection with fallback | Pass | Keyword-only, defaulted `correlation_token` is at `src/claude_teams/agent_output.py:92-99`; token filtering is restricted to `backend_session_id is None` at `:130-137`, and the original candidates remain when the match list is empty. |
| (c) Inject a marker only into the initial argv prompt, including prompt-file launches | Pass | `build_command` appends the suffix directly to `_prompt_arg(request)` at `src/claude_teams/backends/claude_code.py:184-185`; the suffix and token source are at `:188-203`. `build_resume_command` retains only `_prompt_arg(request)` at `:205-243`. |
| (d) Derive and pass the matching Claude token on read | Pass | `src/claude_teams/server_simple.py:983-990` uses `f"{name}@{session_id}"`; this exactly matches spawn-time `SpawnRequest.agent_id` construction at `:1300-1302`. |

## Correctness

Candidate collection and the existing timestamp/session filters are unchanged.
For an unknown ID, selection now scans each eligible transcript for the
specific token, then chooses the newest only among matches. If nothing has
flushed the token yet, it deliberately uses the original candidate list. For a
known ID, the token branch is skipped and the exact session-ID filtering remains
the sole selector. The suffix is a single argv element; no shell interpolation
or quoting path is introduced.

## Parity with Codex reference

Pass. `read_claude_output` implements the same condition, bounded-token helper,
matched-list construction, and “matched if nonempty; otherwise original list”
semantics as `_matching_codex_rollouts` at
`src/claude_teams/agent_output.py:187-201`. Assigning `candidates =
token_matched` (`:136-137`) instead of returning it is behaviorally equivalent:
the local list is immediately consumed by the same `max(..., key=mtime)` call,
and no caller can observe the local reassignment.

## Edge cases / Backwards compat

Pass.

- Existing callers remain compatible: `correlation_token` is optional,
  keyword-only, and defaults to `None` (`agent_output.py:95-99`).
- A `None` or empty token preserves pre-change selection (`agent_output.py:130`).
- A nonempty exact `backend_session_id` keeps its established exact-ID route
  (`agent_output.py:121-126`) and cannot be token-filtered (`:130`).
- Prompt-file launches use `_prompt_arg` before the suffix (`claude_code.py:185`),
  so the marker is placed in the file-read instruction—the first recorded user
  message—without putting the task's raw prompt into argv.
- Resume launches do not add the marker (`claude_code.py:241-242`), avoiding
  repeated marker pollution once the native session ID is known.

## Test Fidelity

| Planned test case | Result | Review |
| --- | --- | --- |
| Disambiguation by token | Pass | `tests/test_agent_output.py:607-654` writes two real JSONL transcripts with eligible start times and opposite freshness, then invokes `read_claude_output`. It asserts ID, message, and selected path for the older token-matching transcript. |
| Fallback when token absent | Pass | `tests/test_agent_output.py:657-686` supplies eligible token-free transcripts and verifies the newer transcript is retained, exercising the nonempty-candidates/empty-match-list fallback. |
| Exact-ID path ignores token | Pass | `tests/test_agent_output.py:689-734` supplies a known target plus newer token-bearing decoy, calls the public reader with both ID and token, and asserts the target response. This exercises the actual exact-ID route. |
| Initial command embeds token; resume does not | Pass | `tests/test_agent_output.py:737-754` calls both production command builders and verifies the request token is appended only for initial spawn. |
| Token survives `prompt_file_path` | Pass | `tests/test_agent_output.py:757-774` invokes the production builder with the Windows prompt-file extra and asserts both path instruction and marker occur in its sole prompt argv element. The existing strengthened backend test also verifies the raw prompt and quotes stay out of that instruction (`tests/test_backends/test_claude_code.py:230-250`). |

## Scope adherence

Pass. `git status --short` reports changes only to the three production files
and the two test files permitted or directly necessitated by the plan. Although
`tests/test_backends/test_claude_code.py` is outside the plan's short “Files
affected” list, its five adjusted prompt assertions are justified: exact prompt
equality would necessarily fail after the required suffix is appended. The
updates preserve the original checks with `startswith(...)` and add marker
assertions where appropriate (`tests/test_backends/test_claude_code.py:157-164,
:199-209, :211-228, :230-250`).

## Critical Issues (blockers)

None.

## Warnings (non-blocking)

None.

## Final Verdict

APPROVED
