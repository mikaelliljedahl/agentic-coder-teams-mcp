# Codex team-tool name hint — implementation

## Red/green evidence

- Added the focused tests from plan §5 before production edits. The red run had eight expected failures: `test_codex_mcp_tool_name_mirrors_codex_sanitizer` (helper absent), `test_spawn_prompt_names_full_team_tools`, `test_resume_prompt_names_full_team_tools`, and `test_team_tool_hint_is_shim_safe` (hint absent), `test_join_prompt_names_both_client_tool_forms_flag_off`, `test_join_happy_path`, and `test_join_prompt_flag_on_both_shells_and_reader` (member tool names absent), and `test_single_send_immediate_safe_shim_no_timer` (queue notice used the bare name). `test_nonce_in_the_correct_transcript_is_delivered` passed: the server still materialized one delivery marker. One initial test setup error in `test_hint_is_codex_only` used an invalid blank Claude model; changing the fixture request to `opus` made that control test pass.
- After the production edits, the flag-off golden failed for all four flag-off settings (`None`, `0`, `true`, empty string). Its diff showed only the new tool-names line after `Join protocol:`. The existing Codex shim/native argv assertions then failed because they still expected prompts without the final hint; these were updated.
- The focused affected-file run passed: **249 passed**. The first whole-repo pytest run identified two more exact resume argv assertions in `tests/test_agent_output.py`; both expected the old unhinted prompt. They were updated. The final whole-repo run passed.

## Final design

`codex_mcp_tool_name` mirrors Codex's standard server-key sanitizer (`[^A-Za-z0-9_]` to `_`) and `mcp__` prefix, using the existing `win-agent-teams` key. The Codex backend appends the hint as the last paragraph on spawn and resume, after the correlation or delivery marker and before `_prompt_arg` chooses native argv or the JSON cmd-shim fallback. It is always appended once per constructed command. Claude Code and Pi command builders are unchanged.

The backend-neutral join prompt names the four protocol tools for both documented MCP keys, `win-agent-teams` and `win-agent-teams-external`, and states the Codex and Claude Code naming rules. Its flag-on Codex wake step names both `external_set_wake` forms; the `join_team` response repeats the member-tool guidance. The Codex queue notice names both `external_read` forms, derived with `codex_mcp_tool_name`. The spawned-worker hint still uses the single `win-agent-teams` key because the Codex identity override uses that key.

### Exact final Codex hint

> win-agent-teams: your lead and any agents you spawn are reachable only through the win-agent-teams MCP tools. Message your lead with mcp__win_agent_teams__send_message. Work from your lead arrives as a new prompt, not in an inbox; mcp__win_agent_teams__read_messages reads only messages sent to you by agents you spawned or external members you invited yourself. spawn_agent, list_agents and follow_up_agent use the same mcp__win_agent_teams__ prefix. In code mode call them as tools.mcp__win_agent_teams__send_message. Do not use Codex's built-in collaboration tools such as collaboration.send_message or collaboration.list_agents for this: they only reach Codex-internal subagents, and your lead is not one.

### Exact join-prompt line

> Tool names: use this MCP server's key. Codex replaces non-[A-Za-z0-9_] key characters with _: for win-agent-teams call mcp__win_agent_teams__join_team, mcp__win_agent_teams__external_read, mcp__win_agent_teams__external_send, mcp__win_agent_teams__leave_team; for win-agent-teams-external call mcp__win_agent_teams_external__join_team, mcp__win_agent_teams_external__external_read, mcp__win_agent_teams_external__external_send, mcp__win_agent_teams_external__leave_team. Claude Code keeps the key: mcp__win-agent-teams__join_team or mcp__win-agent-teams-external__join_team with the same tool suffixes. Use the MCP tools, not Codex collaboration tools.

## Golden diff

`tests/fixtures/native_wake/flag_off.json` changes only its `prompt` key. The review fix replaced the prior tool-names line with the exact line above. A parsed comparison against `HEAD` verified the `tools` value is identical and the final prompt equals the original prompt with exactly this one line inserted after `Join protocol:`. The feature intentionally changes the flag-off prompt baseline; flag-on still adds only its wake step.

## Deviations from plan

The review fix deliberately extends the approved external-member design to the README-recommended `win-agent-teams-external` key; the original plan assumed only `win-agent-teams` there. Spawned-worker routing remains as planned. Two additional exact argv assertions in `tests/test_agent_output.py` needed updates after the first whole-repo run, beyond the test files enumerated in §3. The initial control-test setup correction is noted under red/green evidence.

## Review fixes

The post-implementation review's seven findings were resolved as dispositioned:

1. External-member prompts, `join_team` instructions, the flag-on wake step, and the Codex queue notice now cover both configured server keys. The naming rule and concrete names are present in the member-facing text. Red first: six focused tests failed on the old hint wording or missing isolated-key names; after the production changes those assertions passed. The flag-off golden then failed for all four settings on the changed line, before its `prompt` key was updated.
2. The hint now says the lead and agents the worker spawns are reachable through team tools, matching the direction guard.
3. The hint now includes external members invited by the worker in the `read_messages` scope.
4. Tests assert the hint is exactly once and last on spawn and resume, and check the constant directly for newline, carriage return, `<>|&^!()%"`.
5. The queue notice derives both `external_read` names with `codex_mcp_tool_name`.
6. `_prompt_arg` documents that the appended paragraph makes every shim-launched spawn and resume prompt use JSON wrapping.
7. **Deliberate choice:** the `spawn_agent` and `follow_up_agent` tool docstrings remain unchanged. The hint is aimed at the spawned agent, and keeping the docstrings unchanged keeps the `tools` golden stable.

The review-fix focused run passed: **398 passed** across `test_codex.py`, `test_join_team.py`, `test_codex_member_wake.py`, `test_follow_up_delivery.py`, `test_native_wake_flag_off.py`, and `test_agent_output.py`.

## Validation

All commands ran from the repository root on Linux:

| Command | Final result |
| --- | --- |
| `uv run ruff format --check .` | Green — 91 files already formatted |
| `uv run ruff check .` | Green — all checks passed |
| `uv run ty check` | Green — all checks passed |
| `uv run pytest` | Green — 1838 passed, 4 skipped |

## Smoke

**PASS — Linux, 2026-09-26 13:49Z.** A headless Claude Sonnet lead ran this worktree's MCP server with native wake enabled and spawned `hint-smoke` (`codex`, `cheapest`) with the verbatim prompt `Call send_message(text='SMOKE-UP-1') to your lead, then stop.` The lead inbox received `{from: hint-smoke, text: SMOKE-UP-1}` on the first turn.

The rollout at `~/.codex/sessions/2026/09/26/rollout-2026-09-26T15-49-16-01a0ddfa-3325-7bd0-adbb-76eef07c034c.jsonl` had no `collaboration.*` tool call. Its first exec used `ALL_TOOLS.find` for the win-agent-teams `send_message` tool and fetched its schema. Its second exec called `tools.mcp__win_agent_teams__send_message({text:"SMOKE-UP-1"})`, which returned `{"success":true,"to":"team-lead"}`. This smoke predates the review-fix wording changes; the hint mechanism is unchanged.
