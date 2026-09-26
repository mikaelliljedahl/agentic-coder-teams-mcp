# Independent post-implementation review: Codex tool-name hint

Reviewer: Claude Opus (post-implementation), 2026-09-26.
Scope: `plan.md` (incl. §7 dispositions), `plan-review.md`, `implementation.md`,
and the full uncommitted diff in `wt-codex-tool-name-hint` against `471a175`.

**Verdict: APPROVE WITH CHANGES.** The spawned-worker path (spawn and resume
hint) is correct, Codex-only, appended exactly once as the last paragraph, and
safe for the cmd.exe shim. Delivery-receipt and correlation matching are
unaffected. The golden change is exactly the intended line. The one real defect
is in the external-member surfaces: they hardcode the `win-agent-teams` server
key, but the README's recommended isolated member setup uses a different key
(finding 1). Fix finding 1 before the PR. Findings 2 to 7 can be fixed now or
dispositioned.

## Verification performed

- Re-ran the focused suites: `test_codex.py`, `test_join_team.py`,
  `test_codex_member_wake.py`, `test_follow_up_delivery.py`,
  `test_native_wake_flag_off.py`, `test_agent_output.py`: **398 passed**.
  `ruff format --check`, `ruff check`, and `ty check` are all green (Linux). I
  did not re-run the full pytest suite. `implementation.md` reports 1838 passed,
  4 skipped.
- Golden: I parsed `flag_off.json` at `HEAD` and in the worktree. `tools` is
  identical and the key set is unchanged. The new `prompt` equals the old one
  with exactly the join-prompt line inserted after `Join protocol:\n`.
- Hint placement: `build_command` wraps `_correlated_prompt(request)` and
  `build_resume_command` wraps `request.prompt`, each once, before
  `_prompt_arg` (`codex.py:394-400`, `:434-440`). Nothing persists the built
  argv and feeds it back as the next prompt: the server's
  `_materialize_prompt` output never contains the hint. So a follow-up cannot
  accumulate hints.
- Hint content: this matches the `send_message` docstring
  (`server_simple.py:3538ff`). A lead's message to a spawned child takes the
  resume path and never enters the child's inbox. The hint says "Work from
  your lead arrives as a new prompt, not in an inbox". It limits
  `read_messages` to messages from the worker's own children. Plan-review F1
  is resolved.
- Shim: the hint contains no `\n \r < > | & ^ ! ( ) % "`. Its only
  punctuation is `: ; . , '` and `_`. Because of the `\n\n` separator, every
  prompt sent through the shim is now JSON-wrapped. This is intended and
  documented. `json.dumps` leaves `[` and `]` unchanged, so the delivery and
  correlation markers survive verbatim inside the wrapped text.
- Receipt matching: `receipt_nonces` runs `_MARKER_RE.findall` over the whole
  user record (`delivery.py:235`). The correlation scan is a substring search
  for the token (`agent_output.py:416-460`). Neither depends on the marker
  being last.
- Changed exact-argv tests (`test_agent_output.py:808,831`,
  `test_codex.py:833-863`): the expected values were updated to include the
  hint, and none was loosened. The rename
  `test_single_line_prompt_not_wrapped_via_cmd_shim` →
  `..._wrapped_via_cmd_shim_after_hint` follows from the intended shape change
  in plan §4. No test was deleted.
- Plan §5 cases 1 to 8 are all present and meaningful. Red evidence is recorded
  in `implementation.md`: 8 expected failures, the golden failure, and the
  argv failures.

## Findings

1. **Major — External-member tool names hardcode the wrong server key for the
   documented isolated setup.**
   - Where: `server_simple.py:876-885` (`_join_tool_names_line`, used by the
     join prompt and by the `join_team` response at `:955`),
     `server_simple.py:860` (step 6), and `native_wake.py:599` (codex queue
     notice).
   - The problem: README `:234-237` tells the external member to run a
     separate profile "whose MCP configuration contains only a
     `win-agent-teams-external` entry with `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`".
     In that setup the member sees these names:
     - Codex: `mcp__win_agent_teams_external__join_team`
     - Claude Code: `mcp__win-agent-teams-external__join_team`

     The prompt says `mcp__win_agent_teams__join_team` and
     `mcp__win-agent-teams__`. The exact name we hand the member does not exist
     in the recommended configuration. The member-wake queue notice
     (`native_wake.py:599`) has the same problem: it is sent to exactly this
     kind of external Codex session.
   - Why the plan missed it: §1.5's argument ("a different key already breaks
     identity") holds only for spawned Codex workers. That is where
     `_MCP_SERVER_NAME` drives the `-c mcp_servers.win-agent-teams.env`
     override. External members join by token and do not depend on that key.
     Plan-review did not catch this either.
   - Fix: state the naming rule rather than one exact key on the member
     surfaces. For example: "Tool names: these are the win-agent-teams MCP
     tools. In Codex they are `mcp__<server key, non-alphanumerics as _>__<tool>`,
     e.g. `mcp__win_agent_teams__join_team` or
     `mcp__win_agent_teams_external__join_team`. In Claude Code they are
     `mcp__<server key>__<tool>`. Not Codex's built-in `collaboration` tools."
     Keep it single-line and shim-safe. The `<`/`>` characters are fine in the
     join prompt, which is pasted and never launched through a shim. In the
     queue notice, avoid `<>` and name both concrete spellings instead.
     Regenerate the golden and adjust `test_join_team.py` and
     `test_codex_member_wake.py`. Alternatively, change the README to recommend
     the key `win-agent-teams` for the isolated profile. That also works,
     because it is a separate profile, but it is a docs and ops change that
     existing users must repeat.

2. **Minor — "teammates are reachable" overstates what a worker can do.**
   - Where: `codex.py:661`.
   - The problem: `send_message` refuses siblings ("Anyone else — a sibling
     ... — is REFUSED"). The phrase "your lead and teammates are reachable only
     through the win-agent-teams MCP tools" invites a worker to message a
     sibling and get refused.
   - Fix: "your lead (and any agents you spawn) are reachable only through
     ...".

3. **Nit — `read_messages` scope is slightly incomplete.**
   - Where: `codex.py:665-666`.
   - The problem: the hint says "messages sent to you by agents you spawned
     yourself". External members registered by this worker also land in its
     inbox through `external_send`.
   - Fix: "by agents or members you spawned or invited yourself", or leave as
     is. It is harmless for typical workers.

4. **Nit — The shim-safety test checks a split fragment, not the constant.**
   - Where: `tests/test_backends/test_codex.py:920-922`.
   - The problem: `hint = complete_prompt.split("\n\n")[-1]` makes the `\n`
     check tautological, and `%` and `"` are not checked. cmd.exe expands
     `%VAR%`, and `"` toggles quoting.
   - Fix: also assert directly on `codex_module._TEAM_TOOL_HINT`, and add `%`
     and `"` to the forbidden set.
   - Related gap: no test asserts that the hint occurs exactly once on spawn or
     resume. Add `prompt.count(codex_module._TEAM_TOOL_HINT) == 1` and
     `prompt.endswith(codex_module._TEAM_TOOL_HINT)` to tests 2 and 3. That
     pins "once, last", which the plan requires.

5. **Nit — The queue notice hardcodes the name instead of using the helper.**
   - Where: `native_wake.py:599`.
   - The problem: plan §2.3 says names come from the §2.1 helper. The module
     already imports lazily from `backends.codex` (`native_wake.py:492`), so
     the helper is available.
   - Fix: use `codex_mcp_tool_name("external_read")`. This becomes moot if
     finding 1 is fixed by naming both spellings.

6. **Nit — The single-line shim branch in `_prompt_arg` is now effectively
   dead.**
   - Where: `codex.py:577-603`.
   - The problem: every built Codex prompt now contains `\n\n`, so via the
     shim the JSON wrap always applies. The docstring still frames the wrap as
     applying to "a multi-line prompt".
   - Fix: update the docstring (and the `_launches_via_cmd_shim` wording if
     desired) to say every spawn and resume prompt is wrapped via the shim
     because of the hint. The code can stay.

7. **Nit — Tool docstrings do not mention the hint.**
   - Where: `spawn_agent` and `follow_up_agent` docstrings (unchanged).
   - The problem: consuming agents read only the docstrings. A lead could
     usefully learn that Codex prompts automatically get a final tool-name
     paragraph, so it need not spell out `mcp__...` names itself. It would also
     explain the extra text when the lead inspects a transcript. The plan
     deliberately skipped this so the `tools` golden stays untouched. That is
     an acceptable disposition, but it should be stated in `implementation.md`
     as a conscious choice, not left implicit. If it is added later, update the
     `tools` golden in the same change.

## implementation.md accuracy

The document is honest and complete:
- Red evidence is present. It includes the setup error in the control test
  and its fix.
- The extra `test_agent_output.py` updates are disclosed as a deviation.
- The exact hint and join line match the code.
- Gate results match my re-run of the lint and type gates.

One item is still open: **Smoke is `TODO`.** Plan §5 treats the Linux
first-turn smoke as the only evidence that the model actually avoids
`collaboration.send_message`. Record it before merge.

## Dispositions (lead, 2026-09-26)

| # | Disposition |
|---|---|
| 1 | Accepted (major). External-member surfaces (join-prompt tool-names line, `join_team` response, step 6, codex queue notice) must not assert a single hardcoded key. State the naming rule (Codex: `mcp__` + key with non-`[A-Za-z0-9_]` chars as `_` + `__` + tool; Claude Code keeps the key as is) and give concrete names for both documented keys, `win-agent-teams` and the README-recommended isolated `win-agent-teams-external`, both derived via `codex_mcp_tool_name`. The spawned-worker hint keeps the single spelling (spawned Codex workers always get the `win-agent-teams` key via the identity override). |
| 2 | Accepted. "your lead and any agents you spawn". |
| 3 | Accepted. Mention that `read_messages` also returns messages from external members you invited. |
| 4 | Accepted. Assert shim safety on `_TEAM_TOOL_HINT` itself incl. `%` and `"`; add exactly-once + at-end assertions for spawn and resume. |
| 5 | Accepted. Build the queue-notice names via `codex_mcp_tool_name`. |
| 6 | Accepted. Update `_prompt_arg` docstring. |
| 7 | Accepted as a deliberate choice: tool docstrings unchanged (the hint is aimed at the spawned agent, not the lead; keeps the `tools` golden stable). Recorded in `implementation.md`. |
| Smoke | Done by lead, PASS — recorded in `implementation.md`. |
