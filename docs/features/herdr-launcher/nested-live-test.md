# Nested live test — an agent spawning an agent through Herdr

Run on 2026-09-11 against a real Herdr 0.8.2 server, disposable session
`nested` (stopped and deleted afterwards). Driver:
`scripts/herdr_nested_check.py`, executed by a spawned codex agent
(`herdr-nested-tester`) in this worktree.

The driver calls `server_simple.spawn_agent` **in-process** with the Herdr
launcher enabled, rather than driving the process manager directly. That is the
point of the test: hooks, the `state-<agent>.json` marker and the inbox are the
server's work, so only this path shows whether the on-disk contract survives a
new launcher.

## Result

| Check | Result |
| --- | --- |
| Herdr launcher actually selected | PASS (`HerdrProcessManager`) |
| Spawn returned a pid | PASS (`1694637`) |
| Agent got its own Herdr tab | PASS (`herdr-grandchild@65e3ef04-…`) |
| Pane showed agent content | PASS (2739 chars) — but see caveat |
| **`state-herdr-grandchild.json` written** | **PASS** |
| Grandchild reported upstream | FAIL — see cause below |
| Kill closed the tab, marker and process gone | PASS |
| Test session stopped and deleted | PASS |

### The load-bearing evidence

`spawn_agent` writes **no** state marker: `_write_state_marker` is not called
anywhere in its body, and the marker is produced only by
`claude_teams.hooks emit`, which runs as a lifecycle hook **inside the spawned
agent's own process**, reading a hook payload from stdin.

So `state-herdr-grandchild.json` appearing proves the codex CLI genuinely
started inside the Herdr pane and fired a lifecycle hook. That is what the
whole disk contract rests on, and it survived the launcher change.

### Caveat on the pane check

The pane dump showed the *command line* (which itself contains the word
"codex"), not a painted codex TUI, because the driver read the pane immediately
after `pane run`. The assertion was therefore weaker than it looked and has been
corrected: the check now only asserts that the pane has content, the driver
waits before reading, and the comment records that the marker — not the pane
text — is the real proof. Left as a caution: a check that passes for the wrong
reason is worse than one that fails.

### Why "grandchild reported upstream" failed

Two defects, **both in the test driver, neither in the launcher**:

1. The driver watched `inbox-team-lead.jsonl`. But it ran *inside* a spawned
   agent whose `AGENT_NAME` is `herdr-nested-tester`, and a child sends to its
   **parent's** inbox — `inbox-herdr-nested-tester.jsonl`. This is precisely the
   rule `CLAUDE.md` states ("anything acting on 'the lead' must use the agent's
   OWN identity — never assume `team-lead`"), and the driver broke it. Fixed: it
   now derives the inbox from `AGENT_NAME`.
2. That inbox was created but empty, so the grandchild had not replied within
   the 180s budget either. The window is now 240s.

The message round trip therefore remains **unproven under this launcher**. It is
protocol behaviour rather than launcher behaviour, but it has not been
demonstrated end-to-end here and should not be claimed.

## Incidental findings (not caused by this change)

- **claude-code agents stall on this machine.** Two attempts to run this test
  with `backend="claude-code"` opened a `foot` window, started `claude`, and
  then stopped progressing; the debug log ends at "Policy limits: Cache still
  valid" and no state marker ever appears. codex agents ran normally throughout.
  This looks environmental and predates the launcher work, but it is worth
  investigating separately.
- **A spawned agent gets a minimal per-agent MCP config.** The spawner writes
  `mcp/<agent>.mcp.json` containing only `win-agent-teams`; servers registered
  in the user's `~/.claude.json` are not visible to spawned agents. An earlier
  attempt to have the child use a second, herdr-configured MCP server failed for
  this reason.
- The tester agent's own written report vanished from the worktree before it
  could be read, and it never sent its DONE line. This document was written from
  the driver's own captured output instead. The tester's cleanup step is the
  likely culprit; source changes were verified intact afterwards.
