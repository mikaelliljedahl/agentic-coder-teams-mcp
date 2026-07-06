# Plan: fix Codex spawn creating extra junk WT tabs (+ silent prompt truncation)

## Follow-on: unify codex onto the claude wrapper (fixes lingering `[process exited]` tabs)
After the `\;` fix landed, a second issue remained: codex tabs stay open showing
`[process exited with code N] … press Enter to restart` after the agent is killed
or errors, because codex was launched directly (no wrapper) and WT's default
`closeOnExit: graceful` keeps a non-zero-exit tab. Claude avoids this via a
powershell wrapper ending in `exit 0`.

The direct-launch design (commit 8219d25) assumed codex's TUI + state hooks break
under the powershell wrapper. Two runtime spikes (`scratchpad/spike_codex_wrapper.py`,
`spike_codex_wrapper_hooks.py`) disproved this on the current code: codex's TUI runs
**and** its state hooks fire (`state-<agent>.json` written) under the wrapper, and
killing the codex subtree makes the wrapper hit `exit 0` so WT auto-closes the tab.
The old constraint was really the since-fixed hook-quoting bug (single-quoted TOML
literals + `commandWindows` launcher).

**Change:** codex now takes the same wrapper path as claude by default
(`_spawn_in_terminal_tab`); legacy direct-launch is kept behind
`WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1` as a fallback. This also makes the `\;`
escaping redundant for the default path (prompt is baked into the .ps1, never on
the wt line) while keeping it correct for the fallback. `_terminate_tab` already
keys on `wrapper_path`, so kill/auto-close needs no change. Needs a live MCP smoke
after restart (MCP callback + waiting-marker + kill-auto-close + follow_up).

## Symptom
Spawning a **Codex** agent in a Windows Terminal tab opens the intended tab **plus
several extra "nonsense" tabs** the user must close by hand. Claude agents are fine.

## Root cause (reproduced — see `spike_codex_wt_tabs.py`)
`wt.exe` uses `;` as its **command delimiter**: each `;` on the wt command line
starts another sub-command (a new tab).

- **Claude** is launched via a generated `.ps1` wrapper, so the wt command line is
  only `powershell -File <wrapper>` — the prompt (and any `;`) is baked into the
  file, never exposed to wt. **Immune.**
- **Codex** is launched **directly**: `wt … -- codex <argv>` (chosen because
  Codex's TUI/state-hooks die when parented by a powershell wrapper —
  `process_manager.py:804-812`). Every codex argv token, including the **free-form
  user prompt**, reaches wt's parser. wt splits the prompt on `;` **even inside the
  double-quoted token** that `list2cmdline` produces.

Two consequences, both proven by the spike's recorder (a codex.exe stand-in that
logs its argv):

1. **Junk tabs** — the fragments after each `;` (`then run the tests`,
   `report back…`) become extra `new-tab` sub-commands.
2. **Silent prompt truncation** — Codex only receives the text *before* the first
   `;`. Spike: recorder saw `"Implement the parser"` instead of the full prompt.
   (This second bug is arguably worse and currently invisible to the user.)

Only `;` triggers it; the spike's real command also carries `{ } , = ' "` and
newlines and none of those split.

## Fix (minimal, proven)
Escape wt's delimiter `;` → `\;` in the codex argv tokens that go **after `--`**.
wt strips the backslash and hands a literal `;` to Codex without starting a new
sub-command. Spike Part C confirms: **exactly 1 tab, full prompt with clean `;`**.

### Change
`src/claude_teams/backends/process_manager.py`, `_spawn_in_terminal_tab`, the
`is_codex` branch (currently line 812):

```python
# before
wt_cmd = [*wt_head, "-d", request.cwd, "--", *cmd]
# after
wt_cmd = [*wt_head, "-d", request.cwd, "--", *self._escape_wt_passthrough(cmd)]
```

Add a small helper:

```python
@staticmethod
def _escape_wt_passthrough(cmd: list[str]) -> list[str]:
    r"""Escape wt.exe's ';' command-delimiter so codex argv reaches the child
    verbatim. wt splits its command line on ';' (even inside a quoted token),
    truncating the prompt and spawning junk tabs; '\;' passes a literal ';'
    through. Only used for the direct codex launch — the claude path bakes argv
    into a .ps1 wrapper and never exposes it to wt."""
    return [tok.replace(";", r"\;") for tok in cmd]
```

### Why scoped to the `is_codex` branch
- Claude's `.ps1` path must **not** be escaped (its argv never touches wt).
- Non-wt codex launches (Linux terminals, `WIN_AGENT_TEAMS_NO_WT_TABS=1`) don't
  enter `_spawn_in_terminal_tab`, so they're untouched.
- Both initial spawn and `resume` flow through this one chokepoint, so both are
  covered.

### Edge cases considered
- **Trailing backslash before closing quote** (`\"` hazard): we only insert `\`
  *before* `;`, never at a token's end (unless the token ends in `;`, which yields
  `…\;"` — backslash precedes `;`, not `"`). Safe.
- **Other wt metachars**: empirically only `;` splits; the escape is still safe if
  future config values contain `;`.
- Escaping is applied to **all** post-`--` tokens (flags + `-c` values + prompt),
  which future-proofs config values without affecting today's `;`-free ones.

## Tests
Extend `tests/test_backends/test_base_runtime.py` (existing codex-tab tests ~L620–650):
1. **Codex prompt with `;`** → assert the post-`--` prompt token in the `Popen`
   argv is `\;`-escaped and the full prompt is preserved (no truncation).
2. **Regression**: Claude tab launch argv is unchanged (no escaping applied).
3. Unit-test `_escape_wt_passthrough` directly (pure string logic, cross-platform).

These are argv-construction tests (Popen mocked), so they run on **Linux too** —
run the suite on the Lubuntu VM before calling the branch green.

## Smoke test (after restarting Claude Code)
The MCP server runs from this repo's `.venv` in **editable** mode
(`~/.claude/.mcp.json` -> `…/.venv/Scripts/python.exe -m claude_teams.server_simple`,
loading `src/claude_teams/…`). The working-tree fix is already on disk, so a
Claude Code restart reloads it — no reinstall needed.

**State at restart:** branch `fix/codex-wt-semicolon-tabs`, changes **uncommitted**
(process_manager.py + test_base_runtime.py + this plan + spike). Commit only after
smoke passes.

Steps:
1. Restart Claude Code (restarts the win-agent-teams MCP server -> loads the fix).
2. Spawn a **codex** agent whose prompt deliberately contains semicolons, e.g.
   prompt: `List your steps: first inspect the repo; then summarise it; finally
   reply to lead with the summary.`
3. **PASS criteria:**
   - Exactly **one** WT tab opens (titled `<agent>@<team>`); **no** junk tabs
     (`then summarise it`, `finally reply…`) and no `0x80070002` error tabs.
   - The codex agent received the **full** prompt (all three steps), not just the
     text before the first `;`. Confirm via its reply / actions.
4. Kill the agent; confirm clean teardown.

## Deliverables / branch
- Branch off `origin/main` (e648d22), e.g. `fix/codex-wt-semicolon-tabs`.
- `spike_codex_wt_tabs.py` documents the repro + fix; remove before merge (or keep
  under a `spikes/` dir if the team wants it retained).
- Manual smoke: spawn a real codex agent whose prompt contains `;` → one tab, full
  prompt.
```
