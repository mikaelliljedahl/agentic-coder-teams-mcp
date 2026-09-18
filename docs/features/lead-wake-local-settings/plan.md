# Plan: project-scope wake hooks go to `.claude/settings.local.json`

## Problem

`install_lead_wake(scope="project")` (the default) and
`install_member_wake(scope="project")` write their `Stop` group into the project
`.claude/settings.json`. That file is conventionally **checked in**. The group
is machine- and conversation-specific: it bakes the absolute interpreter path
(`C:/…/.venv/Scripts/python.exe`), the absolute `--session-dir`, and for
lead-wake the `--owner-host-pid` / `--owner-host-token` of one process.

Observed in the field: a Windows install was committed to a shared repo. Every
clone and every git worktree on a Linux machine then ran the hook on each turn
end and printed
`Stop hook error: … C:/…/python.exe: No such file or directory`.
A user-level setting cannot suppress it, because Claude Code merges hooks across
scopes.

## Current behavior

- `_lead_wake_settings_path(scope)` (`server_simple.py`) returns
  `<cwd>/.claude/settings.json` for `project`, `~/.claude/settings.json` for
  `user`. It is shared by `install_lead_wake`, `install_member_wake` and
  `_wake_binding_status`.
- `install_member_wake` writes non-atomically with `path.write_text`.

## Proposed design

1. `_lead_wake_settings_path("project")` returns
   `<cwd>/.claude/settings.local.json` — the per-user, per-checkout file Claude
   Code reads with the same hook semantics and that Claude Code itself keeps out
   of git. `user` scope is unchanged (`~/.claude/settings.json` is never shared).
2. New `_legacy_project_settings_path()` → `<cwd>/.claude/settings.json`.
3. **Migration.** New helper `_strip_legacy_project_group(strip)` where `strip`
   is `_install_wake_hook` or `_install_member_wake_hook` called with
   `remove=True`. It reads the legacy file; if stripping changes the config, it
   rewrites the file atomically and returns the path, else returns `None` and
   leaves the file byte-identical (never creates it). Called for
   `scope="project"` on both install and remove, in both tools, *after* the
   prerequisite refusals and after the new file is written successfully (so a
   refusal still touches nothing). A legacy write failure is non-fatal: the
   result carries `"legacy_cleanup": "failed"`; on success
   `"migrated_from": "<path>"`.
   Only the tool's own group is stripped (lead tool → lead group, member tool →
   member group); unrelated hooks are preserved verbatim.
4. `_wake_binding_status` scans project-local, legacy project, then user. A
   group found only in the legacy file is classified exactly as today
   (`stale`/`legacy`/`bound`), so the existing "re-run install_lead_wake" hint
   fires and the re-install performs the migration.
5. `install_member_wake` switches to `_write_json_object_atomic` (same helper,
   no behavior change beyond atomicity) since it now shares the migration path.
6. Docstrings of both tools (the only thing a consuming agent reads) state the
   new file, why (machine-specific, must not be committed), and the migration.
   Update `README.md`, `INSTALL.md`, `docs/reference/agent-messaging-protocol.md`
   and `.claude/skills/*` mentions of the project path.

## Files affected

- `src/claude_teams/server_simple.py`
- `tests/test_install_lead_wake.py`, `tests/test_wake_binding_status.py`,
  `tests/test_member_wake.py`, `tests/test_tool_descriptions.py`
- `README.md`, `INSTALL.md`, `docs/reference/agent-messaging-protocol.md`,
  skills that name the path
- `docs/features/lead-wake-local-settings/*`

## Risks

- **A repo that does not ignore `settings.local.json`.** Claude Code adds it to
  the global git ignore when it creates the file, but we create it ourselves.
  Mitigation: docstring says it must not be committed; out of scope to edit
  `.gitignore` from an MCP tool.
- **Migration rewrites a tracked file** (JSON re-serialised with `indent=2`).
  This is the same serialisation the original install already applied, and the
  resulting diff (hook removed) is the desired one.
- **Both files carry a group during a half-migrated state** → hook would run
  twice. Migration runs on every project-scope install, so the state is
  transient; the hook itself is idempotent per turn.
- `hooks.Stop` precedence: local and project settings merge, so moving the group
  does not change when it fires.

## Test cases (red first)

1. project install writes `.claude/settings.local.json`; `.claude/settings.json`
   is not created.
2. project install with a legacy lead group in `settings.json` (plus an unrelated
   `Stop` group and an unrelated key): legacy group removed, unrelated content
   preserved, result has `migrated_from`.
3. project install with a legacy file that has no wake group: file
   byte-identical, no `migrated_from`.
4. project `remove=True` strips the group from both files.
5. prerequisite refusal (`no_active_session`) leaves both files untouched even
   when the legacy file has a group.
6. `scope="user"` unchanged; never touches project files.
7. `_wake_binding_status`: group only in legacy project file → `stale`; group in
   `settings.local.json` bound to this host → `bound`.
8. member-wake project scope: writes local file, migrates legacy member group,
   leaves a legacy *lead* group alone (and vice versa).
9. Tool description mentions `.claude/settings.local.json`.

## Plan-review dispositions (Codex, `plan-review.md`)

1. **Two-file failure policy — ACCEPTED (policy defined, no rollback).** Order
   stays local-first, legacy-second: the reverse order could leave *no* hook.
   A legacy cleanup failure keeps the install a success (the new hook works) but
   the result carries `legacy_cleanup: "failed"` and `legacy_path`. The window
   where both files hold a group is allowed: a leftover legacy group is normally
   bound to another process and therefore silent; same-process duplicates issue
   the same instruction twice. Every project-scope call re-attempts the strip, so
   a retry repairs it. Tested with an injected legacy `replace` failure + retry.
   Rollback of the new group is rejected as over-engineering.
2. **Keep the file out of git — ACCEPTED.** New best-effort helper
   `_ensure_locally_ignored(path)`: inside a git work tree, if
   `git check-ignore -q` says the file is not ignored, append the repo-relative
   pattern to `$(git rev-parse --git-path info/exclude)` (local, never
   committed, shared by worktrees). Result field `git_ignore`:
   `already_ignored | excluded | not_a_repo | failed`. Never raises, never
   blocks the install.
3. **Status classification — ACCEPTED.** Locations are scan locations only;
   classification is unchanged (`bound` > `stale` > `legacy` > `absent`). Test 7
   reworded: foreign-bound group in legacy file → `stale`; unbound → `legacy`;
   matching group in the local file → `bound`.
4. **Member remove without a live session — ACCEPTED.** `remove=True` requires
   only a syntactically valid UUID; install keeps live-session validation.
5. **Result shape — ACCEPTED.** Purely additive optional keys on both tools
   (`migrated_from`, `legacy_cleanup`, `legacy_path`, `git_ignore`); existing
   keys and the member tool's shape (no `success` key) are preserved. New-file
   write failure keeps today's behavior.
6. **Byte-identical no-op — ACCEPTED.** The helper detects the target group
   first and returns without writing when absent; after a real removal only
   semantic JSON preservation is promised.
7. **Missing tests — ACCEPTED** (local write failure leaves legacy bytes; legacy
   failure + retry; both files holding a group; corrupt legacy input; user scope
   leaves both project files untouched; hook counts asserted).
8. **Docstrings — ACCEPTED**, including `_wake_binding_status`.
9. **External contract — ACCEPTED as documentation.** Claude Code's settings
   docs list `.claude/settings.local.json` as the personal, git-ignored project
   scope, merged with the other scopes (hooks included); recorded in
   `implementation.md`. No automated integration test (needs a live Claude).
