# Implementation: project-scope wake hooks go to `.claude/settings.local.json`

## Red

`tests/test_wake_local_settings.py` was written first against unchanged
production code: `13 failed, 4 passed` (the 4 passing cases assert behavior
that already held: user scope, refusal ordering, malformed-id rejection, and
the absent `git_ignore` key for user scope).

## Green

`uv run pytest tests/test_wake_local_settings.py` → `17 passed`. Six existing
tests that pinned the old project path (`tests/test_install_lead_wake.py` ×5,
`tests/test_member_wake.py` ×1) and the tool-description test were updated to
the new path; `tests/test_wake_binding_status.py` gained the project-local scan
cases and keeps its legacy-path cases (that file is still scanned).

## Final design

- `_lead_wake_settings_path("project")` → `<cwd>/.claude/settings.local.json`;
  `_legacy_project_settings_path()` → `<cwd>/.claude/settings.json`.
- `_strip_legacy_project_group(has_group, strip)` detects the tool's own group
  first and rewrites the legacy file only when one exists (byte-identical
  no-op otherwise; a corrupt file reads as `{}` and is left alone).
- `_ensure_locally_ignored(path)` appends `/.claude/settings.local.json` to
  `.git/info/exclude` when `git check-ignore` says the file is not ignored.
- Both install tools call `_project_scope_extras` *after* the local file was
  written, so refusals and local write failures touch nothing. Additive result
  keys: `migrated_from`, `legacy_cleanup` + `legacy_path`, `git_ignore`.
- `install_member_wake` now writes atomically and returns
  `settings_write_failed` on I/O failure; `remove=True` tolerates
  `session_not_found`.
- `_wake_binding_status` scans project-local, legacy project, user;
  classification is unchanged.

## Deviations from the plan

None beyond the accepted plan-review dispositions recorded in `plan.md`.
`_git` names `errors="replace"` to satisfy `tests/test_subprocess_decoding.py`.

## External contract

Claude Code's settings documentation lists `.claude/settings.local.json` as the
personal project scope ("not checked in"), merged with user and project
settings; hooks from all scopes run. Not covered by an automated test (needs a
live Claude host).

## Validation

```bash
uv run ruff format --check .   # 83 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1546 passed, 4 skipped (Linux)
```

Not run on Windows.

## Post-implementation review dispositions (Codex, `implementation-review.md`)

Each accepted finding got a failing test first (`6 failed` before the fixes).

1. **Tracked file reported as excluded — FIXED.** `git ls-files --error-unmatch`
   runs first; new status `tracked`, documented in both tool descriptions with
   the `git rm --cached` remedy.
2. **Removal creating an unignored file — FIXED.** `remove=True` no longer
   writes a settings file that does not exist (both tools).
3. **Legacy read / malformed shape — FIXED.** Read, shape check and write share
   one `try`; `OSError`/`UnicodeDecodeError` → `legacy_cleanup: "failed"`;
   a non-mapping `hooks` or non-list `Stop` is a byte-identical no-op.
4. **Member `settings_write_failed` result — ACCEPTED AS INTENTIONAL CHANGE.**
   This supersedes plan disposition 5 for that one case: the previous behavior
   was an unhandled `OSError` out of a non-atomic write, which no caller could
   rely on. The structured refusal matches `install_lead_wake`, is in the
   registered tool description, and is asserted there.
5. **Layout coverage — PARTIALLY FIXED.** Added linked-worktree and
   nested-cwd-with-spaces tests that assert `git check-ignore`, plus an
   anchoring assertion. **Windows is NOT covered**: no Windows host or CI job
   was available; drive-letter paths through `git rev-parse --git-path` remain
   unverified.
6. **Member stale-session migration — FIXED** (test added).
7. **Registered descriptions — FIXED.** Both tools are asserted via
   `mcp.get_tool(...)`.
8. **Live Claude smoke — manual step.** After install, end a lead turn in a repo
   that also has project and user `Stop` hooks and confirm all three fire.

Final gates after the fixes: format/lint/ty clean, `1546 passed, 4 skipped`.
