# Post-implementation review: project-scope wake hooks go to `.claude/settings.local.json`

VERDICT: APPROVED WITH CHANGES

Focused validation: `uv run pytest tests/test_wake_local_settings.py tests/test_install_lead_wake.py tests/test_member_wake.py tests/test_wake_binding_status.py -q` passed with **94 passed, 1 skipped**.

1. **MAJOR — `_ensure_locally_ignored` reports success for a file that is already tracked.**

   **Evidence:** The accepted disposition promises a local exclusion result (`docs/features/lead-wake-local-settings/plan.md:111-117`). The helper checks `git check-ignore` and, if it is nonzero, appends an exclude pattern and returns `excluded` (`src/claude_teams/server_simple.py:6534-6557`). `git check-ignore` does not treat an already-indexed path as ignored by default, and adding `.git/info/exclude` does not remove a tracked path from the index. The only git test covers a new, untracked repository (`tests/test_wake_local_settings.py:323-338`).

   **Recommendation:** Detect index membership with `git ls-files --error-unmatch` (or an equivalent explicit check) before appending. Return a documented `tracked`/`failed` status and make the tool description tell the user that an already tracked local settings file must be removed from the index manually. Add a test that pre-stages `.claude/settings.local.json` and verifies the result does not claim it is safely excluded.

2. **MAJOR — Project-scope removal can create an unignored local settings file.**

   **Evidence:** The atomic writer always creates the target path (`src/claude_teams/server_simple.py:6582-6591`). Lead removal invokes project extras with `ignore=False` (`server_simple.py:6665-6679`), and member removal likewise disables ignore setup (`server_simple.py:6823-6834`). The tool descriptions nevertheless call the project file “git-ignored” (`server_simple.py:6609-6619`, `6773-6781`). Thus a first-ever `remove=True` in an unignored repository writes an empty `settings.local.json` without adding an exclusion.

   **Recommendation:** Either avoid creating a file when removal is a no-op, or run the ignore helper for every project-scope operation that writes/creates the local file, including removal. Add lead and member tests that call project removal in a fresh unignored repository and assert the resulting file is ignored (or is not created).

3. **MAJOR — Legacy read and malformed-shape failures are not handled by the accepted best-effort policy.**

   **Evidence:** The disposition says cleanup failure remains non-fatal and is reported in the result (`plan.md:102-110`). `_strip_legacy_project_group` calls `_read_json_object` outside its `try` and assumes `hooks` is a mapping (`server_simple.py:6503-6511`). `_read_json_object` catches only `JSONDecodeError`; filesystem read errors propagate (`server_simple.py:549-556`). A valid-but-malformed settings object such as `{"hooks": []}` also raises at `.get("Stop")`. Because this helper runs after the local write, the tool can leave the new hook installed and then raise instead of returning `legacy_cleanup: "failed"`. The added corrupt-file test covers only invalid JSON (`tests/test_wake_local_settings.py:155-164`).

   **Recommendation:** Guard `hooks` with `isinstance(hooks, dict)`, and catch legacy read/transform `OSError` (and any deliberately supported decoding failure) into the same `legacy_cleanup` result used for write failure. Add tests for an unreadable legacy path and valid JSON with a non-mapping `hooks` value, asserting the local install result remains well-shaped and the legacy file is not rewritten.

4. **MAJOR — The accepted result-shape disposition is not implemented for member settings-write failures.**

   **Evidence:** The plan disposition explicitly preserves the member tool’s existing shape and says new-file write failure keeps today’s behavior (`docs/features/lead-wake-local-settings/plan.md:124-127`). The implementation now catches `OSError` and returns `{"success": false, "reason": "settings_write_failed"}` (`src/claude_teams/server_simple.py:6823-6826`); the implementation note records that as a behavior change (`docs/features/lead-wake-local-settings/implementation.md:30-32`). The new test asserts the changed result (`tests/test_wake_local_settings.py:299-314`), but no compatibility decision or migration note supersedes the disposition.

   **Recommendation:** Either restore the prior exception behavior, or explicitly revise the disposition and tool contract to make the structured failure an intentional API change. If retaining it, add a compatibility note and test the registered MCP result contract rather than only the implementation function’s return value.

5. **MAJOR — The ignore implementation is not tested for the worktree, nested-cwd, or Windows cases that determine whether the safety fix works.**

   **Evidence:** `_ensure_locally_ignored` derives the Git working directory from `path.parent.parent`, invokes Git with the absolute path, computes a repo-relative pattern with `Path.relative_to`, and writes the Git-reported exclude path (`server_simple.py:6535-6554`). The only positive test initializes a top-level POSIX repository (`tests/test_wake_local_settings.py:323-338`), while the implementation record says Windows was not run (`docs/features/lead-wake-local-settings/implementation.md:48-57`). The plan disposition claims the exclusion is shared by worktrees (`plan.md:111-117`) but provides no linked-worktree or subdirectory proof.

   **Recommendation:** Add tests for a linked worktree, a `cwd` below the repository top level, and a path containing spaces. Add Windows CI or a Windows smoke test covering drive-letter paths, Git’s path output, and the generated anchored pattern. Ensure the test verifies `git check-ignore` from both the nested cwd and the worktree, not merely that a line was appended to the exclude file.

6. **MINOR — Member removal after session loss is implemented, but migration cleanup is not tested on that path.**

   **Evidence:** The accepted disposition requires removal with only a syntactically valid UUID (`plan.md:122-123`), and the code correctly permits `session_not_found` for removal (`server_simple.py:6810-6817`). The test verifies removal from the local file after deleting the session (`tests/test_wake_local_settings.py:277-288`), but it does not place a member group in the legacy project file or assert `migrated_from`/legacy cleanup in this stale-session case.

   **Recommendation:** Extend the missing-session test to seed both local and legacy member groups, then assert both are removed and the result reports the migration. Add the analogous legacy-cleanup failure/retry test for member scope if the shared helper is intended to be covered through both tool entry points.

7. **MINOR — The tool-description tests do not test the contract that MCP consumers actually receive, and they omit the member tool.**

   **Evidence:** Repository conventions make the tool description the consuming agent’s contract (`CLAUDE.md:72-74`). The current test inspects `install_lead_wake.__doc__` directly (`tests/test_tool_descriptions.py:122-129`), even though the same test file documents that the registered FastMCP description is distinct from `func.__doc__` (`tests/test_tool_descriptions.py:132-143`). There is no corresponding member-wake description assertion. The source docstrings contain the new path and migration text (`server_simple.py:6609-6619`, `6773-6784`), but that does not prove the registered descriptions contain it.

   **Recommendation:** Query the registered descriptions for both install tools and assert the local path, migration fields, cleanup-failure behavior, removal-with-missing-session rule, and user-scope exception. Keep the direct docstring test only as a supplemental implementation check.

8. **NIT — The accepted external Claude Code disposition is documented, not integration-tested, as intended.**

   **Evidence:** The implementation record explicitly documents the settings-local merge contract and notes that a live Claude host is required for automation (`docs/features/lead-wake-local-settings/implementation.md:41-46`). The focused tests validate file contents and helper behavior, not Claude’s actual hook loading (`tests/test_wake_local_settings.py:95-118`).

   **Recommendation:** Keep this accepted limitation, but retain a manual smoke-test step in release/implementation notes that verifies a local `Stop` hook is loaded alongside project and user hooks on a supported Claude Code version.
