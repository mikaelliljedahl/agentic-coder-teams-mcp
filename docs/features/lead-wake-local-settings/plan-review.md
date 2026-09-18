# Plan review: project-scope wake hooks go to `.claude/settings.local.json`

VERDICT: APPROVED WITH CHANGES

1. **MAJOR — The two-file migration is not failure-safe and can leave duplicate hooks indefinitely.**

   **Evidence:** The plan writes the new file first, then removes the legacy group, and explicitly makes legacy cleanup failure non-fatal (`docs/features/lead-wake-local-settings/plan.md:34-43`). It acknowledges that both files can hold a group but assumes the hook is idempotent (`plan.md:75-77`). The current atomic helper is atomic only for one path (`src/claude_teams/server_simple.py:6478-6488`), while the lead and member installers currently each write one selected path (`server_simple.py:6601-6608`, `server_simple.py:6683-6688`).

   **Recommendation:** Define an explicit recovery policy before implementation. At minimum, if legacy removal fails after the local write succeeds, report a failed operation (not an ordinary successful install), identify that the legacy hook remains, and attempt to roll back the newly written group. Also test a failure injected into the legacy replace and a retry. A crash between the two independent replaces cannot be made fully atomic; the plan must choose and document whether that window is allowed, how duplicate firing is prevented or tolerated, and how the next invocation repairs it. Do not rely on “idempotent per turn” without an implementation-level test or evidence.

2. **MAJOR — The plan does not guarantee that the new file is actually kept out of version control.**

   **Evidence:** The stated risk is that the tool creates `.claude/settings.local.json` itself, while the mitigation is only a docstring warning and the decision not to edit `.gitignore` (`plan.md:68-71`). The implementation’s atomic writer creates the parent and replaces the requested path (`server_simple.py:6478-6488`), so the claimed Claude Code ignore side effect is not established by this tool.

   **Recommendation:** Make ignore protection part of the design: preferably add `.claude/settings.local.json` to the repository’s ignore policy, or use a clearly documented, reliable local-exclude mechanism. Add a verification that the generated path is ignored. If changing ignore state is intentionally out of scope, state that the feature does not prevent recurrence of the original committed-machine-path incident and surface a prominent warning/result field; a docstring alone is not an adequate safety mitigation.

3. **MAJOR — A legacy file’s location must not determine `_wake_binding_status`; one proposed test encodes the wrong classification.**

   **Evidence:** The plan says a group found only in the legacy project file is `stale` (`plan.md:46-49`, `94-95`). The actual status code classifies from the owner flags and current host: missing binding is `legacy`, a foreign PID/token is `stale`, and a matching PID/token is `bound` (`server_simple.py:6338-6373`). Existing tests demonstrate the distinction: missing owner binding is `legacy` (`tests/test_wake_binding_status.py:141-160`), while a different process is `stale` (`tests/test_wake_binding_status.py:98-139`).

   **Recommendation:** Reword the plan and tests so project-local, legacy-project, and user are only scan locations. Test each classification in each location. Preserve an explicit precedence such as matching `bound` first, then any foreign bound group as `stale`, then an unbound group as `legacy`, otherwise `absent`; do not turn “legacy path” into the `legacy` or `stale` state by itself.

4. **MAJOR — `install_member_wake(remove=True)` may be unable to perform the promised legacy cleanup after the joined session is gone.**

   **Evidence:** The plan promises migration on remove (`plan.md:34-45`, `90`), but the current member tool validates the joined session before resolving the settings path or constructing the update (`server_simple.py:6677-6685`). The existing tests establish that an unknown session returns `session_not_found` (`tests/test_member_wake.py:566-578`). A stale member hook is precisely a case where the joined session may no longer be available.

   **Recommendation:** For removal, accept a syntactically valid session ID and derive the path without requiring a live session; the removal helper does not need the session directory contents. Keep live-session validation for installation. If preserving the current prerequisite is intentional, state it in the tool docstring and add a test showing that cleanup is expected to remain impossible after session deletion; do not describe remove as unconditionally stripping both files.

5. **MAJOR — Backwards compatibility of result shapes and failure semantics is underspecified.**

   **Evidence:** The lead tool has a stable success object with `success`, `action`, `path`, `reader`, `scope`, and binding fields (`server_simple.py:6609-6620`), whereas the member tool’s successful result has no `success` field (`server_simple.py:6689-6695`). The plan adds `migrated_from` and `legacy_cleanup` but does not specify whether those are additive on both tools or whether a cleanup failure changes `success`/exception behavior (`plan.md:41-43`, `50-51`).

   **Recommendation:** Specify an additive result contract separately for lead and member tools. Preserve existing keys and the member tool’s existing success shape; add optional migration fields only. Define whether cleanup failure is `success: false`, a normal action with a warning field, or an exception, and keep new-file write failures consistent with the current behavior. Add tests for both tools’ successful, migrated, no-op, and cleanup-failed result shapes.

6. **MAJOR — “No legacy wake group” cannot automatically imply byte-identical preservation with the proposed helper.**

   **Evidence:** The plan requires no rewrite when stripping finds no group (`plan.md:36-38`, `88-89`). The existing strip helper reconstructs the hooks map and removes an empty `Stop` key and possibly an empty `hooks` key even when no wake group was present (`server_simple.py:6407-6419`). Any actual rewrite through the atomic writer serializes normalized JSON without preserving original whitespace/newline bytes (`server_simple.py:6478-6484`).

   **Recommendation:** Make the migration helper first detect whether the target group exists; return without writing for a true no-op. Define preservation as semantic JSON preservation when a removal is needed, not “verbatim” bytes for unrelated content. Test a legacy file with custom whitespace/newline plus an empty `Stop` structure, and assert both no-op bytes and parsed-content preservation after a real removal.

7. **MAJOR — The proposed tests miss the important partial-failure and duplicate-state cases.**

   **Evidence:** The listed tests cover ordinary migration, refusal ordering, user scope, and basic status (`plan.md:81-98`), but none injects failure in the new-file write, failure in legacy cleanup, a process/crash-like state with groups in both files, or a retry after partial migration. The existing lead suite already has an atomic replace failure pattern for the selected file (`tests/test_install_lead_wake.py:391-412`), but there is no corresponding two-path test. Member installation currently writes directly (`server_simple.py:6687-6688`), so its new atomic failure behavior also needs direct coverage.

   **Recommendation:** Add focused tests for: local write failure leaving legacy bytes untouched; local success plus legacy replace failure; retry repairing both files; both files initially containing the same group; existing local and legacy unrelated groups/keys; member atomic replace failure; corrupt/unreadable legacy input; and user-scope operations leaving both project files untouched. Assert the resulting hook count as well as the returned status.

8. **MINOR — The docstring update must cover the actual authoritative contracts, not only the plan’s listed prose files.**

   **Evidence:** The current lead docstring still says project scope writes `.claude/settings.json` (`server_simple.py:6505-6507`), and the member docstring says the same (`server_simple.py:6649-6652`). Repository conventions say the consuming agent reads MCP tool descriptions rather than README or this repository guidance (`CLAUDE.md:72-74`). The `_wake_binding_status` docstring also says `absent` means no group in “either scope” (`server_simple.py:6338-6342`), which becomes inaccurate after adding a separate legacy-project scan.

   **Recommendation:** Update both install-tool docstrings with the exact local path, why it is local, the user-scope exception, migration behavior, and partial-cleanup warning. Update the status docstring and any exposed binding description to name all scanned locations and preserve the state meanings. Keep README/INSTALL/protocol/skill edits targeted to actual path references rather than duplicating a broader documentation rewrite.

9. **NIT — Clarify the external Claude Code contract with a verification step.**

   **Evidence:** The central premise that `.claude/settings.local.json` loads hooks with the same merge/ordering semantics is asserted but not evidenced in the plan (`plan.md:29-32`, `78-79`); the repository tests only inspect JSON files and helper output (`tests/test_install_lead_wake.py:106-123`, `tests/test_wake_binding_status.py:77-96`).

   **Recommendation:** Record the supported Claude Code settings behavior and version/source in the implementation notes, and add a small manual or integration smoke test that installs a local `Stop` hook alongside project/user hooks and observes the expected merge behavior. The unit tests should still remain the primary regression coverage, but this external contract should not be left as an unverified assumption.
