# Implementation — codex-trust-prompt

## Part A: first-marker diagnosis

### Red/green evidence

- Red first: `uv run pytest -q tests/test_startup_diagnosis.py tests/test_correlation_transport.py::test_launch_metadata_precedes_marker_from_minimal_adapter tests/test_follow_up_delivery.py::test_follow_up_launch_metadata_precedes_child_marker` — **17 failed**. The failures showed the missing hook capability, launch fields, diagnosis helper and status fields, and tool descriptions.
- After implementation, the focused status, spawn and follow-up run passed: **36 passed**.
- First whole-repo pass found nine lint errors in new code, three test typing errors, and eight tests with exact old payload or tool-description snapshots. Those were corrected without changing the diagnostic predicate.
- Final whole-repo pass: **1882 passed, 6 skipped**.
- Review-fix red run: `uv run pytest -q tests/test_native_wake_flag_off.py::test_golden_is_ascii_for_windows_default_encoding tests/test_follow_up_delivery.py::test_follow_up_launch_metadata_precedes_child_marker tests/test_follow_up_delivery.py::test_follow_up_ignores_raising_optional_hook_capability tests/test_startup_diagnosis.py::test_list_agents_doc_only_names_returned_state_field` — **4 failed**. The ASCII assertion exposed the non-ASCII fixture, the optional capability raised through follow-up, and the list docstring named `stalled`. The tightened ordering test initially returned `agent_busy` because its live old-process fixture needed a waiting marker; after that test setup correction it passed with a strictly increasing fake clock.
- Review-fix focused green run: **133 passed** across the native-wake golden, follow-up ordering and capability tests, startup diagnosis tests, and Pi backend tests.
- The first whole-repo review-fix pass had format and lint failures confined to the new tests (two files needing formatting, `TRY003`, and `E501`); these were fixed. `ty check` passed, and pytest reported **1885 passed, 6 skipped**.

### Final design

Spawn captures TTY mode and the optional backend hook argv immediately before `b.spawn`, then captures `launch_started_at` on the line before process startup. Follow-up does the same after shutting down the old PID and immediately before `resume`; the finalizer stores the metadata when the replacement process is recorded. `spawned_at` keeps its prior capture point. Claude Code, Codex and Pi expose state-hook argv through `state_hook_args`; the shared base returns `[]`, and adapters without the optional method also yield `[]` and `hooks_wired=false`.

One helper uses the persisted launch metadata and raw marker for `no_marker_since_launch` and `startup_hint` on both `check_agent` forms, both `list_agents` forms and `agent_status`. The tri-state predicate and wall-clock limitation are in all three tool descriptions and the reference protocol. A stale `waiting` marker still controls public `state`; the new diagnosis does not alter `state`, `stalled` or `heartbeat_age_s`.

### Review fixes and Linux smoke

- No behavioral deviation from rev 3 Part A and round 3 dispositions 3 and 6. README was not changed because the approved plan lists its backend notes under Part B, not Part A.
- The `flag_off.json` golden was regenerated with `ensure_ascii=True` and is pure ASCII. Its diff against HEAD changes only the `check_agent`, `list_agents`, and `agent_status` description values. The golden reader now specifies UTF-8; no other golden reader without an explicit encoding was found.
- The follow-up test now uses a strictly increasing clock and checks `old PID stopped < launch_started_at < marker ts`; it also seeds old launch fields and verifies all three are replaced. A throwing third-party `state_hook_args` now logs at debug level, yields `[]`, records `hooks_wired=false`, and still allows follow-up to resume. Pi's builder and capability share one state-extension argv helper. The `list_agents` docstring no longer names `stalled`, and the reference doc covers delivered or unconfirmed follow-ups that record a new PID.
- Linux smoke (lead run, 2026-09-26): a worktree MCP server with a headless Sonnet lead spawned `trust-blocked` (Codex cheapest, interactive via herdr) in the untrusted cwd `<scratchpad>/untrusted-smoke`, and `trust-ok` in this trusted worktree. Both records had `interactive=true` and `hooks_wired=true`.
- After about 60 seconds, `trust-blocked` had no state marker. `check_agent` compact and full, `agent_status`, and `list_agents` compact and full all reported `no_marker_since_launch=true` with the hint: "No state marker since launch 56s ago. Likely causes: Codex's folder-trust prompt for <cwd> (look at the agent's terminal and answer it), a login prompt, or a slow start. Hooks may also have failed." `state` was idle (unknown in `agent_status`) and `stalled` was false, unchanged behavior.
- `trust-ok` wrote a Stop marker; all surfaces showed `no_marker_since_launch=false` and `startup_hint=null`. `~/.codex/config.toml` was byte-identical by SHA-256 before and after. Both agents were killed. **Result: PASS.**

### Validation commands

```text
uv run ruff format --check .   # pass: 93 files already formatted
uv run ruff check .            # pass: All checks passed
uv run ty check                # pass: All checks passed
uv run pytest                  # pass: 1885 passed, 6 skipped
```
