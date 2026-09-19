# Post-Implementation Review: Wake Coordinators for Parked Agents

VERDICT: REJECTED

The implementation is close and most accepted plan-review dispositions are correctly realized: `SubagentStop` is inert and unwired, every emitted marker gets a UUID generation, legacy markers use one-buffer byte fingerprints, ack updates are locked/merged and atomically replaced, wakes print before best-effort acknowledgement, the lock timeout is bounded on Windows, ambient readers are validated, `.watch/` is outside the flat snapshot, and `--no-parked` restores edge-only startup behavior. However, acknowledged generations are only filtered during startup seeding, so the advertised per-generation suppression can be bypassed by a later edge. The repository's required full test gate is also currently red. Those two issues prevent approval.

1. **MAJOR — An already acknowledged generation can be delivered again when the same marker bytes/generation produce a later file edge.**

   Evidence: `src/claude_teams/cli.py:724-733`; `src/claude_teams/cli.py:758-776`; `src/claude_teams/cli.py:787-796`; `src/claude_teams/server_simple.py:1597-1604`; `docs/reference/agent-messaging-protocol.md:1529-1542`.

   The ack map is consulted only while seeding pre-existing markers at startup. Changed state files are registered solely through `_waiting_agent`, and the final settled `_parked_candidate` is emitted without comparing its generation with the reader's ack. A concrete single-watcher sequence is: deliver and ack `g1`; re-arm; rewrite/touch `state-worker.json` while preserving `g1`; the mtime/size identity changes, the edge enters `pending_waits`, and `g1` is delivered again. The same applies to a byte-identical legacy marker rewrite. This contradicts both the generation-equality design and the consuming-agent statement that a delivered generation is suppressed on later watches. It also contradicts the plan's accepted legacy-risk statement that an identical-byte rewrite is indistinguishable from no rewrite.

   Recommendation: apply generation suppression to every candidate, not only startup candidates. Keep the reader's ack map for the invocation and, before registering and again before emitting a settled candidate, drop it when `acked[path.name] == candidate.gen` (or use a helper that performs this consistently). Decide explicitly whether `--no-parked` deliberately bypasses this suppression to preserve raw edge-only compatibility; the default mode must honor it. Add regressions that first ack `g1`, then rewrite/touch the marker with the same `g1` (and separately identical legacy bytes) while a default watch is active and assert timeout; rewriting with `g2` must wake.

2. **MAJOR — The required full pytest gate is red, contrary to the implementation record.**

   Evidence: `docs/features/watch-parked-agents/implementation.md:57-64`; `tests/test_follow_up_delivery.py:244`.

   `uv run pytest -q` completed with `1 failed, 1544 passed, 4 skipped`. The failure is `test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record`: actual reason `agent_busy`, expected `resume_not_confirmed`. Running that test alone reproduces the failure. The changed production logic for this feature is not in the follow-up path (the `server_simple.py` diff is docstring-only), so this appears unrelated to the watch change, but the current worktree is not green and the implementation record's `1545 passed` claim is not true for the reviewed state.

   Recommendation: investigate and resolve or explicitly disposition the reproducible failure before merge, rerun the full suite, and update `implementation.md` with the final command/output. Do not report the branch as gate-green until the current head passes. The focused command requested for this review did pass: `122 passed, 2 skipped`.

3. **MINOR — Several contract/docstrings remain internally inaccurate despite the round-2 disposition.**

   Evidence: `src/claude_teams/server_simple.py:1575-1578`; `src/claude_teams/cli.py:623-629`; `src/claude_teams/cli.py:669-673`; `tests/test_cli_watch_parked.py:1-7`; `tests/test_tool_descriptions.py:189-202`; `docs/reference/agent-messaging-protocol.md:1537-1542`.

   `_DISK_CONTRACT_NOTE` still presents the marker schema as only `{state,event,ts}`, although `gen` is now present on every new marker. The watch option help says “once per reader per park” without the documented at-least-once qualification. The main watch docstring still says a transition preceding the initial snapshot may explain exit 2, which is no longer generally true in default parked mode (it is true for `--no-parked`, an already acknowledged park, or an unfinished settle). Most visibly, the new test module's top-level documentation says “the ack is written before the wake is printed,” the exact ordering rejected in round 2 and opposite to `_emit_parked_wake`. The tool-description test checks only keywords and therefore cannot catch these contradictory statements or the promised suppression bug in finding 1.

   Recommendation: add `gen` to the disk schema; qualify “once” as successful-ack suppression with possible at-least-once duplicates; correct the timeout explanation; and change the test-module sentence to print-then-ack. Strengthen the registered-description assertion to pin the qualified suppression/deduplication sentence rather than only independent keywords.

4. **MINOR — The `.watch/` output-edge test is timing-vacuous, and ack-failure/ordering coverage is skipped on Windows.**

   Evidence: `tests/test_cli_watch_parked.py:234-267`; `tests/test_cli_watch_parked.py:302-353`; `docs/features/watch-parked-agents/implementation.md:66-68`.

   `test_ack_files_are_never_an_output_edge` starts with an unacknowledged parked marker and settle `0`, so the watch exits immediately for that marker; the other-reader ack is written only after a `0.08s` sleep, normally after the watch has already exited. The assertion therefore does not demonstrate that an ack write occurring during a live `--pattern "*"` watch is ignored. The filesystem-permission ack-failure test is skipped on Windows and when running as root, so those environments do not pin print-then-ack or non-fatal failure behavior. The helper merge/replace tests are good, but they do not exercise `_emit_parked_wake` ordering.

   Recommendation: for the `.watch/` test, run a broad-pattern watch with no actionable parked marker, write only `.watch/ack-other.json` while it is active, and assert timeout with empty stdout. Add a platform-independent unit test that monkeypatches `_emit_wake` and `_acknowledge` to record call order, plus one that makes `_acknowledge` raise `OSError`/`FileLockTimeoutError` and asserts one waiting JSON record, warning on stderr, and exit 0. This avoids permission-bit and 20–80 ms scheduling dependence.

5. **MINOR — Windows-specific behavior is designed defensively but not directly validated for this feature.**

   Evidence: `src/claude_teams/cli.py:38-45`; `src/claude_teams/cli.py:510-549`; `src/claude_teams/cli.py:684-699`; `docs/features/watch-parked-agents/implementation.md:66-68`.

   Safe-reader validation prevents both slash styles from entering ack paths, unique UUID temp names avoid collisions, and the stable sidecar lock is compatible with atomic data-file replacement. `FileLockTimeoutError` is converted to `OSError` and therefore reaches the non-fatal wake path. These choices look correct. However, the implementation record confirms no Windows run, and the only end-to-end ack-failure test is explicitly skipped there. Also remember that the passed five-second timeout affects the `msvcrt` branch only; the reused POSIX `flock` remains blocking by design.

   Recommendation: add the platform-independent simulated lock-timeout test from finding 4 and run the focused watch/ack tests on Windows before merge. No Windows-specific redesign is indicated.

6. **NIT — The remaining accepted dispositions and the focused implementation are otherwise sound.**

   Evidence: `src/claude_teams/hooks.py:19-29`; `src/claude_teams/hooks.py:108-119`; `src/claude_teams/hooks.py:260-280`; `src/claude_teams/hooks.py:320-397`; `src/claude_teams/cli.py:457-564`; `src/claude_teams/cli.py:623-733`; `tests/test_hooks_parked_marker.py:35-104`; `tests/test_cli_watch_parked.py:66-248`; `tests/test_cli_watch_parked.py:270-366`.

   `_parked_candidate` reads once and hashes the same bytes it parses; `_read_acked` tolerates malformed schema and filters values; `_acknowledge` performs locked load/merge, unique same-directory temp creation, atomic replacement, and cleanup; `_emit_parked_wake` correctly prints before best-effort ack; startup seeding uses the initial pattern-filtered snapshot and enters the existing settle/priority machinery; reader validation covers the ambient source; and hook generation is shared by running and waiting events. Removing `SubagentStop` from the event sets also removes it from Claude, Codex POSIX, and Codex Windows wiring while the legacy watcher filter remains.

   Recommendation: retain these parts. The implementation is not over-engineered: generation ids, legacy fingerprints, the internal subdirectory, and the cross-process merge lock each address a demonstrated compatibility or concurrency requirement. The minimal path to approval is to fix finding 1, make the contract/test corrections in findings 3–4, and restore the full pytest gate.

Validation performed during this review: focused pytest `122 passed, 2 skipped`; `ruff format --check` passed; `ruff check` passed; `ty check` passed; `git diff --check` passed; full pytest `1 failed, 1544 passed, 4 skipped` as detailed above.
