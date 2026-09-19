# Plan Review Round 2: Wake Coordinators for Parked Agents

VERDICT: APPROVED WITH CHANGES

V2 fixes the two original root causes and is substantially simpler and safer than v1. Generation equality, legacy byte fingerprints, the `.watch/` subdirectory, ambient-reader validation, and `--no-parked` are appropriate; none is over-engineered for the stated once-per-reader behavior. Approval requires correcting the delivery ordering/contract and testing the locked same-reader merge. The other recommendations below are clarifications that should be incorporated while making those changes.

1. **MAJOR — Ack-before-print contradicts the claimed at-least-once guarantee and can permanently lose a wake.**

   Evidence: `docs/features/watch-parked-agents/plan.md:65-73`; `docs/features/watch-parked-agents/plan.md:78-79`; `docs/features/watch-parked-agents/plan.md:158-159`; `src/claude_teams/cli.py:472-474`; `src/claude_teams/cli.py:669-675`.

   The plan says the ack is persisted before `_emit_wake`. If the process exits, is killed, or stdout fails after the ack replacement but before the JSON reaches stdout, the next watch suppresses that generation permanently. That is an at-most-once failure mode, not at-least-once. Allowing delivery after an ack write failure only handles the opposite branch; it does not repair the acknowledged-but-unprinted crash window. Thus round-1 dispositions 3 and 7 are not fully resolved as written.

   Recommendation: print and flush the success JSON first, then attempt the ack, logging ack failure to stderr and still exiting 0. A crash between print and ack may duplicate the next delivery, which is exactly the acceptable at-least-once tradeoff. Update lines 68-70 and the implementation/test description accordingly. If the project instead chooses ack-before-print, label it at-most-once and explicitly accept lost wakes—but that is a poor fit for this feature's purpose.

2. **MAJOR — The same-reader locked read/modify/write guarantee still lacks the test requested in round 1.**

   Evidence: `docs/features/watch-parked-agents/plan.md:52-59`; `docs/features/watch-parked-agents/plan.md:109-110`; `docs/features/watch-parked-agents/plan.md:145-150`; `docs/features/watch-parked-agents/plan.md:160-162`; `src/claude_teams/filelock.py:70-84`.

   Test 18 covers different readers, which use different files and therefore does not exercise the lock. Test 21 covers successive invocations, not concurrent writers. Neither proves the disposition's central claim that two same-reader read/modify/write operations cannot lose different ack entries. Duplicate delivery from simultaneous same-reader watchers is documented and acceptable, but lost merged entries are not.

   Recommendation: add a deterministic concurrency test against the ack helper, or run two same-reader watches concurrently with disjoint marker patterns (for example `state-a.json` and `state-b.json`) and a barrier that forces both toward the update. Assert the final single ack file contains both entries and valid JSON. Keep the documented single-live-watcher rule and permissible duplicate delivery.

3. **MINOR — Specify atomic ack replacement and a single byte read for legacy candidate identity.**

   Evidence: `docs/features/watch-parked-agents/plan.md:43-50`; `docs/features/watch-parked-agents/plan.md:52-59`; `src/claude_teams/filelock.py:72-76`.

   V2 specifies a locked read/modify/write but no longer explicitly says the data file is written via unique temporary file plus atomic `replace`. The reused lock's own contract assumes precisely that pattern. A direct write can leave corrupt data after a crash; tolerant loading would then discard every previously acked entry. Similarly, “ONE parse” is not enough for legacy markers if hashing and JSON parsing perform separate filesystem reads: an intervening replacement could make the fingerprint identify different bytes from those parsed.

   Recommendation: state that the ack helper performs locked load/merge plus unique same-directory temp write and atomic replace. State that `_parked_candidate` reads the marker bytes once, hashes those exact bytes when needed, and decodes/parses that same buffer. Add a focused atomic-write/temp-cleanup test; the concurrent merge test from finding 2 then covers the lock interaction.

4. **MINOR — The tool-docstring wording overpromises exactly-once behavior.**

   Evidence: `docs/features/watch-parked-agents/plan.md:68-73`; `docs/features/watch-parked-agents/plan.md:85-93`; `docs/features/watch-parked-agents/plan.md:109-110`.

   Section C says a fresh watch wakes “once per reader per park,” while section B correctly allows duplicates after ack failure and from two live same-reader watchers. Even after fixing output-before-ack, a crash in the delivery/ack window also permits a duplicate. Since MCP tool descriptions are the consuming agent's contract, “once” is too strong.

   Recommendation: describe this as “a successfully acknowledged generation is suppressed on later watches for that reader” and explicitly say delivery is at-least-once, so duplicates are possible after interruption/ack failure or unsupported concurrent same-reader watches. Pin that qualified wording in `tests/test_tool_descriptions.py` rather than merely checking generic contract words.

5. **MINOR — Ack lock timeout/blocking behavior should be included in the failure policy.**

   Evidence: `docs/features/watch-parked-agents/plan.md:56-59`; `docs/features/watch-parked-agents/plan.md:69-73`; `src/claude_teams/filelock.py:10-14`; `src/claude_teams/filelock.py:31-33`; `src/claude_teams/filelock.py:43-58`.

   On Windows, `file_lock` can raise `FileLockTimeoutError` after 30 seconds; that must be handled like any other ack failure. On POSIX, `flock` blocks without a timeout, so a live holder can delay process completion beyond the watch deadline and owner checks. The lock is held only for a short merge, so this is unlikely, but the plan currently calls the success/exit behavior unchanged without acknowledging the blocking path.

   Recommendation: explicitly include lock acquisition/timeout failures in the non-fatal ack-failure path. Prefer passing a short bounded timeout on Windows. It is acceptable to retain the repository's existing blocking POSIX lock model, provided the plan does not claim the watch deadline bounds post-wake acknowledgement work.

6. **NIT — Round-1 dispositions 1, 2, 5, 6, 8, and 10 are resolved; dispositions 3, 4, 7, and 9 need only the changes above.**

   Evidence: `docs/features/watch-parked-agents/plan.md:33-39`; `docs/features/watch-parked-agents/plan.md:43-47`; `docs/features/watch-parked-agents/plan.md:52-55`; `docs/features/watch-parked-agents/plan.md:74-81`; `docs/features/watch-parked-agents/plan.md:83-105`; `docs/features/watch-parked-agents/plan.md:118-170`; `src/claude_teams/cli.py:404-421`; `src/claude_teams/cli.py:477-488`.

   Disposition-by-disposition verification:

   - Round-1 finding 1 is resolved: making `SubagentStop` inert removes the race and the false `waiting` state without synchronization.
   - Finding 2 is resolved: UUID generations remove timestamp ordering, while hashing the exact legacy bytes provides a workable compatibility identity.
   - Finding 3 is structurally resolved by one candidate record, subject to correcting delivery ordering in finding 1 above.
   - Finding 4 is only partially resolved: the lock design prevents lost merges, but the test plan does not verify it (finding 2 above).
   - Finding 5 is resolved: `_snapshot_mtimes` scans only direct files, so `.watch/` and its contents cannot become `reason="output"`, even for `--pattern "*"`.
   - Finding 6 is resolved: validating the final resolved identity before path construction covers explicit readers, ambient nested leads, and Windows separators.
   - Finding 7 is resolved except for the mislabeled delivery order; `--no-parked`, unchanged JSON, and exit behavior are otherwise explicit and well tested.
   - Finding 8 is resolved: kill-time cleanup is correctly cut.
   - Finding 9 is substantially resolved by tests 6-21, except for the missing same-reader concurrent merge test.
   - Finding 10 is resolved: the affected files now include the MCP docstring source and a contract test, subject to the qualified wording in finding 4 above.

   Recommendation: make the four minimal edits above—reverse wake/ack order, add the same-reader merge test, require atomic ack replacement/single-buffer parsing, and qualify the docstring's “once” language. With those changes, the plan is ready for implementation; no redesign or scope reduction is needed.
