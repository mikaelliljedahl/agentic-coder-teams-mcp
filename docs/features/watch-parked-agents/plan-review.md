# Plan Review: Wake Coordinators for Parked Agents

VERDICT: REJECTED

The two observed root causes are correctly identified, and a persistent per-reader acknowledgement is a better fit than coupling acknowledgement to `read_messages`/`agent_status`, adding a caller-managed `--since`, or comparing only with watch start. However, the proposed design does not yet guarantee its central properties under concurrent hook/watch activity, and its timestamp rules are internally inconsistent. The plan should be revised before implementation.

1. **BLOCKER — Design A does not actually guarantee that `SubagentStop` “never downgrades” a parked marker.**

   Evidence: `docs/features/watch-parked-agents/plan.md:33-41`; `src/claude_teams/hooks.py:70-75`; `src/claude_teams/hooks.py:98-106`.

   The proposed read-check-return is separate from the atomic replacement. If `Stop` and `SubagentStop` hook processes both read an older `running`, missing, or corrupt marker, both decide to write and a later `SubagentStop` replacement can still overwrite the `Stop`. Atomic replacement prevents torn JSON, not lost-update races. Moreover, the code and watcher documentation say `SubagentStop` is parent-agent churn rather than a parent lifecycle state (`src/claude_teams/cli.py:37-41`, `445-469`), so preserving its current write over `running` continues to make status falsely report `waiting`.

   Recommendation: prefer the smaller semantic fix: make `SubagentStop` a no-op in the state-marker emitter (remove it from the waiting-state mapping while retaining hook wiring if needed). That cannot overwrite either `running` or a genuine `Stop` and eliminates the read/modify/write race. If backward compatibility requires retaining `SubagentStop` markers, specify and implement a cross-process, cross-platform serialized update/CAS protocol and add a deterministic concurrent `Stop`/`SubagentStop` test. The guard itself would not mask a genuine state under the documented meaning of `SubagentStop`; the risky assumption is treating delivery of a later running event as infallible (`plan.md:86-87`).

2. **BLOCKER — The acknowledgement comparison is internally contradictory and fails for legacy markers.**

   Evidence: `docs/features/watch-parked-agents/plan.md:48-61`; `src/claude_teams/cli.py:445-469`; `tests/test_cli_watch.py:607-613`.

   The plan says a candidate requires `marker_ts > acknowledged_ts`, with both missing values mapped to `0`, but also says a missing/non-numeric marker timestamp is “never ack-able” and always a candidate. Under the stated comparison, an unacknowledged legacy marker has `0 > 0`, which is false, so it will not wake even once. After storing `0`, it still cannot be both acknowledged and always eligible. Numeric `time.time()` ordering also does not identify generations safely: wall clocks can move backwards, and equal timestamps are possible, so a genuine later park can be suppressed by `>` despite the Windows/platform-neutral claim.

   Recommendation: define a generation identity rather than an ordering. The cleanest contract is a unique marker generation/id written on each actionable `Stop`, with the ack storing that id and eligibility defined as inequality. If the marker schema must remain unchanged, define a stable fingerprint of the exact marker bytes (or a documented composite identity) and explicitly support legacy/missing timestamps. Add clock-rollback, equal-timestamp, missing-timestamp, and non-numeric-timestamp tests.

3. **MAJOR — The marker generation being acknowledged is not defined atomically with the settled wake.**

   Evidence: `docs/features/watch-parked-agents/plan.md:50-59`; `src/claude_teams/cli.py:651-675`.

   During settle, the current code rereads the marker only to decide actionability. The plan then says to write “the woken marker's `ts`” without saying whether that value was captured during candidate registration, the final settle read, or a later read. If the marker changes between the final actionability check and a later timestamp read, the watcher can acknowledge a newer park that it never emitted. Conversely, acknowledging an old value is safe only if the newer generation remains detectably unequal and eligible on the next invocation.

   Recommendation: replace `_waiting_agent` for this path with one parse that returns an immutable candidate record containing agent, generation, and path. At settle completion, parse once, decide, and acknowledge exactly that returned generation. Never reread the marker merely to choose what to acknowledge. Specify failure ordering: persist the ack before emitting success for at-most-once behavior, or emit before ack for at-least-once behavior; exact-once cannot be claimed across process crashes.

4. **MAJOR — Concurrent watchers using the same reader can duplicate wakes and lose ack entries.**

   Evidence: `docs/features/watch-parked-agents/plan.md:45-59`; `src/claude_teams/cli.py:577-611`; `src/claude_teams/cli.py:669-675`.

   Per-reader files correctly isolate different readers, but two watcher processes for the same reader can both load the same ack, both wake for the same marker, and both perform read/modify/atomic-replace. If they wake for different agents, the last replacement can discard the other process's entry. Unique temp files and `replace` prevent partial files but do not serialize the update.

   Recommendation: either establish and enforce/document a single-live-watcher-per-reader invariant, or use a cross-process lock around ack eligibility plus update. A lock must work on Windows as well as Linux and must not be held through the 15-second settle period. Add tests for two readers (each must wake once), a nested lead whose reader is not `team-lead`, and two simultaneous watchers sharing one reader. This is one reason acking in `read_messages` or `agent_status` is inferior: neither operation is guaranteed to occur for a parked child, and `agent_status` has no natural per-reader consumption semantics.

5. **MAJOR — A session-local ack file becomes a spurious `reason="output"` source under supported broad patterns.**

   Evidence: `docs/features/watch-parked-agents/plan.md:48-49`; `src/claude_teams/cli.py:636-645`; `tests/test_cli_watch.py:577-604`; `docs/reference/agent-messaging-protocol.md:1559-1565`.

   With `--pattern "*"`, every `watch-ack-<reader>.json` write is classified as ordinary output because only `state-*.json` receives semantic classification. One watcher can therefore wake another watcher—or a subsequent broad-pattern watch—on internal bookkeeping rather than user output. The plan's file-enumeration note does not address this behavior.

   Recommendation: explicitly exclude all watch-internal ack/temp/lock files from snapshots and output classification regardless of `--pattern`, or store acknowledgements in a dedicated internal subdirectory that watch never traverses. Pin this with a broad-pattern regression test involving an ack update from another reader/process.

6. **MAJOR — Reader resolution must be safe for ack writes and must preserve nested-lead identity.**

   Evidence: `src/claude_teams/cli.py:477-488`; `src/claude_teams/cli.py:581-591`; `tests/test_cli_watch.py:301-316`; `tests/test_cli_watch.py:670-688`; `CLAUDE.md:78-94`.

   The explicit `--reader` is validated, but the ambient `AGENT_NAME` branch is not. Today that branch constructs paths used for reads; the plan would additionally create/replace an ack path derived from it. On Windows, backslashes are separators, making validation before any write especially important. The ack must use the resolved current reader, not a hard-coded `team-lead`, because nested agents are leads for their own children.

   Recommendation: validate the final resolved reader regardless of whether it came from `--reader`, `AGENT_NAME`, or the fallback, before constructing any inbox or ack path. Add an unsafe ambient-reader test plus ack-path tests for both nested ambient identity and explicit `--reader` override.

7. **MAJOR — Backward-compatibility and ack failure behavior need an explicit contract.**

   Evidence: `docs/features/watch-parked-agents/plan.md:62-64`; `src/claude_teams/cli.py:553-573`; `docs/reference/agent-messaging-protocol.md:1546-1557`; `tests/test_cli_watch.py:260-269`.

   Default behavior intentionally changes a pre-existing waiting marker from timeout/exit 2 to JSON success/exit 0; `--no-parked` is a reasonable compatibility escape hatch. The emitted success object can remain byte-for-schema compatible (`reason`, `agent`, `path`), but the plan does not say what happens if the ack file cannot be read, created, locked, or replaced. Returning exit 0 without a durable ack gives duplicate future wakes; failing after output risks a success line with a nonzero exit; acknowledging before output risks a lost notification if stdout fails.

   Recommendation: document the delivery guarantee and error policy in the plan before coding. Preserve the existing one-line JSON shapes and exit codes for successful wakes, exit 2 with empty stdout for timeouts, and exit 4 for owner loss. Add exact-output/exit tests, unwritable/replace-failure tests, timeout-before-settle with no ack, `--no-parked` followed by a genuine new edge, and pre-existing parked-marker priority against both messages and output.

8. **MINOR — `kill_agent` acknowledgement cleanup is unnecessary coupling and its proposed test has no assertion.**

   Evidence: `docs/features/watch-parked-agents/plan.md:90-93`; `docs/features/watch-parked-agents/plan.md:114`.

   “Removes the entry (or leaves the file valid)” does not choose behavior. Scanning and rewriting every reader's ack during kill introduces more lost-update races and makes kill depend on watcher internals. With a correct generation-inequality scheme, an old entry for a deleted marker is harmless, and a later same-name agent writes a different generation.

   Recommendation: cut kill-time ack mutation from this feature. Allow stale entries and document them as harmless; normal session cleanup removes the whole session. If bounded file growth is a real requirement, design pruning separately under the same ack lock and give it precise tests.

9. **MAJOR — The test plan does not cover the protocol's highest-risk interleavings.**

   Evidence: `docs/features/watch-parked-agents/plan.md:95-114`; `tests/test_cli_watch.py:345-365`; `tests/test_cli_watch.py:497-604`; `tests/test_hooks.py:72-159`.

   The listed happy-path tests are useful, especially message priority and corrupt-file tolerance, but they omit: concurrent `Stop`/`SubagentStop`; a marker changing during settle/ack; two simultaneous same-reader watchers; independent multiple readers; nested-reader ack paths; clock rollback/equal timestamps; missing/non-numeric timestamps; broad `--pattern`; ack I/O failure; timeout before settle; and more than one pre-existing parked marker across successive invocations. Existing settle tests cover transient and overlapping edge-triggered waits, not these pre-existing/ack interactions.

   Recommendation: add deterministic tests for every omitted case above. Retain the existing tests that assert `SubagentStop` edge filtering and settle priority, updating only those whose old pre-existing-marker expectation is intentionally replaced. Test cases should assert exact JSON, exit code, stdout emptiness on timeout, and exact ack contents—not merely that a path substring appears.

10. **MAJOR — The consuming-agent contract must be updated in MCP tool docstrings, not merely prose documentation.**

    Evidence: `docs/features/watch-parked-agents/plan.md:66-78`; `CLAUDE.md:72-74`; `src/claude_teams/cli.py:553-573`; `docs/reference/agent-messaging-protocol.md:1510-1557`.

    The plan correctly names the `spawn_agent` watch-command note and `agent_watch_paths`, but the affected-files section omits the source file(s) containing those tool docstrings, and it does not enumerate the contract they must expose. The current CLI/protocol contract explicitly says pre-snapshot waiting may time out; that statement becomes false by default.

    Recommendation: add the MCP tool-description source files and any docstring contract tests to “Files affected.” Require the tool descriptions to state: default parked-marker recovery; per-reader once-only acknowledgement semantics; reader identity for nested leads; the settle delay for pre-existing markers; message > output > waiting priority; `--no-parked`; unchanged success JSON; timeout/owner exit codes; and whether acknowledgement is at-most-once or at-least-once. Update the reference section and marker table consistently. README/skill prose is secondary, not a substitute.
