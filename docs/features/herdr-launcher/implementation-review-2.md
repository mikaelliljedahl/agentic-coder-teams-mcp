# Independent implementation re-review: Herdr launcher, round 2

## Summary

The revision substantially improves the implementation. The local identity state machine now has coherent non-zombie classifications, silent success is command-specific, both JSON streams are inspected with error precedence, server absence is no longer inferred from query failure, spawn rollback is much stronger, and the Popen seam can exercise real lifecycle behavior. The focused 101-test suite and all static gates pass.

One end-to-end ownership hole remains a blocker. `HerdrOwnershipUnprovenError` protects the record only if `kill_process` is reached. Both `kill_agent` and follow-up first call Boolean `owns_process`; when the token is already unreadable, that collapses `OWNERSHIP_INDETERMINATE` to false. `kill_agent` then deletes the durable record without calling the new protected path, while follow-up can start a resumed worker beside the unproven live original. The process-manager-only tests cannot catch either caller failure.

I also found two localized correctness regressions: `HERDR_CONFIG_PATH` is a config **file** path but is used as a directory, and the new local liveness helper has dropped the existing zombie check. Neither requires redesign, but both need correction.

Validation performed:

- `uv run pytest tests/test_backends/test_process_manager_herdr.py -q`: **101 passed**.
- focused `ruff check`, `ruff format --check`, and repository `ty check`: passed.
- `git diff --check ed1a4b0..HEAD`: passed.
- Read-only Herdr 0.8.2 verification confirmed `session list --json` uses `sessions[].socket_path`, and `herdr --help` describes `HERDR_CONFIG_PATH` as overriding the config **file path**.

## Round-1 finding disposition

### MAJOR 1 — `_probe` did not establish PID/token state first: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2721-2778`; tests `tests/test_backends/test_process_manager_herdr.py:1262-1319`.

`_local_identity` now runs before the pane query. A dead PID, unreadable live token, and readable mismatching token become `PID_GONE`, `INDETERMINATE`, and `IDENTITY_MISMATCH` respectively. Once PID/token match, a missing pane, different pane shell PID, control-plane failure, and matching pane become `PANE_GONE`, `PANE_GONE`, `INDETERMINATE`, and `OWNED`. This closes the original false-life/false-death cases and keeps `_tracked_alive` safe for the mixin short-circuit.

The new `_pid_is_live` implementation introduces a separate zombie regression, reported below; it does not negate the corrected ordering and state definitions for ordinary processes.

### MAJOR 2 — Endpoint rebinding was unreachable: PARTIALLY RESOLVED

**Code:** `_ensure_server`/`_note_endpoint` at `src/claude_teams/backends/process_manager.py:2507-2530`; `_probe` at `:2746-2753`; tests at `tests/test_backends/test_process_manager_herdr.py:1033-1059,1530-1541`.

The endpoint is now genuinely re-resolved on every spawn, so a later spawn can update `self.socket_endpoint`; an existing record then rebinds only after its pane/PID/token proof. That is a real production path, unlike round 1’s manual-only assignment.

It is still not a general handoff path. A handoff followed by health/send/capture/kill but no new spawn never refreshes the manager endpoint, so the record remains stamped with the old path and no rebind is logged. The original manual-attribute rebinding test remains, while the new discovery test stops after changing the manager cache and does not prove an existing record’s full transition. This residual is operational/observability rather than a PID-safety hole because every object operation still re-proves pane/PID/token through the immutable session selector.

### MAJOR 3 — Silent success was global and stream parsing incomplete: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2388-2408,2468-2505,3121-3163`; call site `:2690-2692`; tests `tests/test_backends/test_process_manager_herdr.py:1384-1417`.

Only `pane run` opts into empty exit-0 success. Other enveloped calls reject silence. Both streams are parsed and a structured error wins even when stdout contains success. Legitimate Herdr errors on stderr continue to work; a non-JSON stderr warning beside valid stdout is ignored, which is the sensible compatibility choice. A success object printed only on stderr is also accepted; that is permissive but not unsafe and preserves the existing either-stream boundary.

### MAJOR 4 — Server discovery conflated failure with absence: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2507-2613`; tests `tests/test_backends/test_process_manager_herdr.py:1475-1541`.

Only a well-formed literal `running: false` is absence. Timeout/malformed/unavailable responses propagate, running status is type-checked, the socket fallback comes from `session list --json`, Popen `OSError` is named, and cached reachability is checked on each spawn. The extra finite status subprocess adds latency to every spawn and a transient timeout now fails that spawn, but that is the correct fail-closed trade-off: using the cache or auto-starting would treat uncertainty as endpoint authority or absence. A bounded retry could improve availability later without weakening this rule.

The read-only real CLI output confirms the implemented `socket_path` field. The separate config-lock regression below is not a session-list issue.

### MAJOR 5 — Failed/expired server children leaked: PARTIALLY RESOLVED

**Closing code:** constructor and lifecycle at `src/claude_teams/backends/process_manager.py:2350-2354,2532-2536,2580-2627`; tests `tests/test_backends/test_process_manager_herdr.py:1423-1472`.

The child is retained immediately, early exit is polled/reaped, a never-ready child is terminated, and later ensures poll a retained child. The Popen fake now exposes terminate/kill/wait rather than restating liveness through `poll` alone.

One reaping edge remains: after `terminate()` times out, `_terminate_server_child` calls `kill()` but never `wait()` and has already cleared `_server_child`. A stubborn child can therefore still become an untracked zombie. This is a localized follow-up below.

### MAJOR 6 — Failed spawn rollback was incomplete: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2679-2719,3069-3099`; tests `tests/test_backends/test_process_manager_herdr.py:1547-1621`.

The rollback scope now begins as soon as a tab ID can be recovered from a partial response. Pane validation, run, token capture, and provenance write are all inside it; the manager record is committed only afterward. Cleanup failure is attached to the original exception. Tests cover partial create, provenance failure, and rollback failure.

### MAJOR 7 — Kill removed the record before stop authorization: NOT RESOLVED end to end

**Process-manager fix:** `src/claude_teams/backends/process_manager.py:2842-2894`; direct tests `tests/test_backends/test_process_manager_herdr.py:1343-1381`.

Within `HerdrProcessManager`, this is fixed: `_force_kill_settled` keeps the record and raises when a live PID’s token is unreadable. But the real public caller at `src/claude_teams/server_simple.py:5876-5895` first calls Boolean `owns_process`. If the token is unreadable from the outset, the mixin correctly returns `OWNERSHIP_INDETERMINATE`, `owns_process` collapses that to false, and `kill_process` is skipped. Lines 5882-5884 nevertheless delete the durable record and return success. The exact abandoned-agent failure therefore remains through a path the new exception cannot intercept. See BLOCKER 1.

### MAJOR 8 — Graceful shutdown used pane mismatch as process-exit proof: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2896-2932`; tests `tests/test_backends/test_process_manager_herdr.py:800-849,1325-1340`.

Completion now depends only on original PID identity. A different/missing pane does not imply exit; token mismatch or a non-live PID does. An unreadable token on a live PID conservatively waits to the deadline, which can be longer than the former immediate return but is intentional: the former prompt return was the bug. A moved live process receives token-checked SIGINT and gets the normal grace period before force kill.

The process-manager method is correct. The caller’s preceding Boolean ownership gate can still skip it and resume concurrently on an indeterminate token; that is part of BLOCKER 1, not a flaw in `_original_process_gone`.

### MINOR 1 — Refused send/capture did not record a reason: RESOLVED

**Closing code:** `src/claude_teams/backends/process_manager.py:2983-3003`; test `tests/test_backends/test_process_manager_herdr.py:1624-1638`.

Both operations share the same state-recording gate and retain their public signatures.

### MINOR 2 — Startup lock was not under the effective Herdr config directory: NOT RESOLVED

**Code:** `src/claude_teams/backends/process_manager.py:2643-2656`; test `tests/test_backends/test_process_manager_herdr.py:1641-1648`.

The default path is improved, but `HERDR_CONFIG_PATH` is misinterpreted as a directory. It is documented by Herdr as a config file path. This becomes a functional failure (and can create a directory where a config file should be), detailed as MAJOR 1 below.

### MINOR 3 — Strict JSON decoding was also applied to pane display text: NOT RESOLVED

**Code:** `src/claude_teams/backends/process_manager.py:2410-2434,2455-2466`.

The text seam still passes through `_invoke(errors="strict")`, and the implementation comment still calls all output machine protocol. There is no invalid-byte capture test. Keep strict decoding for JSON, but give terminal display text an explicit tolerant policy consistent with `read_log_tail(errors="replace")`.

### MINOR 4 — README visibility claim and implementation evidence were stale: PARTIALLY RESOLVED

**Locations:** `README.md:280-283`; `docs/features/herdr-launcher/implementation.md:59-70,156-169`; `nested-live-test.md`.

The implementation update correctly states that the focused suite is now 101 tests and adds better live evidence. The earlier validation block still says 73 new tests/1461 total, and README still claims an unpinned default is necessarily “the session you are sitting in” even though the manager never establishes attached-client identity. The nested report also retains the earlier “five of six” summary despite listing more checks and later documenting a successful rerun.

## New and remaining findings

### BLOCKER 1 — Boolean ownership gates still abandon or duplicate an indeterminate live agent

**Locations:** `src/claude_teams/server_simple.py:4518-4527,5876-5895`; `_PidOwnershipMixin` at `src/claude_teams/backends/process_manager.py:329-383`; new exception path at `:2871-2894`.

The mixin intentionally distinguishes `OWNERSHIP_INDETERMINATE` because an unreadable token on a live PID is not evidence that the owner is gone. The two lifecycle callers collapse it too early:

- `kill_agent`: `owns_process == False` skips `kill_process`, then removes the durable agent record and returns success. The live process and the manager’s private record survive, but no durable agent remains to address or retry.
- follow-up/resume: when `plan.alive` is true but `owns_process == False` solely because ownership is indeterminate, the condition skips both graceful and force shutdown and proceeds directly to `backend.resume`. Old and new workers can run concurrently on one conversation.

If the first token read happens to succeed and a later read fails, `_force_kill_settled` now raises before durable deletion, and ordinary exception unwinding releases the registry lock; that narrower path is sane and safer than the former leak. The problem is that persistent or first-read uncertainty never reaches it. No server-level test exercises the new exception or the initial indeterminate gate.

**Concrete fix:** use the three-valued `ownership_probe` at both lifecycle decisions. For `OWNERSHIP_OURS`, run graceful/kill. For `OWNERSHIP_NOT_OURS`, the original process is gone/reused and cleanup/resume may proceed. For `OWNERSHIP_INDETERMINATE`, return a structured retriable refusal and preserve the durable record/lease; do not resume. Catch a later `HerdrOwnershipUnprovenError` (preferably a backend-neutral ownership-unproven exception) into the same result. Add server-level tests proving kill retains `agents.json` and follow-up does not invoke resume when the initial probe is indeterminate, plus the transition where the first probe is ours and the kill-time token becomes unreadable.

### MAJOR 1 — `HERDR_CONFIG_PATH` is treated as a directory, breaking startup under the documented override

**Locations:** `src/claude_teams/backends/process_manager.py:2643-2656`; `tests/test_backends/test_process_manager_herdr.py:1641-1648`.

`herdr --help` says `HERDR_CONFIG_PATH` “overrides config file path.” The code uses `Path(config)` as `root`, yielding `<config-file>/win-agent-teams-….lock`. On an existing config file, `file_lock`’s parent `mkdir` fails with `FileExistsError`/`NotADirectoryError`. If the file does not yet exist, the launcher can create a directory at the intended config-file location, preventing Herdr from creating/reading the file later.

The test encodes the same wrong assumption by setting the variable to `tmp_path / "cfg"` and asserting that is the lock parent.

**Concrete fix:** use `Path(config).expanduser().parent` for the override and the Herdr config directory for the default. Test with `HERDR_CONFIG_PATH=<tmp>/cfg/config.toml`, create that file, acquire the actual `file_lock`, and assert the lock is a sibling under `<tmp>/cfg`.

### MAJOR 2 — The new liveness helper regresses zombie handling

**Locations:** `_pid_is_live` at `src/claude_teams/backends/process_manager.py:2258-2270`; callers at `:2755-2778,2871-2894,2919-2932`; existing zombie-aware implementation at `:2939-2968`.

`_pid_is_live` uses only `kill(pid, 0)`. A zombie therefore counts as live, and its `/proc/<pid>/stat` creation token remains readable. `_local_identity` can classify a dead zombie as `PANE_GONE`/`INDETERMINATE` and `health_check` reports it alive; `_original_process_gone` can wait the entire graceful timeout; `_force_kill_settled` can “kill” the unsignalable zombie and pop the record. The manager’s existing `_pid_alive` explicitly rejects zombies, but the new static helpers bypass it.

The tests monkeypatch `_pid_is_live`, while the older `_probe_setup(pid_alive=...)` still monkeypatches the now-unused manager `_pid_alive`; some older death tests therefore depend accidentally on host PID 4242 being absent. No test covers a zombie.

**Concrete fix:** make these instance methods use the manager’s zombie-aware `_pid_alive`, or consolidate one shared PID-state helper that includes the `/proc` zombie check. Update `_probe_setup` to patch the boundary actually used and add a zombie fixture/fake for probe, graceful completion, and force-kill settlement.

### MINOR 1 — Force-killed server children are still not reaped

**Location:** `src/claude_teams/backends/process_manager.py:2615-2627`; fake at `tests/test_backends/test_process_manager_herdr.py:857-880`.

After a terminate wait times out, the code calls `child.kill()` but does not perform the final `wait()`. Because `_server_child` was cleared before cleanup, the later reaper cannot collect it. The fake’s `wait` never raises, so this branch is untestable as written.

**Concrete fix:** after `kill()`, call a bounded/unbounded final `wait` appropriate for a child just sent SIGKILL, retaining the Popen until it is reaped. Give the fake a scripted first `TimeoutExpired` and assert the kill plus final wait.

### MINOR 2 — Existing-agent endpoint rebind remains dependent on an unrelated later spawn

**Location:** `src/claude_teams/backends/process_manager.py:2507-2530,2721-2753`.

This is the residual from round-1 MAJOR 2. Refreshing once per spawn is reasonable for latency, but the code and documentation should not claim that `_probe` notices handoff immediately. Either refresh boundedly before an existing record’s first object operation after handoff, or explicitly define rebinding as eventual-on-next-spawn and test that contract. No PID-safety change is required.

### MINOR 3 — The checked-in nested driver cannot reproduce all of the documented assertions

**Locations:** `scripts/herdr_nested_check.py:110-118,144-165`; `docs/features/herdr-launcher/nested-live-test.md:16-36,84-113`.

The current tab-presence assertion iterates tab dictionaries and evaluates `CHILD in tab`, which tests dictionary keys, not `tab["label"]`. The prior commit had a `labels` list; HEAD regressed it. More importantly, `_tab_labels` maps every malformed/error response to an empty list, so the final “kill closed the tab” check passes if Herdr is simply unavailable. The document’s stronger “marker and process gone” claim is not checked by the driver at all.

The marker and inbox paths are constant across runs and the driver does not prove they were absent before spawn or newer than the spawn. A failed prior run can therefore satisfy later existence/content checks. The historical report says the disposable session was deleted, and its timestamped `Stop` marker plus the Codex TUI line and received message are persuasive evidence for that run; the checked-in script is not yet a robust reproducer.

**Concrete fix:** restore label extraction; make Herdr query errors fail the check; record the spawn start time and require a valid marker with `event`/`ts` newer than it; establish that marker/inbox evidence is absent or use a unique session/child token; after kill assert PID death, marker deletion, and either a successful tab list without the label or the specific expected workspace-not-found code.

### MINOR 4 — Plain-text capture still shares strict JSON decoding

**Location:** `src/claude_teams/backends/process_manager.py:2410-2434,2455-2466`.

This is unresolved round-1 MINOR 3. Keep strict decoding for JSON; explicitly decode terminal output for display with replacement (or another documented tolerant policy), and test invalid pane bytes.

### MINOR 5 — User-facing visibility/evidence wording remains overstated

**Locations:** `README.md:280-283`; `docs/features/herdr-launcher/implementation.md:59-70,163-169`; `docs/features/herdr-launcher/nested-live-test.md:16-25,84-113`.

README still equates the selected/default server with the user’s attached session without consulting `status client`. The implementation document contains both the old 73-test validation and the new 101-test count, and summarizes “five of six” despite the nested document’s larger table and later all-pass rerun. Update these as documentation cleanup; they do not affect process safety.

## State-machine and caller audit

For non-zombie processes, every `_HerdrProbe` state is reachable and correctly scoped:

| State | Reachable condition | Health/object/destruction consequence |
| --- | --- | --- |
| `OWNED` | live matching token + pane reports same shell PID | alive; object operations allowed |
| `PANE_GONE` | live matching token + pane absent or reports another PID | degraded-alive; no pane/tab operation; token-checked PID stop |
| `PID_GONE` | token unreadable and PID not live | dead; no operation |
| `IDENTITY_MISMATCH` | corrupt null stored token or readable token mismatch | dead/original gone; no operation |
| `INDETERMINATE` | live unreadable token, or matching token plus non-not-found pane failure | degraded-alive; no object operation; PID signal only after fresh token match |

No new deterministic path signals a known foreign/reused PID. Both graceful and force PID signals require a fresh non-null token equal to the stored token. The remaining token-check-to-signal race is the same non-atomic OS limitation already present in neighboring managers. Herdr object operations retain their analogous probe-to-command race but are always preceded by full proof.

`graceful_shutdown` can now consume the full timeout when a live token remains unreadable, when a moved/unmanaged process ignores SIGINT, or when a pane command fails while the original process remains. Those are honest outcomes, not regressions: returning early would falsely authorize resume. A dead/reused original returns promptly from `_original_process_gone`. The zombie exception is MAJOR 2.

Per-spawn endpoint revalidation costs one bounded Herdr status subprocess. If status becomes unreachable between spawns, the new spawn fails rather than using stale authority or attempting a duplicate server. That is the safe behavior; it should be surfaced clearly and may later receive a bounded retry.

## Test-quality assessment: would each round-1 major regress again?

| Round-1 major | Would current tests catch a reintroduction? | Assessment |
| --- | --- | --- |
| 1 — local identity ordering | **Mostly yes** | New cross-product tests catch unreadable-live, dead+timeout, and mismatch+timeout. They miss zombies, and old `pid_alive` injection is stale. |
| 2 — unreachable rebinding | **Partly** | A new test catches a cache that is never refreshed on a later spawn. Existing-record rebinding is still driven by manual cache assignment; handoff without another spawn is untested. |
| 3 — global silence/one stream | **Yes** | Dedicated tests fail if silence becomes general or stderr error loses precedence. |
| 4 — failure-as-absence/guessed socket | **Yes for the reported bugs** | Tests cover status failure, malformed status, session-list socket, and cache refresh. The real read-only CLI confirms `socket_path`. |
| 5 — server child leaks | **Partly** | Timeout termination and later `poll` are tested; terminate-timeout → kill → final reap is not representable by the fake. |
| 6 — incomplete spawn rollback | **Yes** | Partial response, provenance failure, and cleanup failure are behavioral assertions across the rollback boundary. |
| 7 — premature record deletion | **No end to end** | Direct manager tests catch its old pop order, but no test crosses `server_simple.kill_agent`; the initial Boolean gate still reproduces the leak. |
| 8 — false graceful completion | **Yes in the manager, not the caller** | Direct tests prove a live original returns false and a gone original true. No follow-up test proves indeterminate ownership prevents resume. |

The fakes are materially better: server-child cleanup and status discovery can now be scripted, and several tests assert externally meaningful transitions rather than mapping `expect` directly to a canned result. Remaining weak spots are the stale `_pid_alive` seam, manual rebinding, absence of server-level ownership tests, and the nested driver issues above.

## Nested live-test evidence

The state-marker argument is technically sound for the recorded fresh run. Normal `spawn_agent` does not write a spawned Codex agent’s marker; `claude_teams.hooks emit` writes it from a lifecycle hook configured into that agent. A fresh marker containing `{"event":"Stop"}` therefore proves the Codex CLI started far enough inside the Herdr pane to run its lifecycle hook. It does not alone prove the model performed the requested work, but the separately observed Codex TUI status bar and upstream `GRANDCHILD ALIVE` message complete that chain.

The report appropriately admits the first run’s inbox error and distinguishes the weak early pane-content check. Its historical evidence supports spawn → Codex run/hook → message for the rerun. It does **not** by itself prove pane move, handoff, attached-client capture, all three backends, or ownership-indeterminate behavior. The committed driver also does not currently substantiate its strong kill cleanup claim or protect itself from stale artifacts, as MINOR 3 explains.

## Verdict

VERDICT: CHANGES REQUIRED — 1 BLOCKER
