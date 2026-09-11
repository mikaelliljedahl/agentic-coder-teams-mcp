# Independent plan re-review: Herdr launcher, round 3

## Summary

Revision 3 closes all six findings raised as new findings in round 2. In particular, token capture is now mandatory and non-null, the richer probe is projected explicitly by caller, the speculative generation field is gone, the repository lock is reused at the correct cross-team scope, daemon launch has its own seam, and session identity no longer invents an environment variable.

The central PID-reuse proof is now sound. `OWNED` can short-circuit `_PidOwnershipMixin.ownership_probe` safely because it requires a responding pane at the stored endpoint, exact `shell_pid`, and a current non-null Linux creation token equal to the non-null spawn token. A replacement process cannot satisfy that conjunction under the same creation-token invariant the rest of this repository relies upon.

Two major lifecycle details still need revision before implementation. `_HerdrProbe.GONE` conflates “stored pane ID is absent” with “agent PID is dead,” which can report a moved-but-live pane agent dead. Also, using socket inode identity makes a supported `--handoff` permanently degrade otherwise provably identical live panes, despite the plan explicitly recognizing that handoff can replace the endpoint while preserving pane processes. These are not foreign-PID signalling holes, so the blocker count is zero, but they prevent approval.

## Round-2 finding disposition

### BLOCKER 1 — Nullable tokens and mixin fallback: RESOLVED

**Closing text:** Plan §“Spawn,” lines 238-262; §“Ownership, liveness, kill, capture, send,” lines 264-315; tests 20-24 and 32-35, lines 403-418.

Spawn now treats `creation_token(handle) is None` as failure, closes the created tab, and never registers the process. `HerdrProcessInfo.creation_token` is explicitly non-empty. `OWNED` requires both stored and live tokens to be non-null before equality, so `None == None` cannot forge ownership.

The plan also now describes the actual inherited fallback rather than claiming a Herdr control error automatically becomes `OWNERSHIP_INDETERMINATE`: after `_tracked_alive` returns false, `ownership_probe` still evaluates the expected PID token exactly as `process_manager.py:332-350` specifies. This is correct for lease/claim reclaim.

### MAJOR 1 — Strict `_tracked_alive` cuts both ways: RESOLVED

**Closing text:** Plan lines 266-315 and Risks 4, lines 373-376; caller-projection tests 32-35.

The revision no longer uses the strict boolean as the sole health/lifecycle answer. It defines `_HerdrProbe`, maps only `OWNED` to `_tracked_alive=True`, reports `INDETERMINATE` as degraded-alive when PID and token still match, withholds pane/tab operations unless object identity is proven, and permits only token-revalidated PID signalling during a control-plane failure. This addresses the transient-CLI false-negative and wrong-object-operation concerns from round 2.

A different false-negative remains because `GONE` itself is too broad; see New MAJOR 1.

### MAJOR 2 — Undefined server generation: RESOLVED

**Closing text:** Plan §“Session policy,” lines 226-234; Risks 1-2, lines 360-368; live matrix, lines 442-450.

The generation guard is removed rather than renamed. No guessed Herdr JSON field is used. Restart and handoff behavior is assigned to the live matrix, while ownership uses the concrete endpoint/pane/PID/token conjunction.

Removal does not reopen the PID-reuse hole. After a server restart, even if a pane ID is reused, a new process must also have the old numeric PID and the old `/proc/<pid>/stat` start-time token to become `OWNED`. The repository already treats that token as PID-reuse-distinguishing. If the socket endpoint changes, the endpoint check rejects `OWNED` even earlier. If a stale socket path is reused and the endpoint comparison falls back to path, the PID/token conjunction still prevents false ownership.

There remains a handoff availability problem caused by the chosen endpoint identity, not by removal of generation; see New MAJOR 2.

### MAJOR 3 — Auto-start locking: RESOLVED

**Closing text:** Plan §“Session policy,” lines 202-224; files affected, lines 347-356; tests 42-45, lines 427-433.

The design now explicitly reuses `claude_teams.filelock.file_lock`, matching `src/claude_teams/filelock.py:70-84`. It derives a lock path from the target Herdr session under the Herdr config directory, rather than from a win-agent-teams session, so independent teams targeting the same endpoint contend on the same file. The path exists before the server, the status check is repeated inside the critical section, and release is in `finally`.

The plan accurately reflects the helper’s platform behavior: POSIX acquisition is blocking `fcntl.flock` (`filelock.py:57-58`), while the timeout loop is Windows-only. Only readiness after acquisition is claimed bounded. No `filelock.py` change is needed.

### MAJOR 4 — One subprocess seam cannot cover daemon startup: RESOLVED

**Closing text:** Plan §“Subprocess seams,” lines 327-345; test preamble and auto-start/construction cases, lines 383-440.

Finite JSON commands use `_run_herdr`; the long-running detached daemon uses `_popen_herdr_server` returning a `Popen`-like object with `poll`/`wait`. Token/PID, clock/sleep, and lock boundaries are separately injectable. This is implementable: finite protocol fixtures, early exit/reaping, readiness timing, ownership, and startup races no longer need to be forced through one incompatible fake.

Tests 46-47 are correctly narrowed to exact command/env construction. The live matrix, not a fake subprocess, owns shell round-trip proof.

### MINOR 1 — Session/socket discovery speculation: RESOLVED

**Closing text:** Plan §“Session routing,” lines 157-190; startup failure handling, lines 219-224.

The plan uses only `WIN_AGENT_TEAMS_HERDR_SESSION` as an optional configured name, never treats `HERDR_ENV=1` as a name, resolves a concrete endpoint through the available JSON status/session commands, and allows failure reporting to fall back to session plus command when no socket row exists. That removes the invented `HERDR_SESSION` dependency and the impossible promise that a socket is always discoverable.

## Ownership and restart analysis

### Why the `OWNED` short-circuit is safe

For an in-memory record, `_PidOwnershipMixin._has_live_registry_entry` calls `_tracked_alive`; only `OWNED` returns true. Revision 3 defines `OWNED` as the conjunction of:

1. the resolved endpoint matches the stored endpoint;
2. `process-info` succeeds for the stored pane ID;
3. its `shell_pid` equals the stored PID;
4. the stored token is non-null;
5. a fresh token for that PID is non-null and equals the stored token.

Therefore the mixin’s lines 332-333 may return `OWNERSHIP_OURS` without a second token read: `_tracked_alive` has already performed the token proof. Pane identity alone cannot win, and two null reads cannot win.

For a control-plane error, `_tracked_alive` is false and the mixin falls through. A matching live expected token returns `OURS`; an unreadable live token returns `INDETERMINATE`; a dead PID or mismatch returns `NOT_OURS`. That is safe for reclaim: only the last state permits another caller to take the lease/claim.

For destruction, the manager re-probes. `OWNED` permits the Herdr object operation; `INDETERMINATE` permits no tab/pane command and only a freshly token-validated signal; `IDENTITY_MISMATCH` touches nothing. The token is revalidated again before the post-close force kill. No stated path authorizes signalling a known reused/foreign PID. The tiny check-to-`os.kill` race is the same process-identity limitation already accepted by the repository’s common PID/token model, not a Herdr-specific regression.

### Effect of removing generation

A fresh server can reuse the textual pane ID only if Herdr violates or resets its advertised non-reuse behavior. Even then, it cannot produce false `OWNED` merely through that ID: its process must also match both the old numeric PID and old creation token. Socket-inode reuse does not change that conclusion. A legitimate handoff can preserve the real PID/token and pane, which correctly means the agent is still the same process; the remaining issue is allowing the manager to re-establish endpoint identity safely, discussed below.

## New findings

### MAJOR 1 — `GONE` conflates missing pane identity with dead process liveness

**Concern:** Plan `_HerdrProbe` definition, lines 269-275; health projection, lines 290-294; graceful shutdown, lines 309-315; live matrix line 446.

`GONE` is defined as “pane definitively absent, **or PID dead**,” and `health_check` reports every `GONE` dead. Pane absence does not prove PID death. Herdr’s documented `pane move` behavior gives a concrete counterexample: moving a pane gives it a new workspace-qualified pane ID while the process continues. The manager’s stored pane ID becomes absent, yet its stored PID and creation token can still match a live agent. A user closing/rearranging topology or a handoff-related remap can create the same logical state.

This creates exactly the live-agent false negative the round-two review asked the richer probe to prevent. It can also leave lifecycle behavior underspecified: `graceful_shutdown` says `GONE` does nothing, while `kill_process` gives concrete branches only for `OWNED` and `INDETERMINATE`. If pane-absent/PID-live is treated as dead, `kill_agent` can remove the record while leaving the original agent running. The inherited `ownership_probe` would independently return `OURS` when the PID/token still match, so the manager’s health and ownership answers would contradict each other.

**Suggested fix:** Separate object absence from process death, for example `PANE_GONE` and `PID_GONE`, or calculate PID/token liveness before projecting. A missing pane plus matching live PID/token should be degraded-alive/unmanaged, should remain non-reclaimable, and should allow only token-checked PID shutdown/kill—not pane/tab commands. Only a dead PID or a token mismatch should report process death. Add tests for pane not found + matching live token, pane move/new ID, and the corresponding health, reclaim, graceful, and force-kill projections.

### MAJOR 2 — Socket inode identity makes legitimate handoff permanently unmanaged

**Concern:** Plan endpoint identity, lines 176-188; removal rationale, lines 226-234; `OWNED` conjunction, lines 277-280; Risks 1 and 4; live matrix lines 446-448.

The plan correctly says `herdr --handoff` can replace the endpoint while pane processes survive. It nevertheless stores `(st_dev, st_ino)` whenever the socket can be statted and requires the current endpoint to equal that stored value for `OWNED`. If handoff replaces the socket inode while retaining panes/PIDs, every existing record becomes permanently non-`OWNED`. Health can remain degraded-alive through token fallback, but `send-keys`, `tab close`, and normal pane capture/input can no longer be used. `kill_agent` falls back to PID signalling and cannot fulfill the planned “confirm the tab closes” behavior; a dead or orphaned tab may remain.

This is safe against foreign PID signalling but unnecessarily turns a supported continuity event into permanent control-plane loss. It also acts like the generation guard revision 3 says it removed: socket inode replacement is effectively an endpoint generation marker.

**Suggested fix:** Use the canonical socket path/session selector as stable session identity and use pane/PID/token for object/process identity, or support safe endpoint rebinding. Rebinding is provable when the newly resolved endpoint returns the same stored pane ID, the same `shell_pid`, and the same non-null creation token; only then update `info.socket_endpoint`. Define the behavior before implementation and verify both inode-preserving and inode-replacing handoff in the live matrix.

### MAJOR 3 — `send` is a mutating pane operation but has no stated ownership projection

**Concern:** Plan destruction projection, lines 300-305; concrete lifecycle methods, lines 307-325.

The plan carefully withholds `pane send-keys` during `graceful_shutdown` unless the object is `OWNED`, but the public `send` method directly issues `pane send-text`/`send-keys` without saying it first requires `OWNED`. `send` mutates terminal input and has the same wrong-pane risk as shutdown input. `capture` is read-only, but should at least refuse or report degraded when the stored endpoint/pane cannot be proven rather than silently reading an unrelated object.

Within one uninterrupted Herdr session, non-reused IDs make a stale pane ID fail rather than hit another pane. Across restart/handoff, however, the entire reason for the endpoint/PID/pane conjunction is to avoid treating a textual ID as sufficient. Applying it only to kill paths leaves the interface internally inconsistent.

**Suggested fix:** State that `send` requires `_HerdrProbe.OWNED`; all other states send nothing (with the existing void return, log or raise a named error consistently with current manager conventions). Gate `capture` on verified object identity too, allowing an explicit empty/error diagnostic policy. Add tests proving `INDETERMINATE`, `GONE`, and `IDENTITY_MISMATCH` never send text or keys.

### MINOR 1 — Constructor purity conflicts with endpoint resolution “at construction”

**Concern:** Plan launcher selection, lines 153-155; session routing, lines 159-188.

The constructor is required to perform no `shutil.which`, server probe, or subprocess, but session routing says the manager resolves its session once “at construction” via `herdr session list --json` / `status server --json`. Both cannot be true. The intended split appears to be: capture and validate the optional configured session name in the pure constructor; resolve and freeze the socket endpoint lazily during the first `spawn_process` server-ensure step.

**Suggested fix:** State that split explicitly and define behavior before any process has been spawned. Tests should assert constructor purity and first-spawn endpoint resolution separately.

### MINOR 2 — `capture(lines=None)` still claims an unverified tmux-equivalent history mode

**Concern:** Plan capture rule, lines 320-323; live matrix lines 447-448.

Installed `herdr pane read --help` shows `--lines <N>` as optional but does not say omission means the pane’s full available history. Revision 3 still states that `lines=None` “requests full available history (tmux semantics).” The live matrix can measure 0.8.2 behavior, but the design should not make the stronger semantic claim before that evidence exists.

**Suggested fix:** Define `lines=None` mechanically as omitting `--lines` and returning whatever full/default snapshot Herdr 0.8.2 supplies; document any difference from tmux after the live check. This is diagnostic-only and non-blocking.

## Verdict

VERDICT: CHANGES REQUIRED — 0 BLOCKERS
