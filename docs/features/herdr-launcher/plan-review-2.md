# Independent plan re-review: Herdr launcher, round 2

## Summary

Revision 2 materially improves the design. It fully closes five of the ten round-one findings and partially closes the other five. The explicit session prefix, strict opt-in selection, complete manager interface, corrected env-validation layer, and decision not to override `resolve_agent_pid` are sound.

One blocker remains in the revised ownership design: it never requires the creation token captured at spawn to be non-null. Under its stated equality check, `None == None` can make `_tracked_alive` return true and cause `_PidOwnershipMixin.ownership_probe` to short-circuit to `OWNERSHIP_OURS`. The revision also treats `_tracked_alive` as though its boolean can express “indeterminate,” which is not how the existing mixin works. The server-generation and auto-start-lock designs remain too abstract to implement or test as written.

This review read revision 2, round-one review, the existing locking implementation in `src/claude_teams/filelock.py` and its call sites, and read-only Herdr help for `api`, `session list`, and `status`. No Herdr state-changing or server command was run.

## Round-one finding disposition

### BLOCKER 1 — Named-session routing: RESOLVED

**Closing text:** Plan §“Session routing,” lines 156-174; `HerdrProcessInfo` in §“Spawn,” lines 203-205; tests 7-10, lines 303-305.

The revision defines one mandatory `_herdr_argv` constructor, puts the top-level `--session <name>` prefix before all commands, makes the selected session immutable, records it in process info, and explicitly includes cleanup/status/server-start calls. That closes the original wrong-session `tab close` risk for a configured session.

There is a separate ambiguity in how an unconfigured manager discovers the current Herdr session; see New MAJOR 3. It does not negate the concrete configured-session fix.

### BLOCKER 2 — PID ownership proof: PARTIALLY RESOLVED

**Addressing text:** Plan §“Ownership, liveness, kill, capture, send,” lines 225-248; `HerdrProcessInfo`, lines 203-205; tests 18-23, lines 315-319.

The plan now stores a token, checks session/generation, validates `process-info`, requires `shell_pid == info.pid`, and revalidates the token before a bare-PID kill. Those are the correct ingredients. It does not require the stored token to be non-null, and it incorrectly assumes every failed compound check maps to an indeterminate ownership result. `_PidOwnershipMixin` does not preserve the reason `_tracked_alive` returned false; it falls through to its own token logic. See New BLOCKER 1.

### MAJOR 1 — `HERDR_ENV` auto-selection/import behavior: RESOLVED

**Closing text:** Plan §“Launcher selection — strictly opt-in,” lines 136-154; tests 1-6, lines 299-301.

Herdr is now selected only by explicit `WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`. `HERDR_ENV=1` alone cannot select it. The selector and constructor are specified as pure, with binary/server work deferred until spawn. This preserves existing Linux behavior and avoids import-time subprocess side effects.

### MAJOR 2 — Server/client states and auto-start supervision: PARTIALLY RESOLVED

**Addressing text:** Plan §“Session policy,” lines 176-199; Risks 1 and 3, lines 280-288; tests 35-39, lines 332-335.

The revision correctly separates attached-client, headless-server, and no-server states, retracts the visibility claim for headless mode, specifies detached non-inherited stdio, retention/reaping, early-exit checking, readiness polling, and documents the server-wide blast radius.

The startup lock has no defined canonical path or named primitive, and the “server generation” has no concrete acquisition algorithm. These are load-bearing details, not implementation trivia. See New MAJOR 2 and New MAJOR 3.

### MAJOR 3 — Transient `foreground_processes[0]` as agent PID: RESOLVED

**Closing text:** Plan §“Ownership, liveness, kill, capture, send,” lines 238-243; current-behavior verification, lines 72-76.

The plan no longer overrides `resolve_agent_pid`; it returns the persisted shell/agent handle. This is correct because `_build_posix_shell_command` ends in `exec {shlex.join(cmd)}`. A successful `exec` replaces the shell process image without changing its PID, while foreground children created later by hooks/tools are transient and should not replace the authoritative handle.

This conclusion is conditional on the live-matrix checks already listed at lines 344-347: `pane run` must execute in that root shell, `shell_pid` must identify it, and the PID must survive the `exec`. If `exec` fails, the interactive shell may remain alive even though no agent started; that is a spawn-confirmation limitation, not a reason to use `foreground_processes[0]`. The existing protocol likewise says a returned PID does not prove prompt receipt; the first lifecycle marker remains the meaningful evidence.

### MAJOR 4 — JSON/error and capture semantics: PARTIALLY RESOLVED

**Addressing text:** Plan §“The `_run_herdr` boundary,” lines 256-267; capture rules, lines 249-252; tests 25-30 and 33, lines 321-330.

The revision now specifies timeout, return-code and error-envelope checks, typed required fields, a named exception, and empty-success versus malformed capture behavior. It also acknowledges that the visible fallback may be only a viewport.

Two gaps remain. First, “every other error is indeterminate” is not representable through `_tracked_alive: bool` or `health_check: tuple[bool, str]` without an additional internal status model; the existing mixin discards why `_tracked_alive` returned false. Second, installed `pane read --help` only shows that `--lines` is optional; it does not establish that omitting it means “full available history.” The plan should define the actual argv and return behavior for `lines=None` as the maximum Herdr makes available, without asserting equivalence to tmux unless the live matrix verifies it.

### MAJOR 5 — Test coverage/seams/live assumptions: PARTIALLY RESOLVED

**Addressing text:** Plan §“Test cases,” lines 295-340, and §“Live matrix,” lines 342-350.

Coverage is now broad and the isolated named-session live matrix includes all three backends, resume, process/PID behavior, restart behavior, PTY behavior, and capture. That resolves the round-one coverage omissions.

However, the statement that unit tests fake a single `_run_herdr` seam is inconsistent with the design. Detached server startup must use `subprocess.Popen`, not the synchronous JSON-returning `_run_herdr`. Token behavior calls the module-level `creation_token`; concurrency depends on a real/fake file-lock seam; readiness depends on time/polling. Tests 6, 18-23, and 37-39 therefore need additional explicit seams. See New MAJOR 4.

### MINOR 1 — Incomplete manager interface: RESOLVED

**Closing text:** Plan §“The interface a manager must satisfy,” lines 38-70; tests 23-24.

The revised table includes all public manager methods used by callers and separately names `_processes`, `_pid_alive`, and `_tracked_alive` as subclass obligations. It correctly explains the mixin short-circuit.

### MINOR 2 — Wrong env-validation layer: RESOLVED

**Closing text:** Plan current behavior, lines 72-76; Spawn step 3, lines 210-214; tests 14-15.

The plan now relies on upstream `_SAFE_ENV_KEY` validation, does not misuse `_validate_safe_name`, and tests difficult values as single argv tokens.

### MINOR 3 — Overstated “verified” claims: PARTIALLY RESOLVED

**Addressing text:** Plan §“Observed Herdr behavior,” lines 78-117; Risks 4-5, lines 289-293; live matrix, lines 342-350.

Most wording is now appropriately scoped to one observed 0.8.2 run, capture completeness is qualified, and pane-ID non-reuse is no longer used as PID proof. The remaining server-generation and socket-discovery claims are still not backed by a defined CLI field or algorithm. See New MAJOR 2 and New MINOR 1.

## New findings

### BLOCKER 1 — Nullable tokens and the mixin fallback still make the ownership proof unsound

**Concern:** Plan §“Spawn” step 6, lines 218-219; §“Ownership, liveness, kill, capture, send,” lines 227-248; §“The `_run_herdr` boundary,” lines 265-267; `_PidOwnershipMixin.ownership_probe` in `src/claude_teams/backends/process_manager.py:311-365`.

The compound check says:

```text
creation_token(str(info.pid)) equals info.creation_token
```

but the plan never says `info.creation_token` must be a non-empty token or that spawn fails when token capture returns `None`. In Python, `None == None` is true. If both the stored and current token reads fail, the other pane checks can pass and `_tracked_alive` can return true. The mixin then immediately returns `OWNERSHIP_OURS` at lines 332-333, authorizing destructive behavior with no PID-reuse proof. This directly reopens round-one BLOCKER 2.

There is a second semantic mismatch. When `_tracked_alive` returns false because `process-info` timed out or returned a non-not-found error, `ownership_probe` does not return “indeterminate” on that basis. It does this:

- no `expected_token` → `OWNERSHIP_NOT_OURS`;
- dead PID → `OWNERSHIP_NOT_OURS`;
- live PID with matching token → `OWNERSHIP_OURS`;
- live PID with unreadable token → `OWNERSHIP_INDETERMINATE`;
- token mismatch → `OWNERSHIP_NOT_OURS`.

Thus line 266’s promise that every non-not-found Herdr error is indeterminate is false for the inherited API.

The correct trade-off differs by caller:

- **Lease/claim reclaim (`ownership_probe`)**: the existing mixin is safe once a real expected token exists. A live matching token returns `OURS`; an unreadable live token returns `INDETERMINATE`; neither permits reclaim. A transient Herdr CLI failure must not be converted to `NOT_OURS` while the original PID/token still matches.
- **Destruction (`owns_process`)**: a matching creation token is sufficient authority to signal that PID, but not by itself to issue `tab close` or `pane send-keys`. Those Herdr object operations need separately proven session/generation/object identity. If Herdr identity is indeterminate, the safe fallback is a token-revalidated PID operation only, not a tab/pane operation.
- **Read-only health (`health_check`)**: reporting dead on every transient Herdr timeout is an avoidable false negative. If the original PID and token still match, process liveness is true even if Herdr control is temporarily unavailable; return alive with a degraded/indeterminate detail. Definitive pane absence plus a dead/mismatched PID is dead. A boolean health API cannot faithfully encode all four internal states, so the manager needs an internal richer probe and an explicit projection.

**Suggested fix:** Require token capture to return a non-empty token before registration; otherwise close the newly created tab and fail spawn. Require both stored and live tokens to be non-null before equality. Define an internal Herdr probe enum/result such as `owned`, `gone`, `identity_mismatch`, and `indeterminate`, with error detail. Let `_tracked_alive` project only exact `owned` to true, but have `health_check`, pane/tab lifecycle operations, and PID fallback consume the richer result according to the caller-specific rules above. Add explicit tests for `stored_token is None`, `live_token is None`, transient control failure plus matching token, and the distinct health/destruction/reclaim outcomes.

### MAJOR 1 — The compound `_tracked_alive` can cause health false negatives, while not actually enforcing pane-operation fail-closed behavior

**Concern:** Plan lines 227-248 and 265-267.

Making `_tracked_alive` strict is correct for the mixin’s in-memory ownership shortcut. It is not sufficient as the manager’s one liveness primitive. A single CLI timeout, malformed response, or temporarily unavailable socket makes it false. The plan says tracked `health_check` uses `_tracked_alive` first but does not specify a token-aware fallback for a tracked handle, so a live agent can be reported dead. That can make status and delivery confirmation conclude the child died when the only failure was the control plane.

At the same time, `owns_process` will often recover to `OURS` through the matching token after `_tracked_alive` fails. Unless `graceful_shutdown` and `kill_process` run their own richer object-identity probe, they may still send keys or close a tab after the supposed fail-closed check failed. The strict boolean therefore fails in opposite directions for different callers.

**Suggested fix:** Do not make `health_check` a raw alias of `_tracked_alive`. Specify the richer probe and caller projections described above. Treat timeout/non-not-found control errors as degraded live when PID+token match for health, as non-reclaimable for leases, and as prohibiting pane/tab commands while still allowing a token-checked OS-PID kill when destruction was authorized.

### MAJOR 2 — “Server generation” is not defined by any discoverable Herdr CLI contract

**Concern:** Plan §“Server-generation guard,” lines 193-199; process info lines 203-205; compound check line 230; tests 22 and live matrix restart check.

Read-only help exposes `herdr status server --json`, `herdr status client --json`, `herdr session list --json`, and `herdr api snapshot`, but it names no server generation, start-time, boot ID, server PID, or socket-identity field. The plan’s parenthetical “server start time / socket identity” is a choice between two different mechanisms, not an implementable definition.

A filesystem socket inode/ctime could be used as a locally derived endpoint identity, but the plan does not say so or establish its behavior across ordinary restart, live handoff/update (`herdr --handoff` exists), stale socket files, or named sessions. A server-reported start time/PID may or may not exist in JSON; `--help` does not promise it. Marking all entries dead on a generation change is particularly risky if a supported handoff changes the server endpoint while preserving pane processes.

The generation guard is also not needed for PID safety if the manager verifies session, pane `shell_pid`, and a non-null Linux creation token: a fresh server reusing a pane ID cannot make a different process match both the old PID and old `/proc/<pid>/stat` start time.

**Suggested fix:** Either remove generation from the ownership proof and rely on the concrete pane/PID/token conjunction, or specify one exact, obtainable generation value and validate it in the isolated live matrix across restart and handoff. Do not implement against a guessed JSON field. If socket `(st_dev, st_ino)` is chosen, define when it is sampled, what stat/read failures mean, and why endpoint replacement implies old panes died on this Herdr version.

### MAJOR 3 — Auto-start locking lacks a canonical shared path and ignores the repository’s existing primitive

**Concern:** Plan §“Session routing,” lines 156-174; §“Session policy,” lines 186-191; Risk 3; files affected, lines 269-276.

This repository already has exactly one documented cross-process advisory lock implementation: `claude_teams.filelock.file_lock` (`src/claude_teams/filelock.py:1-14, 70-84`), used by `delivery_store` and backed by the same `lock_handle`/`unlock_handle` model as the agent registry. The plan neither names it nor lists `filelock.py` as affected.

More importantly, “a file lock under the session dir” is ambiguous. If it means the win-agent-teams agent-session directory, two independent team sessions targeting the same Herdr session take different locks and can both start the same Herdr server. The lock must be keyed by the canonical Herdr session/socket identity and shared by every MCP process targeting it. If it means the Herdr directory returned by `session list`, a never-before-created named session may not yet appear, so the plan needs a stable pre-start lock location. The critical section must re-check server status after acquiring the lock, then perform `Popen`, early-exit/readiness handling, and release in `finally`.

On POSIX, the existing `file_lock(timeout_s=...)` uses blocking `flock`; its timeout parameter is only effective on Windows (`filelock.py:10-14, 43-59`). That is consistent with current repository locking but should not be described as a bounded acquisition. The readiness poll can still be bounded while the lock wait is not.

**Suggested fix:** Explicitly reuse `claude_teams.filelock.file_lock`, define a canonical per-Herdr-session lock path available before the server exists (respecting `HERDR_CONFIG_PATH`/session selection), and hold it across a status recheck and the bounded start/readiness transaction. Add a two-distinct-win-agent-session concurrency test proving both processes converge on the same lock. If bounded POSIX lock acquisition is required, that is a deliberate shared-helper change and must be scoped/tested rather than invented privately in the manager.

### MAJOR 4 — The 41-case suite requires more than the claimed single `_run_herdr` seam

**Concern:** Plan §“The `_run_herdr` boundary,” lines 256-263; test preamble line 297; tests 6, 18-23, and 35-39.

`_run_herdr` can fake synchronous JSON control calls. It cannot safely represent the long-running detached `herdr server` child: using `subprocess.run` would block for the server lifetime, while `Popen` returns no command JSON to validate. Early-exit/reaping tests need a fake `Popen` object with `poll`/`wait`. Ownership tests need a separate seam for module-level `creation_token` and `_pid_alive`. Concurrent-start tests need controllable locking plus time/readiness ordering; import-purity tests need a subprocess sentinel outside `_run_herdr` because they prove the helper was never reached or bypassed.

Backend tests 40-41 can capture the final `pane run` argv through `_run_herdr`, but that proves construction, not an actual shell byte round-trip. The live matrix appropriately owns the latter proof; the unit-test wording should not call it byte-for-byte execution proof.

**Suggested fix:** Define at least `_run_herdr` for finite control commands and `_popen_herdr_server` (or an injected Popen factory) for daemon launch, and explicitly monkeypatch/inject creation-token, PID-liveness, clock/sleep, and lock boundaries where relevant. Test the real `file_lock` separately with two processes if cross-process serialization itself is a promised property. Reword tests 40-41 as exact command/env construction characterization.

### MINOR 1 — Session-name discovery and failure socket reporting remain speculative

**Concern:** Plan lines 158-160 and 186-191.

`HERDR_ENV=1` is a boolean context marker, not a session name. The plan says session identity comes from `WIN_AGENT_TEAMS_HERDR_SESSION` else `HERDR_SESSION`/`HERDR_ENV` context, but it never defines which concrete variable contains a name or how an unpinned invocation is recorded as an immutable identity. The installed `herdr --skill` documents `HERDR_WORKSPACE_ID`, `HERDR_TAB_ID`, and `HERDR_PANE_ID`, but does not document `HERDR_SESSION` as a public variable. Unqualified CLI calls may correctly route through inherited caller context, but that is not the same as having a session name suitable for `info.session` equality.

Likewise, `session list --json` is available, but help does not guarantee that a brand-new named session which failed during startup already has a list row/socket path. The promise to always report a discovered socket path may be impossible in the earliest failure window.

**Suggested fix:** Define the exact authoritative context input supported by Herdr 0.8.2, or represent the session as a resolved socket endpoint rather than an assumed env name. For startup failure, report a discovered socket path when available and otherwise report the selected session plus the authoritative command, without requiring a guessed path.

## Verdict

VERDICT: CHANGES REQUIRED — 1 BLOCKER
