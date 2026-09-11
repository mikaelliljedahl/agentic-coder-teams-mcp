# Independent plan re-review: Herdr launcher, round 4

## Summary

Revision 4 resolves all five round-3 findings. The ownership/liveness state machine is now coherent, the PID-reuse proof remains sound without a server-generation field, pane moves do not become false process deaths, endpoint continuity can be proven across handoff, every mutating Herdr object operation is identity-gated, and the constructor/capture contracts are implementable as written.

No blocker or structural major remains. I found three minor wording/hardening items that should be handled during implementation: one stale sentence still says `HERDR_ENV` supplies a session name, rebinding should explicitly be constrained/logged/bounded, and the Risks section retains stale state/kill wording. None requires changing the design’s structure.

## Round-3 finding disposition

### MAJOR 1 — `GONE` conflates missing pane identity with dead process liveness: RESOLVED

**Closing text:** Plan §“Ownership, liveness, kill, capture, send,” lines 286-348; caller-projection tests 33-37, lines 457-464; live matrix pane-move check, lines 493-500.

The enum now separates `PANE_GONE` from `PID_GONE`. PID/token liveness is evaluated before projecting pane absence. A missing stored pane with a matching live PID/token is degraded-alive and unmanaged, not dead; only a dead PID or token mismatch becomes process death. That aligns `health_check` with inherited `ownership_probe`, which also treats a matching live PID/token as ours/non-reclaimable.

The lifecycle path is complete. `PANE_GONE` withholds pane/tab commands, uses token-checked `SIGINT` for graceful shutdown, and uses the token-revalidated PID force path if needed. The record is therefore still removable even though its old pane ID cannot be managed.

### MAJOR 2 — Socket inode identity breaks handoff: RESOLVED

**Closing text:** Plan §“Session routing,” lines 177-205; §“Session policy,” lines 243-251; `OWNED` definition, lines 295-298; test 39 and live matrix, lines 465-466 and 493-500.

Socket inode identity is removed. Stable routing identity is the canonical socket path/session selector. When that endpoint changes, continuity is not assumed: the new endpoint must report the stored pane ID, exact stored shell PID, and a fresh non-null creation token equal to the stored non-null token before the record is rebound.

That conjunction is sufficient for accidental safety under this repository’s process-identity model. A different honest Herdr server or second session may coincidentally reuse a textual pane ID and numeric PID, but it cannot make a different process have the old Linux `/proc` start-time token. If the old process itself survived a legitimate handoff and appears through the new endpoint, adoption is correct: it is the same agent process.

A malicious same-user socket service could inspect `/proc`, lie in its JSON, and echo all three values. Rebinding is not a security boundary against that attacker; neither are environment identity, writable registry files, or the rest of this same-user architecture. The implementation should say so and apply the minor hardening in New MINOR 2, but the design need not add authentication for this feature.

### MAJOR 3 — `send`/`capture` are not identity-gated: RESOLVED

**Closing text:** Plan lines 328-365; tests 38 and 43, lines 464 and 471-474.

All mutating Herdr object operations—`tab close`, `pane send-keys`, and `pane send-text`—now require `OWNED`. The public `send` method sends nothing in every other state and records the reason. `capture` is also gated; it returns an empty diagnostic result with a recorded reason rather than reading a pane whose identity is not proven. This closes the stale/reused-pane input and disclosure paths.

### MINOR 1 — Constructor purity conflicts with endpoint resolution: RESOLVED

**Closing text:** Plan lines 183-192 and test 6, lines 428-431.

The constructor now captures and validates only the optional configured session name. Binary/server/endpoint discovery is lazy during the first spawn’s server-ensure step, after which the canonical socket path is frozen. This is consistent with the pure-constructor and import-time selection requirements.

### MINOR 2 — `capture(lines=None)` overclaims tmux-equivalent history: RESOLVED

**Closing text:** Plan lines 357-365; test 42, lines 471-472; live matrix lines 498-500.

`lines=None` is now defined mechanically as omitting `--lines` and returning the snapshot Herdr 0.8.2 supplies. No full-history or tmux-equivalence guarantee is made before the live matrix measures it.

## State-machine and ownership audit

### `OWNED` and the mixin short-circuit

The `OWNED` conjunction is safe for `_PidOwnershipMixin.ownership_probe`’s in-memory short-circuit at `process_manager.py:332-333`:

1. The call is routed through the stored/resolved session selector.
2. That endpoint successfully resolves the stored pane ID.
3. The pane reports the stored `shell_pid`.
4. The stored creation token is non-null, enforced at spawn.
5. A fresh creation token for that PID is non-null and equal.

Only this state makes `_tracked_alive` true. Therefore the mixin cannot return `OWNERSHIP_OURS` from pane identity alone or from two null tokens.

### Restart, handoff, stale entries, and ID reuse

Removing the generation guard does not create a false-`OWNED` path:

- **Ordinary server death/restart:** old pane processes die by the stated Herdr lifecycle. A stale record therefore fails the PID/token proof. Even if the fresh server reuses the pane ID and numeric PID, the replacement process has a different creation token.
- **Stale endpoint or second honest session:** matching pane/PID text is insufficient; the independently read local creation token must also match. A different process fails.
- **Legitimate handoff:** if the pane and process survive, all three continuity values match and rebinding is correct. If any differs, no rebind occurs.
- **Pane move:** the old pane ID is absent but the PID/token still match, producing `PANE_GONE`, not false ownership or false death.

The repository already treats Linux process start time as the PID-reuse discriminator. The remaining token-check-to-signal race is common to the existing managers and is not reopened by endpoint rebinding.

### Per-state projection matrix

| Probe state | `health_check` | Inherited reclaim/`ownership_probe` | `graceful_shutdown` | `kill_process` | `send` | `capture` |
| --- | --- | --- | --- | --- | --- | --- |
| `OWNED` | alive | short-circuits `OURS` | pane Ctrl-C, then poll | tab close, token-check before PID fallback | allowed | allowed |
| `PANE_GONE` | alive, degraded | matching PID/token → `OURS` | no pane op; token-checked SIGINT | no tab op; token-checked PID kill | no-op + reason | `""` + reason |
| `PID_GONE` | dead | dead → `NOT_OURS` | no-op/report | no-op/report | no-op + reason | `""` + reason |
| `IDENTITY_MISMATCH` | dead | mismatch/dead → `NOT_OURS`; if original PID/token still match, conservatively `OURS` | touch nothing | touch nothing | no-op + reason | `""` + reason |
| `INDETERMINATE` | alive/degraded only when PID/token still match; otherwise token result governs | match → `OURS`; unreadable live token → `INDETERMINATE`; mismatch/dead → `NOT_OURS` | no pane op; token-checked SIGINT | no tab op; token-checked PID kill | no-op + reason | `""` + reason |

The `IDENTITY_MISMATCH` reclaim row deserves emphasis: a pane can mismatch while the original PID still exists elsewhere. The inherited mixin then returns `OURS`, which is the conservative and correct reclaim answer—do not start concurrent work—while manager lifecycle methods still refuse to touch the mismatching Herdr object.

### `PANE_GONE` removal and double-kill analysis

`PANE_GONE` does not leak the process record. The outer server sees matching PID ownership, calls graceful shutdown, and, if that fails, calls force kill. Both paths revalidate the creation token immediately before signalling. If SIGINT already ended the agent, the later force stage sees a dead PID and sends nothing. If the numeric PID was recycled between stages, the new creation token differs and sends nothing. Thus the path neither strands a known-live agent merely because its pane moved nor double-kills a replacement process.

As in the existing documented contract, `kill_agent` success cannot prove an OS kill succeeded under permissions/kernel errors; that is not a new Herdr record-leak condition.

## Implementability audit

The design is implementable with the stated boundaries:

- pure selector/constructor;
- lazy endpoint resolution;
- shared `claude_teams.filelock.file_lock` at a stable per-Herdr-session path, with status re-check inside the lock;
- `_run_herdr` for bounded finite JSON commands;
- `_popen_herdr_server` for the retained long-running daemon child;
- injectable token/PID, clock/sleep, and lock boundaries;
- command-specific JSON validation and isolated live checks for Herdr lifecycle facts.

All five probe states now have defined projections for health, reclaim, graceful shutdown, force kill, send, and capture. No source-file or protocol-contract change beyond the listed process-manager/launcher documentation work is implied.

## New non-blocking findings

### MINOR 1 — One stale launcher-selection sentence still assigns a session name to `HERDR_ENV`

**Concern:** Plan lines 147-152 versus lines 177-190 and 207.

The launcher-selection section still says `HERDR_ENV=1` “only supplies the default session name once herdr has been explicitly selected.” The detailed routing design correctly says `HERDR_ENV` is a boolean only and is never used as a session name. These statements contradict each other.

**Suggested fix:** During implementation cleanup, replace the stale phrase with: “`HERDR_ENV=1` neither selects the launcher nor supplies a session name.” The detailed routing design is otherwise unambiguous.

### MINOR 2 — Rebinding should explicitly preserve the immutable selector, log transitions, and use one bounded proof attempt

**Concern:** Plan lines 202-205 and test 39.

The three-value proof is sufficient against accidental cross-session adoption only if the candidate endpoint was resolved from the same immutable configured session selector (`WIN_AGENT_TEAMS_HERDR_SESSION`, or the same unpinned default). The plan implies this but does not state it in the rebinding rule. An implementation must not scan every `session list` row and adopt whichever row reports matching-looking data.

Rebinding is an important lifecycle event and should be observable. It should be logged with old/new canonical paths and the successful pane/PID/token proof. The attempt should consist of bounded `_run_herdr` calls and may repeat on later handoffs, but must never loop indefinitely or update the stored endpoint before all proof fields validate. A failed/ambiguous proof remains non-owned.

**Suggested fix:** Add those constraints as implementation acceptance details and a unit test that a different session selector is never considered, even if its fake response claims the same tuple. No new architecture or authentication mechanism is needed.

### MINOR 3 — Risks still uses removed state names and understates PID fallback

**Concern:** Plan Risks 1, lines 401-406.

The text says endpoint change is reported as `INDETERMINATE/GONE`, but `GONE` no longer exists; the relevant states are `PANE_GONE` and `PID_GONE`. It also says kill is “scoped to `tab close`,” while the operative design intentionally uses token-checked OS signals for `PANE_GONE` and `INDETERMINATE`.

**Suggested fix:** Update the risk text to the current enum and say: “Herdr object destruction is scoped to proven-owned tabs; degraded/unmanaged agents use token-revalidated PID signalling.”

## Verdict

VERDICT: APPROVED — 0 BLOCKERS
