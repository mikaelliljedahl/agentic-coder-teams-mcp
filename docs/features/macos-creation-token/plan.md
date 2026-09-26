# macOS process creation tokens

## History

Revision 3 dispositions every finding in `plan-review.md` (r2 review); see
"Review dispositions" at the end.


Revision 1 (`plan-r1.md`, reviewed in `plan-review-r1.md`) bundled two changes:
a macOS creation-token reader and a redesign of `kill_agent` that verifies the
kill landed. The review returned CHANGES REQUIRED with two blockers, both in the
kill-verification half (signalling an unproven sidecar PID; treating an
unreadable process as dead). This revision narrows the feature to the
creation-token reader only. The kill-verification work is deferred to its own
feature and is not part of this PR.

The r1 implementation existed before any plan (local macOS adaptations). This
revision is implemented fresh from `origin/main` with red-green TDD; the r1
code is reference material only.

## Scope

`process_manager.creation_token(handle)` returns an immutable per-process token
on Windows (`GetProcessTimes` creation FILETIME,
`src/claude_teams/backends/process_manager.py` `_read_windows_creation_token`)
and on Linux (`/proc/<pid>/stat` field 22, `_read_linux_creation_token`). On
macOS, which has no `/proc`, it falls through to the Linux reader and returns
`None` for every PID.

In scope: a macOS reader so `creation_token` returns a real token on Darwin.

Out of scope:
- `kill_agent` kill verification, `pid_alive`/`wait_pid_exit`, zombie handling,
  sidecar/grandchild termination (r1 findings 1, 2, 3, 5, 6, 7) — separate
  feature.
- `procinfo.resolve_nearest_host()` returning an empty chain on macOS
  (`tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain`) —
  separate fix. It stays red on macOS after this change and is reported as such.

## Current behavior on macOS (origin/main e5c8e98)

- Every `creation_token()` call returns `None`, so every agent record, lease
  holder and delivery claim stores a `None` token.
- `ownership_probe` (`process_manager.py` `_PidOwnershipMixin.ownership_probe`)
  returns `NOT_OURS` for a tokenless expectation and `INDETERMINATE` for a live
  PID with an unreadable token. So after any MCP-server restart, `kill_agent`
  never signals a live agent (orphan), and lease/claim reclaim treats live
  holders as indeterminate or not ours.
- Full suite on macOS arm64: 13 failed, 2 errors. The creation-token-dependent
  ones are in `tests/test_pid_reuse.py`, `tests/test_delivery_integrity.py`,
  `tests/test_follow_up_delivery.py`.

## Design

In `src/claude_teams/backends/process_manager.py`:

1. `_parse_darwin_kinfo_start_time(raw: bytes) -> tuple[int, int] | None` —
   pure function, no platform dependency, unit-testable on Linux CI.
   - `struct kinfo_proc` (LP64, little-endian: arm64 and x86_64) begins with
     `kp_proc.p_un.__p_starttime`, a `struct user64_timeval`: 64-bit
     `tv_sec` then 32-bit `tv_usec` then 4 bytes padding. Decode with `<qi` at
     offset 0 (r1 used `<qq`, which folds padding into microseconds).
   - Return `None` if `len(raw)` is shorter than the 12-byte prefix, if
     `tv_sec <= 0`, or if `tv_usec` is outside `[0, 1_000_000)`.
2. `_read_darwin_creation_token(pid: int) -> str | None` —
   `sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})` via `ctypes`, with
   `argtypes`/`restype` declared. Two-call size-then-data pattern. A missing
   PID answers with a zero-length reply. Any load error, sysctl error, or
   short/invalid reply returns `None`. Token format `"<sec>.<usec:06d>"`.
   `sysctl` rather than libproc `proc_pidinfo`, which returns nothing for
   PID 1 and for zombies.
3. `creation_token()` dispatches to it when `sys.platform == "darwin"`, before
   the Linux fallback.

4. **Prerequisite (plan-review.md finding 1):** fix
   `_PidOwnershipMixin._pid_health_with_token`. Today, with an expected token
   set, an *unreadable* current token is reported dead
   (`"process not found or token unreadable"`). Readable Darwin tokens make
   stored tokens common on macOS, so a transient read failure after an MCP
   restart would report a live agent dead, skip `follow_up_agent`'s live/busy
   guard, and allow a concurrent resume of the same conversation. The same
   hazard already exists on Linux and Windows. New rule: token unreadable and
   `self._pid_alive(handle)` true → `(True, "process alive; token unreadable,
   identity unverified")`; token unreadable and PID not alive →
   `(False, "process not found")`. Mismatch and match are unchanged. This only
   moves an unknown case from "dead" to "alive", i.e. it can block a resume or
   keep prompt files longer, never authorize a signal: destructive paths still
   gate on `owns_process`.

### Why `None` is safe here

`None` already means "token unreadable" on every platform, and the existing
consumers are fail-closed about it: `ownership_probe` classifies a live PID
with an unreadable token as `INDETERMINATE` using the manager's own
`_pid_alive`, never as dead or ours. This change does not alter that policy or
any liveness check; it only makes the token readable where it previously never
was. With item 4 in place, a transient re-read failure of a stored token never
reads as death. This is a caller-specific argument, not a universal one — see
"Inherited limitations".

### Inherited limitations (pre-existing, out of scope, disposition: follow-up)

These exist on origin/main on every platform and are neither introduced nor
fixed here:

- **Failed initial capture.** If `creation_token()` fails at spawn,
  reservation or claim time, the record stores `None`. `ownership_probe`
  classifies a tokenless expectation as `NOT_OURS` before checking liveness, so
  such a live lease/claim holder is reclaimable
  (`leases.py` reclaim path, `server_simple.py` claim reclaim). Preferred
  future fix: refuse lease/claim acquisition when identity capture fails.
- **Legacy tokenless records** created before this change stay tokenless; no
  backfill from a possibly reused PID.
- **POSIX `_pid_alive` treats every `OSError` as absent**, including `EPERM`.
  Agents run as the same user, so this is not hit in practice.
- Numeric-PID check-then-signal races remain; token matching is not an atomic
  identity guarantee.

## Files affected

- `src/claude_teams/backends/process_manager.py`
- `tests/test_darwin_creation_token.py` (new)
- `tests/test_pid_health_with_token.py` (new) — or the existing test module
  that covers `_pid_health_with_token`, if one exists

## Risks

1. Hard-coded ABI offset. Mitigated by decoding only the documented
   `user64_timeval` prefix, validating ranges, portable parser tests with
   nonzero padding, and a native macOS check against an independent start time.
2. Only 64-bit little-endian Darwin is supported; 32-bit Darwin is not a
   supported Python target.
3. Behavior change on all platforms from item 4: an agent whose token cannot
   be re-read but whose PID is alive now reports alive instead of dead. It is
   the conservative direction. Known limitation (implementation-review.md
   SHOULD 1): if a dead agent's PID is reused by a process whose token *and*
   liveness both fail with access denied (Windows protected/elevated process,
   Linux `hidepid`), the agent reads as "alive, identity unverified"
   indefinitely — follow-up queues, cleanup waits. Accepted: the alternative is
   the concurrent-resume hazard this item removes.
4. CI has a Linux job (ruff format, ruff check, ty, pytest) and a Windows pytest
   job (`.github/workflows/ci.yml`); neither runs Darwin. Native macOS results
   are recorded by hand in `implementation.md` with OS version and architecture.

## Test cases

### `tests/test_darwin_creation_token.py`

Portable (run on Linux and Windows CI; only the native class is skipped off
Darwin, never the whole module):
- parser decodes `tv_sec`/`tv_usec` with nonzero padding bytes present;
- parser rejects empty, short (< 12 bytes), `tv_sec <= 0`, `tv_usec` negative
  and `tv_usec >= 1_000_000`;
- the real ctypes reader against a fake CDLL/sysctl: load failure, first-call
  failure, zero size, second-call failure (including ENOMEM), successful second
  call with zero or short output, returned length greater than capacity,
  invalid prefix, success, token formatting with leading-zero microseconds.
  Assert the MIB `{CTL_KERN=1, KERN_PROC=14, KERN_PROC_PID=1, pid}`, null
  new-value arguments, and the declared signature
  (`argtypes = [POINTER(c_int), c_uint, c_void_p, POINTER(c_size_t),
  c_void_p, c_size_t]`, `restype = c_int`). Parse only the returned bytes;
- dispatch: Windows precedence, Darwin branch, Linux fallback, asserted with
  reader sentinels. Platform predicates are pinned through module-local
  facades (e.g. `_IS_WINDOWS` / `_IS_DARWIN` helpers) or tightly scoped
  patches — never by mutating global `os.name` across pytest/pathlib.

### `_pid_health_with_token` (finding 1 regression)
- empty manager registry, stored valid token, live PID, injected read failure
  → alive, reason names unverified identity;
- same with dead PID → dead;
- token mismatch → dead (reuse); token match → alive; no stored token →
  bare PID liveness (unchanged);
- `follow_up_agent`/resume path: with a bound resumable session, no sidecar,
  live PID and injected token-read failure, no resume is started and no
  live-child artifacts are deleted.

macOS-only (`skipif(sys.platform != "darwin")`):
- a live child process has a token; it is stable across calls;
- two concurrent children have different tokens;
- the token's seconds match the child's start time measured independently
  (`ps -o lstart=` or equivalent) within 1 s;
- a PID that does not exist returns `None`;
- PID 1 returns a token.

Existing tests that must turn green on macOS: the creation-token cases in
`tests/test_pid_reuse.py`, `tests/test_delivery_integrity.py`,
`tests/test_follow_up_delivery.py`.

## Validation

All four gates across the whole repo, on macOS locally:
`uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
`uv run pytest`. Linux and Windows via PR CI. Any remaining red (expected:
`test_procinfo` on macOS) is named in `implementation.md` and the PR.

## Review dispositions (plan-review.md, r2)

| # | Severity | Disposition |
|---|---|---|
| 1 | BLOCKER | Accepted. Design item 4 fixes `_pid_health_with_token`; regression tests added to the plan. Lands in this PR because enabling Darwin tokens activates it. |
| 2 | SHOULD | Accepted as documentation: "Why None is safe" is now caller-specific and the inherited limitations are listed with a follow-up disposition. Refusing acquisition on failed capture is deferred to a separate feature to keep this PR small. |
| 3 | SHOULD | Accepted in full: real ctypes reader tests with a fake CDLL, signature and MIB assertions, returned-length handling, portable dispatch tests via module-local platform facades. |
| r1 #4 | — | Resolved (`<qi`, range validation, native start-time check; arm64 and x86_64 reported separately, x86_64 marked unverified if not run). |
| r1 #8, #9 | — | Resolved / excluded by scope. |
