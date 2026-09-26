# macOS process creation tokens: implementation review

Reviewer: Claude Opus (opposite family to the Codex implementer).
Scope: uncommitted diff against `origin/main` e5c8e98 in
`/Users/henrikekblad/code/wt-macos-liveness` (`process_manager.py`,
`tests/test_backends/test_base_runtime.py`, `tests/test_follow_up_delivery.py`,
new `tests/test_darwin_creation_token.py`), checked against `plan.md` rev 3 and
its review dispositions.

**Verdict: APPROVED WITH CHANGES** (0 BLOCKER, 2 SHOULD, 4 NIT)

## Plan conformance

- Design 1 (`_parse_darwin_kinfo_start_time`): `<qi` at offset 0, 12-byte
  minimum, `sec <= 0` and `usec` outside `[0, 1_000_000)` rejected. Matches.
- Design 2 (`_read_darwin_creation_token`): declared `argtypes`/`restype`,
  two-call size-then-data, rejects zero size, rejects a returned length greater
  than capacity, parses only `buffer.raw[: size.value]`, and formats
  `"<sec>.<usec:06d>"`. Load errors (`OSError`, `AttributeError`) return `None`.
  Matches.
- Design 3: dispatch runs Windows first, then Darwin, then Linux, through the
  module-local predicates `_creation_token_is_windows` and
  `_creation_token_is_darwin`. Nothing mutates `os.name` globally. Matches
  review finding 3.
- Design 4 (`_pid_health_with_token`): an unreadable token on a live PID now
  returns `(True, "…identity unverified")`, and an unreadable token on a dead
  PID returns `(False, "process not found")`. Match, mismatch and tokenless
  behaviour are unchanged. Matches review finding 1.
- Test cases promised by the plan:
  - The parser tests cover padding and every rejection case.
  - The fake-CDLL tests cover MIB `[1,14,1,pid]`, count 4, null `newp` and
    `newlen=0` on both calls, and the exact signature. They also cover load
    failure, first-call failure, zero size, second-call failure (including
    `ENOMEM`), zero or short output, returned length > capacity, an invalid
    prefix, and a leading-zero `usec` (`123.000042`).
  - The three dispatch tests use reader sentinels.
  - The `_pid_health_with_token` cases are all there: unverified-alive, dead,
    mismatch, match, and tokenless.
  - The follow-up regression is there: empty registry, live PID, token read
    fails, asserts `queued`, no `resume_calls`, prompt artifact intact.
  - The native macOS tests are all there.
  - All promised tests are present.
  - Placement deviation: the plan named `tests/test_pid_health_with_token.py`
    "or the existing module". The tests went into `test_base_runtime.py`, which
    the plan allowed.

## Safety trace of the `_pid_health_with_token` change

`_pid_health_with_token` is reached only through `health_check` (at
`process_manager.py:857`, `:1599`, `:2010`, `:3149`), `_tab_health` (`:1425`)
and the WT tab settle poll (`:1319`, `:1323`). `health_check` is consumed
through `server_simple._agent_alive` (`:1724`) and the resume confirmation
child probe (`:4575`). `_agent_alive` consumers:

- delivery settle (`:2505`): a live agent keeps the record `unconfirmed` instead
  of failing it.
- `check_agent` (`:3934`), `list_agents` (`:6108`) and `agent_status`
  (`:6140`): display only.
- follow-up prep (`:4287`): live means queue and do not resume. This is the
  intended fix.
- artifact cleanup (`:5925`): `child_exited` is now False, so artifacts are
  kept longer.
- session-active check (`:1322`) and the `lead_wake` live-children count
  (`lead_wake.py:220`).

None of these paths sends a signal. `kill_agent` and the other destructive
paths gate on `owns_process`/`ownership_probe`, which this diff does not touch,
and `ownership_probe` already classifies live+unreadable as `INDETERMINATE`.
The change can therefore only turn "dead" into "alive". It cannot authorize a
signal. The WT tab settle poll now reports a live-but-unreadable wrapper as
settled, so it does not retry in a new console. That is also the conservative
direction: fewer duplicate spawns.

`creation_token()` callers (`server_simple.py:1533, 3353, 4114, 4434, 4897,
6381, 6731`, `lead_wake.py:98`, and the internal uses) are only affected in that
Darwin now returns a string instead of `None`. That is the intended outcome, and
it is what turns the previously red pid-reuse, delivery-integrity and follow-up
tests green on macOS.

## Findings

1. **SHOULD: the "unknown means alive" rule can become permanent on Windows.
   The plan's risk section does not mention this.**
   `process_manager.py:497-501`. On Windows, `_read_windows_creation_token`
   returns `None` on `ACCESS_DENIED` (`process_manager.py:199-230`), and
   `_windows_pid_alive` returns True on `ACCESS_DENIED`
   (`process_manager.py:984`). Suppose a dead agent's PID is reused by a
   protected or elevated process. The record then reads "alive, identity
   unverified" indefinitely, where before it read dead. Consequences:
   - `follow_up_agent` queues forever;
   - `lead_wake._live_children` counts the agent as a live child, which works
     against the "no unbounded wake loop" intent in the comment at
     `lead_wake.py:224-226`;
   - artifact cleanup never runs.

   Linux with `hidepid` has the same issue. macOS is essentially unaffected,
   because `kern.proc.pid` is readable for any user's process. The
   plan-review blocker required this direction, so this is not a blocker.

   Fix: add this persistent-unreadable case to plan Risk 3 and to the
   `implementation.md` deviations/limitations. In the PR text, name it as an
   inherited follow-up alongside the other items. Optionally make
   `lead_wake` treat an "identity unverified" liveness as not live, since that
   path is advisory.
2. **SHOULD: name the raw sysctl MIB constants.**
   `process_manager.py:293`: `(ctypes.c_int * 4)(1, 14, 1, pid)`, and the
   literal `4` at `:295` and `:299`. The file already names its Windows
   constants (`_PROCESS_QUERY_LIMITED_INFORMATION`, `_STILL_ACTIVE`,
   `_ERROR_ACCESS_DENIED`) and the new `_DARWIN_START_TIME_PREFIX_BYTES`. Fix:
   add `_CTL_KERN = 1`, `_KERN_PROC = 14` and `_KERN_PROC_PID = 1` next to the
   other constants, build `mib` from them, and pass `len(mib)` instead of `4`.
   This is behaviour-preserving, and the existing MIB assertion in the test
   keeps it covered.
3. **NIT: `ctypes.CDLL(None)` and the `argtypes` assignment run on every call.**
   `process_manager.py:281-292`. This is a fresh `dlopen(NULL)` and a fresh
   function object per call. It is not a shared-state race, because each
   `CDLL` instance is new. It mirrors the per-call style of
   `_read_windows_creation_token`. It is cheap but wasteful, since
   `creation_token` runs inside poll loops. Optional fix: build a configured
   `sysctl` once with `functools.cache` (tests would then need
   `cache_clear()`). Acceptable as is for consistency with the Windows reader.
4. **NIT: the reader docstring is thin compared with its Windows and Linux
   siblings.** `process_manager.py:278-279`. The neighbouring readers explain
   why their field is immutable and what `None` means ("caller must fail
   closed"). Fix: add two or three lines covering these points:
   - `kp_proc.p_starttime` is immutable;
   - why `sysctl` and not `proc_pidinfo` (fails for PID 1 and zombies);
   - a missing PID returns a zero-length reply, giving `None`;
   - LP64 little-endian only.
5. **NIT: the native test's `ps lstart` parse depends on locale.**
   `tests/test_darwin_creation_token.py:141-148`. `strptime("%a %b …")` assumes
   English day and month names. On a host whose `LANG`/`LC_TIME` produces
   localized `ps` output, the test fails for a reason unrelated to the code. Fix:
   pass `env={**os.environ, "LC_ALL": "C"}` to `check_output`.
6. **NIT: the fake-CDLL tests patch the global `ctypes.CDLL`.**
   `tests/test_darwin_creation_token.py:90,111` patch `pm.ctypes.CDLL`, which is
   the process-wide `ctypes` module. `monkeypatch` scopes and restores it
   correctly, and the tests pass under the sequential runner. If this becomes
   a problem, use a module-local `_load_libc()` seam, which would pair well with
   the cache in NIT 3. No action needed now.

## Portability (Linux / Windows CI)

The portable tests use a pure-Python fake (`ctypes.cast`, `memmove`,
`set_errno` and `byref` all exist on every platform). The platform facades are
patched, and `os.name` and `sys.platform` are never mutated. Only the two native
tests carry `skipif(sys.platform != "darwin")`. Nothing Darwin-specific runs at
import: `ctypes.CDLL(None)` is only evaluated inside the reader, which dispatch
never reaches off Darwin. On Windows, `CDLL(None)` would raise `TypeError`,
which is not in the caught tuple, but that path is unreachable by dispatch and
tests patch `CDLL`. The follow-up regression test patches `_pid_alive` and
`creation_token` on whatever manager the host selects, and it passes an empty
registry, so it is manager-agnostic. I expect it to pass on both CI jobs.

## Gate results (macOS 26.5 arm64, run by reviewer)

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | PASS: 87 files already formatted |
| `uv run ruff check .` | PASS: all checks passed |
| `uv run ty check` | PASS: all checks passed |
| `uv run pytest` | RED: 1 failed, 1726 passed, 4 skipped |
| `uv run pytest tests/test_darwin_creation_token.py -v` | 22 passed. Both native tests ran (not skipped). 0 skipped. |

The only failure is `tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain`
(`resolve_nearest_host(...).chain == ()`). It fails the same way on `origin/main`
and is out of scope per the plan. The full suite is therefore not green on
macOS, and the PR must say so. Linux and Windows results still need PR CI.
x86_64 Darwin was not tested.

## Summary

The implementation follows the approved plan and all accepted dispositions.
The ABI decode is correct and defensive. The health change is strictly
conservative for signalling. Every promised test exists and asserts what the
plan specifies. Before the upstream PR, name the MIB constants and record the
"permanently unverified-alive" consequence (finding 1) as a known limitation.
The NITs are optional.
