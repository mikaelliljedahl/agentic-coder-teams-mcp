# macOS process creation tokens: implementation

## Red and green evidence

After `uv sync`, the focused red run was:

```text
uv run pytest -q tests/test_darwin_creation_token.py tests/test_backends/test_base_runtime.py -k 'darwin or token_health' tests/test_follow_up_delivery.py -k 'unreadable_token_on_live_bound_child or token_health or parser or ctypes_reader or dispatch_precedence'
21 failed, 5 passed, 134 deselected
```

Failures were `test_parser_ignores_nonzero_padding`, all six `test_parser_rejects_invalid_prefix` cases, all eight `test_ctypes_reader_contract` cases, `test_ctypes_reader_load_failure`, all three `test_dispatch_precedence` cases, `test_token_health_without_registry[None-True-True-identity unverified]`, and `test_unreadable_token_on_live_bound_child_does_not_resume_or_clean_artifacts`. The Darwin functions/platform predicates did not exist; unreadable-token health reported a live PID dead; the follow-up regression returned `failed` instead of `queued`.

After the source change, the focused run (`uv run pytest -q tests/test_darwin_creation_token.py tests/test_backends/test_base_runtime.py tests/test_follow_up_delivery.py -k 'unreadable_token_on_live_bound_child or token_health or parser or ctypes_reader or dispatch_precedence or native_'`) gave `28 passed, 132 deselected`. A subsequent small test-only addition covered `ENOMEM` on the second sysctl call; the final focused run gave `29 passed, 132 deselected`, and the final whole-repo run includes it.

## Final design

- Parse the 12-byte start-time prefix of Darwin LP64 `kinfo_proc` with `<qi`, rejecting short or invalid seconds/microseconds; trailing padding is ignored.
- Read `KERN_PROC_PID` with a declared `ctypes` `sysctl` signature and two calls. Errors, zero/short replies, oversized returned lengths, and invalid prefixes yield `None`; valid times yield `sec.usec` with six microsecond digits.
- Dispatch Windows first, Darwin second, Linux otherwise using local platform predicates that portable tests can patch without changing global `os.name`.
- When a stored token cannot be re-read, `_pid_health_with_token` checks PID liveness. A live PID remains alive with identity unverified; a dead PID reports not found. Token match/mismatch and tokenless behavior remain as before. This does not alter `ownership_probe` or destructive signal authorization.

The plan's regression tests live in `tests/test_backends/test_base_runtime.py` and `tests/test_follow_up_delivery.py`, alongside the existing runtime and bound-session patterns. There are no behavioral deviations from design items 1–4. The refactor pass only formatted the tests and named the parser's size/range constants.

## Native evidence and validation

`sw_vers`: macOS 26.5.2, build 25F84. `uname -m`: arm64. Native tests passed for live children, stable and distinct tokens, start seconds compared to `/bin/ps -o lstart=` within one second, missing PID, and PID 1. x86_64 execution is **unverified**.

Final whole-repository gates:

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | PASS — 87 files already formatted |
| `uv run ruff check .` | PASS — all checks passed |
| `uv run ty check` | PASS — all checks passed |
| `uv run pytest` | RED — 1 failed, 1726 passed, 4 skipped in 42.44 s |

The only remaining failure is `tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain`: `procinfo.resolve_nearest_host(os.getpid()).chain` is empty on this macOS host. It predates this feature: the same single test failed with the same assertion in a detached `/tmp` worktree at `origin/main` (`e5c8e98`), run as `PYTHONPATH=/tmp/wat-macos-origin-main/src /Users/henrikekblad/code/wt-macos-liveness/.venv/bin/pytest -q tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain` (`1 failed`). The temporary worktree was removed. Per the plan, macOS host-chain resolution is a separate follow-up; this gate remains red and is not counted as passing.

Inherited tokenless-capture, legacy records, POSIX permission-error, and PID check/signal race limitations remain as documented in `plan.md`.

## Post-review changes (implementation-review.md)

| # | Severity | Disposition |
|---|---|---|
| 1 | SHOULD | Documented as a known limitation in plan.md Risk 3 and in the PR description: an unreadable-token PID reused by an access-denied process reads as alive indefinitely. Accepted trade-off against concurrent resume. |
| 2 | SHOULD | Fixed: sysctl MIB values named `_CTL_KERN`, `_KERN_PROC`, `_KERN_PROC_PID`; `len(mib)` instead of a literal `4`. |
| 3 | NIT | Not changed: per-call `CDLL(None)` mirrors the Windows reader's per-call style; cost is negligible. |
| 4 | NIT | Fixed: reader docstring states immutability, why sysctl over libproc, and that `None` means fail closed. |
| 5 | NIT | Fixed: native test runs `ps` with `LC_ALL=C`. |
| 6 | NIT | Not changed, as the review recommends. |

Gates after these changes (macOS 26.5.2 arm64): `ruff format --check` pass,
`ruff check` pass, `ty check` pass, `pytest` 1 failed / 1726 passed / 4 skipped —
the only failure is the pre-existing
`tests/test_procinfo.py::test_real_os_walk_returns_a_plausible_chain`
(also red on origin/main e5c8e98; out of scope).
