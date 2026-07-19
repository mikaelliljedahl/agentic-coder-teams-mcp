# Increment 2 plan review — CHANGES REQUIRED

## Summary

The diagnostic analysis is substantially correct: a clean reconstruction of
`HEAD` has exactly **12 Linux** diagnostics and **13 win32** diagnostics, and
the proposed type-level changes reduce Linux to zero.  The proposed Linux `qa`
job plus Windows test job also implements the requested platform policy.

This plan is not yet implementation-ready because it omits required Ruff work.
The literal constant `getattr` and `setattr` replacements trigger Ruff B009 and
B010, respectively; and the proposed `ruff format --check .` gate fails the
current repository on eight already-unformatted files.  Without explicitly
adding the suppressions and formatting changes, the new required gate will be
red immediately.

## Score

**78/100 — CHANGES REQUIRED.**

## Coverage

All twelve Linux type diagnostics are accounted for precisely:

| Group | Linux count | Verified result |
|---|---:|---|
| `_popen` overload/return plus `log_handle` calls | 6 | `**kwargs: Any` alone leaves 6 total diagnostics |
| `ctypes.WinDLL` / `get_last_error` | 3 | dynamic lookup removes all three |
| `_tracked_alive` concrete members | 3 | concrete `cast` removes all three |
| **Total** | **12** | **0 after the complete, lint-clean patch** |

The two `winerror` assignments are also accounted for. They are not Linux
diagnostics because the Increment-1 `ty: ignore` is used there, but they become
two `unused-ignore-comment` warnings under win32. `setattr` removes both when
paired with the needed Ruff B010 suppression.

The expected residual win32 set is exactly two `signal.SIGKILL` diagnostics at
`src/claude_teams/backends/process_manager.py:1605` and `:2082` in the current
worktree (the plan's pre-edit locations are `:1593` and `:2070`). No type
diagnostic was missed in the stated Linux zero target.

## Technical Verification

### Checker evidence

I used a detached temporary worktree at commit `3aeeb81` so the review did not
mistake the implementer's uncommitted edits for baseline behavior.

* Baseline `uv run ty check`: **Found 12 diagnostics** (Linux).
* Baseline `uv run ty check --python-platform win32`: **Found 13 diagnostics**.
* Changing only `_popen` from `**kwargs: object` to `**kwargs: Any`, and adding
  the `Any` import: **Found 6 diagnostics** (Linux) and **7 diagnostics**
  (win32). This reproduces the plan's 12-to-6 claim.
* Applying all type fixes, importing both `Any` and `cast`, and adding the
  required Ruff handling described below: Linux `ty` reported **All checks
  passed!**; win32 reported exactly the two documented `SIGKILL` errors.
* The lint-clean patched files passed `ruff check`; `pytest -q` reported
  **497 passed, 2 skipped**.

### `_popen`: verified, with a bounded `Any` trade-off

The source wrapper is at
`src/claude_teams/backends/process_manager.py:581-616` (the plan's baseline is
`:573-608`). In the clean baseline, the two `Popen` calls give both
`invalid-return-type` and `no-matching-overload`, while `log_handle.write` and
`.flush` give `call-non-callable` and `unresolved-attribute`.

`**kwargs: Any` causes the two `Popen` overloads to resolve as
`Popen[str]` and makes `log_handle` callable/flushable to `ty`; no cast is
needed for this change. It does, necessarily, stop checking key names and
values passed through this one dynamic subprocess wrapper, and makes the
fallback log handle `Any`. That is an acceptable local trade-off for this
thin forwarding boundary, but it should be stated as a loss of checking rather
than as type safety gained. `Any` is actually used, so it introduces no
unused-import issue.

### `ctypes`: type fix is correct, but the plan omits required Ruff B009

The three baseline failures are `ctypes.WinDLL` at
`src/claude_teams/backends/process_manager.py:96` and `:755`, and
`ctypes.get_last_error` at `:771` (current edited locations are `:96`, `:765`,
and `:783`). `getattr(ctypes, "WinDLL")` and
`getattr(ctypes, "get_last_error")` are clean under both `ty` platforms and
their result is `Any`, which is compatible with the existing dynamic ctypes
function/`argtypes` setup.

When the member exists, `getattr(module, "name")` without a default has the
same lookup result and the same `AttributeError` failure behavior as
`module.name`; it does not change the Windows runtime call. This is the most
direct platform-neutral idiom here. A `ty: ignore` on attribute access is worse
because it becomes unused on win32. A named non-literal attribute string could
avoid B009 but obscures the intent.

However, Ruff reports **B009** for each literal `getattr`; the plan's examples
at `plan.md:35` and `:48` will fail the stated `ruff check .` gate. Add a
localized `# noqa: B009` with a reason to each lookup (as already used by the
other Windows ctypes lookups at
`src/claude_teams/backends/process_manager.py:408`, `:435`, and `:446`). Keep
the suppression on the line containing the actual expression; putting it on a
wrapping `return (` line is ineffective and creates RUF100.

### `_tracked_alive`: casts and concrete types are correct

The base contract deliberately accepts `info: object` at
`src/claude_teams/backends/process_manager.py:236-243`; retaining that exact
parameter type avoids an override/LSP incompatibility. `typing.cast` is a
runtime no-op, so it does not change liveness behavior.

The required concrete mapping is proven by the per-subclass registries:

* `WindowsProcessManager`: `dict[str, ProcessInfo]` at
  `src/claude_teams/backends/process_manager.py:456`, thus cast to `ProcessInfo`
  at the override currently at `:642-644`.
* `TmuxProcessManager`: `dict[str, TmuxProcessInfo]` at `:1308`, thus cast to
  `TmuxProcessInfo` at `:1378-1381`.
* `LinuxTerminalProcessManager`: `dict[str, LinuxTerminalProcessInfo]` at
  `:1677`, thus cast to `LinuxTerminalProcessInfo` at `:1760-1770`.

The final controlled checker run had no override/LSP diagnostic. Removing the
three `# type: ignore[attr-defined]` comments is safe: those comments do not
suppress the current `ty` errors, and repository-wide search found no mypy
configuration or mypy invocation (only `.mypy_cache/` is ignored). The plan
must explicitly import `cast` in addition to `Any`; without it, Ruff F821 and
`ty` unresolved-reference errors result.

### `winerror`: behavior/type result correct, but the plan omits Ruff B010

The two baseline assignments are at
`tests/test_backends/test_process_manager_windows.py:124` and `:146` (currently
`:124` and `:146`). `setattr(err, "winerror", N)` has the same effect as
`err.winerror = N` for this `OSError` instance and is clean on both Linux and
win32 under `ty`; it also removes the two win32 unused-`ty: ignore` warnings.

Literal `setattr` triggers Ruff **B010**, though. Add `# noqa: B010` with a
brief Windows-only reason on each assignment (the current implementation does
this), or use another intentionally dynamic mechanism. The plan's bare form at
`plan.md:53-56` is not sufficient for its own `ruff check .` gate.

### `SIGKILL`: reasonable out-of-scope decision

There are three sites: the Windows manager's non-Windows fallback at
`src/claude_teams/backends/process_manager.py:747`, tmux at `:1605`, and Linux
terminal at `:2082`. Linux `ty` flags none. win32 flags only the tmux and Linux
terminal sites, exactly as the plan says. Leaving them unchanged is reasonable:
the latter two are intrinsically Linux-only implementations, a dynamic lookup
would degrade readability/type checking for no Linux-gate benefit, and this is
documented rather than suppressed. I agree with this scope boundary.

## CI Gate Review

The proposed split is correct for the requested policy:

* `qa` on `ubuntu-latest` runs formatting, lint, Linux `ty`, and the complete
  test suite, so it enforces Linux type cleanliness and all tests passing.
* `tests-windows` on `windows-latest` runs the full suite, preserving a
  cross-platform test requirement without treating win32 `ty` residuals as a
  gate failure.
* The triggers remain `pull_request`, pushes to `main`, and manual dispatch.
  Configuring `qa` as a required check remains a branch-protection setting, as
  correctly noted at `plan.md:67-70`.

`uv sync --group dev` is the right dependency selection: the `dev` group in
`pyproject.toml:28-42` contains pytest, pytest-asyncio, pytest-cov, Ruff, and
ty. `uv lock --check` succeeds. For stricter reproducibility, prefer
`uv sync --locked --group dev` in both jobs; the current form is still
enforcing, but can resolve a stale lock rather than fail fast.

“100% tests” is satisfied if it means a 100%-green test run, which is the
plan's explicit wording and what `uv run pytest` enforces. It does **not** mean
100% line coverage: pytest is invoked without `--cov`, and the repository's
configured coverage floor is 90% at `pyproject.toml:109-123`. If the request
instead means 100% coverage, the plan needs an explicit coverage command and a
policy change.

The gate has one material pre-existing-state problem: on the clean base,
`uv run ruff format --check . --diff` reports **8 files would be reformatted**:

* `src/claude_teams/backends/codex.py`
* `src/claude_teams/backends/contracts.py`
* `src/claude_teams/backends/process_manager.py`
* `src/claude_teams/backends/registry.py`
* `tests/test_agent_output.py`
* `tests/test_backends/test_base_runtime.py`
* `tests/test_backends/test_codex.py`
* `tests/test_cli_watch.py`

Thus the declared `qa` job will reliably fail on the current branch even after
the type work. This is enforcing rather than flaky, but it defeats the planned
green QA baseline. The plan must either include formatting all eight files in
this increment (the preferred solution) or deliberately narrow/defer the
format gate; the latter would not match the plan's stated full-repository gate.

## New Gaps/Risks

1. Missing `cast` import, B009 suppressions, and B010 suppressions would create
   new `ty`/Ruff failures. These are blockers, not cosmetic documentation gaps.
2. The affected-files list at `plan.md:72-76` omits all formatting-only files
   required for `ruff format --check .` to pass.
3. The live worktree already contains uncommitted implementation edits, so a
   direct live checker run showed 4 Linux diagnostics rather than the plan's
   baseline 12. The count claims above were therefore verified against a clean
   detached reconstruction, not inferred from the dirty state.
4. `Any` and ctypes `getattr` intentionally reduce checking at dynamic FFI and
   subprocess boundaries. They do not cause a current downstream diagnostic,
   but future code using these values will not gain static guarantees.

## Improvement Suggestions

1. Amend the exact edit list to import `Any, cast`, use the three casts, and
   include `# noqa: B009` on every literal ctypes `getattr` plus
   `# noqa: B010` on both `setattr` calls. State why each suppression is
   intentional.
2. Add the eight Ruff-format edits to “Files affected”, run
   `uv run ruff format` before adding the format gate, then validate with the
   exact CI commands from a clean checkout.
3. Use `uv sync --locked --group dev` unless intentional lock regeneration in
   CI is desired.
4. State whether “100% tests” means all tests pass (current plan) or 100%
   coverage; add a coverage invocation only for the latter.

## Ready for Implementation?

**No.** Amend the plan for B009/B010, the `cast` import, and the eight-file
format baseline before implementation. Once those changes are included, the
type strategy and Linux/Windows CI design are technically sound.
