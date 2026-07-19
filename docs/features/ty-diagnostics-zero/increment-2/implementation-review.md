# Increment 2 implementation review — APPROVED

## Summary

The implementation resolves every blocker identified in the plan review. The
Linux QA command set is clean, the type checker reaches zero on Linux, win32
has exactly the two documented `SIGKILL` residuals, and the full test suite is
green. The CI workflow is correctly split into an enforcing Linux `qa` job and
a Windows test job, with locked dependency synchronization.

## Score

**99/100 — APPROVED.**

The one point retained is a documentation-only nit: the earlier CI prose at
`plan.md:62-65` still says `uv sync --group dev`, while the amended Ruff section
and actual workflow correctly say `uv sync --locked --group dev`. The executable
configuration is correct.

## Prior Blockers Resolution

### 1. B009 `getattr`: resolved

Every literal ctypes lookup is now on an expression line with `# noqa: B009`
and an immediately preceding Windows-only/type-checker rationale. The three
original diagnostic sites are covered at
`src/claude_teams/backends/process_manager.py:98`, `:765`, and `:783`.
The implementation also consistently updates the three Windows job-object
lookups at `:410`, `:437`, and `:446`, for six B009-suppressed literal ctypes
lookups total. This wider consistency is correct.

`uv run ruff check .` output: **All checks passed!**

### 2. B010 `setattr`: resolved

Both test assignments use `setattr(err, "winerror", N)  # noqa: B010` at
`tests/test_backends/test_process_manager_windows.py:124` and `:146`. The
first has an explicit adjacent explanation of why the Windows-only shape is
being simulated. No `ty: ignore` remains in this test.

### 3. `cast` import: resolved

`src/claude_teams/backends/process_manager.py:15` imports both `Any` and
`cast` alongside the existing typing imports. The clean Ruff and `ty` runs
confirm neither is unresolved or unused.

### 4. Whole-tree formatting: resolved

All eight previously dirty formatter targets are now included in the diff:
`backends/codex.py`, `backends/contracts.py`, `backends/process_manager.py`,
`backends/registry.py`, `tests/test_agent_output.py`,
`tests/test_backends/test_base_runtime.py`, `tests/test_backends/test_codex.py`,
and `tests/test_cli_watch.py`.

`uv run ruff format --check .` output: **45 files already formatted**.
`git diff --check` also produced no whitespace errors.

## Substance Verification

### 5. Linux and win32 type results: passed

* `uv run ty check`: **All checks passed!** (zero Linux diagnostics).
* `uv run ty check --python-platform win32`: **Found 2 diagnostics**, both
  `signal.SIGKILL` unresolved attributes at
  `src/claude_teams/backends/process_manager.py:1607` and `:2084`.

There are no win32 unused-ignore warnings or other diagnostics. These two
sites are the intentionally out-of-scope tmux/Linux-terminal implementations;
the Windows manager's conditional fallback at `:747` remains unflagged.

### 6. `_popen(**kwargs: Any)`: correct

`WindowsProcessManager._popen` is now declared at
`src/claude_teams/backends/process_manager.py:581-616` with `**kwargs: Any`.
It forwards the same kwargs to both unchanged `subprocess.Popen` calls and
uses the same fallback logging branch. No cast is introduced or needed. This
is behavior-preserving and resolves the overload/return plus log-handle
diagnostics, as demonstrated by the zero-Linux `ty` run.

### 7. `_tracked_alive` casts: correct

The base contract remains `info: object` at
`src/claude_teams/backends/process_manager.py:236-243`, preserving the override
signature. The overrides correctly cast only at member access:

* `cast(ProcessInfo, info)` at `:642-644`, matching
  `WindowsProcessManager._processes: dict[str, ProcessInfo]` at `:456`.
* `cast(TmuxProcessInfo, info)` at `:1380-1383`, matching the tmux registry at
  `:1310`.
* `cast(LinuxTerminalProcessInfo, info)` at `:1762-1772`, matching the Linux
  terminal registry at `:1679`.

All three obsolete `# type: ignore[attr-defined]` comments are removed. The
zero-diagnostic `ty` result confirms no override/LSP issue; `cast` is a runtime
no-op, so liveness behavior is unchanged.

### 8. ctypes runtime semantics and win32 result: correct

`getattr(ctypes, "WinDLL")` and `getattr(ctypes, "get_last_error")` without a
default have the same successful lookup and the same `AttributeError` behavior
as direct attribute access when absent. The Windows creation-token path is
called only from the `os.name == "nt"` branch at
`src/claude_teams/backends/process_manager.py:172-186`; Windows job-object and
PID logic also guard their calls. The changes therefore preserve Windows
runtime behavior while making the static lookup platform-neutral. The win32
type run has no ctypes diagnostic.

### 9. `winerror` runtime semantics and tests: correct

`setattr(err, "winerror", N)` has the same effect as the prior assignment for
these locally constructed `OSError` objects. The fallback tests exercise both
the access-denied retry and non-breakaway propagation paths. Targeted command:
`uv run pytest -q tests/test_backends/test_process_manager_windows.py` →
**16 passed**.

### 10. Full suite: passed

`uv run pytest -q` → **497 passed, 2 skipped in 23.44s**. No regression was
observed from the type edits or formatting changes.

### 11. CI workflow and lock: correct

`.github/workflows/ci.yml:13-34` defines `qa` on `ubuntu-latest` with, in
order, `uv sync --locked --group dev`, `ruff format --check .`, `ruff check .`,
Linux `ty check`, and pytest. `.github/workflows/ci.yml:38-50` defines
`tests-windows` on `windows-latest` with the same locked dev sync and pytest.
The workflow remains triggered for PRs, pushes to `main`, and manual dispatch
at `:3-7`. The YAML structure/indentation is valid for two independent jobs;
there is no accidental dependency that would prevent either from enforcing its
own steps.

`uv lock --check` succeeds (`Resolved 95 packages`), and a local
`uv sync --locked --group dev` succeeds (`Checked 92 packages`). The dev group
indeed contains pytest, Ruff, and ty in `pyproject.toml:28-42`. This correctly
creates the requested Linux QA gate while retaining cross-platform test
coverage. As correctly documented, branch protection must still make `qa` a
required GitHub status check.

### 12. Formatting churn scope: verified

Spot-checks confirm layout-only changes, with no logic or literal-value edits:

* `src/claude_teams/backends/codex.py:430-439` only expands an existing
  compound conditional across lines.
* `src/claude_teams/backends/registry.py:17-20` adds only the formatter's blank
  line before a class.
* `tests/test_backends/test_base_runtime.py` and
  `tests/test_backends/test_codex.py` only collapse/expand calls and function
  declarations; assertions, arguments, and string values are unchanged.
* `tests/test_cli_watch.py` only reformats two test signatures.

The full diff likewise shows format-only parenthesis, line-wrap, and blank-line
changes in the remaining formatter files. The process-manager diff separates
the intentional type/ctypes changes from normal Ruff formatting.

## New Regressions

None found. The only nonzero type result is the known, documented win32
`SIGKILL` pair; it is outside the Linux gate by design. All formatter, linter,
lock, type, targeted-test, and full-suite checks pass as expected.

## Final Verdict

**APPROVED — 99/100.** No implementation blockers remain.
