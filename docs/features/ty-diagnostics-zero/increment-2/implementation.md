# Implementation: ty diagnostics → 0 (Increment 2) + Linux QA gate

## Result

`uv run ty check` on Linux: **12 → 0** (`All checks passed!`). Combined with
Increment 1, the full type-debt cleanup is **44 → 0**.

Under `--python-platform win32`, exactly **2** diagnostics remain — both
`signal.SIGKILL` in the tmux / Linux-terminal managers (Linux-only code paths),
intentionally out of scope for the Linux gate and documented, not hidden.

| Gate | Result |
|---|---|
| `uv run ty check` (Linux) | **0** |
| `uv run ruff format --check .` | 45 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run pytest -q` | 497 passed, 2 skipped |

## Changes

### `backends/process_manager.py`
- Import `Any, cast` alongside the existing typing imports.
- `_popen(**kwargs: object)` → `**kwargs: Any` — resolves both `Popen` overloads
  and the `log_handle` `write`/`flush` calls with no cast (6 diagnostics). Thin
  dynamic subprocess wrapper; behavior unchanged.
- `ctypes.WinDLL` (×2) and `ctypes.get_last_error` → `getattr(ctypes, "…")`
  with `# noqa: B009` and a Windows-only rationale (3 diagnostics). Platform-
  agnostic: clean on both Linux and Windows; identical runtime lookup.
- Three `_tracked_alive(self, info: object)` overrides → `cast(ProcessInfo /
  TmuxProcessInfo / LinuxTerminalProcessInfo, info).…` (3 diagnostics). Base
  `info: object` param unchanged (no LSP/override risk); stale
  `# type: ignore[attr-defined]` removed.

### `tests/test_backends/test_process_manager_windows.py`
- **Corrects Increment 1**: `err.winerror = N  # ty: ignore[unresolved-attribute]`
  → `setattr(err, "winerror", N)  # noqa: B010`. The Inc-1 ignore was valid only
  on Linux and would be an `unused-ignore` warning on Windows; `setattr` is clean
  on both and behavior-identical.

### `.github/workflows/ci.yml` — the QA gate
- `qa` (ubuntu-latest): `uv sync --locked --group dev`, `ruff format --check .`,
  `ruff check .`, `ty check`, `pytest`. The required Linux gate.
- `tests-windows` (windows-latest): `uv sync --locked --group dev`, `pytest`.
  Cross-platform test coverage (no ty/lint — those run once, on Linux).
- Triggers unchanged: `pull_request`, push to `main`, `workflow_dispatch`.

### `style: ruff format` (separate commit)
8 pre-existing unformatted files reformatted so the new `ruff format --check`
gate starts green (behavior-preserving, layout-only; verified by review):
`backends/{codex,contracts,process_manager,registry}.py`,
`tests/test_agent_output.py`, `tests/test_backends/{test_base_runtime,test_codex}.py`,
`tests/test_cli_watch.py`.

## Deviations from plan

None. The plan was amended after plan-review (78/100 → the B009/B010 noqa, the
`cast` import, and the 8-file format baseline were already in the code; the doc
was updated to match). Implementation review: **99/100 APPROVED** (only a doc
nit, since fixed).

## Enforcement note (important)

The repository is a **fork**; GitHub disables Actions on forks by default, so the
existing `ci.yml` has never run (`total_count: 0`). To activate the gate the
maintainer must, once, click **"I understand my workflows, go ahead and enable
them"** on the repo's Actions tab, then add branch protection on `main` →
Require status checks → `qa`. Neither is settable from a file / by this agent.
