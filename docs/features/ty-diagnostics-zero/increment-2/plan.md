# Plan: ty diagnostics → 0 (Increment 2) + Linux QA gate

## Scope

Finish the type-debt cleanup: drive `uv run ty check` from **12 → 0** on the
**Linux** target (the CI gate platform), and add a GitHub Actions **QA gate**
that fails a PR unless `ty check` is clean **and** the test suite is 100% green.
Tests remain runnable on both Linux and Windows.

All 12 remaining diagnostics are in `backends/process_manager.py`. This
increment also **corrects an Increment-1 mistake**: the `winerror` fix used
`# ty: ignore`, which is a *valid attribute on Windows* and would surface as an
`unused-ignore` warning there — replaced with a platform-agnostic `setattr`.

Delivered by appending to the existing branch/PR (#26), since the winerror
correction belongs with Increment 1 and the whole effort is one goal:
"type debt → 0 + enforce it."

## Platform reasoning (why these fixes, and why Linux)

`ty`'s diagnostic set is platform-dependent. Confirmed empirically with
`ty check` vs `ty check --python-platform win32`:

| Group | Linux (gate) | win32 |
|---|---|---|
| `ctypes.WinDLL` / `get_last_error` | error | clean |
| `signal.SIGKILL` in `os.kill` | clean | error (Linux-only code paths) |
| `Popen` overload/return, `log_handle`, `_tracked_alive` | error | error |
| Inc-1 `winerror # ty: ignore` | used (valid) | **unused-ignore warning** |

The QA gate runs on **GitHub / Linux**, so "zero" is defined on Linux. Fixes are
chosen to be **platform-agnostic where cheap** (clean on both OSes) so nothing is
fragile and the code stays cross-platform:

- `ctypes` Windows-only members → `getattr(ctypes, "…")` (clean on both).
- `winerror` test assignment → `setattr(err, "winerror", N)` (clean on both;
  reverts the Inc-1 ignore).
- `SIGKILL` (lines 739/1593/2070) → **left untouched**: not flagged on the Linux
  gate, lives in Linux/tmux-only code paths, and rewriting kill logic carries
  behavior risk for no gate benefit. (Under win32 two of these remain, which is
  acceptable because the ty gate is Linux; documented, not hidden.)

## Diagnostic inventory & fixes (12 → 0 on Linux)

| # | Lines | Rule | Fix | Verified |
|---|---|---|---|---|
| 6 | `_popen` 589, 606 (invalid-return + no-matching-overload ×2) and 603/604 (call-non-callable, flush) | `**kwargs: object` defeats `Popen[str]` inference and leaves `log_handle = kwargs.get("stdout")` typed `object` | Change `_popen(**kwargs: object)` → `**kwargs: Any` (import `Any`). `log_handle` becomes `Any`; both `Popen` calls resolve; **no cast needed** | ✅ tested: 12→6 |
| 3 | 96, 755 (`ctypes.WinDLL`), 771 (`ctypes.get_last_error`) | Windows-only stdlib members; ty assumes Linux | `getattr(ctypes, "WinDLL")(...)` and `getattr(ctypes, "get_last_error")()`. Returns `Any` (these are dynamic FFI handles already); clean on Linux and Windows | to verify |
| 3 | 636, 1368, 1758 | `_tracked_alive(self, info: object)` overrides access `.process`/`.pane_id`/`.terminal_process` | Bind `typed = cast(ProcessInfo/TmuxProcessInfo/LinuxTerminalProcessInfo, info)` and use `typed.…`. Keeps the base `info: object` param (no LSP/override risk); remove the now-redundant `# type: ignore[attr-defined]` | to verify |

### Increment-1 correction (not counted above; currently clean on Linux)

- `tests/.../test_process_manager_windows.py` lines 121, 142: replace
  `err.winerror = N  # ty: ignore[unresolved-attribute]` with
  `setattr(err, "winerror", N)`. Removes the Windows `unused-ignore` fragility;
  still clean on the Linux gate.

## CI QA gate (`.github/workflows/ci.yml`)

Restructure the single windows job into:

- **`qa` (ubuntu-latest)** — the required gate: `uv sync --locked --group dev`,
  `ruff format --check .`, `ruff check .`, `uv run ty check`, `uv run pytest`.
- **`tests-windows` (windows-latest)** — cross-platform test coverage:
  `uv sync --locked --group dev`, `uv run pytest`.

Both triggered on `pull_request` (and push to `main`). `ty check` and `ruff`
run only in `qa` (Linux), matching "gate on Linux". Branch protection making
`qa` a **required** status check is a repo setting the maintainer enables — I
cannot set it from a file; called out in the PR.

## Ruff handling (required for the gate — added after plan review)

The literal `getattr`/`setattr` replacements and the new `ruff format --check`
gate need explicit lint work, or the `qa` job is red immediately:

- Each literal `getattr(ctypes, "…")` carries `# noqa: B009` with a reason
  (matching the existing ctypes lookups already in this file); each
  `setattr(err, "winerror", …)` carries `# noqa: B010`. Suppressions sit on the
  expression line (a wrapping `return (` line would be ineffective → RUF100).
- Both `Any` and `cast` are imported in `process_manager.py`.
- `ruff format --check .` fails on **8 pre-existing unformatted files** on the
  base commit. They are reformatted in a **dedicated `style: ruff format` commit**
  so the gate is green and the functional diff stays reviewable
  (behavior-preserving; per CLAUDE.md's cosmetic-pre-existing-breakage rule):
  `backends/codex.py`, `backends/contracts.py`, `backends/process_manager.py`,
  `backends/registry.py`, `tests/test_agent_output.py`,
  `tests/test_backends/test_base_runtime.py`, `tests/test_backends/test_codex.py`,
  `tests/test_cli_watch.py`.
- CI uses `uv sync --locked --group dev` (lock verified current via
  `uv lock --check`) for reproducible, fail-fast installs.
- "100% tests" = a fully green `uv run pytest` (matches the request); coverage is
  not gated here (no `--cov` in pytest addopts).

## Files affected

- `src/claude_teams/backends/process_manager.py`
- `tests/test_backends/test_process_manager_windows.py` (winerror → setattr)
- `.github/workflows/ci.yml`
- The 8 `ruff format` files above (separate style commit)

## Risks

- **`**kwargs: Any` widens `_popen`** — acceptable: it's a thin dynamic wrapper
  and callers already build kwargs dynamically. No runtime change. Verified it
  resolves the overloads without a cast.
- **`getattr(ctypes, …)` returns `Any`** — loses static typing on the FFI
  handles, which were already effectively dynamic (`.OpenProcess.argtypes = …`).
  No runtime change (`getattr(mod, "name")` == `mod.name` when present).
- **`cast` on `_tracked_alive` info** — no-op at runtime; the concrete type is
  guaranteed by the per-subclass `self._processes` dict. Removing the stale
  `# type: ignore[attr-defined]` is safe (no mypy configured in the repo).
- **CI restructure** — must keep `uv sync --locked --group dev` (the dev group carries
  ruff/ty/pytest). Windows test job must still pass with the `setattr`/getattr
  changes (they're behavior-identical).

## Validation

- `uv run ty check` → **0** (Linux). Also spot-check `--python-platform win32`
  drops from 13 to the 2 documented Linux-only-code SIGKILL entries + 0
  unused-ignore (winerror now clean).
- `uv run ruff check src/ tests/` and `ruff format --check` → clean.
- `uv run pytest -q` → no regressions; the process-manager Windows tests
  (breakaway fallback) still pass with `setattr`.

## Out of scope

- Full win32 `ty` cleanliness (the 2 SIGKILL-in-Linux-paths entries) — the gate
  is Linux; noted for a possible future cross-platform ty run.
- Enabling branch protection (repo setting, maintainer action).
