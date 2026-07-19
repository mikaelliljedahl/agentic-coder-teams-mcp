# Implementation: ty diagnostics — Increment 1

## Result

`uv run ty check`: **44 → 12 diagnostics.** All 32 targeted diagnostics in
Increment 1 are resolved. The remaining 12 are the deferred
`backends/process_manager.py` set (Increment 2), unchanged and untouched.

Validation (whole repo):

| Gate | Before | After |
|---|---|---|
| `uv run ty check` | 44 | **12** (all `process_manager.py`) |
| `uv run ruff check src/ tests/` | pass | **pass** |
| `uv run pytest -q` | 484 pass / 2 skip | **497 pass / 2 skip** (+13 new `_safe_float` tests) |

## Red/green evidence

Type-annotation changes are verified by `ty` itself rather than unit-test
red/green. The one behavioral-risk change (`_safe_float`) got a real regression
test:

- **Red basis:** the rejected `isinstance(value, (int, float, str))` guard would
  coerce a non-zero `Decimal`/`Fraction` to `0.0`. `tests/test_safe_float.py`
  asserts `Decimal("2.5") → 2.5` and `Fraction(7, 2) → 3.5`, which that guard
  would fail.
- **Green:** the shipped type-only fix keeps the runtime expression
  `float(value or 0.0)` byte-for-byte; all 13 cases pass.

## Changes (final design)

### Production (3 files)

- `agent_output.py::_content_text` (4) — comprehension → explicit loop with
  `mapping = cast("dict[str, object]", item)`, a typed `text` local, and
  `parts: list[str]`. Filter/order/join behavior identical.
- `server_simple.py` (10):
  - `_annotate(result: dict) -> dict` (5 invalid-return-type). All 5 callers pass
    `run_blocking(_do_*)` whose `_do_*` are declared `-> dict`.
  - `_safe_float` (1) — `return float(cast(Any, value or 0.0))`; runtime
    unchanged, `cast` is a no-op.
  - `permission_mode` (2) — replaced the ty-unhonored `# type: ignore[arg-type]`
    with `cast('Literal["default","require_approval","bypass"]', permission_mode)`
    at both SpawnRequest construction sites.
  - `_follow_up_failure::payload` (1) — annotated `dict[str, object]`.
- `backends/contracts.py::Backend` (1) — added `resume(self, request:
  SpawnRequest, backend_session_id: str) -> SpawnResult` to the Protocol (only
  `resume`, per review). **Compatibility note:** third-party backends registered
  via `BackendRegistry.register` must now provide `resume`; both built-ins
  (`ClaudeCodeBackend`, `CodexBackend`) already inherit `BaseBackend.resume`.

### Tests (12 files)

- 5 record-dict helpers annotated `dict[str, object]` (agent_status, kill_agent,
  restart_safety, stall_signal, agent_output).
- Fake backends (`test_hooks_integration`, `test_spawn_agent_watch_contract`):
  imported `SpawnRequest`, typed `spawn(request: SpawnRequest)` and
  `last_request: SpawnRequest | None`, and at each read site bound `request`,
  `assert request is not None`, `assert request.extra is not None`.
- `test_read_messages::_read(**kwargs: Any)`.
- `tool.description` sites: `assert tool is not None` (session_recovery,
  tool_descriptions).
- `test_process_manager_windows`: `err.winerror = N  # ty: ignore[unresolved-attribute]`
  with a Windows-only-attribute rationale comment.
- `test_cleanup`: `os.utime(stamp, (old, old))` instead of `(t,) * 2`.
- New `tests/test_safe_float.py` (13 cases).

## Deviations from plan

None material. Two refinements during implementation:

1. `_safe_float`: the plan's `coerced: Any = value or 0.0` local did **not**
   silence ty (it still narrowed the assignment type to `~AlwaysFalsy | float`).
   Switched to `float(cast(Any, value or 0.0))`, which is the same runtime
   expression and the same behavior-preserving intent the review approved.
2. `_safe_float` test placement: no `tests/test_server_simple.py` exists, so the
   regression test lives in a new `tests/test_safe_float.py` (as the plan's
   fallback specified).

## Pre-existing breakage note (per CLAUDE.md)

The full-suite run surfaced one intermittent failure,
`tests/test_cli_watch.py::test_watch_no_inbox_preserves_artifact_only_behavior`
(a timing-sensitive test: `--timeout 2` vs a 0.07 s writer thread). It is a
**pre-existing, non-deterministic flake**, not introduced here:

- This change touches no CLI/watch code.
- The test passes in isolation and when its own file runs alone.
- A full-suite run on **clean `main`** (changes stashed) passed
  (484 passed / 2 skip), and a re-run **with** these changes also passed
  (497 passed / 2 skip).

Not addressed in this PR (out of scope; separate flake-hardening if desired).

## Increment 2 (deferred, 12 diagnostics, all `process_manager.py`)

- ctypes Windows-only members (`WinDLL`, `get_last_error`) — 3
- `_popen(**kwargs: object)` defeating `Popen[str]` + `log_handle` `IO[str]`
  narrowing — 6
- three `_tracked_alive(self, info: object)` overrides (Windows / tmux /
  Linux-terminal) — 3; retyping the param risks an LSP/override diagnostic and
  needs a focused decision.
