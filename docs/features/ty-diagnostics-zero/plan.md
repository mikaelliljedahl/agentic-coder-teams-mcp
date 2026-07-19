# Plan: drive `ty check` diagnostics toward zero

## Scope

`uv run ty check` currently reports **44 diagnostics** on `main` (down from a
historical 63). None were introduced by a recent feature; this is standing type
debt. Goal: eliminate it with **minimal, behavior-preserving** fixes — narrow
`Optional`/`object` types with guards or precise annotations, add missing
annotations, and fix genuinely wrong types. Blanket suppressions are only used
where a diagnostic is a true platform/false-positive case, and each such case is
justified inline.

This work is split into two reviewable increments. **This PR delivers
Increment 1 (32 diagnostics).** Increment 2 (12 diagnostics, all in
`backends/process_manager.py`) is documented here and deferred so the
process-manager typing (Windows ctypes + tmux/Linux-terminal `_tracked_alive` +
the coupled `_popen` kwargs) gets its own focused review.

## Current behavior

`ty` runs clean-ish but emits 44 errors. Notable: `ty` does **not** honor
`# type: ignore[...]` comments (it honors `# ty: ignore[...]`), so several
already-suppressed lines still surface (e.g. `info.process  # type: ignore`).

## Diagnostic inventory (44)

### Increment 1 — cross-platform, tests, and `server_simple.py` (32)

| # | File / lines | Rule | Root cause | Fix |
|---|---|---|---|---|
| 4 | `agent_output.py:352,355,356,360` | invalid-argument-type / no-matching-overload | `_content_text` list-comp: after `isinstance(item, dict)` the dict is `dict[Unknown, Unknown]`, so `item["text"]`, `item.get(...)`, and `"".join(parts)` see `Never`/`object` keys & values | **Rewrite the comprehension as an explicit loop** (a comprehension target cannot be annotated). For each `item`, `if not isinstance(item, dict): continue`, then `mapping = cast("dict[str, object]", item)`, `if mapping.get("type") != text_type: continue`, `text = mapping.get("text")`, `if isinstance(text, str): parts.append(text)`. `parts: list[str]` → `"".join(parts)` is clean. Behavior identical to the current filter+join |
| 5 | `server_simple.py:1302,1339,1504,1559,1738` | invalid-return-type | `_annotate(result: object) -> object` returned where `-> dict` expected | Change signature to `_annotate(result: dict) -> dict`; all 5 call sites pass `run_blocking(_do_*)` which returns `dict` |
| 1 | `server_simple.py:909` | invalid-argument-type | `float(value or 0.0)` with `value: object` yields `~AlwaysFalsy | float`, not `ConvertibleToFloat` | **Type-only fix, runtime expression unchanged.** Bind a local `coerced: Any = value or 0.0` then `return float(coerced)`. The `cast`/`Any` is a no-op at runtime, so `float(value or 0.0)` still executes exactly as today for `Decimal`, `Fraction`, `bytes`, custom `SupportsFloat`, `bool`, `0`, `""`, `None` — all preserved under the existing `TypeError/ValueError` fallback. (The earlier `isinstance` guard idea was **rejected** by review: it would coerce non-zero `Decimal`/`Fraction`/bytes/`SupportsFloat` to `0.0` — a behavior change.) Add tests: non-zero `Decimal` → its float value; `""`/`None`/`0` → `0.0` |
| 2 | `server_simple.py:1262,1697` | invalid-argument-type | `permission_mode: str` passed where `Literal["default","require_approval","bypass"]` expected; `# type: ignore[arg-type]` not honored by ty | Replace the ignore with `typing.cast("Literal[...]", permission_mode)` — faithful equivalent of the existing "trust me" intent |
| 1 | `server_simple.py:1703` | unresolved-attribute | `Backend` protocol lacks `resume`, but every concrete backend (`process_base.BaseBackend`) implements it | Add **only** `resume(self, request: SpawnRequest, backend_session_id: str) -> SpawnResult` to the `Backend` Protocol in `contracts.py`. Do **not** add `supports_resume`/`build_resume_command` — the reported error is solely the `resume` call, and `supports_resume` is reached via `getattr`. Both registered backends (`ClaudeCodeBackend`, `CodexBackend`) inherit `BaseBackend.resume`, so no concrete gap is masked. **Compatibility note:** this widens the public structural protocol — third-party backends registered via `BackendRegistry.register` must now provide `resume` even when `supports_resume` is false; documented in the PR |
| 1 | `server_simple.py:1086` | no-matching-overload | `payload.update({...})` where `payload` inferred with narrow value types | Annotate `payload: dict[str, object]` at its construction |
| 5 | `test_agent_output.py:1105`, `test_agent_status.py:33`, `test_kill_agent.py:38`, `test_restart_safety.py:69`, `test_stall_signal.py:33` | no-matching-overload | `agent.update(overrides)`: `agent` inferred narrow, `overrides` is `dict[str,object]` | Annotate the local record dict as `dict[str, object]` in each helper |
| 4 | `test_hooks_integration.py:62,92`, `test_spawn_agent_watch_contract.py:115,116` | unresolved-attribute | fake backend `self.last_request: object`, **and** the fake `spawn(self, request: object)` methods assign `object` into it | Three coordinated edits per fake backend: (a) `from claude_teams.backends.contracts import SpawnRequest` (no import cycle — `contracts` imports neither test nor `server_simple`); (b) change the fake `spawn`/`resume` parameter to `request: SpawnRequest` and type the field `self.last_request: SpawnRequest \| None`; (c) at each read site bind `request = backend.last_request`, `assert request is not None`, and — since `extra` is itself `dict \| None` on `SpawnRequest` — `assert request.extra is not None` before subscripting `request.extra[...]`. (Review flagged that a field annotation + single assert alone leaves the `object` param and the optional-`extra` errors.) |
| 4 | `test_read_messages.py:13` (×4) | invalid-argument-type | `_read(**kwargs: object)` forwarded to typed params | Change helper to `**kwargs: Any` (test forwarder) |
| 2 | `test_session_recovery.py:251`, `test_tool_descriptions.py:66` | unresolved-attribute | `tool: Tool | None` then `tool.description` | Add `assert tool is not None` before use |
| 2 | `test_process_manager_windows.py:121,142` | unresolved-attribute | `err.winerror = N` — `winerror` is Windows-only on `OSError` | Use `object.__setattr__`/annotate, or set via a typed helper; simplest: `err.winerror = N  # ty: ignore[unresolved-attribute]` (Windows-only stdlib attr) with justification |
| 1 | `test_cleanup.py:113` | invalid-argument-type | `(t,) * 2` → `tuple[float, ...]`, `os.utime` wants 2-tuple | Pass an explicit 2-tuple: `(t, t)` |

### Increment 2 — `backends/process_manager.py` (12) — DEFERRED

Note: these are **not all Windows-only** (a review-corrected mislabel). The
ctypes members are Windows-specific, but the three `_tracked_alive` errors span
the Windows manager (`ProcessInfo`, line 636), the **tmux** manager
(`TmuxProcessInfo`, line 1368), and the **Linux-terminal** manager
(`LinuxTerminalProcessInfo`, line 1758).

| # | Lines | Rule | Note |
|---|---|---|---|
| 3 | 96, 755, 771 | unresolved-attribute | `ctypes.WinDLL`, `ctypes.get_last_error` — Windows-only; ty assumes Linux. Needs `# ty: ignore` w/ justification or platform guard |
| 4 | 589, 606 (×2 each) | invalid-return-type / no-matching-overload | `_popen(**kwargs: object)` defeats `Popen[str]` inference. A **result-only** cast is insufficient (review); needs a typed-kwargs design plus an `IO[str]` narrowing for `stdout` |
| 2 | 603, 604 | call-non-callable / unresolved-attribute | `log_handle` is `object` from `kwargs.get`; needs the same `IO[str]` narrowing before `write`/`flush` |
| 3 | 636, 1368, 1758 | unresolved-attribute | `_tracked_alive(self, info: object)` overrides; each accesses a concrete field (`.process`/`.pane_id`/`.terminal_process`). Mechanically fixable by retyping the param, **but** the base override is `info: object`, so narrowing the param risks an LSP/`invalid-override` diagnostic that needs verification — deferred with the rest of the file rather than rushed into this PR |

Rationale for deferral: all 12 live in one subsystem file. The ctypes and
`_popen`-kwargs work is genuinely coupled and platform-specific; the three
`_tracked_alive` annotations are individually trivial but share the override/LSP
question above. Keeping the whole file as **one focused follow-up** (rather than
splitting three lines out) keeps this PR's boundary clean — one file, one
review — and avoids landing a half-typed `process_manager.py`.

## Files affected (Increment 1)

- `src/claude_teams/agent_output.py`
- `src/claude_teams/server_simple.py`
- `src/claude_teams/backends/contracts.py`
- `tests/test_agent_output.py`, `tests/test_agent_status.py`,
  `tests/test_kill_agent.py`, `tests/test_restart_safety.py`,
  `tests/test_stall_signal.py`, `tests/test_hooks_integration.py`,
  `tests/test_spawn_agent_watch_contract.py`, `tests/test_read_messages.py`,
  `tests/test_session_recovery.py`, `tests/test_tool_descriptions.py`,
  `tests/test_backends/test_process_manager_windows.py`, `tests/test_cleanup.py`
- New `_safe_float` regression test lands in `tests/test_server_simple.py` if it
  exists, else a new `tests/test_safe_float.py` (imports `Decimal` from `decimal`)

## Risks

- **`_annotate` signature narrowing** — if any caller passes a non-dict, this
  would newly error. Verified all 5 call sites pass `run_blocking(_do_*)` which
  returns `dict`. Low risk.
- **Adding `resume` to the `Backend` Protocol** — protocols are structural, so
  every concrete backend must actually provide it. `process_base.resume` exists;
  confirm all registered backends inherit it. If any backend lacks `resume`,
  that is a real latent bug the type fix surfaces (good), and must be resolved
  rather than masked.
- **`cast` for `permission_mode`** — hides invalid string values exactly as the
  prior `# type: ignore` did; no behavior change. Upstream validation unchanged.
- **`_safe_float` type-only fix** — the runtime expression `float(value or 0.0)`
  is left byte-for-byte identical; only a no-op `Any`/`cast` local is added, so
  no float-convertible input changes result. A regression test for non-zero
  `Decimal` guards against the rejected `isinstance`-guard approach.
- **`ty: ignore` on `winerror` / (Increment 2) ctypes** — these are genuine
  Windows-only stdlib members; suppression is the correct call on a Linux CI.

## Test cases / validation

TDD is awkward for pure type-annotation changes (the "test" is `ty` itself), so
the discipline here is: **the full runtime test suite must stay green** (proving
behavior preserved) while the `ty` count drops.

- `uv run ty check` → expect **12** remaining (all `process_manager.py`), down
  from 44. Report the exact number.
- `uv run pytest -q` → no new failures vs. baseline.
- `uv run ruff check src/ tests/` → no new lint (whole-repo, per CLAUDE.md).
- For `_content_text`: confirm existing `test_agent_output.py` coverage of text
  extraction still passes (behavior identical).
- For `_safe_float`: **add** a focused test — non-zero `Decimal("2.5")` → `2.5`,
  and `""`/`None`/`0` → `0.0` — proving the type-only change is behavior-preserving.
- For `Backend.resume`: `test_session_recovery` / resume-path tests still pass.

## Out of scope

- Increment 2 (`process_manager.py`, 12 diagnostics) — follow-up PR.
- Any refactor beyond the minimal typing change needed per diagnostic.
