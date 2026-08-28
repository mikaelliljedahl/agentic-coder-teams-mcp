# Backend name aliases + documented `backend` parameter

## Scope

Fix the recurring `Backend 'claude' not found. Available: claude-code, codex, pi`
error seen when an orchestrating Claude Code lead calls `spawn_agent`.

## Current behavior

- `spawn_agent` (`src/claude_teams/server_simple.py`) accepts `backend: str = ""`
  but its docstring never documents the parameter or its legal values. The
  docstring *does* say "For codex, ..." and "For claude-code, ..." inside the
  `model` section, so a caller infers that a backend name exists but has to
  guess its spelling. `"claude"` is the natural guess.
- `BackendRegistry.get()` does an exact dict lookup against
  `_BUILTIN_BACKENDS` keys (`claude-code`, `codex`, `pi`) and raises
  `BackendNotRegisteredError` on anything else.
- The consuming agent only ever reads the MCP tool docstring, so a gap there is
  a product bug, not a caller bug.

## Proposed design

1. **Document `backend` in the `spawn_agent` docstring.** One paragraph naming
   the exact legal values (`claude-code`, `codex`, `pi`) and the empty-string
   default (`claude-code` when installed). This is the real fix.
2. **Explicit alias map in the registry.** `BackendRegistry.resolve_name(name)`
   normalizes case/whitespace and maps a small, hand-written alias table to
   canonical names. `get()` resolves through it. Registered names always win
   over aliases, so a third-party backend literally named `claude` still
   resolves to itself.
   - `claude`, `claudecode`, `claude_code`, `claude-cli`, `claude_code_cli` -> `claude-code`
   - `gpt`, `openai`, `codex-cli` -> `codex`
   No fuzzy matching: a typo must still fail loudly.
3. **Server canonicalizes before storing.** `spawn_agent` resolves the caller's
   backend string to its canonical name *before* it is written to the agent
   record and passed to `_materialize_prompt`, `_hook_extra` and
   `_pi_binding_extra` — otherwise an alias would be persisted and every later
   lookup (resume, follow-up, output reader) would key off `"claude"`.

Explicitly **not** renaming the `claude-code` backend: the key matches the CLI,
`_BUILTIN_BACKENDS`, the README table and the test tree, and `claude` collides
with the model family.

Explicitly **not** adding "did you mean" to `BackendNotRegisteredError`
(dropped by the user).

## Files affected

- `src/claude_teams/backends/registry.py` — alias table + `resolve_name()`.
- `src/claude_teams/server_simple.py` — `spawn_agent` docstring + canonicalization.
- `tests/test_backends/test_registry.py` — alias resolution tests.
- `tests/test_server/` — spawn stores the canonical name.

## Risks

- Silent aliasing can mask a genuine typo. Mitigated by keeping the table small,
  explicit and exact-match only.
- Persisting an alias would corrupt downstream lookups. Mitigated by
  canonicalizing in `spawn_agent` before any use of `backend_name`.

## Test cases

- `resolve_name` maps each alias, is case/whitespace insensitive, and passes an
  unknown name through unchanged (so `get()` still raises for typos).
- A registered backend named like an alias wins over the alias table.
- `get("claude")` returns the `claude-code` backend.
- `get("nope")` still raises `BackendNotRegisteredError`.
- `spawn_agent(backend="claude")` succeeds and records `backend == "claude-code"`.
