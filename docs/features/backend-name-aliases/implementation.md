# Implementation — backend name aliases + documented `backend` parameter

## Final design

1. `spawn_agent`'s docstring now opens with a `backend:` paragraph naming the
   exact legal values (`claude-code`, `codex`, `pi`), stating that the default
   is `claude-code`, pointing at `list_backends()`, and calling out that plain
   `claude` is a model family rather than a backend key.
2. `src/claude_teams/backends/registry.py` gained
   - `_BACKEND_ALIASES`, a hand-written exact-match table
     (`claude`, `claudecode`, `claude_code`, `claude-cli`, `claude_code_cli`,
     `claude-code-cli` -> `claude-code`; `gpt`, `openai`, `codex-cli` -> `codex`);
   - `canonical_backend_name()`, a pure case/whitespace-insensitive mapper that
     returns unknown names unchanged so typos still fail loudly;
   - `BackendRegistry.resolve_name()`, which prefers an actually registered
     backend over the alias table;
   - `get()` now looks up through `resolve_name()`, while the error still
     reports the name the caller passed.
3. `spawn_agent` canonicalizes via `registry.resolve_name()` *before* the name
   reaches `_materialize_prompt`, `_hook_extra`, `_pi_binding_extra` or the
   agent record, so an alias can never be persisted.

## Deviation from the plan

None in substance. The alias resolution is exposed both as a free function
(`canonical_backend_name`) and as a registry method; the free function exists so
the test fake registries can mirror real behavior without duplicating the table.
Six pre-existing `_FakeRegistry` test doubles gained a `resolve_name` delegating
to it.

## Red/green evidence

- New tests in `tests/test_backends/test_registry.py`
  (`TestRegistryNameAliases`) and `tests/test_spawn_agent_watch_contract.py`
  (`test_spawn_agent_records_canonical_backend_for_alias`,
  `test_spawn_agent_docstring_documents_backend_values`).
- Before the change, `_FakeRegistry.get` asserts `backend == "claude-code"`, so
  the alias spawn test failed on that assertion; `resolve_name` did not exist,
  so every registry alias test failed with `AttributeError`.
- After: `1302 passed, 2 skipped`.

## Validation

```
uv run ruff format --check .   # 78 files already formatted
uv run ruff check .            # All checks passed
uv run ty check                # 1 pre-existing diagnostic (see below)
uv run pytest                  # 1302 passed, 2 skipped
```

`ty check` reports one diagnostic, `unresolved-attribute` on
`context.Process` at `tests/test_join_team.py:730`. That file is untouched by
this change and the diagnostic pre-dates it; it is one of the known
Windows-only `ty` findings noted in CLAUDE.md.
