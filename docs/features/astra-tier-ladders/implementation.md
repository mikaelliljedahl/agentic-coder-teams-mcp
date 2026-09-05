# Implementation — Astra GPT tier ladders

## Red/green evidence

- Red first: the rewritten focused tier tests failed against the old ladders and
  old Pi soft-fallback behavior (`uv run pytest tests/test_backends/test_codex.py
  tests/test_backends/test_pi.py`).
- Green focused run: 121 passed.
- `uv run ruff format --check .`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest` with the harness's injected agent environment removed:
  1,390 passed, 2 skipped.
- `uv run ty check`: one pre-existing Windows-only diagnostic at
  `tests/test_join_team.py:730` (`BaseContext.Process`); the Linux-targeted
  check (`uv run ty check --python-platform linux`) passed. The feature does
  not modify that test.

## Final design

- `codex` and `pi` each use one fixed six-tier ladder. Both use Luna for
  `cheapest`/`low`/`medium`, Codex uses Sol @ medium for `high`, Pi uses Luna @
  max for `high`, and both use `gpt-6-astra` @ low/medium for `xhigh`/`max`.
- The legacy `WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS` helper and ladder
  variants are removed; a still-set variable is ignored.
- `BackendModelUnavailableError` accepts an optional backend-specific upgrade
  hint. Codex and Pi now include their install commands when a discovered tier
  model is missing.
- Pi tier launches hard-fail on a non-empty catalog, while explicit raw slugs
  retain the existing soft fallback and empty discovery still skips validation.
  Pi availability comparison normalizes provider prefixes on both requested and
  discovered IDs.
- Spawn-agent documentation, README, backend-authoring guidance, and the
  messaging protocol now describe the per-backend ladders and failure signals.

## Deviations and review dispositions

The implementation follows the approved plan and the independent reviews. The
provider-prefix normalization and regression test were added per plan-review F3
so catalogs containing `openai-codex/gpt-6-astra` do not false-error. The Windows
`ty` diagnostic is pre-existing and platform-specific; no unrelated behavior was
changed to hide it.
