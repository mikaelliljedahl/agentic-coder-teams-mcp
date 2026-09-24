# Plan: `xhigh` back to Astra @ low

## Scope

PR #67 moved the `xhigh` tier on both `codex` and `pi` from Astra @ low to
Sol @ xhigh on cost grounds. The user's actual use of `xhigh` is **tricky code
reviews** (implementation mostly runs on Claude Opus), which changes the
trade-off. Decision (user, 2026-09-24): `xhigh` → `("gpt-6-astra", "low")` on
both backends. Every other tier is unchanged.

| Tier | now (codex = pi) | after |
| --- | --- | --- |
| `high` | Sol @ high | Sol @ high |
| `xhigh` | Sol @ xhigh | **Astra @ low** |
| `max` | Astra @ medium | Astra @ medium |

Rationale to record (replaces the PR #67 `xhigh` paragraph):

- Sol @ xhigh barely separates from `high` (DeepSWE v1.1: 66.6 % vs Sol @ high
  65.3 %), so `xhigh` was nearly a duplicate tier. Astra @ low (67.0 %) gives
  `xhigh` a different, stronger base model; the ladder becomes
  Sol @ high → Astra @ low → Astra @ medium (72.8 %).
- Review-shaped evidence favours Astra: harder cross-file review subset
  (CodeRabbit) 57.1 % vs 47.6 % actionable findings — measured against
  GPT-5.6 Sol, not GPT-6 Sol, so directional only.
- Astra uses markedly fewer tokens per task (≈⅓ of GPT-5.6 Sol on coding
  evals), keeping more of the reviewed code inside codex's 272k default
  window.
- Astra's reported tendency to widen targeted fixes does not apply to
  read-only review. Cost: ~1.6x per task on DeepSWE, 5x per token; acceptable
  for an infrequent review tier.

## Design / files

- `src/claude_teams/backends/codex.py`: `_TIER_LAUNCH["xhigh"]`, ladder comment
  (rationale above), and the stale `default_model` docstring
  "``medium`` (Luna @ xhigh)" → "(Luna @ max)" left over from PR #67.
- `src/claude_teams/backends/pi.py`: `_TIER_LAUNCH["xhigh"]`, ladder comment;
  check `default_model` docstring for the same staleness.
- `src/claude_teams/server_simple.py`: `spawn_agent` docstring `xhigh` bullet →
  `Astra (``gpt-6-astra``) @ low : hard problems, tricky code review`.
- `README.md`: table row, `xhigh` rationale paragraph, pi 0.87.0 note (Astra is
  in that catalog, so `xhigh`/`max` work there again). `ADDING-A-BACKEND.md`
  example table.

## Tests (red first)

1. Codex: `resolve_model("xhigh") == "gpt-6-astra"`; `resolve_launch` pairs
   (full discovery, legacy env, caller effort ignored) → `("gpt-6-astra",
   "low")`; `test_errors_when_sol_tier_unavailable` back to `high` only;
   Luna+Sol-only catalog: `xhigh` and `max` both raise with the upgrade hint;
   the "GPT-5.6 Luna/Sol/Terra plus GPT-6 Astra" catalog matrix drops `xhigh`
   (Astra is present there, so `xhigh` now resolves).
2. Pi: same resolve pairs; stale pi 0.87.0 catalog matrix drops `xhigh`, and a
   positive assertion that `xhigh` resolves on that catalog.
3. Tool description: `xhigh` bullet pins `(``gpt-6-astra``) @ low `.
4. `test_shared_ladder_tiers_match_codex` keeps pinning codex = pi.

## Risks

- Higher cost/quota per `xhigh` run (Astra 5x per token); intended.
- Evidence for Astra @ low vs GPT-6 Sol @ xhigh is thin and near-tied on
  generic benchmarks; the choice rests on the review use case.

## Validation

`uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
`uv run pytest`.

## Plan review disposition (plan-review.md, Codex)

1. [MINOR] fixture mislabelled "GPT-5.6-only" — **accepted**, wording fixed;
   the missing-Astra case (Luna+Sol-only catalog) is kept for `xhigh` and `max`.
