# Implementation: `xhigh` back to Astra @ low

## Final design

- `codex.py` / `pi.py`: `_TIER_LAUNCH["xhigh"] = ("gpt-6-astra", "low")`; both
  backends still share one ladder (pinned by
  `test_shared_ladder_tiers_match_codex`). Ladder comments carry the review
  rationale; pi's comment notes the Astra tiers also work on pi < 0.87.1.
- Fixed the stale `CodexBackend.default_model` docstring left by PR #67
  ("Luna @ xhigh" → "Luna @ max").
- `spawn_agent` description: `xhigh` bullet →
  `Astra (``gpt-6-astra``) @ low : hard problems, tricky review`.
- README: table row, new `xhigh` rationale paragraph (review use case, token
  efficiency in the 272k window, cost), pi 0.87.0 note (`xhigh`/`max` work).
  `ADDING-A-BACKEND.md` example table.

## Red / green evidence

Red (tests only): `11 failed, 156 passed` across `test_codex.py`,
`test_pi.py`, `test_tool_descriptions.py` — every `xhigh` pair assertion, the
codex missing-Astra `[xhigh]` case, `test_errors_when_sol_tier_unavailable`
(now asserts `xhigh` resolves without Sol), the new pi
`test_stale_pi_catalog_still_serves_astra_tiers`, and the description ladder.

Green: `169 passed` on the four targeted files after the production change.

## Deviations from the plan

None.

## Validation

```
uv run ruff format --check .   # 86 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1698 passed, 4 skipped
```

Test count is one lower than after PR #67 by design: three parametrized
`xhigh`-needs-Sol cases were removed and two Astra cases were added.

## Implementation review disposition (implementation-review.md, Codex)

1. [MINOR] token-efficiency claim lacked its baseline — **accepted, fixed**:
   `codex.py` comment and README now say "about one-third as many tokens as
   GPT-5.6 Sol on coding evaluations". Comment/doc-only; gates re-run green.
