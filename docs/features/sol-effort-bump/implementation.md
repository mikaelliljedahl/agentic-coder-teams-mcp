# Implementation: raise the GPT-6 tier ladder one effort step

## Final design

Both `_TIER_LAUNCH` tables now carry the ladder agreed in `plan.md`:

| Tier | codex | pi |
| --- | --- | --- |
| `cheapest` | Luna @ high | Luna @ high |
| `low` | Luna @ xhigh | Luna @ xhigh |
| `medium` | Luna @ max | Luna @ max |
| `medium-fast` | — | Sol @ medium |
| `high` | Sol @ high | Sol @ high |
| `xhigh` | Sol @ xhigh | Sol @ xhigh |
| `max` | Astra @ medium | Astra @ medium |

- `high-fast` removed from pi. `PiBackend._RETIRED_TIERS = {"high-fast": "high"}`
  and `_reject_retired_tier()` run first in both `resolve_model` and
  `resolve_launch`, raising the new `RetiredTierError`
  (`contracts.py`, subclass of `UnsupportedBackendModelError`), so a stale
  caller never reaches the raw-slug passthrough / soft fallback.
- Ladder comments in `codex.py` / `pi.py` rewritten with the benchmark rationale
  (one effort step up for GPT-6, Sol @ xhigh over the 5x-priced Astra @ low,
  Astra @ medium stays top) and the codex 272k note re-pointed at Luna @ max.
- `spawn_agent` docstring now lists every tier with its model and effort on one
  line, states the ladders are identical, lists only `medium-fast` as pi-only,
  and names `high-fast` as removed (`RetiredTierError`, use `high`).
- README table/prose (stability claim now names the one exception; pi 0.87.0
  note: only `max` works), `ADDING-A-BACKEND.md`, `CLAUDE.md` updated.

## Red / green evidence

Red (tests changed first, production untouched):

- `tests/test_backends/test_pi.py`: collection error —
  `ImportError: cannot import name 'RetiredTierError'`.
- Remaining targeted files: `12 failed, 88 passed` — every codex tier pair,
  the new `xhigh` Sol-unavailable cases, default-tier argv (`effort=max`),
  both `medium-fast` resume regressions, and both description tests.

Green: `170 passed` on the four targeted files. Two description-test failures
on the first green run were test-side (expected `` ``slug`` @ effort `` but the
docstring format is `` (``slug``) @ effort ``, and the test forbade any mention
of `high-fast` although the design names it as removed); the assertions were
corrected to pin the tier bullet and the removal notice.

## Deviations from the plan

- Tool description keeps a single sentence naming `high-fast` as removed (plan
  test 6 said "absent"); it is absent as a tier bullet. Deliberate: the
  consuming agent only reads the description, so it should learn the
  replacement.

## Validation

```
uv run ruff format --check .   # 86 files already formatted
uv run ruff check .            # All checks passed!
uv run ty check                # All checks passed!
uv run pytest                  # 1699 passed, 4 skipped
```

## Implementation review disposition (implementation-review.md, Codex)

1. [MAJOR] "every tier one effort step up" overstated — **accepted, fixed**:
   README, `codex.py` and `pi.py` comments now scope the one-step claim to
   `cheapest`..`high` (+ `medium-fast`), state that pi `high` switched Luna @
   max → Sol @ high, and explain `xhigh` (model change to Sol) and `max`
   (unchanged) separately. Doc/comment-only; all four gates re-run green
   (1699 passed, 4 skipped).
