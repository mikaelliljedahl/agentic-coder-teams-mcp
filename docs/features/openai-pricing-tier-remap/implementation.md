# Implementation: OpenAI pricing tier remap

## Final design

`_TIER_LAUNCH` in both `codex.py` and `pi.py`:

| tier   | before                | after                 |
|--------|-----------------------|-----------------------|
| low    | gpt-5.6-terra @ medium| gpt-5.6-luna @ high   |
| medium | gpt-5.6-sol @ low     | gpt-5.6-luna @ xhigh  |
| high   | gpt-5.6-sol @ medium  | unchanged             |
| xhigh  | gpt-5.6-sol @ high    | unchanged             |
| ultra  | gpt-5.6-sol @ xhigh   | unchanged             |

Terra is off the ladder (still reachable as a raw slug). No effort-spec,
default-tier, or server changes: `xhigh` is already an advertised effort on
both backends, `medium` is still the default tier name, and
`server_simple.py`'s `high -> Sol @ medium` comment stays true.

## Deviation from the plan

The first plan draft also moved `high` to Luna @ max. Per user direction,
`high` stays Sol @ medium — it is the backend-dev / code-review working tier
where Sol's quality-per-wall-clock still wins. Consequence: no tier introduces
the `max` effort, so the plan-review MAJOR about an untested `("", "max")` pi
fallback no longer applies.

## Plan-review disposition (`plan-review-1.md`)

- MAJOR (Sol hard-fail coverage lost) — **fixed**: `test_errors_when_sol_tier_unavailable`
  now targets `high`, and both catalog directions are asserted
  (`test_errors_when_luna_tier_unavailable`, `test_sol_tiers_ok_without_luna`,
  `test_luna_tiers_ok_without_sol`).
- MAJOR (pi `max` fallback) — **obsolete** after the `high` decision; the pi
  fallback is covered at tuple level (`("", "xhigh")`) and at command level.
- MINOR (external compatibility) — **accepted as documented precondition**;
  command-level tests now prove the intended slug + effort reach argv on both
  backends.
- MINOR (no executable gate) — **fixed**: verification commands recorded in
  `plan.md` and below.
- NIT (`low` rationale) — **fixed** in `plan.md`.

## Implementation-review disposition (`implementation-review-1.md`)

- MAJOR (plan's "default spawn fails without Luna" is false) — **fixed in
  docs**: `plan.md` now separates `default_model()` (the advertised tier
  `medium`, Luna-dependent) from an omitted `model` argument (blank → backend's
  own configured default, no tier table, no Luna validation). No code change:
  the blank escape hatch is deliberate and already pinned by
  `test_blank_model_defers_to_codex_config` / `test_blank_defers_to_pi_default`.
- MAJOR (pi partial-catalog gap) — **fixed**: added
  `test_partial_catalog_drops_model_rather_than_substituting` (Sol present,
  Luna absent → `low`/`medium` lose the model, `high` unaffected) and switched
  the command-level fallback test to a Sol-only catalog. The plan's decision
  text now states the model-loss behaviour explicitly instead of claiming the
  fallback only fires on a non-GPT catalog. This is characterization coverage
  of behaviour the change makes reachable, so it passes without a production
  edit — it was written to lock in a decision, not to drive one.
- MINOR (plan's verification commands don't run in a worktree) — **fixed**:
  `plan.md` now carries the `PYTHONPATH`-prefixed worktree-safe form.
- NIT (stale `max` in the compatibility sentence) — **fixed**.

## Red/green evidence

Red (tests updated first, tables unchanged):

```
12 failed, 86 passed in 5.75s
```

Failing: codex `test_resolves_tier_to_slug`, `test_tiers_map_to_model_and_effort`,
`test_explicit_effort_ignored_for_tier`, `test_luna_tiers_ok_without_sol`,
`test_errors_when_luna_tier_unavailable`,
`test_build_command_emits_default_tier_launch`; pi
`test_resolve_model_tier_to_slug`, `test_tier_maps_to_model_and_thinking`,
`test_soft_fallback_when_tier_model_absent`,
`test_skips_validation_when_discovery_empty`,
`test_medium_tier_launch_reaches_argv`,
`test_medium_tier_fallback_keeps_thinking_without_model`.

Green after the `_TIER_LAUNCH` edits: `98 passed`.

## Validation commands

The repo venv lives in the primary worktree and its editable install points at
the primary `src/`, so a worktree run must set `PYTHONPATH` to this worktree's
`src` or it silently tests the primary tree's code:

```
PYTHONPATH=<worktree>/src <repo>/.venv/Scripts/python.exe -m pytest tests/test_backends/test_codex.py tests/test_backends/test_pi.py -q
PYTHONPATH=<worktree>/src <repo>/.venv/Scripts/python.exe -m pytest -q
<repo>/.venv/Scripts/python.exe -m ruff check .
```

Results: focused `99 passed` (98 before the review's added pi test); full suite
`1177 passed, 2 skipped`; ruff clean.

One full-suite run showed 3 failures in `tests/test_watch_command_discovery.py`
(`TimeoutExpired` on a 10s subprocess budget) while a Codex agent was saturating
the CPU. They pass in isolation and on a re-run of the full suite; this is
load-sensitive flakiness in those tests, unrelated to this change.
