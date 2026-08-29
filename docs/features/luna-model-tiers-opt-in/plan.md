# Luna model tiers opt-in

## Scope

Make the GPT capability-tier ladder shift one step toward Luna, behind an
opt-in environment variable. Default behavior is unchanged. Covers **both**
GPT-backed backends — `codex` and `pi` — which today carry byte-identical
ladders on purpose, so that a tier keeps meaning the same thing regardless of
which backend runs it.

## Current behavior

`CodexBackend._TIER_LAUNCH` (src/claude_teams/backends/codex.py) and
`PiBackend._TIER_LAUNCH` (src/claude_teams/backends/pi.py) are identical fixed
class-level mappings:

| Tier | Model | Effort |
|------|-------|--------|
| cheapest | gpt-5.6-luna | medium |
| low | gpt-5.6-luna | high |
| medium | gpt-5.6-luna | xhigh |
| high | gpt-5.6-sol | medium |
| xhigh | gpt-5.6-sol | high |
| max | gpt-5.6-sol | xhigh |

Each is read by that backend's `supported_models`, `resolve_model`, and
`resolve_launch`. The two differ only in what an unavailable model does: codex
errors, pi soft-falls-back to its own default model.

## Motivation

Empirically, Luna @ max performs comparably to Sol @ medium. Only genuinely
hard problems (Sol @ high/xhigh) justify switching model family. Shifting the
ladder saves Sol quota, which is an order of magnitude scarcer than Luna's.

## Proposed design

Add a second mapping and select between them at call time:

| Tier | Default (unchanged) | Luna-preferring |
|------|---------------------|-----------------|
| cheapest | luna @ medium | luna @ medium |
| low | luna @ high | luna @ high |
| medium | luna @ xhigh | luna @ xhigh |
| high | sol @ medium | **luna @ max** |
| xhigh | sol @ high | **sol @ medium** |
| max | sol @ xhigh | **sol @ high** |

Selection: env var `WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS`, `1` = opt in.
Any other value (unset, `0`, blank) keeps the current ladder.

Read the env var per call rather than at import time, so a process that sets it
after import (tests, a long-lived MCP server whose env is updated) sees the
change and tests need no module reload.

## Files affected

- `src/claude_teams/backends/contracts.py` — `PREFER_LUNA_ENV` and
  `prefer_luna_tiers()`, shared so the two backends cannot drift apart.
- `src/claude_teams/backends/codex.py` — add `_TIER_LAUNCH_PREFER_LUNA`, a
  `_tier_launch()` accessor gated on the env var, and route the three readers
  (`supported_models`, `resolve_model`, `resolve_launch`) through it.
- `src/claude_teams/backends/pi.py` — the same three changes, mirroring codex.
- `tests/test_backends/test_codex.py`, `tests/test_backends/test_pi.py` — tier
  resolution under both modes, plus a test asserting the two ladders are equal.
- `README.md` — document the env var and the tier table.

## Risks

- Tier *names* are unchanged, so no caller-visible API break; only the
  model+effort a tier resolves to changes, and only under opt-in.
- Codex's `_require_available` still validates the resolved slug, so an install
  without Luna errors loudly instead of downgrading; pi keeps its soft fallback.
- Ordering of `supported_models()` must stay cheapest-first in both maps.
- The two backends' ladders must not diverge — asserted by a test rather than
  left to review.
- The env var is machine-global, so the test suites pin it off via an autouse
  fixture; without that, a developer who exports it sees unrelated tier tests
  fail.

## Test cases

1. Default (env unset): `resolve_launch("high")` -> `("gpt-5.6-sol", "medium")`.
2. Opt-in (`=1`): `resolve_launch("high")` -> `("gpt-5.6-luna", "max")`.
3. Opt-in: `resolve_launch("xhigh")` -> `("gpt-5.6-sol", "medium")`.
4. Opt-in: `resolve_launch("max")` -> `("gpt-5.6-sol", "high")`.
5. `=0` and other non-`1` values behave as default.
6. Cheap tiers (`cheapest`/`low`/`medium`) identical in both modes.
7. `supported_models()` returns the same six names in the same order in both.
8. Raw slug passthrough and blank-model escape hatch unaffected by the flag.
9. Under opt-in, `high` errors on a codex install without Luna (the
   availability check follows the *active* ladder, not the default one).
10. Under opt-in, pi's `high` soft-falls-back to `("", "max")` without Luna.
11. Codex's and pi's active ladders are equal under opt-in.
