# GPT-6 Sol/Luna on the codex and pi tier ladders

## Scope

User intent: "we should run 6 instead of 5.6". Remap the `codex` and `pi`
capability tiers from GPT-5.6 Luna/Sol/Terra to the newly released GPT-6 Luna
(`gpt-6-luna`) and GPT-6 Sol (`gpt-6-sol`). GPT-6 Astra already sits at
`xhigh`/`max`.

In scope: both `_TIER_LAUNCH` tables, the pi upgrade hint, the `spawn_agent`
docstring (the only thing a consuming agent reads), README, ADDING-A-BACKEND.md,
inline comments/docstring examples, and the tests that pin the ladders.

Out of scope: tier names/order, default tier (`medium`), resolution/validation
logic, raw-slug handling, claude-code backend, the unrelated `gpt-reserve` slug.

## Current behavior (verified on `main` @ 1900dcd)

| Tier | codex (`codex.py:122`) | pi (`pi.py:195`) |
| --- | --- | --- |
| `cheapest` | gpt-5.6-luna @ medium | gpt-5.6-luna @ medium |
| `low` | gpt-5.6-luna @ high | gpt-5.6-luna @ high |
| `medium` | gpt-5.6-luna @ xhigh | gpt-5.6-luna @ xhigh |
| `medium-fast` | — | gpt-5.6-terra @ high |
| `high` | gpt-5.6-sol @ medium | gpt-5.6-luna @ max |
| `high-fast` | — | gpt-5.6-sol @ medium |
| `xhigh` | gpt-6-astra @ low | gpt-6-astra @ low |
| `max` | gpt-6-astra @ medium | gpt-6-astra @ medium |

A tier whose model is missing from non-empty live discovery raises
`BackendModelUnavailableError` with an upgrade hint; empty discovery skips
validation.

## Facts gathered

- **codex 0.155.1** `codex debug models`: `gpt-6-sol` (efforts low..ultra,
  272000 ctx), `gpt-6-luna` (low..max, 272000), `gpt-6-astra`, plus all 5.6
  models. No `gpt-6-terra`. So every proposed effort below is accepted.
- **pi 0.87.0** (installed): `--list-models` has no gpt-6-sol/luna.
- **pi 0.87.1** (checked via `npx -y @earendil-works/pi-coding-agent@0.87.1
  --list-models`, global install untouched): exposes `openai-codex gpt-6-luna`
  and `openai-codex gpt-6-sol`, both **272K** native context, 128K max-out.
  Bundled `pi-ai/dist/providers/data/openai-codex.json`: both have
  `thinkingLevelMap` covering low..max, and a priced tier for
  `inputTokensAbove: 272000` (so the API bills >272k input).
- Pricing (pi data, $/Mtok in/out): 6-luna 0.1/0.5 (5.6-luna 0.2/1.2);
  6-sol 2/10 (5.6-sol 4/20; 5.6-terra 2/12); 6-astra 10/50.
- **Pi's "Luna @ 1M" is not a pi default.** pi's bundled data gives
  `gpt-5.6-luna` 272000 too; the 1M window comes from the operator's
  `~/.pi/agent/models.json` `modelOverrides["gpt-5.6-luna"].contextWindow =
  1000000`. The same override does not yet exist for `gpt-6-luna`.
- No benchmark data for GPT-6 Sol/Luna exists in the repo (the Astra plan's
  table covers only 5.6 Luna/Sol and Astra).

## Lead / user decision (received during planning)

Confirmed by the lead, pending the user's final confirmation: `medium-fast` =
`gpt-6-sol @ low`, `high-fast` = `gpt-6-sol @ medium` (the two subtiers keep
their order and don't collapse); every other 5.6 slug becomes its GPT-6
equivalent with the same effort; Astra stays at `xhigh`/`max`. Exception: if
`gpt-6-luna` does NOT have the 1M context in pi, pi `high` can't stay
Luna @ max — flagged as open question 1 below rather than guessed. This matches
D1-D4.

## Design decisions

### D1 — Straight substitution, efforts preserved

5.6-luna → 6-luna and 5.6-sol → 6-sol at the same efforts; Astra unchanged at
`xhigh`/`max`. Rationale: the previous ladders were tuned from a measured
benchmark; we have no GPT-6 Sol/Luna measurements, so re-tuning efforts (e.g.
moving `xhigh` to 6-sol @ high, dropping Astra) would be a guess. This is therefore an intentional, **unbenchmarked**
migration that satisfies the GPT-6-only requirement while preserving effort
labels; it is not a demonstrated non-regression (plan-review F2). GPT-6 tokens
cost about half of their 5.6 counterparts. Follow-up acceptance criterion:
benchmark each ladder point (pass@1 / cost / steps, same suite as the Astra
plan) and re-tune if any tier regresses or the ladder stops ascending.

### D2 — pi `high` stays Luna @ max (now `gpt-6-luna`), defined at 272K

Keep the ladder's shape, but define it against stock pi (plan-review F1): pi's
native `gpt-6-luna` window is 272K, and the ladder/table no longer claims 1M.
The 1M window is documented as an optional, service-dependent operator
`models.json` override (the same kind the operator already has for
`gpt-5.6-luna`). Evidence it is plausible: pi prices `gpt-6-luna` above 272k input
exactly as it does `gpt-5.6-luna`, whose 1M override works today. Without the
override, pi `high` runs at 272K, the same constraint that made codex pick Sol.
**Open question** (below).

### D3 — pi `medium-fast`: `gpt-6-sol @ low` (drop Terra)

Terra has no GPT-6 successor. 6-sol is cheaper than 5.6-terra ($2/$10 vs
$2/$12) and is the GPT-6 "fast" model already used by `high-fast`. Putting
`medium-fast` at 6-sol @ low and `high-fast` at 6-sol @ medium keeps both
subtiers on one model, one effort step apart, and removes all 5.6 models from
the tier ladders per the user's intent. Terra/5.6 models stay reachable as raw
slugs. The latency claim ("roughly 3-4x faster") was measured on 5.6; the
docstring wording is softened to "faster" without the multiplier.
**Open question** (below).

### D4 — Codex `high`: `gpt-6-sol @ medium`

Codex reports a 272k default `context_window` (872k `max_context_window`, which
this integration does not enable or rely on), so the Sol bridge rationale is
unchanged. Stale "262k" wording is updated to "272k default" everywhere.

### D5 — Hard-fail for pi 0.87.0, upgrade path documented

When live discovery is non-empty, stale pi (0.87.0) now fails every tier except
`xhigh`/`max`. That is the
intended loud failure (silent downgrade was the old bug). `_PI_UPGRADE_HINT`
gains the minimum version: "GPT-6 Sol/Luna need pi >= 0.87.1". The spawn_agent
docstring and README state the minimum too. Codex hint unchanged (0.155.1 has
the models; minimum version not determinable without bisecting releases).

## Proposed ladders

| Tier | codex (272k ctx) | pi |
| --- | --- | --- |
| `cheapest` | gpt-6-luna @ medium | gpt-6-luna @ medium |
| `low` | gpt-6-luna @ high | gpt-6-luna @ high |
| `medium` | gpt-6-luna @ xhigh | gpt-6-luna @ xhigh |
| `medium-fast` | — | **gpt-6-sol @ low** |
| `high` | gpt-6-sol @ medium | **gpt-6-luna @ max** (272K native) |
| `high-fast` | — | gpt-6-sol @ medium |
| `xhigh` | gpt-6-astra @ low | gpt-6-astra @ low |
| `max` | gpt-6-astra @ medium | gpt-6-astra @ medium |

## Files affected

- `src/claude_teams/backends/codex.py` — `_TIER_LAUNCH`, ladder comment.
- `src/claude_teams/backends/pi.py` — `_TIER_LAUNCH`, ladder comment,
  `_PI_UPGRADE_HINT`, `--list-models` docstring example.
- `src/claude_teams/server_simple.py` — `spawn_agent` docstring model section.
- `README.md` — tier table + prose.
- `ADDING-A-BACKEND.md` — tier example.
- Tests: `tests/test_backends/test_codex.py`, `tests/test_backends/test_pi.py`,
  `tests/test_tool_descriptions.py`, `tests/test_pi_fast_subtier_resume.py`,
  (`tests/test_resume_session_dir.py` left unchanged — opaque raw-slug data, F6).
- `docs/features/gpt6-tiers/*`.

## Risks

- gpt-6-luna may not accept >272k input even with an override → pi `high`
  degrades on long tasks. Mitigation: documented; open question.
- 6-sol @ low for `medium-fast` is unbenchmarked vs. `medium`.
- Every replaced ladder point is unbenchmarked: capability or ladder order may
  regress at any tier (F2). Follow-up benchmark named in D1.
- The 5.6-era "3-4x faster" multiplier is removed from pi.py, the spawn_agent
  docstring, README and ADDING-A-BACKEND (F5); a test asserts it is gone from
  the registered description.
- Users on pi 0.87.0 hard-fail most tiers until they upgrade (intended; hint
  names the version).
- Docstring tests assert exact phrasing; keep assertions on tier/slug/effort
  lines rather than prose.

## Test cases (red first)

Codex:
1. `resolve_model` for cheapest/low/medium → `gpt-6-luna`, high → `gpt-6-sol`,
   xhigh/max → `gpt-6-astra`.
2. `resolve_launch` pairs for all six tiers with a GPT-6 catalog.
3. Tier ignores caller effort (pairs updated).
4. A 5.6-only catalog (`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-6-astra`) raises
   `BackendModelUnavailableError` naming `gpt-6-luna` for cheapest/low/medium
   and `gpt-6-sol` for high, with the upgrade hint (stale-CLI case).
5. Luna/Sol tiers resolve without Astra; missing Astra still raises.
6. `build_command` for a tier emits `model='gpt-6-luna'` / `'gpt-6-astra'`.
7. Raw `gpt-5.6-terra`/`gpt-5.6-luna` slugs still pass through.

Pi:
1. `resolve_model`/`resolve_launch` for all eight tiers match the table.
2. A pi-0.87.0-shaped catalog (5.6 + astra, no 6-luna/sol) raises for
   `cheapest`, `low`, `medium`, `medium-fast`, `high`, `high-fast`, naming the
   missing slug, and the message contains `0.87.1`.
3. Fast subtiers ignore caller effort; `openai-codex/` qualified ids match.
4. `build_command` qualifies `openai-codex/gpt-6-luna`.
5. Raw `gpt-5.6-terra` slug still passes through / soft-falls.

Docstring (`test_tool_descriptions.py`):
1. `medium-fast` line pins ``gpt-6-sol`` @ low; `high-fast` pins ``gpt-6-sol``
   @ medium.
3. Description contains no `3-4x`.
2. Description mentions `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`, and `0.87.1`,
   and no longer mentions `gpt-5.6`.

## Open questions for user

1. **pi `high` context.** Finding: pi 0.87.1 ships `gpt-6-luna` with a
   **272K** native window — it does NOT have 1M out of the box (neither did
   `gpt-5.6-luna`; its 1M is the operator's `models.json` override). Whether
   the API accepts >272k for `gpt-6-luna` is unverified. Default kept:
   `gpt-6-luna @ max`, assuming the operator
   adds `"gpt-6-luna": {"contextWindow": 1000000, "maxTokens": 128000}` to
   `~/.pi/agent/models.json` (mirroring today's 5.6-luna override). If GPT-6
   Luna does not actually accept >272k, pi `high` should instead become
   `gpt-6-sol @ medium` like codex.
2. **pi `medium-fast`.** Default: `gpt-6-sol @ low`. Alternative: keep
   `gpt-5.6-terra @ high` (benchmarked, but still 5.6).
3. **Re-tuning efforts** once GPT-6 Sol/Luna are benchmarked (possible:
   6-sol @ high could displace Astra @ low at `xhigh`).

## Resolution of open question 1 (user, during implementation)

The user answered: the 1M Luna window is their own local pi setting, and "this
MCP does not need to care about context window". Pi `high` therefore stays
`gpt-6-luna @ max`, and the docs no longer make context-window claims or give
override instructions for pi (supersedes the F1 documentation approach: the
ladder is no longer described in terms of pi's context window at all).
