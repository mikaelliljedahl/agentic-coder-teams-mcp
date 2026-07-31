# OpenAI pricing tier remap (Luna-first ladder)

## Scope

Remap the codex and pi capability-tier ladders (`_TIER_LAUNCH`) to reflect
OpenAI's July 2026 usage-limit change, which makes GPT-5.6 Luna dramatically
more cost-effective and GPT-5.6 Terra no longer worth using.

## Current behavior

Both `CodexBackend` and `PiBackend` map the five tiers as:

| tier   | model         | effort |
|--------|---------------|--------|
| low    | gpt-5.6-terra | medium |
| medium | gpt-5.6-sol   | low    |
| high   | gpt-5.6-sol   | medium |
| xhigh  | gpt-5.6-sol   | high   |
| ultra  | gpt-5.6-sol   | xhigh  |

## Why change

New per-5h message quotas (Pro 20x): Sol 200–2,000, Terra 500–4,000, Luna
5,000–40,000. Measured token usage on comparable tasks:

- Luna @ xhigh ≈ Sol @ low quality: 45k vs 11k tokens (~4x tokens, ~20x
  cheaper quota → ~5x more cost-effective).
- Luna @ max ≈ Sol @ medium quality: 73k vs 18k tokens (same ratio). We do
  *not* remap `high` onto it — see "Proposed design".
- Terra now yields only ~2x Sol's quota (500–4,000 vs 200–2,000), so it no
  longer earns its place as the cheap tier — Luna is far cheaper.

## Proposed design

New ladder, identical for codex and pi (both target the same GPT-5.6 catalog):

| tier   | model         | effort | rationale                                   |
|--------|---------------|--------|---------------------------------------------|
| low    | gpt-5.6-luna  | high   | dirt cheap; one rung below the measured Luna@xhigh≈Sol@low point, i.e. deliberately below general-default quality for quick/low-stakes work |
| medium | gpt-5.6-luna  | xhigh  | ≈ old Sol@low quality at ~1/5 the cost      |
| high   | gpt-5.6-sol   | medium | unchanged — backend dev, code review        |
| xhigh  | gpt-5.6-sol   | high   | unchanged — genuinely hard problems         |
| ultra  | gpt-5.6-sol   | xhigh  | unchanged — top tier                        |

Only `low` and `medium` move to Luna. `high` deliberately stays Sol @ medium:
it is the working tier for backend development and code review, where the
quality-per-wall-clock of Sol still wins over Luna @ max even though Luna is
cheaper per unit of quota.

Terra is removed from the ladder entirely. Raw-slug passthrough
(`model="gpt-5.6-terra"`) still works — only the tier bundles change.

Default tier stays `medium` (now Luna @ xhigh instead of Sol @ low).

Codex `_REASONING_EFFORT_SPEC.options` already contains `max`; pi
`_THINKING_OPTIONS` already contains `max`. No spec changes needed.

## Files affected

- `src/claude_teams/backends/codex.py` — `_TIER_LAUNCH` + ladder comment +
  `default_model` docstring mention of "Sol @ low".
- `src/claude_teams/backends/pi.py` — `_TIER_LAUNCH` + ladder comment.
- `src/claude_teams/server_simple.py` — no change needed: the ~line 2963
  comment example (`Codex "high" -> Sol @ medium`) stays true, and the spawn
  tool docstring is tier-name-based.
- `tests/test_backends/test_codex.py` — tier assertions (resolve_model,
  resolve_launch, availability-error cases that stub Terra/Sol catalogs).
- `tests/test_backends/test_pi.py` — same.

## Risks

- Luna at a high effort is slower wall-clock than Sol at a lower one; accepted
  for `low`/`medium`, which is why `high` and above stay on Sol.
- Installs whose codex account lacks Luna will now hard-fail `low`/`medium`
  spawns (by design: no silent downgrade; error names the missing slug). This
  hits callers that pass a tier explicitly, **not** callers that omit `model`:
  `spawn_agent` defaults `model` to `""`, and a blank model bypasses the tier
  table entirely and defers to the backend's own configured default. The
  advertised `default_model()` value (`medium`) and "what you get when you omit
  the argument" are deliberately different things — see the contract note below.
- Tests that stub discovery with `["gpt-5.6-terra", "gpt-5.6-sol"]` must be
  updated to include Luna or they'd fail for the wrong reason.

## Contract note: "default tier" vs "omitted model"

Two distinct things, unchanged by this feature but worth stating because the
new Luna dependency makes the difference observable:

- `default_model()` returns the tier name `medium`, which now means Luna @
  xhigh. A caller that reads that value and passes it back acquires the Luna
  dependency.
- Omitting `model` on `spawn_agent` sends `""`, which is an escape hatch: both
  backends skip the tier table and defer to their own configured default model
  (codex `config.toml`, pi's login default), with no Luna validation.

We keep the escape hatch as-is. Existing tests pin it
(`test_blank_model_defers_to_codex_config`, `test_blank_defers_to_pi_default`).

## Decision: pi soft-fallback keeps current semantics

Pi drops an unavailable tier model but keeps the tier's thinking level, so
`medium` degrades to `("", "xhigh")` whenever Luna is absent. This triggers on
*any* catalog without Luna, including a **partial** GPT-5.6 catalog that still
has Sol: pi has no model-substitution rule, so it does not fall back to Sol,
and the tier loses its explicit model selection while Sol-backed tiers keep
working. We accept that rather than inventing a substitution or a safe-effort
downgrade — pi's whole fallback premise is "the login's catalog is unknown, let
pi choose", and substituting a model would hide the misconfiguration behind a
different one. Covered by tuple- and command-level tests using a Sol-only
catalog.

Model/effort compatibility at the external boundary (does *this* codex/pi
install accept Luna @ high/xhigh) is validated by the CLI at launch, not by
this repo; codex hard-errors on a missing slug, and a rejected effort surfaces
as a launch failure. Tests here prove only that the intended slug and effort
reach argv.

## Test cases (TDD)

1. `resolve_model("low"/"medium")` → `gpt-5.6-luna`;
   `"high"/"xhigh"/"ultra"` → `gpt-5.6-sol` (codex + pi).
2. `resolve_launch` per-tier pairs match the new table (codex + pi).
3. Tier effort still overrides caller `reasoning_effort` (codex).
4. Codex catalog boundaries, both directions:
   - Luna absent / Sol present → `medium` raises `BackendModelUnavailableError`;
     `high` still resolves.
   - Sol absent / Luna present → `high` raises it; `medium` still resolves.
   - Empty discovery still skips validation (unchanged).
5. Pi soft-fallback, both catalogs: no GPT-5.6 catalog at all, and a partial
   catalog with Sol but no Luna → `medium` → `("", "xhigh")`, `low` →
   `("", "high")`, while `high` still resolves to `("gpt-5.6-sol", "medium")`.
6. Command-level (pi `build_command`): tier `medium` emits
   `--model openai-codex/gpt-5.6-luna` and `--thinking xhigh`; the Sol-only
   fallback emits `--thinking xhigh` and no `--model`.
7. Command-level (codex `build_command`): a Luna launch emits
   `model='gpt-5.6-luna'` with `model_reasoning_effort=xhigh`.
8. Raw slug passthrough unchanged (incl. `gpt-5.6-terra`).
9. Default tier remains `medium`.

## Verification commands

The venv lives in the primary checkout and its editable install points at the
primary `src/`, so a worktree run must set `PYTHONPATH` to this worktree's
`src` or it silently exercises the primary tree's code. From the worktree root,
with `VENV=<primary-checkout>/.venv/Scripts/python.exe`:

```
PYTHONPATH="$PWD/src" "$VENV" -m pytest tests/test_backends/test_codex.py tests/test_backends/test_pi.py -q
PYTHONPATH="$PWD/src" "$VENV" -m pytest -q     # full repo gate
"$VENV" -m ruff check .
```

Red first: the focused backend run must fail on the old mapping before the
tables change, and pass after. `server_simple.py` is comment-only here, so no
server test is expected to move; the full gate confirms it.
