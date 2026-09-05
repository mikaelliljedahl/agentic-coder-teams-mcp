# Astra on the GPT tier ladders; per-backend ladders replace the Luna opt-in

## Scope

1. Put GPT-6 Astra (`gpt-6-astra`, released 2026-09-04) on the `codex` and `pi`
   capability-tier ladders, replacing Sol at the top two tiers.
2. Remove the `WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS` opt-in. Each backend
   gets one fixed ladder; the two differ only at `high`.
3. Give the lead an unmissable signal when the installed CLI does not expose a
   tier's model (stale codex/pi install) — for **pi as well as codex**.

Out of scope: tier *names* and their order (unchanged), the claude-code backend,
raw-slug handling (unchanged: passes through; Sol/Terra remain reachable as raw
slugs).

## Current behavior

- `codex.py` and `pi.py` each hold two identical ladders (`_TIER_LAUNCH`,
  `_TIER_LAUNCH_PREFER_LUNA`); `_tier_launch()` picks one by reading the env var
  per call via `contracts.prefer_luna_tiers()`.
- Default: `high`=Sol@medium, `xhigh`=Sol@high, `max`=Sol@xhigh. Opt-in shifts
  the top three one step toward Luna.
- Codex: a tier whose slug is absent from `codex debug models` raises
  `BackendModelUnavailableError` ("Upgrade the CLI or check account access").
- Pi: a tier whose slug is absent from `pi --list-models` **silently** degrades
  to `("", thinking)` — pi then runs whatever its configured default model is.
  This is the "colleagues ran gpt-5.5 for months" failure mode.

## Why the ladders may now differ per backend

The previous rule "codex and pi must share a ladder so a tier means the same
thing" was about *capability*, not model identity. Luna needs 4–5x the tokens of
Sol for the same task. Codex caps context at 262k, so Luna@max cannot finish
complex tasks there; on pi the operator runs Luna with a 1M window (double cost
above 262k, still cheaper than Sol). The same tier therefore has to map to
*different* models on the two backends to keep its capability constant.

Benchmark (pass@1 / avg cost / steps), single suite, ±1–4%:

| Model @ effort | pass@1 | cost | steps |
| --- | --- | --- | --- |
| luna @ high | 44% | $0.16 | 49 |
| luna @ xhigh | 57% | $0.31 | 71 |
| luna @ max | 67% | $0.61 | 102 |
| sol @ medium | 61% | $1.42 | 31 |
| sol @ high | 69% | $2.66 | 37 |
| sol @ xhigh | 71% | $3.60 | 44 |
| sol @ max | 73% | $6.46 | 61 |
| astra @ low | 67% | $2.19 | 20 |
| astra @ medium | 73% | $4.38 | 26 |
| astra @ high | 73% | $5.72 | 27 |
| astra @ xhigh | 74% | $6.52 | 29 |
| astra @ max | 73% | $12.37 | 28 |

Astra dominates every Sol point above Sol@medium at equal-or-lower cost, and its
effort curve is flat above `medium` — so `astra @ high/xhigh/max` never appear on
a ladder. Sol@medium stays on codex `high` as the only 262k-safe point between
Luna and Astra (31 steps / 18k out-tokens vs Luna@max's 102 / 73k).

## Proposed ladders

| Tier | codex (262k ctx) | pi (Luna @ 1M) |
| --- | --- | --- |
| `cheapest` | luna @ medium | luna @ medium |
| `low` | luna @ high | luna @ high |
| `medium` | luna @ xhigh | luna @ xhigh |
| `high` | **sol @ medium** | **luna @ max** |
| `xhigh` | astra @ low | astra @ low |
| `max` | astra @ medium | astra @ medium |

Default tier stays `medium` on both.

## Design

### `contracts.py`
- Delete `PREFER_LUNA_ENV` and `prefer_luna_tiers()`. The env var, if still set,
  is ignored silently (decision: only the maintainer has it set).
- `BackendModelUnavailableError.__init__` gains `upgrade_hint: str = ""`,
  appended to the message. Keeps the existing positional signature.

### `codex.py`
- One `_TIER_LAUNCH` (table above). Remove `_TIER_LAUNCH_PREFER_LUNA` and
  `_tier_launch()`; call sites use `self._TIER_LAUNCH`.
- `_require_available` passes
  `upgrade_hint="Upgrade codex: npm install -g @openai/codex@latest"`.
- Rewrite the ladder rationale comment (Astra replaces Sol at the top; Sol@medium
  kept for the 262k reason; Sol/Terra reachable as raw slugs).

### `pi.py`
- One `_TIER_LAUNCH` (table above). Remove the opt-in ladder and `_tier_launch()`.
- `resolve_launch`: a **tier** whose slug is absent (discovery non-empty) now
  raises `BackendModelUnavailableError(slug, "pi", available,
  upgrade_hint="Upgrade pi: npm install -g @earendil-works/pi-coding-agent@latest
  (or add the model to your provider config)")`. **Raw slugs keep the soft
  fallback** — that path exists for operators logged into a non-OpenAI provider,
  and a raw slug is an explicit choice. A tier is a *capability request*, and
  serving it with an arbitrary default model is exactly the silent downgrade
  the codex path already refuses.
- Discovery empty → still skip validation (unchanged).

### `server_simple.py`
- `spawn_agent` docstring: tier list is now per backend where they differ
  (`high`: codex = Sol@medium, pi = Luna@max); state that both codex **and pi**
  error on a missing tier model, with the upgrade command in the error. The
  consuming agent only sees this docstring, so this is where the signal is
  documented.
- Comment at ~3252 ("Codex `high` -> Sol @ medium") stays true; no change.

### Docs
- `README.md` §"GPT capability tiers": replace the two-column env table with the
  per-backend table + the context-window rationale; drop the env-var prose;
  update "Unavailable-model behaviour" (pi tiers now error too; raw slugs
  soft-fall-back). Spawn-options row for `model`: fix the pi soft-fallback claim.
- `ADDING-A-BACKEND.md` §"Models & capability tiers": fix stale example (still
  shows five tiers with `ultra`), describe soft-fallback as raw-slug-only on pi.
- `docs/reference/agent-messaging-protocol.md:196`: mention pi and the hint.

## Files affected

- `src/claude_teams/backends/contracts.py`, `codex.py`, `pi.py`
- `src/claude_teams/server_simple.py` (docstring only)
- `tests/test_backends/test_codex.py`, `tests/test_backends/test_pi.py`
- `README.md`, `ADDING-A-BACKEND.md`, `docs/reference/agent-messaging-protocol.md`

## Risks

- **Behaviour change on pi**: a tier spawn that used to silently succeed on a
  stale install now fails. Intended; that is the requested signal. Anyone who
  relied on tiers while logged into a non-OpenAI provider must pass a raw slug
  or blank `model` instead.
- Astra effort `low` must be accepted by codex for `gpt-6-astra` — the benchmark
  exercised it, and codex validates model+effort at launch, so a mismatch would
  surface as a launch error, not a silent change.
- `tests/test_resume_session_dir.py` stubs `resolve_model` to `"gpt-5.6-sol"`;
  it does not go through the ladder, so unaffected (verify).

## Test cases (red first)

codex:
- tiers map to the new pairs (`high`→sol/medium, `xhigh`→astra/low, `max`→astra/medium).
- `supported_models()` order and `default_model()` unchanged.
- env var set to `"1"` has no effect (regression for the removal).
- missing astra with sol+luna present → `BackendModelUnavailableError` for
  `xhigh`/`max`; message contains `npm install -g @openai/codex@latest`.
- luna/sol tiers still resolve when astra absent.

pi:
- tiers map to the new pairs (`high`→luna/max, `xhigh`→astra/low, `max`→astra/medium).
- `PiBackend()._TIER_LAUNCH != CodexBackend()._TIER_LAUNCH` and they differ only
  at `high` (asserted key-by-key).
- env var `"1"` has no effect.
- tier with slug absent → raises, message names `@earendil-works/pi-coding-agent`.
- raw slug absent → still `("", effort)` (soft fallback preserved).
- discovery empty → tier resolves without error (unchanged).
- Delete `TestPiPreferLunaTiers`, the old soft-fallback-for-tier tests, and the
  `_prefer_luna`/`_default_tier_ladder` fixtures (both files).

Then full gates: `uv run ruff format --check .`, `uv run ruff check .`,
`uv run ty check`, `uv run pytest`.
