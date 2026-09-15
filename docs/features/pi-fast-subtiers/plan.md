# Feature: pi-only "fast" subtiers (`medium-fast`, `high-fast`)

## Scope

Add two **pi-only** capability subtiers to `PiBackend._TIER_LAUNCH`:

| New tier | Model | Thinking | Rationale (from the requester) |
| --- | --- | --- | --- |
| `medium-fast` | `gpt-5.6-terra` | `high` | Faster alternative around the `medium` capability point. |
| `high-fast` | `gpt-5.6-sol` | `medium` | 3–4x faster than Luna @ max at roughly the same benchmark. |

Codex is untouched: these are pi-only. `high-fast` deliberately equals codex's
existing `high` pair (Sol @ medium) — that is the 262k-safe fast bridge, offered
on pi as an explicit opt-out of the 1M-context Luna @ max default.

Ladder order becomes (cheapest first, subtiers interleaved at their capability
point):

```
cheapest, low, medium, medium-fast, high, high-fast, xhigh, max
```

`medium-fast` sits after `medium` and `high-fast` after `high`: each is the
"same capability, less latency" sibling of the tier it follows, so the primary
ladder still reads cheapest→max when the subtiers are ignored.

## Current behavior

- `PiBackend._TIER_LAUNCH` has exactly the six shared tier names; `CodexBackend`
  has the same six keys, differing only at `high`.
- `supported_models()` returns `list(_TIER_LAUNCH)` — insertion order is the
  advertised order, so placement in the dict literal is the placement callers see.
- `resolve_model(tier)` returns the tier's slug; `resolve_launch(tier, effort)`
  returns `(slug, thinking)` and hard-errors via `BackendModelUnavailableError`
  when live discovery is non-empty and lacks the slug.
- `tests/test_backends/test_pi.py::test_backend_ladders_differ_only_at_high`
  asserts `set(pi_ladder) == set(codex_ladder)` — this invariant is what the
  change intentionally breaks and must be rewritten, not deleted.

## Proposed design

1. `src/claude_teams/backends/pi.py`
   - Insert the two entries into `_TIER_LAUNCH` at their ladder positions.
   - Extend the ladder comment block to list them and say why they exist
     (latency-optimized siblings, pi-only).
   - `resolve_model` / `resolve_launch` / `_model_available` need **no** code
     change: they are driven entirely by `_TIER_LAUNCH` and the discovery set.
     `gpt-5.6-terra` and `gpt-5.6-sol` are both in pi's live catalog.
2. `src/claude_teams/server_simple.py` — `spawn_agent` docstring is the only
   contract the consuming agent reads (per CLAUDE.md). Add the two subtiers to
   the tier bullet list, marked pi-only, with the "when to pick it" framing
   (same capability, materially faster).
3. `README.md` — add the rows to the ladder table (codex column: "—", i.e. not
   offered) and a sentence on the latency rationale.
4. `ADDING-A-BACKEND.md` — the pi-differs-from-codex note becomes "Pi differs at
   `high` and adds two pi-only subtiers".

## Files affected

- `src/claude_teams/backends/pi.py`
- `src/claude_teams/server_simple.py` (docstring only)
- `README.md`, `ADDING-A-BACKEND.md`
- `tests/test_backends/test_pi.py`

## Risks

- **R1 — shared-key invariant test.** `test_backend_ladders_differ_only_at_high`
  fails by construction. Rewrite it to assert the *intended* relationship: pi's
  keys are a strict superset of codex's; the shared keys differ only at `high`;
  the extra keys are exactly `{medium-fast, high-fast}`.
- **R2 — hyphenated tier names.** `resolve_launch` lowercases and looks up the
  key, so `medium-fast` works; but a hyphen is new in this namespace. Verify no
  caller splits/normalizes a model string on `-` (a raw slug already contains
  hyphens, so this is expected to be safe — confirm by grep).
- **R3 — tier vs raw slug ambiguity.** None: neither name collides with a slug.
- **R4 — partial catalogs.** A login without `gpt-5.6-terra`/`gpt-5.6-sol` now
  hard-errors for these tiers. That matches existing tier semantics (never
  silently downgrade) and is the desired behavior.
- **R5 — docs drift.** The tier ladder is documented in four places; all four
  must be updated together or the MCP docstring and README disagree.

## Test cases (red first)

In `tests/test_backends/test_pi.py`:

1. `test_supported_models_are_tiers` — extended to the 8-name ordered list.
2. `test_resolve_model_tier_to_slug` — `medium-fast` → `gpt-5.6-terra`,
   `high-fast` → `gpt-5.6-sol`.
3. `test_tier_maps_to_model_and_thinking` — `("gpt-5.6-terra", "high")` and
   `("gpt-5.6-sol", "medium")`.
4. New: subtier owns its thinking level, ignoring a caller `reasoning_effort`.
5. New: `medium-fast` / `high-fast` raise `BackendModelUnavailableError` naming
   the missing slug when discovery lacks terra / sol respectively.
6. New: command-level — each subtier's resolved pair reaches argv as
   `--model openai-codex/<slug> --thinking <level>`.
7. Rewrite `test_backend_ladders_differ_only_at_high` per R1, including an
   explicit assertion that codex does **not** expose the subtiers.
8. `tests/test_backends/test_codex.py` — unchanged; its six-tier list assertion
   is now also the regression guard that the subtiers stayed pi-only.

## Validation

```
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

## Plan-review dispositions (see `plan-review.md`, VERDICT: CHANGES REQUIRED)

- **F1 (required) — accepted.** Drop the unqualified "same capability" framing.
  The fast subtiers trade **context window** for latency: Terra 272K and Sol
  372K against Luna's 1M. Every contract surface (spawn_agent docstring, README,
  pi.py ladder comment) must name the window explicitly and warn against
  choosing them for context-heavy work. Wording becomes "comparable benchmark
  quality, 3–4x faster, much smaller context window".
- **F2 (required) — accepted.** `CLAUDE.md` added to files affected; its tier
  summary (`low/medium/high/xhigh/ultra`, "Pi soft-falls-back") is already stale
  and is corrected to the six shared tiers + two pi-only subtiers, hard-error
  for tier models, soft fallback only for raw slugs.
  `docs/reference/agent-messaging-protocol.md` needs no change.
- **F3 (required) — accepted.** `tests/test_tool_descriptions.py` added: assert
  the *registered* `spawn_agent` description carries both hyphenated names,
  their pairs, the pi-only marking and the context-window caveat.
- **F4 (nice-to-have) — accepted.** Cross-backend test also asserts that pi's
  keys filtered to codex's key set equal `list(codex_ladder)` (shared order
  preserved).
- **F5 (nice-to-have) — accepted.** Add a server-path regression proving a
  fast-tier spawn persists the concrete slug/effort and that the resume request
  carries them unchanged (follow-up deliberately does not re-resolve).
