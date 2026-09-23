# Plan: raise the GPT-6 tier ladder one effort step

## Scope

PR #66 put GPT-6 on both tier ladders at the *same* efforts as GPT-5.6,
explicitly "not yet re-benchmarked". In use, GPT-6 needs more effort per tier
than GPT-5.6 did, and published benchmarks agree for Sol (DeepSWE v1.1:
Sol @ medium 56.6 % vs Sol @ high 65.3 %). This change re-ladders both the
`codex` and `pi` backends. Ladder agreed with the user on 2026-09-23:

| Tier | codex before | pi before | **codex after** | **pi after** |
| --- | --- | --- | --- | --- |
| `cheapest` | Luna @ medium | Luna @ medium | **Luna @ high** | **Luna @ high** |
| `low` | Luna @ high | Luna @ high | **Luna @ xhigh** | **Luna @ xhigh** |
| `medium` | Luna @ xhigh | Luna @ xhigh | **Luna @ max** | **Luna @ max** |
| `medium-fast` | — | Sol @ low | — | **Sol @ medium** |
| `high` | Sol @ medium | Luna @ max | **Sol @ high** | **Sol @ high** |
| `high-fast` | — | Sol @ medium | — | **removed** |
| `xhigh` | Astra @ low | Astra @ low | **Sol @ xhigh** | **Sol @ xhigh** |
| `max` | Astra @ medium | Astra @ medium | Astra @ medium | Astra @ medium |

Rationale for the non-mechanical rows:

- **Luna has no step above `max`**, so with `medium` = Luna @ max, pi `high`
  (formerly Luna @ max) would duplicate `medium`. Pi `high` moves to Sol @ high,
  and the two backends' six shared tiers become identical.
- **`high-fast` is removed.** Its reason to exist was "Sol beside a Luna
  `high`"; now `high` is Sol itself. `medium-fast` stays as the pi-only Sol
  sibling of a Luna `medium`.
- **`xhigh` → Sol @ xhigh.** Astra costs 5x Sol per token. DeepSWE v1.1:
  Sol @ xhigh 66.6 % / $1.00 per task vs Astra @ low 67.0 % / $1.60;
  AutomationBench: 33.2 % / $0.27 vs 30.3 % / $1.08; OSWorld 2.0: 60.5 % / $2.21
  vs 62.2 % / $2.55. Sol @ max is not used (+2.2 pts over xhigh on DeepSWE for
  +174 % cost, and lower than xhigh on AutomationBench).
- **`max` stays Astra @ medium** (DeepSWE 72.8 %), the clear capability jump.
- Luna per-effort data is not published; the Luna bump is operator judgement.

The ladders become identical on the six shared tiers; pi adds only
`medium-fast`.

## Current behavior

- `src/claude_teams/backends/codex.py:128` `_TIER_LAUNCH` (six tiers).
- `src/claude_teams/backends/pi.py:197` `_TIER_LAUNCH` (six + `medium-fast`,
  `high-fast`); `supported_models()` returns its keys in order.
- `PiBackend.resolve_launch` (`pi.py:240`): a key not in `_TIER_LAUNCH` is a
  raw slug; an unavailable raw slug **soft-falls to pi's default model**.
  Consequence: after removal, a caller still sending `high-fast` would silently
  run on pi's default (non-empty discovery) or launch `--model high-fast`
  (empty discovery). Tiers must never silently downgrade.
- Codex's comment block explains `high` as the 272k-context-safe bridge because
  "Luna @ max cannot finish complex tasks" in codex's 272k default window.

## Design

1. **Tables.** Update both `_TIER_LAUNCH` dicts to the "after" columns; drop
   `high-fast` from pi. Effort values are all valid (`high`/`xhigh`/`max` are in
   codex `_REASONING_EFFORT_SPEC.options` and pi `_THINKING_OPTIONS`).
2. **Retired tier guard (pi).** Add `_RETIRED_TIERS: ClassVar[dict[str, str]] =
   {"high-fast": "high"}`. Both `resolve_launch` and `resolve_model` raise a new
   `RetiredTierError(UnsupportedBackendModelError)` (in `contracts.py`; message
   names the removed tier and its replacement `high`, plus the supported list)
   for a retired name, before the raw-slug path, so a stale caller fails loudly
   instead of degrading. `spawn_agent` does not catch it, so the tool call
   fails — the intended outcome. Scope: the guard covers new spawns only;
   resume reuses the persisted concrete pair verbatim and never re-resolves a
   tier (records created by `spawn_agent` never store a tier name).
3. **Comments.** Rewrite both ladder comment blocks: new ladder, "raised one
   effort step for GPT-6", the Astra 5x cost reason for Sol @ xhigh, and drop
   the now-false "Astra dominated Sol above medium" and "codex does not expose
   the fast subtiers because its high is already Sol @ medium" sentences.
   Keep the 272k note but update it: codex `medium` is now Luna @ max inside the
   272k default window (accepted risk, see below).
4. **Consumer contract.** `server_simple.py` `spawn_agent` docstring (the only
   thing the consuming agent reads): new ladder, only `medium-fast` as pi-only
   subtier, "ladders are identical on the six shared tiers", `high-fast`
   removed. Also the inline comment at `server_simple.py:3296` citing
   "Codex `high` -> Sol @ medium".
5. **Prose docs.** `README.md` ladder table + prose (including the "tier
   names and order are stable" claim → names are stable except the retired
   `high-fast`; and the pi 0.87.0 note → only `max` works there now, since
   `xhigh` moved to Sol), `ADDING-A-BACKEND.md`
   example and prose, `CLAUDE.md` "Model tiers" bullet.
6. Running agents are unaffected: tiers resolve at spawn and the concrete pair
   is persisted in `agents.json` and reused verbatim on resume.

## Files affected

- `src/claude_teams/backends/codex.py`, `src/claude_teams/backends/pi.py`,
  `src/claude_teams/server_simple.py`
- `README.md`, `ADDING-A-BACKEND.md`, `CLAUDE.md`
- `tests/test_backends/test_codex.py`, `tests/test_backends/test_pi.py`,
  `tests/test_pi_fast_subtier_resume.py`, `tests/test_tool_descriptions.py`

## Test cases (red first)

1. Codex `resolve_model`/`resolve_launch` for every tier return the new pairs
   (full discovery, partial discovery, caller effort ignored). Missing-model
   cases updated: `xhigh` now requires `gpt-6-sol`, only `max` requires Astra.
2. Pi `supported_models()` == `[cheapest, low, medium, medium-fast, high,
   xhigh, max]`; extra-over-codex == `{"medium-fast"}`; shared six tiers equal
   codex's pairs exactly (strengthens the old "differ only at high" test).
3. Pi `resolve_launch` / argv table pins new pairs; `high` now needs
   `gpt-6-sol`.
4. Pi `resolve_launch("high-fast", None)` and `resolve_model("high-fast")`
   raise `RetiredTierError` (an `UnsupportedBackendModelError`) naming `high`,
   for both empty and non-empty discovery; it is not in `supported_models()`.
   Pi `resolve_model("high")` / `("xhigh")` == `gpt-6-sol`.
4b. Pi stale-catalog matrix (pi 0.87.0: Astra, no GPT-6 Sol/Luna): `xhigh`
   and `high` now fail with the Sol-unavailable error; only `max` resolves.
   Partial-catalog assertions updated for the new model owners.
5. `test_pi_fast_subtier_resume.py` pins `medium-fast` → Sol @ medium only.
6. Tool description: `medium-fast` line pins `gpt-6-sol` @ medium; `high-fast`
   absent; codex/pi `high` = Sol @ high and `xhigh` = Sol @ xhigh stated.

## Risks

- **Codex `medium` (the default tier) = Luna @ max in a 272k window.** The old
  ladder comment says Luna @ max cannot finish complex tasks there. Accepted by
  the user; flagged in the comment and README so it can be revisited.
- Higher effort costs more tokens/latency per spawn — intended.
- Removing `high-fast` breaks callers using it; mitigated by the loud retired
  tier error pointing to `high`.
- Benchmarks are third-party summaries of vendor numbers; Luna is unmeasured.

## Validation

`uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
`uv run pytest`.

## Plan review disposition (plan-review.md, Codex)

1. [MAJOR] README "tier names and order are stable" — **accepted**, design 5.
2. [MAJOR] `resolve_model` guard untested — **accepted**, test 4.
3. [MAJOR] pi 0.87.0 note / stale-catalog `xhigh` — **accepted**, design 5 + test 4b.
4. [MINOR] use existing invalid-model exception — **accepted**:
   `RetiredTierError` subclasses `UnsupportedBackendModelError`.
   Persisted-record scope clarified in design 2.
