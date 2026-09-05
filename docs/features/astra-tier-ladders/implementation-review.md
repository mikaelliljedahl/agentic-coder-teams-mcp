# Implementation review — Astra on the GPT tier ladders

Reviewer: Claude Code (Opus), independent post-implementation reviewer.
Scope: full working-tree diff vs `main` (9 files), `plan.md`, `plan-review.md`,
`implementation.md`, plus a local re-run of all four quality gates.
Production code and tests were read only; nothing was modified.

**Verdict: approve.** No blocking findings. Five non-blocking findings, all with
recommended dispositions below; only NB1 is worth a decision before merge, and
"accept as is" is a defensible answer for it.

## Verification performed

| Check | Result |
| --- | --- |
| `uv run ruff format --check .` | 79 files already formatted |
| `uv run ruff check .` | All checks passed |
| `uv run ty check` | 1 diagnostic: `tests/test_join_team.py:730` `BaseContext.Process` — pre-existing, Windows-only, untouched by this change (matches `implementation.md`) |
| `uv run pytest` (focused) | `tests/test_backends/test_{codex,pi}.py` — 121 passed |
| `uv run pytest` (full) | 1390 passed, 2 skipped — **only with the harness's injected `AGENT_NAME`/`AGENT_SESSION_ID`/`AGENT_PARENT_NAME`/`WIN_AGENT_TEAMS_SESSION_DIR` unset**. With them set, `tests/test_join_team.py::test_restart_with_real_lead_binding` fails; that is an environment artifact of running the suite from inside a spawned agent, unrelated to this feature (`test_join_team.py` alone passes cleanly once the vars are cleared). `implementation.md` states this honestly. |

## Requirement-by-requirement

**Fixed ladder table.** `codex._TIER_LAUNCH` (`codex.py:121-127`) and
`pi._TIER_LAUNCH` (`pi.py:181-188`) match the plan's table exactly, including the
single intentional divergence at `high` (codex Sol @ medium, pi Luna @ max). The
Astra slug is `gpt-6-astra` in both ladders, in the tests, in `server_simple.py`,
README and ADDING-A-BACKEND.md — no variant slug slipped in anywhere (`grep` over
the tree confirms). Default tier remains `medium`; tier names and order are
unchanged and still asserted (`test_codex.py:72,85`, `test_pi.py:99,109`). The
new `test_backend_ladders_differ_only_at_high` pins the divergence key-by-key, as
the plan asked.

**Legacy env removal.** `PREFER_LUNA_ENV`, `prefer_luna_tiers()`, both
`_TIER_LAUNCH_PREFER_LUNA` ladders and both `_tier_launch()` methods are gone;
`import os` was dropped from `contracts.py` (plan-review F4) and `ruff` is clean,
so no dangling import or dangling re-export. Call sites use `self._TIER_LAUNCH`
directly. Regression guards exist on both backends
(`test_codex.py::test_legacy_luna_env_has_no_effect`,
`test_pi.py::test_legacy_luna_env_has_no_effect`). No reference to the variable
survives outside historical `docs/features/` records, which is correct.

**Error signal / upgrade hints.** `BackendModelUnavailableError.__init__` gains
`upgrade_hint: str = ""`, appended to the message, keeping the three-positional
signature (`contracts.py`). Codex passes
`npm install -g @openai/codex@latest`; pi passes the
`@earendil-works/pi-coding-agent@latest` hint plus the provider-config
alternative. Both hint strings are asserted via `match=` regexes, so a silently
dropped hint would go red.

**Pi tier hard-fail vs raw-slug soft fallback.** `pi.resolve_launch`
(`pi.py:222-260`) raises for a tier whose model is absent from a non-empty
catalog and keeps `("", effort)` for an unavailable raw slug. Covered by
`test_errors_when_tier_model_absent`,
`test_partial_catalog_errors_for_absent_tier` (the converted form plan-review F2
required, so the earlier partial-catalog decision is superseded rather than
silently dropped), `test_raw_slug_soft_fallback_when_unavailable`, and
`test_skips_validation_when_discovery_empty` — the last correctly asserts that
empty discovery resolves to the tier's slug, not to `""`.

**Provider-prefixed discovery (plan-review F3).** `_model_available` remains the
predicate and `_available_model_ids()` is used only to build the message, exactly
as specified. `test_provider_prefixed_catalog_entry_is_available` covers an
`openai-codex/gpt-6-astra` catalog. See NB1 for a nuance in how the fix was
widened.

**Test-catalog updates (F1/F2).** `gpt-6-astra` was added to pi's `_ALL_MODELS`
and to every codex `_stub_discovery` list that exercises `xhigh`/`max`, and the
obsolete `TestCodexPreferLunaTiers` / `TestPiPreferLunaTiers` classes and the
`_prefer_luna` / `_default_tier_ladder` fixtures are removed from both files. The
Pi build-command test that asserted the tier soft fallback is deleted, correctly,
since that behaviour no longer exists. The remaining build-command tests exercise
`gpt-6-astra` at `max` (`model='gpt-6-astra'`, `model_reasoning_effort=medium`).

**Effort validity.** `xhigh` = Astra @ `low` and `max` = Astra @ `medium` are
inside `codex._REASONING_EFFORT_SPEC.options` (`codex.py:135`), and pi's `high` =
Luna @ `max` is inside `_THINKING_OPTIONS`. The plan's open risk here is closed.

**API / backward compatibility.** Public tier vocabulary, `supported_models()`,
`default_model()`, `resolve_model()` passthrough, blank-model handling and
raw-slug passthrough (including `"ultra"` as a raw slug) are unchanged. The only
intended behaviour break is the pi tier hard-fail, which the plan lists as an
accepted risk. The new error parameter is keyword-defaulted, so existing
three-argument constructions still work.

**Docs.** README's ladder table, ADDING-A-BACKEND.md's tier example (now six
tiers with the correct slugs and the "Pi differs only at high" note), the
`spawn_agent` docstring — the only surface a consuming agent actually reads — and
the protocol doc all describe the per-backend ladders and both upgrade hints. The
protocol doc's re-derived citations were checked line-for-line: `codex.py:243` is
indeed `_require_available` and `pi.py:222` is `resolve_launch`, so plan-review
F5 is genuinely resolved rather than merely restated. The
`server_simple.py:3260` comment ("Codex `high` -> Sol @ medium") remains true.

## Non-blocking findings

### NB1 — `_model_available` now strips the provider prefix from *discovered* ids too, which can accept a foreign-provider model
`pi.py`: the check became
`any(candidate.split("/", 1)[-1] == bare for candidate in available)`.
Plan-review F3 only asked that a discovered `openai-codex/gpt-6-astra` not
false-error; this implementation also makes a catalog entry such as
`some-other-provider/gpt-6-astra` satisfy the `xhigh` tier, after which
`_build_model_args` qualifies the bare slug as `openai-codex/gpt-6-astra`
(`_PI_PROVIDER`) — a model that login may not serve. The failure is loud (pi
rejects it at launch), not a silent downgrade, and it needs a multi-provider
catalog with a same-named model, so severity is low.
**Recommended disposition: accept as is**, or optionally tighten the predicate to
match only entries whose prefix is empty or `_PI_PROVIDER`. Do not block the PR
on it.

### NB2 — Stale `pi --list-models` sample in the `pi.py` docstring
`pi.py:114`'s illustrative output lists `gpt-5.6-sol`/`luna`/`terra` but no
`gpt-6-astra`, so the in-code example no longer shows the catalog the top two
tiers now require. Cosmetic, but it is the example a future backend author reads.
**Recommended disposition: fix while here** (one added sample row) — or leave; no
functional impact.

### NB3 — README prose wraps mid-sentence around the Astra slug
README §"GPT capability tiers": "…between Luna and Astra. Here Astra / is
`gpt-6-astra`. Pi runs Luna / with a 1M window…" reads as an inserted clause with
odd line breaks. Content is correct.
**Recommended disposition: optional copy-edit** before the PR; not a blocker.

### NB4 — `reasoning_effort` row in the README spawn-options table still omits pi
The row lists codex and claude-code effort values but not pi's `--thinking`
options, even though pi's `high` tier now uses `max`. Pre-existing gap, adjacent
to text this change touched.
**Recommended disposition: out of scope** for this PR; note as a follow-up.

### NB5 — Codex upgrade-hint coverage is Astra-only
`test_missing_astra_tier_includes_upgrade_hint` asserts the hint text for
`xhigh`/`max`; the missing-Sol and missing-Luna cases assert only the exception
type. `_require_available` is a single shared path, so the hint cannot regress
for one tier and not another.
**Recommended disposition: no action.**

## Housekeeping (not a review finding)
`git status` shows untracked `.pi/` and `.pi-glla/`. Keep them out of the commit;
the stray `docs/features/astra-tier-ladders/feat/...` path flagged in the plan
review is gone.

## Verdict

**Approve.** The implementation matches the approved plan and resolves every
plan-review finding (F1–F6) with real code and real tests rather than prose. All
four gates are green apart from one pre-existing Windows-only `ty` diagnostic
that `implementation.md` reports accurately. None of NB1–NB5 needs to be fixed
before opening the PR.
