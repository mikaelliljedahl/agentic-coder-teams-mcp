# Plan review — Astra on the GPT tier ladders (independent, opposite-model)

Reviewer: Claude (Opus). Scope: `docs/features/astra-tier-ladders/plan.md` only.
Inspected: `backends/{contracts,codex,pi}.py`, `server_simple.py`, `tests/test_backends/test_{codex,pi}.py`,
`README.md`, `ADDING-A-BACKEND.md`, `docs/reference/agent-messaging-protocol.md`.

Verdict: **approve with required changes** — the design is coherent and the rationale for
per-backend ladders is sound. Six findings must be dispositioned before coding; four of them
would otherwise turn the whole existing test suite red or leave stale docs.

## Blocking findings

### F1 — Test model catalogs contain no `gpt-6-astra`; most existing tier tests will fail
`tests/test_backends/test_pi.py:17-25` defines `_ALL_MODELS` (luna/sol/terra/5.4/5.5/spark) and the
autouse-ish `_models` fixture (`:86-93`) defaults to it. `tests/test_backends/test_codex.py:78-88`
stubs discovery per test, and the tier tests pass `["gpt-5.6-luna", "gpt-5.6-sol"]`
(`test_codex.py:135`, `:152`, `:166`, `:190`, `:196`). Under the new ladder, `xhigh`/`max` resolve to
`gpt-6-astra`, so on codex they raise `BackendModelUnavailableError` and on pi (with the new tier
hard-fail) they raise too — including `TestPiBuildCommand` cases that go through `resolve_launch`.
The plan's "Files affected"/"Test cases" sections never mention updating these catalogs.
**Disposition: required.** Add `gpt-6-astra` to `_ALL_MODELS` and to every codex `_stub_discovery`
list used by tier tests, and say so explicitly in the plan's test section.

### F2 — pi's existing tier soft-fallback tests are contradicted but not all named for deletion
The plan says "delete `TestPiPreferLunaTiers`, the old soft-fallback-for-tier tests". Concretely
that is `test_pi.py:165-169` (`test_soft_fallback_when_tier_model_absent`) **and**
`test_pi.py:171-182` (`test_partial_catalog_drops_model_rather_than_substituting`), the latter added
as the disposition of a prior review finding
(`docs/features/openai-pricing-tier-remap/implementation-review-1.md:23-25`). Deleting it silently
reverses an earlier accepted decision.
**Disposition: required.** Name both tests; convert the partial-catalog test into the *raises* form
(catalog `["gpt-5.6-sol"]` → `low`/`medium` now raise, naming luna) so the partial-catalog case
stays covered, and note in the plan that this supersedes that earlier finding.

### F3 — pi hard-fail must reuse the provider-prefix-tolerant availability check
`pi._model_available` (`pi.py:284-294`) strips a `provider/` prefix before comparing and returns
True on empty discovery; `_available_model_ids` (`:296-302`) returns the raw list. The plan writes
the new raise as `BackendModelUnavailableError(slug, "pi", available)` without saying where
`available` comes from or that the bare-name comparison must be preserved. Implementing it as a
copy of codex's `_require_available` (plain `slug not in available`) would false-error on any
catalog that lists ids as `openai-codex/gpt-6-astra`.
**Disposition: required.** Specify: keep `_model_available` as the predicate, and use
`_available_model_ids()` only to build the message. Add a test with a provider-prefixed catalog
entry that must *not* raise.

### F4 — Removing `prefer_luna_tiers()` leaves `os` unused in `contracts.py`
`contracts.py:29` is the only `os.` use in that module. Deleting the function without dropping the
`import os` fails `ruff check` (F401) — a gate the plan claims to run but does not foresee.
**Disposition: required.** Add "remove the now-unused `import os`" to the `contracts.py` bullet.

## Non-blocking but should be fixed in the plan

### F5 — Stale line-number citation in the protocol doc
`docs/reference/agent-messaging-protocol.md:194-199` cites `codex.py:237-246` for
`BackendModelUnavailableError`. Deleting `_TIER_LAUNCH_PREFER_LUNA` and `_tier_launch()` shifts
those lines by ~60. The plan only says "mention pi and the hint".
**Disposition: fix while editing** — re-derive the citation, and add the pi `resolve_launch`
citation alongside it.

### F6 — `openai-codex/gpt-6-astra` provider qualification is assumed, not verified
`pi._build_model_args` (`pi.py:466-478`) qualifies a bare slug as `openai-codex/<slug>`
(`_PI_PROVIDER`, `pi.py:35`). Whether pi exposes a GPT-6 model under that same provider id is
unverified in the plan; if it is served under a different provider, tier spawns on pi break at
launch (loudly, not silently — so severity is low). The plan's risk list covers the codex
`astra @ low` effort question but not this.
**Disposition: verify before implementing** (one `pi --list-models` run) and record the observed id
in the plan's risks; if the provider differs, the fix is a per-slug provider map, which would widen
scope.

## Confirmed-correct points (no action)
- Tier efforts used by the new ladder (`low`, `medium`, `max`) are already inside
  `codex._REASONING_EFFORT_SPEC.options` (`codex.py:161`) and `pi._THINKING_OPTIONS` (`pi.py:222-224`).
- `server_simple.py:3252-3254`'s "Codex `high` -> Sol @ medium" comment does stay true, as claimed.
- `server_simple.py:3933-3937` (follow-up reuses the concrete resolved model, never the tier name)
  is unaffected by the ladder change.
- `tests/test_resume_session_dir.py` stubs a raw slug and does not traverse the ladder — the plan's
  "verify" note is correct; nothing to change there.
- `tests/test_tool_descriptions.py` asserts nothing about tier text, so the `spawn_agent` docstring
  edit needs no test update.
- Ignoring a still-set `WIN_AGENT_TEAMS_GPT_PREFER_LUNA_MODEL_TIERS` silently is acceptable given a
  single known holder; the planned regression test (env `"1"` has no effect) is the right guard.

## Housekeeping (outside the feature, worth a word)
`git status` shows an untracked stray directory
`docs/features/astra-tier-ladders/feat/astra-tier-ladders` plus `.pi/` and `.pi-glla/`. Do not
commit them; consider removing the stray path before the PR.

## Recommended plan edits, in order
1. Test section: add the `gpt-6-astra` catalog updates (F1) and the explicit list of pi tests to
   delete/convert (F2).
2. `pi.py` design bullet: state the `_model_available`-based check and prefix tolerance (F3).
3. `contracts.py` bullet: drop `import os` (F4).
4. Docs bullet: re-derive the protocol-doc line citation (F5).
5. Risks: add the pi provider-id assumption for astra (F6).

## Implementer disposition

All findings are resolved as follows before implementation:

- **F1:** Update every tier-test discovery catalog to include `gpt-6-astra`, including
  the shared Pi catalog and each Codex catalog that exercises `xhigh`/`max`.
- **F2:** Remove the obsolete Pi tier soft-fallback tests named in the review, and
  replace the partial-catalog case with a hard-failure assertion so the behavior change
  remains covered.
- **F3:** Keep Pi's `_model_available` predicate (including empty-discovery skip and
  provider-prefix stripping) and use the raw discovered IDs only for the error detail;
  add a provider-prefixed catalog regression test.
- **F4:** Remove the unused `os` import along with the deleted contracts environment
  helper.
- **F5:** Refresh the protocol-document source citations after the implementation and
  mention both Codex and Pi tier failures plus the upgrade hints.
- **F6:** Verified before implementation with `pi --list-models`: the installed CLI
  exposes `openai-codex gpt-6-astra`, so the existing bare-slug qualification is
  correct; no provider map is needed.
