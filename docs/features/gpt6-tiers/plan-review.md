# Independent plan review: GPT-6 tier remap

The production-file inventory is complete for the requested scope. An exhaustive
search of `src/`, `tests/`, `README.md`, and `ADDING-A-BACKEND.md` found GPT-5.6
references only in the two tier tables and Pi discovery example
(`src/claude_teams/backends/codex.py:123-126`,
`src/claude_teams/backends/pi.py:118,196-201`), the consuming-agent docstring
(`src/claude_teams/server_simple.py:3188-3189`), the backend guide
(`ADDING-A-BACKEND.md:97-106`), and the five test files named by the plan. The
remaining test references are raw-slug/fixture cases, which the plan correctly
says remain supported. The current README tier section names families rather
than 5.6 slugs, but its table and rationale still need the planned semantic
update (`README.md:683-705`).

1. **F1 — major — Pi `high` is specified as a 1M tier even though the product does not establish that invariant.**

   **Evidence:** D2 admits that stock Pi exposes GPT-6 Luna at 272K and that 1M
   depends on an operator editing `~/.pi/agent/models.json`
   (`docs/features/gpt6-tiers/plan.md:66-73,165-169`). The backend only resolves
   a tier to a slug/effort and checks whether that slug appears in discovery
   (`src/claude_teams/backends/pi.py:261-273`); its availability predicate
   compares model names only (`src/claude_teams/backends/pi.py:275-287`). It
   neither installs, validates, nor reports the context override. The current
   public documentation nevertheless presents `pi (Luna @ 1M)` as a backend
   property (`README.md:683`) and the backend comment says the ladder is tuned
   for Pi's 1M context (`src/claude_teams/backends/pi.py:171-188`). On this
   machine, `~/.pi/agent/models.json:5-8` overrides only `gpt-5.6-luna`; there is
   no GPT-6 Luna override. Thus the proposed code would normally launch
   `gpt-6-luna @ max` at 272K while documentation and tier rationale imply 1M.
   A pricing tier above 272K shows that larger requests may be billable, but it
   does not prove that this repository has established or enforced a 1M limit.

   **Recommended fix:** Resolve the open question before approval. Either define
   `high` against stock behavior (and choose Sol if the long-context bridge is a
   requirement), or explicitly define `high` as Luna @ max with a native 272K
   window and describe the 1M override as optional operator configuration. In
   the latter case, remove `1M` from the fixed ladder/table claims and document
   the exact override plus its unverified/service-dependent status in a clearly
   optional note. Do not make correctness of a shipped tier depend on an
   unmanaged file without runtime validation.

2. **F2 — major — D1 claims capability preservation that its own evidence does not support.**

   **Evidence:** The plan says there are no GPT-6 Sol/Luna benchmarks in the
   repository (`docs/features/gpt6-tiers/plan.md:52-53`) and correctly calls
   re-tuning a guess (`docs/features/gpt6-tiers/plan.md:59-62`), but immediately
   asserts the replacements are "at least as capable per effort" and therefore
   "no weaker" (`docs/features/gpt6-tiers/plan.md:62-64`). Model availability,
   accepted effort names, and lower token prices do not establish equivalent
   tier capability. D3 separately acknowledges that Sol @ low is unbenchmarked
   (`docs/features/gpt6-tiers/plan.md:75-84,128`), so the same uncertainty must
   apply to the straight Luna/Sol substitutions. The old comments explicitly
   describe an ascending cost/quality ladder (`src/claude_teams/backends/codex.py:105-121`,
   `src/claude_teams/backends/pi.py:171-194`), making an unqualified preservation
   claim material to the design.

   **Recommended fix:** Recast D1 as an intentional unbenchmarked migration made
   to satisfy the GPT-6-only requirement while preserving effort labels, not as
   a demonstrated non-regression. Add possible capability/order regression for
   every replaced point to Risks and name a follow-up benchmark/acceptance
   criterion. D3 (`gpt-6-sol @ low`) is otherwise a coherent provisional choice:
   it removes Terra, retains a one-step fast ladder, and is already identified
   as unbenchmarked.

3. **F3 — minor — D4 describes 272K as an absolute Codex cap without accounting for the discovered maximum.**

   **Evidence:** The current source still says 262K
   (`src/claude_teams/backends/codex.py:115-119`), while the plan updates this to
   "caps at 272k" (`docs/features/gpt6-tiers/plan.md:86-88`). On the requested
   local check, Codex 0.155.1 reports both `context_window: 272000` and
   `max_context_window: 872000` for GPT-6 Luna, Sol, and Astra. Therefore 272K is
   the advertised/default context, but the word "caps" is stronger than the
   discovery evidence.

   **Recommended fix:** State that this integration uses or observes a 272K
   default/effective window, and explain whether/why Codex's 872K maximum is not
   enabled or relied upon. Update all stale 262K wording in the source comment,
   tool docstring, README, and backend guide as part of the planned prose pass.
   Keeping Codex `high = gpt-6-sol @ medium` is reasonable once that rationale is
   phrased accurately.

4. **F4 — minor — The stale-Pi test matrix does not cover the full D5 claim and the claim needs its discovery condition.**

   **Evidence:** D5 says Pi 0.87.0 will fail every tier except `xhigh`/`max`
   (`docs/features/gpt6-tiers/plan.md:90-96`), but the proposed stale-catalog test
   names only `medium`, `high`, `medium-fast`, and `high-fast`
   (`docs/features/gpt6-tiers/plan.md:148-152`), omitting `cheapest` and `low`.
   Moreover, Pi deliberately treats an empty discovery result as available
   (`src/claude_teams/backends/pi.py:275-285`), so the hard failure occurs only
   when 0.87.0 successfully returns its non-empty catalog. The installed 0.87.0
   does return that catalog, so the intended behavior is valid under that
   condition.

   **Recommended fix:** Parameterize the stale-catalog test across all six
   affected tiers (`cheapest`, `low`, `medium`, `medium-fast`, `high`,
   `high-fast`), checking the expected missing slug and `0.87.1` hint. Qualify
   D5 and user-facing wording with "when live discovery succeeds/is non-empty,"
   consistent with the resolver contract. The minimum-version hint itself is a
   sensible actionable change.

5. **F5 — minor — The plan does not explicitly remove all obsolete 3–4x latency claims after changing the fast model.**

   **Evidence:** D3 says only that the "docstring wording" will be softened
   (`docs/features/gpt6-tiers/plan.md:81-84`), but the measured 5.6-era multiplier
   appears in four places: the Pi ladder comment
   (`src/claude_teams/backends/pi.py:190-194`), the consuming-agent docstring
   (`src/claude_teams/server_simple.py:3186-3191`), README
   (`README.md:694-698`), and backend guide (`ADDING-A-BACKEND.md:109-113`). The
   files list mentions all four files, but describes the guide change only as a
   tier-example update (`docs/features/gpt6-tiers/plan.md:113-121`). Retaining any
   multiplier would incorrectly transfer a 5.6 Terra/Sol measurement to GPT-6
   Sol @ low/medium.

   **Recommended fix:** Add an explicit implementation item to replace the
   multiplier in all four locations with carefully qualified wording (for
   example, "lower-latency option; not yet benchmarked against its neighbor").
   Add a scoped assertion or final grep that no `3-4x`/`3–4x` claim remains in
   those surfaces.

6. **F6 — nit — `tests/test_resume_session_dir.py` should not be changed solely to erase an unrelated raw slug.**

   **Evidence:** Its fake backend and persisted record use `gpt-5.6-sol` only as
   opaque resume data (`tests/test_resume_session_dir.py:23-55`); the test does
   not pin a tier. The plan calls this a cosmetic update
   (`docs/features/gpt6-tiers/plan.md:119-121`) while also declaring raw 5.6
   slugs supported and out of the tier remap (`docs/features/gpt6-tiers/plan.md:14-15,81-82`).
   Other raw-slug tests must legitimately retain 5.6 references, so changing
   this fixture does not achieve a clean global grep either.

   **Recommended fix:** Leave this test unchanged unless there is a separate
   requirement to modernize arbitrary fixtures. Treat the remaining 5.6 grep
   hits as intentional raw-slug coverage and document that disposition.

VERDICT: CHANGES REQUESTED

---

## Dispositions (implementer: Claude Opus 5.5)

- **F1 — accepted.** The ladder is now defined against stock pi: `high` =
  `gpt-6-luna @ max` with pi's native **272K** window. The "1M" claim is removed
  from the fixed ladder (pi.py comment, spawn_agent docstring, README table
  header, ADDING-A-BACKEND). The 1M `models.json` `contextWindow` override is
  documented as optional, service-dependent operator configuration. Whether pi
  `high` should instead move to `gpt-6-sol @ medium` remains open question 1 for
  the user (the lead's instruction: flag rather than guess).
- **F2 — accepted.** D1 recast as an intentional, unbenchmarked migration that
  preserves effort labels; per-point capability/order regression added to Risks
  with a follow-up benchmark criterion.
- **F3 — accepted.** Verified: codex 0.155.1 reports `context_window` 272000 and
  `max_context_window` 872000 for gpt-6-luna/sol/astra. Wording changed to
  "272k default window" (the integration does not raise codex's context setting),
  and stale "262k" wording is updated everywhere in the prose pass.
- **F4 — accepted.** Stale-catalog test parameterized over all six affected
  tiers; D5 and user-facing wording qualified with "when live discovery is
  non-empty".
- **F5 — accepted.** The "3-4x" multiplier is removed from pi.py, the
  spawn_agent docstring, README and ADDING-A-BACKEND; a docstring test asserts
  no `3-4x` remains in the registered description.
- **F6 — accepted.** `tests/test_resume_session_dir.py` left unchanged; remaining
  5.6 grep hits are intentional raw-slug coverage.
