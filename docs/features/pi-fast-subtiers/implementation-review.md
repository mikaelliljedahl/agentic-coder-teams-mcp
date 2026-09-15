# Independent post-implementation review

1. **F1 — required: The consuming-agent contract still overstates interchangeability when the adapter cannot know whether the fast models can handle the same input.**

   Evidence: `src/claude_teams/server_simple.py:3172-3177` calls each fast subtier the “same job” as its sibling and tells callers to choose solely according to turnaround versus “the last increment of depth.” `README.md:678-682` and `ADDING-A-BACKEND.md:111-113` give the same speed/benchmark framing. That is not an honest selection rule under the owner's stated constraint: per-model context capacity is provider/config state which this MCP neither discovers nor represents. The code correctly makes no behavioral promise about it (`src/claude_teams/backends/pi.py:192-194`), but the user-facing contract still implies that latency/depth is the only material trade-off. A context-heavy task may not be the “same job” for the configured Terra or Sol endpoint even when its benchmark quality is comparable.

   Recommendation: keep the model/effort pairs and latency/benchmark claims, but replace “same job” and the latency-versus-depth-only recommendation with a provider-neutral caveat: context capacity is determined by the installed Pi provider/configuration, is not encoded by these tiers, and should be checked before choosing a fast subtier for context-heavy work. Do not hard-code any context-window numbers.

2. **F2 — required: The documentation contradicts both the new tier set and its own provider/config-state disclaimer.**

   Evidence: after adding two Pi-only keys, the registered `spawn_agent` description still says “The codex and pi ladders differ only at `high`” (`src/claude_teams/server_simple.py:3178-3180`); they now also differ because Pi has `medium-fast` and `high-fast`. Separately, the Pi ladder comment says the adapter deliberately makes no claim about per-model context limits (`src/claude_teams/backends/pi.py:192-194`) while the same comment describes fixed context properties at `src/claude_teams/backends/pi.py:174-175`, `src/claude_teams/backends/pi.py:182`, and `src/claude_teams/backends/pi.py:186-187`. The same fixed-context framing remains in `README.md:664-667`, `README.md:684-686`, and `ADDING-A-BACKEND.md:109-110`. Thus the spawn docstring, README, backend guide, and Pi comment are not mutually self-consistent under the stated provider/config constraint.

   Recommendation: say that the **six shared tiers' mappings** differ only at `high`, then separately identify the two Pi-only keys. Replace or explicitly qualify fixed context claims in contract surfaces as deployment-specific provider/config facts so they agree with the disclaimer; no model-specific window numbers need to be added.

3. **F3 — required: The registered-tool-description test does not pin the documented model/effort pairs or the provider/config caveat.**

   Evidence: `tests/test_tool_descriptions.py:190-200` independently searches for the two tier names, two model slugs, `pi only`, and `faster`. It never asserts `medium-fast -> gpt-5.6-terra @ high` or `high-fast -> gpt-5.6-sol @ medium`, despite plan-review F3 explicitly requiring the registered contract's pairs to be protected. The test would pass if the efforts were swapped, omitted, or attached to the wrong tier. It also permits the incomplete selection wording described in F1 because it checks no provider/config-dependent context caveat. The backend mapping tests do not close this contract gap: they prove the code table, not that the actual FastMCP description accurately explains that table.

   Recommendation: assert each tier/model/effort association as one localized phrase or parsed line in the registered description, plus the Pi-only qualification and a provider/config-dependent context-capacity caveat. Continue reading the registered FastMCP `Tool.description`, as the test already correctly does.

4. **F4 — nice-to-have: The claimed spawn-persistence regression never executes the spawn persistence path.**

   Evidence: `tests/test_pi_fast_subtier_resume.py:38-47` calls `PiBackend.resolve_launch()` directly and asserts its return value; it never calls `server_simple.spawn_agent`, never observes the `SpawnRequest` passed to `PiBackend.spawn`, and never reads an `agents.json` record. `tests/test_pi_fast_subtier_resume.py:54-79` then hand-constructs a record that already contains the desired concrete slug/effort before exercising `_build_resume_request`. Both tests would still pass if `spawn_agent` accidentally persisted the original tier name or the wrong effort at `src/claude_teams/server_simple.py:3329-3345`. The production code itself is correct: it resolves at `src/claude_teams/server_simple.py:3264-3269`, persists the concrete values, and reuses them at `src/claude_teams/server_simple.py:3943-3952`; the gap is that plan-review F5 is not actually regression-tested as claimed.

   Recommendation: replace the first helper-level test with a focused server-path test that invokes `spawn_agent(model="medium-fast", backend="pi")` using a fake launch boundary, then reads the saved agent record and feeds that same record to `_build_resume_request`. Assert the concrete slug and bundled effort at the spawn request, persisted-record, and resume-request boundaries.

The backend table itself matches the requested eight-tier Pi order and both concrete launch pairs (`src/claude_teams/backends/pi.py:195-204`). Hyphenated names are safe in the current path because resolution strips whitespace and performs a direct lowercase dictionary lookup (`src/claude_teams/backends/pi.py:258-273`); no production caller was found that splits tier names on `-`. F2's `CLAUDE.md` correction is present (`CLAUDE.md:111-116`), and F4's shared-order invariant is correctly and non-vacuously covered by the exact Pi order plus the filtered cross-backend order assertions (`tests/test_backends/test_pi.py:99-109`, `tests/test_backends/test_pi.py:163-167`). Focused feature/static validation passed (148 tests, Ruff, and ty). The full suite reached 1513 passed and 4 skipped but also reproduced one unrelated pre-existing PID-sensitive failure in `tests/test_follow_up_delivery.py:228-249` (`agent_busy` rather than `resume_not_confirmed`); the staged feature changes do not touch that path.

VERDICT: CHANGES REQUIRED

## Round 2 verification

1. **F1 — RESOLVED.**

   Evidence: the registered `spawn_agent` contract now says the fast models benchmark close to the neighbouring tier but are “a different model, not a drop-in ... on every input” (`src/claude_teams/server_simple.py:3172-3177`). The same bounded framing appears in `README.md:678-682`, and `src/claude_teams/backends/pi.py:189-193` expressly avoids claiming interchangeability. `ADDING-A-BACKEND.md:111-113` limits itself to latency and benchmark proximity and no longer calls the models capability-equivalent. This is honest without a context-capacity caveat: it makes no context assertion in either direction and directly warns callers that benchmark proximity does not guarantee suitability for every input.

2. **F2 — RESOLVED.**

   Evidence: the formerly false whole-ladder statement is now scoped to “the six tiers both backends share” before identifying `high` as their sole shared-tier mapping difference (`src/claude_teams/server_simple.py:3178-3181`). The newly added Pi comment no longer contains the disclaimer that contradicted the adjacent pre-existing context prose (`src/claude_teams/backends/pi.py:170-193`). The resulting feature wording consistently distinguishes the six shared tiers from the two Pi-only additions (`src/claude_teams/backends/pi.py:170-172`, `ADDING-A-BACKEND.md:89-106`).

3. **F3 — RESOLVED.**

   Evidence: `tests/test_tool_descriptions.py:199-208` finds the registered-description line for each exact hyphenated tier and requires both its model slug and its `@ <effort>` text on that same line. The test therefore fails for a missing, swapped, or misattached effort rather than passing merely because all tokens occur somewhere in the description. It continues to inspect the registered FastMCP description (`tests/test_tool_descriptions.py:190-191`).

4. **F4 — RESOLVED.**

   Evidence: the replacement regression invokes the real `server_simple.spawn_agent` with each tier (`tests/test_pi_fast_subtier_resume.py:71-79`, `tests/test_pi_fast_subtier_resume.py:93-103`) while subclassing the real `PiBackend` and stubbing its process-launch boundary (`tests/test_pi_fast_subtier_resume.py:27-40`). It asserts the concrete pair in the captured spawn request and the record read back from `agents.json` (`tests/test_pi_fast_subtier_resume.py:81-90`), then passes that persisted record into `_build_resume_request` and verifies both the resume request and final Pi argv (`tests/test_pi_fast_subtier_resume.py:105-121`). This now protects every boundary claimed by the disposition.

No new problem was introduced by these fixes. The corrected registered-contract and Pi persistence tests pass. The previously reported delivery failure **still reproduces on this reviewer environment**, including with `tests/test_follow_up_delivery.py` run alone: `test_immediately_exiting_child_is_not_confirmed_and_leaves_the_record` returns `agent_busy` instead of `resume_not_confirmed` (`tests/test_follow_up_delivery.py:228-249`), yielding 40 passed and 1 failed. That test uses a fixed pre-existing record PID of `123` (`tests/test_follow_up_delivery.py:139-157`), so the outcome is host-process-state sensitive; none of the Round 2 fixes or staged production changes alter this delivery path. It is therefore not a new feature problem and does not change this verification verdict.

VERDICT: APPROVED
