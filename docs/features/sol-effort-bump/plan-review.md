# Plan review: sol-effort-bump

Verdict: CHANGES REQUESTED

## Findings

1. [MAJOR] The README promises tier names and order are stable, which conflicts with retiring `high-fast`. The README currently says the tier names and order “are stable” (`README.md:678-681`), while the plan removes `high-fast` (`plan.md:18,27-29`). The plan says to update the README table and prose, but does not explicitly resolve this compatibility claim. Revise it to state that tier mappings or availability can change, or explicitly document this removal as an exception.

2. [MAJOR] The red tests do not pin the planned `resolve_model` behavior. The design requires both `resolve_launch` and `resolve_model` to reject `high-fast` (`plan.md:59-65`), but the retired-tier test only calls `resolve_launch` (`plan.md:100-101`), and the Pi mapping tests only specify `resolve_launch`/argv for the new pairs (`plan.md:98-99`). `PiBackend.resolve_model` has its own raw-slug fallthrough when a name is absent from `_TIER_LAUNCH` (`pi.py:228-239`), so it would return `"high-fast"` if its guard were omitted. Add tests that `resolve_model("high-fast")` gives the retirement error and that changed tier slugs (`high` and `xhigh`) resolve to Sol. Keep the launch tests for both empty and non-empty discovery to protect against the silent fallback.

3. [MAJOR] The README’s Pi minimum-version statement becomes false for `xhigh`. It currently says pi 0.87.0 lacks every tier except `xhigh`/`max` (`README.md:717-718`). The plan moves `xhigh` from Astra to Sol (`plan.md:19`), and the stale pi catalog used by the existing test includes Astra but not Sol (`tests/test_backends/test_pi.py:236-248`). Update the README prose to say `xhigh` now also requires Sol availability, and extend the stale-catalog test matrix to assert that `xhigh` fails with the Sol-unavailable error. The proposed test list mentions `high` requiring Sol but omits this `xhigh` ownership change (`plan.md:98-99`).

4. [MINOR] Prefer the existing invalid-model exception for the retired tier, or explicitly justify the plain `ValueError`. `spawn_agent` does not catch an exception from `resolve_launch`: it awaits `run_blocking(_do_spawn)` and returns the result (`server_simple.py:3299,3412`), so a `ValueError` will fail the tool call instead of returning a successful spawn. That is appropriate for a removed input, and `BackendModelUnavailableError` would be the wrong category because the selected model is not missing. However, the code already uses `UnsupportedBackendModelError`, a `ValueError` subclass, for invalid model input (`contracts.py:61-71`, `claude_code.py:101-109`). Use it (or a small specialized subclass retaining “removed; use `high`” guidance) so callers can handle invalid model selections consistently.

## Resume and persisted records

Normally spawned agents are safe: `spawn_agent` resolves the tier before storing `resolved_model` (`server_simple.py:3294-3299,3374-3376`), and the existing resume regression verifies persistence and verbatim reuse of the concrete pair (`tests/test_pi_fast_subtier_resume.py:73-90,93-121`). I found no repository fixture with `"model": "high-fast"`. Resume itself does not call either tier resolver; it uses a stored string verbatim (`server_simple.py:3973-3980`). Thus an abnormal or legacy record whose model field literally is `high-fast` would bypass the new guard and be passed onward as a raw model. This is not a risk for records created through the current spawn path, but the plan should keep that scope clear and avoid implying the guard also validates persisted records.

## Repo-wide scan and test adequacy

The core red-green cases are well targeted for the ladder mapping, shared-tier equality, removal from discovery, and `medium-fast` persistence. In addition to the gaps above, update the existing Pi partial-catalog assertions for the new model owners (`tests/test_backends/test_pi.py:216-223`) and remove `high-fast` from the Pi argv and resume parameter tables (`tests/test_backends/test_pi.py:324-337`; `tests/test_pi_fast_subtier_resume.py:21-24`).

A repo-wide search excluding `docs/features/` and `.venv` found the expected stale references in the Pi backend, spawn-agent description, README, backend guide, CLAUDE.md, and the named tests. `.claude/skills/agent-orchestration/SKILL.md` only gives generic model-selection guidance and has no tier mapping to update. The CLI and `list_backends` expose `supported_models()` dynamically (`src/claude_teams/cli.py:106-113`; `src/claude_teams/server_simple.py:6297-6305`), so they need no hard-coded ladder edit.
