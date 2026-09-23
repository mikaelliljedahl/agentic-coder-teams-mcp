# Implementation review: sol-effort-bump

Verdict: CHANGES REQUESTED

## Findings

1. [MAJOR] The README and backend comments overstate the effort increase as applying uniformly to every tier. `README.md:703-710` says every tier is one effort step above its GPT-5.6 predecessor; the same universal claim appears in `src/claude_teams/backends/codex.py:122-128` and `src/claude_teams/backends/pi.py:187-189`. In the approved mapping, `max` remains Astra @ medium, while `xhigh` moves from Astra @ low to Sol @ xhigh, so those are not one-step increases at the same tier/model. Keep the selected mappings, but qualify the rationale to identify the tiers where the one-step comparison applies and explain `xhigh`/`max` as the cost and capability choices documented below.

## Checks confirmed

- Both tables match the approved “after” ladder exactly: Codex at `src/claude_teams/backends/codex.py:133-140`; Pi at `src/claude_teams/backends/pi.py:194-202`. Pi has no `high-fast` key.
- Pi’s `_reject_retired_tier()` runs before raw-slug handling in both resolvers (`src/claude_teams/backends/pi.py:238-243,266-282`), and normalizes with `.lower()` (`pi.py:284-290`). Its tests reject the retired tier with empty and non-empty discovery (`tests/test_backends/test_pi.py:196-202`) and directly exercise `resolve_model` (`tests/test_backends/test_pi.py:129-133`). The exception is actionable and correctly subclasses `UnsupportedBackendModelError` (`src/claude_teams/backends/contracts.py:74-89`).
- The spawn path lets `RetiredTierError` reach the tool caller (`src/claude_teams/server_simple.py:3299,3412`); resume reuses the stored concrete model and effort, and current spawn records do not store a tier name (`server_simple.py:3973-3981`; `tests/test_pi_fast_subtier_resume.py:73-90,93-120`). The CLI and `list_backends` enumerate `supported_models()` dynamically (`src/claude_teams/cli.py:106-113`; `server_simple.py:6297-6305`), so they cannot list `high-fast` after its removal.
- The `spawn_agent` description lists every tier with its model and effort, describes `medium-fast` as the sole Pi-only tier, and explains the retired-tier error (`src/claude_teams/server_simple.py:3183-3205`). README, backend guide, and CLAUDE.md otherwise match the implementation, including the Pi 0.87.0 availability note. A repo-wide search excluding `docs/features/` and `.venv` found no remaining old pair mappings; remaining `high-fast` references only identify the retired name or test its rejection.
- The implementation deviation is sound: the tool description mentions `high-fast` only to say it was removed and how to recover, not as a selectable tier. The test distinguishes the bullet list from that notice (`tests/test_tool_descriptions.py:243-257`), so the consuming agent receives useful migration guidance without being offered a retired tier.
- Ran `uv run pytest -q tests/test_backends tests/test_tool_descriptions.py tests/test_pi_fast_subtier_resume.py`: **532 passed**.
