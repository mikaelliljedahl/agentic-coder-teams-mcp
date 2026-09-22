# Independent implementation review: GPT-6 tier remap

1. **F1 — major — Pi context-window claims remain despite the user's final decision that this MCP must make none.**

   **Evidence:** The final resolution in `docs/features/gpt6-tiers/plan.md` says the MCP no longer makes context-window claims for Pi. The staged implementation still says the ladder is "tuned to its context window" in `README.md:681`, says Pi's context window is whatever the operator configuration provides in `README.md:706-707`, repeats that Pi's context window is its own configuration in `ADDING-A-BACKEND.md:109-110`, and makes the same claim in the production ladder commentary at `src/claude_teams/backends/pi.py:187-190`. These statements are not needed to explain or implement the approved `high = gpt-6-luna @ max` mapping, and they directly contradict the superseding resolution. The consuming-agent `spawn_agent` docstring correctly avoids this claim.

   **Recommended fix:** Remove all Pi context-window wording. In the README introduction, describe the ladders simply as backend-specific fixed mappings; retain the Codex-only 272k rationale where useful. In the backend guide and Pi source comment, state only that Pi uses Luna @ max at `high`, without assigning ownership, configuration, size, or semantics to Pi's context window. Add a focused grep/assertion if desired so this explicit decision cannot regress.

The staged `_TIER_LAUNCH` tables otherwise exactly match the approved tier/slug/effort mappings, including Pi `high = gpt-6-luna @ max`, `medium-fast = gpt-6-sol @ low`, and `high-fast = gpt-6-sol @ medium`. `_PI_UPGRADE_HINT` names Pi 0.87.1. The `spawn_agent` description clearly and accurately pins the shared ladder, the Pi-only fast tiers, the Codex/Pi `high` difference, discovery behavior, and the minimum Pi version without making a context-window claim. Tests meaningfully pin every tier's slug and bundled effort, cover stale non-empty catalogs and raw-slug compatibility, and exercise representative argv construction. Remaining GPT-5.6/Terra references outside historical feature documents are either explicit raw-slug compatibility coverage/documentation, stale-catalog fixtures, migration rationale, or comments supporting negative assertions; no obsolete tier mapping remains. No `262k`, affirmative `3-4x`/`3–4x`, or `1M` claim remains outside historical feature documentation (the `3-4x` test occurrence is a negative assertion).

Validation: `uv run pytest -q` — 1688 passed, 4 skipped. `uv run ruff check .` — passed.

VERDICT: CHANGES REQUESTED

---

## Dispositions (implementer: Claude Opus 5.5)

- **F1 — accepted, fixed.** Removed every pi context-window statement:
  README intro now says "each backend has its own fixed ladder", the README
  rationale says only "Pi uses Luna @ max at `high`", ADDING-A-BACKEND.md says
  "Pi uses Luna @ max", and the pi.py ladder comment no longer mentions context
  windows. The Codex-only 272k rationale is retained. Gates rerun green.
