# External-member wake — post-implementation review (Claude Opus, independent)

Reviewer: Claude Opus 4.8 (orchestrator), independent of the Fable implementer.
Scope: `member_wake` + `install_member_wake` against approved plan v2
([`plan.md`](plan.md), incl. §8 review-1 dispositions), the tests, and the diff.

**Workflow-deviation note (transparent):** CLAUDE.md wants an opposite-family
(GPT/Codex) reviewer at each gate; those tokens are exhausted this session. The
implementer was Fable; this authoritative post-implementation gate is Opus —
genuinely independent of the implementer and a different model. That is the best
available independence under the constraint and is recorded here rather than
silently self-approved.

## Verdict: APPROVED — 95/100. No blockers. One adjacent finding (outside this module).

## Plan conformance

Every plan v2 element is present and faithful:

- **M0–M5 + M2b** decision path implemented exactly (`member_wake.evaluate_member`).
- **Reuse-by-import** of `WakeDecision`, `_scan_senders`, `_is_armed`,
  `_apply_guard` (and its `_read_guard`/`_write_guard`/`_cursor_advanced`
  machinery), `_read_payload` — no duplication, `lead_wake` behavior unchanged.
- **Member-specific new code where lead_wake was hardcoded (review-1 Major 3):**
  `_member_kill_switch_on` (correct truth table — verified below), a
  `win-agent-teams/member-wake` `_log_line`, `_member_read_reason`
  (`external_read`, not `read_messages`), `_member_arm_reason` (reader-scoped
  watch command + `leave_team` escape hatch).
- **M2 membership gate** keys on `name`/`backend=="external"`/
  `spawned_by_source=="join_ticket"`, live only on `status=="running"`; absent /
  `left` / `_TERMINAL_STATUSES` / unreadable all fail open. Matches the real
  record written by `join_team` and mutated by `leave_team`.
- **M2b abandoned-team TTL (review-1 Major 1)** fails open when no joined-side
  activity within `WIN_AGENT_TEAMS_MEMBER_WAKE_TTL_SECONDS` (default 6h).
- **`install_member_wake` defaults `scope="user"` (review-1 Major 2)**; docstring
  states the Stop-hook harness requirement, no-credential guarantee, pull-only,
  and the leave/remove/kill-switch escape hatches.
- **Coexistence** verified: lead group keys on `claude_teams.lead_wake`, member
  group on `claude_teams.member_wake`; neither string is a substring of the
  other, so `_install_wake_hook` and `_install_member_wake_hook` never clobber
  each other. Tests 31–33 assert both directions.
- **No token at rest:** hook argv + settings carry only member name + joined dir
  (test 34 asserts it).
- **`_watch_command_bash`** extended with keyword-only `reader=None` — existing
  callers unaffected (test 38); reader-scoped rendering verified (test 37).

## Deviations from plan v2 — all reviewed and accepted

1. **M2b activity scan also includes `agents.json` mtime.** Correct and
   necessary: a freshly joined quiet team may have no state marker or inbox yet,
   and scanning only the plan's `state-*`/`inbox-*` pair would spuriously mark a
   live team stale and silently disarm the wake. `agents.json` always exists and
   is rewritten on membership changes. Accepted.
2. **Missing/blank `--member` fails open at M1 instead of an argparse error.**
   Correct: `argparse` `required=True` exits non-zero, which would violate the
   never-unstoppable contract. Failing open to allow is the safe choice.
   Accepted.
3. **Guard-cap fail-open logs code `D6`** (reused `lead_wake._apply_guard`
   verbatim) rather than an `M`-code. Cosmetic; tests assert on `action`, not the
   code. Accepted.

## Gates (independently re-run by the reviewer)

- `uv run ruff format --check` → 75 files already formatted.
- `uv run ruff check` → All checks passed.
- `uv run ty check` → All checks passed.
- `uv run pytest -q` → **1 failed, 1162 passed, 3 skipped.** The single failure
  is `tests/test_agent_output.py::test_spawn_agent_persists_output_lookup_metadata`
  — **pre-existing and unrelated**, independently confirmed by re-running it with
  this feature's changes `git stash`ed (fails identically on the clean tree; the
  known live-PID `create_token` local flake). This feature adds **zero** new red.
- `tests/test_member_wake.py` → 39 passed, covering all 20 plan §5 cells.

## Adjacent finding (NOT in `member_wake`; non-blocking for this feature)

**`lead_wake._live_subagent_names` treats a `left` member as still live.** It
skips only `_TERMINAL_STATUSES = {"killed"}`, so after a member calls
`leave_team` (status → `left`), the *upstream* lead-wake hook keeps counting that
member as a live child and keeps demanding an armed watcher on the lead — even
though the member has permanently left and its inbox is drained. This is a real
inconsistency in the wake family: `member_wake`'s M2 correctly treats `left` as
done, while `lead_wake`'s D2 does not, so a `leave_team` does **not** quiet the
lead-wake loop as a user would expect. Observed live this session (the
`visual-qa` member left, yet the lead-wake nag continued).

**Recommendation:** in `lead_wake._live_subagent_names`, also exclude external
members whose `status == "left"` from the live set (a permanently-left member is
not something the lead must wait for). Small, behavior-correct, and directly
serves the intended effect of `leave_team`. Because it is a behavior change to
pre-existing `lead_wake` code (with its own tests), it is surfaced here for an
explicit decision rather than folded in silently: fix in this same branch, or a
separate follow-up.

**Resolution (accepted by the user — fix in this branch):** `lead_wake.py`
`_live_subagent_names` now skips `status == "left"` alongside
`_TERMINAL_STATUSES`, with a new RED→GREEN test
`tests/test_lead_wake.py::TestLiveSubagentScoping::test_left_external_member_is_not_live`
(a `left` member now yields a D2 allow instead of a D5 block). Full suite after
the fix: `1163 passed, 3 skipped, 1 pre-existing failure` (the unrelated
`test_agent_output` flake); ruff/ty clean.
