# Plan review — claude-lead-inbox-wake

Reviewer: Claude Fable 5 (orchestrating session).
Reviewed: `plan.md` (Design B, `background_tasks` variant), against `prd.md`,
`prd-review.md`, `spike-results.md`, and the live source tree.

**Process note (required disposition):** the repo workflow mandates an
independent plan review by the opposite model family (Codex). Codex quota is
exhausted this week; the user explicitly directed that Fable (this session)
performs the review. Deviation is intentional, user-approved, recorded here.

## Verdict

**Approve with one required fix (F1) and one required note (F2).** The plan is
well-grounded: I re-opened every load-bearing citation and they are accurate —
`write_claude_settings` does emit a single `Stop` group today
(hooks.py:136–149), `_shell_quote_command` per-token double-quotes
(hooks.py:126–133), `_watch_argv` yields `... "claude_teams.cli" "watch"
<dir>` with `str(session_dir)` not `as_posix()` (server_simple.py:849–854, as
the plan flags), `IDENTITY = AGENT_NAME or ROOT_LEAD_NAME`
(server_simple.py:61,70), `AGENT_NAME=request.name` at spawn
(claude_code.py:304). The design's decision table is sound, fail-open posture
is thorough, and the load-bearing `background_tasks` assumption is handled
correctly (Task-0 re-confirm + pid/marker fallback). Only one finding blocks a
clean TDD start; it changes which primitive the guard test is written against,
so it must be fixed in the plan, not deferred.

## Findings

### F1 (required fix) — the progress guard's data source does not provide cursor/total

§2.4 specifies the hook reads unread via `messaging.unread_sender_counts`
(messaging.py:104). I verified its signature: it returns **`dict[str, int]`**
(sender → unread count) and nothing else — no per-sender `cursor` or `total`.

§2.5's progress guard, however, stores a snapshot of per-sender
`{"total": N, "cursor": M}` and detects a *productive* wake by whether **any
sender's cursor advanced**. That cursor value is not available from
`unread_sender_counts`. As written, §2.4 and §2.5 are inconsistent — the guard
cannot compute "cursor advanced" from the data source the plan gives it.

Two ways to resolve; pick one in the plan:

- **(preferred) Source both signals from the same richer scan.** The hook
  computes per-sender `{total, cursor, unread}` itself using the primitives
  `messaging.read_inbox_by_sender` + `messaging.load_inbox_cursors`
  (messaging.py:10) — exactly the inline computation the `inbox-status` CLI
  command already does (cli.py:406–411). Derive unread (D3/D4/D5) **and** the
  cursor snapshot (guard) from one read. This keeps a single read-only scan,
  no cursor writes (FR8 intact), and gives the guard the cursor it needs.
- **(inferior) Redefine progress in terms of unread deltas.** Rejected as the
  recommendation: unread count is an imperfect progress proxy — a sender's
  unread can stay flat when a new message arrives in the same window the lead
  drains one, so "unread unchanged" would false-positive as "no progress" and
  trip the guard toward fail-open prematurely. Cursor is the correct signal;
  the plan already knows this (§2.5), so the fix is to source the cursor, not
  to abandon it.

Impact on TDD: test #9 (`test_wake_progress_guard_resets_after_productive_
wake`) and the guard schema in §2.5 must be written against the cursor value
from the richer scan. Small change, but it must land in the plan first so the
red test targets the right primitive.

### F2 (required note, no redesign) — a persistent arming false-negative consumes the no-progress budget

Trace D5 under a false-negative arming match (the model *did* start the
watcher, but §2.4's predicate fails to recognise it — e.g. an unanticipated
command wrapper): every subsequent `Stop` has no unread and re-blocks with the
arm instruction; cursor never advances (there is nothing to read), so with
`stop_hook_active` true the no-progress counter climbs and the guard fail-opens
at N=3. The lead then goes deaf despite a watcher actually running.

This is acceptable *safety* behaviour (better deaf-but-stoppable than
unstoppable), and the plan's separator-normalised, token-set match makes it
unlikely — but the plan currently frames a false-negative as merely "at worst
the lead arms a second watcher" (§2.4). That undersells it: a *persistent*
false-negative degrades determinism to a 3-turn window. **Required:** add one
sentence to §2.4 or §5 acknowledging that a persistent arming-match
false-negative is bounded by the no-progress guard (goes fail-open after N),
and that the manual smoke test (§4 step 5) must confirm the real
`_watch_command_bash` rendering is recognised by the match predicate — i.e.
verify the happy path of arming detection end-to-end, not just the unit-level
token match. No code redesign; this is honesty about the failure mode plus one
concrete smoke-test assertion.

### F3 (info, no change) — assessed sound

- Module split (`claude_teams.lead_wake` separate from `hooks.py` to avoid the
  import cycle; wiring helpers stay in `hooks.py`): correct call — `hooks.py`
  is imported by both `cli.py` and `server_simple.py`, so pulling
  `server_simple` into it would cycle.
- OQ5 resolution (two `Stop` groups, no merge) matches spike probe d.
- `install_lead_wake` over a copy-paste snippet: right, for the
  `sys.executable`-rendering and idempotency reasons given.
- Env-var names/defaults, kill-switch-read-at-runtime, backward-compat with
  older-server sessions: all sound.
- The four §8 contradiction disclosures (`effort` absent, pid marker
  superseded, single→two `Stop` group test change, `_watch_argv` has no
  `--reader`) all check out against source.

## Required-revision summary

| # | Severity | Action |
|---|----------|--------|
| F1 | required fix | Source per-sender `{total, cursor, unread}` from `read_inbox_by_sender` + `load_inbox_cursors` (as `inbox-status` does, cli.py:406–411) so the guard has the cursor it needs; update §2.4, §2.5, and test #9 accordingly |
| F2 | required note | State in §2.4/§5 that a persistent arming false-negative is bounded by the no-progress guard (fail-open after N), and add a smoke-test assertion that the real `_watch_command_bash` rendering is recognised by the match predicate |

Once F1–F2 are applied, the plan is approved to proceed to red-green TDD
implementation.
