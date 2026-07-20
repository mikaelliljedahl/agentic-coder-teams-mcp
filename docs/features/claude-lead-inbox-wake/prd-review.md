# PRD review — claude-lead-inbox-wake

Reviewer: Claude Fable 5 (orchestrating session).
Reviewed: `prd.md` as written 2026-07-20.

**Process note (required disposition):** the repo workflow mandates an
independent review by the opposite model family (Codex). Codex quota is
exhausted this week; the user explicitly directed that Fable (this session)
reviews Opus's work for this feature instead. This deviation is intentional,
user-approved, and recorded here rather than silently absorbed.

## Verdict

**Approve with required revisions.** The PRD is well-grounded in the repo
(disk contract, hooks wiring, Pi-wake parity, nested-lead anti-regression) and
the verified Stop-hook semantics section is exactly the right groundwork. But
the document defers the single decision that determines whether the feature is
worth building — blocking vs non-blocking design — as a side question (OQ3),
while several requirements and one user story silently assume answers to it.
The findings below must be resolved in the PRD (not punted wholesale to the
plan) because they change requirement text, not just mechanism.

## Findings

### F1 (major) — The non-blocking design must be a first-class candidate, and OQ3 frames it wrong

OQ3 asks whether a hook-spawned process can trigger a harness re-invocation.
Almost certainly it cannot: the harness wakes on completion of *tracked*
background tasks the model started via its Bash tool; a process forked by a
hook is invisible to that tracking. If the plan spikes only that hypothesis,
it will conclude "non-blocking impossible" and default to the UI-blocking
design. That is a false dichotomy.

There is a third design the PRD does not name — call it **Design B,
"verified arming"**:

1. On `Stop` with live subagents, the hook checks whether the background
   watcher is actually running (pid/marker file under the session dir,
   written by the `watch` CLI or a tiny wrapper).
2. If armed → allow the stop. The turn ends, the UI stays typeable, and the
   already-running tracked watcher wakes the harness on the next message —
   the existing recipe's wake path, unchanged.
3. If **not** armed → block with a reason containing the exact command:
   "run this as a background Bash task now: `<python> -m claude_teams.cli
   watch <dir> …`". Following a direct injected instruction is near-certain
   even on Opus-low — the failure mode was *remembering unprompted*, not
   *refusing when told*.
4. Next `Stop`, the hook re-verifies. Not armed again → block again. The
   lead can never go to sleep unarmed; determinism lives in the
   *verification*, not in the model.

Design B gives: no UI freeze (R1 gone), no 600 s ceiling (overnight waits
fine, watcher runs for hours), no 8-block-cap pressure in normal operation
(a successful arming is progress), at the cost of one short extra model turn
per arming. Design A (block-in-hook) gives zero extra turns but freezes the
UI and cannot wait past ~10 min.

**Required change:** present Design A and Design B side by side in the PRD's
proposed-capability section, rewrite OQ3 as "choose A, B, or hybrid
(short in-hook grace wait, then arm-and-release), decided by an early spike",
and add the arming-verification marker (pid file) to the disk-contract
considerations if B is on the table.

### F2 (major) — U4 (overnight lead) is unsatisfiable under Design A; the PRD contradicts itself

U4 promises a lead left overnight "wakes on the next reply". Under Design A,
FR13(a) allow-on-timeout means the lead is asleep and deaf ≤ 10 minutes after
its last turn — it will *not* wake on a reply at hour 3. FR13(b) re-block hits
the 8-cap after ~8×max-wait (< ~80 min), then stops anyway. So as written,
U4 fails under every Design-A disposition. Design B satisfies U4 trivially.

**Required change:** either (a) descope U4 to "fails safe overnight; wakes on
the next human nudge" and say so, or (b) keep U4 and note it forces Design B
(or a hybrid) — but the PRD must stop promising a story its lead candidate
design cannot deliver.

### F3 (major) — FR14 as written defeats successive wakes

`stop_hook_active: true` is set on *any* continuation caused by a Stop-hook
block — including a legitimate message-wake. FR14's blanket "take the
cap-safe path (shorten/skip the wait, or allow)" therefore fires on the very
next Stop after each successful wake: in a long session (wake → drain → stop
→ wake …) every second hook invocation would skip its wait, and the feature
degrades to first-wake-only. The 8-cap counts consecutive blocks **without
progress**; a wake that led to `read_messages` (cursor advanced) *is*
progress.

**Required change:** refine FR14 to a progress-based rule: fail toward
allow only when repeated blocks show **no cursor advance** since the previous
block (track last-seen cursor/total in a small state file under the session
dir), not merely because `stop_hook_active` is true. Keep the field as a
belt-and-braces guard combined with the progress check.

### F4 (minor) — FR11 conflicts with FR8's wording

FR11 wants the block reason to name the unread sender(s); FR8 forbids the
hook to "read/consume messages" and limits it to `session-dir` /
`inbox-status` / `watch`. If `inbox-status` doesn't expose senders, FR11 is
unimplementable under FR8 as written. The Pi wake's actual invariant is
"never *advance the cursor*", not "never open the file".

**Required change:** either relax FR8 to "MUST NOT write the cursor or
mutate any session file; a read-only scan of `inbox-<reader>.jsonl` for
metadata (sender, count) is permitted", or extend `inbox-status` to expose
senders and keep FR8 strict, or downgrade FR11 to counts-only. Pick one in
the PRD; don't leave the contradiction to the implementer.

### F5 (minor) — OQ6 (Claude Desktop honors hooks) must be the *first* spike, not an open question among nine

The user's primary lead runs in Claude Desktop's embedded Claude Code
harness. If that harness ignored user/project settings hooks, the feature
would miss its main consumer. (Expectation: it honors them — this very
session runs the harness with settings support — but it must be confirmed
empirically before implementation effort is spent.) **Required change:** mark
OQ6 and the OQ3 spike as pre-plan gating checks, ordered first.

### F6 (info) — Sound as written; no change required

- Verified-semantics section (§2), incl. the reference-vs-guide discrepancy
  candor and the `session_id` trap (R6/FR5): good.
- Identity via `session-dir` CLI + env, nested-lead anti-regression (FR5–FR7,
  AC3): exactly right, matches the Pi bugfix lesson.
- Fail-open posture (FR7), kill switch (FR17), Windows quoting reuse
  (FR18–FR19), tool-docstring rule (FR26): good.
- N3 keeping the background-watcher recipe as documented fallback: note that
  under Design B it is not a *fallback* but the *wake path itself* — wording
  will need a touch-up once F1 is resolved.

## Required-revision summary

| # | Severity | Action |
|---|----------|--------|
| F1 | major | Add Design B ("verified arming") as first-class; rewrite OQ3 as A/B/hybrid spike |
| F2 | major | Reconcile U4 with the chosen design space (descope or bind to B) |
| F3 | major | FR14 → progress-based (cursor-advance) guard, not blanket `stop_hook_active` skip |
| F4 | minor | Resolve FR8↔FR11 contradiction explicitly |
| F5 | minor | Promote OQ6 + OQ3 spike to ordered pre-plan gating checks |

Once F1–F5 are applied, the PRD is approved to proceed to `plan.md`.
