# Plan review — Pi-lead inbox wake (consolidated)

Independent plan review per CLAUDE.md step 2. Implementer will be **Claude**, so the
reviewer was an independent **Codex** agent (opposite family), driven through the
`win-agent-teams` MCP. Reviewer verified every technical claim against the actual
source (repo Python + upstream Pi TypeScript API).

## Final verdict

**APPROVED WITH NOTES — 96/100.** No implementation blocker remains. The plan
([plan.md](plan.md), v4) is ready for red-green TDD implementation.

## Iteration history

| Iter | Artifact | Score | Verdict | What it forced |
|------|----------|-------|---------|----------------|
| 1 | `plan-review-1.md` | 56 | CHANGES REQUESTED | Feasibility gate (extension can't call MCP → shell out); cursor-ownership; wrong Pi injection API; discovery/recovery nuance; test location. |
| 2 | `plan-review-2.md` | 76 | CHANGES REQUESTED | Fallback self-loop still unbounded; unread-count ACK race; CLI contracts; `ExecOptions` has no `env`; `sendMessage` returns void; `isIdle` context-only. |
| 3 | `plan-review-3.md` | 85 | CHANGES REQUESTED | ACK_STALLED late-drain/new-sender transitions; owned `AbortController` (no shutdown `AbortSignal`); `display:boolean` typo; one-injection-per-generation. |
| 4 | `plan-review-4.md` | 96 | **APPROVED WITH NOTES** | — (all blockers resolved; ACK_STALLED correctness traced; lifecycle verified). |

Finding-by-finding dispositions are recorded in [plan.md](plan.md) §9. The exact,
upstream-verified Pi API the plan is written against is pinned in [plan.md](plan.md)
§10.

## Non-blocking notes carried to implementation (from `plan-review-4.md` §6)

These do not block; handle them during coding / final doc cleanup:

1. **R6 wording** — corrected in v4 to "extension-owned controller aborted from
   `session_shutdown`" (done).
2. **ACK_STALLED session-change precedence** — checking `session-dir` *before*
   evaluating old-session status would rebind sooner. Not a correctness blocker
   (pre-injection validation already prevents a wrong-session injection; worst case
   one old watcher runs until timeout `T`). Consider reordering during
   implementation.
3. **Pin the exact Pi dependency immediately** — commit the exact resolved
   `@earendil-works/pi-coding-agent` version (no range) with the first extension
   scaffold; make the §10 signature check part of CI.
4. **Reuse one safe-name invariant** — `watch --reader` and `inbox-status --reader`
   should share the repo's existing safe-agent-name predicate (hooks/process-mgmt)
   rather than a divergent local copy.
5. **§9 finding-1 wording** — updated in v4 to the precise `inbox-status`/
   generation-ACK terms (done).

## Separation of duties

Implementer (planned): Claude. Reviewer (this review): Codex agent
`codex-pi-lead-plan-review` via `win-agent-teams`. The post-implementation review
(CLAUDE.md step 4) must be done by the opposite family from whoever implements.
