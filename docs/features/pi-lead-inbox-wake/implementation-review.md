# Implementation review — Pi-lead inbox wake (consolidated)

Independent post-implementation review per CLAUDE.md step 4. Implementer: **Claude**
subagents (TDD). Reviewer: independent **Codex** agent `codex-pi-lead-impl-review`
(opposite family), via the `win-agent-teams` MCP. Reviewer ran the suites and read
the production source (not just the summaries).

## Final verdict

**APPROVED WITH NOTES — 96/100.** No blocker remains. The only outstanding
feature-specific validation is the deliberately deferred **live Pi/model smoke
run** (needs credentials); committed package-loadability evidence covers the rest.

## Iteration history

| Iter | Artifact | Score | Verdict | Outcome |
|------|----------|-------|---------|---------|
| 1 | `implementation-review-1.md` | 76 | CHANGES REQUESTED | 4 blockers: cross-platform inbox-path bug; T5d hollow; parseWatch exit-0 stderr; T9 evidence absent. |
| 2 | `implementation-review-2.md` | 96 | **APPROVED WITH NOTES** | All 4 blockers resolved + all 4 non-blocking follow-ups; suites re-run green. |

## Blockers resolved (iter 1 → iter 2)

1. **Cross-platform inbox-path (real bug)** — `isLeadInboxPath()` normalizes
   separators on both sides; `state-machine.ts:147` uses it. A Windows `session_dir`
   + backslash wake path now injects (regression test proven).
2. **T5d fidelity** — rewritten so bare unread stays constant while
   `min(cursor,total)` reaches the captured target; a naive "unread decreased" ACK
   now fails it.
3. **parseWatch exit-0 stderr** — non-empty stderr on exit 0 is now `malformed`
   (backoff, no wake), per plan §3.1; test added.
4. **T9 delivery evidence** — `test/index.test.ts` imports and drives the real
   default-export factory (registration + one live child + abort); `README.md`
   documents load + smoke procedure. Live run explicitly deferred.

Non-blocking follow-ups (T11 counts live children; unsafe-reader exit tests both
commands; T2b follows with real `read_messages`; prettier claim corrected) — all
resolved.

## Quality gates at approval (independently re-run)

- Python `uv run pytest -q`: **500 passed, 2 skipped**.
- `uv run ruff check .`: **green**.
- Extension `npx tsc --noEmit`: **green**; `npx vitest run`: **50 passed (5 suites)**;
  `npx eslint .`: **green**; scoped prettier: **green**.
- **Pre-existing repo gates left red (NOT this feature):** `uv run ty check` — 44
  diagnostics (confirmed 44 with and without this feature's Python changes; zero on
  the added lines); `ruff format --check .` — 7 pre-existing files. Both surfaced,
  not absorbed into the feature branch.

## Remaining before "done"

- **Live Pi smoke test PASSED (2026-07-19, Codex worker → Pi lead woke immediately
  and read)** — validated interactively by the user: a Codex worker spawned by a Pi
  lead (running `pi -e pi-extensions/win-agent-teams-wake`) called `send_message`,
  and the Pi lead was woken immediately and read the message with no manual input.
  Procedure in `pi-extensions/win-agent-teams-wake/README.md`.
- **Commit + PR** (CLAUDE.md step 5) — pending user go; review artifacts
  (`plan-review-*.md`, `implementation-review-*.md`) are intermediate and must stay
  untracked.

## Separation of duties

Plan review: Codex. Implementation: Claude subagents. Implementation review: Codex.
Reviewer never implemented; implementer never reviewed.
