# Implementation status — message delivery protocol

Living handoff document. Update it as work lands; it is the source of truth for
"where are we", not conversation memory.

## Where the work lives

- Worktree: `C:\code\github\win-agent-teams-mcp\wt-message-delivery-protocol`
- Branch: `feature/message-delivery-protocol`
- Primary worktree (`agentic-coder-teams-mcp/`) is **not** touched — other agents
  may be using it.
- Read `requirements.md` (R1–R8, acceptance criteria) and `plan.md` (phases,
  design, tests) before doing anything. `plan.md` is revision 5, approved in
  direction by five independent Codex reviews (42 → 84/100) plus a contradiction
  sweep.
- `plan-review-*.md` and `consistency-sweep.md` are review artefacts: they stay
  **untracked** and are never committed.

## Done

| Commit | What |
|---|---|
| `refactor:` | **Phase 0** — removed dead `busy_hint` (field, unreachable branch, 10 test call sites). Narrowed the noqa to `PLR0911` since branch count fell below threshold. |
| `docs:` | requirements.md, plan.md, `docs/reference/agent-messaging-protocol.md`, and the `.gitignore` narrowing that makes `docs/features` + `docs/reference` visible to git. |
| `fix:` | **A7** — `waiting` marker now resolved *before* the transcript-derived checks, so an agent parked at a Stop hook with no `last_message` is no longer reported `agent_busy`. Red-green with a real failing test. |
| `docs:` | synced watch-command return fields after `8e5aa32`. |
| `docs:` | folded PR #30 (kill purges inbox + cursor) into B1 and B4. |
| `docs:` | fixed 7 reference claims invalidated by Phase 0 and A7. |

## Next, in dependency order

**Phase A (remaining).** Everything reads the correlation id, so A1b is first.

1. **A1b** — per-spawn correlation id: generate before `backend.spawn`, persist
   on the agent record, preserve through resume/CAS, carry via
   `SpawnRequest.extra`, make Codex's `_correlated_prompt` consume it instead of
   deriving its own (prevents double markers), load it in `_read_agent_output`.
2. **A1** — server owns final prompt materialization. Sensitivity test on the
   *user prompt only*, then append the marker: single-line on argv,
   newline-delimited in the sidecar. Ordering is load-bearing — reversing it
   routes every Claude spawn through a file read.
3. **A2** — four sequential gates (metadata → scan → count → session-id) plus
   gate 0 for sidecar-pending. Two-tier enumeration + validated-binding cache.
   Legacy records **refuse** follow-up.
4. **A3 / A4 / A4b** — child liveness as early-failure only; nonce confirmation
   against named receipt records; per-target operation lease with
   `holder_create_token` fencing, atomic temp+replace storage, refuse-on-kill
   with a CLI operator escape.
5. **A5 / A6** — unique prompt files with lifecycle rules; explicit consumer
   decisions for all **five** binding outcomes.

**Then** C1+C2 (direction guard — deliberately ahead of Phase B), then Phase B,
then C3+C4.

## Quality gates — all four, whole repo

```
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

There is no `.venv` in a fresh worktree; `uv sync` first.

**Known pre-existing red:** `ty check` reports 2 diagnostics for
`signal.SIGKILL`, which does not exist on Windows. Present on clean `main` too.
Reported, deliberately not silently fixed — it needs a decision on whether that
gate is meant to pass on Windows at all. **Do not describe the repo as green
while this stands.**

## Working rules learned the hard way

1. **Rebase before every phase.** `main` moved four times during the planning
   work and three of those changed facts the plan depended on: the watcher
   settle window (1.5 s → **15.0 s**), the coordination tools' return shapes,
   and kill's inbox-purge semantics. Check `git log HEAD..origin/main` and read
   what landed — do not just rebase mechanically.
2. **Update the reference in the same commit as the behaviour.** Phase 0 and A7
   each falsified claims in `agent-messaging-protocol.md` and were committed
   without fixing it. A trusted, wrong reference is worse than none.
3. **Patching a document breaks its other sections.** Three review iterations in
   a row found stale directives introduced by fixing something else — mostly
   test cases still encoding a rejected rule. After editing any normative rule,
   grep for short distinctive tokens (not whole sentences: markdown wraps them)
   and re-read the tests that reference it.
4. In `agent-messaging-protocol.md`, **symbol names are authoritative, line
   numbers are indicative.** 112 of 113 `server_simple.py` line citations had
   drifted within a few PRs while every named value stayed correct. Converting
   the whole document to symbol-based references is an open, offered follow-up.

## Decisions already taken — do not silently revisit

- **R1** is bounded in-call delivery with an explicitly cooperative tail. A
  persistent dispatcher was considered and **rejected** as a non-goal: it is the
  only way to guarantee the tail for a passive sender, but it means adding a
  daemon to a project built on per-agent MCP servers and a one-shot watcher.
- **R2** is an accident guard, **not** a security boundary. `IDENTITY` is read
  from an env var by the caller's own process; a worker can trivially forge it.
  Ship it, but never describe it as authorization.
- **R8 allows no legacy exception.** Agents spawned before correlation existed
  cannot be followed up and must be killed and respawned. Accepted cost; the
  alternative lets a nonce be confirmed in the wrong conversation and reported
  as `delivered` — the original bug with a false receipt attached.
- **Guaranteed-path messages never enter the actionable inbox.** They go to a
  separate audit store. Since PR #30, kill purges a sender's inbox messages
  entirely, so an inbox-resident audit record would be destroyed by killing the
  sender — this is now the only design that satisfies R4.
