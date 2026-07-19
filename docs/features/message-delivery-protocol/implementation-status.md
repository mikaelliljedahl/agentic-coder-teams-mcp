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
| `feat:` | **A1b/A1** — per-spawn correlation id, server-owned prompt materialization. |
| `refactor:` | shared Windows-aware force-kill helper (clears the old `ty` red). |
| `feat:` | **A2 + A6** — the validation ladder and all five consumer decisions. |
| `test:` | the three coverage gaps the A1 review flagged (persisted-malformed-id restart path, legacy refusal wording, Codex direct-launch fail-loud). |
| `feat:` | **A3 + A4 + A4b + A5** — `delivery.py` (nonce confirmation, rotation-aware receipt scanner), `leases.py` (per-target operation lease), the three-phase `follow_up_agent`, kill refusal + CLI operator escape, unique prompt files. |
| `feat:` | **C1 + C2** — `spawned_by`/`spawned_by_source` on the record, the downstream-only direction guard ahead of every side effect, and `win-agent-teams adopt` (CLI-only, token + generation gated). |

## Next, in dependency order

**Phase A and C1+C2 are complete.** Next: Phase B, then C3+C4.

C1+C2 note for whoever picks up Phase B: **every agent record now needs
`spawned_by`**, and a test fixture that omits it gets `parent_unknown` rather
than reaching the code under test. Five existing fixtures were updated for
this; new ones should set `"spawned_by": "team-lead"` (the default test
`IDENTITY`) unless the test is about the guard itself.

Phase B builds directly on two things A3–A5 established and should reuse rather
than re-derive:

- `pending_delivery` on the agent record already is the "unconfirmed attempt"
  state B1's status store needs, and `_reconcile_pending_delivery` already
  implements "a retry reconciles before re-sending".
- `_LEASE_QUEUE_WAIT_SECONDS` is the bounded in-call wait R1 describes. B1's
  `queued(phase=pending)` tail is the same return shape `follow_up_agent`
  already produces when that budget expires.

## Quality gates — all four, whole repo

```
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

There is no `.venv` in a fresh worktree; `uv sync` first.

All four gates are **green** as of the A3-A5 commit (778 passed, 1 skipped).

Known intermittent, pre-existing and NOT caused by this work:
`tests/test_cli_watch.py::test_watch_settle_wakes_persistent_waiting` fails
roughly 1 run in 3 on a wall-clock settle race. Re-run to confirm before
treating it as a real failure. Every new timing test added for A3/A4/A4b takes
an injected clock and poll interval for exactly this reason.

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
- **A2's tier 1 does not short-circuit the count gate.** The plan reads as
  though a tier-1 token hit binds immediately, but the plan's own test requires
  `ambiguous` when two transcripts carry the marker and one of them is the
  stored one. The stored transcript therefore joins the candidate set as an
  extra, cutoff-free candidate rather than returning early. Steady-state cost is
  covered by the validated-binding cache, which does no scan at all on a hit.
- **A2's gate 0 is evaluated inside the count gate.** It is numbered 0 because
  it changes the meaning of an observation, but it only ever reinterprets *zero
  matches*, and it cannot precede the metadata gate: a `legacy` record has no
  token to scan for.
- **`agent_status` keeps liveness precedence over `state="unknown"`.** A dead
  process still reports `dead`; `unknown` replaces only the mtime-recency guess.
- **Guaranteed-path messages never enter the actionable inbox.** They go to a
  separate audit store. Since PR #30, kill purges a sender's inbox messages
  entirely, so an inbox-resident audit record would be destroyed by killing the
  sender — this is now the only design that satisfies R4.
