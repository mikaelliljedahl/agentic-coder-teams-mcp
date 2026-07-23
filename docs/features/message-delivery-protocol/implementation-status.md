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
| `feat:` | **Phase B** — `filelock.py` (the registry's lock, extracted and shared) and `delivery_store.py` (the durable per-sender audit store). |
| `feat:` | **Phase B** — the bounded wait (B2), the one-budget delivery loop (B0), the state machine (B1), `no_delivery_path` (B3), `delivery_status` + `deliver_pending`, kill-time reconciliation, and the reference update. |
| `feat:` | **C3** — `_classify_recipient` (five classes), the downstream send routed through the shared `_guaranteed_send`, and the refusals that replace the reroute. |
| `test:` | **C4** — the watcher contract (R3) characterised: wake-without-consume, cursor clamping, exit-2-does-not-strand. No watcher code changed. |

## Status: implemented, awaiting re-review

**Every phase has landed — Phase 0, A, B, C1+C2, and C3+C4**, plus all eleven
findings from the Phase A review and all eight from the final review. The
remaining step is a clean re-review, not more implementation; open follow-ups
are listed under "Offered, not done" below.

**Read this before trusting a green suite.** Three separate reviews found live
critical defects while all four gates were green — six in Phase A, four in the
final pass. The pattern is consistent and worth internalising: the defects live
in concurrency, crash windows and error states, which unit tests on an injected
clock do not reach. The final round needed a genuinely threaded test (two
callers racing one idempotency key) and induced `OSError`s to go red at all.

Two defect *classes* also recurred at new sites after being fixed once: a
persistence helper swallowing `OSError` and reporting success (`save_leases`,
then `save_records`), and an error collapsing into "absence" rather than
staying uncertain (candidate enumeration, then `_scan_for_nonce`). When fixing
either shape, grep for the other instances rather than assuming the one in
front of you is the only one.

**Round 3 found both classes again, and this is now the governing rule of the
feature: *an error is not an absence, and a failed write is not a success.***
The second re-review found the repair had stopped at the write primitive — the
*loaders* still failed open (`load_records`, `load_leases`, `_to_lease`), three
call sites still swallowed failed writes (`_release_delivery_claim`,
`_discard_delivery_record`, `release_lease`'s discarded result), liveness
collapsed "gone" and "could not tell" into one bool, and kill deleted the only
metadata a later rescan needed. All are fixed; see
`code-review-final-2.md` (untracked) and the commits.

A full sweep of `src/` for both shapes was run and is recorded in the round-3
report. Several further instances exist **outside** this feature's surface and
were deliberately not changed here — notably `_ensure_lead_token` minting a
fresh unpersisted token from a corrupt `session-meta.json` (which bricks the
CLI operator escape), `session_info` reporting a corrupt registry as a healthy
zero-agent session, `_session_has_live_agent` returning `False` on a read error
immediately before `shutil.rmtree`, `_bind_path` mapping a read error to the
terminal `BINDING_UNVERIFIED` instead of the retriable `BINDING_INDETERMINATE`,
and `_follow_rotation` silently dropping an unreadable candidate below the
`len(candidates) > 1 → ambiguous` guard. Each needs a policy decision the
current specs do not settle. Do not "fix" them incidentally; plan them.

### What C3 changed, for anyone reading the diff later

- `_message_recipient` is **gone**, replaced by `_classify_recipient`, which
  returns one of `child` / `spawner` / `sibling` / `unrelated` / `unknown`. Only
  the first two are deliverable. The old "unknown name is routed to the lead
  with a warning" rule is what R5 forbids: nothing consumed the warning, so a
  typo became a real-looking upstream message.
- `send_message` gained an `idempotency_key` parameter, used only by the
  downstream branch. Its **result shape now depends on the recipient**: upstream
  returns `{success, to}`, downstream returns the full B1 delivery schema.
- `follow_up_agent`'s body was extracted into `_guaranteed_send` and both tools
  call it. They must not be two implementations that happen to agree.
- `sibling` and `unrelated` are separate classes on purpose. Both refuse
  identically, but reporting a grandchild as "sibling" would be a lie in a
  client-visible field.
- Four existing tests encoded the removed reroute and were updated, not
  deleted: two `_message_recipient` unit tests, the `send_message` reroute test
  in `test_agent_output.py`, and the tool-description test.

### C4 is characterisation, and says so

`tests/test_watcher_contract.py` passed on the first run by design — it pins
behaviour that already held. No watcher code was touched: wake priority, the
settle window, and the `before = after` output-edge coupling are all unchanged.
Mutation testing was used instead of red to show the tests bite (below).

### Mutation results (C3 + C4)

Applied to a copy, run, restored from the copy — never `git checkout`.

| # | Mutation | Result |
|---|---|---|
| M1 | named spawner no longer short-circuits before the parentage check | killed |
| M2 | unknown recipient rerouted upstream (the old rule) | killed |
| M3 | guaranteed send also appends to the inbox (B4 breach) | killed |
| M4 | sibling/unrelated routed instead of refused | killed |
| M5 | any record with a `spawned_by` counts as my child | killed |
| M6 | wake advances the cursor (consumes) | killed |
| M7 | cursor clamp (`min`) removed | killed |

No survivors. `server_simple.py` was verified afterwards by **AST body
comparison** against `HEAD` (127 → 129 symbols; only `_message_recipient`
removed, `_classify_recipient`/`_guaranteed_send`/`_spawner_target` added,
`send_message`/`follow_up_agent` changed) and by checking all 14 registered
`@mcp.tool()` docstrings are present and non-empty.

## Round-3 changes (second re-review)

| Area | What changed |
|---|---|
| `delivery_store.load_records`, `leases.load_leases`, `leases._to_lease` | Absence (`FileNotFoundError`) still reads empty; unreadable or malformed now raises `DeliveryStoreUnreadableError` / `LeaseStoreUnreadableError`. `DeliveryStoreUnreadableError` subclasses `DeliveryStoreError` so every existing fail-closed handler inherits the behaviour. |
| `process_manager.ownership_probe` | New three-valued ownership: `ours` / `not_ours` / `indeterminate`. `owns_process` is now a thin `== ours` wrapper, so destructive gates are unchanged. Only `not_ours` may authorize a reclaim. |
| `leases._holder_reclaimable`, `reserve_lease`, `reconcile_lease` | Take `holder_probe` instead of `holder_live`. Indeterminate never reclaims. `reconcile_lease` raises `LeaseNotPersistedError` rather than reporting a reclaim it did not persist. `drop_agent` returns a bool. |
| `server_simple._claim_is_held` (was `_holder_is_live`) | Per-call `claim_id` plus an in-process `_ACTIVE_CLAIM_IDS` registry. A claim stamped with our PID that we are no longer working is reclaimable — which closes the wedge — and reclaiming it cannot permit concurrent work. A foreign holder that is merely unprovable stays held. |
| `_release_delivery_claim` | Deregisters in-process first and unconditionally, returns whether the write landed, logs at warning. |
| `_discard_delivery_record` | Returns success; a failed C2 rollback annotates the refusal (`record_discarded: false`) instead of promising nothing changed. |
| `_scan_target` / `target_snapshot` | The target's registry record is copied onto the durable row at attempt time and refreshed post-resume, so a rescan survives `kill_agent` deleting the agent. Scanning only — liveness still comes from the real record. |
| `_release_lease_or_warn` | The lease-side twin of the claim wedge: `release_lease`'s result is no longer discarded. |
| `kill_agent` | Fails closed with `lease_store_unavailable` when the lease store is unreadable or a reclaim is unpersisted. |
| `cli.lease_force` | Revalidate → fence → kill → clear now run inside **one** registry transaction, with the operation-id check first. Validating only at the final CAS left the fence and the kill unprotected against a legitimate lease handoff. |

## Offered, not done

- Converting `agent-messaging-protocol.md`'s ~113 `server_simple.py` line
  citations to symbol-based references. Symbol names are already authoritative;
  the line numbers drift within a few PRs.
- Caller-identity enforcement on `kill_agent` — same hazard class as R2, but
  lifecycle rather than delivery. Explicitly a non-goal here.
- A recipient-visible, non-actionable history query for guaranteed-path
  messages. R5 permits it only as a separate query, never as unread delivery.

## Historical notes from earlier phases

Phase B notes, kept because they explain why the code looks the way it does:

- **`follow_up_agent` now takes a required `idempotency_key`.** Every existing
  test call site was updated; a new one that omits it gets a validation error
  before any waiting, not a delivery.
- **`agent_busy` no longer exists as a returned reason.** Both former sites are
  bounded waits. A test that wants the old fast refusal should set
  `_DELIVERY_CALL_BUDGET_SECONDS = 0.0` and assert
  `queued(phase="pending")` — otherwise it burns the whole real budget in wall
  time, which is how the suite briefly went from 45 s to 178 s.
- **`agent_not_found` and `backend_session_missing` became `no_delivery_path`**
  with `state="record_removed"` / `state="no_backend_session"` (B3/R7).
- **`_LEASE_QUEUE_WAIT_SECONDS` and `_DELIVERY_CONFIRM_BOUND_SECONDS` are
  gone**, replaced by the single `_DELIVERY_CALL_BUDGET_SECONDS`. Keeping
  per-step bounds alongside a total would have let one call spend the
  advertised budget several times over.
- The direction guard is now evaluated **twice**: read-only up front (so a
  refusal still leaves the session byte-identical, which creating the durable
  record first would have broken) and again under the registry lock, which
  remains authoritative.

C1+C2 note for whoever picks up Phase B: **every agent record now needs
`spawned_by`**, and a test fixture that omits it gets `parent_unknown` rather
than reaching the code under test. Five existing fixtures were updated for
this; new ones should set `"spawned_by": "team-lead"` (the default test
`IDENTITY`) unless the test is about the guard itself.

Phase B did reuse the two things A3–A5 established, as intended:

- `pending_delivery` on the agent record is still the per-target "unconfirmed
  attempt" state, and `_reconcile_pending_delivery` is still what stops a retry
  re-sending. `deliveries.json` sits beside it as the *sender-side* record; the
  two answer different questions and both are needed.
- The bounded in-call wait R1 describes is now `_DELIVERY_CALL_BUDGET_SECONDS`
  (the old `_LEASE_QUEUE_WAIT_SECONDS` was folded into it). The FIFO queue wait
  and the busy wait share that one budget and produce the same
  `queued(phase="pending")` tail.

## Quality gates — all four, whole repo

```
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

There is no `.venv` in a fresh worktree; `uv sync` first.

All four gates are **green** as of the final-review fix commits (924 passed,
1 skipped).

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
5. **Never edit `server_simple.py` with a scripted whole-file operation.** Two
   separate agents damaged it two steps running. Once, a slice-replacement
   anchored on `mcp_config_path = _write_mcp_config(...)` — a line that occurs
   in **both** `_do_spawn` and `_prepare` — matched the wrong one and silently
   deleted ~600 lines spanning `send_message`, `read_messages` and
   `check_agent`. Once, a `git checkout src/claude_teams/server_simple.py`
   meant to revert a single mutation reverted the whole file and wiped a
   finished implementation. Both were recovered, but only because the author
   noticed. The file is ~2700 lines with repeated idioms, so anchors that look
   unique usually are not.

   Rules: prefer targeted edits; if you must script one, assert the anchor
   matches exactly once *before* writing; to revert a mutation, restore from a
   copy taken beforehand, never `git checkout` a file with uncommitted work.
   Afterwards verify with a symbol-set diff against `HEAD` **and** body hashes
   of the functions outside your intended scope — a green suite alone does not
   prove nothing was lost, because a deleted `@mcp.tool()` docstring changes
   the contract calling agents read (`_with_disk_note` appends to the
   *registered* description) while every test still passes.

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
