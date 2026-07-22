# Implementation review — claude-lead-inbox-wake

Reviewer: Claude Fable 5 (orchestrating session).
Reviewed: the working-tree implementation on `feature/claude-lead-inbox-wake`
against the approved `plan.md`, with gates re-run independently.

**Process note (required disposition):** the repo workflow mandates independent
review by the opposite model family (Codex). Codex quota is exhausted this
week; the user directed that Fable (this session) reviews. Deviation is
intentional, user-approved, recorded.

## Verdict

**Approve — ready for PR.** The implementation is faithful to the approved
plan, the decision core is correct (I traced D0–D6 and the guard state machine
by hand), the one net-new externally-visible surface (`install_lead_wake`) is
safe and idempotent, and the whole-repo gates are green on independent re-run.
Two non-blocking notes below; neither requires a change before PR.

## Independent gate re-run (not the implementer's report)

Re-ran on this Linux VM (the CI/Linux gate):

- `uv run ruff check` → **All checks passed!**
- `uv run ruff format --check .` → **55 files already formatted**
- `uv run python -m pytest -q` → **688 passed, 3 skipped** (skips are the
  Windows-only `cmd` tests — expected on Linux)

Matches the implementer's claim. No red anywhere; no pre-existing breakage.

## What I verified against source (not just the report)

- **Decision table D0–D6** (`lead_wake.evaluate`, lead_wake.py:308–368) matches
  plan §2.2 row-for-row: D0 kill-switch, D1 fail-open-no-session, D2
  no-live-subagents, D3 unread→block(read), D4 armed→allow, D5 not-armed→block
  (arm), D6 guard fail-open.
- **F1 fix landed correctly.** `_scan_senders` (lead_wake.py:129–147) composes
  `read_inbox_by_sender` + `load_inbox_cursors` and derives `{total,cursor,
  unread}` in one read-only scan — it does **not** use `unread_sender_counts`.
  `_cursor_advanced` (lead_wake.py:245–264) keys progress on **cursor**, with an
  explicit comment on why an unread-delta would wrongly fail-open. Exactly the
  F1 resolution.
- **Guard state machine** (`_apply_guard`, lead_wake.py:267–305): I hand-traced
  first-block (count 0, fires), repeated no-progress continuations
  (`stop_hook_active` gates the increment; blocks at 1→2→3 then D6 allow at cap
  3, safely under the harness's hard 8), and the productive-wake path (cursor
  advance resets to 0 and the block proceeds unshortened — no first-wake-only
  degradation). Correct.
- **Never-unstoppable invariant** (`main`, lead_wake.py:400–420): the hook only
  ever prints a `block` decision or nothing, never `{"continue":false}`, never
  exits non-zero; D0/D1/D2/D6 are all fail-open. Confirmed.
- **Identity/nested-lead** (`_resolve_identity`, lead_wake.py:76–87): `AGENT_NAME`
  authoritative over the baked `--reader`, falls back to `team-lead` — a nested
  lead reads its own inbox (the Pi team-lead-only bug is not repeated). Never
  uses the Stop stdin `session_id`.
- **hooks wiring** (hooks.py): `_wake_hook_matcher` appended to `Stop` only; all
  other events keep the single `emit` group. Clean; matches plan §2.7 and spike
  probe d (no merge).
- **`install_lead_wake`** (server_simple.py): `_install_wake_hook` rebuilds
  `Stop` by filtering out any existing wake-token group then re-appending —
  idempotent (two installs → one group), `remove=True` drops only the wake
  group, unrelated events and unrelated `Stop` groups preserved, `scope`
  validated to `{project,user}`, reads existing settings via `_read_json_object`
  so a user's other settings survive. Safe.
- **Arming match** (`_command_matches_session`, lead_wake.py:150–165):
  separator-insensitive and session-scoped (token set + session-dir/basename),
  so a watcher for a different session does not count as armed.

## Notes (non-blocking)

- **N1 (observation, no fix).** The no-progress counter is not reset on the
  D4 (armed→allow) transition, so a stale count could carry across an
  arm-then-lose-watcher episode and bring D6 fail-open ~1 turn sooner. In
  practice this is self-correcting: any message drained while armed advances the
  cursor, and the next block's `_cursor_advanced` check resets the counter. The
  residual case (armed but nothing drained, then watcher lost) errs toward
  *stoppable*, which is the design principle. Safe to leave; a one-line reset on
  the D4 path would be a pure-nicety refinement, not a correctness fix.
- **N2 (TDD-discipline candor).** The implementer disclosed that for the
  cluster-C guard tests (#8/#9) test and production code first landed together,
  with the red captured retroactively (breaking `_apply_guard`, observing
  `'block'=='allow'` / `2==0`, then reverting). The red is real and
  reproducible and the behavior is correct and covered, but it was not observed
  strictly-before-code on the first pass for that one cluster. Accepted: the
  disclosure is exactly the honesty the workflow asks for, and the resulting
  guard logic is verified correct here independently.

## Outstanding (carried to PR description, not blockers)

- **GC1** (Claude Desktop's embedded harness runs the settings `Stop` hook) —
  the 1-minute manual M1 check remains the user's to run; every headless signal
  on harness 2.1.215 points to it holding.
- **M3** (harness re-invokes the idle lead on watcher exit; real
  `_watch_command_bash` recognised end-to-end) — interactive, unprovable
  headless; lives in the README manual smoke test. Same assumption the shipped
  watch recipe already relies on.

Approved to commit and open the PR.

---

## Addendum — post-merge rebase + Fynd 1 fix (reviewed after interactive test)

The interactive smoke test surfaced a real bug the headless spikes and the
single-session unit fakes structurally could not (a coverage gap I own): the
wake hook, auto-wired onto every spawned agent, let a **leaf worker count its
own registry record as a live subagent** → it hit D5 and armed a watcher for
its own inbox. Confirmed from live disk state (agents.json held only `worker1`;
`wake-progress-worker1.json` was written).

Also rebased the branch onto `origin/main` = `cf2748a` (PR #34, "auto-load Pi
wake + fix nested-lead targeting & worker identity"), which had squash-merged
after our branch was cut.

Verified independently on HEAD `6311fea`:

- **Rebase clean, nothing dropped.** History: `cf2748a` → `e5f9078` →
  `5ab60f7` → `6311fea`. Both features' symbols coexist in server_simple.py
  (`_write_pi_mcp_config`/`_AGENT_PARENT_NAME` from #34; `install_lead_wake`/
  `_group_has_wake_token` from us) and README (both the Pi-setup and the
  Claude-lead-wake sections). #34 did not touch `_DISK_CONTRACT_NOTE`.
- **Fynd 1 fix correct** (`_live_subagent_names(session_dir, identity)`,
  lead_wake.py): excludes self (`name == identity`), excludes terminal, and
  when any record carries a `parent` scopes to `parent == identity`; legacy
  fallback (no parent anywhere) still self-excludes. Traced: worker1 → `live=[]`
  → D2 allow (bug gone); top-level team-lead → `live=[worker1]` → proceeds to
  arm for its real child (correct). Parent recorded as `"parent": IDENTITY` on
  the spawn record (server_simple.py:1591), consistent with #34's `parent_name`
  convention.
- **Regression tests** added (`TestLiveSubagentScoping`): only-self→D2,
  self+sibling→D2, self+live-child→not-D2, terminal-child→D2, legacy-leaf→D2 —
  closing the coverage gap.
- **Gates green independently:** ty clean, ruff clean, 718 passed / 3 skipped.

Fynd 2 (top-level lead hook was never installed in the first test) was a
test-setup gap, not a code defect; the retest pre-installs the hook. Fynd 3
(worker's reply never reached any inbox) is orthogonal to this feature and is
re-checked in the retest now that #34's worker-identity fix is in the branch.

Fynd 1 fix approved.
