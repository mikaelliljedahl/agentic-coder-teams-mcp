# Independent plan review — semantic agent watcher

**Reviewer:** Claude Code (Opus) — `semantic-watch-plan-review`
**Artifact reviewed:** `docs/features/semantic-agent-watch/plan.md`
**Code reviewed:** `src/claude_teams/cli.py`, `src/claude_teams/server_simple.py`,
`src/claude_teams/backends/{claude_code,codex,process_base,process_manager}.py`,
`tests/test_cli_watch.py`, and the referenced contract/description test files.
**Date:** 2026-07-19

---

## Verdict

**Changes requested (conditional approval).**

The design is sound in its core idea — separate edge detection from readiness, keep
`(mtime_ns, size)` polling, make state-marker watching semantic, and add independent
inbox detection without consuming messages. The scope, risks, and TDD outline are
unusually well thought through, and the existing `test_cli_watch.py` suite stays green
under the new semantics (every existing test either ends in `waiting`, uses a non-state
custom pattern, or asserts the timeout path).

However, there is **one blocking finding**: the plan's central "nested lead identity is
respected" acceptance criterion cannot be met as designed, because `AGENT_NAME` is **not**
present in a spawned Claude Code agent's process/shell environment — so the watcher's
`os.environ.get("AGENT_NAME")` derivation silently resolves to the wrong inbox for any
nested Claude orchestrator. TDD case #6 as written would pass while the real requirement
fails. This must be resolved or explicitly re-scoped before implementation.

Several non-blocking findings and concrete recommendations follow.

---

## Blocking findings

### B1. `AGENT_NAME` is not in a spawned Claude Code agent's environment → nested identity routing is wrong

**Where:** Design §2 ("Derive the watcher reader identity"), Risks → "Identity mismatch"
("Backend launch code already supplies that environment for spawned agents"), Acceptance
criterion "Nested lead identity is respected", TDD case #6.

**Claim in the plan:** a nested orchestrator's background shell inherits `AGENT_NAME`, so
`reader = os.environ.get("AGENT_NAME","").strip() or "team-lead"` picks
`inbox-{nested-name}.jsonl`.

**What the code actually does:**

- `ClaudeCodeBackend.build_env` (`src/claude_teams/backends/claude_code.py:272-289`)
  returns **only** `CLAUDECODE`, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, and an optional
  capability var. It does **not** set `AGENT_NAME`.
- `AGENT_NAME` is injected only into the *MCP server subprocess* via the written
  `--mcp-config` env block (`server_simple.py:1144-1148`), never into the Claude Code CLI
  process itself.
- `process_manager.spawn_process` builds the child env as
  `merged_env = os.environ.copy(); merged_env.update(env)` (`process_manager.py:470-471`),
  and `env` is `build_env(request)`. So the spawned Claude process inherits the *spawner's*
  `AGENT_NAME`, not its own.

Consequences:

- **Root lead:** spawner (root MCP server) has no `AGENT_NAME` → child shell has none →
  watcher defaults to `team-lead`. Root works (its inbox *is* `inbox-team-lead.jsonl`).
- **Nested Claude orchestrator `orch`:** its own MCP subprocess has `AGENT_NAME=orch`, so
  when `orch` spawns a grandchild, `os.environ.copy()` leaks `AGENT_NAME=orch` into the
  grandchild's shell — but `orch`'s **own** Claude CLI process (the one that runs the
  `watch` background task) was spawned by the root, so it has **no** `AGENT_NAME`. The
  watcher `orch` mounts therefore resolves `reader = "team-lead"` and watches the *root's*
  inbox, never `inbox-orch.jsonl`. Messages addressed to `orch` never wake it.

**Contrast — Codex already does this correctly:** `CodexBackend.build_env`
(`codex.py:558-584`) sets `"AGENT_NAME": request.name` (plus `AGENT_SESSION_ID`,
`AGENT_PARENT_NAME`), and `merged_env.update(env)` makes it win over any inherited value.
So a nested *Codex* orchestrator's shell does carry the correct `AGENT_NAME`, but a nested
*Claude Code* one does not. The plan's identity assumption holds for Codex and fails for
Claude Code — the exact family the plan is built to wake.

**Why the TDD case does not catch it:** case #6 ("`AGENT_NAME=parent-agent` selects
`inbox-parent-agent.jsonl`") sets the env var inside the test harness. It proves the
watcher *honours* `AGENT_NAME` when present, but not that the environment is *populated*
in production. It gives false confidence in the acceptance criterion.

**Required disposition (choose one, and cover it with a test):**

1. **Preferred — make the launch path supply identity.** Add
   `"AGENT_NAME": request.name` (and, for symmetry with Codex, `AGENT_SESSION_ID` /
   `AGENT_PARENT_NAME`) to `ClaudeCodeBackend.build_env`. `merged_env.update(env)`
   guarantees it wins over any inherited value, fixing both the "absent" (root-spawned) and
   "leaked-from-parent" (nested-spawned) cases. Add `src/claude_teams/backends/claude_code.py`
   and a `build_env` assertion test to the plan's file list. **Verify** this does not
   disturb the nested MCP identity: the per-agent `--mcp-config` env still sets
   `AGENT_NAME` for the MCP subprocess, so that path is unaffected; only the CLI process
   and its child shells gain the (correct) value. Note this in the plan's risks.
2. **Alternative — make identity explicit on the command.** Add an optional
   `--reader NAME` (falling back to the env derivation) and have the coordinator pass its
   own name. This keeps backends untouched but expands the public CLI surface the plan
   otherwise tries to minimise, and pushes correctness onto every caller.
3. **Re-scope.** If nested Claude orchestration is genuinely out of scope for this feature,
   remove the "nested lead identity is respected" acceptance criterion and TDD case #6,
   and state the limitation explicitly. (I recommend against this — the plan's stated Goal
   names "a spawned agent acting as a nested orchestrator".)

This is blocking because an unresolved choice here means the implementation can ship,
pass its own tests, and still not satisfy the requirement it was written to prove.

---

## Non-blocking findings

### N1. Default-on inbox watching silently changes behavior for custom-pattern callers (backward compatibility)

Design §5 keeps inbox watching on even when a custom `--pattern` is supplied, and defers
`--no-inbox`. Today `win-agent-teams watch <dir> --pattern output-*.md` wakes **only** on
that artifact. After this change the same invocation *also* wakes on any message to the
derived reader. A coordinator using a custom pattern purely to await a build artifact will
now get spurious wakes from unrelated inbox traffic. TDD case #8 only asserts the output
file still wakes; it does not pin down whether inbox is additionally active for
custom-pattern callers, so the behavior change is neither decided nor tested.

**Recommendation:** decide explicitly and encode it in a test. Least-surprise option:
when a *custom* pattern is supplied, default inbox watching **off** (or require an opt-in),
while keeping it **on** for the default `state-*.json` case. If you keep it globally on,
add `--no-inbox` now (the plan's own "demonstrate a need" bar is met by this case) and a
test for it.

### N2. Codex post-wake recipe in `_DISK_CONTRACT_NOTE` becomes incorrect for message wakes

`_DISK_CONTRACT_NOTE` (`server_simple.py:864-884`) tells a **Codex** coordinator: "On
return, read the marker JSON directly from disk as the primary post-change read." Under the
new semantics a wake can be triggered by an inbox message with *no* marker change, so a
Codex coordinator that only re-reads the marker will miss the reason it was woken and will
never call `read_messages`. The plan says "update tool descriptions" but does not call out
this specific correction.

**Recommendation:** the plan should explicitly require the note (and `agent_watch_paths` /
`spawn_agent` docstrings) to instruct coordinators to branch on the emitted wake `reason`
(`message` → `read_messages`; `waiting` → status/marker read; `output` → inspect file), and
add/extend a `test_tool_descriptions.py` assertion for the `reason` contract. This is the
actual contract surface consumed by agents (per the note's own comment), so getting it
right matters as much as the code.

### N3. Timeout (exit 2) is the only recovery for an already-`waiting` agent — must be documented so coordinators don't hang

The edge-triggered design (correctly) does not wake on a `waiting` marker that is already
present at watcher startup. There is a TOCTOU window: the coordinator's pre-watch status
check sees `running`, the agent transitions to `waiting` before the watch baseline snapshot,
and no further edge occurs → the watch blocks to `--timeout` and exits 2. This is a latency
bug, not a hang, **only if** the coordinator re-checks status on exit 2. The plan relies on
"the coordinator should perform its normal status check before mounting the watcher" but
does not require documenting the exit-2 re-check.

**Recommendation:** state in the contract note that exit 2 means "no *new* edge within the
window — re-run the status check, the marker may already be `waiting`," and add a test that
a pre-existing `waiting` marker present at startup does **not** wake (locking in the
edge-trigger guarantee that the whole stale-marker rationale depends on). See also T3.

### N4. Two independent implementations of message-validity/unread rules will drift

The watcher runs in a separate `win-agent-teams watch` process and, per the plan, adds its
own "read-only unread calculation" using "the same validity rules as `read_messages`". The
server already has `_sender_message_count` / `_sender_unread_count`
(`server_simple.py:1037-1061`) **and** the grouping+clamping logic inside `read_messages`
(`server_simple.py:1432-1459`). Note these are subtly different: `_sender_message_count`
matches `msg.get("from") == sender` for a *named* sender, whereas `read_messages` (the
canonical consumer) skips any message whose `from` is not a non-empty string and groups all
senders. The watcher needs the *enumerate-all-unread-senders* behavior, i.e. the
`read_messages` grouping, not `_sender_message_count`. A hand-rolled second copy risks
diverging from the consumer it must agree with.

**Recommendation:** extract a pure, `Path`-based helper (e.g.
`unread_senders(inbox_path: Path, cursor_path: Path) -> dict[str,int]`) that both
`read_messages`/`_sender_unread_count` and the watcher import, so the validity rules
(dict + non-empty-string `from`) and the `consumed = min(cursor, total)` clamping
(`server_simple.py:1060`, `1457-1459`) are defined once. `cli.py` already imports
`server_simple` at module load (`cli.py:15`), so reuse costs nothing structurally. This
keeps the watcher's "same rules as `read_messages`" claim true by construction.

### N5. Cross-process concurrent read of the inbox while it is appended (Windows) is untested

The watcher reads `inbox-{reader}.jsonl` in a *different process* from the MCP server that
appends to it via `send_message` (`server_simple.py:1359-1360`). The existing
`_sender_message_count` reads happen in the same process as the writer; the watcher makes
this genuinely cross-process on Windows. CPython opens files with shared read/write access
so this should work, and the plan's "partial append → ignore malformed final line → next
poll re-evaluates" mitigation is correct. But there is no test exercising a read racing an
append.

**Recommendation:** the plan already lists malformed-row tolerance (case #7); add an
explicit assertion that a trailing partial/garbage line is ignored *and* that a subsequent
completed append (new size/mtime) is then detected. This directly exercises the
partial-append risk on the actual watched file.

---

## TDD adequacy

The nine listed red cases are good and mostly prove what they claim. Gaps worth closing so
the suite actually proves the requirements (not just the happy paths):

- **T1 (proves B1 end-to-end, or documents the boundary):** if disposition B1.1 is chosen,
  add a `ClaudeCodeBackend.build_env` test asserting `AGENT_NAME == request.name`. Case #6
  alone is insufficient (it stubs the env).
- **T2 (multi-predicate precedence):** the plan specifies `message > waiting > output`
  when several fire in one interval, but no test proves it. Add one where an inbox append
  and a `waiting` transition land in the same poll and assert `reason == "message"`.
- **T3 (edge-trigger guarantee):** assert a `waiting` marker already present at watcher
  startup does **not** wake (only a *transition* to `waiting` does). This is the crux of
  the stale-marker/`follow_up_agent` rationale (Risks → "Stale waiting marker") and is
  currently unproven; a regression here reintroduces false immediate wakes.
- **T4 (wake JSON shape):** assert the emitted object's schema for each reason
  (`{"reason":"waiting","agent":...,"path":...}`, `{"reason":"message","from":[...],
  "path":...}`, `{"reason":"output","path":...}`). Coordinators and N2's contract note
  branch on `reason`, so its shape is part of the contract and must be pinned.
- **T5 (semantic filtering under a custom state pattern):** §5 claims "a custom pattern that
  selects state files still applies semantic state filtering." Add a test with
  `--pattern state-*.json` (or a narrower state glob) proving a `running` write is ignored.
- **T6 (backward-compat of stdout):** the existing assertions (`"state-worker.json" in
  stdout`, `"output-report.md" in stdout`) still pass because the JSON embeds the path —
  good — but add an explicit note/test that the machine-readable line is a single JSON
  object, so a future change cannot silently revert the format the note advertises.

Also confirm the plan's intent that the **existing** `test_cli_watch.py` cases remain
unmodified. My read: all five pass unchanged under the new semantics (they end in
`waiting`, use a non-state pattern, or hit the timeout). If any existing test is expected to
change, the plan should say so; silent edits to existing green tests would be a red flag.

---

## Races / correctness (assessed OK, noted for the implementer)

- **Send-before-watch:** the "check unread once before the loop" step (Design §3) closes
  this correctly; combined with re-checking on inbox `(mtime_ns,size)` change, no append is
  missed. ✔
- **Cursor-only change does not wake:** correct — the watcher tracks the inbox file, not the
  `.pos.json`, and `read_messages` only rewrites the cursor, leaving the inbox mtime
  untouched. ✔
- **Baseline advance after ignored `running`:** required (Design §1) and correct; without it
  a stuck `running` edge would re-fire. A transient `running→waiting→running` inside one
  0.5 s poll can be collapsed to the last-seen `running` and thus not wake — acceptable,
  since a truly idle agent stays `waiting`. ✔
- **Over-large stored cursor:** even without explicit clamping, `count > consumed` stays
  false, so no false wake; matching `_sender_unread_count`'s `min(cursor,total)` (via N4's
  shared helper) is cleaner but not a correctness risk. ✔

---

## Windows-specific notes

- `(mtime_ns, size)` snapshotting is already the established, tested approach
  (`_snapshot_mtimes`, `cli.py:64-81`) and handles same-second atomic rewrites — keep it. ✔
- Path construction for `inbox-{reader}.jsonl` from the passed `session_dir` is fine via
  `pathlib`. ✔
- Cross-process read-during-append: see N5.
- No new subprocess/console interaction is introduced by the watcher, so the
  Windows-Terminal / console spawn machinery is unaffected. ✔

---

## Recommended concrete plan changes (summary)

1. **Resolve B1.** Adopt disposition B1.1 (set `AGENT_NAME`/`AGENT_SESSION_ID`/
   `AGENT_PARENT_NAME` in `ClaudeCodeBackend.build_env`, mirroring `CodexBackend`), add
   `src/claude_teams/backends/claude_code.py` to "Files expected to change", add a
   `build_env` test (T1), and record the reasoning + the "MCP subprocess identity
   unaffected" verification in Risks. If B1.1 is rejected, pick B1.2 or B1.3 and update the
   acceptance criteria/TDD accordingly. **Do not leave this implicit.**
2. **Decide the custom-pattern × inbox interaction (N1):** default inbox off under a custom
   pattern, or add `--no-inbox` now; add a test either way.
3. **Extend the description/doc work (N2):** require the `_DISK_CONTRACT_NOTE` (and
   `spawn_agent` / `agent_watch_paths` docstrings) to teach coordinators to branch on the
   wake `reason`, especially the Codex foreground recipe, and assert it in
   `test_tool_descriptions.py`.
4. **Document exit-2 recovery + add the edge-trigger test (N3/T3).**
5. **Share the unread/validity logic (N4):** extract one `Path`-based helper used by both
   the server and the watcher rather than a parallel implementation.
6. **Close the TDD gaps T2–T6.**

None of items 2–6 blocks starting implementation once B1 is dispositioned; they should all
be reflected in the updated plan and its TDD list before green begins.

---

## What is good (keep as-is)

- Edge-triggered semantics with baseline advancement; not treating a startup-present
  `waiting` marker as an edge (avoids the `follow_up_agent` false-wake).
- Watcher never advances cursors; `read_messages` remains the sole consumer — clean
  ownership, and the intentional immediate re-wake on undrained inbox is the right default.
- Preserving `(mtime_ns,size)` polling and exit-2-silent-on-timeout for caller
  compatibility.
- Embedding the path inside the wake JSON so existing substring-based stdout assertions and
  callers keep working.
- Honest risk section (stale marker, inbox parsing cost, partial append) — the analysis is
  strong; the one place it overreaches is the "backend launch code already supplies
  `AGENT_NAME`" claim, which B1 corrects.
