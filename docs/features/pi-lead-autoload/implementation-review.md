# pi-lead-autoload — independent post-implementation review

Reviewer: Claude Code (Opus), independent of the implementer.
Scope: `git diff main...HEAD` on branch `feature/pi-lead-autoload` (commit 9659bf0),
reviewed against `docs/features/pi-lead-autoload/design.md` and the locked user
decision (every spawned Pi agent gets the wake watcher; it watches ONLY its own
identity; reader = `AGENT_NAME or team-lead` via the CLI default WITHOUT
`--reader`; `--reader team-lead` never hardcoded).

## Verdict

**NOT approvable as-is — one blocker.** The extension-side §6 rework is correct
and well-structured, but the design (and the locked decision) rest on a premise
that is only half true: "the CLI default without `--reader` is
`AGENT_NAME or team-lead`". That is true for `watch`, but **`inbox-status`
hardcodes its `--reader` default to `"team-lead"`**, so for a nested lead every
status probe reads the wrong inbox. The end-to-end nested-lead wake therefore
still does not work against the real CLI, and the mocked TS tests cannot see it.
Everything else is in good shape; fixing the blocker is a small, contained
change.

## Findings

### F1 — BLOCKER: `inbox-status` without `--reader` probes `team-lead`'s inbox, breaking the nested-lead wake end-to-end

- Where: `src/claude_teams/cli.py:376-377` (unchanged by this branch) vs.
  `pi-extensions/win-agent-teams-wake/src/cli.ts` `runInboxStatus` /
  `src/state-machine.ts` (all `runInboxStatus` call sites pass
  `this.readerOverride`, normally `undefined` → no `--reader`).
- The `watch` command's reader default is env-based
  (`cli.py:259`: `os.environ.get("AGENT_NAME","").strip() or "team-lead"`), and
  `session-dir` reports the same env-based `IDENTITY`
  (`server_simple.py:70`). But `inbox-status` declares
  `reader: str = typer.Option("team-lead", "--reader", ...)` — a **hardcoded**
  `team-lead` default, not the ambient identity.
- Consequence for a nested Pi lead (`AGENT_NAME=worker-1`), traced through
  `state-machine.ts watching()`:
  1. `runWatch` (no `--reader`) correctly wakes on `inbox-worker-1.jsonl`
     (env default), and `isOwnInboxPath(w.path, dir, discovery.identity)`
     correctly matches.
  2. `runInboxStatus` (no `--reader`) then snapshots
     `inbox-team-lead.jsonl` + team-lead's cursor — the WRONG inbox.
  3. If team-lead's inbox has nothing unread (the normal case),
     `captureGeneration()` returns `{}` → the "arrive-then-consume" branch:
     **no injection, the nested lead is never woken** — the §6 bug survives in
     a new form.
  4. Worse, that branch calls `this.watchBackoff.reset()` and re-arms, and the
     next `watch` exits ~immediately (its initial unread check on the OWN inbox,
     `cli.py:264-274`, still fires because the message is unconsumed). Result:
     a **hot subprocess loop** (watch → session-dir → inbox-status, no backoff)
     until the message is drained by other means.
  5. If team-lead's inbox DOES have unread messages (shared session dir), the
     nested lead injects a **spurious wake naming team-lead's senders** and then
     runs its ACK budget against team-lead's cursor.
- Fix (either satisfies the locked decision; (a) is recommended and also what
  the design text assumes):
  - (a) Change `inbox_status`'s `--reader` default in `cli.py` to the same
    env-based `AGENT_NAME or team-lead` used by `watch`/`session-dir` (keep
    `_require_safe_reader` on the resolved value). One-line semantic change +
    a CLI test pinning it.
  - (b) Have the state machine pass `--reader <discovery.identity>` explicitly
    on `inbox-status` (and optionally `watch`). Not "hardcoded team-lead", so
    still compliant, but leaves the CLI's inconsistent default in place.
- Note: the extension README added by this branch ("shells out to the CLI
  without `--reader`, letting the CLI apply that same ambient default") is
  currently a false claim for `inbox-status`; it becomes true with fix (a).

### F2 — SHOULD-FIX: the TS test harness masks F1; no test pins the CLI default the design depends on

- Where: `pi-extensions/win-agent-teams-wake/test/harness.ts` /
  `test/state-machine.test.ts` ("§6: a nested lead ... wakes on its own inbox",
  "§6: shells out WITHOUT --reader").
- The harness returns canned `inbox-status` payloads regardless of reader, i.e.
  it models `inbox-status` as identity-aware — which the real CLI is not. The
  headline nested-lead test therefore passes with the wrong-inbox bug present.
  This is exactly the "assertion that would pass even if the code were wrong"
  class.
- Recommendation: alongside the F1 fix, add a Python CLI test asserting
  `inbox-status` WITHOUT `--reader` resolves the reader from `AGENT_NAME`
  (and falls back to `team-lead`), mirroring the existing `watch` reader tests.
  That pins the cross-language contract the extension now relies on. If fix (b)
  is chosen instead, change the TS assertion to require
  `--reader <identity>` on inbox-status calls.

### F3 — SHOULD-FIX (minor): no test for the new server-side wiring

- Where: `src/claude_teams/server_simple.py` `_pi_wake_extension_dir()` and the
  `_hook_extra` pi branch adding `pi_wake_extension_path`.
- The backend `_extension_args` side is tested (`tests/test_backends/test_pi.py`,
  both new tests are real assertions), but nothing covers: the
  `WIN_AGENT_TEAMS_PI_WAKE_EXTENSION` override, the missing-dir → `None`
  (spawn-still-proceeds) path, or that `_hook_extra("pi")` actually emits both
  keys. This mirrors a pre-existing gap for the state extension, but the branch
  adds new behavior with zero coverage of its wiring.

### F4 — NIT: wake extension is silently gated behind `WIN_AGENT_TEAMS_STATE_HOOKS`

- Where: `server_simple.py` `_hook_extra` — `WIN_AGENT_TEAMS_STATE_HOOKS=0`
  returns `{}` before the wake path is added, so disabling *state hooks* also
  disables the *wake* feature. Possibly intended as a single kill switch, but
  it is undocumented; a one-line mention in the `_hook_extra` docstring (or the
  extension README) would prevent surprise.

### F5 — NIT: empty-string reader override is inconsistent

- Where: `cli.ts` `readerArgs("")` omits `--reader` (falsy), while
  `state-machine.ts` `activeReader()` returns `""` (`??` keeps empty strings),
  so the path guard would look for `inbox-.jsonl` while the CLI uses its
  ambient default. Nonsensical input, unreachable from the shipped wiring;
  normalize (`opts.reader || undefined`) or ignore.

## What checked out clean

- **Activation guard** (`index.ts`): exactly per design §2 — plain `pi` with
  neither env var registers no handlers and starts no loop (verified: the guard
  runs before any `pi.on`/exec wiring); `WIN_AGENT_TEAMS_SESSION_DIR` (injected
  unconditionally-when-known by `backends/pi.py build_env`) activates spawned
  agents; `WIN_AGENT_TEAMS_LEAD === "1"` is a strict opt-in. The guard tests
  assert real observable behavior (registered handlers), with proper env
  save/restore.
- **§6 machine rework**: `discovering()`/`refresh()` now gate on
  `r.kind === "ok"` with the inbox bound to `discovery.identity`;
  `activeReader()` is only called when `discovery` is non-null (after a
  successful discover/refresh) so the non-null assertion is safe;
  `refresh()`'s session-change detection via `sessionId` is sound (identity is
  env-derived and cannot change within a session). `--reader team-lead` is
  nowhere hardcoded; `DEFAULT_READER` survives only as the
  `parseInboxStatus` JSON fallback, as documented.
- **Root-lead regression check**: none. A root lead has no `AGENT_NAME`, so
  `watch` and `inbox-status` both resolve to `team-lead` with or without the
  old explicit flag — behavior is bit-identical to the merged feature,
  including for F1's code path.
- **Backend `-e` wiring**: `_extension_args` emits per-extension `-e` pairs,
  included in both `build_command` and `build_resume_command` before the prompt
  args; paths travel as single argv elements (same mechanism as the proven
  state extension), so no Windows quoting/newline exposure; omission of either
  key degrades gracefully. Python tests assert both presence and exact absence.
- **Cross-platform**: `ownInboxPath`/`isOwnInboxPath` keep the
  separator-normalization from blocker 1 of the wake feature, now with a
  required identity parameter; Windows backslash and mixed-style cases remain
  tested, plus a nested-lead case.
- **package.json**: `"pi": {"extensions": ["./index.ts"]}` matches the design's
  §1a prerequisite; `private: true` retained per design.
- **Deviations** declared in implementation.md (no `pi-lead` subcommand, no
  `.mcp.json` change) are design-sanctioned options, not omissions.

## Conclusion

Fix F1 (recommend option (a): env-based default for `inbox-status --reader`,
matching `watch`/`session-dir`) plus the F2 pinning test, then this is
approvable. F3 is cheap and worth doing in the same pass; F4/F5 are optional.
The deferred Codex cross-review still applies per the repo workflow.
