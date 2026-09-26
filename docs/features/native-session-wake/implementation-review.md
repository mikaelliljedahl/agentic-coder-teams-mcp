# Native session wake — post-implementation review

Reviewer: Claude Opus (independent post-implementation review). Scope: the
uncommitted working tree on `feat/native-session-wake` (base `f63571a`), read
against `plan.md` rev 4, `plan-review.md` rounds 1–4 including R4-A, and
`implementation.md`.

VERDICT: APPROVED

There are no blockers and no majors. The findings below are minors and nits,
and they can be fixed before the PR without another review round. Every live
smoke is still outstanding (see the end of this file).

## What was verified

- **Flag-off identity was checked independently, not taken from the report.**
  I extracted `e5c8e98:src` with `git archive` and imported it through
  `PYTHONPATH`. That base produces tool descriptions identical to
  `tests/fixtures/native_wake/flag_off.json`: 21 tools normally and 5 under
  `EXTERNAL_ONLY`. Its `_build_join_prompt` output is also byte-identical to the
  golden prompt. The current tree with the flag off lists the same 21 and 5
  tools. With the flag on there are 22 tools; the extra one is
  `external_set_wake`.
- **The flag-off identity tests make real comparisons.**
  - The send result is compared as a literal dict, against a member that has a
    `codex_wake` record. The runner and `subprocess.run` are made to fail if
    called.
  - The spawn and resume environment is compared as `observed == [env]` for all
    three backends. No backend overrides `spawn` or `resume`, so testing
    `BaseBackend` covers production.
  - `main()` fails if `Thread.start` is called. There is no `native_wake` key in
    `session_info`, and no `native-wake-*` file is created.
- **Import-time side effects are inert.** Import only creates a
  `threading.Event`, a `Lock` and an idle `CodexMemberWake()`. Every other site
  is gated, including the docstring mutation (`server_simple.py:414-426`),
  `_DISK_CONTRACT_NOTE`, the join prompt, the scrub, `session_info`,
  `watch_member` and `session_activated`. Moving the `return` after the
  `_agents_transaction` block in `send_message`, `external_send` and
  `external_read` does not change behaviour: the result, the save semantics and
  the refusal early-returns are unchanged.
- **Lock order.**
  - Sends follow per-member → short agents lock (`_native_member_snapshot`) →
    release → revalidate (short) → queue. The agents lock is never held across
    the subprocess or the post.
  - `watch_member` runs after the `with` block. The registry lock is a leaf.
  - Test 26c uses a spy to assert that the per-member lock is never requested
    while the agents lock is held. The 200×200 stress test has a bounded join.
- **The rest of the design matches the plan.**
  - Generation keying and the R4-A tombstone (`native_wake.py:530-544`).
  - Catch-up only on a transition: a new `_Target` carries `catchup=True`.
  - The activation event is consumed before the snapshot is taken
    (`native_wake.py:335-337`).
  - Backoff after every failure, starting when the attempt completes.
  - H1b is evaluated before any env read or host resolution.
  - The owner lock is `flock(LOCK_NB)`, or `seek(0)` plus `LK_NBLCK` on Windows.
    It returns `False` only for contention.
- **Queue safety.** The queue call runs with `stdin=DEVNULL`, `cwd=Path.home()`,
  `CODEX_HOME` set to the reported home, a real timeout and UTF-8 capture. The
  notice interpolates only a sanitised `[A-Za-z0-9_-]` sender.
  - SQLite is opened read-only (`mode=ro`) with `timeout=0.5`, only if the file
    exists, and is closed before the queue call.
  - The test compares the mtime, size and file set before and after, and checks
    that no db file is created.
- **Spot-checked tests would fail on regression.**
  - If activation re-created the target, `test_activation_catchup…` would post
    twice.
  - Without generation keying, `test_generation_state_clear_and_failure` would
    return `coalesced`.
  - Without revalidation, `test_revalidate_before_subprocess` would call the
    runner.
  - Without the R4-A predicate, the tombstone send would hit the spies that are
    set to fail.
  - Without the event, `test_event_during_scan_survives` would lose S2.
  - No test uses a real Claude socket or a real `codex queue`. The only socket
    is a `tmp_path` `AF_UNIX` listener.

## Findings

1. **[MINOR] The "Linux/macOS" claim is false: Claude wake cannot work on
   macOS.**
   - Evidence: `server_simple.py:362`, `README.md:238`,
     `.claude/skills/external-member-join/SKILL.md:132`,
     `.claude/skills/external-member-invite/SKILL.md:103`, and the protocol
     reference.
   - Cause: `procinfo.resolve_nearest_host` (`procinfo.py:317-322`) walks
     `/proc` on every non-Windows OS, and there is no Darwin reader. On macOS,
     H2 therefore always returns `host_not_claude`.
   - Impact: the path fails safe, because the watcher still works. But tool
     descriptions, which are the only contract consuming agents read, promise a
     wake that never happens.
   - Fix: say "Linux-only (native Windows and macOS use the watcher)"
     everywhere. Add a test that pins H2's result when `/proc` is absent.
     Alternatively, add a `ps`/`sysctl` host reader as a follow-up (F6).
2. **[MINOR] The `codex queue` subprocess inherits the lead's full environment,
   including `CLAUDE_CODE_MESSAGING_SOCKET`/`_TOKEN` and the
   `AGENT_*`/`WIN_AGENT_TEAMS_*` identity variables.**
   - Evidence: `native_wake.py:588`.
   - Risk: if `codex queue` starts, rather than attaches to, an app-server, a
     long-lived Codex process then carries the lead's Claude channel and team
     identity. That is the same stale-environment class that §2.4 scrubs for
     spawns.
   - Fix: pop `CLAUDE_CODE_MESSAGING_SOCKET` and `CLAUDE_CODE_MESSAGING_TOKEN`
     from `environ` before the runner call, and consider `AGENT_NAME`,
     `AGENT_SESSION_ID`, `AGENT_PARENT_NAME` and `WIN_AGENT_TEAMS_SESSION_DIR`
     too. Assert their absence in `test_single_send_immediate_safe_shim_no_timer`.
3. **[MINOR] The notifier does per-member registry I/O every tick, even when the
   Claude channel is unavailable.**
   - Evidence: `native_wake.py:340` runs `member_alive`, which takes
     `_agents_file_lock` and parses `agents.json`, before the
     `channel.reason != "available"` return at `:346`.
   - When this happens: a Codex-hosted member server needs the flag on (for
     `external_set_wake`) and starts the notifier on Linux as
     `host_not_claude`. It then takes the lead's agents lock once a second, for
     no purpose.
   - Fix: return early from `tick()` when the channel is unavailable (or skip
     starting the notifier then), while still draining `_members`. It would also
     help to rate-limit `_LOG.exception` when `agents.json` is corrupt, so stderr
     is not spammed at 1 Hz.
4. **[MINOR] The contract does not cover a member MCP restart.**
   - A member target exists only after the member's server has handled an
     `external_read`, `external_send` or `external_set_wake` call (`watch_member`
     at `server_simple.py:3888`, `:3959`, `:4028`).
   - After a restart of a Claude-hosted member's MCP server, no notice arrives
     until the member makes one such call.
   - The restart sentence only covers the lead (`session_info`/`resume_session`).
   - Fix: add one sentence to the flag-on `external_read` note and to the
     `external-member-join` skill: "after an MCP restart, call external_read once
     to re-arm notices".
5. **[MINOR] There is no test of a member target that actually posts.**
   - Evidence: `test_no_target_no_recovery_member_drop`
     (`tests/test_native_wake.py:289`) only asserts that the target is dropped.
   - Nothing pins that a live member target posts with the `external_read … member_token`
     instruction, takes `native-wake-member-<name>.lock`, or does its baseline
     catch-up.
   - Fix: add one test that calls `watch_member`, appends to
     `inbox-member.jsonl`, ticks past the coalesce window, and asserts one post
     containing `external_read` and the member lock file name.
6. **[NIT] The `send_message` wording change uses a fragile `str.replace`, and
   its result is ungrammatical.**
   - Evidence: `server_simple.py:419-424`.
   - The result reads "No idempotency key, lease, durable delivery row, process
     resume is involved; no wake unless …". If the source sentence changes, the
     replace silently becomes a no-op, and nothing asserts that it applied.
   - Fix: replace with "…, or process resume is involved; there is no wake
     unless …". In `test_flag_on_tool_contract_and_watch_retained`, assert that
     `"or wake is involved"` is absent from the flag-on `send_message`
     description.
7. **[NIT] Test 15a does not assert what the notice says.**
   - `test_resume_auto_adopt_signals_and_wakes_before_tick` does not assert
     that the notice names the sender, which the plan requires.
   - Its "within one second before a tick" proof is also timing-dependent: the
     thread's first `tick()` can run after `resume_session`.
   - Fix: capture the text and assert `"alice (1)"`. Wait until the first idle
     tick has run, for example by polling
     `notifier.targets == {}` plus a short sleep, before calling
     `resume_session`.
8. **[NIT] Two lock-file names can collide.** A lead identity named
   `member-<x>` and a member named `<x>` both map to
   `native-wake-member-<x>.lock` (`native_wake.py:370-371`). This is harmless
   today, because both would contend on one lock at worst. Use distinct
   directories or a separator that cannot occur in a name, for example
   `native-wake.lead.<id>.lock`.
9. **[NIT] `codex_home` is trusted member input run as the lead.**
   `CODEX_HOME` selects which `config.toml` `codex queue` loads. A member on the
   same machine is already the same OS user, so this is not an escalation, but
   note it in the `external_set_wake` docstring and the protocol reference. The
   same applies to the redundant `record is None` test at
   `native_wake.py:535`: cosmetic.

No scope creep was found. `lead_wake.py`, `member_wake.py` and `watch` are
untouched, and no follow-up (F1–F5, §9.1) was implemented. All plan items in
§3 and §4 are present. The plan listed extending the filelock tests; that
coverage lives in `test_native_wake.py` instead, which is acceptable.

## Gate results (run by the reviewer, Linux, whole repo)

| Command | Result |
|---|---|
| `uv run ruff format --check .` | PASS: 91 files already formatted |
| `uv run ruff check .` | PASS: all checks passed |
| `uv run ty check` | PASS: all checks passed |
| `uv run pytest -q` | PASS: 1823 passed, 4 skipped (47.8 s) |

## Manual smokes still required before the PR

None of these has been run against this implementation. The 2026-09-25 smoke
run was of a prototype.

- **S1**: Claude Desktop lead ↔ Codex Desktop member round trip with the flag
  on and no human nudge.
- **S2**: a spawned Claude child's MCP server has its own non-empty socket
  (stem = child PID) after the scrub. This also confirms that an empty
  inherited value does not suppress Claude's own export.
- **S3**: Codex and Pi children report `host_not_claude`.
- **S4**: a `resume_session` switch, plus a lead restart with a backlog where
  the catch-up notice arrives immediately.
- **S5**: a bypass-mode lead gets notices without approval.
- **S6**: with `crossSessionInbound=refuse`, the notice is dropped silently and
  the watcher still wakes the lead.
- **S7**: with the flag off, there are no notices, lock files or queue rows.
- **W1**: Windows with the flag off: byte-identical behaviour and the full suite
  green.
- **W2**: Windows `codex queue` wake through the native binary and the `.cmd`
  fallback, with the notice intact.
- **W3**: Windows `unsupported_platform`, no pipe I/O, and the scrubbed spawn
  environment.
- **V1 and V2** (open, not blocking): an unloaded thread, and a queue during a
  busy turn. Record the results in the `external_set_wake` docstring.
- **If finding 1 is kept as "macOS"**: a macOS check that H2 resolves `claude`.
  It currently cannot.
