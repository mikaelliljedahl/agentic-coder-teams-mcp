# Implementation review — codex-trust-prompt

## Part A

Reviewer: Claude Opus (independent post-implementation review), 2026-09-26.
Scope: the uncommitted diff on `fix/codex-trust-prompt` against HEAD `c2868be`, plus the untracked
`tests/test_startup_diagnosis.py`. I reviewed it against `plan.md` rev 3 §3, all three rounds of
`plan-review.md` (round-3 dispositions 3 and 6 in particular) and `implementation.md`.

### Verification performed

- Focused run: `tests/test_startup_diagnosis.py`, `test_correlation_transport.py`, `test_follow_up_delivery.py`,
  `test_agent_status.py`, `test_agent_output.py`, `test_join_team.py` and `test_native_wake_flag_off.py`.
  **361 passed.**
- Full suite: **1882 passed, 6 skipped**. `ruff format --check`, `ruff check` and `ty check` all pass on Linux.
- A semantic JSON diff of `tests/fixtures/native_wake/flag_off.json` against HEAD. Only
  `tools.check_agent`, `tools.list_agents` and `tools.agent_status` differ.
- I decoded the new fixture as cp1252, which is what `Path.read_text()` does on Windows under Python 3.12.
  The em dashes come out garbled, so the golden no longer matches there (finding 1).
- `grep trust_cwd|projects=` over `src/` finds nothing. **Part B is not implemented here.** That is correct.

### Checklist results

| Area | Result |
|---|---|
| Launch-start capture (spawn) | `server_simple.py:3516-3518`. Mode and hook fields are computed first, `launch_started_at = time.time()` is taken on the line before `b.spawn`, and both are persisted in the appended record (`:3540-3541`). `spawned_at` is unchanged. OK. |
| Launch-start capture (follow-up) | `:4848-4853`. The capture happens after the old-PID graceful shutdown or kill (`:4839-4846`) and immediately before `plan.backend.resume`. It is passed to `_finalize_follow_up` (`:4398`), which refreshes the record on delivered/unconfirmed. On a failed resume the record is not written, so the metadata is discarded, as before. OK. |
| Marker comparison | `_startup_diagnosis` (`:6360-6414`) treats a marker as absent when `ts < launch_started_at` or `ts` is non-numeric/bool (via `_marker_timestamp`). The comparison is diagnostic-only; `_resolve_agent_state`/`_heartbeat_fields` inputs are untouched. OK. |
| Missing capability (R3 disp. 3) | `_state_hook_args` uses `getattr(...)`/`callable` and yields `[]`. Spawn with a minimal `_FakeBackend` records `hooks_wired=False` (`test_correlation_transport.py`). `ty` is green. The claude kill switch `WIN_AGENT_TEAMS_STATE_HOOKS=0` returns `[]`, so `hooks_wired=False`. OK (see finding 3 for the exception path). |
| Surface consistency | All five surfaces call the same helper with the record and the raw marker. `_empty_agent_check` returns `None`/`None` in both modes. No new binding resolution was added; the spy count is 3, all pre-existing. `state`, `stalled` and `heartbeat_age_s` are unchanged. OK. |
| Legacy / external | A missing `launch_started_at` or a non-bool `hooks_wired` gives `None`. `backend == "external"` gives `None`. Join records carry no launch fields. OK. |
| Docstrings (R3 disp. 6) | The exact tri-state predicate, the env override, `heuristic`, `folder-trust` and the clock caveat appear in all three tools. The headless hint says "CLI startup problem or hook failure". OK, with nits 6 and 7. |
| Part B | Absent. OK. |

### Findings

1. **Major — the `flag_off.json` golden was re-serialised with literal UTF-8, which breaks the golden test on Windows.**
   `tests/fixtures/native_wake/flag_off.json` (14 changed lines).
   - **What changed:** the file is 28 lines of diff, but only three values changed semantically: the three tool
     descriptions that gained the new fields. The rest of the diff converts every `—`/`→` escape into
     a literal `—`/`→`, apparently because the file was regenerated with `ensure_ascii=False`. At HEAD the file
     was pure ASCII; it now has 14 lines with non-ASCII bytes.
   - **Why it matters:** `tests/test_native_wake_flag_off.py:20` reads the file with `read_text()` and no
     encoding. On Windows under Python 3.12 (the `.python-version`) that decodes as cp1252: `—` becomes `â€”`,
     and `test_tools_list_identical`'s `== expected` fails. Linux CI stays green, so the regression hides on
     exactly the platform the repo targets.
   - **Why the change is needed at all:** the fixture is a golden of every tool description. Changing three
     docstrings legitimately requires updating those three values; nothing else should change.
   - **Fix:** regenerate with the default `json.dumps(..., ensure_ascii=True, indent=2)` (matching HEAD's
     format) so the diff is limited to the three description strings. Optionally also make the loader
     `read_text(encoding="utf-8")` so the test is encoding-independent.

2. **Minor — the follow-up ordering test cannot detect a capture after `resume`.**
   `tests/test_follow_up_delivery.py:401-431`. The `env` fixture freezes `server_simple.time.time` at a constant
   `1_000.0` (`:198`). As a result, `launch_started_at <= marker_ts` holds whether the capture happens before or
   after `resume()`: both are 1000.0. Plan §3.A5(3) asks the test to prove the capture precedes the child
   marker.
   **Fix:** inside this test, patch `server_simple.time.time` with a strictly increasing counter (for example
   `itertools.count(1000.0, 0.001)`) and assert `launch_started_at < marker_ts`. Optionally also record the
   `graceful_shutdown`/`kill_process` call time and assert the capture comes after it, which covers "after the
   old PID shuts down".

3. **Minor — in follow-up, an exception from an optional adapter capability now fires after the old PID is stopped.**
   `server_simple.py:4848-4850` (`_launch_mode_fields`), via `_state_hook_args` at `:297-300`.
   - **Why it matters:** the call sits outside the inner `try`, so a third-party adapter whose
     `state_hook_args` raises would kill the old agent, skip `resume` and `_finalize_follow_up`, and propagate.
     The outer `finally` still releases the lease. Built-in adapters are pure and cannot hit this. Still, the
     disposition-3 intent was that an unknown adapter must never break a launch because of this diagnostic.
   - **Fix:** make `_state_hook_args` fail-safe with `try: ... except Exception: logger.debug(...); return []`,
     which records `hooks_wired=False`. Alternatively compute `launch_fields` before the old-PID shutdown. Add
     one test with an adapter whose `state_hook_args` raises.

4. **Minor — the end-to-end tests do not cover a follow-up refreshing an existing `launch_started_at`.**
   The follow-up test starts from a record without launch fields. **Fix:** seed the record with
   `launch_started_at=1.0, hooks_wired=True, launch_interactive=False` and assert that all three values are
   replaced after a delivered follow-up. This is the path where a stale value would silently mask or mis-date
   a new launch.

5. **Nit — Pi's `state_hook_args` duplicates rather than reuses the builder's logic.**
   `backends/pi.py:520-523`. The plan says the method should return "the exact argv helper each builder uses".
   Pi re-implements the `pi_state_extension_path` branch of `_extension_args`. It is equivalent today, but the
   two can drift. **Fix:** extract `_state_extension_args(request)`, call it from both `_extension_args` and
   `state_hook_args`, and keep the wake extension separate.

6. **Nit — the `list_agents` docstring mentions a field that `list_agents` does not return.**
   `server_simple.py:6483` says "Neither field changes `state` or `stalled`", but `list_agents` has no
   `stalled` field. **Fix:** drop `or stalled` there.

7. **Nit — the reference doc is imprecise in two places.**
   - `docs/reference/agent-messaging-protocol.md:96` says "A successful follow-up refreshes all three fields".
     The finalizer also refreshes them on `delivery_unconfirmed` with a live new PID. **Fix:** say "a follow-up
     that records a new PID (delivered or unconfirmed)".
   - `:677` exceeds the file's wrap width. **Fix:** re-wrap it.

8. **Nit — `implementation.md` omits the fixture re-serialisation and the outstanding Linux smoke.**
   **Fix:** after fixing finding 1, note that `flag_off.json` changes only the three tool descriptions.
   Plan §3.A5 requires the Linux smoke (interactive codex in a fresh `/tmp` dir, which should show `True` plus
   the Codex hint) before the PR, and it is still marked TODO. Record its result before opening PR 1.

### Verdict

**APPROVE WITH CHANGES.**
- **Before PR 1:** fix finding 1 (a Windows-only red golden) and run the Linux smoke (finding 8).
- **Should fix in this PR:** findings 2-4. Each is small and test-local or a one-line guard.
- **Optional:** findings 5-7.

The core design matches rev 3 Part A and round-3 dispositions 3 and 6. Capture points precede process start on
both paths. The diagnostic uses only the record and the raw marker, all five surfaces agree, public
`state`/`stalled`/`heartbeat_age_s` are untouched, legacy and external records yield `None`, and Part B is
absent.

## Dispositions (lead, 2026-09-26) — Part A

| # | Disposition |
|---|---|
| 1 | Accepted (major). Regenerate `flag_off.json` with `ensure_ascii=True`, so only the three description values differ from main. Read goldens with `encoding="utf-8"` in `tests/test_native_wake_flag_off.py`, and check any other golden readers. |
| 2 | Accepted. The follow-up ordering test uses a monotonically increasing fake clock and asserts capture happens after the old-PID stop and before `resume`. |
| 3 | Accepted. The capability helper catches exceptions from third-party `state_hook_args` and returns `[]`, which gives `hooks_wired=False`. Test it, including that follow-up still resumes. |
| 4 | Accepted. Add a test that a follow-up replaces existing launch fields on the record. |
| 5 | Accepted. Pi shares its hook-arg logic with the builder. `list_agents` docstring drops the `stalled` mention. Fix the reference-doc wording. Smoke recorded in `implementation.md` (lead run: PASS). |

---

# Part B: opt-in Codex `trust_cwd` (post-implementation review, Claude Opus, 2026-09-26)

Scope: uncommitted diff on `feat/codex-trust-cwd` over origin/main dd922ff, plus untracked
`tests/test_trust_cwd.py`, reviewed against plan rev 3 §4, plan-review round-3 dispositions 1, 2, 4 and 5, and
`implementation.md` Part B. I re-ran `pytest` on the trust, codex, follow-up, agent-output and native-wake golden
suites (346 passed), plus `ruff check`, `ruff format --check` and `ty check`. All were green. I did not run a live
Codex smoke.

## Security checklist

| Requirement | Result |
|---|---|
| Default off, no escape hatch | **Holds.** The parameter defaults to `False`, and nothing reads an env var for it. The override is emitted only when `extra["codex_trust_cwd"] == "1"` (`codex.py:470`), and that is set only by `spawn_agent(trust_cwd=True)` (`server_simple.py:3588-3596`) or a record with `trust_cwd is True` (`:4403`). The bool check is strict, so the string `"true"` does not enable it. A non-trust follow-up asserts that the key is absent (`test_follow_up_delivery.py`, the `nonce_in_the_correct_transcript` test). |
| Refusal before any session or process | **Holds.** The preflight runs before `_active_session_id(create=True)` (`server_simple.py:3504-3533`). `test_trust_refusal_creates_no_session` checks for no session dir and no spawn request. |
| Override construction | **Holds, with finding 2.** It uses a TOML literal with the key `Path(cwd).resolve()`, ASCII-lowercased only on Windows. `'`, `"`, C0 and DEL are rejected in both the raw and the resolved path. Nothing writes `config.toml`. |
| Fail-closed transports and precedence | **Holds.** The order is backend, then headless, then shim, then direct launch, then path (`server_simple.py:323-341`). The builder re-checks independently (`codex.py:468-483`). |
| Pinning | **Holds.** The binary and mode are pinned into `extra`. The full resume command is built under the lease while the old PID is alive. Phase 2 re-checks mode and resolution before `_mark_attempt_sent` and before the old PID is stopped (`server_simple.py:4977-5008`). |
| Follow-up ordering and refusal shape | **Holds, with finding 4.** The check runs after `_reconcile_pending_delivery`/`_answer_reconciled_attempt` and before `if alive`. The refusal is wrapped in `_with_public_status(record)`, so it carries `status`, `phase` and `idempotency_key`. The reason is not a C2 reason, so the row stays queued/pending. No lease is held, and the old PID is untouched. |
| Lease-release helper | **Holds.** It takes `(session_id, agent_name, operation_id)`, keeps its two-attempt read-back, and is used in both the new catch (`:4897-4899`) and the finalizer (`:5091`). A test covers the failed first release. |
| `flag_off.json` | **Holds.** Pure ASCII, and only the `spawn_agent` value changed. |

## Findings

1. **Major (pre-existing, but Part B now approves this transport): PowerShell smart quotes break the W1 wrapper
   quoting.** Location: `src/claude_teams/backends/process_manager.py:167-169` and `src/claude_teams/backends/codex.py:52-56`.
   PowerShell treats U+2018, U+2019, U+201A and U+201B as single-quote characters. `_powershell_quote` doubles only
   ASCII `'`. A cwd such as `C:\x’; calc; ’` therefore ends the literal early in `Set-Location`, in `-C` and in the new
   `projects=` token. `unsafe_trust_path('/tmp/a\u2019b')` returns `False` (verified).
   The `-C`/`Set-Location` exposure pre-dates Part B and also affects prompts and env values. Part B, however,
   declares W1 "allowed" on the basis of a quote check that misses these characters.
   **Fix:** in this PR, also reject U+2018-U+201B and U+201C-U+201E in `unsafe_trust_path` and add them to the
   `test_trust_cwd_builder_rejects_unsafe_path` parameters. In a separate PR, make `_powershell_quote` double every
   single-quote variant, with a unit test. Add a `’` cwd case to the Windows smoke.

2. **Minor: TOCTOU between path validation and key construction.** Location: `src/claude_teams/backends/codex.py:478-480`.
   `unsafe_trust_path` resolves the path, and `_trust_args` then resolves it a second time to build the key. If a symlink
   in the path is retargeted between the two calls, an unvalidated resolved string (for example one containing `'`)
   reaches the TOML key and could inject extra `projects` entries.
   **Fix:** resolve once (`key = str(Path(request.cwd).resolve())`), then validate `request.cwd` and `key` with a
   character-only helper, and build the token from that same validated `key`.

3. **Minor: `CODEX_HOME` forwarding changes behavior for everyone, and relative values are not normalized.**
   Location: `src/claude_teams/backends/codex.py:686-689`.
   - It is safe for users who never set `CODEX_HOME`, because nothing is added.
   - For users who set it on the MCP server and launch through tmux, herdr or an existing WT window, children now use
     the server's home instead of the terminal's. This is arguably a fix, but it is not gated on `trust_cwd`.
     README and CHANGELOG do not mention it; only the reference doc does.
   - A relative `CODEX_HOME` would resolve against the agent's cwd, which is repository-controlled. A repository could
     then supply a user-level `config.toml`.

   **Fix:** export `os.path.abspath(codex_home)` and add a short README note about the propagation. Add a test that
   `CODEX_HOME` is absent from `build_env` when it is unset.

4. **Minor: the `_prepare` trust refusal drops a newly bound session id.** Location:
   `src/claude_teams/server_simple.py:4766-4776`. Every other refusal in `_prepare` goes through `_refuse`/`_wait`,
   which call `_save_agents_transaction` when `changed`. The trust refusal returns without doing so.
   **Fix:** before returning, add `if changed: _save_agents_transaction(session_id, agents)`.

5. **Minor: the successful trusted spawn is untested at the server level.** Location: `tests/test_trust_cwd.py`.
   There are refusal tests only. The follow-up tests build the record by hand (`_trusted_record`).
   **Fix:** add a `spawn_agent(trust_cwd=True)` success test. It should assert that the record has
   `trust_cwd: True` and `launch_interactive: True`, and that `request.extra` carries `codex_trust_cwd`, the pinned
   `codex_trust_binary` and `codex_trust_interactive="1"`. It should also assert that the default spawn records
   `trust_cwd: False` with no `codex_trust_*` extras.

6. **Minor (process): the Linux isolated-`CODEX_HOME` integration smoke has not been run.** Location:
   `docs/features/codex-trust-prompt/implementation.md`, Part B TODO. Plan §4.B7 requires it before the PR. It is the
   only check that installed Codex 0.157.1 accepts the `projects={...}` override and that it beats an explicit
   `untrusted` entry.
   **Fix:** run it, record per-step `cmp` results, and keep the PR draft until the Windows W1-W5 smoke also passes.

7. **Nit: a changed but safe binary reports `trust_cwd_unsafe_transport`.** Location:
   `src/claude_teams/server_simple.py:4996-4997`. For example, a Codex upgrade between the two phases gets this reason,
   with the remedy "Use the native Codex binary…". That is misleading.
   **Fix:** add a reason `trust_cwd_binary_changed` ("Codex resolution changed during the call; retry"), or put that
   text in `detail`.

8. **Nit: the direct-launch refusal is not limited to Windows.** Location:
   `src/claude_teams/backends/process_manager.py:58-60`. Plan §4.B1 rule 4 says "on Windows". On Linux, a stray
   `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1` refuses trust even though that path is unused. This is fail-closed and
   matches the docstring.
   **Fix:** gate it on `os.name == "nt"`, or record the deliberate deviation in `implementation.md`.

9. **Nit: the tests are narrower than the plan in two places.**
   - Windows lowercasing is not tested for non-ASCII characters staying unchanged (for example `Ä`).
   - The builder-level check for `\x1f` and a CR is not parameterized.

   **Fix:** add both parameters to the existing tests.

## Tests and existing-test changes

- `test_agent_output.py`: one added `"trust_cwd": False` key in an exact-record assertion. This is required by the new
  field and weakens nothing.
- `test_codex.py`: additions only.
- `test_follow_up_delivery.py`: additions only, plus one new negative assertion on the default path. No existing
  assertion was removed or loosened.
- Red-first evidence is recorded in `implementation.md` (13 failed, then F5 `KeyError`).
- The plan's B1-B8 cases are covered, except the success-path spawn noted in finding 5.

## Verdict

**APPROVE WITH CHANGES.** There are no blockers: the security model is correct, and the plan's contract for default
off, fail closed, pinning, follow-up ordering and lease release is implemented as approved.
- **Before opening PR 2:** fix findings 1 (the smart-quote rejection in `unsafe_trust_path`), 2 and 5, and run the
  Linux smoke (finding 6).
- **Should fix in this PR:** findings 3 and 4.
- **Optional:** findings 7-9.
- **Separate follow-up PR:** the general `_powershell_quote` fix from finding 1.
- Keep PR 2 in draft until Windows W1-W5 pass.

## Dispositions (lead, 2026-09-26) — Part B

Linux isolated-`CODEX_HOME` smoke (lead): trust override, CODEX_HOME forwarding and wrong-backend refusal PASS; follow-up FAILED with `binding_unverified` → new finding 10.

| # | Disposition |
|---|---|
| 1 | Accepted (major). `unsafe_trust_path` also rejects U+2018–U+201B (PowerShell single-quote equivalents). The general `_powershell_quote` hardening for `-C`/`Set-Location` is spun off as a separate follow-up (noted in implementation.md). |
| 2 | Accepted. Resolve once; validate and build the key from the same string. |
| 3 | Accepted. Forward `CODEX_HOME` as an absolute path (resolved against the server's cwd, never the agent's); test unset and relative; README note. |
| 4 | Accepted. Follow-up trust refusal persists a newly bound session id like the other refusals. |
| 5 | Accepted. Server-level success test for `spawn_agent(trust_cwd=True)` asserting the saved record and the pinned request fields. |
| 6 | Accepted. Smoke recorded in implementation.md (lead); follow-up re-run after finding 10. |
| 7 | Accepted. A changed-but-safe binary gets its own refusal reason, not `unsafe_transport`. |
| 8 | Accepted. Limit the direct-launch refusal to Windows. |
| 9 | Accepted. Add the missing test parameters. |
| 10 | **New (lead, major).** `agent_output.py:495` hardcodes `Path.home()/".codex"/"sessions"`. With `CODEX_HOME` set (now forwarded to children), rollouts land under `$CODEX_HOME/sessions`, so binding never verifies and `follow_up_agent` refuses with `binding_unverified`. Fix: resolve the Codex home the same way as the forwarded value (`CODEX_HOME` absolute, else `~/.codex`) everywhere rollouts are read. Red test with a non-default home. |
