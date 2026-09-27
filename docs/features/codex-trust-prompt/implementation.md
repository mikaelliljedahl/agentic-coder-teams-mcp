# Implementation — codex-trust-prompt

## Part A: first-marker diagnosis

### Red/green evidence

- Red first: `uv run pytest -q tests/test_startup_diagnosis.py tests/test_correlation_transport.py::test_launch_metadata_precedes_marker_from_minimal_adapter tests/test_follow_up_delivery.py::test_follow_up_launch_metadata_precedes_child_marker` — **17 failed**. The failures showed the missing hook capability, launch fields, diagnosis helper and status fields, and tool descriptions.
- After implementation, the focused status, spawn and follow-up run passed: **36 passed**.
- First whole-repo pass found nine lint errors in new code, three test typing errors, and eight tests with exact old payload or tool-description snapshots. Those were corrected without changing the diagnostic predicate.
- Final whole-repo pass: **1882 passed, 6 skipped**.
- Review-fix red run: `uv run pytest -q tests/test_native_wake_flag_off.py::test_golden_is_ascii_for_windows_default_encoding tests/test_follow_up_delivery.py::test_follow_up_launch_metadata_precedes_child_marker tests/test_follow_up_delivery.py::test_follow_up_ignores_raising_optional_hook_capability tests/test_startup_diagnosis.py::test_list_agents_doc_only_names_returned_state_field` — **4 failed**. The ASCII assertion exposed the non-ASCII fixture, the optional capability raised through follow-up, and the list docstring named `stalled`. The tightened ordering test initially returned `agent_busy` because its live old-process fixture needed a waiting marker; after that test setup correction it passed with a strictly increasing fake clock.
- Review-fix focused green run: **133 passed** across the native-wake golden, follow-up ordering and capability tests, startup diagnosis tests, and Pi backend tests.
- The first whole-repo review-fix pass had format and lint failures confined to the new tests (two files needing formatting, `TRY003`, and `E501`); these were fixed. `ty check` passed, and pytest reported **1885 passed, 6 skipped**.

### Final design

Spawn captures TTY mode and the optional backend hook argv immediately before `b.spawn`, then captures `launch_started_at` on the line before process startup. Follow-up does the same after shutting down the old PID and immediately before `resume`; the finalizer stores the metadata when the replacement process is recorded. `spawned_at` keeps its prior capture point. Claude Code, Codex and Pi expose state-hook argv through `state_hook_args`; the shared base returns `[]`, and adapters without the optional method also yield `[]` and `hooks_wired=false`.

One helper uses the persisted launch metadata and raw marker for `no_marker_since_launch` and `startup_hint` on both `check_agent` forms, both `list_agents` forms and `agent_status`. The tri-state predicate and wall-clock limitation are in all three tool descriptions and the reference protocol. A stale `waiting` marker still controls public `state`; the new diagnosis does not alter `state`, `stalled` or `heartbeat_age_s`.

### Review fixes and Linux smoke

- No behavioral deviation from rev 3 Part A and round 3 dispositions 3 and 6. README was not changed because the approved plan lists its backend notes under Part B, not Part A.
- The `flag_off.json` golden was regenerated with `ensure_ascii=True` and is pure ASCII. Its diff against HEAD changes only the `check_agent`, `list_agents`, and `agent_status` description values. The golden reader now specifies UTF-8; no other golden reader without an explicit encoding was found.
- The follow-up test now uses a strictly increasing clock and checks `old PID stopped < launch_started_at < marker ts`; it also seeds old launch fields and verifies all three are replaced. A throwing third-party `state_hook_args` now logs at debug level, yields `[]`, records `hooks_wired=false`, and still allows follow-up to resume. Pi's builder and capability share one state-extension argv helper. The `list_agents` docstring no longer names `stalled`, and the reference doc covers delivered or unconfirmed follow-ups that record a new PID.
- Linux smoke (lead run, 2026-09-26): a worktree MCP server with a headless Sonnet lead spawned `trust-blocked` (Codex cheapest, interactive via herdr) in the untrusted cwd `<scratchpad>/untrusted-smoke`, and `trust-ok` in this trusted worktree. Both records had `interactive=true` and `hooks_wired=true`.
- After about 60 seconds, `trust-blocked` had no state marker. `check_agent` compact and full, `agent_status`, and `list_agents` compact and full all reported `no_marker_since_launch=true` with the hint: "No state marker since launch 56s ago. Likely causes: Codex's folder-trust prompt for <cwd> (look at the agent's terminal and answer it), a login prompt, or a slow start. Hooks may also have failed." `state` was idle (unknown in `agent_status`) and `stalled` was false, unchanged behavior.
- `trust-ok` wrote a Stop marker; all surfaces showed `no_marker_since_launch=false` and `startup_hint=null`. `~/.codex/config.toml` was byte-identical by SHA-256 before and after. Both agents were killed. **Result: PASS.**

### Validation commands

```text
uv run ruff format --check .   # pass: 93 files already formatted
uv run ruff check .            # pass: All checks passed
uv run ty check                # pass: All checks passed
uv run pytest                  # pass: 1885 passed, 6 skipped
```

### CI fix (Windows)

The first CI run of PR #72 failed on `tests-windows`:
`tests/test_agent_output.py::test_spawn_agent_persists_output_lookup_metadata` expected
`launch_interactive: True`, but the Windows runner has no interactive console, so
`provides_tty` returns False. The test now pins `process_manager.provides_tty`
to True, as `test_follow_up_delivery.py` already does. No production change.
Linux gates re-run: format, ruff and ty all pass; pytest 1885 passed, 6 skipped.

## Part B: opt-in Codex project trust

### Red/green evidence

- Red first: `uv run pytest -q tests/test_trust_cwd.py tests/test_backends/test_codex.py -k 'trust_cwd or spawn_doc'` had **13 failed**: missing parameter, preflight, argv override, path refusal and security description.
- Follow-up tests then covered changed mode, changed binary before old-PID shutdown, receipt reconciliation, override carry-forward, and retry/read-back after a failed first lease release. The final focused suite had **197 passed** (including the ASCII tool golden).
- Round-3 F5 red: `test_build_env_passes_isolated_codex_home_to_child` failed with `KeyError: 'CODEX_HOME'`; it passed after `build_env` exported the server's isolated home to the child wrapper.
- The first whole-repo pytest run was **5 failed, 1906 passed, 6 skipped**. Four simulated Windows Terminal tests exposed a regression in the direct-launch flag accessor, and one exact spawn-record assertion lacked the new `trust_cwd: false` field. Both were corrected; the affected tests then passed (**32 passed**).
- Review-fix red run: `uv run pytest -q tests/test_trust_cwd.py tests/test_backends/test_codex.py tests/test_agent_output.py tests/test_follow_up_delivery.py -k 'trust or isolated_home or changed_safe_binary or direct_launch_flag'` reported **11 failed, 30 passed**. Ten failures covered the accepted path, isolated-home, persistence, and refusal findings; one direct-launch test had an incomplete OS stub (`environ` missing). After correcting the stub and implementation, the same run reported **41 passed**.

### Final design and deviations

`trust_cwd` defaults to false and is persisted on the agent record. A true value runs a structured preflight before session creation, with backend, headless, shim, direct-launch, and unsafe-path precedence. The Codex builders add one process-only TOML literal `projects` override before the prompt. The key is the resolved cwd, ASCII-lowercased on Windows. Raw and resolved paths reject quotes, C0 controls and DEL. No code writes Codex's `config.toml`.

Follow-up reconciles a prior receipt before the trust preflight, which runs before the busy/idle branch and lease reservation. A refusal keeps the pending row's public status and identity and leaves the old PID untouched. The approved binary and mode are pinned in the request; the full resume command is built while the old PID is alive, then resolution and mode are checked again immediately before marking an attempt sent or stopping that PID. The builder also rejects unsafe trust transports. The extracted `(session_id, agent_name, operation_id)` lease-release helper retries and reads back in both the request-build catch and the delivery finalizer.

`CodexBackend.build_env` also forwards an explicitly set `CODEX_HOME` to the child. This is required for the isolated Windows Terminal smoke because an existing tab does not inherit the MCP server's environment. There is no behavioral deviation from the approved Part B contract; this environment propagation implements round-3 finding 5.

Review fixes: `CODEX_HOME` is converted to an absolute path using the server's cwd, and both the child environment and rollout readers use the shared home resolver. The builder resolves the trust key once and validates the exact key it emits. Unsafe paths now include PowerShell single-quote variants U+2018-U+201B; the builder also checks the raw path before resolving it, so NUL gets the structured trust error. The follow-up trust refusal persists a newly discovered backend session id, and a changed safe binary has its own retry reason. The direct-launch flag affects trust preflight only on Windows. Tests cover successful trusted and default server spawns, isolated-home output and binding, unset/relative `CODEX_HOME`, non-ASCII Windows key preservation, CR and U+001F paths, and changed-binary refusal.

Separate follow-up: harden the general PowerShell `_powershell_quote` helper for smart single quotes in `-C`, `Set-Location`, prompts and environment values. This is broader than the Part B trust-path guard and is not changed here.

### Validation and remaining smokes

The reference protocol and README describe the opt-in and security scope. The `flag_off.json` golden remains pure ASCII and changes semantically only in the `spawn_agent` description.

- **Linux isolated-`CODEX_HOME` smoke (lead, 2026-09-26):** A worktree MCP server was started with `CODEX_HOME` set to an isolated home. Its `config.toml` was a copy with an appended `[projects.<dir>] trust_level="untrusted"` entry; `auth.json` was symlinked. Interactive cheapest-tier Codex agents were launched through herdr:

  | Agent | Setup | Result |
  |---|---|---|
  | `tb-noflag` | Fresh cwd, no flag | No marker; `no_marker_since_launch=true` with the folder-trust hint. |
  | `tb-flag` | Fresh cwd, `trust_cwd=True` | Reached a marker, ran, and printed `CODEX_HOME=<isolated home>`. |
  | `tb-untrusted` | Explicit-untrusted cwd, `trust_cwd=True` | Reached a marker and answered OK. |
  | Wrong backend | claude-code with `trust_cwd=True` | Refused with `trust_cwd_unsupported_backend`. |

  The isolated `config.toml` was byte-identical afterwards (`cmp`). The real user config gained no `tb-*` entries. An unrelated `/tmp/claude-1000/nd-scratch` trust entry appeared from another session and is not caused by this feature. Follow-up on `tb-flag` failed with `binding_unverified` because rollout lookup ignored `CODEX_HOME`; fixed by finding 10 and re-run below.

  **Follow-up re-run after finding 10 (lead, 2026-09-26): PASS.** Setup was the same isolated `CODEX_HOME`. The lead spawned `tb-fu` in a fresh cwd with `trust_cwd=True`. After 45 s `check_agent(full=True)` showed `binding: bound`, `backend_session_id` set, and `last_message: "FIRST"`. `follow_up_agent(idempotency_key="tb-fu-2")` returned `status: "delivered"` with the same backend session. After resume, `check_agent` showed `last_message: "SECOND"` with `no_marker_since_launch: false`. The isolated and the real `config.toml` were both byte-identical before and after (`cmp`).
- **DONE (2026-09-27, all five PASS; see "Windows transport smoke W1-W5" below) — Windows transport smoke (Windows operator):** Use a real Windows Codex install with a native `codex.exe`. In PowerShell, create an isolated home, copy `auth.json` and `config.toml` from the normal Codex home, set `$env:CODEX_HOME` to the isolated home **before starting the lead and MCP server**, and save a byte copy of that isolated `config.toml`. Create a cwd named `trust smoke space ;%!&^`; add both lowercase and mixed-case `[projects.'<absolute cwd>']` entries with `trust_level = 'untrusted'` to the isolated config, then refresh the byte copy. Confirm the spawned child reports the isolated `CODEX_HOME` (ask it to print the environment variable and compare the absolute path), including from an already-open Windows Terminal window. For W1, clear `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH` and `WIN_AGENT_TEAMS_NO_WT_TABS`, restart the lead/server with that environment, spawn `trust_cwd=true` in that cwd, and wait for a state marker. For W3, set `WIN_AGENT_TEAMS_NO_WT_TABS=1`, restart the lead/server, repeat the spawn and marker check. After each W1/W3 spawn and follow-up, run `fc.exe /b <saved-config-copy> <isolated-home>\config.toml` and require exit code 0. For W2, set `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1`, restart, and require `trust_cwd_unsafe_transport` with no process. For W4, set `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0`, restart, and require `trust_cwd_headless`. For W5, place a **copy** of the installed `codex.cmd` shim alone in a new empty directory at the front of `PATH` (no adjacent `node_modules`), restart, and require `trust_cwd_unsafe_transport` before launch. Restore the environment and remove the disposable home and shim copy when done. Keep the PR draft until W1-W5 pass.

  PowerShell setup and byte check for that TODO (run in the shell that starts the lead):

  ```powershell
  $isolatedHome = Join-Path $env:TEMP ("wat-trust-" + [guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $isolatedHome | Out-Null
  Copy-Item (Join-Path $HOME ".codex\auth.json") $isolatedHome
  Copy-Item (Join-Path $HOME ".codex\config.toml") $isolatedHome
  $config = Join-Path $isolatedHome "config.toml"
  $cwd = Join-Path $env:TEMP "trust smoke space ;%!&^"
  New-Item -ItemType Directory -Path $cwd -Force | Out-Null
  $lower = $cwd.ToLowerInvariant()
  $upper = $cwd.ToUpperInvariant()
  @("", "[projects.'$lower']", "trust_level = 'untrusted'", "[projects.'$upper']", "trust_level = 'untrusted'") |
    Add-Content -Path $config -Encoding utf8
  $before = Join-Path $isolatedHome "config.before"
  Copy-Item $config $before
  $env:CODEX_HOME = $isolatedHome
  # After each W1/W3 spawn and follow-up:
  & fc.exe /b $before $config
  if ($LASTEXITCODE -ne 0) { throw "Codex changed isolated config.toml" }
  ```

Final whole-repo gates (after Part B review fixes):

```text
uv run ruff format --check .   # pass: 95 files already formatted
uv run ruff check .            # pass: All checks passed
uv run ty check                # pass: All checks passed
uv run pytest                  # pass: 1933 passed, 6 skipped
```

### CI fix (Windows), Part B

The first CI run of PR #73 failed six `tests/test_backends/test_codex.py` Part B tests on `tests-windows`. All six were test portability bugs; the production behavior was correct. The tests expected POSIX separators and non-drive-rooted paths, and one expected mixed case where Windows builds the ASCII-lowercased key. On Windows the key is the resolved native path (backslashes, drive letter), lowercased to match Codex's lookup, and `CODEX_HOME` is forwarded as an absolute (drive-rooted) path. The tests now derive their expectations the same way, or compare separators neutrally. No production change. A second CI run caught one more case: a fake `Path.resolve` matched `str(path)` against a POSIX string. It now compares `Path` objects. Linux gates were re-run: format, ruff and ty pass; pytest 1933 passed, 6 skipped.

### Merge with `main` (native downstream delivery, #74)

`main` gained native downstream delivery (the follow-up split into `_prepare` / `_commit_attempt` with native carriers such as `codex_queue`) and the Windows Terminal quoting hardening. Resolution:

- `spawn_agent` and `_build_resume_request` keep both the `codex_trust_*` extras and `_dispatch_extra(...)`.
- In `_prepare`, native eligibility is computed first; the `trust_cwd` preflight runs only when `native_method is None`, i.e. when the follow-up relaunches Codex through resume. A native carrier never restarts Codex, so it is not refused by trust checks. The resume-request build keeps the PR's lease-releasing `try/except`, and the trust `extra` is added only when a request exists.
- In `_commit_attempt`, the trust re-check runs only for `METHOD_RESUME` plans (native plans have `request=None`) and still runs before the attempt is marked sent.
- `_release_lease_or_warn` keeps the PR's `(session_id, agent_name, operation_id)` signature because `_prepare` releases before a plan exists; `main`'s three stage-2 call sites pass `prep.plan.agent_name, prep.plan.operation_id`.
- `process_manager` takes `main`'s direct-launch block (blocker check, `_wt_argv`) unchanged, gated by the raw `_env_flag(_CODEX_DIRECT_LAUNCH_ENV)`. The trust preflight (`server_simple._trust_cwd_preflight`) and `codex.py` use the PR's `codex_direct_launch_enabled()` (flag and `os.name == "nt"`). A direct-launch request is still refused for `trust_cwd` even though `main` may fall back to the wrapper at runtime.
- CI fix after the merge: the first resolution gated the `process_manager` block with `codex_direct_launch_enabled()`. `main`'s `tests/test_backends/test_wt_command_line.py` drives the WT-tab path on Linux by setting only the env flag, so the added `os.name` check turned direct launch off on Linux and the `qa` job failed 5 tests (`test_direct_launch_encodes_cwd_and_prompt`, the 3 `test_direct_launch_with_percent_pair_uses_wrapper_tab` cases, and `test_percent_pair_in_direct_and_wrapper_falls_back_to_classic_console`). Windows passed. Restoring `main`'s `_env_flag(...)` gate fixes it with no production change, because the WT-tab path runs only on Windows. Red was reproduced on Windows by making `process_manager.os.name` report `"posix"` through a pytest plugin (5 failed, 53 passed), and those tests then passed (58 passed).

Gates after the merge (Windows): `ruff format --check` pass, `ruff check` pass, `ty check` pass, `pytest` 2874 passed, 9 skipped.

## Merge review findings

An independent Codex review of the merge resolution (`45c89cd` + `ef1b370`) is saved as [merge-review.md](merge-review.md). Verdict: approve with non-blocking issues; 0 blocker, 0 major, 2 minor. Both minor findings are accepted and addressed.

### Finding 1 (minor, accepted): native metadata used the raw `CODEX_HOME`

`server_simple._effective_codex_home()` (sole caller: `_native_record_fields`, which pins `codex_home` on the agent record for the `codex_queue` carrier) returned the raw, stripped `CODEX_HOME`. The launch environment (`CodexBackend.build_env`) and the rollout readers use the shared `codex_home()` resolver, which resolves a relative value against the server's cwd. `verify_codex_thread()` rejects a relative home as `home_missing`, so with a relative `CODEX_HOME` native delivery was silently dropped and follow-up fell back to resume.

Fix: `_effective_codex_home()` now returns `str(codex_home())`. Compatibility:
- `CODEX_HOME` unset or empty: unchanged, `str(Path.home() / ".codex")` (pinned by `test_default_codex_home`).
- Absolute `CODEX_HOME`: the resolved form of the same path. On Windows a rooted path without a drive (`/new-home`) now becomes `C:\new-home`, which is what the child actually uses; `test_recovery_metadata_is_recorded_with_the_flag_off` now expects `str(Path("/new-home").resolve())`, so it holds on Linux and Windows.
- Whitespace-only `CODEX_HOME`: previously stripped to the default; now resolved like the launch path does (the child inherits the same value), so the record matches the child instead of disagreeing with it.

Red/green:
- Red: the new `tests/test_native_record_fields.py::test_relative_codex_home_is_pinned_resolved` (relative `CODEX_HOME=isolated-home`, spawn through `spawn_agent`, assert the record equals `str(codex_home())` and `verify_codex_thread` accepts it) failed with `assert 'isolated-home' == 'C:\...\isolated-home'`.
- Green: after the fix, `tests/test_native_record_fields.py` 19 passed; together with native selection, Codex dispatch, agent output and the Codex backend tests, 406 passed.

### Finding 2 (minor, accepted): trust_cwd x native carrier boundaries were unpinned

New file `tests/test_trust_cwd_native.py`, built on the fixtures of `tests/test_native_codex_dispatch.py` / `tests/test_native_selection.py` (real delivery and lease stores, the real `codex queue` runner with a fake `Popen`). The target is an eligible `codex_queue` Codex agent with `trust_cwd=True, launch_interactive=True`.

| Case | Test | Pins |
| --- | --- | --- |
| a | `test_trusted_native_delivery_skips_the_relaunch_preflight[launch_mode_changed]`, `[direct_launch]` | Delivered via `codex_queue` while the resume environment would be refused (`provides_tty` false, or `codex_direct_launch_enabled()` true, patched so it runs on Linux CI); no binary discovery, no resume, same PID/token/epoch, trust fields kept, lease released, 1 attempt. |
| b | `test_stage_two_loss_then_trust_refusal_never_resumes` | Thread verification lost at stage 2; the retry meets `_prepare`'s preflight: `trust_cwd_launch_mode_changed`, row `pending`, 0 attempts, no resume request built, no queue run, old child untouched, lease released. |
| c | `test_not_enqueued_native_attempt_then_trust_refusal` | `Popen` construction fails (`native_not_enqueued`); the retry is refused by `_prepare`: row `pending`, reason `native_not_enqueued`, 1 attempt (the reverted native one), not unresolved-native, no resume request built, old child untouched, lease released. |
| d | `test_resume_fallback_carries_trust_extras_and_a_new_epoch` | Same native failure, safe environment: resume request carries `codex_trust_cwd`, `codex_trust_binary`, `codex_trust_interactive` and a `dispatch_epoch` above the old one; record keeps `trust_cwd`/`launch_interactive` and takes the new epoch; 2 attempts, method `resume`. |
| e | `test_changed_binary_at_fallback_commit_is_refused` | Binary pinned in `_prepare` differs at commit: `trust_cwd_binary_changed`, discovery called exactly twice, no resume, 1 attempt, old child untouched, lease released. |

These pin existing behaviour, so they passed on first run (6 passed; also 6 passed with `WIN_AGENT_TEAMS_NATIVE_WAKE=1 WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` in the environment). Guard evidence, each mutation applied alone to `server_simple.py` and reverted (restored byte-identical):

| Mutation | Result |
| --- | --- |
| M1: drop `native_method is None and` from the `_prepare` trust preflight | 5 failed (a x2, b, c, e), 1 passed |
| M2: skip the `_prepare` preflight on the resume retry (`... and native_allowed and ...`) | 4 failed (b, c, d, e), 2 passed |
| M3: drop the commit-time `current_binary != pinned` recheck | 1 failed (e) |
| M4: drop `codex_trust_binary` pinning of the resume request extras | 1 failed (d) |

### Gates (Windows, after both findings)

- `uv run ruff format --check .`: pass (116 files).
- `uv run ruff check .`: pass.
- `uv run ty check`: pass.
- `uv run pytest` with `WIN_AGENT_TEAMS_NATIVE_WAKE` / `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM` unset: 2881 passed, 9 skipped (2874 + 7 new tests).
- With both flags set to `1` in the environment: 5 failed, 2876 passed, 9 skipped. The same 5 fail identically on `ef1b370`'s `server_simple.py`, so they are pre-existing and not caused by this change: `test_agent_output.py::test_spawn_agent_persists_output_lookup_metadata`, `test_backends/test_base_runtime.py::TestBaseBackendSpawn::test_calls_process_manager_with_command_and_env` and `::test_env_values_are_passed_unquoted_to_process_manager`, `test_backends/test_codex.py::TestCodexMcpIdentity::test_build_command_injects_identity_env_override`, `test_join_team.py::test_external_only_mode`. They assert exact env/record contents and do not clear the ambient native flags; CI never sets them. Recommended follow-up (separate PR): an autouse fixture that clears the native flags for tests that do not opt in.

### Skip count 9 vs 10

The run after `ef1b370` recorded 2873 passed, 10 skipped, against 2874 passed, 9 skipped after `45c89cd`. `ef1b370` changes no test and no skip condition. The extra skip is `tests/test_watch_command_discovery.py::test_watch_command_bash_executes_and_times_out_quietly`, which probes `bash -c "exit 0"` at runtime and skips when bash is not usable. Reproduced: run from PowerShell (no `bash` on PATH) it skips (17 passed, 1 skipped); run from Git Bash it passes (18 passed). The difference is the shell the suite was launched from, not the change. It is expected, and CI on Linux always has bash.

## Windows transport smoke W1-W5 (2026-09-27)

Host: Windows 11 Pro 26200, native `codex.exe` (`%LOCALAPPDATA%\Programs\OpenAI\Codex\bin`), no npm `codex.cmd` installed. Code under test: `ef1b370` (merge + CI fix; the later review commits change only native metadata and tests). Harness: [`evidence/trust_smoke.py`](evidence/trust_smoke.py). For every step it starts a **fresh** win-agent-teams MCP server over stdio with that step's environment (the documented "restart the lead/server" step), in its own lead workspace so it gets its own session. It follows the setup above: an isolated `CODEX_HOME` with copied `auth.json`/`config.toml`, the cwd `trust smoke space ;%!&^` with lowercase and uppercase `untrusted` project entries, and `fc.exe /b` against the saved copy after every spawn and follow-up. W5 uses a minimal `codex.cmd` (`@echo off` + native exe `%*`) alone in a fresh directory prepended to `PATH`, because no installed shim exists; `discover_binary()` then resolves the `.cmd` since no `node_modules` sits next to it.

| Step | Environment | Result |
|---|---|---|
| W1 | WT tab (flags cleared) | PASS. Child printed the isolated `CODEX_HOME` (absolute path match), marker `waiting`, `binding: bound`, `no_marker_since_launch: false`. Follow-up `delivered`, then `SECOND`. `fc.exe` exit 0 after spawn and after follow-up. |
| W3 | `WIN_AGENT_TEAMS_NO_WT_TABS=1` (new console) | PASS, same checks as W1. |
| W2 | `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1` | PASS. `{"success": false, "reason": "trust_cwd_unsafe_transport"}`; `list_agents` has no `w2`. |
| W4 | `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` | PASS. `trust_cwd_headless`; no agent registered. |
| W5 | `codex.cmd` first on `PATH`, no flags | PASS. `trust_cwd_unsafe_transport` before launch; no agent registered. |

W1 and W3 ran twice, once per follow-up path:

- With `WIN_AGENT_TEAMS_NATIVE_WAKE=1` / `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` (the Desktop configuration), the follow-up went in place via `method: "codex_queue"` (`replaced_existing: false`, same PID).
- With both flags unset (`SMOKE_RESUME=1`), the follow-up took the resume path: Codex was relaunched with the trust overrides (`replaced_existing: true`, new PID, same backend session) and answered `SECOND`. The isolated `config.toml` was still byte-identical.

The real `~/.codex/config.toml` was byte-identical before and after every run. The disposable home and shim directory were removed by the harness.

Reproduce: `.venv\Scripts\python.exe docs\features\codex-trust-prompt\evidence\trust_smoke.py [W1 W2 W3 W4 W5]` (set `SMOKE_RESUME=1` for the resume path). W1 opens a Windows Terminal tab and W3 a console window.

## Linux smoke after merging main (2026-09-27)

The earlier Linux smokes ran on 2026-09-26, before `origin/main` (#74, #75
native downstream) was merged in. This re-run covers the merged code.

- **Host:** Linux (Omarchy), codex-cli 0.157.1, herdr launcher
  (`WIN_AGENT_TEAMS_LINUX_LAUNCHER=herdr`).
- **Code under test:** `0c55c08`.
- **Harness:** [`evidence/trust_smoke_linux.py`](evidence/trust_smoke_linux.py).
  It is the Linux counterpart of `trust_smoke.py`, and each step starts a
  fresh MCP server over stdio. The setup:
  - an isolated `CODEX_HOME`, with copies of `auth.json` and `config.toml`;
  - the cwd `trust smoke space ;%!&^`, with an explicit
    `trust_level = 'untrusted'` entry;
  - a byte comparison of the isolated `config.toml` after every spawn and
    follow-up.

| Step | Setup | Result |
|---|---|---|
| L0 | Control: no `trust_cwd` | PASS. After 75 s: no marker, `no_marker_since_launch: true`, and the folder-trust hint names the cwd. The prompt really does block, so the flag is what unblocks it. |
| L1 | `trust_cwd=True`, native flags on | PASS. The child printed the isolated `CODEX_HOME` and `FIRST`, with `binding: bound`. The follow-up was `delivered` via `method: "codex_queue"` (`replaced_existing: false`, same PID), then `SECOND`. The config was byte-identical after both steps. |
| L1R | `trust_cwd=True`, native flags off | PASS. The follow-up took the resume path (`replaced_existing: true`, new PID, same backend session), then `SECOND`. The config was byte-identical. |
| L2 | `backend=claude-code` | PASS. Refused with `trust_cwd_unsupported_backend`, and no agent was registered. |
| L4 | cwd containing U+2019 | PASS. Refused with `trust_cwd_unsafe_path`, and no agent was registered. |

The real `~/.codex/config.toml` was byte-identical before and after
(`real_config_unchanged: true`; SHA-256 unchanged). All agents were killed,
and the disposable home was removed by the harness.

Reproduce:

```bash
.venv/bin/python docs/features/codex-trust-prompt/evidence/trust_smoke_linux.py [L0 L1 L1R L2 L4]
```

## Merge re-review

The Codex re-review of `cee9d98`/`72bc7ca`/`758e3cf` approved with 0 new findings ([merge-rereview.md](merge-rereview.md)). CI on `758e3cf`: `qa` and `tests-windows` pass.
