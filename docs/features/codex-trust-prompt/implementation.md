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
- **TODO — Windows transport smoke (Windows operator):** Use a real Windows Codex install with a native `codex.exe`. In PowerShell, create an isolated home, copy `auth.json` and `config.toml` from the normal Codex home, set `$env:CODEX_HOME` to the isolated home **before starting the lead and MCP server**, and save a byte copy of that isolated `config.toml`. Create a cwd named `trust smoke space ;%!&^`; add both lowercase and mixed-case `[projects.'<absolute cwd>']` entries with `trust_level = 'untrusted'` to the isolated config, then refresh the byte copy. Confirm the spawned child reports the isolated `CODEX_HOME` (ask it to print the environment variable and compare the absolute path), including from an already-open Windows Terminal window. For W1, clear `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH` and `WIN_AGENT_TEAMS_NO_WT_TABS`, restart the lead/server with that environment, spawn `trust_cwd=true` in that cwd, and wait for a state marker. For W3, set `WIN_AGENT_TEAMS_NO_WT_TABS=1`, restart the lead/server, repeat the spawn and marker check. After each W1/W3 spawn and follow-up, run `fc.exe /b <saved-config-copy> <isolated-home>\config.toml` and require exit code 0. For W2, set `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1`, restart, and require `trust_cwd_unsafe_transport` with no process. For W4, set `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0`, restart, and require `trust_cwd_headless`. For W5, place a **copy** of the installed `codex.cmd` shim alone in a new empty directory at the front of `PATH` (no adjacent `node_modules`), restart, and require `trust_cwd_unsafe_transport` before launch. Restore the environment and remove the disposable home and shim copy when done. Keep the PR draft until W1-W5 pass.

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

The first CI run of PR #73 failed six `tests/test_backends/test_codex.py` Part B tests on `tests-windows`. All six were test portability bugs; the production behavior was correct. The tests expected POSIX separators and non-drive-rooted paths, and one expected mixed case where Windows builds the ASCII-lowercased key. On Windows the key is the resolved native path (backslashes, drive letter), lowercased to match Codex's lookup, and `CODEX_HOME` is forwarded as an absolute (drive-rooted) path. The tests now derive their expectations the same way, or compare separators neutrally. No production change. Linux gates were re-run: format, ruff and ty pass; pytest 1933 passed, 6 skipped.
