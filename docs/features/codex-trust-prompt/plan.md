# Plan (rev 3): Codex folder-trust prompt blocks spawned agents silently

Branch `fix/codex-trust-prompt` (worktree `wt-codex-trust-prompt`, from origin/main c2868be). `CX/` = `codex-rs/`
at `rust-v0.156.1` (installed CLI `codex-cli 0.157.1`); `SS` = `src/claude_teams/server_simple.py`;
`PM` = `src/claude_teams/backends/process_manager.py`. Rev 3 answers `plan-review.md` rounds 1-2 (§5).

## 1. Problem, scope, delivery

A `codex` agent spawned in a cwd Codex has never trusted stops on the TUI folder-trust screen: no state marker,
0% CPU, `check_agent` says `idle`; a human had to answer (PR #69 review;
`docs/features/native-session-wake/smoke-run-2026-09-25.md:41-42`). Answering persists trust to `~/.codex/config.toml`.

Two independently shippable parts, two PRs from this feature directory:
- **Part A** (PR 1, mergeable after the four Linux gates): a factual `no_marker_since_launch` diagnosis with a
  hedged, backend-aware hint on all five status surfaces. Needs nothing from Part B.
- **Part B** (PR 2, draft until the Windows smoke passes): opt-in, per-launch, non-persistent Codex `trust_cwd`.
Out of scope: pre-spawn advisory, env-wide opt-in, `trust_cwd` for claude-code/pi (explicit error), new `state`
values, transcript-activity heuristics, writing any user config file.

## 2. Research (verified)

- **When the screen shows.** The TUI runs `check_directory_trust` (`CX/tui/src/lib.rs:1747-1775`) before `App::run`
  (`:1950-1975`), so no hook can fire while it is up. It is skipped only for `trust_level == Trusted`
  (`CX/tui/src/onboarding/directory_trust.rs:69-70`); unset or `untrusted` shows it (`:71-82`). No CLI flag skips it,
  including `--dangerously-bypass-approvals-and-sandbox` (`CX/utils/cli/src/shared_options.rs:51-59`). `codex exec`
  never renders it (`CX/exec/src/lib.rs:962-970`): only interactive launches block.
- **Two lookups.** Active project: canonical then raw cwd, then main-repo root (`CX/config/src/config_toml.rs:850-875`;
  worktrees -> main checkout, `CX/git-utils/src/trust.rs:13-130`). On Windows the *lookup* key is ASCII-lowercased and
  `projects.get(lowercase)` wins before any case-insensitive match (`config_toml.rs:878-918`). Per-layer: every dir
  from cwd to the project root loads `<dir>/.codex/config.toml`, trusted by own key, else project-root key, else
  main-repo key (`CX/config/src/loader/mod.rs:1035-1074`, `:1640-1712`); linked worktrees may merge main-checkout
  hooks (`:1760-1810`). Precedence: SessionFlags 30 > project 25 > user 20, but legacy managed file/MDM 40/50 win
  (`CX/config/src/config_layer_source.rs:35-51`).
- **`-c`.** Merged before trust is decided (`CX/config/src/loader/mod.rs:338-339`, `:367`), never written back; only a
  human choice persists (`CX/tui/src/onboarding/onboarding_screen.rs:687-735`). Dotted keys split on every `.` and keep
  quotes (`CX/config/src/overrides.rs:22`, `CX/utils/cli/src/config_override.rs:56-58`); an inline-table value under key
  `projects` parses as TOML (`config_override.rs:95-102`) and deep-merges (`CX/config/src/merge.rs:94-98`).
- **What trust gates.** Project `.codex/config.toml` (MCP servers, providers), project hooks (can run before the
  prompt is handled) and exec policies (`CX/config/src/loader/mod.rs:1082-1100`); default sandbox/approval
  (`CX/core/src/config/permissions.rs:51-62`), which our `bypass` mode already overrides (`codex.py:276-278`).
  `--dangerously-bypass-hook-trust` (`codex.py:449-481`) does not touch directory trust.
- **This repo.** Hooks write `{"state","event","ts": time.time()}` from the child (`src/claude_teams/hooks.py:84-116`).
  With no marker `_resolve_agent_state` gives `idle` (`SS:318-348`) and `_heartbeat_fields` `(None, False)` (`SS:295-313`).
  `spawned_at` is set after `b.spawn` returns (`SS:3481`, `:3502`; Windows waits for the tab, `PM:1074-1217`) and after
  follow-up delivery (`SS:4343`), so it cannot date a launch. Hook argv comes from
  `ClaudeCodeBackend._hooks_settings_args` (kill switch `WIN_AGENT_TEAMS_STATE_HOOKS`, `backends/claude_code.py:251-264`),
  `CodexBackend._hook_override_args` (`codex.py:449-494`) and `PiBackend._extension_args` (`backends/pi.py:498-517`; the
  extension path may be absent, `SS:2970-2985`).
- **Other backends.** pi passes `-a` except in `require_approval` (`backends/pi.py:357-363`, `process_base.py:74-75`);
  claude-code has a workspace trust dialog inherited from trusted ancestors (2.1.283 strings; `bypassPermissions`
  effect unverified). Both get Part A only.

## 3. Part A: no-marker diagnosis (PR 1)

### 3.A1 Launch metadata
Persisted on the agent record by both launch paths, computed immediately before the process starts:
- `launch_started_at = time.time()`: captured on the line before `b.spawn(request)` (`SS:3481`) and stored in the
  record appended at `SS:3490-3520`; on follow-up, on the line before `plan.backend.resume(...)` (`SS:4794`, after the
  old PID is shut down at `:4786-4792`) and passed into the finalizer's `agent.update` (`SS:4337-4350`).
- `launch_interactive = process_manager.provides_tty(backend_name, is_interactive=b.is_interactive)` (`PM:392-405`,
  `PM:1008-1018`), the same call `CodexBackend._headless` makes (`codex.py:344-361`).
- `hooks_wired = bool(b.state_hook_args(request))`: a new `BaseBackend.state_hook_args` (default `[]`) returning the
  exact argv helper each builder uses (claude `_hooks_settings_args`, codex `_hook_override_args`, pi the
  `pi_state_extension_path` part of `_extension_args`). Same env, same request, so it equals what is in the argv.
- `spawned_at` is left as is. Legacy records without `launch_started_at` get `null` fields, never `true`.

### 3.A2 Definition
`_startup_diagnosis(agent, marker, alive, now) -> {"no_marker_since_launch": bool | None, "startup_hint": str | None}`.
Inputs are the record and the raw marker only; **no binding or transcript resolution**. For this diagnostic, a marker
without numeric `ts`, or with `ts < launch_started_at`, counts as absent. `no_marker_since_launch` is:
- `None` when the record has no `launch_started_at`/`hooks_wired` (legacy) or the backend is `external`;
- `True` iff alive, `hooks_wired`, `now - launch_started_at >= _first_marker_seconds()` (default 45 s, env
  `WIN_AGENT_TEAMS_FIRST_MARKER_SECONDS`) and no marker counts;
- `False` otherwise.
`startup_hint` is set only when `True`, always hedged:
- interactive codex: "No state marker since launch Ns ago. Likely causes: Codex's folder-trust prompt for <cwd>
  (look at the agent's terminal and answer it), a login prompt, or a slow start. Hooks may also have failed.";
- interactive claude-code: same shape, naming the workspace-trust dialog first;
- interactive pi: naming the project-trust selector (possible when launched with `require_approval`);
- headless (any): "No state marker since launch Ns ago: slow start, failed hooks, or a crashed CLI. Check the log."
Public `state`, `stalled`, `heartbeat_age_s` and marker handling for them are unchanged.

**Wall-clock limitation.** The parent's `time.time()` and the child's hook `ts` share a host clock but not a
monotonic one; a backward step (NTP, manual change) between capture and the first hook can make a genuine first
marker look stale, giving a false `True`. Documented in the docstring ("heuristic; clock changes can mislead").
A launch identifier in markers is deferred.

### 3.A3 Surfaces (all call the same helper with `agent` + `_read_state_marker`)
`check_agent` compact and `full=True` via `_compact_check_view(..., agent=...)` (`SS:1959`, caller `SS:4131-4180`);
`_empty_agent_check` (`SS:1792`) adds `null` fields in both modes; `agent_status` via `_agent_status_row` (`SS:6386`);
`list_agents(full=False)` via `_list_agents_row` (`SS:6296`); `list_agents(full=True)` in the loop at `SS:6356-6380`.
Docstrings of all three tools and `docs/reference/agent-messaging-protocol.md` define both fields.

### 3.A4 Files
`SS` (metadata capture, helper, five surfaces, docstrings); `backends/process_base.py` (`BaseBackend.state_hook_args`);
`claude_code.py`, `codex.py`, `pi.py` (overrides); tests below; reference doc; `implementation.md`.

### 3.A5 Tests, red-first order
1. `state_hook_args`: claude with/without `hooks_settings_path` and with `WIN_AGENT_TEAMS_STATE_HOOKS=0`; codex with
   overrides, empty, and both kill switches; pi with/without state extension path.
2. `spawn_agent` records `launch_started_at` < time of a marker written inside a fake `spawn()`; not flagged.
3. Follow-up: marker written inside fake `resume()` before finalisation; not flagged; `spawned_at` still set as today.
4. Stale marker (`ts < launch_started_at`, `state=waiting`) + 60 s: `True`; public `state` still `waiting`.
5. Boundary: marker `ts == launch_started_at` counts; `ts = launch_started_at - 0.001` does not; `now - launch == 45`
   is `True`, `44.999` is `False`; simulated backward clock step documents the false `True`.
6. Dead, external, legacy record: `False`/`None` as defined; `hooks_wired=False` stays `False` even if env changes later.
7. Hint text per backend and mode (codex/claude/pi interactive, headless); `None` when not `True`.
8. The same live record + stale marker gives identical values on all five surfaces, with no binding call (spy).
9. `state`/`stalled`/`heartbeat_age_s` unchanged versus main for every fixture above.
10. Tool descriptions mention `no_marker_since_launch`, "folder-trust", "heuristic".
Then focused tests, full suite, the four gates; Linux smoke: codex in a fresh `/tmp` dir shows `True` + codex hint.

## 4. Part B: Codex `trust_cwd` (PR 2, draft until Windows smoke)

### 4.B1 Contract
`spawn_agent(..., trust_cwd: bool = False)`; recorded on the agent. One `_trust_cwd_preflight(backend, cwd, mode)` returns
`None` or a reason. Precedence, first match wins:
1. backend is not codex -> `trust_cwd_unsupported_backend`;
2. launch is headless (`launch_interactive` false, incl. W4) -> `trust_cwd_headless` (trust would change project loading
   without clearing any prompt, so it is refused, not ignored);
3. binary is the `codex.cmd` shim (W5) -> `trust_cwd_unsafe_transport`;
4. `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1` on Windows (W2) -> `trust_cwd_unsafe_transport`;
5. path contains `'`, `"`, any C0 control (U+0000-U+001F, incl. newline) or DEL -> `trust_cwd_unsafe_path`.
Spawn runs it in `_do_spawn` before `_active_session_id(create=True)` (`SS:3409`): no session, name, prompt file or
process is created on refusal. The result is a structured error dict (reason + remedy), not an exception.

### 4.B2 Follow-up ordering
In `follow_up_agent._prepare`, insert the preflight **after the `if not binding.bound` refusal (`SS:4549-4554`) and
before the busy/idle branch (`SS:4592`)**, i.e. before `reserve_lease` (`SS:4661`), `_mark_attempt_sent` (`SS:4783`)
and old-PID shutdown (`SS:4786-4792`). It uses the stored `trust_cwd`, the recorded `launch_interactive` and the
*current* `provides_tty`: if they differ -> `trust_cwd_launch_mode_changed` (follow-up keeps the launch mode; no
silent switch to headless or interactive). Refusal goes through the existing `_refuse(reason, retriable=True, detail=...)`
(`SS:4527-4542`): the old agent stays alive, no lease is held, and the delivery record is untouched, so the caller can
retry after fixing the environment. Separately, wrap `_build_resume_request` (`SS:4689`), which runs after the lease is
granted, in `try/except` that calls `_release_lease_or_warn` (`SS:4850`) before re-raising.

### 4.B3 Encoding
Append `-c projects={ '<key>' = { trust_level = 'trusted' } }` before the prompt in both builders when
`extra["codex_trust_cwd"]=="1"`. `<key>` = `str(Path(cwd).resolve())`, ASCII-lowercased on Windows to match Codex's
lookup key (`config_toml.rs:878-918`). TOML literal strings only; with §4.B1 rule 5 no `'` or `"` can occur.

### 4.B4 Transports (all Codex launch paths)
| # | Path | `trust_cwd=True` |
|---|---|---|
| P1 | Linux terminal emulator: `shlex` via `_build_posix_shell_command` (`PM:153-157`) into `bash -lc` (`PM:2204`) | allowed |
| P2 | tmux (`PM:1541-1592`), same shlex command | allowed |
| P3 | herdr (`PM:3006-3025`), same shlex command | allowed |
| W1 | Windows Terminal tab + `.ps1` wrapper, default (`PM:1140-1150`, `_powershell_quote` `PM:160-162`), native exe | allowed after Windows smoke |
| W2 | WT direct launch (`PM:1122-1137`, `_escape_wt_passthrough` `PM:1223-1235`) | refused, rule 4 |
| W3 | `CREATE_NEW_CONSOLE` Popen (no WT, `WIN_AGENT_TEAMS_NO_WT_TABS=1`, WT failure; `PM:713-760`), native exe | allowed after Windows smoke |
| W4 | non-interactive Popen, `WIN_AGENT_TEAMS_INTERACTIVE_CONSOLE=0` (`PM:1020-1033`) -> headless | refused, rule 2 |
| W5 | `codex.cmd` shim fallback (`codex.py:284-335`, `:567-575`), any window type | refused, rule 3 (rule 2 first if headless) |

### 4.B5 `spawn_agent` docstring text (draft)
> trust_cwd (codex only, interactive launches only, default False): asks Codex to treat `cwd` as a trusted project for
> this launch and its follow-ups, via a per-process `-c projects=…` override; nothing is written to
> `~/.codex/config.toml`. Without it, an interactive Codex stops on its folder-trust prompt for a new cwd and no state
> marker appears until a human answers. Trusting runs repository-controlled code: `<cwd>/.codex/config.toml` (MCP
> servers, model providers), project hooks (which can run before your prompt is handled) and exec policies; for a
> linked worktree, main-checkout hooks may also load. Ancestor `.codex` directories are not trusted by it. It is
> intended to override a user-level `untrusted` entry for the cwd (Windows: not yet verified for mixed-case entries),
> and managed policy can override it, in which case the prompt still shows. Refused with a structured error for
> other backends, headless launches, the `codex.cmd` shim, `WIN_AGENT_TEAMS_CODEX_DIRECT_LAUNCH=1`, a cwd containing
> `'`, `"` or control characters, and on follow-up when the launch mode changed. Use only for checkouts you trust.

### 4.B6 Files
`backends/codex.py` (`_trust_args`, preflight helpers); `PM` (`codex_direct_launch_enabled()` accessor); `SS`
(`spawn_agent` param/record/preflight/docstring, `_prepare` preflight, `_build_resume_request` lease cleanup);
tests below; reference doc; README backend notes; `implementation.md` part B.

### 4.B7 Tests, red-first order
1. Builders: default emits no `projects=`; flag emits exactly one token before the prompt, spawn and resume; key is
   resolved path, lowercased under a Windows monkeypatch; paths with spaces, `.`, `;`, `%`, `!`, `&`, `^`, `\` stay one token.
2. Preflight precedence table: each rule alone, plus rule 2 vs 3 (headless shim -> `trust_cwd_headless`), rule 3 vs 5.
3. `'`, `"`, `\n`, `\t`, `\x7f` -> `trust_cwd_unsafe_path`.
4. `spawn_agent` refusal creates no session dir, no record, no prompt file, no process.
5. Follow-up with unsafe transport or changed mode: old PID alive, no lease in the leases file, delivery record unchanged
   and retryable, backend `resume` never called.
6. `_build_resume_request` raising after lease grant releases the lease.
7. Follow-up of a `trust_cwd` agent re-sends the override; of a default agent does not.
8. Tool description mentions "repository-controlled", "managed", "interactive launches only".
Integration (Linux, installed 0.157.1, isolated `CODEX_HOME` with auth copied): config with an existing `projects`
entry and an explicit `untrusted` dir; fresh cwd without the flag -> prompt + Part A hint; with the flag -> marker; the
`untrusted` dir with the flag -> marker; follow-up of the trusted agent -> marker; `cmp` shows `config.toml` byte-identical
after every step. Windows (real machine, before leaving draft): W1 and W3 with native exe, cwd with spaces and `;%!&^`,
plus user entries `c:\…` and `C:\…` including `untrusted` -> marker arrives; W2, W4, W5 refused. Then the four gates.

## 5. Review resolution map
| Finding | Where |
|---|---|
| R1 F1 timestamp race | §3.A1 capture points; tests A2, A3 |
| R1 F2 false positives | §3.A1 `launch_interactive`/`hooks_wired`, §3.A2 hedged hint, headless variant; A6, A7 |
| R1 F3 trust scope | §2 two lookups and precedence; §4.B5 |
| R1 F4 Windows argv | §4.B1 rules, §4.B4 table, Windows smoke |
| R1 F5 security text | §2 "What trust gates"; §4.B5 |
| R1 F6 advisory | dropped (§1) |
| R1 F7 surfaces | §3.A3; test A8 |
| R1 F8 tests | §3.A5, §4.B7 red-first lists, isolated `CODEX_HOME`, Windows gate |
| R1 F9 research | §2 (only TUI blocks; `<dir>/.codex/config.toml`) |
| R2 F1 follow-up ordering | §4.B2 insertion point, retryable refusal, lease cleanup; B5, B6 |
| R2 F2 surfaces disagree | §3.A2 no binding; stale marker absent for diagnostic only; A4, A8 |
| R2 F3 mtime/legacy | activity suppression dropped; field is factual (§3.A2) |
| R2 F4 Windows key case | §4.B3 lowercase key; §4.B5 qualified; Windows smoke mixed case |
| R2 F5 headless no-op | §4.B1 rule 2 refusal; §4.B2 mode kept; W4/W5 precedence in §4.B4; B2 |
| R2 F6 hooks/clock | §3.A1 `state_hook_args` incl. claude kill switch; §3.A2 wall-clock note; A1, A5 |
| R2 F7 `"` in path | §4.B1 rule 5; B3 |
| Q1-Q4 | per-spawn only; Codex-only error; no new state; no advisory (§1) |

## 6. Risks
- 0.156.1 source vs 0.157.1 installed: the Part B integration smoke is the check; repeat on Codex upgrades.
- Part A can say `True` for a slow, hooks-wired start or after a backward clock step; the fields are additive, hedged,
  and `state`/`stalled` are untouched.
- `Path.resolve()` may differ from Codex's WSL normalisation (`CX/utils/path-utils/src/lib.rs:18-21`); a mismatch leaves
  the prompt visible (fail-safe) and Part A reports it.
