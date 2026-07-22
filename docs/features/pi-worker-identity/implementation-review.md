# Post-implementation review: Fix Pi worker identity clobbering (session hijack)

Reviewer: Claude Code (Opus family), independent of the implementer. Read-only
review of the uncommitted diff on branch `feature/pi-lead-autoload` against
`docs/features/pi-worker-identity/plan.md`, `plan-review.md` (F1–F8), and
`implementation.md`. Verified by reading the full call paths in
`src/claude_teams/server_simple.py` / `src/claude_teams/backends/pi.py` and by
running the whole-repo gates.

## Gates (whole repo, Linux)

- `uv run ruff check .` — **clean** ("All checks passed!")
- `uv run ty check` — **clean**
- `uv run python -m pytest -q` — **687 passed, 3 skipped**

## Focus-area verification

### 1. Identity resolution (`_resolve_identity`, server_simple.py:79-102)

Verified all three contract cases by reading the function and the tests:

- ROOT human lead (empty `AGENT_NAME`, no `WIN_AGENT_TEAMS_SESSION_DIR`) →
  `("team-lead", False)`; guard inactive. Regression tests
  `test_resolve_identity_root_lead_is_team_lead`,
  `test_root_lead_send_message_works`,
  `test_recovery_hint_retained_for_root_lead` pass.
- Named worker → its own name, `unresolved=False`, regardless of the signal.
- Empty `AGENT_NAME` + `WIN_AGENT_TEAMS_SESSION_DIR` set → sentinel
  `_UNRESOLVED_IDENTITY` (`"\x00unresolved-identity"`, line 76) +
  `unresolved=True`. The NUL guarantees the sentinel can never equal a real
  inbox name, binding identity, or `ROOT_LEAD_NAME`.

No path masquerades as `team-lead` under the subagent signal. Cross-checked the
signal's provenance: `WIN_AGENT_TEAMS_SESSION_DIR` is set **only** by
`PiBackend.build_env` (pi.py:423); the two pi extensions and the wake test only
*read* it; no `mcp.json` env block carries it. The pi-lead-autoload wake
feature's root-lead opt-in uses a *different* variable
(`WIN_AGENT_TEAMS_LEAD=1`, `pi-extensions/win-agent-teams-wake/index.ts:33`), so
a human pi lead opting into wake does NOT trip the guard. False-positive
residue is limited to a human launching a lead from a shell that literally
inherited a spawned worker's env — in that shell `AGENT_NAME` is inherited
non-empty too, so identity resolves to the worker name rather than unresolved.
Acceptable and consistent with plan-review Q4.

Residual false negatives (documented, accepted): (a) a clobbering config that
sets `AGENT_NAME` to a non-empty wrong value (e.g. literally `"team-lead"`)
still masquerades — env is authoritative by design; (b) the signal is pi-only,
so a hypothetical Claude/Codex identity clobber would still fall back to
`team-lead` — but those backends always deliver a literal `AGENT_NAME`
(`--mcp-config` file / `-c` identity args), per plan-review.

### 2. Guard coverage

Enumerated every identity-bearing entry point and its behavior under
unresolved identity:

| Tool | Guarded? | Behavior when unresolved |
|---|---|---|
| `send_message` (1610) | yes, before `_active_session_id()` | structured refusal, no inbox write |
| `read_messages` (1681) | yes, before session/inbox/cursor access | refusal; test proves lead cursor file never created |
| `resume_session` (2130) | yes, first statement of `_do_resume`, before `_session_id` rebind / `_persist_session_binding` | refusal, no rebind |
| `follow_up_agent` (1881) | no explicit guard | safe by construction: `_active_session_id()` → `""` (unresolved early-return) → `session_not_found` refusal before any `_write_mcp_config`/resume |
| `check_agent` / `kill_agent` / `list_agents` / `agent_status` / `agent_watch_paths` / `session_info` | no | session resolves to `""`; agent lookups miss (`_agents_file("")` does not exist) → clean failures/no-ops; read-only w.r.t. the lead |
| `spawn_agent` (1509) | **no — see Finding R1** | creates a NEW session + binding, then crashes at process launch |

### 3. `_recover_session_id` (F2)

Verified top-to-bottom (640-690): the unresolved early-return (654-659) sits
**after** the `_AGENT_SESSION_ID` return and **before** the binding-key match
(660-668), the candidate scan, the single-candidate auto-adopt (674-682), and
the nudge (683-689). `_pending_recovery` is reset to `{}` at entry, so the
child surfaces no nudge either.
`test_recover_session_id_no_silent_autoadopt_for_unresolved_child` is a real
red-first test: it seeds exactly the single-candidate/single-lead-history state
that old code would silently adopt (or nudge about), and asserts neither
`adopted_session` nor `recovery_hint` appears. An unresolved child can never
adopt a session silently or via nudge; `resume_session` (the explicit path) is
guarded separately.

### 4. Per-agent `--mcp-config`

- `_write_pi_mcp_config` (1263-1298): literal `AGENT_SESSION_ID`/`AGENT_NAME`/
  `AGENT_PARENT_NAME` + `CLAUDE_TEAMS_PERMISSION_MODE=bypass` in `env`,
  `directTools: true` at **server-entry** level (F4 honored). No `${` anywhere
  (test-asserted). Path `<session_dir>/mcp/<agent>.pi.mcp.json` — distinct from
  Claude's `<agent>.mcp.json` (collision test present) and under the session
  dir, not the repo tree. JSON written via `json.dumps`, no shell involvement.
- `PiBackend._mcp_config_args` (pi.py:324-340): returns
  `["--mcp-config", str(path)]` as **discrete argv tokens**, wired into both
  `build_command` (291) and `build_resume_command` (319); omitted entirely when
  the extra is absent (test-asserted). Windows-safe: pi launches via direct
  `node <cli.js>` (bypasses the `.cmd` shim), and even on the shim fallback the
  path carries no newline or `< > | & ^`.
- F5 invariant (higher-precedence project sources still shallow-override
  individual keys) is documented in the writer's docstring.

### 5. No regression to wake / Claude paths

- `_write_mcp_config` (Claude) byte-for-byte untouched; pi-specific keys proven
  not to leak into it (`test_pi_config_distinct_from_claude_config`).
- `_hook_extra` pi branch still calls `_ensure_pi_mcp_config()` and still emits
  `pi_state_extension_path` + `pi_wake_extension_path` when state hooks are on
  (updated guards tests pass); the per-agent config is written **before** the
  `WIN_AGENT_TEAMS_STATE_HOOKS=0` kill switch and returned even when the
  extensions are suppressed (`test_hook_extra_writes_pi_config_with_state_hooks_off`).
- The resume/follow-up path also flows through `_hook_extra`, so
  `build_resume_command` sees `pi_mcp_config_path` — resumed pi workers keep
  literal identity too.

### 6. Test quality

The new tests are real, not vacuous:

- The auto-adopt test fails against old code on *both* assertions regardless of
  whether auto-adopt is enabled (old code either adopts or writes the nudge).
- The `read_messages` test seeds the lead inbox and asserts the lead **cursor
  file does not exist** after the call — a concrete no-side-effect proof, not
  just a refusal-shape check.
- Config tests assert literal values, `${` absence, filename, server-entry
  `directTools`, and discrete argv positioning (`cmd[idx+1] == path`).
- Root-lead regression tests pin `team-lead` resolution, working
  `send_message`, and the preserved recovery nudge.

Gap: no test covers `spawn_agent` (or `follow_up_agent`) under unresolved
identity — the one write-path left unguarded (Finding R1/R3).

### 7. Docs accuracy

README and ADDING-A-BACKEND correctly describe the implemented two-mechanism
design (human pi lead → interpolated `~/.pi/agent/mcp.json`; spawned worker →
per-agent literal `--mcp-config`), the later-wins/whole-env-replacement merge
hazard, the `identity_unresolved` refusal behavior, and the never-hardcode
`AGENT_*` warning. The `"lead"` → `"team-lead"` drift is fixed. No superseded
`${}`-for-workers claim remains.

## Findings

### R1 (should-fix, not a blocker) — `spawn_agent` is unguarded

`server_simple.py:1509-1510`: `_do_spawn` calls
`_active_session_id(create=True)` with no `_require_resolved_identity()` check.
Failure scenario for an unresolved child that calls `spawn_agent`: the
unresolved early-return in `_recover_session_id` yields `""`, so `create=True`
**creates a brand-new session** (`_create_session`, 1233-1240) including a
binding file with `identity = "\x00unresolved-identity"`; the spawn then dies
at process launch with an unhandled `ValueError` ("embedded null byte") when
`AGENT_PARENT_NAME = IDENTITY` (SpawnRequest `lead_session_id`, 1546 →
`build_env`, pi.py:419) reaches `subprocess` env. Net effect: **no hijack** (the
orphan session is new and its sentinel-identity binding is excluded from the
lead's `_candidate_sessions`/`_distinct_binding_sessions`, both filtered on
`identity == IDENTITY`), but the child gets a raw traceback instead of the
structured refusal, and stray session/binding files accumulate.
Fix (one line + one test): add the same
`refusal = _require_resolved_identity(); if refusal is not None: return refusal`
at the top of `_do_spawn`.

### R2 (nit) — implementation.md slightly overstates "inert"

`implementation.md` (F3 disposition) says the sentinel renders
`spawn_agent`/`follow_up_agent` "inert". True for `follow_up_agent`
(`session_not_found` before any side effect); for `spawn_agent` the reality is
"orphan session dir + binding, then unhandled ValueError" (R1). Fix R1 and the
sentence becomes accurate; otherwise amend the doc.

### R3 (nit) — missing test for the unguarded spawn path

No test exercises `spawn_agent` under unresolved identity. Add one alongside
the R1 fix asserting the structured refusal and that no new session directory
is created.

### R4 (nit, accepted residual) — guard signal is pi-only and value-blind

(a) A clobbering config that sets `AGENT_NAME` to a wrong **non-empty** value
still masquerades — unavoidable, env is authoritative; (b) Claude/Codex have no
subagent signal, but always deliver literal identity. Both already implied by
the plan review; no code change requested — recorded here so the limitation is
explicit.

## Verdict

**Commit-ready for draft PR #34.** No blockers: the hijack bug is closed on
every path (spoofed `send_message`, inbox consumption, explicit
`resume_session`, and the silent single-candidate auto-adopt), the root-lead
and Claude/wake paths are regression-tested, and all whole-repo gates are green
(ruff clean, ty clean, 687 passed / 3 skipped). Recommend applying **R1** (one
guard line in `spawn_agent` + one test, per R3) before or immediately after
pushing to the draft — it is a robustness/UX fix, not a security hole.

**Acceptance gate before the PR leaves draft** (unchanged from the plan): the
live pi smoke test — spawn a real pi worker, verify its `send_message` records
`from: <its own name>`, and verify no `resume_session`/adoption ever touches
the lead's session. Pinned stack: pi 0.80.10 + pi-mcp-adapter 2.11.0.
