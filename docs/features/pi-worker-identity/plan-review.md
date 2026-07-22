# Plan review: Fix Pi worker identity clobbering (session hijack)

Reviewer: Claude Code (Opus family), independent of the plan author. Read-only
review of `docs/features/pi-worker-identity/plan.md` against the source tree
(branch `feature/pi-lead-autoload`, clean) and the installed pi stack
(pi 0.80.10, pi-mcp-adapter 2.11.0, decompiled bundles in `/tmp/jiti/`).

## Verification of the root-cause analysis

All of the plan's mechanism claims were independently re-verified. The analysis
is correct and precisely cited:

- **Config source order** — confirmed in `pi-mcp-adapter-config` `getConfigSources`
  (line 195): shared-global (`~/.config/mcp/mcp.json`), pi-global
  (`~/.pi/agent/mcp.json`, readPath replaced by the override), shared-project
  (`<cwd>/.mcp.json`), pi-project (`<cwd>/.pi/mcp.json`). `loadMcpConfig`
  (line 186) merges in that order with later-wins semantics.
- **Shallow server-entry merge** — confirmed: `mergeServerMaps` (line 259) does
  `merged[name] = {...base[name], ...definition}`, so a later source with an
  `env` key replaces the entire env map. An empty `env: {AGENT_NAME: "", ...}`
  in `<cwd>/.mcp.json` therefore wipes any earlier-source identity env.
- **Subprocess env** — confirmed: `server-manager` `resolveEnv` (line 476)
  builds `{...process.env, ...interpolateEnvRecord(definition.env)}`; the
  correct inherited `AGENT_NAME` is overridden by the merged config's literal
  `""`. `interpolateEnvVars` (utils line 62) maps `${VAR}` →
  `process.env[VAR] ?? ""`, and a literal `""` stays `""`.
- **`--mcp-config` redirects only source #2** — confirmed:
  `getPiGlobalConfigPath(overridePath)` (config line 92) replaces the pi-global
  readPath; sources #3/#4 still merge afterwards. So the plan is right that the
  per-agent file alone is insufficient while a clobbering `env` block exists in
  `<cwd>/.mcp.json`.
- **Flag safety** — confirmed: the adapter registers the flag
  (`index` line 85 `pi.registerFlag("mcp-config", ...)`) and `initializeMcp`
  reads it (`init` line 37). Live check on this machine:
  `pi --mcp-config /nonexistent.json --version` exits 0 on pi 0.80.10 — pi does
  not reject the flag, even with a bogus path.
- **Server-side citations** — all spot-checked and accurate:
  `IDENTITY` fallback (`server_simple.py:70`), `_write_mcp_config`
  (1169-1186), `_ensure_pi_mcp_config` (1211-1249) and its call before the
  kill switch (1305-1306), `send_message` `from: IDENTITY` (1500),
  `resume_session` unauthenticated rebind (2012-2019), `_recover_session_id`
  early return on `_AGENT_SESSION_ID` (584-585) and the nudge (609-615),
  `PiBackend.build_command`/`build_resume_command`/`build_env`
  (`backends/pi.py:281-322, 390-406`), `_hook_extra` also runs on the
  follow-up/resume path (1859) so `build_resume_command` will see the new
  extra key.
- **Guard signal** — confirmed: `WIN_AGENT_TEAMS_SESSION_DIR` is set only by
  `PiBackend.build_env` (pi.py:405), appears in no `mcp.json` `env` block, and
  survives into the server subprocess via `resolveEnv`'s `process.env` spread.
  Claude/Codex `build_env` do not set it, and Claude/Codex workers always carry
  a literal `AGENT_NAME`, so the guard cannot fire for them.

## Findings

### F1 (should-fix, factual error with design consequences) — `.mcp.json` is gitignored, not tracked

Plan line 55-56 states the repo `.mcp.json` is "tracked, not gitignored".
This is wrong: `.gitignore:16` ignores `.mcp.json`, `git ls-files` does not
list it, and it has no commit history. Consequences the plan must absorb:

- **Fix #3 cannot ship in the PR.** Editing
  `/home/mikael/code/agentic-coder-teams-mcp/.mcp.json` is a local machine
  repair, invisible to the diff and unenforceable by tests. The PR's durable
  deliverables are the per-agent config (#1/#2), the guards (#4/#5), and docs.
- **Any user's hand-written project `.mcp.json`/`.pi/mcp.json` with an
  `AGENT_*` env block re-creates the bug**, and the repo cannot fix their
  files. The defense-in-depth guard is therefore not a "never fires in the
  healthy path" nicety — it is the only protection on such machines and should
  be treated (and tested) as a first-class behavior: pi worker still launches,
  identity-bearing tools refuse loudly with an actionable reason.
- Silver lining: the shipped README snippets (lines ~36-54) contain **no**
  `env` block, so the shipped docs never taught this mistake; the local file
  was hand-added. See F7 for the doc change that actually matters.

Recommendation: correct the plan text; reclassify fix #3 as "local environment
remediation, documented in implementation.md", and consider an optional
spawn-time preflight (warn in the spawn result when
`<agent_cwd>/.mcp.json` or `<agent_cwd>/.pi/mcp.json` declares a
`win-agent-teams` entry with any `AGENT_*` key) so the failure is diagnosable
at the lead, not only inside the broken child.

### F2 (should-fix) — Guard #5 as written misses the *silent* hijack paths

The observed incident went through the `recovery_hint` → `resume_session`
nudge, and #5 suppresses exactly that. But `_recover_session_id` has two other
adoption paths that a mis-identified child (IDENTITY == "team-lead", cwd ==
lead's cwd) can hit **with no nudge and no tool call**:

- the exact binding-key match (589-594) — in practice defused because
  `_binding_key` (284-290) includes the parent PID, which differs for the
  child's MCP server, but this is incidental, not designed;
- **single-candidate auto-adopt (600-608)** — `_candidate_sessions` filters
  only on `identity+cwd` (492-497, no PID), so with a single-lead history the
  child *silently* adopts the lead's session. This is a worse variant of the
  hijack than the one observed.

Guard #4's sentinel IDENTITY does close both paths (binding metas store
`identity: "team-lead"`, which no longer matches the sentinel) — but only if
the plan makes this explicit. Recommendation: specify that when identity is
unresolved, `_recover_session_id` returns `""` immediately after the
`_AGENT_SESSION_ID` check (before binding-key match and candidate scan), and
add a red-first test for the auto-adopt path (child env + single candidate →
no adoption), not only for the nudge suppression.

### F3 (should-fix) — Guard #4 covers too few tools; `read_messages` reads the lead's inbox

The plan guards `send_message` and `resume_session`. But with the current
fallback, a mis-identified child also:

- **reads and cursors the lead's inbox** — `read_messages` uses
  `_inbox_file(session_id, IDENTITY)` (1570-1572), so the child *consumes* the
  lead's messages (marks them read via the cursor), which is as damaging as
  the `from:` spoof;
- spawns grandchildren recorded with `lead_session_id=IDENTITY` (1427) and
  writes `_write_mcp_config(..., IDENTITY)` as the parent (1405).

With the sentinel IDENTITY these degrade to harmless-but-confusing (reading an
empty sentinel inbox; children whose parent is a sentinel). Recommendation:
either centralize a `_require_resolved_identity()` check applied to every
identity-bearing tool (`send_message`, `read_messages`, `resume_session`,
`spawn_agent`, `follow_up_agent`), or explicitly document in the plan that the
sentinel makes the remaining tools inert and add one test asserting
`read_messages` under unresolved identity does NOT touch
`inbox-team-lead.jsonl` / its cursor.

### F4 (nit) — `directTools` placement wording

Plan step #1 says "top-level `directTools: true`". The adapter reads
`directTools` either per server definition (`direct-tools` line 123,
`definition.directTools`) or as `config.settings.directTools` (line 108). The
repo `.mcp.json` puts it inside the server entry; the per-agent writer should
do the same (server-entry level), not document-top-level. Cosmetic, but the
implementer copying the plan literally would produce a key the adapter ignores.

### F5 (nit) — Remaining `.mcp.json` keys still shallow-override the per-agent entry

After removing only the `env` key, `<cwd>/.mcp.json`'s `win-agent-teams` entry
still merges after the per-agent file and overrides `command`, `args`, `cwd`,
`lifecycle`, `directTools`. Today the values are identical/benign (and `env`
absent means ours survives — the merge replaces only keys that are present),
so this is fine; but implementation.md should note the invariant: the
per-agent file only reliably owns keys that project-level sources do not
declare. If the per-agent file ever needs to win on `command`/`args`, this
design does not deliver that.

### F6 (nit, hardening opportunity) — Session id is derivable from the guard signal

`WIN_AGENT_TEAMS_SESSION_DIR` is `~/.claude/agent-sessions/<session-id>`. When
`AGENT_SESSION_ID` is empty but the signal is present, the server could derive
the session id from the directory basename (validated UUID-shaped) instead of
merely refusing. Optional; refusing is acceptable, but this would restore a
degraded-but-correct child instead of a bricked one on machines with a
clobbering user config (see F1). If adopted, keep identity (name) unresolved —
only the session binding is derivable.

### F7 (should-fix) — Open decision (b) is mis-framed: README has no env block to drop

The README `mcpServers` snippets (Windows ~line 36, Linux ~line 49) already
contain **no** `env` block, so there is nothing to "drop". What the docs do
need:

1. an explicit warning (README + ADDING-A-BACKEND) never to add `AGENT_*` keys
   to a project `.mcp.json` / `.pi/mcp.json` `win-agent-teams` entry, with one
   line on why (shallow env replacement in the pi adapter merge);
2. an update to the pi section (README lines ~106-111), which currently
   documents the `${AGENT_*}` interpolation design that this plan supersedes
   for spawned workers (the `~/.pi/agent/mcp.json` interpolation file remains
   for human-launched pi leads — say so);
3. README line ~365 says lead identity is `"lead"`; it is `"team-lead"`
   (server_simple.py:69). Pre-existing doc drift worth fixing in passing.

### F8 (nit) — Adapter drift risk is real but bounded; pin what was verified

The mechanism depends on adapter internals (flag name, source order, shallow
merge). Verified here: pi 0.80.10 + pi-mcp-adapter 2.11.0. The plan already
proposes pinning in implementation.md and a smoke test — keep that, and make
the smoke test assert the *observable* contract (spawned pi worker's
`send_message` records its own name), not adapter internals.

## Answers to the specific review questions

1. **Merge order / necessity+sufficiency**: Confirmed. `--mcp-config` replaces
   only source #2's read path; #3 and #4 still merge after it. Removing the
   local `.mcp.json` env block + the per-agent literal file is necessary and
   sufficient **on this machine** (no `~/.config/mcp/mcp.json`, no repo
   `.pi/mcp.json` — both verified absent). Source #4 (`<cwd>/.pi/mcp.json`)
   would clobber again if a user creates one with an env block — covered only
   by the guard + docs warning (F1/F7). Note when an override path is given,
   source #1 (`~/.config/mcp/mcp.json`) is still read (lowest precedence) —
   harmless, as the plan says.
2. **Root-lead safety**: Confirmed safe. `_AGENT_NAME` etc. use
   `os.environ.get(..., "")` (61-63); absent keys behave identically to empty
   strings, and nothing else depends on the literals existing. The root lead
   (human-launched, no `WIN_AGENT_TEAMS_SESSION_DIR` — the variable exists
   nowhere except pi's `build_env` and the pi extensions) still resolves to
   `team-lead` and the guard stays inactive. Note: there is no
   `WIN_AGENT_TEAMS_LEAD` variable in the codebase; root-lead-ness is purely
   "no AGENT_NAME".
3. **Claude path**: Untouched. `_write_mcp_config` is not modified; the new
   file name `<agent>.pi.mcp.json` cannot collide with `<agent>.mcp.json`;
   Claude workers always get a literal `AGENT_NAME`, so guards never fire.
4. **Guard correctness**: The signal is sound. False positives require a human
   to launch a root lead from a shell that already exports
   `WIN_AGENT_TEAMS_SESSION_DIR` (e.g. a terminal opened inside a spawned
   agent's environment) — acceptable, worth one sentence in the tool error.
   False negative: a user env block that explicitly sets
   `WIN_AGENT_TEAMS_SESSION_DIR: ""` would blind the guard — document, don't
   engineer around. Nested leads are safe: `process_manager` spawns children
   with `os.environ.copy() + build_env` (process_manager.py:478), so every
   spawned agent's server sees both the signal and (post-fix) a literal name.
5. **Cross-platform**: Sound. Direct `node <cli.js>` launch bypasses the shim;
   `--mcp-config <path>` as two discrete argv tokens follows the existing
   `--session-dir` precedent that already round-trips Windows paths. The shim
   fallback carries no newline/`<>|&^` in the path. No concerns.
6. **Test plan**: Good core, four gaps: (a) build_command/build_resume_command
   emit **no** `--mcp-config` when `pi_mcp_config_path` is absent
   (compat/degradation); (b) unresolved-identity child does NOT silently
   auto-adopt a single candidate session (F2 — the most dangerous path is
   currently untested); (c) `read_messages` under unresolved identity does not
   read/cursor the lead's inbox (F3); (d) an explicit Claude regression test
   is promised in Risks but missing from the numbered list — add it (existing
   `_write_mcp_config` tests may already cover; verify).
7. **Open decisions**: see below.

## Recommendations on the two open decisions

- **(a) Guard failure mode: refuse-at-tool-call.** Agree with the plan.
  A hard startup crash surfaces as an opaque adapter "server failed" with no
  channel to explain itself; the worker silently loses all team tools. A
  structured refusal (`{"success": false, "reason": "identity_unresolved",
  "hint": ...}`) is visible to the model, actionable (report via final
  output), keeps read-only/diagnostic tools usable, and is directly unit
  -testable. Pair it with a `logger.error` at import time so the log tells the
  operator which config clobbered identity.
- **(b) README env block**: reframed per F7 — there is no env block in the
  README snippet to drop. Instead: add the "never put AGENT_* in project MCP
  configs" warning, update the pi identity section to describe the per-agent
  `--mcp-config` mechanism, and fix the `"lead"` → `"team-lead"` drift. In
  scope for this PR (docs are the only durable channel to the file that
  actually caused the bug — see F1).

## Verdict

**Sound — implement with the amendments above.** The root-cause analysis is
verified correct in every particular, the chosen mechanism is right, and the
cross-platform reasoning holds. Required changes before/during implementation:

1. Correct the "tracked" claim and reclassify fix #3 as local remediation +
   docs warning (F1, F7).
2. Make the unresolved-identity early-return in `_recover_session_id` cover
   the binding-key and auto-adopt paths, with a test for silent auto-adopt
   (F2).
3. Extend or explicitly disposition the guard for `read_messages` (and note
   the sentinel's effect on `spawn_agent`/`follow_up_agent`) (F3).
4. Put `directTools` at server-entry level in the per-agent file (F4).
5. Add the four missing tests (question 6).

The live pi smoke (spawn → `send_message` records own name, no lead-session
adoption) remains the deferred acceptance check, as the plan states.
