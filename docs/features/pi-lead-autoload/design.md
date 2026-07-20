# Pi lead auto-load design

Goal: make the `win-agent-teams-wake` Pi extension start **automatically** for
**any Pi agent that participates in a win-agent-teams session** — the way
`pi install npm:pi-mcp-adapter` auto-loads on every `pi` start — without a
manual `-e <path>` flag and with minimal MCP setup, while leaving an ordinary
`pi` session (run for unrelated work) completely unaffected.

This is a **design note only**. Nothing here is implemented.

## Architecture note — "lead" is a role at every nesting level

A Pi agent in win-agent-teams is not either "the lead" or "a worker". Any Pi
agent can spawn children, so **any** Pi agent can be a lead for the agents below
it (nested orchestration — the equivalent of Claude Code subagents, which Pi
lacks natively and win-agent-teams supplies). Concretely:

- **Root lead**: a human-launched bare `pi`. No `AGENT_NAME`; its own inbox
  identity is `team-lead`.
- **Mid-level Pi subagent-as-lead**: spawned by the pi backend, so it **has**
  `AGENT_NAME=<its name>` and `WIN_AGENT_TEAMS_SESSION_DIR` in env, yet it can
  spawn its own level-2 children and must be woken when *they* post to *its*
  inbox (`inbox-<AGENT_NAME>.jsonl`).

The wake extension is therefore **not** a root-lead-only feature. Its identity
("whose inbox do I watch") is the agent's **own** identity —
`AGENT_NAME` when set, else `team-lead` — which is exactly the default the
`watch` / `inbox-status` CLI uses when invoked **without** `--reader`. The
merged extension hardcodes `--reader team-lead` and fails closed unless
`identity === "team-lead"`; that is an **over-constrained bug** — it only ever
wakes a root lead and silently never wakes a nested Pi lead. See
[§6 Correction to the merged feature](#6-correction-to-the-merged-feature).

## 0. Grounding — what exists today

- **Wake extension**: `pi-extensions/win-agent-teams-wake/` (root `index.ts`
  default-export `activate(pi)` factory; runtime dep-free; shells out to the
  `win-agent-teams` CLI). On `session_start` it starts a single-flight state
  machine `DISCOVERING → WATCHING → ACK_WAIT → ACK_STALLED`
  (`src/state-machine.ts`), torn down on `session_shutdown`.
- **Fail-closed identity check** (`src/state-machine.ts` `discovering()` /
  `refresh()`): it runs `win-agent-teams session-dir`, and only advances to
  `WATCHING` when `r.discovery.identity === this.reader`, where the reader is
  fixed at `DEFAULT_READER = "team-lead"` (`src/cli.ts:12`; the machine's
  `reader` defaults to it and nothing overrides it). Every `watch` /
  `inbox-status` shell-out passes `--reader team-lead`, and `isLeadInboxPath`
  guards against `inbox-team-lead.jsonl`. Any non-`team-lead` / no-session
  result keeps it in `DISCOVERING`, re-polling on a 30 s-capped backoff. **This
  is the over-constrained bug**: a nested Pi lead has `identity=<AGENT_NAME>`,
  so `discovering()` never advances and it is never woken.
- **Session-presence env the backend injects** (`backends/pi.py` `build_env`):
  every spawned Pi agent gets `AGENT_NAME`, `AGENT_SESSION_ID`,
  `AGENT_PARENT_NAME`, and — when a session dir is known —
  `WIN_AGENT_TEAMS_SESSION_DIR`. A bare human-launched root lead has **none** of
  these. This asymmetry drives the guard design (§2).
- **Worker-side precedent**: `pi-extensions/win-agent-teams-state/index.ts` is
  a **no-op unless** `WIN_AGENT_TEAMS_SESSION_DIR` **and** `AGENT_NAME` are
  present in env. It is loaded by the pi backend via `-e`
  (`src/claude_teams/backends/pi.py:356` `_extension_args` → `["-e", <path>]`).
- **`pi install` capabilities** (`pi install --help`): accepts
  `npm:<pkg>`, `git:…`, `https://…`, `ssh://…`, **and `./local/path`**.
  `-l/--local` writes to project `.pi/settings.json`; without it, global
  `~/.pi/agent/settings.json`.
- **Auto-load mechanism** (`~/.pi/agent/settings.json`): a top-level
  `"packages": ["npm:pi-mcp-adapter"]` array. On each `pi` start every entry is
  loaded. `pi list` confirms the resolved source
  (`npm:pi-mcp-adapter → ~/.pi/agent/npm/node_modules/pi-mcp-adapter`).
- **What makes a directory a loadable pi package**: `pi-mcp-adapter`'s
  `package.json` carries a `"pi": { "extensions": ["./index.ts"] }` field. The
  wake extension's `package.json` **does not** have this field yet — it only
  has `"main": "index.ts"`. This is the single most important prerequisite.
- **MCP config precedence** (pi-mcp-adapter README, verbatim order):
  1. `~/.config/mcp/mcp.json`
  2. `<Pi agent dir>/mcp.json` (`~/.pi/agent/mcp.json`)
  3. `.mcp.json`
  4. `.pi/mcp.json`
  The repo already ships a project `.mcp.json` with the `win-agent-teams`
  server (`command` = repo `.venv` python, `cwd` = repo root, empty
  `AGENT_*` env → lead identity, `lifecycle: keep-alive`, `directTools: true`).

## 1. Auto-load — two delivery paths for two agent kinds

There are two ways a Pi agent starts, and each needs its own auto-load path:

- **Spawned Pi agents (workers + nested subagents-as-lead)** are launched by the
  pi backend, which already loads the **state** extension via `-e`
  (`backends/pi.py` `_extension_args`). The cleanest coverage is to have the
  backend load the **wake** extension the *same way* — a second `-e <wake dir>`
  for every spawned Pi agent. Then any spawned Pi that becomes a lead for
  level-2 children is covered with no per-agent setup, and the guard (§2)
  activates it via the injected `WIN_AGENT_TEAMS_SESSION_DIR`. See §5 task 3.
- **The root lead** is a bare human-launched `pi` — no backend involved, so `-e`
  can't reach it. This is where the `packages` auto-load (below) applies. It is
  loaded for every `pi` start but stays a no-op unless `WIN_AGENT_TEAMS_LEAD=1`
  (§2).

Both paths load the *same* extension directory; they differ only in how the
extension gets in front of the process. The subsections below cover the
`packages` path (root lead); the backend `-e` path is a one-line backend change
(§5 task 3).

`pi` auto-loads by reading the `"packages"` array in
`~/.pi/agent/settings.json` at every start. To add the wake extension there,
three options; recommendation follows.

### (a) `pi install <repo path>` — RECOMMENDED
Run once, from anywhere:

```bash
pi install /home/mikael/code/agentic-coder-teams-mcp/pi-extensions/win-agent-teams-wake
```

This appends an entry to `packages` pointing at the local directory (mirroring
how `npm:pi-mcp-adapter` was added). It requires the directory to be a valid
pi package — i.e. the `package.json` **must** gain the
`"pi": { "extensions": ["./index.ts"] }` field first (see task list). Because
the extension is dep-free (only node builtins + the pinned type-only
devDependency), no `node_modules` install is needed at runtime.

- Pro: uses pi's own tooling; survives `pi update`; matches the adapter flow.
- Con: absolute-path entry is machine-specific (fine for a single-dev setup);
  the repo path must remain stable.

### (b) Publish to npm + `pi install npm:win-agent-teams-wake`
Cleanest for multi-machine distribution, but the package is currently
`"private": true` and unpublished. Out of scope for a local one-time setup;
note as a future path if this ships to other users.

### (c) Manual `settings.json` `packages` entry
Hand-edit `~/.pi/agent/settings.json`:

```json
"packages": [
  "npm:pi-mcp-adapter",
  "/home/mikael/code/agentic-coder-teams-mcp/pi-extensions/win-agent-teams-wake"
]
```

Functionally identical to (a) but bypasses `pi install`'s validation/trust
handling. Use only if `pi install` from a path misbehaves.

**Recommendation: (a).** The exact `packages` string it produces is the
absolute repo path above (option (c) shows the literal form).

## 2. Activation guard — "in a win-agent-teams session", not "is team-lead"

The activation condition is: **this Pi agent participates in a win-agent-teams
session** (at any nesting level). Every such agent can spawn children and so
needs the wake. The condition is **not** "identity == team-lead" — that would
(and today does) exclude every nested Pi lead.

The guard must run **early** (before the state machine starts) so that a plain
`pi` with no teams context is a **true no-op**: no background loop, no
`session-dir` shell-out, no dependency on the `win-agent-teams` CLI being on
`PATH`. Mirrors the `win-agent-teams-state` no-op precedent.

### Two distinct cases the guard must accept

1. **Spawned Pi agent** (including a nested subagent-as-lead): the backend
   injects `WIN_AGENT_TEAMS_SESSION_DIR` (and `AGENT_NAME`). Cheap env check —
   no subprocess.
2. **Root lead** (bare human-launched `pi`): **no** teams env is present, yet it
   *is* a lead. The only authoritative signal is the on-disk session, i.e. a
   `win-agent-teams session-dir` probe returning a live session (exit 0).

### Recommended guard (layered, cheapest-first)

In `index.ts` `activate(pi)`, before starting the loop:

```ts
// 1. Fast path: spawned agent — backend injected the session dir.
const inSession = !!process.env.WIN_AGENT_TEAMS_SESSION_DIR;

// 2. Root-lead opt-in: bare `pi` has no teams env, so gate the one-time
//    session-dir probe behind an explicit flag to keep plain `pi` a no-op.
const rootLeadOptIn = process.env.WIN_AGENT_TEAMS_LEAD === "1";

if (!inSession && !rootLeadOptIn) return; // plain pi → true no-op
```

- For a **spawned** agent the env var alone activates it — zero cost, and it is
  automatically correct for a nested subagent-as-lead.
- For the **root lead** we still want auto-load without a per-project MCP flag,
  but a bare `pi` genuinely has no teams marker, so blindly probing
  `session-dir` on *every* `pi` start (teams or not) is the churn we reject.
  Gate that single probe behind `WIN_AGENT_TEAMS_LEAD=1`, set by the tiny lead
  launcher (§4). The existing in-loop `DISCOVERING` `session-dir` check then
  confirms a real session and supplies the session dir.

Why env-based, not a marker file: env is the established pattern in this repo
(`win-agent-teams-state` keys off `WIN_AGENT_TEAMS_SESSION_DIR`/`AGENT_NAME`),
is set per-invocation, and needs no filesystem lookup.

> Note the inversion vs. the worker state extension: that one activates *only
> when `AGENT_NAME` is present*. Here `AGENT_NAME` may be **absent** (root lead)
> or **present** (nested lead) — both must activate — so the wake guard keys on
> `WIN_AGENT_TEAMS_SESSION_DIR` (present for any spawned agent) plus the
> root-lead opt-in, **not** on `AGENT_NAME`.

### Reader identity — own inbox, not hardcoded `team-lead`

Once active, the extension must watch **its own** inbox. The correct reader is
`AGENT_NAME || "team-lead"` — precisely what the CLI defaults to when `watch` /
`inbox-status` are called **without** `--reader`. The cleanest implementation is
to **stop passing `--reader` at all** and let the CLI apply its own default
(which already resolves ambient identity), keeping `--reader` only as an
optional explicit override. The in-loop identity gate then becomes "a session
exists" rather than "identity == team-lead". Details in §6.

Result: a plain `pi` anywhere → immediate no-op. A spawned Pi agent (worker or
nested lead) → activates automatically (backend env) and watches
`inbox-<AGENT_NAME>.jsonl`. A root lead launched via the flag → activates and
watches `inbox-team-lead.jsonl`.

## 3. MCP config: global vs project

The `win-agent-teams` MCP server needs the **repo cwd** (its `.mcp.json` sets
`cwd` to the repo root; the server resolves session dirs / spawns relative to
it) and the empty `AGENT_*` env that marks the caller as the lead.

- A **global** entry (`~/.config/mcp/mcp.json`) would apply to *every* pi
  session on the machine — pi would try to start the win-agent-teams server
  (or at least register its tools) everywhere. With `lifecycle: keep-alive` it
  connects at startup even in unrelated sessions; even `lazy` still injects the
  tool surface. Worse, a global entry has a **fixed** `cwd`, so it cannot be
  the "current repo" — it would always point at this one repo regardless of
  where `pi` runs. That breaks the identity/cwd model.
- The **project** `.mcp.json` already in the repo resolves `cwd` correctly and
  only loads when `pi` runs inside the repo.

**Recommendation: keep the MCP server in the project `.mcp.json` (status quo).
Do not globalize it.** It already scopes cleanly to the repo, and pi's
precedence means project `.mcp.json` is picked up automatically when the lead
runs `pi` in the repo — no per-invocation MCP flag needed. The only coupling to
the activation flag is the wake *extension*, not the MCP server; the MCP server
is naturally gated by "am I running pi inside this repo".

(If multi-repo lead support is ever needed, the right answer is a project
`.mcp.json` per repo with `cwd` interpolated, not a global entry — pi-mcp-adapter
supports `${VAR}`/`~` interpolation in `cwd`, but not "the directory pi was
launched from" as a token, so per-repo files remain simplest.)

## 4. Minimal one-time setup

Spawned Pi agents (workers and nested subagents-as-lead) need **no user setup**
once §5 task 3 lands — the pi backend loads the wake extension via `-e` for
them automatically, and the injected `WIN_AGENT_TEAMS_SESSION_DIR` activates it.

For the **root lead** (bare `pi`), a user does this **once**:

1. Add the `"pi": { "extensions": ["./index.ts"] }` field to the wake
   extension's `package.json` (shipped by the repo; not a user step once merged).
2. Register the package globally:
   ```bash
   pi install /home/mikael/code/agentic-coder-teams-mcp/pi-extensions/win-agent-teams-wake
   ```
3. Define a root-lead launcher that sets the opt-in flag (shell alias/function,
   or a `win-agent-teams pi-lead` helper):
   ```bash
   alias pi-lead='WIN_AGENT_TEAMS_LEAD=1 pi'
   ```

Thereafter:

- `pi-lead` in the repo → wake extension activates (flag → session-dir probe
  confirms the lead session) **and** project `.mcp.json` gives it the
  `win-agent-teams` MCP tools. Fully automatic.
- A **spawned** Pi agent (any level) → wake + state extensions loaded by the
  backend; wake activates off `WIN_AGENT_TEAMS_SESSION_DIR` and watches its own
  `inbox-<AGENT_NAME>.jsonl`. No setup.
- Plain `pi` in the repo → MCP tools present (project `.mcp.json`) but the wake
  watcher stays dormant (no flag, no injected session dir). MCP tools are inert
  until called.
- Plain `pi` anywhere else → unaffected: no flag, no teams env, no project
  `.mcp.json`.

If even "MCP tools visible in a plain in-repo `pi`" is undesirable, gate the
`.mcp.json` entry's `lifecycle` to `lazy` (currently `keep-alive`) so the server
never *starts* until a tool is called — but leaving it is low-cost.

## 5. Implementation task list (design only — not done here)

1. **`pi-extensions/win-agent-teams-wake/package.json`**: add
   `"pi": { "extensions": ["./index.ts"] }` so `pi install <dir>` / `packages`
   treats it as a loadable extension. (Prerequisite for §1a.) Consider dropping
   `"private": true` only if npm publishing is later chosen.
2. **`pi-extensions/win-agent-teams-wake/index.ts`**: add the early layered
   guard (§2): activate when `WIN_AGENT_TEAMS_SESSION_DIR` is set **or**
   `WIN_AGENT_TEAMS_LEAD === "1"`; otherwise return (true no-op). Add focused
   tests: (a) no loop start when neither is set; (b) activates for a spawned
   agent (session-dir env only, no flag); (c) activates for the flagged root
   lead.
3. **`src/claude_teams/backends/pi.py`** (`_extension_args`): also load the wake
   extension for spawned Pi agents — append a second `-e <wake dir>` alongside
   the existing state-extension `-e`. Thread the wake path in via `request.extra`
   (mirror `pi_state_extension_path` → add `pi_wake_extension_path`) and wire it
   where the server populates `_hook_extra` for the pi backend. This is what
   covers nested subagents-as-lead without per-agent setup.
4. **Generalize the extension beyond `team-lead`** — see §6 (drop the hardcoded
   `--reader team-lead` and the fail-closed-unless-`team-lead` gate; watch the
   ambient `AGENT_NAME`-or-`team-lead` identity).
5. **Root-lead launcher**: document `alias pi-lead='WIN_AGENT_TEAMS_LEAD=1 pi'`,
   or add a `win-agent-teams pi-lead` subcommand that execs `pi` with the flag
   set. A subcommand is more discoverable/cross-shell; the alias is zero-code.
6. **Docs**:
   - README: describe the wake feature as covering *any* Pi lead at any nesting
     level (root + nested), loaded via backend `-e` for spawned agents and via
     `packages` + `WIN_AGENT_TEAMS_LEAD` for the root lead. Keep the worker
     `-e win-agent-teams-state` description.
   - Note MCP config stays project-scoped (`.mcp.json`) and why global is rejected.
7. **No change to `.mcp.json`** required (keep project scope); optionally flip
   `lifecycle` to `lazy`.

## 6. Correction to the merged feature

The just-merged wake extension is over-constrained to the root lead and must be
generalized so it also wakes nested Pi subagents-as-lead. Code changes:

1. **Drop the hardcoded `--reader team-lead`.** In `src/state-machine.ts` the
   machine's `reader` defaults to `DEFAULT_READER = "team-lead"` and is passed
   into every `runWatch` / `runInboxStatus` / `isLeadInboxPath` call. Stop
   forcing a reader: call the CLI **without** `--reader` so it applies its own
   ambient default (`AGENT_NAME` or `team-lead`), and derive the inbox-path
   guard from the identity the CLI actually reports. Keep an optional
   `reader` override (`WakeMachineOptions.reader` / a `--reader` passthrough)
   for explicit cases only; do not default it to `team-lead`.
2. **Replace the fail-closed `identity === "team-lead"` gate.** In
   `discovering()` and `refresh()` (`src/state-machine.ts:~120-135`), the
   advance/stay condition is currently `r.discovery.identity === this.reader`
   with `this.reader === "team-lead"`. Change the gate to "a live session
   exists" (`r.kind === "ok"`), and treat `r.discovery.identity` as **the**
   reader — i.e. bind the watched inbox to the reported identity rather than
   comparing it to a hardcoded constant. A spawned agent reports
   `identity=<AGENT_NAME>`; a root lead reports `identity=team-lead`; both now
   pass.
3. **Recompute the inbox-path guard from the reported identity.**
   `isLeadInboxPath` / `leadInboxPath` (`src/cli.ts:25,38`) default the reader
   to `team-lead`; drive them from `discovery.identity` so a spawned agent's
   wake matches `inbox-<AGENT_NAME>.jsonl`, not `inbox-team-lead.jsonl`. (The
   "lead" naming in these helpers is now a misnomer — consider renaming to
   `ownInboxPath` / `isOwnInboxPath` for clarity, non-blocking.)
4. **Tests**: extend the state-machine tests to cover a `session-dir` result
   with `identity=<AGENT_NAME>` (nested lead) reaching `WATCHING` and waking on
   `inbox-<AGENT_NAME>.jsonl`, in addition to the existing `team-lead` case.

These changes are behavior-generalizing (they widen when the extension wakes)
and should ship together with the auto-load work, since the two spawned-agent
delivery path (§5 task 3) is pointless while the identity gate still rejects a
nested lead.
