# win-agent-teams-wake

A [Pi](https://github.com/earendil-works/pi) extension that wakes **any Pi lead**
— at any nesting level — when a child posts to its `win-agent-teams` inbox. It
runs a single-flight, cursor-aware state machine that shells out to the
read-only `win-agent-teams` CLI (`session-dir` / `inbox-status` / `watch`) and
injects one steering turn per message generation, telling the lead to call
`read_messages`.

"Lead" is a role at every nesting level, not a fixed identity: a spawned Pi
subagent can itself spawn children and must be woken when *they* post to *its*
inbox. The extension watches the agent's **own** identity — `AGENT_NAME` when
set (a spawned agent / nested lead → `inbox-<AGENT_NAME>.jsonl`), else
`team-lead` (a human-launched root lead → `inbox-team-lead.jsonl`). It shells out
to the CLI **without** `--reader`, letting the CLI apply that same ambient
default; a `WakeMachineOptions.reader` override exists only for explicit cases.

The extension never consumes the inbox itself — the lead remains the only cursor
writer. See `docs/features/pi-lead-inbox-wake/plan.md` for the wake-loop design
and `docs/features/pi-lead-autoload/design.md` for auto-load + the nested-lead
generalization.

## Activation guard (no-op unless in a win-agent-teams session)

`activate()` returns immediately — no handlers, no loop, no dependency on the
CLI being on `PATH` — unless one of these is present in the environment:

- `WIN_AGENT_TEAMS_SESSION_DIR` — injected by the pi backend into every spawned
  agent (worker or nested subagent-as-lead). This is the normal activation path.
- `WIN_AGENT_TEAMS_LEAD=1` — the explicit opt-in for a bare human-launched root
  lead, set by a launcher such as `alias pi-lead='WIN_AGENT_TEAMS_LEAD=1 pi'`.

A plain `pi` run for unrelated work sets neither and stays a true no-op.

## Auto-load

- **Spawned Pi agents**: the pi backend loads this extension via `-e`
  automatically (alongside `win-agent-teams-state`), so nested leads are covered
  with no per-agent setup. The injected `WIN_AGENT_TEAMS_SESSION_DIR` activates
  it.
- **Root lead**: register the package once so `pi` auto-loads it from
  `~/.pi/agent/settings.json` `packages` (the `package.json` here carries the
  `"pi": { "extensions": ["./index.ts"] }` field that makes the directory a
  loadable pi package):

  ```bash
  pi install /path/to/agentic-coder-teams-mcp/pi-extensions/win-agent-teams-wake
  ```

  Then launch the root lead with the opt-in flag (`pi-lead` alias above). The
  `win-agent-teams` MCP server stays project-scoped in the repo's `.mcp.json`
  (not global) — see the design doc for why.

- **Kill switch**: `WIN_AGENT_TEAMS_STATE_HOOKS=0` disables the pi *state*
  extension **and** this wake extension together — the server's `_hook_extra`
  returns early before adding either `-e` path, so the two share one switch.
  There is no separate flag to disable wake while keeping state reporting.

## Requirements

- Node.js 20+ (developed on Node 24).
- Pi, pinned to the exact version this extension is written against:
  `@earendil-works/pi-coding-agent@0.80.10` (see `package.json` `devDependencies`).
- The `win-agent-teams` CLI on `PATH` (this repository's Python package,
  runnable via `uv run win-agent-teams ...`).

## Install / build

From this directory:

```bash
npm install          # installs the pinned Pi package + toolchain
npm run typecheck    # tsc --noEmit
npm test             # vitest run
npm run lint         # eslint .
npm run format       # prettier --check on index.ts src/**/*.ts test/**/*.ts
```

The package entry point is the root `index.ts` (`main` in `package.json`), whose
**default export** is the Pi `ExtensionFactory` (`(pi: ExtensionAPI) => void`).
The implementation modules live under `src/` and are imported by `index.ts`. Pi
0.80.10 loads a default-exported factory module directly; no separate manifest
file is required.

## Loading the extension in Pi

Load this extension directory with Pi's `-e` / `--extension` flag, pointing at
the directory (Pi resolves `main` from `package.json`), following the same
convention as the sibling `win-agent-teams-state` extension:

```bash
pi -e pi-extensions/win-agent-teams-wake
```

On `session_start` the factory creates an owned `AbortController` and starts one
loop; on `session_shutdown` it aborts and awaits that loop. Only `pi.exec`,
`pi.sendMessage`, and `pi.on("session_start"|"session_shutdown")` are used.

## Manual smoke test (requires a live Pi + model credentials)

This is the end-to-end delivery check (plan §6, T9). It has **not** been run in
CI or in this repository's automated suite — the automated proof of loadability
is `test/index.test.ts`, which drives the factory with a fake `ExtensionAPI`.
Run the steps below with your own Pi and model configuration:

1. `npm install` here so `@earendil-works/pi-coding-agent@0.80.10` is resolved.
2. **Give the lead MCP access.** Pi has no native MCP client, so install the
   adapter:

   ```bash
   pi install npm:pi-mcp-adapter
   ```

3. **Register `win-agent-teams` over stdio** in a location the adapter reads —
   `.mcp.json` at the repo root (or `~/.config/mcp/mcp.json`). Leave the identity
   env vars empty so the server resolves the session as **team-lead**:

   ```json
   {
     "mcpServers": {
       "win-agent-teams": {
         "command": "<repo>/.venv/bin/python",
         "args": ["-m", "claude_teams.server_simple"],
         "cwd": "<repo>",
         "env": {
           "AGENT_NAME": "",
           "AGENT_SESSION_ID": "",
           "AGENT_PARENT_NAME": ""
         },
         "directTools": true,
         "lifecycle": "keep-alive"
       }
     }
   }
   ```

   (`.venv/bin/win-agent-teams serve` works in place of the `python -m` command.)

4. **Put `.venv/bin` on `PATH`.** The extension shells out to the bare
   `win-agent-teams` command, so it must be resolvable — export the venv `bin`
   onto `PATH`, or launch Pi via `uv run`.
5. Start a bare Pi session as the **team lead** with this extension loaded,
   launched from the repo cwd with no `AGENT_NAME` (identity = team-lead):

   ```bash
   pi -a -e pi-extensions/win-agent-teams-wake
   ```

   Confirm `win-agent-teams session-dir` reports the lead identity for that
   session. Note: the `directTools` cache warms after one restart of the lead, so
   the MCP tools may only appear on the second launch.
6. From the team, spawn a worker agent (`spawn_agent`).
7. Have the worker `send_message` to the team lead's inbox.
8. Observe that the lead session receives a single injected steering turn
   (`customType: "win-agent-teams/wake"`) naming the sender(s), and that the
   lead then calls `read_messages` and drains the inbox.
9. Confirm no repeated wake fires while the generation stays unread, and that a
   later message from a new sender produces exactly one additional wake.

Record the observed result (Pi version, model, sender, wake content) when you
run it; do not treat this document as evidence that the live run has occurred.
