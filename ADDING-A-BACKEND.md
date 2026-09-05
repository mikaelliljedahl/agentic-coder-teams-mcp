# Adding a backend

How to add support for a new agentic CLI (a "backend") to win-agent-teams. This
is the reference for the mechanics; the surrounding plan→implement→review→smoke→PR
workflow lives in [DEVELOPMENT.md](DEVELOPMENT.md).

A backend is an adapter that lets the server **spawn** a specific CLI as a team
agent and, ideally, lets that agent **talk back** through the MCP tools. The three
shipped backends — `claude-code`, `codex`, `pi` — are the reference patterns; read
them alongside this guide (`src/claude_teams/backends/{claude_code,codex,pi}.py`).

## 1. The one hard dependency: the disk contract

Everything the coordinator relies on is on disk under
`~/.claude/agent-sessions/<session-id>/` (`server_simple._session_dir`). A spawned
agent participates by honoring two files:

- **State marker** `state-<agent>.json` — `{ "state": "running"|"waiting",
  "event": "<lifecycle-event>", "ts": <epoch-float> }`
  (`server_simple._state_marker_file`, schema in `hooks.py`). The agent writes
  `running` while working and `waiting` when it settles/idles. The coordinator
  file-watches this to know when to look.
- **Inbox** `inbox-<agent>.jsonl` — one `{"from","text","ts"}` line per message
  (`server_simple._inbox_file`). Written/read by the `send_message`/`read_messages`
  MCP tools; messaging is **pull**, never pushed.

Identity is carried in three env vars the agent's process must expose so the MCP
server binds it to the right inbox/session: `AGENT_NAME`, `AGENT_SESSION_ID`,
`AGENT_PARENT_NAME`. `AGENT_NAME` falls back to `team-lead` when absent.
Because any agent can spawn children, "lead" is a per-level role, not `team-lead`
specifically — see the nested-orchestration section in `CLAUDE.md`.

Two capability levels:

- **Full team participant** (`claude-code`, `codex`, `pi`) — connects to the
  win-agent-teams MCP server (so it can `send_message`/`spawn_agent`/…) **and**
  reports state. `is_interactive = True`.
- **One-shot worker** — just runs the prompt and exits; no messaging, state
  degrades to liveness + activity-recency (see §7). `is_interactive = False`.

What breaks if the agent can't speak MCP: it can neither send nor receive
messages, and — unless it still writes the state marker out-of-band — its state
degrades and `follow_up_agent` idle-detection weakens
(`server_simple._resolve_agent_state`).

## 2. The `Backend` protocol & `BaseBackend`

Every backend satisfies the `Backend` Protocol in
`src/claude_teams/backends/contracts.py`. Subclass `BaseBackend`
(`src/claude_teams/backends/process_base.py`), which implements the whole process
lifecycle (`spawn`, `resume`, `health_check`, `kill`, `capture`, `send`, …) on top
of `process_manager`. You override only the CLI-specific pieces.

| Method | BaseBackend default | Override? |
|--------|--------------------|-----------|
| `name` / `binary_name` | from `_name` / `_binary_name` | set the class attrs |
| `is_interactive` | `False` | `True` for a full participant |
| `is_available` / `discover_binary` | `shutil.which(_binary_name)` | override to resolve a native/entry binary (see §4) |
| `build_command` | `NotImplementedError` | **must implement** |
| `build_resume_command` | `NotImplementedError` | implement iff `supports_resume` |
| `supported_models` / `default_model` / `resolve_model` | `NotImplementedError` | **must implement** |
| `resolve_launch` | resolve model + pass effort through | override if a tier bundles effort |
| `build_env` | `{}` | override to inject identity/PATH |
| `default_permission_args` / `bypass_permission_args` | `[]` | override for autonomous bypass |
| `supports_resume` | `False` | `True` to enable `follow_up_agent` |
| `reasoning_effort_spec` / `agent_select_spec` / `discover_agents` | `None` / `[]` | optional |

`SpawnRequest` (contracts.py) is the backend-agnostic input: `agent_id`, `name`,
`team_name` (= session id), `prompt`, `model`, `cwd`, `lead_session_id` (= parent
identity), `permission_mode`, `reasoning_effort`, and a free-form `extra` dict the
server fills with per-backend wiring (§6).

## 3. The backend class, step by step

Create `src/claude_teams/backends/<name>.py`:

```python
class MyBackend(BaseBackend):
    _name = "mybackend"
    _binary_name = "myagent"

    @property
    def is_interactive(self) -> bool:
        return True  # full participant
```

### Models & capability tiers

The MCP caller picks capability, not a raw model slug. `codex` and `pi` each
expose six **tiers** (`cheapest/low/medium/high/xhigh/max`) that bundle a concrete
model + reasoning effort; `supported_models()` returns the tier names and
`resolve_launch()` maps a tier to `(model, effort)`:

```python
_TIER_LAUNCH = {  # fixed per-backend mapping; Codex shown here
    "cheapest": ("gpt-5.6-luna", "medium"),
    "low":      ("gpt-5.6-luna", "high"),
    "medium":   ("gpt-5.6-luna", "xhigh"),
    "high":     ("gpt-5.6-sol", "medium"),
    "xhigh":    ("gpt-6-astra", "low"),
    "max":      ("gpt-6-astra", "medium"),
}
# Pi differs only at high: ("gpt-5.6-luna", "max").
```

The production ladders are backend-specific at `high`: Codex uses Sol @ medium
for its 262k context limit, while Pi uses Luna @ max with its 1M window. See the
README's ladder table for the complete fixed mapping. `resolve_launch(model,
effort)` returns the `(model, effort)` pair the spawn uses.

- **Hard-fail tiers** (`codex.py` and `pi.py` `resolve_launch`): when live
  discovery is non-empty, a tier whose model isn't installed raises
  `BackendModelUnavailableError` with an upgrade hint — no silent downgrade.
- **Pi raw-slug soft-fallback** (`pi.py` `resolve_launch`): an unavailable
  explicit raw slug returns `("", effort)` so the CLI uses its own default
  model. A raw slug is an explicit provider-specific escape hatch; do not apply
  this fallback to capability tiers.

Discover the installed models live and cache per process (`codex._discover_codex_model_slugs`,
`pi._discover_pi_model_ids`); treat an empty discovery result as "unknown → skip
validation" so a discovery hiccup never blocks a spawn. When a tier owns the
effort, ignore any caller-supplied `reasoning_effort`.

### `build_command` / `build_resume_command`

Assemble the argv. It must inject: permission args, the cwd, the resolved
model/effort, **identity/MCP wiring** (§6), the **state-hook wiring** (§7), and the
prompt. See `pi.build_command` for the compact shape and `codex.build_command` for
the headless/interactive split.

Decide interactive vs headless with
`process_manager.provides_tty(self._name, is_interactive=self.is_interactive)`
(`codex._headless`, `pi._headless`): a real TTY (WT tab / new console / tmux)
runs the interactive TUI; the non-console spawn path runs the CLI's headless
entrypoint (`codex exec`, `pi -p --mode json`).

For resume, prefer a **deterministic** session id you set at spawn so the follow-up
re-targets the same conversation (`pi` uses `--session-id <agent>` + `--continue`);
codex resolves its own session id from the rollout and passes it to
`codex … resume <id>`.

### `build_env`

Return the identity env (`AGENT_NAME`/`AGENT_SESSION_ID`/`AGENT_PARENT_NAME`) plus
anything the CLI needs. Keys are validated against `_SAFE_ENV_KEY`
(`process_base._spawn_with_command`). `codex.build_env` also replicates its npm
shim's PATH effects when it bypasses the shim (§4).

### Permission args

`default_permission_args` / `bypass_permission_args` become the flags for
`permission_mode`. `require_approval` yields none. Examples: claude
`--permission-mode bypassPermissions`, codex `--dangerously-bypass-approvals-and-sandbox`,
pi `-a` (trust project; pi has no permission popups).

## 4. Windows launch gotchas

Most agent CLIs are installed as an npm `.cmd` shim that routes through
`cmd.exe`. `cmd.exe` expands `%*` and **truncates an argv token at the first
newline** and mangles `< > | & ^ ( )` — so a multi-line prompt cannot survive the
shim verbatim. Two proven escapes:

1. **Bypass the shim** by launching the real binary/entry directly, so argv goes
   through `CreateProcess` / `CommandLineToArgvW` intact:
   - codex resolves the bundled native `codex.exe` (`codex._resolve_native_codex`).
   - pi resolves `node` + the package's `dist/cli.js` (`pi._launcher` /
     `_resolve_entry`).
2. **Carry the prompt in a file** instead of argv:
   - claude writes a UTF-8 sidecar and passes a "read your prompt from
     `<path>`" instruction (`server_simple._write_prompt_file_extra`,
     `claude_code._prompt_arg`).
   - pi falls back to a `@<file>` include only when it is forced onto the shim
     (`pi._prompt_args`).

Prefer (1); keep (2) as the fallback when the native entry can't be resolved.

## 5. Register the backend

Add it to `_BUILTIN_BACKENDS` in `src/claude_teams/backends/registry.py`:

```python
_BUILTIN_BACKENDS = {
    "claude-code": "claude_teams.backends.claude_code.ClaudeCodeBackend",
    "codex":       "claude_teams.backends.codex.CodexBackend",
    "pi":          "claude_teams.backends.pi.PiBackend",
    "mybackend":   "claude_teams.backends.mybackend.MyBackend",
}
```

Backends load lazily and are only registered if `is_available()` returns True, so
a missing binary silently drops the backend rather than erroring.

## 6. Server glue — identity & MCP access

`spawn_agent` (`server_simple.py`) builds an `extra` dict passed on the
`SpawnRequest`; add your backend's wiring there. How each existing backend gets the
win-agent-teams tools:

- **claude-code** — a per-agent `--mcp-config <path>` file
  (`server_simple._write_mcp_config`) whose `env` carries the identity.
- **codex** — a per-process `-c mcp_servers.win-agent-teams.env={…}` override
  carrying the identity (`codex._mcp_identity_args`); avoids racing on the shared
  `~/.codex/config.toml`.
- **pi** — pi has no built-in MCP, so it uses the official `pi-mcp-adapter`
  package. A **human-launched pi lead** reads a generated `~/.pi/agent/mcp.json`
  (`server_simple._ensure_pi_mcp_config`) whose `env` uses `${AGENT_NAME}`
  interpolation, resolved from the pi process's own env (`pi.build_env`); one
  shared static file, race-free. A **spawned pi worker** additionally gets a
  per-agent `--mcp-config <session_dir>/mcp/<agent>.pi.mcp.json` file
  (`server_simple._write_pi_mcp_config`) with **literal** `AGENT_*` identity, so a
  worker's identity does not depend on `${AGENT_*}` interpolation.

  > **WARNING — never hardcode `AGENT_*` in a project MCP config.** Do not add an
  > `AGENT_NAME` / `AGENT_SESSION_ID` / `AGENT_PARENT_NAME` `env` block to a
  > project `.mcp.json` / `.pi/mcp.json` `win-agent-teams` entry. The
  > pi-mcp-adapter merges config sources later-wins and replaces the whole `env`
  > map per server entry, so an empty/literal `AGENT_*` there clobbers the correct
  > per-agent identity and forces the server to refuse identity-bearing tools
  > (`identity_unresolved`).

Add a per-backend branch to `server_simple._hook_extra` for any spawn-time setup
(pi's branch ensures the mcp.json and points at the state extension).

### Output fallback reader

So `check_agent`/`follow_up_agent` can recover `last_message` and
`backend_session_id` even when the agent never messaged, add a
`read_<name>_output(...)` in `src/claude_teams/agent_output.py` and return it
from your binder's `legacy_read` (see §6b). There is no longer a
`server_simple._read_agent_output` dispatch — the binding ladder replaced it.
The reader parses the CLI's own session/rollout log. When you control the storage
location and session id (as pi does via `--session-dir`/`--session-id`), enumeration
is a single directory listing (`agent_output.read_pi_output`). When you don't
(codex), match by cwd + start-time (`agent_output.read_codex_output`).

These readers are the **legacy** path only. They are reached through
`_TranscriptBinder.legacy_read`, for records written before correlation existed.
Everything spawned today goes through the binding ladder below instead.

## 6b. The message-delivery protocol

A backend that only implements the above will spawn fine and then refuse every
follow-up. Binding an agent to *its own* transcript — not merely to a plausible
one — needs three more pieces. Getting any of them wrong produces a false
`delivered`: the server reports a message as received when it was not.

**1. Correlation marker in the prompt.** The server mints a random
`correlation_id` **per spawn** (`agent_output.new_correlation_id`) and persists it
on the agent record. A per-agent *derived* token is not sufficient: a killed
agent's name can be reused, so a derived token would name two conversations.

Who appends the marker depends on how the prompt reaches the agent:

- If the CLI takes the prompt **verbatim** (codex, pi), the backend appends it —
  see `CodexBackend._correlated_prompt` / `PiBackend._correlated_prompt`, both
  reading `extra[CORRELATION_FIELD]`. The server leaves such prompts alone.
- If the server has to choose the transport (claude-code, which may route a
  CLI-sensitive prompt through a sidecar file), the **server** owns it in
  `server_simple._materialize_prompt`. The backend must then pass the prompt
  through untouched, or the argv path gets two markers.

Mark the **resume** as well as the spawn. A resume whose transcript cannot be
identified is exactly the unverifiable receipt the protocol exists to prevent.
Never mint an id for a record that has none — that is a `legacy` record, and a
fresh id would match nothing in the conversation that already exists.

**2. A binder.** Subclass `agent_output._TranscriptBinder` and register it in
`_make_binder`. It enumerates candidate transcripts, parses a session id, and
reads the last assistant message. There is deliberately **no newest-mtime
fallback**: zero token matches is `unverified`, two is `ambiguous`. Recency is
not identity, and a wrong `backend_session_id` pinned once never self-corrects.

If your backend needs lookup metadata that is not derivable from `cwd` — as pi
does, since the server chooses its storage dir — persist it on the agent record
at spawn and read it from the `record` argument (`PI_SESSION_DIR_FIELD`). Do not
re-derive the layout inside the reader.

**3. A named receipt record.** Add a branch to `delivery.receipt_nonces` naming
the record class that means *the agent received this text as input*: `type:
"user"` for claude-code, the user `response_item` for codex, `type: "message"`
with `message.role == "user"` for pi. Assistant output, tool invocations, and CLI
diagnostics are **not** receipts — a nonce there proves the text was echoed or
logged, not that it entered the agent's context.

An unlisted backend yields no receipt, which makes delivery `unconfirmed` rather
than `delivered`. That is the correct direction to fail, but it means follow-up
never confirms, so a backend without a receipt branch is effectively read-only.

## 7. State reporting

The coordinator watches `state-<agent>.json`. How each backend writes it (all
gated by `WIN_AGENT_TEAMS_STATE_HOOKS`, default on):

- **claude-code** — an injected settings file mapping every lifecycle event to a
  `python -m claude_teams.hooks emit …` command (`hooks.write_claude_settings`,
  `claude_code._hooks_settings_args`).
- **codex** — inline `-c hooks.<Event>=…` overrides **plus**
  `--dangerously-bypass-hook-trust` (codex silently skips injected hooks without
  it); on Windows a bare-path `.cmd` launcher avoids `cmd.exe` mangling
  (`hooks.codex_hook_overrides` / `write_codex_launcher`, `codex._hook_override_args`).
- **pi** — pi has no hook CLI, so a bundled zero-dep extension
  (`pi-extensions/win-agent-teams-state`, loaded via `-e`) writes the marker from
  pi's `session_start`/`turn_start`/`tool_call` → `running` and `agent_settled` →
  `waiting`. Path resolved by `server_simple._pi_state_extension_dir` (overridable
  with `WIN_AGENT_TEAMS_PI_EXTENSION`).

`hooks.emit` reads the hook payload on stdin and maps the event name to
`running`/`waiting` (`_RUNNING_EVENTS`/`_WAITING_EVENTS`). If your CLI can run a
command on lifecycle events, reuse `emit`; if not, write the marker directly (same
schema) as pi's extension does. Without any marker, `state` degrades to liveness +
activity-recency (`server_simple._resolve_agent_state`) and there is no `waiting`
signal.

## 8. Tests

Add `tests/test_backends/test_<name>.py`, mirroring `test_pi.py` / `test_codex.py`.
Patterns to reuse:

- Stub model discovery by monkeypatching the module's `_discover_*` function and
  asserting tier→(model, effort), soft/hard fallback, and blank-model behavior.
- Control the launcher (`monkeypatch.setattr(MyBackend, "_launcher", …)`) so tests
  don't need the CLI on PATH, and `process_manager.provides_tty` to exercise the
  interactive vs headless command shapes.
- For the output reader, write a small fixture JSONL in `tmp_path` and assert the
  parsed `last_message`/`backend_session_id` (see `test_pi.TestReadPiOutput`).

Run `ruff check` and the full suite; then smoke-test a real spawn per DEVELOPMENT.md
(including the Lubuntu VM run).

## 9. Checklist

- [ ] `backends/<name>.py` subclassing `BaseBackend`; `_name`/`_binary_name`/`is_interactive`
- [ ] Model tiers: `supported_models`/`default_model`/`resolve_model`/`resolve_launch` (+ live discovery, hard-fail or soft-fallback)
- [ ] `build_command` (+ headless split) and `build_resume_command` (deterministic session id)
- [ ] `build_env` with identity; permission args; `supports_resume`
- [ ] Windows launch: native/entry resolution or prompt-file transport
- [ ] Register in `registry._BUILTIN_BACKENDS`
- [ ] Server glue: MCP identity injection + `_hook_extra` branch
- [ ] `read_<name>_output` in `agent_output.py` + `_read_agent_output` branch
- [ ] Correlation marker on **both** spawn and resume (backend-side if the prompt is verbatim, else server-side)
- [ ] `_TranscriptBinder` subclass registered in `_make_binder` (no newest-mtime fallback)
- [ ] Named receipt record branch in `delivery.receipt_nonces`
- [ ] State reporting writing `state-<agent>.json`
- [ ] `tests/test_backends/test_<name>.py`
- [ ] README backend table + a setup section
