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

The MCP caller picks capability, not a raw model slug. Both `codex` and `pi`
expose five **tiers** (`low/medium/high/xhigh/ultra`) that each bundle a concrete
model + reasoning effort; `supported_models()` returns the tier names and
`resolve_launch()` maps a tier to `(model, effort)`:

```python
_TIER_LAUNCH = {                       # pi.py:120
    "low":   ("gpt-5.6-terra", "medium"),
    "medium":("gpt-5.6-sol",   "low"),
    ...
}
```

`resolve_launch(model, effort)` returns the `(model, effort)` pair the spawn uses.
Two policies to choose from:

- **Hard-fail** (`codex.py` `resolve_launch` / `_require_available`): a tier whose
  model isn't installed raises `BackendModelUnavailableError` — no silent
  downgrade.
- **Soft-fallback** (`pi.py` `resolve_launch`): an unavailable tier model returns
  `("", effort)` so the CLI uses its own default model. Use this when the user may
  be logged into a provider whose catalog differs.

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
  package + a generated `~/.pi/agent/mcp.json` (`server_simple._ensure_pi_mcp_config`)
  whose `env` uses `${AGENT_NAME}` interpolation, resolved from each pi process's
  own env (`pi.build_env`). One shared static file, race-free.

Add a per-backend branch to `server_simple._hook_extra` for any spawn-time setup
(pi's branch ensures the mcp.json and points at the state extension).

### Output fallback reader

So `check_agent`/`follow_up_agent` can recover `last_message` and
`backend_session_id` even when the agent never messaged, add a
`read_<name>_output(...)` in `src/claude_teams/agent_output.py` and a branch in
`server_simple._read_agent_output` (today it handles `codex`, `claude-code`, `pi`).
The reader parses the CLI's own session/rollout log. When you control the storage
location and session id (as pi does via `--session-dir`/`--session-id`), binding is
trivial — glob the one file, read the header id, scan backward for the last
assistant text (`agent_output.read_pi_output`). When you don't (codex), match by
cwd + start-time and, if needed, a correlation token appended to the prompt
(`agent_output.read_codex_output`, `codex_correlation_token`).

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
- [ ] State reporting writing `state-<agent>.json`
- [ ] `tests/test_backends/test_<name>.py`
- [ ] README backend table + a setup section
